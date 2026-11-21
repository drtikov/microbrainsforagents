#!/usr/bin/env python
"""Idempotent installer for Aetheris PreToolUse/PostToolUse hooks in Claude settings.

Merges the hook entries into `.claude/settings.local.json` (default, gitignored)
or `.claude/settings.json` (`--shared`) without clobbering unrelated configuration.
`--remove` strips only the aetheris hook entries.

Usage:
    python install_hook.py [--strict-ast] [--shadow] [--shared] [--auto-allow]
                           [--profile strict|dev] [--policy FILE]
                           [--session-state FILE] [--journal-path FILE]
                           [--project-dir DIR] [--async-post] [--remove]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SKILL_DIR.parents[2]
SETTINGS_LOCAL_PATH = PROJECT_DIR / ".claude" / "settings.local.json"
SETTINGS_SHARED_PATH = PROJECT_DIR / ".claude" / "settings.json"

PRE_TOOL_MATCHER = "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|WebFetch|WebSearch|mcp__.*"
POST_TOOL_MATCHER = "Bash|Read|WebFetch|WebSearch|mcp__.*"

# Forward slashes + double quotes: hooks run through Git Bash on Windows or
# sh / cmd.exe; single-quoted paths break under cmd.exe and unescaped backslashes
# are eaten as escapes under Git Bash.
_PY = sys.executable.replace("\\", "/")


def _quote_cmd_token(tok: str) -> str:
    """Double-quote a token when it contains spaces or shell metacharacters."""
    s = str(tok)
    if not s or any(ch in s for ch in (' ', '\t', '\n', '"', "'", '&', '|', ';', '<', '>', '(', ')')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _hook_cmd(shared: bool, event_args: list[str], extra: list[str]) -> list[str]:
    py_tok = "python" if shared else _quote_cmd_token(_PY)
    quoted_extra = [_quote_cmd_token(x) if not x.startswith("--") else x for x in extra]
    return [py_tok, "-m", "aetheris_microbrains.hook", *event_args, *quoted_extra]


def _hook_entry(
    cmd: list[str],
    timeout: int = 10,
    is_async: bool = False,
    on_failure: str | None = None,
) -> dict:
    entry: dict = {
        "type": "command",
        "command": " ".join(cmd),
        "timeout": timeout,
    }
    if on_failure is not None:
        entry["onFailure"] = on_failure
    if is_async:
        entry["async"] = True
    return entry


def _is_aetheris_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if "aetheris_microbrains.hook" in str(entry.get("command", "")):
        return True
    # Matcher wrappers nest the command under "hooks"; recurse to identify wrappers.
    return any(
        _is_aetheris_entry(h) for h in entry.get("hooks", []) if isinstance(h, dict)
    )


def _strip_aetheris_from_event_list(entries: list) -> list:
    """Remove only Aetheris hook entries while preserving unrelated user hooks in shared wrappers."""
    kept: list = []
    for entry in entries:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        inner = entry.get("hooks")
        if isinstance(inner, list):
            filtered = [h for h in inner if not _is_aetheris_entry(h)]
            if filtered:
                updated = dict(entry)
                updated["hooks"] = filtered
                kept.append(updated)
        elif not _is_aetheris_entry(entry):
            kept.append(entry)
    return kept


def _atomic_write_json(target: Path, data: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2) + "\n")
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_validated_settings(target: Path) -> tuple[dict | None, int]:
    """Load and validate Claude settings JSON structure; return `(settings_dict, exit_code)`."""
    if not target.exists():
        return {}, 0
    try:
        settings = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: {target} is not valid JSON ({exc}); refusing to overwrite.", file=sys.stderr)
        return None, 1
    if not isinstance(settings, dict):
        print(f"ERROR: {target} top-level JSON must be an object; refusing to overwrite.", file=sys.stderr)
        return None, 1
    if "hooks" in settings:
        hooks = settings["hooks"]
        if not isinstance(hooks, dict):
            print(f"ERROR: {target} 'hooks' field must be an object; refusing to overwrite.", file=sys.stderr)
            return None, 1
        for ev in ("PreToolUse", "PostToolUse"):
            if ev in hooks and not isinstance(hooks[ev], list):
                print(f"ERROR: {target} 'hooks.{ev}' must be a list; refusing to overwrite.", file=sys.stderr)
                return None, 1
    return settings, 0


def install(
    strict_ast: bool,
    shadow: bool,
    shared: bool = False,
    settings_path: Path | None = None,
    auto_allow: bool = False,
    profile: str | None = None,
    policy: str | None = None,
    session_state: str | None = None,
    journal_path: str | None = None,
    async_post: bool = False,
    project_dir: Path | None = None,
) -> int:
    if settings_path is not None:
        target = settings_path
    elif project_dir is not None:
        target = Path(project_dir) / ".claude" / ("settings.json" if shared else "settings.local.json")
    else:
        target = SETTINGS_SHARED_PATH if shared else SETTINGS_LOCAL_PATH

    settings, rc = _load_validated_settings(target)
    if rc != 0 or settings is None:
        return rc

    pre_extra = ["--log-decisions"]
    post_extra = ["--log-decisions"]
    if strict_ast:
        pre_extra.append("--strict-ast")
    if shadow:
        pre_extra.append("--shadow")
        post_extra.append("--shadow")
    if auto_allow:
        pre_extra.append("--auto-allow")
    if profile:
        pre_extra.extend(["--profile", profile])
    if policy:
        pre_extra.extend(["--policy", policy])
        post_extra.extend(["--policy", policy])
    if session_state:
        pre_extra.extend(["--session-state", session_state])
        post_extra.extend(["--session-state", session_state])
    if journal_path:
        pre_extra.extend(["--journal-path", journal_path])
        post_extra.extend(["--journal-path", journal_path])

    hooks = settings.setdefault("hooks", {})
    pre = _strip_aetheris_from_event_list(hooks.get("PreToolUse", []))
    pre.append({
        "matcher": PRE_TOOL_MATCHER,
        "hooks": [_hook_entry(_hook_cmd(shared, [], pre_extra), on_failure=None if shadow else "block")],
    })
    post = _strip_aetheris_from_event_list(hooks.get("PostToolUse", []))
    post.append({
        "matcher": POST_TOOL_MATCHER,
        "hooks": [_hook_entry(_hook_cmd(shared, ["--event", "post-tool-use"], post_extra), is_async=async_post)],
    })
    hooks["PreToolUse"] = pre
    hooks["PostToolUse"] = post

    _atomic_write_json(target, settings)
    mode = "shadow (observe-only)" if shadow else "enforce"
    ast_mode = "strict-AST" if strict_ast else "default-AST"
    scope = "shared" if shared else "local"
    print(f"Installed Aetheris hooks ({mode}, {ast_mode}, {scope}) -> {target}")
    print('Probe:  echo \'{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}\' | python -m aetheris_microbrains.hook')
    return 0


def remove(
    shared: bool = False,
    settings_path: Path | None = None,
    project_dir: Path | None = None,
) -> int:
    if settings_path is not None:
        target = settings_path
    elif project_dir is not None:
        target = Path(project_dir) / ".claude" / ("settings.json" if shared else "settings.local.json")
    else:
        target = SETTINGS_SHARED_PATH if shared else SETTINGS_LOCAL_PATH

    if not target.exists():
        print(f"No {target.name}; nothing to remove.")
        return 0
    settings, rc = _load_validated_settings(target)
    if rc != 0 or settings is None:
        return rc
    hooks = settings.get("hooks", {})
    for event in ("PreToolUse", "PostToolUse"):
        if event in hooks:
            hooks[event] = _strip_aetheris_from_event_list(hooks[event])
    _atomic_write_json(target, settings)
    print(f"Removed Aetheris hooks from {target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict-ast", action="store_true", help="Enable strict AST import/dynamic-call allowlist")
    ap.add_argument("--shadow", action="store_true", help="Observe-only: log decisions, never block")
    ap.add_argument("--auto-allow", action="store_true", help="Emit permissionDecision='allow' on safe verdicts (skips Claude Code prompts for gate-approved calls)")
    ap.add_argument("--shared", action="store_true", help="Target shared .claude/settings.json with portable `python`")
    ap.add_argument("--profile", choices=["strict", "dev"], default=None, help="Policy profile (`strict` or `dev`)")
    ap.add_argument("--policy", default=None, help="Path to JSON/TOML/YAML PolicyConfig file")
    ap.add_argument("--session-state", default=None, help="Path for cross-turn SessionSecurityState JSON")
    ap.add_argument("--journal-path", default=None, help="Override path for decision_journal.jsonl")
    ap.add_argument("--project-dir", default=None, help="Target project root containing `.claude/` (defaults to repo root)")
    ap.add_argument("--async-post", action="store_true", help="Run PostToolUse asynchronously instead of blocking before output ingestion")
    ap.add_argument("--remove", action="store_true", help="Uninstall the Aetheris hooks")
    args = ap.parse_args()
    proj = Path(args.project_dir) if args.project_dir else None
    if args.remove:
        return remove(shared=args.shared, project_dir=proj)
    return install(
        args.strict_ast,
        args.shadow,
        shared=args.shared,
        auto_allow=args.auto_allow,
        profile=args.profile,
        policy=args.policy,
        session_state=args.session_state,
        journal_path=args.journal_path,
        async_post=args.async_post,
        project_dir=proj,
    )


if __name__ == "__main__":
    raise SystemExit(main())
