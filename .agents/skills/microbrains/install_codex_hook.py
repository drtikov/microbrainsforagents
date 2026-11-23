"""Install, verify, or remove Aetheris Microbrains hooks in OpenAI Codex CLI (`hooks.json`).

Codex expects root-level hook event arrays in `<project>/.codex/hooks.json` or
`$CODEX_HOME/hooks.json` (default `~/.codex/hooks.json`) rather than Claude Code's
`{"hooks": ...}` wrapper.

Usage:
    python .agents/skills/microbrains/install_codex_hook.py --shadow          # observe-only first
    python .agents/skills/microbrains/install_codex_hook.py                   # enforce mode
    python .agents/skills/microbrains/install_codex_hook.py --user --profile dev
    python .agents/skills/microbrains/install_codex_hook.py --self-test
    python .agents/skills/microbrains/install_codex_hook.py --remove
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

SKILL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SKILL_DIR.parents[2]

CODEX_PRE_MATCHER = (
    "Bash|shell|shell_command|exec_command|apply_patch|Edit|Write|fwrite|"
    "Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
)
CODEX_POST_MATCHER = (
    "Bash|shell|shell_command|exec_command|apply_patch|Edit|Write|fwrite|"
    "Read|Grep|WebFetch|WebSearch|mcp__.*"
)

FULL_LIFECYCLE_SPECS: tuple[tuple[str, str, str | None], ...] = (
    ("SessionStart", "session-start", "startup|resume|compact"),
    ("UserPromptSubmit", "user-prompt-submit", None),
    ("Stop", "stop", None),
)
_ALL_MANAGED_EVENTS: tuple[str, ...] = ("PreToolUse", "PostToolUse") + tuple(
    ev for ev, _slug, _m in FULL_LIFECYCLE_SPECS
)

_PY = sys.executable.replace("\\", "/")


def _quote_cmd_token(tok: str) -> str:
    s = str(tok).replace("\\", "/")
    if not s or any(ch in s for ch in (" ", "\t", "\n", '"', "'", "&", "|", ";", "<", ">", "(", ")")):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _hook_cmd(shared: bool, event_args: list[str], extra: list[str]) -> list[str]:
    py_tok = "python" if shared else _quote_cmd_token(_PY)
    quoted_extra = [_quote_cmd_token(x) if not x.startswith("--") else x for x in extra]
    return [py_tok, "-m", "aetheris_microbrains.hook", "--host", "codex", *event_args, *quoted_extra]


def _hook_entry(cmd: list[str], timeout: int = 10) -> dict:
    return {
        "type": "command",
        "command": " ".join(cmd),
        "timeout": timeout,
    }


def _is_aetheris_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if "aetheris_microbrains.hook" in str(entry.get("command", "")):
        return True
    return any(_is_aetheris_entry(h) for h in entry.get("hooks", []) if isinstance(h, dict))


def _strip_aetheris_from_event_list(entries: list) -> list:
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


def _resolve_hooks_path(
    *,
    hooks_path: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    user_scope: bool = False,
) -> Path:
    if hooks_path is not None:
        return Path(hooks_path)
    if user_scope:
        codex_home = str(os.getenv("CODEX_HOME") or "").strip()
        base = Path(codex_home).expanduser() if codex_home else (Path.home() / ".codex")
        return base / "hooks.json"
    root = Path(project_dir) if project_dir is not None else PROJECT_DIR
    return root / ".codex" / "hooks.json"


def _load_validated_hooks_json(target: Path) -> tuple[dict | None, int]:
    if not target.exists():
        return {}, 0
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: {target} is not valid JSON ({exc}); refusing to overwrite.", file=sys.stderr)
        return None, 1
    if not isinstance(data, dict):
        print(f"ERROR: {target} top-level JSON must be an object; refusing to overwrite.", file=sys.stderr)
        return None, 1
    for ev in _ALL_MANAGED_EVENTS:
        if ev in data and not isinstance(data[ev], list):
            print(f"ERROR: {target} '{ev}' must be a list; refusing to overwrite.", file=sys.stderr)
            return None, 1
    return data, 0


def install(
    *,
    strict_ast: bool = False,
    shadow: bool = False,
    shared: bool = False,
    user_scope: bool = False,
    hooks_path: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    auto_allow: bool = False,
    codex_escalate_deny: bool = False,
    profile: Optional[str] = None,
    policy: Optional[str] = None,
    session_state: Optional[str] = "auto",
    journal_path: Optional[str] = None,
    context_guard: bool = False,
    full_lifecycle: bool = False,
    dry_run: bool = False,
) -> int:
    target = _resolve_hooks_path(hooks_path=hooks_path, project_dir=project_dir, user_scope=user_scope)
    data, rc = _load_validated_hooks_json(target)
    if rc != 0 or data is None:
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
    if codex_escalate_deny:
        pre_extra.append("--codex-escalate-deny")
    if context_guard:
        pre_extra.append("--context-guard")
        post_extra.append("--context-guard")
    if profile:
        pre_extra.extend(["--profile", profile])
    if policy:
        pre_extra.extend(["--policy", policy])
        post_extra.extend(["--policy", policy])
    if session_state and str(session_state).lower() != "none":
        pre_extra.extend(["--session-state", session_state])
        post_extra.extend(["--session-state", session_state])
    if journal_path:
        pre_extra.extend(["--journal-path", journal_path])
        post_extra.extend(["--journal-path", journal_path])

    pre = _strip_aetheris_from_event_list(data.get("PreToolUse", []))
    post = _strip_aetheris_from_event_list(data.get("PostToolUse", []))

    pre.append({
        "matcher": CODEX_PRE_MATCHER,
        "hooks": [_hook_entry(_hook_cmd(shared, [], pre_extra), timeout=10)],
    })
    post.append({
        "matcher": CODEX_POST_MATCHER,
        "hooks": [_hook_entry(_hook_cmd(shared, ["--event", "post-tool-use"], post_extra), timeout=10)],
    })
    data["PreToolUse"] = pre
    data["PostToolUse"] = post

    if full_lifecycle:
        lc_extra = ["--log-decisions"]
        if shadow:
            lc_extra.append("--shadow")
        if session_state and str(session_state).lower() != "none":
            lc_extra.extend(["--session-state", session_state])
        if journal_path:
            lc_extra.extend(["--journal-path", journal_path])
        for ev_name, ev_slug, ev_matcher in FULL_LIFECYCLE_SPECS:
            ev_list = _strip_aetheris_from_event_list(data.get(ev_name, []))
            wrapper: dict = {"hooks": [_hook_entry(_hook_cmd(shared, ["--event", ev_slug], lc_extra), timeout=10)]}
            if ev_matcher is not None:
                wrapper["matcher"] = ev_matcher
            ev_list.append(wrapper)
            data[ev_name] = ev_list

    if dry_run:
        print(f"DRY RUN - would write Codex hooks to {target}:")
        print(json.dumps(data, indent=2))
        return 0

    _atomic_write_json(target, data)
    mode = "shadow (observe-only)" if shadow else "enforce"
    scope = "user" if user_scope else "project"
    print(f"Installed Aetheris Codex hooks ({mode}, {scope}) -> {target}")
    return 0


def remove(
    *,
    user_scope: bool = False,
    hooks_path: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    target = _resolve_hooks_path(hooks_path=hooks_path, project_dir=project_dir, user_scope=user_scope)
    if not target.exists():
        print(f"No {target}; nothing to remove.")
        return 0
    data, rc = _load_validated_hooks_json(target)
    if rc != 0 or data is None:
        return rc
    for ev in _ALL_MANAGED_EVENTS:
        if ev in data:
            data[ev] = _strip_aetheris_from_event_list(data[ev])
    if dry_run:
        print(f"DRY RUN - would write to {target}:")
        print(json.dumps(data, indent=2))
        return 0
    _atomic_write_json(target, data)
    print(f"Removed Aetheris Codex hooks from {target}")
    return 0


def self_test(
    *,
    user_scope: bool = False,
    hooks_path: Optional[Path] = None,
    project_dir: Optional[Path] = None,
) -> int:
    import importlib

    repo_root = str(PROJECT_DIR if project_dir is None else Path(project_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    failures: list[str] = []
    try:
        hook_mod = importlib.import_module("aetheris_microbrains.hook")
        deny_v = hook_mod.evaluate_hook_payload(
            {"tool_name": "shell_command", "tool_input": {"command": "rm -rf /"}},
            host="codex",
            log_decisions=False,
        )
        allow_v = hook_mod.evaluate_hook_payload(
            {"tool_name": "shell_command", "tool_input": {"command": "git status"}},
            host="codex",
            log_decisions=False,
        )
        esc_v = hook_mod.evaluate_hook_payload(
            {"tool_name": "shell_command", "tool_input": {"command": "custom_unknown_tool --flag"}},
            host="codex",
            log_decisions=False,
        )
        if not (
            deny_v.get("exit_code") == 2
            and deny_v.get("permissionDecision") == "deny"
            and allow_v.get("exit_code") == 0
            and allow_v.get("permissionDecision") == "allow"
            and esc_v.get("host_action") == "defer_to_host_policy"
            and esc_v.get("permissionDecision") is None
        ):
            failures.append("codex-probe-contract")
    except Exception as exc:
        print(f"[FAIL] Codex probe failed: {exc}")
        failures.append("probe-error")

    target = _resolve_hooks_path(hooks_path=hooks_path, project_dir=project_dir, user_scope=user_scope)
    if not target.is_file():
        print(f"[FAIL] missing {target}; run install_codex_hook.py first")
        failures.append("not-installed")
    else:
        data, rc = _load_validated_hooks_json(target)
        if rc != 0 or data is None:
            failures.append("invalid-json")
        else:
            pre = [e for e in data.get("PreToolUse", []) if _is_aetheris_entry(e)]
            post = [e for e in data.get("PostToolUse", []) if _is_aetheris_entry(e)]
            if len(pre) != 1 or len(post) != 1:
                failures.append("hook-entries")
            elif pre[0].get("matcher") != CODEX_PRE_MATCHER or post[0].get("matcher") != CODEX_POST_MATCHER:
                failures.append("matcher-mismatch")

    if failures:
        print(f"CODEX SELF-TEST FAILED: {', '.join(failures)}")
        return 1
    print(f"CODEX SELF-TEST PASSED ({target})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict-ast", action="store_true", help="Enable strict AST import/dynamic-call allowlist")
    ap.add_argument("--shadow", action="store_true", help="Observe-only pilot mode: log without blocking")
    ap.add_argument("--auto-allow", action="store_true", help="Emit explicit permissionDecision='allow' on safe verdicts")
    ap.add_argument(
        "--codex-escalate-deny",
        action="store_true",
        help="Treat escalated actions as deny instead of deferring to Codex approval policy",
    )
    ap.add_argument("--profile", choices=["strict", "dev"], default=None, help="Policy profile (`strict` or `dev`)")
    ap.add_argument("--policy", default=None, help="Path to JSON/TOML/YAML PolicyConfig file")
    ap.add_argument("--session-state", default="auto", help="SessionSecurityState path or 'auto' (default: auto)")
    ap.add_argument("--no-session-state", action="store_true", help="Disable SessionSecurityState tracking")
    ap.add_argument("--journal-path", default=None, help="Override path for decision_journal.jsonl")
    ap.add_argument("--context-guard", action="store_true", help="Enable context-hygiene gating")
    ap.add_argument("--full-lifecycle", action="store_true", help="Also register SessionStart, UserPromptSubmit, Stop")
    ap.add_argument("--shared", action="store_true", help="Use portable `python` executable instead of current interpreter path")
    ap.add_argument("--user", action="store_true", help="Install to user `$CODEX_HOME/hooks.json` (`~/.codex/hooks.json`)")
    ap.add_argument("--project-dir", default=None, help="Target project root containing `.codex/` (defaults to repo root)")
    ap.add_argument("--hooks-path", default=None, help="Explicit path to Codex hooks.json")
    ap.add_argument("--dry-run", action="store_true", help="Preview hooks.json without writing")
    ap.add_argument("--self-test", action="store_true", help="Verify Codex contract and installed hooks.json")
    ap.add_argument("--remove", action="store_true", help="Remove Aetheris entries from Codex hooks.json")
    args = ap.parse_args(argv)

    proj = Path(args.project_dir) if args.project_dir else None
    hp = Path(args.hooks_path) if args.hooks_path else None
    if args.self_test:
        return self_test(user_scope=args.user, hooks_path=hp, project_dir=proj)
    if args.remove:
        return remove(user_scope=args.user, hooks_path=hp, project_dir=proj, dry_run=args.dry_run)

    sess_arg = None if args.no_session_state else args.session_state
    return install(
        strict_ast=args.strict_ast,
        shadow=args.shadow,
        shared=args.shared,
        user_scope=args.user,
        hooks_path=hp,
        project_dir=proj,
        auto_allow=args.auto_allow,
        codex_escalate_deny=args.codex_escalate_deny,
        profile=args.profile,
        policy=args.policy,
        session_state=sess_arg,
        journal_path=args.journal_path,
        context_guard=args.context_guard,
        full_lifecycle=args.full_lifecycle,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
