#!/usr/bin/env python
"""Idempotent installer for Aetheris PreToolUse/PostToolUse hooks in Claude settings.

Merges the hook entries into `.claude/settings.local.json` (default, gitignored)
or `.claude/settings.json` (`--shared`) without clobbering unrelated configuration.
`--remove` strips only the aetheris hook entries.

Usage:
    python install_hook.py [--strict-ast] [--shadow] [--shared] [--remove]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SKILL_DIR.parents[2]
SETTINGS_LOCAL_PATH = PROJECT_DIR / ".claude" / "settings.local.json"
SETTINGS_SHARED_PATH = PROJECT_DIR / ".claude" / "settings.json"

# Forward slashes + shlex.quote: hooks run through bash on Windows (Git Bash), where
# backslash paths are eaten as escape characters and unquoted spaces split the path.
_PY = sys.executable.replace("\\", "/")


def _hook_cmd(shared: bool, event_args: list[str], extra: list[str]) -> list[str]:
    py_tok = "python" if shared else shlex.quote(_PY)
    return [py_tok, "-m", "aetheris_microbrains.hook", *event_args, *extra]


def _hook_entry(cmd: list[str], timeout: int = 10) -> dict:
    return {
        "type": "command",
        "command": " ".join(cmd),
        "timeout": timeout,
    }


def _is_aetheris_entry(entry: dict) -> bool:
    if "aetheris_microbrains.hook" in str(entry.get("command", "")):
        return True
    # Matcher wrappers nest the command under "hooks"; recurse so install()
    # replaces existing entries instead of appending duplicates.
    return any(
        _is_aetheris_entry(h) for h in entry.get("hooks", []) if isinstance(h, dict)
    )


def install(
    strict_ast: bool,
    shadow: bool,
    shared: bool = False,
    settings_path: Path | None = None,
) -> int:
    target = settings_path or (SETTINGS_SHARED_PATH if shared else SETTINGS_LOCAL_PATH)
    settings: dict = {}
    if target.exists():
        try:
            settings = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ERROR: {target} is not valid JSON ({exc}); refusing to overwrite.", file=sys.stderr)
            return 1

    extra = ["--log-decisions"]
    if strict_ast:
        extra.append("--strict-ast")
    if shadow:
        extra.append("--shadow")

    hooks = settings.setdefault("hooks", {})
    pre = [e for e in hooks.get("PreToolUse", []) if not _is_aetheris_entry(e)]
    pre.append({"matcher": "*", "hooks": [_hook_entry(_hook_cmd(shared, [], extra))]})
    post = [e for e in hooks.get("PostToolUse", []) if not _is_aetheris_entry(e)]
    post.append({"matcher": "*", "hooks": [_hook_entry(_hook_cmd(shared, ["--event", "post-tool-use"], extra))]})
    hooks["PreToolUse"] = pre
    hooks["PostToolUse"] = post

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    mode = "shadow (observe-only)" if shadow else "enforce"
    ast_mode = "strict-AST" if strict_ast else "default-AST"
    scope = "shared" if shared else "local"
    print(f"Installed Aetheris hooks ({mode}, {ast_mode}, {scope}) -> {target}")
    print("Probe:  echo '{\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"rm -rf /tmp/aetheris_probe\"}}' | python -m aetheris_microbrains.hook")
    return 0


def remove(shared: bool = False, settings_path: Path | None = None) -> int:
    target = settings_path or (SETTINGS_SHARED_PATH if shared else SETTINGS_LOCAL_PATH)
    if not target.exists():
        print(f"No {target.name}; nothing to remove.")
        return 0
    try:
        settings = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: {target} is not valid JSON ({exc}); refusing to overwrite.", file=sys.stderr)
        return 1
    hooks = settings.get("hooks", {})
    for event in ("PreToolUse", "PostToolUse"):
        kept = []
        for entry in hooks.get(event, []):
            inner = entry.get("hooks", [])
            inner = [h for h in inner if not _is_aetheris_entry(h)]
            if inner:
                entry["hooks"] = inner
                kept.append(entry)
        hooks[event] = kept
    target.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Removed Aetheris hooks from {target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict-ast", action="store_true", help="Enable strict AST import/dynamic-call allowlist")
    ap.add_argument("--shadow", action="store_true", help="Observe-only: log decisions, never block")
    ap.add_argument("--shared", action="store_true", help="Target shared .claude/settings.json with portable `python`")
    ap.add_argument("--remove", action="store_true", help="Uninstall the Aetheris hooks")
    args = ap.parse_args()
    return remove(shared=args.shared) if args.remove else install(args.strict_ast, args.shadow, shared=args.shared)


if __name__ == "__main__":
    raise SystemExit(main())
