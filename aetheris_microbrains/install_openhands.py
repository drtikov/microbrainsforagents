"""Idempotent installer for Aetheris Microbrains hooks in `.openhands/hooks.json`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MARKER = "aetheris_microbrains.hook"
TOOL_MATCHER = "terminal|file_editor|execute_bash|str_replace_editor|bash|edit|write|read|mcp__.*"


def _python_exe() -> str:
    return sys.executable.replace("\\", "/")


def _hook_entry(event_args: List[str], extra: List[str]) -> Dict[str, Any]:
    return {
        "type": "command",
        "command": " ".join([f'"{_python_exe()}"', "-m", "aetheris_microbrains.hook", *event_args, *extra]),
        "timeout": 10,
        "async": False,
    }


def _build_entries(
    *,
    profile: str = "dev",
    policy: Optional[str] = None,
    shadow: bool = False,
    no_session_state: bool = False,
    no_log: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extra: List[str] = ["--profile", profile]
    if policy:
        extra += ["--policy", f'"{str(Path(policy).resolve()).replace(chr(92), "/")}"']
    if shadow:
        extra += ["--shadow"]
    session = [] if no_session_state else ["--session-state", "auto"]
    log = [] if no_log else ["--log-decisions"]
    pre = {
        "matcher": TOOL_MATCHER,
        "hooks": [_hook_entry(["--host", "openhands", *session], extra + log)],
    }
    post = {
        "matcher": TOOL_MATCHER,
        "hooks": [_hook_entry(["--host", "openhands", "--event", "post-tool-use", *session], extra + log)],
    }
    return pre, post


def _hooks_path(*, user: bool = False, workspace: Optional[Path] = None) -> Path:
    if user:
        return Path.home() / ".openhands" / "hooks.json"
    base = Path(workspace).expanduser().resolve() if workspace is not None else Path.cwd()
    return base / ".openhands" / "hooks.json"


def _is_aetheris_group(group: Any) -> bool:
    if not isinstance(group, dict):
        return False
    for h in group.get("hooks", []):
        if MARKER in json.dumps(h):
            return True
    return False


def _merge(config: Dict[str, Any], pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    hooks = out.get("hooks")
    use_wrapper = isinstance(hooks, dict) or ("pre_tool_use" not in out and "post_tool_use" not in out)
    if use_wrapper:
        hooks_dict = dict(hooks) if isinstance(hooks, dict) else {}
        pre_key, post_key = "PreToolUse", "PostToolUse"
        target = hooks_dict
    else:
        hooks_dict = out
        pre_key, post_key = "pre_tool_use", "post_tool_use"
        target = out

    for key, entry in ((pre_key, pre), (post_key, post)):
        existing = target.get(key)
        groups = [g for g in existing if not _is_aetheris_group(g)] if isinstance(existing, list) else []
        groups.append(entry)
        target[key] = groups

    if use_wrapper:
        out["hooks"] = hooks_dict
    return out


def _strip(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    hooks = out.get("hooks")
    container = dict(hooks) if isinstance(hooks, dict) else out
    for key in ("PreToolUse", "PostToolUse", "pre_tool_use", "post_tool_use"):
        existing = container.get(key)
        if isinstance(existing, list):
            groups = [g for g in existing if not _is_aetheris_group(g)]
            if groups:
                container[key] = groups
            else:
                container.pop(key, None)
    if isinstance(hooks, dict):
        out["hooks"] = container
    return out


def _write(path: Path, config: Dict[str, Any], dry_run: bool) -> None:
    text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    if dry_run:
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def self_test() -> int:
    from .hook import evaluate_hook_payload

    deny_term = evaluate_hook_payload(
        {"tool_name": "terminal", "tool_input": {"command": "rm -rf /"}},
        host="openhands",
        log_decisions=False,
    )
    allow_term = evaluate_hook_payload(
        {"tool_name": "terminal", "tool_input": {"command": "git status"}},
        host="openhands",
        log_decisions=False,
    )
    deny_cfg = evaluate_hook_payload(
        {
            "tool_name": "file_editor",
            "tool_input": {"command": "create", "path": ".openhands/hooks.json", "file_text": "{}"},
        },
        host="openhands",
        log_decisions=False,
    )
    deny_ast = evaluate_hook_payload(
        {
            "tool_name": "file_editor",
            "tool_input": {"command": "create", "path": "broken.py", "file_text": "def broken(:\n"},
        },
        host="openhands",
        log_decisions=False,
    )
    ok = (
        deny_term.get("decision") == "deny"
        and deny_term.get("exit_code") == 2
        and allow_term.get("decision") == "allow"
        and allow_term.get("exit_code") == 0
        and deny_cfg.get("decision") == "deny"
        and deny_ast.get("decision") == "deny"
    )
    print("OPENHANDS SELF-TEST PASSED" if ok else "OPENHANDS SELF-TEST FAILED")
    return 0 if ok else 1


def install(
    *,
    workspace: Optional[Path] = None,
    user: bool = False,
    profile: str = "dev",
    policy: Optional[str] = None,
    shadow: bool = False,
    no_session_state: bool = False,
    no_log: bool = False,
    remove: bool = False,
    dry_run: bool = False,
) -> int:
    path = _hooks_path(user=user, workspace=workspace)
    config: Dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config = loaded

    if remove:
        config = _strip(config)
    else:
        pre, post = _build_entries(
            profile=profile,
            policy=policy,
            shadow=shadow,
            no_session_state=no_session_state,
            no_log=no_log,
        )
        config = _merge(config, pre, post)

    _write(path, config, dry_run=dry_run)
    where = "user" if user else "workspace"
    action = "removed from" if remove else "written to"
    if not dry_run:
        print(f"Aetheris OpenHands hooks {action} {where} config -> {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Install Aetheris Microbrains hooks into .openhands/hooks.json.")
    p.add_argument("--user", action="store_true", help="Write to ~/.openhands/hooks.json instead of <repo>/.openhands/hooks.json")
    p.add_argument("--project-dir", default=None, help="Target repository for workspace scope")
    p.add_argument("--profile", choices=["strict", "dev"], default="dev")
    p.add_argument("--policy", default=None, help="Policy file passed through to the gate")
    p.add_argument("--shadow", action="store_true", help="Observe-only: never block")
    p.add_argument("--no-session-state", action="store_true")
    p.add_argument("--no-log", action="store_true", help="Do not request --log-decisions")
    p.add_argument("--remove", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    ws = Path(args.project_dir).expanduser().resolve() if args.project_dir else None
    return install(
        workspace=ws,
        user=bool(args.user),
        profile=str(args.profile),
        policy=args.policy,
        shadow=bool(args.shadow),
        no_session_state=bool(args.no_session_state),
        no_log=bool(args.no_log),
        remove=bool(args.remove),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    sys.exit(main())
