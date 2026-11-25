"""Idempotent installer for Aetheris Microbrains hooks in ZCode configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MARKER = "aetheris_microbrains.hook"
PRE_TOOL_MATCHER = "Bash|Edit|Write|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
POST_TOOL_MATCHER = "Bash|Edit|Write|Read|Grep|WebFetch|WebSearch|mcp__.*"


def _python_exe() -> str:
    return sys.executable.replace("\\", "/")


def _hook_entry(event_args: List[str], extra: List[str], *, use_rust_shim: bool = False) -> Dict[str, Any]:
    from .core.rust_shim import build_hook_command

    cmd, full_args = build_hook_command(
        "zcode",
        event_args=event_args,
        extra_args=extra,
        use_rust_shim=use_rust_shim,
        auto_daemon=True,
    )
    return {
        "type": "process",
        "command": cmd,
        "args": full_args,
        "timeoutMs": 10000,
        "statusMessage": "Aetheris Microbrains gate",
    }


def _build_entries(
    *,
    profile: str = "strict",
    policy: Optional[str] = None,
    shadow: bool = False,
    no_session_state: bool = False,
    no_log: bool = False,
    use_rust_shim: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extra: List[str] = []
    if profile == "dev":
        extra += ["--profile", "dev"]
    if policy:
        extra += ["--policy", str(Path(policy).resolve()).replace("\\", "/")]
    if shadow:
        extra += ["--shadow"]
    session = [] if no_session_state else ["--session-state", "auto"]
    log = [] if no_log else ["--log-decisions"]
    pre = {
        "matcher": PRE_TOOL_MATCHER,
        "hooks": [_hook_entry([*session], extra + log, use_rust_shim=use_rust_shim)],
    }
    post = {
        "matcher": POST_TOOL_MATCHER,
        "hooks": [_hook_entry(["--event", "post-tool-use", *session], extra + log, use_rust_shim=use_rust_shim)],
    }
    return pre, post


def _config_path(*, user: bool = False, workspace: Optional[Path] = None) -> Path:
    if user:
        return Path.home() / ".zcode" / "cli" / "config.json"
    base = Path(workspace).expanduser().resolve() if workspace is not None else Path.cwd()
    return base / ".zcode" / "config.json"


def _is_aetheris_group(group: Any) -> bool:
    if not isinstance(group, dict):
        return False
    for h in group.get("hooks", []):
        blob = json.dumps(h)
        if MARKER in blob or "zcode_gate.py" in blob or "aetheris-gate" in blob:
            return True
    return False


def _merge(config: Dict[str, Any], pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    hooks = out.get("hooks")
    hooks_dict = dict(hooks) if isinstance(hooks, dict) else {}
    hooks_dict["enabled"] = True
    events = hooks_dict.get("events")
    events_dict = dict(events) if isinstance(events, dict) else {}
    for event, entry in (("PreToolUse", pre), ("PostToolUse", post)):
        existing = events_dict.get(event)
        groups = [g for g in existing if not _is_aetheris_group(g)] if isinstance(existing, list) else []
        groups.append(entry)
        events_dict[event] = groups
    hooks_dict["events"] = events_dict
    out["hooks"] = hooks_dict
    return out


def _strip(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    hooks = out.get("hooks")
    if not isinstance(hooks, dict):
        return out
    hooks_dict = dict(hooks)
    events = hooks_dict.get("events")
    if isinstance(events, dict):
        events_dict = dict(events)
        for event in ("PreToolUse", "PostToolUse"):
            existing = events_dict.get(event)
            if isinstance(existing, list):
                groups = [g for g in existing if not _is_aetheris_group(g)]
                if groups:
                    events_dict[event] = groups
                else:
                    events_dict.pop(event, None)
        if events_dict:
            hooks_dict["events"] = events_dict
        else:
            hooks_dict.pop("events", None)
    out["hooks"] = hooks_dict
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

    deny_v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        host="zcode",
        log_decisions=False,
    )
    allow_v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        host="zcode",
        log_decisions=False,
    )
    cfg_write = evaluate_hook_payload(
        {"tool_name": "Write", "tool_input": {"file_path": ".zcode/config.json", "content": "{}"}},
        host="zcode",
        log_decisions=False,
    )
    ok = (
        deny_v.get("decision") == "deny"
        and deny_v.get("exit_code") == 2
        and (deny_v.get("hookSpecificOutput") or {}).get("permissionDecision") == "deny"
        and allow_v.get("decision") == "allow"
        and allow_v.get("exit_code") == 0
        and cfg_write.get("decision") == "deny"
    )
    print("ZCODE SELF-TEST PASSED" if ok else "ZCODE SELF-TEST FAILED")
    return 0 if ok else 1


def install(
    *,
    workspace: Optional[Path] = None,
    user: bool = False,
    profile: str = "strict",
    policy: Optional[str] = None,
    shadow: bool = False,
    no_session_state: bool = False,
    no_log: bool = False,
    use_rust_shim: bool = False,
    remove: bool = False,
    dry_run: bool = False,
) -> int:
    path = _config_path(user=user, workspace=workspace)
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
            use_rust_shim=use_rust_shim,
        )
        config = _merge(config, pre, post)

    _write(path, config, dry_run=dry_run)
    where = "user" if user else "workspace"
    action = "removed from" if remove else "written to"
    if not dry_run:
        print(f"Aetheris ZCode hooks {action} {where} config -> {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Install Aetheris Microbrains hooks into ZCode config.")
    p.add_argument("--user", action="store_true", help="Write to ~/.zcode/cli/config.json instead of <repo>/.zcode/config.json")
    p.add_argument("--project-dir", default=None, help="Target repository for workspace scope")
    p.add_argument("--profile", choices=["strict", "dev"], default="strict")
    p.add_argument("--policy", default=None, help="Policy file passed through to the gate")
    p.add_argument("--shadow", action="store_true", help="Observe-only: never block")
    p.add_argument("--no-session-state", action="store_true")
    p.add_argument("--no-log", action="store_true", help="Do not request --log-decisions")
    p.add_argument("--use-rust-shim", action="store_true", help="Use native `aetheris-gate --auto-daemon` binary when available")
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
        use_rust_shim=bool(args.use_rust_shim),
        remove=bool(args.remove),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    sys.exit(main())
