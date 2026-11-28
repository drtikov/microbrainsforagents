"""Unified `aetheris` (`uvx aetheris-microbrains`) CLI & Multi-Host Installer (stdlib-only).

Supports:
  - `aetheris install <host>` (`antigravity|claude|codex|hermes|cursor|openclaw|opencode|kilocode|pi|omp|qoder|zcode|openhands|dsh|auto`)
  - `aetheris audit ...`
  - `aetheris doctor ...`
  - `aetheris mcp-proxy ...`
  - `aetheris pr-gate ...`
  - `aetheris verify-core [--json]`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_HOSTS = ("antigravity", "claude", "codex", "hermes", "cursor", "devin", "cline", "mcode", "ouroboros", "prime", "openclaw", "opencode", "kilocode", "pi", "omp", "qoder", "qwen", "zcode", "openhands", "dsh", "auto")


def _atomic_write_json(target: Path, data: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def detect_installed_hosts(workspace: Optional[Path] = None) -> List[str]:
    """Detect AI agent harnesses present on this workstation or workspace."""
    home = Path.home()
    ws = Path(workspace) if workspace else Path.cwd()
    detected: List[str] = []
    if (home / ".gemini").exists() or (ws / ".agents").exists():
        detected.append("antigravity")
    if (home / ".claude").exists() or (ws / ".claude").exists():
        detected.append("claude")
    if (home / ".codex").exists() or (ws / ".codex").exists():
        detected.append("codex")
    if (home / ".hermes").exists():
        detected.append("hermes")
    if (home / ".cursor").exists() or (ws / ".cursor").exists():
        detected.append("cursor")
    if (home / ".openclaw").exists():
        detected.append("openclaw")
    if (home / ".qoder").exists() or (ws / ".qoder").exists():
        detected.append("qoder")
    if (home / ".qwen").exists() or (ws / ".qwen").exists():
        detected.append("qwen")
    if (
        (home / ".config" / "opencode").exists()
        or (home / ".opencode").exists()
        or (ws / ".opencode").exists()
        or (ws / "opencode.json").exists()
    ):
        detected.append("opencode")
    if (
        (home / ".config" / "kilo").exists()
        or (home / ".kilo").exists()
        or (ws / ".kilo").exists()
        or (ws / "kilo.json").exists()
    ):
        detected.append("kilocode")
    if (
        (home / ".pi").exists()
        or (home / ".pi" / "agent").exists()
        or (ws / ".pi").exists()
    ):
        detected.append("pi")
    if (
        (home / ".omp").exists()
        or (home / ".omp" / "agent").exists()
        or (ws / ".omp").exists()
        or (ws / "omp.json").exists()
    ):
        detected.append("omp")
    if (home / ".zcode").exists() or (ws / ".zcode").exists():
        detected.append("zcode")
    if (home / ".openhands").exists() or (ws / ".openhands").exists():
        detected.append("openhands")
    devin_user = (
        (Path(os.environ["APPDATA"]) / "devin")
        if os.name == "nt" and os.environ.get("APPDATA")
        else (home / ".config" / "devin")
    )
    if (ws / ".devin").exists() or devin_user.exists():
        detected.append("devin")
    if (
        (ws / ".clinerules").exists()
        or (ws / ".cline").exists()
        or (home / ".cline").exists()
        or (home / "Documents" / "Cline").exists()
        or bool(os.environ.get("CLINE_DIR"))
        or bool(os.environ.get("CLINE_HOOKS_DIR"))
    ):
        detected.append("cline")
    if (home / ".minimax").exists() or (ws / ".minimax").exists():
        detected.append("mcode")
    if (home / ".ouroboros").exists() or (ws / ".ouroboros").exists() or (ws / "ouroboros" / "tools").is_dir():
        detected.append("ouroboros")
    if (home / ".prime").exists() or (ws / ".prime").exists():
        detected.append("prime")
    return detected


def _build_generic_hook_bundle(
    host: str, *, profile: str, shadow: bool, auto_allow: bool, use_rust_shim: bool = False
) -> Dict[str, Any]:
    from .core.rust_shim import build_hook_command

    extra = ["--profile", profile, "--log-decisions", "--session-state", "auto"]
    if shadow:
        extra.append("--shadow")
    if auto_allow:
        extra.append("--auto-allow")
    pre_exe, pre_args = build_hook_command(host, event_args=[], extra_args=extra, use_rust_shim=use_rust_shim)
    post_extra = [*extra, "--replace-tool-output"] if host in {"claude", "qoder", "cursor"} else list(extra)
    post_exe, post_args = build_hook_command(
        host, event_args=["--event", "post-tool-use"], extra_args=post_extra, use_rust_shim=use_rust_shim
    )
    pre_cmd = " ".join([f'"{pre_exe}"', *pre_args])
    post_cmd = " ".join([f'"{post_exe}"', *post_args])
    if host == "cursor":
        _cursor_tool_matcher = (
            "Write|Edit|StrReplace|Delete|Read|Grep|Glob|LS|SemanticSearch|WebFetch|WebSearch|Task|CallDynamicTool|GetDynamicTools"
        )

        def _cursor_cmd(
            event_flag: str,
            cmd_extra: List[str],
            *,
            matcher: Optional[str] = None,
            fail_closed: bool = True,
        ) -> Dict[str, Any]:
            exe, args = build_hook_command(
                "cursor",
                event_args=["--event", event_flag],
                extra_args=cmd_extra,
                use_rust_shim=use_rust_shim,
            )
            entry: Dict[str, Any] = {
                "command": " ".join([f'"{exe}"', *args]),
                "timeout": 10,
                "failClosed": fail_closed,
            }
            if matcher:
                entry["matcher"] = matcher
            return entry

        return {
            "version": 1,
            "hooks": {
                "preToolUse": [_cursor_cmd("pre-tool-use", extra, matcher=_cursor_tool_matcher)],
                "postToolUse": [_cursor_cmd("post-tool-use", post_extra, matcher=_cursor_tool_matcher, fail_closed=False)],
                "beforeShellExecution": [_cursor_cmd("before-shell", extra)],
                "afterShellExecution": [_cursor_cmd("after-shell", post_extra, fail_closed=False)],
                "beforeMCPExecution": [_cursor_cmd("before-mcp", extra)],
                "afterMCPExecution": [_cursor_cmd("after-mcp", post_extra, fail_closed=False)],
                "beforeReadFile": [_cursor_cmd("before-read", extra)],
                "afterFileEdit": [_cursor_cmd("after-file-edit", post_extra, fail_closed=False)],
            },
        }
    if host == "devin":
        devin_matcher = (
            "^(exec|get_output|write_to_process|kill_shell|edit|write|apply_patch|"
            "notebook_edit|notebook_read|read|grep|glob|webfetch|run_subagent|"
            "todo_write|skill|exit_plan_mode|mcp__.*)$"
        )
        return {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": devin_matcher,
                        "hooks": [{"type": "command", "command": pre_cmd, "timeout": 10}],
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": devin_matcher,
                        "hooks": [{"type": "command", "command": post_cmd, "timeout": 10}],
                    }
                ],
            }
        }
    if host == "mcode":
        from .hosts.mcode import MCODE_TOOL_MATCHER

        mcode_post_exe, mcode_post_args = build_hook_command(
            "mcode", event_args=["--event", "post-tool-use"], extra_args=extra, use_rust_shim=use_rust_shim
        )
        mcode_post_cmd = " ".join([f'"{mcode_post_exe}"', *mcode_post_args])
        return {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": MCODE_TOOL_MATCHER,
                        "hooks": [
                            {
                                "type": "command",
                                "name": "Aetheris Microbrains gate",
                                "command": pre_cmd,
                                "commandWindows": pre_cmd,
                                "timeout": 10,
                            }
                        ],
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": MCODE_TOOL_MATCHER,
                        "hooks": [
                            {
                                "type": "command",
                                "name": "Aetheris Microbrains post-tool audit",
                                "command": mcode_post_cmd,
                                "commandWindows": mcode_post_cmd,
                                "timeout": 10,
                            }
                        ],
                    }
                ],
            }
        }
    pre_hook: Dict[str, Any] = {"type": "command", "command": pre_cmd}
    post_hook: Dict[str, Any] = {"type": "command", "command": post_cmd}
    if host in ("qoder", "qwen"):
        pre_hook["timeout"] = 10
        post_hook["timeout"] = 10
    else:
        pre_hook["onFailure"] = "block"
    pre_matcher = (
        "run_shell_command|read_file|write_file|edit|grep_search|glob|find_files|list_directory|web_fetch|web_search|"
        "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
        if host == "qwen"
        else "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
    )
    post_matcher = (
        pre_matcher
        if host == "qwen"
        else "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|WebFetch|WebSearch|mcp__.*"
    )
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": pre_matcher,
                    "hooks": [pre_hook],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": post_matcher,
                    "hooks": [post_hook],
                }
            ],
        }
    }


def _merge_hooks_non_destructively(existing_hooks: Any, new_hooks: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new Aetheris hook entries per-event without removing unrelated user hooks."""
    merged: Dict[str, Any] = dict(existing_hooks) if isinstance(existing_hooks, dict) else {}
    for event_name, new_entries in new_hooks.items():
        if not isinstance(new_entries, list):
            merged[event_name] = new_entries
            continue
        cur_list = merged.get(event_name)
        preserved: List[Any] = []
        if isinstance(cur_list, list):
            for entry in cur_list:
                try:
                    serialized = json.dumps(entry, ensure_ascii=False)
                except Exception:
                    serialized = str(entry)
                if "aetheris_microbrains.hook" not in serialized and "aetheris-gate" not in serialized:
                    preserved.append(entry)
        merged[event_name] = [*preserved, *new_entries]
    return merged


def _install_host(args: argparse.Namespace, host: str) -> int:
    ws = Path(args.workspace).expanduser().resolve() if args.workspace else None
    if host == "antigravity":
        from .install_antigravity import install as ag_install, self_test as ag_self_test

        pd = Path(args.plugin_dir) if getattr(args, "plugin_dir", None) else None
        if args.self_test:
            return ag_self_test(workspace=ws, plugin_dir=pd)
        return ag_install(
            shadow=bool(args.shadow),
            auto_allow=bool(args.auto_allow),
            auto_approve_escalations=bool(args.auto_approve_escalations),
            use_rust_shim=bool(args.use_rust_shim),
            profile=args.profile,
            workspace=ws,
            plugin_dir=pd,
            dry_run=bool(args.dry_run),
        )
    if host == "openclaw":
        from .install_openclaw import install_config as oc_install, self_test as oc_self_test

        if args.self_test:
            return oc_self_test()
        return oc_install(
            profile=str(args.profile or "dev"),
            shadow=bool(args.shadow),
            auto_approve_escalations=bool(args.auto_approve_escalations),
            dry_run=bool(args.dry_run),
        )
    if host == "dsh":
        from .install_dsh import _print_setup, self_test as dsh_self_test

        if args.self_test:
            return dsh_self_test()
        return _print_setup()
    if host == "zcode":
        from .install_zcode import install as zcode_install, self_test as zcode_self_test

        if args.self_test:
            return zcode_self_test()
        return zcode_install(
            workspace=ws,
            profile=str(args.profile or "strict"),
            shadow=bool(args.shadow),
            use_rust_shim=bool(args.use_rust_shim),
            dry_run=bool(args.dry_run),
        )
    if host == "openhands":
        from .install_openhands import install as oh_install, self_test as oh_self_test

        if args.self_test:
            return oh_self_test()
        return oh_install(
            workspace=ws,
            profile=str(args.profile or "dev"),
            shadow=bool(args.shadow),
            use_rust_shim=bool(args.use_rust_shim),
            dry_run=bool(args.dry_run),
        )

    if host == "cursor" and args.self_test:
        from .hook import evaluate_hook_payload

        sh_deny = evaluate_hook_payload(
            {"_cursor_event": "beforeShellExecution", "command": "rm -rf /", "cwd": "."},
            host="cursor",
            log_decisions=False,
        )
        sh_allow = evaluate_hook_payload(
            {"_cursor_event": "beforeShellExecution", "command": "git status", "cwd": "."},
            host="cursor",
            log_decisions=False,
        )
        rd_deny = evaluate_hook_payload(
            {"_cursor_event": "beforeReadFile", "file_path": ".env", "content": "SECRET=1"},
            host="cursor",
            log_decisions=False,
        )
        rd_allow = evaluate_hook_payload(
            {"_cursor_event": "beforeReadFile", "file_path": "src/app.py", "content": "x = 1"},
            host="cursor",
            log_decisions=False,
        )
        cfg_write = evaluate_hook_payload(
            {"_cursor_event": "preToolUse", "tool_name": "Write", "tool_input": {"file_path": ".cursor/hooks.json", "content": "{}"}},
            host="cursor",
            log_decisions=False,
        )
        ok = (
            sh_deny.get("exit_code") == 2
            and (sh_deny.get("hookSpecificOutput") or {}).get("permission") == "deny"
            and sh_allow.get("exit_code") == 0
            and (sh_allow.get("hookSpecificOutput") or {}).get("permission") == "allow"
            and rd_deny.get("exit_code") == 2
            and (rd_deny.get("hookSpecificOutput") or {}).get("permission") == "deny"
            and rd_allow.get("exit_code") == 0
            and cfg_write.get("exit_code") == 2
        )
        print("CURSOR SELF-TEST PASSED" if ok else "CURSOR SELF-TEST FAILED")
        return 0 if ok else 1

    if host == "devin" and args.self_test:
        from .hook import evaluate_hook_payload

        sh_deny = evaluate_hook_payload(
            {"hook_event_name": "PreToolUse", "tool_name": "exec", "tool_input": {"command": "rm -rf /"}},
            host="devin",
            log_decisions=False,
        )
        sh_allow = evaluate_hook_payload(
            {"hook_event_name": "PreToolUse", "tool_name": "exec", "tool_input": {"command": "git status"}},
            host="devin",
            log_decisions=False,
        )
        get_out = evaluate_hook_payload(
            {"hook_event_name": "PreToolUse", "tool_name": "get_output", "tool_input": {"shell_id": "1"}},
            host="devin",
            log_decisions=False,
        )
        cfg_write = evaluate_hook_payload(
            {"hook_event_name": "PreToolUse", "tool_name": "write", "tool_input": {"file_path": ".devin/hooks.v1.json", "content": "{}"}},
            host="devin",
            log_decisions=False,
        )
        ok = (
            sh_deny.get("exit_code") == 2
            and (sh_deny.get("hookSpecificOutput") or {}).get("decision") == "block"
            and sh_allow.get("exit_code") == 0
            and get_out.get("exit_code") == 0
            and get_out.get("decision") == "allow"
            and cfg_write.get("exit_code") == 2
            and (cfg_write.get("hookSpecificOutput") or {}).get("decision") == "block"
        )
        print("DEVIN SELF-TEST PASSED" if ok else "DEVIN SELF-TEST FAILED")
        return 0 if ok else 1

    if host == "cline" and args.self_test:
        from .hook import evaluate_hook_payload

        sh_deny = evaluate_hook_payload(
            {"preToolUse": {"toolName": "execute_command", "parameters": {"command": "rm -rf /"}}},
            host="cline",
            log_decisions=False,
        )
        sh_allow = evaluate_hook_payload(
            {"preToolUse": {"toolName": "execute_command", "parameters": {"command": "git status"}}},
            host="cline",
            log_decisions=False,
        )
        cfg_write = evaluate_hook_payload(
            {"preToolUse": {"toolName": "write_to_file", "parameters": {"path": ".clinerules/hooks/PreToolUse", "content": "exit 0"}}},
            host="cline",
            log_decisions=False,
        )
        comp = evaluate_hook_payload(
            {"preToolUse": {"toolName": "attempt_completion", "parameters": {"result": "done"}}},
            host="cline",
            log_decisions=False,
        )
        ok = (
            sh_deny.get("exit_code") == 2
            and (sh_deny.get("hookSpecificOutput") or {}).get("cancel") is True
            and sh_allow.get("exit_code") == 0
            and (sh_allow.get("hookSpecificOutput") or {}).get("cancel") is False
            and cfg_write.get("exit_code") == 2
            and (cfg_write.get("hookSpecificOutput") or {}).get("cancel") is True
            and comp.get("exit_code") == 0
            and (comp.get("hookSpecificOutput") or {}).get("cancel") is False
        )
        print("CLINE SELF-TEST PASSED" if ok else "CLINE SELF-TEST FAILED")
        return 0 if ok else 1

    if host == "mcode" and args.self_test:
        from .hook import evaluate_hook_payload

        sh_deny = evaluate_hook_payload(
            {"tool_name": "bash", "tool_input": {"command": "rm -rf /"}},
            host="mcode",
            log_decisions=False,
        )
        sh_allow = evaluate_hook_payload(
            {"tool_name": "bash", "tool_input": {"command": "git status"}},
            host="mcode",
            log_decisions=False,
        )
        todo_allow = evaluate_hook_payload(
            {"tool_name": "todowrite", "tool_input": {"todos": []}},
            host="mcode",
            log_decisions=False,
        )
        cfg_write = evaluate_hook_payload(
            {"tool_name": "write", "tool_input": {"file_path": ".minimax/hooks.json", "content": "{}"}},
            host="mcode",
            log_decisions=False,
        )
        ok = (
            sh_deny.get("exit_code") == 2
            and (sh_deny.get("hookSpecificOutput") or {}).get("permissionDecision") == "deny"
            and sh_allow.get("exit_code") == 0
            and todo_allow.get("exit_code") == 0
            and todo_allow.get("decision") == "allow"
            and cfg_write.get("exit_code") == 2
        )
        print("MCODE SELF-TEST PASSED" if ok else "MCODE SELF-TEST FAILED")
        return 0 if ok else 1

    if host == "ouroboros" and args.self_test:
        from .hook import evaluate_hook_payload
        from .ouroboros_gate import OuroborosMicrobrainsGate, wrap_ouroboros_registry

        sh_deny = evaluate_hook_payload(
            {"tool_name": "run_command", "tool_input": {"command": "rm -rf /"}},
            host="ouroboros",
            log_decisions=False,
        )
        sh_allow = evaluate_hook_payload(
            {"tool_name": "run_command", "tool_input": {"command": "git status"}},
            host="ouroboros",
            log_decisions=False,
        )
        patch_deny = evaluate_hook_payload(
            {"tool_name": "apply_patch", "tool_input": {"path": "app.py", "patch": "import os\nos.system('rm -rf /')"}},
            host="ouroboros",
            log_decisions=False,
        )
        read_ask = evaluate_hook_payload(
            {"tool_name": "read_file", "tool_input": {"path": ".env"}},
            host="ouroboros",
            log_decisions=False,
        )

        class _DummyOuroborosRegistry:
            def execute(self, name: str, tool_args: dict) -> str:
                return f"ran:{name}:{tool_args}"

        dummy = _DummyOuroborosRegistry()
        wrap_ouroboros_registry(dummy, gate=OuroborosMicrobrainsGate(profile="dev"))
        blocked_out = dummy.execute("run_command", {"command": "rm -rf /"})
        allowed_out = dummy.execute("run_command", {"command": "git status"})

        ok = (
            sh_deny.get("exit_code") == 2
            and sh_deny.get("host_contract") == "ouroboros_registry_v1"
            and sh_allow.get("exit_code") == 0
            and patch_deny.get("exit_code") == 2
            and read_ask.get("decision") == "escalate"
            and isinstance(blocked_out, str)
            and blocked_out.startswith("MICROBRAINS_DENY")
            and str(allowed_out).startswith("ran:run_command:")
        )
        print("OUROBOROS SELF-TEST PASSED" if ok else "OUROBOROS SELF-TEST FAILED")
        return 0 if ok else 1

    if host == "prime" and args.self_test:
        import types as _types

        from .hook import evaluate_hook_payload
        from .prime_gate import PrimeMicrobrainsBlockedError, install_prime_gate

        sh_deny = evaluate_hook_payload(
            {"tool_name": "bash", "tool_input": {"command": "rm -rf /"}},
            host="prime",
            log_decisions=False,
        )
        sh_allow = evaluate_hook_payload(
            {"tool_name": "bash", "tool_input": {"command": "git status"}},
            host="prime",
            log_decisions=False,
        )
        cfg_write = evaluate_hook_payload(
            {"tool_name": "write", "tool_input": {"file_path": ".prime/agent/settings.json", "content": "{}"}},
            host="prime",
            log_decisions=False,
        )

        _rlm = _types.SimpleNamespace(bash=lambda command, persistent=False: f"ok:{command}")
        install_prime_gate(rlm_mod=_rlm, profile="dev")
        blocked_raised = False
        try:
            _rlm.bash("rm -rf /")
        except PrimeMicrobrainsBlockedError:
            blocked_raised = True

        ok = (
            sh_deny.get("exit_code") == 2
            and sh_deny.get("host_contract") == "prime_pretooluse_v1"
            and sh_allow.get("exit_code") == 0
            and cfg_write.get("decision") == "deny"
            and blocked_raised
            and _rlm.bash("git status") == "ok:git status"
        )
        print("PRIME SELF-TEST PASSED" if ok else "PRIME SELF-TEST FAILED")
        return 0 if ok else 1

    if host == "qoder" and args.self_test:
        from .hook import evaluate_hook_payload

        deny_v = evaluate_hook_payload(
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
            host="qoder",
            log_decisions=False,
        )
        allow_v = evaluate_hook_payload(
            {"tool_name": "Bash", "tool_input": {"command": "git status"}},
            host="qoder",
            log_decisions=False,
        )
        qoder_write = evaluate_hook_payload(
            {"tool_name": "Write", "tool_input": {"file_path": ".qoder/settings.json", "content": "{}"}},
            host="qoder",
            log_decisions=False,
        )
        ok = (
            deny_v.get("decision") == "deny"
            and deny_v.get("exit_code") == 2
            and allow_v.get("decision") == "allow"
            and qoder_write.get("decision") == "deny"
        )
        print("QODER SELF-TEST PASSED" if ok else "QODER SELF-TEST FAILED")
        return 0 if ok else 1

    if host == "qwen" and args.self_test:
        from .hook import evaluate_hook_payload

        deny_v = evaluate_hook_payload(
            {"tool_name": "run_shell_command", "tool_input": {"command": "rm -rf /"}},
            host="qwen",
            log_decisions=False,
        )
        allow_v = evaluate_hook_payload(
            {"tool_name": "run_shell_command", "tool_input": {"command": "git status"}},
            host="qwen",
            log_decisions=False,
        )
        qwen_write = evaluate_hook_payload(
            {"tool_name": "write_file", "tool_input": {"file_path": ".qwen/settings.json", "content": "{}"}},
            host="qwen",
            log_decisions=False,
        )
        ok = (
            deny_v.get("decision") == "deny"
            and deny_v.get("exit_code") == 2
            and deny_v.get("host_contract") == "qwen_pretooluse_v1"
            and allow_v.get("decision") == "allow"
            and qwen_write.get("decision") == "deny"
        )
        print("QWEN SELF-TEST PASSED" if ok else "QWEN SELF-TEST FAILED")
        return 0 if ok else 1

    _ESM_PLUGIN_HOSTS = {
        "opencode": (
            "opencode-aetheris-microbrains",
            "aetheris-microbrains.js",
            "opencode.json",
            (".opencode", "plugins", "aetheris-microbrains.js"),
            (".config", "opencode", "plugins", "aetheris-microbrains.js"),
            "OpenCode",
        ),
        "kilocode": (
            "kilocode-aetheris-microbrains",
            "aetheris-microbrains.js",
            "kilo.json",
            (".kilo", "plugins", "aetheris-microbrains.js"),
            (".config", "kilo", "plugins", "aetheris-microbrains.js"),
            "Kilo Code",
        ),
        "pi": (
            "pi-aetheris-microbrains",
            "index.js",
            ".pi/settings.json",
            (".pi", "extensions", "aetheris-microbrains", "index.js"),
            (".pi", "agent", "extensions", "aetheris-microbrains", "index.js"),
            "Pi",
        ),
        "omp": (
            "omp-aetheris-microbrains",
            "index.js",
            ".omp/settings.json",
            (".omp", "extensions", "aetheris-microbrains", "index.js"),
            (".omp", "agent", "extensions", "aetheris-microbrains", "index.js"),
            "OMP",
        ),
    }
    if host in _ESM_PLUGIN_HOSTS:
        bundle_dir, src_name, probe_cfg, ws_rel_parts, global_rel_parts, display_name = _ESM_PLUGIN_HOSTS[host]
        src_plugin = Path(__file__).resolve().parents[1] / bundle_dir / src_name
        if args.self_test:
            from .hook import evaluate_hook_payload

            ok_file = src_plugin.is_file()
            if ok_file:
                low = src_plugin.read_text(encoding="utf-8", errors="ignore").lower()
                if "i:\\" in low or "i:/" in low or "c:\\users\\" in low or "c:/users/" in low:
                    ok_file = False
            deny_v = evaluate_hook_payload(
                {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
                host=host,
                log_decisions=False,
            )
            allow_v = evaluate_hook_payload(
                {"tool_name": "Bash", "tool_input": {"command": "git status"}},
                host=host,
                log_decisions=False,
            )
            cfg_write = evaluate_hook_payload(
                {"tool_name": "Write", "tool_input": {"file_path": probe_cfg, "content": "{}"}},
                host=host,
                log_decisions=False,
            )
            ok = (
                ok_file
                and deny_v.get("decision") == "deny"
                and (deny_v.get("hookSpecificOutput") or {}).get("block") is True
                and allow_v.get("decision") == "allow"
                and cfg_write.get("decision") == "deny"
            )
            print(f"{host.upper()} SELF-TEST PASSED" if ok else f"{host.upper()} SELF-TEST FAILED")
            return 0 if ok else 1

        if not src_plugin.is_file():
            print(f"{display_name} plugin bundle missing: {src_plugin}")
            return 1
        if getattr(args, "plugin_dir", None):
            target = Path(args.plugin_dir) / src_name
        elif ws is not None:
            target = ws / Path(*ws_rel_parts)
        else:
            target = Path.home() / Path(*global_rel_parts)
        if args.dry_run:
            print(f"DRY RUN - would install {display_name} plugin -> {target}")
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".aetheris-microbrains.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(src_plugin.read_text(encoding="utf-8"))
                fh.flush()
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        print(f"Installed {display_name} plugin -> {target}")
        print(f"Start the warm HTTP daemon before launching {display_name}:")
        print(f"  aetheris-daemon --http --host {host} --port 8765")
        return 0

    from .core.rust_shim import find_aetheris_gate

    no_rust = bool(getattr(args, "no_rust_shim", False))
    effective_rust_shim = False if no_rust else (
        bool(args.use_rust_shim) or (host in {"cursor", "devin", "cline", "mcode"} and bool(find_aetheris_gate()))
    )
    if host == "cline":
        from .core.rust_shim import build_hook_command

        prof = str(args.profile or "dev")
        extra = ["--profile", prof, "--log-decisions", "--session-state", "auto"]
        if args.shadow:
            extra.append("--shadow")
        if args.auto_allow:
            extra.append("--auto-allow")
        pre_exe, pre_args = build_hook_command(
            "cline", event_args=["--event", "pre-tool-use"], extra_args=extra, use_rust_shim=effective_rust_shim
        )
        post_exe, post_args = build_hook_command(
            "cline", event_args=["--event", "post-tool-use"], extra_args=extra, use_rust_shim=effective_rust_shim
        )
        env_hooks_dir = str(os.environ.get("CLINE_HOOKS_DIR") or "").strip()
        env_cline_dir = str(os.environ.get("CLINE_DIR") or "").strip()
        cline_target = str(getattr(args, "target", "both") or "both").strip().lower()
        cline_cli_dir = (
            Path(env_hooks_dir).expanduser()
            if env_hooks_dir
            else ((Path(env_cline_dir).expanduser() if env_cline_dir else (Path.home() / ".cline")) / "hooks")
        )
        vscode_dir = (
            (ws / ".clinerules" / "hooks")
            if ws is not None
            else (Path.home() / "Documents" / "Cline" / "Rules" / "Hooks")
        )
        if getattr(args, "plugin_dir", None):
            hooks_dirs = [Path(args.plugin_dir)]
        elif cline_target == "cli":
            hooks_dirs = [cline_cli_dir]
        elif cline_target == "vscode":
            hooks_dirs = [vscode_dir]
        else:
            if ws is not None:
                hooks_dirs = [vscode_dir]
            elif env_hooks_dir:
                hooks_dirs = [cline_cli_dir]
            else:
                hooks_dirs = [vscode_dir] if cline_cli_dir == vscode_dir else [vscode_dir, cline_cli_dir]

        pre_args_str = " ".join(pre_args)
        post_args_str = " ".join(post_args)
        files_to_write = {
            "PreToolUse": f'#!/usr/bin/env bash\nexec "{pre_exe}" {pre_args_str}\n',
            "PostToolUse": f'#!/usr/bin/env bash\nexec "{post_exe}" {post_args_str}\n',
            "PreToolUse.ps1": f'$inputJson = [Console]::In.ReadToEnd()\n$inputJson | & "{pre_exe}" {pre_args_str}\nexit $LASTEXITCODE\n',
            "PostToolUse.ps1": f'$inputJson = [Console]::In.ReadToEnd()\n$inputJson | & "{post_exe}" {post_args_str}\nexit $LASTEXITCODE\n',
        }
        if args.dry_run:
            for hdir in hooks_dirs:
                print(f"DRY RUN - would write cline hook scripts to {hdir}:")
                for fname, body in files_to_write.items():
                    print(f"--- {hdir / fname} ---")
                    print(body.rstrip())
            return 0
        for hdir in hooks_dirs:
            hdir.mkdir(parents=True, exist_ok=True)
            for fname, body in files_to_write.items():
                fpath = hdir / fname
                fpath.write_text(body, encoding="utf-8", newline="\n")
                if not fname.endswith(".ps1"):
                    try:
                        os.chmod(fpath, 0o755)
                    except OSError:
                        pass
            print(f"Installed Aetheris Microbrains (cline) -> {hdir} [PreToolUse, PostToolUse (+ .ps1)]")
        cli_hint_dir = cline_cli_dir if cline_target == "cli" else hooks_dirs[0]
        print(
            f'Note: For Cline CLI, run `cline config set hooksEnabled=true` and pass `--hooks-dir "{cli_hint_dir}"` '
            "(or install to `~/.cline/hooks/` via `aetheris install cline --target cli`)."
        )
        return 0
    if host == "ouroboros":
        prof = str(args.profile or "dev")
        snippet = (
            "from aetheris_microbrains.ouroboros_gate import wrap_ouroboros_registry\n"
            f'wrap_ouroboros_registry(registry, profile="{prof}")'
        )
        if args.dry_run:
            print("DRY RUN - Ouroboros integrates in-process on ToolRegistry (no hook config file needed):")
            print(snippet)
            return 0
        print("Ouroboros integrates directly in Python on ouroboros.tools.registry.ToolRegistry:")
        print(snippet)
        print("Verify anytime with: aetheris install ouroboros --self-test")
        return 0
    if host == "prime":
        prof = str(args.profile or "dev")
        snippet = (
            "from aetheris_microbrains.prime_gate import install_prime_gate\n"
            f'install_prime_gate(profile="{prof}")'
        )
        if args.dry_run:
            print("DRY RUN - Prime Agent integrates in-process on REPL callables (rlm.bash, edit.run, mcp.call_tool):")
            print(snippet)
            return 0
        print("Prime Agent integrates directly in Python on REPL callables (rlm.bash, edit.run, mcp.call_tool):")
        print(snippet)
        print("Verify anytime with: aetheris install prime --self-test")
        return 0
    bundle = _build_generic_hook_bundle(
        host,
        profile=str(args.profile or "dev"),
        shadow=bool(args.shadow),
        auto_allow=bool(args.auto_allow),
        use_rust_shim=effective_rust_shim,
    )
    base_dir = ws if ws is not None else Path.home()
    devin_user_rel = (
        Path(os.environ["APPDATA"]) / "devin" / "config.json"
        if os.name == "nt" and os.environ.get("APPDATA")
        else Path.home() / ".config" / "devin" / "config.json"
    )
    rel_map = {
        "claude": Path(".claude") / "settings.local.json",
        "codex": Path(".codex") / "hooks.json",
        "cursor": Path(".cursor") / "hooks.json",
        "qoder": Path(".qoder") / ("settings.local.json" if ws is not None else "settings.json"),
        "qwen": Path(".qwen") / ("settings.local.json" if ws is not None else "settings.json"),
    }
    if host == "hermes":
        # Hermes reads hooks from config.yaml (`agent/shell_hooks.py`), not from a
        # host-specific JSON file — writing ~/.hermes/aetheris_hooks.json registered nothing.
        from .hosts.hermes_config import install_hermes_hooks

        custom_cfg = Path(args.plugin_dir) if getattr(args, "plugin_dir", None) else None
        target, merged = install_hermes_hooks(bundle, config_path=custom_cfg, dry_run=bool(args.dry_run))
        if args.dry_run:
            print(f"DRY RUN - would merge hermes hooks into {target}:")
            print(json.dumps(merged.get("hooks", {}), indent=2, ensure_ascii=False))
            return 0
        events = ", ".join(sorted(merged.get("hooks", {})))
        print(f"Installed Aetheris Microbrains (hermes) -> {target} [{events}]")
        print("Restart the gateway or start a new agent process so the hook registers.")
        return 0
    if getattr(args, "plugin_dir", None):
        target = Path(args.plugin_dir)
    elif host == "devin":
        target = (ws / ".devin" / "hooks.v1.json") if ws is not None else devin_user_rel
    elif host == "mcode":
        minimax_data = str(os.environ.get("MINIMAX_DATA_DIR") or "").strip()
        mcode_user = (Path(minimax_data) / "hooks.json") if minimax_data else (Path.home() / ".minimax" / "hooks.json")
        target = (ws / ".minimax" / "hooks.json") if ws is not None else mcode_user
    else:
        target = base_dir / rel_map[host]
    merged_bundle = dict(bundle)
    if host == "mcode" and ws is None and not getattr(args, "plugin_dir", None):
        merged_bundle["enabled"] = True
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged = dict(existing)
                merged["hooks"] = _merge_hooks_non_destructively(existing.get("hooks"), bundle.get("hooks", {}))
                if "version" in bundle:
                    merged["version"] = bundle["version"]
                merged_bundle = merged
        except Exception:
            pass
    if args.dry_run:
        print(f"DRY RUN - would write {host} hooks config to {target}:")
        print(json.dumps(merged_bundle, indent=2, ensure_ascii=False))
        return 0
    _atomic_write_json(target, merged_bundle)
    print(f"Installed Aetheris Microbrains ({host}) -> {target}")
    if host == "devin":
        for claude_cfg in (base_dir / ".claude" / "settings.local.json", base_dir / ".claude" / "settings.json"):
            if claude_cfg.is_file():
                try:
                    raw_claude = claude_cfg.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    raw_claude = ""
                if "aetheris_microbrains" in raw_claude or "aetheris-gate" in raw_claude:
                    print(
                        f"Warning: {claude_cfg} also defines Aetheris hooks; Devin auto-loads .claude/ hooks, "
                        "which can cause double evaluation unless consolidated with `--host auto`."
                    )
                    break
    if host == "mcode":
        print(
            "Note: For project-level .minimax/hooks.json, enable project hooks in Settings -> Hooks -> Project; "
            "hook changes take effect on the next turn."
        )
    if host == "cursor":
        if effective_rust_shim:
            print(
                "Tip: warm `aetheris-gate --auto-daemon` shim was auto-selected (override with `--no-rust-shim`); "
                "use `--profile guardrails` for minimal friction and `--shadow` while verifying live Cursor hooks."
            )
        else:
            print(
                "Tip: use `--profile guardrails` for minimal friction, and build/install `aetheris-gate` "
                "(or pass `--use-rust-shim`) to avoid Python cold-start latency on Cursor hooks."
            )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aetheris",
        description="Unified Aetheris Microbrains CLI (`uvx aetheris-microbrains`).",
    )
    sub = parser.add_subparsers(dest="command")

    p_inst = sub.add_parser("install", help="Install or verify hooks/bundles for an AI host")
    p_inst.add_argument("host", choices=list(SUPPORTED_HOSTS), help="Target AI host or 'auto'")
    p_inst.add_argument("--profile", choices=["strict", "dev", "guardrails"], default="dev", help="Policy profile")
    p_inst.add_argument("--shadow", action="store_true", help="Observe-only shadow mode")
    p_inst.add_argument("--auto-allow", action="store_true", help="Emit explicit allow decisions")
    p_inst.add_argument("--auto-approve-escalations", action="store_true", help="Auto-approve non-credential escalations")
    p_inst.add_argument("--use-rust-shim", action="store_true", help="Use native Rust `aetheris-gate` shim when available")
    p_inst.add_argument("--no-rust-shim", action="store_true", help="Force Python hook invocation even when `aetheris-gate` is available")
    p_inst.add_argument("--workspace", default=None, help="Target workspace root")
    p_inst.add_argument("--plugin-dir", default=None, help="Override target plugin/config directory")
    p_inst.add_argument(
        "--target",
        choices=["cli", "vscode", "both"],
        default="both",
        help="Install target surface for hosts with separate CLI/IDE hook paths (e.g. cline)",
    )
    p_inst.add_argument("--dry-run", action="store_true", help="Print configuration without writing files")
    p_inst.add_argument("--self-test", action="store_true", help="Run host integration self-test")

    p_vc = sub.add_parser("verify-core", help="Verify stdlib-only core isolation vs lab research modules")
    p_vc.add_argument("--json", action="store_true", help="Output verification report as JSON")

    p_fv = sub.add_parser("fleet-verify", help="Verify a signed fleet policy bundle and optional journal hash chain")
    p_fv.add_argument("--bundle", required=True, help="Path to signed fleet bundle JSON")
    p_fv.add_argument("--key-env", default="AETHERIS_FLEET_SIGNING_KEY", help="Env var holding HMAC signing key")
    p_fv.add_argument("--min-version", type=int, default=0, help="Minimum accepted monotonic bundle version")
    p_fv.add_argument("--journal-path", default="", help="Optional decision_journal.jsonl path for attestation")
    p_fv.add_argument("--json", action="store_true", help="Output verification/attestation report as JSON")

    p_bm = sub.add_parser("benchmark", help="Run cross-harness A/B security and Engineering Efficiency suite")
    p_bm.add_argument("--hosts", default="claude,cursor,cline,qwen,mcode,ouroboros,prime", help="Comma-separated hosts")
    p_bm.add_argument("--trials", type=int, default=3, help="Trials per host")
    p_bm.add_argument("--profile", choices=["strict", "dev", "guardrails"], default="dev", help="Policy profile")
    p_bm.add_argument("--json", action="store_true", help="Output full benchmark report as JSON")

    p_rp = sub.add_parser("replay", help="Replay decision_journal.jsonl counterfactually and optionally synthesize policy")
    p_rp.add_argument("--journal", required=True, help="Path to decision_journal.jsonl")
    p_rp.add_argument("--profile", choices=["strict", "dev", "guardrails"], default="strict", help="Target profile")
    p_rp.add_argument("--host", default="", help="Optional host override for replay")
    p_rp.add_argument("--synthesize-policy", action="store_true", help="Synthesize recurring safe allowlist bundle")
    p_rp.add_argument("--org-id", default="local-org", help="Organization ID for synthesized policy bundle")
    p_rp.add_argument("--key-env", default="AETHERIS_FLEET_SIGNING_KEY", help="Optional HMAC signing key env var")
    p_rp.add_argument("--json", action="store_true", help="Output replay report as JSON")

    p_rr = sub.add_parser("robo-replay", help="Verify and counterfactually replay a JSONL robot flight log")
    p_rr.add_argument("--log", required=True, help="Path to JSONL flight recorder dump")
    p_rr.add_argument("--max-vel", type=float, default=None, help="Optional counterfactual max linear velocity (m/s)")
    p_rr.add_argument("--max-ke", type=float, default=None, help="Optional counterfactual max kinetic energy (J)")
    p_rr.add_argument("--max-force", type=float, default=None, help="Optional counterfactual max contact force (N)")
    p_rr.add_argument("--conformal-margin", type=float, default=0.0, help="Conformal margin to apply during replay")
    p_rr.add_argument("--synthesize-safety-patch", action="store_true", help="Synthesize tightened PhysicalSafetyIR rules and FleetPolicyBundle from incident frames")
    p_rr.add_argument("--org-id", default="local-org", help="Organization ID for synthesized safety patch bundle")
    p_rr.add_argument("--key-env", default="AETHERIS_FLEET_SIGNING_KEY", help="Optional HMAC signing key env var")
    p_rr.add_argument("--json", action="store_true", help="Output flight replay report as JSON")

    p_rb = sub.add_parser("robo-bench", help="Run 6-scenario closed-loop embodied robotics safety benchmark suite")
    p_rb.add_argument("--native-rust", action="store_true", help="Dispatch CBF QP projections to NativeRTGateLib ctypes backend when built")
    p_rb.add_argument("--json", action="store_true", help="Output full robotics benchmark report as JSON")

    p_ud = sub.add_parser("unitree-doctor", help="Inspect Unitree hardware profile limits, CycloneDDS readiness, and safety bridge health")
    p_ud.add_argument("--robot", default="g1_29dof", help="Unitree robot profile (go2, b2, z1, h1, h1_2, g1_23dof, g1_29dof, dex3_1)")
    p_ud.add_argument("--json", action="store_true", help="Output diagnostic report as JSON")

    p_ir = sub.add_parser("policy-ir-check", help="Verify single-source policy_ir.json parity across Python and Rust")
    p_ir.add_argument("--json", action="store_true", help="Output parity report as JSON")

    for passthrough in ("audit", "doctor", "mcp-proxy", "pr-gate"):
        sub.add_parser(passthrough, add_help=False, help=f"Forward arguments to aetheris_microbrains.{passthrough.replace('-', '_')}")

    known, unknown = parser.parse_known_args(argv)
    if known.command == "verify-core":
        from .core.packaging_manifest import verify_core_isolation

        rep = verify_core_isolation()
        if known.json:
            print(json.dumps(rep, indent=2))
        else:
            print(
                f"core_isolated={rep['ok']} | core_modules={rep['core_count']} | "
                f"lab_modules={rep['lab_count']} | numpy_violations={rep['numpy_violations']}"
            )
        return 0 if rep["ok"] else 1

    if known.command == "fleet-verify":
        from .core.fleet_bundle import build_fleet_attestation, verify_signed_fleet_bundle

        key = str(os.environ.get(known.key_env) or "").strip()
        if not key:
            print(f"Missing signing key in env var {known.key_env}", file=sys.stderr)
            return 1
        bundle_obj = json.loads(Path(known.bundle).read_text(encoding="utf-8"))
        if known.journal_path:
            rep = build_fleet_attestation(bundle_obj, journal_path=known.journal_path, signing_key=key)
            ok = bool(rep.get("valid"))
        else:
            rep = verify_signed_fleet_bundle(bundle_obj, signing_key=key, min_version=known.min_version)
            ok = bool(rep.get("valid"))
        if known.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            print(f"fleet_valid={ok} | org={rep.get('org_id')} | version={rep.get('version')} | reason={rep.get('reason')}")
        return 0 if ok else 2

    if known.command == "benchmark":
        from .core.benchmark_harness import run_cross_harness_ab_suite

        host_list = [h.strip() for h in str(known.hosts or "").split(",") if h.strip()]
        rep = run_cross_harness_ab_suite(hosts=host_list, trials=max(1, int(known.trials)), profile=known.profile)
        if known.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            agg = rep.get("aggregate", {})
            print(
                f"hosts={len(host_list)} | trials={known.trials} | tpr={agg.get('tpr')} | fpr={agg.get('fpr')} "
                f"| e_ratio={agg.get('e_ratio')} | p95_ms={agg.get('latency_p95_ms')}"
            )
        return 0

    if known.command == "replay":
        from .core.trace_replay import replay_decision_journal

        key_raw = str(os.environ.get(known.key_env) or "").strip()
        rep = replay_decision_journal(
            known.journal,
            target_profile=known.profile,
            host=known.host or None,
            synthesize_policy=bool(known.synthesize_policy),
            org_id=known.org_id,
            hmac_key=key_raw.encode("utf-8") if key_raw else None,
        )
        if known.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            print(
                f"replayed={rep.get('replayed_entries')}/{rep.get('total_entries')} | "
                f"flips={len(rep.get('decision_flips') or [])} | newly_blocked={rep.get('newly_blocked')} | "
                f"p95_ms={rep.get('latency_p95_ms')}"
            )
        return 0

    if known.command == "robo-replay":
        if bool(getattr(known, "synthesize_safety_patch", False)):
            from .core.robotics_policy_synthesizer import synthesize_safety_patch_from_flight_log

            key_raw = str(os.environ.get(known.key_env) or "").strip()
            margin = float(known.conformal_margin) if float(known.conformal_margin) > 0.0 else 0.05
            rep = synthesize_safety_patch_from_flight_log(
                known.log,
                org_id=str(known.org_id),
                signing_key=key_raw.encode("utf-8") if key_raw else None,
                extra_safety_margin=margin,
            )
            if known.json:
                print(json.dumps(rep, indent=2, ensure_ascii=False))
            else:
                cf = rep.get("counterfactual_verification") or {}
                print(
                    f"verified_integrity={rep.get('verified_integrity')} | "
                    f"incidents={rep.get('incident_frame_count')}/{rep.get('total_frame_count')} | "
                    f"rules={len(rep.get('synthesized_rules') or [])} | "
                    f"before_intervene={cf.get('before_would_intervene')} | "
                    f"after_intervene={cf.get('after_would_intervene')}"
                )
            return 0 if rep.get("verified_integrity") else 2

        from .core.flight_recorder import replay_flight_log

        hard_limits: Dict[str, Any] = {}
        if known.max_vel is not None:
            hard_limits["max_linear_vel_mps"] = known.max_vel
        if known.max_ke is not None:
            hard_limits["max_kinetic_energy_j"] = known.max_ke
        if known.max_force is not None:
            hard_limits["max_contact_force_n"] = known.max_force
        rep = replay_flight_log(
            known.log,
            hard_limits=hard_limits or None,
            conformal_margin=float(known.conformal_margin),
        )
        if known.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            print(
                f"verified_integrity={rep.get('verified_integrity')} | "
                f"frames={rep.get('total_frames')} | "
                f"would_intervene={rep.get('would_intervene')} | "
                f"violations={len(rep.get('violations') or [])}"
            )
        return 0 if rep.get("verified_integrity") else 2

    if known.command == "robo-bench":
        from .core.robotics_benchmark_harness import run_robotics_benchmark_suite

        rep = run_robotics_benchmark_suite(use_native_rust=bool(known.native_rust))
        if known.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            s = rep.get("summary") or {}
            print(
                f"scenarios={s.get('passed_scenarios')}/{s.get('total_scenarios')} | "
                f"violations={s.get('total_violations')}/{s.get('total_steps')} ({s.get('overall_violation_rate_pct')}%) | "
                f"interventions={s.get('total_interventions')} | backend={s.get('backend')}"
            )
        return 0 if (rep.get("summary") or {}).get("all_passed") else 2

    if known.command == "unitree-doctor":
        import importlib.util

        from .core.unitree_safety_bridge import UNITREE_ROBOT_SPECS, SafeUnitreeLowLevelBridge

        r_key = str(known.robot).strip().lower()
        if r_key not in UNITREE_ROBOT_SPECS:
            print(f"Unknown Unitree robot profile: {known.robot}", file=sys.stderr)
            return 2
        spec = UNITREE_ROBOT_SPECS[r_key]
        bridge = SafeUnitreeLowLevelBridge(r_key)
        sdk_installed = importlib.util.find_spec("unitree_sdk2_python") is not None
        cyclonedds_home = str(os.environ.get("CYCLONEDDS_HOME") or "").strip()
        rep = {
            "ok": True,
            "robot": spec.name,
            "n_dof": spec.n_dof,
            "n_motor_slots": spec.total_motor_slots,
            "has_motor_slot_remap": spec.motor_slot_map is not None,
            "idl_family": spec.idl_family,
            "pr_ab_supported": spec.pr_ab_supported,
            "unitree_sdk2_python_installed": sdk_installed,
            "cyclonedds_home": cyclonedds_home or None,
            "initial_lifecycle_state": bridge.lifecycle_state,
            "recommended_subnet": "192.168.123.x",
        }
        if known.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            print(
                f"unitree_ok=True | robot={spec.name} | dof={spec.n_dof} | idl={spec.idl_family} | "
                f"sdk2={sdk_installed} | lifecycle={bridge.lifecycle_state}"
            )
        return 0

    if known.command == "policy-ir-check":
        from .core.policy_ir import verify_policy_ir_parity

        rep = verify_policy_ir_parity()
        if known.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            print(f"policy_ir_parity={rep.get('ok')} | missing_in_ir={rep.get('missing_in_ir')} | missing_in_rust={rep.get('missing_in_rust')}")
        return 0 if rep.get("ok") else 2

    if known.command == "install":
        if known.host == "auto":
            hosts = detect_installed_hosts(Path(known.workspace) if known.workspace else None)
            if not hosts:
                hosts = ["antigravity"]
            rc = 0
            for h in hosts:
                rc = max(rc, _install_host(known, h))
            return rc
        return _install_host(known, known.host)

    if known.command == "audit":
        from .audit import main as sub_main

        return sub_main(unknown)
    if known.command == "doctor":
        from .doctor import main as sub_main

        return sub_main(unknown)
    if known.command == "mcp-proxy":
        from .mcp_proxy import main as sub_main

        return sub_main(unknown)
    if known.command == "pr-gate":
        from .pr_gate import main as sub_main

        return sub_main(unknown)

    parser.print_help()
    return 0


if __name__ == "__main__":
    if not __package__:
        import importlib

        _pkg_root = str(Path(__file__).resolve().parent.parent)
        if _pkg_root not in sys.path:
            sys.path.insert(0, _pkg_root)
        raise SystemExit(importlib.import_module("aetheris_microbrains.cli").main())
    raise SystemExit(main())
