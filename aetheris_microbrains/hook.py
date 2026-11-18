"""Drop-in PreToolUse Hook CLI & Adapter for Claude Code, Codex, and custom harnesses.

Reads a JSON tool-call payload from stdin (or via `evaluate_hook_payload()`), runs
Aetheris Microbrains' 4-layer AST / secret / allowlist-first shlex gate, and outputs
a structured hook decision (`allow`, `deny`, or `ask` on escalate).

Usage (e.g. in Claude Code / Codex PreToolUse hook config):
    python -m aetheris_microbrains.hook
    python -m aetheris_microbrains.hook --strict-ast
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional

_HOOK_REDACT_PATTERNS = (
    (re.compile(r"sk-proj-[A-Za-z0-9_\-]{12,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
)


def _redact_error_text(text: str) -> str:
    out = str(text or "")
    for pat, repl in _HOOK_REDACT_PATTERNS:
        out = pat.sub(repl, out)
    return out


def _load_decision_engine():
    from .core.layered_decision_engine import PolicyConfig, decide
    return PolicyConfig, decide


def _normalize_hook_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Claude Code / Codex / generic PreToolUse JSON into `decide()` input."""
    tool_raw = str(
        payload.get("tool_name")
        or payload.get("tool")
        or payload.get("name")
        or ""
    ).strip()
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else payload

    tool_lower = tool_raw.lower()
    if tool_lower in {"bash", "shell", "execute_shell", "run_command", "terminal"}:
        tool_name = "execute_shell"
    elif tool_lower in {"edit", "str_replace_editor", "edit_file", "multiedit"}:
        tool_name = "edit_file"
    elif tool_lower in {"write", "write_file", "repo_write"}:
        tool_name = "write_file"
    elif tool_lower in {"read", "read_file", "repo_read", "view_file"}:
        tool_name = "read_file"
    elif tool_lower in {"grep", "grep_search", "search_code"}:
        tool_name = "search_code"
    elif tool_lower in {"glob", "list_dir"}:
        tool_name = "list_dir"
    else:
        tool_name = tool_raw

    normalized: Dict[str, Any] = {"tool": tool_name}
    if "command" in tool_input or "cmd" in tool_input:
        normalized["cmd"] = tool_input.get("command") or tool_input.get("cmd")
    if "file_path" in tool_input or "path" in tool_input:
        normalized["path"] = tool_input.get("file_path") or tool_input.get("path")

    # Build diff text from edit/write payloads (new_string, content, diff)
    diff = tool_input.get("diff") or tool_input.get("payload_diff")
    if diff is None and "new_string" in tool_input:
        diff = str(tool_input.get("new_string") or "")
    elif diff is None and "content" in tool_input and tool_name in {"write_file", "edit_file"}:
        diff = str(tool_input.get("content") or "")
    if diff is not None:
        normalized["diff"] = diff

    if "query" in tool_input:
        normalized["query"] = tool_input["query"]
    return normalized


def evaluate_hook_payload(
    payload: Dict[str, Any],
    strict_ast: bool = False,
    file_content: Optional[str] = None,
    log_decisions: bool = False,
    shadow_mode: bool = False,
    host: str = "claude",
    codex_escalate_deny: bool = False,
) -> Dict[str, Any]:
    """Evaluate a PreToolUse hook payload and return host-adapted verdict + exit_code."""
    PolicyConfig, decide = _load_decision_engine()
    resolved_host = str(host or "claude").strip().lower()
    if resolved_host not in {"claude", "codex", "generic"}:
        resolved_host = "claude"

    norm = _normalize_hook_payload(payload)
    policy = PolicyConfig(strict_ast_allowlist=strict_ast)
    ctx: Dict[str, Any] = {}
    if file_content is not None:
        ctx["file_content"] = file_content
    if log_decisions or shadow_mode:
        ctx["log_decisions"] = True
    if shadow_mode:
        ctx["shadow_mode"] = True
    verdict = decide(norm, context=ctx or None, policy=policy)
    dec = verdict["decision"]
    reason = _redact_error_text(f"{verdict['reason_code']}: {verdict['label']}")
    host_contract = {
        "claude": "claude_pretooluse_v1",
        "codex": "codex_pretooluse_no_ask_v1",
        "generic": "generic_v1",
    }[resolved_host]

    perm: Optional[str]
    if shadow_mode:
        # Strict passive shadow mode: never block or inject host allow/deny regardless of host
        perm = None
        exit_code = 0
        host_action = "shadow_observe"
        hook_specific = None
    elif resolved_host == "codex":
        if dec == "deny" or (dec == "escalate" and codex_escalate_deny):
            perm = "deny"
            exit_code = 2
            host_action = "deny"
            hook_specific = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        elif dec == "allow":
            perm = "allow"
            exit_code = 0
            host_action = "allow"
            hook_specific = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": reason,
            }
        else:
            # Codex PreToolUse does not accept permissionDecision="ask"; defer silently to host flow
            perm = None
            exit_code = 0
            host_action = "defer_to_host_approval"
            hook_specific = {"hookEventName": "PreToolUse"}
    elif resolved_host == "generic":
        perm = "allow" if dec == "allow" else ("deny" if dec == "deny" else "ask")
        exit_code = 2 if dec == "deny" else 0
        host_action = dec
        hook_specific = None
    else:
        perm = "allow" if dec == "allow" else ("deny" if dec == "deny" else "ask")
        exit_code = 2 if dec == "deny" else 0
        host_action = perm
        hook_specific = {
            "hookEventName": "PreToolUse",
            "permissionDecision": perm,
            "permissionDecisionReason": reason,
        }

    out: Dict[str, Any] = {
        "hookSpecificOutput": hook_specific,
        "host": resolved_host,
        "host_contract": host_contract,
        "host_action": host_action,
        "authorizes_execution": False,
        "decision": dec,
        "decision_kind": verdict.get("decision_kind"),
        "permissionDecision": perm,
        "reason": reason,
        "label": verdict["label"],
        "reason_code": verdict["reason_code"],
        "layer": verdict["layer"],
        "latency_ms": verdict["latency_ms"],
        "exit_code": exit_code,
    }
    if "escalation_contract" in verdict:
        out["escalation_contract"] = verdict["escalation_contract"]
    if "journal_written" in verdict:
        out["journal_written"] = verdict["journal_written"]
    if "journal_error" in verdict:
        out["journal_error"] = verdict["journal_error"]
    if shadow_mode:
        out["shadow_decision"] = verdict.get("shadow_decision", dec)
        out["shadow_reason_code"] = verdict.get("shadow_reason_code", verdict["reason_code"])
    return out


def _format_hook_stdout(hook_specific: Dict[str, Any]) -> str:
    return json.dumps({"hookSpecificOutput": hook_specific})


def main(argv: Optional[list[str]] = None) -> int:
    shadow_mode = bool(os.getenv("AETHERIS_HOOK_SHADOW") == "1")
    host = str(os.getenv("AETHERIS_HOOK_HOST") or "claude").strip().lower()
    if host not in {"claude", "codex", "generic"}:
        host = "claude"
    try:
        parser = argparse.ArgumentParser(description="Aetheris Microbrains PreToolUse Hook")
        parser.add_argument("--strict-ast", action="store_true", help="Enable strict AST import/dynamic-call allowlist")
        parser.add_argument("--log-decisions", action="store_true", help="Persist hook decisions to journal for cross-process drift detection")
        parser.add_argument(
            "--auto-allow",
            action="store_true",
            help="Explicitly emit permissionDecision='allow' on safe verdicts (bypasses Claude Code user prompts); default is silent pass-through",
        )
        parser.add_argument(
            "--shadow",
            action="store_true",
            help="Audit/shadow mode: log what Microbrains would deny/ask to the decision journal without blocking (passive even on errors)",
        )
        parser.add_argument(
            "--host",
            choices=["claude", "codex", "generic"],
            default=host,
            help="Host PreToolUse contract format (claude emits allow/ask/deny; codex omits unsupported 'ask')",
        )
        parser.add_argument(
            "--codex-escalate-deny",
            action="store_true",
            help="With --host codex, treat escalated actions as deny (exit 2) instead of deferring to Codex approval",
        )
        args = parser.parse_args(argv)
        log_decisions = bool(args.log_decisions or os.getenv("AETHERIS_LOG_DECISIONS") == "1")
        auto_allow = bool(args.auto_allow or os.getenv("AETHERIS_HOOK_AUTO_ALLOW") == "1")
        shadow_mode = bool(args.shadow or os.getenv("AETHERIS_HOOK_SHADOW") == "1")
        host = str(args.host)
        codex_escalate_deny = bool(args.codex_escalate_deny)

        raw = sys.stdin.read().strip()
        if not raw:
            if shadow_mode:
                print("empty_hook_payload (shadow passthrough)", file=sys.stderr)
                return 0
            if host == "codex":
                if codex_escalate_deny:
                    print("empty_hook_payload", file=sys.stderr)
                    print(
                        _format_hook_stdout(
                            {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "deny",
                                "permissionDecisionReason": "empty_hook_payload",
                            }
                        )
                    )
                    return 2
                return 0
            if host == "generic":
                print(json.dumps({"decision": "escalate", "reason_code": "empty_hook_payload", "host": "generic", "exit_code": 0}))
                return 0
            print(
                _format_hook_stdout(
                    {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                        "permissionDecisionReason": "empty_hook_payload",
                    }
                )
            )
            return 0
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            err_msg = _redact_error_text(f"invalid_hook_json:{exc.msg} (line {exc.lineno}, col {exc.colno})")
            print(err_msg, file=sys.stderr)
            if shadow_mode:
                return 0
            if host == "generic":
                print(json.dumps({"decision": "deny", "reason_code": err_msg, "host": "generic", "exit_code": 2}))
                return 2
            print(
                _format_hook_stdout(
                    {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": err_msg,
                    }
                )
            )
            return 2

        if not isinstance(payload, dict):
            err_msg = _redact_error_text("invalid_hook_payload:expected_json_object")
            print(err_msg, file=sys.stderr)
            if shadow_mode:
                return 0
            if host == "generic":
                print(json.dumps({"decision": "deny", "reason_code": err_msg, "host": "generic", "exit_code": 2}))
                return 2
            print(
                _format_hook_stdout(
                    {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": err_msg,
                    }
                )
            )
            return 2

        result = evaluate_hook_payload(
            payload,
            strict_ast=args.strict_ast,
            log_decisions=log_decisions,
            shadow_mode=shadow_mode,
            host=host,
            codex_escalate_deny=codex_escalate_deny,
        )
        if shadow_mode:
            if result.get("journal_written") is False:
                j_err = _redact_error_text(str(result.get("journal_error") or "unknown"))
                print(f"shadow_journal_write_failed:{j_err}", file=sys.stderr)
            # Passive shadow mode: logs to decision journal and returns control cleanly
            return 0
        exit_code = int(result["exit_code"])
        if host == "generic":
            generic_out = {
                k: result[k]
                for k in (
                    "decision",
                    "decision_kind",
                    "authorizes_execution",
                    "host_contract",
                    "label",
                    "reason_code",
                    "layer",
                    "latency_ms",
                    "exit_code",
                    "escalation_contract",
                    "journal_written",
                    "journal_error",
                )
                if k in result and result[k] is not None
            }
            generic_out["host"] = "generic"
            if exit_code == 2:
                print(result["reason"], file=sys.stderr)
            print(json.dumps(generic_out))
            return exit_code

        perm = result["permissionDecision"]
        if exit_code == 2:
            print(result["reason"], file=sys.stderr)
            if result["hookSpecificOutput"] is not None:
                print(_format_hook_stdout(result["hookSpecificOutput"]))
            return 2
        if host == "codex" and result.get("host_action") == "defer_to_host_approval":
            return 0
        if perm == "allow" and not auto_allow:
            # Silent pass-through: let host apply normal user permission rules
            return 0
        if result["hookSpecificOutput"] is not None:
            print(_format_hook_stdout(result["hookSpecificOutput"]))
        return exit_code
    except Exception as exc:
        err_msg = _redact_error_text(f"hook_internal_error:{type(exc).__name__}:{exc}")
        print(err_msg, file=sys.stderr)
        if shadow_mode:
            return 0
        if host == "generic":
            print(json.dumps({"decision": "deny", "reason_code": err_msg, "host": "generic", "exit_code": 2}))
            return 2
        print(
            _format_hook_stdout(
                {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": err_msg,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
