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
import sys
from typing import Any, Dict, Optional

from .core.layered_decision_engine import PolicyConfig, decide


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
) -> Dict[str, Any]:
    """Evaluate a PreToolUse hook payload and return normalized verdict + exit_code."""
    norm = _normalize_hook_payload(payload)
    policy = PolicyConfig(strict_ast_allowlist=strict_ast)
    ctx: Dict[str, Any] = {}
    if file_content is not None:
        ctx["file_content"] = file_content
    if log_decisions:
        ctx["log_decisions"] = True
    verdict = decide(norm, context=ctx or None, policy=policy)
    dec = verdict["decision"]
    perm = "allow" if dec == "allow" else ("deny" if dec == "deny" else "ask")
    exit_code = 2 if dec == "deny" else 0
    reason = f"{verdict['reason_code']}: {verdict['label']}"
    hook_specific = {
        "hookEventName": "PreToolUse",
        "permissionDecision": perm,
        "permissionDecisionReason": reason,
    }
    return {
        "hookSpecificOutput": hook_specific,
        "decision": dec,
        "permissionDecision": perm,
        "reason": reason,
        "label": verdict["label"],
        "reason_code": verdict["reason_code"],
        "layer": verdict["layer"],
        "latency_ms": verdict["latency_ms"],
        "exit_code": exit_code,
    }


def _format_hook_stdout(hook_specific: Dict[str, Any]) -> str:
    return json.dumps({"hookSpecificOutput": hook_specific})


def main(argv: Optional[list[str]] = None) -> int:
    try:
        parser = argparse.ArgumentParser(description="Aetheris Microbrains PreToolUse Hook")
        parser.add_argument("--strict-ast", action="store_true", help="Enable strict AST import/dynamic-call allowlist")
        parser.add_argument("--log-decisions", action="store_true", help="Persist hook decisions to journal for cross-process drift detection")
        parser.add_argument(
            "--auto-allow",
            action="store_true",
            help="Explicitly emit permissionDecision='allow' on safe verdicts (bypasses Claude Code user prompts); default is silent pass-through",
        )
        args = parser.parse_args(argv)
        log_decisions = bool(args.log_decisions or os.getenv("AETHERIS_LOG_DECISIONS") == "1")
        auto_allow = bool(args.auto_allow or os.getenv("AETHERIS_HOOK_AUTO_ALLOW") == "1")

        raw = sys.stdin.read().strip()
        if not raw:
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
            err_msg = f"invalid_hook_json:{exc}"
            print(err_msg, file=sys.stderr)
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

        result = evaluate_hook_payload(payload, strict_ast=args.strict_ast, log_decisions=log_decisions)
        exit_code = int(result["exit_code"])
        perm = result["permissionDecision"]
        if exit_code == 2:
            print(result["reason"], file=sys.stderr)
            print(_format_hook_stdout(result["hookSpecificOutput"]))
            return 2
        if perm == "allow" and not auto_allow:
            # Silent pass-through: let Claude Code apply normal user permission rules
            return 0
        print(_format_hook_stdout(result["hookSpecificOutput"]))
        return exit_code
    except Exception as exc:
        err_msg = f"hook_internal_error:{type(exc).__name__}:{exc}"
        print(err_msg, file=sys.stderr)
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
