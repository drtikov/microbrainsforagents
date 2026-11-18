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
) -> Dict[str, Any]:
    """Evaluate a PreToolUse hook payload and return normalized verdict + exit_code."""
    norm = _normalize_hook_payload(payload)
    policy = PolicyConfig(strict_ast_allowlist=strict_ast)
    ctx = {"file_content": file_content} if file_content is not None else None
    verdict = decide(norm, context=ctx, policy=policy)
    dec = verdict["decision"]
    perm = "allow" if dec == "allow" else ("deny" if dec == "deny" else "ask")
    exit_code = 2 if dec == "deny" else 0
    return {
        "decision": dec,
        "permissionDecision": perm,
        "reason": f"{verdict['reason_code']}: {verdict['label']}",
        "label": verdict["label"],
        "reason_code": verdict["reason_code"],
        "layer": verdict["layer"],
        "latency_ms": verdict["latency_ms"],
        "exit_code": exit_code,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aetheris Microbrains PreToolUse Hook")
    parser.add_argument("--strict-ast", action="store_true", help="Enable strict AST import/dynamic-call allowlist")
    args = parser.parse_args(argv)

    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"decision": "escalate", "permissionDecision": "ask", "reason": "empty_hook_payload"}))
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"decision": "deny", "permissionDecision": "deny", "reason": f"invalid_hook_json:{exc}"}))
        return 2

    result = evaluate_hook_payload(payload, strict_ast=args.strict_ast)
    out = {k: v for k, v in result.items() if k != "exit_code"}
    print(json.dumps(out))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
