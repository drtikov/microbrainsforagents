"""Native Devin (`devin_hook_v1`) payload normalization and response formatting (stdlib-only)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

_DEVIN_TOOL_MAP: Dict[str, str] = {
    "exec": "Bash",
    "get_output": "Read",
    "read": "Read",
    "notebook_read": "Read",
    "grep": "Grep",
    "glob": "Glob",
    "edit": "Edit",
    "apply_patch": "Edit",
    "multiedit": "Edit",
    "notebook_edit": "Edit",
    "write": "Write",
    "webfetch": "WebFetch",
    "run_subagent": "Task",
    "skill": "Task",
    "exit_plan_mode": "Task",
    "todo_write": "TodoWrite",
}


def detect_devin_env() -> bool:
    """Return True when running inside a Devin hook environment."""
    return bool(str(os.environ.get("DEVIN_PROJECT_DIR") or "").strip())


def _flatten_tool_response(resp: Any) -> str:
    """Normalize Devin's structured ``tool_response`` object into inspectable text."""
    if isinstance(resp, dict):
        parts = []
        if resp.get("output") is not None:
            parts.append(str(resp.get("output")))
        if resp.get("error"):
            parts.append(f"error: {resp.get('error')}")
        if parts:
            return "\n".join(parts)
    if resp is None:
        return ""
    return str(resp)


def to_kernel_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Devin PreToolUse/PostToolUse JSON onto Microbrains' kernel schema."""
    out = dict(payload)
    tool_raw = str(payload.get("tool_name") or "").strip()
    t_low = tool_raw.lower()
    if t_low in _DEVIN_TOOL_MAP:
        out["tool_name"] = _DEVIN_TOOL_MAP[t_low]
    elif t_low == "write_to_process":
        out["tool_name"] = "write_to_process"
    elif t_low == "kill_shell":
        out["tool_name"] = "kill_shell"

    if not out.get("session_id"):
        out["session_id"] = out.get("prompt_id") or ""

    if "tool_response" in payload and not isinstance(payload.get("tool_response"), str):
        flat = _flatten_tool_response(payload.get("tool_response"))
        out["tool_response"] = flat
        out.setdefault("tool_output", flat)
    return out


def format_pre_response(
    *,
    decision: str,
    reason: str,
    updated_input: Optional[Dict[str, Any]] = None,
    auto_allow: bool = False,
    auto_approve_escalations: bool = False,
    shadow_mode: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """Emit Devin-native PreToolUse stdout (``decision: approve|block`` + optional ``updatedInput``)."""
    if shadow_mode:
        return {}, 0

    if decision == "deny":
        return {"decision": "block", "reason": reason}, 2

    if decision == "escalate":
        if auto_approve_escalations:
            return {"decision": "approve", "reason": reason}, 0
        # No native `ask` on Devin PreToolUse: exit 0 with empty stdout defers to Devin's prompt.
        return {}, 0

    if updated_input:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": updated_input,
            }
        }, 0
    if auto_allow:
        return {"decision": "approve", "reason": reason}, 0
    return {}, 0


def format_post_response(*, additional_context: Optional[str]) -> Dict[str, Any]:
    """Emit Devin PostToolUse warning context (Devin has no ``updatedToolOutput`` redaction)."""
    if additional_context:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": additional_context,
            }
        }
    return {}
