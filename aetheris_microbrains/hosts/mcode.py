"""Native MiniMax Code (`mcode_pretooluse_v1`) payload normalization & response helpers (stdlib-only)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Native lowercase MCode tool names (`[A-Za-z0-9_.:/-]+`, no `*` wildcard).
MCODE_TOOL_MATCHER = "bash|read|write|edit|grep|glob|web_fetch|web_search|task"

NATIVE_TOOL_ALIASES: Dict[str, str] = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "grep": "Grep",
    "glob": "Glob",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
    "task": "Task",
    "todowrite": "TodoWrite",
}


def to_kernel_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an MCode PreToolUse/PostToolUse payload onto Microbrains' kernel schema."""
    out = dict(payload or {})
    raw = str(out.get("tool_name") or out.get("toolName") or out.get("tool") or "").strip()
    canonical = NATIVE_TOOL_ALIASES.get(raw.lower(), raw)
    if canonical and canonical != raw:
        out["tool_name"] = canonical
    if not out.get("cwd") and out.get("workspace"):
        out["cwd"] = out["workspace"]
    return out


def format_pre_response(
    *,
    decision: str,
    reason: str,
    updated_input: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], int]:
    """Render an engine verdict as MCode PreToolUse ``hookSpecificOutput`` + exit code."""
    perm = "allow" if decision == "allow" else ("deny" if decision == "deny" else "ask")
    exit_code = 2 if decision == "deny" else 0
    hso: Dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": perm,
        "permissionDecisionReason": reason,
    }
    if isinstance(updated_input, dict) and updated_input:
        hso["updatedInput"] = updated_input
    return hso, exit_code


def format_post_response(*, additional_context: Optional[str]) -> Optional[Dict[str, Any]]:
    """Render MCode PostToolUse warning context (tool-result replacement is unsupported)."""
    if additional_context:
        return {
            "hookEventName": "PostToolUse",
            "additionalContext": additional_context,
        }
    return None
