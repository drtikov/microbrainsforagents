"""Native Cline (`cline_hook_v1`) payload normalization and response formatting (stdlib-only)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_CLINE_TOOL_MAP: Dict[str, str] = {
    "execute_command": "Bash",
    "run_commands": "Bash",
    "write_to_file": "Write",
    "replace_in_file": "Edit",
    "apply_patch": "Edit",
    "read_file": "Read",
    "search_files": "Grep",
    "list_files": "Glob",
    "list_code_definition_names": "Glob",
    "browser_action": "WebFetch",
    "web_fetch": "WebFetch",
    "ask_followup_question": "Task",
    "attempt_completion": "Task",
    "plan_mode_respond": "Task",
    "new_task": "Task",
    "load_mcp_documentation": "Task",
    "focus_chain": "Task",
}

_SEARCH_REPLACE_RE = re.compile(
    r"(?:<<<<<<<|-------)\s*SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n(?:\+\+\+\+\+\+\+|>>>>>>>)\s*REPLACE",
    re.DOTALL,
)


def _extract_search_replace_blocks(diff_text: str) -> List[Dict[str, str]]:
    """Parse Cline SEARCH/REPLACE diff blocks into ``old_string``/``new_string`` pairs."""
    return [
        {"old_string": m.group(1), "new_string": m.group(2)}
        for m in _SEARCH_REPLACE_RE.finditer(diff_text)
    ]


def to_kernel_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Cline PreToolUse/PostToolUse hook JSON onto Microbrains' kernel schema."""
    out = dict(payload)

    # Extract raw tool name + parameters across all 3 Cline stdin schemas
    tool_raw = ""
    params: Optional[Dict[str, Any]] = None
    post_result: Any = None

    pre_block = payload.get("preToolUse")
    post_block = payload.get("postToolUse")
    tool_block = payload.get("tool")

    if isinstance(pre_block, dict):
        tool_raw = str(pre_block.get("toolName") or pre_block.get("tool_name") or pre_block.get("name") or "").strip()
        if isinstance(pre_block.get("parameters"), dict):
            params = dict(pre_block["parameters"])
    elif isinstance(post_block, dict):
        tool_raw = str(post_block.get("toolName") or post_block.get("tool_name") or post_block.get("name") or "").strip()
        if isinstance(post_block.get("parameters"), dict):
            params = dict(post_block["parameters"])
        post_result = post_block.get("result")
    elif isinstance(tool_block, dict):
        tool_raw = str(tool_block.get("name") or tool_block.get("toolName") or tool_block.get("tool_name") or "").strip()
        if isinstance(tool_block.get("parameters"), dict):
            params = dict(tool_block["parameters"])

    if not tool_raw:
        tool_raw = str(payload.get("toolName") or payload.get("tool_name") or "").strip()
    if params is None:
        if isinstance(payload.get("parameters"), dict):
            params = dict(payload["parameters"])
        elif isinstance(payload.get("tool_input"), dict):
            params = dict(payload["tool_input"])
        else:
            params = {}

    t_low = tool_raw.lower()
    if t_low in _CLINE_TOOL_MAP:
        out["tool_name"] = _CLINE_TOOL_MAP[t_low]
    elif t_low in {"use_mcp_tool", "access_mcp_resource"}:
        srv = str(params.get("server_name") or params.get("server") or "unknown").strip()
        mtool = str(params.get("tool_name") or ("resource" if t_low == "access_mcp_resource" else "call")).strip()
        srv_clean = srv.replace(":", "_").replace("/", "_").replace(" ", "_")
        mtool_clean = mtool.replace(":", "_").replace("/", "_").replace(" ", "_")
        out["tool_name"] = f"mcp__{srv_clean}__{mtool_clean}"
    elif tool_raw:
        out["tool_name"] = tool_raw

    # Convert Cline SEARCH/REPLACE diff markers into clean code edit params for AST inspection
    if t_low == "replace_in_file" and isinstance(params.get("diff"), str):
        blocks = _extract_search_replace_blocks(params["diff"])
        if blocks:
            if len(blocks) == 1:
                params.setdefault("old_string", blocks[0]["old_string"])
                params.setdefault("new_string", blocks[0]["new_string"])
                params["diff"] = blocks[0]["new_string"]
            else:
                params["edits"] = blocks
                params["diff"] = "\n".join(b["new_string"] for b in blocks)

    out["tool_input"] = params

    if not out.get("session_id"):
        out["session_id"] = str(payload.get("taskId") or payload.get("task_id") or "")
    if not out.get("cwd") and isinstance(payload.get("workspaceRoots"), list) and payload["workspaceRoots"]:
        out["cwd"] = str(payload["workspaceRoots"][0])

    if post_result is None and "tool_result" in payload:
        post_result = payload.get("tool_result")
    if post_result is not None and not out.get("tool_output"):
        out["tool_output"] = str(post_result)
    return out


def format_pre_response(
    *,
    decision: str,
    reason: str,
    shadow_mode: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """Emit Cline PreToolUse JSON (``cancel`` + ``errorMessage``/``contextModification``)."""
    if shadow_mode:
        return {"cancel": False}, 0
    if decision == "deny":
        return {"cancel": True, "errorMessage": reason}, 2
    if decision == "escalate":
        return {"cancel": False, "contextModification": f"[Aetheris Microbrains Escalation] {reason}"}, 0
    return {"cancel": False}, 0


def format_post_response(*, additional_context: Optional[str]) -> Dict[str, Any]:
    """Emit Cline PostToolUse JSON (informational ``contextModification`` warning)."""
    if additional_context:
        return {"cancel": False, "contextModification": additional_context}
    return {"cancel": False}
