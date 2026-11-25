"""Native Cursor (`cursor_hook_v2`) payload normalization and response formatting (stdlib-only)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

_EVENT_ALIASES: Dict[str, str] = {
    "before-shell": "beforeShellExecution",
    "beforeshellexecution": "beforeShellExecution",
    "after-shell": "afterShellExecution",
    "aftershellexecution": "afterShellExecution",
    "before-mcp": "beforeMCPExecution",
    "beforemcpexecution": "beforeMCPExecution",
    "after-mcp": "afterMCPExecution",
    "aftermcpexecution": "afterMCPExecution",
    "before-read": "beforeReadFile",
    "beforereadfile": "beforeReadFile",
    "beforetabfileread": "beforeTabFileRead",
    "after-file-edit": "afterFileEdit",
    "afterfileedit": "afterFileEdit",
    "pre-tool-use": "preToolUse",
    "pretooluse": "preToolUse",
    "post-tool-use": "postToolUse",
    "posttooluse": "postToolUse",
}


def infer_event(payload: Dict[str, Any], cli_event: str = "pre-tool-use") -> str:
    """Infer the canonical Cursor hook event name from payload or CLI flag."""
    raw_cli = str(cli_event or "pre-tool-use").strip()
    cli_low = raw_cli.lower()
    if cli_low in _EVENT_ALIASES and cli_low not in {"pre-tool-use", "post-tool-use"}:
        return _EVENT_ALIASES[cli_low]

    name = str(payload.get("hook_event_name") or "").strip()
    if name:
        return _EVENT_ALIASES.get(name.lower(), name)

    if "command" in payload and "tool_name" not in payload and "file_path" not in payload:
        return "afterShellExecution" if cli_low == "post-tool-use" else "beforeShellExecution"
    if "file_path" in payload and "content" in payload:
        return "beforeReadFile"
    if "file_path" in payload and "edits" in payload:
        return "afterFileEdit"
    return _EVENT_ALIASES.get(cli_low, raw_cli)


def to_kernel_payload(payload: Dict[str, Any], event: str) -> Dict[str, Any]:
    """Translate Cursor hook JSON onto Microbrains' pre/post hook schema."""
    out = dict(payload)
    if not out.get("session_id"):
        out["session_id"] = out.get("conversation_id") or out.get("conversationId") or ""

    event_l = str(event or "").strip().lower()
    if event_l in {"beforeshellexecution", "aftershellexecution"}:
        cmd = payload.get("command") or ""
        out["tool_name"] = "Shell"
        out["tool_input"] = {"command": cmd, "working_directory": payload.get("cwd")}
        if event_l.startswith("after"):
            out["tool_output"] = payload.get("output") or payload.get("tool_output") or ""
        return out

    if event_l in {"beforereadfile", "beforetabfileread"}:
        out["tool_name"] = "Read"
        out["tool_input"] = {"file_path": payload.get("file_path")}
        if "content" in payload:
            out["file_content"] = payload.get("content")
        return out

    if event_l == "afterfileedit":
        edits = payload.get("edits") or []
        first = edits[0] if isinstance(edits, list) and edits else {}
        out["tool_name"] = "Write"
        out["tool_input"] = {
            "file_path": payload.get("file_path"),
            "old_string": first.get("old_string", "") if isinstance(first, dict) else "",
            "new_string": first.get("new_string", "") if isinstance(first, dict) else "",
        }
        return out

    if event_l in {"beforemcpexecution", "aftermcpexecution"}:
        name = str(payload.get("tool_name") or "")
        if not name.startswith("MCP:"):
            name = f"MCP:{name}"
        raw_in = payload.get("tool_input")
        if isinstance(raw_in, str):
            try:
                raw_in = json.loads(raw_in)
            except json.JSONDecodeError:
                raw_in = {"_raw": raw_in}
        out["tool_name"] = name
        out["tool_input"] = raw_in if isinstance(raw_in, dict) else {}
        if event_l.startswith("after"):
            out["tool_output"] = payload.get("result_json") or payload.get("tool_output") or ""
        return out

    t = str(payload.get("tool_name") or "")
    t_low = t.lower()
    if t_low == "shell":
        out["tool_name"] = "Bash"
    elif t_low in {"write", "writefile"}:
        out["tool_name"] = "Write"
    elif t_low in {"edit", "editfile", "strreplace", "str_replace", "search_replace"}:
        out["tool_name"] = "Edit"
    elif t_low in {"read", "readfile", "readlints", "read_lints", "getdynamictools", "get_dynamic_tools"}:
        out["tool_name"] = "Read"
    elif t_low in {"ls", "listdir", "list_dir", "glob", "filesearch"}:
        out["tool_name"] = "Glob"
    elif t_low in {"grep", "ripgrep", "semanticsearch", "semantic_search", "codebasesearch", "codebase_search"}:
        out["tool_name"] = "Grep"
    elif t_low in {"delete", "deletefile", "delete_file"}:
        out["tool_name"] = "delete_file"
    elif t.startswith("MCP:"):
        out["tool_name"] = "mcp__" + t[4:].replace(":", "__").replace("/", "__")
    inp = payload.get("tool_input")
    if not isinstance(inp, dict):
        out["tool_input"] = {"command": payload.get("command")} if payload.get("command") else {}
    return out


def format_pre_response(
    *,
    event: str,
    decision: str,
    reason: str,
    updated_input: Optional[Dict[str, Any]] = None,
    shadow_mode: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """Emit dual Cursor-native stdout + nested ``hookSpecificOutput``."""
    if shadow_mode:
        return {"permission": "allow"}, 0

    event_l = str(event or "").strip().lower()
    ask_ok = event_l in {"beforeshellexecution", "beforemcpexecution"}
    if decision == "allow":
        perm, code = "allow", 0
    elif decision == "deny":
        perm, code = "deny", 2
    elif ask_ok:
        perm, code = "ask", 0
    else:
        # preToolUse / beforeReadFile: Cursor does not enforce `ask`, so fail closed
        perm, code = "deny", 2

    hso: Dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": perm,
        "permissionDecisionReason": reason,
    }
    body: Dict[str, Any] = {
        "permission": perm,
        "hookSpecificOutput": hso,
    }
    if perm != "allow":
        body["user_message"] = reason
        body["agent_message"] = reason
    if updated_input and perm == "allow" and event_l == "pretooluse":
        body["updated_input"] = updated_input
        hso["updatedInput"] = updated_input
    return body, code


def format_post_response(
    *,
    additional_context: Optional[str],
    redacted: Optional[str],
) -> Dict[str, Any]:
    """Emit Cursor-native PostToolUse / after* output with optional scrubbing fields."""
    out: Dict[str, Any] = {}
    if additional_context:
        hso: Dict[str, Any] = {
            "hookEventName": "PostToolUse",
            "additionalContext": additional_context,
        }
        out["additional_context"] = additional_context
        out["hookSpecificOutput"] = hso
        if redacted is not None:
            out["updated_mcp_tool_output"] = redacted
            hso["updatedMCPToolOutput"] = redacted
            hso["updatedToolOutput"] = redacted
    return out
