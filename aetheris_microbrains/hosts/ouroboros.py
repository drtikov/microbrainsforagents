"""Native Ouroboros (`ouroboros_registry_v1`) tool payload adapter (stdlib-only)."""

from __future__ import annotations

import shlex
from typing import Any, Dict, Iterable, List, Mapping, Optional

# Ouroboros native tool names (aligned with `razzant/ouroboros` ToolRegistry)
# mapped onto Microbrains' canonical kernel tool surface.
OUROBOROS_TOOL_ALIASES: Dict[str, str] = {
    # Shell & script execution
    "run_command": "Bash",
    "run_shell": "Bash",
    "run_script": "Bash",
    "start_service": "Bash",
    "stop_service": "Bash",
    "bash": "Bash",
    # File inspection & listing
    "read_file": "Read",
    "repo_read": "Read",
    "read": "Read",
    "list_files": "Glob",
    "glob": "Glob",
    # File writes & edits
    "write_file": "Write",
    "repo_write": "Write",
    "write": "Write",
    "edit_text": "Edit",
    "apply_patch": "Edit",
    "edit_batch": "Edit",
    "edit": "Edit",
    # Code search
    "search_code": "Grep",
    "query_code": "Grep",
    "grep": "Grep",
    # Web & browser
    "web_search": "WebSearch",
    "browse_page": "WebFetch",
    "browser_action": "WebFetch",
    # Read-only Ouroboros inspection tools (still path-checked if file_path/path is set)
    "service_status": "Read",
    "service_logs": "Read",
    "vcs_read": "Read",
    "github_read": "Read",
    "query_data": "Read",
    "inspect_subagent": "Read",
    "get_task_result": "Read",
    "peek_task": "Read",
    "recent_tasks": "Read",
    "live_roots": "Read",
    "tree_read": "Read",
    "journal_read": "Read",
    "workpad_read": "Read",
    "memory_read": "Read",
    "memory_map": "Read",
    "knowledge_read": "Read",
    "knowledge_list": "Read",
    "chat_history": "Read",
    "list_projects": "Read",
    # Ouroboros internal control-plane / state bookkeeping (benign no-shell)
    "plan_task": "TodoWrite",
    "finish_task": "TodoWrite",
    "wait_task": "TodoWrite",
    "wait_tasks": "TodoWrite",
    "await_messages": "TodoWrite",
    "update_focus": "TodoWrite",
    "tree_note": "TodoWrite",
    "journal_write": "TodoWrite",
    "workpad_write": "TodoWrite",
    "memory_mark": "TodoWrite",
    "knowledge_write": "TodoWrite",
    "update_scratchpad": "TodoWrite",
    "update_identity": "TodoWrite",
    "update_world": "TodoWrite",
    "chronicle_write": "TodoWrite",
    "compact_context": "TodoWrite",
    "send_user_message": "TodoWrite",
    "show_card": "TodoWrite",
}


def _coerce_shell_command(val: Any) -> Optional[str]:
    """Coerce string or argv token list/tuple into a shell command string."""
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        tokens = [str(tok) for tok in val if tok is not None]
        return shlex.join(tokens) if tokens else ""
    return None


def _coalesce_edit_batch(edits: Iterable[Any]) -> str:
    snippets: List[str] = []
    for item in edits:
        if not isinstance(item, Mapping):
            continue
        for key in ("new_string", "new_text", "new", "content", "replacement"):
            val = item.get(key)
            if isinstance(val, str) and val:
                snippets.append(val)
                break
    return "\n".join(snippets)


def to_kernel_payload(
    payload: Dict[str, Any],
    *,
    tool_aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Translate an Ouroboros ``ToolRegistry`` call payload onto Microbrains' kernel schema."""
    out = dict(payload or {})
    raw = str(out.get("tool_name") or out.get("toolName") or out.get("tool") or "").strip()
    raw_low = raw.lower()
    custom_aliases = (
        {str(k).strip().lower(): str(v).strip() for k, v in tool_aliases.items() if str(k).strip() and str(v).strip()}
        if tool_aliases
        else {}
    )
    canonical = custom_aliases.get(raw_low) or OUROBOROS_TOOL_ALIASES.get(raw_low, raw)
    if canonical and canonical != raw:
        out["tool_name"] = canonical

    raw_input = out.get("tool_input")
    if not isinstance(raw_input, dict):
        alt = out.get("args") if isinstance(out.get("args"), dict) else out.get("arguments")
        raw_input = alt if isinstance(alt, dict) else {}
    ti = dict(raw_input)

    raw_cmd = ti.get("command") if ti.get("command") else ti.get("cmd")
    coerced_cmd = _coerce_shell_command(raw_cmd)
    if coerced_cmd is not None:
        ti["command"] = coerced_cmd
    if raw_low == "run_script" and not ti.get("command"):
        script_body = ti.get("script") or ti.get("code") or ti.get("content")
        if isinstance(script_body, str) and script_body:
            ti["command"] = script_body

    if not ti.get("file_path"):
        path_val = ti.get("path") or ti.get("target_path")
        if isinstance(path_val, str) and path_val:
            ti["file_path"] = path_val
    if not ti.get("path"):
        dir_val = ti.get("directory") or ti.get("dir") or ti.get("file_path")
        if isinstance(dir_val, str) and dir_val:
            ti["path"] = dir_val

    if not ti.get("old_string"):
        for key in ("old_text", "old", "old_str"):
            val = ti.get(key)
            if isinstance(val, str):
                ti["old_string"] = val
                break
    if not ti.get("new_string"):
        for key in ("new_text", "new", "new_str"):
            val = ti.get(key)
            if isinstance(val, str):
                ti["new_string"] = val
                break

    if raw_low == "apply_patch" and not ti.get("new_string"):
        patch_body = ti.get("patch") or ti.get("diff") or ti.get("content")
        if isinstance(patch_body, str) and patch_body:
            ti["new_string"] = patch_body

    if raw_low == "edit_batch" and not ti.get("new_string") and isinstance(ti.get("edits"), list):
        combined = _coalesce_edit_batch(ti["edits"])
        if combined:
            ti["new_string"] = combined
        if not ti.get("file_path"):
            for item in ti["edits"]:
                if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                    ti["file_path"] = item["path"]
                    break

    if canonical == "Write" and not ti.get("content") and isinstance(ti.get("text"), str):
        ti["content"] = ti["text"]

    if canonical == "Grep" and not ti.get("pattern") and isinstance(ti.get("query"), str):
        ti["pattern"] = ti["query"]

    out["tool_input"] = ti

    if not out.get("cwd"):
        for cwd_key in ("workspace", "repo_dir", "drive_root"):
            cwd_val = out.get(cwd_key) or ti.get(cwd_key)
            if isinstance(cwd_val, str) and cwd_val:
                out["cwd"] = cwd_val
                break

    return out


def extract_schema_name(schema: Mapping[str, Any]) -> str:
    """Extract the tool name from an OpenAI-style or flat Ouroboros tool schema dict."""
    fn = schema.get("function")
    if isinstance(fn, Mapping) and fn.get("name"):
        return str(fn["name"])
    return str(schema.get("name") or "")


def extract_schema_description(schema: Mapping[str, Any]) -> str:
    """Extract the description from an OpenAI-style or flat Ouroboros tool schema dict."""
    fn = schema.get("function")
    if isinstance(fn, Mapping) and fn.get("description"):
        return str(fn["description"])
    return str(schema.get("description") or "")


def format_block_error(tool_name: str, verdict: Mapping[str, Any]) -> str:
    """Format a blocked/escalated Microbrains verdict into an Ouroboros tool-result string."""
    dec = str(verdict.get("decision") or "deny").upper()
    code = str(verdict.get("reason_code") or "POLICY_BLOCK")
    reason = str(verdict.get("reason") or "Blocked by Aetheris Microbrains policy.")
    return f"MICROBRAINS_{dec} ({tool_name}) [{code}]: {reason}"
