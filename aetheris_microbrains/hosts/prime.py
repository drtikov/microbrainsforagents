"""Native Prime Agent (`prime_pretooluse_v1`) payload adapter (stdlib-only)."""

from __future__ import annotations

import shlex
from typing import Any, Dict, Mapping, Optional

PRIME_TOOL_ALIASES: Dict[str, str] = {
    "bash": "Bash",
    "shell": "Bash",
    "run_command": "Bash",
    "run_shell": "Bash",
    "edit": "Edit",
    "edit.run": "Edit",
    "read": "Read",
    "read_file": "Read",
    "write": "Write",
    "write_file": "Write",
}


def _coerce_shell_command(val: Any) -> Optional[str]:
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        tokens = [str(tok) for tok in val if tok is not None]
        return shlex.join(tokens) if tokens else ""
    return None


def to_kernel_payload(
    payload: Dict[str, Any],
    *,
    tool_aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Translate a Prime Agent REPL tool payload onto Microbrains' kernel schema."""
    out = dict(payload or {})
    raw = str(out.get("tool_name") or out.get("toolName") or out.get("tool") or "").strip()
    raw_low = raw.lower()
    if raw_low.startswith("mcp__"):
        canonical = raw
    else:
        custom_aliases = (
            {
                str(k).strip().lower(): str(v).strip()
                for k, v in tool_aliases.items()
                if str(k).strip() and str(v).strip()
            }
            if tool_aliases
            else {}
        )
        canonical = custom_aliases.get(raw_low) or PRIME_TOOL_ALIASES.get(raw_low, raw)
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

    if not ti.get("file_path"):
        path_val = ti.get("path") or ti.get("target_path")
        if isinstance(path_val, str) and path_val:
            ti["file_path"] = path_val
    if not ti.get("path"):
        path_val = ti.get("file_path")
        if isinstance(path_val, str) and path_val:
            ti["path"] = path_val

    if not ti.get("old_string"):
        for key in ("old_str", "old_text", "old"):
            val = ti.get(key)
            if isinstance(val, str):
                ti["old_string"] = val
                break
    if not ti.get("new_string"):
        for key in ("new_str", "new_text", "new", "file_text"):
            val = ti.get(key)
            if isinstance(val, str):
                ti["new_string"] = val
                break

    if canonical == "Write" and not ti.get("content") and isinstance(ti.get("text"), str):
        ti["content"] = ti["text"]

    out["tool_input"] = ti

    if not out.get("cwd") and isinstance(out.get("workspace"), str) and out["workspace"]:
        out["cwd"] = out["workspace"]

    return out


def format_block_error(tool_name: str, verdict: Mapping[str, Any]) -> str:
    """Format a blocked/escalated Microbrains verdict for Prime Agent diagnostics."""
    dec = str(verdict.get("decision") or "deny").upper()
    code = str(verdict.get("reason_code") or "POLICY_BLOCK")
    reason = str(verdict.get("reason") or "Blocked by Aetheris Microbrains policy.")
    return f"MICROBRAINS_{dec} ({tool_name}) [{code}]: {reason}"
