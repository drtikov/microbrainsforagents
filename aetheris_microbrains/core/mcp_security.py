"""MCP Supply-Chain Security, Indirect Prompt-Injection Screening & Session Taint Tracking.

Pure Python stdlib module (< 1 ms per call, zero external dependencies):
  1. `detect_prompt_injection(text)`: Screens MCP tool metadata and PostToolUse outputs
     for invisible Unicode tag smuggling, BiDi overrides, spoofed system/IMPORTANT delimiters,
     override instructions, and covert credential-exfiltration directives.
  2. `scan_mcp_tools(tools)`: Scans MCP tool schemas/descriptions for poisoning and tool
     shadowing against built-in tool names, computing canonical SHA-256 definitions.
  3. `MCPManifestPinner`: Pins canonical tool hashes atomically on disk to detect silent
     MCP server rug pulls on reconnect.
  4. `SessionSecurityState`: Cross-turn taint state enforcing the "lethal trifecta" guard
     (untrusted external ingestion / injection + sensitive local secret read -> egress/mutation veto).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

_UNICODE_TAG_RE = re.compile(r"[\U000E0000-\U000E007F]")
_BIDI_OVERRIDE_RE = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_SPOOFED_DELIM_RE = re.compile(
    r"(?i)(?:</?IMPORTANT>|</?SYSTEM>|\[SYSTEM\]|<\|im_start\|>\s*system|###\s*System\s*Prompt\b)"
)
_OVERRIDE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "injection_ignore_instructions",
        re.compile(r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions\b"),
    ),
    (
        "injection_conceal_from_user",
        re.compile(r"(?i)\bdo\s+not\s+(?:tell|inform|alert|mention(?:\s+this)?\s+to)\s+(?:the\s+)?user\b"),
    ),
    (
        "injection_covert_precondition",
        re.compile(r"(?is)\bbefore\s+(?:calling|running|using|executing)\s+this\s+tool\b[^\n]{0,120}\b(?:read|cat|send|pass|include|copy|env|ssh)\b"),
    ),
    (
        "injection_covert_secret_exfil",
        re.compile(
            r"(?is)\b(?:read|cat|send|post|exfiltrat\w*|pass|include|attach|curl)\b"
            r"[^\n]{0,80}(?:\.env\b|\.ssh/|id_rsa|credentials\.json|~/\.aws)"
        ),
    ),
)

_DEFAULT_BUILTIN_TOOLS: Set[str] = {
    "bash",
    "execute_shell",
    "run_command",
    "read_file",
    "repo_read",
    "write_file",
    "repo_write",
    "edit_file",
    "str_replace_editor",
    "delete_file",
    "web_fetch",
    "search_code",
}

_UNTRUSTED_INGEST_TOOLS: Set[str] = {
    "web_fetch",
    "webfetch",
    "web_search",
    "http_request",
    "browser_open",
    "fetch_url",
}


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())
        if os.name != "nt":
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def detect_prompt_injection(text: Any) -> Dict[str, Any]:
    """Screen text for indirect prompt injection markers and invisible Unicode smuggling."""
    raw = "" if text is None else str(text)
    if not raw:
        return {
            "injection_detected": False,
            "severity": "none",
            "findings": [],
            "authorizes_execution": False,
        }

    findings: List[str] = []
    high = False

    if _UNICODE_TAG_RE.search(raw):
        findings.append("unicode_tag_smuggling")
        high = True
    if _BIDI_OVERRIDE_RE.search(raw):
        findings.append("unicode_bidi_override")
        high = True
    if len(_ZERO_WIDTH_RE.findall(raw)) >= 3:
        findings.append("excessive_zero_width_chars")
    if _SPOOFED_DELIM_RE.search(raw):
        findings.append("spoofed_instruction_delimiter")
        high = True
    for code, pat in _OVERRIDE_PATTERNS:
        if pat.search(raw):
            findings.append(code)
            high = True

    dedup = list(dict.fromkeys(findings))
    detected = len(dedup) > 0
    severity = "high" if high else ("medium" if detected else "none")
    return {
        "injection_detected": detected,
        "severity": severity,
        "findings": dedup,
        "authorizes_execution": False,
    }


def _collect_schema_strings(node: Any, depth: int = 0, out: Optional[List[str]] = None) -> List[str]:
    acc = out if out is not None else []
    if depth > 8 or node is None:
        return acc
    if isinstance(node, str):
        if node:
            acc.append(node)
        return acc
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k:
                acc.append(k)
            _collect_schema_strings(v, depth + 1, acc)
        return acc
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_schema_strings(item, depth + 1, acc)
    return acc


def _canonical_tool_hash(tool_def: Dict[str, Any]) -> str:
    canonical_obj = {
        "name": str(tool_def.get("name") or "").strip(),
        "description": str(tool_def.get("description") or ""),
        "inputSchema": tool_def.get("inputSchema") if "inputSchema" in tool_def else tool_def.get("parameters", {}),
    }
    encoded = json.dumps(canonical_obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def scan_mcp_tools(
    tools: Sequence[Dict[str, Any]],
    builtin_tool_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Static security scan of MCP tool manifests for tool poisoning and tool shadowing."""
    builtins_norm = set(_DEFAULT_BUILTIN_TOOLS)
    if builtin_tool_names:
        for b in builtin_tool_names:
            clean_b = str(b or "").strip().lower()
            if clean_b:
                builtins_norm.add(clean_b)

    flagged_tools: List[Dict[str, Any]] = []
    shadowed_tools: List[str] = []
    tool_hashes: Dict[str, str] = {}
    all_findings: List[str] = []

    for entry in tools:
        if not isinstance(entry, dict):
            continue
        t_name = str(entry.get("name") or entry.get("tool_name") or "").strip()
        if not t_name:
            continue
        t_low = t_name.lower()
        tool_hashes[t_name] = _canonical_tool_hash(entry)

        tool_findings: List[str] = []
        if t_low in builtins_norm or t_low.replace("-", "_") in builtins_norm:
            if not entry.get("is_builtin", False):
                shadowed_tools.append(t_name)
                tool_findings.append("mcp_tool_shadowing")

        strings = [t_name, str(entry.get("description") or "")]
        schema_obj = entry.get("inputSchema") if "inputSchema" in entry else entry.get("parameters")
        if schema_obj is not None:
            _collect_schema_strings(schema_obj, out=strings)

        for s in strings:
            inj = detect_prompt_injection(s)
            if inj["injection_detected"]:
                tool_findings.extend(inj["findings"])

        dedup_tf = list(dict.fromkeys(tool_findings))
        if dedup_tf:
            flagged_tools.append({"name": t_name, "findings": dedup_tf})
            all_findings.extend(dedup_tf)

    return {
        "safe": len(flagged_tools) == 0,
        "tool_count": len(tool_hashes),
        "flagged_tools": flagged_tools,
        "injections_found": len(flagged_tools),
        "shadowed_tools": list(dict.fromkeys(shadowed_tools)),
        "tool_hashes": tool_hashes,
        "findings": list(dict.fromkeys(all_findings)),
        "authorizes_execution": False,
    }


class MCPManifestPinner:
    """Persistent SHA-256 pinner for MCP tool definitions to detect rug-pull mutations."""

    def __init__(self, pin_path: Optional[Path] = None):
        default_pins = Path(__file__).resolve().parents[1] / "memory" / "mcp_manifest_pins.json"
        self.pin_path = Path(pin_path) if pin_path else default_pins

    def _load(self) -> Dict[str, Any]:
        if not self.pin_path.is_file():
            return {}
        try:
            data = json.loads(self.pin_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def pin_server(
        self,
        server_name: str,
        tools: Sequence[Dict[str, Any]],
        builtin_tool_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        scan = scan_mcp_tools(tools, builtin_tool_names=builtin_tool_names)
        data = self._load()
        data[str(server_name)] = {
            "tool_hashes": scan["tool_hashes"],
            "pinned_at": round(time.time(), 6),
        }
        _atomic_write_json(self.pin_path, data)
        return {
            "status": "pinned",
            "server": str(server_name),
            "safe": scan["safe"],
            "tool_hashes": scan["tool_hashes"],
            "scan_report": scan,
            "authorizes_execution": False,
        }

    def verify_server(
        self,
        server_name: str,
        tools: Sequence[Dict[str, Any]],
        builtin_tool_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        scan = scan_mcp_tools(tools, builtin_tool_names=builtin_tool_names)
        data = self._load()
        pinned_rec = data.get(str(server_name))
        if not isinstance(pinned_rec, dict):
            return {
                "verified": False,
                "unpinned_server": True,
                "rug_pull_detected": False,
                "modified_tools": [],
                "added_tools": list(scan["tool_hashes"].keys()),
                "removed_tools": [],
                "scan_report": scan,
                "authorizes_execution": False,
            }

        pinned_hashes = pinned_rec.get("tool_hashes", {})
        cur_hashes = scan["tool_hashes"]
        modified = sorted(k for k in cur_hashes if k in pinned_hashes and cur_hashes[k] != pinned_hashes[k])
        added = sorted(k for k in cur_hashes if k not in pinned_hashes)
        removed = sorted(k for k in pinned_hashes if k not in cur_hashes)
        rug_pull = bool(modified or added or removed)
        return {
            "verified": bool(not rug_pull and scan["safe"]),
            "unpinned_server": False,
            "rug_pull_detected": rug_pull,
            "modified_tools": modified,
            "added_tools": added,
            "removed_tools": removed,
            "scan_report": scan,
            "authorizes_execution": False,
        }


class SessionSecurityState:
    """Cross-turn session taint tracker implementing the 'lethal trifecta' guard."""

    def __init__(self, state_path: Optional[Path] = None, ttl_sec: float = 86400.0):
        self.state_path = Path(state_path) if state_path else None
        self.ttl_sec = float(ttl_sec)
        self.untrusted_external_ingested: bool = False
        self.sensitive_local_read: bool = False
        self.prompt_injection_seen: bool = False
        self.taint_events: List[str] = []
        if self.state_path is not None:
            self._load()

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        if _fcntl is None or self.state_path is None:
            yield
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_path.with_name(f".{self.state_path.name}.lock")
        fh = open(lock_path, "a+", encoding="utf-8")
        try:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
            yield
        finally:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
            finally:
                fh.close()

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            with self._state_lock():
                if not self.state_path.is_file():
                    return
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                updated_at = data.get("updated_at")
                if (
                    self.ttl_sec > 0
                    and isinstance(updated_at, (int, float))
                    and (time.time() - float(updated_at)) > self.ttl_sec
                ):
                    return
                self.untrusted_external_ingested = bool(data.get("untrusted_external_ingested", False))
                self.sensitive_local_read = bool(data.get("sensitive_local_read", False))
                self.prompt_injection_seen = bool(data.get("prompt_injection_seen", False))
                raw_ev = data.get("taint_events", [])
                self.taint_events = [str(x) for x in raw_ev[-20:]] if isinstance(raw_ev, list) else []
        except Exception:
            pass

    def save(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "untrusted_external_ingested": self.untrusted_external_ingested,
            "sensitive_local_read": self.sensitive_local_read,
            "prompt_injection_seen": self.prompt_injection_seen,
            "taint_events": self.taint_events[-20:],
            "updated_at": round(time.time(), 6),
        }
        with self._state_lock():
            _atomic_write_json(self.state_path, payload)

    def _append_event(self, tag: str) -> None:
        self.taint_events.append(tag)
        if len(self.taint_events) > 20:
            self.taint_events = self.taint_events[-20:]

    @staticmethod
    def _is_external_tool(tool_name: str) -> bool:
        t_low = str(tool_name or "").strip().lower()
        return (
            t_low in _UNTRUSTED_INGEST_TOOLS
            or t_low.startswith("mcp_")
            or "fetch" in t_low
            or "browse" in t_low
        )

    def record_pre_tool(self, tool_name: str, target_path: str = "", is_protected: bool = False) -> None:
        clean_tool = str(tool_name or "").strip().lower()
        if is_protected:
            self.sensitive_local_read = True
            self._append_event(f"sensitive_read:{clean_tool}")
        if self._is_external_tool(clean_tool):
            self.untrusted_external_ingested = True
            self._append_event(f"untrusted_ingest:{clean_tool}")
        self.save()

    def record_post_tool(
        self,
        tool_name: str,
        secrets_found: bool = False,
        injection_detected: bool = False,
    ) -> None:
        clean_tool = str(tool_name or "").strip().lower()
        if self._is_external_tool(clean_tool):
            self.untrusted_external_ingested = True
            self._append_event(f"post_external:{clean_tool}")
        if secrets_found:
            self.sensitive_local_read = True
            self._append_event(f"post_secret:{clean_tool}")
        if injection_detected:
            self.prompt_injection_seen = True
            self.untrusted_external_ingested = True
            self._append_event(f"post_injection:{clean_tool}")
        self.save()

    def evaluate_taint_gate(self, tool_name: str, is_egress_or_mutation: bool) -> Dict[str, Any]:
        if not is_egress_or_mutation:
            return {"decision": "pass"}
        if (self.untrusted_external_ingested or self.prompt_injection_seen) and self.sensitive_local_read:
            return {
                "decision": "deny",
                "reason_code": "layer0_session_taint_lethal_trifecta",
                "label": "lethal_trifecta_blocked",
            }
        if self.prompt_injection_seen:
            return {
                "decision": "escalate",
                "reason_code": "layer0_session_prompt_injection_taint_escalate",
                "label": "prompt_injection_session_taint",
            }
        return {"decision": "pass"}
