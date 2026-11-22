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
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .runtime_paths import file_lock

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None

_UNICODE_TAG_RE = re.compile(r"[\U000E0000-\U000E007F]")
_BIDI_OVERRIDE_RE = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u00AD\u2060\u180E]")
_VARIATION_SELECTOR_RE = re.compile(r"[\uFE00-\uFE0F\U000E0100-\U000E01EF]")
_SUPP_VARIATION_SELECTOR_RE = re.compile(r"[\U000E0100-\U000E01EF]")
_HOMOGLYPH_TABLE = str.maketrans(
    {
        "а": "a", "е": "e", "ё": "e", "о": "o", "р": "p", "с": "c", "х": "x", "і": "i", "у": "y", "к": "k",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
        "Х": "X", "І": "I",
        "α": "a", "ε": "e", "ο": "o", "ρ": "p", "ν": "v", "κ": "k", "ι": "i", "τ": "t",
    }
)
_SPOOFED_DELIM_RE = re.compile(
    r"(?i)(?:</?IMPORTANT>|</?SYSTEM>|</?system-reminder>|</?instructions>|</?tool_response>|"
    r"\[SYSTEM\]|\[/?INST\]|<<SYS>>|<</SYS>>|<\|im_start\|>\s*system|<\|endoftext\|>|<\|eot_id\|>|"
    r"###\s*(?:System(?:\s*Prompt)?|Instructions)\b)"
)
_OVERRIDE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "injection_ignore_instructions",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?"
            r"(?:the\s+)?(?:previous|prior|above|system|safety)\s+"
            r"(?:instructions|rules|policy|prompt|directives)\b"
        ),
    ),
    (
        "injection_conceal_from_user",
        re.compile(r"(?i)\bdo\s+not\s+(?:tell|inform|alert|mention(?:\s+this)?\s+to)\s+(?:the\s+)?user\b"),
    ),
    (
        "injection_covert_precondition",
        re.compile(
            r"(?is)\bbefore\s+(?:calling|running|using|executing)\s+this\s+tool\b"
            r"[\s\S]{0,120}\b(?:read|cat|send|pass|include|copy|env|ssh)\b"
        ),
    ),
    (
        "injection_covert_secret_exfil",
        re.compile(
            r"(?is)\b(?:read|cat|send|post|exfiltrat\w*|pass|include|attach|curl)\b"
            r"[\s\S]{0,80}(?:\.env\b|\.ssh/|id_rsa|credentials\.json|~/\.aws)"
        ),
    ),
    (
        "injection_new_system_instructions",
        re.compile(r"(?i)\b(?:new|updated)\s+system\s+(?:instructions|prompt|directives)\b"),
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
    "websearch",
    "http_request",
    "browser_open",
    "fetch_url",
}
_SHELL_EXTERNAL_INGEST_CMDS: Tuple[str, ...] = (
    "curl", "wget", "http ", "lynx", "w3m", "gh issue", "gh pr", "gh api",
    "git fetch", "git pull", "git clone", "invoke-webrequest", "invoke-restmethod", "iwr ", "irm ",
)


def _canonicalize_injection_text(raw: str) -> str:
    """Strip invisible smuggling characters and NFKC+homoglyph-normalize before phrase matching."""
    stripped = _UNICODE_TAG_RE.sub("", raw)
    stripped = _BIDI_OVERRIDE_RE.sub("", stripped)
    stripped = _VARIATION_SELECTOR_RE.sub("", stripped)
    stripped = _ZERO_WIDTH_RE.sub("", stripped)
    norm = unicodedata.normalize("NFKC", stripped)
    return norm.translate(_HOMOGLYPH_TABLE)


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
    """Screen text for indirect prompt injection markers and invisible Unicode/homoglyph smuggling."""
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
    if _SUPP_VARIATION_SELECTOR_RE.search(raw) or len(_VARIATION_SELECTOR_RE.findall(raw)) >= 2:
        findings.append("unicode_variation_selector_smuggling")
        high = True
    if len(_ZERO_WIDTH_RE.findall(raw)) >= 3:
        findings.append("excessive_zero_width_chars")

    canon = _canonicalize_injection_text(raw)
    if _SPOOFED_DELIM_RE.search(canon) or _SPOOFED_DELIM_RE.search(raw):
        findings.append("spoofed_instruction_delimiter")
        high = True
    for code, pat in _OVERRIDE_PATTERNS:
        if pat.search(canon) or pat.search(raw):
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
    trust_is_builtin_field: bool = False,
) -> Dict[str, Any]:
    """Static security scan of MCP tool manifests for poisoning, split-payloads, and shadowing."""
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
    combined_descs: List[str] = []

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
            if not (trust_is_builtin_field and entry.get("is_builtin", False)):
                shadowed_tools.append(t_name)
                tool_findings.append("mcp_tool_shadowing")

        desc = str(entry.get("description") or "")
        if desc:
            combined_descs.append(desc)
        strings = [t_name, desc]
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

    if len(combined_descs) > 1:
        # Cross-tool aggregate catches ShareLock split prompt injections across multiple tools.
        agg_text = " ".join(combined_descs)[:65536]
        agg_inj = detect_prompt_injection(agg_text)
        if agg_inj["injection_detected"]:
            split_new = [f for f in agg_inj["findings"] if f not in all_findings]
            if split_new:
                all_findings.append("mcp_split_prompt_injection")
                all_findings.extend(split_new)
                flagged_tools.append({"name": "<manifest_aggregate>", "findings": ["mcp_split_prompt_injection", *split_new]})

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
        if pin_path is not None:
            self.pin_path = Path(pin_path)
        else:
            from .runtime_paths import get_state_dir

            self.pin_path = get_state_dir() / "mcp_manifest_pins.json"
        self.corrupted: bool = False

    def _load(self) -> Dict[str, Any]:
        if not self.pin_path.is_file():
            return {}
        try:
            data = json.loads(self.pin_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self.corrupted = True
                return {}
            return data
        except Exception:
            self.corrupted = True
            return {}

    def pin_server(
        self,
        server_name: str,
        tools: Sequence[Dict[str, Any]],
        builtin_tool_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        scan = scan_mcp_tools(tools, builtin_tool_names=builtin_tool_names)
        with file_lock(self.pin_path):
            data = self._load()
            data[str(server_name)] = {
                "tool_hashes": scan["tool_hashes"],
                "pinned_at": round(time.time(), 6),
            }
            self.corrupted = False
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
        if self.corrupted:
            return {
                "verified": False,
                "unpinned_server": False,
                "rug_pull_detected": True,
                "corrupted_pin_store": True,
                "modified_tools": [],
                "added_tools": list(scan["tool_hashes"].keys()),
                "removed_tools": [],
                "scan_report": scan,
                "authorizes_execution": False,
            }
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
    """Cross-turn session taint & IFC provenance tracker implementing the 'lethal trifecta' guard."""

    def __init__(self, state_path: Optional[Path] = None, ttl_sec: float = 86400.0):
        self.state_path = Path(state_path) if state_path else None
        self.ttl_sec = float(ttl_sec)
        self.untrusted_external_ingested: bool = False
        self.sensitive_local_read: bool = False
        self.prompt_injection_seen: bool = False
        self.corrupted: bool = False
        self.taint_events: List[str] = []
        self.taint_artifacts: List[Dict[str, Any]] = []
        self.turn_count: int = 0
        if self.state_path is not None:
            self._load()

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        if self.state_path is None:
            yield
            return
        with file_lock(self.state_path):
            yield

    def _is_expired(self, data: Dict[str, Any]) -> bool:
        updated_at = data.get("updated_at")
        return bool(
            self.ttl_sec > 0
            and isinstance(updated_at, (int, float))
            and (time.time() - float(updated_at)) > self.ttl_sec
        )

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            with self._state_lock():
                if not self.state_path.is_file():
                    return
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self.corrupted = True
                return
            if self._is_expired(data):
                return
            self.untrusted_external_ingested = bool(data.get("untrusted_external_ingested", False))
            self.sensitive_local_read = bool(data.get("sensitive_local_read", False))
            self.prompt_injection_seen = bool(data.get("prompt_injection_seen", False))
            self.turn_count = int(data.get("turn_count", 0) or 0)
            raw_ev = data.get("taint_events", [])
            self.taint_events = [str(x) for x in raw_ev[-20:]] if isinstance(raw_ev, list) else []
            raw_art = data.get("taint_artifacts", [])
            self.taint_artifacts = [dict(x) for x in raw_art[-50:] if isinstance(x, dict)] if isinstance(raw_art, list) else []
        except (json.JSONDecodeError, ValueError, OSError):
            self.corrupted = True

    def save(self) -> None:
        if self.state_path is None:
            return
        with self._state_lock():
            if self.state_path.is_file():
                try:
                    disk = json.loads(self.state_path.read_text(encoding="utf-8"))
                    if not isinstance(disk, dict):
                        self.corrupted = True
                        return
                    if not self._is_expired(disk):
                        self.untrusted_external_ingested |= bool(disk.get("untrusted_external_ingested", False))
                        self.sensitive_local_read |= bool(disk.get("sensitive_local_read", False))
                        self.prompt_injection_seen |= bool(disk.get("prompt_injection_seen", False))
                        self.turn_count = max(self.turn_count, int(disk.get("turn_count", 0) or 0))
                        disk_events = [str(x) for x in disk.get("taint_events", []) if x] if isinstance(disk.get("taint_events"), list) else []
                        merged = list(dict.fromkeys(disk_events + self.taint_events))
                        self.taint_events = merged[-20:]
                        disk_art = [dict(x) for x in disk.get("taint_artifacts", []) if isinstance(x, dict)] if isinstance(disk.get("taint_artifacts"), list) else []
                        seen_art = {(a.get("source"), a.get("tool"), a.get("integrity"), a.get("confidentiality")) for a in disk_art}
                        merged_art = list(disk_art)
                        for a in self.taint_artifacts:
                            key = (a.get("source"), a.get("tool"), a.get("integrity"), a.get("confidentiality"))
                            if key not in seen_art:
                                seen_art.add(key)
                                merged_art.append(a)
                        self.taint_artifacts = merged_art[-50:]
                except (json.JSONDecodeError, ValueError, OSError):
                    self.corrupted = True
                    return
            payload = {
                "untrusted_external_ingested": self.untrusted_external_ingested,
                "sensitive_local_read": self.sensitive_local_read,
                "prompt_injection_seen": self.prompt_injection_seen,
                "turn_count": self.turn_count,
                "taint_events": self.taint_events[-20:],
                "taint_artifacts": self.taint_artifacts[-50:],
                "updated_at": round(time.time(), 6),
            }
            _atomic_write_json(self.state_path, payload)

    def _append_event(self, tag: str) -> None:
        self.taint_events.append(tag)
        if len(self.taint_events) > 20:
            self.taint_events = self.taint_events[-20:]

    def _record_artifact(self, source: str, tool: str, integrity: str, confidentiality: str) -> None:
        rec = {
            "source": str(source or tool)[:120],
            "tool": str(tool or ""),
            "integrity": integrity,
            "confidentiality": confidentiality,
            "turn": self.turn_count,
            "ts": round(time.time(), 3),
        }
        self.taint_artifacts.append(rec)
        if len(self.taint_artifacts) > 50:
            self.taint_artifacts = self.taint_artifacts[-50:]

    @staticmethod
    def _is_external_tool(tool_name: str) -> bool:
        t_low = str(tool_name or "").strip().lower()
        return (
            t_low in _UNTRUSTED_INGEST_TOOLS
            or t_low.startswith("mcp_")
            or "fetch" in t_low
            or "browse" in t_low
        )

    @staticmethod
    def _cmd_ingests_external(cmd_str: str) -> bool:
        low = f" {str(cmd_str or '').lower()} "
        return any(tok in low for tok in _SHELL_EXTERNAL_INGEST_CMDS)

    def record_external_shell(self, cmd_str: str) -> None:
        if self._cmd_ingests_external(cmd_str):
            self.turn_count += 1
            self.untrusted_external_ingested = True
            self._append_event("untrusted_shell_ingest")
            self._record_artifact(str(cmd_str)[:80], "execute_shell", "untrusted", "public")
            self.save()

    def provenance_summary(self) -> Dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "untrusted_external_ingested": self.untrusted_external_ingested,
            "sensitive_local_read": self.sensitive_local_read,
            "prompt_injection_seen": self.prompt_injection_seen,
            "artifact_count": len(self.taint_artifacts),
            "untrusted_sources": [a.get("source") for a in self.taint_artifacts if a.get("integrity") == "untrusted"][-10:],
            "secret_sources": [a.get("source") for a in self.taint_artifacts if a.get("confidentiality") == "secret"][-10:],
        }

    def reset_taint(self) -> None:
        """Explicit operator reset of cross-turn session taint state."""
        self.untrusted_external_ingested = False
        self.sensitive_local_read = False
        self.prompt_injection_seen = False
        self.corrupted = False
        self.taint_events = []
        self.taint_artifacts = []
        self.turn_count = 0
        if self.state_path is not None:
            with self._state_lock():
                _atomic_write_json(
                    self.state_path,
                    {
                        "untrusted_external_ingested": False,
                        "sensitive_local_read": False,
                        "prompt_injection_seen": False,
                        "turn_count": 0,
                        "taint_events": [],
                        "taint_artifacts": [],
                        "updated_at": round(time.time(), 6),
                    },
                )

    def record_pre_tool(self, tool_name: str, target_path: str = "", is_protected: bool = False) -> None:
        clean_tool = str(tool_name or "").strip().lower()
        self.turn_count += 1
        if is_protected:
            self.sensitive_local_read = True
            self._append_event(f"sensitive_read:{clean_tool}")
            self._record_artifact(target_path or clean_tool, clean_tool, "trusted", "secret")
        if self._is_external_tool(clean_tool):
            self.untrusted_external_ingested = True
            self._append_event(f"untrusted_ingest:{clean_tool}")
            self._record_artifact(target_path or clean_tool, clean_tool, "untrusted", "public")
        self.save()

    def record_post_tool(
        self,
        tool_name: str,
        secrets_found: bool = False,
        injection_detected: bool = False,
        is_protected_path: bool = False,
    ) -> None:
        clean_tool = str(tool_name or "").strip().lower()
        self.turn_count += 1
        if self._is_external_tool(clean_tool):
            self.untrusted_external_ingested = True
            self._append_event(f"post_external:{clean_tool}")
            self._record_artifact(clean_tool, clean_tool, "untrusted", "public")
        if is_protected_path:
            self.sensitive_local_read = True
            self._append_event(f"post_protected_read:{clean_tool}")
            self._record_artifact(clean_tool, clean_tool, "trusted", "secret")
        if secrets_found:
            self.sensitive_local_read = True
            self._append_event(f"post_secret:{clean_tool}")
            self._record_artifact(clean_tool, clean_tool, "trusted", "secret")
        if injection_detected:
            self.prompt_injection_seen = True
            self.untrusted_external_ingested = True
            self._append_event(f"post_injection:{clean_tool}")
            self._record_artifact(clean_tool, clean_tool, "untrusted", "secret" if self.sensitive_local_read else "public")
        self.save()

    def evaluate_taint_gate(self, tool_name: str, is_egress_or_mutation: bool) -> Dict[str, Any]:
        if not is_egress_or_mutation:
            return {"decision": "pass"}
        if self.corrupted:
            return {
                "decision": "deny",
                "reason_code": "layer0_session_state_corrupted",
                "label": "session_state_corrupted",
            }
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
