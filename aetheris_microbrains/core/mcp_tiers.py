"""MCP tool risk tiers.

Classifies `mcp__<server>__<tool>` calls against operator trust so pinned
servers get tiered verdicts instead of blanket fail-closed escalation:

- pinned/trusted server + explicit read-only tool name -> structural allow (`layer1_mcp_trusted_readonly`)
- pinned/trusted server + write-ish or unknown tool name -> escalate (`layer1_mcp_write_tool_escalate`)
- unpinned, drifted, or missing-config server          -> unchanged `layer1_uninspected_tool` (ask)

Pins live in `<state_dir>/mcp_pins.json` and bind to a fingerprint of the
server's `.mcp.json` launch definition, so editing the server config (command,
args, URL) or losing `.mcp.json` invalidates the pin (fail closed). Note: this
guards server launch-config drift and complements `MCPManifestPinner` in
`aetheris_microbrains.core.mcp_security` (which cryptographically pins full
tool schemas against remote rug-pull mutations). Operators can also trust
servers declaratively via `PolicyConfig.trusted_mcp_servers`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .runtime_paths import file_lock

# Tool-name fragments that indicate state-changing, code-execution, or
# exfiltration operations, matched case-insensitively on the tool segment.
_MCP_WRITEISH_MARKERS: Tuple[str, ...] = (
    "write", "edit", "create", "delete", "remove", "update", "insert",
    "send", "post", "put", "patch", "exec", "run", "deploy", "install",
    "uninstall", "approve", "pay", "transfer", "invite", "grant", "revoke",
    "publish", "merge", "close", "cancel", "mkdir", "upload", "import",
    "flush", "reset", "rotate", "kill", "stop", "start", "restart",
    "apply", "commit", "push", "rebase",
    "drop", "truncate", "destroy", "purge", "wipe", "shell", "eval",
    "bash", "cmd", "sql", "query", "mutate", "set", "enable", "disable",
    "chmod", "chown", "move", "mv", "copy", "cp", "rename", "fork",
    "archive", "comment", "reply", "dm", "mail", "trigger", "dispatch",
    "schedule", "sign", "mint", "withdraw", "charge", "refund", "export",
    "download", "proxy", "forward", "exfil", "call", "invoke",
)

# Strict read-only leading verb allowlist for trusted servers. Any tool name
# whose leading verb is NOT in this allowlist (or matches a write-ish marker)
# escalates rather than auto-allowing.
_MCP_READONLY_PREFIXES: Tuple[str, ...] = (
    "get", "list", "read", "search", "find", "fetch", "lookup", "describe",
    "inspect", "view", "show", "head", "stat", "check", "count", "exists",
    "status", "info", "whoami", "ping", "version", "health", "diff", "log",
    "blame", "browse", "resolve", "validate", "lint",
)

_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_MCP_PIN_CACHE: Dict[str, Any] = {}


def parse_mcp_tool_name(tool_name: str) -> Optional[Tuple[str, str]]:
    """Split `mcp__<server>__<tool>` into `(server, tool)`; None for non-MCP names."""
    parts = str(tool_name or "").split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        server = parts[1].strip().lower()
        tool = "__".join(parts[2:]).strip()
        if server and tool:
            return server, tool
    return None


def _leading_tool_verb(tool_segment: str) -> str:
    s = str(tool_segment or "").strip()
    if not s:
        return ""
    s = _CAMEL_BOUNDARY_RE.sub(r"\1_\2", s).lower()
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", s) if tok]
    return tokens[0] if tokens else ""


def is_writeish_mcp_tool(tool_segment: str) -> bool:
    """True when the tool name fragment suggests a state-changing or non-read-only operation."""
    norm_tokens = [tok for tok in re.split(r"[^a-z0-9]+", _CAMEL_BOUNDARY_RE.sub(r"\1_\2", str(tool_segment or "")).lower()) if tok]
    return any(tok in _MCP_WRITEISH_MARKERS for tok in norm_tokens)


def is_readonly_mcp_tool(tool_segment: str) -> bool:
    """True ONLY when the leading verb is explicitly read-only and no write-ish markers appear."""
    if is_writeish_mcp_tool(tool_segment):
        return False
    return _leading_tool_verb(tool_segment) in _MCP_READONLY_PREFIXES


def server_pin_fingerprint(server_cfg: Any) -> str:
    """Stable sha256 over the canonical JSON of a server definition."""
    try:
        canonical = json.dumps(server_cfg, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        canonical = repr(server_cfg)
    return hashlib.sha256(canonical.encode("utf-8", "ignore")).hexdigest()


def get_pin_store_path(state_dir: Optional[Path] = None) -> Path:
    """Resolve `<state_dir>/mcp_pins.json` (honors an explicit state dir)."""
    if state_dir is not None:
        return Path(state_dir) / "mcp_pins.json"
    from .runtime_paths import get_state_dir

    return get_state_dir() / "mcp_pins.json"


def _stat_sig(path: Path) -> Tuple[int, int, int]:
    st = path.stat()
    return (int(st.st_mtime_ns), int(st.st_size), int(getattr(st, "st_ino", 0)))


def load_mcp_pins(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Read the pin store with stat-signature caching; missing/corrupt files return {}."""
    path = get_pin_store_path(state_dir)
    try:
        sig = _stat_sig(path)
    except OSError:
        _MCP_PIN_CACHE.pop(str(path), None)
        return {}
    cached = _MCP_PIN_CACHE.get(str(path))
    if cached is not None and cached["sig"] == sig:
        return dict(cached["pins"])
    try:
        pins = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(pins, dict):
            pins = {}
    except Exception:
        pins = {}
    _MCP_PIN_CACHE[str(path)] = {"sig": sig, "pins": pins}
    return dict(pins)


def _atomic_write_pins(path: Path, pins: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(pins, indent=2, sort_keys=True) + "\n")
            fh.flush()
        if os.name != "nt":
            try:
                os.chmod(tmp_name, 0o600)
            except OSError:
                pass
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    _MCP_PIN_CACHE.pop(str(path), None)


def save_mcp_pin(server: str, server_cfg: Any, state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Pin `server` to its current config fingerprint (operator action)."""
    path = get_pin_store_path(state_dir)
    with file_lock(path):
        pins = load_mcp_pins(state_dir)
        updated = dict(pins)
        key = str(server).strip().lower()
        prev = updated.get(key) if isinstance(updated.get(key), dict) else {}
        entry: Dict[str, Any] = {
            "fingerprint": server_pin_fingerprint(server_cfg),
            "pinned_at": time.time(),
        }
        if isinstance(prev.get("tool_hashes"), dict):
            entry["tool_hashes"] = prev["tool_hashes"]
        updated[key] = entry
        _atomic_write_pins(path, updated)
    return updated


def save_mcp_manifest_pin(
    server: str,
    server_cfg: Any,
    tools: Sequence[Dict[str, Any]],
    state_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pin both launch config AND cryptographic tool-schema hashes for an MCP server."""
    from .mcp_security import MCPManifestPinner, scan_mcp_tools

    scan = scan_mcp_tools(tools)
    if not scan["safe"]:
        return {
            "status": "rejected_unsafe_manifest",
            "server": str(server),
            "safe": False,
            "scan_report": scan,
        }
    path = get_pin_store_path(state_dir)
    with file_lock(path):
        pins = load_mcp_pins(state_dir)
        updated = dict(pins)
        updated[str(server).strip().lower()] = {
            "fingerprint": server_pin_fingerprint(server_cfg),
            "tool_hashes": scan["tool_hashes"],
            "pinned_at": time.time(),
        }
        _atomic_write_pins(path, updated)
    manifest_path = path.with_name("mcp_manifest_pins.json")
    MCPManifestPinner(manifest_path).pin_server(str(server).strip().lower(), tools)
    return {
        "status": "pinned",
        "server": str(server).strip().lower(),
        "safe": True,
        "tool_hashes": scan["tool_hashes"],
        "scan_report": scan,
    }


def verify_mcp_manifest(
    server: str,
    tools: Sequence[Dict[str, Any]],
    state_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify live MCP tool schemas against pinned manifest hashes (anti-rug-pull)."""
    from .mcp_security import MCPManifestPinner, scan_mcp_tools

    path = get_pin_store_path(state_dir)
    pins = load_mcp_pins(state_dir)
    key = str(server).strip().lower()
    pinned = pins.get(key)
    if isinstance(pinned, dict) and isinstance(pinned.get("tool_hashes"), dict):
        scan = scan_mcp_tools(tools)
        pinned_hashes = pinned["tool_hashes"]
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
        }
    manifest_path = path.with_name("mcp_manifest_pins.json")
    return MCPManifestPinner(manifest_path).verify_server(key, tools)


def remove_mcp_pin(server: str, state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Drop `server` from the pin store (operator action)."""
    path = get_pin_store_path(state_dir)
    with file_lock(path):
        pins = load_mcp_pins(state_dir)
        updated = {k: v for k, v in pins.items() if k != str(server).strip().lower()}
        _atomic_write_pins(path, updated)
    return updated


def load_mcp_config(project_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Read `.mcp.json` server definitions from the project root with lowercase keys."""
    root = Path(project_dir) if project_dir else Path.cwd()
    for name in (".mcp.json", "mcp.json"):
        cand = root / name
        try:
            if cand.is_file():
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    servers = data.get("mcpServers", data)
                    if isinstance(servers, dict):
                        return {
                            str(k).strip().lower(): v
                            for k, v in servers.items()
                            if str(k).strip()
                        }
                    return {}
        except Exception:
            continue
    return {}


def mcp_tool_tier(
    tool_name: str,
    trusted_servers: Tuple[str, ...] = (),
    state_dir: Optional[Path] = None,
    project_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Tier an `mcp__` tool call: `trusted_readonly` | `trusted_writeish` | `unpinned`.

    A pin is honored ONLY when its server is defined in `.mcp.json` AND its
    fingerprint matches the current definition; missing configs or drifted
    configs fail closed (`unpinned` + `pin_stale=True`). Non-read-only or
    unrecognized tool verbs on trusted servers return `trusted_writeish` (ask).
    Declarative `PolicyConfig.trusted_mcp_servers` cannot go stale by config.
    """
    parsed = parse_mcp_tool_name(tool_name)
    if parsed is None:
        return {"tier": "unpinned", "server": "", "tool": str(tool_name or "")}
    server, tool_seg = parsed
    readonly = is_readonly_mcp_tool(tool_seg)
    writeish = not readonly

    trusted = {str(s).strip().lower() for s in (trusted_servers or ()) if str(s).strip()}
    pins = load_mcp_pins(state_dir)
    pinned_entry = pins.get(server)
    if pinned_entry is None and server not in trusted:
        return {"tier": "unpinned", "server": server, "tool": tool_seg, "writeish": writeish}

    pin_stale = False
    if pinned_entry is not None and server not in trusted:
        servers_cfg = load_mcp_config(project_dir)
        if server not in servers_cfg:
            return {
                "tier": "unpinned",
                "server": server,
                "tool": tool_seg,
                "writeish": writeish,
                "pin_stale": True,
                "pin_missing_config": True,
            }
        expected = pinned_entry.get("fingerprint") if isinstance(pinned_entry, dict) else None
        if not expected or expected != server_pin_fingerprint(servers_cfg.get(server)):
            return {
                "tier": "unpinned",
                "server": server,
                "tool": tool_seg,
                "writeish": writeish,
                "pin_stale": True,
            }
        pinned_tools = pinned_entry.get("tool_hashes") if isinstance(pinned_entry, dict) else None
        if isinstance(pinned_tools, dict):
            known_tools_lower = {str(k).strip().lower() for k in pinned_tools.keys()}
            if tool_seg.strip().lower() not in known_tools_lower:
                return {
                    "tier": "unpinned",
                    "server": server,
                    "tool": tool_seg,
                    "writeish": writeish,
                    "pin_stale": True,
                    "rug_pull_detected": True,
                }

    return {
        "tier": "trusted_readonly" if readonly else "trusted_writeish",
        "server": server,
        "tool": tool_seg,
        "writeish": writeish,
        "pin_stale": pin_stale,
        "source": "pin" if pinned_entry is not None else "policy",
    }
