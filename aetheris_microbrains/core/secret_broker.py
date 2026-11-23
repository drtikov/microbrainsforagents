"""Out-of-Band Secret Vault Broker for Aetheris MCP Gateway (stdlib-only, network-free).

Ensures raw API tokens and credentials stay in an operator-managed vault outside
the LLM's context window:
  1. The agent passes opaque handles (`$SECRET:<NAME>` or `{{aetheris_secret:<NAME>}}`).
  2. The broker enforces per-server and per-tool authorization scopes before resolution.
  3. Resolved secret values are injected ONLY into the outbound downstream call payload
     and stripped (raw, base64, and URL-encoded forms) from any returned response text.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote as _url_quote

from .runtime_paths import file_lock, get_state_dir

_SECRET_HANDLE_RE = re.compile(
    r"\$SECRET:([A-Za-z_][A-Za-z0-9_]*)|\{\{aetheris_secret:([A-Za-z_][A-Za-z0-9_]*)\}\}"
)


@dataclass
class SecretSpec:
    """Vault entry binding a secret handle name to a static value or env var + scope."""

    name: str
    value: Optional[str] = None
    env_var: Optional[str] = None
    allowed_servers: Tuple[str, ...] = ()
    allowed_tools: Tuple[str, ...] = ()

    def resolve_value(self) -> Optional[str]:
        if self.value is not None and str(self.value) != "":
            return str(self.value)
        if self.env_var:
            val = os.environ.get(str(self.env_var))
            if val:
                return val
        return None


def _atomic_write_json(target: Path, data: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
        if os.name != "nt":
            try:
                os.chmod(tmp_name, 0o600)
            except OSError:
                pass
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class SecretBroker:
    """Scoped out-of-band credential resolver & response scrubber for MCP Gateway."""

    def __init__(
        self,
        vault_path: Optional[Path | str] = None,
        bindings: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.vault_path = Path(vault_path) if vault_path is not None else (get_state_dir() / "secret_vault.json")
        self._specs: Dict[str, SecretSpec] = {}
        self._load_vault()
        if bindings:
            for k, cfg in bindings.items():
                if isinstance(cfg, dict):
                    self.register_spec(
                        k,
                        value=cfg.get("value"),
                        env_var=cfg.get("env_var"),
                        allowed_servers=cfg.get("allowed_servers") or (),
                        allowed_tools=cfg.get("allowed_tools") or (),
                    )

    def _load_vault(self) -> None:
        if not self.vault_path.is_file():
            return
        try:
            raw = json.loads(self.vault_path.read_text(encoding="utf-8"))
        except Exception:
            return
        entries = raw.get("secrets") if isinstance(raw, dict) and isinstance(raw.get("secrets"), dict) else raw
        if not isinstance(entries, dict):
            return
        for k, cfg in entries.items():
            if isinstance(cfg, dict):
                self.register_spec(
                    k,
                    value=cfg.get("value"),
                    env_var=cfg.get("env_var"),
                    allowed_servers=cfg.get("allowed_servers") or (),
                    allowed_tools=cfg.get("allowed_tools") or (),
                )

    def register_spec(
        self,
        name: str,
        *,
        value: Optional[str] = None,
        env_var: Optional[str] = None,
        allowed_servers: Sequence[str] = (),
        allowed_tools: Sequence[str] = (),
    ) -> None:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("secret name must be non-empty")
        self._specs[clean_name] = SecretSpec(
            name=clean_name,
            value=str(value) if value is not None else None,
            env_var=str(env_var) if env_var else None,
            allowed_servers=tuple(str(s).strip().lower() for s in allowed_servers if str(s).strip()),
            allowed_tools=tuple(str(t).strip().lower() for t in allowed_tools if str(t).strip()),
        )

    def save_secret(
        self,
        name: str,
        value: str,
        *,
        allowed_servers: Sequence[str] = (),
        allowed_tools: Sequence[str] = (),
        env_var: Optional[str] = None,
    ) -> None:
        """Persist a secret handle specification atomically with owner-only permissions."""
        self.register_spec(
            name,
            value=value,
            env_var=env_var,
            allowed_servers=allowed_servers,
            allowed_tools=allowed_tools,
        )
        with file_lock(self.vault_path):
            payload: Dict[str, Any] = {"version": 1, "secrets": {}}
            if self.vault_path.is_file():
                try:
                    cur = json.loads(self.vault_path.read_text(encoding="utf-8"))
                    if isinstance(cur, dict) and isinstance(cur.get("secrets"), dict):
                        payload["secrets"] = dict(cur["secrets"])
                except Exception:
                    pass
            spec = self._specs[str(name).strip()]
            rec: Dict[str, Any] = {
                "allowed_servers": list(spec.allowed_servers),
                "allowed_tools": list(spec.allowed_tools),
            }
            if spec.value is not None:
                rec["value"] = spec.value
            if spec.env_var is not None:
                rec["env_var"] = spec.env_var
            payload["secrets"][spec.name] = rec
            _atomic_write_json(self.vault_path, payload)

    @classmethod
    def find_handles(cls, obj: Any) -> List[str]:
        """Extract unique secret handle names referenced anywhere inside `obj`."""
        found: List[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, str):
                for m in _SECRET_HANDLE_RE.finditer(node):
                    hname = m.group(1) or m.group(2)
                    if hname and hname not in found:
                        found.append(hname)
            elif isinstance(node, dict):
                for v in node.values():
                    _walk(v)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    _walk(item)

        _walk(obj)
        return found

    def resolve_for_call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Validate scope and substitute `$SECRET:<NAME>` handles for downstream execution."""
        srv_low = str(server_name or "").strip().lower()
        tool_low = str(tool_name or "").strip().lower()
        args_obj = arguments if isinstance(arguments, dict) else {}
        hdr_obj = headers if isinstance(headers, dict) else {}

        handles = self.find_handles(args_obj) + [
            h for h in self.find_handles(hdr_obj) if h not in self.find_handles(args_obj)
        ]
        resolved_values: Dict[str, str] = {}
        for hname in handles:
            spec = self._specs.get(hname)
            if spec is None:
                return {
                    "ok": False,
                    "reason_code": "mcp_proxy_unknown_secret_handle",
                    "label": f"Unknown secret handle: {hname}",
                }
            if spec.allowed_servers and srv_low not in spec.allowed_servers:
                return {
                    "ok": False,
                    "reason_code": "mcp_proxy_secret_scope_violation",
                    "label": f"Secret handle {hname} is not authorized for server '{server_name}'",
                }
            if spec.allowed_tools and tool_low not in spec.allowed_tools:
                return {
                    "ok": False,
                    "reason_code": "mcp_proxy_secret_scope_violation",
                    "label": f"Secret handle {hname} is not authorized for tool '{server_name}:{tool_name}'",
                }
            secret_val = spec.resolve_value()
            if secret_val is None:
                return {
                    "ok": False,
                    "reason_code": "mcp_proxy_unknown_secret_handle",
                    "label": f"Secret handle {hname} has no bound value",
                }
            resolved_values[hname] = secret_val

        def _subst(node: Any) -> Any:
            if isinstance(node, str):
                return _SECRET_HANDLE_RE.sub(
                    lambda m: resolved_values.get(m.group(1) or m.group(2), m.group(0)),
                    node,
                )
            if isinstance(node, dict):
                return {k: _subst(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_subst(item) for item in node]
            return node

        return {
            "ok": True,
            "arguments": _subst(args_obj),
            "headers": _subst(hdr_obj) if headers is not None else None,
            "resolved_handles": list(resolved_values.keys()),
            "_resolved_values": list(resolved_values.values()),
        }

    def scrub_known_secrets(self, text: str) -> Tuple[str, int]:
        """Strip raw, base64, and URL-encoded occurrences of active vault values from `text`."""
        out = str(text or "")
        if not out:
            return out, 0
        hits = 0
        for spec in self._specs.values():
            val = spec.resolve_value()
            if not val or len(val) < 4:
                continue
            b64 = base64.b64encode(val.encode("utf-8")).decode("ascii")
            url_enc = _url_quote(val, safe="")
            for variant in (val, b64, url_enc):
                if variant and len(variant) >= 4 and variant in out:
                    cnt = out.count(variant)
                    out = out.replace(variant, "[REDACTED_VAULT_SECRET]")
                    hits += cnt
        return out, hits
