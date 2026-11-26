"""Central HostAdapter registry for multi-agent normalization & wire contracts (stdlib-only)."""

from __future__ import annotations

import os
from typing import Dict, FrozenSet, Optional, Tuple

from .base import HostAdapter

HOST_REGISTRY: Dict[str, HostAdapter] = {}


def register_host(adapter: HostAdapter) -> HostAdapter:
    """Register or override a host adapter in ``HOST_REGISTRY``."""
    HOST_REGISTRY[adapter.name.strip().lower()] = adapter
    return adapter


_BUILTIN_ADAPTERS: Tuple[HostAdapter, ...] = (
    HostAdapter(
        name="claude",
        contract="claude_pretooluse_v1",
        family="claude_compat",
        supports_http_daemon=True,
        supports_replace_tool_output=True,
    ),
    HostAdapter(
        name="codex",
        contract="codex_pretooluse_no_ask_v1",
        family="codex",
    ),
    HostAdapter(
        name="hermes",
        contract="hermes_shell_hook_v1",
        family="hermes",
    ),
    HostAdapter(
        name="antigravity",
        contract="antigravity_pretooluse_v1",
        family="antigravity",
        returns_hso_directly_http=True,
    ),
    HostAdapter(
        name="openclaw",
        contract="openclaw_before_tool_call_v1",
        family="openclaw",
        supports_http_daemon=True,
        returns_hso_directly_http=True,
    ),
    HostAdapter(
        name="opencode",
        contract="opencode_plugin_v1",
        family="plugin_block",
        supports_http_daemon=True,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
    ),
    HostAdapter(
        name="kilocode",
        contract="kilocode_plugin_v1",
        family="plugin_block",
        supports_http_daemon=True,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
    ),
    HostAdapter(
        name="pi",
        contract="pi_extension_v1",
        family="plugin_block",
        supports_http_daemon=True,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
    ),
    HostAdapter(
        name="omp",
        contract="omp_extension_v1",
        family="plugin_block",
        supports_http_daemon=True,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
    ),
    HostAdapter(
        name="qoder",
        contract="qoder_pretooluse_v1",
        family="claude_compat",
        supports_http_daemon=True,
        supports_replace_tool_output=True,
    ),
    HostAdapter(
        name="qwen",
        contract="qwen_pretooluse_v1",
        family="claude_compat",
        supports_http_daemon=True,
        supports_replace_tool_output=False,
    ),
    HostAdapter(
        name="zcode",
        contract="zcode_pretooluse_v1",
        family="claude_compat",
        supports_http_daemon=True,
        supports_replace_tool_output=True,
    ),
    HostAdapter(
        name="openhands",
        contract="openhands_hook_v1",
        family="claude_compat",
        supports_http_daemon=True,
        supports_replace_tool_output=True,
    ),
    HostAdapter(
        name="cursor",
        contract="cursor_hook_v2",
        family="cursor",
        supports_replace_tool_output=True,
    ),
    HostAdapter(
        name="devin",
        contract="devin_hook_v1",
        family="devin",
        supports_replace_tool_output=False,
    ),
    HostAdapter(
        name="cline",
        contract="cline_hook_v1",
        family="cline",
        supports_replace_tool_output=False,
    ),
    HostAdapter(
        name="mcode",
        contract="mcode_pretooluse_v1",
        family="claude_compat",
        supports_http_daemon=True,
        supports_replace_tool_output=False,
    ),
    HostAdapter(
        name="ouroboros",
        contract="ouroboros_registry_v1",
        family="claude_compat",
        supports_http_daemon=True,
        supports_replace_tool_output=True,
    ),
    HostAdapter(
        name="generic",
        contract="generic_v1",
        family="generic",
    ),
)

for _adapter in _BUILTIN_ADAPTERS:
    register_host(_adapter)


def get_host_adapter(host: Optional[str], default: str = "claude") -> HostAdapter:
    """Return the registered ``HostAdapter`` for ``host``, falling back to ``default``."""
    key = str(host or default).strip().lower()
    if key == "auto":
        key = "devin" if str(os.environ.get("DEVIN_PROJECT_DIR") or "").strip() else default
    if key in HOST_REGISTRY:
        return HOST_REGISTRY[key]
    return HOST_REGISTRY[default]


def resolve_host(host: Optional[str], default: str = "claude") -> str:
    """Normalize a host identifier to a registered name."""
    return get_host_adapter(host, default=default).name


def list_hosts() -> Tuple[str, ...]:
    """Return registered host names in registration order."""
    return tuple(HOST_REGISTRY.keys())


SUPPORTED_HOOK_HOSTS: FrozenSet[str] = frozenset(HOST_REGISTRY.keys())
HTTP_DAEMON_HOSTS: FrozenSet[str] = frozenset(k for k, v in HOST_REGISTRY.items() if v.supports_http_daemon)
PLUGIN_BLOCK_HOSTS: FrozenSet[str] = frozenset(k for k, v in HOST_REGISTRY.items() if v.family == "plugin_block")
HTTP_DIRECT_HSO_HOSTS: FrozenSet[str] = frozenset(k for k, v in HOST_REGISTRY.items() if v.returns_hso_directly_http)
REPLACE_OUTPUT_HOSTS: FrozenSet[str] = frozenset(k for k, v in HOST_REGISTRY.items() if v.supports_replace_tool_output)
POST_CLAUDE_COMPAT_HOSTS: FrozenSet[str] = frozenset(
    k for k, v in HOST_REGISTRY.items() if v.family in {"claude_compat", "codex", "cursor", "devin"}
)
