"""Base data model for Aetheris Microbrains host adapters (stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostAdapter:
    """Declarative capability & wire-contract descriptor for a coding agent host."""

    name: str
    contract: str
    family: str = "claude_compat"
    supports_http_daemon: bool = False
    supports_replace_tool_output: bool = False
    plugin_exit_code_passthrough: bool = False
    returns_hso_directly_http: bool = False
