"""Host adapter registry package (stdlib-only)."""

from .base import HostAdapter
from .registry import (
    HOST_REGISTRY,
    HTTP_DAEMON_HOSTS,
    HTTP_DIRECT_HSO_HOSTS,
    PLUGIN_BLOCK_HOSTS,
    POST_CLAUDE_COMPAT_HOSTS,
    REPLACE_OUTPUT_HOSTS,
    SUPPORTED_HOOK_HOSTS,
    get_host_adapter,
    list_hosts,
    register_host,
    resolve_host,
)

__all__ = [
    "HostAdapter",
    "HOST_REGISTRY",
    "HTTP_DAEMON_HOSTS",
    "HTTP_DIRECT_HSO_HOSTS",
    "PLUGIN_BLOCK_HOSTS",
    "POST_CLAUDE_COMPAT_HOSTS",
    "REPLACE_OUTPUT_HOSTS",
    "SUPPORTED_HOOK_HOSTS",
    "get_host_adapter",
    "list_hosts",
    "register_host",
    "resolve_host",
]
