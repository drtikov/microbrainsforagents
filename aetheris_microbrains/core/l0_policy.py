"""Modular Layer-0 policy facade (stdlib-only)."""

from __future__ import annotations

from .layered_decision_engine import (
    DEFAULT_PROTECTED_PATTERNS,
    _AGENT_CONTROL_PLANE_SEGMENTS,
    _PROTECTED_EXACT_BASENAMES,
    _is_agent_control_plane_path,
)
from .policy_ir import (
    load_policy_ir,
    verify_policy_ir_parity,
)

__all__ = [
    "DEFAULT_PROTECTED_PATTERNS",
    "_PROTECTED_EXACT_BASENAMES",
    "_AGENT_CONTROL_PLANE_SEGMENTS",
    "_is_agent_control_plane_path",
    "load_policy_ir",
    "verify_policy_ir_parity",
]
