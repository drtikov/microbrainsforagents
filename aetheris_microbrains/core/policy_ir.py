"""Single-source Policy IR loader, Rust table compiler, and Python/Rust parity verifier (stdlib-only)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

_DEFAULT_IR_PATH = Path(__file__).resolve().parent / "policy_ir.json"
_RUST_LAYER0_REL = Path("rust") / "aetheris-gate" / "src" / "layer0.rs"

_QUOTED_STR_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_FN_SUPPORTED_HOST_RE = re.compile(
    r"pub\s+fn\s+is_supported_host\b.*?matches!\s*\((.*?)\)\s*\}",
    re.DOTALL,
)
_FN_CONTROL_PLANE_RE = re.compile(
    r"fn\s+is_control_plane_segment\b.*?matches!\s*\((.*?)\)\s*\}",
    re.DOTALL,
)
_CONST_EXACT_BASENAMES_RE = re.compile(
    r"const\s+EXACT_BASENAMES\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\];",
    re.DOTALL,
)


def load_policy_ir(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and validate the canonical declarative Layer-0 Policy IR JSON."""
    ir_path = Path(path).expanduser().resolve() if path is not None else _DEFAULT_IR_PATH
    raw = json.loads(ir_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Policy IR must be a JSON object, got {type(raw).__name__}")
    required = (
        "version",
        "supported_hosts",
        "control_plane_segments",
        "protected_exact_basenames",
        "protected_patterns",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Policy IR missing required field(s): {', '.join(missing)}")
    return raw


def _format_rust_const_array(name: str, items: Iterable[str]) -> str:
    lines = [f"pub const {name}: &[&str] = &["]
    for item in items:
        lines.append(f"    {json.dumps(str(item))},")
    lines.append("];")
    return "\n".join(lines)


def render_rust_policy_tables(ir: Optional[Dict[str, Any]] = None) -> str:
    """Compile Policy IR tables into valid Rust ``pub const`` slice definitions."""
    data = load_policy_ir() if ir is None else ir
    version = str(data.get("version") or "unknown")
    sections = [
        f"//! Generated from aetheris_microbrains/core/policy_ir.json (version {version}).",
        "//! Do not edit by hand; regenerate via aetheris_microbrains.core.policy_ir.",
        "",
        _format_rust_const_array("SUPPORTED_HOSTS", data["supported_hosts"]),
        "",
        _format_rust_const_array("CONTROL_PLANE_SEGMENTS", data["control_plane_segments"]),
        "",
        _format_rust_const_array("PROTECTED_EXACT_BASENAMES", data["protected_exact_basenames"]),
        "",
    ]
    return "\n".join(sections)


def _extract_rust_literals(pattern: re.Pattern[str], source: str, label: str) -> Set[str]:
    match = pattern.search(source)
    if not match:
        raise ValueError(f"Could not locate Rust table {label!r} in layer0.rs")
    body = match.group(1)
    # Strip line comments before extracting string literals.
    cleaned = re.sub(r"//[^\n]*", "", body)
    return set(_QUOTED_STR_RE.findall(cleaned))


def _resolve_repo_root(repo_root: Optional[Union[str, Path]]) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    # aetheris_microbrains/core/policy_ir.py -> repo root is 2 parents up.
    return Path(__file__).resolve().parents[2]


def verify_policy_ir_parity(
    repo_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify parity across policy_ir.json, Python registries/Layer-0, and Rust layer0.rs."""
    from ..hosts.registry import HOST_REGISTRY
    from .layered_decision_engine import (
        _AGENT_CONTROL_PLANE_SEGMENTS,
        _PROTECTED_EXACT_BASENAMES,
    )

    root = _resolve_repo_root(repo_root)
    ir_path = root / "aetheris_microbrains" / "core" / "policy_ir.json"
    ir = load_policy_ir(ir_path if ir_path.is_file() else None)

    ir_hosts: Set[str] = {str(x) for x in ir["supported_hosts"]}
    ir_ctrl: Set[str] = {str(x) for x in ir["control_plane_segments"]}
    ir_base: Set[str] = {str(x) for x in ir["protected_exact_basenames"]}

    rust_ext = ir.get("rust_superset_extensions") or {}
    ext_ctrl: Set[str] = {str(x) for x in rust_ext.get("control_plane_segments", ())}
    ext_base: Set[str] = {str(x) for x in rust_ext.get("protected_exact_basenames", ())}

    py_hosts: Set[str] = set(HOST_REGISTRY.keys())
    py_ctrl: Set[str] = set(_AGENT_CONTROL_PLANE_SEGMENTS)
    py_base: Set[str] = set(_PROTECTED_EXACT_BASENAMES)

    rust_path = root / _RUST_LAYER0_REL
    if not rust_path.is_file():
        return {
            "ok": False,
            "error": f"Rust layer0.rs not found at {rust_path}",
            "missing_in_ir": {
                "supported_hosts": sorted(py_hosts - ir_hosts),
                "control_plane_segments": sorted(py_ctrl - ir_ctrl),
                "protected_exact_basenames": sorted(py_base - ir_base),
            },
            "missing_in_python": {
                "supported_hosts": sorted(ir_hosts - py_hosts),
                "control_plane_segments": sorted(ir_ctrl - py_ctrl),
                "protected_exact_basenames": sorted(ir_base - py_base),
            },
            "missing_in_rust": {
                "supported_hosts": sorted(ir_hosts),
                "control_plane_segments": sorted(ir_ctrl),
                "protected_exact_basenames": sorted(ir_base),
            },
        }

    rust_src = rust_path.read_text(encoding="utf-8")
    rs_hosts = _extract_rust_literals(_FN_SUPPORTED_HOST_RE, rust_src, "is_supported_host")
    rs_ctrl = _extract_rust_literals(_FN_CONTROL_PLANE_RE, rust_src, "is_control_plane_segment")
    rs_base = _extract_rust_literals(_CONST_EXACT_BASENAMES_RE, rust_src, "EXACT_BASENAMES")

    missing_in_ir: Dict[str, List[str]] = {
        "supported_hosts": sorted((py_hosts | rs_hosts) - ir_hosts),
        "control_plane_segments": sorted((py_ctrl | (rs_ctrl - ext_ctrl)) - ir_ctrl),
        "protected_exact_basenames": sorted((py_base | (rs_base - ext_base)) - ir_base),
    }
    missing_in_python: Dict[str, List[str]] = {
        "supported_hosts": sorted(ir_hosts - py_hosts),
        "control_plane_segments": sorted(ir_ctrl - py_ctrl),
        "protected_exact_basenames": sorted(ir_base - py_base),
    }
    missing_in_rust: Dict[str, List[str]] = {
        "supported_hosts": sorted(ir_hosts - rs_hosts),
        "control_plane_segments": sorted(ir_ctrl - rs_ctrl),
        "protected_exact_basenames": sorted(ir_base - rs_base),
    }

    ok = not any(
        items
        for group in (missing_in_ir, missing_in_python, missing_in_rust)
        for items in group.values()
    )
    return {
        "ok": ok,
        "missing_in_ir": missing_in_ir,
        "missing_in_python": missing_in_python,
        "missing_in_rust": missing_in_rust,
    }


__all__ = [
    "load_policy_ir",
    "render_rust_policy_tables",
    "verify_policy_ir_parity",
]
