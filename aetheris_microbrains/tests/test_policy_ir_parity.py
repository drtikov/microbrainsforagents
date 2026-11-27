"""Tests for Single-Source Policy IR compiler, parity verifier, and modular Layer-0 facade."""

from __future__ import annotations

import ast
from pathlib import Path

from aetheris_microbrains.core import l0_policy
from aetheris_microbrains.core.layered_decision_engine import (
    DEFAULT_PROTECTED_PATTERNS,
    _AGENT_CONTROL_PLANE_SEGMENTS,
    _PROTECTED_EXACT_BASENAMES,
    _is_agent_control_plane_path,
)
from aetheris_microbrains.core.policy_ir import (
    load_policy_ir,
    render_rust_policy_tables,
    verify_policy_ir_parity,
)

_CORE_DIR = Path(__file__).resolve().parents[1] / "core"


def test_load_policy_ir_schema_and_hosts() -> None:
    ir = load_policy_ir()
    assert ir["version"] == "0.9.0"
    assert len(ir["supported_hosts"]) == 21
    assert "kimi" in ir["supported_hosts"]
    assert "claude" in ir["supported_hosts"]
    assert set(ir["control_plane_segments"]) == set(_AGENT_CONTROL_PLANE_SEGMENTS)
    assert set(ir["protected_exact_basenames"]) == set(_PROTECTED_EXACT_BASENAMES)
    assert ".env" in ir["protected_patterns"]
    assert "kimi.plugin.json" in ir["protected_patterns"]


def test_render_rust_policy_tables_valid_syntax() -> None:
    rendered = render_rust_policy_tables()
    assert "pub const SUPPORTED_HOSTS: &[&str] = &[" in rendered
    assert "pub const CONTROL_PLANE_SEGMENTS: &[&str] = &[" in rendered
    assert "pub const PROTECTED_EXACT_BASENAMES: &[&str] = &[" in rendered
    assert '"kimi",' in rendered
    assert '".kimi-code",' in rendered
    assert '"kimi.plugin.json",' in rendered
    assert rendered.count("];") == 3


def test_l0_policy_facade_reexports() -> None:
    assert l0_policy.DEFAULT_PROTECTED_PATTERNS is DEFAULT_PROTECTED_PATTERNS
    assert l0_policy._PROTECTED_EXACT_BASENAMES is _PROTECTED_EXACT_BASENAMES
    assert l0_policy._AGENT_CONTROL_PLANE_SEGMENTS is _AGENT_CONTROL_PLANE_SEGMENTS
    assert l0_policy._is_agent_control_plane_path is _is_agent_control_plane_path
    assert l0_policy.load_policy_ir is load_policy_ir
    assert l0_policy.verify_policy_ir_parity is verify_policy_ir_parity
    assert l0_policy._is_agent_control_plane_path(".kimi-code/config.toml") is True
    assert l0_policy._is_agent_control_plane_path(".env") is False


def test_zero_numpy_ast_imports_in_policy_ir_and_l0_facade() -> None:
    forbidden = {"numpy", "scipy", "torch", "pandas", "sklearn"}
    for filename in ("policy_ir.py", "l0_policy.py"):
        src = (_CORE_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in forbidden, f"{filename} imports forbidden module {root_mod}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in forbidden, f"{filename} imports forbidden module {root_mod}"


def test_verify_policy_ir_parity_ok() -> None:
    report = verify_policy_ir_parity()
    assert report["ok"] is True, f"Policy IR parity failed: {report}"
    for key in ("missing_in_ir", "missing_in_python", "missing_in_rust"):
        for section, items in report[key].items():
            assert items == [], f"{key}.{section} not empty: {items}"
