"""Tests for HostSpec registry parity, hook config compilation, and ASTDiffCache."""

from __future__ import annotations

import ast
import concurrent.futures
import time
from pathlib import Path
from typing import Set

import pytest

from aetheris_microbrains.core.ast_cache import (
    ASTDiffCache,
    get_default_ast_cache,
    make_cache_key,
)
from aetheris_microbrains.core.ast_inspector import analyze_code_diff
from aetheris_microbrains.hosts.registry import HOST_REGISTRY
from aetheris_microbrains.hosts.specs import (
    HOST_SPECS,
    HostSpec,
    compile_hook_config_from_spec,
    get_host_spec,
    validate_host_specs_parity,
)


def _extract_imports(source_path: Path) -> Set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_zero_numpy_and_stdlib_only_imports() -> None:
    pkg_root = Path(__file__).resolve().parents[1]
    specs_path = pkg_root / "hosts" / "specs.py"
    cache_path = pkg_root / "core" / "ast_cache.py"

    for path in (specs_path, cache_path):
        imports = _extract_imports(path)
        assert "numpy" not in imports
        assert "np" not in imports
        assert "torch" not in imports
        assert "scipy" not in imports


def test_host_specs_parity_with_registry() -> None:
    parity = validate_host_specs_parity()
    assert parity["ok"] is True
    assert parity["mismatches"] == []
    assert parity["spec_count"] == len(HOST_REGISTRY)
    assert parity["registry_count"] == len(HOST_REGISTRY)
    assert set(HOST_SPECS.keys()) == set(HOST_REGISTRY.keys())

    for name, adapter in HOST_REGISTRY.items():
        spec = get_host_spec(name)
        assert isinstance(spec, HostSpec)
        assert spec.contract == adapter.contract
        assert spec.family == adapter.family
        assert spec.supports_http_daemon == adapter.supports_http_daemon
        assert spec.supports_replace_tool_output == adapter.supports_replace_tool_output
        assert spec.plugin_exit_code_passthrough == adapter.plugin_exit_code_passthrough
        assert spec.returns_hso_directly_http == adapter.returns_hso_directly_http

    assert get_host_spec("  CLAUDE ") == HOST_SPECS["claude"]
    assert get_host_spec("nonexistent-host") is None


def test_compile_hook_config_from_spec_hosts() -> None:
    pre = 'python -m aetheris_microbrains.hook --host claude --profile dev'
    post = 'python -m aetheris_microbrains.hook --host claude --event post-tool-use'

    claude_cfg = compile_hook_config_from_spec("claude", pre_cmd=pre, post_cmd=post)
    assert claude_cfg["hooks"]["PreToolUse"][0]["hooks"][0]["onFailure"] == "block"
    assert "timeout" not in claude_cfg["hooks"]["PreToolUse"][0]["hooks"][0]
    assert claude_cfg["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == post

    qwen_cfg = compile_hook_config_from_spec("qwen", pre_cmd=pre, post_cmd=post)
    assert "run_shell_command" in qwen_cfg["hooks"]["PreToolUse"][0]["matcher"]
    assert qwen_cfg["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"] == 10
    assert "onFailure" not in qwen_cfg["hooks"]["PreToolUse"][0]["hooks"][0]

    qoder_cfg = compile_hook_config_from_spec("qoder", pre_cmd=pre, post_cmd=post)
    assert qoder_cfg["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"] == 10
    assert qoder_cfg["hooks"]["PostToolUse"][0]["hooks"][0]["timeout"] == 10

    mcode_ws = compile_hook_config_from_spec("mcode", pre_cmd=pre, post_cmd=post, is_workspace=True)
    assert "enabled" not in mcode_ws
    assert mcode_ws["hooks"]["PreToolUse"][0]["matcher"] == HOST_SPECS["mcode"].tool_matcher
    assert mcode_ws["hooks"]["PreToolUse"][0]["hooks"][0]["commandWindows"] == pre
    mcode_user = compile_hook_config_from_spec("mcode", pre_cmd=pre, post_cmd=post, is_workspace=False)
    assert mcode_user["enabled"] is True

    devin_cfg = compile_hook_config_from_spec("devin", pre_cmd=pre, post_cmd=post)
    assert devin_cfg["hooks"]["PreToolUse"][0]["matcher"].startswith("^(exec|")
    assert devin_cfg["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"] == 10

    cursor_cfg = compile_hook_config_from_spec("cursor", pre_cmd=pre, post_cmd=post)
    assert cursor_cfg["version"] == 1
    assert cursor_cfg["hooks"]["preToolUse"][0]["failClosed"] is True
    assert cursor_cfg["hooks"]["postToolUse"][0]["failClosed"] is False
    assert "beforeShellExecution" in cursor_cfg["hooks"]
    assert "afterFileEdit" in cursor_cfg["hooks"]

    pre_only = compile_hook_config_from_spec("claude", pre_cmd=pre)
    assert "PreToolUse" in pre_only["hooks"]
    assert "PostToolUse" not in pre_only["hooks"]

    with pytest.raises(ValueError):
        compile_hook_config_from_spec("unknown_host_xyz", pre_cmd=pre)


def test_ast_cache_deep_copy_isolation() -> None:
    cache = ASTDiffCache(max_entries=8)
    key = make_cache_key("src/app.py", "+x = 1", original_source="x = 0", policy_fingerprint="p1")
    original_payload = {
        "language": "python",
        "syntax_valid": True,
        "findings": ["f1"],
        "nested": {"items": [1, 2]},
    }
    cache.put(key, original_payload)

    # Mutating caller's dict after put must not affect cached entry.
    original_payload["findings"].append("mutated_after_put")
    original_payload["nested"]["items"].append(999)

    fetched_a = cache.get(key)
    assert fetched_a is not None
    assert fetched_a["findings"] == ["f1"]
    assert fetched_a["nested"]["items"] == [1, 2]

    # Mutating returned dict from get must not affect subsequent gets.
    fetched_a["findings"].append("mutated_after_get")
    fetched_a["nested"]["items"].append(888)

    fetched_b = cache.get(key)
    assert fetched_b is not None
    assert fetched_b["findings"] == ["f1"]
    assert fetched_b["nested"]["items"] == [1, 2]


def test_ast_cache_lru_eviction_and_stats() -> None:
    cache = ASTDiffCache(max_entries=2)
    k1 = make_cache_key("a.py", "+a = 1")
    k2 = make_cache_key("b.py", "+b = 2")
    k3 = make_cache_key("c.py", "+c = 3")

    assert k1 != k2 != k3
    assert len(k1) == 64

    # Distinguish original_source=None vs original_source=""
    assert make_cache_key("a.py", "+a = 1", original_source=None) != make_cache_key(
        "a.py", "+a = 1", original_source=""
    )

    assert cache.get(k1) is None  # miss 1
    cache.put(k1, {"id": 1})
    cache.put(k2, {"id": 2})

    # Access k1 so k2 becomes least-recently-used.
    assert cache.get(k1) == {"id": 1}  # hit 1
    cache.put(k3, {"id": 3})  # evicts k2

    assert cache.get(k2) is None  # miss 2
    assert cache.get(k1) == {"id": 1}  # hit 2
    assert cache.get(k3) == {"id": 3}  # hit 3

    st = cache.stats()
    assert st["hits"] == 3
    assert st["misses"] == 2
    assert st["size"] == 2
    assert st["max_entries"] == 2
    assert st["hit_rate"] == pytest.approx(3 / 5)

    cache.clear()
    st_cleared = cache.stats()
    assert st_cleared == {"hits": 0, "misses": 0, "size": 0, "max_entries": 2, "hit_rate": 0.0}


def test_ast_cache_thread_safety_and_default_singleton() -> None:
    assert get_default_ast_cache() is get_default_ast_cache()

    cache = ASTDiffCache(max_entries=32)

    def _worker(idx: int) -> None:
        key = make_cache_key(f"mod_{idx % 10}.py", f"+val = {idx % 10}")
        cache.put(key, {"idx": idx % 10, "findings": []})
        got = cache.get(key)
        assert got is not None
        assert got["idx"] == idx % 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_worker, range(120)))

    st = cache.stats()
    assert st["size"] <= 32
    assert st["hits"] == 120


def test_ast_cache_speedup_on_450_line_python_diff() -> None:
    lines = ["def compute_item(x: int) -> int:", "    total = 0"]
    for i in range(446):
        lines.append(f"    total += (x + {i}) % 7")
    lines.append("    return total")
    diff_text = "\n".join(f"+{line}" for line in lines)
    assert len(lines) == 449 or len(diff_text.splitlines()) >= 449
    # Pad to at least 450 lines.
    diff_text = "+# header comment\n" + diff_text
    assert len(diff_text.splitlines()) == 450

    cache = ASTDiffCache(max_entries=16)
    key = make_cache_key("big_module.py", diff_text, original_source=None, allow_suppression=True)

    t0 = time.perf_counter()
    cold_result = analyze_code_diff("big_module.py", diff_text, allow_suppression=True)
    cache.put(key, cold_result)
    cold_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    warm_result = cache.get(key)
    warm_elapsed = time.perf_counter() - t1

    assert warm_result == cold_result
    assert cold_result["syntax_valid"] is True
    assert cold_result["security_clean"] is True
    assert warm_elapsed < cold_elapsed
