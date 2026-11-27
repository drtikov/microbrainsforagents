"""Unit tests for Deterministic PostToolUse Structural Output Compressor."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from aetheris_microbrains.core.economics_ledger import estimate_tokens
from aetheris_microbrains.core.output_compressor import compress_tool_output


def test_small_output_passthrough() -> None:
    sample = "tests/test_smoke.py::test_ok PASSED\n==== 1 passed in 0.05s ===="
    res = compress_tool_output(sample, tool_name="bash", min_chars=1200)

    assert res["compressed"] is False
    assert res["text"] == sample
    assert res["original_chars"] == len(sample)
    assert res["compressed_chars"] == len(sample)
    assert res["original_tokens"] == estimate_tokens(sample)
    assert res["compressed_tokens"] == res["original_tokens"]
    assert res["tokens_saved"] == 0
    assert res["savings_pct"] == 0.0
    assert res["artifact_path"] is None
    assert res["strategy"] == "passthrough"


def test_large_passing_pytest_log_reduction_and_summary() -> None:
    lines = [
        "==== test session starts ====",
        "platform win32 -- Python 3.12.0, pytest-8.3.0",
        "collected 180 items",
        "",
    ]
    for idx in range(1, 181):
        pct = int((idx / 180) * 100)
        lines.append(f"tests/test_suite_alpha.py::test_case_{idx:03d} PASSED [{pct:3d}%]")
    lines.append("")
    lines.append("==== 180 passed in 2.34s ====")
    raw = "\n".join(lines)

    res = compress_tool_output(raw, tool_name="pytest", min_chars=1200, max_output_chars=2400)

    assert res["compressed"] is True
    assert "test_log_distill" in res["strategy"]
    assert res["savings_pct"] > 50.0
    assert res["tokens_saved"] > 0
    assert res["compressed_tokens"] < res["original_tokens"]
    assert "==== 180 passed in 2.34s ====" in res["text"]
    assert "180 passing/progress lines collapsed" in res["text"]


def test_failing_pytest_preserves_assertion_and_failed_verbatim() -> None:
    lines = [
        "==== test session starts ====",
        "collected 121 items",
        "",
    ]
    for idx in range(1, 80):
        lines.append(f"tests/test_core.py::test_pass_{idx:03d} PASSED")
    lines.append("tests/test_core.py::test_critical_invariant FAILED")
    for idx in range(80, 121):
        lines.append(f"tests/test_core.py::test_pass_{idx:03d} PASSED")
    lines.extend(
        [
            "",
            "==== FAILURES ====",
            "____ test_critical_invariant ____",
            "Traceback (most recent call last):",
            '  File "tests/test_core.py", line 42, in test_critical_invariant',
            "    assert result == 42",
            "E       AssertionError: expected 42, got 0",
            "==== short test summary info ====",
            "FAILED tests/test_core.py::test_critical_invariant - AssertionError: expected 42, got 0",
            "==== 1 failed, 120 passed in 1.85s ====",
        ]
    )
    raw = "\n".join(lines)

    res = compress_tool_output(raw, tool_name="pytest", min_chars=1200, max_output_chars=2400)

    assert res["compressed"] is True
    assert "test_log_distill" in res["strategy"]
    assert "tests/test_core.py::test_critical_invariant FAILED" in res["text"]
    assert "Traceback (most recent call last):" in res["text"]
    assert "E       AssertionError: expected 42, got 0" in res["text"]
    assert "FAILED tests/test_core.py::test_critical_invariant - AssertionError: expected 42, got 0" in res["text"]
    assert "==== 1 failed, 120 passed in 1.85s ====" in res["text"]
    assert "passing/progress lines collapsed" in res["text"]
    assert res["savings_pct"] > 50.0


def test_repeated_compiler_warnings_deduplication() -> None:
    warning_a = "src/engine/parser.c:118:12: warning: implicit conversion loses integer precision [-Wshorten-64-to-32]"
    warning_b = "src/engine/lexer.c:45:9: warning: unused variable 'scratch_buf' [-Wunused-variable]"

    lines = ["Building target engine_core..."]
    lines.extend([warning_a] * 35)
    for _ in range(10):
        lines.append(warning_b)
        lines.append("src/engine/ast.c:99:4: note: expanded from macro 'CHECK_NODE'")
    lines.append("Build finished with 45 warnings.")
    raw = "\n".join(lines)

    res = compress_tool_output(raw, tool_name="gcc", min_chars=1200, max_output_chars=2400)

    assert res["compressed"] is True
    assert "dedup" in res["strategy"]
    assert f"{warning_a}  [repeated 35\u00d7]" in res["text"]
    assert f"{warning_b}  [repeated 10\u00d7]" in res["text"]
    assert "Build finished with 45 warnings." in res["text"]
    assert res["tokens_saved"] > 0


def test_artifact_persistence_to_tmp_path(tmp_path: Path) -> None:
    raw = "\n".join(f"log entry {i:04d}: payload={'x' * 40}" for i in range(120))
    expected_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    res = compress_tool_output(
        raw,
        tool_name="shell",
        min_chars=1200,
        max_output_chars=1600,
        state_dir=tmp_path,
        store_raw=True,
    )

    assert res["compressed"] is True
    assert "head_tail" in res["strategy"]
    assert res["artifact_path"] is not None

    artifact_file = Path(res["artifact_path"])
    assert artifact_file == tmp_path / "compressed_outputs" / f"{expected_digest}.log"
    assert artifact_file.is_file()
    assert artifact_file.read_text(encoding="utf-8") == raw
    assert f"[Full raw output saved to {artifact_file}]" in res["text"]
    assert res["compressed_chars"] <= 1600
    assert res["tokens_saved"] > 0


def test_zero_numpy_and_pure_stdlib_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[1] / "core" / "output_compressor.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported_roots.add(node.module.split(".")[0])

    assert "numpy" not in imported_roots
    assert "scipy" not in imported_roots
    assert "torch" not in imported_roots
    assert imported_roots <= {
        "__future__",
        "collections",
        "hashlib",
        "pathlib",
        "re",
        "typing",
        "aetheris_microbrains",
    }
