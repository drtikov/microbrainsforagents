"""Tests for the DeepSeek Harness (DSH) Cordis MCP bundle and verification helper."""

from __future__ import annotations

import json
from pathlib import Path

from aetheris_microbrains.install_dsh import BUNDLE_DIR_NAME, PACKAGE_NAME, self_test


def test_dsh_bundle_structure_and_sanitization():
    repo_root = Path(__file__).resolve().parents[2]
    bundle_dir = repo_root / BUNDLE_DIR_NAME

    required = (
        "package.json",
        "cordis.patch.yml",
        "README.md",
        "icon.svg",
        "locale/en.json",
        "docs/DETAIL.md",
        "docs/preview.svg",
    )
    for rel in required:
        assert (bundle_dir / rel).is_file(), f"Missing DSH bundle file: {rel}"

    pkg = json.loads((bundle_dir / "package.json").read_text(encoding="utf-8"))
    assert pkg["name"] == PACKAGE_NAME
    assert pkg["dsh"]["bundle"]["patch"] == "./cordis.patch.yml"

    patch_txt = (bundle_dir / "cordis.patch.yml").read_text(encoding="utf-8")
    assert "serverName: microbrains" in patch_txt
    assert "transport: streamable-http" in patch_txt
    assert "url: http://127.0.0.1:8766/mcp" in patch_txt
    assert "failOnStartupError: false" in patch_txt

    for rel in required:
        low = (bundle_dir / rel).read_text(encoding="utf-8", errors="ignore").lower()
        assert "i:\\" not in low and "i:/" not in low
        assert "c:\\users\\" not in low and "c:/users/" not in low

    bench_low = (repo_root / "benchmark_dsh_mcp_ab.py").read_text(encoding="utf-8").lower()
    assert "i:\\" not in bench_low and "c:\\users\\" not in bench_low

    assert self_test(bundle_dir=bundle_dir) == 0
