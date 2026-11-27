"""Tests for signed fleet policy bundles, attestations, and cross-harness A/B benchmark harness."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from aetheris_microbrains.core.benchmark_harness import run_cross_harness_ab_suite
from aetheris_microbrains.core.fleet_bundle import (
    build_fleet_attestation,
    create_signed_fleet_bundle,
    verify_signed_fleet_bundle,
)
from aetheris_microbrains.core.runtime_paths import append_jsonl_bounded

CORE_DIR = Path(__file__).resolve().parents[1] / "core"


def test_fleet_bundle_and_benchmark_harness_zero_numpy():
    for fname in ("fleet_bundle.py", "benchmark_harness.py"):
        src = (CORE_DIR / fname).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=fname)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "numpy", f"numpy imported in {fname}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "numpy", f"numpy imported in {fname}"


def test_signed_fleet_bundle_verify_and_fail_closed_cases():
    signing_key = "test-fleet-signing-secret-v1"
    bundle = create_signed_fleet_bundle(
        org_id="acme-sec",
        version=5,
        policy={"profile": "strict", "strict_ast_allowlist": True},
        mcp_pins={"github": {"sha256": "a" * 64}},
        signing_key=signing_key,
        ttl_sec=3600.0,
        issued_at=1_700_000_000.0,
    )

    ok = verify_signed_fleet_bundle(
        bundle,
        signing_key=signing_key,
        min_version=5,
        now=1_700_001_000.0,
    )
    assert ok["valid"] is True
    assert ok["reason"] == "ok"
    assert ok["org_id"] == "acme-sec"
    assert ok["version"] == 5
    assert ok["bundle_sha256"] == bundle["bundle_sha256"]

    # Policy tampering fails closed
    tampered_policy = copy.deepcopy(bundle)
    tampered_policy["policy"]["profile"] = "dev"
    res_tampered = verify_signed_fleet_bundle(
        tampered_policy,
        signing_key=signing_key,
        now=1_700_001_000.0,
    )
    assert res_tampered["valid"] is False
    assert res_tampered["reason"] == "invalid_signature"

    # Wrong signing key fails closed
    res_wrong_key = verify_signed_fleet_bundle(
        bundle,
        signing_key="wrong-key",
        now=1_700_001_000.0,
    )
    assert res_wrong_key["valid"] is False
    assert res_wrong_key["reason"] == "invalid_signature"

    # Expired bundle fails closed
    res_expired = verify_signed_fleet_bundle(
        bundle,
        signing_key=signing_key,
        now=1_700_003_601.0,
    )
    assert res_expired["valid"] is False
    assert res_expired["reason"] == "bundle_expired"

    # Rollback attempt (version < min_version) fails closed
    res_rollback = verify_signed_fleet_bundle(
        bundle,
        signing_key=signing_key,
        min_version=6,
        now=1_700_001_000.0,
    )
    assert res_rollback["valid"] is False
    assert res_rollback["reason"] == "version_rollback"


def test_build_fleet_attestation_with_hash_chained_journal(tmp_path: Path):
    signing_key = b"binary-fleet-key-material"
    bundle = create_signed_fleet_bundle(
        org_id="acme-sec",
        version=2,
        policy={"profile": "dev"},
        mcp_pins={"fs": {"sha256": "b" * 64}},
        signing_key=signing_key,
        ttl_sec=86400.0,
        issued_at=1_700_000_000.0,
    )

    journal_path = tmp_path / "decisions.jsonl"
    append_jsonl_bounded(
        journal_path,
        json.dumps({"tool": "execute_shell", "decision": "allow", "reason_code": "allowlist_ok"}),
        hash_chain=True,
    )
    append_jsonl_bounded(
        journal_path,
        json.dumps({"tool": "execute_shell", "decision": "deny", "reason_code": "layer1_rm_rf"}),
        hash_chain=True,
    )

    att = build_fleet_attestation(
        bundle,
        journal_path=journal_path,
        signing_key=signing_key,
        now=1_700_000_100.0,
    )
    assert att["valid"] is True
    assert att["reason"] == "ok"
    assert att["org_id"] == "acme-sec"
    assert att["version"] == 2
    assert att["bundle_sha256"] == bundle["bundle_sha256"]
    assert att["bundle_verification"]["valid"] is True
    assert att["journal_verification"]["ok"] is True
    assert att["journal_verification"]["chained_entries"] == 2
    assert att["journal_head_hash"] == att["journal_verification"]["head_hash"]
    assert len(att["attestation_sha256"]) == 64

    # Corrupting a chained journal entry causes attestation to fail closed
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    tampered_entry = json.loads(lines[0])
    tampered_entry["decision"] = "deny"
    lines[0] = json.dumps(tampered_entry)
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bad_att = build_fleet_attestation(
        bundle,
        journal_path=journal_path,
        signing_key=signing_key,
        now=1_700_000_100.0,
    )
    assert bad_att["valid"] is False
    assert bad_att["journal_verification"]["ok"] is False


def test_cross_harness_ab_suite_stage1_4():
    report = run_cross_harness_ab_suite(
        hosts=("claude", "qwen", "ouroboros", "prime"),
        trials=3,
    )
    assert report["hosts"] == ["claude", "qwen", "ouroboros", "prime"]
    agg = report["aggregate"]
    assert agg["tpr"] == 1.0
    assert agg["fpr"] == 0.0
    assert agg["e_ratio"] > 1.0
    assert agg["e_gated"] > agg["e_baseline"]
    assert agg["token_savings_pct"] > 0.0
    assert agg["modeled_gated_tokens"] < agg["modeled_baseline_tokens"]
    assert agg["latency_p50_ms"] <= agg["latency_p95_ms"] <= agg["latency_p99_ms"]
    assert agg["ci95_halfwidth_ms"] >= 0.0
    assert agg["ci95_low_ms"] <= agg["latency_mean_ms"] <= agg["ci95_high_ms"]

    for host_name in ("claude", "qwen", "ouroboros", "prime"):
        hm = report["per_host"][host_name]
        assert hm["tpr"] == 1.0, f"{host_name} tpr != 1.0"
        assert hm["fpr"] == 0.0, f"{host_name} fpr != 0.0"
        assert hm["e_ratio"] > 1.0, f"{host_name} e_ratio <= 1.0"
        assert hm["ci95_halfwidth_ms"] >= 0.0
