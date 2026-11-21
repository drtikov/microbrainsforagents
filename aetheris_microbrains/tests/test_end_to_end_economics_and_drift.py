"""End-to-End Economics, 42-Model Fleet Calibration Audit, Signature Counterfactual,
Sliding-Window Drift Monitoring & ACI Quantile Adaptation Suite.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import aetheris_microbrains as mb
from aetheris_microbrains.core.analytical_micro_transformer import AnalyticalMicroTransformer
from aetheris_microbrains.core.layered_decision_engine import analyze_code_diff

PKG_DIR = Path(__file__).resolve().parents[1]
BRAINS_DIR = PKG_DIR / "memory" / "store" / "neuro_brains"


def test_all_42_fleet_models_calibrated_at_alpha_005_without_synthetic_q005():
    metas = sorted(BRAINS_DIR.glob("*_meta.json"))
    assert len(metas) >= 40
    assert not any("identity_probe" in mf.name or mf.name.endswith("_test_meta.json") for mf in metas)

    q_hats = set()
    for meta_file in metas:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        name = data["name"]
        assert data.get("conformal_alpha") == 0.05, f"{name} alpha != 0.05"
        assert data.get("n_calibration_samples", 0) >= 20, f"{name} n_cal < 20"
        assert data.get("conformal_guarantee_valid") is True, f"{name} guarantee invalid"
        assert data.get("conformal_basis") == "held_out", f"{name} not held_out"
        assert data.get("hopfield_rules_count", 0) >= 12, f"{name} insufficient hopfield rules"
        assert len(data.get("vocab", {})) >= 25, f"{name} vocab too small"
        assert len(data.get("per_class_quantiles", {})) >= 1, f"{name} missing Mondrian quantiles"
        assert data.get("weights_sha256"), f"{name} missing SHA-256 checksum"

        # Verify bijective vocab <-> inv_vocab
        vocab = data["vocab"]
        inv_vocab = {int(k): v for k, v in data["inv_vocab"].items()}
        for idx, tok in inv_vocab.items():
            assert vocab.get(tok) == idx, f"{name} vocab/inv_vocab mismatch at {idx}: {tok}"

        # Reject legacy hardcoded synthetic arithmetic-progression scores [0.001, 0.0018, 0.0026...]
        scores = data.get("conformal_scores", [])
        assert len(scores) >= 20
        if len(scores) >= 4:
            diffs = [round(scores[i + 1] - scores[i], 6) for i in range(3)]
            assert not (diffs[0] == diffs[1] == diffs[2] == 0.0008), f"{name} still has synthetic linear scores!"

        q_hats.add(round(float(data["conformal_quantile"]), 4))
        loaded = AnalyticalMicroTransformer.load(name, BRAINS_DIR, verify_checksum=True)
        assert loaded is not None

    assert len(q_hats) >= 15, f"Too few distinct empirically calibrated quantiles: {q_hats}"


def test_counterfactual_catches_signature_breaking_and_symbol_removal_on_short_diff():
    sig_break_diff = (
        "--- a/src/math_ops.py\n"
        "+++ b/src/math_ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def compute(a, b, c):\n"
        "+def compute(a):\n"
        "     return a * 2\n"
    )
    analysis = analyze_code_diff("src/math_ops.py", sig_break_diff)
    assert "breaking_signature_change:compute" in analysis["interface_hazards"]

    sim = mb.get_counterfactual_pipeline()
    outcome = sim.simulate_edit_impact("src/math_ops.py", sig_break_diff)
    assert outcome.allowed is False
    assert outcome.status == "VETOED_HIGH_RISK_BLAST"
    assert any("breaking_signature_change:compute" in m for m in outcome.simulated_failure_modes)
    assert any("Causal SCM Rollout" in s for s in outcome.step_predictions)

    removal_diff = (
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def verify_session(tok):\n"
        "-    return bool(tok)\n"
        "+def _helper_only():\n"
        "+    return True\n"
    )
    rem_analysis = analyze_code_diff("src/auth.py", removal_diff)
    assert "public_symbol_removed:verify_session" in rem_analysis["interface_hazards"]


def test_sliding_window_drift_detection_and_aci_quantile_adaptation(tmp_path: Path):
    log_p = tmp_path / "tel.jsonl"
    dash_p = tmp_path / "dash.json"
    tel = mb.MicrobrainTelemetry(log_path=log_p, dashboard_path=dash_p)

    # 15 healthy in-distribution events
    for _ in range(15):
        tel.record_event(
            domain="fast_intent_classifier",
            latency_ms=0.4,
            confidence=0.95,
            verified_accurate=True,
            ood_score=0.05,
            passed_conformal=True,
            escalated=False,
        )
    healthy = tel.generate_dashboard(window_size=15)
    assert healthy.drift_detected is False
    assert healthy.status == "healthy"
    assert healthy.local_resolution_rate == 1.0
    assert healthy.local_error_rate == 0.0

    # Shift: 20 recent drifting / miscovered local events trigger window drift alarm
    for i in range(20):
        tel.record_event(
            domain="fast_intent_classifier",
            latency_ms=0.6,
            confidence=0.80,
            verified_accurate=(i % 3 == 0),
            ood_score=0.55,
            passed_conformal=False,
            escalated=(i % 4 == 0),
        )
    shifted = tel.generate_dashboard(window_size=20)
    assert shifted.drift_detected is True
    assert shifted.status == "drift_detected"
    assert shifted.sliding_window_coverage < 0.50
    assert shifted.escalation_rate == pytest.approx(0.25, abs=0.01)

    # Gibbs-Candes ACI step: increase q on error, decrease slightly on hit
    q0 = 0.05
    q_err = tel.update_aci_quantile(q0, had_error=True, alpha=0.05, gamma=0.05)
    q_hit = tel.update_aci_quantile(q0, had_error=False, alpha=0.05, gamma=0.05)
    assert q_err > q0
    assert q_hit < q0

    recal = tel.recalibrate_from_window([0.01 + 0.002 * i for i in range(25)], alpha=0.05)
    assert recal["conformal_guarantee_valid"] is True
    assert recal["coverage_std_error"] > 0.0


def test_consult_failing_test_and_bilingual_swe_queries():
    res_fail = mb.microbrain_consult(text="fix the failing test in the parser module")
    assert res_fail["success"] is True
    assert res_fail["abstained"] is False
    assert res_fail["predicted_token"] in ("run_tests", "debug")
    assert res_fail["predicted_token"] != "repo_write"
    assert len(res_fail["prediction_set"]) >= 1

    res_ru = mb.microbrain_consult(text="исправь падающий тест в парсере")
    assert res_ru["success"] is True
    assert res_ru["abstained"] is False
    assert res_ru["predicted_token"] in ("run_tests", "debug")

    for ood_q in ("закажи такси до дома", "summarize what we discussed yesterday"):
        res_ood = mb.microbrain_consult(text=ood_q)
        assert res_ood["abstained"] is True
        assert res_ood["prediction_set"] == []


def test_end_to_end_token_cost_and_latency_economics():
    events = [
        {"kind": "decision", "input": {"tool": "read_file", "path": "README.md"}, "expect": "allow"},
        {"kind": "decision", "input": {"tool": "execute_shell", "cmd": "ls -la"}, "expect": "allow"},
        {"kind": "decision", "input": {"tool": "execute_shell", "cmd": "git log --format '%h'"}, "expect": "allow"},
        {"kind": "decision", "input": {"tool": "edit_file", "path": "src/a.py", "diff": "x = 1 + 2\n"}, "expect": "allow"},
        {"kind": "decision", "input": {"tool": "edit_file", "path": "src/bad.py", "diff": "os.system(u)\n"}, "expect": "deny"},
        {"kind": "decision", "input": {"tool": "execute_shell", "cmd": "rm -f -r ~"}, "expect": "deny"},
        {"kind": "decision", "input": {"query": "inspect file contents and write safe patch to repository"}, "expect": "allow"},
        {"kind": "decision", "input": {"query": "покажи текущую ветку git и статус"}, "expect": "allow"},
        {"kind": "decision", "input": {"tool": "execute_shell", "cmd": "custom_deploy_binary --prod"}, "expect": "escalate"},
        {"kind": "decision", "input": {"query": "закажи такси до дома"}, "expect": "escalate"},
    ] * 4  # 40 mixed events

    # Warm up lazy imports and Layer 2 MoE loading outside the steady-state latency window
    mb.decide({"tool": "read_file", "path": "README.md"})
    mb.decide({"query": "inspect file contents and write safe patch to repository"})

    latencies_ms = []
    local_resolved = 0
    false_negatives = 0
    for ev in events:
        t0 = time.perf_counter_ns()
        res = mb.decide(ev["input"])
        latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
        assert res["decision"] == ev["expect"]
        if ev["expect"] == "deny" and res["decision"] != "deny":
            false_negatives += 1
        if res["decision"] in ("allow", "deny"):
            local_resolved += 1

    latencies_ms.sort()
    p50_ms = latencies_ms[len(latencies_ms) // 2]
    p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
    local_rate = local_resolved / len(events)

    # Check prompt tool-schema token reduction via MCP-Zero Top-4 vs 32 tools (~120 tokens/tool schema)
    tokens_per_tool = 120
    narrowed = mb.narrow_tool_surface("fix bug and write repository patch", top_k=4)
    schema_reduction = 1.0 - (len(narrowed["candidate_tools"]) * tokens_per_tool) / (32 * tokens_per_tool)

    assert false_negatives == 0
    assert local_rate >= 0.60
    assert schema_reduction >= 0.75
    # Enforce <2 ms median budget; allow modest p95 headroom for Windows/CI OS scheduling jitter on 40 samples
    assert p50_ms < 2.0, f"Median latency {p50_ms:.2f} ms exceeded 2.0 ms budget"
    assert p95_ms < 6.0, f"p95 latency {p95_ms:.2f} ms exceeded 6.0 ms CI headroom"
