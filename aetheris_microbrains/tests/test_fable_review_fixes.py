"""Regression tests pinning all 7 findings from Fable's technical review.

Verifies:
1. ContinualLearningPipeline uses persistent `W_embed_anchor`, non-zero first-step EWC
   regularization, empirical diagonal Fisher over Hopfield rules, and strict set coverage.
2. WeightSpaceVSA.find_nearest_domains() ranks by actual projected weights rather than domain names.
3. AnalyticalMicroTransformer.infer_conformal(..., alpha=...) honors caller alpha dynamically.
4. CriticalEnsemble.evaluate_with_cascade() never returns AUTHORIZED / llm_invoked=True without an LLM.
5. Secret leak detection (`secret_leak_pattern_found`) & interface hazard rejection in CriticalEnsemble.
6. Colony SWE benchmark has zero hardcoded OOD stems (`эльфийск`, `астральн`, `martian`, `weather`).
7. Shipped WeightSpaceVSA & model bundles load cleanly with `allow_pickle=False`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import aetheris_microbrains as mb
from aetheris_microbrains.core.analytical_micro_transformer import AnalyticalMicroTransformer
from aetheris_microbrains.core.continual_learning_pipeline import ContinualLearningPipeline
from aetheris_microbrains.core.critical_ensemble_abstention import CriticalEnsemble
from aetheris_microbrains.core.layered_decision_engine import analyze_code_diff
from aetheris_microbrains.core.weight_space_vsa import WeightSpaceVSA
from aetheris_microbrains.tests import test_colony_swe_shadow as colony_bench

PKG_DIR = Path(__file__).resolve().parents[1]
BRAINS_DIR = PKG_DIR / "memory" / "store" / "neuro_brains"
VSA_PATH = PKG_DIR / "memory" / "store" / "weight_space_vsa.npz"


def test_fable_1_continual_learning_ewc_anchor_and_empirical_fisher(tmp_path: Path):
    for suffix in ("_weights.npz", "_meta.json"):
        shutil.copy2(
            BRAINS_DIR / f"neuro_code_edit_verifier{suffix}",
            tmp_path / f"neuro_code_edit_verifier{suffix}",
        )

    brain = AnalyticalMicroTransformer.load("neuro_code_edit_verifier", tmp_path)
    assert brain is not None
    assert hasattr(brain, "W_embed_anchor")
    assert brain.W_embed_anchor.shape == brain.W_embed.shape

    pipeline = ContinualLearningPipeline(store_dir=tmp_path, audit_log_path=tmp_path / "audit.json", ewc_lambda=0.5)

    # Empirical Fisher over Hopfield rules must differ significantly from uniform 0.1*w^2 proxy
    emp_fisher = pipeline.compute_fisher_information({"W_embed": brain.W_embed}, brain=brain)["W_embed"]
    proxy_fisher = np.maximum(1e-4, np.square(brain.W_embed) * 0.1).astype(np.float32)
    assert not np.allclose(emp_fisher, proxy_fisher), "Fisher fell back to parameter-magnitude proxy!"
    assert float(np.max(emp_fisher)) > 2e-4
    # Active Hopfield token rows must carry strictly greater Fisher importance than unused tail rows
    active_mean = float(np.mean(emp_fisher[: len(brain.vocab)]))
    idle_mean = float(np.mean(emp_fisher[len(brain.vocab) + 1 :]))
    assert active_mean > idle_mean

    # Single update on an existing in-vocab token must yield non-zero EWC penalty immediately
    res = pipeline.apply_plasticity_feedback(
        "code_edit_verifier",
        ["syntax_tree_malformed"],
        learning_rate=0.05,
        promote=False,
    )
    assert res.status == "success"
    assert res.ewc_penalty > 0.0, "EWC penalty must be strictly positive even on the first token step!"
    assert res.weights_delta_l2 > 0.0

    # Strict set-membership coverage: target NOT in prediction_set must count as miss even if top_candidate matches
    fake_holdout = [(["syntax_tree_valid"], "__definitely_not_in_set__")]
    assert pipeline._measure_holdout_coverage(brain, fake_holdout) == 0.0


def test_fable_2_vsa_find_nearest_domains_uses_actual_weights():
    vsa = WeightSpaceVSA(dimension=1024, seed=99)
    rng = np.random.default_rng(7)
    base_w = {"W_embed": rng.normal(size=(32, 32)).astype(np.float32)}
    # Domain B has almost identical weights to Domain A, despite completely unrelated name
    near_w = {"W_embed": (base_w["W_embed"] + 0.02 * rng.normal(size=(32, 32))).astype(np.float32)}
    far_w = {"W_embed": rng.normal(scale=2.0, size=(32, 32)).astype(np.float32)}

    vsa.store_brain_weights("alpha_domain", base_w)
    vsa.store_brain_weights("unrelated_name_clone", near_w)
    vsa.store_brain_weights("alpha_domain_orthogonal", far_w)

    nearest = vsa.find_nearest_domains("alpha_domain", top_k=2)
    assert len(nearest) == 2
    assert nearest[0]["domain"] == "unrelated_name_clone"
    assert nearest[0]["similarity"] > 0.85


def test_fable_3_infer_conformal_respects_caller_alpha():
    brain = AnalyticalMicroTransformer.load("neuro_fast_intent_classifier", BRAINS_DIR)
    assert brain is not None and len(brain.conformal_scores) >= 20

    res_strict = brain.infer_conformal(["fix"], alpha=0.01)
    res_loose = brain.infer_conformal(["fix"], alpha=0.40)
    assert res_strict["alpha"] == pytest.approx(0.01)
    assert res_loose["alpha"] == pytest.approx(0.40)
    assert res_strict["conformal_quantile"] >= res_loose["conformal_quantile"]
    assert res_strict["conformal_quantile"] != brain.conformal_quantile or res_loose["conformal_quantile"] != brain.conformal_quantile


def test_fable_4_escalate_to_llm_honesty_without_llm_callback():
    ensemble = CriticalEnsemble()
    # 550-line benign diff triggers patch_quality_predictor DENY (1 DENY, 3 ALLOW -> ABSTAIN_DELEGATE_TO_LLM)
    big_diff = "\n".join(f"+val_{i} = {i}" for i in range(520))

    no_llm = ensemble.evaluate_with_cascade("edit_file", "src/data_gen.py", big_diff)
    assert no_llm["final_verdict"] != "AUTHORIZED"
    assert no_llm["final_verdict"] == "ESCALATED_TO_LLM"
    assert no_llm["llm_invoked"] is False
    assert no_llm["requires_external_llm"] is True

    # Force pure uncertainty escalation on benign small edit via low variance abstention
    called_with = []

    def _fake_llm(payload: dict) -> dict:
        called_with.append(payload)
        return {"verdict": "AUTHORIZED_BY_LLM"}

    # Missing model forces estimator abstention -> cascade passes tier1-4 -> invokes callback
    orig_loader = ensemble._get_model
    ensemble._models_cache["neuro_patch_quality_predictor"] = None  # trigger model_missing abstain
    ensemble._get_model = lambda n: None if n == "neuro_patch_quality_predictor" else orig_loader(n)
    with_llm = ensemble.evaluate_with_cascade("edit_file", "src/clean.py", "def ok():\n    return 1\n", llm_callback=_fake_llm)
    assert len(called_with) == 1
    assert with_llm["llm_invoked"] is True
    assert with_llm["final_verdict"] == "AUTHORIZED_BY_LLM"


def test_fable_5_secret_leak_detection_and_interface_hazard_ensemble():
    secret_diff = (
        "--- a/src/config.py\n"
        "+++ b/src/config.py\n"
        "@@ -1,1 +1,2 @@\n"
        '+OPENAI_KEY = "sk-proj-1234567890abcdef1234567890abcdef"\n'
    )
    analysis = analyze_code_diff("src/config.py", secret_diff)
    assert analysis["security_clean"] is False
    assert "secret_leak_pattern_found" in analysis["findings"]

    ensemble = CriticalEnsemble()
    consensus = ensemble.evaluate_critical_action("edit_file", "src/config.py", secret_diff)
    assert consensus.status == "CONSENSUS_REJECTED"
    v_vote = next(v for v in consensus.votes if v.estimator_name == "code_edit_verifier")
    assert v_vote.verdict == "DENY"

    # Breaking signature removal must not be rubber-stamped by CriticalEnsemble
    sig_break_diff = (
        "--- a/src/api.py\n"
        "+++ b/src/api.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def public_handler(req, ctx, timeout=5):\n"
        "+def public_handler(req):\n"
        "     return req\n"
    )
    sig_res = ensemble.evaluate_critical_action("edit_file", "src/api.py", sig_break_diff)
    assert sig_res.status != "CONSENSUS_APPROVED"


def test_fable_6_colony_benchmark_has_no_hardcoded_ood_stems():
    hybrid = colony_bench.HybridMicrobrainWithConformalGate()
    for forbidden_ood_stem in ("астральн", "эльфийск", "философское", "philosophy", "martian", "weather", "погоду"):
        assert forbidden_ood_stem not in hybrid.critical_invariants
        assert forbidden_ood_stem not in colony_bench.PureKeywordBaseline.CRITICAL_KEYWORDS

    res = colony_bench.run_3way_ablation_benchmark()
    hybrid_metrics = res["3. Microbrain Hybrid (Weights + Conformal Gate)"]
    assert hybrid_metrics["missed_escalations"] == "0/50"
    assert float(hybrid_metrics["strict_accuracy"].split("(")[1].replace("%)", "")) >= 80.0


def test_fable_7_vsa_loads_with_allow_pickle_false():
    with np.load(VSA_PATH, allow_pickle=False) as raw:
        assert "weight_keys" in raw.files
        assert "weight_values" in raw.files

    loaded = WeightSpaceVSA.load(VSA_PATH)
    assert len(loaded.domain_weight_vectors) == 42
    nearest = loaded.find_nearest_domains("tool_routing_v2", top_k=3)
    assert len(nearest) == 3
    assert all(item["similarity"] > 0.0 for item in nearest)


def test_rm_rf_top_level_and_ensemble_route_ood_escalation():
    # Destructive `rm -rf` on any top-level/root path must be caught by Layer 1
    res = mb.decide({"tool": "run_command", "cmd": "rm -rf /ouroboros"})
    assert res["decision"] == "deny"

    ens = CriticalEnsemble().evaluate_critical_action("run_command", "rm -rf /ouroboros")
    assert ens.status == "CONSENSUS_REJECTED"

    # Ensemble route on completely unknown OOD tokens must abstain & escalate
    route_ood = mb.run_neuro_route(
        action="ensemble_route",
        tokens=["random_divergent_intent_xyz"],
        min_agreement=0.99,
    )
    assert route_ood["escalate_to_llm"] is True
    assert route_ood["fallback"] is True


def test_fable_8_zero_numbered_placeholder_tokens_across_all_42_models():
    import json
    import re
    from collections import Counter

    metas = sorted(BRAINS_DIR.glob("*_meta.json"))
    assert len(metas) == 42
    for mf in metas:
        data = json.loads(mf.read_text(encoding="utf-8"))
        vocab_keys = [k for k in data["vocab"] if not str(k).startswith("<")]
        assert len(data["vocab"]) >= 50, f"{mf.stem} vocab below 50: {len(data['vocab'])}"
        stems = Counter(re.sub(r"_\d+$", "", t) for t in vocab_keys if re.search(r"_\d+$", t))
        bad_series = {s: c for s, c in stems.items() if c >= 2}
        assert not bad_series, f"{mf.stem} still contains numbered placeholder series: {bad_series}"
        for forbidden in ("intent_class_1", "risk_factor_level_1", "subtask_step_1", "blast_radius_files_1", "quality_metric_1"):
            assert forbidden not in data["vocab"], f"{mf.stem} contains {forbidden}"


def test_fable_9_moe_router_fitted_w_gate_and_empirical_per_expert_quantiles():
    moe = mb.ToolRoutingMoE(d_model=256)
    assert moe.w_gate_fitted is True
    # Empirical per-expert quantiles must NOT all collapse to identical 0.0420/0.0120 constants
    qs = set(moe.per_expert_quantiles.values())
    assert len(qs) >= 3, f"Per-expert quantiles appear constant/uncalibrated: {moe.per_expert_quantiles}"
    assert 0.042 not in qs or len(qs) > 1
    assert all(cov >= 0.90 for cov in moe.per_expert_coverage.values())


def test_fable_10_examples_01_to_05_run_cleanly_and_no_syntaxerror_hack():
    import subprocess
    import sys
    repo_root = PKG_DIR.parent
    ex04 = (repo_root / "examples" / "04_counterfactual_safety_gate.py").read_text(encoding="utf-8")
    assert "SyntaxError" not in ex04, "examples/04 still contains literal SyntaxError substring hack!"

    ex01 = (repo_root / "examples" / "01_quickstart_consult.py").read_text(encoding="utf-8")
    assert "44" not in ex01 and 'result.get("prediction")' not in ex01

    bench_mod_path = repo_root / "examples" / "05_reproducible_25step_benchmark.py"
    assert bench_mod_path.exists()
    proc = subprocess.run(
        [sys.executable, str(bench_mod_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Aetheris Microbrains v0.6.3 Reproducible 25-Step Agent Benchmark" in proc.stdout
    assert "Token Reduction" in proc.stdout

    # Also verify structured return contract of benchmark helper
    sys.path.insert(0, str(repo_root))
    from examples import __file__ as _ex_check  # noqa: F401
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench25", str(bench_mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    report = mod.run_25step_benchmark(trials=2)
    m = report["aetheris_microbrains"]
    assert m["dangerous_missed_fn"] == 0
    assert m["benign_false_positive_fp"] == 0
    assert m["token_reduction_pct"] >= 60.0
    assert m["p95_latency_ms"] < 2.0


def test_fable_11_save_does_not_pollute_cwd_with_memory_store(tmp_path: Path):
    import os
    from aetheris_microbrains.core import analytical_micro_transformer as amt_mod

    orig_cwd = os.getcwd()
    pkg_tmp = tmp_path / "pkg_store"
    pkg_tmp.mkdir()
    cwd_tmp = tmp_path / "user_cwd"
    cwd_tmp.mkdir()
    orig_store = amt_mod.STORE_DIR
    orig_canon = list(amt_mod.CANONICAL_STORE_DIRS)
    try:
        os.chdir(cwd_tmp)
        amt_mod.STORE_DIR = pkg_tmp
        amt_mod.CANONICAL_STORE_DIRS[:] = [pkg_tmp, cwd_tmp / "memory" / "store" / "neuro_brains"]
        b = AnalyticalMicroTransformer(name="tmp_pkg_only_test", d_model=16, n_heads=2, d_mlp=32)
        b.compile_logic_association("ping", "pong")
        # Calling save() with default directory=None must NOT create memory/store/neuro_brains in user CWD
        b.save(directory=None)
        assert (pkg_tmp / "tmp_pkg_only_test_weights.npz").exists()
        assert not (cwd_tmp / "memory" / "store" / "neuro_brains").exists()
    finally:
        amt_mod.STORE_DIR = orig_store
        amt_mod.CANONICAL_STORE_DIRS[:] = orig_canon
        os.chdir(orig_cwd)


def test_fable_12_full_file_patched_ast_context_parsing():
    original_src = (
        "import os\n\n"
        "def compute(x, y):\n"
        "    total = (x + y\n"
        "             + 1)\n"
        "    return total\n"
    )
    # 1. Valid unified diff applied to full file -> patched_file_ast_verified=True & syntax_valid=True
    good_diff = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -3,4 +3,4 @@\n"
        " def compute(x, y):\n"
        "-    total = (x + y\n"
        "+    total = (x * 2 + y\n"
        "              + 1)\n"
        "     return total\n"
    )
    res_good = analyze_code_diff("mod.py", good_diff, original_source=original_src)
    assert res_good["patched_file_ast_verified"] is True
    assert res_good["syntax_valid"] is True
    assert res_good["security_clean"] is True

    # 2. Diff dropping a closing paren (or breaking full-file AST structure) fails full-file ast.parse
    bad_diff = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -3,4 +3,4 @@\n"
        " def compute(x, y):\n"
        "     total = (x + y\n"
        "-             + 1)\n"
        "+             + 1\n"
        "     return total\n"
    )
    res_bad = analyze_code_diff("mod.py", bad_diff, original_source=original_src)
    assert res_bad["patched_file_ast_verified"] is True
    assert res_bad["syntax_valid"] is False
    assert "patched_file_syntax_error" in res_bad["findings"]


def test_fable_13_critical_ensemble_multi_feature_structural_tokens():
    ce = CriticalEnsemble()
    diff_text = (
        "--- a/src/calc.py\n"
        "+++ b/src/calc.py\n"
        "@@ -1,2 +1,5 @@\n"
        " def add(a, b):\n"
        "+    val = a + b\n"
        "+    if val > 0:\n"
        "+        return val\n"
        "     return 0\n"
    )
    res = ce.evaluate_critical_action("apply_patch", "src/calc.py", diff_text)
    assert res.status == "CONSENSUS_APPROVED"
    for v in res.votes:
        toks = v.details.get("tokens", [])
        assert len(toks) >= 3, f"{v.estimator_name} received trivial <=2 tokens: {toks}"

    verifier_vote = next(v for v in res.votes if v.estimator_name == "code_edit_verifier")
    verifier_toks = verifier_vote.details["tokens"]
    assert any(t.startswith("ast_node_") for t in verifier_toks)
    assert "ast_interface_stable" in verifier_toks


def test_fable_14_counterfactual_reasoner_and_blast_radius_work_outside_ouroboros(tmp_path: Path):
    from aetheris_microbrains.core.causal_world_model import StructuralCausalModel
    from aetheris_microbrains.core.counterfactual_reasoner import _find_downstream_dependents, simulate_counterfactual

    # Build a standalone non-ouroboros project layout (`src/...`) with 4 importing modules
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target_mod = src_dir / "helper_lib.py"
    target_mod.write_text("def run_ok(x):\n    return x + 1\n", encoding="utf-8")
    for idx in range(4):
        (src_dir / f"consumer_{idx}.py").write_text("from src.helper_lib import run_ok\n", encoding="utf-8")

    deps = _find_downstream_dependents(tmp_path, str(target_mod))
    assert len(deps) >= 4

    scm = StructuralCausalModel(tmp_path)
    assert scm._eval_blast_radius(str(target_mod)) >= 4

    # Destructive flag-reordered shell must be blocked
    sh_bad = simulate_counterfactual(tmp_path, "run_command", "rm -fr /var/data", recent_tools=["repo_read"])
    assert sh_bad.is_safe is False
    assert any("irreversible_action" in h for h in sh_bad.hazards)

    # Obfuscated getattr sink must be blocked via analyze_code_diff
    code_bad = simulate_counterfactual(
        tmp_path,
        "repo_write",
        str(target_mod),
        proposed_content="getattr(os, 'sys' + 'tem')('id')\n",
        recent_tools=["repo_read"],
    )
    assert code_bad.is_safe is False
    assert any("security_violation" in h for h in code_bad.hazards)


def test_fable_15_live_tool_registry_enforcement_in_moe_and_decide():
    moe = mb.ToolRoutingMoE(d_model=256)

    # Empty live registry cannot return any executable tool or claim conformal pass
    empty_res = moe.route("read file src/app.py", available_tools=[])
    assert empty_res.selected_tool == "<none>"
    assert empty_res.conformal_passed is False

    # Narrow custom registry only permits registered tools; unrelated request escalates instead of misfiring
    restricted = ["git_status"]
    mismatch = moe.route("delete production sql table", available_tools=restricted)
    assert mismatch.selected_tool in restricted
    assert mismatch.conformal_passed is False

    matched = moe.route("check git status in repository", available_tools=restricted)
    assert matched.selected_tool == "git_status"
    assert matched.conformal_passed is True

    # Layer 0 decide() denies tools outside `available_tools`
    denied = mb.decide({"tool": "run_command", "cmd": "ls -la"}, context={"available_tools": ["read_file", "git_status"]})
    assert denied["decision"] == "deny"
    assert denied["reason_code"] == "layer0_unregistered_tool"
    assert denied["layer"] == 0


def test_fable_16_clopper_pearson_bound_and_nonconformity_drift_monitor():
    # Exact Clopper-Pearson upper bound matches 1 - alpha**(1/n)
    ub22 = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(0, 22, alpha=0.05)
    assert 0.12 <= ub22 <= 0.14
    ub100 = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(0, 100, alpha=0.05)
    assert abs(ub100 - 0.0295) < 0.002

    # Rolling nonconformity & abstention monitor triggers drift_detected after OOD burst
    b = AnalyticalMicroTransformer(name="tmp_drift_test", d_model=32, n_heads=2, d_mlp=64)
    b.compile_logic_association("read_file_token", "allow_read")
    b.compile_logic_association("write_file_token", "allow_write")
    cal_data = [(["read_file_token"], "allow_read")] * 11 + [(["write_file_token"], "allow_write")] * 11
    cal_rep = b.calibrate_conformal(cal_data, alpha=0.05, in_sample=False)
    assert cal_rep["upper_error_bound_95"] == ub22

    in_dist = b.infer_conformal(["read_file_token"])
    assert in_dist["upper_error_bound_95"] == ub22
    assert in_dist["drift_detected"] is False

    last = None
    for i in range(12):
        last = b.infer_conformal([f"unseen_ood_token_{i}"])
    assert last is not None
    assert last["abstained"] is True
    assert last["drift_detected"] is True
    assert last["rolling_abstention_rate"] >= 0.80


def test_fable_17_cascade_ensemble_no_fake_tokens_and_real_routing_match():
    from aetheris_microbrains.core.cascade_ensemble import CascadeEnsembleRouter

    router = CascadeEnsembleRouter()
    ood_verdict = router.evaluate_action_pipeline(
        task_prompt="frobnicate quantumFlux capacitor orthogonally",
        target_tool="custom_unregistered_gizmo",
    )
    assert ood_verdict.authorized is False
    assert ood_verdict.verdict == "DEGRADED_FALLBACK"
    assert ood_verdict.escalation_payload is not None
    tier2 = next(t for t in ood_verdict.tier_results if t.tier_name == "Tier2_ToolRouting")
    assert tier2.passed is False

    safe_verdict = router.evaluate_action_pipeline(
        task_prompt="list files in directory",
        target_tool="ls",
        tool_args={"cmd": "ls -la"},
    )
    assert safe_verdict.authorized is True
    assert safe_verdict.verdict == "AUTHORIZED"

    bad_verdict = router.evaluate_action_pipeline(
        task_prompt="wipe disk",
        target_tool="execute_shell",
        tool_args={"cmd": "rm -rf /"},
    )
    assert bad_verdict.authorized is False
    assert bad_verdict.verdict == "BLOCKED_PRECONDITION_SUKI"


def test_fable_18_bilingual_intent_persistence_and_grounded_metacognitive_calibrator():
    from aetheris_microbrains.core.intent_persistence_tracker import IntentPersistenceTracker
    from aetheris_microbrains.core.metacognitive_calibrator import calibrate_action_confidence

    tracker = IntentPersistenceTracker("исправь ошибку маршрутизатора сообщений")
    assert len(tracker.root_terms) >= 3
    aligned = tracker.register_step("отладка маршрутизатора", "анализ сообщений")
    assert len(aligned.overlapping_terms) >= 1
    for _ in range(3):
        drifted = tracker.register_step("покупка авиабилетов и прогноз погоды", "отвлеченная задача")
    assert drifted.is_drifting is True
    assert drifted.neural_signal == "reanchoring_triggered"

    cal_unc = calibrate_action_confidence("git_operations", raw_confidence=0.80, context_familiarity=0.40)
    assert cal_unc.neural_signal == "entropy_floor_enforced"
    cal_high = calibrate_action_confidence("file_inspection", raw_confidence=0.95, context_familiarity=0.90)
    assert cal_high.neural_signal == "calibration_audit_passed"
    assert cal_high.should_proceed is True


def test_fable_19_universal_swe_protected_paths():
    from aetheris_microbrains.core.layered_decision_engine import is_protected_path

    for sensitive in (".env", "config/.env.production", ".git/hooks/pre-commit", ".git/config", ".ssh/id_rsa", "credentials.json"):
        assert is_protected_path(sensitive) is True
        dec = mb.decide({"tool": "edit_file", "path": sensitive, "diff": "+SECRET=1\n"})
        assert dec["decision"] == "deny"
        assert dec["layer"] == 0
        ens = CriticalEnsemble().evaluate_critical_action("edit_file", sensitive, "+x=1")
        assert ens.status == "CONSENSUS_REJECTED"

    for harmless in ("src/env_loader.py", "tests/test_env.py", "src/math/axioms.py"):
        assert is_protected_path(harmless) is False


def test_fable_20_static_data_table_vs_cyclomatic_complexity_blast_radius():
    from aetheris_microbrains.core.counterfactual_simulation_pipeline import CounterfactualSimulationPipeline

    rows = "\n".join(f'    {{"id": {i}, "label": "item_{i}"}},' for i in range(250))
    static_code = f"STATIC_TABLE = [\n{rows}\n]\n"
    diff_res = analyze_code_diff("src/data_table.py", static_code)
    assert diff_res["syntax_valid"] is True
    assert diff_res["security_clean"] is True
    assert diff_res["cyclomatic_complexity"] == 0
    assert diff_res["is_static_data_or_doc"] is True

    pipe = CounterfactualSimulationPipeline()
    out_static = pipe.simulate_edit_impact("src/data_table.py", static_code)
    assert out_static.allowed is True
    assert out_static.status == "APPROVED_FOR_PATCH"

    branches = "\n".join(f"    if x == {i}:\n        y += {i}" for i in range(30))
    complex_code = f"def dispatch(x):\n    y = 0\n{branches}\n    return y\n"
    complex_res = analyze_code_diff("src/complex_flow.py", complex_code)
    assert complex_res["cyclomatic_complexity"] >= 30
    assert complex_res["is_static_data_or_doc"] is False


def test_fable_21_decision_journal_recalibration_and_decide_telemetry():
    from aetheris_microbrains.core.layered_decision_engine import recalibrate_from_decision_journal

    res = mb.decide("read file src/app.py")
    assert "upper_error_bound_95" in res
    assert 0.0 < res["upper_error_bound_95"] < 0.5
    assert isinstance(res["drift_detected"], bool)

    synthetic_journal = [
        {"decision": "allow", "layer": 2, "latency_ms": 0.4, "verified_correct": True, "nonconformity_score": 0.03}
        for _ in range(22)
    ]
    cal_summary = recalibrate_from_decision_journal(synthetic_journal, alpha=0.05)
    assert cal_summary["n_episodes"] == 22
    assert cal_summary["verified_episodes"] == 22
    assert cal_summary["verified_errors"] == 0
    assert cal_summary["recalibrated_q_hat"] is not None
    assert cal_summary["router_updated"] is True
    assert 0.12 <= cal_summary["upper_error_bound_95"] <= 0.14


def test_fable_22_zero_proposal_n_comments_in_codebase():
    import re

    offenders = []
    for pyf in sorted(PKG_DIR.rglob("*.py")):
        text = pyf.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\bProposal\s+\d+", text):
            offenders.append(str(pyf.relative_to(PKG_DIR)))
    assert not offenders, f"Found residual Proposal references in: {offenders}"




