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
    assert "Aetheris Microbrains v0.6.4 Local 25-Step Smoke Test" in proc.stdout
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
    assert "reanchoring_triggered" in str(drifted.neural_signal)

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


def test_fable_23_expanded_calibration_and_external_drop_rate():
    import importlib.util
    import json

    meta_files = sorted((PKG_DIR / "memory" / "store" / "neuro_brains").glob("*_meta.json"))
    assert len(meta_files) == 42
    for mf in meta_files:
        data = json.loads(mf.read_text(encoding="utf-8"))
        assert data.get("n_calibration_samples", 0) >= 100, f"{mf.name} n_cal={data.get('n_calibration_samples')}"
        assert data.get("conformal_alpha") == 0.05

    bench_path = PKG_DIR.parent / "examples" / "06_real_llm_ab_benchmark.py"
    assert bench_path.exists()
    spec = importlib.util.spec_from_file_location("bench06", str(bench_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rep = mod.evaluate_drop_rate_and_net_savings(top_k_values=(4,))
    assert rep["full_32_tool_schema_tokens"] >= 1500
    row4 = rep["sweep"][0]
    assert row4["top_k"] == 4
    assert row4["recall_at_k_pct"] >= 95.0
    assert row4["drop_rate_pct"] <= 5.0
    assert row4["net_saved_pct"] >= 35.0
    assert row4["p95_ms"] < 10.0


def test_fable_24_semantic_embedding_hybrid_narrowing():
    import numpy as np

    descriptions = {
        "docker_ps": "List running containers and health status",
        "repo_read": "Read source file from repository",
        "git_status": "Show git branch and working tree status",
        "run_tests": "Execute pytest unit tests",
        "db_query": "Run read-only SQL analytics query",
    }

    # Test zero-dependency char-trigram description similarity
    res_desc = mb.narrow_tool_surface(
        "inspect file in repository",
        available_tools=list(descriptions.keys()),
        top_k=2,
        tool_descriptions=descriptions,
    )
    assert res_desc["mode"] == "narrowed"
    assert res_desc["semantic_hybrid"] is True
    assert "repo_read" in res_desc["candidate_tools"]

    # Test custom embed_fn cosine hybrid boosting semantic match into Top-2
    def _mock_embed(texts):
        vecs = []
        for t in texts:
            tl = t.lower()
            v = [
                1.0 if ("container" in tl or "health" in tl) else 0.0,
                1.0 if ("file" in tl or "read" in tl) else 0.0,
                1.0 if ("sql" in tl or "query" in tl) else 0.0,
            ]
            vecs.append(v)
        return np.asarray(vecs, dtype=np.float64)

    res_emb = mb.narrow_tool_surface(
        "inspect health of running container",
        available_tools=list(descriptions.keys()),
        top_k=2,
        tool_descriptions=descriptions,
        embed_fn=_mock_embed,
    )
    assert res_emb["mode"] == "narrowed"
    assert "docker_ps" in res_emb["candidate_tools"]


def test_fable_25_strict_ast_allowlist_mode():
    strict_pol = mb.PolicyConfig(strict_ast_allowlist=True)

    # Pure stdlib math/json diff -> allow
    safe_res = mb.decide(
        {"tool": "edit_file", "path": "src/calc.py", "diff": "import math\ndef area(r):\n    return math.pi * r * r\n"},
        policy=strict_pol,
    )
    assert safe_res["decision"] == "allow"

    # Unapproved third-party/network import -> escalate under strict AST allowlist
    net_res = mb.decide(
        {"tool": "edit_file", "path": "src/fetch.py", "diff": "import urllib.request\ndef f(u):\n    return u\n"},
        policy=strict_pol,
    )
    assert net_res["decision"] == "escalate"
    assert net_res["reason_code"] == "layer1_strict_ast_allowlist_escalate"
    assert "strict_ast_unapproved_import:urllib" in net_res["label"]

    # Dynamic getattr call -> escalate under strict AST allowlist
    dyn_res = mb.decide(
        {"tool": "edit_file", "path": "src/dyn.py", "diff": "def f(obj, name):\n    return getattr(obj, name)\n"},
        policy=strict_pol,
    )
    assert dyn_res["decision"] == "escalate"
    assert "strict_ast_dynamic_call:getattr" in dyn_res["label"]


def test_fable_26_pretooluse_hook_cli():
    import json
    import subprocess
    import sys

    # Allowlisted Bash tool
    ok_res = mb.evaluate_hook_payload({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    assert ok_res["decision"] == "allow"
    assert ok_res["permissionDecision"] == "allow"
    assert ok_res["exit_code"] == 0

    # Dangerous Bash tool -> exit_code 2, deny
    bad_res = mb.evaluate_hook_payload({"tool_name": "Bash", "tool_input": {"command": "rm -fr /"}})
    assert bad_res["decision"] == "deny"
    assert bad_res["permissionDecision"] == "deny"
    assert bad_res["exit_code"] == 2

    # Unapproved import under --strict-ast via CLI stdin/stdout -> Claude Code hookSpecificOutput
    proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--strict-ast"],
        input=json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "src/a.py", "new_string": "import socket\n"}}),
        capture_output=True,
        text=True,
        cwd=str(PKG_DIR.parent),
    )
    assert proc.returncode == 0
    out = json.loads(proc.stdout.strip())
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert "strict_ast" in out["hookSpecificOutput"]["permissionDecisionReason"]

    # Deny via CLI stdin/stdout -> exit 2 + reason in stderr for model feedback
    deny_proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook"],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}),
        capture_output=True,
        text=True,
        cwd=str(PKG_DIR.parent),
    )
    assert deny_proc.returncode == 2
    assert "layer1_shell_parser_veto" in deny_proc.stderr
    deny_out = json.loads(deny_proc.stdout.strip())
    assert deny_out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert deny_out["hookSpecificOutput"]["permissionDecision"] == "deny"

    # Uninspected external/MCP tool without cmd/diff/query -> ask (never auto-allow)
    mcp_proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook"],
        input=json.dumps({"tool_name": "mcp__github__create_issue", "tool_input": {"title": "x"}}),
        capture_output=True,
        text=True,
        cwd=str(PKG_DIR.parent),
    )
    assert mcp_proc.returncode == 0
    mcp_out = json.loads(mcp_proc.stdout.strip())
    assert mcp_out["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert "layer1_uninspected_tool" in mcp_out["hookSpecificOutput"]["permissionDecisionReason"]


def test_fable_27_mondrian_per_class_cal_sizing():
    import json

    meta_files = sorted((PKG_DIR / "memory" / "store" / "neuro_brains").glob("*_meta.json"))
    assert len(meta_files) == 42
    max_cal = 0
    for mf in meta_files:
        data = json.loads(mf.read_text(encoding="utf-8"))
        n_cal = data.get("n_calibration_samples", 0)
        assert n_cal >= 120, f"{mf.name} n_cal={n_cal}"
        max_cal = max(max_cal, n_cal)
    assert max_cal >= 200


def test_fable_28_fit_from_traces_and_baseline_comparison():
    import importlib.util

    moe = mb.ToolRoutingMoE(d_model=256)
    traces = [
        {"query": "read source file config.py", "tool": "repo_read"},
        {"query": "inspect lines in server.py", "tool": "repo_read"},
        {"query": "commit staged bugfix changes", "tool": "git_commit"},
        {"query": "run pytest unit tests", "tool": "run_tests"},
        {"query": "execute shell command ls", "tool": "execute_shell"},
    ] * 4
    fit_rep = moe.fit_from_traces(traces)
    assert fit_rep["n_traces"] == 20
    assert fit_rep["train_top1_acc"] >= 0.80
    assert moe.route("commit staged bugfix changes").selected_tool in {"git_commit", "repo_write"}

    bench_path = PKG_DIR.parent / "examples" / "06_real_llm_ab_benchmark.py"
    spec = importlib.util.spec_from_file_location("bench06_b", str(bench_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.evaluate_baselines_comparison(top_k=4)
    assert len(rows) == 3
    mb_row = next(r for r in rows if "Microbrains" in r["baseline"])
    assert mb_row["recall_at_k_pct"] >= 95.0
    assert mb_row["cp95_upper_drop_pct"] < 15.0


def test_fable_29_non_python_yaml_js_hardening():
    yaml_bad = mb.decide({
        "tool": "edit_file",
        "path": "config/deploy.yaml",
        "diff": "payload: !!python/object/apply:os.system ['id']\n",
    })
    assert yaml_bad["decision"] == "deny"
    assert yaml_bad["label"] == "yaml_unsafe_deserialization_tag"

    js_bad = mb.decide({
        "tool": "edit_file",
        "path": "src/runner.ts",
        "diff": "const cp = require('child_process'); cp.execSync(cmd);\n",
    })
    assert js_bad["decision"] == "deny"
    assert js_bad["label"] == "js_dangerous_sink"


def test_fable_30_conformal_set_valid_and_coverage_target():
    res = mb.microbrain_consult("syntax error in AST parser during code edit")
    assert res["conformal_set_valid"] is True
    assert res["coverage_target"] == 0.95
    assert res["conformal_guarantee_valid"] is True
    assert isinstance(res["prediction_set"], list)


def test_fable_31_active_inference_negation_protection():
    from aetheris_microbrains.core.active_inference import _action_category

    assert _action_category("build a new router") == "focused_execution"
    assert _action_category("don't build it yet") == "generic_action"
    assert _action_category("I don't need a full breakdown") == "generic_action"
    assert _action_category("не анализируй это пока") == "generic_action"
    assert _action_category("анализируй это") == "logical_breakdown"


def test_fable_32_intent_tracker_negation_and_observational_directive():
    from aetheris_microbrains.core.intent_persistence_tracker import IntentPersistenceTracker

    tracker = IntentPersistenceTracker("почини баг логина")
    # Negated refusal must NOT score 0.0 drift
    neg_ru = tracker.register_step("не чини баг, просто отчитайся")
    assert neg_ru.drift_score >= 0.40
    assert "чини" not in neg_ru.overlapping_terms

    tracker_en = IntentPersistenceTracker("fix the login bug")
    neg_en = tracker_en.register_step("don't fix the bug, just report it")
    assert neg_en.drift_score >= 0.40
    assert "fix" not in neg_en.overlapping_terms

    # Observational directive (no imperative 'Abandon secondary rabbits')
    tracker_en.register_step("unrelated weather chat")
    drift_turn = tracker_en.register_step("completely off-topic cooking recipe")
    if drift_turn.reanchoring_directive:
        assert "Abandon secondary rabbits" not in drift_turn.reanchoring_directive


def test_fable_33_metacognitive_calibrator_q_hat_provenance():
    from aetheris_microbrains.core.metacognitive_calibrator import calibrate_action_confidence

    res = calibrate_action_confidence("regex_replace", raw_confidence=0.97, context_familiarity=0.3).to_dict()
    assert "q_hat" in res
    assert res["conformal_basis"] in {"held_out", "empirical_fallback"}
    assert 0.0 < res["q_hat"] < 1.0


def test_fable_34_cascade_escalation_payload_on_block_verdicts():
    from aetheris_microbrains.core.cascade_ensemble import CascadeEnsembleRouter

    router = CascadeEnsembleRouter()
    blocked_const = router.evaluate_action_pipeline(
        task_prompt="edit protected bible",
        target_tool="write_file",
        tool_args={"path": "memory/bible.md", "diff": "+# x\n"},
    )
    assert blocked_const.verdict == "BLOCKED_CONSTITUTIONAL"
    assert blocked_const.escalation_payload is not None
    assert blocked_const.escalation_payload["delegate_to"] == "HUMAN_SUPERVISOR"

    blocked_suki = router.evaluate_action_pipeline(
        task_prompt="run destructive shell",
        target_tool="execute_shell",
        tool_args={"cmd": "rm -rf /"},
    )
    assert blocked_suki.verdict == "BLOCKED_PRECONDITION_SUKI"
    assert blocked_suki.escalation_payload is not None
    assert blocked_suki.escalation_payload["delegate_to"] == "POLICY_REVIEWER"


def test_fable_35_negation_do_and_clause_window():
    from aetheris_microbrains.core.active_inference import _action_category

    # Affirmative 'do' imperative must NOT be treated as negation
    assert _action_category("do fix the bug") == "careful_repair"
    assert _action_category("do not fix the bug") == "generic_action"
    # Medium-distance negation (6 words between don't and fix) within same clause
    assert _action_category("Don't you think that we should probably fix the login flow") == "generic_action"
    # Clause boundary resets negation scope
    assert _action_category("Don't panic, fix the login flow") == "careful_repair"


def test_fable_36_uninspected_tools_external_policy_and_telemetry(tmp_path):
    from aetheris_microbrains.core.layered_decision_engine import PolicyConfig, _DECISION_RING_BUFFER

    # Uninspected tools (MCP, NotebookEdit, edit_file without diff) must escalate, not allow
    for tool in ("NotebookEdit", "mcp__slack__post_message", "edit_file"):
        res = mb.decide({"tool": tool, "path": "src/a.py"})
        assert res["decision"] == "escalate", f"{tool} unexpectedly returned {res}"
        assert res["reason_code"] == "layer1_uninspected_tool"
        assert res["upper_error_bound_95"] is None

    # Explicit READ_ONLY tool still allows and has no conformal upper_error_bound_95
    ro = mb.decide({"tool": "read_file", "path": "src/a.py"})
    assert ro["decision"] == "allow"
    assert ro["reason_code"] == "layer1_structural_verified"
    assert ro["upper_error_bound_95"] is None

    # PolicyConfig.require_approval_for_external is enforced at Layer 0
    ext_pol = PolicyConfig(require_approval_for_external=True)
    ext_res = mb.decide({"tool": "execute_shell", "cmd": "ls -la"}, policy=ext_pol)
    assert ext_res["decision"] == "escalate"
    assert ext_res["reason_code"] == "layer0_external_requires_confirmation"

    # Cross-process drift_detected reads persistent journal when ring buffer is empty
    jpath = tmp_path / "decision_journal.jsonl"
    jpath.write_text(
        "\n".join(
            '{"decision": "escalate", "reason_code": "layer2_conformal_abstention_ood"}'
            for _ in range(10)
        ) + "\n",
        encoding="utf-8",
    )
    _DECISION_RING_BUFFER.clear()
    drift_res = mb.decide("quantum flux hyperdrive calibrator xyzzy", context={"journal_path": str(jpath)})
    assert drift_res["drift_detected"] is True


def test_fable_37_hook_prefilter_mode_protected_read_and_fail_closed(monkeypatch, capsys):
    import json
    import subprocess
    import sys
    from aetheris_microbrains import hook as hook_mod

    # 1. Default prefilter mode: safe allow exits 0 with empty stdout (does not bypass Claude Code prompts)
    safe_proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook"],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}}),
        capture_output=True,
        text=True,
        cwd=str(PKG_DIR.parent),
    )
    assert safe_proc.returncode == 0
    assert safe_proc.stdout.strip() == ""

    # 2. Explicit --auto-allow emits hookSpecificOutput permissionDecision='allow'
    auto_proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--auto-allow"],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}}),
        capture_output=True,
        text=True,
        cwd=str(PKG_DIR.parent),
    )
    assert auto_proc.returncode == 0
    auto_out = json.loads(auto_proc.stdout.strip())
    assert auto_out["hookSpecificOutput"]["permissionDecision"] == "allow"

    # 3. Reading protected paths (.env / .ssh/id_rsa) must escalate ('ask'), never auto-allow
    for prot_path in (".env", "/home/user/.ssh/id_rsa", "config/credentials.json"):
        res = mb.decide({"tool": "read_file", "path": prot_path})
        assert res["decision"] == "escalate", f"{prot_path} returned {res}"
        assert res["reason_code"] == "layer0_protected_path_read_requires_confirmation"
        hres = mb.evaluate_hook_payload({"tool_name": "Read", "tool_input": {"file_path": prot_path}})
        assert hres["permissionDecision"] == "ask"

    # 4. Internal exception in hook.main() fails closed with exit code 2 and stderr reason
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated_hook_crash")

    monkeypatch.setattr(hook_mod, "evaluate_hook_payload", _boom)
    monkeypatch.setattr(sys, "stdin", type("S", (), {"read": lambda self: '{"tool_name":"Bash","tool_input":{"command":"ls"}}'})())
    rc = hook_mod.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "hook_internal_error:RuntimeError:simulated_hook_crash" in captured.err









