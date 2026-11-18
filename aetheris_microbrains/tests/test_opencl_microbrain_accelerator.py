"""
Tests for the 11-Proposal GT 710 OpenCL Micro-Brain Co-Processor Suite:
1. EGGROLL Low-Rank Neuroevolution in Resident VRAM (`eggroll_evolve_population` & `AutonomousBrainNursery.evolve_population_eggroll`)
2. Closed-Form Continuous-Time (CfC) Liquid Reservoir (`cfc_step` & `LiquidStateReservoir.step_cfc_continuous`)
3. Hyperdimensional Computing (VSA) 10,240-Bit Associative Memory (`hdc_search` & `MicroMoEOrchestrator.route_query_hdc`)
4. TTT-Linear Test-Time Training Fast Weights (`ttt_linear_step` & `MicroBrainHypernetwork.adapt_test_time_ttt`)
5. Hybrid CPU/OpenCL Dispatch (`hardware_router.recommended_microbrain_dispatch` & `BitLinearLayer.forward`)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.bitnet_ternary_engine import BitLinearLayer
from core.brain_moe_orchestrator import MicroMoEOrchestrator
from core.brain_nursery import AutonomousBrainNursery
from core.hardware_router import (
    get_best_execution_provider,
    get_hardware_report,
    recommended_microbrain_dispatch,
)
from core.liquid_state_reservoir import LiquidStateReservoir
from core.microbrain_hypernetwork import MicroBrainHypernetwork
from core.opencl_microbrain_accelerator import OpenCLMicroBrainAccelerator


def test_hardware_router_and_opencl_detection():
    """Verify OpenCL GPU detection and batch-size crossover routing."""
    accel = OpenCLMicroBrainAccelerator.get_instance()
    status = accel.status()
    assert "available" in status
    assert "gpu_available" in status
    assert "device_name" in status
    # Dispatch is per-op via the measured crossover table, not a single batch
    # threshold - the old decorative crossover_batch_size key was removed.
    assert "crossover_batch_size" not in status
    assert "crossovers" in status

    report = get_hardware_report()
    assert "opencl_available" in report
    assert "gpu_detected" in report
    assert "gpu_name" in report

    # Single-sample queries route to CPU to avoid PCIe 2.0 launch overhead
    assert recommended_microbrain_dispatch(batch_size=1) == "CPU"
    assert recommended_microbrain_dispatch(batch_size=16) == "CPU"

    # Batch >= 32 routes to OpenCL_GPU when available
    if accel.gpu_available:
        assert recommended_microbrain_dispatch(batch_size=32) == "OpenCL_GPU"
        assert recommended_microbrain_dispatch(batch_size=256) == "OpenCL_GPU"


def test_proposal_5_bitnet_hybrid_cpu_and_opencl_parity():
    """BitLinearLayer dispatches using empirical crossover table (default CPU for B=64, GPU when forced or large)."""
    rng = np.random.default_rng(42)
    in_features, out_features = 64, 48
    weights = rng.standard_normal((out_features, in_features)).astype(np.float32)
    layer = BitLinearLayer(in_features, out_features, weights=weights)

    # Single-sample -> CPU path
    x_single = rng.standard_normal(in_features).astype(np.float32)
    out_single = layer.forward(x_single)
    assert out_single.shape == (out_features,)
    assert layer.last_backend == "CPU"

    # Batched B=64 stays on CPU under the honest bootstrap (BitNet unmeasured/unfavorable on GPU)
    x_batch = rng.standard_normal((64, in_features)).astype(np.float32)
    out_batch = layer.forward(x_batch)
    assert out_batch.shape == (64, out_features)
    assert layer.last_backend == "CPU"

    # With force_gpu=True (or small crossover threshold), executes on OpenCL GPU with numerical parity
    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        out_gpu = layer.forward(x_batch, force_gpu=True)
        assert layer.last_backend == "OpenCL_GPU"
        x_q, scale_x = layer._quantize_activations(x_batch)
        combined_scale = float(layer.gamma_w / scale_x)
        cpu_expected = (
            np.dot(x_q.astype(np.int32), layer.ternary_weight.T.astype(np.int32)).astype(np.float32)
            * combined_scale
        )
        np.testing.assert_allclose(out_gpu, cpu_expected, rtol=1e-4, atol=1e-4)


def test_proposal_1_eggroll_low_rank_neuroevolution_rollout(tmp_path, monkeypatch):
    """EGGROLL low-rank population neuroevolution evaluates 256 daughter brains in parallel on GPU."""
    import core.brain_nursery as nursery_mod

    monkeypatch.setattr(nursery_mod, "STORE_DIR", tmp_path)
    nursery = AutonomousBrainNursery()

    brain = nursery.grow_brain_for_domain(
        "eggroll_test",
        rules={"alpha": "beta", "beta": "gamma"},
        d_model=32,
        n_heads=2,
    )
    assert brain.d_model == 32

    rng = np.random.default_rng(123)
    x_samples = rng.standard_normal((16, 32)).astype(np.float32)
    y_targets = rng.standard_normal((16, 32)).astype(np.float32)

    evo_res = nursery.evolve_population_eggroll(
        "eggroll_test",
        x_samples=x_samples,
        y_targets=y_targets,
        pop_size=256,
        sigma=0.35,
        seed=42,
        force_gpu=True,  # explicit hardware exercise; honest dispatch defaults to CPU at pop=256
    )
    assert evo_res["population_size"] == 256
    assert evo_res["best_mse"] <= evo_res["mean_population_mse"]
    assert evo_res["improvement_ratio"] > 0.0

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert evo_res["backend"] == "OpenCL_GPU"
        # The device string must be the accelerator's OWN reported device —
        # the old assert ("GT 710" in device or len(device) > 0) passed for
        # ANY non-empty string, including placeholders.
        assert evo_res["device"] == accel.device_name
        assert evo_res["device"] != "CPU"


def test_proposal_2_cfc_liquid_state_reservoir_continuous_step():
    """Closed-Form Continuous-Time (CfC) liquid reservoir step with variable delta_t on GPU."""
    reservoir = LiquidStateReservoir(reservoir_dim=128, input_dim=16, spectral_radius=0.92, leak_rate=0.3)
    rng = np.random.default_rng(101)

    u_vec = rng.standard_normal(16).astype(np.float32)
    res_fast = reservoir.step_cfc_continuous(u_vec, delta_t=0.05, event_tag="fast_reflex", force_gpu=True)
    assert res_fast["status"] == "stepped_cfc"
    assert res_fast["delta_t"] == pytest.approx(0.05, rel=1e-3)
    assert res_fast["reservoir_energy"] > 0.0
    assert len(res_fast["full_state"]) == 128

    res_slow = reservoir.step_cfc_continuous(u_vec, delta_t=2.5, event_tag="slow_deliberation", force_gpu=True)
    assert res_slow["step_count"] == 2
    assert res_slow["delta_t"] == pytest.approx(2.5, rel=1e-3)

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert res_fast["backend"] == "OpenCL_GPU"
        assert res_slow["backend"] == "OpenCL_GPU"


def test_proposal_3_hdc_10240_bit_associative_memory_and_moe_router(tmp_path, monkeypatch):
    """10,240-bit Hyperdimensional Computing (VSA) bitwise XOR + popcount search and MoE routing."""
    accel = OpenCLMicroBrainAccelerator.get_instance()

    # Create 512 distinct memory hypervectors in the 10,240-bit space (320 x uint32)
    rng = np.random.default_rng(777)
    memory_bank = rng.integers(0, 2**32, size=(512, 320), dtype=np.uint32)

    # Plant a target hypervector at index 137 with 5 flipped bits
    target_hv = memory_bank[137].copy()
    target_hv[0] ^= np.uint32(0b1010101010)  # 5 bits flipped out of 10,240

    sims, backend = accel.hdc_search(target_hv, memory_bank, force_gpu=True)
    best_idx = int(np.argmax(sims))
    assert best_idx == 137
    expected_sim = 1.0 - (5.0 / 10240.0)
    assert float(sims[137]) == pytest.approx(expected_sim, abs=1e-5)

    if accel.gpu_available:
        assert backend == "OpenCL_GPU"

    # Also test MicroMoEOrchestrator.route_query_hdc
    import core.brain_nursery as nursery_mod

    monkeypatch.setattr(nursery_mod, "STORE_DIR", tmp_path)
    # monkeypatch restores the previous singleton at teardown — a bare
    # assignment here leaked a tmp-path nursery into every later suite.
    monkeypatch.setattr(AutonomousBrainNursery, "_instance", None)
    nursery = AutonomousBrainNursery.get_instance()
    nursery.grow_brain_for_domain("astrophysics", rules={"quasar": "pulsar", "redshift": "galaxy"})
    nursery.grow_brain_for_domain("biochemistry", rules={"enzyme": "peptide", "ribosome": "atp"})

    moe = MicroMoEOrchestrator()
    hdc_route = moe.route_query_hdc(["enzyme", "peptide", "atp"], top_k=2)
    assert hdc_route["success"] is True
    assert hdc_route["chosen_expert"] == "neuro_biochemistry"
    assert hdc_route["hdc_similarity"] > 0.5
    if accel.gpu_available:
        # Honest crossover dispatch: a 2-expert codebook stays on CPU — the PCIe
        # launch overhead outweighs the work at this size (no force_gpu here).
        assert hdc_route["backend"] == "CPU"


def test_proposal_4_ttt_linear_fast_weight_adaptation():
    """TTT-Linear online self-supervised gradient descent reduces reconstruction loss in VRAM."""
    hypernet = MicroBrainHypernetwork(latent_dim=64, hidden_dim=32)

    context = "causal manifold surgical repair AST localization"
    outcome = "verified minimal patch passing regression tests"

    step1 = hypernet.adapt_test_time_ttt(context, outcome, lr=0.15, force_gpu=True)
    step2 = hypernet.adapt_test_time_ttt(context, outcome, lr=0.15, force_gpu=True)
    step3 = hypernet.adapt_test_time_ttt(context, outcome, lr=0.15, force_gpu=True)

    assert step1["status"] == "ttt_adapted"
    assert step3["ttt_steps"] == 3
    # Self-supervised reconstruction loss monotonically decreases across online gradient updates
    assert step3["reconstruction_loss"] < step1["reconstruction_loss"]

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert step1["backend"] == "OpenCL_GPU"

    # Verify synthesize_weights incorporates the TTT fast-weight matrix
    synth = hypernet.synthesize_weights(context, out_dim=16, in_dim=16)
    assert synth["status"] == "synthesized_success"
    assert synth["weights_shape"] == [16, 16]


def test_proposal_6_blelloch_parallel_associative_scan_mamba_ssm():
    """Blelloch / Hillis-Steele O(log N) parallel associative scan matches sequential SSM step-by-step."""
    from core.linear_state_space_microbrain import LinearStateSpaceMicroBrain

    rng = np.random.default_rng(606)
    seq = rng.standard_normal((128, 32)).astype(np.float32)

    ssm_seq = LinearStateSpaceMicroBrain(d_model=32, d_state=16)
    res_seq = ssm_seq.process_sequence(seq.tolist())

    ssm_par = LinearStateSpaceMicroBrain(d_model=32, d_state=16)
    res_par = ssm_par.process_sequence_parallel_scan(seq, force_gpu=True)

    assert res_par["status"] == "sequence_processed_parallel"
    assert res_par["total_tokens"] == 128
    assert res_par["parallel_scan_depth_log2"] == 7  # log2(128) = 7 steps instead of 128 sequential steps
    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert res_par["backend"] == "OpenCL_GPU"

    # Verify numerical parity between O(N) sequential loop and O(log N) GPU SRAM parallel scan
    np.testing.assert_allclose(res_par["final_output"], res_seq["final_output"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ssm_par.hidden_state, ssm_seq.hidden_state, rtol=1e-4, atol=1e-4)

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert res_par["backend"] == "OpenCL_GPU"


def test_proposal_7_difflogic_bitplane_logic_gate_tree():
    """Bitplane Differentiable Logic Gate Tree evaluates 64 samples in 2 uint32 words on GPU."""
    from core.mixture_of_depths_router import MixtureOfDepthsRouter

    router = MixtureOfDepthsRouter(latent_dim=64, exit_threshold=0.5)
    rng = np.random.default_rng(707)

    # Train gates via differentiable De Morgan relaxation toward XOR (gate 6: a ^ b)
    a_prob = np.array([0.0, 0.0, 1.0, 1.0] * 16, dtype=np.float32)
    b_prob = np.array([0.0, 1.0, 0.0, 1.0] * 16, dtype=np.float32)
    xor_target = np.array([0.0, 1.0, 1.0, 0.0] * 16, dtype=np.float32)

    loss_first = router.adapt_difflogic_gates(a_prob, b_prob, xor_target, lr=1.2)["loss"]
    for _ in range(15):
        loss_last = router.adapt_difflogic_gates(a_prob, b_prob, xor_target, lr=1.2)["loss"]
    assert loss_last < loss_first

    # Evaluate a batch of 64 samples (packed into 2 x uint32 words per feature) on GPU
    batch_inputs = rng.standard_normal((64, 64)).astype(np.float32)
    bitplane_res = router.forward_difflogic_bitplane(batch_inputs, force_gpu=True)
    assert bitplane_res["status"] == "difflogic_bitplane_evaluated"
    assert bitplane_res["batch_size"] == 64
    assert bitplane_res["packed_uint32_words"] == 2
    assert len(bitplane_res["early_exit_flags"]) == 64

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert bitplane_res["backend"] == "OpenCL_GPU"


def test_proposal_8_kuramoto_hopfield_micro_transformer():
    """Kuramoto oscillatory phase synchronization + collision-free Modern Hopfield associative MLP."""
    from core.analytical_micro_transformer import AnalyticalMicroTransformer

    brain = AnalyticalMicroTransformer(name="kuramoto_hopfield_test", d_model=32, n_heads=2, d_mlp=8)

    # Compile 12 distinct rules into a brain with d_mlp=8 (legacy modulo k_idx % 8 would collide and overwrite!)
    rules = {f"trigger_{i}": f"target_{i}" for i in range(12)}
    for k_tok, v_tok in rules.items():
        brain.compile_logic_association(k_tok, v_tok, binding_gain=6.0)

    assert len(brain.hopfield_keys) == 12

    # Verify early rules (which collided modulo 8 in the old MLP) are retrieved accurately via Modern Hopfield + Kuramoto
    res_0 = brain.forward_kuramoto_hopfield(["trigger_0"], sync_steps=8, beta=14.0, force_gpu=True)
    res_8 = brain.forward_kuramoto_hopfield(["trigger_8"], sync_steps=8, beta=14.0, force_gpu=True)
    assert res_0["predicted_token"] == "target_0"
    assert res_8["predicted_token"] == "target_8"
    assert 0.0 <= res_0["kuramoto_order_r"] <= 1.0001

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert res_0["backend"] == "OpenCL_GPU"


def test_proposal_9_predictive_coding_sram_variational_relaxation():
    """Iterative Variational Free-Energy Relaxation in GPU SRAM reduces free energy F."""
    from core.predictive_coding_engine import HierarchicalPredictiveCodingEngine

    pc = HierarchicalPredictiveCodingEngine(layer_dims=[32, 24, 16])
    rng = np.random.default_rng(909)
    pos_batch = rng.standard_normal((32, 32)).astype(np.float32)

    res = pc.relax_and_adapt_pff(pos_batch, relax_steps=20, state_lr=0.08, force_gpu=True)
    assert res["status"] == "pff_sram_relaxed"
    assert res["batch_size"] == 32
    assert res["final_free_energy"] < res["initial_free_energy"]
    assert res["free_energy_reduction_pct"] > 0.0

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert res["backend"] == "OpenCL_GPU"


def test_proposal_10_active_inference_monte_carlo_efe_rollout():
    """1,024-particle multi-step Expected Free Energy (EFE) rollout on GT 710 GPU."""
    from core.active_inference import ActiveInferenceCluster

    cluster = ActiveInferenceCluster()
    state = {
        "drives": {"safety": 0.45, "curiosity": 0.65, "integrity": 0.60},
        "homeostasis": 0.55,
    }
    res = cluster.run_monte_carlo_efe_rollout(
        state,
        goal="fix broken parser bug carefully",
        n_particles=256,
        horizon=8,
        noise_scale=0.04,
        force_gpu=True,  # explicit hardware exercise; honest dispatch defaults to CPU
    )
    assert res["status"] == "mc_efe_complete"
    assert res["total_particles"] == 1024  # 4 personas * 256 particles
    assert res["horizon"] == 8
    assert res["winning_persona"] in ("aura", "zeno", "iris", "eris")
    assert len(res["ranked_personas"]) == 4

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert res["backend"] == "OpenCL_GPU"


def test_proposal_11_multiscale_vietoris_rips_geodesic_slerp():
    """Multi-scale Vietoris-Rips persistence filtration (16 x 32 = 512 graphs) on GPU."""
    from core.topological_persistence_merger import TopologicalPersistenceMerger

    merger = TopologicalPersistenceMerger()
    rng = np.random.default_rng(1111)
    wa = rng.standard_normal(128).astype(np.float32)
    wb = rng.standard_normal(128).astype(np.float32)

    res = merger.multiscale_geodesic_slerp_merge(wa, wb, n_t=16, n_eps=32, force_gpu=True)
    assert res["status"] == "multiscale_geodesic_merged"
    assert res["grid_evaluations"] == 512
    assert 0.0 < res["optimal_t"] < 1.0
    assert len(res["beta0_barcode"]) == 32
    assert len(res["beta1_barcode"]) == 32
    assert len(res["fused_weights"]) == 128

    accel = OpenCLMicroBrainAccelerator.get_instance()
    if accel.gpu_available:
        assert res["backend"] == "OpenCL_GPU"


# ==============================================================================
# Wave-3 Measured GT 710 Co-Processor Proposals 1 - 8
# ==============================================================================


def test_wave3_proposal_1_autotuned_crossover_dispatch():
    """Per-kernel crossover autotuner; honest bootstrap keeps small batches on CPU."""
    accel = OpenCLMicroBrainAccelerator.get_instance()
    crossover = accel.get_crossover("bitnet_batch_gemm")
    assert crossover >= 1024  # Honest bootstrap: BitNet GPU measured 5-8x slower at batch 32-4096

    # Check should_use_gpu dispatch decisions
    assert accel.should_use_gpu("bitnet_batch_gemm", batch_size=32, force_gpu=False) is False
    assert accel.should_use_gpu("bitnet_batch_gemm", batch_size=32, force_gpu=True) == accel.gpu_available

    # Dynamic crossover adjustment (restored in finally — order-independent)
    orig = accel.get_crossover("bitnet_batch_gemm")
    try:
        accel.set_crossover("bitnet_batch_gemm", 16)
        assert accel.get_crossover("bitnet_batch_gemm") == 16
        assert accel.should_use_gpu("bitnet_batch_gemm", batch_size=32, force_gpu=False) == accel.gpu_available
    finally:
        accel.set_crossover("bitnet_batch_gemm", orig)


def test_wave3_proposal_2_resident_vram_codebooks_and_weights():
    """Persistent VRAM buffers eliminate repeated PCIe 2.0 uploads."""
    accel = OpenCLMicroBrainAccelerator.get_instance()

    # 1. Resident HDC codebook
    rng = np.random.default_rng(202)
    cb_hv = rng.integers(0, 2**32, size=(64, accel.HDC_WORDS), dtype=np.uint32)
    meta = [{"idx": i, "tag": f"goal_{i}"} for i in range(64)]

    reg_res = accel.register_resident_hdc_codebook("test_experience_cb", cb_hv, metadata=meta)
    assert reg_res["name"] == "test_experience_cb"
    assert reg_res["n_items"] == 64

    q_hv = cb_hv[12].copy()
    search_res = accel.search_resident_hdc_codebook("test_experience_cb", q_hv, top_k=3, force_gpu=True)
    assert search_res["found"] is True
    # pcie_codebook_bytes_uploaded == 0 was removed: the field is a design
    # constant for resident buffers (it can never be nonzero), so asserting
    # it laundered a tautology into a contract.  Assert the backend actually
    # used instead — the thing force_gpu=True is supposed to control.
    if accel.gpu_available:
        assert search_res["backend"] == "OpenCL_GPU"
    assert search_res["top_indices"][0] == 12
    assert search_res["top_matches"][0]["tag"] == "goal_12"

    # 2. Resident BitNet ternary weights
    w_ternary = rng.choice([-1, 0, 1], size=(48, 64)).astype(np.int8)
    reg_w = accel.register_resident_bitnet_weights("test_w_key", w_ternary)
    x_q = rng.choice([-1, 0, 1], size=(32, 64)).astype(np.int8)
    out, backend = accel.bitnet_forward(
        x_q, w_ternary, combined_scale=0.05, force_gpu=True, resident_weight_key="test_w_key"
    )
    assert out.shape == (32, 48)


def test_wave3_proposal_3_true_eggroll_es_step_on_device_rng(tmp_path, monkeypatch):
    """True EGGROLL on-device counter RNG rank-r ES gradient step in VRAM."""
    accel = OpenCLMicroBrainAccelerator.get_instance()
    rng = np.random.default_rng(303)
    dim = 32
    w_base = rng.standard_normal((dim, dim)).astype(np.float32)
    x_samples = rng.standard_normal((16, dim)).astype(np.float32)
    y_targets = rng.standard_normal((16, dim)).astype(np.float32)

    es_res = accel.eggroll_es_step(
        w_base=w_base,
        x_samples=x_samples,
        y_targets=y_targets,
        pop_size=64,
        rank_r=2,
        sigma=0.25,
        lr=0.20,
        seed=42,
        force_gpu=True,
    )
    assert es_res["population_size"] == 64
    assert es_res["rank_r"] == 2
    # pcie_perturbation_bytes_uploaded == 0 removed (tautology — see the
    # resident-codebook test); the discriminating assertion is the backend.
    if accel.gpu_available:
        assert es_res["backend"] == "OpenCL_GPU"
        assert es_res["device"] == accel.device_name
    assert es_res["updated_weights"].shape == (dim, dim)
    assert es_res["behavior_descriptors"].shape == (64, 8)

    # Test through nursery
    import core.brain_nursery as nursery_mod

    monkeypatch.setattr(nursery_mod, "STORE_DIR", tmp_path)
    nursery = AutonomousBrainNursery()
    nursery.grow_brain_for_domain("eggroll_es_domain", rules={"x": "y"})
    n_res = nursery.evolve_population_eggroll_es(
        "eggroll_es_domain",
        x_samples=x_samples,
        y_targets=y_targets,
        pop_size=64,
        rank_r=2,
        force_gpu=True,
    )
    assert n_res["brain_name"] == "neuro_eggroll_es_domain"


def test_wave3_proposal_4_evolution_surrogate_prefilter():
    """Chronological 80/20 train/test split and honesty gate on evolution surrogate."""
    from core.evolution_surrogate_prefilter import EvolutionSurrogatePrefilter, ResidentHDCExperienceMemory

    mem = ResidentHDCExperienceMemory.get_instance()
    assert mem.n_records >= 4

    surrogate = EvolutionSurrogatePrefilter.get_instance()
    assert surrogate.status in ("calibrated_active", "unmeasured")

    risk_eval = surrogate.evaluate_goal_risk("causal manifold surgical repair regression fix")
    assert "risk_score" in risk_eval
    assert "near_duplicate" in risk_eval
    assert "held_out_auc" in risk_eval

    # Honesty gate verification: if held_out_auc <= 0.55, it must abstain as 'unmeasured'
    if surrogate.held_out_auc <= 0.55:
        assert risk_eval["status"] == "unmeasured"
        assert risk_eval["advising"] is False
    else:
        assert risk_eval["status"] == "calibrated_active"
        assert risk_eval["advising"] is True


def test_wave3_proposal_5_split_conformal_calibration_and_calibrated_abstention():
    """Split-conformal calibration with guaranteed singleton coverage & OOD abstention."""
    from core.analytical_micro_transformer import AnalyticalMicroTransformer

    brain = AnalyticalMicroTransformer(name="conformal_brain_test", d_model=32, n_heads=2)
    # Compile 6 rules
    rules = {f"in_{i}": f"out_{i}" for i in range(6)}
    for k, v in rules.items():
        brain.compile_logic_association(k, v)

    # Calibration set of known exchangeable examples
    cal_data = [([f"in_{i}"], f"out_{i}") for i in range(6)]
    cal_res = brain.calibrate_conformal(cal_data, alpha=0.10)
    assert cal_res["status"] == "calibrated"
    assert 0.0 <= cal_res["quantile_q_hat"] <= 1.0

    # In-distribution inference -> singleton prediction (no abstention)
    inf_in = brain.infer_conformal(["in_2"])
    assert inf_in["singleton"] is True
    assert inf_in["abstained"] is False
    assert inf_in["reason"] == "singleton"
    assert inf_in["predicted_token"] == "out_2"

    # Out-of-distribution (OOD) random noise input -> abstains
    inf_ood = brain.infer_conformal(["random_unknown_token_xyz_999"])
    assert inf_ood["abstained"] is True
    assert inf_ood["reason"] in ("ambiguous", "ood", "ood_coverage_breach")


def test_wave3_proposal_6_resident_hdc_experience_memory():
    """Large-scale resident HDC memory for near-duplicate and nearest-failure lookup."""
    from core.evolution_surrogate_prefilter import ResidentHDCExperienceMemory

    mem = ResidentHDCExperienceMemory.get_instance()
    # Test near-duplicate query for identical text
    dup_res = mem.detect_near_duplicate("causal manifold surgical repair regression fix")
    assert dup_res["similarity"] > 0.85
    assert dup_res["is_near_duplicate"] is True

    # Test dissimilar query
    dist_res = mem.detect_near_duplicate("astronomy telescope exoplanet spectrum detection")
    assert dist_res["similarity"] < 0.85
    assert dist_res["is_near_duplicate"] is False


def test_wave3_proposal_7_dominated_novelty_search_quality_diversity(tmp_path, monkeypatch):
    """Dominated Novelty Search (DNS, 2025 SOTA) GPU Pareto quality-diversity."""
    accel = OpenCLMicroBrainAccelerator.get_instance()
    rng = np.random.default_rng(707)

    pop_size = 32
    desc_dim = 8
    # Candidate 0 has highest fitness; candidates 1..31 have lower fitness
    fitness = np.linspace(-10.0, -1.0, pop_size, dtype=np.float32)
    fitness[0] = 50.0  # Dominant global elite
    descriptors = rng.standard_normal((pop_size, desc_dim)).astype(np.float32)

    dns_res = accel.dominated_novelty_search(fitness, descriptors, k_nearest=4, force_gpu=True)
    assert dns_res["pop_size"] == 32
    assert len(dns_res["dns_scores"]) == 32

    # Global un-dominated elite gets maximum novelty bonus (+10.0)
    assert dns_res["dns_scores"][0] > 10.0

    # Test nursery integration
    import core.brain_nursery as nursery_mod

    monkeypatch.setattr(nursery_mod, "STORE_DIR", tmp_path)
    nursery = AutonomousBrainNursery()
    nursery.grow_brain_for_domain("dns_domain", rules={"a": "b"})
    dns_evo = nursery.evolve_population_dns(
        "dns_domain",
        x_samples=rng.standard_normal((16, 32)).astype(np.float32),
        y_targets=rng.standard_normal((16, 32)).astype(np.float32),
        pop_size=32,
        k_nearest=3,
        force_gpu=True,
    )
    assert dns_evo["status"] == "dns_evolved"
    assert dns_evo["brain_name"] == "neuro_dns_domain"


def test_wave3_proposal_8_dirichlet_epistemic_info_gain():
    """Dirichlet/Beta expected KL divergence discriminates uncertain vs explored beliefs."""
    accel = OpenCLMicroBrainAccelerator.get_instance()
    init_drives = np.array([0.5, 0.5, 0.5, 0.7], dtype=np.float32)
    shifts = np.zeros((2, 4), dtype=np.float32)
    weights = np.ones((2, 4), dtype=np.float32)
    weights[:, 1] = 1.2  # Curiosity drive active
    prefs = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32)

    # Policy 0: untried, highly uncertain prior (alpha=1.0, beta=1.0)
    # Policy 1: exhaustively observed, low-uncertainty belief (alpha=50.0, beta=50.0)
    beliefs = np.array([[1.0, 1.0], [50.0, 50.0]], dtype=np.float32)

    mc_res = accel.active_inference_mc_rollout(
        init_drives=init_drives,
        policy_shifts=shifts,
        persona_weights=weights,
        target_prefs=prefs,
        n_particles=128,
        horizon=4,
        belief_counts=beliefs,
    )
    pol0 = mc_res["policy_metrics"][0]
    pol1 = mc_res["policy_metrics"][1]

    # Untried policy 0 MUST yield strictly higher epistemic information gain (nats of Bayesian surprise)
    assert pol0["dirichlet_kl_nats"] > pol1["dirichlet_kl_nats"]
    assert pol0["epistemic_info_gain"] > pol1["epistemic_info_gain"]
def test_get_best_execution_provider_returns_a_real_provider():
    """The helper was imported by this module but never exercised; pin its contract.

    It must return a valid ONNX Runtime provider name (or the honest CPU
    fallback when onnxruntime is absent) — never an empty/garbage string.
    """
    provider = get_best_execution_provider(batch_size=64)
    valid = {
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    }
    assert provider in valid
    if provider != "CPUExecutionProvider":
        import onnxruntime as _ort  # noqa: F401  (only reachable when installed)

        assert provider in _ort.get_available_providers()
