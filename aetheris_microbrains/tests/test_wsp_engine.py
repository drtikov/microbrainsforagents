"""
Comprehensive test suite for the 4 Advanced Pillars of Weight-Space Programming in Aetheris:
1. Concept-Level Sparse Autoencoders (SAE)
2. Execution-Driven Neural Repair (Auto Failure Loop Interception)
3. Epigenetic LoRA Lenses (Sub-millisecond mode switching)
4. Formal Spectral Invariants (Lyapunov Stability & Bound Protection)
"""

import json
import numpy as np
import pytest

from aetheris_microbrains.evolution.wsp_substrate import CPUMicroRouter, ROMEAssociativeMemory
from aetheris_microbrains.evolution.wsp_sae import ConceptSparseAutoencoder
from aetheris_microbrains.evolution.wsp_lenses import EpigeneticLensManager
from aetheris_microbrains.evolution.wsp_engine import WSPEngine, WSPSpectralInvariantViolation


@pytest.fixture
def clean_wsp_env(tmp_path, monkeypatch):
    test_store = tmp_path / "memory_store"
    test_store.mkdir(parents=True, exist_ok=True)
    lenses_dir = test_store / "wsp_lenses"
    lenses_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_substrate.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_substrate.BASE_WEIGHTS_FILE", test_store / "wsp_base_weights.npz")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_substrate.ACTIVE_PATCHES_FILE", test_store / "wsp_active_patches.npz")

    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_sae.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_sae.SAE_DICTIONARY_FILE", test_store / "wsp_sae_dictionary.npz")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_sae.CONCEPT_REGISTRY_FILE", test_store / "wsp_concept_registry.json")

    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_lenses.LENSES_DIR", lenses_dir)

    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_engine.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_engine.WSP_LEDGER_FILE", test_store / "wsp_ledger.jsonl")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_engine.REPETITIVE_FAILURES_FILE", test_store / "wsp_failure_tracker.json")

    return test_store


# -----------------------------------------------------------------------------
# Pillar 1: Sparse Autoencoder Concept Steering
# -----------------------------------------------------------------------------
def test_sae_concept_decomposition_and_steering(clean_wsp_env):
    sae = ConceptSparseAutoencoder(d_model=32, expansion_factor=4, k_active=4, seed=42)
    x = np.random.randn(32).astype(np.float32)

    z_sparse = sae.encode(x)
    assert z_sparse.shape == (128,)
    # Exactly k_active nonzeros
    assert np.count_nonzero(z_sparse) == 4

    x_hat = sae.decode(z_sparse)
    assert x_hat.shape == (32,)

    # Steer concept 0 (core safety)
    x_steered = sae.steer_concept(x, concept_idx=0, factor=3.0)
    assert x_steered.shape == (32,)
    assert not np.allclose(x_hat, x_steered)


# -----------------------------------------------------------------------------
# Pillar 2: Execution-Driven Neural Repair
# -----------------------------------------------------------------------------
def test_execution_driven_neural_repair_loop_trigger(clean_wsp_env):
    router = CPUMicroRouter(d_in=32, d_hidden=16, d_out=4, seed=42)
    engine = WSPEngine(router=router)

    # First two failures track count without triggering repair
    r1 = engine.report_execution_failure("tool_execution", "SyntaxErrorInDynamicTool")
    assert r1["count"] == 1
    assert not r1["repair_triggered"]

    r2 = engine.report_execution_failure("tool_execution", "SyntaxErrorInDynamicTool")
    assert r2["count"] == 2
    assert not r2["repair_triggered"]

    # 3rd identical failure triggers automatic emergency SVD repair
    r3 = engine.report_execution_failure("tool_execution", "SyntaxErrorInDynamicTool")
    assert r3["count"] == 3
    assert r3["repair_triggered"]
    assert r3["repair_details"] is not None
    assert r3["repair_details"]["operation"] == "damp_dominant_eigenvalue"


# -----------------------------------------------------------------------------
# Pillar 3: Epigenetic LoRA Lenses
# -----------------------------------------------------------------------------
def test_epigenetic_lens_switching(clean_wsp_env):
    router = CPUMicroRouter(d_in=128, d_hidden=64, d_out=16, seed=42)
    engine = WSPEngine(router=router)

    # Verify default lenses exist
    lenses = engine.lens_manager.list_lenses()
    lens_names = [l["name"] for l in lenses]
    assert "lens_audit_guard" in lens_names
    assert "lens_creative_research" in lens_names

    x = np.random.randn(128).astype(np.float32)
    logits_base = router.forward(x, use_patches=False)

    # Activate audit guard lens (< 0.1ms)
    act_res = engine.activate_lens("lens_audit_guard")
    assert act_res["lens_name"] == "lens_audit_guard"
    logits_audit = router.forward(x, use_patches=True)
    assert not np.allclose(logits_base, logits_audit)

    # Reset to base
    engine.clear_lens()
    logits_cleared = router.forward(x, use_patches=True)
    assert np.allclose(logits_base, logits_cleared)


# -----------------------------------------------------------------------------
# Pillar 4: Formal Spectral Invariants (Safety Gate)
# -----------------------------------------------------------------------------
def test_formal_spectral_invariant_enforcement(clean_wsp_env):
    router = CPUMicroRouter(d_in=32, d_hidden=16, d_out=4, seed=42)
    engine = WSPEngine(router=router, max_frobenius_ratio=1.2)

    # Normal small patch passes verification
    u = np.ones(32, dtype=np.float32) / np.sqrt(32)
    v = np.ones(16, dtype=np.float32) / np.sqrt(16)
    res = engine.synthesize_rank1_patch("W1", u, v, alpha=0.01)
    assert res["operation"] == "synthesize_rank1_patch"

    # Explosive patch (alpha=50.0) violates Frobenius norm ratio invariant
    with pytest.raises(WSPSpectralInvariantViolation):
        engine.synthesize_rank1_patch("W1", u, v, alpha=50.0)
