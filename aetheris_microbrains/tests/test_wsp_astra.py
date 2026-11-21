"""
Unit tests for Astra-Style Weight-Space Programming Improvements:
1. AstraSandboxTracer: cyclic loop/attractor diagnosis.
2. AstraColumnSubtraction: subtractive directional projection on top aligned columns.
3. InMemoryABProbe: zero-side-effect evaluation battery before patch commitment.
"""

import numpy as np
import pytest

from aetheris_microbrains.evolution.wsp_substrate import CPUMicroRouter
from aetheris_microbrains.evolution.wsp_engine import WSPEngine
from aetheris_microbrains.evolution.wsp_sandbox_tracer import AstraSandboxTracer
from aetheris_microbrains.evolution.wsp_ab_probe import InMemoryABProbe


@pytest.fixture
def clean_astra_env(tmp_path, monkeypatch):
    test_store = tmp_path / "memory_store"
    test_store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_substrate.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_substrate.BASE_WEIGHTS_FILE", test_store / "wsp_base_weights.npz")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_substrate.ACTIVE_PATCHES_FILE", test_store / "wsp_active_patches.npz")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_engine.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_engine.WSP_LEDGER_FILE", test_store / "wsp_ledger.jsonl")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_lenses.LENSES_DIR", test_store / "wsp_lenses")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_sae.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_sae.SAE_DICTIONARY_FILE", test_store / "wsp_sae_dictionary.npz")
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_sae.CONCEPT_REGISTRY_FILE", test_store / "wsp_concept_registry.json")
    return test_store


def test_astra_sandbox_tracer_attractor_detection(clean_astra_env):
    tracer = AstraSandboxTracer(d_hidden=32, buffer_capacity=30)

    # Simulate cyclic loop: alternating between two vectors with noise
    base_v = np.ones(32, dtype=np.float32)
    for i in range(12):
        step_act = base_v * (1.5 if i % 2 == 0 else 0.5) + np.random.randn(32).astype(np.float32) * 0.05
        tracer.record_activation(step_act)

    diag = tracer.diagnose_attractor()
    assert diag is not None
    assert "is_cyclic_rut" in diag
    assert "v_attractor" in diag
    assert diag["v_attractor"].shape == (32,)
    assert diag["attractor_energy_ratio"] > 0.4


def test_astra_column_subtraction_surgical_edit(clean_astra_env):
    router = CPUMicroRouter(d_in=32, d_hidden=16, d_out=4, seed=42)
    engine = WSPEngine(router=router)

    # Attractor aligned with hidden output space (shape 16,)
    v_attractor = np.zeros(16, dtype=np.float32)
    v_attractor[0] = 1.0

    # Initial projection of W1 columns onto attractor: (d_in, d_hidden) @ (d_hidden, 1) -> (d_in, 1)
    proj_before = np.abs(np.matmul(router.W1_effective, v_attractor)).sum()

    # Apply Astra column subtraction on top 4 aligned columns
    res = engine.apply_astra_column_subtraction(
        v_attractor=v_attractor,
        layer_name="W1",
        n_columns=4,
        beta=0.5,
    )

    assert res["operation"] == "astra_column_subtraction"
    assert res["columns_patched"] == 4
    assert res["params_touched"] > 0

    proj_after = np.abs(np.matmul(router.W1_effective, v_attractor)).sum()
    assert proj_after < proj_before


def test_in_memory_ab_probe_evaluation(clean_astra_env):
    router = CPUMicroRouter(d_in=32, d_hidden=16, d_out=4, seed=42)
    probe = InMemoryABProbe(router=router, n_probes=5, seed=42)

    small_delta_W1 = np.random.randn(32, 16).astype(np.float32) * 0.005
    small_delta_W2 = np.zeros((16, 4), dtype=np.float32)

    eval_res = probe.evaluate_candidate_patch(small_delta_W1, small_delta_W2)
    assert eval_res["passed"] is True
    assert eval_res["eval_time_ms"] < 5.0

    assert np.all(router.delta_W1 == 0.0)
    assert np.all(router.delta_W2 == 0.0)
