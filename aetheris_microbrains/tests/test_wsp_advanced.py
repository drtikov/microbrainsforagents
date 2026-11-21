"""
Tests for Weight-Space Task Arithmetic, TIES/DARE Merging, Model Soups,
and CPU Epistemic Uncertainty Oracle.
"""

import numpy as np
import pytest

from aetheris_microbrains.evolution.wsp_merger import WeightSpaceMerger
from aetheris_microbrains.evolution.wsp_epistemic_oracle import EpistemicOracle
from aetheris_microbrains.evolution.wsp_lenses import EpigeneticLensManager


@pytest.fixture
def clean_merger_env(tmp_path, monkeypatch):
    test_store = tmp_path / "memory_store"
    test_store.mkdir(parents=True, exist_ok=True)
    lenses_dir = test_store / "wsp_lenses"
    lenses_dir.mkdir(parents=True, exist_ok=True)
    soups_dir = test_store / "wsp_soups"
    soups_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_epistemic_oracle.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_epistemic_oracle.ORACLE_WEIGHTS_FILE", test_store / "wsp_epistemic_oracle.npz")

    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_merger.STORE_DIR", test_store)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_merger.SOUPS_DIR", soups_dir)
    monkeypatch.setattr("aetheris_microbrains.evolution.wsp_lenses.LENSES_DIR", lenses_dir)

    return {"store": test_store, "lenses": lenses_dir, "soups": soups_dir}


def test_dare_stochastic_dropping():
    delta = np.ones((100, 100), dtype=np.float32)
    # 50% drop rate
    dare_delta = WeightSpaceMerger.apply_dare(delta, drop_rate=0.5, seed=42)

    # Approximately half should be zeros
    n_zeros = np.count_nonzero(dare_delta == 0.0)
    assert 4000 <= n_zeros <= 6000

    # Non-zero elements should be rescaled by 1 / (1 - 0.5) = 2.0
    non_zeros = dare_delta[dare_delta != 0.0]
    assert np.allclose(non_zeros, 2.0)


def test_ties_merging_resolves_conflict():
    # Model 1 wants +1.0 in coord [0, 0]
    d1 = np.zeros((4, 4), dtype=np.float32)
    d1[0, 0] = 1.0
    d1[1, 1] = 0.8

    # Model 2 also wants +0.8 in [0, 0], but opposes [1, 1] with -0.9
    d2 = np.zeros((4, 4), dtype=np.float32)
    d2[0, 0] = 0.8
    d2[1, 1] = -0.9

    # Model 3 agrees with Model 1 on positive direction for [1, 1]
    d3 = np.zeros((4, 4), dtype=np.float32)
    d3[0, 0] = 0.9
    d3[1, 1] = 0.7

    merged = WeightSpaceMerger.ties_merge([d1, d2, d3], k_sparsity=0.5)
    # [0, 0] should be positive average
    assert merged[0, 0] > 0.0
    # [1, 1]: 2 models wanted positive (+0.8, +0.7 = +1.5), 1 wanted negative (-0.9).
    # Elected sign is positive. Negative model 2 should be filtered out.
    assert merged[1, 1] > 0.0


def test_model_soup_brewing(clean_merger_env):
    merger = WeightSpaceMerger(soups_dir=clean_merger_env["soups"])

    attempts = [
        {"delta_W1": np.ones((8, 4), dtype=np.float32) * 0.1, "delta_W2": np.ones((4, 2), dtype=np.float32) * 0.2},
        {"delta_W1": np.ones((8, 4), dtype=np.float32) * 0.15, "delta_W2": np.ones((4, 2), dtype=np.float32) * 0.18},
    ]
    scores = [0.85, 0.95]

    res = merger.brew_model_soup(attempts, scores, name="soup_test")
    assert res["name"] == "soup_test"
    assert (clean_merger_env["soups"] / "soup_test.npz").exists()


def test_epigenetic_hybrid_lens_merging(clean_merger_env):
    mgr = EpigeneticLensManager(lenses_dir=clean_merger_env["lenses"])
    mgr._ensure_default_lenses()

    # Merge audit guard and creative research into hybrid lens
    res = mgr.merge_lenses(
        lens_names=["lens_audit_guard", "lens_creative_research"],
        merged_name="lens_hybrid_audit_creative",
        method="ties",
    )
    assert res["merged_name"] == "lens_hybrid_audit_creative"

    # Verify loaded hybrid lens
    loaded = mgr.load_lens("lens_hybrid_audit_creative")
    assert loaded["delta_W1"].shape == (128, 64)
    assert loaded["delta_W2"].shape == (64, 16)


def test_cpu_epistemic_oracle(clean_merger_env):
    oracle = EpistemicOracle(d_in=32, d_repr=16, seed=42)
    x = np.random.randn(32).astype(np.float32)

    # Initial query on unfamiliar vector should have uncertainty
    u1 = oracle.compute_uncertainty(x)
    assert "uncertainty_score" in u1
    assert "confidence" in u1
    init_err = u1["uncertainty_score"]

    # Register experience (learning on CPU in < 0.02ms)
    for _ in range(5):
        oracle.register_familiar_experience(x)

    # Re-evaluating familiar vector should show reduced uncertainty and higher confidence
    u2 = oracle.compute_uncertainty(x)
    assert u2["uncertainty_score"] < init_err
    assert u2["confidence"] > u1["confidence"]

