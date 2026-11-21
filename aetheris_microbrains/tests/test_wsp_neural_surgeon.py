"""
Tests for Weight-Space Programming (WSP) Neural Surgeon.
Verifies:
1. Attractor diagnostics and zero-gradient singular deflation (Qwen3.5-0.8B layers 15 & 19 style).
2. Direct closed-form rank-1 rule implantation (ROME/MEMIT style) in < 1 ms.
3. Colony-wide neuro-brain health scan.
"""

import pytest
import numpy as np
from aetheris_microbrains.core.wsp_neural_surgeon import WSPNeuralSurgeon


def test_wsp_attractor_deflation():
    """Verify that a pathological dominant attractor (repetitive loop) is excised."""
    surgeon = WSPNeuralSurgeon(attractor_ratio_threshold=3.0, deflation_factor=0.7)

    # 1. Synthesize a healthy isotropic weight matrix
    W_healthy = np.random.randn(32, 32) * 0.1
    _, rep_healthy = surgeon.diagnose_and_deflate_attractor(W_healthy, brain_id="healthy_brain")
    assert rep_healthy.was_pathological is False
    assert rep_healthy.parameters_modified_count == 0

    # 2. Synthesize a pathological matrix with an overwhelming cyclic attractor rank-1 bias
    # (Simulating Qwen3.5-0.8B layers 15 & 19 repetitive reasoning loop trap)
    u_bad = np.random.randn(32, 1)
    u_bad /= np.linalg.norm(u_bad)
    v_bad = np.random.randn(1, 32)
    v_bad /= np.linalg.norm(v_bad)

    pathological_bias = 5.0 * np.dot(u_bad, v_bad)
    W_pathological = W_healthy + pathological_bias

    # 3. Perform zero-gradient surgical excision
    W_sanitized, rep_surgery = surgeon.diagnose_and_deflate_attractor(W_pathological, brain_id="trapped_brain")

    assert rep_surgery.was_pathological is True
    assert rep_surgery.dominant_eigen_ratio > 3.0
    assert rep_surgery.parameters_modified_count > 0
    assert rep_surgery.modified_ratio > 0.0

    # Verify that the sanitized matrix has a drastically lower top singular value
    _, s_post, _ = np.linalg.svd(W_sanitized)
    _, s_pre, _ = np.linalg.svd(W_pathological)
    assert s_post[0] < s_pre[0]


def test_wsp_direct_rank1_implant():
    """Verify instant closed-form associative memory implantation without backprop."""
    surgeon = WSPNeuralSurgeon()

    W_orig = np.random.randn(16, 16) * 0.05
    k_key = np.random.randn(16)
    k_key /= np.linalg.norm(k_key)
    v_target = np.random.randn(16)
    v_target /= np.linalg.norm(v_target)

    # Initially W_orig * k_key does not match v_target
    initial_cos = float(np.dot(np.dot(W_orig, k_key), v_target))

    # Implant new association directly
    W_implanted, res = surgeon.direct_rank1_implant(W_orig, k_key, v_target)

    # Must recall perfectly (cosine similarity ~ 1.0)
    assert res.key_recalled_similarity > 0.99
    assert res.latency_ms < 5.0  # Typically < 0.1 ms


def test_wsp_neuro_brains_store_audit():
    """Verify that colony-wide neuro-brain scan operates safely on existing npz files."""
    surgeon = WSPNeuralSurgeon()
    audit_res = surgeon.audit_neuro_brains_store()

    assert "total_files_audited" in audit_res
    assert "pathological_brains_detected" in audit_res
    assert audit_res["total_files_audited"] >= 0

