"""Regression tests for the 2026-11-14 microbrain audit fixes.

Live findings being pinned (full audit of the 5 microbrain modules):
1. ``_counter_rademacher_matrix`` claimed to be a bit-exact host mirror of the
   OpenCL ``eggroll_counter_rademacher`` kernel but used different constants
   AND different mixing (measured 47.9% sign mismatch) - whenever the ES
   rollout ran on GPU, the gradient paired kernel-measured fitness with
   unrelated perturbation directions.
2. ``rips_multiscale_slerp_search`` silently truncated inputs to 32 points
   (register union-find cap) with zero disclosure; 40-pt and 32-pt barcodes
   were identical.
3. The Kuramoto GPU kernel raced: each work-item ran its own multi-step sync
   loop reading other threads' phases mid-flight, diverging from the CPU
   path's synchronous Jacobi update.
4. ``synthesize_weights`` presented an untrained fixed hash projection as
   "task-specialized" weights (measured 0.484 correlation between unrelated
   task descriptions) without an untrained-status disclosure.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aetheris_microbrains.core.opencl_microbrain_accelerator import (
    OpenCLMicroBrainAccelerator,
    get_opencl_accelerator,
)

_ACCEL_SRC = (
    Path(__file__).resolve().parents[1] / "core" / "opencl_microbrain_accelerator.py"
).read_text(encoding="utf-8")

M32 = 0xFFFFFFFF


def _kernel_rademacher_ref(seed, pair_id, r_idx, coord, stream):
    """Pure-Python transcription of the OpenCL eggroll_counter_rademacher kernel.

    Sum the four field products in uint32, XOR the seed, then one
    xor-shift-multiply mix.  This is the GPU ground truth the host matrix
    must reproduce bit-for-bit.
    """
    s = (
        pair_id * 374761393
        + r_idx * 668265263
        + coord * 2147483647
        + stream * 1274126177
    ) & M32
    h = (seed & M32) ^ s
    h = ((h ^ (h >> 13)) * 1274126177) & M32
    h ^= h >> 16
    return 1.0 if h & 1 else -1.0


class TestHostMirrorBitExactness:
    def test_host_matrix_matches_kernel_hash_exactly(self):
        seed, n_pairs, rank_r, dim = 2026, 24, 2, 32
        for stream in (0, 1):
            host = OpenCLMicroBrainAccelerator._counter_rademacher_matrix(
                seed, n_pairs, rank_r, dim, stream
            )
            assert host.shape == (n_pairs, rank_r, dim)
            for p in range(n_pairs):
                for r in range(rank_r):
                    for c in range(dim):
                        assert host[p, r, c] == _kernel_rademacher_ref(
                            seed, p, r, c, stream
                        ), f"stream={stream} pair={p} r={r} coord={c}"

    def test_kernel_and_host_share_the_hash_constants(self):
        # Tripwire: if either side's constants change, the mirror breaks.
        for const in ("374761393", "668265263", "2147483647", "1274126177"):
            assert _ACCEL_SRC.count(const) >= 2, const  # kernel + host

    def test_es_gpu_candidate_fitness_matches_cpu_when_gpu_present(self):
        """The direct consequence of the fix: kernel-measured per-candidate MSE
        (GPU) now describes the SAME perturbations the host reconstructs."""
        acc = get_opencl_accelerator()
        if not acc.gpu_available:
            pytest.skip("no OpenCL GPU on this box")
        rng = np.random.default_rng(3)
        dim = 16
        w = rng.uniform(-1, 1, size=(dim, dim)).astype(np.float32)
        xs = rng.uniform(-1, 1, size=(8, dim)).astype(np.float32)
        ys = np.tanh(xs @ w.T / 4.0).astype(np.float32)
        r_gpu = acc.eggroll_es_step(w, xs, ys, pop_size=8, seed=3, force_gpu=True)
        r_cpu = acc.eggroll_es_step(w, xs, ys, pop_size=8, seed=3, force_gpu=False)
        assert r_gpu["backend"] == "OpenCL_GPU"
        assert np.allclose(
            np.asarray(r_gpu["candidate_mses"], dtype=np.float64),
            np.asarray(r_cpu["candidate_mses"], dtype=np.float64),
            atol=1e-3,
        )


class TestEggrollReturnShape:
    def test_candidate_mses_key_is_not_duplicated(self):
        acc = get_opencl_accelerator()
        rng = np.random.default_rng(7)
        w = rng.uniform(-1, 1, size=(16, 16)).astype(np.float32)
        xs = rng.uniform(-1, 1, size=(8, 16)).astype(np.float32)
        res = acc.eggroll_es_step(
            w, xs, np.tanh(xs @ w.T / 4.0).astype(np.float32), pop_size=8, seed=0
        )
        keys = list(res.keys())
        assert keys.count("candidate_mses") == 1
        assert res["mse_reduced"] in (True, False)


class TestRipsDisclosure:
    def test_truncation_is_disclosed_and_behavior_unchanged(self):
        acc = get_opencl_accelerator()
        rng = np.random.default_rng(5)
        ua = rng.uniform(0, 1, size=(40, 6)).astype(np.float32)
        ub = rng.uniform(0, 1, size=(40, 6)).astype(np.float32)
        r40 = acc.rips_multiscale_slerp_search(ua, ub, theta=0.7)
        assert r40["points_evaluated"] == 32
        assert r40["points_truncated"] is True
        r32 = acc.rips_multiscale_slerp_search(
            ua[:32].copy(), ub[:32].copy(), theta=0.7
        )
        assert r32["points_truncated"] is False
        # Document the pinned behavior: the barcode describes only the first 32
        assert r40["beta0_barcode"] == r32["beta0_barcode"]
        assert r40["beta1_barcode"] == r32["beta1_barcode"]


class TestKuramotoSynchronousSemantics:
    def test_kernel_source_uses_ping_pong_single_steps(self):
        assert "kuramoto_hopfield_sync_step" in _ACCEL_SRC
        assert "kuramoto_hopfield_readout" in _ACCEL_SRC
        # the racing in-kernel multi-step loop is gone
        assert "for (int step = 0; step < sync_steps; step++)" not in _ACCEL_SRC
        assert "ph_a, ph_b = ph_b, ph_a" in _ACCEL_SRC
        # dispatch op key unchanged (crossover table + persisted receipts keep working)
        assert 'should_use_gpu("kuramoto_hopfield_step"' in _ACCEL_SRC

    def test_gpu_phases_match_cpu_synchronous_jacobi_when_gpu_present(self):
        acc = get_opencl_accelerator()
        if not acc.gpu_available:
            pytest.skip("no OpenCL GPU on this box")
        rng = np.random.default_rng(11)
        seq_len, d_model = 6, 8
        x = rng.standard_normal((seq_len, d_model)).astype(np.float32)
        phases0 = np.broadcast_to(
            np.linspace(0.0, 0.35, seq_len, dtype=np.float32)[:, None],
            (seq_len, d_model),
        ).copy()
        omega = np.full((seq_len, d_model), 0.05, dtype=np.float32)
        coupling = (np.maximum(0.0, np.matmul(x, x.T)) / float(seq_len)).astype(np.float32)
        hk = rng.standard_normal((5, d_model)).astype(np.float32)
        hv = rng.standard_normal((5, d_model)).astype(np.float32)
        kwargs = dict(sync_steps=6, dt=0.15, beta=8.0)
        cpu = acc.kuramoto_hopfield_attention(
            x, phases0.copy(), omega, coupling, hk, hv, force_gpu=False, **kwargs
        )
        gpu = acc.kuramoto_hopfield_attention(
            x, phases0.copy(), omega, coupling, hk, hv, force_gpu=True, **kwargs
        )
        assert gpu["backend"] == "OpenCL_GPU"
        assert np.allclose(np.asarray(cpu["phases"]), np.asarray(gpu["phases"]), atol=1e-3)
        assert np.allclose(np.asarray(cpu["output"]), np.asarray(gpu["output"]), atol=1e-3)


class TestHypernetDisclosure:
    def test_synthesize_discloses_untrained_generator(self):
        from aetheris_microbrains.core.microbrain_hypernetwork import MicroBrainHypernetwork

        hn = MicroBrainHypernetwork()
        res = hn.synthesize_weights("compile the rust module and report errors")
        assert res["generator_trained"] is False
        assert "W_ttt_fast" in res["generator_note"]
        assert res["status"] == "synthesized_success"  # existing consumers unaffected

    def test_module_docstring_carries_the_honesty_note(self):
        src = (
            Path(__file__).resolve().parents[1] / "core" / "microbrain_hypernetwork.py"
        ).read_text(encoding="utf-8")
        assert "never trained" in src
        assert "no production caller" in src  # adapt_test_time_ttt disclosure


class TestStatusAndDocHonesty:
    def test_status_no_longer_reports_fake_crossover_batch_size(self):
        acc = get_opencl_accelerator()
        st = acc.status()
        assert "crossover_batch_size" not in st
        assert "crossovers" in st

    def test_accelerator_docstring_kernel_count_updated(self):
        assert "Implements 5 SOTA" not in _ACCEL_SRC
        assert "16 OpenCL kernels" in _ACCEL_SRC


class TestParallelScanChunkDisclosure:
    def test_chunk_count_reported_for_multi_chunk_sequences(self):
        from aetheris_microbrains.core.linear_state_space_microbrain import LinearStateSpaceMicroBrain

        ssm = LinearStateSpaceMicroBrain(d_model=8, d_state=4)
        seq = np.zeros((600, 8), dtype=np.float32)
        res = ssm.process_sequence_parallel_scan(seq)
        assert res["parallel_scan_chunks"] == 3
        assert res["parallel_scan_depth_log2"] == 8  # per chunk, capped at 256


class TestKanReshapeCollapse:
    def test_fit_still_learns_after_the_refactor(self):
        from aetheris_microbrains.core.kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain

        kan = KolmogorovArnoldMicroBrain(in_features=2, out_features=1, degree=4)
        X = np.random.default_rng(0).uniform(-1, 1, size=(120, 2))
        Y = np.sin(X[:, :1])
        res = kan.fit_symbolic(X, Y)
        assert res["mse_reduced"] is True
        assert res["post_mse"] < 1e-2
