"""Real-time sensorimotor loop, 2-bit packed LUT matvec, ONNX int8 export, and certified NIR/DVS tests."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_AETHER_DIR = _PKG_DIR.parent

for p in [str(_PKG_DIR), str(_AETHER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

from aetheris_microbrains.core.bitnet_ternary_engine import BitLinearLayer
from aetheris_microbrains.core.liquid_state_reservoir import LiquidStateReservoir
from aetheris_microbrains.core.nir_export import (
    DVSEventSpikeEncoder,
    export_reservoir_nir_certified,
)
from aetheris_microbrains.core.sensorimotor_loop import AsyncMultiRateSensorimotorLoop
from aetheris_microbrains.core.ternary_edge_runtime import (
    PackedTernaryLinear,
    export_packed_linear_onnx_int8,
)


# ---------------------------------------------------------------------------
# AsyncMultiRateSensorimotorLoop + Dead-man TTL watchdog
# ---------------------------------------------------------------------------

class TestAsyncMultiRateSensorimotorLoop:
    def test_slow_cognitive_does_not_block_reflex_ticks(self):
        cog_done = threading.Event()
        cog_calls = {"count": 0}

        def reflex_fn(obs):
            return {"action": [1.0, 0.0]}

        def slow_cognitive_fn(obs):
            cog_calls["count"] += 1
            time.sleep(0.04)  # 40 ms slow path
            cog_done.set()
            return {"mode": "vla_ready", "seq": cog_calls["count"]}

        def planner_fn(obs):
            return {"goal": [0.5, 0.5]}

        with AsyncMultiRateSensorimotorLoop(
            dt=0.01,
            reflex_fn=reflex_fn,
            cognitive_fn=slow_cognitive_fn,
            planner_fn=planner_fn,
            cognitive_period_ms=50.0,
            planner_period_ms=100.0,
            cognitive_ttl_ms=500.0,
            planner_ttl_ms=1000.0,
        ) as loop:
            # First tick triggers async workers; reflex itself must remain fast (< 2 ms).
            # Measure several reflex ticks while cognitive_fn is sleeping 40 ms.
            reflex_latencies = []
            for i in range(5):
                out = loop.tick({"vector": [float(i)]})
                reflex_latencies.append(out["reflex_ms"])
                assert "cognitive_age_ms" in out
                assert "watchdog_active" in out
                assert out["action"]["action"] == [1.0, 0.0]

            # All reflex calls stayed well below the 40 ms cognitive sleep
            assert max(reflex_latencies) < 2.0, f"reflex blocked: {reflex_latencies}"
            assert cog_done.wait(timeout=1.0), "async cognitive worker did not finish"

            # Give worker thread a moment to publish the snapshot tuple
            deadline = time.perf_counter() + 0.5
            latest = loop.tick({"vector": [99.0]})
            while not latest["cognitive"] and time.perf_counter() < deadline:
                time.sleep(0.005)
                latest = loop.tick({"vector": [99.0]})

            assert latest["cognitive"].get("mode") == "vla_ready"
            assert latest["watchdog_active"] is False

    def test_watchdog_ttl_expiration_ramps_linearly_to_backup(self):
        release_cog = threading.Event()

        def reflex_fn(obs):
            return {"action": [10.0, 0.0]}

        def backup_fn(obs):
            return {"action": [0.0, 10.0]}

        def stalled_cognitive_fn(obs):
            release_cog.wait(timeout=1.0)
            return {"mode": "recovered"}

        loop = AsyncMultiRateSensorimotorLoop(
            dt=0.005,
            reflex_fn=reflex_fn,
            cognitive_fn=stalled_cognitive_fn,
            backup_fn=backup_fn,
            cognitive_ttl_ms=25.0,
            watchdog_ramp_ms=40.0,
            auto_start=True,
        )
        try:
            out_fresh = loop.tick({"vector": [1.0]})
            assert out_fresh["watchdog_expired"] is False
            assert out_fresh["decay"] == 1.0
            assert np.allclose(out_fresh["action"]["action"], [10.0, 0.0])

            # Wait past ttl_ms into the linear ramp window
            time.sleep(0.045)
            out_mid = loop.tick({"vector": [2.0]})
            assert out_mid["watchdog_expired"] is True
            assert out_mid["watchdog_active"] is True
            assert out_mid["cognitive_age_ms"] > 25.0
            assert 0.0 < out_mid["decay"] < 1.0
            a_mid = np.asarray(out_mid["action"]["action"], dtype=np.float64)
            expected_mid = out_mid["decay"] * np.array([10.0, 0.0]) + (1.0 - out_mid["decay"]) * np.array([0.0, 10.0])
            assert np.allclose(a_mid, expected_mid, atol=1e-5)

            # Wait past ttl_ms + ramp_ms -> full handoff (decay == 0.0)
            time.sleep(0.045)
            out_full = loop.tick({"vector": [3.0]})
            assert out_full["watchdog_expired"] is True
            assert out_full["decay"] == 0.0
            assert np.allclose(out_full["action"]["action"], [0.0, 10.0])
        finally:
            release_cog.set()
            loop.stop()


# ---------------------------------------------------------------------------
# PackedTernaryLinear LUT matvec & ONNX int8 export
# ---------------------------------------------------------------------------

class TestPackedLutAndOnnxInt8:
    @pytest.mark.parametrize("in_f,out_f,batch", [(16, 8, 4), (13, 7, 3), (4, 1, 1), (1, 5, 2)])
    def test_matvec_packed_lut_matches_matvec_quantized(self, in_f, out_f, batch):
        rng = np.random.RandomState(100 + in_f * 7 + out_f)
        W = rng.randn(out_f, in_f).astype(np.float32)
        bl = BitLinearLayer(in_features=in_f, out_features=out_f, weights=W)
        pt = PackedTernaryLinear.from_bitlinear(bl)

        x = rng.randn(batch, in_f).astype(np.float32)
        ref = pt.matvec_quantized(x)
        lut = pt.matvec_packed_lut(x)
        assert lut.shape == ref.shape
        assert lut.dtype == np.float32
        assert np.allclose(lut, ref, atol=1e-6)

        # Zero-input edge case
        z = np.zeros((batch, in_f), dtype=np.float32)
        assert np.allclose(pt.matvec_packed_lut(z), pt.matvec_quantized(z), atol=1e-6)

    def test_onnx_int8_export_raises_install_hint_when_missing(self, tmp_path):
        try:
            import onnx  # noqa: F401
            pytest.skip("onnx installed; negative-path test not applicable")
        except ImportError:
            pt = PackedTernaryLinear(in_features=8, out_features=4)
            with pytest.raises(RuntimeError, match="pip install onnx"):
                export_packed_linear_onnx_int8(pt, tmp_path / "int8.onnx")

    def test_onnx_int8_export_and_ort_validation_when_present(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        rng = np.random.RandomState(17)
        W = rng.randn(6, 8).astype(np.float32)
        bias = rng.randn(6).astype(np.float32)
        pt = PackedTernaryLinear(in_features=8, out_features=6, weights=W, bias=bias)

        onnx_path = tmp_path / "packed_int8.onnx"
        receipt = export_packed_linear_onnx_int8(pt, onnx_path, opset=17)
        assert receipt["status"] == "exported"
        assert receipt["weights_mode"] == "int8_matmul_integer"
        assert onnx_path.is_file()

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        op_types = [n.op_type for n in model.graph.node]
        assert "DynamicQuantizeLinear" in op_types
        assert "MatMulInteger" in op_types
        assert "Cast" in op_types
        assert "Mul" in op_types
        assert "Add" in op_types

        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = rng.randn(3, 8).astype(np.float32)
        y_ort = sess.run(None, {"X": x})[0]
        y_float = pt.forward(x, quantize_activations=False)
        assert y_ort.shape == (3, 6)
        assert np.allclose(y_ort, y_float, atol=0.15)


# ---------------------------------------------------------------------------
# Certified NIR export & DVS event spike encoder
# ---------------------------------------------------------------------------

class TestCertifiedNirAndDvsEncoder:
    def test_certified_nir_tau_phys_and_schemas(self, tmp_path):
        res = LiquidStateReservoir(reservoir_dim=16, input_dim=6, leak_rate=0.25)
        res.fit_readout(np.random.RandomState(3).randn(8, 16), np.random.RandomState(3).randn(8, 2))

        for ntype, expected_nir in [("li", "nir.LI"), ("lif", "nir.LIF"), ("cubalif", "nir.CubaLIF")]:
            out = export_reservoir_nir_certified(res, dt=0.002, include_readout=True, neuron_type=ntype)
            assert out["tau_phys"] == pytest.approx(0.002 / 0.25)
            assert out["nodes"]["reservoir"]["nir_type"] == expected_nir
            assert out["nodes"]["readout"]["weight_shape"] == [2, 16]
            assert out["nodes"]["output"]["output_type"] == [2]

        try:
            import nir  # noqa: F401
            hdf5_file = tmp_path / "res.nir.h5"
            out_h5 = export_reservoir_nir_certified(res, dt=0.001, hdf5_path=hdf5_file, neuron_type="li")
            assert out_h5["hdf5_written"] is True
            assert hdf5_file.is_file()
        except ImportError:
            with pytest.raises(RuntimeError, match="pip install nir"):
                export_reservoir_nir_certified(res, dt=0.001, hdf5_path=tmp_path / "res.nir.h5")

    def test_dvs_event_spike_encoder_modes_and_reservoir_compat(self):
        events = [
            (2, 3, 1000.0, 1),
            (2, 3, 2000.0, 1),
            (5, 1, 5000.0, -1),
            (7, 7, 9000.0, 0),
            (99, 99, 9500.0, 1),  # out-of-bounds ignored safely
        ]

        enc_st = DVSEventSpikeEncoder(width=8, height=8, polarity_mode="two_channel")
        tensor = enc_st.encode_events(events, n_bins=4, t_start_us=0.0, t_end_us=10000.0, mode="spatiotemporal")
        assert tensor.shape == (4, 2, 8, 8)
        assert tensor[0, 1, 3, 2] > 0.0  # ON event at (2, 3) in bin 0
        assert tensor[2, 0, 1, 5] > 0.0  # OFF event at (5, 1) in bin 2

        enc_res = DVSEventSpikeEncoder(width=8, height=8, out_dim=12, polarity_mode="signed")
        seq = enc_res.encode_events(events, n_bins=3, mode="rate")
        assert seq.shape == (3, 12)
        assert seq.dtype == np.float32

        res = LiquidStateReservoir(reservoir_dim=20, input_dim=12)
        for frame in seq:
            step_out = res.step(frame, event_tag="dvs_frame")
            assert step_out["status"] == "stepped"

        empty_bin = enc_res.encode_events([], n_bins=1, mode="binary")
        assert empty_bin.shape == (12,)
        assert np.all(empty_bin == 0.0)
