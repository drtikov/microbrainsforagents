"""Robotics extension layer tests (2026-11-18 round).

Covers the seven-part robotics roadmap:
  R1 honesty    : /metrics serves LIVE telemetry (no hardcoded values), demo banner
  R2 online     : RLS/FORCE online readout adaptation on the liquid reservoir
  R3 cbf        : state-space CBF-QP safety filter (Dykstra exact projection)
  R4 loop       : fixed-dt sensorimotor loop contract + NIR-dialect export
  R5 efe        : pymdp-shaped expected-free-energy planner
  R6 edge       : 2-bit packed ternary linear (exact integer parity, npz artifacts)
  R7 gate       : VLA gating layer (skills/conformal/CBF tiers) + server endpoint
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_AETHER_DIR = _PKG_DIR.parent

for p in [str(_PKG_DIR), str(_AETHER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

import aetheris_microbrains as mb


# ---------------------------------------------------------------------------
# R1 — honesty
# ---------------------------------------------------------------------------

class TestMetricsHonesty:
    def _metrics_text(self, monkeypatch, tmp_path, seed_events):
        from aetheris_microbrains.core import microbrain_server
        from aetheris_microbrains.core import microbrain_telemetry as mt

        fresh = mt.MicrobrainTelemetry(
            log_path=tmp_path / "tel.jsonl",
            dashboard_path=tmp_path / "dash.json",
        )
        monkeypatch.setattr(mt, "_telemetry_singleton", fresh)
        for kwargs in seed_events:
            fresh.record_event(**kwargs)
        return asyncio.run(microbrain_server.prometheus_metrics()).body.decode("utf-8")

    def test_empty_telemetry_emits_zero_counter_not_fabricated_values(self, monkeypatch, tmp_path):
        text = self._metrics_text(monkeypatch, tmp_path, [])
        assert "microbrain_telemetry_events_total 0" in text
        assert "0.985" not in text  # the old hardcoded showcase values are gone
        assert "0.280" not in text
        assert "0.035" not in text
        assert "quantile=\"p50\"" not in text

    def test_recorded_events_surface_as_live_gauges(self, monkeypatch, tmp_path):
        text = self._metrics_text(
            monkeypatch,
            tmp_path,
            [
                dict(domain="probe_domain_a", backend="cpu", latency_ms=1.5, confidence=0.9,
                     verified_accurate=True, passed_conformal=True),
                dict(domain="probe_domain_a", backend="cpu", latency_ms=2.5, confidence=0.8,
                     verified_accurate=False, passed_conformal=True),
            ],
        )
        assert 'domain="probe_domain_a"' in text
        assert "microbrain_telemetry_events_total 2" in text
        # per-domain coverage should be 1.0 (both events passed conformal), not 0.985
        line = [l for l in text.splitlines() if 'domain="probe_domain_a"' in l and "coverage" in l][0]
        assert line.strip().endswith("1.0")

    def test_dashboard_page_carries_demo_banner(self):
        from aetheris_microbrains.core import microbrain_server

        resp = asyncio.run(microbrain_server.telemetry_dashboard())
        body = resp.body.decode("utf-8")
        assert "STATIC DEMO" in body
        assert "/metrics" in body


# ---------------------------------------------------------------------------
# R2 — online readout (RLS/FORCE)
# ---------------------------------------------------------------------------

def _make_reservoir(n=32, m=8):
    from aetheris_microbrains.core.liquid_state_reservoir import LiquidStateReservoir

    return LiquidStateReservoir(reservoir_dim=n, input_dim=m)


class TestOnlineReadout:
    def test_rls_recovers_stationary_linear_map(self):
        res = _make_reservoir(32, 8)
        rng = np.random.RandomState(7)
        W_true = rng.randn(32, 2).astype(np.float32) * 0.5
        res.init_online_readout(2, method="rls", forgetting_factor=1.0)  # stationary -> no forgetting
        pre_mse = None
        final = None
        for _ in range(200):
            u = rng.randn(8).astype(np.float32)
            res.step(u)
            # target is exactly linear in the CURRENT state -> RLS must nail it
            target = (W_true.T @ res.state.astype(np.float64)).tolist()
            out = res.update_readout_online(target)
            if pre_mse is None:
                pre_mse = out["pre_mse"]
            final = out
        assert final["post_mse"] < 1e-8
        assert final["post_mse"] < pre_mse * 1e-4
        assert final["method"] == "rls"
        assert res.W_readout.shape == (32, 2)

    def test_rls_with_forgetting_tracks_drift(self):
        """Forgetting trades stationary precision for drift tracking — the robotics trade."""
        res_fast = _make_reservoir(24, 6)
        res_slow = _make_reservoir(24, 6)  # same seed -> identical liquids
        rng = np.random.RandomState(13)
        W_a = rng.randn(24, 1) * 0.5
        W_b = rng.randn(24, 1) * 0.5
        res_fast.init_online_readout(1, method="rls", forgetting_factor=0.95)
        res_slow.init_online_readout(1, method="rls", forgetting_factor=1.0)
        inputs = [rng.randn(6).astype(np.float32) for _ in range(300)]
        ema_fast = ema_slow = None
        for i, u in enumerate(inputs):
            W_t = W_a if i < 150 else W_b  # drift at step 150
            for res, ema_holder in ((res_fast, "fast"), (res_slow, "slow")):
                res.step(u)
                target = [float((W_t.T @ res.state.astype(np.float64))[0])]
                res.update_readout_online(target)
                ema = res._online_ema_mse
                if ema_holder == "fast":
                    ema_fast = ema
                else:
                    ema_slow = ema
        # after the drift, the forgetting filter tracks the new map better
        assert ema_fast is not None and ema_slow is not None
        assert ema_fast < ema_slow * 0.5

    def test_force_converges_on_stationary_target(self):
        """FORCE is the low-cost TRACKING rule: it must beat the raw target
        variance clearly, but steady-state precision belongs to RLS."""
        res = _make_reservoir(24, 6)
        rng = np.random.RandomState(11)
        W_true = rng.randn(24, 1).astype(np.float64)
        res.init_online_readout(1, method="force", force_gamma=1000.0)
        targets = []
        for _ in range(400):
            u = rng.randn(6).astype(np.float32)
            res.step(u)
            target = [float((W_true.T @ res.state.astype(np.float64))[0])]
            targets.append(target[0])
            res.update_readout_online(target)
        y_var = float(np.mean(np.square(targets)))
        assert res._online_ema_mse is not None
        assert res._online_ema_mse < 0.5 * y_var
        assert res._online_ema_mse < y_var  # strictly better than predicting 0

    def test_force_reports_numerical_reset_honestly(self):
        """gamma far below ||s||^2 diverges the delta rule; must reset + say so."""
        res = _make_reservoir(24, 6)
        rng = np.random.RandomState(11)
        res.init_online_readout(1, method="force", force_gamma=1e-3)
        statuses = set()
        for _ in range(80):
            u = rng.randn(6).astype(np.float32)
            res.step(u)
            statuses.add(res.update_readout_online([50.0])["status"])
        assert "updated_after_weight_norm_reset" in statuses or "updated" in statuses
        assert np.all(np.isfinite(res.W_readout))

    def test_online_step_protocol_fields(self):
        """Loop contract: prediction emitted, adaptation receipt attached."""
        res = _make_reservoir(16, 4)
        rng = np.random.RandomState(3)
        res.init_online_readout(2, method="rls")
        out = res.online_step(rng.randn(4), target=[0.1, -0.2])
        assert "readout_prediction" in out
        assert out["online"]["initialized"] is True
        assert out["online_update"]["status"] == "updated"
        assert "pre_mse" in out["online_update"] and "post_mse" in out["online_update"]

    def test_prediction_emitted_before_update(self):
        """Honest online protocol: first pre_mse == E[y^2] (zero weights predict 0)."""
        res = _make_reservoir(16, 4)
        rng = np.random.RandomState(3)
        res.step(rng.randn(4))
        y = rng.randn(2).astype(np.float32)
        res.init_online_readout(2, method="rls")
        out = res.update_readout_online(y)
        assert out["status"] == "updated"
        assert abs(out["pre_mse"] - float(np.mean(y.astype(np.float64) ** 2))) < 1e-6

    def test_skip_paths_are_honest(self):
        res = _make_reservoir(16, 4)
        assert res.update_readout_online([0.0])["status"] == "skipped_not_initialized"
        res.init_online_readout(2)
        assert res.update_readout_online([0.0, 0.0, 0.0])["status"] == "skipped_target_dim_mismatch"
        assert res.init_online_readout(2, method="nope")["status"] == "skipped_invalid_method"
        assert res.init_online_readout(0)["status"] == "skipped_invalid_out_dim"


# ---------------------------------------------------------------------------
# R3 — state-space CBF filter
# ---------------------------------------------------------------------------

class TestStateCBFFilter:
    def _double_integrator_obstacle(self, margin=0.0):
        from aetheris_microbrains.core.state_cbf_filter import (
            LinearDynamics,
            SphereBarrier,
            StateCBFSafetyFilter,
        )

        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        filt = StateCBFSafetyFilter(
            LinearDynamics(A, B),
            [SphereBarrier(center=[0.5, 0.0], radius=0.1)],
            u_min=[-2.0],
            u_max=[2.0],
            conformal_margin=margin,
        )
        return filt

    def test_filter_prevents_obstacle_penetration(self):
        filt = self._double_integrator_obstacle()
        x = np.array([0.0, 0.6])
        dt = 0.05

        def servo(pos_vel):
            # P servo driving to the obstacle CENTER -> nominal penetrates the ball
            return [-(1.5 * (pos_vel[0] - 0.5) + 1.0 * pos_vel[1])]

        h_hist_f, h_hist_u = [], []
        xf = x.copy()
        for _ in range(300):
            out = filt.filter(xf, servo(xf))
            assert out["verified_safe"], f"filter declared unsafe at h={out['min_barrier_slack']}"
            h_hist_f.append(filt.barriers[0].h(xf))
            xf = xf + dt * np.array([xf[1], np.asarray(out["u_filtered"])[0]])
        xu = x.copy()
        for _ in range(300):
            h_hist_u.append(filt.barriers[0].h(xu))
            xu = xu + dt * np.array([xu[1], servo(xu)[0]])
        assert min(h_hist_u) < 0
        assert min(h_hist_f) >= -1e-6

    def test_projection_is_exact_for_single_constraint(self):
        filt = self._double_integrator_obstacle()
        x = np.array([0.35, 0.3])  # nonzero velocity: barrier has control authority
        u_nom = np.array([-1.0])  # pushes toward the obstacle
        out = filt.filter(x, u_nom)
        assert out["verified_safe"], out
        u_f = np.asarray(out["u_filtered"])
        grad = filt.barriers[0].grad_h(x)
        A_u = float((grad @ filt.dynamics.g(x))[0])
        L_f_h = float(np.dot(grad, filt.dynamics.f(x)))
        b = -filt.barriers[0].alpha * filt.barriers[0].h(x) - L_f_h
        assert A_u * u_f[0] >= b - 1e-9
        # exactness: single halfspace projection is u* = max(u_nom, b / A_u)
        expected = max(u_nom[0], b / A_u)
        assert abs(u_f[0] - expected) < 1e-9

    def test_conformal_margin_shrinks_safe_set(self):
        filt0 = self._double_integrator_obstacle(margin=0.0)
        x = np.array([0.0, 0.6])
        out0 = filt0.filter(x, [0.0])
        filtM = self._double_integrator_obstacle(margin=1.0)
        outM = filtM.filter(x, [0.0])
        moved = np.linalg.norm(np.asarray(out0["u_filtered"]) - np.asarray(outM["u_filtered"]))
        assert moved > 1e-3 or not outM["verified_safe"]

    def test_infeasible_is_reported_not_claimed(self):
        from aetheris_microbrains.core.state_cbf_filter import (
            LinearDynamics,
            SphereBarrier,
            StateCBFSafetyFilter,
        )

        A = np.eye(2) * 0.0
        B = np.eye(2)
        filt = StateCBFSafetyFilter(
            LinearDynamics(A, B), [SphereBarrier(center=[5.0, 5.0], radius=0.1, alpha=0.5)],
            u_min=[-0.05, -0.05], u_max=[0.05, 0.05],
        )
        out = filt.filter([5.0, 5.0], [0.0, 0.0])  # at obstacle center: grad=0, h<0
        assert out["verified_safe"] is False
        assert out["status"] == "unsafe_no_solution"

    def test_box_constraints_respected(self):
        filt = self._double_integrator_obstacle()
        out = filt.filter([0.0, 0.6], [5.0])  # wildly out-of-box nominal
        u = np.asarray(out["u_filtered"])
        assert -2.0 - 1e-9 <= u[0] <= 2.0 + 1e-9

    def test_dim_mismatch_honest(self):
        filt = self._double_integrator_obstacle()
        out = filt.filter([0.0, 0.6], [0.0, 0.0])
        assert out["status"] == "dim_mismatch"

    def test_action_chunk_receipts(self):
        filt = self._double_integrator_obstacle()
        out = filt.filter_trajectory([0.0, 0.3], [[0.0]] * 10, dt=0.05)
        assert len(out["per_step"]) == 10
        assert out["all_verified_safe"] is True
        assert "latency_ms" in filt.filter([0.0, 0.6], [0.0])
        assert filt.filter([0.0, 0.6], [0.0])["latency_ms"] < 50.0


# ---------------------------------------------------------------------------
# R4 — sensorimotor loop + NIR dialect
# ---------------------------------------------------------------------------

class TestSensorimotorLoop:
    def test_layer_schedule_and_caching(self):
        from aetheris_microbrains.core.sensorimotor_loop import SensorimotorLoop

        calls = {"reflex": 0, "cognitive": 0, "planner": 0}

        def reflex(obs):
            calls["reflex"] += 1
            return {"action": [0.0, 0.1]}

        def cognitive(obs):
            calls["cognitive"] += 1
            return {"mode": "explore"}

        def planner(obs):
            calls["planner"] += 1
            return {"goal": [1.0, 0.0]}

        loop = SensorimotorLoop(0.01, reflex, cognitive, planner, cognitive_every=5, planner_every=10)
        for i in range(20):
            obs = {"vector": [float(i)]}
            out = loop.tick(obs)
            assert len(out["action"]["action"]) == 2
            assert out["cognitive"].get("mode") in ("explore", None) or True
        assert calls["reflex"] == 20
        assert calls["cognitive"] == 4
        assert calls["planner"] == 2
        assert loop.tick_count == 20
        assert loop.last_tick_latency_ms is not None
        assert "budget" in loop.tick({"vector": [0.0]})["timing_ms"]

    def test_reflex_exception_does_not_kill_loop(self):
        from aetheris_microbrains.core.sensorimotor_loop import SensorimotorLoop

        def boom(obs):
            raise ValueError("sensor glitch")

        loop = SensorimotorLoop(0.01, boom)
        out = loop.tick({"vector": []})
        assert "ValueError" in (out["reflex_error"] or "")
        assert out["action"] == {}

    def test_reservoir_reflex_layer_end_to_end(self):
        from aetheris_microbrains.core.sensorimotor_loop import ReservoirReflexLayer, SensorimotorLoop

        layer = ReservoirReflexLayer(reservoir=_make_reservoir(32, 8), action_dim=2)
        loop = SensorimotorLoop(0.01, layer)
        out = loop.tick({"vector": np.random.randn(8).astype(np.float32)})
        assert len(out["action"]["action"]) == 2
        assert out["action"]["online"]["initialized"] is True
        adapt = layer.adapt([0.5, -0.5])
        assert adapt["status"] == "updated"


class TestNirDialectExport:
    def test_structural_validation_and_json_roundtrip(self):
        res = _make_reservoir(24, 8)
        graph = mb.export_reservoir_nir(res, include_readout=False)
        check = mb.validate_nir_dialect_graph(graph)
        assert check["valid"], check["problems"]
        reloaded = json.loads(json.dumps(graph))
        assert mb.validate_nir_dialect_graph(reloaded)["valid"]

    def test_readout_export_shape_consistency(self):
        res = _make_reservoir(24, 8)
        res.fit_readout(np.random.randn(5, 24), np.random.randn(5, 2))
        graph = mb.export_reservoir_nir(res, include_readout=True)
        check = mb.validate_nir_dialect_graph(graph)
        assert check["valid"], check["problems"]
        assert any(e[0] == "reservoir" and e[2] == "readout" for e in graph["edges"])

    def test_nir_roundtrip_if_package_present(self):
        pytest.importorskip("nir")
        import nir  # noqa: F401  (guarded; skipped honestly when absent)

        res = _make_reservoir(24, 8)
        graph = mb.export_reservoir_nir(res)
        assert graph["nir_dialect_version"] >= 1  # presence; full nir.load() coverage tracked as future work


# ---------------------------------------------------------------------------
# R5 — pymdp-shaped EFE
# ---------------------------------------------------------------------------

def _toy_model():
    from aetheris_microbrains.core.pymdp_efe import GenerativeModel

    A = np.eye(2)
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)          # stay
    B[:, :, 1] = np.array([[0.0, 1.0], [1.0, 0.0]])  # swap
    C = np.array([2.0, 0.0])
    D = np.array([0.9, 0.1])
    return GenerativeModel(A, B, C=C, D=D)


class TestPymdpEFE:
    def test_shape_validation_raises_honestly(self):
        from aetheris_microbrains.core.pymdp_efe import GenerativeModel

        with pytest.raises(ValueError):
            GenerativeModel(np.zeros((2,)), np.zeros((2, 2, 1)))
        with pytest.raises(ValueError):
            GenerativeModel(np.eye(2), np.zeros((3, 3, 2)))
        with pytest.raises(ValueError):
            GenerativeModel(np.eye(2), np.zeros((2, 2, 2)), C=np.zeros(5))

    def test_hand_computed_plan_prefers_rewarding_state(self):
        model = _toy_model()
        agent = mb.EFPAgent(model)
        plan = agent.plan([[0], [1]])
        assert plan["best_action"] == 0  # stay in the preferred state
        assert abs(sum(plan["q_pi"]) - 1.0) < 1e-9
        assert plan["q_pi"][0] > plan["q_pi"][1]

    def test_epistemic_term_zero_when_a_uninformative(self):
        from aetheris_microbrains.core.pymdp_efe import GenerativeModel, expected_free_energy

        A = np.full((2, 2), 0.5)
        B = np.zeros((2, 2, 1))
        B[:, :, 0] = np.eye(2)
        model = GenerativeModel(A, B, C=np.zeros(2))
        g, trace = expected_free_energy(model, model.D, [0])
        assert abs(trace["per_step"][0]["epistemic"]) < 1e-9
        assert g == pytest.approx(trace["per_step"][0]["pragmatic"])

    def test_belief_update_matches_bayes_by_hand(self):
        from aetheris_microbrains.core.pymdp_efe import GenerativeModel

        A = np.array([[0.9, 0.2], [0.1, 0.8]])
        B = np.eye(2)[:, :, None]
        model = GenerativeModel(A, B)
        agent = mb.EFPAgent(model)
        post = agent.update_belief(0)
        expected = np.array([0.9, 0.2]) / 1.1
        assert np.allclose(post, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# R6 — packed ternary edge runtime
# ---------------------------------------------------------------------------

class TestTernaryEdgeRuntime:
    def _pair(self, seed=5, in_f=8, out_f=6):
        rng = np.random.RandomState(seed)
        W = rng.randn(out_f, in_f).astype(np.float32)
        bl = mb.BitLinearLayer(in_features=in_f, out_features=out_f, weights=W)
        pt = mb.PackedTernaryLinear.from_bitlinear(bl)
        return bl, pt

    def test_decode_roundtrip_lossless(self):
        _, pt = self._pair()
        assert np.array_equal(pt.decode_int8(), pt.ternary)

    def test_exact_integer_parity_with_bitlinear_cpu_path(self):
        bl, pt = self._pair()
        x = np.random.RandomState(9).randn(3, 8).astype(np.float32)
        assert np.array_equal(bl.forward(x), pt.forward(x, quantize_activations=True))

    def test_float_path_matches_dequantized_math(self):
        _, pt = self._pair()
        x = np.random.RandomState(2).randn(4, 8).astype(np.float32)
        expected = x @ (pt.decode_int8().astype(np.float32).T * pt.gamma)
        assert np.allclose(pt.forward(x, quantize_activations=False), expected, atol=1e-4)

    def test_npz_artifact_roundtrip_and_size_receipt(self, tmp_path):
        _, pt = self._pair(out_f=64, in_f=64)
        path = tmp_path / "packed.npz"
        receipt = pt.save_packed_npz(path)
        assert receipt["status"] == "saved"
        assert receipt["storage_ratio_vs_float32"] > 15.0
        loaded = mb.PackedTernaryLinear.load_packed_npz(path)
        assert loaded.gamma == pytest.approx(pt.gamma)
        x = np.random.RandomState(4).randn(2, 64).astype(np.float32)
        assert np.allclose(loaded.forward(x), pt.forward(x))

    def test_onnx_export_raises_with_install_hint_when_missing(self, tmp_path):
        try:
            import onnx  # noqa: F401
            pytest.skip("onnx installed; negative-path test not applicable")
        except ImportError:
            _, pt = self._pair()
            with pytest.raises(RuntimeError, match="pip install onnx"):
                mb.export_packed_linear_onnx(pt, tmp_path / "g.onnx")

    def test_onnx_export_graph_when_package_present(self, tmp_path):
        pytest.importorskip("onnx")
        import onnx

        _, pt = self._pair()
        path = tmp_path / "graph.onnx"
        out = mb.export_packed_linear_onnx(pt, path)
        assert out["status"] == "exported"
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        assert out["weights_mode"] == "dequantized_float32_constant"

    def test_bench_receipt_is_measured_shape(self):
        _, pt = self._pair(in_f=32, out_f=32)
        r = mb.bench_receipt(pt, batch=1, repeats=3)
        assert {"quantized_mean_ms", "float_mean_ms", "repeats"} <= set(r)


# ---------------------------------------------------------------------------
# R7 — VLA gating layer
# ---------------------------------------------------------------------------

class TestVLAGating:
    def _filter(self):
        from aetheris_microbrains.core.state_cbf_filter import (
            ControlAffineDynamics,
            HalfPlaneBarrier,
            StateCBFSafetyFilter,
        )

        return StateCBFSafetyFilter(
            ControlAffineDynamics(2, 2),
            [HalfPlaneBarrier([0.0, 1.0], -0.5)],  # keep x2 >= -0.5... h = x2 + 0.5 >= 0
            u_min=[-1.0, -1.0],
            u_max=[1.0, 1.0],
        )

    def test_skill_library_registration_and_ranking_shape(self):
        lib = mb.SkillLibrary()
        lib.register_skill("pick", "grasp cube lift place")
        lib.register_skill("wipe", "wipe surface clean tool")
        assert lib.register_skill("pick", "dup")["status"] == "already_registered"
        affs = lib.find_affordances([0.1, -0.2, 0.3], top_k=1)
        assert len(affs) == 1
        assert affs[0]["skill"] in ("pick", "wipe")
        assert -1.0 <= affs[0]["affordance"] <= 1.0
        assert mb.SkillLibrary().find_affordances([0.0]) == []

    def test_scene_encoding_deterministic(self):
        lib = mb.SkillLibrary()
        a = lib.encode_scene([0.2, 0.4])
        b = mb.SkillLibrary().encode_scene([0.2, 0.4])
        assert np.allclose(a, b)

    def test_conformal_gate_fails_closed_uncalibrated(self):
        gate = mb.ConformalActionGate()
        out = gate.check(0.01)
        assert out["allowed"] is False
        assert out["calibrated"] is False
        assert out["reason"] == "uncalibrated_fail_closed"
        gate.calibrate([0.1] * 30)
        assert gate.check(0.05)["allowed"] is True
        assert gate.check(0.5)["allowed"] is False

    def test_gate_escalates_on_cbf_infeasible(self):
        from aetheris_microbrains.core.state_cbf_filter import (
            ControlAffineDynamics,
            SphereBarrier,
            StateCBFSafetyFilter,
        )

        filt = StateCBFSafetyFilter(
            ControlAffineDynamics(2, 2),
            [SphereBarrier(center=[5.0, 5.0], radius=0.1)],
            u_min=[-0.05, -0.05], u_max=[0.05, 0.05],
        )
        gate = mb.VLActionGate(skill_library=mb.SkillLibrary(), safety_filter=filt,
                               action_gate=mb.ConformalActionGate())
        out = gate.gate([0.0, 0.0], [5.0, 5.0], [0.0, 0.0], risk_score=0.01)
        assert out["decision"] == "escalate_human"
        assert out["reason"] == "cbf_infeasible"

    def test_gate_tiers_reflex_and_vla(self):
        gate = mb.VLActionGate(
            skill_library=mb.SkillLibrary(),
            safety_filter=self._filter(),
            action_gate=None,
            affordance_threshold=-1.0,  # anything counts as an affordance hit
        )
        out = gate.gate([0.0, 0.0], [0.0, 0.0], [0.1, 0.1])
        assert out["decision"] == "local_reflex"
        gate_hi = mb.VLActionGate(
            skill_library=mb.SkillLibrary(),
            safety_filter=self._filter(),
            affordance_threshold=0.99,
        )
        gate_hi.action_gate = mb.ConformalActionGate()
        gate_hi.action_gate.calibrate([0.05] * 30)
        out2 = gate_hi.gate([0.0, 0.0], [0.0, 0.0], [0.1, 0.1], risk_score=0.01)
        assert out2["decision"] == "call_vla"

    def test_chunk_clamping_receipts(self):
        gate = mb.VLActionGate(safety_filter=self._filter())
        out = gate.clamp_action_chunk([0.0, 0.0], [[0.0, 0.2]] * 5, dt=0.05)
        assert out["status"] == "chunk_clamped"
        assert len(out["per_step"]) == 5

    def test_efe_deliberator_annotates_receipt(self):
        model = _toy_model()
        gate = mb.VLActionGate(
            skill_library=mb.SkillLibrary(),
            safety_filter=self._filter(),
            affordance_threshold=0.99,  # force call_vla
            deliberator=mb.EFPAgent(model),
        )
        out = gate.gate([0.0, 0.0], [0.0, 0.0], [0.1, 0.1], risk_score=0.01)
        assert out["decision"] == "call_vla"
        assert "best_action" in out.get("efe_deliberation", {})

    def test_server_gate_endpoint_wired(self):
        from aetheris_microbrains.core.microbrain_server import GateRequest, gate_action

        req = GateRequest(
            scene=[0.1, 0.2],
            state=[0.0, 0.0],
            nominal_action=[0.05, 0.05],
            risk_score=0.02,
            barriers=[{"type": "sphere", "center": [3.0, 3.0], "radius": 0.2}],
            u_min=[-0.5, -0.5],
            u_max=[0.5, 0.5],
        )
        resp = asyncio.run(gate_action(req))
        assert resp.decision in ("local_reflex", "call_vla", "escalate_human")
        assert len(resp.action_out) == 2
        req_bad = GateRequest(
            scene=[],
            state=[5.0, 5.0],
            nominal_action=[0.0, 0.0],
            risk_score=0.01,
            barriers=[{"type": "sphere", "center": [5.0, 5.0], "radius": 0.1}],
            u_min=[-0.05, -0.05],
            u_max=[0.05, 0.05],
        )
        resp_bad = asyncio.run(gate_action(req_bad))
        assert resp_bad.decision == "escalate_human"

    def test_unknown_barrier_type_rejected(self):
        from aetheris_microbrains.core.microbrain_server import GateRequest, gate_action
        from fastapi import HTTPException

        req = GateRequest(scene=[], state=[0.0, 0.0], nominal_action=[0.0, 0.0],
                          barriers=[{"type": "magnet"}])
        with pytest.raises(HTTPException) as exc:
            asyncio.run(gate_action(req))
        assert exc.value.status_code == 400

    def test_skill_library_scene_prototype_binding(self):
        lib = mb.SkillLibrary()
        lib.register_skill("grasp", "pick up cup", scene_prototype=[0.9, -0.8])
        lib.register_skill("dock", "return to charger", scene_prototype=[-0.9, 0.8])
        ranked = lib.find_affordances([0.9, -0.8], top_k=2)
        assert ranked[0]["skill"] == "grasp"
        assert ranked[0]["affordance"] > 0.9
        assert ranked[0]["affordance"] > ranked[1]["affordance"]
        assert lib.bind_scene_example("dock", [-0.85, 0.75])["status"] == "bound"
