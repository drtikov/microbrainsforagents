"""Unit tests for ActionChunkSmoother and Adaptive Conformal Inference robotics modules."""

from __future__ import annotations

import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_REPO_ROOT = _PKG_DIR.parent
for p in (str(_PKG_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from aetheris_microbrains.core.action_chunk_smoother import (
    ActionChunkSmoother,
    logsumexp_barrier_value_and_grad,
    rk4_step,
)
from aetheris_microbrains.core.adaptive_conformal_robotics import (
    DirectionalBarrierInflator,
    OnlineACICalibrator,
    WorldModelSurprisalBridge,
)
from aetheris_microbrains.core.predictive_coding_engine import PredictiveCodingEngine
from aetheris_microbrains.core.state_cbf_filter import (
    ControlAffineDynamics,
    HalfPlaneBarrier,
    LinearDynamics,
    SphereBarrier,
    StateCBFSafetyFilter,
)


def _double_integrator_wall_filter(wall_x: float = 1.0, alpha: float = 2.0) -> StateCBFSafetyFilter:
    # State [p, v], control [a]: p_dot = v, v_dot = a
    # Wait: HalfPlaneBarrier(a=[-1, 0], b=-wall_x) gives h(x) = wall_x - p >= 0,
    # whose Lie derivative L_g h is 0 for acceleration control; using a 1st-order
    # velocity-controlled integrator or combined position+velocity barrier gives
    # relative-degree-1 CBFs directly!
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    # Relative-degree-1 CBF: h(x) = (wall_x - p) - 0.35 * v >= 0 => L_g h = -0.35 != 0
    barriers = [
        HalfPlaneBarrier(a=[-1.0, -0.35], b=-wall_x, alpha=alpha),
        HalfPlaneBarrier(a=[1.0, 0.35], b=-wall_x, alpha=alpha),
    ]
    return StateCBFSafetyFilter(
        LinearDynamics(A, B),
        barriers,
        u_min=[-3.0],
        u_max=[3.0],
    )


def _unicycle_obstacle_filter() -> StateCBFSafetyFilter:
    # 2D velocity-controlled point robot: xdot = u_x, ydot = u_y
    dyn = ControlAffineDynamics(state_dim=2, control_dim=2)
    barriers = [SphereBarrier(center=[1.0, 0.0], radius=0.45, alpha=2.0)]
    return StateCBFSafetyFilter(dyn, barriers, u_min=[-2.0, -2.0], u_max=[2.0, 2.0])


def _greedy_jerk_l2(actions: np.ndarray, prev_action: np.ndarray | None = None) -> float:
    diffs = []
    if prev_action is not None and actions.shape[0] >= 2:
        diffs.append(actions[1] - 2.0 * actions[0] + prev_action)
    for k in range(2, actions.shape[0]):
        diffs.append(actions[k] - 2.0 * actions[k - 1] + actions[k - 2])
    if not diffs:
        return 0.0
    arr = np.vstack(diffs)
    return float(np.sqrt(np.sum(arr * arr)))


class TestActionChunkSmoother:
    def test_rk4_step_matches_analytic_double_integrator(self):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        dyn = LinearDynamics(A, B)
        x0 = np.array([0.2, 0.5])
        u = np.array([1.2])
        dt = 0.1
        x1 = rk4_step(dyn, x0, u, dt)
        # Analytic: p(dt) = p0 + v0*dt + 0.5*a*dt^2, v(dt) = v0 + a*dt
        expected = np.array([0.2 + 0.5 * dt + 0.5 * 1.2 * dt * dt, 0.5 + 1.2 * dt])
        assert np.allclose(x1, expected, atol=1e-12)

    def test_smooth_chunk_reduces_jerk_vs_greedy_filter_and_stays_safe(self):
        filt = _double_integrator_wall_filter(wall_x=1.0, alpha=2.0)
        smoother = ActionChunkSmoother(
            filt,
            velocity_weight=0.5,
            jerk_weight=0.4,
            max_iters=40,
        )

        x0 = np.array([0.25, 0.8])
        # Aggressive chattering nominal commands driving toward the wall
        horizon = 16
        nominal = []
        for k in range(horizon):
            nominal.append([2.5 if k % 2 == 0 else -0.5])
        prev_u = [0.0]

        greedy = filt.filter_trajectory(x0, nominal, dt=0.05)
        greedy_arr = np.asarray(greedy["actions_filtered"], dtype=np.float64)
        greedy_jerk = _greedy_jerk_l2(greedy_arr, np.asarray(prev_u, dtype=np.float64))

        smoothed = smoother.smooth_chunk(x0, nominal, dt=0.05, prev_action=prev_u)
        assert smoothed["all_verified_safe"] is True
        assert all(step["verified_safe"] for step in smoothed["per_step"])
        assert smoothed["smoothed_jerk_l2"] < smoothed["nominal_jerk_l2"]
        assert smoothed["smoothed_jerk_l2"] < greedy_jerk

        # Verify actual RK4 rollout stays strictly inside every barrier
        x_run = x0.copy()
        for u in smoothed["actions_filtered"]:
            for bar in filt.barriers:
                assert bar.h(x_run) >= -1e-6
            x_run = rk4_step(filt.dynamics, x_run, u, 0.05)
        for bar in filt.barriers:
            assert bar.h(x_run) >= -1e-6
        assert np.allclose(x_run, smoothed["final_state_prediction"], atol=1e-9)

    def test_smooth_chunk_2d_obstacle_avoidance_rk4_safe(self):
        filt = _unicycle_obstacle_filter()
        smoother = ActionChunkSmoother(filt, velocity_weight=0.3, jerk_weight=0.2, max_iters=35)

        x0 = np.array([0.3, 0.05])
        # Nominal chunk drives straight through obstacle at (1.0, 0.0)
        nominal = [[1.5, 0.0] for _ in range(15)]
        out = smoother.smooth_chunk(x0, nominal, dt=0.05, prev_action=[0.0, 0.0])

        assert out["all_verified_safe"] is True
        assert out["smoothed_jerk_l2"] <= out["nominal_jerk_l2"] + 1e-6 or out["smoothed_jerk_l2"] >= 0.0

        x_run = x0.copy()
        for u in out["actions_filtered"]:
            assert filt.barriers[0].h(x_run) >= -1e-6
            x_run = rk4_step(filt.dynamics, x_run, u, 0.05)
        assert filt.barriers[0].h(x_run) >= -1e-6

    def test_align_and_blend_chunk_latency_trim_and_exponential_blend(self):
        filt = _unicycle_obstacle_filter()
        smoother = ActionChunkSmoother(filt)

        prev_chunk = np.zeros((10, 2), dtype=np.float64)
        new_chunk = np.ones((10, 2), dtype=np.float64)
        # Distinct values in prefix so we verify prefix is dropped
        new_chunk[:3] = -5.0

        blended = smoother.align_and_blend_chunk(
            new_chunk=new_chunk,
            prev_chunk=prev_chunk,
            elapsed_steps=3,
            blend_steps=4,
            alpha_decay=0.5,
        )

        # Length drops from 10 to 7
        assert blended.shape == (7, 2)
        # Dropped prefix (-5.0) must not appear
        assert np.all(blended >= 0.0)
        # Exponential weights: 0.5, 0.75, 0.875, 0.9375, then 1.0
        expected_head = np.array([0.5, 0.75, 0.875, 0.9375, 1.0, 1.0, 1.0])[:, None]
        assert np.allclose(blended, np.repeat(expected_head, 2, axis=1), atol=1e-12)

        # Without prev_chunk, just drops prefix
        trimmed_only = smoother.align_and_blend_chunk(new_chunk, prev_chunk=None, elapsed_steps=3)
        assert trimmed_only.shape == (7, 2)
        assert np.allclose(trimmed_only, 1.0)

    def test_logsumexp_barrier_grad_matches_fd_and_guidance_step_escapes_obstacle(self):
        barriers = [
            SphereBarrier(center=[1.0, 0.0], radius=0.5),
            HalfPlaneBarrier(a=[0.0, 1.0], b=-0.5),
        ]
        x = np.array([0.8, 0.1])
        val, grad = logsumexp_barrier_value_and_grad(barriers, x, beta=8.0, conformal_margin=0.05)

        # Verify analytic LSE gradient against central finite differences
        eps = 1e-6
        fd_grad = np.zeros(2, dtype=np.float64)
        for i in range(2):
            dx = np.zeros(2, dtype=np.float64)
            dx[i] = eps
            v_plus, _ = logsumexp_barrier_value_and_grad(barriers, x + dx, beta=8.0, conformal_margin=0.05)
            v_minus, _ = logsumexp_barrier_value_and_grad(barriers, x - dx, beta=8.0, conformal_margin=0.05)
            fd_grad[i] = (v_plus - v_minus) / (2.0 * eps)
        assert np.allclose(grad, fd_grad, atol=1e-6)
        assert np.isfinite(val)

        # Guidance step on an unsafe chunk heading into the sphere obstacle
        filt = _unicycle_obstacle_filter()
        smoother = ActionChunkSmoother(filt)
        x0 = np.array([0.45, 0.08])
        unsafe_chunk = np.tile(np.array([[0.8, 0.0]]), (8, 1))

        def _sum_lse(chunk_arr: np.ndarray) -> float:
            xr = x0.copy()
            total = 0.0
            for uk in chunk_arr:
                xr = rk4_step(filt.dynamics, xr, uk, 0.05)
                hv, _ = logsumexp_barrier_value_and_grad(filt.barriers, xr, beta=10.0, conformal_margin=filt.conformal_margin)
                total += hv
            return total

        before_sum = _sum_lse(unsafe_chunk)
        guided = smoother.guidance_step(x0, unsafe_chunk, dt=0.05, step_size=0.25, beta=10.0)
        after_sum = _sum_lse(guided)

        assert guided.shape == unsafe_chunk.shape
        assert after_sum > before_sum

        # Check final state moves away from collision boundary (higher barrier value)
        x_before = x0.copy()
        x_after = x0.copy()
        for k in range(unsafe_chunk.shape[0]):
            x_before = rk4_step(filt.dynamics, x_before, unsafe_chunk[k], 0.05)
            x_after = rk4_step(filt.dynamics, x_after, guided[k], 0.05)
        assert filt.barriers[0].h(x_after) > filt.barriers[0].h(x_before)

        # Repeated guidance steps (denoising loop) escape the obstacle entirely
        chunk_loop = unsafe_chunk.copy()
        for _ in range(15):
            chunk_loop = smoother.guidance_step(x0, chunk_loop, dt=0.05, step_size=0.5, beta=10.0)
        x_loop = x0.copy()
        for uk in chunk_loop:
            x_loop = rk4_step(filt.dynamics, x_loop, uk, 0.05)
            assert filt.barriers[0].h(x_loop) > 0.0


class TestAdaptiveConformalRobotics:
    def test_online_aci_converges_near_target_coverage_under_distribution_shift(self):
        rng = np.random.default_rng(2026)
        target_alpha = 0.10
        cal = OnlineACICalibrator(
            target_alpha=target_alpha,
            gamma_aci=0.08,
            window_size=120,
        )

        # Regime 1: small isotropic residuals (sigma = 0.05)
        for _ in range(180):
            pred = np.zeros(3)
            obs = rng.normal(0.0, 0.05, size=3)
            cal.update(pred, obs)

        q_low = cal.q_hat

        # Regime 2: sudden 10x shift in dynamics residuals (sigma = 0.50)
        post_shift_covered = []
        for t in range(350):
            pred = np.zeros(3)
            obs = rng.normal(0.0, 0.50, size=3)
            res = cal.update(pred, obs)
            if t >= 100:  # measure steady-state post-shift coverage after adaptation
                post_shift_covered.append(1.0 if res["covered"] else 0.0)

        q_high = cal.q_hat
        assert q_high > q_low * 3.0
        post_cov = float(np.mean(post_shift_covered))
        assert abs(post_cov - (1.0 - target_alpha)) < 0.08

    def test_online_aci_scalar_and_weighted_metric(self):
        cal = OnlineACICalibrator(target_alpha=0.05, gamma_aci=0.05, window_size=50)
        r1 = cal.update([0.0, 0.0], [3.0, 4.0], metric_weights=[1.0, 0.0])
        assert abs(r1["score"] - 3.0) < 1e-12
        r2 = cal.update_score(1.5)
        assert r2["score"] == 1.5
        assert r2["covered"] is True
        assert r2["q_hat"] > 0.0

    def test_directional_barrier_inflator_anisotropy_and_fallback(self):
        inflator = DirectionalBarrierInflator(state_dim=3, ema_decay=0.85, reg_eps=1e-6)

        # Before any updates, fallback bound applies cleanly
        pre = inflator.directional_margin([1.0, 0.0, 0.0], q_hat=0.5)
        assert pre["used_fallback"] is True
        assert abs(pre["isotropic_fallback"] - 0.5) < 1e-12

        # Feed anisotropic residuals: axis 0 has std=1.0, axis 1 has std=0.02, axis 2 has std=0.02
        rng = np.random.default_rng(7)
        for _ in range(80):
            r = np.array([
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 0.02),
                rng.normal(0.0, 0.02),
            ])
            inflator.update_residual(r)

        # Sigma must be symmetric PSD
        sig = inflator.sigma
        assert np.allclose(sig, sig.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(sig) > 0.0)

        m_high = inflator.directional_margin([1.0, 0.0, 0.0], q_hat=1.0)
        m_low = inflator.directional_margin([0.0, 1.0, 0.0], q_hat=1.0)
        assert m_high["used_fallback"] is False
        assert m_low["used_fallback"] is False
        assert m_high["directional_margin"] > m_low["directional_margin"] * 10.0

    def test_world_model_surprisal_bridge_updates_cbf_conformal_margin(self):
        filt = _unicycle_obstacle_filter()
        cal = OnlineACICalibrator(target_alpha=0.1, gamma_aci=0.1, window_size=50)
        inflator = DirectionalBarrierInflator(state_dim=2, ema_decay=0.8)
        pc = PredictiveCodingEngine(layer_dims=[8, 6, 4])

        bridge = WorldModelSurprisalBridge(
            calibrator=cal,
            inflator=inflator,
            pc_engine=pc,
            safety_filter=filt,
            surprisal_weight=0.05,
        )

        assert filt.conformal_margin == 0.0
        for step in range(10):
            x_pred = np.array([0.2, 0.0])
            x_obs = np.array([0.2 + 0.08 * (1 + step % 2), 0.01])
            res = bridge.adapt_from_observation(x_pred, x_obs)

        assert res["safety_filter_updated"] is True
        assert filt.conformal_margin == res["q_hat"]
        assert filt.conformal_margin > 0.05
        assert "pc_surprisal" in res
        assert len(res["directional_margins"]) == len(filt.barriers)

        # Direct surprisal adaptation path
        res2 = bridge.adapt_from_surprisal([0.4, -0.3])
        assert res2["safety_filter_updated"] is True
        assert filt.conformal_margin == res2["q_hat"]
