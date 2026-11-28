"""Tests for time-varying / dynamic-obstacle CBFs, HOCBFs, KalmanObstacleTracker,
and compositional smooth Signal Temporal Logic (STL) barrier trees.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_AETHER_DIR = _PKG_DIR.parent

for _p in (str(_PKG_DIR), str(_AETHER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from aetheris_microbrains.core.action_chunk_smoother import rk4_step
from aetheris_microbrains.core.semantic_cbf_compiler import (
    KalmanObstacleTracker,
    MovingObstacleBarrier,
    STLAndBarrier,
    STLOrBarrier,
    STLUntilBarrier,
    SemanticCBFCompiler,
    SubspaceSphereBarrier,
    TimeVaryingBarrier,
)
from aetheris_microbrains.core.state_cbf_filter import (
    ControlAffineDynamics,
    EulerLagrangeDynamics,
    HighOrderBarrier,
    LinearDynamics,
    SphereBarrier,
    StateCBFSafetyFilter,
)


class TestDynamicMovingObstacleCBF:
    def test_first_order_stationary_robot_evades_fast_approaching_human(self):
        # 3D velocity-controlled robot at [0, 0.08, 0] with u_nominal = [0, 0, 0];
        # human obstacle starts at [1.0, 0.0, 0.0] moving at [-2.5, 0.0, 0.0] m/s, radius = 0.45 m.
        dyn = LinearDynamics(np.zeros((3, 3)), np.eye(3))
        x0 = np.array([0.0, 0.08, 0.0], dtype=np.float64)
        u_zero = [0.0, 0.0, 0.0]
        p_obs0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_obs = np.array([-2.5, 0.0, 0.0], dtype=np.float64)
        radius = 0.45

        # 1) Static SphereBarrier ignores obstacle velocity: L_f h = 0 and h > 0 => zero evasion at t=0
        static_bar = SphereBarrier(center=p_obs0, radius=radius, alpha=1.5)
        static_filt = StateCBFSafetyFilter(dyn, [static_bar], u_min=[-4.0] * 3, u_max=[4.0] * 3)
        out_static_t0 = static_filt.filter(x0, u_zero)
        assert out_static_t0["verified_safe"] is True
        assert np.allclose(out_static_t0["u_filtered"], 0.0, atol=1e-12)

        # Rolling out the static barrier (re-centering without dh_dt) collides when human sweeps past
        x_stat = x0.copy()
        dt = 0.02
        min_h_static = float("inf")
        for step in range(45):
            p_t = p_obs0 + v_obs * (step * dt)
            step_bar = SphereBarrier(center=p_t, radius=radius, alpha=0.8)
            step_filt = StateCBFSafetyFilter(dyn, [step_bar], u_min=[-2.2] * 3, u_max=[2.2] * 3)
            res = step_filt.filter(x_stat, u_zero)
            x_stat = x_stat + dt * np.asarray(res["u_filtered"], dtype=np.float64)
            p_next = p_obs0 + v_obs * ((step + 1) * dt)
            min_h_static = min(min_h_static, float(np.sum((x_stat - p_next) ** 2) - radius**2))
        assert min_h_static < -0.02

        # 2) Dynamic MovingObstacleBarrier includes dh_dt < 0 => immediate evasion at t=0 & zero collisions
        dyn_bar = MovingObstacleBarrier(center=p_obs0, radius=radius, velocity=v_obs, alpha=0.8)
        assert dyn_bar.dh_dt(x0) < -4.0  # -2 * (0 - 1.0) * (-2.5) = -5.0
        dyn_filt = StateCBFSafetyFilter(dyn, [dyn_bar], u_min=[-2.2] * 3, u_max=[2.2] * 3)
        out_dyn_t0 = dyn_filt.filter(x0, u_zero)
        assert out_dyn_t0["verified_safe"] is True
        assert np.linalg.norm(out_dyn_t0["u_filtered"]) > 1.5  # proactive immediate evasion!

        x_dyn = x0.copy()
        dyn_bar.set_time(0.0)
        min_h_dyn = float("inf")
        for _ in range(45):
            res = dyn_filt.filter(x_dyn, u_zero)
            assert res["verified_safe"] is True
            x_dyn = x_dyn + dt * np.asarray(res["u_filtered"], dtype=np.float64)
            dyn_bar.advance_time(dt)
            min_h_dyn = min(min_h_dyn, dyn_bar.h(x_dyn))
        assert min_h_dyn >= -1e-4

    def test_hocbf_double_integrator_exact_relative_velocity_and_proactive_evasion(self):
        # Verify exact algebraic identity on 1-DOF double integrator:
        # h(q, t) = (q - v_o * t)^2 - r^2  =>  L_f^2 h + 2 * grad_dh_dt . f + d2h_dt2 == 2 * (qdot - v_o)^2
        dyn_1d = EulerLagrangeDynamics(n_dof=1)
        v_o = -2.3
        mov_1d = MovingObstacleBarrier(center=[1.1], radius=0.35, velocity=[v_o], alpha=2.0)
        hocbf_1d = HighOrderBarrier(mov_1d, alpha1=2.0, alpha2=2.5)
        filt_1d = StateCBFSafetyFilter(dyn_1d, [hocbf_1d], u_min=[-20.0], u_max=[20.0])

        mov_1d.set_time(0.2)
        q_val, dq_val = 0.15, 0.4
        x_probe = np.array([q_val, dq_val], dtype=np.float64)
        A, b, _ = filt_1d._halfspaces(x_probe)
        p_t = 1.1 + v_o * 0.2
        h_val = (q_val - p_t) ** 2 - 0.35**2
        h_dot_exact = 2.0 * (q_val - p_t) * (dq_val - v_o)
        h_ddot_drift_exact = 2.0 * (dq_val - v_o) ** 2
        expected_A = 2.0 * (q_val - p_t)
        expected_b = -(h_ddot_drift_exact + (2.0 + 2.5) * h_dot_exact + (2.0 * 2.5) * h_val)
        assert A[0, 0] == pytest.approx(expected_A, abs=1e-9)
        assert b[0] == pytest.approx(expected_b, abs=1e-9)

        # 2-DOF torque-controlled stationary robot at q=[0, 0.06], qdot=[0, 0], nominal u=[0, 0].
        # Initial state lies strictly inside C1 intersect C2 (h(x0, 0) > 0 and psi_1(x0, 0) > 0).
        dyn_2d = EulerLagrangeDynamics(n_dof=2)
        x0 = np.array([0.0, 0.06, 0.0, 0.0], dtype=np.float64)
        p0 = np.array([1.5, 0.0], dtype=np.float64)
        v0 = np.array([-2.6, 0.0], dtype=np.float64)
        a0 = np.array([-0.4, 0.0], dtype=np.float64)
        r_safe = 0.42

        # Static HOCBF has L_f h = 0, L_f^2 h = 0 at qdot = 0 -> commands ZERO evasion torque at t=0
        static_hocbf = HighOrderBarrier(SphereBarrier(center=p0, radius=r_safe), alpha1=4.5, alpha2=4.5)
        static_filt = StateCBFSafetyFilter(dyn_2d, [static_hocbf], u_min=[-12.0, -12.0], u_max=[12.0, 12.0])
        res_static_0 = static_filt.filter(x0, [0.0, 0.0])
        assert np.allclose(res_static_0["u_filtered"], [0.0, 0.0], atol=1e-12)

        # Static HOCBF rolled out against moving obstacle never moves until h < 0 and collides deeply
        x_s = x0.copy()
        dt = 0.015
        min_h_s = float("inf")
        for k in range(80):
            t_k = k * dt
            p_k = p0 + v0 * t_k + 0.5 * a0 * (t_k * t_k)
            sh = HighOrderBarrier(SphereBarrier(center=p_k, radius=r_safe), alpha1=4.5, alpha2=4.5)
            sf = StateCBFSafetyFilter(dyn_2d, [sh], u_min=[-12.0, -12.0], u_max=[12.0, 12.0])
            out_s = sf.filter(x_s, [0.0, 0.0])
            u_s = np.asarray(out_s["u_filtered"], dtype=np.float64)
            x_s = rk4_step(dyn_2d, x_s, u_s, dt)
            t_next = (k + 1) * dt
            p_next = p0 + v0 * t_next + 0.5 * a0 * (t_next * t_next)
            min_h_s = min(min_h_s, float(np.sum((x_s[:2] - p_next) ** 2) - r_safe**2))
        assert min_h_s < -0.05

        # Dynamic HOCBF with dh_dt, d2h_dt2, grad_dh_dt accelerates immediately at t=0 and never collides
        mov_2d = MovingObstacleBarrier(
            center=p0,
            radius=r_safe,
            velocity=v0,
            acceleration=a0,
            alpha=4.5,
        )
        dyn_hocbf = HighOrderBarrier(mov_2d, alpha1=4.5, alpha2=4.5)
        psi1_0 = float(dyn_hocbf.grad_h(x0) @ dyn_2d.f(x0) + dyn_hocbf.dh_dt(x0) + 4.5 * dyn_hocbf.h(x0))
        assert dyn_hocbf.h(x0) > 0.0 and psi1_0 > 0.0
        dyn_filt = StateCBFSafetyFilter(dyn_2d, [dyn_hocbf], u_min=[-12.0, -12.0], u_max=[12.0, 12.0])

        res_dyn_0 = dyn_filt.filter(x0, [0.0, 0.0])
        assert res_dyn_0["verified_safe"] is True
        assert np.linalg.norm(res_dyn_0["u_filtered"]) > 4.0  # immediate evasion acceleration from rest!

        x_d = x0.copy()
        mov_2d.set_time(0.0)
        min_h_d = float("inf")
        for _ in range(80):
            out_d = dyn_filt.filter(x_d, [0.0, 0.0])
            assert out_d["verified_safe"] is True
            u_d = np.asarray(out_d["u_filtered"], dtype=np.float64)
            x_d = rk4_step(dyn_2d, x_d, u_d, dt)
            mov_2d.advance_time(dt)
            min_h_d = min(min_h_d, dyn_hocbf.h(x_d))
        assert min_h_d > 0.0

    def test_grad_dh_dt_g_term_and_filter_trajectory_restores_barrier_time(self):
        # Verify mixed control term grad_dh_dt @ G(x) when G(x) has non-zero position row
        dyn = ControlAffineDynamics(
            state_dim=2,
            control_dim=1,
            f=lambda x: np.array([x[1], 0.0]),
            g=lambda _x: np.array([[0.25], [1.0]]),
        )
        mov = MovingObstacleBarrier(center=[1.0], radius=0.3, velocity=[-2.0], state_slice=slice(0, 1), state_dim=2)
        hocbf = HighOrderBarrier(mov, alpha1=2.0, alpha2=2.0, state_slice=slice(0, 2))
        filt = StateCBFSafetyFilter(dyn, [hocbf], u_min=[-10.0], u_max=[10.0])

        x = np.array([0.0, 0.0], dtype=np.float64)
        A, _, _ = filt._halfspaces(x)
        # L_g L_f h: H_h=[[2,0],[0,0]], f=[0,0] => 0; gh=[-2,0], J_f=[[0,1],[0,0]], G=[[0.25],[1]] => gh @ J_f @ G = -2.0
        # grad_dh_dt = [+4.0, 0.0] => grad_dh_dt @ G = 4.0 * 0.25 = +1.0 => total A = -1.0
        assert A[0, 0] == pytest.approx(-1.0, abs=1e-8)

        # Verify filter_trajectory advances barrier time across horizon and restores current_time afterward
        mov3 = MovingObstacleBarrier(center=[0.9, 0.0, 0.0], radius=0.4, velocity=[-2.0, 0.0, 0.0], alpha=1.5)
        mov3.set_time(0.12)
        filt3 = StateCBFSafetyFilter(LinearDynamics(np.zeros((3, 3)), np.eye(3)), [mov3], u_min=[-4.0] * 3, u_max=[4.0] * 3)
        chunk_out = filt3.filter_trajectory([0.0, 0.05, 0.0], [[0.0, 0.0, 0.0]] * 8, dt=0.05)
        assert chunk_out["all_verified_safe"] is True
        assert mov3.current_time == pytest.approx(0.12, abs=1e-12)

    def test_semantic_compiler_compiles_velocity_entity_to_moving_obstacle_barrier(self):
        comp = SemanticCBFCompiler()
        scene = {
            "entities": {
                "static_table": {"center": [1.0, 1.0, 0.0], "radius": 0.3},
                "walking_human": {
                    "center": [1.2, 0.0, 0.0],
                    "radius": 0.3,
                    "velocity": [-1.1, 0.2, 0.0],
                    "acceleration": [0.0, -0.05, 0.0],
                },
            }
        }
        bars = comp.compile_rules(
            [
                'avoid(entity="static_table", min_dist_m=0.2)',
                'avoid(entity="walking_human", min_dist_m=0.4, alpha=1.2)',
            ],
            scene_graph=scene,
            pos_slice=slice(0, 3),
            state_dim=6,
        )
        assert isinstance(bars[0], SubspaceSphereBarrier)
        assert isinstance(bars[1], MovingObstacleBarrier)
        assert bars[1].r == pytest.approx(0.7)
        np.testing.assert_allclose(bars[1].v0, [-1.1, 0.2, 0.0])
        np.testing.assert_allclose(bars[1].a0, [0.0, -0.05, 0.0])
        assert bars[1].inflate(0.15).r == pytest.approx(0.85)


class TestKalmanObstacleTracker:
    def test_constant_velocity_convergence_and_closed_loop_barrier_integration(self):
        rng = np.random.default_rng(42)
        tracker = KalmanObstacleTracker(
            dim=3,
            model="cv",
            process_noise=0.2,
            measurement_noise=0.025,
            initial_vel_var=4.0,
            confidence_sigma=2.0,
        )
        p_true0 = np.array([1.8, 0.05, 0.0], dtype=np.float64)
        v_true = np.array([-1.6, 0.0, 0.0], dtype=np.float64)
        dt = 0.025

        # Warm-up 15 frames of noisy 3D detections and check velocity + uncertainty convergence
        first_vel_unc = None
        for k in range(15):
            t_k = k * dt
            z_noisy = p_true0 + v_true * t_k + rng.normal(0.0, 0.025, size=3)
            est = tracker.update(z_noisy, dt=dt)
            if k == 1:
                first_vel_unc = est["vel_uncertainty_radius"]
        assert first_vel_unc is not None
        assert tracker.velocity_uncertainty_radius() < first_vel_unc
        assert tracker.position_uncertainty_radius() < 0.08
        assert np.linalg.norm(tracker.vel - v_true) < 0.22

        # Continue online tracking while filtering a stationary robot out of the human's path
        dyn = LinearDynamics(np.zeros((3, 3)), np.eye(3))
        x_robot = np.array([0.0, 0.12, 0.0], dtype=np.float64)
        base_r = 0.35
        min_true_clearance = float("inf")
        for k in range(15, 70):
            t_k = k * dt
            p_true = p_true0 + v_true * t_k
            z_noisy = p_true + rng.normal(0.0, 0.025, size=3)
            tracker.update(z_noisy, dt=dt)
            mov_bar = tracker.to_moving_barrier(
                base_radius=base_r,
                alpha=1.5,
                sigma=2.0,
                vel_horizon_s=dt,
                state_slice=slice(0, 3),
                state_dim=3,
            )
            filt = StateCBFSafetyFilter(dyn, [mov_bar], u_min=[-3.5] * 3, u_max=[3.5] * 3)
            out = filt.filter(x_robot, [0.0, 0.0, 0.0])
            assert out["verified_safe"] is True
            x_robot = x_robot + dt * np.asarray(out["u_filtered"], dtype=np.float64)
            p_true_next = p_true0 + v_true * ((k + 1) * dt)
            min_true_clearance = min(
                min_true_clearance,
                float(np.linalg.norm(x_robot - p_true_next) - base_r),
            )
        assert min_true_clearance > 0.0  # zero true collisions despite noisy observations!

    def test_constant_acceleration_mode_tracks_accelerating_target(self):
        rng = np.random.default_rng(7)
        tracker = KalmanObstacleTracker(
            dim=3,
            model="ca",
            process_noise=0.4,
            measurement_noise=0.01,
        )
        p0 = np.array([0.5, -0.2, 0.1], dtype=np.float64)
        v0 = np.array([-0.8, 0.3, 0.0], dtype=np.float64)
        a_true = np.array([-0.6, 0.0, 0.2], dtype=np.float64)
        dt = 0.04
        for k in range(45):
            t = k * dt
            p_t = p0 + v0 * t + 0.5 * a_true * (t * t)
            tracker.update(p_t + rng.normal(0.0, 0.01, size=3), dt=dt)
        t_last = 44 * dt
        assert np.linalg.norm(tracker.vel - (v0 + a_true * t_last)) < 0.2
        assert np.linalg.norm(tracker.acc - a_true) < 0.35


class TestTimeVaryingBarrierBackwardCompatibility:
    def test_single_and_two_arg_dh_dt_and_second_derivatives(self):
        tv = TimeVaryingBarrier(
            target=[1.0, 0.0, 0.0],
            target_radius=0.2,
            initial_radius=1.2,
            horizon_s=4.0,
            alpha=2.0,
            current_time=1.5,
            state_slice=slice(0, 3),
            state_dim=6,
        )
        x = np.zeros(6, dtype=np.float64)
        # 1-arg BarrierConstraint hook uses current_time=1.5
        dh_1arg = tv.dh_dt(x)
        # 2-arg legacy signature at t=1.5 and t=2.5
        dh_2arg_same = tv.dh_dt(x, 1.5)
        dh_2arg_other = tv.dh_dt(x, 2.5)
        assert dh_1arg == pytest.approx(dh_2arg_same, abs=1e-12)
        assert dh_1arg != pytest.approx(dh_2arg_other, abs=1e-6)

        # Verify d2h_dt2 matches finite-difference derivative of dh_dt
        eps = 1e-5
        fd_d2h = (tv.dh_dt(x, 1.5 + eps) - tv.dh_dt(x, 1.5 - eps)) / (2.0 * eps)
        assert tv.d2h_dt2(x) == pytest.approx(fd_d2h, abs=1e-7)
        assert tv.d2h_dt2(x, 5.0) == pytest.approx(0.0, abs=1e-12)
        assert tv.grad_dh_dt(x).shape == (6,)
        assert np.allclose(tv.grad_dh_dt(x), 0.0)

        # HOCBF wrapper over TimeVaryingBarrier drives 2nd-order Euler-Lagrange system to target
        tv_2d = TimeVaryingBarrier(
            target=[0.8, 0.0],
            target_radius=0.25,
            initial_radius=1.4,
            horizon_s=3.0,
            alpha=2.5,
            state_slice=slice(0, 2),
        )
        hocbf = HighOrderBarrier(tv_2d, alpha1=2.5, alpha2=2.5, state_slice=slice(0, 2))
        dyn = EulerLagrangeDynamics(n_dof=2)
        filt = StateCBFSafetyFilter(dyn, [hocbf], u_min=[-10.0, -10.0], u_max=[10.0, 10.0])
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        dt = 0.03
        for _ in range(105):
            out = filt.filter(state, [0.0, 0.0])
            assert out["verified_safe"] is True
            u = np.asarray(out["u_filtered"], dtype=np.float64)
            state = state + dt * (dyn.f(state) + dyn.g(state) @ u)
            hocbf.advance_time(dt)
        assert np.linalg.norm(state[:2] - np.array([0.8, 0.0])) <= 0.26


class TestCompositionalSTLBarrierTrees:
    def test_stl_and_barrier_sound_inner_approximation_and_exact_derivatives(self):
        b1 = MovingObstacleBarrier(center=[0.6, 0.0, 0.0], radius=0.35, velocity=[-0.4, 0.1, 0.0])
        b2 = TimeVaryingBarrier(
            target=[0.0, 0.0, 0.0],
            target_radius=0.4,
            initial_radius=1.5,
            horizon_s=4.0,
            current_time=0.8,
        )
        b1.set_time(0.8)
        and_bar = STLAndBarrier([b1, b2], eta=20.0, alpha=1.5)

        rng = np.random.default_rng(19)
        pts = rng.uniform(-1.2, 1.2, size=(80, 3))
        saw_positive = False
        for pt in pts:
            h_and = and_bar.h(pt)
            h1, h2 = b1.h(pt), b2.h(pt)
            assert h_and <= min(h1, h2) + 1e-12
            if h_and >= 0.0:
                saw_positive = True
                assert h1 >= 0.0 and h2 >= 0.0
        assert saw_positive

        # Check exact chain-rule grad_h, dh_dt, d2h_dt2, grad_dh_dt against finite differences
        x_probe = np.array([-0.2, 0.3, 0.1], dtype=np.float64)
        eps = 1e-6
        fd_grad = np.zeros(3, dtype=np.float64)
        for i in range(3):
            ei = np.zeros(3, dtype=np.float64)
            ei[i] = eps
            fd_grad[i] = (and_bar.h(x_probe + ei) - and_bar.h(x_probe - ei)) / (2.0 * eps)
        np.testing.assert_allclose(and_bar.grad_h(x_probe), fd_grad, atol=1e-5)

        t0 = 0.8
        and_bar.set_time(t0 + eps)
        hp = and_bar.h(x_probe)
        dhp = and_bar.dh_dt(x_probe)
        gp = and_bar.grad_h(x_probe)
        and_bar.set_time(t0 - eps)
        hm = and_bar.h(x_probe)
        dhm = and_bar.dh_dt(x_probe)
        gm = and_bar.grad_h(x_probe)
        and_bar.set_time(t0)

        assert and_bar.dh_dt(x_probe) == pytest.approx((hp - hm) / (2.0 * eps), abs=1e-5)
        assert and_bar.d2h_dt2(x_probe) == pytest.approx((dhp - dhm) / (2.0 * eps), abs=1e-4)
        np.testing.assert_allclose(and_bar.grad_dh_dt(x_probe), (gp - gm) / (2.0 * eps), atol=1e-4)
        assert and_bar.inflate(0.12).h(x_probe) == pytest.approx(and_bar.h(x_probe) - 0.12, abs=1e-12)

    def test_stl_or_barrier_soundness_and_reaches_either_goal_a_or_goal_b(self):
        comp = SemanticCBFCompiler()
        spec = (
            "OR("
            "eventually(target=[1.0, 0.7, 0.0], radius=0.3, initial_radius=1.6, within_s=2.5, alpha=2.0), "
            "eventually(target=[1.0, -0.7, 0.0], radius=0.3, initial_radius=1.6, within_s=2.5, alpha=2.0), "
            "eta=25.0, alpha=2.0"
            ")"
        )
        or_bar = comp.compile_stl_tree(spec, pos_slice=slice(0, 3), state_dim=3)
        assert isinstance(or_bar, STLOrBarrier)
        g_a, g_b = or_bar.barriers[0], or_bar.barriers[1]

        # Soundness check: h_or(x) <= max(h_A(x), h_B(x)) everywhere
        rng = np.random.default_rng(11)
        for pt in rng.uniform(-1.5, 2.0, size=(50, 3)):
            val = or_bar.h(pt)
            assert val <= max(g_a.h(pt), g_b.h(pt)) + 1e-12
            if val >= 0.0:
                assert (g_a.h(pt) >= 0.0) or (g_b.h(pt) >= 0.0)

        dyn = LinearDynamics(np.zeros((3, 3)), np.eye(3))
        dt = 0.05

        # Branch 1: starting at y = +0.20 guides robot into Goal A ([1.0, +0.7, 0.0])
        or_bar.set_time(0.0)
        filt = StateCBFSafetyFilter(dyn, [or_bar], u_min=[-2.5] * 3, u_max=[2.5] * 3)
        x_a = np.array([0.0, 0.20, 0.0], dtype=np.float64)
        for _ in range(52):
            res = filt.filter(x_a, [0.0, 0.0, 0.0])
            assert res["verified_safe"] is True
            x_a = x_a + dt * np.asarray(res["u_filtered"], dtype=np.float64)
            or_bar.advance_time(dt)
        assert g_a.h(x_a) >= 0.0
        assert g_b.h(x_a) < 0.0

        # Branch 2: starting at y = -0.20 guides robot into Goal B ([1.0, -0.7, 0.0])
        or_bar.set_time(0.0)
        x_b = np.array([0.0, -0.20, 0.0], dtype=np.float64)
        for _ in range(52):
            res = filt.filter(x_b, [0.0, 0.0, 0.0])
            assert res["verified_safe"] is True
            x_b = x_b + dt * np.asarray(res["u_filtered"], dtype=np.float64)
            or_bar.advance_time(dt)
        assert g_b.h(x_b) >= 0.0
        assert g_a.h(x_b) < 0.0

    def test_stl_until_barrier_and_nested_tree_compilation(self):
        comp = SemanticCBFCompiler()
        scene = {
            "entities": {
                "walking_human": {
                    "center": [0.55, 0.0, 0.0],
                    "radius": 0.15,
                    "velocity": [0.0, -0.25, 0.0],
                },
                "dock_station": {"center": [1.1, 0.35, 0.0], "radius": 0.28},
            }
        }
        nested_expr = (
            "AND("
            'avoid(entity="walking_human", min_dist_m=0.15, alpha=2.0), '
            "UNTIL("
            "keep_in_box(min=[-0.35, -0.35, -0.35], max=[0.35, 0.35, 0.35], alpha=2.5), "
            'eventually(entity="dock_station", initial_radius=1.8, within_s=3.6, alpha=2.5), '
            "t_start=0.0, t_switch=1.0, t_end=3.6, alpha=2.5"
            "), "
            "eta=25.0, alpha=2.5"
            ")"
        )
        root_bar = comp.compile_stl_tree(nested_expr, scene_graph=scene, pos_slice=slice(0, 3), state_dim=3)
        assert isinstance(root_bar, STLAndBarrier)
        mov_child = root_bar.barriers[0]
        until_child = root_bar.barriers[1]
        assert isinstance(mov_child, MovingObstacleBarrier)
        assert isinstance(until_child, STLUntilBarrier)

        dyn = LinearDynamics(np.zeros((3, 3)), np.eye(3))
        filt = StateCBFSafetyFilter(dyn, [root_bar], u_min=[-3.0] * 3, u_max=[3.0] * 3)
        x = np.array([0.0, 0.15, 0.0], dtype=np.float64)
        dt = 0.04
        root_bar.set_time(0.0)
        for step in range(92):
            t_cur = step * dt
            # Nominal tries to leave hold box early during [0, 1.0], then stays zero
            u_nom = [1.2, 0.0, 0.0] if t_cur < 0.9 else [0.0, 0.0, 0.0]
            out = filt.filter(x, u_nom)
            assert out["verified_safe"] is True, f"step={step}, t={t_cur}, out={out}"
            if t_cur <= 1.0:
                # Hold box [-0.35, 0.35]^3 must remain strictly satisfied throughout [t_start, t_switch]
                assert np.all(np.abs(x) <= 0.35 + 1e-4)
            assert mov_child.h(x) >= -1e-4
            x = x + dt * np.asarray(out["u_filtered"], dtype=np.float64)
            root_bar.advance_time(dt)

        # By t_end=3.6, robot has transitioned out of the hold box and reached dock_station (radius 0.28)
        assert np.linalg.norm(x - np.array([1.1, 0.35, 0.0])) <= 0.28 + 1e-3

        # Equivalent dict specification also compiles cleanly, and unknown entities fail closed
        dict_tree = {
            "op": "UNTIL",
            "hold": {"rule_type": "keep_in_box", "params": {"min": [-0.4, -0.4, -0.4], "max": [0.4, 0.4, 0.4]}},
            "goal": {"op": "OR", "args": ['eventually(entity="dock_station", within_s=3.0)']},
            "t_start": 0.0,
            "t_switch": 1.0,
            "t_end": 3.0,
        }
        dict_bar = comp.compile_stl_tree(dict_tree, scene_graph=scene)
        assert isinstance(dict_bar, STLUntilBarrier)

        with pytest.raises(ValueError, match="ghost_obstacle"):
            comp.compile_stl_tree('AND(avoid(entity="ghost_obstacle", min_dist_m=0.2))', scene_graph=scene)


def test_no_forbidden_markers_in_owned_files():
    forbidden = "Prop" + "osal"
    pat = re.compile(forbidden, re.IGNORECASE)
    for rel in (
        "core/state_cbf_filter.py",
        "core/semantic_cbf_compiler.py",
        "tests/test_dynamic_cbf_and_stl_trees.py",
    ):
        text = (_PKG_DIR / rel).read_text(encoding="utf-8")
        assert pat.search(text) is None, f"Forbidden marker in {rel}"
