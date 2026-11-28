"""Tests for native Rust backend parity in StateCBFSafetyFilter,
CLFCBFSafetyFilter exponential stability & slack relaxation, and
DifferentiableCBFProjection implicit KKT Jacobians & VJP.
"""

from __future__ import annotations

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

from aetheris_microbrains.core.differentiable_clf_cbf import (
    CLFCBFSafetyFilter,
    DifferentiableCBFProjection,
    QuadraticCLF,
)
from aetheris_microbrains.core.rust_shim import NativeRTGateLib
from aetheris_microbrains.core.state_cbf_filter import (
    HalfPlaneBarrier,
    LinearDynamics,
    SphereBarrier,
    StateCBFSafetyFilter,
)


class TestNativeRustBackendParity:
    def test_native_rust_matches_python_dykstra_to_1e9(self):
        dyn = LinearDynamics(
            A=np.array([[0.0, 0.1], [-0.1, 0.0]]),
            B=np.eye(2),
        )
        barriers = [
            SphereBarrier(center=[1.0, 0.2], radius=0.6, alpha=1.2),
            HalfPlaneBarrier(a=[1.0, 1.0], b=0.5, alpha=0.8),
        ]
        filt_py = StateCBFSafetyFilter(
            dyn, barriers, u_min=[-1.5, -1.5], u_max=[1.5, 1.5], use_native_rust=False
        )
        filt_rs = StateCBFSafetyFilter(
            dyn, barriers, u_min=[-1.5, -1.5], u_max=[1.5, 1.5], use_native_rust=True
        )

        x = np.array([1.7, 0.35])
        u_nom = np.array([-2.0, -0.8])
        out_py = filt_py.filter(x, u_nom)
        out_rs = filt_rs.filter(x, u_nom)

        assert out_py["solver_backend"] == "python_numpy"
        if NativeRTGateLib.is_available():
            assert out_rs["solver_backend"] == "native_rust"
        else:
            assert out_rs["solver_backend"] == "python_numpy"

        assert out_rs["verified_safe"] == out_py["verified_safe"]
        assert np.allclose(out_rs["u_filtered"], out_py["u_filtered"], atol=1e-9)

    def test_native_rust_falls_back_gracefully_when_limits_exceeded(self):
        # control_dim = 64 is within MAX_DIM = 64 (native_rust when built)
        dyn_ok = LinearDynamics(np.zeros((64, 64)), np.eye(64))
        bar_ok = HalfPlaneBarrier(a=[1.0] + [0.0] * 63, b=0.0, alpha=1.0)
        filt_ok = StateCBFSafetyFilter(dyn_ok, [bar_ok], u_min=-2.0 * np.ones(64), u_max=2.0 * np.ones(64), use_native_rust=True)
        out_ok = filt_ok.filter(np.zeros(64), np.array([-1.0] + [0.5] * 63))
        assert out_ok["verified_safe"] is True
        expected_backend = "native_rust" if NativeRTGateLib.is_available() else "python_numpy"
        assert out_ok["solver_backend"] == expected_backend

        # control_dim = 65 exceeds MAX_DIM = 64 -> seamless fallback to python_numpy
        dyn = LinearDynamics(np.zeros((65, 65)), np.eye(65))
        bar = HalfPlaneBarrier(a=[1.0] + [0.0] * 64, b=0.0, alpha=1.0)
        filt = StateCBFSafetyFilter(dyn, [bar], u_min=-2.0 * np.ones(65), u_max=2.0 * np.ones(65), use_native_rust=True)
        out = filt.filter(np.zeros(65), np.array([-1.0] + [0.5] * 64))
        assert out["verified_safe"] is True
        assert out["solver_backend"] == "python_numpy"
        assert out["u_filtered"][0] >= -1e-7


class TestCLFCBFSafetyFilter:
    def test_free_space_exponential_convergence_with_zero_slack(self):
        dyn = LinearDynamics(np.zeros((2, 2)), np.eye(2))
        x_goal = np.array([0.0, 0.0])
        clf = QuadraticCLF(x_goal=x_goal, gamma=2.0, slack_weight=100.0)
        filt = CLFCBFSafetyFilter(dyn, barriers=[], clf=clf, u_min=[-5.0, -5.0], u_max=[5.0, 5.0])

        x = np.array([2.0, -1.5])
        dt = 0.02
        V0 = clf.V(x)
        V_hist = [V0]
        deltas = []
        for _ in range(150):
            # Nominal stabilizing control satisfies CLF without slack in free space
            u_nom = -1.5 * (x - x_goal)
            out = filt.filter(x, u_nom)
            assert out["verified_safe"] is True
            assert out["status"] == "safe"
            deltas.append(out["clf_slack_delta"])
            u = np.asarray(out["u_filtered"], dtype=np.float64)
            x = x + dt * (dyn.f(x) + dyn.g(x) @ u)
            V_hist.append(clf.V(x))

        assert max(deltas) == pytest.approx(0.0, abs=1e-9)
        t_final = 150 * dt
        assert V_hist[-1] <= V0 * np.exp(-clf.gamma * t_final) + 1e-4
        assert V_hist[-1] < 1e-3

    def test_obstacle_preserves_safety_and_activates_clf_slack(self):
        dyn = LinearDynamics(np.zeros((2, 2)), np.eye(2))
        x_goal = np.array([0.0, 0.0])
        clf = QuadraticCLF(x_goal=x_goal, gamma=2.0, slack_weight=100.0)
        # Obstacle sits directly between initial state [2.0, 0.05] and goal [0.0, 0.0]
        obstacle = SphereBarrier(center=[1.0, 0.0], radius=0.6, alpha=2.5)
        filt = CLFCBFSafetyFilter(
            dyn,
            barriers=[obstacle],
            clf=clf,
            u_min=[-4.0, -4.0],
            u_max=[4.0, 4.0],
        )

        x = np.array([2.0, 0.08])
        dt = 0.02
        h_vals = []
        deltas = []
        for _ in range(250):
            u_nom = -2.0 * (x - x_goal)
            out = filt.filter(x, u_nom)
            assert out["verified_safe"] is True
            h_vals.append(obstacle.h(x))
            deltas.append(out["clf_slack_delta"])
            u = np.asarray(out["u_filtered"], dtype=np.float64)
            x = x + dt * (dyn.f(x) + dyn.g(x) @ u)

        # Strict CBF safety preserved for the entire rollout
        assert min(h_vals) >= -1e-4
        # CLF slack automatically activated (> 0) when obstacle blocked direct descent
        assert max(deltas) > 1e-2
        # Robot steers around the obstacle and reaches neighborhood of x_goal
        assert np.linalg.norm(x - x_goal) < 0.35


class TestDifferentiableCBFProjection:
    @staticmethod
    def _central_fd_jacobians(
        filt: StateCBFSafetyFilter,
        x: np.ndarray,
        u_nom: np.ndarray,
        eps: float = 1e-6,
    ):
        d_x = x.shape[0]
        d_u = u_nom.shape[0]
        m = len(filt.barriers)

        du_du_nom_fd = np.zeros((d_u, d_u), dtype=np.float64)
        for j in range(d_u):
            e = np.zeros(d_u, dtype=np.float64)
            e[j] = eps
            up = np.asarray(filt.filter(x, u_nom + e)["u_filtered"], dtype=np.float64)
            um = np.asarray(filt.filter(x, u_nom - e)["u_filtered"], dtype=np.float64)
            du_du_nom_fd[:, j] = (up - um) / (2.0 * eps)

        du_dx_fd = np.zeros((d_u, d_x), dtype=np.float64)
        for k in range(d_x):
            e = np.zeros(d_x, dtype=np.float64)
            e[k] = eps
            up = np.asarray(filt.filter(x + e, u_nom)["u_filtered"], dtype=np.float64)
            um = np.asarray(filt.filter(x - e, u_nom)["u_filtered"], dtype=np.float64)
            du_dx_fd[:, k] = (up - um) / (2.0 * eps)

        du_dalpha_fd = np.zeros((d_u, m), dtype=np.float64)
        for i, bar in enumerate(filt.barriers):
            a0 = bar.alpha
            bar.alpha = a0 + eps
            up = np.asarray(filt.filter(x, u_nom)["u_filtered"], dtype=np.float64)
            bar.alpha = a0 - eps
            um = np.asarray(filt.filter(x, u_nom)["u_filtered"], dtype=np.float64)
            bar.alpha = a0
            du_dalpha_fd[:, i] = (up - um) / (2.0 * eps)

        return du_du_nom_fd, du_dx_fd, du_dalpha_fd

    def test_interior_inactive_case(self):
        dyn = LinearDynamics(np.array([[0.1, -0.2], [0.2, 0.1]]), np.eye(2))
        bar = SphereBarrier(center=[0.0, 0.0], radius=1.0, alpha=1.0)
        filt = StateCBFSafetyFilter(dyn, [bar], u_min=[-3.0, -3.0], u_max=[3.0, 3.0])

        x = np.array([2.5, 1.0])
        u_nom = np.array([0.5, -0.4])
        res = DifferentiableCBFProjection.forward_and_jacobians(x, u_nom, filt)

        assert res["verified_safe"] is True
        assert res["n_active"] == 0
        assert np.allclose(res["u_star"], u_nom, atol=1e-12)
        assert np.allclose(res["du_du_nom"], np.eye(2), atol=1e-12)
        assert np.allclose(res["du_dx"], np.zeros((2, 2)), atol=1e-12)
        assert np.allclose(res["du_dalpha"], np.zeros((2, 1)), atol=1e-12)

        g = np.array([1.2, -0.7])
        gu, gx, ga = DifferentiableCBFProjection.vjp(g, res)
        assert np.allclose(gu, g, atol=1e-12)
        assert np.allclose(gx, np.zeros(2), atol=1e-12)
        assert np.allclose(ga, np.zeros(1), atol=1e-12)

    def test_single_active_cbf_halfspace_matches_fd_and_vjp(self):
        dyn = LinearDynamics(
            A=np.array([[-0.2, 0.4], [-0.3, 0.1]]),
            B=np.array([[1.0, 0.2], [-0.1, 1.0]]),
        )
        bar = SphereBarrier(center=[0.0, 0.0], radius=1.0, alpha=0.8)
        filt = StateCBFSafetyFilter(dyn, [bar], u_min=[-5.0, -5.0], u_max=[5.0, 5.0])

        x = np.array([1.15, 0.45])
        u_nom = np.array([-2.0, -1.2])
        res = DifferentiableCBFProjection.forward_and_jacobians(x, u_nom, filt)
        assert res["verified_safe"] is True
        assert res["active_cbf_indices"] == [0]
        assert res["n_active"] == 1

        fd_u, fd_x, fd_a = self._central_fd_jacobians(filt, x, u_nom)
        assert np.allclose(res["du_du_nom"], fd_u, atol=1e-4)
        assert np.allclose(res["du_dx"], fd_x, atol=1e-4)
        assert np.allclose(res["du_dalpha"], fd_a, atol=1e-4)

        g = np.array([0.8, -1.3])
        gu, gx, ga = DifferentiableCBFProjection.vjp(g, res)
        assert np.allclose(gu, fd_u.T @ g, atol=1e-4)
        assert np.allclose(gx, fd_x.T @ g, atol=1e-4)
        assert np.allclose(ga, fd_a.T @ g, atol=1e-4)

    def test_intersecting_halfspace_and_box_bound_matches_fd_and_vjp(self):
        dyn = LinearDynamics(
            A=np.array([[0.15, -0.1], [0.2, -0.25]]),
            B=np.array([[1.0, 0.3], [0.0, 1.0]]),
        )
        bar1 = SphereBarrier(center=[0.0, 0.0], radius=1.0, alpha=1.2)
        bar2 = HalfPlaneBarrier(a=[0.0, 1.0], b=-5.0, alpha=0.5)  # inactive second barrier
        filt = StateCBFSafetyFilter(
            dyn,
            [bar1, bar2],
            u_min=[-2.0, -2.0],
            u_max=[1.0, 2.0],
        )

        # Pick x and u_nom so bar1 is active AND u_star[0] saturates at u_max[0] = 1.0
        x = np.array([0.85, 0.58])
        u_nom = np.array([2.5, -1.8])
        res = DifferentiableCBFProjection.forward_and_jacobians(x, u_nom, filt)

        assert res["verified_safe"] is True
        assert 0 in res["active_cbf_indices"]
        assert 0 in res["active_box_upper"]
        assert res["n_active"] == 2

        fd_u, fd_x, fd_a = self._central_fd_jacobians(filt, x, u_nom)
        assert np.allclose(res["du_du_nom"], fd_u, atol=1e-4)
        assert np.allclose(res["du_dx"], fd_x, atol=1e-4)
        assert np.allclose(res["du_dalpha"], fd_a, atol=1e-4)

        g = np.array([-0.5, 2.0])
        gu, gx, ga = DifferentiableCBFProjection.vjp(g, res)
        assert np.allclose(gu, fd_u.T @ g, atol=1e-4)
        assert np.allclose(gx, fd_x.T @ g, atol=1e-4)
        assert np.allclose(ga, fd_a.T @ g, atol=1e-4)

    def test_redundant_active_constraints_pruned_cleanly(self):
        dyn = LinearDynamics(np.zeros((2, 2)), np.eye(2))
        # Two identical/collinear barriers active simultaneously
        b1 = HalfPlaneBarrier(a=[1.0, 0.0], b=0.5, alpha=1.0)
        b2 = HalfPlaneBarrier(a=[2.0, 0.0], b=1.0, alpha=1.0)
        filt = StateCBFSafetyFilter(dyn, [b1, b2], u_min=[-2.0, -2.0], u_max=[2.0, 2.0])

        x = np.array([0.5, 0.0])  # h = 0 for both, so both active at u_0 = 0
        u_nom = np.array([-1.0, 0.4])
        res = DifferentiableCBFProjection.forward_and_jacobians(x, u_nom, filt)
        assert res["verified_safe"] is True
        assert res["n_active"] == 1
        assert np.all(np.isfinite(res["du_du_nom"]))
        assert np.allclose(res["du_du_nom"], np.array([[0.0, 0.0], [0.0, 1.0]]), atol=1e-6)
