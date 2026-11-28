"""Tests for High-Order CBFs (HOCBF), Euler-Lagrange dynamics, whole-body barriers,
conformal inflation composition, soft-barrier relaxation, and Simplex fallback.
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

from aetheris_microbrains.core.state_cbf_filter import (
    EulerLagrangeDynamics,
    HalfPlaneBarrier,
    HighOrderBarrier,
    LinearDynamics,
    SphereBarrier,
    StateCBFSafetyFilter,
)
from aetheris_microbrains.core.whole_body_barriers import (
    CapsulePairBarrier,
    KineticEnergyBarrier,
    ManipulabilityBarrier,
    SimplexBackupController,
    TiltConeBarrier,
    WrenchConeBarrier,
)


class TestConformalInflationExactness:
    def test_sphere_barrier_inflate_exact_radius_shift(self):
        bar = SphereBarrier(center=[1.0, 0.0], radius=0.5, alpha=0.8)
        delta = 0.25
        inflated = bar.inflate(delta)

        assert inflated.r == pytest.approx(0.75)
        # Point at distance 0.6 from center: outside r=0.5, inside r+delta=0.75
        x = np.array([1.6, 0.0])
        assert bar.h(x) == pytest.approx(0.6**2 - 0.5**2)
        assert bar.h(x) > 0.0
        assert inflated.h(x) == pytest.approx(0.6**2 - 0.75**2)
        assert inflated.h(x) < 0.0

        # StateCBFSafetyFilter with conformal_margin=delta must produce the exact same
        # halfspace (A, b) as filtering directly against bar.inflate(delta) with margin=0.
        dyn = LinearDynamics(np.zeros((2, 2)), np.eye(2))
        filt_margin = StateCBFSafetyFilter(dyn, [bar], conformal_margin=delta)
        filt_manual = StateCBFSafetyFilter(dyn, [inflated], conformal_margin=0.0)

        A_m, b_m, _ = filt_margin._halfspaces(x)
        A_e, b_e, _ = filt_manual._halfspaces(x)
        assert np.allclose(A_m, A_e, atol=1e-12)
        assert np.allclose(b_m, b_e, atol=1e-12)

    def test_halfplane_barrier_inflate_shifts_b_cleanly(self):
        hp = HalfPlaneBarrier(a=[1.0, 0.0], b=0.2, alpha=1.0)
        inf = hp.inflate(0.3)
        assert inf.b == pytest.approx(0.5)
        x = np.array([0.4, 0.0])
        assert hp.h(x) == pytest.approx(0.2)
        assert inf.h(x) == pytest.approx(-0.1)


class TestEulerLagrangeAndHOCBF:
    def test_relative_degree_1_lacks_authority_while_hocbf_brakes_wall(self):
        dyn = EulerLagrangeDynamics(n_dof=1)  # x = [q, qdot], u = tau
        # Wall at q <= 1.0  =>  h(x) = -q + 1.0 >= 0
        wall_r1 = HalfPlaneBarrier(a=[-1.0, 0.0], b=-1.0, alpha=1.0, relative_degree=1)
        x_probe = np.array([0.6, 1.2])
        # First-order CBF has L_g h = grad_h @ g(x) = [-1, 0] @ [0, 1] = 0
        L_g_h_r1 = wall_r1.grad_h(x_probe) @ dyn.g(x_probe)
        assert np.allclose(L_g_h_r1, 0.0)

        # HOCBF (relative_degree=2) recovers control authority via L_g L_f h = -1
        wall_hocbf = HighOrderBarrier(wall_r1, alpha1=2.0, alpha2=2.0)
        assert wall_hocbf.relative_degree == 2
        filt = StateCBFSafetyFilter(dyn, [wall_hocbf], u_min=[-10.0], u_max=[10.0])

        A, b, _ = filt._halfspaces(x_probe)
        assert abs(A[0, 0]) > 0.5  # nonzero braking authority: A = [[-1.0]]

        # Closed-loop rollout with aggressive nominal acceleration into the wall
        x = np.array([0.0, 1.0])
        dt = 0.02
        h_vals = []
        braked = False
        for _ in range(150):
            u_nom = [4.0]  # tries to accelerate into wall q=1.0
            out = filt.filter(x, u_nom)
            assert out["verified_safe"], out
            u = np.asarray(out["u_filtered"])
            if u[0] < 0.0:
                braked = True
            h_vals.append(wall_hocbf.h(x))
            xdot = dyn.f(x) + dyn.g(x) @ u
            x = x + dt * xdot
        assert braked
        assert min(h_vals) >= -1e-4

    def test_2dof_nonlinear_euler_lagrange_avoids_sphere_obstacle(self):
        # 2-DOF manipulator/point-mass with configuration-dependent M(q), C(q, dq), g(q)
        def M_fn(q):
            return np.array([[2.0 + 0.3 * np.cos(q[1]), 0.1], [0.1, 1.2]])

        def C_fn(q, dq):
            return np.array([[-0.15 * np.sin(q[1]) * dq[1], 0.0], [0.15 * np.sin(q[1]) * dq[0], 0.0]])

        def g_fn(q):
            return np.array([0.5 * np.sin(q[0]), 0.3 * np.sin(q[1])])

        dyn = EulerLagrangeDynamics(n_dof=2, M_fn=M_fn, C_fn=C_fn, g_fn=g_fn)
        # Position-only sphere obstacle centered at q = [1.0, 0.0], r = 0.35
        obs = SphereBarrier(center=[1.0, 0.0], radius=0.35, alpha=2.5)
        hocbf = HighOrderBarrier(obs, alpha1=2.5, alpha2=3.0)
        filt = StateCBFSafetyFilter(dyn, [hocbf], u_min=[-15.0, -15.0], u_max=[15.0, 15.0])

        # Verify lie_derivatives helper matches _halfspaces
        x_test = np.array([0.4, 0.1, 0.8, -0.1])
        Lf, Lf2, LgLf = dyn.lie_derivatives(hocbf, x_test)
        A, b, _ = filt._halfspaces(x_test)
        assert np.allclose(A[0], LgLf, atol=1e-9)
        expected_b = -Lf2 - (2.5 + 3.0) * Lf - 2.5 * 3.0 * hocbf.h(x_test)
        assert b[0] == pytest.approx(expected_b, abs=1e-9)

        # Rollout driving straight through obstacle center [1.0, 0.0] toward [2.0, 0.0]
        x = np.array([0.0, 0.05, 0.6, 0.0])
        dt = 0.015
        min_h = float("inf")
        for _ in range(180):
            q, dq = x[:2], x[2:]
            u_nom = 4.0 * (np.array([2.0, 0.0]) - q) - 1.5 * dq + g_fn(q)
            out = filt.filter(x, u_nom)
            assert out["verified_safe"], out
            min_h = min(min_h, hocbf.h(x))
            u = np.asarray(out["u_filtered"])
            x = x + dt * (dyn.f(x) + dyn.g(x) @ u)
        assert min_h >= -1e-3


class TestRelaxationAndSimplexFallback:
    def test_stage_a_soft_barrier_relaxation_preserves_hard_barrier(self):
        dyn = LinearDynamics(np.zeros((2, 2)), np.eye(2))
        # Hard barrier requires u[0] >= 0.2 (at x=[0,0], h=0 -> L_g h . u >= 0 -> wait: set h=-0.4, alpha=0.5 => u[0] >= 0.2)
        hard_bar = HalfPlaneBarrier(a=[1.0, 0.0], b=0.4, alpha=0.5, hard=True)
        # Conflicting soft barrier demands u[0] <= -0.5 (a=[-1, 0], b=1.0, x=[0,0] => h=-1 => -u[0] >= 0.5 => u[0] <= -0.5)
        soft_bar = HalfPlaneBarrier(a=[-1.0, 0.0], b=1.0, alpha=0.5, hard=False)

        filt = StateCBFSafetyFilter(dyn, [hard_bar, soft_bar], u_min=[-1.0, -1.0], u_max=[1.0, 1.0])
        out = filt.filter([0.0, 0.0], [0.0, 0.0])
        assert out["status"] == "safe_soft_relaxed"
        assert out["verified_safe"] is True
        assert out["relaxed_soft_barriers"] == 1
        assert out["u_filtered"][0] >= 0.2 - 1e-6

    def test_stage_b_simplex_backup_engages_when_hard_barriers_infeasible(self):
        def g_fn(q):
            return np.array([0.8, -0.4])

        dyn = EulerLagrangeDynamics(n_dof=1)  # 1-DOF just for simple state? Use 2-DOF!
        dyn2 = EulerLagrangeDynamics(n_dof=2, g_fn=g_fn)
        backup = SimplexBackupController(n_dof=2, kd=3.0, g_fn=g_fn)

        # Two conflicting hard barriers on u
        b1 = HalfPlaneBarrier(a=[0.0, 0.0, 1.0, 0.0], b=1.0, alpha=1.0, hard=True)   # u[0] >= 1.0
        b2 = HalfPlaneBarrier(a=[0.0, 0.0, -1.0, 0.0], b=1.0, alpha=1.0, hard=True)  # u[0] <= -1.0

        # Without backup controller -> unsafe_no_solution
        filt_no_backup = StateCBFSafetyFilter(dyn2, [b1, b2], u_min=[-5.0, -5.0], u_max=[5.0, 5.0])
        x = np.array([0.2, -0.3, 1.0, -0.5])
        out_none = filt_no_backup.filter(x, [0.0, 0.0])
        assert out_none["status"] == "unsafe_no_solution"
        assert out_none["verified_safe"] is False

        # With SimplexBackupController -> simplex_braking_engaged
        filt_simplex = StateCBFSafetyFilter(
            dyn2,
            [b1, b2],
            u_min=[-5.0, -5.0],
            u_max=[5.0, 5.0],
            backup_controller=backup,
        )
        out_sim = filt_simplex.filter(x, [2.0, 2.0])
        assert out_sim["status"] == "simplex_braking_engaged"
        assert out_sim["backup_engaged"] is True
        assert out_sim["verified_safe"] is False
        expected_tau = -3.0 * np.array([1.0, -0.5]) + g_fn(x[:2])
        assert np.allclose(out_sim["u_filtered"], expected_tau, atol=1e-9)

        # Box clipping is respected on backup action too
        filt_tight = StateCBFSafetyFilter(
            dyn2,
            [b1, b2],
            u_min=[-1.0, -1.0],
            u_max=[1.0, 1.0],
            backup_controller=backup,
        )
        out_clip = filt_tight.filter(x, [0.0, 0.0])
        assert out_clip["status"] == "simplex_braking_engaged"
        assert np.allclose(out_clip["u_filtered"], np.clip(expected_tau, -1.0, 1.0), atol=1e-9)


class TestWholeBodyBarriers:
    def test_capsule_pair_barrier_clearance_grad_and_hocbf(self):
        # 1-DOF translating link segment along x: p0=[q[0], 0, 0], p1=[q[0], 0, 1]
        def link_fk(q):
            return ([q[0], 0.0, 0.0], [q[0], 0.0, 1.0])

        # Opponent vertical human-arm segment at x=1.0, y=0.0, z in [0, 1]
        opp = ([1.0, 0.0, 0.0], [1.0, 0.0, 1.0])
        bar = CapsulePairBarrier(link_fk, opp, r_safe=0.25, n_dof=1)

        assert bar.h(np.array([0.5, 0.0])) == pytest.approx(0.5**2 - 0.25**2, abs=1e-9)
        assert bar.h(np.array([0.85, 0.0])) < 0.0

        grad = bar.grad_h(np.array([0.5, 0.3]))
        # d^2 = (1 - q)^2 => dh/dq = -2(1 - q) = -1.0, dh/dqdot = 0
        assert grad[0] == pytest.approx(-1.0, abs=1e-5)
        assert grad[1] == pytest.approx(0.0, abs=1e-9)

        inf = bar.inflate(0.1)
        assert inf.r_safe == pytest.approx(0.35)
        assert inf.h(np.array([0.7, 0.0])) < 0.0 < bar.h(np.array([0.7, 0.0]))

        # Compose with HighOrderBarrier + EulerLagrangeDynamics
        dyn = EulerLagrangeDynamics(n_dof=1)
        hocbf = HighOrderBarrier(bar, alpha1=2.0, alpha2=2.0)
        filt = StateCBFSafetyFilter(dyn, [hocbf], u_min=[-10.0], u_max=[10.0])
        out = filt.filter(np.array([0.6, 0.8]), [5.0])
        assert out["verified_safe"] is True
        assert out["u_filtered"][0] < 0.0

    def test_manipulability_barrier_detects_singularity(self):
        # Planar 2-link arm with L1 = L2 = 1.0
        def jacobian_2link(q):
            t1, t2 = q[0], q[1]
            s1, c1 = np.sin(t1), np.cos(t1)
            s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
            return np.array([[-s1 - s12, -s12], [c1 + c12, c12]])

        bar = ManipulabilityBarrier(jacobian_2link, eps_sing=0.05, n_dof=2)
        # At q2 = pi/2, det(J J^T) = (L1 L2 sin(q2))^2 = 1.0
        assert bar.h(np.array([0.2, np.pi / 2, 0.0, 0.0])) == pytest.approx(1.0 - 0.05, abs=1e-6)
        # Near singularity q2 = 0.05, det(J J^T) = sin(0.05)^2 ~ 0.0025 < 0.05
        assert bar.h(np.array([0.2, 0.05, 0.0, 0.0])) < 0.0

        grad = bar.grad_h(np.array([0.2, 0.4, 0.0, 0.0]))
        assert abs(grad[0]) < 1e-6  # manipulability invariant to base joint q1
        assert grad[1] > 0.0        # increasing q2 toward pi/2 increases manipulability
        assert np.allclose(grad[2:], 0.0)

        inf = bar.inflate(0.1)
        assert inf.h(np.array([0.2, np.pi / 2])) == pytest.approx(bar.h(np.array([0.2, np.pi / 2])) - 0.1)

    def test_tilt_cone_barrier_enforces_cone(self):
        # End-effector normal tilted by angle q[0] around y-axis: [sin(q0), 0, cos(q0)]
        def normal_fn(q):
            return np.array([np.sin(q[0]), 0.0, np.cos(q[0])])

        bar = TiltConeBarrier(normal_fn, max_tilt_rad=np.deg2rad(30.0))
        assert bar.h(np.array([0.0])) == pytest.approx(1.0 - np.cos(np.deg2rad(30.0)), abs=1e-9)
        assert bar.h(np.array([np.deg2rad(20.0)])) > 0.0
        assert bar.h(np.array([np.deg2rad(45.0)])) < 0.0

        grad = bar.grad_h(np.array([0.2]))
        assert grad[0] == pytest.approx(-np.sin(0.2), abs=1e-5)
        assert bar.inflate(0.05).h(np.array([0.0])) == pytest.approx(bar.h(np.array([0.0])) - 0.05)

    def test_kinetic_energy_barrier_iso15066_tightens_with_distance(self):
        M_const = np.diag([2.0, 2.0])
        # Distance to human at q_human = [1.0, 0.0]
        bar = KineticEnergyBarrier(
            n_dof=2,
            M_fn=M_const,
            distance_fn=lambda q: float(np.linalg.norm(q - np.array([1.0, 0.0]))),
            e_max_far=8.0,
            e_max_near=0.4,
            d_safe=0.5,
        )
        # Same speed ||dq|| = 1.0 => KE = 0.5 * 2 * 1 = 1.0 J
        x_far = np.array([-3.0, 0.0, 1.0, 0.0])   # d = 4.0 -> E_max ~ 7.88 > 1.0 (safe)
        x_near = np.array([0.95, 0.0, 1.0, 0.0])  # d = 0.05 -> E_max ~ 0.475 < 1.0 (unsafe)
        assert bar.max_energy(x_far[:2]) > bar.max_energy(x_near[:2])
        assert bar.h(x_far) > 0.0
        assert bar.h(x_near) < 0.0

        # Filter reduces velocity-increasing torque when near human
        dyn = EulerLagrangeDynamics(n_dof=2, M_fn=M_const)
        filt = StateCBFSafetyFilter(dyn, [bar], u_min=[-10.0, -10.0], u_max=[10.0, 10.0])
        out = filt.filter(x_near, [5.0, 0.0])
        assert out["verified_safe"] is True
        assert out["u_filtered"][0] < 0.0

    def test_wrench_cone_barrier_friction_and_normal_bounds(self):
        bar = WrenchConeBarrier(mu=0.5, f_max=20.0, f_min=1.0, normal_axis=2)

        # Safe inside cone: fz=10, |ft|=3 <= 0.5*10=5, 1 <= 10 <= 20
        assert bar.h(np.array([3.0, 0.0, 10.0])) == pytest.approx(2.0, abs=1e-5)
        # Violates Coulomb friction: |ft|=6 > 5
        assert bar.h(np.array([6.0, 0.0, 10.0])) == pytest.approx(-1.0, abs=1e-5)
        # Exceeds f_max=20
        assert bar.h(np.array([0.0, 0.0, 24.0])) == pytest.approx(-4.0, abs=1e-5)
        # Below f_min=1 (loss of contact)
        assert bar.h(np.array([0.0, 0.0, 0.2])) == pytest.approx(-0.8, abs=1e-5)

        grad_cone = bar.grad_h(np.array([6.0, 0.0, 10.0]))
        assert grad_cone[0] == pytest.approx(-1.0, abs=1e-4)
        assert grad_cone[2] == pytest.approx(0.5, abs=1e-4)

        inf = bar.inflate(0.5)
        assert inf.h(np.array([3.0, 0.0, 10.0])) == pytest.approx(bar.h(np.array([3.0, 0.0, 10.0])) - 0.5)
