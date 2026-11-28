"""Tests for floating-base URDF centroidal kinematics/momentum, LIPM Capture Point CBF,
multi-foot GRF friction-pyramid allocation, and swing-foot clearance HOCBF.
"""

from __future__ import annotations

import math
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

from aetheris_microbrains.core.legged_centroidal_cbf import (
    CapturePointDCMBarrier,
    GRFFrictionPyramidAllocator,
    SupportPolygonBalanceCBF,
    SwingFootClearanceBarrier,
    support_polygon_from_feet,
)
from aetheris_microbrains.core.state_cbf_filter import StateCBFSafetyFilter, _fd_jacobian
from aetheris_microbrains.core.urdf_spatial_kinematics import (
    URDFRobotModel,
    exp_so3,
    hat_so3,
)


# ---------------------------------------------------------------------------
# URDF Fixtures: Floating-Base Branched Legged Robot & Swing Leg
# ---------------------------------------------------------------------------

FLOATING_BIPED_URDF = """<?xml version="1.0"?>
<robot name="floating_biped">
  <link name="torso">
    <inertial>
      <origin xyz="0 0 0.12" rpy="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="0.12" ixy="0.002" ixz="0.0" iyy="0.10" iyz="0.0" izz="0.08"/>
    </inertial>
  </link>

  <joint name="left_hip_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="0 0.12 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="80.0" velocity="6.0"/>
  </joint>
  <link name="left_thigh">
    <inertial>
      <origin xyz="0 0 -0.18" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.025" ixy="0" ixz="0" iyy="0.025" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="left_knee_pitch" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shank"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="0.1" effort="60.0" velocity="8.0"/>
  </joint>
  <link name="left_shank">
    <inertial>
      <origin xyz="0 0 -0.17" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_hip_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_thigh"/>
    <origin xyz="0 -0.12 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="80.0" velocity="6.0"/>
  </joint>
  <link name="right_thigh">
    <inertial>
      <origin xyz="0 0 -0.18" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.025" ixy="0" ixz="0" iyy="0.025" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="right_knee_pitch" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shank"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="0.1" effort="60.0" velocity="8.0"/>
  </joint>
  <link name="right_shank">
    <inertial>
      <origin xyz="0 0 -0.17" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.002"/>
    </inertial>
  </link>
</robot>
"""

SWING_LEG_3DOF_URDF = """<?xml version="1.0"?>
<robot name="swing_leg_3dof">
  <link name="hip_base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="hip_pitch" type="revolute">
    <parent link="hip_base"/>
    <child link="thigh"/>
    <origin xyz="0 0 0.72" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="90.0" velocity="6.0"/>
  </joint>
  <link name="thigh">
    <inertial>
      <origin xyz="0 0 -0.16" rpy="0 0 0"/>
      <mass value="1.8"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="knee_pitch" type="revolute">
    <parent link="thigh"/>
    <child link="shank"/>
    <origin xyz="0 0 -0.32" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="0.2" effort="75.0" velocity="8.0"/>
  </joint>
  <link name="shank">
    <inertial>
      <origin xyz="0 0 -0.16" rpy="0 0 0"/>
      <mass value="1.1"/>
      <inertia ixx="0.012" ixy="0" ixz="0" iyy="0.012" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="ankle_pitch" type="revolute">
    <parent link="shank"/>
    <child link="foot"/>
    <origin xyz="0 0 -0.32" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="40.0" velocity="8.0"/>
  </joint>
  <link name="foot">
    <inertial>
      <origin xyz="0.03 0 -0.02" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>
</robot>
"""


# ---------------------------------------------------------------------------
# 1. Floating-Base URDF CoM Jacobian & Centroidal Momentum Matrix Tests
# ---------------------------------------------------------------------------


class TestFloatingBaseCentroidalKinematics:
    def test_floating_base_com_jacobian_vs_fd_and_cmm_identities(self):
        robot_fb = URDFRobotModel.from_urdf_string(FLOATING_BIPED_URDF, floating_base=True)
        robot_fixed = URDFRobotModel.from_urdf_string(FLOATING_BIPED_URDF, floating_base=False)

        assert robot_fb.floating_base is True
        assert robot_fixed.floating_base is False
        assert robot_fb.n_joints == 4
        assert robot_fb.n_v == 10
        assert robot_fb.n_q == 10
        assert robot_fixed.n_v == 4
        assert robot_fb.total_mass == pytest.approx(14.4, abs=1e-12)

        rng = np.random.RandomState(42)
        eps = 1e-6

        for trial in range(6):
            p_b = rng.uniform(-0.5, 0.5, size=3)
            # Include exact zero base rotation on trial 0 to verify raw coordinate FD parity too
            phi_b = np.zeros(3) if trial == 0 else rng.uniform(-0.6, 0.6, size=3)
            q_j = rng.uniform(-0.9, 0.9, size=robot_fb.n_joints)
            q = np.concatenate([p_b, phi_b, q_j])
            v = rng.uniform(-1.2, 1.2, size=robot_fb.n_v)

            # Manual total CoM check
            fk = robot_fb.forward_kinematics(q)
            p_com = robot_fb.center_of_mass(q)
            expected_com = np.zeros(3)
            for node in robot_fb._nodes:
                T_k = fk[node.link_name]
                expected_com += node.mass * (T_k[:3, 3] + T_k[:3, :3] @ node.com_xyz)
            expected_com /= robot_fb.total_mass
            assert np.allclose(p_com, expected_com, atol=1e-13)

            # Analytical CoM Jacobian vs central FD on SE(3) x R^{n_j}
            J_com = robot_fb.com_jacobian(q)
            assert J_com.shape == (3, robot_fb.n_v)
            J_fd = np.zeros_like(J_com)
            for i in range(robot_fb.n_v):
                ei = np.zeros(robot_fb.n_v)
                ei[i] = 1.0
                q_plus = robot_fb.integrate_configuration(q, ei, dt=eps)
                q_minus = robot_fb.integrate_configuration(q, ei, dt=-eps)
                J_fd[:, i] = (robot_fb.center_of_mass(q_plus) - robot_fb.center_of_mass(q_minus)) / (
                    2.0 * eps
                )
            assert np.max(np.abs(J_com - J_fd)) < 1e-7

            if trial == 0:
                # At phi_b = 0, raw coordinate central FD also matches J_com
                J_coord_fd = _fd_jacobian(robot_fb.center_of_mass, q, eps=eps)
                assert np.max(np.abs(J_com - J_coord_fd)) < 1e-7

            # Orin-Goswami Centroidal Momentum Matrix linear-part equality: A_G[3:6] == m_total * J_com
            A_G = robot_fb.centroidal_momentum_matrix(q)
            assert A_G.shape == (6, robot_fb.n_v)
            assert np.allclose(A_G[3:6, :], robot_fb.total_mass * J_com, atol=1e-12)

            # Verify h_G = A_G @ v equals physical angular + linear momentum about CoM
            h_G = A_G @ v
            assert np.allclose(h_G[3:6], robot_fb.total_mass * (J_com @ v), atol=1e-12)

            # Also check fixed-base com_jacobian and A_G[3:6]
            J_com_fix = robot_fixed.com_jacobian(q_j)
            A_G_fix = robot_fixed.centroidal_momentum_matrix(q_j)
            assert J_com_fix.shape == (3, 4)
            assert A_G_fix.shape == (6, 4)
            assert np.allclose(A_G_fix[3:6, :], robot_fixed.total_mass * J_com_fix, atol=1e-12)
            J_fix_fd = _fd_jacobian(robot_fixed.center_of_mass, q_j, eps=eps)
            assert np.max(np.abs(J_com_fix - J_fix_fd)) < 1e-7

    def test_floating_base_rnea_crba_and_centroidal_momentum_conservation(self):
        robot = URDFRobotModel.from_urdf_string(FLOATING_BIPED_URDF, floating_base=True)
        rng = np.random.RandomState(123)
        q = np.concatenate(
            [
                np.array([0.1, -0.05, 0.85]),
                np.array([0.12, -0.18, 0.25]),
                np.array([0.3, -0.6, -0.25, -0.4]),
            ]
        )
        v = rng.uniform(-0.8, 0.8, size=robot.n_v)
        qddot = rng.uniform(-1.0, 1.0, size=robot.n_v)
        gravity = (0.0, 0.0, -9.81)

        M = robot.mass_matrix_crba(q)
        assert M.shape == (robot.n_v, robot.n_v)
        assert np.allclose(M, M.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 1e-6)

        # CRBA + RNEA consistency: M qddot + C(q, v) v + g(q) == RNEA(q, v, qddot)
        tau_rnea = robot.inverse_dynamics_rnea(q, v, qddot, gravity=gravity)
        tau_bias = robot.inverse_dynamics_rnea(q, v, np.zeros(robot.n_v), gravity=gravity)
        assert np.allclose(tau_rnea, M @ qddot + tau_bias, atol=1e-10)

        # Free-floating zero-gravity, zero-torque momentum conservation: h_G(t) = A_G(q) v = const
        dt = 0.0005
        q_cur = q.copy()
        v_cur = v.copy()
        h0 = robot.centroidal_momentum_matrix(q_cur) @ v_cur

        for _ in range(120):
            # RK2 midpoint integration of M(q) dv/dt + C(q, v) v = 0
            def _accel(q_s: np.ndarray, v_s: np.ndarray) -> np.ndarray:
                M_s = robot.mass_matrix_crba(q_s)
                c_s = robot.coriolis_torques(q_s, v_s)
                return np.linalg.solve(M_s, -c_s)

            a1 = _accel(q_cur, v_cur)
            q_mid = robot.integrate_configuration(q_cur, v_cur, 0.5 * dt)
            v_mid = v_cur + 0.5 * dt * a1
            a2 = _accel(q_mid, v_mid)
            q_cur = robot.integrate_configuration(q_cur, v_mid, dt)
            v_cur = v_cur + dt * a2

        h_final = robot.centroidal_momentum_matrix(q_cur) @ v_cur
        assert np.max(np.abs(h_final - h0)) < 1e-6


# ---------------------------------------------------------------------------
# 2. CapturePointDCMBarrier & SupportPolygonBalanceCBF Tests
# ---------------------------------------------------------------------------


class TestSupportPolygonBalanceCBF:
    def test_capture_point_barrier_gradients_hessian_and_inflation(self):
        feet = np.array(
            [
                [0.25, 0.15, 0.0],
                [-0.25, 0.15, 0.0],
                [-0.25, -0.15, 0.0],
                [0.25, -0.15, 0.0],
            ]
        )
        A, b = support_polygon_from_feet(feet)
        assert A.shape == (4, 2)
        assert np.allclose(np.linalg.norm(A, axis=1), 1.0, atol=1e-12)

        # Single-edge barrier with dynamic z_com=None on 6D centroidal state [p(3), v(3)]
        bar_dyn = CapturePointDCMBarrier(A[0], b[0], z_com=None, gravity=9.81, alpha=2.5)
        x6 = np.array([0.05, -0.02, 0.78, 0.35, -0.25, 0.10])
        grad_an = bar_dyn.grad_h(x6)
        grad_fd = _fd_jacobian(lambda s: np.array([bar_dyn.h(s)]), x6, eps=1e-6).flatten()
        assert np.max(np.abs(grad_an - grad_fd)) < 1e-7

        hess_an = bar_dyn.hess_h(x6)
        hess_fd = _fd_jacobian(bar_dyn.grad_h, x6, eps=1e-5)
        assert np.max(np.abs(hess_an - 0.5 * (hess_fd + hess_fd.T))) < 1e-6

        # Multi-edge barrier + inflation
        bar_multi = CapturePointDCMBarrier.from_feet(
            feet[:2], foot_half_size=(0.08, 0.04), z_com=0.8
        )
        x_in = np.array([0.0, 0.15, 0.0, 0.0])
        h_orig = bar_multi.h(x_in)
        assert h_orig > 0.0
        bar_inf = bar_multi.inflate(0.02)
        assert bar_inf.h(x_in) == pytest.approx(h_orig - 0.02, abs=1e-12)

    def test_support_polygon_cbf_rejects_unsafe_push_and_zmp_commands(self):
        # Quadruped stance rectangle: [-0.22, 0.22] x [-0.14, 0.14]
        feet = np.array(
            [
                [0.22, 0.14, 0.0],
                [-0.22, 0.14, 0.0],
                [-0.22, -0.14, 0.0],
                [0.22, -0.14, 0.0],
            ]
        )
        cbf = SupportPolygonBalanceCBF.from_feet(
            feet,
            z_com=0.75,
            alpha=3.0,
            state_mode="lipm_2d",
            enforce_zmp_in_polygon=True,
        )

        dt = 0.005
        n_steps = 200
        x0 = np.array([0.02, 0.01, 0.50, 0.30], dtype=np.float64)
        # Adversarial nominal ZMP command placed far outside the opposite side,
        # which accelerates DCM out of the front-right corner
        u_unsafe = np.array([-0.60, -0.50], dtype=np.float64)

        # Unfiltered rollout exits the support polygon
        x_unfilt = x0.copy()
        min_margin_unfilt = float("inf")
        for step in range(n_steps):
            push = np.array([0.0, 0.0, 0.35, 0.20]) if 10 <= step < 30 else None
            xi_u = cbf.capture_point(x_unfilt)
            min_margin_unfilt = min(min_margin_unfilt, cbf.polygon_margin(xi_u))
            x_unfilt = cbf.step_dynamics(x_unfilt, u_unsafe, dt, perturbation=push)
        assert min_margin_unfilt < -0.1

        # CBF-filtered rollout stays strictly inside the stance foot polygon
        x = x0.copy()
        min_margin_filt = float("inf")
        redirected = False
        for step in range(n_steps):
            push = np.array([0.0, 0.0, 0.35, 0.20]) if 10 <= step < 30 else None
            out = cbf.filter(x, u_unsafe)
            assert out["verified_safe"] is True, out
            u_safe = np.asarray(out["u_filtered"], dtype=np.float64)
            assert cbf.is_inside_polygon(u_safe, tol=1e-6)
            if np.linalg.norm(u_safe - u_unsafe) > 1e-3:
                redirected = True
            xi = cbf.capture_point(x)
            min_margin_filt = min(min_margin_filt, cbf.polygon_margin(xi))
            x = cbf.step_dynamics(x, u_safe, dt, perturbation=push)

        xi_final = cbf.capture_point(x)
        min_margin_filt = min(min_margin_filt, cbf.polygon_margin(xi_final))
        assert redirected is True
        assert min_margin_filt >= -1e-4

        # Conformal margin shrinks allowable DCM safe region
        cbf.set_conformal_margin(0.03)
        xi_near_edge = np.array([0.20, 0.0, 0.0, 0.0])  # +x margin = 0.02 < 0.03
        assert min(b.inflate(0.03).h(xi_near_edge) for b in cbf.barriers) < 0.0


# ---------------------------------------------------------------------------
# 3. GRFFrictionPyramidAllocator Tests (Biped & Quadruped)
# ---------------------------------------------------------------------------


class TestGRFFrictionPyramidAllocator:
    def _assert_cone_satisfied(
        self, forces: np.ndarray, active_mask: np.ndarray, mu: float, f_min: float, f_max: float
    ) -> None:
        for i, active in enumerate(active_mask):
            fx, fy, fz = forces[i]
            if not active:
                assert np.allclose(forces[i], 0.0, atol=1e-12)
                continue
            assert fz >= f_min - 1e-5
            assert fz <= f_max + 1e-5
            assert abs(fx) <= mu * fz + 1e-5
            assert abs(fy) <= mu * fz + 1e-5

    def test_biped_2_feet_under_lateral_push_and_cop_bounds(self):
        mu = 0.5
        f_min = 10.0
        f_max = 450.0
        cop_bounds = (-0.08, 0.08, -0.04, 0.04)
        allocator = GRFFrictionPyramidAllocator(
            mu=mu, f_min=f_min, f_max=f_max, cop_bounds=cop_bounds
        )

        feet = np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]])
        com = np.array([0.0, 0.0, 0.80])
        mg = 250.0

        # Achievable lateral push perturbation within friction cone and ZMP/CoP polygon
        f_des = np.array([35.0, -30.0, mg])
        tau_des = np.array([-4.0, -16.0, 2.0])
        res = allocator.allocate(feet, com, f_des, tau_des)

        assert res["feasible"] is True
        assert res["max_violation"] <= 1e-5
        self._assert_cone_satisfied(res["forces"], np.array([True, True]), mu, f_min, f_max)
        assert res["wrench_error"] < 1e-2

        # Check rectangular CoP bounds on each foot
        for c in range(2):
            fz_c = res["forces"][c, 2]
            tx_c, ty_c = res["local_torques"][c]
            cop_x = -ty_c / fz_c
            cop_y = tx_c / fz_c
            assert -0.08 - 1e-5 <= cop_x <= 0.08 + 1e-5
            assert -0.04 - 1e-5 <= cop_y <= 0.04 + 1e-5

        # Excessive lateral push exceeding mu * mg: wrench cannot be matched exactly,
        # but solver MUST strictly enforce friction pyramid and unilateral normal force
        res_extreme = allocator.allocate(feet, com, [0.0, 300.0, mg], [0.0, 0.0, 0.0])
        assert res_extreme["feasible"] is True
        self._assert_cone_satisfied(
            res_extreme["forces"], np.array([True, True]), mu, f_min, f_max
        )

    def test_quadruped_4_feet_and_three_legged_stance_lateral_push(self):
        mu = 0.6
        f_min = 5.0
        f_max = 300.0
        allocator = GRFFrictionPyramidAllocator(mu=mu, f_min=f_min, f_max=f_max)

        feet = np.array(
            [
                [0.25, 0.16, 0.0],
                [0.25, -0.16, 0.0],
                [-0.25, 0.16, 0.0],
                [-0.25, -0.16, 0.0],
            ]
        )
        com = np.array([0.02, -0.01, 0.38])
        f_des = np.array([40.0, -70.0, 220.0])
        tau_des = np.array([4.0, -8.0, 6.0])

        res4 = allocator.allocate(feet, com, f_des, tau_des)
        assert res4["feasible"] is True
        self._assert_cone_satisfied(res4["forces"], np.ones(4, dtype=bool), mu, f_min, f_max)
        assert np.allclose(res4["net_force"], f_des, atol=1e-2)
        assert np.allclose(res4["net_torque"], tau_des, atol=1e-2)

        # 3-legged trot/swing configuration (foot 2 lifted) under lateral push
        mask3 = np.array([True, True, False, True])
        com3 = np.array([0.05, -0.04, 0.38])
        f_des3 = np.array([15.0, -20.0, 220.0])
        tau_des3 = np.array([-1.5, -2.0, 0.5])
        res3 = allocator.allocate(feet, com3, f_des3, tau_des3, active_mask=mask3)
        assert res3["feasible"] is True
        self._assert_cone_satisfied(res3["forces"], mask3, mu, f_min, f_max)
        assert np.allclose(res3["net_force"], f_des3, atol=1e-2)
        assert np.allclose(res3["net_torque"], tau_des3, atol=1e-2)


# ---------------------------------------------------------------------------
# 4. SwingFootClearanceBarrier HOCBF Tests
# ---------------------------------------------------------------------------


class TestSwingFootClearanceBarrier:
    def test_swing_foot_clearance_grad_hess_and_terrain_step_rollout(self):
        robot = URDFRobotModel.from_urdf_string(SWING_LEG_3DOF_URDF)
        dyn = robot.to_euler_lagrange_dynamics()

        # Smooth terrain step rising from z=0.0 (x < 0.06) to z=0.14 (x > 0.16)
        def terrain_z(x_w: float, _y_w: float) -> float:
            return 0.07 * (1.0 + math.tanh(25.0 * (x_w - 0.11)))

        def terrain_grad(x_w: float, _y_w: float) -> tuple[float, float]:
            s = 1.0 / math.cosh(25.0 * (x_w - 0.11))
            return 0.07 * 25.0 * s * s, 0.0

        c_min = 0.04
        foot_bar = SwingFootClearanceBarrier(
            robot=robot,
            foot_link="foot",
            local_point=(0.0, 0.0, -0.02),
            terrain_height_fn=terrain_z,
            terrain_grad_fn=terrain_grad,
            c_min=c_min,
            alpha=8.0,
            alpha2=10.0,
            relative_degree=2,
            hard=True,
        )

        # Verify analytical gradient and subspace Hessian against central FD
        x_test = np.array([0.4, -0.7, 0.3, 0.1, -0.2, 0.05], dtype=np.float64)
        grad_an = foot_bar.grad_h(x_test)
        grad_fd = _fd_jacobian(lambda s: np.array([foot_bar.h(s)]), x_test, eps=1e-6).flatten()
        assert np.allclose(grad_an[3:], 0.0, atol=1e-15)
        assert np.max(np.abs(grad_an - grad_fd)) < 1e-6

        hess_an = foot_bar.hess_h(x_test)
        hess_fd = _fd_jacobian(foot_bar.grad_h, x_test, eps=1e-5)
        assert np.max(np.abs(hess_an - 0.5 * (hess_fd + hess_fd.T))) < 1e-5

        # Verify inflate shifts c_min
        assert foot_bar.inflate(0.03).h(x_test) == pytest.approx(foot_bar.h(x_test) - 0.03, abs=1e-12)

        # Closed-loop swing rollout over the terrain step
        filt = StateCBFSafetyFilter(
            dyn,
            [foot_bar],
            u_min=dyn.u_min,
            u_max=dyn.u_max,
        )

        # Initial bent-leg configuration safely above terrain, swinging forward & downward onto the step
        x0 = np.array([-0.60, 0.95, -0.35, 0.0, 0.0, 0.0], dtype=np.float64)
        q_unsafe_goal = np.array([-0.45, 0.10, 0.0], dtype=np.float64)
        assert foot_bar.h(x0) > 0.05
        assert foot_bar.h(np.concatenate([q_unsafe_goal, np.zeros(3)])) < -0.05

        dt = 0.005
        n_steps = 120

        # Unfiltered PD controller drives the swing foot into the terrain step
        x_u = x0.copy()
        min_h_unfilt = float("inf")
        for _ in range(n_steps):
            q_u, dq_u = x_u[:3], x_u[3:]
            u_nom = np.clip(45.0 * (q_unsafe_goal - q_u) - 8.0 * dq_u + dyn.grav(q_u), dyn.u_min, dyn.u_max)
            min_h_unfilt = min(min_h_unfilt, foot_bar.h(x_u))
            x_u = x_u + dt * (dyn.f(x_u) + dyn.g(x_u) @ u_nom)
        assert min_h_unfilt < 0.0

        # Filtered controller keeps swing foot strictly above terrain + c_min
        x_f = x0.copy()
        min_h_filt = float("inf")
        intervened = False
        for _ in range(n_steps):
            q_cur, dq_cur = x_f[:3], x_f[3:]
            u_nom = 45.0 * (q_unsafe_goal - q_cur) - 8.0 * dq_cur + dyn.grav(q_cur)
            u_nom_clipped = np.clip(u_nom, dyn.u_min, dyn.u_max)
            out = filt.filter(x_f, u_nom)
            assert out["verified_safe"] is True, out
            u_safe = np.asarray(out["u_filtered"], dtype=np.float64)
            if np.linalg.norm(u_safe - u_nom_clipped) > 1e-3:
                intervened = True
            min_h_filt = min(min_h_filt, foot_bar.h(x_f))
            x_f = x_f + dt * (dyn.f(x_f) + dyn.g(x_f) @ u_safe)

        min_h_filt = min(min_h_filt, foot_bar.h(x_f))
        assert intervened is True
        assert min_h_filt >= -1e-4
