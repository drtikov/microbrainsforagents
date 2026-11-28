"""Tests for pure-NumPy SE(3) Lie algebra, Featherstone URDF kinematics/dynamics,
analytical Envelope-Theorem (Danskin) capsule gradients, and CBF safety-filter rollouts.
"""

from __future__ import annotations

import sys
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

from aetheris_microbrains.core.state_cbf_filter import HighOrderBarrier, StateCBFSafetyFilter
from aetheris_microbrains.core.urdf_spatial_kinematics import (
    URDFRobotModel,
    adjoint_se3,
    exp_se3,
    exp_so3,
    hat_so3,
    spatial_cross_force,
    spatial_cross_motion,
    vee_so3,
)
from aetheris_microbrains.core.whole_body_barriers import (
    CapsulePairBarrier,
    _fd_grad_subspace,
    _segment_segment_distance_and_witnesses,
    _segment_segment_distance_sq,
)


# ---------------------------------------------------------------------------
# URDF XML Fixtures (3-DoF, 6-DoF, N-DoF chains, and multi-primitive checks)
# ---------------------------------------------------------------------------

URDF_3DOF_SPATIAL_XML = """<?xml version="1.0"?>
<robot name="spatial_3dof_arm">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="4.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.05"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.10"/>
      </geometry>
    </collision>
  </link>

  <joint name="shoulder_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.12" rpy="0 0 0.15"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" effort="60.0" velocity="3.0"/>
  </joint>

  <link name="link1">
    <inertial>
      <origin xyz="0.02 0 0.15" rpy="0.1 0 0"/>
      <mass value="2.2"/>
      <inertia ixx="0.025" ixy="0.001" ixz="0.0" iyy="0.024" iyz="0.0" izz="0.006"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.28"/>
      </geometry>
    </collision>
  </link>

  <joint name="shoulder_pitch" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.30" rpy="0.05 -0.1 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="2.4" effort="45.0" velocity="3.0"/>
  </joint>

  <link name="link2">
    <inertial>
      <origin xyz="0 0.01 0.16" rpy="0 0.1 0"/>
      <mass value="1.6"/>
      <inertia ixx="0.018" ixy="0.0" ixz="0.001" iyy="0.017" iyz="0.0" izz="0.004"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.16" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.30"/>
      </geometry>
    </collision>
  </link>

  <joint name="elbow_pitch" type="revolute">
    <parent link="link2"/>
    <child link="link3"/>
    <origin xyz="0 0 0.32" rpy="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="30.0" velocity="4.0"/>
  </joint>

  <link name="link3">
    <inertial>
      <origin xyz="0 0 0.14" rpy="0 0 0"/>
      <mass value="1.1"/>
      <inertia ixx="0.010" ixy="0.0" ixz="0.0" iyy="0.010" iyz="0.0" izz="0.002"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.14" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.26"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def _build_6dof_spatial_urdf_xml() -> str:
    return """<?xml version="1.0"?>
<robot name="spatial_6dof_manipulator">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.06" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.06"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.06" rpy="0 0 0"/>
      <geometry><cylinder radius="0.07" length="0.12"/></geometry>
    </collision>
  </link>

  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.14" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="80.0" velocity="2.5"/>
  </joint>
  <link name="l1">
    <inertial>
      <origin xyz="0.01 0 0.12" rpy="0.1 0.05 0"/>
      <mass value="3.2"/>
      <inertia ixx="0.03" ixy="0.001" ixz="0" iyy="0.03" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.12" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.22"/></geometry>
    </collision>
  </link>

  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0.04 0.24" rpy="0.1 -0.05 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" effort="75.0" velocity="2.5"/>
  </joint>
  <link name="l2">
    <inertial>
      <origin xyz="0 0 0.18" rpy="0 0 0.1"/>
      <mass value="2.6"/>
      <inertia ixx="0.035" ixy="0" ixz="0.001" iyy="0.034" iyz="0" izz="0.008"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.18" rpy="0 0 0"/>
      <geometry><cylinder radius="0.045" length="0.34"/></geometry>
    </collision>
  </link>

  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 -0.03 0.36" rpy="0 0.08 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="55.0" velocity="3.0"/>
  </joint>
  <link name="l3">
    <inertial>
      <origin xyz="0.01 0 0.15" rpy="0 0 0"/>
      <mass value="1.9"/>
      <inertia ixx="0.020" ixy="0" ixz="0" iyy="0.021" iyz="0.0005" izz="0.005"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry><cylinder radius="0.04" length="0.28"/></geometry>
    </collision>
  </link>

  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.30" rpy="0.05 0 0.1"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="30.0" velocity="4.0"/>
  </joint>
  <link name="l4">
    <inertial>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.003"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry><cylinder radius="0.035" length="0.14"/></geometry>
    </collision>
  </link>

  <joint name="j5" type="revolute">
    <parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.16" rpy="0 -0.1 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.2" upper="2.2" effort="25.0" velocity="4.0"/>
  </joint>
  <link name="l5">
    <inertial>
      <origin xyz="0 0 0.07" rpy="0 0 0"/>
      <mass value="0.9"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.002"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.07" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.12"/></geometry>
    </collision>
  </link>

  <joint name="j6" type="revolute">
    <parent link="l5"/><child link="l6"/>
    <origin xyz="0 0 0.14" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="18.0" velocity="5.0"/>
  </joint>
  <link name="l6">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="0.6"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.0015"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><cylinder radius="0.025" length="0.10"/></geometry>
    </collision>
  </link>

  <joint name="ee_fixed" type="fixed">
    <parent link="l6"/><child link="tool0"/>
    <origin xyz="0.02 -0.01 0.11" rpy="0.1 0.2 -0.15"/>
  </joint>
  <link name="tool0">
    <inertial>
      <origin xyz="0 0 0.02" rpy="0 0 0"/>
      <mass value="0.25"/>
      <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0004"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.02" rpy="0 0 0"/>
      <geometry><sphere radius="0.03"/></geometry>
    </collision>
  </link>
</robot>
"""


def _build_ndof_chain_urdf_xml(n_dof: int) -> str:
    axes = ["0 0 1", "0 1 0", "1 0 0"]
    parts = [
        '<?xml version="1.0"?>',
        f'<robot name="chain_{n_dof}dof">',
        '  <link name="l0">',
        '    <collision><origin xyz="0 0 0.04"/><geometry><cylinder radius="0.03" length="0.08"/></geometry></collision>',
        '  </link>',
    ]
    for i in range(1, n_dof + 1):
        ax = axes[(i - 1) % 3]
        parts.append(
            f'  <joint name="j{i}" type="revolute">'
            f'<parent link="l{i - 1}"/><child link="l{i}"/>'
            f'<origin xyz="0.01 0 0.10" rpy="0.05 -0.03 0.02"/>'
            f'<axis xyz="{ax}"/>'
            f'<limit lower="-3.14" upper="3.14" effort="40.0" velocity="3.0"/>'
            f'</joint>'
        )
        parts.append(
            f'  <link name="l{i}">'
            f'<inertial><origin xyz="0 0 0.05"/><mass value="0.5"/>'
            f'<inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.001"/></inertial>'
            f'<collision><origin xyz="0 0 0.05"/><geometry><cylinder radius="0.02" length="0.09"/></geometry></collision>'
            f'</link>'
        )
    parts.append("</robot>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 1. SE(3) & Spatial Algebra Tests
# ---------------------------------------------------------------------------


class TestSE3AndSpatialAlgebra:
    def test_so3_se3_exponentials_and_adjoint_homomorphism(self):
        rng = np.random.RandomState(42)
        omega = rng.randn(3) * 0.7
        p = rng.randn(3) * 0.4
        assert np.allclose(hat_so3(omega) @ p, np.cross(omega, p), atol=1e-14)
        assert np.allclose(vee_so3(hat_so3(omega)), omega, atol=1e-14)

        R = exp_so3(omega)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)
        # Near-zero Taylor branch continuity
        R_tiny = exp_so3([1e-11, -2e-11, 3e-11])
        assert np.allclose(R_tiny, np.eye(3) + hat_so3([1e-11, -2e-11, 3e-11]), atol=1e-20)

        xi1 = rng.randn(6) * 0.5
        xi2 = rng.randn(6) * 0.5
        T1 = exp_se3(xi1)
        T2 = exp_se3(xi2)
        # Ad_{T1 T2} == Ad_{T1} Ad_{T2}
        assert np.allclose(adjoint_se3(T1 @ T2), adjoint_se3(T1) @ adjoint_se3(T2), atol=1e-13)

        # Conjugation identity: T1 * exp(eps * xi2^) * T1^{-1} == exp(eps * (Ad_{T1} xi2)^)
        eps = 1e-5
        lhs = T1 @ exp_se3(xi2, theta=eps) @ np.linalg.inv(T1)
        rhs = exp_se3(adjoint_se3(T1) @ xi2, theta=eps)
        assert np.allclose(lhs, rhs, atol=1e-12)

    def test_spatial_cross_motion_and_force_duality(self):
        rng = np.random.RandomState(17)
        v1 = rng.randn(6)
        v2 = rng.randn(6)
        f = rng.randn(6)

        crm = spatial_cross_motion(v1)
        crf = spatial_cross_force(v1)
        assert crm.shape == (6, 6) and crf.shape == (6, 6)
        assert np.allclose(crf, -crm.T, atol=1e-15)
        assert np.allclose(crm @ v2, spatial_cross_motion(v1, v2), atol=1e-14)
        assert np.allclose(crf @ f, spatial_cross_force(v1, f), atol=1e-14)
        # Power duality: <f, v1 x v2> + <v1 x* f, v2> == 0
        power_sum = float(f @ spatial_cross_motion(v1, v2) + spatial_cross_force(v1, f) @ v2)
        assert abs(power_sum) < 1e-13


# ---------------------------------------------------------------------------
# 2. URDF Parsing Security, Primitives, FK & Analytical Jacobians
# ---------------------------------------------------------------------------


class TestURDFKinematicsAndJacobians:
    def test_urdf_safe_xml_rejects_entity_expansion_and_loads_file(self, tmp_path):
        malicious_xml = """<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;">
]>
<robot name="bad"><link name="base"/></robot>
"""
        with pytest.raises(ValueError, match="forbidden"):
            URDFRobotModel.from_urdf_string(malicious_xml)

        urdf_file = tmp_path / "arm3.urdf"
        urdf_file.write_text(URDF_3DOF_SPATIAL_XML, encoding="utf-8")
        robot = URDFRobotModel.from_urdf_file(urdf_file)
        assert robot.n_dof == 3
        assert robot.u_max.tolist() == pytest.approx([60.0, 45.0, 30.0])
        assert robot.u_min.tolist() == pytest.approx([-60.0, -45.0, -30.0])

    def test_box_sphere_cylinder_capsule_fitting_and_prismatic_joint(self):
        xml = """<?xml version="1.0"?>
<robot name="primitives_bot">
  <link name="base">
    <collision>
      <origin xyz="0.1 -0.2 0.3" rpy="0 0 0"/>
      <geometry><box size="0.2 0.4 1.0"/></geometry>
    </collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="slider"/>
    <origin xyz="0 0 0.5" rpy="0.2 -0.3 0.1"/>
    <axis xyz="0.6 0.0 0.8"/>
    <limit lower="-0.5" upper="0.8" effort="120.0" velocity="1.5"/>
  </joint>
  <link name="slider">
    <inertial>
      <origin xyz="0 0 0"/><mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.1"/><geometry><sphere radius="0.08"/></geometry>
    </collision>
  </link>
</robot>
"""
        bot = URDFRobotModel.from_urdf_string(xml)
        box_cap = bot.collision_capsules["base"][0]
        assert box_cap.radius == pytest.approx(0.5 * np.hypot(0.2, 0.4))
        assert np.linalg.norm(box_cap.local_p1 - box_cap.local_p0) == pytest.approx(1.0)
        sph_cap = bot.collision_capsules["slider"][0]
        assert np.allclose(sph_cap.local_p0, sph_cap.local_p1)
        assert sph_cap.radius == pytest.approx(0.08)

        # Verify analytical Jacobian of prismatic joint with rotated origin & non-axis-aligned axis
        q0 = np.array([0.35])
        pt_local = np.array([0.12, -0.07, 0.2])
        p_w, J_p = bot.link_point_jacobian(q0, "slider", pt_local)
        eps = 1e-7
        p_plus, _ = bot.link_point_jacobian(q0 + eps, "slider", pt_local)
        p_minus, _ = bot.link_point_jacobian(q0 - eps, "slider", pt_local)
        assert np.allclose(J_p[:, 0], (p_plus - p_minus) / (2.0 * eps), atol=1e-9)
        J_ee = bot.end_effector_jacobian(q0)
        assert np.allclose(J_ee[:3, 0], 0.0, atol=1e-15)

    @pytest.mark.parametrize("fixture_kind", ["3dof", "6dof"])
    def test_analytical_jacobians_match_central_finite_differences(self, fixture_kind):
        xml = URDF_3DOF_SPATIAL_XML if fixture_kind == "3dof" else _build_6dof_spatial_urdf_xml()
        robot = URDFRobotModel.from_urdf_string(xml)
        rng = np.random.RandomState(101 if fixture_kind == "3dof" else 202)
        eps = 1e-7

        for _ in range(5):
            q = rng.uniform(-1.5, 1.5, size=robot.n_dof)
            local_pt = rng.uniform(-0.15, 0.25, size=3)

            for lname in robot.link_names:
                p_w, J_analytic = robot.link_point_jacobian(q, lname, local_pt)
                fk0 = robot.forward_kinematics(q)[lname]
                assert np.allclose(p_w, fk0[:3, :3] @ local_pt + fk0[:3, 3], atol=1e-13)

                J_fd = np.zeros((3, robot.n_dof), dtype=np.float64)
                for j in range(robot.n_dof):
                    dq = np.zeros(robot.n_dof, dtype=np.float64)
                    dq[j] = eps
                    T_p = robot.forward_kinematics(q + dq)[lname]
                    T_m = robot.forward_kinematics(q - dq)[lname]
                    pt_p = T_p[:3, :3] @ local_pt + T_p[:3, 3]
                    pt_m = T_m[:3, :3] @ local_pt + T_m[:3, 3]
                    J_fd[:, j] = (pt_p - pt_m) / (2.0 * eps)

                assert np.max(np.abs(J_analytic - J_fd)) < 1e-6

            # End-effector 6xN geometric Jacobian [J_angular; J_linear]
            ee_name = robot.end_effector_link_name
            J_ee = robot.end_effector_jacobian(q, ee_name)
            T0 = robot.forward_kinematics(q)[ee_name]
            R0 = T0[:3, :3]
            J_ee_fd = np.zeros((6, robot.n_dof), dtype=np.float64)
            for j in range(robot.n_dof):
                dq = np.zeros(robot.n_dof, dtype=np.float64)
                dq[j] = eps
                T_p = robot.forward_kinematics(q + dq)[ee_name]
                T_m = robot.forward_kinematics(q - dq)[ee_name]
                dR = (T_p[:3, :3] - T_m[:3, :3]) / (2.0 * eps)
                J_ee_fd[:3, j] = vee_so3(dR @ R0.T)
                J_ee_fd[3:6, j] = (T_p[:3, 3] - T_m[:3, 3]) / (2.0 * eps)

            assert np.max(np.abs(J_ee - J_ee_fd)) < 1e-6


# ---------------------------------------------------------------------------
# 3. CRBA SPD Mass Matrix & RNEA Power Conservation
# ---------------------------------------------------------------------------


class TestCRBAAndRNEADynamics:
    @pytest.mark.parametrize("fixture_kind", ["3dof", "6dof"])
    def test_crba_spd_rnea_consistency_and_power_conservation(self, fixture_kind):
        xml = URDF_3DOF_SPATIAL_XML if fixture_kind == "3dof" else _build_6dof_spatial_urdf_xml()
        robot = URDFRobotModel.from_urdf_string(xml)
        rng = np.random.RandomState(303 if fixture_kind == "3dof" else 404)
        gravity = (0.15, -0.25, -9.81)

        for _ in range(5):
            q = rng.uniform(-1.4, 1.4, size=robot.n_dof)
            qdot = rng.uniform(-1.2, 1.2, size=robot.n_dof)
            qddot = rng.uniform(-2.0, 2.0, size=robot.n_dof)

            M = robot.mass_matrix_crba(q)
            assert M.shape == (robot.n_dof, robot.n_dof)
            assert np.allclose(M, M.T, atol=1e-14)
            eigvals = np.linalg.eigvalsh(M)
            assert float(np.min(eigvals)) > 1e-5  # strictly positive-definite

            # RNEA decomposition: rnea(q, qdot, qddot, g) == M(q) qddot + rnea(q, qdot, 0, g)
            tau_full = robot.inverse_dynamics_rnea(q, qdot, qddot, gravity=gravity)
            tau_bias = robot.inverse_dynamics_rnea(q, qdot, np.zeros(robot.n_dof), gravity=gravity)
            assert np.allclose(tau_full, M @ qddot + tau_bias, atol=1e-12)

            # Power conservation: qdot^T (M qddot + C qdot + g) == d/dt (0.5 qdot^T M qdot + V(q))
            dt_fd = 1e-7
            q_plus = q + dt_fd * qdot + 0.5 * (dt_fd * dt_fd) * qddot
            qdot_plus = qdot + dt_fd * qddot
            q_minus = q - dt_fd * qdot + 0.5 * (dt_fd * dt_fd) * qddot
            qdot_minus = qdot - dt_fd * qddot

            E_plus = robot.kinetic_energy(q_plus, qdot_plus) + robot.potential_energy(q_plus, gravity)
            E_minus = robot.kinetic_energy(q_minus, qdot_minus) + robot.potential_energy(q_minus, gravity)
            dE_dt = (E_plus - E_minus) / (2.0 * dt_fd)
            mech_power = float(np.dot(qdot, tau_full))
            assert mech_power == pytest.approx(dE_dt, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Envelope-Theorem Capsule Gradients, Speedup & Closed-Loop CBF Rollout
# ---------------------------------------------------------------------------


class TestAnalyticalCapsuleBarriersAndRollout:
    def test_witness_segment_distance_and_danskin_gradient_accuracy(self):
        robot6 = URDFRobotModel.from_urdf_string(_build_6dof_spatial_urdf_xml())
        barriers = robot6.build_self_collision_barriers(min_link_separation=2, margin=0.02)
        assert len(barriers) > 5

        rng = np.random.RandomState(505)
        for _ in range(8):
            q = rng.uniform(-1.3, 1.3, size=robot6.n_dof)
            dq = rng.uniform(-0.8, 0.8, size=robot6.n_dof)
            x = np.concatenate([q, dq])

            for bar in barriers:
                p0, p1 = bar.link_fk_fn(q)
                q0, q1 = bar.opponent_segment(q)
                d2, s_star, t_star, p_star, q_star = _segment_segment_distance_and_witnesses(
                    p0, p1, q0, q1
                )
                assert 0.0 <= s_star <= 1.0
                assert 0.0 <= t_star <= 1.0
                assert np.allclose(p_star, p0 + s_star * (p1 - p0), atol=1e-14)
                assert np.allclose(q_star, q0 + t_star * (q1 - q0), atol=1e-14)
                assert d2 == pytest.approx(_segment_segment_distance_sq(p0, p1, q0, q1), abs=1e-14)

                # Skip exact-parallel degenerate segments if any occur
                d1_v = p1 - p0
                d2_v = q1 - q0
                denom = float(np.dot(d1_v, d1_v) * np.dot(d2_v, d2_v) - np.dot(d1_v, d2_v) ** 2)
                if np.dot(d1_v, d1_v) > 1e-12 and np.dot(d2_v, d2_v) > 1e-12 and denom <= 1e-10:
                    continue

                grad_analytic = bar.grad_h(x)
                grad_fd = _fd_grad_subspace(bar.h, x, robot6.n_dof, eps=1e-6)
                assert grad_analytic.shape == (2 * robot6.n_dof,)
                assert np.allclose(grad_analytic[robot6.n_dof :], 0.0, atol=1e-15)
                assert np.max(np.abs(grad_analytic - grad_fd)) < 1e-5

        # Verify inflate() forwards link_jacobians_fn, opponent_jacobians_fn, broadphase_cutoff
        bar_cut = robot6.build_self_collision_barriers(broadphase_cutoff=0.25)[0]
        inf = bar_cut.inflate(0.05)
        assert inf.link_jacobians_fn is bar_cut.link_jacobians_fn
        assert inf.opponent_jacobians_fn is bar_cut.opponent_jacobians_fn
        assert inf.broadphase_cutoff == pytest.approx(0.25)
        assert inf.r_safe == pytest.approx(bar_cut.r_safe + 0.05)

        # Verify AABB broadphase cutoff skips Jacobian evaluation when separated beyond cutoff
        jac_calls = {"count": 0}

        def counted_jac(q):
            jac_calls["count"] += 1
            return (np.zeros((3, 1)), np.zeros((3, 1)))

        far_bar = CapsulePairBarrier(
            link_fk_fn=lambda q: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.2]),
            opponent_segment=([2.0, 0.0, 0.0], [2.0, 0.0, 0.2]),
            r_safe=0.1,
            n_dof=1,
            link_jacobians_fn=counted_jac,
            broadphase_cutoff=0.5,
        )
        assert far_bar.is_broadphase_culled(np.array([0.0, 0.0])) is True
        g_culled = far_bar.grad_h(np.array([0.0, 0.0]))
        assert np.allclose(g_culled, 0.0)
        assert jac_calls["count"] == 0

    def test_high_dof_analytical_capsule_gradient_dramatically_faster_than_fd(self):
        n_dof = 28
        robot_hi = URDFRobotModel.from_urdf_string(_build_ndof_chain_urdf_xml(n_dof))
        # Base vs tip capsule barrier
        bar_analytic = robot_hi.build_self_collision_barriers(min_link_separation=n_dof - 1)[0]
        bar_fd = CapsulePairBarrier(
            link_fk_fn=bar_analytic.link_fk_fn,
            opponent_segment=bar_analytic.opponent_segment,
            r_safe=bar_analytic.r_safe,
            n_dof=n_dof,
            eps=1e-6,
        )

        rng = np.random.RandomState(77)
        x = np.concatenate([rng.uniform(-0.4, 0.4, size=n_dof), np.zeros(n_dof)])

        # Accuracy parity on 28-DoF chain
        g_an = bar_analytic.grad_h(x)
        g_fd = bar_fd.grad_h(x)
        assert np.max(np.abs(g_an - g_fd)) < 1e-5

        # Warmup + timing comparison
        for _ in range(3):
            bar_analytic.grad_h(x)
            bar_fd.grad_h(x)

        n_iters = 25
        t0 = time.perf_counter()
        for _ in range(n_iters):
            bar_analytic.grad_h(x)
        dt_analytic = time.perf_counter() - t0

        t1 = time.perf_counter()
        for _ in range(n_iters):
            bar_fd.grad_h(x)
        dt_fd = time.perf_counter() - t1

        assert dt_analytic * 3.0 < dt_fd, (
            f"Expected >=3x speedup on {n_dof}-DoF chain, got analytic={dt_analytic:.4f}s vs fd={dt_fd:.4f}s"
        )

    def test_urdf_euler_lagrange_and_state_cbf_safety_filter_rollout(self):
        robot = URDFRobotModel.from_urdf_string(URDF_3DOF_SPATIAL_XML)
        dyn = robot.to_euler_lagrange_dynamics()
        assert dyn.n_dof == 3
        assert np.allclose(dyn.u_max, [60.0, 45.0, 30.0])
        assert np.allclose(dyn.u_min, [-60.0, -45.0, -30.0])

        # Analytical capsule barrier between forearm link3 and a vertical obstacle post in front of the arm,
        # composed alongside URDF whole-body self-collision barriers
        l3_cap = robot.collision_capsules["link3"][0]
        l3_idx = robot._link_name_to_idx["link3"]
        obstacle_post = ([0.36, 0.0, 0.20], [0.36, 0.0, 0.95])
        obs_bar = CapsulePairBarrier(
            link_fk_fn=lambda q: robot._link_segment_fk(q, l3_idx, l3_cap.local_p0, l3_cap.local_p1),
            opponent_segment=obstacle_post,
            r_safe=l3_cap.radius + 0.05,
            alpha=2.0,
            alpha2=3.0,
            relative_degree=2,
            n_dof=robot.n_dof,
            link_jacobians_fn=lambda q: robot._link_segment_jacobians(
                q, l3_idx, l3_cap.local_p0, l3_cap.local_p1
            ),
        )
        self_bars = robot.build_self_collision_barriers(
            min_link_separation=2,
            margin=0.02,
            relative_degree=2,
            alpha=2.0,
            alpha2=3.0,
            broadphase_cutoff=0.35,
        )
        hocbf_check = HighOrderBarrier(obs_bar, alpha1=2.0, alpha2=3.0)
        assert hocbf_check.relative_degree == 2

        filt = StateCBFSafetyFilter(
            dyn,
            [obs_bar] + self_bars,
            u_min=dyn.u_min,
            u_max=dyn.u_max,
        )

        x0 = np.array([0.0, 0.25, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        q_goal = np.array([0.0, 0.95, 0.55], dtype=np.float64)
        assert obs_bar.h(x0) > 0.02
        assert obs_bar.h(np.concatenate([q_goal, np.zeros(3)])) < 0.0

        dt = 0.005
        n_steps = 160

        # Unfiltered nominal PD + gravity-compensation rollout penetrates the obstacle post
        x_unfiltered = x0.copy()
        min_unfiltered_h = float("inf")
        for _ in range(n_steps):
            q_u, dq_u = x_unfiltered[:3], x_unfiltered[3:]
            u_nom_u = np.clip(10.0 * (q_goal - q_u) - 4.0 * dq_u + dyn.grav(q_u), dyn.u_min, dyn.u_max)
            min_unfiltered_h = min(min_unfiltered_h, obs_bar.h(x_unfiltered))
            x_unfiltered = x_unfiltered + dt * (dyn.f(x_unfiltered) + dyn.g(x_unfiltered) @ u_nom_u)
        assert min_unfiltered_h < 0.0

        # Filtered closed-loop rollout stays verified_safe and respects all barriers + effort bounds
        x = x0.copy()
        min_obs_h = float("inf")
        min_self_h = float("inf")
        intervened = False

        for _ in range(n_steps):
            q_cur, dq_cur = x[:3], x[3:]
            u_nom = 10.0 * (q_goal - q_cur) - 4.0 * dq_cur + dyn.grav(q_cur)
            u_nom_clipped = np.clip(u_nom, dyn.u_min, dyn.u_max)
            out = filt.filter(x, u_nom)
            assert out["verified_safe"] is True, out
            u_f = np.asarray(out["u_filtered"], dtype=np.float64)
            assert np.all(u_f >= dyn.u_min - 1e-6) and np.all(u_f <= dyn.u_max + 1e-6)
            if np.linalg.norm(u_f - u_nom_clipped) > 1e-3:
                intervened = True

            min_obs_h = min(min_obs_h, obs_bar.h(x))
            for sb in self_bars:
                min_self_h = min(min_self_h, sb.h(x))

            x = x + dt * (dyn.f(x) + dyn.g(x) @ u_f)

        assert intervened is True
        assert min_obs_h >= -1e-4
        assert min_self_h >= -1e-4
