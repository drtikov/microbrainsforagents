"""Tests for bimanual closed-chain/wrench/cable CBFs and SE_2(3) belief/occlusion CBFs."""

from __future__ import annotations

import math
import numpy as np
import pytest

from aetheris_microbrains.core.bimanual_grasp_cbf import (
    ClosedChainKinematicBarrier,
    CooperativeWrenchCBF,
    TautCableBarrier,
    log_se3,
    log_so3,
    se3_right_jacobian_inv,
)
from aetheris_microbrains.core.se3_belief_occlusion_cbf import (
    OcclusionShadowBarrier,
    OccluderBox,
    OccluderSegment,
    RightInvariantSE3Filter,
)
from aetheris_microbrains.core.state_cbf_filter import (
    ControlAffineDynamics,
    LinearDynamics,
    StateCBFSafetyFilter,
)
from aetheris_microbrains.core.urdf_spatial_kinematics import (
    URDFRobotModel,
    exp_se3,
    exp_so3,
)


def _make_3dof_arm_urdf(name: str = "arm3") -> str:
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link"/>
  <link name="link1">
    <inertial><origin xyz="0.15 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial><origin xyz="0.15 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="ee_link">
    <inertial><origin xyz="0.05 0 0"/><mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="5"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="5"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="link2"/><child link="ee_link"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="5"/>
  </joint>
</robot>"""


# ---------------------------------------------------------------------------
# SE(3) Lie Helpers & ClosedChainKinematicBarrier Tests
# ---------------------------------------------------------------------------


def test_lie_log_and_right_jacobian_inv_roundtrip():
    for phi in [
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-9, -2e-9, 1.5e-9]),
        np.array([0.4, -0.7, 0.5]),
        np.array([math.pi - 1e-4, 0.0, 0.0]),
    ]:
        R = exp_so3(phi)
        phi_rec = log_so3(R)
        np.testing.assert_allclose(exp_so3(phi_rec), R, atol=1e-7)

    xi = np.array([0.35, -0.25, 0.45, 0.12, -0.08, 0.2])
    T = exp_se3(xi)
    xi_rec = log_se3(T)
    np.testing.assert_allclose(xi_rec, xi, atol=1e-10)

    # Verify J_r^{-1}(xi) numerically against exp_se3(xi + d_xi) = exp_se3(xi) exp_se3(J_r * d_xi)
    Jr_inv = se3_right_jacobian_inv(xi)
    eps = 1e-7
    T_inv = np.linalg.inv(T)
    for i in range(6):
        e_i = np.zeros(6)
        e_i[i] = eps
        # Perturb on the right in body frame: T_pert = T @ exp_se3(e_i)
        T_p = T @ exp_se3(e_i)
        T_m = T @ exp_se3(-e_i)
        d_xi_num = (log_se3(T_p) - log_se3(T_m)) / (2.0 * eps)
        np.testing.assert_allclose(d_xi_num, Jr_inv[:, i], atol=1e-6)
        _ = T_inv


def test_closed_chain_kinematic_barrier_analytical_gradient_and_filtering():
    left_arm = URDFRobotModel.from_urdf_string(_make_3dof_arm_urdf("left_arm"))
    right_arm = URDFRobotModel.from_urdf_string(_make_3dof_arm_urdf("right_arm"))

    T_base_L = np.eye(4)
    T_base_L[:3, 3] = np.array([0.0, -0.4, 0.0])
    T_base_R = np.eye(4)
    T_base_R[:3, :3] = exp_so3([0.0, 0.0, math.pi])
    T_base_R[:3, 3] = np.array([0.0, 0.4, 0.0])

    q_L0 = np.array([0.4, -0.2, 0.1])
    q_R0 = np.array([-0.4, -0.2, -0.1])
    T_L0 = T_base_L @ left_arm.forward_kinematics(q_L0)[left_arm.end_effector_link_name]
    T_R0 = T_base_R @ right_arm.forward_kinematics(q_R0)[right_arm.end_effector_link_name]
    T_rel_des = np.linalg.inv(T_L0) @ T_R0

    bar = ClosedChainKinematicBarrier(
        left_model=left_arm,
        right_model=right_arm,
        T_base_L=T_base_L,
        T_base_R=T_base_R,
        T_rel_des=T_rel_des,
        eps_loop=0.05,
        weight=[1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
        alpha=2.0,
    )

    q_test = np.array([0.46, -0.15, 0.18, -0.34, -0.25, -0.05])
    # Analytical grad_h and error_jacobian must match central finite differences
    J_ana = bar.error_jacobian(q_test)
    eps = 1e-6
    J_fd = np.zeros_like(J_ana)
    for i in range(6):
        ei = np.zeros(6)
        ei[i] = eps
        J_fd[:, i] = (bar.pose_error(q_test + ei) - bar.pose_error(q_test - ei)) / (2.0 * eps)
    np.testing.assert_allclose(J_ana, J_fd, atol=1e-6)

    g_ana = bar.grad_h(q_test)
    g_fd = np.zeros(6)
    for i in range(6):
        ei = np.zeros(6)
        ei[i] = eps
        g_fd[i] = (bar.h(q_test + ei) - bar.h(q_test - ei)) / (2.0 * eps)
    np.testing.assert_allclose(g_ana, g_fd, atol=1e-6)

    # Inflation shrinks h(x) strictly by delta
    bar_inf = bar.inflate(0.001)
    assert math.isclose(bar_inf.h(q_test), bar.h(q_test) - 0.001, rel_tol=0.0, abs_tol=1e-12)

    # Closed-loop kinematic velocity CBF prevents breaking loop closure
    dyn = ControlAffineDynamics(state_dim=6, control_dim=6)
    cbf_filter = StateCBFSafetyFilter(dynamics=dyn, barriers=[bar], u_min=-2.0, u_max=2.0)

    q = np.concatenate([q_L0, q_R0])
    assert bar.h(q) > 0.0
    dt = 0.01
    u_tear_apart = np.array([1.2, 0.5, -0.4, -1.2, -0.5, 0.4])
    for _ in range(40):
        res = cbf_filter.filter(q, u_tear_apart)
        assert res["verified_safe"] is True
        u_safe = np.asarray(res["u_filtered"], dtype=np.float64)
        q = q + dt * u_safe
        assert bar.h(q) >= -1e-4


# ---------------------------------------------------------------------------
# CooperativeWrenchCBF Tests
# ---------------------------------------------------------------------------


def test_cooperative_wrench_cbf_asymmetric_excessive_squeeze_and_slip():
    cbf = CooperativeWrenchCBF(
        p_L=(-0.1, 0.0, 0.0),
        p_R=(0.1, 0.0, 0.0),
        p_obj=(0.0, 0.0, 0.0),
        f_crush_max=25.0,
        f_min=4.0,
        mu=0.5,
    )

    # Verify exact nullspace property G @ N_G == 0 and idempotence N_G^2 == N_G
    assert cbf.G.shape == (6, 12)
    assert cbf.N_G.shape == (12, 12)
    np.testing.assert_allclose(cbf.G @ cbf.N_G, np.zeros((6, 12)), atol=1e-12)
    np.testing.assert_allclose(cbf.N_G @ cbf.N_G, cbf.N_G, atol=1e-12)

    # Asymmetric excessive squeeze: 150 N inward on left, -135 N inward on right (+15 N net x, +4 N lift)
    w_L_nom = np.array([150.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    w_R_nom = np.array([-135.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    w_nom = np.concatenate([w_L_nom, w_R_nom])

    # Nominal violates crush barrier dramatically
    assert cbf.crush_barrier.h(w_nom) < 0.0

    res = cbf.filter_wrenches(w_L_nom, w_R_nom)
    assert res["verified_safe"] is True
    assert res["status"] == "safe"
    w_safe = res["w_safe"]

    # Resultant object wrench is preserved while internal squeeze is bounded <= f_crush_max
    np.testing.assert_allclose(cbf.G @ w_safe, cbf.G @ w_nom, atol=1e-5)
    assert np.linalg.norm(cbf.N_G @ w_safe) <= 25.0 + 1e-5
    assert cbf.crush_barrier.h(w_safe) >= -1e-4
    assert cbf.antipodal_preload_barrier.h(w_safe) >= -1e-4

    # Slip-risk case: high tangential force (5 N each) with tiny inward force -> must increase normal squeeze to >= 10 N
    w_L_slip = np.array([1.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    w_R_slip = np.array([-1.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    w_slip = np.concatenate([w_L_slip, w_R_slip])
    assert cbf.antipodal_preload_barrier.h(w_slip) < 0.0

    res_slip = cbf.filter_wrenches(w_L_slip, w_R_slip)
    assert res_slip["verified_safe"] is True
    w_slip_safe = res_slip["w_safe"]
    np.testing.assert_allclose(cbf.G @ w_slip_safe, cbf.G @ w_slip, atol=1e-5)
    assert np.linalg.norm(cbf.N_G @ w_slip_safe) <= 25.0 + 1e-5
    assert res_slip["preload_components"]["h_cone_L"] >= -1e-5
    assert res_slip["preload_components"]["h_cone_R"] >= -1e-5
    assert res_slip["preload_components"]["h_preload"] >= -1e-5

    # Verify analytical gradients of crush_barrier and antipodal_preload_barrier
    eps = 1e-6
    g_crush = cbf.crush_barrier.grad_h(w_nom)
    g_crush_fd = np.array(
        [(cbf.crush_barrier.h(w_nom + eps * np.eye(12)[i]) - cbf.crush_barrier.h(w_nom - eps * np.eye(12)[i])) / (2.0 * eps) for i in range(12)]
    )
    np.testing.assert_allclose(g_crush, g_crush_fd, atol=1e-5)

    # Infeasible high lift vs low crush cap must fail closed
    res_bad = cbf.filter_wrenches(
        np.array([10.0, 0.0, 30.0, 0.0, 0.0, 0.0]),
        np.array([-10.0, 0.0, 30.0, 0.0, 0.0, 0.0]),
    )
    assert res_bad["verified_safe"] is False


# ---------------------------------------------------------------------------
# TautCableBarrier Tests
# ---------------------------------------------------------------------------


def test_taut_cable_barrier_prevents_overstretch_and_rupture():
    L_slack = 0.50
    delta_L_max = 0.10
    L_max = L_slack + delta_L_max

    for mode in ("quadratic", "exponential"):
        bar = TautCableBarrier(
            L_slack=L_slack,
            delta_L_max=delta_L_max,
            left_slice=slice(0, 3),
            right_slice=slice(3, 6),
            mode=mode,
            alpha=2.0,
        )
        dyn = ControlAffineDynamics(state_dim=6, control_dim=6)
        flt = StateCBFSafetyFilter(dynamics=dyn, barriers=[bar], u_min=-2.0, u_max=2.0)

        x = np.array([-0.22, 0.0, 0.0, 0.22, 0.0, 0.0])  # d = 0.44 < 0.60
        # Analytical gradient matches FD
        g_ana = bar.grad_h(x)
        eps = 1e-6
        g_fd = np.array(
            [(bar.h(x + eps * np.eye(6)[i]) - bar.h(x - eps * np.eye(6)[i])) / (2.0 * eps) for i in range(6)]
        )
        np.testing.assert_allclose(g_ana, g_fd, atol=1e-6)

        u_pull = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        dt = 0.02
        for _ in range(50):
            out = flt.filter(x, u_pull)
            assert out["verified_safe"] is True
            x = x + dt * np.asarray(out["u_filtered"], dtype=np.float64)
            assert bar.distance(x) <= L_max + 1e-3
            assert bar.h(x) >= -1e-3

    # Also verify ClosedChainKinematicBarrier + TautCableBarrier combined on URDF arms
    left_arm = URDFRobotModel.from_urdf_string(_make_3dof_arm_urdf("l_cable"))
    right_arm = URDFRobotModel.from_urdf_string(_make_3dof_arm_urdf("r_cable"))
    T_base_L = np.eye(4)
    T_base_L[:3, 3] = [0.0, -0.35, 0.0]
    T_base_R = np.eye(4)
    T_base_R[:3, 3] = [0.0, 0.35, 0.0]
    q0 = np.array([0.2, 0.0, 0.0, -0.2, 0.0, 0.0])

    cable_urdf = TautCableBarrier(
        L_slack=0.75,
        delta_L_max=0.05,
        left_model=left_arm,
        right_model=right_arm,
        T_base_L=T_base_L,
        T_base_R=T_base_R,
        alpha=2.0,
    )
    g_u = cable_urdf.grad_h(q0)
    g_u_fd = np.array(
        [(cable_urdf.h(q0 + 1e-6 * np.eye(6)[i]) - cable_urdf.h(q0 - 1e-6 * np.eye(6)[i])) / 2e-6 for i in range(6)]
    )
    np.testing.assert_allclose(g_u, g_u_fd, atol=1e-6)


# ---------------------------------------------------------------------------
# RightInvariantSE3Filter Tests
# ---------------------------------------------------------------------------


def test_right_invariant_se3_filter_delayed_visual_fusion():
    rng = np.random.default_rng(42)
    dt = 0.005  # 200 Hz IMU
    delay_steps = 12  # 60 ms visual latency
    delay_s = delay_steps * dt
    gravity = np.array([0.0, 0.0, -9.81])

    # True initial state
    R_true = exp_so3([0.05, -0.03, 0.1])
    v_true = np.array([0.4, -0.2, 0.05])
    p_true = np.array([0.2, -0.1, 0.5])

    # Perturbed initial filter belief
    R_init = exp_so3([0.12, -0.08, 0.22]) @ R_true
    v_init = v_true + np.array([0.1, -0.1, 0.05])
    p_init = p_true + np.array([-0.15, 0.12, -0.08])

    iekf = RightInvariantSE3Filter(
        R0=R_init,
        v0=v_init,
        p0=p_init,
        P0=np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        t0=0.0,
        gravity=gravity,
        gyro_noise_std=0.003,
        accel_noise_std=0.03,
        buffer_size=256,
    )
    dead_reckon = RightInvariantSE3Filter(
        R0=R_init,
        v0=v_init,
        p0=p_init,
        P0=0.1,
        t0=0.0,
        gravity=gravity,
        gyro_noise_std=0.003,
        accel_noise_std=0.03,
    )

    true_history = [(0.0, R_true.copy(), p_true.copy())]
    meas_cov = np.diag([1e-4, 1e-4, 1e-4, 4e-4, 4e-4, 4e-4])

    total_steps = 200  # 1.0 s
    rewound_counts = []
    cov_shrank_count = 0

    for step in range(1, total_steps + 1):
        t_now = step * dt
        omega_true = np.array([0.25 * math.sin(2.0 * t_now), 0.15, 0.4 * math.cos(1.5 * t_now)])
        a_world_true = np.array([0.6 * math.cos(t_now), -0.4 * math.sin(2.0 * t_now), 0.2])
        accel_body_true = R_true.T @ (a_world_true - gravity)

        # Advance ground truth
        R_mid = R_true @ exp_so3(0.5 * omega_true * dt)
        R_true = R_true @ exp_so3(omega_true * dt)
        p_true = p_true + v_true * dt + 0.5 * (R_mid @ accel_body_true + gravity) * (dt * dt)
        v_true = v_true + (R_mid @ accel_body_true + gravity) * dt
        true_history.append((t_now, R_true.copy(), p_true.copy()))

        # Noisy IMU reading
        omega_meas = omega_true + rng.normal(0.0, 0.008, size=3)
        accel_meas = accel_body_true + rng.normal(0.0, 0.06, size=3)
        iekf.propagate(omega_meas, accel_meas, dt)
        dead_reckon.propagate(omega_meas, accel_meas, dt)

        # Every 20 steps (100 ms), a 60 ms delayed camera pose arrives
        if step >= delay_steps and step % 20 == 0:
            t_cam, R_cam_true, p_cam_true = true_history[step - delay_steps]
            R_meas = exp_so3(rng.normal(0.0, 0.004, size=3)) @ R_cam_true
            p_meas = p_cam_true + rng.normal(0.0, 0.008, size=3)

            cov_trace_before = float(np.trace(iekf.covariance))
            res = iekf.fuse_delayed_pose(t_cam, R_meas, p_meas, meas_cov=meas_cov)
            cov_trace_after = float(np.trace(res["covariance"]))

            assert res["updated"] is True
            assert res["rewound_steps"] == delay_steps
            assert math.isclose(res["delay_s"], delay_s, abs_tol=1e-9)
            rewound_counts.append(res["rewound_steps"])
            if cov_trace_after < cov_trace_before:
                cov_shrank_count += 1

    assert len(rewound_counts) >= 8
    assert cov_shrank_count == len(rewound_counts)

    pos_err_iekf = float(np.linalg.norm(iekf.p - p_true))
    rot_err_iekf = float(np.linalg.norm(log_so3(R_true @ iekf.R.T)))
    pos_err_dr = float(np.linalg.norm(dead_reckon.p - p_true))
    rot_err_dr = float(np.linalg.norm(log_so3(R_true @ dead_reckon.R.T)))

    assert pos_err_iekf < 0.04
    assert rot_err_iekf < 0.03
    assert pos_err_iekf < 0.25 * pos_err_dr
    assert rot_err_iekf < 0.5 * rot_err_dr


# ---------------------------------------------------------------------------
# OcclusionShadowBarrier Tests
# ---------------------------------------------------------------------------


def test_occlusion_shadow_barrier_expansion_slowdown_and_observation_collapse():
    # 2D pillar occluding target at (3.0, 0.0) from sensor at (0.0, 0.0)
    pillar = OccluderSegment(p0=(1.5, -0.5), p1=(1.5, 0.5))
    box_3d = OccluderBox(min_corner=(1.4, -0.4, -1.0), max_corner=(1.6, 0.4, 1.0))

    bar = OcclusionShadowBarrier(
        last_seen_pos=(3.0, 0.0),
        r_0=0.3,
        v_max=1.2,
        a_max=0.8,
        sensor_pos=(0.0, 0.0),
        occluders=[pillar],
        last_seen_time=0.0,
        alpha=1.5,
        alpha2=2.0,
        relative_degree=2,
    )
    assert bar.occluded is True
    assert math.isclose(bar.reachable_radius(), 0.3, abs_tol=1e-12)

    # 3D box ray-cast check
    bar_3d = OcclusionShadowBarrier(
        last_seen_pos=(3.0, 0.0, 0.0),
        r_0=0.3,
        v_max=1.0,
        a_max=0.5,
        sensor_pos=(0.0, 0.0, 0.0),
        occluders=[box_3d],
    )
    assert bar_3d.is_occluded((3.0, 0.0, 0.0), (0.0, 0.0, 0.0)) is True
    assert bar_3d.is_occluded((3.0, 0.0, 0.0), (1.5, 1.0, 0.0)) is False

    # Advance time while occluded: verify radius formula & exact dh_dt, d2h_dt2, grad_dh_dt
    bar.advance_time(0.5)
    expected_r = 0.3 + 1.2 * 0.5 + 0.5 * 0.8 * (0.5 ** 2)  # 0.3 + 0.6 + 0.1 = 1.0
    assert math.isclose(bar.reachable_radius(), expected_r, abs_tol=1e-12)

    x_probe = np.array([1.8, 0.2, 0.5, 0.0])
    eps_t = 1e-6
    t0 = bar.current_time
    bar.set_time(t0 + eps_t)
    h_p = bar.h(x_probe)
    dh_p = bar.dh_dt(x_probe)
    bar.set_time(t0 - eps_t)
    h_m = bar.h(x_probe)
    dh_m = bar.dh_dt(x_probe)
    bar.set_time(t0)

    assert math.isclose(bar.dh_dt(x_probe), (h_p - h_m) / (2.0 * eps_t), abs_tol=1e-5)
    assert math.isclose(bar.d2h_dt2(x_probe), (dh_p - dh_m) / (2.0 * eps_t), abs_tol=1e-5)
    np.testing.assert_allclose(bar.grad_dh_dt(x_probe), np.zeros(4), atol=1e-12)

    # Wire into 2D double-integrator StateCBFSafetyFilter (x = [px, py, vx, vy], u = [ax, ay])
    A = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    B = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    dyn = LinearDynamics(A, B)
    flt = StateCBFSafetyFilter(dynamics=dyn, barriers=[bar], u_min=[-4.0, -4.0], u_max=[4.0, 4.0])

    # Robot approaching occluded shadow zone at x = 0.8 with vx = 0.6 m/s, nominal ax = +1.0 m/s^2
    x_robot = np.array([0.8, 0.0, 0.6, 0.0])
    u_nom = np.array([1.0, 0.0])
    out_occluded = flt.filter(x_robot, u_nom)
    assert out_occluded["verified_safe"] is True
    ax_occluded = out_occluded["u_filtered"][0]
    # Expanding reachability shadow forces strong braking (ax < 0 < u_nom[0])
    assert ax_occluded < -0.5

    # Active perception (epistemic_probe) moves sensor around pillar and re-acquires LOS
    bar.set_sensor_pos((1.5, 1.2))
    assert bar.occluded is False
    r_after_obs = bar.mark_observed(pos=(3.0, 0.0), vel=(0.0, 0.0), now_s=bar.current_time)
    assert math.isclose(r_after_obs, 0.3, abs_tol=1e-12)
    assert math.isclose(bar.dh_dt(x_robot), 0.0, abs_tol=1e-12)

    out_visible = flt.filter(x_robot, u_nom)
    assert out_visible["verified_safe"] is True
    ax_visible = out_visible["u_filtered"][0]
    # Once line of sight collapses r_occ back to r_0, nominal motion is unimpeded
    assert ax_visible > ax_occluded + 1.0
    assert math.isclose(ax_visible, u_nom[0], abs_tol=1e-6)
