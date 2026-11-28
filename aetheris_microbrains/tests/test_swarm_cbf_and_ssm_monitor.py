"""Tests for DecentralizedSwarmCBF and ProprioceptiveSSMMonitor.

Covers:
  - 2-agent and 4-agent symmetric head-on antipodal swaps in 2D and 3D
    (single-integrator and double-integrator HOCBF), proving deadlock/collision
    without circulation vs 100% swap completion with certified clearance.
  - CapabilityToken-weighted right-of-way evasion burden sharing (fragile
    low-energy robot vs unladen high-energy robot).
  - ProprioceptiveSSMMonitor O(1) constant memory across 500 1-kHz ticks,
    singleton isolation, classification of nominal / incipient slip / backlash /
    thermal-torque degradation, DirectionalBarrierInflator integration, and
    closed-loop HighOrderBarrier early braking under 50% actuator torque loss.
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

from aetheris_microbrains.core.adaptive_conformal_robotics import DirectionalBarrierInflator
from aetheris_microbrains.core.capability_tokens import (
    CapabilityCaveat,
    attenuate_capability,
    mint_root_capability,
)
from aetheris_microbrains.core.linear_state_space_microbrain import LinearStateSpaceMicroBrain
from aetheris_microbrains.core.proprioceptive_ssm_monitor import ProprioceptiveSSMMonitor
from aetheris_microbrains.core.state_cbf_filter import (
    EulerLagrangeDynamics,
    HalfPlaneBarrier,
    HighOrderBarrier,
    StateCBFSafetyFilter,
)
from aetheris_microbrains.core.swarm_cbf_coordinator import (
    DecentralizedSwarmCBF,
    SwarmAgentDescriptor,
    compute_capability_priority,
    compute_responsibility_share,
)


class TestCapabilityPriorityAndDecentralizedPartition:
    def test_priority_monotonic_under_capability_attenuation_and_partition_of_unity(self):
        secret = "swarm-root-secret-key"
        unconstrained = SwarmAgentDescriptor("free_bot", [0.0, 0.0])
        assert compute_capability_priority(unconstrained) == pytest.approx(1.0)

        tok_high_e = mint_root_capability(
            secret,
            subject="agile_carrier",
            caveat=CapabilityCaveat(max_kinetic_energy_j=25.0, max_linear_vel_mps=2.0, depth_remaining=4),
            now=1000.0,
        )
        tok_mid = attenuate_capability(
            tok_high_e,
            CapabilityCaveat(max_kinetic_energy_j=4.0, max_contact_force_n=15.0, depth_remaining=2),
        )
        tok_fragile = attenuate_capability(
            tok_mid,
            CapabilityCaveat(max_kinetic_energy_j=0.5, max_linear_vel_mps=0.2, max_contact_force_n=5.0, depth_remaining=0),
        )

        ag_high = SwarmAgentDescriptor("high", [0.0, 0.0], capability_token=tok_high_e)
        ag_mid = SwarmAgentDescriptor("mid", [1.0, 0.0], capability_token=tok_mid)
        ag_fragile = SwarmAgentDescriptor("fragile", [2.0, 0.0], capability_token=tok_fragile)

        p_high = compute_capability_priority(ag_high)
        p_mid = compute_capability_priority(ag_mid)
        p_fragile = compute_capability_priority(ag_fragile)

        assert 0.0 < p_high < p_mid < p_fragile

        for a, b in [(ag_high, ag_mid), (ag_high, ag_fragile), (ag_mid, ag_fragile)]:
            w_ab = compute_responsibility_share(a, b)
            w_ba = compute_responsibility_share(b, a)
            assert 0.0 < w_ab < 1.0
            assert 0.0 < w_ba < 1.0
            assert w_ab + w_ba == pytest.approx(1.0, abs=1e-12)

    def test_pairwise_halfspace_sums_to_centralized_cbf_and_hocbf_without_neighbor_u(self):
        secret = "partition-check"
        tok_fragile = mint_root_capability(
            secret,
            subject="glassware",
            caveat=CapabilityCaveat(max_kinetic_energy_j=0.5, max_linear_vel_mps=0.2),
            now=1000.0,
        )
        tok_agile = mint_root_capability(
            secret,
            subject="runner",
            caveat=CapabilityCaveat(max_kinetic_energy_j=25.0),
            now=1000.0,
        )
        a1 = SwarmAgentDescriptor("a1", [-0.45, 0.05], velocity=[0.8, 0.0], radius=0.25, capability_token=tok_fragile)
        a2 = SwarmAgentDescriptor("a2", [0.45, -0.05], velocity=[-1.1, 0.1], radius=0.25, capability_token=tok_agile)

        for mode in ("single_integrator", "double_integrator"):
            coord = DecentralizedSwarmCBF(dim=2, dynamics_mode=mode, alpha=2.5, alpha1=2.0, alpha2=3.0)
            row_12, b_12, meta_12 = coord.pairwise_halfspace(a1, a2)
            row_21, b_21, meta_21 = coord.pairwise_halfspace(a2, a1)

            assert np.allclose(row_12, -row_21, atol=1e-12)
            assert meta_12["joint_rhs"] == pytest.approx(meta_21["joint_rhs"], abs=1e-12)
            assert b_12 + b_21 == pytest.approx(meta_12["joint_rhs"], abs=1e-12)


class TestSwarmAntipodalSwaps2DAnd3D:
    def test_2_agent_antipodal_swap_2d_deadlocks_without_circulation_and_succeeds_with_cbf(self):
        radius = 0.25
        d_safe = 2.0 * radius
        starts = [[-2.0, 0.0], [2.0, 0.0]]
        goals = [[2.0, 0.0], [-2.0, 0.0]]

        # 1. Unfiltered nominal controllers collide head-on at the origin
        p0 = np.array(starts[0], dtype=np.float64)
        p1 = np.array(starts[1], dtype=np.float64)
        min_unfiltered_dist = float("inf")
        for _ in range(200):
            v0 = np.clip(1.4 * (np.array(goals[0]) - p0), -1.5, 1.5)
            v1 = np.clip(1.4 * (np.array(goals[1]) - p1), -1.5, 1.5)
            p0 = p0 + 0.02 * v0
            p1 = p1 + 0.02 * v1
            min_unfiltered_dist = min(min_unfiltered_dist, float(np.linalg.norm(p0 - p1)))
        assert min_unfiltered_dist < 0.05  # Direct head-on collision

        # 2. CBF without circulation stalls in symmetric head-on deadlock
        agents_no_circ = [
            SwarmAgentDescriptor(f"r{i}", starts[i], radius=radius, u_min=[-1.5, -1.5], u_max=[1.5, 1.5])
            for i in range(2)
        ]
        coord_no_circ = DecentralizedSwarmCBF(
            dim=2,
            dynamics_mode="single_integrator",
            enable_circulation=False,
            alpha=2.5,
        )
        res_deadlock = coord_no_circ.step_swarm_simulation(
            agents_no_circ, goals=goals, dt=0.02, n_steps=350, goal_tolerance=0.08
        )
        assert res_deadlock["all_reached_goals"] is False
        assert max(res_deadlock["goal_errors"].values()) > 2.0

        # 3. DecentralizedSwarmCBF (with circulation) completes 100% of swap in both dynamics modes
        for mode, n_steps, u_lim in [("single_integrator", 350, 1.6), ("double_integrator", 450, 8.0)]:
            agents = [
                SwarmAgentDescriptor(
                    f"r{i}",
                    starts[i],
                    radius=radius,
                    u_min=[-u_lim, -u_lim],
                    u_max=[u_lim, u_lim],
                )
                for i in range(2)
            ]
            coord = DecentralizedSwarmCBF(dim=2, dynamics_mode=mode, alpha=2.5, alpha1=2.2, alpha2=3.2)
            res = coord.step_swarm_simulation(agents, goals=goals, dt=0.02, n_steps=n_steps, goal_tolerance=0.08)
            assert res["all_reached_goals"] is True, f"Failed swap in {mode}: {res['goal_errors']}"
            assert res["min_pairwise_distance"] >= d_safe - 1e-4
            assert res["circulation_steps"] > 0
            assert res["all_verified_safe"] is True

    def test_4_agent_symmetric_swap_2d_zero_deadlocks_and_safe_clearance(self):
        radius = 0.22
        d_safe = 2.0 * radius
        starts = [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]]
        goals = [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]]

        # Without circulation: 4-way symmetric deadlock at origin
        coord_no_circ = DecentralizedSwarmCBF(dim=2, dynamics_mode="single_integrator", enable_circulation=False)
        res_stall = coord_no_circ.step_swarm_simulation(
            [SwarmAgentDescriptor(f"a{i}", starts[i], radius=radius) for i in range(4)],
            goals=goals,
            dt=0.02,
            n_steps=400,
            goal_tolerance=0.08,
        )
        assert res_stall["all_reached_goals"] is False
        assert min(res_stall["goal_errors"].values()) > 2.0

        # With circulation: roundabout resolves 4-way intersection in single and double integrator modes
        for mode, n_steps, u_lim in [("single_integrator", 420, 1.8), ("double_integrator", 550, 9.0)]:
            coord = DecentralizedSwarmCBF(dim=2, dynamics_mode=mode, alpha=2.5, alpha1=2.2, alpha2=3.4)
            agents = [
                SwarmAgentDescriptor(f"a{i}", starts[i], radius=radius, u_min=[-u_lim] * 2, u_max=[u_lim] * 2)
                for i in range(4)
            ]
            res = coord.step_swarm_simulation(agents, goals=goals, dt=0.02, n_steps=n_steps, goal_tolerance=0.08)
            assert res["all_reached_goals"] is True, f"4-agent 2D {mode} failed: {res['goal_errors']}"
            assert res["min_pairwise_distance"] >= d_safe - 1e-4

    def test_2_and_4_agent_symmetric_swap_3d_helicity_and_zero_deadlocks(self):
        radius = 0.22
        d_safe = 2.0 * radius
        coord_check = DecentralizedSwarmCBF(dim=3, dynamics_mode="single_integrator")

        # Verify antipodal chirality consistency t(-r) == -t(r) and orthogonality across 3D directions
        probe_dirs = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.17, -0.29, 0.94],  # Collinear with primary helicity axis -> fallback branch
        ]
        for d_raw in probe_dirs:
            r_hat = np.asarray(d_raw, dtype=np.float64) / np.linalg.norm(d_raw)
            t_pos = coord_check.circulation_tangent(r_hat)
            t_neg = coord_check.circulation_tangent(-r_hat)
            assert np.linalg.norm(t_pos) == pytest.approx(1.0, abs=1e-9)
            assert float(np.dot(t_pos, r_hat)) == pytest.approx(0.0, abs=1e-9)
            assert np.allclose(t_neg, -t_pos, atol=1e-9)

        # 2-agent 3D vertical z-axis swap: stalls without circulation, succeeds with circulation
        starts_2 = [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]
        goals_2 = [[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]

        coord_3d_nocirc = DecentralizedSwarmCBF(dim=3, dynamics_mode="single_integrator", enable_circulation=False)
        res_2_stall = coord_3d_nocirc.step_swarm_simulation(
            [SwarmAgentDescriptor(f"z{i}", starts_2[i], radius=radius) for i in range(2)],
            goals=goals_2,
            dt=0.02,
            n_steps=350,
        )
        assert res_2_stall["all_reached_goals"] is False

        for mode, n_steps, u_lim in [("single_integrator", 360, 1.8), ("double_integrator", 480, 9.0)]:
            coord_3d = DecentralizedSwarmCBF(dim=3, dynamics_mode=mode, alpha=2.5, alpha1=2.2, alpha2=3.4)
            res_2 = coord_3d.step_swarm_simulation(
                [
                    SwarmAgentDescriptor(f"z{i}", starts_2[i], radius=radius, u_min=[-u_lim] * 3, u_max=[u_lim] * 3)
                    for i in range(2)
                ],
                goals=goals_2,
                dt=0.02,
                n_steps=n_steps,
                goal_tolerance=0.08,
            )
            assert res_2["all_reached_goals"] is True, f"2-agent 3D {mode} failed: {res_2['goal_errors']}"
            assert res_2["min_pairwise_distance"] >= d_safe - 1e-4

        # 4-agent 3D antipodal swap intersecting simultaneously at (0, 0, 0)
        starts_4 = [
            [-2.0, 0.0, -0.8],
            [2.0, 0.0, 0.8],
            [0.0, -2.0, 0.8],
            [0.0, 2.0, -0.8],
        ]
        goals_4 = [
            [2.0, 0.0, 0.8],
            [-2.0, 0.0, -0.8],
            [0.0, 2.0, -0.8],
            [0.0, -2.0, 0.8],
        ]
        res_4_stall = coord_3d_nocirc.step_swarm_simulation(
            [SwarmAgentDescriptor(f"q{i}", starts_4[i], radius=radius) for i in range(4)],
            goals=goals_4,
            dt=0.02,
            n_steps=400,
        )
        assert res_4_stall["all_reached_goals"] is False

        for mode, n_steps, u_lim in [("single_integrator", 440, 1.9), ("double_integrator", 580, 9.5)]:
            coord_3d = DecentralizedSwarmCBF(dim=3, dynamics_mode=mode, alpha=2.5, alpha1=2.2, alpha2=3.4)
            res_4 = coord_3d.step_swarm_simulation(
                [
                    SwarmAgentDescriptor(f"q{i}", starts_4[i], radius=radius, u_min=[-u_lim] * 3, u_max=[u_lim] * 3)
                    for i in range(4)
                ],
                goals=goals_4,
                dt=0.02,
                n_steps=n_steps,
                goal_tolerance=0.08,
            )
            assert res_4["all_reached_goals"] is True, f"4-agent 3D {mode} failed: {res_4['goal_errors']}"
            assert res_4["min_pairwise_distance"] >= d_safe - 1e-4


class TestCapabilityWeightedRightOfWay:
    def test_fragile_payload_agent_deviates_minimally_while_unladen_robot_evades_90_percent_plus(self):
        secret = "warehouse-hmac-secret"
        fragile_token = mint_root_capability(
            secret,
            subject="beaker_carrier",
            caveat=CapabilityCaveat(max_kinetic_energy_j=0.5, max_linear_vel_mps=0.2),
            now=2000.0,
        )
        unladen_token = mint_root_capability(
            secret,
            subject="empty_amr",
            caveat=CapabilityCaveat(max_kinetic_energy_j=25.0),
            now=2000.0,
        )

        radius = 0.25
        d_safe = 2.0 * radius
        fragile_agent = SwarmAgentDescriptor(
            "fragile",
            position=[-1.5, 0.0],
            radius=radius,
            u_min=[-0.5, -0.5],
            u_max=[0.5, 0.5],
            capability_token=fragile_token,
        )
        unladen_agent = SwarmAgentDescriptor(
            "unladen",
            position=[1.5, 0.0],
            radius=radius,
            u_min=[-2.0, -2.0],
            u_max=[2.0, 2.0],
            capability_token=unladen_token,
        )

        omega_fragile = compute_responsibility_share(fragile_agent, unladen_agent)
        omega_unladen = compute_responsibility_share(unladen_agent, fragile_agent)
        assert omega_fragile < 0.05  # omega_ij << 0.5
        assert omega_unladen > 0.95
        assert omega_fragile + omega_unladen == pytest.approx(1.0)

        coord = DecentralizedSwarmCBF(dim=2, dynamics_mode="single_integrator", alpha=2.5)
        goals = {"fragile": [1.5, 0.0], "unladen": [-1.5, 0.0]}
        res = coord.step_swarm_simulation(
            [fragile_agent, unladen_agent],
            goals=goals,
            dt=0.02,
            n_steps=900,
            goal_tolerance=0.06,
        )

        assert res["all_reached_goals"] is True, res["goal_errors"]
        assert res["min_pairwise_distance"] >= d_safe - 1e-4

        traj_fragile = res["trajectories"]["fragile"]
        traj_unladen = res["trajectories"]["unladen"]
        max_lat_fragile = float(np.max(np.abs(traj_fragile[:, 1])))
        max_lat_unladen = float(np.max(np.abs(traj_unladen[:, 1])))

        evasion_fraction_unladen = max_lat_unladen / (max_lat_fragile + max_lat_unladen)
        assert max_lat_fragile < 0.06
        assert max_lat_unladen > 0.45
        assert evasion_fraction_unladen >= 0.90


class TestProprioceptiveSSMMonitorAndHOCBFDerating:
    def test_dedicated_ssm_instance_and_constant_o1_memory_over_500_ticks(self):
        prev_singleton = LinearStateSpaceMicroBrain._instance
        monitor = ProprioceptiveSSMMonitor(n_joints=4, d_state=16)

        assert monitor.ssm is not prev_singleton
        assert LinearStateSpaceMicroBrain._instance is prev_singleton

        first_out = monitor.step(
            tau_cmd=[1.0, -0.5, 0.2, 0.0],
            tau_meas=[0.99, -0.50, 0.20, 0.0],
            q_des=[0.1, 0.2, -0.1, 0.0],
            q=[0.1, 0.2, -0.1, 0.0],
            qdot_des=[0.0, 0.0, 0.0, 0.0],
            qdot=[0.0, 0.0, 0.0, 0.0],
            winding_temp_c=36.0,
            tactile_hf_energy=0.02,
        )
        mem_tick_1 = first_out["memory_bytes"]
        shape_tick_1 = monitor.ssm.hidden_state.shape
        buf_ids_tick_1 = (id(monitor._u_buf), id(monitor.torque_scale), id(monitor.covariance_inflation))

        for k in range(1, 500):
            phase = 0.02 * k
            s = math.sin(phase)
            out = monitor.step(
                tau_cmd=[1.0 * s, -0.5 * s, 0.3, 0.1],
                tau_meas=[0.99 * s, -0.495 * s, 0.298, 0.1],
                q_des=[0.2 * s, -0.1 * s, 0.05, 0.0],
                q=[0.199 * s, -0.099 * s, 0.05, 0.0],
                qdot_des=[0.2 * math.cos(phase), -0.1 * math.cos(phase), 0.0, 0.0],
                qdot=[0.2 * math.cos(phase), -0.1 * math.cos(phase), 0.0, 0.0],
                winding_temp_c=38.0,
                tactile_hf_energy=0.03,
            )

        assert monitor.ticks == 500
        assert monitor.ssm.processed_tokens_count == 500
        assert out["memory_bytes"] == mem_tick_1
        assert monitor.state_memory_bytes() == mem_tick_1
        assert monitor.ssm.hidden_state.shape == shape_tick_1
        assert id(monitor._u_buf) == buf_ids_tick_1[0]
        assert out["health_status"] == ProprioceptiveSSMMonitor.HEALTH_NOMINAL
        assert out["grasp_force_delta_n"] == 0.0
        assert np.allclose(out["torque_scale"], 1.0, atol=0.05)

    def test_incipient_slip_detection_and_bounded_grasp_force_delta(self):
        monitor = ProprioceptiveSSMMonitor(
            n_joints=3,
            nominal_grasp_force_n=6.0,
            max_grasp_force_n=20.0,
            slip_hf_threshold=0.30,
        )
        # Nominal grasp frame -> zero force delta
        nom = monitor.step(
            torque_error=[0.05, 0.02, 0.01],
            pos_error=[0.001, 0.0, 0.0],
            vel_error=[0.0, 0.0, 0.0],
            winding_temp_c=35.0,
            tactile_hf_energy=0.04,
        )
        assert nom["health_status"] == "nominal"
        assert nom["grasp_force_delta_n"] == 0.0

        # Incipient slip burst -> positive force delta bounded by max_grasp_force_n
        slip_out = None
        for _ in range(8):
            slip_out = monitor.step(
                torque_error=[0.1, -0.1, 0.05],
                pos_error=[0.004, 0.002, 0.0],
                vel_error=[0.08, 0.03, 0.0],
                winding_temp_c=36.0,
                tactile_hf_energy=0.95,
                tangential_slip_rate=0.12,
            )
        assert slip_out is not None
        assert slip_out["health_status"] == "incipient_gripper_slip"
        assert slip_out["grasp_force_delta_n"] > 0.0
        assert slip_out["recommended_grasp_force_n"] <= 20.0
        assert slip_out["grasp_force_delta_n"] <= 20.0 - 6.0 + 1e-9

        # Extreme HF energy near grasp saturation respects max_grasp_force_n cap strictly
        capped = monitor.step(
            torque_error=[0.1, 0.0, 0.0],
            vel_error=[0.2, 0.0, 0.0],
            tactile_hf_energy=50.0,
            tangential_slip_rate=2.0,
            current_grasp_force_n=19.2,
        )
        assert capped["health_status"] == "incipient_gripper_slip"
        assert 0.0 < capped["grasp_force_delta_n"] <= 0.8 + 1e-9
        assert capped["recommended_grasp_force_n"] == pytest.approx(20.0, abs=1e-9)

    def test_backlash_phase_lag_detection_inflates_directional_barrier_covariance(self):
        monitor = ProprioceptiveSSMMonitor(n_joints=2, nominal_max_torque=10.0)
        inflator = DirectionalBarrierInflator(state_dim=4, ema_decay=0.85)
        wall_joint0 = HighOrderBarrier(HalfPlaneBarrier(a=[-1.0, 0.0], b=-1.0), alpha1=2.0, alpha2=2.0)
        wall_joint1 = HighOrderBarrier(HalfPlaneBarrier(a=[0.0, -1.0], b=-1.0), alpha1=2.0, alpha2=2.0)

        # Joint 0 has 75-degree oscillatory phase lag between tau_cmd and qdot (cable stretch / backlash);
        # Joint 1 tracks cleanly in phase with zero lag.
        lag_rad = math.radians(75.0)
        last_out = None
        for k in range(80):
            wt = 0.22 * k
            tau0 = 4.0 * math.sin(wt)
            qd0 = 0.8 * math.sin(wt - lag_rad)
            qd0_des = 0.8 * math.sin(wt)

            tau1 = 2.5 * math.sin(wt)
            qd1 = 0.5 * math.sin(wt)  # Zero phase lag on Joint 1

            last_out = monitor.step(
                tau_cmd=[tau0, tau1],
                tau_meas=[tau0 * 0.98, tau1 * 0.98],
                q_des=[0.0, 0.0],
                q=[-0.03 * math.cos(wt - lag_rad), 0.0],
                qdot_des=[qd0_des, qd1],
                qdot=[qd0, qd1],
                winding_temp_c=37.0,
                tactile_hf_energy=0.02,
                inflator=inflator,
            )

        assert last_out is not None
        assert last_out["health_status"] == "joint_backlash_or_cable_stretch"
        cov_inf = np.asarray(last_out["covariance_inflation"])
        assert cov_inf[0] > 0.02
        assert cov_inf[1] == pytest.approx(0.0, abs=1e-6)

        # DirectionalBarrierInflator inflates Joint 0 barrier much more than healthy Joint 1 barrier
        x_probe = np.array([0.4, 0.4, 0.0, 0.0])
        m0 = inflator.directional_margin(wall_joint0.grad_h(x_probe), q_hat=1.0)
        m1 = inflator.directional_margin(wall_joint1.grad_h(x_probe), q_hat=1.0)
        assert m0["effective_margin"] > 10.0 * m1["effective_margin"]

    def test_thermal_and_torque_degradation_derating_enables_hocbf_early_braking_before_wall(self):
        # Verify thermal-only derating on joint 1 as well
        therm_mon = ProprioceptiveSSMMonitor(n_joints=2, min_authority=0.2, temp_derate_start_c=75.0, temp_critical_c=115.0)
        therm_res = therm_mon.step(winding_temp_c=[40.0, 95.0], tactile_hf_energy=0.01)
        assert therm_res["health_status"] == "actuator_thermal_or_torque_degradation"
        assert therm_res["torque_scale"][0] == pytest.approx(1.0)
        assert therm_res["torque_scale"][1] == pytest.approx(0.6, abs=1e-6)

        # Closed-loop 1-DOF manipulator/carriage rushing toward wall q <= 2.0 with 50% physical torque loss
        # (degraded actuator saturates at [-5.0, +5.0] N*m instead of nominal [-10.0, +10.0] N*m).
        wall_q = 2.0
        nom_u_min = [-10.0]
        nom_u_max = [10.0]
        degraded_torque_cap = 5.0  # 50% actuator torque loss

        # Ingest sustained 50% torque deficit frames into ProprioceptiveSSMMonitor
        monitor = ProprioceptiveSSMMonitor(n_joints=1, min_authority=0.2, nominal_max_torque=10.0)
        deg_out = None
        for _ in range(55):
            deg_out = monitor.step(
                tau_cmd=[10.0],
                tau_meas=[degraded_torque_cap],  # 5.0 N*m delivered vs 10.0 N*m commanded
                q_des=[0.2],
                q=[0.18],
                qdot_des=[1.0],
                qdot=[0.95],
                winding_temp_c=68.0,
                tactile_hf_energy=0.02,
            )
        assert deg_out is not None
        assert deg_out["health_status"] == "actuator_thermal_or_torque_degradation"
        assert deg_out["torque_scale"][0] == pytest.approx(0.50, abs=0.02)

        def _run_wall_rollout(apply_derating: bool):
            dyn = EulerLagrangeDynamics(n_dof=1)
            wall = HighOrderBarrier(
                HalfPlaneBarrier(a=[-1.0, 0.0], b=-wall_q),
                alpha1=5.8,
                alpha2=5.8,
            )
            safety_filter = StateCBFSafetyFilter(dyn, [wall], u_min=nom_u_min, u_max=nom_u_max)
            if apply_derating:
                # Call multiple times to also prove idempotence of HOCBF alpha scaling
                derate_info = monitor.apply_derating_to_filter(safety_filter, nom_u_min, nom_u_max)
                derate_info = monitor.apply_derating_to_filter(safety_filter, nom_u_min, nom_u_max)
                assert derate_info["u_max"][0] == pytest.approx(10.0 * deg_out["torque_scale"][0], abs=1e-6)
                assert derate_info["u_min"][0] == pytest.approx(-10.0 * deg_out["torque_scale"][0], abs=1e-6)

            x = np.array([0.0, 3.2], dtype=np.float64)
            dt = 0.005
            first_brake_q = None
            max_q = float(x[0])
            min_h = wall.h(x)
            all_verified = True

            for _ in range(450):
                u_nom = [6.0]  # Nominal policy keeps pushing into the wall
                res = safety_filter.filter(x, u_nom)
                all_verified = all_verified and bool(res["verified_safe"])
                u_cmd = float(res["u_filtered"][0])
                if first_brake_q is None and u_cmd < 0.0:
                    first_brake_q = float(x[0])

                # Physical plant suffers 50% torque loss (deliverable torque capped at +-5.0 N*m)
                u_delivered = float(np.clip(u_cmd, -degraded_torque_cap, degraded_torque_cap))
                xdot = dyn.f(x) + dyn.g(x) @ np.array([u_delivered])
                x = x + dt * xdot
                max_q = max(max_q, float(x[0]))
                min_h = min(min_h, wall.h(x))

            return {
                "first_brake_q": first_brake_q,
                "max_q": max_q,
                "min_h": min_h,
                "all_verified": all_verified,
                "u_min": safety_filter.u_min.copy(),
                "u_max": safety_filter.u_max.copy(),
            }

        unaware = _run_wall_rollout(apply_derating=False)
        derated = _run_wall_rollout(apply_derating=True)

        # Verify bounds shrunk by ~50%
        assert derated["u_max"][0] < 5.3
        assert derated["u_min"][0] > -5.3

        # Un-derated filter brakes too late for 50% authority and crashes past q = 2.0 (h < 0)
        assert unaware["max_q"] > wall_q + 0.05
        assert unaware["min_h"] < -0.05

        # Monitor-derated filter initiates braking strictly earlier (smaller q) and stops safely before wall!
        assert derated["first_brake_q"] is not None and unaware["first_brake_q"] is not None
        assert derated["first_brake_q"] < unaware["first_brake_q"] - 0.35
        assert derated["max_q"] <= wall_q + 1e-6
        assert derated["min_h"] >= -1e-6
        assert derated["all_verified"] is True
