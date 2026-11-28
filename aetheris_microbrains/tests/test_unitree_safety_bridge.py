"""Tests for Unitree robot profiles, Motor-PD CBF adapter, and SafeUnitreeLowLevelBridge."""

from __future__ import annotations

import numpy as np

from aetheris_microbrains.core.unitree_safety_bridge import (
    UNITREE_ROBOT_SPECS,
    MotorPDCBFAdapter,
    SafeUnitreeLowLevelBridge,
    UnitreeDDSWireShim,
    unitree_crc32,
)


def test_unitree_specs_and_crc32():
    expected = {"go2": 12, "b2": 12, "z1": 6, "h1": 19, "h1_2": 27, "g1_23dof": 23, "g1_29dof": 29, "dex3_1": 7}
    for name, dof in expected.items():
        assert UNITREE_ROBOT_SPECS[name].n_dof == dof
        assert len(UNITREE_ROBOT_SPECS[name].joint_names) == dof
    c1 = unitree_crc32([1, 2, 3, 4])
    c2 = unitree_crc32([1, 2, 3, 5])
    assert c1 == unitree_crc32([1, 2, 3, 4])
    assert c1 != c2


def test_motor_pd_cbf_clamps_torque_and_halfspaces():
    spec = UNITREE_ROBOT_SPECS["go2"]
    adapter = MotorPDCBFAdapter(spec, dt=0.002)
    q = np.zeros(12)
    dq = np.zeros(12)
    kp = np.full(12, 60.0)
    kd = np.full(12, 2.0)
    # Extreme step jump q_des = 2.0 rad -> 120 N*m > 23.7 N*m tau_max
    out = adapter.filter_pd_target(q=q, dq=dq, q_des_nom=np.full(12, 2.0), kp=kp, kd=kd)
    assert out["verified_safe"] is True
    assert np.all(np.abs(out["tau_applied"]) <= np.asarray(spec.tau_max) + 1e-6)


def test_safe_unitree_bridge_homing_thermal_and_watchdog():
    bridge = SafeUnitreeLowLevelBridge("g1_29dof", dt=0.002, ramp_duration_s=0.01, watchdog_ttl_ms=50.0, mode_pr=1)
    assert bridge.lifecycle_state == "UNINITIALIZED"

    # Before lowstate -> damping
    r0 = bridge.step(None, {"q": [1.0] * 29}, now_s=0.0)
    assert r0["lifecycle_state"] == "UNINITIALIZED"
    assert all(k == 0.0 for k in r0["kp"])

    low_state = {
        "mode_machine": 3,
        "imu_state": {"rpy": [0.0, 0.0, 0.0]},
        "motor_state": [{"q": 0.0, "dq": 0.0, "temperature": 95.0} for _ in range(29)],
    }
    r1 = bridge.step(low_state, {"q": [1.0] * 29}, now_s=0.01)
    assert r1["lifecycle_state"] == "HOMING_RAMP"
    assert r1["mode_machine"] == 3
    assert r1["mode_pr"] == 1
    assert r1["thermal_min_scale"] < 0.6

    # Advance past ramp
    for i in range(1, 8):
        r = bridge.step(low_state, {"q": [0.2] * 29}, now_s=0.01 + i * 0.002)
    assert r["lifecycle_state"] == "ACTIVE_CBF"

    # Watchdog expiry -> DAMPING_STOP
    r_wd = bridge.step(low_state, {"q": [0.2] * 29}, now_s=0.20)
    assert r_wd["lifecycle_state"] == "DAMPING_STOP"
    assert r_wd["status"] == "watchdog_expired"
    assert all(k == 0.0 for k in r_wd["kp"])
    assert r_wd["kd"] == list(UNITREE_ROBOT_SPECS["g1_29dof"].kd_damping_default)


class _FakeMotorStateIDL:
    def __init__(self, q: float = 0.0, dq: float = 0.0, temperature=(35, 35)):
        self.q = q
        self.dq = dq
        self.temperature = temperature


class _FakeHGLowStateIDL:
    def __init__(self, n_slots: int = 35, temperature=(45, 20)):
        self.mode_pr = 1
        self.mode_machine = 3
        self.imu_state = type("IMU", (), {"rpy": [0.0, 0.0, 0.0]})()
        self.motor_state = [_FakeMotorStateIDL(temperature=list(temperature)) for _ in range(n_slots)]


class _CapturingChannelPublisher:
    def __init__(self):
        self.messages = []

    def Write(self, msg):
        self.messages.append(msg)


def test_hg_int16_pair_temperature_and_h1_motor_slot_remap_and_wire_shim():
    # 1. HG IDL int16[2] temperature (list) must not crash _extract_lowstate
    bridge = SafeUnitreeLowLevelBridge("g1_29dof", dt=0.002, ramp_duration_s=0.006, mode_pr=1)
    ls_cool = _FakeHGLowStateIDL(35, temperature=(45, 20))
    r1 = bridge.step(ls_cool, {"q": [0.1] * 29}, now_s=0.0)
    assert r1["status"] == "ok"
    assert r1["thermal_min_scale"] == 1.0

    # Driver/MOSFET overheating in second slot of int16[2] must still derate
    ls_hot_driver = _FakeHGLowStateIDL(35, temperature=(35, 95))
    r2 = bridge.step(ls_hot_driver, {"q": [0.1] * 29}, now_s=0.002)
    assert r2["thermal_min_scale"] < 0.6

    # 2. H1 20-slot scatter map (motor index 9 unused; waist_yaw at slot 6; right_hip_yaw at slot 8)
    h1_spec = UNITREE_ROBOT_SPECS["h1"]
    assert h1_spec.total_motor_slots == 20
    assert h1_spec.effective_motor_slots[0] == 8  # right_hip_yaw -> slot 8
    assert h1_spec.effective_motor_slots[10] == 6  # waist_yaw -> slot 6
    assert 9 not in h1_spec.effective_motor_slots

    h1_ls = _FakeHGLowStateIDL(35, temperature=(35, 35))
    h1_ls.motor_state[8].q = 0.25  # physical right_hip_yaw
    h1_ls.motor_state[9].q = 99.0  # unused hardware slot should be ignored
    h1_bridge = SafeUnitreeLowLevelBridge("h1", dt=0.002, ramp_duration_s=0.004)
    chan_pub = _CapturingChannelPublisher()
    shim = UnitreeDDSWireShim(chan_pub, "h1")
    cmd_q = [0.0] * 19
    cmd_q[10] = 0.15  # logical waist_yaw
    out_h1 = h1_bridge.step(h1_ls, {"q": cmd_q}, now_s=0.0, publisher=shim)
    assert abs(h1_bridge.q_init[0] - 0.25) < 1e-9
    assert out_h1["status"] == "ok"
    assert len(chan_pub.messages) == 1
    wire_msg = chan_pub.messages[0]
    assert not isinstance(wire_msg, dict)
    assert wire_msg.crc != 0
    assert wire_msg.motor_cmd[9].mode == 0
    assert wire_msg.motor_cmd[9].q == 0.0
    assert wire_msg.motor_cmd[6].mode == 1
    assert wire_msg.motor_cmd[8].mode == 1
