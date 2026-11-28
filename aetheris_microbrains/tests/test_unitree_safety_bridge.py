"""Tests for Unitree robot profiles, Motor-PD CBF adapter, and SafeUnitreeLowLevelBridge."""

from __future__ import annotations

import struct

import numpy as np

from aetheris_microbrains.core.unitree_safety_bridge import (
    UNITREE_ROBOT_SPECS,
    MotorPDCBFAdapter,
    SafeUnitreeLowLevelBridge,
    UnitreeDDSWireShim,
    unitree_crc32,
)


def test_unitree_specs_and_crc32():
    expected = {
        "go2": 12,
        "b2": 12,
        "z1": 6,
        "h1": 19,
        "h1_2": 27,
        "g1_23dof": 23,
        "g1_29dof": 29,
        "dex3_1": 7,
        "dex3_1_r": 7,
        "dex3_1_l": 7,
    }
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
    q = np.asarray(spec.nominal_stand_q, dtype=np.float64)
    dq = np.zeros(12)
    kp = np.full(12, 60.0)
    kd = np.full(12, 2.0)
    # Extreme step jump q_des = 2.0 rad (backward calf hyper-extension + torque overflow)
    out = adapter.filter_pd_target(q=q, dq=dq, q_des_nom=np.full(12, 2.0), kp=kp, kd=kd)
    assert out["verified_safe"] is True
    assert np.all(np.abs(out["tau_applied"]) <= np.asarray(spec.tau_max) + 1e-6)
    # Go2 calf joints (indices 2, 5, 8, 11) must stay strictly in negative flexion range [-2.7227, -0.8378]
    for calf_idx in (2, 5, 8, 11):
        assert out["q_des_safe"][calf_idx] <= -0.8378 + 1e-6
        assert out["q_des_safe"][calf_idx] >= -2.7227 - 1e-6

    # Asymmetric knee limits on H1 and G1 block backward hyper-extension, and wrist torque is capped safely
    h1_spec = UNITREE_ROBOT_SPECS["h1"]
    g1_spec = UNITREE_ROBOT_SPECS["g1_29dof"]
    assert h1_spec.q_min[3] > -0.3 and h1_spec.q_max[3] > 2.0  # H1 right_knee
    assert g1_spec.q_min[3] > -0.1 and g1_spec.q_max[3] > 2.8  # G1 left_knee
    assert g1_spec.tau_max[20] <= 5.0  # G1 wrist torque must stay small vs hip (88 N*m)


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
    assert wire_msg.crc == out_h1["crc"]
    assert wire_msg.motor_cmd[9].mode == 0
    assert wire_msg.motor_cmd[9].q == 0.0
    assert wire_msg.motor_cmd[6].mode == 1
    assert wire_msg.motor_cmd[8].mode == 1

    direct_wire = h1_bridge.to_wire_lowcmd(out_h1)
    assert direct_wire.crc == wire_msg.crc
    assert direct_wire.motor_cmd[6].mode == 1


def test_h1_2_z1_dex3_limits_and_c_struct_crc_wire_parity():
    # 1. H1-2: 27-DoF explicit sequential JointIndex + asymmetric knee & 300 N*m limit
    h1_2 = UNITREE_ROBOT_SPECS["h1_2"]
    assert h1_2.joint_names[0] == "left_hip_yaw"
    assert h1_2.joint_names[3] == "left_knee"
    assert h1_2.motor_slot_map == tuple(range(27))
    assert h1_2.n_motor_slots == 27
    assert -0.15 < h1_2.q_min[3] < 0.0 and h1_2.q_max[3] > 2.1
    assert h1_2.tau_max[3] == 300.0

    # 2. Z1: shoulder joint_2 >= 0 (60 N*m) and elbow joint_3 <= 0 (30 N*m)
    z1 = UNITREE_ROBOT_SPECS["z1"]
    z1_pd = MotorPDCBFAdapter(z1, dt=0.002)
    z1_out = z1_pd.filter_pd_target(
        q=[0.0, 0.5, -0.5, 0.0, 0.0, 0.0],
        dq=[0.0] * 6,
        q_des_nom=[0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        kp=[40.0] * 6,
        kd=[2.0] * 6,
    )
    assert z1_out["q_des_safe"][1] >= -1e-6
    assert z1_out["q_des_safe"][2] <= 1e-6
    assert z1.tau_max[1] == 60.0 and z1.tau_max[2] == 30.0

    # 3. Dex3-1 right vs left asymmetric finger ranges & 1.4 N*m finger torque cap
    dex_r = UNITREE_ROBOT_SPECS["dex3_1_r"]
    dex_l = UNITREE_ROBOT_SPECS["dex3_1_l"]
    assert dex_r.tau_max[0] == 2.45 and dex_r.tau_max[3] == 1.4
    assert dex_r.q_min[3] == 0.0 and dex_r.q_max[3] > 1.5
    assert dex_l.q_max[3] == 0.0 and dex_l.q_min[3] < -1.5

    # 4. Pure-Python C-struct CRC parity with official unitree_sdk2py crc.py (<2B2x+'B3x5fI'*35) and unitree_go (<4B4IH2x+'B3x5f3I'*20+'4B55BxI')
    go2_bridge = SafeUnitreeLowLevelBridge("go2", dt=0.002, ramp_duration_s=0.004)
    go2_ls = {
        "imu_state": {"rpy": [0.0, 0.0, 0.0]},
        "motor_state": [{"q": float(q0), "dq": 0.0, "temperature": 35.0} for q0 in UNITREE_ROBOT_SPECS["go2"].nominal_stand_q],
    }
    go2_out = go2_bridge.step(go2_ls, {"q": list(UNITREE_ROBOT_SPECS["go2"].nominal_stand_q)}, now_s=0.0)
    go2_wire = go2_bridge.to_wire_lowcmd(go2_out)
    assert go2_wire.crc == go2_out["crc"]

    go_args = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(20):
        mc = go2_wire.motor_cmd[i]
        go_args.extend((mc.mode, mc.q, mc.dq, mc.tau, mc.kp, mc.kd, 0, 0, 0))
    go_args.extend([0] * 59)  # 4B bms + 55B remote/led/fan/gpio
    go_args.append(0)  # LowCmd_.reserve
    go_bytes = struct.pack("<4B4IH2x" + "B3x5f3I" * 20 + "4B55BxI", *go_args)
    assert len(go_bytes) == 808
    go_words = list(struct.unpack("<202I", go_bytes))
    assert go2_wire.crc == unitree_crc32(go_words)

    g1_bridge = SafeUnitreeLowLevelBridge("g1_29dof", dt=0.002, ramp_duration_s=0.004, mode_pr=1)
    g1_ls = {
        "mode_machine": 5,
        "imu_state": {"rpy": [0.0, 0.0, 0.0]},
        "motor_state": [{"q": 0.0, "dq": 0.0, "temperature": 35.0} for _ in range(35)],
    }
    g1_out = g1_bridge.step(g1_ls, {"q": [0.1] * 29}, now_s=0.0)
    g1_wire = g1_bridge.to_wire_lowcmd(g1_out)
    assert g1_wire.crc == g1_out["crc"]

    hg_args = [g1_wire.mode_pr, g1_wire.mode_machine]
    for i in range(35):
        mc = g1_wire.motor_cmd[i]
        hg_args.extend((mc.mode, mc.q, mc.dq, mc.tau, mc.kp, mc.kd, 0))
    hg_args.extend((0, 0, 0, 0))
    hg_bytes = struct.pack("<2B2x" + "B3x5fI" * 35 + "4I", *hg_args)
    assert len(hg_bytes) == 1000
    hg_words = list(struct.unpack("<250I", hg_bytes))
    assert g1_wire.crc == unitree_crc32(hg_words)
