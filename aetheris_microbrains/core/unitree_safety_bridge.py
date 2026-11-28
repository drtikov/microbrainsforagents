"""Unitree Robotics hardware profiles, Motor-PD CBF adapter, and safe low-level bridge.

Supports Go2 (12 DoF), B2 (12 DoF), Z1 (6 DoF), H1 (19 DoF), H1-2 (27 DoF),
G1 (23/29 DoF), and Dex3-1 (7 DoF) with zero mandatory CycloneDDS/SDK2 dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import threading
import time
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .state_cbf_filter import StateCBFSafetyFilter


@dataclass(frozen=True)
class UnitreeRobotSpec:
    name: str
    n_dof: int
    joint_names: Tuple[str, ...]
    q_min: Tuple[float, ...]
    q_max: Tuple[float, ...]
    dq_max: Tuple[float, ...]
    tau_max: Tuple[float, ...]
    kp_safe_max: Tuple[float, ...]
    kd_damping_default: Tuple[float, ...]
    nominal_stand_q: Tuple[float, ...]
    idl_family: str  # "unitree_go" or "unitree_hg"
    pr_ab_supported: bool = False
    fall_tilt_rad: float = 0.85
    motor_slot_map: Optional[Tuple[int, ...]] = None
    n_motor_slots: Optional[int] = None

    @property
    def effective_motor_slots(self) -> Tuple[int, ...]:
        """Hardware motor_state/motor_cmd slot index for each logical joint."""
        if self.motor_slot_map is not None:
            return self.motor_slot_map
        return tuple(range(self.n_dof))

    @property
    def total_motor_slots(self) -> int:
        """Total physical hardware motor slots in LowState/LowCmd frames."""
        if self.n_motor_slots is not None:
            return int(self.n_motor_slots)
        return self.n_dof


def _expand_per_joint(val: Union[float, Sequence[float]], n: int) -> Tuple[float, ...]:
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        seq = tuple(float(v) for v in val)
        if len(seq) != n:
            raise ValueError(f"Expected {n} per-joint values, got {len(seq)}")
        return seq
    return (float(val),) * n


def _make_spec(
    name: str,
    joint_names: Sequence[str],
    q_min: Union[float, Sequence[float]],
    q_max: Union[float, Sequence[float]],
    dq_max: Union[float, Sequence[float]],
    tau_max: Union[float, Sequence[float]],
    kp_max: Union[float, Sequence[float]],
    kd_damp: Union[float, Sequence[float]],
    idl_family: str,
    pr_ab_supported: bool = False,
    stand_q: Optional[Sequence[float]] = None,
    fall_tilt_rad: float = 0.85,
    motor_slot_map: Optional[Sequence[int]] = None,
    n_motor_slots: Optional[int] = None,
) -> UnitreeRobotSpec:
    n = len(joint_names)
    return UnitreeRobotSpec(
        name=name,
        n_dof=n,
        joint_names=tuple(joint_names),
        q_min=_expand_per_joint(q_min, n),
        q_max=_expand_per_joint(q_max, n),
        dq_max=_expand_per_joint(dq_max, n),
        tau_max=_expand_per_joint(tau_max, n),
        kp_safe_max=_expand_per_joint(kp_max, n),
        kd_damping_default=_expand_per_joint(kd_damp, n),
        nominal_stand_q=tuple(float(v) for v in stand_q) if stand_q is not None else (0.0,) * n,
        idl_family=idl_family,
        pr_ab_supported=pr_ab_supported,
        fall_tilt_rad=float(fall_tilt_rad),
        motor_slot_map=tuple(int(s) for s in motor_slot_map) if motor_slot_map is not None else None,
        n_motor_slots=int(n_motor_slots) if n_motor_slots is not None else None,
    )


def _table_bounds(
    joint_names: Sequence[str],
    rules: Mapping[str, Tuple[float, float, float, float, float, float]],
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Build per-joint (q_min, q_max, dq_max, tau_max, kp_max, kd_damp) from suffix/name lookup."""
    q_mins, q_maxs, dq_maxs, tau_maxs, kp_maxs, kd_damps = [], [], [], [], [], []
    for j in joint_names:
        matched: Optional[Tuple[float, float, float, float, float, float]] = None
        if j in rules:
            matched = rules[j]
        else:
            for key, vals in rules.items():
                if j.endswith(key):
                    matched = vals
                    break
        if matched is None:
            raise KeyError(f"Missing per-joint limit rule for {j!r}")
        lo, hi, dqm, tm, kpm, kdd = matched
        q_mins.append(lo)
        q_maxs.append(hi)
        dq_maxs.append(dqm)
        tau_maxs.append(tm)
        kp_maxs.append(kpm)
        kd_damps.append(kdd)
    return q_mins, q_maxs, dq_maxs, tau_maxs, kp_maxs, kd_damps


_GO2_JOINTS = tuple(f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf"))
_GO2_STAND = (0.0, 0.67, -1.3) * 4
_H1_JOINTS = (
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "waist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
)
# Official Unitree H1JointIndex mapping (H1_NUM_MOTOR = 20; hardware slot 9 is unused).
_H1_MOTOR_SLOT_MAP = (
    8,   # right_hip_yaw -> motor 8
    0,   # right_hip_roll -> motor 0
    1,   # right_hip_pitch -> motor 1
    2,   # right_knee -> motor 2
    11,  # right_ankle -> motor 11
    7,   # left_hip_yaw -> motor 7
    3,   # left_hip_roll -> motor 3
    4,   # left_hip_pitch -> motor 4
    5,   # left_knee -> motor 5
    10,  # left_ankle -> motor 10
    6,   # waist_yaw -> motor 6
    12, 13, 14, 15,  # right_shoulder_pitch/roll/yaw, right_elbow
    16, 17, 18, 19,  # left_shoulder_pitch/roll/yaw, left_elbow
)
_G1_23_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
)
_G1_29_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)

_GO2_LIMITS = _table_bounds(
    _GO2_JOINTS,
    {
        "_hip": (-1.0472, 1.0472, 30.1, 23.7, 80.0, 2.0),
        "_thigh": (-1.5708, 3.4907, 30.1, 23.7, 80.0, 2.0),
        # Negative-only flexion on Go2 calf prevents backward knee hyper-extension into chassis
        "_calf": (-2.7227, -0.8378, 15.7, 45.43, 80.0, 2.0),
    },
)
_B2_LIMITS = _table_bounds(
    _GO2_JOINTS,
    {
        "_hip": (-0.87, 0.87, 23.0, 140.0, 200.0, 5.0),
        "_thigh": (-0.94, 4.69, 23.0, 140.0, 200.0, 5.0),
        "_calf": (-2.82, -0.43, 16.0, 200.0, 200.0, 5.0),
    },
)
_H1_LIMITS = _table_bounds(
    _H1_JOINTS,
    {
        "_hip_yaw": (-0.43, 0.43, 23.0, 200.0, 200.0, 5.0),
        "_hip_roll": (-0.43, 0.43, 23.0, 200.0, 200.0, 5.0),
        "_hip_pitch": (-3.14, 2.53, 23.0, 200.0, 200.0, 5.0),
        # Asymmetric knee flexion prevents backward knee snap
        "_knee": (-0.26, 2.05, 14.0, 200.0, 200.0, 5.0),
        "_ankle": (-0.87, 0.52, 9.0, 40.0, 60.0, 2.5),
        "waist_yaw": (-2.35, 2.35, 23.0, 200.0, 200.0, 5.0),
        "_shoulder_pitch": (-2.87, 2.87, 9.0, 40.0, 60.0, 2.0),
        "left_shoulder_roll": (-0.34, 3.11, 9.0, 40.0, 60.0, 2.0),
        "right_shoulder_roll": (-3.11, 0.34, 9.0, 40.0, 60.0, 2.0),
        "_shoulder_yaw": (-1.3, 4.45, 20.0, 18.0, 40.0, 1.5),
        "_elbow": (-1.25, 2.61, 20.0, 18.0, 40.0, 1.5),
    },
)
_G1_LIMIT_RULES = {
    "_hip_pitch": (-2.5307, 2.8798, 32.0, 88.0, 150.0, 4.0),
    "left_hip_roll": (-0.5236, 2.9671, 32.0, 88.0, 150.0, 4.0),
    "right_hip_roll": (-2.9671, 0.5236, 32.0, 88.0, 150.0, 4.0),
    "_hip_yaw": (-2.7576, 2.7576, 32.0, 88.0, 150.0, 4.0),
    # Asymmetric knee range prevents backward hyper-extension
    "_knee": (-0.0873, 2.8798, 20.0, 88.0, 150.0, 4.0),
    "_ankle_pitch": (-0.8727, 0.5236, 37.0, 35.0, 60.0, 2.0),
    "_ankle_roll": (-0.2618, 0.2618, 37.0, 35.0, 60.0, 2.0),
    "waist_yaw": (-2.618, 2.618, 32.0, 88.0, 150.0, 4.0),
    "waist_roll": (-0.52, 0.52, 37.0, 50.0, 100.0, 3.0),
    "waist_pitch": (-0.52, 0.52, 37.0, 50.0, 100.0, 3.0),
    "_shoulder_pitch": (-3.0892, 2.6704, 37.0, 25.0, 60.0, 2.0),
    "left_shoulder_roll": (-1.5882, 2.2515, 37.0, 25.0, 60.0, 2.0),
    "right_shoulder_roll": (-2.2515, 1.5882, 37.0, 25.0, 60.0, 2.0),
    "_shoulder_yaw": (-2.618, 2.618, 37.0, 25.0, 60.0, 2.0),
    "_elbow": (-1.0472, 2.0944, 37.0, 25.0, 60.0, 2.0),
    "_wrist_roll": (-1.9722, 1.9722, 37.0, 5.0, 40.0, 1.0),
    "_wrist_pitch": (-1.6144, 1.6144, 37.0, 5.0, 40.0, 1.0),
    "_wrist_yaw": (-1.6144, 1.6144, 37.0, 5.0, 40.0, 1.0),
}
_G1_23_LIMITS = _table_bounds(_G1_23_JOINTS, _G1_LIMIT_RULES)
_G1_29_LIMITS = _table_bounds(_G1_29_JOINTS, _G1_LIMIT_RULES)

# Official Unitree Z1 (z1.urdf): joint2 strictly [0, 170 deg] & 60 Nm; joint3 strictly [-165 deg, 0]
_Z1_JOINTS = tuple(f"joint_{i}" for i in range(1, 7))
_Z1_LIMITS = _table_bounds(
    _Z1_JOINTS,
    {
        "joint_1": (-2.6180, 2.6180, 3.1415, 30.0, 60.0, 2.5),
        "joint_2": (0.0, 2.9671, 3.1415, 60.0, 80.0, 3.0),
        "joint_3": (-2.8798, 0.0, 3.1415, 30.0, 60.0, 2.5),
        "joint_4": (-1.5184, 1.5184, 3.1415, 30.0, 50.0, 2.0),
        "joint_5": (-1.3439, 1.3439, 3.1415, 30.0, 50.0, 2.0),
        "joint_6": (-2.7925, 2.7925, 3.1415, 30.0, 50.0, 2.0),
    },
)

# Official Unitree H1-2 JointIndex (0..26 sequential) and h1_2_handless.urdf limits
_H1_2_JOINTS = (
    "left_hip_yaw", "left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
_H1_2_MOTOR_SLOT_MAP = tuple(range(27))
_H1_2_LIMITS = _table_bounds(
    _H1_2_JOINTS,
    {
        "_hip_yaw": (-0.43, 0.43, 23.0, 200.0, 200.0, 5.0),
        "_hip_pitch": (-3.14, 2.5, 23.0, 200.0, 200.0, 5.0),
        "left_hip_roll": (-0.43, 3.14, 23.0, 200.0, 200.0, 5.0),
        "right_hip_roll": (-3.14, 0.43, 23.0, 200.0, 200.0, 5.0),
        "_knee": (-0.12, 2.19, 14.0, 300.0, 200.0, 5.0),
        "_ankle_pitch": (-0.8973, 0.5236, 9.0, 60.0, 80.0, 2.5),
        "_ankle_roll": (-0.2618, 0.2618, 9.0, 40.0, 80.0, 2.5),
        "waist_yaw": (-2.35, 2.35, 23.0, 200.0, 200.0, 5.0),
        "_shoulder_pitch": (-3.14, 1.57, 9.0, 40.0, 60.0, 2.0),
        "left_shoulder_roll": (-0.38, 3.4, 9.0, 40.0, 60.0, 2.0),
        "right_shoulder_roll": (-3.4, 0.38, 9.0, 40.0, 60.0, 2.0),
        "left_shoulder_yaw": (-2.66, 3.01, 20.0, 18.0, 50.0, 1.5),
        "right_shoulder_yaw": (-3.01, 2.66, 20.0, 18.0, 50.0, 1.5),
        "_elbow": (-0.95, 3.18, 20.0, 18.0, 50.0, 1.5),
        "left_wrist_roll": (-3.01, 2.75, 31.4, 19.0, 40.0, 1.0),
        "right_wrist_roll": (-2.75, 3.01, 31.4, 19.0, 40.0, 1.0),
        "_wrist_pitch": (-0.4625, 0.4625, 31.4, 19.0, 40.0, 1.0),
        "_wrist_yaw": (-1.27, 1.27, 31.4, 19.0, 40.0, 1.0),
    },
)

# Official Unitree Dex3-1 hand limits (dex3_1_r.urdf / dex3_1_l.urdf)
_DEX3_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
_DEX3_R_LIMITS = _table_bounds(
    _DEX3_JOINTS,
    {
        "thumb_0": (-1.0472, 1.0472, 3.14, 2.45, 15.0, 0.3),
        "thumb_1": (-1.0472, 0.6109, 12.0, 1.4, 15.0, 0.3),
        "thumb_2": (-1.7453, 0.0, 12.0, 1.4, 15.0, 0.3),
        "index_0": (0.0, 1.5708, 12.0, 1.4, 15.0, 0.3),
        "index_1": (0.0, 1.7453, 12.0, 1.4, 15.0, 0.3),
        "middle_0": (0.0, 1.5708, 12.0, 1.4, 15.0, 0.3),
        "middle_1": (0.0, 1.7453, 12.0, 1.4, 15.0, 0.3),
    },
)
_DEX3_L_LIMITS = _table_bounds(
    _DEX3_JOINTS,
    {
        "thumb_0": (-1.0472, 1.0472, 3.14, 2.45, 15.0, 0.3),
        "thumb_1": (-0.6109, 1.0472, 12.0, 1.4, 15.0, 0.3),
        "thumb_2": (0.0, 1.7453, 12.0, 1.4, 15.0, 0.3),
        "index_0": (-1.5708, 0.0, 12.0, 1.4, 15.0, 0.3),
        "index_1": (-1.7453, 0.0, 12.0, 1.4, 15.0, 0.3),
        "middle_0": (-1.5708, 0.0, 12.0, 1.4, 15.0, 0.3),
        "middle_1": (-1.7453, 0.0, 12.0, 1.4, 15.0, 0.3),
    },
)
_DEX3_ENVELOPE_LIMITS = _table_bounds(
    _DEX3_JOINTS,
    {
        "thumb_0": (-1.0472, 1.0472, 3.14, 2.45, 15.0, 0.3),
        "thumb_1": (-1.0472, 1.0472, 12.0, 1.4, 15.0, 0.3),
        "thumb_2": (-1.7453, 1.7453, 12.0, 1.4, 15.0, 0.3),
        "index_0": (-1.5708, 1.5708, 12.0, 1.4, 15.0, 0.3),
        "index_1": (-1.7453, 1.7453, 12.0, 1.4, 15.0, 0.3),
        "middle_0": (-1.5708, 1.5708, 12.0, 1.4, 15.0, 0.3),
        "middle_1": (-1.7453, 1.7453, 12.0, 1.4, 15.0, 0.3),
    },
)

UNITREE_ROBOT_SPECS: Dict[str, UnitreeRobotSpec] = {
    "go2": _make_spec("go2", _GO2_JOINTS, *_GO2_LIMITS, "unitree_go", False, _GO2_STAND),
    "b2": _make_spec("b2", _GO2_JOINTS, *_B2_LIMITS, "unitree_go", False, _GO2_STAND),
    "z1": _make_spec("z1", _Z1_JOINTS, *_Z1_LIMITS, "unitree_go", False, fall_tilt_rad=3.14),
    "h1": _make_spec(
        "h1",
        _H1_JOINTS,
        *_H1_LIMITS,
        "unitree_hg",
        False,
        motor_slot_map=_H1_MOTOR_SLOT_MAP,
        n_motor_slots=20,
    ),
    "h1_2": _make_spec(
        "h1_2",
        _H1_2_JOINTS,
        *_H1_2_LIMITS,
        "unitree_hg",
        True,
        motor_slot_map=_H1_2_MOTOR_SLOT_MAP,
        n_motor_slots=27,
    ),
    "g1_23dof": _make_spec("g1_23dof", _G1_23_JOINTS, *_G1_23_LIMITS, "unitree_hg", True),
    "g1_29dof": _make_spec("g1_29dof", _G1_29_JOINTS, *_G1_29_LIMITS, "unitree_hg", True),
    "dex3_1": _make_spec("dex3_1", _DEX3_JOINTS, *_DEX3_ENVELOPE_LIMITS, "unitree_hg", False, fall_tilt_rad=3.14),
    "dex3_1_r": _make_spec("dex3_1_r", _DEX3_JOINTS, *_DEX3_R_LIMITS, "unitree_hg", False, fall_tilt_rad=3.14),
    "dex3_1_l": _make_spec("dex3_1_l", _DEX3_JOINTS, *_DEX3_L_LIMITS, "unitree_hg", False, fall_tilt_rad=3.14),
}


def unitree_crc32(data_words: Sequence[int]) -> int:
    """Compute Unitree SDK2 32-bit word CRC32 (polynomial 0x04C11DB7)."""
    crc = 0xFFFFFFFF
    poly = 0x04C11DB7
    for word in data_words:
        w = int(word) & 0xFFFFFFFF
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if w & xbit:
                crc ^= poly
            xbit >>= 1
    return crc & 0xFFFFFFFF


def _bytes_to_u32_words(packed: bytes) -> List[int]:
    """Convert little-endian packed bytes (multiple of 4) into uint32 words."""
    n_words = len(packed) // 4
    return list(struct.unpack(f"<{n_words}I", packed[: n_words * 4]))


def _pack_hg_lowcmd_bytes(
    mode_pr: int,
    mode_machine: int,
    slots: Sequence[Tuple[int, float, float, float, float, float, int]],
    lowcmd_reserve: Sequence[int] = (0, 0, 0, 0),
) -> bytes:
    """Pack unitree_hg::msg::dds_::LowCmd_ pre-CRC payload (<2B2x + B3x5fI*N + 4I)."""
    args: List[Any] = [int(mode_pr) & 0xFF, int(mode_machine) & 0xFF]
    for m, qv, dqv, tv, kpv, kdv, res in slots:
        args.extend((int(m) & 0xFF, float(qv), float(dqv), float(tv), float(kpv), float(kdv), int(res) & 0xFFFFFFFF))
    args.extend(int(lowcmd_reserve[i]) & 0xFFFFFFFF if i < len(lowcmd_reserve) else 0 for i in range(4))
    return struct.pack(f"<2B2x{'B3x5fI' * len(slots)}4I", *args)


def _pack_go_lowcmd_bytes(
    slots: Sequence[Tuple[int, float, float, float, float, float, Tuple[int, int, int]]],
    *,
    head: Sequence[int] = (0, 0),
    level_flag: int = 0,
    frame_reserve: int = 0,
    sn: Sequence[int] = (0, 0),
    version: Sequence[int] = (0, 0),
    bandwidth: int = 0,
    bms_off: int = 0,
    bms_reserve: Sequence[int] = (0, 0, 0),
    wireless_remote: Sequence[int] = (),
    led: Sequence[int] = (),
    fan: Sequence[int] = (0, 0),
    gpio: int = 0,
    lowcmd_reserve: int = 0,
) -> bytes:
    """Pack unitree_go::msg::dds_::LowCmd_ pre-CRC payload (<4B4IH2x + B3x5f3I*N + 4B55BxI)."""
    args: List[Any] = [
        int(head[0]) & 0xFF if len(head) > 0 else 0,
        int(head[1]) & 0xFF if len(head) > 1 else 0,
        int(level_flag) & 0xFF,
        int(frame_reserve) & 0xFF,
        int(sn[0]) & 0xFFFFFFFF if len(sn) > 0 else 0,
        int(sn[1]) & 0xFFFFFFFF if len(sn) > 1 else 0,
        int(version[0]) & 0xFFFFFFFF if len(version) > 0 else 0,
        int(version[1]) & 0xFFFFFFFF if len(version) > 1 else 0,
        int(bandwidth) & 0xFFFF,
    ]
    for m, qv, dqv, tv, kpv, kdv, res in slots:
        args.extend((
            int(m) & 0xFF,
            float(qv),
            float(dqv),
            float(tv),
            float(kpv),
            float(kdv),
            int(res[0]) & 0xFFFFFFFF if len(res) > 0 else 0,
            int(res[1]) & 0xFFFFFFFF if len(res) > 1 else 0,
            int(res[2]) & 0xFFFFFFFF if len(res) > 2 else 0,
        ))
    args.append(int(bms_off) & 0xFF)
    args.extend(int(bms_reserve[i]) & 0xFF if i < len(bms_reserve) else 0 for i in range(3))
    args.extend(int(wireless_remote[i]) & 0xFF if i < len(wireless_remote) else 0 for i in range(40))
    args.extend(int(led[i]) & 0xFF if i < len(led) else 0 for i in range(12))
    args.extend(int(fan[i]) & 0xFF if i < len(fan) else 0 for i in range(2))
    args.append(int(gpio) & 0xFF)
    args.append(int(lowcmd_reserve) & 0xFFFFFFFF)
    return struct.pack(f"<4B4IH2x{'B3x5f3I' * len(slots)}4B55BxI", *args)


def pack_lowcmd_crc_words(
    q: Sequence[float],
    dq: Sequence[float],
    tau: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    mode_pr: int = 0,
    mode_machine: int = 0,
    *,
    motor_mode: int = 1,
    motor_slot_map: Optional[Sequence[int]] = None,
    n_slots: Optional[int] = None,
    idl_family: str = "unitree_hg",
) -> List[int]:
    """Pack LowCmd fields into IEEE-754 float32 + C-struct little-endian uint32 words for CRC32."""
    total_slots = int(n_slots) if n_slots is not None else (35 if idl_family == "unitree_hg" else 20)
    slots = tuple(int(s) for s in motor_slot_map) if motor_slot_map is not None else tuple(range(len(q)))
    slot_data: Dict[int, Tuple[int, float, float, float, float, float]] = {}
    for logical_idx in range(min(len(q), len(slots))):
        hw_idx = slots[logical_idx]
        if 0 <= hw_idx < total_slots:
            slot_data[hw_idx] = (
                int(motor_mode) & 0xFF,
                float(q[logical_idx]),
                float(dq[logical_idx]) if logical_idx < len(dq) else 0.0,
                float(tau[logical_idx]) if logical_idx < len(tau) else 0.0,
                float(kp[logical_idx]) if logical_idx < len(kp) else 0.0,
                float(kd[logical_idx]) if logical_idx < len(kd) else 0.0,
            )

    if idl_family == "unitree_go":
        go_slots = [(*slot_data.get(hw_idx, (0, 0.0, 0.0, 0.0, 0.0, 0.0)), (0, 0, 0)) for hw_idx in range(total_slots)]
        return _bytes_to_u32_words(_pack_go_lowcmd_bytes(go_slots))
    hg_slots = [(*slot_data.get(hw_idx, (0, 0.0, 0.0, 0.0, 0.0, 0.0)), 0) for hw_idx in range(total_slots)]
    return _bytes_to_u32_words(_pack_hg_lowcmd_bytes(mode_pr, mode_machine, hg_slots))


def _dykstra_project(
    u_nom: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iters: int = 120,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    m = A.shape[0]
    n_sets = m + 1
    s_list = [np.zeros_like(u_nom) for _ in range(n_sets)]
    w = u_nom.copy()
    for _ in range(max_iters):
        conv = True
        for i in range(m):
            y = w + s_list[i]
            ai = A[i]
            an2 = float(ai @ ai)
            viol = float(b[i]) - float(ai @ y)
            if viol > 0.0 and an2 > 1e-18:
                proj = y + (viol / an2) * ai
                conv = False
            else:
                proj = y.copy()
            s_new = y - proj
            if not np.allclose(s_new, s_list[i], atol=tol):
                conv = False
            s_list[i] = s_new
            w = proj
        y = w + s_list[m]
        proj = np.clip(y, lo, hi)
        if not np.allclose(proj, y, atol=tol):
            conv = False
        s_list[m] = y - proj
        w = proj
        if conv:
            break
    slack = float(np.min(A @ w - b)) if m > 0 else 0.0
    box_ok = bool(np.all(w >= lo - 1e-6) and np.all(w <= hi + 1e-6))
    return w, bool(np.all(np.isfinite(w)) and box_ok and (m == 0 or slack >= -1e-6))


class MotorPDCBFAdapter:
    """Translates torque-space CBFs A_tau * tau >= b_tau into safe PD position targets q_des."""

    def __init__(self, spec: UnitreeRobotSpec, dt: float = 0.002):
        self.spec = spec
        self.dt = max(1e-5, float(dt))
        self.q_min = np.asarray(spec.q_min, dtype=np.float64)
        self.q_max = np.asarray(spec.q_max, dtype=np.float64)
        self.dq_max = np.asarray(spec.dq_max, dtype=np.float64)
        self.tau_max = np.asarray(spec.tau_max, dtype=np.float64)

    def filter_pd_target(
        self,
        q: Sequence[float],
        dq: Sequence[float],
        q_des_nom: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        dq_des: Optional[Sequence[float]] = None,
        tau_ff: Optional[Sequence[float]] = None,
        A_tau: Optional[np.ndarray] = None,
        b_tau: Optional[np.ndarray] = None,
        q_des_prev: Optional[Sequence[float]] = None,
        tau_max_override: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        n = self.spec.n_dof
        q_arr = np.asarray(q, dtype=np.float64).flatten()[:n]
        dq_arr = np.asarray(dq, dtype=np.float64).flatten()[:n]
        q_nom = np.asarray(q_des_nom, dtype=np.float64).flatten()[:n]
        kp_arr = np.maximum(0.0, np.asarray(kp, dtype=np.float64).flatten()[:n])
        kd_arr = np.maximum(0.0, np.asarray(kd, dtype=np.float64).flatten()[:n])
        dq_d = np.zeros(n, dtype=np.float64) if dq_des is None else np.asarray(dq_des, dtype=np.float64).flatten()[:n]
        t_ff = np.zeros(n, dtype=np.float64) if tau_ff is None else np.asarray(tau_ff, dtype=np.float64).flatten()[:n]
        t_lim = self.tau_max if tau_max_override is None else np.asarray(tau_max_override, dtype=np.float64).flatten()[:n]

        lo = self.q_min.copy()
        hi = self.q_max.copy()
        kp_safe = np.where(kp_arr > 1e-6, kp_arr, 1e-6)
        tau_margin = np.where(kp_arr > 1e-6, t_lim / kp_safe, np.inf)
        lo = np.maximum(lo, q_arr - tau_margin)
        hi = np.minimum(hi, q_arr + tau_margin)

        if q_des_prev is not None:
            qp = np.asarray(q_des_prev, dtype=np.float64).flatten()[:n]
            step = self.dq_max * self.dt
            lo = np.maximum(lo, qp - step)
            hi = np.minimum(hi, qp + step)
        lo = np.minimum(lo, hi)

        if A_tau is not None and b_tau is not None and np.asarray(A_tau).size > 0:
            At = np.asarray(A_tau, dtype=np.float64)
            bt = np.asarray(b_tau, dtype=np.float64).flatten()
            tau_bias = t_ff - kp_arr * q_arr + kd_arr * (dq_d - dq_arr)
            A_q = At * kp_arr[None, :]
            b_q = bt - At @ tau_bias
            q_safe, ok = _dykstra_project(q_nom, A_q, b_q, lo, hi)
        else:
            q_safe = np.clip(q_nom, lo, hi)
            ok = bool(np.all(np.isfinite(q_safe)))

        tau_applied = t_ff + kp_arr * (q_safe - q_arr) + kd_arr * (dq_d - dq_arr)
        return {
            "q_des_safe": q_safe,
            "tau_applied": tau_applied,
            "verified_safe": ok,
        }


class SafeUnitreeLowLevelBridge:
    """4-state lifecycle interlock and safety bridge for Unitree rt/lowcmd.

    Supported robots
    ----------------
    Go2 (12 DoF), H1 (19 DoF), H1-2 (21 DoF), G1 (23/29 DoF),
    Z1 (6 DoF arm), Dex3-1 (7 DoF hand envelope).

    Scope boundary — Dex3-1 / DHHand
    ----------------------------------
    This bridge targets the **body motor bus** (``rt/lowcmd`` / ``rt/lowstate``)
    carried over CycloneDDS as ``unitree_hg.msg.dds_.LowCmd_`` /
    ``unitree_hg.msg.dds_.LowState_``.  The Dex3-1 hand motor array
    (``HGHandCmd_``, 7-slot ``MotorCmd_``, published on the separate topic
    ``rt/dex3/righthand/cmd`` / ``rt/dex3/lefthand/cmd``) is **out of scope**:
    joint-limit enforcement and CRC packing for hand commands require a
    dedicated ``Dex3HandBridge`` that speaks the ``HGHandCmd_`` IDL.
    Per-joint position/velocity limits for Dex3-1 fingers are defined in
    ``_DEX3_R_LIMITS`` / ``_DEX3_L_LIMITS`` above and are available for
    re-use by such a bridge; they are not applied here.
    """

    def __init__(
        self,
        robot_model: Union[str, UnitreeRobotSpec] = "g1_29dof",
        dt: float = 0.002,
        ramp_duration_s: float = 2.0,
        safety_filter: Optional[StateCBFSafetyFilter] = None,
        energy_tank: Optional[Any] = None,
        ssm_monitor: Optional[Any] = None,
        watchdog_ttl_ms: float = 100.0,
        mode_pr: int = 0,
        thermal_derate_start_c: float = 75.0,
        thermal_critical_c: float = 105.0,
    ):
        if isinstance(robot_model, str):
            key = robot_model.strip().lower()
            if key not in UNITREE_ROBOT_SPECS:
                raise ValueError(f"Unknown Unitree robot model: {robot_model!r}")
            self.spec = UNITREE_ROBOT_SPECS[key]
        else:
            self.spec = robot_model

        self.dt = max(1e-5, float(dt))
        self.ramp_duration_s = max(self.dt, float(ramp_duration_s))
        self.safety_filter = safety_filter
        self.energy_tank = energy_tank
        self.ssm_monitor = ssm_monitor
        self.watchdog_ttl_ms = max(1.0, float(watchdog_ttl_ms))
        if self.spec.pr_ab_supported and int(mode_pr) not in (0, 1):
            raise ValueError("mode_pr must be 0 (PR) or 1 (AB)")
        self.mode_pr = int(mode_pr)
        self.thermal_derate_start_c = float(thermal_derate_start_c)
        self.thermal_critical_c = max(self.thermal_derate_start_c + 1.0, float(thermal_critical_c))

        self.pd_adapter = MotorPDCBFAdapter(self.spec, dt=self.dt)
        self._lock = threading.Lock()
        self.lifecycle_state: str = "UNINITIALIZED"
        self.q_init: Optional[np.ndarray] = None
        self.q_des_prev: Optional[np.ndarray] = None
        self.mode_machine: int = 0
        self.ramp_elapsed_s: float = 0.0
        self.last_lowstate_s: Optional[float] = None
        self.last_step_s: Optional[float] = None
        self.last_packet_rx_s: Optional[float] = None
        self._last_packet_token: Optional[Tuple[Any, ...]] = None
        self._last_step_monotonic_s: Optional[float] = None
        self.step_call_count: int = 0
        self.fresh_packet_count: int = 0
        self.stale_packet_count: int = 0
        self.e_stopped: bool = False

    def e_stop(self) -> None:
        with self._lock:
            self.lifecycle_state = "DAMPING_STOP"
            self.e_stopped = True

    @staticmethod
    def _temp_to_scalar(raw: Any, default: float = 35.0) -> float:
        """Normalize a motor temperature field to scalar degrees Celsius.

        Real SDK2 IDL shapes: unitree_hg MotorState.temperature is int16[2]
        (motor, driver), unitree_go MotorState.temperature is a uint8 scalar,
        and dict-style feeds may pass either form. Takes max across array
        elements so either motor winding or MOSFET overheating triggers derating.
        """
        try:
            if isinstance(raw, (list, tuple)) or (hasattr(raw, "__len__") and not isinstance(raw, (str, bytes))):
                vals = [float(v) for v in raw if v is not None and math.isfinite(float(v))]
                return float(max(vals)) if vals else default
            val = float(raw)
            return val if math.isfinite(val) else default
        except (TypeError, ValueError):
            return default

    def _extract_lowstate(self, low_state: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
        n = self.spec.n_dof
        if isinstance(low_state, Mapping):
            motors = low_state.get("motor_state") or []
            use_slot_map = self.spec.motor_slot_map is not None and len(motors) >= self.spec.total_motor_slots
            slots = self.spec.effective_motor_slots if use_slot_map else tuple(range(n))
            q = np.array([float(motors[s]["q"]) if s < len(motors) else 0.0 for s in slots], dtype=np.float64)
            dq = np.array([float(motors[s]["dq"]) if s < len(motors) else 0.0 for s in slots], dtype=np.float64)
            temps = np.array(
                [self._temp_to_scalar(motors[s].get("temperature", 35.0)) if s < len(motors) else 35.0 for s in slots],
                dtype=np.float64,
            )
            mm = int(low_state.get("mode_machine", 0))
            rpy = (low_state.get("imu_state") or {}).get("rpy", (0.0, 0.0, 0.0))
        else:
            motors = getattr(low_state, "motor_state", ())
            use_slot_map = self.spec.motor_slot_map is not None and len(motors) >= self.spec.total_motor_slots
            slots = self.spec.effective_motor_slots if use_slot_map else tuple(range(n))
            q = np.array([float(motors[s].q) if s < len(motors) else 0.0 for s in slots], dtype=np.float64)
            dq = np.array([float(motors[s].dq) if s < len(motors) else 0.0 for s in slots], dtype=np.float64)
            temps = np.array(
                [self._temp_to_scalar(getattr(motors[s], "temperature", 35.0)) if s < len(motors) else 35.0 for s in slots],
                dtype=np.float64,
            )
            mm = int(getattr(low_state, "mode_machine", 0))
            imu = getattr(low_state, "imu_state", None)
            rpy = getattr(imu, "rpy", (0.0, 0.0, 0.0)) if imu is not None else (0.0, 0.0, 0.0)
        roll, pitch = float(rpy[0]), float(rpy[1])
        return q, dq, temps, mm, roll, pitch

    def to_wire_lowcmd(
        self,
        cmd_dict: Mapping[str, Any],
        low_cmd_msg: Optional[Any] = None,
        *,
        motor_mode: int = 1,
        crc_calc: Optional[Any] = None,
    ) -> Any:
        """Convert a bridge output dict into a CRC-signed Unitree LowCmd IDL object."""
        msg = low_cmd_msg if low_cmd_msg is not None else _allocate_lowcmd_idl(self.spec)
        return populate_unitree_lowcmd_idl(
            cmd_dict,
            msg,
            spec=self.spec,
            motor_mode=motor_mode,
            crc_calc=crc_calc,
        )

    @staticmethod
    def _get_lowstate_field(low_state: Any, name: str) -> Any:
        if isinstance(low_state, Mapping):
            return low_state.get(name)
        return getattr(low_state, name, None)

    @classmethod
    def _extract_rx_time_s(cls, low_state: Any, packet_time_s: Optional[float]) -> Optional[float]:
        """Extract explicit monotonic RX packet timestamp if provided by caller/subscriber."""
        raw = packet_time_s
        if raw is None:
            for key in ("packet_time_s", "rx_time_s", "stamp_s", "timestamp_s"):
                raw = cls._get_lowstate_field(low_state, key)
                if raw is not None:
                    break
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("nan")

    @classmethod
    def _coerce_opt_int(cls, val: Any) -> Optional[int]:
        if val is None or isinstance(val, bool):
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _build_packet_token(
        cls,
        low_state: Any,
        q: np.ndarray,
        dq: np.ndarray,
        temps: np.ndarray,
        mm: int,
        roll: float,
        pitch: float,
    ) -> Tuple[Any, ...]:
        """Build a packet-freshness fingerprint separating fresh DDS frames from frozen caches."""
        tick = cls._coerce_opt_int(cls._get_lowstate_field(low_state, "tick"))
        seq = cls._coerce_opt_int(cls._get_lowstate_field(low_state, "seq"))
        crc = cls._coerce_opt_int(cls._get_lowstate_field(low_state, "crc"))
        has_nonzero_wire_counter = any(v is not None and v != 0 for v in (tick, seq, crc))
        if has_nonzero_wire_counter:
            return ("wire", tick, seq, crc)
        return (
            "payload",
            tick,
            seq,
            crc,
            q.tobytes(),
            dq.tobytes(),
            temps.tobytes(),
            int(mm),
            round(roll, 12),
            round(pitch, 12),
        )

    def _update_packet_freshness_locked(
        self,
        low_state: Any,
        q: np.ndarray,
        dq: np.ndarray,
        temps: np.ndarray,
        mm: int,
        roll: float,
        pitch: float,
        t_now: float,
        packet_time_s: Optional[float],
    ) -> Tuple[Optional[str], bool, float]:
        """Track fresh sensor packet arrivals independently of step() call cadence.

        Returns ``(failure_reason, is_fresh, lowstate_age_ms)``.
        """
        rx_ts = self._extract_rx_time_s(low_state, packet_time_s)
        if rx_ts is not None and not math.isfinite(rx_ts):
            return "non_finite_lowstate", False, 0.0

        if self.last_lowstate_s is not None and (t_now - self.last_lowstate_s) * 1000.0 > self.watchdog_ttl_ms:
            age_ms = max(0.0, (t_now - self.last_lowstate_s) * 1000.0)
            return "watchdog_expired", False, age_ms

        if rx_ts is not None:
            rx_age_ms = (t_now - rx_ts) * 1000.0
            if rx_age_ms > self.watchdog_ttl_ms:
                return "watchdog_expired", False, max(0.0, rx_age_ms)
            is_fresh = self.last_packet_rx_s is None or rx_ts > self.last_packet_rx_s
            if is_fresh:
                self.last_packet_rx_s = rx_ts
                self.last_lowstate_s = min(t_now, rx_ts)
                self.fresh_packet_count += 1
            else:
                if self.last_lowstate_s is None:
                    self.last_lowstate_s = min(t_now, rx_ts)
                self.stale_packet_count += 1
            return None, is_fresh, max(0.0, (t_now - self.last_lowstate_s) * 1000.0)

        token = self._build_packet_token(low_state, q, dq, temps, mm, roll, pitch)
        is_fresh = self._last_packet_token is None or token != self._last_packet_token
        if is_fresh:
            self._last_packet_token = token
            self.last_lowstate_s = t_now
            self.fresh_packet_count += 1
        else:
            self.stale_packet_count += 1
        age_ms = 0.0 if self.last_lowstate_s is None else max(0.0, (t_now - self.last_lowstate_s) * 1000.0)
        return None, is_fresh, age_ms

    def step(
        self,
        low_state: Optional[Any],
        nominal_cmd: Mapping[str, Any],
        now_s: Optional[float] = None,
        publisher: Optional[Any] = None,
        packet_time_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            t_now = time.monotonic() if now_s is None else float(now_s)
            if not math.isfinite(t_now):
                self.lifecycle_state = "DAMPING_STOP"
                return self._build_damping_cmd("non_finite_lowstate", publisher, packet_fresh=False)

            self.last_step_s = t_now
            self._last_step_monotonic_s = time.monotonic()
            self.step_call_count += 1
            n = self.spec.n_dof

            if low_state is None:
                age_ms = None if self.last_lowstate_s is None else max(0.0, (t_now - self.last_lowstate_s) * 1000.0)
                if self.last_lowstate_s is not None and (t_now - self.last_lowstate_s) * 1000.0 > self.watchdog_ttl_ms:
                    self.lifecycle_state = "DAMPING_STOP"
                reason = "no_lowstate" if self.lifecycle_state == "UNINITIALIZED" else "watchdog_expired"
                return self._build_damping_cmd(reason, publisher, lowstate_age_ms=age_ms, packet_fresh=False)

            q, dq, temps, mm, roll, pitch = self._extract_lowstate(low_state)
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and math.isfinite(roll) and math.isfinite(pitch)):
                self.lifecycle_state = "DAMPING_STOP"
                return self._build_damping_cmd("non_finite_lowstate", publisher, packet_fresh=False)

            fresh_err, is_fresh, lowstate_age_ms = self._update_packet_freshness_locked(
                low_state, q, dq, temps, mm, roll, pitch, t_now, packet_time_s
            )
            if fresh_err is not None:
                self.lifecycle_state = "DAMPING_STOP"
                return self._build_damping_cmd(fresh_err, publisher, lowstate_age_ms=lowstate_age_ms, packet_fresh=False)

            if abs(roll) > self.spec.fall_tilt_rad or abs(pitch) > self.spec.fall_tilt_rad or self.e_stopped:
                self.lifecycle_state = "DAMPING_STOP"
                return self._build_damping_cmd("fall_tilt_or_estop", publisher, lowstate_age_ms=lowstate_age_ms, packet_fresh=is_fresh)

            if self.lifecycle_state == "UNINITIALIZED":
                self.q_init = q.copy()
                self.q_des_prev = q.copy()
                self.mode_machine = mm
                self.ramp_elapsed_s = 0.0
                self.lifecycle_state = "HOMING_RAMP"
            elif self.lifecycle_state == "DAMPING_STOP":
                return self._build_damping_cmd("damping_latched", publisher, lowstate_age_ms=lowstate_age_ms, packet_fresh=is_fresh)

            q_nom = np.asarray(nominal_cmd.get("q", q), dtype=np.float64).flatten()[:n]
            dq_nom = np.asarray(nominal_cmd.get("dq", np.zeros(n)), dtype=np.float64).flatten()[:n]
            tau_nom = np.asarray(nominal_cmd.get("tau", np.zeros(n)), dtype=np.float64).flatten()[:n]
            kp_nom = np.minimum(np.asarray(self.spec.kp_safe_max), np.maximum(0.0, np.asarray(nominal_cmd.get("kp", self.spec.kp_safe_max), dtype=np.float64).flatten()[:n]))
            kd_nom = np.maximum(0.0, np.asarray(nominal_cmd.get("kd", self.spec.kd_damping_default), dtype=np.float64).flatten()[:n])

            if not (np.all(np.isfinite(q_nom)) and np.all(np.isfinite(kp_nom))):
                self.lifecycle_state = "DAMPING_STOP"
                return self._build_damping_cmd("non_finite_nominal_cmd", publisher, lowstate_age_ms=lowstate_age_ms, packet_fresh=is_fresh)

            # Thermal derating from motor temperatures
            tau_max = np.asarray(self.spec.tau_max, dtype=np.float64).copy()
            denom = max(1.0, self.thermal_critical_c - self.thermal_derate_start_c)
            therm_scale = np.clip(1.0 - 0.7 * np.maximum(0.0, temps - self.thermal_derate_start_c) / denom, 0.25, 1.0)
            tau_max = tau_max * therm_scale

            # Homing ramp
            if self.lifecycle_state == "HOMING_RAMP":
                self.ramp_elapsed_s += self.dt
                s = min(1.0, self.ramp_elapsed_s / self.ramp_duration_s)
                alpha = 3.0 * s * s - 2.0 * s * s * s
                q_target = (1.0 - alpha) * self.q_init + alpha * q_nom
                kp_eff = alpha * kp_nom
                if s >= 1.0 - 1e-9:
                    self.lifecycle_state = "ACTIVE_CBF"
            else:
                q_target = q_nom
                kp_eff = kp_nom

            A_tau, b_tau = None, None
            if self.safety_filter is not None:
                x_full = np.concatenate([q, dq])
                A_tau, b_tau, _ = self.safety_filter._halfspaces(x_full)

            pd_out = self.pd_adapter.filter_pd_target(
                q=q, dq=dq, q_des_nom=q_target, kp=kp_eff, kd=kd_nom,
                dq_des=dq_nom, tau_ff=tau_nom, A_tau=A_tau, b_tau=b_tau,
                q_des_prev=self.q_des_prev, tau_max_override=tau_max,
            )
            q_safe = pd_out["q_des_safe"]
            self.q_des_prev = q_safe.copy()

            words = pack_lowcmd_crc_words(
                q_safe,
                dq_nom,
                tau_nom,
                kp_eff,
                kd_nom,
                self.mode_pr,
                self.mode_machine,
                motor_mode=1,
                motor_slot_map=self.spec.effective_motor_slots,
                n_slots=(35 if self.spec.idl_family == "unitree_hg" else 20),
                idl_family=self.spec.idl_family,
            )
            crc = unitree_crc32(words)
            out = {
                "lifecycle_state": self.lifecycle_state,
                "robot_model": self.spec.name,
                "idl_family": self.spec.idl_family,
                "mode_pr": self.mode_pr,
                "mode_machine": self.mode_machine,
                "motor_mode": 1,
                "q": q_safe.tolist(),
                "dq": dq_nom.tolist(),
                "tau": tau_nom.tolist(),
                "kp": kp_eff.tolist(),
                "kd": kd_nom.tolist(),
                "tau_applied": pd_out["tau_applied"].tolist(),
                "thermal_min_scale": float(np.min(therm_scale)),
                "verified_safe": pd_out["verified_safe"],
                "packet_fresh": is_fresh,
                "lowstate_age_ms": round(lowstate_age_ms, 4),
                "crc": crc,
                "status": "ok",
            }
            if publisher is not None and hasattr(publisher, "Write"):
                publisher.Write(out)
            return out

    def _build_damping_cmd(
        self,
        reason: str,
        publisher: Optional[Any],
        *,
        lowstate_age_ms: Optional[float] = None,
        packet_fresh: bool = False,
    ) -> Dict[str, Any]:
        n = self.spec.n_dof
        zeros = [0.0] * n
        kd = list(self.spec.kd_damping_default)
        words = pack_lowcmd_crc_words(
            zeros,
            zeros,
            zeros,
            zeros,
            kd,
            self.mode_pr,
            self.mode_machine,
            motor_mode=0,
            motor_slot_map=self.spec.effective_motor_slots,
            n_slots=(35 if self.spec.idl_family == "unitree_hg" else 20),
            idl_family=self.spec.idl_family,
        )
        out = {
            "lifecycle_state": self.lifecycle_state,
            "robot_model": self.spec.name,
            "idl_family": self.spec.idl_family,
            "mode_pr": self.mode_pr,
            "mode_machine": self.mode_machine,
            "motor_mode": 0,
            "q": zeros,
            "dq": zeros,
            "tau": zeros,
            "kp": zeros,
            "kd": kd,
            "verified_safe": False,
            "packet_fresh": packet_fresh,
            "lowstate_age_ms": None if lowstate_age_ms is None else round(float(lowstate_age_ms), 4),
            "crc": unitree_crc32(words),
            "status": reason,
        }
        if publisher is not None and hasattr(publisher, "Write"):
            publisher.Write(out)
        return out


class _FallbackMotorCmdIDL:
    """Lightweight duck-typed MotorCmd IDL slot used when unitree_sdk2py is not installed."""

    __slots__ = ("mode", "q", "dq", "tau", "kp", "kd", "reserve")

    def __init__(self, idl_family: str = "unitree_hg") -> None:
        self.mode: int = 0
        self.q: float = 0.0
        self.dq: float = 0.0
        self.tau: float = 0.0
        self.kp: float = 0.0
        self.kd: float = 0.0
        self.reserve: Union[int, List[int]] = 0 if idl_family == "unitree_hg" else [0, 0, 0]


class _FallbackBmsCmdIDL:
    """Lightweight duck-typed BmsCmd_ IDL struct for unitree_go fallback."""

    __slots__ = ("off", "reserve")

    def __init__(self) -> None:
        self.off: int = 0
        self.reserve: List[int] = [0, 0, 0]


class _FallbackLowCmdIDL:
    """Lightweight duck-typed LowCmd_ IDL struct used when unitree_sdk2py is not installed."""

    def __init__(self, n_slots: Optional[int] = None, idl_family: str = "unitree_hg") -> None:
        self.idl_family = idl_family
        count = int(n_slots) if n_slots is not None else (35 if idl_family == "unitree_hg" else 20)
        self.motor_cmd: List[_FallbackMotorCmdIDL] = [_FallbackMotorCmdIDL(idl_family) for _ in range(count)]
        if idl_family == "unitree_go":
            self.head: List[int] = [0, 0]
            self.level_flag: int = 0
            self.frame_reserve: int = 0
            self.sn: List[int] = [0, 0]
            self.version: List[int] = [0, 0]
            self.bandwidth: int = 0
            self.bms_cmd = _FallbackBmsCmdIDL()
            self.wireless_remote: List[int] = [0] * 40
            self.led: List[int] = [0] * 12
            self.fan: List[int] = [0, 0]
            self.gpio: int = 0
            self.reserve: Union[int, List[int]] = 0
        else:
            self.mode_pr: int = 0
            self.mode_machine: int = 0
            self.reserve = [0, 0, 0, 0]
        self.crc: int = 0


def _allocate_lowcmd_idl(spec: UnitreeRobotSpec) -> Any:
    """Instantiate real unitree_sdk2py LowCmd_ IDL if available, else fallback struct."""
    try:
        from unitree_sdk2py.idl.default import (  # type: ignore[import-not-found]
            unitree_go_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )

        if spec.idl_family == "unitree_hg":
            return unitree_hg_msg_dds__LowCmd_()
        return unitree_go_msg_dds__LowCmd_()
    except ImportError:
        n_slots = 35 if spec.idl_family == "unitree_hg" else 20
        return _FallbackLowCmdIDL(n_slots=max(n_slots, spec.total_motor_slots), idl_family=spec.idl_family)


def _compute_idl_wire_crc(low_cmd_msg: Any, crc_calc: Optional[Any] = None) -> int:
    """Compute real SDK2 CRC(low_cmd_msg) if available, else pure-Python IDL struct CRC32."""
    if crc_calc is not None and hasattr(crc_calc, "Crc"):
        return int(crc_calc.Crc(low_cmd_msg)) & 0xFFFFFFFF
    try:
        from unitree_sdk2py.utils.crc import CRC  # type: ignore[import-not-found]

        return int(CRC().Crc(low_cmd_msg)) & 0xFFFFFFFF
    except (ImportError, OSError) as exc:
        warnings.warn(
            f"unitree_sdk2py CRC unavailable ({exc}); using pure-Python IDL struct-packed fallback CRC32",
            RuntimeWarning,
            stacklevel=2,
        )
        is_go = (
            getattr(low_cmd_msg, "__idl_typename__", "") == "unitree_go.msg.dds_.LowCmd_"
            or getattr(low_cmd_msg, "idl_family", "") == "unitree_go"
            or hasattr(low_cmd_msg, "head")
        )
        mcs = getattr(low_cmd_msg, "motor_cmd", ())
        if is_go:
            go_slots = []
            for mc in mcs:
                res = getattr(mc, "reserve", (0, 0, 0))
                res_tup = tuple(res) if isinstance(res, (list, tuple)) else (int(res), 0, 0)
                go_slots.append((
                    int(getattr(mc, "mode", 0)) & 0xFF,
                    float(getattr(mc, "q", 0.0)),
                    float(getattr(mc, "dq", 0.0)),
                    float(getattr(mc, "tau", 0.0)),
                    float(getattr(mc, "kp", 0.0)),
                    float(getattr(mc, "kd", 0.0)),
                    (
                        int(res_tup[0]) & 0xFFFFFFFF if len(res_tup) > 0 else 0,
                        int(res_tup[1]) & 0xFFFFFFFF if len(res_tup) > 1 else 0,
                        int(res_tup[2]) & 0xFFFFFFFF if len(res_tup) > 2 else 0,
                    ),
                ))
            bms = getattr(low_cmd_msg, "bms_cmd", None)
            tail_res = getattr(low_cmd_msg, "reserve", 0)
            packed = _pack_go_lowcmd_bytes(
                go_slots,
                head=getattr(low_cmd_msg, "head", (0, 0)),
                level_flag=getattr(low_cmd_msg, "level_flag", 0),
                frame_reserve=getattr(low_cmd_msg, "frame_reserve", 0),
                sn=getattr(low_cmd_msg, "sn", (0, 0)),
                version=getattr(low_cmd_msg, "version", (0, 0)),
                bandwidth=getattr(low_cmd_msg, "bandwidth", 0),
                bms_off=getattr(bms, "off", 0) if bms is not None else 0,
                bms_reserve=getattr(bms, "reserve", (0, 0, 0)) if bms is not None else (0, 0, 0),
                wireless_remote=getattr(low_cmd_msg, "wireless_remote", ()),
                led=getattr(low_cmd_msg, "led", ()),
                fan=getattr(low_cmd_msg, "fan", (0, 0)),
                gpio=getattr(low_cmd_msg, "gpio", 0),
                lowcmd_reserve=int(tail_res[0]) if isinstance(tail_res, (list, tuple)) else int(tail_res),
            )
            return unitree_crc32(_bytes_to_u32_words(packed))

        mode_pr = int(getattr(low_cmd_msg, "mode_pr", 0)) & 0xFF
        mode_machine = int(getattr(low_cmd_msg, "mode_machine", 0)) & 0xFF
        hg_slots = []
        for mc in mcs:
            res = getattr(mc, "reserve", 0)
            res_int = int(res[0]) if isinstance(res, (list, tuple)) else int(res)
            hg_slots.append((
                int(getattr(mc, "mode", 0)) & 0xFF,
                float(getattr(mc, "q", 0.0)),
                float(getattr(mc, "dq", 0.0)),
                float(getattr(mc, "tau", 0.0)),
                float(getattr(mc, "kp", 0.0)),
                float(getattr(mc, "kd", 0.0)),
                res_int & 0xFFFFFFFF,
            ))
        tail = getattr(low_cmd_msg, "reserve", (0, 0, 0, 0))
        tail_seq = tail if isinstance(tail, (list, tuple)) else (int(tail), 0, 0, 0)
        packed = _pack_hg_lowcmd_bytes(mode_pr, mode_machine, hg_slots, tail_seq)
        return unitree_crc32(_bytes_to_u32_words(packed))


def populate_unitree_lowcmd_idl(
    cmd_dict: Mapping[str, Any],
    low_cmd_msg: Any,
    spec: Optional[Union[str, UnitreeRobotSpec]] = None,
    *,
    motor_mode: int = 1,
    crc_calc: Optional[Any] = None,
) -> Any:
    """Scatter a SafeUnitreeLowLevelBridge command dict into a Unitree LowCmd_ IDL struct with wire CRC."""
    resolved_spec: Optional[UnitreeRobotSpec]
    if isinstance(spec, UnitreeRobotSpec):
        resolved_spec = spec
    elif isinstance(spec, str):
        resolved_spec = UNITREE_ROBOT_SPECS[spec.strip().lower()]
    else:
        rm = str(cmd_dict.get("robot_model") or "").strip().lower()
        resolved_spec = UNITREE_ROBOT_SPECS.get(rm)

    if hasattr(low_cmd_msg, "mode_pr"):
        low_cmd_msg.mode_pr = int(cmd_dict.get("mode_pr", 0))
    if hasattr(low_cmd_msg, "mode_machine"):
        low_cmd_msg.mode_machine = int(cmd_dict.get("mode_machine", 0))

    q = cmd_dict.get("q") or ()
    dq = cmd_dict.get("dq") or ()
    tau = cmd_dict.get("tau") or ()
    kp = cmd_dict.get("kp") or ()
    kd = cmd_dict.get("kd") or ()
    n = len(q)

    motor_cmds = getattr(low_cmd_msg, "motor_cmd", ())
    slots = resolved_spec.effective_motor_slots if resolved_spec is not None else tuple(range(n))
    active_slots = set(slots[:n])

    # Zero/disable any unused motor slots (e.g. H1 hardware motor index 9)
    for hw_idx in range(len(motor_cmds)):
        if hw_idx not in active_slots:
            mc = motor_cmds[hw_idx]
            mc.mode = 0
            mc.q = 0.0
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = 0.0

    for logical_idx in range(min(n, len(slots))):
        hw_idx = int(slots[logical_idx])
        if hw_idx >= len(motor_cmds):
            continue
        mc = motor_cmds[hw_idx]
        mc.mode = int(motor_mode)
        mc.q = float(q[logical_idx])
        mc.dq = float(dq[logical_idx]) if logical_idx < len(dq) else 0.0
        mc.tau = float(tau[logical_idx]) if logical_idx < len(tau) else 0.0
        mc.kp = float(kp[logical_idx]) if logical_idx < len(kp) else 0.0
        mc.kd = float(kd[logical_idx]) if logical_idx < len(kd) else 0.0

    low_cmd_msg.crc = _compute_idl_wire_crc(low_cmd_msg, crc_calc=crc_calc)
    return low_cmd_msg


class UnitreeDDSWireShim:
    """Adapter wrapping a Unitree ChannelPublisher to convert bridge dicts into wire LowCmd_ IDL messages."""

    def __init__(
        self,
        channel_publisher: Any,
        robot_model: Union[str, UnitreeRobotSpec] = "g1_29dof",
        *,
        lowcmd_factory: Optional[Callable[[], Any]] = None,
        crc_calc: Optional[Any] = None,
        motor_mode: int = 1,
    ) -> None:
        if isinstance(robot_model, str):
            key = robot_model.strip().lower()
            if key not in UNITREE_ROBOT_SPECS:
                raise ValueError(f"Unknown Unitree robot model: {robot_model!r}")
            self.spec = UNITREE_ROBOT_SPECS[key]
        else:
            self.spec = robot_model
        self.channel_publisher = channel_publisher
        self.lowcmd_factory = lowcmd_factory
        self.crc_calc = crc_calc
        self.motor_mode = int(motor_mode)
        self.last_idl_msg: Optional[Any] = None

    def Write(self, cmd_or_dict: Any) -> Any:
        """Convert dict command to CRC-signed LowCmd_ IDL and forward to ChannelPublisher.Write."""
        if isinstance(cmd_or_dict, Mapping):
            msg = self.lowcmd_factory() if self.lowcmd_factory is not None else _allocate_lowcmd_idl(self.spec)
            cmd_mode = int(cmd_or_dict.get("motor_mode", self.motor_mode))
            populate_unitree_lowcmd_idl(
                cmd_or_dict,
                msg,
                spec=self.spec,
                motor_mode=cmd_mode,
                crc_calc=self.crc_calc,
            )
        else:
            msg = cmd_or_dict
        self.last_idl_msg = msg
        return self.channel_publisher.Write(msg)


class UnitreeLowStateBuffer:
    """Subscriber-side DDS freshness buffer pairing with SafeUnitreeLowLevelBridge.

    Register ``buffer.on_low_state`` with Unitree SDK2's ``ChannelSubscriber``
    and call ``buffer.step_bridge(...)`` inside the 500 Hz control thread so
    sensor arrival time is tracked at DDS RX independently of control-loop
    call cadence.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._latest: Optional[Tuple[int, float, Any]] = None
        self._rx_seq: int = 0

    def on_low_state(self, msg: Any, rx_time_s: Optional[float] = None) -> None:
        """CycloneDDS ChannelSubscriber callback storing stamped sensor packets."""
        if msg is None:
            return
        rx_s = float(self._clock() if rx_time_s is None else rx_time_s)
        self._rx_seq += 1
        self._latest = (self._rx_seq, rx_s, msg)

    def latest(self) -> Tuple[Optional[Any], Optional[float], int]:
        """Return ``(msg, rx_time_s, rx_seq)`` for the newest received LowState."""
        snap = self._latest
        if snap is None:
            return None, None, 0
        return snap[2], snap[1], snap[0]

    def age_ms(self, now_s: Optional[float] = None) -> Optional[float]:
        """Milliseconds elapsed since the latest DDS LowState packet arrived."""
        snap = self._latest
        if snap is None:
            return None
        t_now = float(self._clock() if now_s is None else now_s)
        return max(0.0, (t_now - snap[1]) * 1000.0)

    def step_bridge(
        self,
        bridge: SafeUnitreeLowLevelBridge,
        nominal_cmd: Mapping[str, Any],
        *,
        now_s: Optional[float] = None,
        publisher: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Step ``bridge`` with the latest buffered packet and its RX timestamp."""
        t_now = float(self._clock() if now_s is None else now_s)
        snap = self._latest
        msg = snap[2] if snap is not None else None
        rx_s = snap[1] if snap is not None else None
        return bridge.step(
            msg,
            nominal_cmd,
            now_s=t_now,
            publisher=publisher,
            packet_time_s=rx_s,
        )


class UnitreeExternalWatchdog:
    """Out-of-loop supervisor thread enforcing damping if the Python control loop stalls."""

    def __init__(
        self,
        bridge: SafeUnitreeLowLevelBridge,
        publisher: Optional[Any] = None,
        *,
        timeout_ms: Optional[float] = None,
        check_interval_s: float = 0.005,
        keepalive_damping: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bridge = bridge
        self.publisher = publisher
        self.timeout_ms = max(1.0, float(bridge.watchdog_ttl_ms if timeout_ms is None else timeout_ms))
        self.check_interval_s = max(0.001, float(check_interval_s))
        self.keepalive_damping = bool(keepalive_damping)
        self._clock = clock
        self._stop_event = threading.Event()
        self._tripped_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._arm_time_s: float = 0.0
        self._arm_step_count: int = 0
        self.tripped: bool = False
        self.trip_count: int = 0
        self.last_damping_cmd: Optional[Dict[str, Any]] = None

    def start(self) -> "UnitreeExternalWatchdog":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._tripped_event.clear()
        self.tripped = False
        self.trip_count = 0
        self.last_damping_cmd = None
        with self.bridge._lock:
            self._arm_time_s = float(self._clock())
            self._arm_step_count = self.bridge.step_call_count
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="unitree-external-watchdog",
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        t = self._thread
        self._thread = None
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    def wait_tripped(self, timeout: float = 1.0) -> bool:
        return self._tripped_event.wait(timeout=timeout)

    def check_once(self, now_s: Optional[float] = None) -> bool:
        """Check whether the monitored control loop missed its heartbeat deadline."""
        now = float(self._clock() if now_s is None else now_s)
        with self.bridge._lock:
            if (
                self.bridge._last_step_monotonic_s is not None
                and self.bridge.step_call_count > self._arm_step_count
            ):
                ref_s = max(self._arm_time_s, self.bridge._last_step_monotonic_s)
            else:
                ref_s = self._arm_time_s
            age_ms = max(0.0, (now - ref_s) * 1000.0)
            if age_ms <= self.timeout_ms:
                return False
            self.bridge.lifecycle_state = "DAMPING_STOP"
            if not self.tripped or self.keepalive_damping:
                self.last_damping_cmd = self.bridge._build_damping_cmd(
                    "loop_watchdog_expired",
                    self.publisher,
                    lowstate_age_ms=age_ms,
                    packet_fresh=False,
                )
            self.tripped = True
            self.trip_count += 1
            self._tripped_event.set()
            return True

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.check_interval_s):
            self.check_once()
            if self.tripped and not self.keepalive_damping:
                return

    def __enter__(self) -> "UnitreeExternalWatchdog":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
