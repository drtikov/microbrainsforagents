"""Unitree Robotics hardware profiles, Motor-PD CBF adapter, and safe low-level bridge.

Supports Go2 (12 DoF), B2 (12 DoF), Z1 (6 DoF), H1 (19 DoF), H1-2 (27 DoF),
G1 (23/29 DoF), and Dex3-1 (7 DoF) with zero mandatory CycloneDDS/SDK2 dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
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


def _make_spec(
    name: str,
    joint_names: Sequence[str],
    q_min: float,
    q_max: float,
    dq_max: float,
    tau_max: float,
    kp_max: float,
    kd_damp: float,
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
        q_min=(float(q_min),) * n,
        q_max=(float(q_max),) * n,
        dq_max=(float(dq_max),) * n,
        tau_max=(float(tau_max),) * n,
        kp_safe_max=(float(kp_max),) * n,
        kd_damping_default=(float(kd_damp),) * n,
        nominal_stand_q=tuple(float(v) for v in stand_q) if stand_q is not None else (0.0,) * n,
        idl_family=idl_family,
        pr_ab_supported=pr_ab_supported,
        fall_tilt_rad=float(fall_tilt_rad),
        motor_slot_map=tuple(int(s) for s in motor_slot_map) if motor_slot_map is not None else None,
        n_motor_slots=int(n_motor_slots) if n_motor_slots is not None else None,
    )


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

UNITREE_ROBOT_SPECS: Dict[str, UnitreeRobotSpec] = {
    "go2": _make_spec("go2", _GO2_JOINTS, -2.72, 2.72, 30.0, 23.7, 80.0, 2.0, "unitree_go", False, _GO2_STAND),
    "b2": _make_spec("b2", _GO2_JOINTS, -2.8, 2.8, 23.0, 140.0, 200.0, 5.0, "unitree_go", False, _GO2_STAND),
    "z1": _make_spec("z1", tuple(f"joint_{i}" for i in range(1, 7)), -2.8, 2.8, 3.14, 30.0, 60.0, 2.5, "unitree_go", False),
    "h1": _make_spec(
        "h1",
        _H1_JOINTS,
        -2.8,
        2.8,
        20.0,
        200.0,
        200.0,
        5.0,
        "unitree_hg",
        False,
        motor_slot_map=_H1_MOTOR_SLOT_MAP,
        n_motor_slots=20,
    ),
    "h1_2": _make_spec("h1_2", tuple(f"joint_{i}" for i in range(27)), -3.0, 3.0, 20.0, 200.0, 200.0, 5.0, "unitree_hg", True),
    "g1_23dof": _make_spec("g1_23dof", _G1_23_JOINTS, -2.8, 2.8, 25.0, 88.0, 150.0, 4.0, "unitree_hg", True),
    "g1_29dof": _make_spec("g1_29dof", _G1_29_JOINTS, -2.8, 2.8, 25.0, 88.0, 150.0, 4.0, "unitree_hg", True),
    "dex3_1": _make_spec("dex3_1", ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"), -0.1, 1.75, 12.0, 2.5, 15.0, 0.3, "unitree_hg", False, fall_tilt_rad=3.14),
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


def pack_lowcmd_crc_words(
    q: Sequence[float],
    dq: Sequence[float],
    tau: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    mode_pr: int = 0,
    mode_machine: int = 0,
) -> List[int]:
    """Pack LowCmd fields into deterministic uint32 words for CRC32."""
    words: List[int] = [int(mode_pr) & 0xFF, int(mode_machine) & 0xFF]
    for arr in (q, dq, tau, kp, kd):
        for v in arr:
            words.append(int(round(float(v) * 1000.0)) & 0xFFFFFFFF)
    return words


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
    """4-state lifecycle interlock and safety bridge for Unitree rt/lowcmd."""

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
        self.lifecycle_state: str = "UNINITIALIZED"
        self.q_init: Optional[np.ndarray] = None
        self.q_des_prev: Optional[np.ndarray] = None
        self.mode_machine: int = 0
        self.ramp_elapsed_s: float = 0.0
        self.last_lowstate_s: Optional[float] = None
        self.e_stopped: bool = False

    def e_stop(self) -> None:
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

    def step(
        self,
        low_state: Optional[Any],
        nominal_cmd: Mapping[str, Any],
        now_s: Optional[float] = None,
        publisher: Optional[Any] = None,
    ) -> Dict[str, Any]:
        t_now = time.monotonic() if now_s is None else float(now_s)
        n = self.spec.n_dof

        if low_state is None:
            if self.last_lowstate_s is not None and (t_now - self.last_lowstate_s) * 1000.0 > self.watchdog_ttl_ms:
                self.lifecycle_state = "DAMPING_STOP"
            return self._build_damping_cmd("no_lowstate" if self.lifecycle_state == "UNINITIALIZED" else "watchdog_expired", publisher)

        q, dq, temps, mm, roll, pitch = self._extract_lowstate(low_state)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and math.isfinite(roll) and math.isfinite(pitch)):
            self.lifecycle_state = "DAMPING_STOP"
            return self._build_damping_cmd("non_finite_lowstate", publisher)

        if self.last_lowstate_s is not None and (t_now - self.last_lowstate_s) * 1000.0 > self.watchdog_ttl_ms:
            self.lifecycle_state = "DAMPING_STOP"
            return self._build_damping_cmd("watchdog_expired", publisher)

        self.last_lowstate_s = t_now
        if abs(roll) > self.spec.fall_tilt_rad or abs(pitch) > self.spec.fall_tilt_rad or self.e_stopped:
            self.lifecycle_state = "DAMPING_STOP"
            return self._build_damping_cmd("fall_tilt_or_estop", publisher)

        if self.lifecycle_state == "UNINITIALIZED":
            self.q_init = q.copy()
            self.q_des_prev = q.copy()
            self.mode_machine = mm
            self.ramp_elapsed_s = 0.0
            self.lifecycle_state = "HOMING_RAMP"
        elif self.lifecycle_state == "DAMPING_STOP":
            return self._build_damping_cmd("damping_latched", publisher)

        q_nom = np.asarray(nominal_cmd.get("q", q), dtype=np.float64).flatten()[:n]
        dq_nom = np.asarray(nominal_cmd.get("dq", np.zeros(n)), dtype=np.float64).flatten()[:n]
        tau_nom = np.asarray(nominal_cmd.get("tau", np.zeros(n)), dtype=np.float64).flatten()[:n]
        kp_nom = np.minimum(np.asarray(self.spec.kp_safe_max), np.maximum(0.0, np.asarray(nominal_cmd.get("kp", self.spec.kp_safe_max), dtype=np.float64).flatten()[:n]))
        kd_nom = np.maximum(0.0, np.asarray(nominal_cmd.get("kd", self.spec.kd_damping_default), dtype=np.float64).flatten()[:n])

        if not (np.all(np.isfinite(q_nom)) and np.all(np.isfinite(kp_nom))):
            self.lifecycle_state = "DAMPING_STOP"
            return self._build_damping_cmd("non_finite_nominal_cmd", publisher)

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

        words = pack_lowcmd_crc_words(q_safe, dq_nom, tau_nom, kp_eff, kd_nom, self.mode_pr, self.mode_machine)
        crc = unitree_crc32(words)
        out = {
            "lifecycle_state": self.lifecycle_state,
            "robot_model": self.spec.name,
            "idl_family": self.spec.idl_family,
            "mode_pr": self.mode_pr,
            "mode_machine": self.mode_machine,
            "q": q_safe.tolist(),
            "dq": dq_nom.tolist(),
            "tau": tau_nom.tolist(),
            "kp": kp_eff.tolist(),
            "kd": kd_nom.tolist(),
            "tau_applied": pd_out["tau_applied"].tolist(),
            "thermal_min_scale": float(np.min(therm_scale)),
            "verified_safe": pd_out["verified_safe"],
            "crc": crc,
            "status": "ok",
        }
        if publisher is not None and hasattr(publisher, "Write"):
            publisher.Write(out)
        return out

    def _build_damping_cmd(self, reason: str, publisher: Optional[Any]) -> Dict[str, Any]:
        n = self.spec.n_dof
        zeros = [0.0] * n
        kd = list(self.spec.kd_damping_default)
        words = pack_lowcmd_crc_words(zeros, zeros, zeros, zeros, kd, self.mode_pr, self.mode_machine)
        out = {
            "lifecycle_state": self.lifecycle_state,
            "robot_model": self.spec.name,
            "idl_family": self.spec.idl_family,
            "mode_pr": self.mode_pr,
            "mode_machine": self.mode_machine,
            "q": zeros,
            "dq": zeros,
            "tau": zeros,
            "kp": zeros,
            "kd": kd,
            "verified_safe": False,
            "crc": unitree_crc32(words),
            "status": reason,
        }
        if publisher is not None and hasattr(publisher, "Write"):
            publisher.Write(out)
        return out


class _FallbackMotorCmdIDL:
    """Lightweight duck-typed MotorCmd IDL slot used when unitree_sdk2py is not installed."""

    __slots__ = ("mode", "q", "dq", "tau", "kp", "kd", "reserve")

    def __init__(self) -> None:
        self.mode: int = 0
        self.q: float = 0.0
        self.dq: float = 0.0
        self.tau: float = 0.0
        self.kp: float = 0.0
        self.kd: float = 0.0
        self.reserve: List[int] = [0, 0, 0]


class _FallbackLowCmdIDL:
    """Lightweight duck-typed LowCmd_ IDL struct used when unitree_sdk2py is not installed."""

    def __init__(self, n_slots: int = 35) -> None:
        self.mode_pr: int = 0
        self.mode_machine: int = 0
        self.motor_cmd: List[_FallbackMotorCmdIDL] = [_FallbackMotorCmdIDL() for _ in range(n_slots)]
        self.reserve: List[int] = [0, 0, 0, 0]
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
        return _FallbackLowCmdIDL(n_slots=max(n_slots, spec.total_motor_slots))


def _f32_to_u32(val: float) -> int:
    return int.from_bytes(struct.pack("<f", float(val)), "little")


def _compute_idl_wire_crc(low_cmd_msg: Any, crc_calc: Optional[Any] = None) -> int:
    """Compute real SDK2 CRC(low_cmd_msg) if available, else deterministic float32-packed CRC32."""
    if crc_calc is not None and hasattr(crc_calc, "Crc"):
        return int(crc_calc.Crc(low_cmd_msg)) & 0xFFFFFFFF
    try:
        from unitree_sdk2py.utils.crc import CRC  # type: ignore[import-not-found]

        return int(CRC().Crc(low_cmd_msg)) & 0xFFFFFFFF
    except (ImportError, OSError) as exc:
        warnings.warn(
            f"unitree_sdk2py CRC unavailable ({exc}); using fallback float32-packed CRC32 (not hardware-wire-compatible)",
            RuntimeWarning,
            stacklevel=2,
        )
        words: List[int] = [
            int(getattr(low_cmd_msg, "mode_pr", 0)) & 0xFF,
            int(getattr(low_cmd_msg, "mode_machine", 0)) & 0xFF,
        ]
        for mc in getattr(low_cmd_msg, "motor_cmd", ()):
            words.append(int(getattr(mc, "mode", 0)) & 0xFFFFFFFF)
            words.append(_f32_to_u32(getattr(mc, "q", 0.0)))
            words.append(_f32_to_u32(getattr(mc, "dq", 0.0)))
            words.append(_f32_to_u32(getattr(mc, "tau", 0.0)))
            words.append(_f32_to_u32(getattr(mc, "kp", 0.0)))
            words.append(_f32_to_u32(getattr(mc, "kd", 0.0)))
        return unitree_crc32(words)


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
            populate_unitree_lowcmd_idl(
                cmd_or_dict,
                msg,
                spec=self.spec,
                motor_mode=self.motor_mode,
                crc_calc=self.crc_calc,
            )
        else:
            msg = cmd_or_dict
        self.last_idl_msg = msg
        return self.channel_publisher.Write(msg)
