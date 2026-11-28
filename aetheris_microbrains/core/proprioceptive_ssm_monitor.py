"""Proprioceptive high-frequency State-Space Model (SSM) health & safety monitor.

Runs a dedicated per-monitor ``LinearStateSpaceMicroBrain`` (Mamba-3 complex
rotational ZOH/trapezoidal recurrence) in strict ``O(1)`` time and ``O(1)``
memory per 1 kHz control tick — without touching the global SSM singleton — to
classify proprioceptive residual streams into actionable safety states:

  - ``"nominal"``
  - ``"incipient_gripper_slip"`` -> bounded ``grasp_force_delta_n > 0``
  - ``"joint_backlash_or_cable_stretch"`` -> per-joint covariance inflation for
    ``DirectionalBarrierInflator``
  - ``"actuator_thermal_or_torque_degradation"`` -> per-joint ``torque_scale``
    derating in ``[min_authority, 1.0]`` pushed into ``StateCBFSafetyFilter``
    via ``apply_derating_to_filter``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .adaptive_conformal_robotics import DirectionalBarrierInflator
from .linear_state_space_microbrain import LinearStateSpaceMicroBrain
from .state_cbf_filter import BarrierConstraint, HighOrderBarrier, StateCBFSafetyFilter


def _as_joint_vec(
    val: Optional[Union[float, Sequence[float], np.ndarray]],
    n_joints: int,
    default: float = 0.0,
) -> np.ndarray:
    if val is None:
        return np.full(n_joints, float(default), dtype=np.float64)
    arr = np.asarray(val, dtype=np.float64).flatten()
    if arr.size == 1:
        return np.full(n_joints, float(arr[0]), dtype=np.float64)
    if arr.shape[0] != n_joints:
        raise ValueError(f"Expected vector of length {n_joints} (or scalar), got shape {arr.shape}")
    return arr


class ProprioceptiveSSMMonitor:
    """1 kHz O(1)-memory proprioceptive fault, slip, backlash, and derating monitor."""

    HEALTH_NOMINAL = "nominal"
    HEALTH_SLIP = "incipient_gripper_slip"
    HEALTH_BACKLASH = "joint_backlash_or_cable_stretch"
    HEALTH_DEGRADATION = "actuator_thermal_or_torque_degradation"

    def __init__(
        self,
        n_joints: int = 6,
        d_model: Optional[int] = None,
        d_state: int = 16,
        dt: float = 0.001,
        min_authority: float = 0.15,
        nominal_max_torque: Union[float, Sequence[float]] = 10.0,
        nominal_grasp_force_n: float = 5.0,
        max_grasp_force_n: float = 30.0,
        slip_hf_threshold: float = 0.35,
        slip_gain_n: float = 12.0,
        backlash_phase_threshold: float = 0.20,
        backlash_osc_min: float = 0.04,
        covariance_inflation_gain: float = 0.30,
        temp_nominal_c: float = 35.0,
        temp_derate_start_c: float = 75.0,
        temp_critical_c: float = 115.0,
        derating_detect_threshold: float = 0.88,
    ):
        if int(n_joints) < 1:
            raise ValueError("n_joints must be >= 1")
        self.n_joints = int(n_joints)
        self.dt = max(1e-5, float(dt))
        self.min_authority = float(np.clip(min_authority, 0.01, 1.0))

        self.nominal_max_torque = np.maximum(
            1e-3,
            _as_joint_vec(nominal_max_torque, self.n_joints, default=10.0),
        )
        self.nominal_grasp_force_n = max(0.0, float(nominal_grasp_force_n))
        self.max_grasp_force_n = max(self.nominal_grasp_force_n, float(max_grasp_force_n))

        self.slip_hf_threshold = max(1e-4, float(slip_hf_threshold))
        self.slip_gain_n = max(0.0, float(slip_gain_n))
        self.backlash_phase_threshold = float(np.clip(backlash_phase_threshold, 0.02, 0.95))
        self.backlash_osc_min = max(1e-5, float(backlash_osc_min))
        self.covariance_inflation_gain = max(0.0, float(covariance_inflation_gain))

        self.temp_nominal_c = float(temp_nominal_c)
        self.temp_derate_start_c = float(temp_derate_start_c)
        self.temp_critical_c = max(self.temp_derate_start_c + 1.0, float(temp_critical_c))
        self.derating_detect_threshold = float(np.clip(derating_detect_threshold, self.min_authority, 0.99))

        resolved_d_model = max(16, 5 * self.n_joints + 4) if d_model is None else max(8, int(d_model))
        resolved_d_state = max(4, int(d_state))

        # Dedicated per-monitor SSM instance (never touches LinearStateSpaceMicroBrain._instance)
        self.ssm = LinearStateSpaceMicroBrain(d_model=resolved_d_model, d_state=resolved_d_state)

        # Pre-allocated fixed-size O(1) recurrent buffers
        self._u_buf = np.zeros(self.ssm.d_model, dtype=np.float32)
        self._tau_mean = np.zeros(self.n_joints, dtype=np.float64)
        self._qdot_mean = np.zeros(self.n_joints, dtype=np.float64)
        self._tau_ac_fast = np.zeros(self.n_joints, dtype=np.float64)
        self._tau_ac_lag = np.zeros(self.n_joints, dtype=np.float64)
        self._qdot_ac_fast = np.zeros(self.n_joints, dtype=np.float64)
        self._qdot_ac_lag = np.zeros(self.n_joints, dtype=np.float64)
        self._tau_var_ema = np.zeros(self.n_joints, dtype=np.float64)
        self._qdot_var_ema = np.zeros(self.n_joints, dtype=np.float64)
        self._vel_err_var_ema = np.zeros(self.n_joints, dtype=np.float64)
        self._quad_wronskian_ema = np.zeros(self.n_joints, dtype=np.float64)
        self._torque_ratio_ema = np.ones(self.n_joints, dtype=np.float64)
        self._tactile_hf_ema: float = 0.0

        # Current actionable outputs (fixed-size O(1) vectors)
        self.torque_scale = np.ones(self.n_joints, dtype=np.float64)
        self.covariance_inflation = np.zeros(self.n_joints, dtype=np.float64)
        self.state_covariance_inflation = np.zeros(2 * self.n_joints, dtype=np.float64)
        self.grasp_force_delta_n: float = 0.0
        self.health_status: str = self.HEALTH_NOMINAL
        self.ticks: int = 0

        # Baseline HOCBF alpha cache keyed by barrier object id so repeated derating is idempotent
        self._nominal_barrier_alphas: Dict[int, Tuple[float, float, float]] = {}

    def reset(self) -> None:
        """Reset SSM hidden state and all fixed-size O(1) residual statistics."""
        self.ssm.reset()
        self._u_buf.fill(0.0)
        self._tau_mean.fill(0.0)
        self._qdot_mean.fill(0.0)
        self._tau_ac_fast.fill(0.0)
        self._tau_ac_lag.fill(0.0)
        self._qdot_ac_fast.fill(0.0)
        self._qdot_ac_lag.fill(0.0)
        self._tau_var_ema.fill(0.0)
        self._qdot_var_ema.fill(0.0)
        self._vel_err_var_ema.fill(0.0)
        self._quad_wronskian_ema.fill(0.0)
        self._torque_ratio_ema.fill(1.0)
        self._tactile_hf_ema = 0.0
        self.torque_scale.fill(1.0)
        self.covariance_inflation.fill(0.0)
        self.state_covariance_inflation.fill(0.0)
        self.grasp_force_delta_n = 0.0
        self.health_status = self.HEALTH_NOMINAL
        self.ticks = 0

    def state_memory_bytes(self) -> int:
        """Exact byte count of recurrent SSM and EMA buffers (constant across all ticks)."""
        arrays = (
            self.ssm.hidden_state,
            self.ssm.prev_u,
            self._u_buf,
            self._tau_mean,
            self._qdot_mean,
            self._tau_ac_fast,
            self._tau_ac_lag,
            self._qdot_ac_fast,
            self._qdot_ac_lag,
            self._tau_var_ema,
            self._qdot_var_ema,
            self._vel_err_var_ema,
            self._quad_wronskian_ema,
            self._torque_ratio_ema,
            self.torque_scale,
            self.covariance_inflation,
            self.state_covariance_inflation,
        )
        return int(sum(arr.nbytes for arr in arrays))

    def step(
        self,
        *,
        tau_cmd: Optional[Sequence[float]] = None,
        tau_meas: Optional[Sequence[float]] = None,
        q_des: Optional[Sequence[float]] = None,
        q: Optional[Sequence[float]] = None,
        qdot_des: Optional[Sequence[float]] = None,
        qdot: Optional[Sequence[float]] = None,
        torque_error: Optional[Sequence[float]] = None,
        pos_error: Optional[Sequence[float]] = None,
        vel_error: Optional[Sequence[float]] = None,
        winding_temp_c: Optional[Union[float, Sequence[float]]] = None,
        tactile_hf_energy: float = 0.0,
        tangential_slip_rate: Optional[float] = None,
        current_grasp_force_n: Optional[float] = None,
        inflator: Optional[DirectionalBarrierInflator] = None,
    ) -> Dict[str, Any]:
        """Ingest one 1 kHz proprioceptive residual frame in O(1) time and O(1) memory."""
        n = self.n_joints

        tau_c = None if tau_cmd is None else _as_joint_vec(tau_cmd, n)
        tau_m = None if tau_meas is None else _as_joint_vec(tau_meas, n)
        q_d = None if q_des is None else _as_joint_vec(q_des, n)
        q_cur = None if q is None else _as_joint_vec(q, n)
        qd_d = None if qdot_des is None else _as_joint_vec(qdot_des, n)
        qd_cur = None if qdot is None else _as_joint_vec(qdot, n)

        if torque_error is not None:
            e_tau = _as_joint_vec(torque_error, n)
        elif tau_c is not None and tau_m is not None:
            e_tau = tau_c - tau_m
        else:
            e_tau = np.zeros(n, dtype=np.float64)

        if pos_error is not None:
            e_pos = _as_joint_vec(pos_error, n)
        elif q_d is not None and q_cur is not None:
            e_pos = q_d - q_cur
        else:
            e_pos = np.zeros(n, dtype=np.float64)

        if vel_error is not None:
            e_vel = _as_joint_vec(vel_error, n)
        elif qd_d is not None and qd_cur is not None:
            e_vel = qd_d - qd_cur
        else:
            e_vel = np.zeros(n, dtype=np.float64)

        temp_vec = _as_joint_vec(winding_temp_c, n, default=self.temp_nominal_c)
        hf_val = max(0.0, float(tactile_hf_energy)) if math.isfinite(float(tactile_hf_energy)) else 0.0

        # --- 1. O(1) Oscillatory Phase-Lag Wronskian (tau_cmd vs qdot) ---
        drive_sig = tau_c if tau_c is not None else e_tau
        resp_sig = qd_cur if qd_cur is not None else -e_vel

        if self.ticks == 0:
            self._tau_mean[:] = drive_sig
            self._qdot_mean[:] = resp_sig
        else:
            self._tau_mean += 0.04 * (drive_sig - self._tau_mean)
            self._qdot_mean += 0.04 * (resp_sig - self._qdot_mean)

        tau_ac = drive_sig - self._tau_mean
        qdot_ac = resp_sig - self._qdot_mean

        # Two-timescale low-pass filters: identical zero-lag signals give Wronskian == 0 identically,
        # while phase-lagged oscillation yields constant positive sin(phi) quadrature energy.
        self._tau_ac_fast = 0.65 * self._tau_ac_fast + 0.35 * tau_ac
        self._tau_ac_lag = 0.92 * self._tau_ac_lag + 0.08 * tau_ac
        self._qdot_ac_fast = 0.65 * self._qdot_ac_fast + 0.35 * qdot_ac
        self._qdot_ac_lag = 0.92 * self._qdot_ac_lag + 0.08 * qdot_ac

        wronskian_inst = (
            self._tau_ac_lag * self._qdot_ac_fast - self._tau_ac_fast * self._qdot_ac_lag
        )
        self._quad_wronskian_ema = 0.88 * self._quad_wronskian_ema + 0.12 * wronskian_inst
        self._tau_var_ema = 0.90 * self._tau_var_ema + 0.10 * (tau_ac * tau_ac)
        self._qdot_var_ema = 0.90 * self._qdot_var_ema + 0.10 * (qdot_ac * qdot_ac)
        self._vel_err_var_ema = 0.90 * self._vel_err_var_ema + 0.10 * (e_vel * e_vel)

        denom_lag = np.sqrt(self._tau_var_ema * self._qdot_var_ema) + 1e-8
        # Scale factor ~3.2 normalizes the (0.65 vs 0.92) two-filter sin(theta_2 - theta_1) gain to [0, 1]
        phase_lag_score = np.clip(3.2 * np.abs(self._quad_wronskian_ema) / denom_lag, 0.0, 1.0)

        # Deadband chatter term when tau_cmd oscillates strongly while output shaft stalls in deadzone
        tau_rms = np.sqrt(self._tau_var_ema)
        qdot_rms = np.sqrt(self._qdot_var_ema)
        vel_err_rms = np.sqrt(self._vel_err_var_ema)
        deadband_mask = (tau_rms > 0.15 * self.nominal_max_torque) & (vel_err_rms > self.backlash_osc_min) & (
            qdot_rms < 0.25 * vel_err_rms
        )
        effective_lag_score = np.where(
            (tau_rms > 0.05 * self.nominal_max_torque) & (qdot_rms > self.backlash_osc_min),
            phase_lag_score,
            np.where(deadband_mask, 0.65, 0.0),
        )

        # --- 2. O(1) Actuator Torque & Thermal Derating Estimation ---
        tau_load_min = 0.08 * self.nominal_max_torque
        if tau_c is not None and tau_m is not None:
            loaded = np.abs(tau_c) >= tau_load_min
            # Only attribute deficit when actuator under-delivers commanded magnitude or opposes sign,
            # and suppress updates during oscillatory sign-reversal backlash chatter.
            instant_ratio = np.clip(
                1.0 - np.abs(e_tau) / (np.abs(tau_c) + 1e-6),
                self.min_authority,
                1.0,
            )
        else:
            loaded = np.abs(e_tau) >= tau_load_min
            instant_ratio = np.clip(
                1.0 - np.abs(e_tau) / self.nominal_max_torque,
                self.min_authority,
                1.0,
            )

        non_backlash_loaded = loaded & (effective_lag_score < self.backlash_phase_threshold)
        self._torque_ratio_ema = np.where(
            non_backlash_loaded,
            0.85 * self._torque_ratio_ema + 0.15 * instant_ratio,
            self._torque_ratio_ema,
        )

        temp_span = self.temp_critical_c - self.temp_derate_start_c
        temp_excess = np.maximum(0.0, temp_vec - self.temp_derate_start_c)
        thermal_scale = np.clip(
            1.0 - (1.0 - self.min_authority) * (temp_excess / temp_span),
            self.min_authority,
            1.0,
        )
        self.torque_scale[:] = np.clip(
            np.minimum(self._torque_ratio_ema, thermal_scale),
            self.min_authority,
            1.0,
        )

        # --- 3. Multi-Timescale Recurrent SSM Update (O(1) time & memory) ---
        self._tactile_hf_ema = 0.60 * self._tactile_hf_ema + 0.40 * hf_val
        tangential_sig = (
            max(0.0, float(tangential_slip_rate))
            if tangential_slip_rate is not None
            else float(np.max(np.abs(e_vel)) + 0.5 * np.max(np.abs(e_pos)))
        )

        self._u_buf.fill(0.0)
        d_m = self.ssm.d_model
        norm_tau_err = e_tau / self.nominal_max_torque
        norm_temp = np.maximum(0.0, temp_vec - self.temp_nominal_c) / max(
            self.temp_critical_c - self.temp_nominal_c, 1.0
        )

        # Pack multi-channel normalized residuals into fixed pre-allocated input buffer
        idx = 0
        for block in (norm_tau_err, e_pos, e_vel, norm_temp, effective_lag_score):
            end = min(d_m, idx + n)
            if end > idx:
                self._u_buf[idx:end] = block[: end - idx].astype(np.float32)
            idx = end
        if idx < d_m:
            self._u_buf[idx] = np.float32(hf_val)
        if idx + 1 < d_m:
            self._u_buf[idx + 1] = np.float32(tangential_sig)
        if idx + 2 < d_m:
            self._u_buf[idx + 2] = np.float32(np.mean(np.abs(norm_tau_err)))
        if idx + 3 < d_m:
            self._u_buf[idx + 3] = np.float32(np.max(norm_temp))

        ssm_out = self.ssm.step_mamba3(self._u_buf)
        h_mat = self.ssm.hidden_state
        d_s = self.ssm.d_state
        third = max(1, d_s // 3)
        slow_energy = float(np.linalg.norm(h_mat[:, :third]))
        osc_coherence = float(ssm_out["phase_coherence"])
        fast_energy = float(np.linalg.norm(h_mat[:, -third:]))
        fast_tactile = float(np.linalg.norm(h_mat[min(idx, d_m - 1), -third:]))

        self.ticks += 1

        # --- 4. Multi-Timescale Health Classification & Actionable Mitigations ---
        slip_detected = bool(
            (hf_val >= self.slip_hf_threshold or self._tactile_hf_ema >= self.slip_hf_threshold)
            and (tangential_sig > 1e-3 or hf_val >= 1.5 * self.slip_hf_threshold)
        )
        backlash_detected = bool(
            np.max(effective_lag_score) >= self.backlash_phase_threshold and osc_coherence > 1e-5
        )
        degradation_detected = bool(
            np.min(self.torque_scale) <= self.derating_detect_threshold
            or np.max(temp_vec) >= self.temp_derate_start_c + 2.0
        )

        # Gripper slip -> bounded positive grasp force delta
        base_grasp = (
            self.nominal_grasp_force_n
            if current_grasp_force_n is None
            else max(0.0, float(current_grasp_force_n))
        )
        headroom_n = max(0.0, self.max_grasp_force_n - base_grasp)
        if slip_detected and headroom_n > 0.0:
            excess_hf = max(0.0, max(hf_val, self._tactile_hf_ema) - self.slip_hf_threshold) + 0.05
            raw_delta = self.slip_gain_n * (excess_hf + 0.25 * fast_tactile + 0.5 * min(tangential_sig, 2.0))
            self.grasp_force_delta_n = float(min(headroom_n, max(1e-3, raw_delta)))
        else:
            self.grasp_force_delta_n = 0.0
        recommended_grasp_n = float(min(self.max_grasp_force_n, base_grasp + self.grasp_force_delta_n))

        # Backlash / cable stretch -> per-joint covariance inflation for DirectionalBarrierInflator
        if backlash_detected:
            active_lag = np.maximum(0.0, effective_lag_score - 0.5 * self.backlash_phase_threshold)
            amp = self.covariance_inflation_gain * active_lag * (1.0 + np.abs(e_pos) + 0.25 * vel_err_rms)
            self.covariance_inflation[:] = np.where(
                effective_lag_score >= self.backlash_phase_threshold,
                np.maximum(1e-3, amp),
                0.0,
            )
        else:
            self.covariance_inflation *= 0.5
            self.covariance_inflation[self.covariance_inflation < 1e-6] = 0.0

        self.state_covariance_inflation[:n] = self.covariance_inflation
        self.state_covariance_inflation[n:] = 2.0 * self.covariance_inflation

        if inflator is not None and np.any(self.covariance_inflation > 0.0):
            self.push_to_inflator(inflator)

        active_anomalies: List[str] = []
        if slip_detected:
            active_anomalies.append(self.HEALTH_SLIP)
        if degradation_detected:
            active_anomalies.append(self.HEALTH_DEGRADATION)
        if backlash_detected:
            active_anomalies.append(self.HEALTH_BACKLASH)

        # Primary health state selection
        if slip_detected and not degradation_detected and not backlash_detected:
            self.health_status = self.HEALTH_SLIP
        elif backlash_detected and not degradation_detected and not slip_detected:
            self.health_status = self.HEALTH_BACKLASH
        elif degradation_detected and not slip_detected and not backlash_detected:
            self.health_status = self.HEALTH_DEGRADATION
        elif active_anomalies:
            # Deterministic priority when simultaneous anomalies occur: slip -> degradation -> backlash
            self.health_status = active_anomalies[0]
        else:
            self.health_status = self.HEALTH_NOMINAL

        return {
            "health_status": self.health_status,
            "classification": self.health_status,
            "active_anomalies": active_anomalies,
            "grasp_force_delta_n": self.grasp_force_delta_n,
            "recommended_grasp_force_n": recommended_grasp_n,
            "max_grasp_force_n": self.max_grasp_force_n,
            "covariance_inflation": self.covariance_inflation.tolist(),
            "state_covariance_inflation": self.state_covariance_inflation.tolist(),
            "torque_scale": self.torque_scale.tolist(),
            "ssm_features": {
                "slow_energy": round(slow_energy, 6),
                "oscillatory_coherence": round(osc_coherence, 6),
                "fast_energy": round(fast_energy, 6),
                "state_energy": float(ssm_out["state_energy"]),
            },
            "residual_stats": {
                "torque_error_norm": float(np.linalg.norm(e_tau)),
                "pos_error_norm": float(np.linalg.norm(e_pos)),
                "vel_error_norm": float(np.linalg.norm(e_vel)),
                "max_winding_temp_c": float(np.max(temp_vec)),
                "tactile_hf_energy": hf_val,
                "max_phase_lag_score": float(np.max(effective_lag_score)),
            },
            "ticks": self.ticks,
            "memory_bytes": self.state_memory_bytes(),
        }

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        """Alias for ``step``."""
        return self.step(**kwargs)

    def push_to_inflator(self, inflator: DirectionalBarrierInflator) -> np.ndarray:
        """Push per-joint backlash uncertainty inflation into a ``DirectionalBarrierInflator``."""
        if inflator.state_dim == self.n_joints:
            return inflator.update_residual(self.covariance_inflation)
        if inflator.state_dim == 2 * self.n_joints:
            return inflator.update_residual(self.state_covariance_inflation)
        vec = np.zeros(inflator.state_dim, dtype=np.float64)
        k = min(inflator.state_dim, self.state_covariance_inflation.shape[0])
        vec[:k] = self.state_covariance_inflation[:k]
        return inflator.update_residual(vec)

    def _effective_barrier_scale(self, bar: BarrierConstraint) -> float:
        base = getattr(bar, "base_barrier", bar)
        if hasattr(base, "a"):
            a_vec = np.asarray(getattr(base, "a"), dtype=np.float64).flatten()
            active = np.abs(a_vec[: self.n_joints]) > 1e-9
            if np.any(active):
                return float(np.min(self.torque_scale[: a_vec[: self.n_joints].shape[0]][active]))
        return float(np.min(self.torque_scale))

    def apply_derating_to_filter(
        self,
        safety_filter: StateCBFSafetyFilter,
        nominal_u_min: Sequence[float],
        nominal_u_max: Sequence[float],
    ) -> Dict[str, Any]:
        """Scale ``safety_filter.u_min``/``u_max`` and adapt HOCBF braking rates to match authority.

        When an actuator's deliverable torque shrinks by ``s in [min_authority, 1]``,
        kinematic stopping distance ``v^2 / (2 s a_max)`` grows by ``1 / s``.
        Scaling both the control box ``[s u_min, s u_max]`` and relative-degree-2
        HOCBF class-K decay rates ``(s alpha1, s alpha2)`` preserves exact viability
        and causes ``StateCBFSafetyFilter`` to initiate braking earlier at ``1 / s``
        distance from obstacles/walls.
        """
        u_min_nom = _as_joint_vec(nominal_u_min, self.n_joints)
        u_max_nom = _as_joint_vec(nominal_u_max, self.n_joints)
        if np.any(u_min_nom > u_max_nom):
            raise ValueError("nominal_u_min must be elementwise <= nominal_u_max")

        derated_u_min = u_min_nom * self.torque_scale
        derated_u_max = u_max_nom * self.torque_scale
        safety_filter.u_min = derated_u_min.copy()
        safety_filter.u_max = derated_u_max.copy()

        adapted_barriers = 0
        for bar in safety_filter.barriers:
            rel_deg = int(getattr(bar, "relative_degree", 1))
            if rel_deg == 2:
                bid = id(bar)
                if bid not in self._nominal_barrier_alphas:
                    a1_nom = float(getattr(bar, "alpha1", getattr(bar, "alpha", 0.5)))
                    a_nom = float(getattr(bar, "alpha", a1_nom))
                    a2_nom = float(getattr(bar, "alpha2", 1.0))
                    self._nominal_barrier_alphas[bid] = (a1_nom, a_nom, a2_nom)
                a1_base, a_base, a2_base = self._nominal_barrier_alphas[bid]
                s_bar = self._effective_barrier_scale(bar)
                if isinstance(bar, HighOrderBarrier) or hasattr(bar, "alpha1"):
                    bar.alpha1 = s_bar * a1_base
                bar.alpha = s_bar * a_base
                bar.alpha2 = s_bar * a2_base
                adapted_barriers += 1

        return {
            "torque_scale": self.torque_scale.tolist(),
            "u_min": derated_u_min.tolist(),
            "u_max": derated_u_max.tolist(),
            "min_torque_scale": float(np.min(self.torque_scale)),
            "adapted_hocbf_barriers": adapted_barriers,
            "derated": bool(np.any(self.torque_scale < 1.0 - 1e-6)),
        }
