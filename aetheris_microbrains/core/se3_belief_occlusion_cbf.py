"""Right-Invariant SE_2(3) EKF for delayed pose fusion and occlusion shadow CBFs.

Provides deterministic NumPy state-estimation and active-perception safety
primitives:

- ``RightInvariantSE3Filter``: Right-Invariant EKF on ``SE_2(3)`` tracking
  ``(R_t in SO(3), v_t in R^3, p_t in R^3)`` under high-rate IMU propagation
  ``(omega, accel, dt)`` and out-of-sequence delayed camera/SLAM pose updates
  via a bounded state/IMU ring buffer that rewinds, updates in ``se_2(3)``,
  and re-propagates forward to ``t_now``.
- ``OcclusionShadowBarrier``: line-of-sight ray-caster over 2D/3D occluders
  maintaining a time-expanding reachable shadow zone
  ``r_occ(t) = r_0 + v_max * (t - t_seen) + 0.5 * a_max * (t - t_seen)^2``
  with exact ``advance_time(dt)``, ``dh_dt(x)``, ``d2h_dt2(x)``,
  ``grad_dh_dt(x)``, and ``mark_observed(pos, vel, now_s)`` to collapse
  uncertainty when active perception (``epistemic_probe``) re-acquires sight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .bimanual_grasp_cbf import log_so3
from .state_cbf_filter import BarrierConstraint
from .urdf_spatial_kinematics import exp_so3, hat_so3


# ---------------------------------------------------------------------------
# SE_2(3) Lie Group Helpers
# ---------------------------------------------------------------------------


def _so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian ``J_l(phi)`` of SO(3) with Taylor expansion near 0."""
    phi_v = np.asarray(phi, dtype=np.float64).flatten()
    theta2 = float(np.dot(phi_v, phi_v))
    theta = math.sqrt(theta2)
    K = hat_so3(phi_v)
    if theta < 1e-8:
        c2 = 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0
        c3 = (1.0 / 6.0) - theta2 / 120.0 + (theta2 * theta2) / 5040.0
    else:
        c2 = (1.0 - math.cos(theta)) / theta2
        c3 = (theta - math.sin(theta)) / (theta2 * theta)
    return np.eye(3, dtype=np.float64) + c2 * K + c3 * (K @ K)


def exp_se2_3(xi: Sequence[float]) -> np.ndarray:
    """Exponential map ``se_2(3) -> SE_2(3)`` (5x5) for ``xi = [phi(3); rho_v(3); rho_p(3)]``."""
    xi_arr = np.asarray(xi, dtype=np.float64).flatten()
    if xi_arr.shape[0] != 9:
        raise ValueError(f"exp_se2_3 expects a 9-vector, got shape {xi_arr.shape}")
    phi = xi_arr[0:3]
    rho_v = xi_arr[3:6]
    rho_p = xi_arr[6:9]
    R = exp_so3(phi)
    Jl = _so3_left_jacobian(phi)
    X = np.eye(5, dtype=np.float64)
    X[:3, :3] = R
    X[:3, 3] = Jl @ rho_v
    X[:3, 4] = Jl @ rho_p
    return X


def pack_se2_3(R: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble 5x5 ``SE_2(3)`` matrix from ``(R, v, p)``."""
    X = np.eye(5, dtype=np.float64)
    X[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    X[:3, 3] = np.asarray(v, dtype=np.float64).flatten()[:3]
    X[:3, 4] = np.asarray(p, dtype=np.float64).flatten()[:3]
    return X


def _reproject_so3(R: np.ndarray) -> np.ndarray:
    """Nearest proper rotation matrix in SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0.0:
        U[:, -1] *= -1.0
        R_clean = U @ Vt
    return R_clean


def _normalize_cov(cov: Union[float, Sequence[float], np.ndarray], dim: int) -> np.ndarray:
    arr = np.asarray(cov, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr) * np.eye(dim, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"Covariance vector length {arr.shape[0]} != {dim}")
        return np.diag(arr)
    if arr.shape == (dim, dim):
        return 0.5 * (arr + arr.T)
    raise ValueError(f"Covariance shape {arr.shape} incompatible with dimension {dim}")


# ---------------------------------------------------------------------------
# RightInvariantSE3Filter
# ---------------------------------------------------------------------------


@dataclass
class _BufferedIMUStep:
    """Single ring-buffer snapshot storing post-propagation state and the IMU packet that produced it."""

    timestamp_s: float
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    P: np.ndarray
    omega: np.ndarray
    accel: np.ndarray
    dt: float


class RightInvariantSE3Filter:
    """Right-Invariant Extended Kalman Filter on ``SE_2(3)`` with out-of-sequence delay buffer.

    Tracks ``(R_t in SO(3), v_t in R^3, p_t in R^3)`` and 9x9 right-invariant
    Lie-algebra covariance ``P_t`` (error ``eta = X X_hat^{-1} = exp(xi^)`` with
    ``xi = [phi, rho_v, rho_p]``). Maintains a bounded ring buffer of past states
    and IMU packets so delayed visual/SLAM pose observations ``fuse_delayed_pose``
    rewind to the nearest buffered state, apply the Lie-algebra Kalman update
    ``X <- exp(delta_xi^) X``, and re-propagate stored IMU packets to ``t_now``.
    """

    def __init__(
        self,
        R0: Optional[np.ndarray] = None,
        v0: Optional[Sequence[float]] = None,
        p0: Optional[Sequence[float]] = None,
        P0: Union[float, Sequence[float], np.ndarray] = 1e-2,
        t0: float = 0.0,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        gyro_noise_std: float = 0.005,
        accel_noise_std: float = 0.05,
        pos_diffusion_std: float = 1e-4,
        buffer_size: int = 512,
    ):
        if buffer_size < 2:
            raise ValueError("buffer_size must be >= 2")
        self.gravity = np.asarray(gravity, dtype=np.float64).flatten()[:3]
        self.gyro_noise_std = float(gyro_noise_std)
        self.accel_noise_std = float(accel_noise_std)
        self.pos_diffusion_std = float(pos_diffusion_std)
        self.buffer_size = int(buffer_size)

        self._Q_imu = np.diag(
            [self.gyro_noise_std ** 2] * 3 + [self.accel_noise_std ** 2] * 3
        )
        self._g_skew = hat_so3(self.gravity)

        self.R = np.eye(3, dtype=np.float64) if R0 is None else _reproject_so3(R0)
        self.v = np.zeros(3, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).flatten()[:3].copy()
        self.p = np.zeros(3, dtype=np.float64) if p0 is None else np.asarray(p0, dtype=np.float64).flatten()[:3].copy()
        self.P = _normalize_cov(P0, 9)
        self.t = float(t0)

        self._buffer: Deque[_BufferedIMUStep] = deque(maxlen=self.buffer_size)
        self._buffer.append(
            _BufferedIMUStep(
                timestamp_s=self.t,
                R=self.R.copy(),
                v=self.v.copy(),
                p=self.p.copy(),
                P=self.P.copy(),
                omega=np.zeros(3, dtype=np.float64),
                accel=np.zeros(3, dtype=np.float64),
                dt=0.0,
            )
        )

    @property
    def X(self) -> np.ndarray:
        """Current 5x5 ``SE_2(3)`` state matrix."""
        return pack_se2_3(self.R, self.v, self.p)

    @property
    def covariance(self) -> np.ndarray:
        """Current 9x9 world-frame state covariance ``Sigma_t`` over ``[dphi, dv, dp]``."""
        J_w = np.eye(9, dtype=np.float64)
        J_w[3:6, 0:3] = -hat_so3(self.v)
        J_w[6:9, 0:3] = -hat_so3(self.p)
        S = J_w @ self.P @ J_w.T
        return 0.5 * (S + S.T)

    def reset(
        self,
        R0: Optional[np.ndarray] = None,
        v0: Optional[Sequence[float]] = None,
        p0: Optional[Sequence[float]] = None,
        P0: Union[float, Sequence[float], np.ndarray] = 1e-2,
        t0: float = 0.0,
    ) -> None:
        self.R = np.eye(3, dtype=np.float64) if R0 is None else _reproject_so3(R0)
        self.v = np.zeros(3, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).flatten()[:3].copy()
        self.p = np.zeros(3, dtype=np.float64) if p0 is None else np.asarray(p0, dtype=np.float64).flatten()[:3].copy()
        self.P = _normalize_cov(P0, 9)
        self.t = float(t0)
        self._buffer.clear()
        self._buffer.append(
            _BufferedIMUStep(
                timestamp_s=self.t,
                R=self.R.copy(),
                v=self.v.copy(),
                p=self.p.copy(),
                P=self.P.copy(),
                omega=np.zeros(3, dtype=np.float64),
                accel=np.zeros(3, dtype=np.float64),
                dt=0.0,
            )
        )

    def _step_dynamics(
        self,
        R: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        P: np.ndarray,
        omega: np.ndarray,
        accel: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagate one IMU step in ``SE_2(3)`` and right-invariant covariance."""
        if dt <= 0.0:
            return R.copy(), v.copy(), p.copy(), P.copy()

        d_theta = omega * dt
        R_mid = R @ exp_so3(0.5 * d_theta)
        R_next = _reproject_so3(R @ exp_so3(d_theta))
        a_world = R_mid @ accel + self.gravity
        p_next = p + v * dt + 0.5 * a_world * (dt * dt)
        v_next = v + a_world * dt

        # Exact nilpotent state transition Phi = exp(A_c * dt)
        Phi = np.eye(9, dtype=np.float64)
        Phi[3:6, 0:3] = self._g_skew * dt
        Phi[6:9, 0:3] = 0.5 * self._g_skew * (dt * dt)
        Phi[6:9, 3:6] = np.eye(3, dtype=np.float64) * dt

        # Right-invariant body-noise input matrix G_c(X) = Ad_X[:, :6]
        Gc = np.zeros((9, 6), dtype=np.float64)
        Gc[0:3, 0:3] = R
        Gc[3:6, 0:3] = hat_so3(v) @ R
        Gc[3:6, 3:6] = R
        Gc[6:9, 0:3] = hat_so3(p) @ R

        Qd = (Phi @ Gc @ self._Q_imu @ Gc.T @ Phi.T) * dt
        if self.pos_diffusion_std > 0.0:
            Qd[6:9, 6:9] += (self.pos_diffusion_std ** 2) * dt * np.eye(3, dtype=np.float64)

        P_next = Phi @ P @ Phi.T + Qd
        P_next = 0.5 * (P_next + P_next.T)
        return R_next, v_next, p_next, P_next

    def propagate(
        self,
        omega: Sequence[float],
        accel: Sequence[float],
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagate filter forward by ``dt`` seconds with body-frame IMU ``(omega, accel)``."""
        dt_f = float(dt)
        if dt_f <= 0.0:
            raise ValueError(f"dt must be positive, got {dt_f}")
        w_vec = np.asarray(omega, dtype=np.float64).flatten()[:3]
        a_vec = np.asarray(accel, dtype=np.float64).flatten()[:3]

        self.R, self.v, self.p, self.P = self._step_dynamics(
            self.R, self.v, self.p, self.P, w_vec, a_vec, dt_f
        )
        self.t += dt_f
        self._buffer.append(
            _BufferedIMUStep(
                timestamp_s=self.t,
                R=self.R.copy(),
                v=self.v.copy(),
                p=self.p.copy(),
                P=self.P.copy(),
                omega=w_vec.copy(),
                accel=a_vec.copy(),
                dt=dt_f,
            )
        )
        return self.R.copy(), self.v.copy(), self.p.copy(), self.covariance

    def _apply_pose_update(
        self,
        R: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        P: np.ndarray,
        R_meas: np.ndarray,
        p_meas: np.ndarray,
        R_cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Right-invariant Kalman update on ``SE_2(3)``: ``X <- exp(delta_xi^) X``."""
        R_m = _reproject_so3(R_meas)
        p_m = np.asarray(p_meas, dtype=np.float64).flatten()[:3]

        r_R = log_so3(R_m @ R.T)
        r_p = p_m - p
        innov = np.concatenate([r_R, r_p])

        H = np.zeros((6, 9), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3, dtype=np.float64)
        H[3:6, 0:3] = -hat_so3(p)
        H[3:6, 6:9] = np.eye(3, dtype=np.float64)

        S = H @ P @ H.T + R_cov
        S = 0.5 * (S + S.T) + 1e-12 * np.eye(6, dtype=np.float64)
        K = np.linalg.solve(S, H @ P).T

        delta_xi = K @ innov
        X_curr = pack_se2_3(R, v, p)
        X_upd = exp_se2_3(delta_xi) @ X_curr

        R_upd = _reproject_so3(X_upd[:3, :3])
        v_upd = X_upd[:3, 3].copy()
        p_upd = X_upd[:3, 4].copy()

        I_KH = np.eye(9, dtype=np.float64) - K @ H
        P_upd = I_KH @ P @ I_KH.T + K @ R_cov @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)
        return R_upd, v_upd, p_upd, P_upd, float(np.linalg.norm(innov))

    def fuse_delayed_pose(
        self,
        timestamp_s: float,
        R_meas: np.ndarray,
        p_meas: Sequence[float],
        meas_cov: Union[float, Sequence[float], np.ndarray] = 1e-4,
    ) -> Dict[str, Any]:
        """Fuse a possibly delayed 6D pose observation ``(R_meas, p_meas)`` at ``timestamp_s``.

        When ``timestamp_s < t_now``, rewinds to the nearest buffered state in
        the ring buffer, performs the Lie-algebra right-invariant Kalman update
        ``X <- exp(delta_xi^) X``, and re-propagates all subsequent IMU packets
        forward to ``t_now``.
        """
        t_obs = float(timestamp_s)
        R_cov = _normalize_cov(meas_cov, 6)
        R_m = np.asarray(R_meas, dtype=np.float64).reshape(3, 3)
        p_m = np.asarray(p_meas, dtype=np.float64).flatten()[:3]

        if not (np.all(np.isfinite(R_m)) and np.all(np.isfinite(p_m)) and np.all(np.isfinite(R_cov))):
            return {
                "updated": False,
                "reason": "nonfinite_measurement",
                "R": self.R.copy(),
                "v": self.v.copy(),
                "p": self.p.copy(),
                "P": self.P.copy(),
                "covariance": self.covariance,
                "timestamp_s": self.t,
            }

        buf_list = list(self._buffer)
        if not buf_list or t_obs >= self.t - 1e-12:
            self.R, self.v, self.p, self.P, innov_norm = self._apply_pose_update(
                self.R, self.v, self.p, self.P, R_m, p_m, R_cov
            )
            if buf_list:
                last = buf_list[-1]
                last.R = self.R.copy()
                last.v = self.v.copy()
                last.p = self.p.copy()
                last.P = self.P.copy()
            return {
                "updated": True,
                "rewound_steps": 0,
                "delay_s": max(0.0, self.t - t_obs),
                "applied_timestamp_s": self.t,
                "timestamp_s": self.t,
                "innovation_norm": innov_norm,
                "R": self.R.copy(),
                "v": self.v.copy(),
                "p": self.p.copy(),
                "P": self.P.copy(),
                "covariance": self.covariance,
            }

        # Find nearest buffered state to t_obs
        best_idx = min(range(len(buf_list)), key=lambda i: abs(buf_list[i].timestamp_s - t_obs))
        entry = buf_list[best_idx]
        R_k, v_k, p_k, P_k, innov_norm = self._apply_pose_update(
            entry.R, entry.v, entry.p, entry.P, R_m, p_m, R_cov
        )
        entry.R = R_k.copy()
        entry.v = v_k.copy()
        entry.p = p_k.copy()
        entry.P = P_k.copy()

        # Re-propagate stored IMU packets forward from best_idx + 1 to t_now
        rewound_steps = len(buf_list) - 1 - best_idx
        for j in range(best_idx + 1, len(buf_list)):
            step_j = buf_list[j]
            R_k, v_k, p_k, P_k = self._step_dynamics(
                R_k, v_k, p_k, P_k, step_j.omega, step_j.accel, step_j.dt
            )
            step_j.R = R_k.copy()
            step_j.v = v_k.copy()
            step_j.p = p_k.copy()
            step_j.P = P_k.copy()

        self.R = R_k.copy()
        self.v = v_k.copy()
        self.p = p_k.copy()
        self.P = P_k.copy()

        return {
            "updated": True,
            "rewound_steps": rewound_steps,
            "delay_s": max(0.0, self.t - t_obs),
            "applied_timestamp_s": entry.timestamp_s,
            "timestamp_s": self.t,
            "innovation_norm": innov_norm,
            "R": self.R.copy(),
            "v": self.v.copy(),
            "p": self.p.copy(),
            "P": self.P.copy(),
            "covariance": self.covariance,
        }


# ---------------------------------------------------------------------------
# OcclusionShadowBarrier & Ray-Cast Visibility Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OccluderBox:
    """Axis-aligned 2D or 3D occluding box ``[min_corner, max_corner]``."""

    min_corner: Sequence[float]
    max_corner: Sequence[float]


@dataclass(frozen=True)
class OccluderSegment:
    """2D or 3D occluding line segment / capsule ``[p0, p1]`` with optional ``radius``."""

    p0: Sequence[float]
    p1: Sequence[float]
    radius: float = 0.0


def _segment_ray_aabb_intersects(
    origin: np.ndarray,
    target: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    eps: float = 1e-12,
) -> bool:
    """Slab test for finite ray ``origin + s * (target - origin)``, ``s in (0, 1)``."""
    d = target - origin
    s_min = 0.0
    s_max = 1.0 - 1e-9
    for axis in range(origin.shape[0]):
        da = float(d[axis])
        o_a = float(origin[axis])
        lo = float(b_min[axis])
        hi = float(b_max[axis])
        if abs(da) <= eps:
            if o_a < lo or o_a > hi:
                return False
        else:
            t1 = (lo - o_a) / da
            t2 = (hi - o_a) / da
            if t1 > t2:
                t1, t2 = t2, t1
            s_min = max(s_min, t1)
            s_max = min(s_max, t2)
            if s_min > s_max:
                return False
    return s_max > 0.0 and s_min <= s_max


def _segment_segment_dist_sq(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, eps: float = 1e-15
) -> float:
    """Exact squared Euclidean distance between two N-D line segments."""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    if a <= eps and e <= eps:
        return float(np.dot(r, r))
    if a <= eps:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    diff = (p0 + s * d1) - (q0 + t * d2)
    return float(np.dot(diff, diff))


def _ray_hits_segment(
    origin: np.ndarray,
    target: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float = 0.0,
) -> bool:
    """Test whether finite ray ``[origin, target]`` intersects segment/capsule ``[p0, p1]``."""
    if radius > 0.0:
        d2 = _segment_segment_dist_sq(origin, target, p0, p1)
        return d2 <= radius * radius

    if origin.shape[0] == 2:
        r = target - origin
        s = p1 - p0
        rxs = float(r[0] * s[1] - r[1] * s[0])
        qp = p0 - origin
        qpxr = float(qp[0] * r[1] - qp[1] * r[0])
        if abs(rxs) <= 1e-12:
            if abs(qpxr) > 1e-12:
                return False
            rr = float(np.dot(r, r))
            if rr <= 1e-15:
                return False
            t0 = float(np.dot(p0 - origin, r)) / rr
            t1 = float(np.dot(p1 - origin, r)) / rr
            lo, hi = min(t0, t1), max(t0, t1)
            return hi > 1e-9 and lo < 1.0 - 1e-9
        t = float(qp[0] * s[1] - qp[1] * s[0]) / rxs
        u = qpxr / rxs
        return (1e-9 < t < 1.0 - 1e-9) and (0.0 <= u <= 1.0)

    # 3D thin segment fallback
    return _segment_segment_dist_sq(origin, target, p0, p1) <= 1e-6


def _ray_hits_sphere(origin: np.ndarray, target: np.ndarray, center: np.ndarray, radius: float) -> bool:
    d = target - origin
    a = float(np.dot(d, d))
    if a <= 1e-15:
        return float(np.linalg.norm(origin - center)) <= radius
    oc = origin - center
    b = 2.0 * float(np.dot(oc, d))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sqrt_disc = math.sqrt(disc)
    s1 = (-b - sqrt_disc) / (2.0 * a)
    s2 = (-b + sqrt_disc) / (2.0 * a)
    return (s2 > 1e-9) and (s1 < 1.0 - 1e-9)


def _ray_hits_occluder(origin: np.ndarray, target: np.ndarray, occ: Any) -> bool:
    dim = origin.shape[0]
    if isinstance(occ, OccluderBox):
        return _segment_ray_aabb_intersects(
            origin,
            target,
            np.asarray(occ.min_corner, dtype=np.float64).flatten()[:dim],
            np.asarray(occ.max_corner, dtype=np.float64).flatten()[:dim],
        )
    if isinstance(occ, OccluderSegment):
        return _ray_hits_segment(
            origin,
            target,
            np.asarray(occ.p0, dtype=np.float64).flatten()[:dim],
            np.asarray(occ.p1, dtype=np.float64).flatten()[:dim],
            radius=float(occ.radius),
        )
    if isinstance(occ, dict):
        kind = str(occ.get("type", "")).lower()
        if kind == "box" or ("min" in occ and "max" in occ):
            return _segment_ray_aabb_intersects(
                origin,
                target,
                np.asarray(occ["min"], dtype=np.float64).flatten()[:dim],
                np.asarray(occ["max"], dtype=np.float64).flatten()[:dim],
            )
        if kind == "sphere" or ("center" in occ and "radius" in occ and "p0" not in occ):
            return _ray_hits_sphere(
                origin,
                target,
                np.asarray(occ["center"], dtype=np.float64).flatten()[:dim],
                float(occ["radius"]),
            )
        p0 = np.asarray(occ["p0"], dtype=np.float64).flatten()[:dim]
        p1 = np.asarray(occ["p1"], dtype=np.float64).flatten()[:dim]
        return _ray_hits_segment(origin, target, p0, p1, radius=float(occ.get("radius", 0.0)))
    if isinstance(occ, (tuple, list)):
        if len(occ) == 2:
            p0 = np.asarray(occ[0], dtype=np.float64).flatten()[:dim]
            p1 = np.asarray(occ[1], dtype=np.float64).flatten()[:dim]
            return _ray_hits_segment(origin, target, p0, p1, radius=0.0)
        if len(occ) == 3:
            p0 = np.asarray(occ[0], dtype=np.float64).flatten()[:dim]
            p1 = np.asarray(occ[1], dtype=np.float64).flatten()[:dim]
            return _ray_hits_segment(origin, target, p0, p1, radius=float(occ[2]))
    raise TypeError(f"Unsupported occluder specification: {type(occ).__name__}")


class OcclusionShadowBarrier(BarrierConstraint):
    """Time-varying reachability CBF for occluded obstacle belief zones.

    Ray-casts from ``sensor_pos`` against ``occluders`` (2D/3D segments or
    boxes) to test line-of-sight visibility. When occluded, the obstacle's
    reachable radius expands as::

        r_occ(t) = r_0 + v_max * (t - t_seen) + 0.5 * a_max * (t - t_seen)^2

    defining the time-varying barrier::

        h(x, t) = ||p_robot(x) - c_occ(t)||^2 - r_occ(t)^2 - margin_offset >= 0

    Implements ``advance_time(dt)``, ``set_time(t)``, ``dh_dt(x)``, ``d2h_dt2(x)``,
    ``grad_dh_dt(x)``, and ``mark_observed(pos, vel, now_s)`` to collapse
    ``r_occ`` back to ``r_0`` when active perception (``epistemic_probe``)
    re-acquires line of sight.
    """

    def __init__(
        self,
        last_seen_pos: Sequence[float],
        r_0: float = 0.3,
        v_max: float = 1.0,
        a_max: float = 0.0,
        sensor_pos: Optional[Sequence[float]] = None,
        occluders: Optional[Sequence[Any]] = None,
        last_seen_vel: Optional[Sequence[float]] = None,
        last_seen_time: float = 0.0,
        current_time: Optional[float] = None,
        propagate_velocity: bool = False,
        only_expand_when_occluded: bool = True,
        state_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
        alpha: float = 1.0,
        alpha2: float = 2.0,
        relative_degree: int = 2,
        hard: bool = True,
        margin_offset: float = 0.0,
    ):
        self.last_seen_pos = np.asarray(last_seen_pos, dtype=np.float64).flatten()
        self.spatial_dim = int(self.last_seen_pos.shape[0])
        if self.spatial_dim not in (2, 3):
            raise ValueError(f"OcclusionShadowBarrier expects 2D or 3D positions, got {self.spatial_dim}D")
        self.last_seen_vel = (
            np.zeros(self.spatial_dim, dtype=np.float64)
            if last_seen_vel is None
            else np.asarray(last_seen_vel, dtype=np.float64).flatten()[: self.spatial_dim]
        )
        self.r_0 = float(r_0)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.sensor_pos = (
            np.zeros(self.spatial_dim, dtype=np.float64)
            if sensor_pos is None
            else np.asarray(sensor_pos, dtype=np.float64).flatten()[: self.spatial_dim]
        )
        self.occluders: List[Any] = list(occluders) if occluders is not None else []
        self.last_seen_time = float(last_seen_time)
        self.current_time = self.last_seen_time if current_time is None else float(current_time)
        self.propagate_velocity = bool(propagate_velocity)
        self.only_expand_when_occluded = bool(only_expand_when_occluded)
        self.state_slice = state_slice
        self.state_dim = None if state_dim is None else int(state_dim)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.margin_offset = float(margin_offset)

        self.occluded: bool = self._evaluate_occlusion()

    def _evaluate_occlusion(
        self,
        target_pos: Optional[np.ndarray] = None,
        sensor_pos: Optional[np.ndarray] = None,
    ) -> bool:
        if not self.occluders:
            return True
        origin = self.sensor_pos if sensor_pos is None else np.asarray(sensor_pos, dtype=np.float64).flatten()[: self.spatial_dim]
        target = self.last_seen_pos if target_pos is None else np.asarray(target_pos, dtype=np.float64).flatten()[: self.spatial_dim]
        return any(_ray_hits_occluder(origin, target, occ) for occ in self.occluders)

    def is_occluded(
        self,
        target_pos: Optional[Sequence[float]] = None,
        sensor_pos: Optional[Sequence[float]] = None,
    ) -> bool:
        """Ray-cast from ``sensor_pos`` to ``target_pos`` against configured occluders."""
        t_arr = None if target_pos is None else np.asarray(target_pos, dtype=np.float64)
        s_arr = None if sensor_pos is None else np.asarray(sensor_pos, dtype=np.float64)
        return self._evaluate_occlusion(target_pos=t_arr, sensor_pos=s_arr)

    def set_sensor_pos(self, sensor_pos: Sequence[float]) -> bool:
        """Update sensor origin and re-evaluate line-of-sight occlusion."""
        self.sensor_pos = np.asarray(sensor_pos, dtype=np.float64).flatten()[: self.spatial_dim]
        self.occluded = self._evaluate_occlusion()
        return self.occluded

    def set_time(self, t: float) -> None:
        self.current_time = float(t)
        self.occluded = self._evaluate_occlusion()

    def advance_time(self, dt: float) -> float:
        self.current_time += float(dt)
        self.occluded = self._evaluate_occlusion()
        return self.current_time

    def mark_observed(
        self,
        pos: Sequence[float],
        vel: Optional[Sequence[float]] = None,
        now_s: Optional[float] = None,
        sensor_pos: Optional[Sequence[float]] = None,
    ) -> float:
        """Re-acquire observation, collapsing reachable shadow radius ``r_occ`` back to ``r_0``."""
        self.last_seen_pos = np.asarray(pos, dtype=np.float64).flatten()[: self.spatial_dim]
        self.last_seen_vel = (
            np.zeros(self.spatial_dim, dtype=np.float64)
            if vel is None
            else np.asarray(vel, dtype=np.float64).flatten()[: self.spatial_dim]
        )
        obs_t = self.current_time if now_s is None else float(now_s)
        self.last_seen_time = obs_t
        self.current_time = obs_t
        if sensor_pos is not None:
            self.sensor_pos = np.asarray(sensor_pos, dtype=np.float64).flatten()[: self.spatial_dim]
        self.occluded = self._evaluate_occlusion()
        return self.reachable_radius(self.current_time)

    def _effective_dt_and_active(self, t: Optional[float] = None) -> Tuple[float, bool]:
        t_eval = self.current_time if t is None else float(t)
        raw_dt = max(0.0, t_eval - self.last_seen_time)
        if self.only_expand_when_occluded and self.occluders and not self.occluded:
            return 0.0, False
        return raw_dt, True

    def reachable_radius(self, t: Optional[float] = None) -> float:
        """Return ``r_occ(t) = r_0 + v_max*(t - t_seen) + 0.5*a_max*(t - t_seen)^2``."""
        dt_occ, active = self._effective_dt_and_active(t)
        if not active or dt_occ <= 0.0:
            return self.r_0
        return float(self.r_0 + self.v_max * dt_occ + 0.5 * self.a_max * dt_occ * dt_occ)

    def _radius_derivatives(self, t: Optional[float] = None) -> Tuple[float, float, float]:
        dt_occ, active = self._effective_dt_and_active(t)
        if not active or (dt_occ <= 0.0 and not self.occluded):
            return self.r_0, 0.0, 0.0
        r = self.r_0 + self.v_max * dt_occ + 0.5 * self.a_max * dt_occ * dt_occ
        dr = self.v_max + self.a_max * dt_occ
        d2r = self.a_max
        return float(r), float(dr), float(d2r)

    def _center_at(self, t: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        dt_occ, active = self._effective_dt_and_active(t)
        if self.propagate_velocity and active and dt_occ > 0.0:
            return self.last_seen_pos + self.last_seen_vel * dt_occ, self.last_seen_vel
        return self.last_seen_pos, np.zeros(self.spatial_dim, dtype=np.float64)

    def _pos_slice(self, n_state: int) -> slice:
        if self.state_slice is not None:
            return self.state_slice
        return slice(0, min(self.spatial_dim, n_state))

    def _slice_pos(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        return arr[self._pos_slice(arr.shape[0])]

    def h(self, x: np.ndarray) -> float:
        pos = self._slice_pos(x)
        center, _ = self._center_at(self.current_time)
        r_occ = self.reachable_radius(self.current_time)
        d = pos - center
        return float(np.dot(d, d) - r_occ * r_occ - self.margin_offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        n = arr.shape[0]
        full_dim = self.state_dim if self.state_dim is not None else n
        sl = self._pos_slice(full_dim)
        center, _ = self._center_at(self.current_time)
        out = np.zeros(full_dim, dtype=np.float64)
        out[sl] = 2.0 * (arr[sl] - center)
        return out

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        _ = eps
        arr = np.asarray(x, dtype=np.float64).flatten()
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        sl = self._pos_slice(full_dim)
        H = np.zeros((full_dim, full_dim), dtype=np.float64)
        idx_stop = full_dim if sl.stop is None else sl.stop
        idx_start = 0 if sl.start is None else sl.start
        for k in range(idx_start, min(idx_stop, full_dim)):
            H[k, k] = 2.0
        return H

    def dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> float:
        t_eval = self.current_time if t is None else float(t)
        pos = self._slice_pos(x)
        center, vel = self._center_at(t_eval)
        r_occ, dr_dt, _ = self._radius_derivatives(t_eval)
        if not self.propagate_velocity:
            return float(-2.0 * r_occ * dr_dt)
        d = pos - center
        return float(-2.0 * np.dot(d, vel) - 2.0 * r_occ * dr_dt)

    def d2h_dt2(self, x: np.ndarray, t: Optional[float] = None) -> float:
        _ = x
        t_eval = self.current_time if t is None else float(t)
        _, vel = self._center_at(t_eval)
        r_occ, dr_dt, d2r_dt2 = self._radius_derivatives(t_eval)
        vel_term = 2.0 * float(np.dot(vel, vel)) if self.propagate_velocity else 0.0
        return float(vel_term - 2.0 * (dr_dt * dr_dt + r_occ * d2r_dt2))

    def grad_dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        t_eval = self.current_time if t is None else float(t)
        arr = np.asarray(x, dtype=np.float64).flatten()
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        out = np.zeros(full_dim, dtype=np.float64)
        if not self.propagate_velocity:
            return out
        _, vel = self._center_at(t_eval)
        sl = self._pos_slice(full_dim)
        out[sl] = -2.0 * vel
        return out

    def inflate(self, delta: float) -> "OcclusionShadowBarrier":
        return OcclusionShadowBarrier(
            last_seen_pos=self.last_seen_pos,
            r_0=self.r_0,
            v_max=self.v_max,
            a_max=self.a_max,
            sensor_pos=self.sensor_pos,
            occluders=self.occluders,
            last_seen_vel=self.last_seen_vel,
            last_seen_time=self.last_seen_time,
            current_time=self.current_time,
            propagate_velocity=self.propagate_velocity,
            only_expand_when_occluded=self.only_expand_when_occluded,
            state_slice=self.state_slice,
            state_dim=self.state_dim,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            margin_offset=self.margin_offset + float(delta),
        )
