"""Semantic Control Barrier Function (CBF) & Signal Temporal Logic (STL) compiler.

Translates declarative physical safety rules and STL specifications
(``avoid``, ``keep_in_box``, ``keep_upright``, ``speed_limit``, ``force_limit``,
``always``, ``eventually``) bound against a dynamic 3D ``scene_graph`` into
live ``BarrierConstraint`` instances compatible with ``StateCBFSafetyFilter``.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .state_cbf_filter import BarrierConstraint, HalfPlaneBarrier, SphereBarrier

try:  # Optional whole-body barrier extension if installed in environment.
    from .whole_body_barriers import TiltConeBarrier as _ExternalTiltConeBarrier
except ImportError:  # pragma: no cover - local fallback provided below
    _ExternalTiltConeBarrier = None

VALID_RULE_TYPES = frozenset(
    {
        "avoid",
        "keep_in_box",
        "keep_upright",
        "speed_limit",
        "force_limit",
        "always",
        "eventually",
    }
)


def _embed_grad(sub_grad: np.ndarray, full_dim: int, state_slice: Optional[slice]) -> np.ndarray:
    if state_slice is None:
        if sub_grad.shape[0] == full_dim:
            return sub_grad
        out = np.zeros(full_dim, dtype=np.float64)
        n = min(full_dim, sub_grad.shape[0])
        out[:n] = sub_grad[:n]
        return out
    out = np.zeros(full_dim, dtype=np.float64)
    out[state_slice] = sub_grad
    return out


class SubspaceSphereBarrier(SphereBarrier):
    """Sphere avoidance barrier operating on a slice of the full state vector."""

    def __init__(
        self,
        center: Sequence[float],
        radius: float,
        alpha: float = 0.5,
        state_slice: Optional[slice] = slice(0, 3),
        state_dim: Optional[int] = None,
        entity: Optional[str] = None,
    ):
        super().__init__(center=center, radius=radius, alpha=alpha)
        self.state_slice = state_slice
        self.state_dim = None if state_dim is None else int(state_dim)
        self.entity = entity

    def _slice_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        if self.state_slice is None:
            return arr[: self.c.shape[0]] if arr.shape[0] >= self.c.shape[0] else arr
        return arr[self.state_slice]

    def h(self, x: np.ndarray) -> float:
        d = self._slice_x(x) - self.c
        return float(np.dot(d, d) - self.r * self.r)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        sub = 2.0 * (self._slice_x(arr) - self.c)
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        return _embed_grad(sub, full_dim, self.state_slice)

    def inflate(self, delta: float) -> "SubspaceSphereBarrier":
        return SubspaceSphereBarrier(
            center=self.c,
            radius=self.r + float(delta),
            alpha=self.alpha,
            state_slice=self.state_slice,
            state_dim=self.state_dim,
            entity=self.entity,
        )


class SubspaceHalfPlaneBarrier(HalfPlaneBarrier):
    """Half-plane barrier operating on a slice of the full state vector."""

    def __init__(
        self,
        a: Sequence[float],
        b: float,
        alpha: float = 0.5,
        state_slice: Optional[slice] = slice(0, 3),
        state_dim: Optional[int] = None,
        entity: Optional[str] = None,
    ):
        a_sub = np.asarray(a, dtype=np.float64).flatten()
        super().__init__(a=a_sub, b=b, alpha=alpha)
        self.a_sub = a_sub
        self.state_slice = state_slice
        self.state_dim = None if state_dim is None else int(state_dim)
        self.entity = entity

    def _slice_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        if self.state_slice is None:
            return arr[: self.a_sub.shape[0]] if arr.shape[0] >= self.a_sub.shape[0] else arr
        return arr[self.state_slice]

    def h(self, x: np.ndarray) -> float:
        return float(np.dot(self.a_sub, self._slice_x(x)) - self.b)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        return _embed_grad(self.a_sub, full_dim, self.state_slice)

    def inflate(self, delta: float) -> "SubspaceHalfPlaneBarrier":
        return SubspaceHalfPlaneBarrier(
            a=self.a_sub,
            b=self.b + float(delta),
            alpha=self.alpha,
            state_slice=self.state_slice,
            state_dim=self.state_dim,
            entity=self.entity,
        )


class NormBoundBarrier(BarrierConstraint):
    """Euclidean norm upper bound: h(x) = bound^2 - ||x_slc||^2 - offset >= 0."""

    def __init__(
        self,
        bound: float,
        alpha: float = 0.5,
        state_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
        label: str = "norm_limit",
        offset: float = 0.0,
    ):
        self.bound = float(bound)
        self.alpha = float(alpha)
        self.state_slice = state_slice
        self.state_dim = None if state_dim is None else int(state_dim)
        self.label = label
        self.offset = float(offset)

    def _slice_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        return arr if self.state_slice is None else arr[self.state_slice]

    def h(self, x: np.ndarray) -> float:
        sub = self._slice_x(x)
        return float(self.bound * self.bound - np.dot(sub, sub) - self.offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        sub = -2.0 * self._slice_x(arr)
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        return _embed_grad(sub, full_dim, self.state_slice)

    def inflate(self, delta: float) -> "NormBoundBarrier":
        return NormBoundBarrier(
            bound=self.bound,
            alpha=self.alpha,
            state_slice=self.state_slice,
            state_dim=self.state_dim,
            label=self.label,
            offset=self.offset + float(delta),
        )


class TiltConeBarrier(BarrierConstraint):
    """Orientation tilt-cone barrier keeping tilt angle within max_tilt_rad."""

    def __init__(
        self,
        max_tilt_deg: float = 15.0,
        alpha: float = 0.5,
        tilt_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
        offset: float = 0.0,
    ):
        self.max_tilt_deg = float(max_tilt_deg)
        self.max_tilt_rad = math.radians(self.max_tilt_deg)
        self.alpha = float(alpha)
        self.tilt_slice = tilt_slice
        self.state_dim = None if state_dim is None else int(state_dim)
        self.offset = float(offset)

    def _slice_x(self, x: np.ndarray) -> Tuple[np.ndarray, slice]:
        arr = np.asarray(x, dtype=np.float64).flatten()
        if self.tilt_slice is not None:
            return arr[self.tilt_slice], self.tilt_slice
        slc = slice(0, min(2, arr.shape[0]))
        return arr[slc], slc

    def h(self, x: np.ndarray) -> float:
        sub, _ = self._slice_x(x)
        return float(self.max_tilt_rad * self.max_tilt_rad - np.dot(sub, sub) - self.offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        sub, slc = self._slice_x(arr)
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        return _embed_grad(-2.0 * sub, full_dim, slc)

    def inflate(self, delta: float) -> "TiltConeBarrier":
        return TiltConeBarrier(
            max_tilt_deg=self.max_tilt_deg,
            alpha=self.alpha,
            tilt_slice=self.tilt_slice,
            state_dim=self.state_dim,
            offset=self.offset + float(delta),
        )


class TimeVaryingBarrier(BarrierConstraint):
    """Time-varying CBF for finite-time STL reachability (shrinking funnel).

    Default funnel:
        h(x, t) = r(t)^2 - ||x_pos - x_target||^2 - offset >= 0
    where r(t) shrinks linearly from ``initial_radius`` at t=0 to
    ``target_radius`` at t=``horizon_s``, and stays at ``target_radius`` for t >= T.
    """

    def __init__(
        self,
        target: Sequence[float],
        target_radius: float = 0.2,
        horizon_s: float = 5.0,
        initial_radius: Optional[float] = None,
        alpha: float = 0.5,
        current_time: float = 0.0,
        state_slice: Optional[slice] = slice(0, 3),
        state_dim: Optional[int] = None,
        offset: float = 0.0,
        radius_schedule: Optional[Callable[[float], Tuple[float, float]]] = None,
    ):
        self.target = np.asarray(target, dtype=np.float64).flatten()
        self.target_radius = float(target_radius)
        self.horizon_s = float(horizon_s)
        if self.horizon_s <= 0.0:
            raise ValueError(f"horizon_s must be positive, got {self.horizon_s}")
        if self.target_radius <= 0.0:
            raise ValueError(f"target_radius must be positive, got {self.target_radius}")
        default_r0 = max(self.target_radius * 5.0, self.target_radius + 2.0)
        self.initial_radius = float(default_r0 if initial_radius is None else initial_radius)
        if self.initial_radius < self.target_radius:
            raise ValueError("initial_radius must be >= target_radius")
        self.alpha = float(alpha)
        self.current_time = float(current_time)
        self.state_slice = state_slice
        self.state_dim = None if state_dim is None else int(state_dim)
        self.offset = float(offset)
        self._radius_schedule = radius_schedule

    def set_time(self, t: float) -> None:
        self.current_time = float(t)

    def advance_time(self, dt: float) -> float:
        self.current_time += float(dt)
        return self.current_time

    def radius_at(self, t: float) -> float:
        r_t, _ = self._radius_and_dr(float(t))
        return r_t

    def _radius_and_dr(self, t: float) -> Tuple[float, float]:
        if self._radius_schedule is not None:
            return self._radius_schedule(t)
        if t <= 0.0:
            return self.initial_radius, (
                -(self.initial_radius - self.target_radius) / self.horizon_s if t == 0.0 else 0.0
            )
        if t >= self.horizon_s:
            return self.target_radius, 0.0
        slope = -(self.initial_radius - self.target_radius) / self.horizon_s
        return self.initial_radius + slope * t, slope

    def _slice_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).flatten()
        if self.state_slice is None:
            return arr[: self.target.shape[0]] if arr.shape[0] >= self.target.shape[0] else arr
        return arr[self.state_slice]

    def h_at(self, x: np.ndarray, t: float) -> float:
        r_t, _ = self._radius_and_dr(float(t))
        d = self._slice_x(x) - self.target
        return float(r_t * r_t - np.dot(d, d) - self.offset)

    def grad_h_t(self, x: np.ndarray, t: float) -> np.ndarray:
        _ = t  # spatial gradient of r(t)^2 - ||x - target||^2 does not depend on t
        arr = np.asarray(x, dtype=np.float64).flatten()
        sub = -2.0 * (self._slice_x(arr) - self.target)
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        return _embed_grad(sub, full_dim, self.state_slice)

    def dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> float:
        _ = x
        t_eval = self.current_time if t is None else float(t)
        r_t, dr_dt = self._radius_and_dr(t_eval)
        return float(2.0 * r_t * dr_dt)

    def d2h_dt2(self, x: np.ndarray, t: Optional[float] = None) -> float:
        _ = x
        t_eval = self.current_time if t is None else float(t)
        r_t, dr_dt = self._radius_and_dr(t_eval)
        if self._radius_schedule is not None:
            eps = 1e-5
            _, dr_p = self._radius_schedule(t_eval + eps)
            _, dr_m = self._radius_schedule(t_eval - eps)
            d2r_dt2 = (float(dr_p) - float(dr_m)) / (2.0 * eps)
            return float(2.0 * dr_dt * dr_dt + 2.0 * r_t * d2r_dt2)
        if t_eval < 0.0 or t_eval >= self.horizon_s:
            return 0.0
        return float(2.0 * dr_dt * dr_dt)

    def grad_dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        _ = t
        arr = np.asarray(x, dtype=np.float64).flatten()
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        return np.zeros(full_dim, dtype=np.float64)

    def h(self, x: np.ndarray) -> float:
        return self.h_at(x, self.current_time)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return self.grad_h_t(x, self.current_time)

    def inflate(self, delta: float) -> "TimeVaryingBarrier":
        return TimeVaryingBarrier(
            target=self.target,
            target_radius=self.target_radius,
            horizon_s=self.horizon_s,
            initial_radius=self.initial_radius,
            alpha=self.alpha,
            current_time=self.current_time,
            state_slice=self.state_slice,
            state_dim=self.state_dim,
            offset=self.offset + float(delta),
            radius_schedule=self._radius_schedule,
        )


def _embed_hess(sub_hess: np.ndarray, full_dim: int, state_slice: Optional[slice]) -> np.ndarray:
    if state_slice is None:
        if sub_hess.shape == (full_dim, full_dim):
            return sub_hess
        out = np.zeros((full_dim, full_dim), dtype=np.float64)
        n = min(full_dim, sub_hess.shape[0])
        out[:n, :n] = sub_hess[:n, :n]
        return out
    out = np.zeros((full_dim, full_dim), dtype=np.float64)
    out[state_slice, state_slice] = sub_hess
    return out


class MovingObstacleBarrier(BarrierConstraint):
    """Dynamic sphere obstacle CBF with exact velocity and acceleration derivatives.

    Enforces ``h(x, t) = ||x_pos - p_obs(t)||^2 - r_eff^2 >= 0`` where::
        p_obs(t) = p0 + v0 * t + 0.5 * a0 * t^2
        v_obs(t) = v0 + a0 * t
        a_obs(t) = a0
    """

    def __init__(
        self,
        center: Sequence[float],
        radius: float,
        velocity: Optional[Sequence[float]] = None,
        acceleration: Optional[Sequence[float]] = None,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        current_time: float = 0.0,
        state_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
        entity: Optional[str] = None,
    ):
        self.p0 = np.asarray(center, dtype=np.float64).flatten()
        d = self.p0.shape[0]
        if float(radius) <= 0.0:
            raise ValueError(f"MovingObstacleBarrier radius must be > 0, got {radius}")
        self.r = float(radius)
        self.v0 = (
            np.zeros(d, dtype=np.float64)
            if velocity is None
            else np.asarray(velocity, dtype=np.float64).flatten()
        )
        self.a0 = (
            np.zeros(d, dtype=np.float64)
            if acceleration is None
            else np.asarray(acceleration, dtype=np.float64).flatten()
        )
        if self.v0.shape != (d,) or self.a0.shape != (d,):
            raise ValueError(
                f"center ({d}D), velocity ({self.v0.shape}), and acceleration ({self.a0.shape}) must match"
            )
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.current_time = float(current_time)
        self.state_slice = state_slice
        self.state_dim = None if state_dim is None else int(state_dim)
        self.entity = entity

    def _effective_slice(self, n_full: int) -> slice:
        if self.state_slice is not None:
            return self.state_slice
        d = self.p0.shape[0]
        return slice(0, min(d, n_full))

    def _slice_x(self, x: np.ndarray) -> Tuple[np.ndarray, int, slice]:
        arr = np.asarray(x, dtype=np.float64).flatten()
        full_dim = self.state_dim if self.state_dim is not None else arr.shape[0]
        slc = self._effective_slice(arr.shape[0])
        return arr[slc], full_dim, slc

    def position_at(self, t: Optional[float] = None) -> np.ndarray:
        tau = self.current_time if t is None else float(t)
        return self.p0 + self.v0 * tau + 0.5 * self.a0 * (tau * tau)

    def velocity_at(self, t: Optional[float] = None) -> np.ndarray:
        tau = self.current_time if t is None else float(t)
        return self.v0 + self.a0 * tau

    def acceleration_at(self, t: Optional[float] = None) -> np.ndarray:
        _ = t
        return self.a0.copy()

    def set_time(self, t: float) -> None:
        self.current_time = float(t)

    def advance_time(self, dt: float) -> float:
        self.current_time += float(dt)
        return self.current_time

    def update_kinematics(
        self,
        center: Sequence[float],
        velocity: Sequence[float],
        acceleration: Optional[Sequence[float]] = None,
        radius: Optional[float] = None,
        reset_time: bool = True,
    ) -> None:
        """Update obstacle state in-place from a live tracker or perception frame."""
        d = self.p0.shape[0]
        self.p0 = np.asarray(center, dtype=np.float64).flatten()
        self.v0 = np.asarray(velocity, dtype=np.float64).flatten()
        self.a0 = (
            np.zeros(d, dtype=np.float64)
            if acceleration is None
            else np.asarray(acceleration, dtype=np.float64).flatten()
        )
        if radius is not None:
            self.r = float(radius)
        if reset_time:
            self.current_time = 0.0

    def h_at(self, x: np.ndarray, t: float) -> float:
        x_pos, _, _ = self._slice_x(x)
        diff = x_pos - self.position_at(t)
        return float(np.dot(diff, diff) - self.r * self.r)

    def h(self, x: np.ndarray) -> float:
        return self.h_at(x, self.current_time)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_pos, full_dim, slc = self._slice_x(x)
        sub = 2.0 * (x_pos - self.position_at(self.current_time))
        return _embed_grad(sub, full_dim, slc)

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        _ = eps
        x_pos, full_dim, slc = self._slice_x(x)
        d = x_pos.shape[0]
        return _embed_hess(2.0 * np.eye(d, dtype=np.float64), full_dim, slc)

    def dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> float:
        tau = self.current_time if t is None else float(t)
        x_pos, _, _ = self._slice_x(x)
        diff = x_pos - self.position_at(tau)
        v_obs = self.velocity_at(tau)
        return float(-2.0 * np.dot(diff, v_obs))

    def d2h_dt2(self, x: np.ndarray, t: Optional[float] = None) -> float:
        tau = self.current_time if t is None else float(t)
        x_pos, _, _ = self._slice_x(x)
        diff = x_pos - self.position_at(tau)
        v_obs = self.velocity_at(tau)
        a_obs = self.acceleration_at(tau)
        return float(2.0 * np.dot(v_obs, v_obs) - 2.0 * np.dot(diff, a_obs))

    def grad_dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        tau = self.current_time if t is None else float(t)
        _, full_dim, slc = self._slice_x(x)
        sub = -2.0 * self.velocity_at(tau)
        return _embed_grad(sub, full_dim, slc)

    def inflate(self, delta: float) -> "MovingObstacleBarrier":
        return MovingObstacleBarrier(
            center=self.p0,
            radius=self.r + float(delta),
            velocity=self.v0,
            acceleration=self.a0,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            current_time=self.current_time,
            state_slice=self.state_slice,
            state_dim=self.state_dim,
            entity=self.entity,
        )


class KalmanObstacleTracker:
    """Linear Kalman filter (constant-velocity or constant-acceleration) for 3D obstacle tracking.

    Tracks ``(pos, vel, cov)`` (plus ``acc`` when ``model='ca'``) from noisy
    position detections ``(z_pos, dt)`` and computes certified position/velocity
    uncertainty radii for ``MovingObstacleBarrier`` inflation.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        model: str = "cv",
        process_noise: float = 0.5,
        measurement_noise: float = 0.05,
        initial_pos: Optional[Sequence[float]] = None,
        initial_vel: Optional[Sequence[float]] = None,
        initial_acc: Optional[Sequence[float]] = None,
        initial_pos_var: float = 0.25,
        initial_vel_var: float = 1.0,
        initial_acc_var: float = 2.0,
        confidence_sigma: float = 2.0,
    ):
        model_norm = str(model).strip().lower()
        if model_norm in ("cv", "constant_velocity"):
            self.model = "cv"
            self._order = 2
        elif model_norm in ("ca", "constant_acceleration"):
            self.model = "ca"
            self._order = 3
        else:
            raise ValueError(f"Unsupported Kalman model {model!r}; expected 'cv' or 'ca'")

        if initial_pos is not None:
            p_init = np.asarray(initial_pos, dtype=np.float64).flatten()
            self.dim = p_init.shape[0] if dim is None else int(dim)
            if p_init.shape[0] != self.dim:
                raise ValueError(f"initial_pos length {p_init.shape[0]} != dim {self.dim}")
            self._initialized = True
        else:
            self.dim = 3 if dim is None else int(dim)
            p_init = np.zeros(self.dim, dtype=np.float64)
            self._initialized = False

        if self.dim <= 0:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        self.process_noise = max(1e-9, float(process_noise))
        self.measurement_noise = max(1e-9, float(measurement_noise))
        self.initial_pos_var = max(1e-9, float(initial_pos_var))
        self.initial_vel_var = max(1e-9, float(initial_vel_var))
        self.initial_acc_var = max(1e-9, float(initial_acc_var))
        self.confidence_sigma = max(0.0, float(confidence_sigma))

        v_init = (
            np.zeros(self.dim, dtype=np.float64)
            if initial_vel is None
            else np.asarray(initial_vel, dtype=np.float64).flatten()
        )
        if v_init.shape != (self.dim,):
            raise ValueError(f"initial_vel shape {v_init.shape} != ({self.dim},)")
        if initial_vel is not None:
            self._initialized = True

        n_s = self._order * self.dim
        self._state = np.zeros(n_s, dtype=np.float64)
        self._state[: self.dim] = p_init
        self._state[self.dim : 2 * self.dim] = v_init
        if self._order == 3 and initial_acc is not None:
            a_init = np.asarray(initial_acc, dtype=np.float64).flatten()
            if a_init.shape != (self.dim,):
                raise ValueError(f"initial_acc shape {a_init.shape} != ({self.dim},)")
            self._state[2 * self.dim : 3 * self.dim] = a_init
            self._initialized = True

        diag_blocks = [
            np.full(self.dim, self.initial_pos_var, dtype=np.float64),
            np.full(self.dim, self.initial_vel_var, dtype=np.float64),
        ]
        if self._order == 3:
            diag_blocks.append(np.full(self.dim, self.initial_acc_var, dtype=np.float64))
        self._cov = np.diag(np.concatenate(diag_blocks))
        self._H = np.zeros((self.dim, n_s), dtype=np.float64)
        self._H[:, : self.dim] = np.eye(self.dim, dtype=np.float64)
        self.steps = 0

    @property
    def pos(self) -> np.ndarray:
        return self._state[: self.dim].copy()

    @property
    def vel(self) -> np.ndarray:
        return self._state[self.dim : 2 * self.dim].copy()

    @property
    def acc(self) -> np.ndarray:
        if self._order == 3:
            return self._state[2 * self.dim : 3 * self.dim].copy()
        return np.zeros(self.dim, dtype=np.float64)

    @property
    def cov(self) -> np.ndarray:
        return self._cov.copy()

    @property
    def pos_cov(self) -> np.ndarray:
        return self._cov[: self.dim, : self.dim].copy()

    @property
    def vel_cov(self) -> np.ndarray:
        return self._cov[self.dim : 2 * self.dim, self.dim : 2 * self.dim].copy()

    def _transition_and_noise(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        dt_f = float(dt)
        if dt_f <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        d = self.dim
        eye_d = np.eye(d, dtype=np.float64)
        q_var = self.process_noise * self.process_noise
        if self._order == 2:
            F = np.block([[eye_d, dt_f * eye_d], [np.zeros((d, d)), eye_d]])
            G_w = np.vstack([0.5 * (dt_f * dt_f) * eye_d, dt_f * eye_d])
            Q = q_var * (G_w @ G_w.T) + 1e-9 * np.eye(2 * d, dtype=np.float64)
            return F, Q
        z = np.zeros((d, d), dtype=np.float64)
        F = np.block(
            [
                [eye_d, dt_f * eye_d, 0.5 * dt_f * dt_f * eye_d],
                [z, eye_d, dt_f * eye_d],
                [z, z, eye_d],
            ]
        )
        G_w = np.vstack([(dt_f**3 / 6.0) * eye_d, 0.5 * (dt_f * dt_f) * eye_d, dt_f * eye_d])
        Q = q_var * (G_w @ G_w.T) + 1e-9 * np.eye(3 * d, dtype=np.float64)
        return F, Q

    def position_uncertainty_radius(self, sigma: Optional[float] = None) -> float:
        k = self.confidence_sigma if sigma is None else float(sigma)
        eig_max = float(np.max(np.linalg.eigvalsh(self.pos_cov)))
        return k * math.sqrt(max(0.0, eig_max))

    def velocity_uncertainty_radius(self, sigma: Optional[float] = None) -> float:
        k = self.confidence_sigma if sigma is None else float(sigma)
        eig_max = float(np.max(np.linalg.eigvalsh(self.vel_cov)))
        return k * math.sqrt(max(0.0, eig_max))

    def uncertainty_radii(self, sigma: Optional[float] = None) -> Tuple[float, float]:
        return self.position_uncertainty_radius(sigma), self.velocity_uncertainty_radius(sigma)

    def predict(self, dt: float) -> Dict[str, Any]:
        F, Q = self._transition_and_noise(dt)
        self._state = F @ self._state
        self._cov = F @ self._cov @ F.T + Q
        self._cov = 0.5 * (self._cov + self._cov.T)
        return {
            "pos": self.pos,
            "vel": self.vel,
            "acc": self.acc,
            "cov": self.cov,
            "pos_uncertainty_radius": self.position_uncertainty_radius(),
            "vel_uncertainty_radius": self.velocity_uncertainty_radius(),
        }

    def update(self, z_pos: Sequence[float], dt: float = 0.05) -> Dict[str, Any]:
        z = np.asarray(z_pos, dtype=np.float64).flatten()
        if z.shape != (self.dim,):
            raise ValueError(f"z_pos shape {z.shape} != expected ({self.dim},)")
        r_var = self.measurement_noise * self.measurement_noise
        R = r_var * np.eye(self.dim, dtype=np.float64)

        if not self._initialized:
            self._state[: self.dim] = z
            self._cov[: self.dim, : self.dim] = R.copy()
            self._initialized = True
            self.steps += 1
        else:
            self.predict(dt)
            S = self._H @ self._cov @ self._H.T + R
            K = np.linalg.solve(S, self._H @ self._cov).T
            innov = z - (self._H @ self._state)
            self._state = self._state + K @ innov
            I_kh = np.eye(self._state.shape[0], dtype=np.float64) - K @ self._H
            self._cov = I_kh @ self._cov @ I_kh.T + K @ R @ K.T
            self._cov = 0.5 * (self._cov + self._cov.T)
            self.steps += 1

        return {
            "pos": self.pos,
            "vel": self.vel,
            "acc": self.acc,
            "cov": self.cov,
            "pos_uncertainty_radius": self.position_uncertainty_radius(),
            "vel_uncertainty_radius": self.velocity_uncertainty_radius(),
        }

    def to_moving_barrier(
        self,
        base_radius: float,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        sigma: Optional[float] = None,
        vel_horizon_s: float = 0.0,
        state_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
        entity: Optional[str] = None,
    ) -> MovingObstacleBarrier:
        pos_unc, vel_unc = self.uncertainty_radii(sigma=sigma)
        r_eff = float(base_radius) + pos_unc + max(0.0, float(vel_horizon_s)) * vel_unc
        return MovingObstacleBarrier(
            center=self.pos,
            radius=r_eff,
            velocity=self.vel,
            acceleration=self.acc if self.model == "ca" else None,
            alpha=alpha,
            alpha2=alpha2,
            relative_degree=relative_degree,
            state_slice=state_slice,
            state_dim=state_dim,
            entity=entity,
        )


class STLAndBarrier(BarrierConstraint):
    """Sound smooth STL conjunction barrier: h_and(x, t) = -(1/eta) * log(sum_i exp(-eta * h_i(x, t))).

    Satisfies ``h_and(x, t) <= min_i h_i(x, t)``, so ``h_and(x, t) >= 0`` strictly
    guarantees ``h_i(x, t) >= 0`` for every child barrier ``i``.
    """

    def __init__(
        self,
        barriers: Sequence[BarrierConstraint],
        eta: float = 20.0,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        offset: float = 0.0,
    ):
        if not barriers:
            raise ValueError("STLAndBarrier requires at least 1 child barrier")
        if float(eta) <= 0.0:
            raise ValueError(f"eta must be > 0, got {eta}")
        self.barriers: List[BarrierConstraint] = list(barriers)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.offset = float(offset)

    def _vals_and_weights(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h_vals = np.asarray([float(b.h(x)) for b in self.barriers], dtype=np.float64)
        m = float(np.min(h_vals))
        exps = np.exp(-self.eta * (h_vals - m))
        s_exp = float(np.sum(exps))
        weights = exps / s_exp
        h_and = m - (math.log(s_exp) / self.eta) - self.offset
        return h_vals, weights, float(h_and)

    def h(self, x: np.ndarray) -> float:
        _, _, val = self._vals_and_weights(x)
        return val

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        _, weights, _ = self._vals_and_weights(x)
        grads = np.vstack([np.asarray(b.grad_h(x), dtype=np.float64).flatten() for b in self.barriers])
        return np.asarray(weights @ grads, dtype=np.float64).flatten()

    def dh_dt(self, x: np.ndarray) -> float:
        _, weights, _ = self._vals_and_weights(x)
        dh_vals = np.asarray([float(b.dh_dt(x)) for b in self.barriers], dtype=np.float64)
        return float(np.dot(weights, dh_vals))

    def d2h_dt2(self, x: np.ndarray) -> float:
        _, weights, _ = self._vals_and_weights(x)
        dh_vals = np.asarray([float(b.dh_dt(x)) for b in self.barriers], dtype=np.float64)
        d2h_vals = np.asarray([float(b.d2h_dt2(x)) for b in self.barriers], dtype=np.float64)
        dh_bar = float(np.dot(weights, dh_vals))
        var_term = float(np.dot(weights, np.square(dh_vals - dh_bar)))
        return float(np.dot(weights, d2h_vals) - self.eta * var_term)

    def grad_dh_dt(self, x: np.ndarray) -> np.ndarray:
        _, weights, _ = self._vals_and_weights(x)
        grads = np.vstack([np.asarray(b.grad_h(x), dtype=np.float64).flatten() for b in self.barriers])
        g_dh = np.vstack([np.asarray(b.grad_dh_dt(x), dtype=np.float64).flatten() for b in self.barriers])
        dh_vals = np.asarray([float(b.dh_dt(x)) for b in self.barriers], dtype=np.float64)
        dh_bar = float(np.dot(weights, dh_vals))
        weighted_diff = weights * (dh_vals - dh_bar)
        return np.asarray((weights @ g_dh) - self.eta * (weighted_diff @ grads), dtype=np.float64).flatten()

    def set_time(self, t: float) -> None:
        seen: set[int] = set()
        for b in self.barriers:
            if id(b) in seen:
                continue
            seen.add(id(b))
            if hasattr(b, "set_time") and callable(b.set_time):
                b.set_time(float(t))
            elif hasattr(b, "current_time"):
                b.current_time = float(t)

    def advance_time(self, dt: float) -> float:
        seen: set[int] = set()
        last_t = 0.0
        for b in self.barriers:
            if id(b) in seen:
                continue
            seen.add(id(b))
            if hasattr(b, "advance_time") and callable(b.advance_time):
                last_t = float(b.advance_time(float(dt)))
        return last_t

    def inflate(self, delta: float) -> "STLAndBarrier":
        return STLAndBarrier(
            barriers=self.barriers,
            eta=self.eta,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            offset=self.offset + float(delta),
        )


class STLOrBarrier(BarrierConstraint):
    """Smooth STL disjunction barrier: h_or(x, t) = (1/eta) * log(sum_i exp(eta * h_i(x, t))) - shift.

    When ``sound_inner=True`` (default), ``shift = ln(N) / eta`` so that
    ``h_or(x, t) <= max_i h_i(x, t)``, ensuring ``h_or(x, t) >= 0 => exists i: h_i(x, t) >= 0``.
    """

    def __init__(
        self,
        barriers: Sequence[BarrierConstraint],
        eta: float = 25.0,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        sound_inner: bool = True,
        offset: float = 0.0,
    ):
        if not barriers:
            raise ValueError("STLOrBarrier requires at least 1 child barrier")
        if float(eta) <= 0.0:
            raise ValueError(f"eta must be > 0, got {eta}")
        self.barriers: List[BarrierConstraint] = list(barriers)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.sound_inner = bool(sound_inner)
        self.offset = float(offset)

    def _vals_and_weights(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h_vals = np.asarray([float(b.h(x)) for b in self.barriers], dtype=np.float64)
        m = float(np.max(h_vals))
        exps = np.exp(self.eta * (h_vals - m))
        s_exp = float(np.sum(exps))
        weights = exps / s_exp
        shift = (math.log(len(self.barriers)) / self.eta) if self.sound_inner else 0.0
        h_or = m + (math.log(s_exp) / self.eta) - shift - self.offset
        return h_vals, weights, float(h_or)

    def h(self, x: np.ndarray) -> float:
        _, _, val = self._vals_and_weights(x)
        return val

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        _, weights, _ = self._vals_and_weights(x)
        grads = np.vstack([np.asarray(b.grad_h(x), dtype=np.float64).flatten() for b in self.barriers])
        return np.asarray(weights @ grads, dtype=np.float64).flatten()

    def dh_dt(self, x: np.ndarray) -> float:
        _, weights, _ = self._vals_and_weights(x)
        dh_vals = np.asarray([float(b.dh_dt(x)) for b in self.barriers], dtype=np.float64)
        return float(np.dot(weights, dh_vals))

    def d2h_dt2(self, x: np.ndarray) -> float:
        _, weights, _ = self._vals_and_weights(x)
        dh_vals = np.asarray([float(b.dh_dt(x)) for b in self.barriers], dtype=np.float64)
        d2h_vals = np.asarray([float(b.d2h_dt2(x)) for b in self.barriers], dtype=np.float64)
        dh_bar = float(np.dot(weights, dh_vals))
        var_term = float(np.dot(weights, np.square(dh_vals - dh_bar)))
        return float(np.dot(weights, d2h_vals) + self.eta * var_term)

    def grad_dh_dt(self, x: np.ndarray) -> np.ndarray:
        _, weights, _ = self._vals_and_weights(x)
        grads = np.vstack([np.asarray(b.grad_h(x), dtype=np.float64).flatten() for b in self.barriers])
        g_dh = np.vstack([np.asarray(b.grad_dh_dt(x), dtype=np.float64).flatten() for b in self.barriers])
        dh_vals = np.asarray([float(b.dh_dt(x)) for b in self.barriers], dtype=np.float64)
        dh_bar = float(np.dot(weights, dh_vals))
        weighted_diff = weights * (dh_vals - dh_bar)
        return np.asarray((weights @ g_dh) + self.eta * (weighted_diff @ grads), dtype=np.float64).flatten()

    def set_time(self, t: float) -> None:
        seen: set[int] = set()
        for b in self.barriers:
            if id(b) in seen:
                continue
            seen.add(id(b))
            if hasattr(b, "set_time") and callable(b.set_time):
                b.set_time(float(t))
            elif hasattr(b, "current_time"):
                b.current_time = float(t)

    def advance_time(self, dt: float) -> float:
        seen: set[int] = set()
        last_t = 0.0
        for b in self.barriers:
            if id(b) in seen:
                continue
            seen.add(id(b))
            if hasattr(b, "advance_time") and callable(b.advance_time):
                last_t = float(b.advance_time(float(dt)))
        return last_t

    def inflate(self, delta: float) -> "STLOrBarrier":
        return STLOrBarrier(
            barriers=self.barriers,
            eta=self.eta,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            sound_inner=self.sound_inner,
            offset=self.offset + float(delta),
        )


class STLUntilBarrier(BarrierConstraint):
    """Smooth temporal STL Until barrier: phi_hold U_[t_start, t_end] phi_goal.

    Strictly enforces ``hold_barrier`` over ``t in [t_start, t_switch]`` and blends
    via a C^2 quintic schedule into ``goal_barrier`` so ``goal_barrier`` is enforced
    through ``t_end`` and beyond.
    """

    def __init__(
        self,
        hold_barrier: BarrierConstraint,
        goal_barrier: BarrierConstraint,
        t_start: float = 0.0,
        t_switch: Optional[float] = None,
        t_end: float = 5.0,
        blend_duration: Optional[float] = None,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        current_time: float = 0.0,
        offset: float = 0.0,
    ):
        self.hold_barrier = hold_barrier
        self.goal_barrier = goal_barrier
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be > t_start ({self.t_start})")
        default_sw = 0.5 * (self.t_start + self.t_end)
        self.t_switch = float(default_sw if t_switch is None else t_switch)
        if not (self.t_start <= self.t_switch <= self.t_end):
            raise ValueError(
                f"t_switch ({self.t_switch}) must lie within [t_start={self.t_start}, t_end={self.t_end}]"
            )
        rem = self.t_end - self.t_switch
        if blend_duration is not None:
            self.blend_duration = max(1e-6, min(float(blend_duration), max(1e-6, rem)))
        elif rem > 1e-6:
            self.blend_duration = 0.5 * rem
        else:
            self.blend_duration = max(1e-6, 0.25 * (self.t_end - self.t_start))
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.current_time = float(current_time)
        self.offset = float(offset)
        self.set_time(self.current_time)

    def _blend_weights(self, t: float) -> Tuple[float, float, float]:
        """Return C^2 quintic blend (s(t), ds/dt, d2s/dt2)."""
        t_a = self.t_switch if self.t_end > self.t_switch else (self.t_end - self.blend_duration)
        t_b = t_a + self.blend_duration
        if t <= t_a:
            return 0.0, 0.0, 0.0
        if t >= t_b:
            return 1.0, 0.0, 0.0
        w = t_b - t_a
        tau = (t - t_a) / w
        tau2 = tau * tau
        tau3 = tau2 * tau
        s = tau3 * (10.0 + tau * (-15.0 + 6.0 * tau))
        ds = (30.0 * tau2 * (1.0 - tau) * (1.0 - tau)) / w
        d2s = (60.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau)) / (w * w)
        return float(s), float(ds), float(d2s)

    def set_time(self, t: float) -> None:
        self.current_time = float(t)
        seen: set[int] = set()
        for child in (self.hold_barrier, self.goal_barrier):
            if id(child) in seen:
                continue
            seen.add(id(child))
            if hasattr(child, "set_time") and callable(child.set_time):
                child.set_time(self.current_time)
            elif hasattr(child, "current_time"):
                child.current_time = self.current_time

    def advance_time(self, dt: float) -> float:
        self.current_time += float(dt)
        seen: set[int] = set()
        for child in (self.hold_barrier, self.goal_barrier):
            if id(child) in seen:
                continue
            seen.add(id(child))
            if hasattr(child, "advance_time") and callable(child.advance_time):
                child.advance_time(float(dt))
            elif hasattr(child, "current_time"):
                child.current_time = self.current_time
        return self.current_time

    def h_at(self, x: np.ndarray, t: float) -> float:
        prev = self.current_time
        try:
            self.set_time(t)
            return self.h(x)
        finally:
            self.set_time(prev)

    def h(self, x: np.ndarray) -> float:
        s, _, _ = self._blend_weights(self.current_time)
        h1 = float(self.hold_barrier.h(x))
        h2 = float(self.goal_barrier.h(x))
        return float((1.0 - s) * h1 + s * h2 - self.offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        s, _, _ = self._blend_weights(self.current_time)
        g1 = np.asarray(self.hold_barrier.grad_h(x), dtype=np.float64).flatten()
        g2 = np.asarray(self.goal_barrier.grad_h(x), dtype=np.float64).flatten()
        return (1.0 - s) * g1 + s * g2

    def dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> float:
        if t is not None and float(t) != self.current_time:
            prev = self.current_time
            try:
                self.set_time(float(t))
                return self._dh_dt_current(x)
            finally:
                self.set_time(prev)
        return self._dh_dt_current(x)

    def _dh_dt_current(self, x: np.ndarray) -> float:
        s, ds, _ = self._blend_weights(self.current_time)
        h1 = float(self.hold_barrier.h(x))
        h2 = float(self.goal_barrier.h(x))
        dh1 = float(self.hold_barrier.dh_dt(x))
        dh2 = float(self.goal_barrier.dh_dt(x))
        return float(ds * (h2 - h1) + (1.0 - s) * dh1 + s * dh2)

    def d2h_dt2(self, x: np.ndarray, t: Optional[float] = None) -> float:
        if t is not None and float(t) != self.current_time:
            prev = self.current_time
            try:
                self.set_time(float(t))
                return self._d2h_dt2_current(x)
            finally:
                self.set_time(prev)
        return self._d2h_dt2_current(x)

    def _d2h_dt2_current(self, x: np.ndarray) -> float:
        s, ds, d2s = self._blend_weights(self.current_time)
        h1 = float(self.hold_barrier.h(x))
        h2 = float(self.goal_barrier.h(x))
        dh1 = float(self.hold_barrier.dh_dt(x))
        dh2 = float(self.goal_barrier.dh_dt(x))
        d2h1 = float(self.hold_barrier.d2h_dt2(x))
        d2h2 = float(self.goal_barrier.d2h_dt2(x))
        return float(d2s * (h2 - h1) + 2.0 * ds * (dh2 - dh1) + (1.0 - s) * d2h1 + s * d2h2)

    def grad_dh_dt(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is not None and float(t) != self.current_time:
            prev = self.current_time
            try:
                self.set_time(float(t))
                return self._grad_dh_dt_current(x)
            finally:
                self.set_time(prev)
        return self._grad_dh_dt_current(x)

    def _grad_dh_dt_current(self, x: np.ndarray) -> np.ndarray:
        s, ds, _ = self._blend_weights(self.current_time)
        g1 = np.asarray(self.hold_barrier.grad_h(x), dtype=np.float64).flatten()
        g2 = np.asarray(self.goal_barrier.grad_h(x), dtype=np.float64).flatten()
        g_dh1 = np.asarray(self.hold_barrier.grad_dh_dt(x), dtype=np.float64).flatten()
        g_dh2 = np.asarray(self.goal_barrier.grad_dh_dt(x), dtype=np.float64).flatten()
        return ds * (g2 - g1) + (1.0 - s) * g_dh1 + s * g_dh2

    def inflate(self, delta: float) -> "STLUntilBarrier":
        return STLUntilBarrier(
            hold_barrier=self.hold_barrier,
            goal_barrier=self.goal_barrier,
            t_start=self.t_start,
            t_switch=self.t_switch,
            t_end=self.t_end,
            blend_duration=self.blend_duration,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            current_time=self.current_time,
            offset=self.offset + float(delta),
        )


@dataclass
class PhysicalSafetyRule:
    """Declarative physical safety or STL specification rule."""

    rule_type: str
    entity: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    alpha: float = 0.5
    hard: bool = True
    raw_spec: Optional[str] = None

    def validate(self) -> None:
        if self.rule_type not in VALID_RULE_TYPES:
            raise ValueError(
                f"Unsupported rule_type {self.rule_type!r}; expected one of {sorted(VALID_RULE_TYPES)}"
            )
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError(f"alpha must be finite and > 0, got {self.alpha}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "entity": self.entity,
            "params": dict(self.params),
            "alpha": float(self.alpha),
            "hard": bool(self.hard),
            "raw_spec": self.raw_spec,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhysicalSafetyRule":
        rule = cls(
            rule_type=str(data["rule_type"]),
            entity=data.get("entity"),
            params=dict(data.get("params") or {}),
            alpha=float(data.get("alpha", 0.5)),
            hard=bool(data.get("hard", True)),
            raw_spec=data.get("raw_spec"),
        )
        rule.validate()
        return rule


class PhysicalSafetyIR:
    """Serializable intermediate representation binding safety rules to a scene graph."""

    def __init__(
        self,
        rules: Optional[Sequence[PhysicalSafetyRule]] = None,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.rules: List[PhysicalSafetyRule] = list(rules or [])
        self.version = str(version)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.validate()

    def validate(self) -> None:
        for rule in self.rules:
            rule.validate()

    def add_rule(self, rule: PhysicalSafetyRule) -> None:
        rule.validate()
        self.rules.append(rule)

    def bind(
        self,
        scene_graph: Optional[Dict[str, Any]],
        fail_on_unresolved: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Bind rules against ``scene_graph['entities']`` and report unresolved entities."""
        entities: Dict[str, Any] = {}
        if isinstance(scene_graph, dict):
            raw_entities = scene_graph.get("entities")
            if isinstance(raw_entities, dict):
                entities = raw_entities

        bound: List[Dict[str, Any]] = []
        unresolved: List[str] = []

        for rule in self.rules:
            rule.validate()
            ref_entity = rule.entity or rule.params.get("entity") or rule.params.get("target_entity")
            resolved_geom: Optional[Dict[str, Any]] = None

            needs_entity = ref_entity is not None
            if rule.rule_type == "avoid" and "center" not in rule.params:
                needs_entity = True
            if rule.rule_type == "eventually" and "target" not in rule.params:
                needs_entity = True

            if needs_entity:
                if not ref_entity or ref_entity not in entities:
                    label = ref_entity or f"<missing_entity:{rule.rule_type}>"
                    unresolved.append(label)
                    continue
                geom = entities[ref_entity]
                if not isinstance(geom, dict):
                    unresolved.append(ref_entity)
                    continue
                resolved_geom = dict(geom)

            bound.append({"rule": rule, "entity_name": ref_entity, "geometry": resolved_geom})

        if unresolved and fail_on_unresolved:
            raise ValueError(f"Unresolved safety entities in scene_graph: {unresolved}")
        return bound, unresolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metadata": dict(self.metadata),
            "rules": [r.to_dict() for r in self.rules],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhysicalSafetyIR":
        rules = [PhysicalSafetyRule.from_dict(r) for r in data.get("rules", [])]
        return cls(
            rules=rules,
            version=str(data.get("version", "1.0")),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, payload: str) -> "PhysicalSafetyIR":
        return cls.from_dict(json.loads(payload))


def _eval_literal(node: ast.AST) -> Any:
    """Safely convert an AST literal (numbers, negated numbers, strings, lists)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        val = _eval_literal(node.operand)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError("Unary operator only supported on numeric literals")
        return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.List):
        return [_eval_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return [_eval_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {_eval_literal(k): _eval_literal(v) for k, v in zip(node.keys, node.values) if k is not None}
    return ast.literal_eval(node)


class SemanticCBFCompiler:
    """Parses declarative safety DSL rules and compiles them to CBF barriers."""

    def parse_rule_string(self, spec: str) -> PhysicalSafetyRule:
        text = spec.strip()
        if not text:
            raise ValueError("Empty safety rule specification")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid safety rule syntax {spec!r}: {exc}") from exc

        node = tree.body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"Safety rule must be a function call, got {spec!r}")

        name = node.func.id
        if name not in VALID_RULE_TYPES:
            raise ValueError(f"Unknown safety rule {name!r}; expected one of {sorted(VALID_RULE_TYPES)}")

        kwargs: Dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError(f"Starred kwargs not allowed in safety rule: {spec!r}")
            kwargs[kw.arg] = _eval_literal(kw.value)

        alpha = float(kwargs.pop("alpha", 0.5))
        hard = bool(kwargs.pop("hard", True))
        entity = kwargs.pop("entity", None)

        if name == "always":
            if len(node.args) != 1:
                raise ValueError(f"always(...) expects 1 predicate argument, got {spec!r}")
            arg = node.args[0]
            if isinstance(arg, ast.Call):
                inner = self.parse_rule_string(ast.unparse(arg))
                return PhysicalSafetyRule(
                    rule_type="always",
                    entity=inner.entity,
                    params={"inner_rule": inner.to_dict()},
                    alpha=alpha if "alpha" in kwargs else inner.alpha,
                    hard=hard,
                    raw_spec=text,
                )
            if isinstance(arg, ast.Compare) and len(arg.ops) == 1 and len(arg.comparators) == 1:
                if not isinstance(arg.left, ast.Name):
                    raise ValueError(f"always(...) comparison LHS must be an identifier: {spec!r}")
                metric = arg.left.id
                rhs = float(_eval_literal(arg.comparators[0]))
                op_node = arg.ops[0]
                op_str = (
                    "<=" if isinstance(op_node, ast.LtE)
                    else "<" if isinstance(op_node, ast.Lt)
                    else ">=" if isinstance(op_node, ast.GtE)
                    else ">" if isinstance(op_node, ast.Gt)
                    else None
                )
                if op_str is None:
                    raise ValueError(f"Unsupported comparison operator in {spec!r}")
                return PhysicalSafetyRule(
                    rule_type="always",
                    entity=entity,
                    params={"metric": metric, "op": op_str, "value": rhs, **kwargs},
                    alpha=alpha,
                    hard=hard,
                    raw_spec=text,
                )
            raise ValueError(f"Unsupported always(...) predicate in {spec!r}")

        if node.args:
            # Allow simple positional convenience for single-arg rules
            if name == "avoid" and len(node.args) == 1 and entity is None:
                entity = str(_eval_literal(node.args[0]))
            elif name == "keep_upright" and len(node.args) == 1 and "max_tilt_deg" not in kwargs:
                kwargs["max_tilt_deg"] = float(_eval_literal(node.args[0]))
            elif name in ("speed_limit", "force_limit") and len(node.args) == 1:
                kwargs["limit"] = float(_eval_literal(node.args[0]))
            else:
                raise ValueError(f"Unexpected positional arguments in {spec!r}")

        rule = PhysicalSafetyRule(
            rule_type=name,
            entity=entity,
            params=kwargs,
            alpha=alpha,
            hard=hard,
            raw_spec=text,
        )
        rule.validate()
        return rule

    def build_ir(
        self,
        rules: Sequence[Union[str, PhysicalSafetyRule]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhysicalSafetyIR:
        parsed: List[PhysicalSafetyRule] = []
        for item in rules:
            if isinstance(item, PhysicalSafetyRule):
                parsed.append(item)
            elif isinstance(item, str):
                parsed.append(self.parse_rule_string(item))
            else:
                raise TypeError(f"Expected PhysicalSafetyRule or str, got {type(item).__name__}")
        return PhysicalSafetyIR(rules=parsed, metadata=metadata)

    def _compile_single(
        self,
        rule: PhysicalSafetyRule,
        geom: Optional[Dict[str, Any]],
        pos_slice: Optional[slice],
        vel_slice: Optional[slice],
        force_slice: Optional[slice],
        tilt_slice: Optional[slice],
        state_dim: Optional[int],
    ) -> List[BarrierConstraint]:
        rt = rule.rule_type
        params = rule.params

        if rt == "avoid":
            center = params.get("center") if "center" in params else (geom or {}).get("center")
            if center is None:
                raise ValueError(f"Entity {rule.entity!r} has no 'center' coordinate")
            ent_radius = float((geom or {}).get("radius", 0.0))
            clearance = float(params.get("min_dist_m", params.get("radius", 0.0)))
            total_r = ent_radius + clearance
            if total_r <= 0.0:
                raise ValueError(f"avoid barrier radius must be > 0, got {total_r}")
            velocity = params.get("velocity") if "velocity" in params else (geom or {}).get("velocity")
            acceleration = (
                params.get("acceleration") if "acceleration" in params else (geom or {}).get("acceleration")
            )
            if velocity is not None or acceleration is not None:
                return [
                    MovingObstacleBarrier(
                        center=center,
                        radius=total_r,
                        velocity=velocity,
                        acceleration=acceleration,
                        alpha=rule.alpha,
                        hard=rule.hard,
                        state_slice=pos_slice,
                        state_dim=state_dim,
                        entity=rule.entity,
                    )
                ]
            return [
                SubspaceSphereBarrier(
                    center=center,
                    radius=total_r,
                    alpha=rule.alpha,
                    state_slice=pos_slice,
                    state_dim=state_dim,
                    entity=rule.entity,
                )
            ]

        if rt == "keep_in_box":
            low = params.get("min") or (geom or {}).get("min")
            high = params.get("max") or (geom or {}).get("max")
            if (low is None or high is None) and "bounds" in params:
                bounds = params["bounds"]
                low = [b[0] for b in bounds]
                high = [b[1] for b in bounds]
            if (low is None or high is None) and geom and "bounds" in geom:
                bounds = geom["bounds"]
                low = [b[0] for b in bounds]
                high = [b[1] for b in bounds]
            if (low is None or high is None) and geom and "center" in geom and "half_extents" in geom:
                c = np.asarray(geom["center"], dtype=np.float64)
                he = np.asarray(geom["half_extents"], dtype=np.float64)
                low = (c - he).tolist()
                high = (c + he).tolist()
            if low is None or high is None:
                raise ValueError("keep_in_box requires 'min'/'max' or 'bounds'")
            low_arr = np.asarray(low, dtype=np.float64).flatten()
            high_arr = np.asarray(high, dtype=np.float64).flatten()
            if low_arr.shape != high_arr.shape or np.any(low_arr > high_arr):
                raise ValueError("keep_in_box bounds invalid: min must be <= max with matching dims")
            dim = low_arr.shape[0]
            barriers: List[BarrierConstraint] = []
            for i in range(dim):
                e_pos = np.zeros(dim, dtype=np.float64)
                e_pos[i] = 1.0
                barriers.append(
                    SubspaceHalfPlaneBarrier(
                        a=e_pos,
                        b=float(low_arr[i]),
                        alpha=rule.alpha,
                        state_slice=pos_slice,
                        state_dim=state_dim,
                        entity=rule.entity,
                    )
                )
                barriers.append(
                    SubspaceHalfPlaneBarrier(
                        a=-e_pos,
                        b=-float(high_arr[i]),
                        alpha=rule.alpha,
                        state_slice=pos_slice,
                        state_dim=state_dim,
                        entity=rule.entity,
                    )
                )
            return barriers

        if rt == "keep_upright":
            max_tilt_deg = float(
                params.get("max_tilt_deg", math.degrees(float(params["max_tilt_rad"])) if "max_tilt_rad" in params else 15.0)
            )
            if _ExternalTiltConeBarrier is not None and callable(params.get("normal_fn")):
                return [
                    _ExternalTiltConeBarrier(
                        normal_fn=params["normal_fn"],
                        max_tilt_rad=math.radians(max_tilt_deg),
                        alpha=rule.alpha,
                        hard=rule.hard,
                    )
                ]
            return [
                TiltConeBarrier(
                    max_tilt_deg=max_tilt_deg,
                    alpha=rule.alpha,
                    tilt_slice=tilt_slice,
                    state_dim=state_dim,
                )
            ]

        if rt == "speed_limit":
            limit = float(params.get("max_speed_mps", params.get("limit", params.get("value", 0.0))))
            if limit < 0.0:
                raise ValueError(f"speed_limit must be >= 0, got {limit}")
            return [
                NormBoundBarrier(
                    bound=limit,
                    alpha=rule.alpha,
                    state_slice=vel_slice,
                    state_dim=state_dim,
                    label="speed_limit",
                )
            ]

        if rt == "force_limit":
            limit = float(params.get("max_force_n", params.get("limit", params.get("value", 0.0))))
            if limit < 0.0:
                raise ValueError(f"force_limit must be >= 0, got {limit}")
            return [
                NormBoundBarrier(
                    bound=limit,
                    alpha=rule.alpha,
                    state_slice=force_slice,
                    state_dim=state_dim,
                    label="force_limit",
                )
            ]

        if rt == "always":
            if "inner_rule" in params:
                inner = PhysicalSafetyRule.from_dict(params["inner_rule"])
                return self._compile_single(
                    inner, geom, pos_slice, vel_slice, force_slice, tilt_slice, state_dim
                )
            metric = str(params.get("metric", "")).lower()
            op = params.get("op", "<=")
            val = float(params["value"])
            if metric in ("speed_mps", "speed", "vel_mps", "velocity") and op in ("<=", "<"):
                return [
                    NormBoundBarrier(
                        bound=val,
                        alpha=rule.alpha,
                        state_slice=vel_slice,
                        state_dim=state_dim,
                        label="speed_limit",
                    )
                ]
            if metric in ("force_n", "force", "contact_force_n") and op in ("<=", "<"):
                return [
                    NormBoundBarrier(
                        bound=val,
                        alpha=rule.alpha,
                        state_slice=force_slice,
                        state_dim=state_dim,
                        label="force_limit",
                    )
                ]
            if metric in ("tilt_deg", "max_tilt_deg") and op in ("<=", "<"):
                return [
                    TiltConeBarrier(
                        max_tilt_deg=val,
                        alpha=rule.alpha,
                        tilt_slice=tilt_slice,
                        state_dim=state_dim,
                    )
                ]
            raise ValueError(f"Unsupported always predicate metric={metric!r}, op={op!r}")

        if rt == "eventually":
            target = params.get("target") if "target" in params else (geom or {}).get("center")
            if target is None:
                raise ValueError("eventually rule requires 'target' or resolvable entity with 'center'")
            within_s = float(params.get("within_s", params.get("horizon_s", 5.0)))
            radius = float(params.get("radius", (geom or {}).get("radius", 0.2)))
            init_r = params.get("initial_radius")
            return [
                TimeVaryingBarrier(
                    target=target,
                    target_radius=radius,
                    horizon_s=within_s,
                    initial_radius=None if init_r is None else float(init_r),
                    alpha=rule.alpha,
                    state_slice=pos_slice,
                    state_dim=state_dim,
                )
            ]

        raise ValueError(f"Unhandled rule_type: {rt!r}")

    def compile_with_receipt(
        self,
        rules: Union[PhysicalSafetyIR, Sequence[Union[str, PhysicalSafetyRule]]],
        scene_graph: Optional[Dict[str, Any]] = None,
        pos_slice: Optional[slice] = slice(0, 3),
        vel_slice: Optional[slice] = None,
        force_slice: Optional[slice] = None,
        tilt_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compile rules into CBF barriers and return a structured audit receipt."""
        ir = rules if isinstance(rules, PhysicalSafetyIR) else self.build_ir(rules)
        bound, unresolved = ir.bind(scene_graph, fail_on_unresolved=False)
        if unresolved:
            return {
                "status": "unresolved_entities",
                "verified_compiled": False,
                "barriers": [],
                "unresolved": unresolved,
                "n_barriers": 0,
                "ir": ir.to_dict(),
            }

        barriers: List[BarrierConstraint] = []
        for entry in bound:
            barriers.extend(
                self._compile_single(
                    rule=entry["rule"],
                    geom=entry["geometry"],
                    pos_slice=pos_slice,
                    vel_slice=vel_slice,
                    force_slice=force_slice,
                    tilt_slice=tilt_slice,
                    state_dim=state_dim,
                )
            )

        return {
            "status": "compiled",
            "verified_compiled": True,
            "barriers": barriers,
            "unresolved": [],
            "n_barriers": len(barriers),
            "ir": ir.to_dict(),
        }

    def compile_rules(
        self,
        rules: Union[PhysicalSafetyIR, Sequence[Union[str, PhysicalSafetyRule]]],
        scene_graph: Optional[Dict[str, Any]] = None,
        pos_slice: Optional[slice] = slice(0, 3),
        vel_slice: Optional[slice] = None,
        force_slice: Optional[slice] = None,
        tilt_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
    ) -> List[BarrierConstraint]:
        """Compile rules into ``BarrierConstraint`` objects; fails closed on unresolved entities."""
        receipt = self.compile_with_receipt(
            rules=rules,
            scene_graph=scene_graph,
            pos_slice=pos_slice,
            vel_slice=vel_slice,
            force_slice=force_slice,
            tilt_slice=tilt_slice,
            state_dim=state_dim,
        )
        if not receipt["verified_compiled"]:
            raise ValueError(f"Unresolved safety entities in scene_graph: {receipt['unresolved']}")
        return list(receipt["barriers"])

    def _compile_leaf_to_single_barrier(
        self,
        rule_or_spec: Union[str, PhysicalSafetyRule],
        scene_graph: Optional[Dict[str, Any]],
        pos_slice: Optional[slice],
        vel_slice: Optional[slice],
        force_slice: Optional[slice],
        tilt_slice: Optional[slice],
        state_dim: Optional[int],
        eta: float,
    ) -> BarrierConstraint:
        rule = self.parse_rule_string(rule_or_spec) if isinstance(rule_or_spec, str) else rule_or_spec
        bars = self.compile_rules(
            [rule],
            scene_graph=scene_graph,
            pos_slice=pos_slice,
            vel_slice=vel_slice,
            force_slice=force_slice,
            tilt_slice=tilt_slice,
            state_dim=state_dim,
        )
        if len(bars) == 1:
            return bars[0]
        return STLAndBarrier(bars, eta=eta, alpha=rule.alpha, hard=rule.hard)

    def _compile_stl_ast_node(
        self,
        node: ast.AST,
        scene_graph: Optional[Dict[str, Any]],
        pos_slice: Optional[slice],
        vel_slice: Optional[slice],
        force_slice: Optional[slice],
        tilt_slice: Optional[slice],
        state_dim: Optional[int],
        default_eta: float,
    ) -> BarrierConstraint:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"STL tree node must be a function call, got {ast.unparse(node)!r}")
        op_raw = node.func.id
        op_upper = op_raw.upper().rstrip("_")

        if op_upper in ("AND", "OR", "UNTIL"):
            kw_map: Dict[str, Any] = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError(f"Starred kwargs not allowed in STL expression: {ast.unparse(node)!r}")
                kw_map[kw.arg] = kw.value

            if op_upper in ("AND", "OR"):
                if not node.args:
                    raise ValueError(f"{op_raw}(...) requires at least 1 child argument")
                eta_val = float(_eval_literal(kw_map.pop("eta"))) if "eta" in kw_map else default_eta
                children = [
                    self._compile_stl_ast_node(
                        arg,
                        scene_graph=scene_graph,
                        pos_slice=pos_slice,
                        vel_slice=vel_slice,
                        force_slice=force_slice,
                        tilt_slice=tilt_slice,
                        state_dim=state_dim,
                        default_eta=eta_val,
                    )
                    for arg in node.args
                ]
                alpha_val = (
                    float(_eval_literal(kw_map.pop("alpha")))
                    if "alpha" in kw_map
                    else float(getattr(children[0], "alpha", 0.5))
                )
                hard_val = bool(_eval_literal(kw_map.pop("hard"))) if "hard" in kw_map else True
                if op_upper == "AND":
                    if kw_map:
                        raise ValueError(f"Unexpected kwargs {sorted(kw_map)} in {op_raw}(...)")
                    return STLAndBarrier(children, eta=eta_val, alpha=alpha_val, hard=hard_val)
                sound_val = bool(_eval_literal(kw_map.pop("sound_inner"))) if "sound_inner" in kw_map else True
                if kw_map:
                    raise ValueError(f"Unexpected kwargs {sorted(kw_map)} in {op_raw}(...)")
                return STLOrBarrier(
                    children,
                    eta=eta_val,
                    alpha=alpha_val,
                    hard=hard_val,
                    sound_inner=sound_val,
                )

            # UNTIL(hold_expr, goal_expr, t_start=..., t_switch=..., t_end=...)
            hold_ast = kw_map.pop("hold", None)
            goal_ast = kw_map.pop("goal", None)
            if len(node.args) == 2 and hold_ast is None and goal_ast is None:
                hold_ast, goal_ast = node.args[0], node.args[1]
            elif node.args or hold_ast is None or goal_ast is None:
                raise ValueError("UNTIL(...) expects (hold_expr, goal_expr, ...) or hold=..., goal=...")

            eta_val = float(_eval_literal(kw_map.pop("eta"))) if "eta" in kw_map else default_eta
            hold_bar = self._compile_stl_ast_node(
                hold_ast,
                scene_graph=scene_graph,
                pos_slice=pos_slice,
                vel_slice=vel_slice,
                force_slice=force_slice,
                tilt_slice=tilt_slice,
                state_dim=state_dim,
                default_eta=eta_val,
            )
            goal_bar = self._compile_stl_ast_node(
                goal_ast,
                scene_graph=scene_graph,
                pos_slice=pos_slice,
                vel_slice=vel_slice,
                force_slice=force_slice,
                tilt_slice=tilt_slice,
                state_dim=state_dim,
                default_eta=eta_val,
            )
            t_start = float(_eval_literal(kw_map.pop("t_start"))) if "t_start" in kw_map else 0.0
            t_end_node = kw_map.pop("t_end", kw_map.pop("within_s", kw_map.pop("horizon_s", None)))
            t_end = (
                float(_eval_literal(t_end_node))
                if t_end_node is not None
                else float(getattr(goal_bar, "horizon_s", 5.0))
            )
            sw_node = kw_map.pop("t_switch", kw_map.pop("switch_time", None))
            t_switch = None if sw_node is None else float(_eval_literal(sw_node))
            blend_node = kw_map.pop("blend_duration", None)
            blend_dur = None if blend_node is None else float(_eval_literal(blend_node))
            alpha_val = (
                float(_eval_literal(kw_map.pop("alpha")))
                if "alpha" in kw_map
                else float(getattr(goal_bar, "alpha", getattr(hold_bar, "alpha", 0.5)))
            )
            hard_val = bool(_eval_literal(kw_map.pop("hard"))) if "hard" in kw_map else True
            if kw_map:
                raise ValueError(f"Unexpected kwargs {sorted(kw_map)} in UNTIL(...)")
            return STLUntilBarrier(
                hold_barrier=hold_bar,
                goal_barrier=goal_bar,
                t_start=t_start,
                t_switch=t_switch,
                t_end=t_end,
                blend_duration=blend_dur,
                alpha=alpha_val,
                hard=hard_val,
            )

        return self._compile_leaf_to_single_barrier(
            ast.unparse(node),
            scene_graph=scene_graph,
            pos_slice=pos_slice,
            vel_slice=vel_slice,
            force_slice=force_slice,
            tilt_slice=tilt_slice,
            state_dim=state_dim,
            eta=default_eta,
        )

    def compile_stl_tree(
        self,
        expr_or_dict: Union[str, Dict[str, Any], PhysicalSafetyRule, BarrierConstraint],
        scene_graph: Optional[Dict[str, Any]] = None,
        pos_slice: Optional[slice] = slice(0, 3),
        vel_slice: Optional[slice] = None,
        force_slice: Optional[slice] = None,
        tilt_slice: Optional[slice] = None,
        state_dim: Optional[int] = None,
        eta: float = 25.0,
    ) -> BarrierConstraint:
        """Compile a nested compositional STL tree (``AND``, ``OR``, ``UNTIL``) into a smooth CBF."""
        if isinstance(expr_or_dict, BarrierConstraint):
            return expr_or_dict
        if isinstance(expr_or_dict, PhysicalSafetyRule):
            return self._compile_leaf_to_single_barrier(
                expr_or_dict,
                scene_graph=scene_graph,
                pos_slice=pos_slice,
                vel_slice=vel_slice,
                force_slice=force_slice,
                tilt_slice=tilt_slice,
                state_dim=state_dim,
                eta=eta,
            )
        if isinstance(expr_or_dict, str):
            text = expr_or_dict.strip()
            if not text:
                raise ValueError("Empty STL tree expression")
            try:
                tree = ast.parse(text, mode="eval")
            except SyntaxError as exc:
                raise ValueError(f"Invalid STL expression syntax {expr_or_dict!r}: {exc}") from exc
            return self._compile_stl_ast_node(
                tree.body,
                scene_graph=scene_graph,
                pos_slice=pos_slice,
                vel_slice=vel_slice,
                force_slice=force_slice,
                tilt_slice=tilt_slice,
                state_dim=state_dim,
                default_eta=float(eta),
            )
        if isinstance(expr_or_dict, dict):
            if "rule_type" in expr_or_dict and "op" not in expr_or_dict and "operator" not in expr_or_dict:
                rule = PhysicalSafetyRule.from_dict(expr_or_dict)
                return self._compile_leaf_to_single_barrier(
                    rule,
                    scene_graph=scene_graph,
                    pos_slice=pos_slice,
                    vel_slice=vel_slice,
                    force_slice=force_slice,
                    tilt_slice=tilt_slice,
                    state_dim=state_dim,
                    eta=eta,
                )
            raw_op = expr_or_dict.get("op", expr_or_dict.get("operator", expr_or_dict.get("type")))
            if raw_op is None:
                raise ValueError(f"STL dict requires 'op' or 'rule_type', got keys {sorted(expr_or_dict)}")
            op_upper = str(raw_op).strip().upper().rstrip("_")
            eta_val = float(expr_or_dict.get("eta", eta))
            if op_upper in ("AND", "OR"):
                raw_children = expr_or_dict.get("args", expr_or_dict.get("children", expr_or_dict.get("rules")))
                if not raw_children:
                    raise ValueError(f"STL {op_upper} dict requires non-empty 'args' or 'children'")
                children = [
                    self.compile_stl_tree(
                        ch,
                        scene_graph=scene_graph,
                        pos_slice=pos_slice,
                        vel_slice=vel_slice,
                        force_slice=force_slice,
                        tilt_slice=tilt_slice,
                        state_dim=state_dim,
                        eta=eta_val,
                    )
                    for ch in raw_children
                ]
                alpha_val = float(expr_or_dict.get("alpha", getattr(children[0], "alpha", 0.5)))
                hard_val = bool(expr_or_dict.get("hard", True))
                if op_upper == "AND":
                    return STLAndBarrier(children, eta=eta_val, alpha=alpha_val, hard=hard_val)
                return STLOrBarrier(
                    children,
                    eta=eta_val,
                    alpha=alpha_val,
                    hard=hard_val,
                    sound_inner=bool(expr_or_dict.get("sound_inner", True)),
                )
            if op_upper == "UNTIL":
                if "hold" in expr_or_dict and "goal" in expr_or_dict:
                    hold_spec, goal_spec = expr_or_dict["hold"], expr_or_dict["goal"]
                else:
                    raw_args = expr_or_dict.get("args", expr_or_dict.get("children"))
                    if not isinstance(raw_args, (list, tuple)) or len(raw_args) != 2:
                        raise ValueError("STL UNTIL dict requires ('hold', 'goal') or 2-element 'args'")
                    hold_spec, goal_spec = raw_args[0], raw_args[1]
                hold_bar = self.compile_stl_tree(
                    hold_spec,
                    scene_graph=scene_graph,
                    pos_slice=pos_slice,
                    vel_slice=vel_slice,
                    force_slice=force_slice,
                    tilt_slice=tilt_slice,
                    state_dim=state_dim,
                    eta=eta_val,
                )
                goal_bar = self.compile_stl_tree(
                    goal_spec,
                    scene_graph=scene_graph,
                    pos_slice=pos_slice,
                    vel_slice=vel_slice,
                    force_slice=force_slice,
                    tilt_slice=tilt_slice,
                    state_dim=state_dim,
                    eta=eta_val,
                )
                t_start = float(expr_or_dict.get("t_start", 0.0))
                t_end = float(
                    expr_or_dict.get(
                        "t_end",
                        expr_or_dict.get("within_s", getattr(goal_bar, "horizon_s", 5.0)),
                    )
                )
                t_sw = expr_or_dict.get("t_switch", expr_or_dict.get("switch_time"))
                blend_dur = expr_or_dict.get("blend_duration")
                alpha_val = float(
                    expr_or_dict.get("alpha", getattr(goal_bar, "alpha", getattr(hold_bar, "alpha", 0.5)))
                )
                return STLUntilBarrier(
                    hold_barrier=hold_bar,
                    goal_barrier=goal_bar,
                    t_start=t_start,
                    t_switch=None if t_sw is None else float(t_sw),
                    t_end=t_end,
                    blend_duration=None if blend_dur is None else float(blend_dur),
                    alpha=alpha_val,
                    hard=bool(expr_or_dict.get("hard", True)),
                )
            raise ValueError(f"Unsupported STL tree operator {raw_op!r}; expected AND, OR, or UNTIL")
        raise TypeError(f"Unsupported expr_or_dict type: {type(expr_or_dict).__name__}")


__all__ = [
    "KalmanObstacleTracker",
    "MovingObstacleBarrier",
    "NormBoundBarrier",
    "PhysicalSafetyIR",
    "PhysicalSafetyRule",
    "STLAndBarrier",
    "STLOrBarrier",
    "STLUntilBarrier",
    "SemanticCBFCompiler",
    "SubspaceHalfPlaneBarrier",
    "SubspaceSphereBarrier",
    "TiltConeBarrier",
    "TimeVaryingBarrier",
    "VALID_RULE_TYPES",
]
