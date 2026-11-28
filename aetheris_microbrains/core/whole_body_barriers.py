"""Whole-body robotics Control Barrier Functions and Simplex backup controller.

Provides deterministic NumPy barrier constraints for physical manipulation and
human-robot interaction:

- ``CapsulePairBarrier``: 3D segment-to-segment capsule clearance barrier.
- ``ManipulabilityBarrier``: Yoshikawa manipulability singularity avoidance.
- ``TiltConeBarrier``: End-effector orientation cone constraint.
- ``KineticEnergyBarrier``: ISO/TS 15066 distance-dependent reflected kinetic energy bound.
- ``WrenchConeBarrier``: Coulomb friction cone + normal contact force limits.
- ``SimplexBackupController``: Passive gravity-compensated damping + braking fallback.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .state_cbf_filter import BarrierConstraint


def _segment_aabb_distance_sq(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> float:
    """Squared Euclidean distance between the axis-aligned bounding boxes of two 3D segments."""
    p_min = np.minimum(p0, p1)
    p_max = np.maximum(p0, p1)
    q_min = np.minimum(q0, q1)
    q_max = np.maximum(q0, q1)
    gap = np.maximum(0.0, np.maximum(p_min - q_max, q_min - p_max))
    return float(np.dot(gap, gap))


def _segment_segment_distance_and_witnesses(
    p0: Sequence[float],
    p1: Sequence[float],
    q0: Sequence[float],
    q1: Sequence[float],
    eps: float = 1e-15,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Exact squared distance and closest-point witnesses between 3D segments [p0, p1] and [q0, q1].

    Returns ``(d2, s_star, t_star, p_star, q_star)`` where
    ``p_star = p0 + s_star * (p1 - p0)`` and ``q_star = q0 + t_star * (q1 - q0)``.
    """
    p0_v = np.asarray(p0, dtype=np.float64).flatten()
    p1_v = np.asarray(p1, dtype=np.float64).flatten()
    q0_v = np.asarray(q0, dtype=np.float64).flatten()
    q1_v = np.asarray(q1, dtype=np.float64).flatten()

    d1 = p1_v - p0_v
    d2_vec = q1_v - q0_v
    r = p0_v - q0_v
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2_vec, d2_vec))
    f = float(np.dot(d2_vec, r))

    if a <= eps and e <= eps:
        return float(np.dot(r, r)), 0.0, 0.0, p0_v.copy(), q0_v.copy()
    if a <= eps:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2_vec))
            denom = a * e - b * b
            if denom > eps:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    p_star = p0_v + s * d1
    q_star = q0_v + t * d2_vec
    diff = p_star - q_star
    return float(np.dot(diff, diff)), float(s), float(t), p_star, q_star


def _segment_segment_distance_sq(
    p0: Sequence[float],
    p1: Sequence[float],
    q0: Sequence[float],
    q1: Sequence[float],
    eps: float = 1e-15,
) -> float:
    """Exact squared Euclidean distance between 3D line segments [p0, p1] and [q0, q1]."""
    d2, _, _, _, _ = _segment_segment_distance_and_witnesses(p0, p1, q0, q1, eps=eps)
    return d2


def _fd_grad_subspace(
    fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    active_dim: Optional[int],
    eps: float,
) -> np.ndarray:
    """Central finite-difference gradient over the first ``active_dim`` coordinates."""
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    n = x_arr.shape[0]
    k = n if active_dim is None else min(n, int(active_dim))
    grad = np.zeros(n, dtype=np.float64)
    for i in range(k):
        ei = np.zeros(n, dtype=np.float64)
        ei[i] = eps
        grad[i] = (float(fn(x_arr + ei)) - float(fn(x_arr - ei))) / (2.0 * eps)
    return grad


class CapsulePairBarrier(BarrierConstraint):
    """3D capsule-to-capsule clearance barrier: h(q) = d(S1(q), S2)^2 - r_safe^2 >= 0."""

    def __init__(
        self,
        link_fk_fn: Callable[[np.ndarray], Tuple[Sequence[float], Sequence[float]]],
        opponent_segment: Union[
            Tuple[Sequence[float], Sequence[float]],
            Callable[[np.ndarray], Tuple[Sequence[float], Sequence[float]]],
        ],
        r_safe: float = 0.1,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        n_dof: Optional[int] = None,
        eps: float = 1e-5,
        link_jacobians_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        opponent_jacobians_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        broadphase_cutoff: Optional[float] = None,
    ):
        self.link_fk_fn = link_fk_fn
        self.opponent_segment = opponent_segment
        self.r_safe = float(r_safe)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.n_dof = None if n_dof is None else int(n_dof)
        self.eps = float(eps)
        self.link_jacobians_fn = link_jacobians_fn
        self.opponent_jacobians_fn = opponent_jacobians_fn
        self.broadphase_cutoff = None if broadphase_cutoff is None else float(broadphase_cutoff)

    def _q(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        return x_arr[: self.n_dof] if self.n_dof is not None else x_arr

    def _segments_at_q(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p0, p1 = self.link_fk_fn(q)
        q0, q1 = self.opponent_segment(q) if callable(self.opponent_segment) else self.opponent_segment
        return (
            np.asarray(p0, dtype=np.float64).flatten(),
            np.asarray(p1, dtype=np.float64).flatten(),
            np.asarray(q0, dtype=np.float64).flatten(),
            np.asarray(q1, dtype=np.float64).flatten(),
        )

    def _broadphase_culled_segments(
        self, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
    ) -> bool:
        if self.broadphase_cutoff is None:
            return False
        cutoff = max(self.broadphase_cutoff, self.r_safe + 1e-6)
        return _segment_aabb_distance_sq(p0, p1, q0, q1) > cutoff * cutoff

    def is_broadphase_culled(self, x: np.ndarray) -> bool:
        """Return True if segment AABBs exceed ``broadphase_cutoff`` (and ``r_safe``)."""
        if self.broadphase_cutoff is None:
            return False
        q = self._q(x)
        p0, p1, q0, q1 = self._segments_at_q(q)
        return self._broadphase_culled_segments(p0, p1, q0, q1)

    def h(self, x: np.ndarray) -> float:
        q = self._q(x)
        p0, p1, q0, q1 = self._segments_at_q(q)
        d2, _, _, _, _ = _segment_segment_distance_and_witnesses(p0, p1, q0, q1)
        return float(d2 - self.r_safe * self.r_safe)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        q = x_arr[: self.n_dof] if self.n_dof is not None else x_arr
        q_len = q.shape[0]

        if self.link_jacobians_fn is not None and (
            self.opponent_jacobians_fn is not None or not callable(self.opponent_segment)
        ):
            p0, p1, q0, q1 = self._segments_at_q(q)
            if self._broadphase_culled_segments(p0, p1, q0, q1):
                return np.zeros(n, dtype=np.float64)
            _, s_star, t_star, p_star, q_star = _segment_segment_distance_and_witnesses(p0, p1, q0, q1)
            J_p0_raw, J_p1_raw = self.link_jacobians_fn(q)
            J_p0 = np.asarray(J_p0_raw, dtype=np.float64).reshape(3, q_len)
            J_p1 = np.asarray(J_p1_raw, dtype=np.float64).reshape(3, q_len)
            J_diff = (1.0 - s_star) * J_p0 + s_star * J_p1
            if self.opponent_jacobians_fn is not None:
                J_q0_raw, J_q1_raw = self.opponent_jacobians_fn(q)
                J_q0 = np.asarray(J_q0_raw, dtype=np.float64).reshape(3, q_len)
                J_q1 = np.asarray(J_q1_raw, dtype=np.float64).reshape(3, q_len)
                J_diff = J_diff - ((1.0 - t_star) * J_q0 + t_star * J_q1)
            grad = np.zeros(n, dtype=np.float64)
            grad[:q_len] = 2.0 * ((p_star - q_star) @ J_diff)
            return grad

        if self.broadphase_cutoff is not None:
            p0, p1, q0, q1 = self._segments_at_q(q)
            if self._broadphase_culled_segments(p0, p1, q0, q1):
                return np.zeros(n, dtype=np.float64)

        return _fd_grad_subspace(self.h, x_arr, self.n_dof, self.eps)

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Configuration-subspace Hessian nabla^2 h(x), zero-padded on velocity coordinates."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        q = x_arr[: self.n_dof] if self.n_dof is not None else x_arr
        q_len = q.shape[0]
        if self.broadphase_cutoff is not None:
            p0, p1, q0, q1 = self._segments_at_q(q)
            if self._broadphase_culled_segments(p0, p1, q0, q1):
                return np.zeros((n, n), dtype=np.float64)
        H_q = np.zeros((q_len, q_len), dtype=np.float64)
        for i in range(q_len):
            dq = np.zeros(q_len, dtype=np.float64)
            dq[i] = eps
            gp = self.grad_h(q + dq)[:q_len]
            gm = self.grad_h(q - dq)[:q_len]
            H_q[:, i] = (gp - gm) / (2.0 * eps)
        H_q = 0.5 * (H_q + H_q.T)
        if q_len == n:
            return H_q
        out = np.zeros((n, n), dtype=np.float64)
        out[:q_len, :q_len] = H_q
        return out

    def inflate(self, delta: float) -> "CapsulePairBarrier":
        return CapsulePairBarrier(
            link_fk_fn=self.link_fk_fn,
            opponent_segment=self.opponent_segment,
            r_safe=self.r_safe + float(delta),
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            n_dof=self.n_dof,
            eps=self.eps,
            link_jacobians_fn=self.link_jacobians_fn,
            opponent_jacobians_fn=self.opponent_jacobians_fn,
            broadphase_cutoff=self.broadphase_cutoff,
        )


class ManipulabilityBarrier(BarrierConstraint):
    """Yoshikawa manipulability barrier: h(q) = det(J(q) J(q)^T) - eps_sing >= 0."""

    def __init__(
        self,
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        eps_sing: float = 1e-3,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        n_dof: Optional[int] = None,
        eps: float = 1e-5,
    ):
        self.jacobian_fn = jacobian_fn
        self.eps_sing = float(eps_sing)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.n_dof = None if n_dof is None else int(n_dof)
        self.eps = float(eps)

    def _q(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        return x_arr[: self.n_dof] if self.n_dof is not None else x_arr

    def h(self, x: np.ndarray) -> float:
        q = self._q(x)
        J = np.asarray(self.jacobian_fn(q), dtype=np.float64)
        det_val = float(np.linalg.det(J @ J.T))
        return det_val - self.eps_sing

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return _fd_grad_subspace(self.h, np.asarray(x, dtype=np.float64).flatten(), self.n_dof, self.eps)

    def inflate(self, delta: float) -> "ManipulabilityBarrier":
        return ManipulabilityBarrier(
            jacobian_fn=self.jacobian_fn,
            eps_sing=self.eps_sing + float(delta),
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            n_dof=self.n_dof,
            eps=self.eps,
        )


class TiltConeBarrier(BarrierConstraint):
    """End-effector tilt cone: h(q) = n_ee(q) . z_world - cos(theta_max) >= 0."""

    def __init__(
        self,
        normal_fn: Callable[[np.ndarray], Sequence[float]],
        max_tilt_rad: float,
        world_up: Sequence[float] = (0.0, 0.0, 1.0),
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        n_dof: Optional[int] = None,
        eps: float = 1e-5,
        margin_offset: float = 0.0,
    ):
        self.normal_fn = normal_fn
        self.max_tilt_rad = float(max_tilt_rad)
        up = np.asarray(world_up, dtype=np.float64).flatten()
        up_norm = float(np.linalg.norm(up))
        self.world_up = up / up_norm if up_norm > 1e-12 else up
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.n_dof = None if n_dof is None else int(n_dof)
        self.eps = float(eps)
        self.margin_offset = float(margin_offset)

    def _q(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        return x_arr[: self.n_dof] if self.n_dof is not None else x_arr

    def h(self, x: np.ndarray) -> float:
        q = self._q(x)
        n_ee = np.asarray(self.normal_fn(q), dtype=np.float64).flatten()
        n_norm = float(np.linalg.norm(n_ee))
        if n_norm > 1e-12:
            n_ee = n_ee / n_norm
        return float(np.dot(n_ee, self.world_up) - np.cos(self.max_tilt_rad) - self.margin_offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return _fd_grad_subspace(self.h, np.asarray(x, dtype=np.float64).flatten(), self.n_dof, self.eps)

    def inflate(self, delta: float) -> "TiltConeBarrier":
        return TiltConeBarrier(
            normal_fn=self.normal_fn,
            max_tilt_rad=self.max_tilt_rad,
            world_up=self.world_up,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            n_dof=self.n_dof,
            eps=self.eps,
            margin_offset=self.margin_offset + float(delta),
        )


class KineticEnergyBarrier(BarrierConstraint):
    """ISO/TS 15066 distance-dependent reflected kinetic energy barrier.

    Models ``h(q, qdot) = E_max(d(q)) - 0.5 * qdot^T M(q) qdot >= 0`` where
    ``E_max(d)`` tightens smoothly from ``e_max_far`` down to ``e_max_near``
    as human-robot separation ``d`` decreases.
    """

    def __init__(
        self,
        n_dof: Optional[int] = None,
        M_fn: Optional[Union[Callable[[np.ndarray], np.ndarray], np.ndarray]] = None,
        distance_fn: Optional[Union[Callable[[np.ndarray], float], float]] = None,
        e_max_far: float = 10.0,
        e_max_near: float = 0.5,
        d_safe: float = 1.0,
        e_max_fn: Optional[Callable[[float], float]] = None,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        eps: float = 1e-5,
        margin_offset: float = 0.0,
    ):
        self.n_dof = None if n_dof is None else int(n_dof)
        self.M_fn = M_fn
        self.distance_fn = distance_fn
        self.e_max_far = float(e_max_far)
        self.e_max_near = float(e_max_near)
        self.d_safe = max(1e-6, float(d_safe))
        self.e_max_fn = e_max_fn
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.eps = float(eps)
        self.margin_offset = float(margin_offset)

    def _split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = self.n_dof if self.n_dof is not None else x_arr.shape[0] // 2
        return x_arr[:n], x_arr[n : 2 * n]

    def _eval_M(self, q: np.ndarray) -> np.ndarray:
        n = q.shape[0]
        if self.M_fn is None:
            return np.eye(n, dtype=np.float64)
        val = self.M_fn(q) if callable(self.M_fn) else self.M_fn
        return np.asarray(val, dtype=np.float64).reshape(n, n)

    def max_energy(self, q: np.ndarray) -> float:
        if self.distance_fn is None:
            d = float(np.linalg.norm(q))
        elif callable(self.distance_fn):
            d = float(self.distance_fn(q))
        else:
            d = float(self.distance_fn)
        if self.e_max_fn is not None:
            return float(self.e_max_fn(d))
        d_pos = max(0.0, d)
        scale = (d_pos * d_pos) / (d_pos * d_pos + self.d_safe * self.d_safe)
        return self.e_max_near + (self.e_max_far - self.e_max_near) * scale

    def h(self, x: np.ndarray) -> float:
        q, dq = self._split(x)
        M_q = self._eval_M(q)
        ke = 0.5 * float(dq @ M_q @ dq)
        return float(self.max_energy(q) - ke - self.margin_offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return _fd_grad_subspace(self.h, np.asarray(x, dtype=np.float64).flatten(), None, self.eps)

    def inflate(self, delta: float) -> "KineticEnergyBarrier":
        return KineticEnergyBarrier(
            n_dof=self.n_dof,
            M_fn=self.M_fn,
            distance_fn=self.distance_fn,
            e_max_far=self.e_max_far,
            e_max_near=self.e_max_near,
            d_safe=self.d_safe,
            e_max_fn=self.e_max_fn,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            eps=self.eps,
            margin_offset=self.margin_offset + float(delta),
        )


class WrenchConeBarrier(BarrierConstraint):
    """Contact friction-cone and normal-force barrier over a state/wrench vector.

    Enforces simultaneously:
      - Unilateral contact:  ``f_n >= f_min``
      - Normal force cap:    ``f_n <= f_max``
      - Coulomb friction:    ``||f_t|| <= mu * f_n``
    via ``h(x) = min(mu * f_n - ||f_t||, f_max - f_n, f_n - f_min) >= 0``.
    """

    def __init__(
        self,
        mu: float = 0.5,
        f_max: float = 100.0,
        f_min: float = 0.0,
        normal_axis: int = 2,
        force_slice: Optional[slice] = None,
        wrench_fn: Optional[Callable[[np.ndarray], Sequence[float]]] = None,
        alpha: float = 0.5,
        alpha2: float = 1.0,
        relative_degree: int = 1,
        hard: bool = True,
        eps: float = 1e-5,
        margin_offset: float = 0.0,
    ):
        self.mu = float(mu)
        self.f_max = float(f_max)
        self.f_min = float(f_min)
        self.normal_axis = int(normal_axis)
        self.force_slice = force_slice
        self.wrench_fn = wrench_fn
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.eps = float(eps)
        self.margin_offset = float(margin_offset)

    def _force_vec(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        w = np.asarray(self.wrench_fn(x_arr), dtype=np.float64).flatten() if self.wrench_fn is not None else x_arr
        if self.force_slice is not None:
            return w[self.force_slice]
        return w[:3] if w.shape[0] >= 3 else w

    def h(self, x: np.ndarray) -> float:
        f = self._force_vec(x)
        fn = float(f[self.normal_axis])
        ft = np.delete(f, self.normal_axis)
        ft_norm = float(np.sqrt(np.dot(ft, ft) + 1e-12) - 1e-6)
        h_cone = self.mu * fn - ft_norm
        h_max = self.f_max - fn
        h_min = fn - self.f_min
        return float(min(h_cone, h_max, h_min) - self.margin_offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return _fd_grad_subspace(self.h, np.asarray(x, dtype=np.float64).flatten(), None, self.eps)

    def inflate(self, delta: float) -> "WrenchConeBarrier":
        return WrenchConeBarrier(
            mu=self.mu,
            f_max=self.f_max,
            f_min=self.f_min,
            normal_axis=self.normal_axis,
            force_slice=self.force_slice,
            wrench_fn=self.wrench_fn,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            eps=self.eps,
            margin_offset=self.margin_offset + float(delta),
        )


class SimplexBackupController:
    """Passive gravity-compensated damping + braking controller for Simplex fallback.

    Computes ``tau_safe = -K_d qdot + g(q)`` (mapped through ``B(q)`` when
    ``B_fn`` is provided and clipped to ``[u_min, u_max]`` if configured).
    """

    def __init__(
        self,
        n_dof: Optional[int] = None,
        kd: Union[float, Sequence[float], np.ndarray] = 5.0,
        g_fn: Optional[Union[Callable[[np.ndarray], np.ndarray], np.ndarray]] = None,
        B_fn: Optional[Union[Callable[[np.ndarray], np.ndarray], np.ndarray]] = None,
        u_min: Optional[Sequence[float]] = None,
        u_max: Optional[Sequence[float]] = None,
    ):
        self.n_dof = None if n_dof is None else int(n_dof)
        self.kd = kd
        self.g_fn = g_fn
        self.B_fn = B_fn
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=np.float64).flatten()
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=np.float64).flatten()

    def _damping_term(self, dq: np.ndarray) -> np.ndarray:
        if np.ndim(self.kd) == 0:
            return float(self.kd) * dq
        kd_arr = np.asarray(self.kd, dtype=np.float64)
        if kd_arr.ndim == 2:
            return kd_arr @ dq
        return kd_arr.flatten() * dq

    def _gravity_term(self, q: np.ndarray) -> np.ndarray:
        if self.g_fn is None:
            return np.zeros_like(q, dtype=np.float64)
        val = self.g_fn(q) if callable(self.g_fn) else self.g_fn
        return np.asarray(val, dtype=np.float64).flatten()

    def compute_safe_action(self, x: np.ndarray) -> np.ndarray:
        """Return passive gravity-compensated damping torque for state x = [q, qdot]."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = self.n_dof if self.n_dof is not None else x_arr.shape[0] // 2
        q = x_arr[:n]
        dq = x_arr[n : 2 * n]
        tau = -self._damping_term(dq) + self._gravity_term(q)
        if self.B_fn is not None:
            B_val = self.B_fn(q) if callable(self.B_fn) else self.B_fn
            B_mat = np.asarray(B_val, dtype=np.float64).reshape(n, -1)
            u_safe, _, _, _ = np.linalg.lstsq(B_mat, tau, rcond=None)
        else:
            u_safe = tau
        if self.u_min is not None:
            u_safe = np.maximum(u_safe, self.u_min)
        if self.u_max is not None:
            u_safe = np.minimum(u_safe, self.u_max)
        return np.asarray(u_safe, dtype=np.float64).flatten()
