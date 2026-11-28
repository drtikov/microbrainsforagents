"""State-space Control Barrier Function (CBF) safety filter for robotics.

P: 2026-11-18 — robotics extension. ``core/cbf_spectral_barrier.py`` applies
correct CBF projection math to WEIGHT matrices (sigma_max bounds); this module
retargets the same mathematics to ROBOT STATE space:

    constraint   h_i(x) >= 0                     (barrier, 0 = safe boundary)
    dynamics     xdot = f(x) + g(x) u            (control-affine)
    condition    dh_i/dx . (f(x) + g(x) u) >= -alpha_i * h_i(x)
    filter       u* = argmin ||u - u_nom||^2     s.t. CBF halfspaces + u box

For control-affine dynamics each CBF condition is an AFFINE halfspace in u:
    L_g h_i(x) . u >= -alpha_i * h_i(x) - L_f h_i(x)
so the safe control set is a convex polytope (intersected with the box), and
min ||u - u_nom||^2 over it is a Euclidean projection. That projection is
computed EXACTLY with Dykstra's cyclic projection algorithm (which converges
to the true projection onto a finite intersection of convex sets, unlike
plain POCS). When the safe set is empty the filter returns
``verified_safe=False`` and never claims a guarantee — the caller is expected
to fail closed (escalate/halt).

Conformal composition: ``set_conformal_margin(delta)`` inflates every barrier
by a nonconformity band delta (e.g. the split-conformal q_hat of a
state-estimation residual), so perception/estimation uncertainty strictly
shrinks the certified safe set before the QP runs.

Everything here is deterministic bounded-work NumPy: no external solver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("aetheris.state_cbf")


def _fd_jacobian(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Jacobian d(fn)/dx at x."""
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    n = x_arr.shape[0]
    f0 = np.asarray(fn(x_arr), dtype=np.float64).flatten()
    m = f0.shape[0]
    J = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        ei = np.zeros(n, dtype=np.float64)
        ei[i] = eps
        fp = np.asarray(fn(x_arr + ei), dtype=np.float64).flatten()
        fm = np.asarray(fn(x_arr - ei), dtype=np.float64).flatten()
        J[:, i] = (fp - fm) / (2.0 * eps)
    return J


class BarrierConstraint(ABC):
    """Interface: h(x) >= 0 keeps the safe set; grad_h(x) is its Jacobian row."""

    relative_degree: int = 1
    hard: bool = True
    alpha: float = 0.5
    alpha2: float = 1.0

    @abstractmethod
    def h(self, x: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad_h(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dh_dt(self, x: np.ndarray) -> float:
        """Partial time derivative dh/dt(x, t) holding state x fixed; 0.0 for static barriers."""
        _ = x
        return 0.0

    def d2h_dt2(self, x: np.ndarray) -> float:
        """Second partial time derivative d^2h/dt^2(x, t) holding state x fixed; 0.0 by default."""
        _ = x
        return 0.0

    def grad_dh_dt(self, x: np.ndarray) -> np.ndarray:
        """Mixed partial gradient nabla_x (dh/dt)(x, t); zeros of state shape by default."""
        arr = np.asarray(x, dtype=np.float64)
        return np.zeros(arr.shape, dtype=np.float64)

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Hessian nabla^2 h(x); defaults to central finite differences over grad_h."""
        H = _fd_jacobian(self.grad_h, x, eps=eps)
        return 0.5 * (H + H.T)

    @abstractmethod
    def inflate(self, delta: float) -> "BarrierConstraint":
        """Return a barrier shifted by -delta (safe set shrinks)."""
        raise NotImplementedError


class HalfPlaneBarrier(BarrierConstraint):
    """h(x) = a.x - b >= 0. Covers walls and per-axis limits (e.g. |v|<=vmax)."""

    def __init__(
        self,
        a: Sequence[float],
        b: float,
        alpha: float = 0.5,
        relative_degree: int = 1,
        hard: bool = True,
        alpha2: float = 1.0,
    ):
        self.a = np.asarray(a, dtype=np.float64).flatten()
        self.b = float(b)
        self.alpha = float(alpha)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.alpha2 = float(alpha2)

    def h(self, x: np.ndarray) -> float:
        return float(np.dot(self.a, np.asarray(x, dtype=np.float64).flatten()) - self.b)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return self.a

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        n = np.asarray(x, dtype=np.float64).flatten().shape[0]
        return np.zeros((n, n), dtype=np.float64)

    def inflate(self, delta: float) -> "HalfPlaneBarrier":
        return HalfPlaneBarrier(
            self.a,
            self.b + float(delta),
            self.alpha,
            relative_degree=self.relative_degree,
            hard=self.hard,
            alpha2=self.alpha2,
        )


class SphereBarrier(BarrierConstraint):
    """h(x) = ||x - c||^2 - r^2 >= 0 (keeps x outside the ball around c)."""

    def __init__(
        self,
        center: Sequence[float],
        radius: float,
        alpha: float = 0.5,
        relative_degree: int = 1,
        hard: bool = True,
        alpha2: float = 1.0,
    ):
        self.c = np.asarray(center, dtype=np.float64).flatten()
        self.r = float(radius)
        self.alpha = float(alpha)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.alpha2 = float(alpha2)

    def h(self, x: np.ndarray) -> float:
        d = np.asarray(x, dtype=np.float64).flatten() - self.c
        return float(np.dot(d, d) - self.r * self.r)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(x, dtype=np.float64).flatten() - self.c)

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        n = np.asarray(x, dtype=np.float64).flatten().shape[0]
        return 2.0 * np.eye(n, dtype=np.float64)

    def inflate(self, delta: float) -> "SphereBarrier":
        return SphereBarrier(
            self.c,
            self.r + float(delta),
            self.alpha,
            relative_degree=self.relative_degree,
            hard=self.hard,
            alpha2=self.alpha2,
        )


class HighOrderBarrier(BarrierConstraint):
    """Relative-degree-2 High-Order CBF (HOCBF) wrapper around any base barrier.

    If ``base_barrier`` operates on a configuration subspace ``q`` smaller than the
    full state ``x = [q, qdot]`` (detected via ``state_slice``, ``q_dim``, or
    ``.a``/``.c`` length), ``grad_h`` and ``hess_h`` are padded with zeros on
    the remaining state coordinates.
    """

    def __init__(
        self,
        base_barrier: BarrierConstraint,
        alpha1: Optional[float] = None,
        alpha2: float = 1.0,
        hard: Optional[bool] = None,
        state_slice: Optional[slice] = None,
        q_dim: Optional[int] = None,
    ):
        self.base_barrier = base_barrier
        self.alpha1 = float(alpha1 if alpha1 is not None else getattr(base_barrier, "alpha", 0.5))
        self.alpha = self.alpha1
        self.alpha2 = float(alpha2)
        self.relative_degree = 2
        self.hard = bool(getattr(base_barrier, "hard", True) if hard is None else hard)
        if state_slice is not None:
            self.state_slice: Optional[slice] = state_slice
        elif q_dim is not None:
            self.state_slice = slice(0, int(q_dim))
        else:
            self.state_slice = None

    def _resolve_slice(self, n_state: int) -> slice:
        if self.state_slice is not None:
            return self.state_slice
        for attr in ("a", "c"):
            if hasattr(self.base_barrier, attr):
                sub_dim = int(np.asarray(getattr(self.base_barrier, attr)).flatten().shape[0])
                if sub_dim < n_state:
                    return slice(0, sub_dim)
        if getattr(self.base_barrier, "n_dof", None) is not None:
            sub_dim = int(getattr(self.base_barrier, "n_dof"))
            if sub_dim < n_state:
                return slice(0, sub_dim)
        return slice(0, n_state)

    def h(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        sl = self._resolve_slice(x_arr.shape[0])
        return float(self.base_barrier.h(x_arr[sl]))

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        sl = self._resolve_slice(n)
        g_sub = np.asarray(self.base_barrier.grad_h(x_arr[sl]), dtype=np.float64).flatten()
        if g_sub.shape[0] == n:
            return g_sub
        out = np.zeros(n, dtype=np.float64)
        out[sl] = g_sub
        return out

    def dh_dt(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        sl = self._resolve_slice(x_arr.shape[0])
        return float(self.base_barrier.dh_dt(x_arr[sl]))

    def d2h_dt2(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        sl = self._resolve_slice(x_arr.shape[0])
        return float(self.base_barrier.d2h_dt2(x_arr[sl]))

    def grad_dh_dt(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        sl = self._resolve_slice(n)
        g_sub = np.asarray(self.base_barrier.grad_dh_dt(x_arr[sl]), dtype=np.float64).flatten()
        if g_sub.shape[0] == n:
            return g_sub
        out = np.zeros(n, dtype=np.float64)
        out[sl] = g_sub
        return out

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        sl = self._resolve_slice(n)
        if hasattr(self.base_barrier, "hess_h"):
            H_sub = np.asarray(self.base_barrier.hess_h(x_arr[sl]), dtype=np.float64)
        else:
            H_sub = _fd_jacobian(self.base_barrier.grad_h, x_arr[sl], eps=eps)
            H_sub = 0.5 * (H_sub + H_sub.T)
        if H_sub.shape == (n, n):
            return H_sub
        out = np.zeros((n, n), dtype=np.float64)
        out[sl, sl] = H_sub
        return out

    def set_time(self, t: float) -> None:
        if hasattr(self.base_barrier, "set_time") and callable(self.base_barrier.set_time):
            self.base_barrier.set_time(float(t))
        elif hasattr(self.base_barrier, "current_time"):
            self.base_barrier.current_time = float(t)

    def advance_time(self, dt: float) -> float:
        if hasattr(self.base_barrier, "advance_time") and callable(self.base_barrier.advance_time):
            return float(self.base_barrier.advance_time(float(dt)))
        return float(getattr(self.base_barrier, "current_time", 0.0))

    def inflate(self, delta: float) -> "HighOrderBarrier":
        return HighOrderBarrier(
            self.base_barrier.inflate(delta),
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            hard=self.hard,
            state_slice=self.state_slice,
        )


class ControlAffineDynamics:
    """xdot = f(x) + g(x) u. Subclass or pass callables."""

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self._f = f
        self._g = g

    def f(self, x: np.ndarray) -> np.ndarray:
        if self._f is not None:
            return np.asarray(self._f(x), dtype=np.float64).flatten()
        return np.zeros(self.state_dim, dtype=np.float64)

    def g(self, x: np.ndarray) -> np.ndarray:
        if self._g is not None:
            return np.asarray(self._g(x), dtype=np.float64)
        return np.eye(self.state_dim, self.control_dim, dtype=np.float64)

    def df_dx(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Jacobian df/dx evaluated at state x (defaults to central finite differences)."""
        return _fd_jacobian(self.f, x, eps=eps)


class LinearDynamics(ControlAffineDynamics):
    """Constant A/B matrices: f(x) = A x, g(x) = B."""

    def __init__(self, A: np.ndarray, B: np.ndarray):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        super().__init__(A.shape[0], B.shape[1], f=lambda x: A @ np.asarray(x, dtype=np.float64), g=lambda x: B)
        self.A, self.B = A, B

    def df_dx(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return self.A


class EulerLagrangeDynamics(ControlAffineDynamics):
    """Second-order rigid-body Euler-Lagrange dynamics in state x = [q, qdot].

    Models::
        M(q) qddot + C(q, qdot) qdot + g(q) = B(q) u
        xdot = f(x) + g_ctrl(x) u
    where ``f(x) = [qdot, M(q)^{-1} (-C(q, qdot) qdot - g(q))]`` and
    ``g_ctrl(x) = [0; M(q)^{-1} B(q)]``.
    """

    def __init__(
        self,
        n_dof: int,
        control_dim: Optional[int] = None,
        M_fn: Optional[Union[Callable[[np.ndarray], np.ndarray], np.ndarray]] = None,
        C_fn: Optional[Union[Callable[[np.ndarray, np.ndarray], np.ndarray], np.ndarray]] = None,
        g_fn: Optional[Union[Callable[[np.ndarray], np.ndarray], np.ndarray]] = None,
        B_fn: Optional[Union[Callable[[np.ndarray], np.ndarray], np.ndarray]] = None,
    ):
        self.n_dof = int(n_dof)
        m_ctrl = self.n_dof if control_dim is None else int(control_dim)
        super().__init__(2 * self.n_dof, m_ctrl)
        self._M_fn = M_fn
        self._C_fn = C_fn
        self._g_fn = g_fn
        self._B_fn = B_fn

    def _split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        return x_arr[: self.n_dof], x_arr[self.n_dof : 2 * self.n_dof]

    def M(self, q: np.ndarray) -> np.ndarray:
        if self._M_fn is None:
            return np.eye(self.n_dof, dtype=np.float64)
        val = self._M_fn(q) if callable(self._M_fn) else self._M_fn
        return np.asarray(val, dtype=np.float64).reshape(self.n_dof, self.n_dof)

    def C_dot(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        if self._C_fn is None:
            return np.zeros(self.n_dof, dtype=np.float64)
        val = self._C_fn(q, dq) if callable(self._C_fn) else self._C_fn
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 2:
            return arr @ dq
        return arr.flatten()

    def grav(self, q: np.ndarray) -> np.ndarray:
        if self._g_fn is None:
            return np.zeros(self.n_dof, dtype=np.float64)
        val = self._g_fn(q) if callable(self._g_fn) else self._g_fn
        return np.asarray(val, dtype=np.float64).flatten()

    def B(self, q: np.ndarray) -> np.ndarray:
        if self._B_fn is None:
            return np.eye(self.n_dof, self.control_dim, dtype=np.float64)
        val = self._B_fn(q) if callable(self._B_fn) else self._B_fn
        return np.asarray(val, dtype=np.float64).reshape(self.n_dof, self.control_dim)

    def f(self, x: np.ndarray) -> np.ndarray:
        q, dq = self._split(x)
        M_q = self.M(q)
        rhs = -self.C_dot(q, dq) - self.grav(q)
        qddot_drift = np.linalg.solve(M_q, rhs)
        return np.concatenate([dq, qddot_drift])

    def g(self, x: np.ndarray) -> np.ndarray:
        q, _ = self._split(x)
        M_q = self.M(q)
        B_q = self.B(q)
        M_inv_B = np.linalg.solve(M_q, B_q)
        top = np.zeros((self.n_dof, self.control_dim), dtype=np.float64)
        return np.vstack([top, M_inv_B])

    def J_f(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """State Jacobian df/dx via central finite differences."""
        return _fd_jacobian(self.f, x, eps=eps)

    def df_dx(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return self.J_f(x, eps=eps)

    def lie_derivatives(
        self, barrier: BarrierConstraint, x: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """Compute (L_f h, L_f^2 h, L_g L_f h) for a barrier at state x."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        f_x = self.f(x_arr)
        G_x = self.g(x_arr)
        J_f = self.J_f(x_arr)
        gh = np.asarray(barrier.grad_h(x_arr), dtype=np.float64).flatten()
        H_h = np.asarray(barrier.hess_h(x_arr), dtype=np.float64)
        L_f_h = float(np.dot(gh, f_x))
        L_f2_h = float(f_x @ H_h @ f_x + gh @ (J_f @ f_x))
        L_g_L_f_h = np.asarray((f_x @ H_h @ G_x) + (gh @ (J_f @ G_x)), dtype=np.float64).flatten()
        return L_f_h, L_f2_h, L_g_L_f_h


class StateCBFSafetyFilter:
    """
    Exact-projection CBF-QP safety filter.

    filter(x, u_nominal) returns the Euclidean projection of u_nominal onto
        { u : L_g h_i u >= -alpha_i h_i - L_f h_i  for all i,  u_min <= u <= u_max }
    with automatic relative-degree-2 HOCBF assembly when ``bar.relative_degree == 2``.
    Projection is computed with Dykstra's algorithm (exact for convex intersections)
    using a bounded iteration budget.

    When the combined constraints are infeasible:
    - Stage A: if soft barriers (``hard=False``) are mixed with hard barriers
      (``hard=True``), retries Dykstra with only the hard barriers and reports
      ``status="safe_soft_relaxed"`` and ``relaxed_soft_barriers``.
    - Stage B: if hard barriers remain infeasible and ``backup_controller`` is
      configured, invokes ``backup_controller.compute_safe_action(x)`` (clipped to
      the control box) and returns ``status="simplex_braking_engaged"``,
      ``backup_engaged=True``, ``verified_safe=False``. When ``backup_controller``
      is ``None``, returns ``status="unsafe_no_solution"`` with ``verified_safe=False``.
    """

    def __init__(
        self,
        dynamics: ControlAffineDynamics,
        barriers: Sequence[BarrierConstraint],
        u_min: Optional[Sequence[float]] = None,
        u_max: Optional[Sequence[float]] = None,
        conformal_margin: float = 0.0,
        max_iters: int = 200,
        tol: float = 1e-9,
        backup_controller: Optional[Any] = None,
        use_native_rust: bool = False,
    ):
        self.dynamics = dynamics
        self.barriers = list(barriers)
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=np.float64).flatten()
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=np.float64).flatten()
        self.conformal_margin = float(conformal_margin)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.backup_controller = backup_controller
        self.use_native_rust = bool(use_native_rust)
        self.calls = 0
        self.unsafe_calls = 0

    def set_conformal_margin(self, delta: float) -> None:
        """Inflate all barriers by the split-conformal band ``delta`` >= 0."""
        self.conformal_margin = max(0.0, float(delta))

    def _halfspaces_with_flags(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[bool]]:
        """Rows (A_i, b_i, alpha_i, hard_i) with A_i u >= b_i, one per inflated barrier."""
        if not self.barriers:
            return (
                np.zeros((0, self.dynamics.control_dim), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                [],
                [],
            )
        f_x = self.dynamics.f(x)
        G_x = self.dynamics.g(x)
        J_f: Optional[np.ndarray] = None
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []
        alphas: List[float] = []
        hard_flags: List[bool] = []
        for bar in self.barriers:
            eff_bar = bar.inflate(self.conformal_margin) if self.conformal_margin > 0.0 else bar
            gh = np.asarray(eff_bar.grad_h(x), dtype=np.float64).flatten()
            L_f_h = float(np.dot(gh, f_x))
            h_eff = float(eff_bar.h(x))
            dh_dt_val = float(eff_bar.dh_dt(x))
            is_hard = bool(getattr(eff_bar, "hard", getattr(bar, "hard", True)))
            rel_deg = int(getattr(eff_bar, "relative_degree", getattr(bar, "relative_degree", 1)))
            if rel_deg == 2:
                if J_f is None:
                    if hasattr(self.dynamics, "df_dx"):
                        J_f = np.asarray(self.dynamics.df_dx(x), dtype=np.float64)
                    elif hasattr(self.dynamics, "J_f"):
                        J_f = np.asarray(self.dynamics.J_f(x), dtype=np.float64)
                    else:
                        J_f = _fd_jacobian(self.dynamics.f, x)
                if hasattr(eff_bar, "hess_h"):
                    H_h = np.asarray(eff_bar.hess_h(x), dtype=np.float64)
                else:
                    H_h = _fd_jacobian(eff_bar.grad_h, x, eps=1e-5)
                    H_h = 0.5 * (H_h + H_h.T)
                L_f2_h = float(f_x @ H_h @ f_x + gh @ (J_f @ f_x))
                L_g_L_f_h = np.asarray((f_x @ H_h @ G_x) + (gh @ (J_f @ G_x)), dtype=np.float64).flatten()
                d2h_dt2_val = float(eff_bar.d2h_dt2(x))
                g_dh_dt = np.asarray(eff_bar.grad_dh_dt(x), dtype=np.float64).flatten()
                L_f_dh_dt = float(np.dot(g_dh_dt, f_x))
                L_g_dh_dt = np.asarray(g_dh_dt @ G_x, dtype=np.float64).flatten()
                alpha1 = float(getattr(eff_bar, "alpha1", getattr(eff_bar, "alpha", 0.5)))
                alpha2 = float(getattr(eff_bar, "alpha2", getattr(bar, "alpha2", 1.0)))
                A_rows.append(L_g_L_f_h + L_g_dh_dt)
                b_rows.append(
                    -(
                        L_f2_h
                        + 2.0 * L_f_dh_dt
                        + d2h_dt2_val
                        + (alpha1 + alpha2) * (L_f_h + dh_dt_val)
                        + alpha1 * alpha2 * h_eff
                    )
                )
                alphas.append(alpha1)
            else:
                L_g_h = np.asarray(gh @ G_x, dtype=np.float64).flatten()
                alpha = float(getattr(eff_bar, "alpha", 0.5))
                A_rows.append(L_g_h)
                b_rows.append(-alpha * h_eff - L_f_h - dh_dt_val)
                alphas.append(alpha)
            hard_flags.append(is_hard)
        return np.vstack(A_rows), np.asarray(b_rows, dtype=np.float64), alphas, hard_flags

    def _halfspaces(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Rows (A_i, b_i) with A_i u >= b_i, one per inflated barrier."""
        A, b, alphas, _ = self._halfspaces_with_flags(x)
        return A, b, alphas

    def _clip_box(self, u: np.ndarray) -> np.ndarray:
        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)
        return u

    def _solve_dykstra(
        self, A: np.ndarray, b: np.ndarray, u_nom: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """Run Dykstra's cyclic projection onto {A u >= b} intersect [u_min, u_max]."""
        m = A.shape[0]
        sets: List[Tuple[str, Any]] = [("halfspace", (A[i], b[i])) for i in range(m)]
        if self.u_min is not None or self.u_max is not None:
            sets.append(("box", None))

        n_sets = len(sets)
        if n_sets == 0:
            result = u_nom.copy()
        else:
            s_list = [np.zeros_like(u_nom) for _ in range(n_sets)]
            w = u_nom.copy()
            for _it in range(self.max_iters):
                converged = True
                for j, (kind, payload) in enumerate(sets):
                    y = w + s_list[j]
                    if kind == "halfspace":
                        a_row, bj = payload
                        a_norm2 = float(a_row @ a_row)
                        viol = bj - float(a_row @ y)
                        if viol > 0.0:
                            if a_norm2 <= 1e-18:
                                proj = y.copy()
                            else:
                                proj = y + (viol / a_norm2) * a_row
                            converged = False
                        else:
                            proj = y.copy()
                    else:
                        proj = self._clip_box(y)
                        if not np.allclose(proj, y, atol=self.tol):
                            converged = False
                    s_new = y - proj
                    if not np.allclose(s_new, s_list[j], atol=self.tol):
                        converged = False
                    s_list[j] = s_new
                    w = proj
                if converged:
                    break
            result = w

        residual = float(np.min(A @ result - b)) if A.size else 0.0
        box_ok = (
            (self.u_min is None or bool(np.all(result >= self.u_min - 1e-6)))
            and (self.u_max is None or bool(np.all(result <= self.u_max + 1e-6)))
        )
        verified = bool(np.all(np.isfinite(result)) and box_ok and (not A.size or residual >= -1e-6))
        return result, residual, verified

    def _solve_projection(
        self, A: np.ndarray, b: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, str]:
        """Dispatch Dykstra projection to native Rust C-ABI when enabled and in-bounds, else Python."""
        if (
            self.use_native_rust
            and 0 < self.dynamics.control_dim <= 16
            and A.shape[0] <= 32
        ):
            try:
                from .rust_shim import NativeRTGateLib

                lib = NativeRTGateLib.get()
            except Exception:
                lib = None
            if lib is not None:
                res = lib.cbf_project(
                    A,
                    b,
                    u,
                    self.u_min,
                    self.u_max,
                    max_iters=self.max_iters,
                    tol=self.tol,
                )
                if res.get("status_code") in (0, 1):
                    return (
                        np.asarray(res["u_filtered"], dtype=np.float64),
                        float(res["min_barrier_slack"]),
                        bool(res["verified_safe"]),
                        "native_rust",
                    )

        result, residual, verified = self._solve_dykstra(A, b, u)
        return result, residual, verified, "python_numpy"

    def filter(self, x: np.ndarray, u_nominal: Sequence[float]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        x = np.asarray(x, dtype=np.float64).flatten()
        u = np.asarray(u_nominal, dtype=np.float64).flatten()
        u_nom_list = u_nominal if isinstance(u_nominal, list) else u.tolist()
        if u.shape[0] != self.dynamics.control_dim:
            return {
                "status": "dim_mismatch",
                "verified_safe": False,
                "u_filtered": u.tolist(),
                "u_nominal": u_nom_list,
            }
        self.calls += 1

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(u))):
            self.unsafe_calls += 1
            return {
                "status": "invalid_nonfinite_input",
                "verified_safe": False,
                "u_filtered": u.tolist(),
                "u_nominal": u_nom_list,
                "min_barrier_slack": -1.0,
                "conformal_margin": self.conformal_margin,
                "n_barriers": len(self.barriers),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        if self.u_min is not None and self.u_max is not None and np.any(self.u_min > self.u_max):
            self.unsafe_calls += 1
            return {
                "status": "infeasible_box",
                "verified_safe": False,
                "u_filtered": u.tolist(),
                "u_nominal": u_nom_list,
                "min_barrier_slack": -1.0,
                "conformal_margin": self.conformal_margin,
                "n_barriers": len(self.barriers),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        A, b, _alphas, hard_flags = self._halfspaces_with_flags(x)
        if (A.size and not np.all(np.isfinite(A))) or (b.size and not np.all(np.isfinite(b))):
            self.unsafe_calls += 1
            return {
                "status": "invalid_nonfinite_input",
                "verified_safe": False,
                "u_filtered": u.tolist(),
                "u_nominal": u_nom_list,
                "min_barrier_slack": -1.0,
                "conformal_margin": self.conformal_margin,
                "n_barriers": A.shape[0] if A.ndim == 2 else len(self.barriers),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        m = A.shape[0]
        result, residual, verified, solver_backend = self._solve_projection(A, b, u)
        if verified:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "safe",
                "verified_safe": True,
                "u_filtered": result.tolist(),
                "u_nominal": u_nom_list,
                "min_barrier_slack": round(residual, 9),
                "conformal_margin": self.conformal_margin,
                "n_barriers": m,
                "solver_backend": solver_backend,
                "latency_ms": round(latency_ms, 4),
            }

        # Stage A: relax soft barriers (hard=False) when mixed with hard barriers
        hard_mask = np.asarray(hard_flags, dtype=bool)
        n_hard = int(np.sum(hard_mask))
        n_soft = m - n_hard
        if n_soft > 0 and n_hard > 0:
            res_hard, slack_hard, ok_hard, hard_backend = self._solve_projection(
                A[hard_mask], b[hard_mask], u
            )
            if ok_hard:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "status": "safe_soft_relaxed",
                    "verified_safe": True,
                    "u_filtered": res_hard.tolist(),
                    "u_nominal": u_nom_list,
                    "min_barrier_slack": round(slack_hard, 9),
                    "conformal_margin": self.conformal_margin,
                    "n_barriers": m,
                    "relaxed_soft_barriers": n_soft,
                    "solver_backend": hard_backend,
                    "latency_ms": round(latency_ms, 4),
                }

        # Stage B: Simplex backup controller fallback when configured
        self.unsafe_calls += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if self.backup_controller is not None:
            u_safe = np.asarray(self.backup_controller.compute_safe_action(x), dtype=np.float64).flatten()
            u_safe = self._clip_box(u_safe)
            backup_slack = float(np.min(A @ u_safe - b)) if A.size else 0.0
            return {
                "status": "simplex_braking_engaged",
                "verified_safe": False,
                "backup_engaged": True,
                "u_filtered": u_safe.tolist(),
                "u_nominal": u_nom_list,
                "min_barrier_slack": round(backup_slack, 9),
                "conformal_margin": self.conformal_margin,
                "n_barriers": m,
                "solver_backend": solver_backend,
                "latency_ms": round(latency_ms, 4),
            }

        return {
            "status": "unsafe_no_solution",
            "verified_safe": False,
            "u_filtered": result.tolist(),
            "u_nominal": u_nom_list,
            "min_barrier_slack": round(residual, 9),
            "conformal_margin": self.conformal_margin,
            "n_barriers": m,
            "solver_backend": solver_backend,
            "latency_ms": round(latency_ms, 4),
        }

    @staticmethod
    def _collect_barrier_tree_nodes(roots: Sequence[Any]) -> List[Any]:
        seen: set[int] = set()
        ordered: List[Any] = []
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            ordered.append(node)
            for attr in ("base_barrier", "hold_barrier", "goal_barrier"):
                child = getattr(node, attr, None)
                if child is not None:
                    stack.append(child)
            children = getattr(node, "barriers", None)
            if isinstance(children, (list, tuple)):
                stack.extend(children)
        return ordered

    def filter_trajectory(
        self,
        x: np.ndarray,
        nominal_actions: Sequence[Sequence[float]],
        dt: float = 0.05,
        advance_barrier_time: bool = True,
        restore_barrier_time: bool = True,
    ) -> Dict[str, Any]:
        """
        Clamp a VLA-style ACTION CHUNK: roll the (propagated, not simulated)
        dynamics forward, filtering each waypoint against the state predicted
        by the *nominal* rollout of the FILTERED actions. Time-varying barriers
        with ``advance_time(dt)`` step forward across the horizon and cleanly
        restore their ``current_time`` afterward when ``restore_barrier_time=True``.
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        dt_f = float(dt)
        filtered: List[List[float]] = []
        receipts: List[Dict[str, Any]] = []
        x_run = x.copy()
        all_safe = True

        all_nodes = self._collect_barrier_tree_nodes(self.barriers)
        saved_times: List[Tuple[Any, float]] = [
            (node, float(node.current_time)) for node in all_nodes if hasattr(node, "current_time")
        ]
        saved_ids = {id(node) for node, _ in saved_times}
        top_advance: List[Any] = []
        seen_top: set[int] = set()
        for bar in self.barriers:
            if id(bar) not in seen_top and callable(getattr(bar, "advance_time", None)):
                seen_top.add(id(bar))
                top_advance.append(bar)
        stateless_advance = [b for b in top_advance if id(b) not in saved_ids and not hasattr(b, "base_barrier")]

        steps_advanced = 0
        try:
            for u_nom in nominal_actions:
                out = self.filter(x_run, u_nom)
                u_f = np.asarray(out["u_filtered"], dtype=np.float64)
                receipts.append({"verified_safe": out["verified_safe"], "status": out["status"]})
                all_safe = all_safe and out["verified_safe"]
                filtered.append(u_f.tolist())
                f_x = self.dynamics.f(x_run)
                G_x = self.dynamics.g(x_run)
                x_run = x_run + dt_f * (f_x + G_x @ u_f)
                if advance_barrier_time and top_advance:
                    for bar in top_advance:
                        bar.advance_time(dt_f)
                    steps_advanced += 1
        finally:
            if restore_barrier_time and steps_advanced > 0:
                for node, orig_t in saved_times:
                    node.current_time = orig_t
                for bar in stateless_advance:
                    bar.advance_time(-steps_advanced * dt_f)

        return {
            "actions_filtered": filtered,
            "all_verified_safe": all_safe,
            "per_step": receipts,
            "final_state_prediction": x_run.tolist(),
        }
