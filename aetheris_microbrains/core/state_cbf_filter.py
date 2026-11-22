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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("aetheris.state_cbf")


class BarrierConstraint(ABC):
    """Interface: h(x) >= 0 keeps the safe set; grad_h(x) is its Jacobian row."""

    @abstractmethod
    def h(self, x: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad_h(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def inflate(self, delta: float) -> "BarrierConstraint":
        """Return a barrier shifted by -delta (safe set shrinks)."""
        raise NotImplementedError


class HalfPlaneBarrier(BarrierConstraint):
    """h(x) = a.x - b >= 0. Covers walls and per-axis limits (e.g. |v|<=vmax)."""

    def __init__(self, a: Sequence[float], b: float, alpha: float = 0.5):
        self.a = np.asarray(a, dtype=np.float64).flatten()
        self.b = float(b)
        self.alpha = float(alpha)

    def h(self, x: np.ndarray) -> float:
        return float(np.dot(self.a, np.asarray(x, dtype=np.float64).flatten()) - self.b)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return self.a

    def inflate(self, delta: float) -> "HalfPlaneBarrier":
        out = HalfPlaneBarrier(self.a, self.b, self.alpha)
        out.b = self.b + float(delta)
        return out


class SphereBarrier(BarrierConstraint):
    """h(x) = ||x - c||^2 - r^2 >= 0 (keeps x outside the ball around c)."""

    def __init__(self, center: Sequence[float], radius: float, alpha: float = 0.5):
        self.c = np.asarray(center, dtype=np.float64).flatten()
        self.r = float(radius)
        self.alpha = float(alpha)

    def h(self, x: np.ndarray) -> float:
        d = np.asarray(x, dtype=np.float64).flatten() - self.c
        return float(np.dot(d, d) - self.r * self.r)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(x, dtype=np.float64).flatten() - self.c)

    def inflate(self, delta: float) -> "SphereBarrier":
        return SphereBarrier(self.c, self.r + float(delta), self.alpha)


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


class LinearDynamics(ControlAffineDynamics):
    """Constant A/B matrices: f(x) = A x, g(x) = B."""

    def __init__(self, A: np.ndarray, B: np.ndarray):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        super().__init__(A.shape[0], B.shape[1], f=lambda x: A @ np.asarray(x, dtype=np.float64), g=lambda x: B)
        self.A, self.B = A, B


class StateCBFSafetyFilter:
    """
    Exact-projection CBF-QP safety filter.

    filter(x, u_nominal) returns the Euclidean projection of u_nominal onto
        { u : L_g h_i u >= -alpha_i h_i - L_f h_i  for all i,  u_min <= u <= u_max }
    computed with Dykstra's algorithm (exact for convex intersections) using a
    bounded iteration budget. Status receipts are honest: when the iteration
    budget ends without satisfying every constraint (empty / nearly-empty safe
    set), ``verified_safe`` is False and the nominal (clipped) command is
    returned with status ``unsafe_no_solution``.
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
    ):
        self.dynamics = dynamics
        self.barriers = list(barriers)
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=np.float64).flatten()
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=np.float64).flatten()
        self.conformal_margin = float(conformal_margin)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.calls = 0
        self.unsafe_calls = 0

    def set_conformal_margin(self, delta: float) -> None:
        """Inflate all barriers by the split-conformal band ``delta`` >= 0."""
        self.conformal_margin = max(0.0, float(delta))

    def _halfspaces(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Rows (A_i, b_i) with A_i u >= b_i, one per inflated barrier."""
        if not self.barriers:
            return (
                np.zeros((0, self.dynamics.control_dim), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                [],
            )
        f_x = self.dynamics.f(x)
        G_x = self.dynamics.g(x)
        A_rows, b_rows, alphas = [], [], []
        for bar in self.barriers:
            gh = bar.grad_h(x).astype(np.float64)
            L_f_h = float(np.dot(gh, f_x))
            L_g_h = gh @ G_x  # row over u
            h_eff = bar.h(x) - self.conformal_margin
            A_rows.append(L_g_h)
            b_rows.append(-float(bar.alpha) * h_eff - L_f_h)
            alphas.append(float(bar.alpha))
        return np.vstack(A_rows), np.asarray(b_rows, dtype=np.float64), alphas

    def _clip_box(self, u: np.ndarray) -> np.ndarray:
        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)
        return u

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

        A, b, _alphas = self._halfspaces(x)
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

        # --- Dykstra's algorithm over halfspaces + box ------------------------
        m = A.shape[0]
        sets: List[Tuple[str, Any]] = [("halfspace", (A[i], b[i])) for i in range(m)]
        if self.u_min is not None or self.u_max is not None:
            sets.append(("box", None))

        n_sets = len(sets)
        if n_sets == 0:
            result = u.copy()
        else:
            s_list = [np.zeros_like(u) for _ in range(n_sets)]
            w = u.copy()
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
        if not verified:
            self.unsafe_calls += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "safe" if verified else "unsafe_no_solution",
            "verified_safe": verified,
            "u_filtered": result.tolist(),
            "u_nominal": u_nom_list,
            "min_barrier_slack": round(residual, 9),
            "conformal_margin": self.conformal_margin,
            "n_barriers": m,
            "latency_ms": round(latency_ms, 4),
        }

    def filter_trajectory(
        self,
        x: np.ndarray,
        nominal_actions: Sequence[Sequence[float]],
        dt: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Clamp a VLA-style ACTION CHUNK: roll the (propagated, not simulated)
        dynamics forward, filtering each waypoint against the state predicted
        by the *nominal* rollout of the FILTERED actions. Returns the clamped
        chunk plus a per-step verified_safe receipt.
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        filtered: List[List[float]] = []
        receipts: List[Dict[str, Any]] = []
        x_run = x.copy()
        all_safe = True
        for u_nom in nominal_actions:
            out = self.filter(x_run, u_nom)
            u_f = np.asarray(out["u_filtered"], dtype=np.float64)
            receipts.append({"verified_safe": out["verified_safe"], "status": out["status"]})
            all_safe = all_safe and out["verified_safe"]
            filtered.append(u_f.tolist())
            f_x = self.dynamics.f(x_run)
            G_x = self.dynamics.g(x_run)
            x_run = x_run + dt * (f_x + G_x @ u_f)
        return {
            "actions_filtered": filtered,
            "all_verified_safe": all_safe,
            "per_step": receipts,
            "final_state_prediction": x_run.tolist(),
        }
