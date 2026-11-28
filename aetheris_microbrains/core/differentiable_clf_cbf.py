"""Joint CLF-CBF-QP safety filter and implicit KKT differentiable CBF projection.

Provides:
1. ``ControlLyapunovConstraint`` & ``QuadraticCLF``: Control Lyapunov Function
   liveness specifications :math:`V(x) \\ge 0` with decay rate :math:`\\gamma > 0`.
2. ``CLFCBFSafetyFilter``: Joint CLF-CBF-QP over augmented decision vector
   :math:`z = [u \\in \\mathbb{R}^{d_u}, \\delta \\in \\mathbb{R}_{\\ge 0}]` solved via
   coordinate-scaled Dykstra projection in metric ``diag([1, ..., 1, slack_weight])``.
   Enforces hard CBF safety halfspaces while softly enforcing exponential CLF decay
   :math:`L_g V(x) u - \\delta \\le -\\gamma V(x) - L_f V(x)`.
3. ``DifferentiableCBFProjection``: Exact implicit KKT active-set differentiation of
   ``StateCBFSafetyFilter`` yielding ``du_du_nom``, ``du_dx``, ``du_dalpha``, and
   ``vjp(grad_u_star, jac_cache)`` for end-to-end policy backpropagation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .state_cbf_filter import (
    BarrierConstraint,
    ControlAffineDynamics,
    StateCBFSafetyFilter,
)


class ControlLyapunovConstraint(ABC):
    """Interface for Control Lyapunov Functions V(x) >= 0 with decay rate gamma."""

    gamma: float = 1.0
    slack_weight: float = 100.0

    @abstractmethod
    def V(self, x: np.ndarray) -> float:
        """Evaluate scalar Lyapunov value V(x) >= 0."""
        raise NotImplementedError

    @abstractmethod
    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient dV/dx at state x."""
        raise NotImplementedError


class QuadraticCLF(ControlLyapunovConstraint):
    """Quadratic CLF V(x) = 0.5 * (x - x_goal)^T P (x - x_goal)."""

    def __init__(
        self,
        x_goal: Sequence[float],
        P: Optional[np.ndarray] = None,
        gamma: float = 1.0,
        slack_weight: float = 100.0,
    ):
        self.x_goal = np.asarray(x_goal, dtype=np.float64).flatten()
        n = self.x_goal.shape[0]
        if P is None:
            self.P = np.eye(n, dtype=np.float64)
        else:
            P_arr = np.asarray(P, dtype=np.float64).reshape(n, n)
            self.P = 0.5 * (P_arr + P_arr.T)
        self.gamma = float(gamma)
        self.slack_weight = float(slack_weight)
        if self.slack_weight <= 0.0:
            raise ValueError("slack_weight must be strictly positive.")

    def V(self, x: np.ndarray) -> float:
        d = np.asarray(x, dtype=np.float64).flatten() - self.x_goal
        return float(0.5 * (d @ self.P @ d))

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        d = np.asarray(x, dtype=np.float64).flatten() - self.x_goal
        return np.asarray(self.P @ d, dtype=np.float64).flatten()


class CLFCBFSafetyFilter:
    """Joint CLF-CBF-QP safety and liveness filter.

    Minimizes::

        0.5 * ||u - u_nom||_2^2 + 0.5 * slack_weight * delta^2

    subject to:
      - CBF safety halfspaces: ``A_cbf u >= b_cbf``
      - CLF liveness halfspace: ``L_g V(x) u - delta <= -gamma * V(x) - L_f V(x)``
      - Control box bounds: ``u_min <= u <= u_max``
      - Non-negative relaxation: ``delta >= 0``

    Solved via scaled Dykstra cyclic projection in coordinates
    ``z_tilde = [u, sqrt(slack_weight) * delta]`` where the weighted quadratic
    objective becomes an exact Euclidean projection.
    """

    def __init__(
        self,
        dynamics: ControlAffineDynamics,
        barriers: Sequence[BarrierConstraint],
        clf: ControlLyapunovConstraint,
        u_min: Optional[Sequence[float]] = None,
        u_max: Optional[Sequence[float]] = None,
        conformal_margin: float = 0.0,
        max_iters: int = 500,
        tol: float = 1e-9,
        backup_controller: Optional[Any] = None,
    ):
        self.dynamics = dynamics
        self.barriers = list(barriers)
        self.clf = clf
        if float(clf.slack_weight) <= 0.0:
            raise ValueError("clf.slack_weight must be strictly positive.")
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=np.float64).flatten()
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=np.float64).flatten()
        self.conformal_margin = float(conformal_margin)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.backup_controller = backup_controller
        self._cbf_helper = StateCBFSafetyFilter(
            dynamics=self.dynamics,
            barriers=self.barriers,
            u_min=self.u_min,
            u_max=self.u_max,
            conformal_margin=self.conformal_margin,
            max_iters=self.max_iters,
            tol=self.tol,
            backup_controller=self.backup_controller,
        )
        self.calls = 0
        self.unsafe_calls = 0

    def set_conformal_margin(self, delta: float) -> None:
        """Inflate all barriers by the split-conformal band ``delta`` >= 0."""
        self.conformal_margin = max(0.0, float(delta))
        self._cbf_helper.set_conformal_margin(self.conformal_margin)

    def _clip_augmented_box(self, z_tilde: np.ndarray) -> np.ndarray:
        out = z_tilde.copy()
        d_u = self.dynamics.control_dim
        if self.u_min is not None:
            out[:d_u] = np.maximum(out[:d_u], self.u_min)
        if self.u_max is not None:
            out[:d_u] = np.minimum(out[:d_u], self.u_max)
        out[d_u] = max(0.0, float(out[d_u]))
        return out

    def _solve_scaled_dykstra(
        self,
        A_cbf: np.ndarray,
        b_cbf: np.ndarray,
        L_g_V: np.ndarray,
        clf_rhs: float,
        u_nom: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, bool]:
        """Run Dykstra in scaled coordinates ``[u, sqrt(w) * delta]``.

        ``clf_rhs = gamma * V(x) + L_f V(x)`` so the CLF inequality is
        ``-L_g V(x) u + delta >= clf_rhs``.
        """
        d_u = self.dynamics.control_dim
        r_w = float(np.sqrt(float(self.clf.slack_weight)))
        m = A_cbf.shape[0]

        A_aug = np.zeros((m + 1, d_u + 1), dtype=np.float64)
        b_aug = np.zeros(m + 1, dtype=np.float64)
        if m > 0:
            A_aug[:m, :d_u] = A_cbf
            b_aug[:m] = b_cbf
        A_aug[m, :d_u] = -L_g_V
        A_aug[m, d_u] = 1.0 / r_w
        b_aug[m] = clf_rhs

        z_nom = np.zeros(d_u + 1, dtype=np.float64)
        z_nom[:d_u] = u_nom

        n_hs = m + 1
        n_sets = n_hs + 1  # halfspaces + augmented box (includes delta >= 0)
        s_list = [np.zeros_like(z_nom) for _ in range(n_sets)]
        w = z_nom.copy()

        for _it in range(self.max_iters):
            converged = True
            for j in range(n_hs):
                y = w + s_list[j]
                a_row = A_aug[j]
                bj = b_aug[j]
                a_norm2 = float(a_row @ a_row)
                viol = bj - float(a_row @ y)
                if viol > 0.0 and a_norm2 > 1e-18:
                    proj = y + (viol / a_norm2) * a_row
                    converged = False
                else:
                    proj = y.copy()
                s_new = y - proj
                if not np.allclose(s_new, s_list[j], atol=self.tol):
                    converged = False
                s_list[j] = s_new
                w = proj

            # Box constraint on u and delta >= 0
            y_box = w + s_list[n_hs]
            proj_box = self._clip_augmented_box(y_box)
            if not np.allclose(proj_box, y_box, atol=self.tol):
                converged = False
            s_box_new = y_box - proj_box
            if not np.allclose(s_box_new, s_list[n_hs], atol=self.tol):
                converged = False
            s_list[n_hs] = s_box_new
            w = proj_box

            if converged:
                break

        u_star = w[:d_u].copy()
        delta_star = max(0.0, float(w[d_u] / r_w))

        cbf_slack = float(np.min(A_cbf @ u_star - b_cbf)) if A_cbf.size else 0.0
        clf_slack = float((-L_g_V @ u_star + delta_star) - clf_rhs)
        box_ok = (
            (self.u_min is None or bool(np.all(u_star >= self.u_min - 1e-6)))
            and (self.u_max is None or bool(np.all(u_star <= self.u_max + 1e-6)))
            and (w[d_u] >= -1e-6 * r_w)
        )
        verified = bool(
            np.all(np.isfinite(w))
            and box_ok
            and (not A_cbf.size or cbf_slack >= -1e-6)
            and (clf_slack >= -1e-6)
        )

        # If nearly anti-parallel CBF/CLF halfspaces slow down cyclic Dykstra,
        # refine with restarted dual FISTA on the exact same scaled QP.
        if not verified:
            w_fista = self._solve_dual_fista(A_aug, b_aug, z_nom)
            if w_fista is not None:
                u_f = w_fista[:d_u].copy()
                delta_f = max(0.0, float(w_fista[d_u] / r_w))
                cbf_s_f = float(np.min(A_cbf @ u_f - b_cbf)) if A_cbf.size else 0.0
                clf_s_f = float((-L_g_V @ u_f + delta_f) - clf_rhs)
                box_ok_f = (
                    (self.u_min is None or bool(np.all(u_f >= self.u_min - 1e-6)))
                    and (self.u_max is None or bool(np.all(u_f <= self.u_max + 1e-6)))
                    and (w_fista[d_u] >= -1e-6 * r_w)
                )
                verified_f = bool(
                    np.all(np.isfinite(w_fista))
                    and box_ok_f
                    and (not A_cbf.size or cbf_s_f >= -1e-6)
                    and (clf_s_f >= -1e-6)
                )
                if verified_f:
                    return u_f, delta_f, cbf_s_f, clf_s_f, True

        return u_star, delta_star, cbf_slack, clf_slack, verified

    def _solve_dual_fista(
        self,
        A_aug: np.ndarray,
        b_aug: np.ndarray,
        z_nom: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Restarted dual FISTA solver for ``min 0.5||z - z_nom||^2 s.t. C z >= d``."""
        d_z = z_nom.shape[0]
        d_u = self.dynamics.control_dim
        C_rows: List[np.ndarray] = [A_aug[i] for i in range(A_aug.shape[0])]
        d_vals: List[float] = [float(b_aug[i]) for i in range(b_aug.shape[0])]

        if self.u_min is not None:
            for k in range(d_u):
                row = np.zeros(d_z, dtype=np.float64)
                row[k] = 1.0
                C_rows.append(row)
                d_vals.append(float(self.u_min[k]))
        if self.u_max is not None:
            for k in range(d_u):
                row = np.zeros(d_z, dtype=np.float64)
                row[k] = -1.0
                C_rows.append(row)
                d_vals.append(-float(self.u_max[k]))

        # Non-negativity on scaled slack coordinate z_tilde[d_u] >= 0
        row_delta = np.zeros(d_z, dtype=np.float64)
        row_delta[d_u] = 1.0
        C_rows.append(row_delta)
        d_vals.append(0.0)

        C_mat = np.vstack(C_rows)
        d_vec = np.asarray(d_vals, dtype=np.float64)
        norms = np.linalg.norm(C_mat, axis=1)
        valid = norms > 1e-12
        if not np.all(valid):
            if np.any(d_vec[~valid] > 1e-9):
                return None
            C_mat = C_mat[valid]
            d_vec = d_vec[valid]
            norms = norms[valid]
        if C_mat.shape[0] == 0:
            return z_nom.copy()

        Cn = C_mat / norms[:, None]
        dn = d_vec / norms
        L = float(np.linalg.norm(Cn, ord=2) ** 2)
        if L <= 1e-18 or not np.isfinite(L):
            return None

        lam = np.zeros(Cn.shape[0], dtype=np.float64)
        y = np.zeros(Cn.shape[0], dtype=np.float64)
        t = 1.0
        max_fista = max(1500, 4 * self.max_iters)
        dual_tol = min(self.tol, 1e-10)
        for it in range(max_fista):
            z_y = z_nom + Cn.T @ y
            grad = Cn @ z_y - dn
            lam_new = np.maximum(0.0, y - grad / L)
            if float(np.dot(y - lam_new, lam_new - lam)) > 0.0:
                t = 1.0
                y = lam_new.copy()
            else:
                t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                y = lam_new + ((t - 1.0) / t_new) * (lam_new - lam)
                t = t_new
            if it > 5 and float(np.max(np.abs(lam_new - lam))) < dual_tol:
                lam = lam_new
                break
            lam = lam_new

        z_sol = z_nom + Cn.T @ lam
        return z_sol if np.all(np.isfinite(z_sol)) else None

    def filter(self, x: np.ndarray, u_nominal: Sequence[float]) -> Dict[str, Any]:
        """Compute joint CLF-CBF-QP control and relaxation slack ``delta``."""
        t0 = time.perf_counter()
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        u_arr = np.asarray(u_nominal, dtype=np.float64).flatten()
        u_nom_list = u_nominal if isinstance(u_nominal, list) else u_arr.tolist()

        if u_arr.shape[0] != self.dynamics.control_dim:
            return {
                "status": "dim_mismatch",
                "verified_safe": False,
                "u_filtered": u_arr.tolist(),
                "u_nominal": u_nom_list,
                "clf_slack_delta": 0.0,
                "V_value": 0.0,
            }
        self.calls += 1

        if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(u_arr))):
            self.unsafe_calls += 1
            return {
                "status": "invalid_nonfinite_input",
                "verified_safe": False,
                "u_filtered": u_arr.tolist(),
                "u_nominal": u_nom_list,
                "clf_slack_delta": 0.0,
                "V_value": float("nan"),
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
                "u_filtered": u_arr.tolist(),
                "u_nominal": u_nom_list,
                "clf_slack_delta": 0.0,
                "V_value": float(self.clf.V(x_arr)),
                "min_barrier_slack": -1.0,
                "conformal_margin": self.conformal_margin,
                "n_barriers": len(self.barriers),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        V_val = float(self.clf.V(x_arr))
        grad_V = np.asarray(self.clf.grad_V(x_arr), dtype=np.float64).flatten()
        f_x = self.dynamics.f(x_arr)
        G_x = self.dynamics.g(x_arr)
        L_f_V = float(np.dot(grad_V, f_x))
        L_g_V = np.asarray(grad_V @ G_x, dtype=np.float64).flatten()
        clf_rhs = float(self.clf.gamma) * V_val + L_f_V

        A_cbf, b_cbf, _alphas, hard_flags = self._cbf_helper._halfspaces_with_flags(x_arr)
        if (
            not np.isfinite(V_val)
            or not np.all(np.isfinite(L_g_V))
            or not np.isfinite(clf_rhs)
            or (A_cbf.size and not np.all(np.isfinite(A_cbf)))
            or (b_cbf.size and not np.all(np.isfinite(b_cbf)))
        ):
            self.unsafe_calls += 1
            return {
                "status": "invalid_nonfinite_input",
                "verified_safe": False,
                "u_filtered": u_arr.tolist(),
                "u_nominal": u_nom_list,
                "clf_slack_delta": 0.0,
                "V_value": V_val,
                "min_barrier_slack": -1.0,
                "conformal_margin": self.conformal_margin,
                "n_barriers": A_cbf.shape[0] if A_cbf.ndim == 2 else len(self.barriers),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        m = A_cbf.shape[0]
        u_star, delta_star, cbf_slack, _clf_slack, verified = self._solve_scaled_dykstra(
            A_cbf, b_cbf, L_g_V, clf_rhs, u_arr
        )
        if verified:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "safe",
                "verified_safe": True,
                "u_filtered": u_star.tolist(),
                "u_nominal": u_nom_list,
                "clf_slack_delta": float(delta_star),
                "V_value": V_val,
                "V_dot_bound": float(L_f_V + L_g_V @ u_star),
                "min_barrier_slack": round(cbf_slack, 9),
                "conformal_margin": self.conformal_margin,
                "n_barriers": m,
                "latency_ms": round(latency_ms, 4),
            }

        # Stage A: relax soft CBF barriers when mixed with hard barriers
        hard_mask = np.asarray(hard_flags, dtype=bool)
        n_hard = int(np.sum(hard_mask))
        n_soft = m - n_hard
        if n_soft > 0 and n_hard > 0:
            u_h, delta_h, slack_h, _clf_h, ok_h = self._solve_scaled_dykstra(
                A_cbf[hard_mask], b_cbf[hard_mask], L_g_V, clf_rhs, u_arr
            )
            if ok_h:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "status": "safe_soft_relaxed",
                    "verified_safe": True,
                    "u_filtered": u_h.tolist(),
                    "u_nominal": u_nom_list,
                    "clf_slack_delta": float(delta_h),
                    "V_value": V_val,
                    "V_dot_bound": float(L_f_V + L_g_V @ u_h),
                    "min_barrier_slack": round(slack_h, 9),
                    "conformal_margin": self.conformal_margin,
                    "n_barriers": m,
                    "relaxed_soft_barriers": n_soft,
                    "latency_ms": round(latency_ms, 4),
                }

        # Stage B: Simplex backup fallback
        self.unsafe_calls += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if self.backup_controller is not None:
            u_safe = np.asarray(self.backup_controller.compute_safe_action(x_arr), dtype=np.float64).flatten()
            u_safe = self._cbf_helper._clip_box(u_safe)
            backup_slack = float(np.min(A_cbf @ u_safe - b_cbf)) if A_cbf.size else 0.0
            delta_backup = max(0.0, float(L_f_V + L_g_V @ u_safe + float(self.clf.gamma) * V_val))
            return {
                "status": "simplex_braking_engaged",
                "verified_safe": False,
                "backup_engaged": True,
                "u_filtered": u_safe.tolist(),
                "u_nominal": u_nom_list,
                "clf_slack_delta": delta_backup,
                "V_value": V_val,
                "V_dot_bound": float(L_f_V + L_g_V @ u_safe),
                "min_barrier_slack": round(backup_slack, 9),
                "conformal_margin": self.conformal_margin,
                "n_barriers": m,
                "latency_ms": round(latency_ms, 4),
            }

        return {
            "status": "unsafe_no_solution",
            "verified_safe": False,
            "u_filtered": u_star.tolist(),
            "u_nominal": u_nom_list,
            "clf_slack_delta": float(delta_star),
            "V_value": V_val,
            "V_dot_bound": float(L_f_V + L_g_V @ u_star),
            "min_barrier_slack": round(cbf_slack, 9),
            "conformal_margin": self.conformal_margin,
            "n_barriers": m,
            "latency_ms": round(latency_ms, 4),
        }


class DifferentiableCBFProjection:
    """Implicit KKT active-set differentiation for ``StateCBFSafetyFilter``.

    Computes exact first-order sensitivity Jacobians of the projected control
    ``u* = argmin_{u in C(x, alpha)} 0.5 * ||u - u_nom||^2`` with respect to
    ``u_nom``, state ``x``, and barrier decay rates ``alpha``, plus VJP backward
    propagation for end-to-end differentiable safe policy learning.
    """

    @staticmethod
    def _prune_linearly_independent(
        candidates: List[Tuple[str, int, np.ndarray]],
        d_u: int,
        tol_orth: float = 1e-8,
    ) -> List[Tuple[str, int, np.ndarray]]:
        """Select a maximal linearly independent subset via reorthogonalized Gram-Schmidt."""
        kept: List[Tuple[str, int, np.ndarray]] = []
        basis: List[np.ndarray] = []
        for kind, idx, row in candidates:
            r_norm = float(np.linalg.norm(row))
            if r_norm <= 1e-12:
                continue
            v = row.copy()
            # Two-pass modified Gram-Schmidt for numerical stability
            for _ in range(2):
                for q in basis:
                    v = v - float(np.dot(v, q)) * q
            v_norm = float(np.linalg.norm(v))
            if v_norm > tol_orth * r_norm and v_norm > 1e-10:
                kept.append((kind, idx, row))
                basis.append(v / v_norm)
                if len(kept) == d_u:
                    break
        return kept

    @staticmethod
    def _select_active_kkt_rows(
        candidates: List[Tuple[str, int, np.ndarray]],
        u_star: np.ndarray,
        u_nom: np.ndarray,
        d_u: int,
    ) -> Tuple[List[Tuple[str, int, np.ndarray]], np.ndarray]:
        """Prune dependent rows and drop any spurious negative-multiplier rows."""
        pool = list(candidates)
        while pool:
            kept = DifferentiableCBFProjection._prune_linearly_independent(pool, d_u)
            if not kept:
                return [], np.zeros(0, dtype=np.float64)
            M_A = np.vstack([row for _, _, row in kept])
            gram = M_A @ M_A.T
            rhs = M_A @ (u_star - u_nom)
            try:
                lam = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                lam, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)
            min_idx = int(np.argmin(lam))
            if lam[min_idx] >= -1e-6:
                return kept, lam
            # Drop the spurious candidate row with the most negative dual variable
            drop_item = kept[min_idx]
            pool = [c for c in pool if not (c[0] == drop_item[0] and c[1] == drop_item[1])]
        return [], np.zeros(0, dtype=np.float64)

    @staticmethod
    def _barrier_alpha_derivatives(
        safety_filter: StateCBFSafetyFilter,
        x: np.ndarray,
    ) -> np.ndarray:
        """Compute exact partial derivatives db_i / d(alpha_i) for each barrier i."""
        m = len(safety_filter.barriers)
        db_dalpha = np.zeros(m, dtype=np.float64)
        if m == 0:
            return db_dalpha
        f_x: Optional[np.ndarray] = None
        for i, bar in enumerate(safety_filter.barriers):
            eff_bar = (
                bar.inflate(safety_filter.conformal_margin)
                if safety_filter.conformal_margin > 0.0
                else bar
            )
            h_eff = float(eff_bar.h(x))
            rel_deg = int(getattr(eff_bar, "relative_degree", getattr(bar, "relative_degree", 1)))
            if rel_deg == 2:
                if f_x is None:
                    f_x = safety_filter.dynamics.f(x)
                gh = np.asarray(eff_bar.grad_h(x), dtype=np.float64).flatten()
                L_f_h = float(np.dot(gh, f_x))
                dh_dt_val = float(eff_bar.dh_dt(x))
                alpha2 = float(getattr(eff_bar, "alpha2", getattr(bar, "alpha2", 1.0)))
                db_dalpha[i] = -(L_f_h + dh_dt_val + alpha2 * h_eff)
            else:
                db_dalpha[i] = -h_eff
        return db_dalpha

    @staticmethod
    def _halfspace_state_jacobians(
        safety_filter: StateCBFSafetyFilter,
        x: np.ndarray,
        m: int,
        d_u: int,
        eps_fd: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Central finite-difference Jacobians (dA/dx, db/dx) of CBF halfspaces."""
        d_x = x.shape[0]
        dA_dx = np.zeros((m, d_u, d_x), dtype=np.float64)
        db_dx = np.zeros((m, d_x), dtype=np.float64)
        if m == 0:
            return dA_dx, db_dx
        for k in range(d_x):
            step = np.zeros(d_x, dtype=np.float64)
            step[k] = eps_fd
            A_p, b_p, _, _ = safety_filter._halfspaces_with_flags(x + step)
            A_m, b_m, _, _ = safety_filter._halfspaces_with_flags(x - step)
            dA_dx[:, :, k] = (A_p - A_m) / (2.0 * eps_fd)
            db_dx[:, k] = (b_p - b_m) / (2.0 * eps_fd)
        return dA_dx, db_dx

    @staticmethod
    def forward_and_jacobians(
        x: Sequence[float],
        u_nom: Sequence[float],
        safety_filter: StateCBFSafetyFilter,
        eps_active: float = 1e-6,
        eps_fd: float = 1e-6,
    ) -> Dict[str, Any]:
        """Solve the CBF-QP via ``safety_filter`` and compute implicit KKT Jacobians.

        Returns a dict containing ``u_star``, ``du_du_nom``, ``du_dx``, ``du_dalpha``,
        plus active-set diagnostics suitable as ``jac_cache`` for ``vjp(...)``.
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        u_arr = np.asarray(u_nom, dtype=np.float64).flatten()
        d_x = x_arr.shape[0]
        d_u = safety_filter.dynamics.control_dim
        m = len(safety_filter.barriers)

        out = safety_filter.filter(x_arr, u_arr)
        u_star = np.asarray(out["u_filtered"], dtype=np.float64).flatten()
        verified = bool(out.get("verified_safe", False))
        status = str(out.get("status", "unsafe_no_solution"))

        if not verified or u_arr.shape[0] != d_u:
            return {
                "u_star": u_star,
                "du_du_nom": np.zeros((d_u, d_u), dtype=np.float64),
                "du_dx": np.zeros((d_u, d_x), dtype=np.float64),
                "du_dalpha": np.zeros((d_u, m), dtype=np.float64),
                "active_cbf_indices": [],
                "active_box_lower": [],
                "active_box_upper": [],
                "n_active": 0,
                "verified_safe": verified,
                "status": status,
            }

        A_all, b_all, _alphas, hard_flags = safety_filter._halfspaces_with_flags(x_arr)
        if status == "safe_soft_relaxed":
            cbf_pool = [i for i, is_hard in enumerate(hard_flags) if is_hard]
        else:
            cbf_pool = list(range(m))

        candidates: List[Tuple[str, int, np.ndarray]] = []
        for i in cbf_pool:
            res_i = float(A_all[i] @ u_star - b_all[i])
            if abs(res_i) <= eps_active and float(np.linalg.norm(A_all[i])) > 1e-12:
                candidates.append(("cbf", i, A_all[i].copy()))

        if safety_filter.u_min is not None:
            for k in range(d_u):
                if abs(u_star[k] - float(safety_filter.u_min[k])) <= eps_active:
                    row = np.zeros(d_u, dtype=np.float64)
                    row[k] = 1.0
                    candidates.append(("box_lo", k, row))

        if safety_filter.u_max is not None:
            for k in range(d_u):
                if abs(float(safety_filter.u_max[k]) - u_star[k]) <= eps_active:
                    row = np.zeros(d_u, dtype=np.float64)
                    row[k] = -1.0
                    candidates.append(("box_hi", k, row))

        kept, lam = DifferentiableCBFProjection._select_active_kkt_rows(
            candidates, u_star, u_arr, d_u
        )
        n_act = len(kept)

        if n_act == 0:
            return {
                "u_star": u_star,
                "du_du_nom": np.eye(d_u, dtype=np.float64),
                "du_dx": np.zeros((d_u, d_x), dtype=np.float64),
                "du_dalpha": np.zeros((d_u, m), dtype=np.float64),
                "active_cbf_indices": [],
                "active_box_lower": [],
                "active_box_upper": [],
                "n_active": 0,
                "verified_safe": True,
                "status": status,
            }

        M_A = np.vstack([row for _, _, row in kept])
        # Build non-singular KKT matrix [[I, -M_A^T], [M_A, 0]]
        K = np.zeros((d_u + n_act, d_u + n_act), dtype=np.float64)
        K[:d_u, :d_u] = np.eye(d_u, dtype=np.float64)
        K[:d_u, d_u:] = -M_A.T
        K[d_u:, :d_u] = M_A

        def _solve_kkt(rhs: np.ndarray) -> np.ndarray:
            try:
                return np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                sol, _, _, _ = np.linalg.lstsq(K, rhs, rcond=None)
                return sol

        # 1. du* / du_nom
        rhs_u = np.zeros((d_u + n_act, d_u), dtype=np.float64)
        rhs_u[:d_u, :] = np.eye(d_u, dtype=np.float64)
        du_du_nom = _solve_kkt(rhs_u)[:d_u, :]

        # 2. du* / dx (includes both active CBF halfspace shifts/rotations and active box bounds)
        cbf_kept = [(j, idx) for j, (kind, idx, _) in enumerate(kept) if kind == "cbf"]
        if cbf_kept:
            dA_dx, db_dx = DifferentiableCBFProjection._halfspace_state_jacobians(
                safety_filter, x_arr, m, d_u, eps_fd=eps_fd
            )
            rhs_x = np.zeros((d_u + n_act, d_x), dtype=np.float64)
            for j, idx in cbf_kept:
                # Top block: sum_j lambda_j * (dA_i / dx)^T
                rhs_x[:d_u, :] += float(lam[j]) * dA_dx[idx, :, :]
                # Bottom block: db_i/dx - u*^T (dA_i / dx)
                rhs_x[d_u + j, :] = db_dx[idx, :] - (u_star @ dA_dx[idx, :, :])
            du_dx = _solve_kkt(rhs_x)[:d_u, :]
        else:
            du_dx = np.zeros((d_u, d_x), dtype=np.float64)

        # 3. du* / dalpha
        if m > 0 and cbf_kept:
            db_dalpha = DifferentiableCBFProjection._barrier_alpha_derivatives(safety_filter, x_arr)
            rhs_alpha = np.zeros((d_u + n_act, m), dtype=np.float64)
            for j, idx in cbf_kept:
                rhs_alpha[d_u + j, idx] = db_dalpha[idx]
            du_dalpha = _solve_kkt(rhs_alpha)[:d_u, :]
        else:
            du_dalpha = np.zeros((d_u, m), dtype=np.float64)

        return {
            "u_star": u_star,
            "du_du_nom": du_du_nom,
            "du_dx": du_dx,
            "du_dalpha": du_dalpha,
            "active_cbf_indices": [idx for kind, idx, _ in kept if kind == "cbf"],
            "active_box_lower": [idx for kind, idx, _ in kept if kind == "box_lo"],
            "active_box_upper": [idx for kind, idx, _ in kept if kind == "box_hi"],
            "n_active": n_act,
            "verified_safe": True,
            "status": status,
        }

    @staticmethod
    def vjp(
        grad_u_star: Sequence[float],
        jac_cache: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backpropagate upstream gradient ``dL/du*`` to ``(grad_u_nom, grad_x, grad_alpha)``."""
        g = np.asarray(grad_u_star, dtype=np.float64).flatten()
        grad_u_nom = np.asarray(jac_cache["du_du_nom"].T @ g, dtype=np.float64).flatten()
        grad_x = np.asarray(jac_cache["du_dx"].T @ g, dtype=np.float64).flatten()
        grad_alpha = np.asarray(jac_cache["du_dalpha"].T @ g, dtype=np.float64).flatten()
        return grad_u_nom, grad_x, grad_alpha
