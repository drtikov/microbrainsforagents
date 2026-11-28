"""Latency-aware action-chunk smoothing and barrier guidance for VLA robotics.

Provides:
  - ``rk4_step``: 4th-order Runge-Kutta integration for ``ControlAffineDynamics``.
  - ``logsumexp_barrier_value_and_grad``: smooth Log-Sum-Exp aggregate barrier
    and exact spatial gradient for diffusion / flow-matching action guidance.
  - ``ActionChunkSmoother``:
      * ``align_and_blend_chunk``: inference-latency prefix trimming and
        exponential head-to-tail blending across consecutive action chunks.
      * ``smooth_chunk``: joint H-step trajectory optimization balancing
        nominal tracking, velocity continuity, and jerk regularization subject
        to RK4-rolled discrete CBF halfspace + box constraints via bounded
        ADMM (Block-Dykstra projection + O(H * d_u) banded Thomas solver).
      * ``guidance_step``: single barrier-guidance gradient ascent step on an
        action chunk along grad_U sum_k h_LSE(x_k).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .state_cbf_filter import BarrierConstraint, ControlAffineDynamics, StateCBFSafetyFilter


def rk4_step(
    dynamics: ControlAffineDynamics,
    x: Sequence[float],
    u: Sequence[float],
    dt: float,
) -> np.ndarray:
    """Integrate control-affine dynamics xdot = f(x) + g(x) u over dt via RK4."""
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64)).flatten()
    dt_f = float(dt)

    def _rhs(state: np.ndarray) -> np.ndarray:
        return np.asarray(dynamics.f(state), dtype=np.float64).flatten() + (
            np.asarray(dynamics.g(state), dtype=np.float64) @ u_arr
        )

    k1 = _rhs(x_arr)
    k2 = _rhs(x_arr + 0.5 * dt_f * k1)
    k3 = _rhs(x_arr + 0.5 * dt_f * k2)
    k4 = _rhs(x_arr + dt_f * k3)
    return x_arr + (dt_f / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def logsumexp_barrier_value_and_grad(
    barriers: Sequence[BarrierConstraint],
    x: Sequence[float],
    beta: float = 10.0,
    conformal_margin: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Compute smooth Log-Sum-Exp aggregate barrier value and exact gradient.

    h_LSE(x) = -(1 / beta) * ln sum_i exp(-beta * h_eff_i(x)),
    where h_eff_i(x) = h_i(x) - conformal_margin.
    """
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    if not barriers:
        return 0.0, np.zeros_like(x_arr, dtype=np.float64)

    beta_f = max(1e-6, float(beta))
    margin_f = float(conformal_margin)

    h_effs = np.array([float(b.h(x_arr)) - margin_f for b in barriers], dtype=np.float64)
    grads = np.vstack([np.asarray(b.grad_h(x_arr), dtype=np.float64).flatten() for b in barriers])

    z = -beta_f * h_effs
    z_max = float(np.max(z))
    exp_shifted = np.exp(z - z_max)
    sum_exp = float(np.sum(exp_shifted))
    weights = exp_shifted / sum_exp

    h_lse = -(z_max + math.log(sum_exp)) / beta_f
    grad_lse = weights @ grads
    return float(h_lse), grad_lse


def _clip_box(
    u: np.ndarray,
    u_min: Optional[np.ndarray],
    u_max: Optional[np.ndarray],
) -> np.ndarray:
    out = u
    if u_min is not None:
        out = np.maximum(out, u_min)
    if u_max is not None:
        out = np.minimum(out, u_max)
    return out


def _project_polytope_dykstra(
    u_target: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    u_min: Optional[np.ndarray],
    u_max: Optional[np.ndarray],
    max_iters: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Euclidean projection onto {u : A u >= b, u_min <= u <= u_max} via Dykstra."""
    m = A.shape[0]
    has_box = u_min is not None or u_max is not None
    if m == 0 and not has_box:
        return u_target.copy()
    if m == 0:
        return _clip_box(u_target, u_min, u_max)

    n_sets = m + (1 if has_box else 0)
    s_list = np.zeros((n_sets, u_target.shape[0]), dtype=np.float64)
    w = u_target.copy()

    for _ in range(max_iters):
        converged = True
        for i in range(m):
            y = w + s_list[i]
            a_row = A[i]
            a_norm2 = float(a_row @ a_row)
            viol = float(b[i]) - float(a_row @ y)
            if viol > 0.0 and a_norm2 > 1e-18:
                proj = y + (viol / a_norm2) * a_row
                converged = False
            else:
                proj = y
            s_new = y - proj
            if not np.allclose(s_new, s_list[i], atol=tol, rtol=0.0):
                converged = False
            s_list[i] = s_new
            w = proj

        if has_box:
            idx = m
            y = w + s_list[idx]
            proj = _clip_box(y, u_min, u_max)
            if not np.allclose(proj, y, atol=tol, rtol=0.0):
                converged = False
            s_new = y - proj
            if not np.allclose(s_new, s_list[idx], atol=tol, rtol=0.0):
                converged = False
            s_list[idx] = s_new
            w = proj

        if converged:
            break

    return w


def _compute_jerk_l2(
    actions: np.ndarray,
    prev_action: Optional[np.ndarray] = None,
) -> float:
    """Compute L2 norm of discrete second-difference (jerk) across an action chunk."""
    horizon = actions.shape[0]
    diffs: List[np.ndarray] = []
    if prev_action is not None and horizon >= 2:
        diffs.append(actions[1] - 2.0 * actions[0] + prev_action)
    for k in range(2, horizon):
        diffs.append(actions[k] - 2.0 * actions[k - 1] + actions[k - 2])
    if not diffs:
        return 0.0
    stacked = np.vstack(diffs)
    return float(np.sqrt(np.sum(stacked * stacked)))


class _BandedSPDSolver:
    """Exact O(H * d_u) LDL^T Thomas solver for symmetric banded (bandwidth <= 2) SPD systems."""

    def __init__(self, diag: np.ndarray, off1: np.ndarray, off2: np.ndarray):
        n = diag.shape[0]
        self.n = n
        self.d = np.zeros(n, dtype=np.float64)
        self.l1 = np.zeros(max(0, n - 1), dtype=np.float64)
        self.l2 = np.zeros(max(0, n - 2), dtype=np.float64)

        for i in range(n):
            di = float(diag[i])
            if i >= 1:
                di -= self.d[i - 1] * (self.l1[i - 1] ** 2)
            if i >= 2:
                di -= self.d[i - 2] * (self.l2[i - 2] ** 2)
            self.d[i] = max(di, 1e-12)

            if i + 1 < n:
                s1 = float(off1[i])
                if i >= 1:
                    s1 -= self.d[i - 1] * self.l1[i - 1] * self.l2[i - 1]
                self.l1[i] = s1 / self.d[i]
            if i + 2 < n:
                self.l2[i] = float(off2[i]) / self.d[i]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve M U = rhs in O(H * d_u) vectorized across control dimensions."""
        n = self.n
        if n == 0:
            return rhs.copy()

        y = np.empty_like(rhs, dtype=np.float64)
        for i in range(n):
            val = rhs[i].copy()
            if i >= 1:
                val -= self.l1[i - 1] * y[i - 1]
            if i >= 2:
                val -= self.l2[i - 2] * y[i - 2]
            y[i] = val

        u = y / self.d[:, None]
        for i in range(n - 2, -1, -1):
            u[i] -= self.l1[i] * u[i + 1]
            if i + 2 < n:
                u[i] -= self.l2[i] * u[i + 2]
        return u


class ActionChunkSmoother:
    """Smooths VLA action chunks under RK4-rolled CBF safety constraints."""

    def __init__(
        self,
        safety_filter: StateCBFSafetyFilter,
        velocity_weight: float = 0.25,
        jerk_weight: float = 0.1,
        max_iters: int = 30,
    ):
        self.safety_filter = safety_filter
        self.velocity_weight = max(0.0, float(velocity_weight))
        self.jerk_weight = max(0.0, float(jerk_weight))
        self.max_iters = max(1, int(max_iters))

    def align_and_blend_chunk(
        self,
        new_chunk: Sequence[Sequence[float]],
        prev_chunk: Optional[Sequence[Sequence[float]]] = None,
        elapsed_steps: int = 0,
        blend_steps: int = 4,
        alpha_decay: float = 0.5,
    ) -> np.ndarray:
        """Drop elapsed_steps prefix for latency alignment and exponentially blend head."""
        new_arr = np.asarray(new_chunk, dtype=np.float64)
        if new_arr.ndim == 1:
            new_arr = new_arr[:, None] if new_arr.size > 0 else new_arr.reshape(0, 1)

        k_drop = max(0, int(elapsed_steps))
        aligned = new_arr[k_drop:].copy()
        if aligned.shape[0] == 0 or prev_chunk is None:
            return aligned

        n_blend = max(0, int(blend_steps))
        if n_blend == 0:
            return aligned

        prev_arr = np.asarray(prev_chunk, dtype=np.float64)
        if prev_arr.ndim == 1:
            prev_arr = prev_arr[:, None] if prev_arr.size > 0 else prev_arr.reshape(0, 1)

        prev_tail = prev_arr[k_drop:]
        if prev_tail.shape[0] == 0:
            return aligned

        decay = float(np.clip(alpha_decay, 0.0, 1.0))
        steps = min(n_blend, aligned.shape[0], prev_tail.shape[0])
        for i in range(steps):
            w_prev = decay ** (i + 1)
            aligned[i] = w_prev * prev_tail[i] + (1.0 - w_prev) * aligned[i]
        return aligned

    def _build_banded_system(
        self,
        u_nom: np.ndarray,
        rho: float,
        prev_action: Optional[np.ndarray],
    ) -> Tuple[_BandedSPDSolver, np.ndarray]:
        horizon, _d_u = u_nom.shape
        lv = self.velocity_weight
        lj = self.jerk_weight

        diag = np.full(horizon, 1.0 + rho, dtype=np.float64)
        off1 = np.zeros(max(0, horizon - 1), dtype=np.float64)
        off2 = np.zeros(max(0, horizon - 2), dtype=np.float64)
        rhs_const = u_nom.copy()

        if lv > 0.0:
            for k in range(1, horizon):
                diag[k] += lv
                diag[k - 1] += lv
                off1[k - 1] -= lv
            if prev_action is not None and horizon >= 1:
                diag[0] += lv
                rhs_const[0] += lv * prev_action

        if lj > 0.0:
            for k in range(2, horizon):
                diag[k - 2] += lj
                diag[k - 1] += 4.0 * lj
                diag[k] += lj
                off1[k - 2] -= 2.0 * lj
                off1[k - 1] -= 2.0 * lj
                off2[k - 2] += lj
            if prev_action is not None and horizon >= 2:
                diag[0] += 4.0 * lj
                diag[1] += lj
                off1[0] -= 2.0 * lj
                rhs_const[0] += 2.0 * lj * prev_action
                rhs_const[1] -= lj * prev_action

        return _BandedSPDSolver(diag, off1, off2), rhs_const

    def smooth_chunk(
        self,
        x0: Sequence[float],
        nominal_actions: Sequence[Sequence[float]],
        dt: float = 0.05,
        prev_action: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Optimize an H-step chunk for smoothness and RK4-rolled CBF safety."""
        x0_arr = np.asarray(x0, dtype=np.float64).flatten()
        u_nom = np.asarray(nominal_actions, dtype=np.float64)
        if u_nom.size == 0:
            return {
                "status": "chunk_smoothed",
                "actions_filtered": [],
                "all_verified_safe": True,
                "per_step": [],
                "final_state_prediction": x0_arr.tolist(),
                "nominal_jerk_l2": 0.0,
                "smoothed_jerk_l2": 0.0,
                "iterations": 0,
            }

        if u_nom.ndim == 1:
            u_nom = u_nom[:, None]
        horizon, d_u = u_nom.shape

        prev_arr: Optional[np.ndarray] = None
        if prev_action is not None:
            prev_arr = np.atleast_1d(np.asarray(prev_action, dtype=np.float64)).flatten()

        nominal_jerk = _compute_jerk_l2(u_nom, prev_arr)

        if (
            d_u != self.safety_filter.dynamics.control_dim
            or not np.all(np.isfinite(x0_arr))
            or not np.all(np.isfinite(u_nom))
            or (prev_arr is not None and (prev_arr.shape[0] != d_u or not np.all(np.isfinite(prev_arr))))
        ):
            return {
                "status": "invalid_input",
                "actions_filtered": u_nom.tolist(),
                "all_verified_safe": False,
                "per_step": [{"verified_safe": False, "status": "invalid_input"} for _ in range(horizon)],
                "final_state_prediction": x0_arr.tolist(),
                "nominal_jerk_l2": nominal_jerk,
                "smoothed_jerk_l2": nominal_jerk,
                "iterations": 0,
            }

        u_min = self.safety_filter.u_min
        u_max = self.safety_filter.u_max
        if u_min is not None and u_max is not None and np.any(u_min > u_max):
            return {
                "status": "infeasible_box",
                "actions_filtered": u_nom.tolist(),
                "all_verified_safe": False,
                "per_step": [{"verified_safe": False, "status": "infeasible_box"} for _ in range(horizon)],
                "final_state_prediction": x0_arr.tolist(),
                "nominal_jerk_l2": nominal_jerk,
                "smoothed_jerk_l2": nominal_jerk,
                "iterations": 0,
            }

        rho = max(1.0, 0.5 + self.velocity_weight + 2.0 * self.jerk_weight)
        solver, rhs_const = self._build_banded_system(u_nom, rho=rho, prev_action=prev_arr)

        z = _clip_box(u_nom.copy(), u_min, u_max)
        y = np.zeros_like(z, dtype=np.float64)
        dykstra_iters = max(40, min(self.safety_filter.max_iters, 120))
        iters_done = 0

        for it in range(self.max_iters):
            iters_done = it + 1
            z_prev = z.copy()

            # U-step: exact O(H * d_u) banded Thomas solve
            u = solver.solve(rhs_const + rho * (z - y))

            # Z-step: roll RK4 trajectory and project each step onto CBF polytope + box
            x_roll = x0_arr.copy()
            z_new = np.empty_like(z)
            for k in range(horizon):
                A_k, b_k, _ = self.safety_filter._halfspaces(x_roll)
                v_k = u[k] + y[k]
                if (A_k.size and not np.all(np.isfinite(A_k))) or (b_k.size and not np.all(np.isfinite(b_k))):
                    z_k = _clip_box(v_k, u_min, u_max)
                else:
                    z_k = _project_polytope_dykstra(
                        v_k,
                        A_k,
                        b_k,
                        u_min,
                        u_max,
                        max_iters=dykstra_iters,
                        tol=self.safety_filter.tol,
                    )
                z_new[k] = z_k
                x_roll = rk4_step(self.safety_filter.dynamics, x_roll, z_k, dt)

            z = z_new
            y = y + (u - z)

            primal_res = float(np.max(np.abs(u - z)))
            dual_res = float(np.max(np.abs(z - z_prev)))
            if primal_res < 1e-5 and dual_res < 1e-5:
                break

        # Final RK4 rollout and exact per-step CBF safety verification
        x_run = x0_arr.copy()
        filtered: List[List[float]] = []
        receipts: List[Dict[str, Any]] = []
        all_safe = True
        for k in range(horizon):
            out = self.safety_filter.filter(x_run, z[k])
            u_f = np.asarray(out["u_filtered"], dtype=np.float64)
            receipts.append({"verified_safe": out["verified_safe"], "status": out["status"]})
            all_safe = all_safe and bool(out["verified_safe"])
            filtered.append(u_f.tolist())
            x_run = rk4_step(self.safety_filter.dynamics, x_run, u_f, dt)

        filtered_arr = np.asarray(filtered, dtype=np.float64)
        smoothed_jerk = _compute_jerk_l2(filtered_arr, prev_arr)
        return {
            "status": "chunk_smoothed" if all_safe else "chunk_unsafe_steps_present",
            "actions_filtered": filtered,
            "all_verified_safe": all_safe,
            "per_step": receipts,
            "final_state_prediction": x_run.tolist(),
            "nominal_jerk_l2": nominal_jerk,
            "smoothed_jerk_l2": smoothed_jerk,
            "iterations": iters_done,
        }

    def guidance_step(
        self,
        x0: Sequence[float],
        chunk: Sequence[Sequence[float]],
        dt: float = 0.05,
        step_size: float = 0.05,
        beta: float = 10.0,
    ) -> np.ndarray:
        """Apply one barrier-guidance gradient ascent step along grad_U sum_k h_LSE(x_k)."""
        u_arr = np.asarray(chunk, dtype=np.float64).copy()
        if u_arr.size == 0:
            return u_arr
        was_1d = u_arr.ndim == 1
        if was_1d:
            u_arr = u_arr[:, None]

        horizon, d_u = u_arr.shape
        x0_arr = np.asarray(x0, dtype=np.float64).flatten()
        n_x = x0_arr.shape[0]
        dynamics = self.safety_filter.dynamics
        barriers = self.safety_filter.barriers
        margin = self.safety_filter.conformal_margin

        if not barriers:
            return u_arr.flatten() if was_1d else u_arr

        # Forward RK4 rollout storing states x_0..x_H
        states = [x0_arr]
        for k in range(horizon):
            states.append(rk4_step(dynamics, states[-1], u_arr[k], dt))

        # Reverse-mode adjoint pass through RK4 steps using central-difference Jacobians
        eps = 1e-6
        grad_u = np.zeros_like(u_arr, dtype=np.float64)
        _, lam = logsumexp_barrier_value_and_grad(barriers, states[horizon], beta=beta, conformal_margin=margin)

        for k in range(horizon - 1, -1, -1):
            xk = states[k]
            uk = u_arr[k]

            jac_u = np.empty((n_x, d_u), dtype=np.float64)
            for j in range(d_u):
                du = np.zeros(d_u, dtype=np.float64)
                du[j] = eps
                jac_u[:, j] = (rk4_step(dynamics, xk, uk + du, dt) - rk4_step(dynamics, xk, uk - du, dt)) / (2.0 * eps)

            grad_u[k] = jac_u.T @ lam

            if k > 0:
                jac_x = np.empty((n_x, n_x), dtype=np.float64)
                for j in range(n_x):
                    dx = np.zeros(n_x, dtype=np.float64)
                    dx[j] = eps
                    jac_x[:, j] = (rk4_step(dynamics, xk + dx, uk, dt) - rk4_step(dynamics, xk - dx, uk, dt)) / (2.0 * eps)
                _, grad_xk = logsumexp_barrier_value_and_grad(barriers, xk, beta=beta, conformal_margin=margin)
                lam = grad_xk + jac_x.T @ lam

        updated = u_arr + float(step_size) * grad_u
        return updated.flatten() if was_1d else updated
