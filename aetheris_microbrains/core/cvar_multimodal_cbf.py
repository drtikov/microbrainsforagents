"""Distributionally robust Conditional Value-at-Risk (CVaR_beta) CBFs for multimodal crowds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .state_cbf_filter import BarrierConstraint, ControlAffineDynamics


@dataclass
class PedestrianHypothesis:
    hypothesis_id: str
    weight: float
    pos: Sequence[float]
    vel: Sequence[float]
    radius: float = 0.4
    alpha: float = 1.0
    uncertainty_margin: float = 0.0


class MultimodalCVaRCBF:
    """Rockafellar-Uryasev linear polytope CVaR_beta CBF over multimodal human intents."""

    def __init__(
        self,
        dynamics: ControlAffineDynamics,
        beta: float = 0.15,
        u_min: Optional[Sequence[float]] = None,
        u_max: Optional[Sequence[float]] = None,
        pos_slice: slice = slice(0, 2),
        max_iters: int = 2000,
        tol: float = 1e-8,
    ):
        self.dynamics = dynamics
        self.beta = float(np.clip(beta, 1e-3, 1.0))
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=np.float64).flatten()
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=np.float64).flatten()
        self.pos_slice = pos_slice
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def filter(
        self,
        x: Sequence[float],
        u_nominal: Sequence[float],
        hypotheses: Sequence[PedestrianHypothesis],
        extra_barriers: Sequence[BarrierConstraint] = (),
    ) -> Dict[str, Any]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        u_nom = np.asarray(u_nominal, dtype=np.float64).flatten()
        d_u = self.dynamics.control_dim
        M = len(hypotheses)
        if M == 0:
            u_out = u_nom.copy()
            if self.u_min is not None:
                u_out = np.maximum(u_out, self.u_min)
            if self.u_max is not None:
                u_out = np.minimum(u_out, self.u_max)
            return {"u_filtered": u_out.tolist(), "cvar_value": 0.0, "var_zeta": 0.0, "verified_safe": True, "status": "safe"}

        weights = np.array([max(0.0, float(h.weight)) for h in hypotheses], dtype=np.float64)
        w_sum = float(np.sum(weights))
        weights = weights / w_sum if w_sum > 0.0 else np.full(M, 1.0 / M)

        f_x = self.dynamics.f(x_arr)
        G_x = self.dynamics.g(x_arr)
        p_robot = x_arr[self.pos_slice]

        # Augmented z = [u (d_u), zeta (1), s_1..s_M (M)]
        dim_z = d_u + 1 + M
        rows_A: List[np.ndarray] = []
        rows_b: List[float] = []

        # 1. CVaR constraint: -zeta - sum_m (w_m / beta) s_m >= 0
        cvar_row = np.zeros(dim_z, dtype=np.float64)
        cvar_row[d_u] = -1.0
        cvar_row[d_u + 1 :] = -weights / self.beta
        rows_A.append(cvar_row)
        rows_b.append(0.0)

        # 2. Epigraph constraints: s_m + zeta - l_m(x, u) >= 0
        # l_m(x, u) = -(L_f h_m + dh_m/dt + L_g h_m u + alpha_m h_m)
        l_g_list: List[np.ndarray] = []
        l_const_list: List[float] = []
        for m_idx, hyp in enumerate(hypotheses):
            p_obs = np.asarray(hyp.pos, dtype=np.float64).flatten()[: len(p_robot)]
            v_obs = np.asarray(hyp.vel, dtype=np.float64).flatten()[: len(p_robot)]
            diff = p_robot - p_obs
            r_eff = float(hyp.radius) + float(hyp.uncertainty_margin)
            h_m = float(np.dot(diff, diff) - r_eff * r_eff)
            gh = np.zeros_like(x_arr)
            gh[self.pos_slice] = 2.0 * diff
            L_f_h = float(np.dot(gh, f_x))
            dh_dt = float(-2.0 * np.dot(diff, v_obs))
            L_g_h = gh @ G_x
            c_m = -(L_f_h + dh_dt + float(hyp.alpha) * h_m)
            l_g_list.append(-L_g_h)
            l_const_list.append(c_m)

            row = np.zeros(dim_z, dtype=np.float64)
            row[:d_u] = L_g_h
            row[d_u] = 1.0
            row[d_u + 1 + m_idx] = 1.0
            rows_A.append(row)
            rows_b.append(c_m)

        for bar in extra_barriers:
            gh = np.asarray(bar.grad_h(x_arr), dtype=np.float64).flatten()
            row = np.zeros(dim_z, dtype=np.float64)
            row[:d_u] = gh @ G_x
            rows_A.append(row)
            rows_b.append(-float(getattr(bar, "alpha", 0.5)) * float(bar.h(x_arr)) - float(np.dot(gh, f_x)))

        A_mat = np.vstack(rows_A)
        b_vec = np.asarray(rows_b, dtype=np.float64)

        z_nom = np.zeros(dim_z, dtype=np.float64)
        z_nom[:d_u] = u_nom

        # Dykstra projection with box: u_min <= u <= u_max, s_m >= 0
        m_hs = A_mat.shape[0]
        s_list = [np.zeros(dim_z, dtype=np.float64) for _ in range(m_hs + 1)]
        w = z_nom.copy()
        for _ in range(self.max_iters):
            conv = True
            for i in range(m_hs):
                y = w + s_list[i]
                ai = A_mat[i]
                an2 = float(ai @ ai)
                viol = float(b_vec[i]) - float(ai @ y)
                if viol > 0.0 and an2 > 1e-18:
                    proj = y + (viol / an2) * ai
                    conv = False
                else:
                    proj = y.copy()
                s_new = y - proj
                if not np.allclose(s_new, s_list[i], atol=self.tol):
                    conv = False
                s_list[i] = s_new
                w = proj
            y = w + s_list[m_hs]
            proj = y.copy()
            if self.u_min is not None:
                proj[:d_u] = np.maximum(proj[:d_u], self.u_min)
            if self.u_max is not None:
                proj[:d_u] = np.minimum(proj[:d_u], self.u_max)
            proj[d_u + 1 :] = np.maximum(0.0, proj[d_u + 1 :])
            if not np.allclose(proj, y, atol=self.tol):
                conv = False
            s_list[m_hs] = y - proj
            w = proj
            if conv:
                break

        u_out = w[:d_u]
        zeta = float(w[d_u])
        losses = [float(np.dot(l_g_list[m], u_out) + l_const_list[m]) for m in range(M)]
        cvar_val = float(zeta + np.sum(weights * np.maximum(0.0, np.asarray(losses) - zeta)) / self.beta)
        min_slack = float(np.min(A_mat @ w - b_vec))
        verified = bool(np.all(np.isfinite(u_out)) and min_slack >= -1e-5 and cvar_val <= 1e-5)
        return {
            "u_filtered": u_out.tolist(),
            "cvar_value": cvar_val,
            "var_zeta": zeta,
            "hypothesis_losses": losses,
            "verified_safe": verified,
            "status": "safe" if verified else "unsafe_no_solution",
        }
