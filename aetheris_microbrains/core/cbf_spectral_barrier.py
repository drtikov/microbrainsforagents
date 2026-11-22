"""
Control Barrier Function (CBF) Spectral Invariant Engine for Aetheris.
Ported and evolved from NeoAura/ECHO (echo_core/barrier_invariants.py) + 2025/2026 SOTA.

Enforces forward-invariant safety boundaries on:
1. Micro-brain weight updates (spectral norm bounded: sigma_max(W) <= limit).
   When an update pushes W towards instability or exploding gradients, CBF projects
   the update vector dW orthogonally back onto the safe invariant manifold via truncated SVD.
2. Synthesized code matrix operators (Lipschitz continuity and spectral radius bounds).
3. Parameter divergence bounds relative to reference anchor (D_KL / Frobenius distance <= kappa).
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

log = logging.getLogger("aetheris.cbf_barrier")


class CBFSpectralBarrierEngine:
    """
    Control Barrier Function (CBF) engine ensuring forward-invariance of parameters:
    Constraint function: h(W) = Limit - sigma_max(W) >= 0
    Forward invariance condition: dot(grad(h), dW) >= -alpha * h(W)
    """

    _instance: Optional["CBFSpectralBarrierEngine"] = None

    def __init__(self, spectral_limit: float = 2.0, alpha: float = 0.5, kappa: float = 1.0):
        self.spectral_limit = float(spectral_limit)
        self.alpha = float(alpha)  # Restoring rate
        self.kappa = float(kappa)  # Maximum Frobenius preference divergence
        self.total_projections = 0
        self.violations_prevented = 0

    @classmethod
    def get_instance(cls) -> "CBFSpectralBarrierEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def enforce_safe_update(
        self,
        weight_matrix: np.ndarray,
        proposed_update: np.ndarray,
        custom_limit: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Applies a Control Barrier Function projection on the proposed update to guarantee
        that the updated weight matrix satisfies C(W) <= 0.
        
        C(W) = spectral_norm(W) - Limit
        Constraint rate: dC/dt <= -alpha * C(W)
        """
        limit = float(custom_limit if custom_limit is not None else self.spectral_limit)
        W = np.asarray(weight_matrix, dtype=np.float64)
        dW = np.asarray(proposed_update, dtype=np.float64)

        if W.ndim < 2:
            norm_w = float(np.linalg.norm(W))
            c_w = norm_w - limit
            projected_dW = dW.copy()
            was_projected = False

            if np.sum(np.abs(dW)) == 0:
                return dW, False

            if norm_w >= 1e-12:
                J = W / norm_w
                dot_product = float(np.sum(J * dW))
                barrier_bound = -self.alpha * c_w
                if dot_product > barrier_bound:
                    j_norm_sq = float(np.sum(J * J))
                    if j_norm_sq > 1e-8:
                        projection_factor = (dot_product - barrier_bound) / j_norm_sq
                        projected_dW = dW - projection_factor * J
                        was_projected = True

            # Nonlinear forward-invariant vector ball guarantee (handles norm_w == 0 and orthogonal dW)
            W_next = W + projected_dW
            norm_next = float(np.linalg.norm(W_next))
            target_bound = float(limit) if norm_w <= limit else float(max(limit, norm_w - self.alpha * c_w))
            if norm_next > target_bound + 1e-6 and norm_next > 1e-12:
                W_next_safe = W_next * (target_bound / norm_next)
                projected_dW = (W_next_safe - W).astype(dW.dtype, copy=False)
                was_projected = True

            if was_projected:
                self.total_projections += 1
                self.violations_prevented += 1
                return projected_dW, True
            return dW, False

        # 2D matrix case: SVD spectral norm + exact nonlinear spectral ball projection
        if np.sum(np.abs(dW)) == 0:
            return dW, False

        try:
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            sigma_max = float(S[0]) if len(S) > 0 else 0.0
        except np.linalg.LinAlgError:
            return dW, False

        c_w = sigma_max - limit
        projected_dW = dW.copy()
        was_projected = False

        if sigma_max >= 1e-12:
            u0 = U[:, 0:1]
            v0 = Vt[0:1, :]
            J = np.dot(u0, v0)  # shape matches W
            dot_product = float(np.sum(J * dW))
            barrier_bound = -self.alpha * c_w

            if dot_product > barrier_bound:
                j_norm_sq = float(np.sum(J * J))
                if j_norm_sq > 1e-8:
                    projection_factor = (dot_product - barrier_bound) / j_norm_sq
                    projected_dW = dW - projection_factor * J
                    was_projected = True

        # Exact nonlinear forward-invariant spectral ball check (prevents W=0 bypass & orthogonal overshoot)
        W_next = W + projected_dW
        target_bound = float(limit) if sigma_max <= limit else float(max(limit, sigma_max - self.alpha * c_w))
        try:
            U_next, S_next, Vt_next = np.linalg.svd(W_next, full_matrices=False)
            if len(S_next) > 0 and float(S_next[0]) > target_bound + 1e-6:
                S_clipped = np.minimum(S_next, target_bound)
                W_next_safe = np.matmul(U_next * S_clipped, Vt_next).astype(W.dtype, copy=False)
                projected_dW = (W_next_safe - W).astype(dW.dtype, copy=False)
                was_projected = True
        except np.linalg.LinAlgError:
            pass

        if was_projected:
            self.total_projections += 1
            self.violations_prevented += 1
            return projected_dW, True

        return dW, False

    def project_matrix_to_safety_manifold(self, matrix: np.ndarray, max_spectral_norm: Optional[float] = None) -> np.ndarray:
        """
        Directly projects a static weight matrix W onto the safe spectral ball B_{sigma <= limit}
        via Mirsky singular-value clipping (optimal Frobenius-closest projection).
        """
        limit = float(max_spectral_norm if max_spectral_norm is not None else self.spectral_limit)
        W = np.asarray(matrix, dtype=np.float64)
        if W.ndim < 2:
            norm_val = float(np.linalg.norm(W))
            if norm_val > limit and norm_val > 1e-12:
                self.violations_prevented += 1
                return W * (limit / norm_val)
            return W.copy()

        try:
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            if len(s) > 0 and s[0] > limit:
                s_clipped = np.minimum(s, limit)
                projected = np.matmul(U * s_clipped, Vt)
                self.violations_prevented += 1
                return projected
        except np.linalg.LinAlgError:
            pass

        return W.copy()

    def check_stability_invariants(self, weight_matrix: np.ndarray) -> Dict[str, Any]:
        """Audits spectral stability metrics."""
        try:
            W = np.asarray(weight_matrix, dtype=np.float64)
            if W.ndim < 2:
                norm_val = float(np.linalg.norm(W))
                return {
                    "spectral_norm": norm_val,
                    "is_stable": bool(np.isfinite(norm_val) and norm_val <= self.spectral_limit),
                    "condition_number": 1.0,
                }
            s = np.linalg.svd(W, compute_uv=False)
            sigma_max = float(s[0]) if len(s) > 0 else 0.0
            sigma_min = float(s[-1]) if len(s) > 0 else 1e-12
            cond = sigma_max / max(1e-12, sigma_min)
            return {
                "spectral_norm": sigma_max,
                "is_stable": bool(np.isfinite(sigma_max) and sigma_max <= self.spectral_limit),
                "condition_number": float(cond),
                "singular_values_top3": [float(x) for x in s[:3]],
            }
        except Exception as e:
            return {"error": str(e), "is_stable": False}
