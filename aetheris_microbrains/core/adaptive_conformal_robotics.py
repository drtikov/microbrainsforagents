"""Adaptive Conformal Inference (ACI) and directional barrier inflation for robotics.

Provides:
  - ``OnlineACICalibrator``: Gibbs & Candes Adaptive Conformal Inference over a
    sliding window of dynamics / perception nonconformity residuals.
  - ``DirectionalBarrierInflator``: online EMA residual covariance tracker
    computing directional barrier margins q_hat * sqrt(grad_h^T Sigma_t grad_h)
    with an isotropic fallback bound.
  - ``WorldModelSurprisalBridge``: connects state prediction residuals (and/or
    ``PredictiveCodingEngine`` free-energy surprisal) + ``OnlineACICalibrator``
    directly to ``StateCBFSafetyFilter.set_conformal_margin(q_hat)``.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from .state_cbf_filter import BarrierConstraint, StateCBFSafetyFilter


class OnlineACICalibrator:
    """Adaptive Conformal Inference calibrator for non-stationary residual streams."""

    def __init__(
        self,
        target_alpha: float = 0.05,
        gamma_aci: float = 0.05,
        window_size: int = 200,
        alpha_min: float = 1e-4,
        alpha_max: float = 0.5,
        initial_q_hat: float = 0.0,
    ):
        self.target_alpha = float(np.clip(target_alpha, 1e-4, 0.99))
        self.gamma_aci = max(0.0, float(gamma_aci))
        self.window_size = max(1, int(window_size))
        self.alpha_min = float(np.clip(alpha_min, 1e-6, 0.5))
        self.alpha_max = float(np.clip(alpha_max, self.alpha_min, 0.99))
        self.alpha_t = float(np.clip(self.target_alpha, self.alpha_min, self.alpha_max))
        self.q_hat = max(0.0, float(initial_q_hat))

        self.scores: Deque[float] = deque(maxlen=self.window_size)
        self.coverage_history: Deque[int] = deque(maxlen=self.window_size)
        self.steps: int = 0
        self.total_covered: int = 0

    def _compute_quantile(self) -> float:
        n = len(self.scores)
        if n == 0:
            return self.q_hat
        arr = np.sort(np.asarray(self.scores, dtype=np.float64))
        k_idx = int(math.ceil((n + 1) * (1.0 - self.alpha_t)))
        if k_idx <= 1:
            return float(arr[0])
        if k_idx <= n:
            return float(arr[k_idx - 1])

        # Safe finite fallback scaling when alpha_t drops below 1 / (n + 1)
        max_s = float(arr[-1])
        ratio = (1.0 / (n + 1)) / max(self.alpha_t, self.alpha_min)
        scale = 1.0 + min(1.0, 0.2 * math.log1p(max(0.0, ratio - 1.0)))
        return float(max_s * scale)

    def update_score(self, score: float) -> Dict[str, Any]:
        """Update ACI level alpha_t and quantile q_hat from a scalar nonconformity score."""
        s_t = float(score)
        if not math.isfinite(s_t):
            self.steps += 1
            self.coverage_history.append(0)
            self.alpha_t = float(
                np.clip(
                    self.alpha_t + self.gamma_aci * (self.target_alpha - 1.0),
                    self.alpha_min,
                    self.alpha_max,
                )
            )
            return {
                "status": "invalid_nonfinite_score",
                "score": s_t,
                "covered": False,
                "alpha_t": self.alpha_t,
                "q_hat": self.q_hat,
                "n_window": len(self.scores),
                "empirical_coverage": self.empirical_coverage,
                "window_coverage": self.window_coverage,
            }

        s_t = max(0.0, s_t)
        covered = bool(s_t <= self.q_hat) if self.steps > 0 or self.q_hat > 0.0 else False
        miss = 0.0 if covered else 1.0

        self.steps += 1
        if covered:
            self.total_covered += 1
        self.coverage_history.append(1 if covered else 0)

        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma_aci * (self.target_alpha - miss),
                self.alpha_min,
                self.alpha_max,
            )
        )
        self.scores.append(s_t)
        self.q_hat = self._compute_quantile()

        return {
            "status": "updated",
            "score": s_t,
            "covered": covered,
            "alpha_t": self.alpha_t,
            "q_hat": self.q_hat,
            "n_window": len(self.scores),
            "empirical_coverage": self.empirical_coverage,
            "window_coverage": self.window_coverage,
        }

    def update(
        self,
        predicted_state: Sequence[float],
        observed_state: Sequence[float],
        metric_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Compute nonconformity score ||x_obs - x_pred||_M and update ACI state."""
        pred = np.asarray(predicted_state, dtype=np.float64).flatten()
        obs = np.asarray(observed_state, dtype=np.float64).flatten()
        if pred.shape != obs.shape:
            raise ValueError(f"State shape mismatch: predicted {pred.shape} vs observed {obs.shape}")

        diff = obs - pred
        if metric_weights is None:
            score = float(np.linalg.norm(diff))
        else:
            w = np.asarray(metric_weights, dtype=np.float64)
            if w.ndim == 1:
                if w.shape[0] != diff.shape[0]:
                    raise ValueError(f"metric_weights dim {w.shape[0]} != state dim {diff.shape[0]}")
                score = float(np.sqrt(np.sum(np.maximum(0.0, w) * (diff * diff))))
            elif w.ndim == 2:
                if w.shape != (diff.shape[0], diff.shape[0]):
                    raise ValueError(f"metric_weights shape {w.shape} != ({diff.shape[0]}, {diff.shape[0]})")
                score = float(np.sqrt(max(0.0, float(diff @ w @ diff))))
            else:
                raise ValueError("metric_weights must be 1D diagonal or 2D matrix")
        return self.update_score(score)

    @property
    def empirical_coverage(self) -> float:
        return float(self.total_covered / self.steps) if self.steps > 0 else 0.0

    @property
    def window_coverage(self) -> float:
        return float(sum(self.coverage_history) / len(self.coverage_history)) if self.coverage_history else 0.0


class DirectionalBarrierInflator:
    """Tracks EMA residual covariance Sigma_t and computes directional barrier margins."""

    def __init__(
        self,
        state_dim: int,
        ema_decay: float = 0.9,
        reg_eps: float = 1e-6,
    ):
        self.state_dim = max(1, int(state_dim))
        self.ema_decay = float(np.clip(ema_decay, 0.0, 0.9999))
        self.reg_eps = max(1e-12, float(reg_eps))
        self._cov_ema = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        self.n_updates: int = 0

    @property
    def sigma(self) -> np.ndarray:
        """Regularized PSD residual covariance Sigma_t."""
        return self._cov_ema + self.reg_eps * np.eye(self.state_dim, dtype=np.float64)

    def update_residual(self, residual: Sequence[float]) -> np.ndarray:
        """Update online EMA residual covariance with residual vector r_t."""
        r = np.asarray(residual, dtype=np.float64).flatten()
        if r.shape[0] != self.state_dim:
            raise ValueError(f"Residual dim {r.shape[0]} != state_dim {self.state_dim}")
        if not np.all(np.isfinite(r)):
            return self.sigma

        outer = np.outer(r, r)
        if self.n_updates == 0:
            self._cov_ema = outer
        else:
            self._cov_ema = self.ema_decay * self._cov_ema + (1.0 - self.ema_decay) * outer
        self._cov_ema = 0.5 * (self._cov_ema + self._cov_ema.T)
        self.n_updates += 1
        return self.sigma

    def update(self, residual: Sequence[float]) -> np.ndarray:
        """Alias for update_residual."""
        return self.update_residual(residual)

    def directional_margin(
        self,
        grad_h: Sequence[float],
        q_hat: float,
    ) -> Dict[str, float]:
        """Compute delta_i(x) = q_hat * sqrt(grad_h^T Sigma_t grad_h) and isotropic bound."""
        g = np.asarray(grad_h, dtype=np.float64).flatten()
        if g.shape[0] != self.state_dim:
            raise ValueError(f"grad_h dim {g.shape[0]} != state_dim {self.state_dim}")

        q = max(0.0, float(q_hat))
        g_norm = float(np.linalg.norm(g))
        isotropic_bound = q * g_norm

        if not np.all(np.isfinite(g)):
            return {
                "directional_margin": isotropic_bound,
                "isotropic_fallback": isotropic_bound,
                "effective_margin": isotropic_bound,
                "grad_norm": g_norm,
                "used_fallback": True,
            }

        sig = self.sigma
        quad = max(0.0, float(g @ sig @ g))
        dir_margin = q * math.sqrt(quad)
        used_fallback = self.n_updates == 0 or not math.isfinite(dir_margin)
        effective = isotropic_bound if used_fallback else dir_margin
        return {
            "directional_margin": float(dir_margin),
            "isotropic_fallback": float(isotropic_bound),
            "effective_margin": float(effective),
            "grad_norm": g_norm,
            "used_fallback": bool(used_fallback),
        }

    def barrier_margins(
        self,
        barriers: Sequence[BarrierConstraint],
        x: Sequence[float],
        q_hat: float,
    ) -> List[Dict[str, float]]:
        """Compute directional margins for every barrier at state x."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        return [self.directional_margin(b.grad_h(x_arr), q_hat) for b in barriers]


class WorldModelSurprisalBridge:
    """Connects prediction residuals / free-energy surprisal + ACI to StateCBFSafetyFilter."""

    def __init__(
        self,
        calibrator: Optional[OnlineACICalibrator] = None,
        inflator: Optional[DirectionalBarrierInflator] = None,
        pc_engine: Optional[Any] = None,
        safety_filter: Optional[StateCBFSafetyFilter] = None,
        surprisal_weight: float = 0.0,
    ):
        self.calibrator = calibrator or OnlineACICalibrator()
        self.inflator = inflator
        self.pc_engine = pc_engine
        self.safety_filter = safety_filter
        self.surprisal_weight = max(0.0, float(surprisal_weight))

    def adapt_from_observation(
        self,
        x_pred: Sequence[float],
        x_obs: Sequence[float],
        safety_filter: Optional[StateCBFSafetyFilter] = None,
        metric_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Update ACI (and optional covariance / PC surprisal) and push q_hat to safety filter."""
        pred = np.asarray(x_pred, dtype=np.float64).flatten()
        obs = np.asarray(x_obs, dtype=np.float64).flatten()
        if pred.shape != obs.shape:
            raise ValueError(f"State shape mismatch: predicted {pred.shape} vs observed {obs.shape}")
        residual = obs - pred

        pc_surprisal: Optional[float] = None
        if self.pc_engine is not None:
            pc_out = self.pc_engine.step_incremental_pc(residual.astype(np.float32))
            pc_surprisal = float(pc_out.get("final_free_energy", 0.0))

        if pc_surprisal is not None and self.surprisal_weight > 0.0:
            if metric_weights is None:
                base_score = float(np.linalg.norm(residual))
            else:
                w = np.asarray(metric_weights, dtype=np.float64)
                base_score = (
                    float(np.sqrt(np.sum(np.maximum(0.0, w) * (residual * residual))))
                    if w.ndim == 1
                    else float(np.sqrt(max(0.0, float(residual @ w @ residual))))
                )
            cal_out = self.calibrator.update_score(base_score + self.surprisal_weight * pc_surprisal)
        else:
            cal_out = self.calibrator.update(pred, obs, metric_weights=metric_weights)

        if self.inflator is not None and residual.shape[0] == self.inflator.state_dim:
            self.inflator.update_residual(residual)

        sf = safety_filter if safety_filter is not None else self.safety_filter
        if sf is not None:
            sf.set_conformal_margin(cal_out["q_hat"])

        out: Dict[str, Any] = dict(cal_out)
        out["conformal_margin_applied"] = cal_out["q_hat"]
        out["safety_filter_updated"] = sf is not None
        if pc_surprisal is not None:
            out["pc_surprisal"] = pc_surprisal
        if self.inflator is not None and sf is not None and sf.barriers and obs.shape[0] == self.inflator.state_dim:
            out["directional_margins"] = self.inflator.barrier_margins(sf.barriers, obs, cal_out["q_hat"])
        return out

    def adapt_from_surprisal(
        self,
        observation_or_error: Sequence[float],
        safety_filter: Optional[StateCBFSafetyFilter] = None,
    ) -> Dict[str, Any]:
        """Update ACI from PredictiveCodingEngine free-energy surprisal and apply margin."""
        if self.pc_engine is None:
            raise ValueError("pc_engine must be provided to adapt_from_surprisal")
        vec = np.asarray(observation_or_error, dtype=np.float32).flatten()
        pc_out = self.pc_engine.step_incremental_pc(vec)
        fe = float(pc_out.get("final_free_energy", 0.0))
        cal_out = self.calibrator.update_score(fe)

        sf = safety_filter if safety_filter is not None else self.safety_filter
        if sf is not None:
            sf.set_conformal_margin(cal_out["q_hat"])

        out: Dict[str, Any] = dict(cal_out)
        out["pc_surprisal"] = fe
        out["conformal_margin_applied"] = cal_out["q_hat"]
        out["safety_filter_updated"] = sf is not None
        return out
