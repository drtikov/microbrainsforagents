"""
Aetheris Hebbian Fast-Weights & Local Online Plasticity.
Implements fast synaptic weight adaptation governed by local activity:
    "Neurons that fire together, wire together."
Fast weights A(t) sit alongside slow structural weights W_base:
    W_effective = W_base + alpha * A(t)
Hebbian Update:
    Delta A = eta * (pre x post^T) - gamma * A  (Hebbian LTP + Passive LTD decay)
Anti-Hebbian Punishment:
    Delta A = - eta * (pre x post^T) on execution failure / crash.
Zero-gradient, instantaneous sub-millisecond adaptation on CPU.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class HebbianPlasticLayer:
    """
    Synaptic layer with local Hebbian fast-weights.
    """

    def __init__(
        self,
        d_in: int = 64,
        d_out: int = 16,
        alpha_fast: float = 0.25,
        eta_learning: float = 0.05,
        gamma_decay: float = 0.01,
        seed: int = 42,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.alpha_fast = alpha_fast
        self.eta = eta_learning
        self.gamma = gamma_decay
        self.rng = np.random.default_rng(seed)

        # Slow base weights (immutable structural core)
        self.W_slow = self.rng.standard_normal((d_in, d_out), dtype=np.float32) / np.sqrt(d_in)
        # Fast plastic weights A(t)
        self.A_fast = np.zeros((d_in, d_out), dtype=np.float32)

        # Buffer for last activation to support credit assignment
        self.last_pre: Optional[np.ndarray] = None
        self.last_post: Optional[np.ndarray] = None

    @property
    def W_effective(self) -> np.ndarray:
        return self.W_slow + self.alpha_fast * self.A_fast

    def forward(self, pre_act: np.ndarray) -> np.ndarray:
        """Forward pass using effective (slow + fast) weights."""
        pre = np.asarray(pre_act, dtype=np.float32).flatten()
        if len(pre) != self.d_in:
            if len(pre) < self.d_in:
                pre = np.pad(pre, (0, self.d_in - len(pre)))
            else:
                pre = pre[: self.d_in]

        post = np.matmul(pre, self.W_effective)
        self.last_pre = pre
        self.last_post = post
        return post

    def _cap_fast_spectrum(self, max_spectral_norm: float = 2.5) -> None:
        """Prevents fast-weight runaway via spectral norm bounding."""
        if self.A_fast.size == 0:
            return
        try:
            s = np.linalg.svd(self.A_fast, compute_uv=False)
            if len(s) > 0 and float(s[0]) > max_spectral_norm:
                self.A_fast = (self.A_fast * (max_spectral_norm / float(s[0]))).astype(np.float32)
        except np.linalg.LinAlgError:
            pass

    def apply_hebbian_reinforcement(self, reward_factor: float = 1.0) -> Dict[str, Any]:
        """
        Reinforces synapses that fired during a successful step using Oja-normalized plasticity:
        Delta A = eta * reward * (pre @ post^T) - gamma * (post^2) * A
        """
        if self.last_pre is None or self.last_post is None:
            return {"status": "no_prior_activation"}

        # Outer product of pre and post activations
        correlation = np.outer(self.last_pre, self.last_post).astype(np.float32)
        # Oja post-synaptic normalization + gentle passive forgetting
        post_sq = np.square(self.last_post)[np.newaxis, :]
        decay = self.gamma * (0.1 + post_sq) * self.A_fast
        # Plastic update
        delta_A = (self.eta * reward_factor * correlation) - decay

        # Element clamp + spectral cap to prevent divergence
        self.A_fast = np.clip(self.A_fast + delta_A, -2.0, 2.0).astype(np.float32)
        self._cap_fast_spectrum(2.5)

        return {
            "status": "reinforced",
            "reward_factor": reward_factor,
            "fast_norm": float(np.linalg.norm(self.A_fast)),
            "delta_max": float(np.max(np.abs(delta_A))),
        }

    def apply_anti_hebbian_punishment(self, penalty_factor: float = 1.5) -> Dict[str, Any]:
        """
        Suppresses synapses that produced a failing step or crash:
        Delta A = - eta * penalty * (pre @ post^T) - gamma * A
        """
        if self.last_pre is None or self.last_post is None:
            return {"status": "no_prior_activation"}

        correlation = np.outer(self.last_pre, self.last_post).astype(np.float32)
        delta_A = (-self.eta * penalty_factor * correlation) - (self.gamma * 0.5 * self.A_fast)
        self.A_fast = np.clip(self.A_fast + delta_A, -2.0, 2.0).astype(np.float32)
        self._cap_fast_spectrum(2.5)

        return {
            "status": "punished",
            "penalty_factor": penalty_factor,
            "fast_norm": float(np.linalg.norm(self.A_fast)),
        }

    def consolidate_fast_to_slow(self, transfer_rate: float = 0.2) -> Dict[str, Any]:
        """
        Complementary Learning Systems (CLS) consolidation: transfers validated fast weights
        into structural slow weights and refreshes the fast plastic buffer.
        """
        rate = max(0.0, min(1.0, float(transfer_rate)))
        delta_slow = (self.alpha_fast * rate * self.A_fast).astype(np.float32)
        self.W_slow = (self.W_slow + delta_slow).astype(np.float32)
        transferred_norm = float(np.linalg.norm(delta_slow))
        self.reset_fast_weights()
        return {
            "status": "consolidated_fast_to_slow",
            "transfer_rate": rate,
            "transferred_norm": round(transferred_norm, 6),
            "w_slow_norm": round(float(np.linalg.norm(self.W_slow)), 6),
        }

    def reset_fast_weights(self) -> None:
        """Clears fast weights (e.g. after sleep consolidation)."""
        self.A_fast.fill(0.0)
        self.last_pre = None
        self.last_post = None

