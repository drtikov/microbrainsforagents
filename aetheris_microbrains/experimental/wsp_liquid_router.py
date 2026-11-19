"""
Aetheris Liquid Time-Constant (LTC) Router & Closed-form Continuous-time (CfC) Substrate.
Implements continuous-time neural dynamics on CPU via NumPy:
    dx/dt = - [1/tau + f(x, I)] * x + f(x, I) * A
Using the analytical Closed-form Continuous-time (CfC) solution:
    h(t) = (h_0 - A) * exp(-t * (1/tau + f(x, I))) + A
Provides input-dependent time constant adaptation (slows down under high epistemic uncertainty,
reacts instantaneously to routine commands).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class LiquidCfCRouter:
    """
    Closed-form Continuous-time (CfC) Liquid Neural Network Router.
    Runs in < 0.05ms on CPU.
    """

    def __init__(
        self,
        d_in: int = 128,
        d_hidden: int = 64,
        d_out: int = 16,
        base_tau: float = 1.0,
        seed: int = 42,
    ):
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.base_tau = base_tau
        self.rng = np.random.default_rng(seed)

        # Static feed-forward projections
        self.W_in = self.rng.standard_normal((d_in, d_hidden), dtype=np.float32) / np.sqrt(d_in)
        self.b_in = np.zeros(d_hidden, dtype=np.float32)

        # Gating network f(x, I) controlling time-constant
        self.W_gate = self.rng.standard_normal((d_in + d_hidden, d_hidden), dtype=np.float32) / np.sqrt(d_in + d_hidden)
        self.b_gate = np.zeros(d_hidden, dtype=np.float32)

        # Asymptotic attractor A
        self.W_attractor = self.rng.standard_normal((d_hidden, d_hidden), dtype=np.float32) / np.sqrt(d_hidden)

        # Output projection
        self.W_out = self.rng.standard_normal((d_hidden, d_out), dtype=np.float32) / np.sqrt(d_hidden)
        self.b_out = np.zeros(d_out, dtype=np.float32)

        # Hidden continuous-time state
        self.h_state = np.zeros(d_hidden, dtype=np.float32)

    def reset_state(self) -> None:
        """Resets hidden trajectory state to zero."""
        self.h_state.fill(0.0)

    def forward(
        self,
        x: np.ndarray,
        dt: float = 0.1,
        uncertainty_factor: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Closed-form Continuous-time (CfC) step:
        h(t + dt) = (h(t) - A) * exp(-dt * (1/tau_eff + f_gate)) + A
        where tau_eff is modulated by uncertainty_factor (slows down when uncertain).
        """
        dt_safe = max(1e-4, float(dt))
        uncertainty_safe = max(0.1, float(uncertainty_factor))

        x_in = np.asarray(x, dtype=np.float32).flatten()
        if len(x_in) != self.d_in:
            if len(x_in) < self.d_in:
                x_in = np.pad(x_in, (0, self.d_in - len(x_in)))
            else:
                x_in = x_in[: self.d_in]

        # Input drive
        u = np.maximum(0.0, np.matmul(x_in, self.W_in) + self.b_in)  # (d_hidden,)

        # Compute dynamic gating & nonlinear rate f(x, I) with clipped sigmoid preventing exp overflow
        gate_input = np.concatenate([x_in, self.h_state])
        gate_logits = np.clip(np.matmul(gate_input, self.W_gate) + self.b_gate, -30.0, 30.0)
        f_gate = 1.0 / (1.0 + np.exp(-gate_logits))  # Sigmoid in [0, 1]

        # Effective time-constant tau: when uncertain, tau increases (decay slows down)
        tau_eff = self.base_tau * uncertainty_safe + 1e-4
        decay_rate = (1.0 / tau_eff) + f_gate

        # Attractor A(x) = tanh(W_attractor @ u)
        attractor = np.tanh(np.matmul(u, self.W_attractor))

        # CfC Analytical solution
        exp_decay = np.exp(-dt_safe * decay_rate)
        self.h_state = ((self.h_state - attractor) * exp_decay + attractor).astype(np.float32)

        # Output logits
        logits = np.matmul(self.h_state, self.W_out) + self.b_out
        route = int(np.argmax(logits))

        telemetry = {
            "dt": dt_safe,
            "mean_tau_eff": float(np.mean(tau_eff)),
            "mean_decay_rate": float(np.mean(decay_rate)),
            "top_route": route,
            "h_norm": float(np.linalg.norm(self.h_state)),
        }
        return logits, telemetry

    def adapt_readout(self, target_route: int, lr: float = 0.05) -> Dict[str, Any]:
        """
        Online supervised readout adaptation on the current liquid hidden state h_state
        via numerically-stable softmax cross-entropy gradient descent.
        """
        target = int(target_route) % self.d_out
        logits = np.matmul(self.h_state, self.W_out) + self.b_out
        shifted = logits - np.max(logits)
        exp_l = np.exp(np.clip(shifted, -30.0, 0.0))
        probs = exp_l / (np.sum(exp_l) + 1e-12)

        target_one_hot = np.zeros(self.d_out, dtype=np.float32)
        target_one_hot[target] = 1.0
        err = (probs - target_one_hot).astype(np.float32)  # (d_out,)

        self.W_out -= float(lr) * np.outer(self.h_state, err)
        self.b_out -= float(lr) * err
        loss = float(-np.log(max(float(probs[target]), 1e-12)))
        return {
            "target_route": target,
            "predicted_route_before": int(np.argmax(logits)),
            "cross_entropy_loss": round(loss, 6),
        }

