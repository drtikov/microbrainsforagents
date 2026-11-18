"""
Aetheris Liquid State Machine & Reservoir Computing Engine.
Inspired by NeoAura's echo_core and Edge Neuromorphic Reservoir Computing (2025-2026).
Tracks multi-step temporal workflows and telemetry dynamics across agent tool executions
using an in-memory sparse recurrent liquid without requiring backpropagation through time.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

class LiquidStateReservoir:
    """A recurrent sparse reservoir that maintains temporal memory over sequences of actions."""

    _instance: Optional[LiquidStateReservoir] = None

    def __init__(self, reservoir_dim: int = 128, input_dim: int = 64, spectral_radius: float = 0.92, leak_rate: float = 0.3):
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

        # Initialize internal reservoir state
        self.state = np.zeros(reservoir_dim, dtype=np.float32)

        # Build sparse deterministic recurrent matrix
        rng = np.random.RandomState(1337)
        raw_W = rng.randn(reservoir_dim, reservoir_dim).astype(np.float32)
        # 80% sparsity
        mask = rng.rand(reservoir_dim, reservoir_dim) < 0.2
        raw_W = raw_W * mask

        # Scale to designated spectral radius
        eigenvalues = np.linalg.eigvals(raw_W)
        max_eig = max(abs(e) for e in eigenvalues) if len(eigenvalues) > 0 else 1.0
        if max_eig > 1e-9:
            self.W_res = raw_W * (spectral_radius / max_eig)
        else:
            self.W_res = raw_W

        self.W_in = rng.randn(reservoir_dim, input_dim).astype(np.float32) * 0.5
        self.W_gate = rng.randn(reservoir_dim, input_dim).astype(np.float32) * 0.3
        # Multi-timescale inverse time constants tau_inv in [0.05, 5.0] for CfC dynamics
        self.tau_inv = np.linspace(0.05, 5.0, reservoir_dim, dtype=np.float32)
        self.W_readout: Optional[np.ndarray] = None
        self.readout_trained: bool = False
        self.step_counter = 0
        self._last_step_ts: Optional[float] = None

    @classmethod
    def get_instance(cls) -> LiquidStateReservoir:
        if cls._instance is None:
            cls._instance = LiquidStateReservoir()
        return cls._instance

    def step(self, input_vector: Union[List[float], np.ndarray], event_tag: str = "agent_step") -> Dict[str, Any]:
        """
        Leaky Integrate-and-Fire Reservoir Update:
        s_t = (1 - alpha) * s_{t-1} + alpha * tanh(W_res @ s_{t-1} + W_in @ u_t)
        """
        u = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(u) < self.input_dim:
            u = np.pad(u, (0, self.input_dim - len(u)))
        else:
            u = u[:self.input_dim]

        pre_activation = np.dot(self.W_res, self.state) + np.dot(self.W_in, u)
        non_linear = np.tanh(pre_activation)
        
        # Leaky integration
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * non_linear
        self.step_counter += 1

        state_energy = float(np.linalg.norm(self.state))
        top_activations = [round(float(x), 4) for x in self.state[:6]]

        res = {
            "status": "stepped",
            "step_count": self.step_counter,
            "event_tag": event_tag,
            "reservoir_energy": round(state_energy, 4),
            "top_state_readout": top_activations,
            "full_state": self.state.tolist(),
        }
        if self.readout_trained and self.W_readout is not None:
            res["readout_prediction"] = np.dot(self.state, self.W_readout).tolist()
        return res

    def fit_readout(
        self,
        state_history: Union[List[List[float]], np.ndarray],
        targets: Union[List[List[float]], np.ndarray],
        l2_reg: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        2025-2026 LFM / Reservoir Readout Adaptation:
        Learns linear readout weights W_readout via closed-form Tikhonov ridge regression
        over collected reservoir trajectories (without BPTT).
        """
        S = np.asarray(state_history, dtype=np.float32)
        if S.ndim == 1:
            S = S[None, :]
        n = S.shape[0]
        if n == 0:
            return {"status": "skipped_empty", "readout_trained": self.readout_trained}
        if S.shape[1] < self.reservoir_dim:
            S = np.pad(S, ((0, 0), (0, self.reservoir_dim - S.shape[1])))
        else:
            S = S[:, : self.reservoir_dim]

        Y = np.asarray(targets, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(n, -1)
        else:
            Y = Y[:n]

        if self.W_readout is not None and self.W_readout.shape == (self.reservoir_dim, Y.shape[1]):
            pre_pred = np.dot(S, self.W_readout)
            pre_mse = float(np.mean((pre_pred - Y) ** 2))
        else:
            pre_mse = float(np.mean(Y ** 2))

        S64 = S.astype(np.float64)
        Y64 = Y.astype(np.float64)
        gram = S64.T @ S64 + float(l2_reg) * np.eye(self.reservoir_dim, dtype=np.float64)
        rhs = S64.T @ Y64
        try:
            W_opt = np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            W_opt, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)
            W_opt = W_opt.astype(np.float32)

        self.W_readout = W_opt
        self.readout_trained = True
        post_pred = np.dot(S, self.W_readout)
        post_mse = float(np.mean((post_pred - Y) ** 2))

        return {
            "status": "readout_fitted",
            "readout_trained": True,
            "samples": int(n),
            "out_dim": int(Y.shape[1]),
            "pre_mse": round(pre_mse, 6),
            "post_mse": round(post_mse, 6),
            "mse_reduced": bool(post_mse <= pre_mse),
        }

    def step_cfc_continuous(
        self,
        input_vector: Union[List[float], np.ndarray],
        delta_t: Optional[float] = None,
        event_tag: str = "agent_step",
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Closed-Form Continuous-Time (CfC) Reservoir Update on OpenCL GPU / CPU:
        h(t + dt) = sigmoid(-|f(x, h)| * tau_inv * dt) * h(t) + (1 - sigmoid(...)) * tanh(g(x, h))
        Natively adapts to irregular wall-clock delays between agent tool calls.
        """
        import time as _time

        now = _time.perf_counter()
        if delta_t is None:
            dt = (now - self._last_step_ts) if self._last_step_ts is not None else 0.1
        else:
            dt = float(delta_t)
        self._last_step_ts = now

        u = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(u) < self.input_dim:
            u = np.pad(u, (0, self.input_dim - len(u)))
        else:
            u = u[: self.input_dim]

        g_drive = np.dot(self.W_res, self.state) + np.dot(self.W_in, u)
        f_drive = np.dot(self.W_gate, u) + self.state

        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        self.state, backend = acc.cfc_step(self.state, f_drive, g_drive, self.tau_inv, dt, force_gpu=force_gpu)
        self.step_counter += 1

        state_energy = float(np.linalg.norm(self.state))
        top_activations = [round(float(x), 4) for x in self.state[:6]]
        res = {
            "status": "stepped_cfc",
            "backend": backend,
            "delta_t": round(float(dt), 4),
            "step_count": self.step_counter,
            "event_tag": event_tag,
            "reservoir_energy": round(state_energy, 4),
            "top_state_readout": top_activations,
            "full_state": self.state.tolist(),
        }
        if self.readout_trained and self.W_readout is not None:
            res["readout_prediction"] = np.dot(self.state, self.W_readout).tolist()
        return res

    # ------------------------------------------------------------------
    # 2026-11-18 robotics extension: ONLINE readout adaptation (RLS/FORCE).
    # The pre-existing fit_readout() is BATCH ridge over collected
    # trajectories; the methods below adapt the readout per step so the
    # reservoir can track non-stationary targets in a closed sensorimotor
    # loop (RLS with forgetting; FORCE rank-1, Sussillo & Abbott style).
    # ------------------------------------------------------------------

    def init_online_readout(
        self,
        out_dim: int,
        method: str = "rls",
        forgetting_factor: float = 0.999,
        force_gamma: float = 1.0,
        l2_delta: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Prepare the readout for per-step online adaptation.

        method="rls":   recursive least squares with forgetting factor
                        lambda = forgetting_factor in (0, 1]; maintains the
                        inverse autocorrelation matrix P (reservoir_dim x
                        reservoir_dim) initialized to I/l2_delta (standard
                        large-P0 RLS: l2_delta must be << data scale or the
                        prior biases low-energy directions).
        method="force": FORCE rank-1 (fixed-P) update, W += s e^T / gamma.
                        No inverse-correlation tracking (O(N) per step);
                        gamma must be >> ||s||^2 or the delta rule diverges
                        (an honest numerical reset guards that case).
        """
        method = str(method).lower()
        if method not in ("rls", "force"):
            return {"status": "skipped_invalid_method", "method": method}
        out_dim = int(out_dim)
        if out_dim < 1:
            return {"status": "skipped_invalid_out_dim", "out_dim": out_dim}
        lam = float(forgetting_factor)
        if not (0.0 < lam <= 1.0):
            return {"status": "skipped_invalid_forgetting", "forgetting_factor": forgetting_factor}

        self.W_readout = np.zeros((self.reservoir_dim, out_dim), dtype=np.float32)
        self.readout_trained = True  # predictions are emitted from now on (zeros until first update)
        self._online_method = method
        self._forgetting_factor = lam
        self._force_gamma = max(float(force_gamma), 1e-6)
        self._P: Optional[np.ndarray] = (
            np.eye(self.reservoir_dim, dtype=np.float64) / max(float(l2_delta), 1e-12)
            if method == "rls"
            else None
        )
        self._online_updates = 0
        self._online_ema_mse: Optional[float] = None
        return {
            "status": "online_readout_ready",
            "method": method,
            "out_dim": out_dim,
            "forgetting_factor": lam if method == "rls" else None,
            "reservoir_dim": self.reservoir_dim,
        }

    def update_readout_online(
        self,
        target: Union[List[float], np.ndarray],
        state_vector: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        One online adaptation step against the CURRENT reservoir state
        (or an explicit state_vector of length reservoir_dim).

        RLS:   k = P s / (lambda + s^T P s);  W += k e^T;  P = (P - k s^T P) / lambda
        FORCE: W += (1/gamma) * s e^T

        The returned pre_mse is the error of the prediction MADE BEFORE this
        update (proper online protocol), post_mse is the error after.
        """
        if not (self.readout_trained and self.W_readout is not None):
            return {"status": "skipped_not_initialized"}
        s = (
            np.asarray(state_vector, dtype=np.float64).flatten()
            if state_vector is not None
            else self.state.astype(np.float64).flatten()
        )
        if s.shape[0] != self.reservoir_dim:
            return {
                "status": "skipped_state_dim_mismatch",
                "expected": self.reservoir_dim,
                "received": int(s.shape[0]),
            }
        y = np.asarray(target, dtype=np.float64).flatten()
        if y.shape[0] != self.W_readout.shape[1]:
            return {
                "status": "skipped_target_dim_mismatch",
                "expected": int(self.W_readout.shape[1]),
                "received": int(y.shape[0]),
            }

        W = self.W_readout.astype(np.float64)
        pre_pred = W.T @ s
        e = y - pre_pred
        pre_mse = float(np.mean(e ** 2))

        if self._online_method == "rls" and self._P is not None:
            P_s = self._P @ s
            denom = max(self._forgetting_factor + float(s @ P_s), 1e-12)
            k = (P_s / denom)[:, None]
            W = W + k @ e[None, :]
            self._P = (self._P - k @ (P_s[None, :])) / self._forgetting_factor
            # numerical hygiene: keep P symmetric, bound its norm
            self._P = 0.5 * (self._P + self._P.T)
            p_norm = float(np.linalg.norm(self._P))
            if not np.isfinite(p_norm) or p_norm > 1e12:
                self._P = np.eye(self.reservoir_dim, dtype=np.float64)
                W = np.zeros_like(W)
                reset_flag = True
            else:
                reset_flag = False
        else:
            W = W + (s[:, None] @ e[None, :]) / self._force_gamma
            w_norm = float(np.linalg.norm(W))
            if not np.isfinite(w_norm) or w_norm > 1e8:
                # FORCE delta rule diverges when gamma << ||s||^2; reset honestly
                W = np.zeros_like(W)
                reset_flag = True
            else:
                reset_flag = False

        self.W_readout = W.astype(np.float32)
        post_pred = self.W_readout.astype(np.float64).T @ s
        post_mse = float(np.mean((y - post_pred) ** 2))
        self._online_updates += 1
        self._online_ema_mse = (
            post_mse
            if self._online_ema_mse is None
            else 0.9 * self._online_ema_mse + 0.1 * post_mse
        )
        return {
            "status": "updated_after_p_numerical_reset" if reset_flag else "updated",
            "method": self._online_method,
            "updates": self._online_updates,
            "pre_mse": round(pre_mse, 8),
            "post_mse": round(post_mse, 8),
            "ema_mse": round(self._online_ema_mse, 8),
        }

    def online_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: Optional[Union[List[float], np.ndarray]] = None,
        delta_t: Optional[float] = None,
        event_tag: str = "robotics_step",
    ) -> Dict[str, Any]:
        """
        Fixed-role sensorimotor tick: step the liquid, emit the readout
        prediction for this state, THEN adapt against the (optional) target.
        The prediction is emitted BEFORE the update (honest online protocol —
        the model never scores a sample it has already trained on).
        """
        if delta_t is not None:
            res = self.step_cfc_continuous(input_vector, delta_t=delta_t, event_tag=event_tag)
        else:
            res = self.step(input_vector, event_tag=event_tag)
        res["online"] = {"initialized": bool(self.readout_trained and self.W_readout is not None)}
        if res["online"]["initialized"]:
            res["online"]["updates"] = self._online_updates
            res["online"]["ema_mse"] = (
                round(self._online_ema_mse, 8) if self._online_ema_mse is not None else None
            )
        if target is not None:
            res["online_update"] = self.update_readout_online(target)
        return res

    def reset_state(self):
        """Flushes the reservoir liquid to ground state."""
        self.state = np.zeros(self.reservoir_dim, dtype=np.float32)
        self.step_counter = 0
        self._last_step_ts = None

