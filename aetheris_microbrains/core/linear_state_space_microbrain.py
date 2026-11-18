"""
Aetheris Linear State Space Micro-Brain Core (Mini-Mamba / S6).
Inspired by Gu & Dao (Mamba / Structured State Space Models) 2025-2026.
Replaces quadratic O(N^2) attention with O(N) linear time and O(1) state memory on CPU.
Ingests long compiler outputs, git diffs, and execution logs without memory explosion.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class LinearStateSpaceMicroBrain:
    """A discretized continuous-time state space model (SSM) running on CPU."""

    _instance: Optional[LinearStateSpaceMicroBrain] = None

    def __init__(self, d_model: int = 32, d_state: int = 16):
        self.d_model = d_model
        self.d_state = d_state

        rng = np.random.RandomState(42)
        # Continuous-time state transition parameter A (HiPPO initialization diagonal)
        self.A_log = -np.log(np.arange(1, d_state + 1, dtype=np.float32)).reshape(1, d_state)
        self.A = -np.exp(self.A_log)  # Strictly negative eigenvalues for stable dynamics

        # Input and output projections
        self.B_proj = rng.randn(d_model, d_state).astype(np.float32) / math.sqrt(d_model)
        self.C_proj = rng.randn(d_model, d_state).astype(np.float32) / math.sqrt(d_model)
        self.D = np.ones(d_model, dtype=np.float32)

        # Delta projection for discretization step size
        self.delta_proj = rng.randn(d_model).astype(np.float32) * 0.1

        # Hidden state memory: shape (d_model, d_state)
        self.hidden_state = np.zeros((d_model, d_state), dtype=np.float32)
        self.prev_u = np.zeros(d_model, dtype=np.float32)
        self.has_prev_u = False
        self.omega_rot = np.linspace(0.1, math.pi * 0.5, max(1, d_state // 2), dtype=np.float32)
        self.readout_trained = False
        self.readout_mse: Optional[float] = None
        self.processed_tokens_count = 0

    @classmethod
    def get_instance(cls) -> LinearStateSpaceMicroBrain:
        if cls._instance is None:
            cls._instance = LinearStateSpaceMicroBrain()
        return cls._instance

    def _discretize(self, u_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-Order Hold (ZOH) Discretization:
        Delta_t = softplus(delta_proj + u_t)
        A_bar = exp(Delta_t * A)
        B_bar = Delta_t * B
        """
        # Delta parameter modulated by input (selective SSM)
        delta = np.log1p(np.exp(np.clip(self.delta_proj + 0.1 * u_t, -10.0, 10.0))) # shape (d_model,)
        delta_expanded = delta[:, None] # shape (d_model, 1)

        # A_bar: shape (d_model, d_state)
        A_bar = np.exp(delta_expanded * self.A)
        # B_bar: shape (d_model, d_state)
        B_bar = delta_expanded * self.B_proj
        return A_bar, B_bar

    def step(self, input_vector: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Recurrent SSM step:
        h_t = A_bar * h_{t-1} + B_bar * u_t
        y_t = sum(C * h_t) + D * u_t
        """
        u = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(u) < self.d_model:
            u = np.pad(u, (0, self.d_model - len(u)))
        else:
            u = u[:self.d_model]

        A_bar, B_bar = self._discretize(u)

        # Elementwise recurrent state update: (d_model, d_state)
        self.hidden_state = A_bar * self.hidden_state + B_bar * u[:, None]
        self.processed_tokens_count += 1

        # Output projection
        y = np.sum(self.hidden_state * self.C_proj, axis=-1) + self.D * u

        energy = float(np.linalg.norm(self.hidden_state))
        return {
            "status": "stepped",
            "tokens_processed": self.processed_tokens_count,
            "state_energy": round(energy, 4),
            "output_vector": y.tolist(),
            "time_complexity": "O(1)_per_token",
            "memory_complexity": "O(1)_constant",
        }

    def process_sequence(self, sequence: List[List[float]]) -> Dict[str, Any]:
        """Ingests a multi-token sequence (e.g. log file stream) in a single linear pass."""
        outputs = []
        for vec in sequence:
            res = self.step(vec)
            outputs.append(res["output_vector"])

        return {
            "status": "sequence_processed",
            "total_tokens": len(sequence),
            "final_state_energy": round(float(np.linalg.norm(self.hidden_state)), 4),
            "final_output": outputs[-1] if outputs else [],
        }

    def process_sequence_parallel_scan(
        self, sequence: Union[List[List[float]], np.ndarray], force_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Work-Efficient Blelloch / Hillis-Steele Parallel Associative Scan
        in GPU __local SRAM (`O(log2 N)` depth instead of `O(N)` sequential steps).
        """
        from .opencl_microbrain_accelerator import get_opencl_accelerator

        seq_arr = np.asarray(sequence, dtype=np.float32)
        if seq_arr.ndim == 1:
            seq_arr = seq_arr[None, :]
        seq_len, raw_dim = seq_arr.shape
        if seq_len == 0:
            return {
                "status": "sequence_processed_parallel",
                "backend": "CPU",
                "total_tokens": 0,
                "final_state_energy": round(float(np.linalg.norm(self.hidden_state)), 4),
                "final_output": [],
            }

        if raw_dim < self.d_model:
            u_seq = np.pad(seq_arr, ((0, 0), (0, self.d_model - raw_dim)))
        else:
            u_seq = seq_arr[:, : self.d_model]

        # Vectorized ZOH discretization across all tokens in the sequence
        deltas = np.log1p(np.exp(np.clip(self.delta_proj[None, :] + 0.1 * u_seq, -10.0, 10.0)))
        a_bar_seq = np.exp(deltas[:, :, None] * self.A[None, :, :]).astype(np.float32)
        b_u_seq = (deltas[:, :, None] * self.B_proj[None, :, :] * u_seq[:, :, None]).astype(np.float32)

        acc = get_opencl_accelerator()
        h_all, backend = acc.ssm_parallel_scan(a_bar_seq, b_u_seq, self.hidden_state, force_gpu=force_gpu)
        self.hidden_state = h_all[-1].copy()
        self.processed_tokens_count += seq_len

        y_all = np.sum(h_all * self.C_proj[None, :, :], axis=-1) + self.D[None, :] * u_seq
        # The scan runs in 256-token SRAM chunks (chained h0); depth_log2 is the
        # Blelloch depth WITHIN one chunk, not for the whole sequence.
        n_chunks = int(math.ceil(seq_len / 256.0))
        parallel_depth = int(math.ceil(math.log2(max(2, min(256, seq_len)))))
        return {
            "status": "sequence_processed_parallel",
            "backend": backend,
            "total_tokens": seq_len,
            "parallel_scan_chunks": n_chunks,
            "parallel_scan_depth_log2": parallel_depth,
            "final_state_energy": round(float(np.linalg.norm(self.hidden_state)), 4),
            "final_output": y_all[-1].tolist(),
            "all_outputs": y_all,
        }

    def step_mamba3(self, input_vector: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Mamba-3 (2026) State Space Model Step:
        1. Exponential-Trapezoidal discretization for higher-order ODE accuracy:
           h_t = A_bar * R(Delta * omega) * h_{t-1} + B1_bar * u_{t-1} + B2_bar * u_t
        2. Complex-valued pairwise rotational state tracking on (2k, 2k+1) dimensions.
        """
        u = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(u) < self.d_model:
            u = np.pad(u, (0, self.d_model - len(u)))
        else:
            u = u[: self.d_model]

        delta = np.log1p(np.exp(np.clip(self.delta_proj + 0.1 * u, -10.0, 10.0)))
        delta_expanded = delta[:, None]  # (d_model, 1)

        A_bar = np.exp(delta_expanded * self.A)
        if self.has_prev_u:
            B2_bar = 0.5 * delta_expanded * self.B_proj
            B1_bar = 0.5 * delta_expanded * A_bar * self.B_proj
        else:
            B2_bar = delta_expanded * self.B_proj
            B1_bar = np.zeros_like(B2_bar)

        # Complex-valued pairwise rotational state transition
        h_rot = self.hidden_state.copy()
        n_pairs = self.d_state // 2
        if n_pairs > 0:
            theta = delta_expanded * self.omega_rot[:n_pairs][None, :]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            h_even = self.hidden_state[:, 0 : 2 * n_pairs : 2]
            h_odd = self.hidden_state[:, 1 : 2 * n_pairs : 2]
            h_rot[:, 0 : 2 * n_pairs : 2] = h_even * cos_t - h_odd * sin_t
            h_rot[:, 1 : 2 * n_pairs : 2] = h_even * sin_t + h_odd * cos_t

        self.hidden_state = A_bar * h_rot + B1_bar * self.prev_u[:, None] + B2_bar * u[:, None]

        # Lyapunov Bounded Stability Normalization (2025-2026):
        # Prevents long-context recurrent state explosion while preserving relative phase
        h_norm = float(np.linalg.norm(self.hidden_state))
        max_bound = float(self.d_model * self.d_state * 2.0)
        if h_norm > max_bound:
            self.hidden_state = self.hidden_state * (max_bound / (h_norm + 1e-9))

        self.prev_u = u.copy()
        self.has_prev_u = True
        self.processed_tokens_count += 1

        y = np.sum(self.hidden_state * self.C_proj, axis=-1) + self.D * u
        energy = float(np.linalg.norm(self.hidden_state))
        phase_coh = 0.0
        if n_pairs > 0:
            mag = np.sqrt(
                self.hidden_state[:, 0 : 2 * n_pairs : 2] ** 2
                + self.hidden_state[:, 1 : 2 * n_pairs : 2] ** 2
            )
            phase_coh = float(np.mean(mag))

        return {
            "status": "stepped_mamba3",
            "discretization": "exponential_trapezoidal_complex",
            "tokens_processed": self.processed_tokens_count,
            "state_energy": round(energy, 4),
            "phase_coherence": round(phase_coh, 4),
            "output_vector": y.tolist(),
            "readout_trained": self.readout_trained,
            "time_complexity": "O(1)_per_token",
            "memory_complexity": "O(1)_constant",
        }

    def process_sequence_mamba3(self, sequence: Union[List[List[float]], np.ndarray]) -> Dict[str, Any]:
        """Runs a multi-token sequence through Mamba-3 complex-rotational trapezoidal recurrence."""
        outputs = []
        last_res: Dict[str, Any] = {}
        for vec in sequence:
            last_res = self.step_mamba3(vec)
            outputs.append(last_res["output_vector"])

        return {
            "status": "sequence_processed_mamba3",
            "discretization": "exponential_trapezoidal_complex",
            "total_tokens": len(outputs),
            "final_state_energy": round(float(np.linalg.norm(self.hidden_state)), 4),
            "phase_coherence": last_res.get("phase_coherence", 0.0),
            "final_output": outputs[-1] if outputs else [],
            "all_outputs": outputs,
        }

    def fit_readout(
        self,
        sequence: Union[List[List[float]], np.ndarray],
        targets: Union[List[List[float]], np.ndarray],
        l2_reg: float = 1e-3,
        use_mamba3: bool = True,
    ) -> Dict[str, Any]:
        """
        Fits linear readout weights (C_proj, D) via closed-form channel-wise ridge regression
        over the recurrent state trajectory collected from `sequence`.
        """
        seq_arr = np.asarray(sequence, dtype=np.float32)
        if seq_arr.ndim == 1:
            seq_arr = seq_arr[None, :]
        T, raw_u_dim = seq_arr.shape
        if T == 0:
            return {"status": "skipped_empty_sequence", "readout_trained": self.readout_trained}

        if raw_u_dim < self.d_model:
            U = np.pad(seq_arr, ((0, 0), (0, self.d_model - raw_u_dim)))
        else:
            U = seq_arr[:, : self.d_model]

        tgt_arr = np.asarray(targets, dtype=np.float32)
        if tgt_arr.ndim == 1:
            tgt_arr = tgt_arr.reshape(T, -1)
        if tgt_arr.shape[1] < self.d_model:
            Y = np.pad(tgt_arr, ((0, 0), (0, self.d_model - tgt_arr.shape[1])))
        else:
            Y = tgt_arr[:, : self.d_model]

        self.reset()
        states = np.zeros((T, self.d_model, self.d_state), dtype=np.float64)
        pre_outputs = np.zeros((T, self.d_model), dtype=np.float64)
        step_fn = self.step_mamba3 if use_mamba3 else self.step
        for t in range(T):
            step_res = step_fn(U[t])
            states[t] = self.hidden_state.astype(np.float64)
            pre_outputs[t] = np.asarray(step_res["output_vector"], dtype=np.float64)

        pre_mse = float(np.mean((pre_outputs - Y.astype(np.float64)) ** 2))

        U_64 = U.astype(np.float64)
        Y_64 = Y.astype(np.float64)
        reg = float(l2_reg) * np.eye(self.d_state + 1, dtype=np.float64)

        for m in range(self.d_model):
            Phi_m = np.column_stack([states[:, m, :], U_64[:, m]])  # (T, d_state + 1)
            gram = Phi_m.T @ Phi_m + reg
            rhs = Phi_m.T @ Y_64[:, m]
            try:
                w_m = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                w_m, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)
            self.C_proj[m, :] = w_m[: self.d_state].astype(np.float32)
            self.D[m] = np.float32(w_m[self.d_state])

        post_outputs = (
            np.sum(states * self.C_proj[None, :, :].astype(np.float64), axis=-1)
            + U_64 * self.D[None, :].astype(np.float64)
        )
        post_mse = float(np.mean((post_outputs - Y_64) ** 2))
        self.readout_trained = True
        self.readout_mse = round(post_mse, 6)

        return {
            "status": "ssm_readout_fitted",
            "samples": int(T),
            "pre_mse": round(pre_mse, 6),
            "post_mse": round(post_mse, 6),
            "mse_reduced": bool(post_mse <= pre_mse),
            "readout_trained": True,
            "use_mamba3": bool(use_mamba3),
        }

    def reset(self):
        """Resets the recurrent hidden state."""
        self.hidden_state = np.zeros((self.d_model, self.d_state), dtype=np.float32)
        self.prev_u = np.zeros(self.d_model, dtype=np.float32)
        self.has_prev_u = False
        self.processed_tokens_count = 0

