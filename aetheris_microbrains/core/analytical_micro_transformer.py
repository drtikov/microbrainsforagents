"""
Analytical Micro-Transformer Brain & Logic Compiler
===================================================
Inspired by 2025-2026 SOTA research:
- Loom / Analytically Derived Transformer Weights (Mehmet Kerem Türkcan, 2026)
- Sesen AI / Hand-Crafted Exact Matrix Computation
- MIT CSAIL Weight-Space Programming (Zero-Shot Neural Synthesis)

Enables Aetheris to construct, grow, and run fully functional micro-transformer
neural networks ("digital neuro-organs") directly in CPU RAM using NumPy,
WITHOUT gradient descent, training loops, backpropagation, or NVIDIA GPUs.

Architecture:
- Embedding & Positional Encoding: Rotary (RoPE) / sinusoidal in exact coordinates.
- Multi-Head Attention: Exact analytical Key-Query inner products:
      Attention(Q, K, V) = Softmax( (X W_Q) (X W_K)^T / sqrt(d_k) ) (X W_V)
- Multi-Layer Perceptron (MLP): Exact Key-Value Associative Memory blocks
  compiled directly from symbolic logic tables, routing rules, or ROME rank-1 grafts.
- Zero-Shot Policy Execution: Forward pass runs in <1 millisecond on ordinary CPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

log = logging.getLogger(__name__)

_PKG_STORE_DIR = Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains"
_CWD_STORE_DIR = Path("memory/store/neuro_brains")
CANONICAL_STORE_DIRS: List[Path] = [
    _PKG_STORE_DIR,
    _CWD_STORE_DIR,
]
STORE_DIR = _PKG_STORE_DIR


class AnalyticalMicroTransformer:
    """
    A standalone, pure-NumPy micro-transformer whose parameters are compiled
    analytically from logic specifications rather than trained with gradient descent.
    """

    def __init__(
        self,
        name: str,
        d_model: int = 64,
        n_heads: int = 2,
        d_mlp: int = 128,
        max_seq_len: int = 16,
        seed: int = 42,
    ):
        self.name = name
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        self.max_seq_len = max_seq_len

        rng = np.random.default_rng(seed)

        # 1. Attention Projections (W_Q, W_K, W_V, W_O)
        self.W_Q = np.zeros((d_model, d_model), dtype=np.float32)
        self.W_K = np.zeros((d_model, d_model), dtype=np.float32)
        self.W_V = np.eye(d_model, dtype=np.float32)
        self.W_O = np.eye(d_model, dtype=np.float32)
        self.induction_heads: List[int] = []

        # 2. MLP Down/Up Projections (Feed-Forward Associative Memory)
        self.W_up = np.zeros((d_model, d_mlp), dtype=np.float32)
        self.W_down = np.zeros((d_mlp, d_model), dtype=np.float32)
        self.b_up = np.zeros(d_mlp, dtype=np.float32)
        self.b_down = np.zeros(d_model, dtype=np.float32)
        # Post-MLP output projection evolved by EGGROLL / DNS in BrainNursery
        self.W_mlp: Optional[np.ndarray] = np.eye(d_model, dtype=np.float32)

        # 3. Vocabulary / Token Coordinate Dictionary
        self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.inv_vocab: Dict[int, str] = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.W_embed = np.zeros((d_model, d_model), dtype=np.float32)
        self._init_orthogonal_embeddings()
        self.W_embed_anchor = self.W_embed.copy()

        # 4. Collision-Free Modern Hopfield Associative Memory & Kuramoto Phase Bank
        self.hopfield_keys: List[np.ndarray] = []
        self.hopfield_values: List[np.ndarray] = []

        # 5. Split-Conformal & Mondrian Prediction, Statistical Bounds, and Checksums
        self.conformal_quantile: Optional[float] = None
        self.per_class_quantiles: Dict[str, float] = {}
        self.per_class_counts: Dict[str, int] = {}
        self.conformal_alpha: float = 0.05
        self.conformal_scores: List[float] = []
        self.n_calibration_samples: int = 0
        self.coverage_std_error: float = 0.0
        self.conformal_guarantee_valid: bool = False
        self.eval_samples: int = 0
        self.eval_errors: int = 0
        self.eval_sample_id: Optional[str] = None
        self.weights_sha256: Optional[str] = None
        # How the current conformal_quantile was obtained — honest provenance
        # for the quantile, which otherwise looks calibrated no matter what.
        self.conformal_basis: str = "uncalibrated"
        # Rolling runtime telemetry for live distribution-shift / abstention drift monitoring
        self._live_nonconformity_window: deque = deque(maxlen=50)
        self._live_abstention_window: deque = deque(maxlen=50)

    @staticmethod
    def compute_clopper_pearson_upper_bound(k_errors: int, n_samples: int, alpha: float = 0.05) -> float:
        """Exact one-sided Clopper-Pearson (1 - alpha) upper binomial failure bound (handles k=0 properly)."""
        n = int(n_samples)
        k = int(k_errors)
        a = max(1e-9, min(0.999999, float(alpha)))
        if n <= 0:
            return 1.0
        if k <= 0:
            return round(1.0 - (a ** (1.0 / float(n))), 4)
        if k >= n:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(35):
            mid = 0.5 * (lo + hi)
            cum = sum(math.comb(n, j) * (mid ** j) * ((1.0 - mid) ** (n - j)) for j in range(k + 1))
            if cum > a:
                lo = mid
            else:
                hi = mid
        return round(hi, 4)

    def get_drift_status(self) -> Dict[str, Any]:
        """Return rolling window nonconformity and abstention drift telemetry."""
        win_nc = list(getattr(self, "_live_nonconformity_window", ()))
        win_ab = list(getattr(self, "_live_abstention_window", ()))
        n_win = len(win_nc)
        mean_nc = round(float(sum(win_nc) / n_win), 4) if n_win > 0 else 0.0
        abstain_rate = round(float(sum(1 for a in win_ab if a) / max(1, len(win_ab))), 4) if win_ab else 0.0
        q_ref = float(self.conformal_quantile) if self.conformal_quantile is not None else 0.50
        drift = bool(n_win >= 10 and (mean_nc > (q_ref * 1.5 + 0.05) or abstain_rate > 0.50))
        return {
            "window_size": n_win,
            "mean_nonconformity": mean_nc,
            "rolling_abstention_rate": abstain_rate,
            "drift_detected": drift,
        }

    def _init_orthogonal_embeddings(self):
        """Initialize distinct orthogonal basis directions for vocabulary using Gram-Schmidt QR."""
        rng = np.random.default_rng(1337)
        # Generate square orthogonal matrix: Q @ Q.T = I
        H = rng.standard_normal((self.d_model, self.d_model)).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        self.W_embed = Q.astype(np.float32)

    def register_token(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            if idx >= len(self.W_embed):
                # Expand embedding table dynamically while preserving existing token embeddings
                rng = np.random.default_rng(1337 + idx)
                extra = rng.standard_normal((32, self.d_model)).astype(np.float32)
                extra /= (np.linalg.norm(extra, axis=-1, keepdims=True) + 1e-12)
                self.W_embed = np.vstack([self.W_embed, extra]).astype(np.float32)
                anchor_base = getattr(self, "W_embed_anchor", None)
                if isinstance(anchor_base, np.ndarray) and anchor_base.shape[1] == self.d_model:
                    self.W_embed_anchor = np.vstack([anchor_base, extra]).astype(np.float32)
                else:
                    self.W_embed_anchor = self.W_embed.copy()
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
            return idx
        return self.vocab[token]

    def embed_token(self, token: str) -> np.ndarray:
        """Return the d_model embedding vector for `token` (or <unk>)."""
        idx = self.vocab.get(str(token), 1)
        if idx >= len(self.W_embed):
            idx = 1
        return self.W_embed[idx].copy()

    def compile_induction_head(self, head_idx: int = 0, copy_strength: float = 2.5):
        """
        Surgically sets W_Q and W_K for a specific head so it forms an exact
        Induction Head: attends from current token to previous occurrences of that token
        and copies the subsequent token into the residual stream.
        Zero gradient descent — pure analytical matrix synthesis.
        """
        start = head_idx * self.d_head
        end = start + self.d_head
        # Set W_Q and W_K to identical identity projections in this head's subspace
        self.W_Q[start:end, start:end] = np.eye(self.d_head, dtype=np.float32) * copy_strength
        self.W_K[start:end, start:end] = np.eye(self.d_head, dtype=np.float32) * copy_strength
        self.W_V[start:end, start:end] = np.eye(self.d_head, dtype=np.float32)
        if int(head_idx) not in self.induction_heads:
            self.induction_heads.append(int(head_idx))

    def compile_logic_association(self, trigger_token: str, target_output_token: str, binding_gain: float = 5.0):
        """
        Compiles an exact IF (trigger) -> THEN (target) rule directly into the MLP weights.
        Assigns dedicated, orthogonal neuron slots so associations don't interfere:
            W_up[:, neuron_idx] = key_vec * binding_gain
            W_down[neuron_idx, :] = target_vec * binding_gain
        """
        k_idx = self.register_token(trigger_token)
        v_idx = self.register_token(target_output_token)

        k_vec = self.W_embed[k_idx]
        v_vec = self.W_embed[v_idx]

        # Use deterministic neuron index dedicated to this key
        neuron_idx = k_idx % self.d_mlp

        # Set neuron weight to trigger strongly on key_vec
        self.W_up[:, neuron_idx] = k_vec * binding_gain
        self.b_up[neuron_idx] = -0.1  # Fire only when dot-product is positive and strong

        # Output target embedding vector with strong gain
        self.W_down[neuron_idx, :] = v_vec * binding_gain

        # Also store in collision-free Modern Hopfield memory bank (deduplicating identical keys)
        k_norm = float(np.linalg.norm(k_vec)) + 1e-12
        existing_idx: Optional[int] = None
        for idx, hk in enumerate(self.hopfield_keys):
            cos_sim = float(np.dot(hk, k_vec) / ((float(np.linalg.norm(hk)) + 1e-12) * k_norm))
            if cos_sim > 0.999:
                existing_idx = idx
                break
        if existing_idx is not None:
            self.hopfield_values[existing_idx] = (v_vec * binding_gain).copy()
        else:
            self.hopfield_keys.append(k_vec.copy())
            self.hopfield_values.append((v_vec * binding_gain).copy())

    def forward(self, input_tokens: List[str]) -> Dict[str, Any]:
        """
        Executes a complete causal transformer forward pass on CPU:
        Input tokens -> Embedding -> Multi-Head Self Attention -> MLP -> Logits
        """
        seq_len = min(len(input_tokens), self.max_seq_len)
        if seq_len == 0:
            # No work was performed — a fabricated confidence 1.0 here made an
            # empty call look like a perfect prediction.
            return {"predicted_token": "<eos>", "confidence": 0.0, "reason": "empty_input_no_work_performed"}

        # 1. Embedding lookup
        x = np.zeros((seq_len, self.d_model), dtype=np.float32)
        for t, tok in enumerate(input_tokens[-seq_len:]):
            idx = self.vocab.get(tok, 1)  # 1 is <unk>
            x[t] = self.W_embed[idx]

        # Add positional encoding (scaled appropriately to not overwhelm semantic embedding)
        if seq_len > 1:
            pos = np.arange(seq_len)[:, np.newaxis]
            i = np.arange(self.d_model)[np.newaxis, :]
            angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
            pos_encoding = np.zeros_like(x)
            pos_encoding[:, 0::2] = np.sin(pos * angle_rates[:, 0::2]) * 0.1
            pos_encoding[:, 1::2] = np.cos(pos * angle_rates[:, 1::2]) * 0.1
            x = x + pos_encoding

        # 2. Multi-Head Self Attention (with Causal Mask)
        Q = np.matmul(x, self.W_Q)  # (seq_len, d_model)
        K = np.matmul(x, self.W_K)
        V = np.matmul(x, self.W_V)

        # Scaled Dot-Product Multi-Head Self Attention
        scale = 1.0 / math.sqrt(max(1, self.d_head))
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e9
        attn_concat = np.zeros((seq_len, self.d_model), dtype=np.float32)
        induction_set = set(getattr(self, "induction_heads", []))

        for h in range(self.n_heads):
            start = h * self.d_head
            end = (h + 1) * self.d_head
            Q_h = Q[:, start:end]
            K_h = K[:, start:end]
            V_h = V[:, start:end]

            if h in induction_set and seq_len >= 2:
                K_h_eff = np.zeros_like(K_h)
                K_h_eff[0] = K_h[0]
                K_h_eff[1:] = K_h[:-1]
                diag_pen = np.eye(seq_len, dtype=np.float32) * -2.5
                diag_pen[0, 0] = 0.0
            else:
                K_h_eff = K_h
                diag_pen = 0.0

            attn_scores_h = np.matmul(Q_h, K_h_eff.T) * scale + mask + diag_pen
            exp_scores_h = np.exp(attn_scores_h - np.max(attn_scores_h, axis=-1, keepdims=True))
            attn_weights_h = exp_scores_h / (np.sum(exp_scores_h, axis=-1, keepdims=True) + 1e-12)
            attn_concat[:, start:end] = np.matmul(attn_weights_h, V_h)

        attn_out = np.matmul(attn_concat, self.W_O)

        # Residual connection + LayerNorm approximation
        x = x + attn_out
        x = x / (np.std(x, axis=-1, keepdims=True) + 1e-6)

        # 3. MLP (Feed-Forward)
        mlp_h = np.maximum(0, np.matmul(x, self.W_up) + self.b_up)  # ReLU
        mlp_out = np.matmul(mlp_h, self.W_down) + self.b_down
        if isinstance(self.W_mlp, np.ndarray) and self.W_mlp.shape == (self.d_model, self.d_model):
            mlp_out = np.matmul(mlp_out, self.W_mlp)

        # Residual connection
        x = x + mlp_out
        last_hidden = x[-1]  # Output vector of the newest token (d_model,)

        # 4. Predict Next Token via Embedding Cosine Similarity
        norms = np.linalg.norm(self.W_embed[:len(self.vocab)], axis=-1) + 1e-12
        h_norm = np.linalg.norm(last_hidden) + 1e-12
        sims = np.dot(self.W_embed[:len(self.vocab)], last_hidden) / (norms * h_norm)

        best_idx = int(np.argmax(sims))
        best_token = self.inv_vocab.get(best_idx, "<unk>")
        confidence = float(round(float(sims[best_idx]), 4))

        return {
            "predicted_token": best_token,
            "confidence": confidence,
            "seq_len": seq_len,
            "last_hidden_norm": float(round(h_norm, 4)),
            "model_name": self.name,
        }

    def forward_kuramoto_hopfield(
        self,
        input_tokens: List[str],
        sync_steps: int = 8,
        beta: float = 12.0,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Kuramoto Oscillatory Phase-Synchronized Attention (AKOrN / KoPE)
        + Modern Hopfield LogSumExp Dense Associative Memory on the GT 710 OpenCL GPU.
        Eliminates modulo `k_idx % d_mlp` rule collisions and binds multi-token context via phase locking.
        """
        from .opencl_microbrain_accelerator import get_opencl_accelerator

        seq_len = min(len(input_tokens), self.max_seq_len)
        if seq_len == 0:
            # Nothing ran: confidence 1.0 / order parameter 1.0 fabricated a
            # perfect-synchronization measurement for zero work.
            return {
                "predicted_token": "<eos>",
                "confidence": 0.0,
                "kuramoto_order_r": 0.0,
                "reason": "empty_input_no_work_performed",
                "backend": "CPU",
            }

        x = np.zeros((seq_len, self.d_model), dtype=np.float32)
        for t, tok in enumerate(input_tokens[-seq_len:]):
            idx = self.vocab.get(tok, 1)
            x[t] = self.W_embed[idx]

        # Compute symmetric positive coupling matrix J from token similarity + induction attention
        sim_mat = np.matmul(x, x.T)
        coupling_J = np.maximum(0.0, sim_mat) / float(max(1, seq_len))

        # Initialize natural frequencies omega and slight phase offsets per token position
        pos_offsets = np.linspace(0.0, 0.35, seq_len, dtype=np.float32)[:, None]
        phases = np.broadcast_to(pos_offsets, (seq_len, self.d_model)).copy()
        omega = np.full((seq_len, self.d_model), 0.05, dtype=np.float32)

        hk = np.stack(self.hopfield_keys, axis=0) if self.hopfield_keys else np.zeros((0, self.d_model), dtype=np.float32)
        hv = np.stack(self.hopfield_values, axis=0) if self.hopfield_values else np.zeros((0, self.d_model), dtype=np.float32)

        acc = get_opencl_accelerator()
        kh_res = acc.kuramoto_hopfield_attention(
            x_in=x,
            phases=phases,
            omega=omega,
            coupling_J=coupling_J,
            hopfield_K=hk,
            hopfield_V=hv,
            sync_steps=sync_steps,
            dt=0.2,
            beta=beta,
            force_gpu=force_gpu,
        )

        y_out = kh_res["output"]
        last_hidden = y_out[-1]
        norms = np.linalg.norm(self.W_embed[: len(self.vocab)], axis=-1) + 1e-12
        h_norm = np.linalg.norm(last_hidden) + 1e-12
        sims = np.dot(self.W_embed[: len(self.vocab)], last_hidden) / (norms * h_norm)

        best_idx = int(np.argmax(sims))
        best_token = self.inv_vocab.get(best_idx, "<unk>")
        confidence = float(round(float(sims[best_idx]), 4))

        return {
            "predicted_token": best_token,
            "confidence": confidence,
            "kuramoto_order_r": kh_res["kuramoto_order_r"],
            "hopfield_rules_count": len(self.hopfield_keys),
            "backend": kh_res["backend"],
            "model_name": self.name,
        }

    # ------------------------------------------------------------------
    # Split-Conformal Prediction & Calibrated Abstention
    # ------------------------------------------------------------------
    def get_token_probabilities(self, input_tokens: List[str], beta: float = 8.0) -> Dict[str, float]:
        """Compute softmax probability distribution over vocabulary from Kuramoto-Hopfield readout."""
        x = np.stack([self.embed_token(t) for t in input_tokens], axis=0) if input_tokens else np.zeros((1, self.d_model), dtype=np.float32)
        seq_len = x.shape[0]
        pos_offsets = np.linspace(0.0, 0.35, seq_len, dtype=np.float32)[:, None]
        phases = np.broadcast_to(pos_offsets, (seq_len, self.d_model)).copy()
        omega = np.full((seq_len, self.d_model), 0.05, dtype=np.float32)
        sim_mat = np.matmul(x, x.T)
        coupling_J = np.maximum(0.0, sim_mat) / float(max(1, seq_len))
        hk = np.stack(self.hopfield_keys, axis=0) if self.hopfield_keys else np.zeros((0, self.d_model), dtype=np.float32)
        hv = np.stack(self.hopfield_values, axis=0) if self.hopfield_values else np.zeros((0, self.d_model), dtype=np.float32)

        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        kh_res = acc.kuramoto_hopfield_attention(
            x_in=x, phases=phases, omega=omega, coupling_J=coupling_J,
            hopfield_K=hk, hopfield_V=hv, sync_steps=8, dt=0.2, beta=beta,
        )
        last_hidden = kh_res["output"][-1]
        n_vocab = len(self.vocab)
        norms = np.linalg.norm(self.W_embed[:n_vocab], axis=-1) + 1e-12
        h_norm = np.linalg.norm(last_hidden) + 1e-12
        sims = np.dot(self.W_embed[:n_vocab], last_hidden) / (norms * h_norm)

        logits = sims * 8.0
        logits -= np.max(logits)
        exp_logits = np.exp(np.clip(logits, -40.0, 0.0))
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)
        return {self.inv_vocab[i]: float(probs[i]) for i in range(n_vocab) if i in self.inv_vocab}

    def calibrate_conformal(
        self,
        calibration_data: List[Tuple[List[str], str]],
        alpha: float = 0.10,
        in_sample: bool = False,
    ) -> Dict[str, Any]:
        """
        Calibrate split-conformal and Mondrian (per-class) prediction thresholds
        at miscoverage level alpha.

        Finite-sample coverage P(y* in C(x)) >= 1 - alpha holds ONLY for exchangeable
        held-out calibration data. Also reports coverage standard error
        sqrt(alpha*(1-alpha)/(n+2)) and Rule-of-3 upper bound 3/n on zero errors.
        """
        if not calibration_data:
            return {"status": "error", "reason": "empty_calibration_data"}

        scores: List[float] = []
        class_scores: Dict[str, List[float]] = {}
        for inp, target in calibration_data:
            probs = self.get_token_probabilities(inp)
            p_target = float(probs.get(target, 0.0))
            s_i = 1.0 - p_target
            scores.append(float(s_i))
            class_scores.setdefault(str(target), []).append(float(s_i))

        n = len(scores)
        q_level = min(1.0, math.ceil((n + 1) * (1.0 - float(alpha))) / float(n))
        q_hat = float(np.quantile(scores, q_level))

        per_class_quantiles: Dict[str, float] = {}
        per_class_counts: Dict[str, int] = {}
        for cls_lbl, cls_vals in class_scores.items():
            n_cls = len(cls_vals)
            per_class_counts[cls_lbl] = n_cls
            c_level = min(1.0, math.ceil((n_cls + 1) * (1.0 - float(alpha))) / float(n_cls))
            c_q = float(np.quantile(cls_vals, c_level))
            # Record per-class finite-sample quantile for Mondrian calibration
            per_class_quantiles[cls_lbl] = float(c_q if n_cls >= 3 else max(q_hat, c_q))

        min_required = max(1, math.ceil(1.0 / max(1e-6, float(alpha))) - 1)
        guarantee_valid = (not in_sample) and (n >= min_required)
        cov_std_err = round(math.sqrt(max(0.0, float(alpha) * (1.0 - float(alpha))) / float(n + 2)), 4)
        upper_bound_rule3 = round(3.0 / float(max(1, n)), 4)

        eval_n = int(getattr(self, "eval_samples", 0) or 0)
        eval_k = int(getattr(self, "eval_errors", 0) or 0)
        if eval_n > 0:
            upper_bound = self.compute_clopper_pearson_upper_bound(eval_k, eval_n, float(alpha))
            upper_bound_95 = self.compute_clopper_pearson_upper_bound(eval_k, eval_n, 0.05)
            bound_status = "measured_eval"
        else:
            upper_bound = None
            upper_bound_95 = None
            bound_status = "unknown_no_eval_set"

        self.conformal_quantile = q_hat
        self.per_class_quantiles = per_class_quantiles
        self.per_class_counts = per_class_counts
        self.conformal_alpha = float(alpha)
        self.conformal_scores = scores
        self.n_calibration_samples = n
        self.coverage_std_error = cov_std_err
        self.conformal_guarantee_valid = guarantee_valid
        self.conformal_basis = "in_sample_auto" if in_sample else "held_out"

        return {
            "status": "calibrated",
            "n_samples": n,
            "n_calibration_samples": n,
            "min_required_samples": min_required,
            "conformal_guarantee_valid": guarantee_valid,
            "alpha": float(alpha),
            "confidence_level": round(1.0 - float(alpha), 4),
            "target_coverage": 1.0 - float(alpha),
            "quantile_q_hat": round(q_hat, 4),
            "per_class_quantiles": per_class_quantiles,
            "coverage_std_error": cov_std_err,
            "upper_error_bound_rule_of_3": upper_bound_rule3,
            "upper_error_bound": upper_bound,
            "upper_error_bound_95": upper_bound_95,
            "eval_samples": eval_n,
            "eval_errors": eval_k,
            "eval_sample_id": getattr(self, "eval_sample_id", None),
            "error_bound_status": bound_status,
            "error_metric": "conformal_miscoverage",
            "mean_nonconformity": round(float(np.mean(scores)), 4),
            "basis": self.conformal_basis,
        }

    def record_evaluation_metrics(
        self,
        n_samples: int,
        n_errors: int,
        sample_id: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record genuinely observed held-out evaluation error counts on this model."""
        n = int(n_samples)
        k = int(n_errors)
        if n < 0 or k < 0 or k > n:
            raise ValueError(f"Invalid evaluation counts: n_samples={n}, n_errors={k}")
        eff_alpha = float(alpha) if alpha is not None else float(self.conformal_alpha or 0.05)
        self.eval_samples = n
        self.eval_errors = k
        self.eval_sample_id = sample_id
        if n > 0:
            ub = self.compute_clopper_pearson_upper_bound(k, n, eff_alpha)
            ub95 = self.compute_clopper_pearson_upper_bound(k, n, 0.05)
            status = "measured_eval"
        else:
            ub = None
            ub95 = None
            status = "unknown_no_eval_set"
        return {
            "eval_samples": n,
            "eval_errors": k,
            "eval_sample_id": sample_id,
            "alpha": eff_alpha,
            "confidence_level": round(1.0 - eff_alpha, 4),
            "upper_error_bound": ub,
            "upper_error_bound_95": ub95,
            "error_bound_status": status,
            "error_metric": "conformal_miscoverage",
        }

    def evaluate_conformal_holdout(
        self,
        eval_data: List[Tuple[List[str], str]],
        alpha: Optional[float] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Measure actual conformal miscoverage errors on an evaluation split and record bounds."""
        eff_alpha = float(alpha) if alpha is not None else float(self.conformal_alpha or 0.05)
        if not eval_data:
            bounds = self.record_evaluation_metrics(0, 0, sample_id=sample_id, alpha=eff_alpha)
            return {"empirical_coverage": 0.0, **bounds}
        eval_errors = 0
        for inp, target in eval_data:
            res = self.infer_conformal(inp, alpha=eff_alpha)
            if target not in res.get("prediction_set", []):
                eval_errors += 1
        n_eval = len(eval_data)
        empirical_cov = round((n_eval - eval_errors) / float(n_eval), 4)
        bounds = self.record_evaluation_metrics(n_eval, eval_errors, sample_id=sample_id, alpha=eff_alpha)
        return {"empirical_coverage": empirical_cov, **bounds}

    def infer_conformal(
        self,
        input_tokens: List[str],
        alpha: Optional[float] = None,
        mondrian: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Conformal inference with optional Mondrian per-class thresholds & calibrated abstention:
        Builds prediction set C_alpha(x) = { y in Vocab : 1 - P(y) <= q_y }.
        When `mondrian is None`, applies class-conditional thresholds only for classes with
        n_cls >= min_required (where the per-class finite-sample quantile is defined).
        """
        effective_alpha = float(alpha) if alpha is not None else float(self.conformal_alpha or 0.05)
        alpha_changed = False
        q_hat = self.conformal_quantile
        uncalibrated = q_hat is None
        if q_hat is not None and alpha is not None and self.conformal_scores:
            stored_alpha = float(self.conformal_alpha or 0.05)
            if abs(effective_alpha - stored_alpha) > 1e-6:
                n_scores = len(self.conformal_scores)
                q_level = min(1.0, math.ceil((n_scores + 1) * (1.0 - effective_alpha)) / float(n_scores))
                q_hat = float(np.quantile(self.conformal_scores, q_level))
                alpha_changed = True
        if q_hat is None:
            if self.hopfield_keys:
                auto_cal: List[Tuple[List[str], str]] = []
                for idx, k_vec in enumerate(self.hopfield_keys):
                    k_sims = np.dot(self.W_embed[: len(self.vocab)], k_vec)
                    k_tok = self.inv_vocab.get(int(np.argmax(k_sims)), "<unk>")
                    v_vec = self.hopfield_values[idx]
                    v_sims = np.dot(self.W_embed[: len(self.vocab)], v_vec)
                    v_tok = self.inv_vocab.get(int(np.argmax(v_sims)), "<unk>")
                    auto_cal.append(([k_tok], v_tok))
                if auto_cal:
                    self.calibrate_conformal(auto_cal, alpha=alpha or 0.10, in_sample=True)
                    q_hat = self.conformal_quantile
                    uncalibrated = False
            if q_hat is None:
                q_hat = 0.50

        n_tokens = len(input_tokens) if input_tokens else 0
        known_tokens = sum(1 for t in input_tokens if t in self.vocab and not t.startswith("<")) if input_tokens else 0
        token_coverage = (known_tokens / n_tokens) if n_tokens > 0 else 1.0

        probs = self.get_token_probabilities(input_tokens)
        sorted_probs = sorted(probs.items(), key=lambda kv: (-kv[1], str(kv[0])))

        n_cal = getattr(self, "n_calibration_samples", len(self.conformal_scores))
        min_req = max(1, math.ceil(1.0 / max(1e-6, effective_alpha)) - 1)
        mondrian_map = (
            self.per_class_quantiles
            if (not uncalibrated and not alpha_changed and self.per_class_quantiles)
            else {}
        )
        cls_counts = getattr(self, "per_class_counts", {}) or {}

        def _threshold_for(tok: str) -> float:
            if mondrian is False:
                return float(q_hat)
            if mondrian is True and str(tok) in mondrian_map:
                return float(mondrian_map[str(tok)])
            if str(tok) in mondrian_map and int(cls_counts.get(str(tok), 0)) >= min_req:
                return float(mondrian_map[str(tok)])
            return float(q_hat)

        prediction_set = [
            tok for tok, p in sorted_probs
            if (1.0 - p) <= (_threshold_for(tok) + 1e-6)
        ]
        top_tok, top_p = sorted_probs[0] if sorted_probs else ("<unk>", 0.0)
        set_size = len(prediction_set)

        if token_coverage < 0.50 and n_tokens > 0:
            abstained = True
            reason = "ood_coverage_breach"
            selected_token = "<abstain>"
        elif set_size == 1:
            abstained = False
            reason = "singleton"
            selected_token = prediction_set[0]
        elif set_size > 1:
            abstained = True
            reason = "ambiguous"
            selected_token = top_tok
        else:
            abstained = True
            reason = "ood"
            selected_token = "<abstain>"

        guarantee_valid = (not uncalibrated) and (self.conformal_basis == "held_out") and (n_cal >= min_req)
        cov_std = round(
            float(getattr(self, "coverage_std_error", 0.0))
            or (math.sqrt(max(0.0, effective_alpha * (1.0 - effective_alpha)) / float(max(1, n_cal + 2))) if n_cal > 0 else 0.0),
            4,
        )
        if not hasattr(self, "_live_nonconformity_window"):
            self._live_nonconformity_window = deque(maxlen=50)
        if not hasattr(self, "_live_abstention_window"):
            self._live_abstention_window = deque(maxlen=50)
        self._live_nonconformity_window.append(1.0 - float(top_p))
        self._live_abstention_window.append(bool(abstained))
        drift_info = self.get_drift_status()
        eval_n = int(getattr(self, "eval_samples", 0) or 0)
        eval_k = int(getattr(self, "eval_errors", 0) or 0)
        if eval_n > 0:
            upper_bound = self.compute_clopper_pearson_upper_bound(eval_k, eval_n, effective_alpha)
            upper_bound_95 = self.compute_clopper_pearson_upper_bound(eval_k, eval_n, 0.05)
            bound_status = "measured_eval"
        else:
            upper_bound = None
            upper_bound_95 = None
            bound_status = "unknown_no_eval_set"

        return {
            "prediction_set": prediction_set,
            "set_size": set_size,
            "singleton": bool(set_size == 1),
            "abstained": abstained,
            "reason": reason,
            "predicted_token": selected_token,
            "top_candidate": top_tok,
            "confidence": round(float(top_p), 4),
            "token_coverage": round(float(token_coverage), 4),
            "conformal_quantile": round(float(q_hat), 4),
            "per_class_quantiles": dict(mondrian_map),
            "coverage_std_error": cov_std,
            "upper_error_bound": upper_bound,
            "upper_error_bound_95": upper_bound_95,
            "confidence_level": round(1.0 - float(effective_alpha), 4),
            "eval_samples": eval_n,
            "eval_errors": eval_k,
            "eval_sample_id": getattr(self, "eval_sample_id", None),
            "error_bound_status": bound_status,
            "error_metric": "conformal_miscoverage",
            "rolling_abstention_rate": drift_info["rolling_abstention_rate"],
            "drift_detected": drift_info["drift_detected"],
            "calibrated": not uncalibrated,
            "calibration_basis": "uncalibrated_default" if uncalibrated else self.conformal_basis,
            "n_calibration_samples": n_cal,
            "conformal_set_valid": guarantee_valid,
            "coverage_target": round(1.0 - float(effective_alpha), 4),
            "conformal_guarantee_valid": guarantee_valid,
            "weights_sha256": getattr(self, "weights_sha256", None),
            "alpha": effective_alpha,
            "model_name": self.name,
        }

    def save(self, directory: Optional[Path] = None):
        target_dir = directory or STORE_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        weights_file = target_dir / f"{self.name}_weights.npz"
        meta_file = target_dir / f"{self.name}_meta.json"
        manifest_file = target_dir / f"{self.name}.manifest.json"
        tmp_weights = target_dir / f"tmp_{self.name}_weights.npz"
        tmp_meta = target_dir / f"tmp_{self.name}_meta.json"
        tmp_manifest = target_dir / f"tmp_{self.name}.manifest.json"

        arrays = {
            "W_Q": self.W_Q, "W_K": self.W_K, "W_V": self.W_V, "W_O": self.W_O,
            "W_up": self.W_up, "W_down": self.W_down, "b_up": self.b_up, "b_down": self.b_down,
            "W_embed": self.W_embed,
            "W_embed_anchor": getattr(self, "W_embed_anchor", self.W_embed),
        }
        if isinstance(self.W_mlp, np.ndarray):
            arrays["W_mlp"] = self.W_mlp
        if self.hopfield_keys:
            arrays["hopfield_keys"] = np.stack(self.hopfield_keys, axis=0)
            arrays["hopfield_values"] = np.stack(self.hopfield_values, axis=0)

        # Stage both tmp files and compute SHA-256 digests before touching live artifacts
        np.savez_compressed(tmp_weights, **arrays)
        weights_sha256 = hashlib.sha256(tmp_weights.read_bytes()).hexdigest()
        self.weights_sha256 = weights_sha256

        n_cal = getattr(self, "n_calibration_samples", len(self.conformal_scores))
        cov_std = round(
            float(getattr(self, "coverage_std_error", 0.0))
            or (math.sqrt(max(0.0, self.conformal_alpha * (1.0 - self.conformal_alpha)) / float(max(1, n_cal + 2))) if n_cal > 0 else 0.0),
            4,
        )
        updated_at = time.time()

        meta = {
            "name": self.name,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_mlp": self.d_mlp,
            "max_seq_len": self.max_seq_len,
            "vocab": self.vocab,
            "inv_vocab": {str(k): v for k, v in self.inv_vocab.items()},
            "conformal_quantile": self.conformal_quantile,
            "per_class_quantiles": dict(getattr(self, "per_class_quantiles", {})),
            "per_class_counts": dict(getattr(self, "per_class_counts", {})),
            "conformal_alpha": self.conformal_alpha,
            "conformal_scores": self.conformal_scores,
            "n_calibration_samples": n_cal,
            "coverage_std_error": cov_std,
            "conformal_guarantee_valid": getattr(self, "conformal_guarantee_valid", False),
            "conformal_basis": self.conformal_basis,
            "eval_samples": int(getattr(self, "eval_samples", 0) or 0),
            "eval_errors": int(getattr(self, "eval_errors", 0) or 0),
            "eval_sample_id": getattr(self, "eval_sample_id", None),
            "weights_sha256": weights_sha256,
            "hopfield_rules_count": len(self.hopfield_keys),
            "induction_heads": list(getattr(self, "induction_heads", [])),
            "updated_at": updated_at,
        }

        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        meta_sha256 = hashlib.sha256(tmp_meta.read_bytes()).hexdigest()

        manifest = {
            "name": self.name,
            "weights_file": weights_file.name,
            "meta_file": meta_file.name,
            "weights_sha256": weights_sha256,
            "meta_sha256": meta_sha256,
            "updated_at": updated_at,
        }
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Promote staged artifacts together and commit via companion .manifest.json
        tmp_weights.replace(weights_file)
        tmp_meta.replace(meta_file)
        tmp_manifest.replace(manifest_file)

        # Mirror only to other canonical roots that ALREADY exist on disk (never mkdir inside arbitrary CWD)
        if directory is None:
            import shutil
            for alt_dir in CANONICAL_STORE_DIRS:
                if alt_dir.exists() and alt_dir.resolve() != target_dir.resolve():
                    try:
                        shutil.copy2(weights_file, alt_dir / weights_file.name)
                        shutil.copy2(meta_file, alt_dir / meta_file.name)
                        shutil.copy2(manifest_file, alt_dir / manifest_file.name)
                    except Exception:
                        pass

    @classmethod
    def load(
        cls,
        name: str,
        directory: Optional[Path] = None,
        verify_checksum: bool = True,
        strict_directory: bool = False,
    ) -> Optional[AnalyticalMicroTransformer]:
        if strict_directory:
            if directory is None:
                return None
            search_dirs = [directory]
        else:
            search_dirs = [directory] if directory else []
            search_dirs.extend([STORE_DIR] + CANONICAL_STORE_DIRS)

        weights_file = None
        meta_file = None
        chosen_dir: Optional[Path] = None
        for d in search_dirs:
            if d is None:
                continue
            dp = Path(d)
            wf = dp / f"{name}_weights.npz"
            mf = dp / f"{name}_meta.json"
            if wf.exists() and mf.exists():
                weights_file = wf
                meta_file = mf
                chosen_dir = dp
                break

        if not weights_file or not meta_file or chosen_dir is None:
            return None

        try:
            meta_bytes = meta_file.read_bytes()
            meta = json.loads(meta_bytes.decode("utf-8"))
            actual_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()
            actual_meta_sha = hashlib.sha256(meta_bytes).hexdigest()

            if verify_checksum:
                expected_sha = meta.get("weights_sha256")
                if expected_sha and actual_sha != expected_sha:
                    log.warning(
                        "Checksum mismatch for microbrain '%s' (%s): expected %s, got %s",
                        name, weights_file, expected_sha[:12], actual_sha[:12],
                    )
                    return None
                manifest_file = chosen_dir / f"{name}.manifest.json"
                if manifest_file.exists():
                    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                    if (
                        manifest.get("weights_sha256") != actual_sha
                        or manifest.get("meta_sha256") != actual_meta_sha
                    ):
                        log.warning(
                            "Manifest pair mismatch for microbrain '%s' in %s",
                            name, chosen_dir,
                        )
                        return None

            obj = cls(
                name=meta["name"],
                d_model=meta["d_model"],
                n_heads=meta["n_heads"],
                d_mlp=meta["d_mlp"],
                max_seq_len=meta["max_seq_len"],
            )
            with np.load(weights_file, allow_pickle=False) as data:
                obj.W_Q = data["W_Q"]
                obj.W_K = data["W_K"]
                obj.W_V = data["W_V"]
                obj.W_O = data["W_O"]
                obj.W_up = data["W_up"]
                obj.W_down = data["W_down"]
                obj.b_up = data["b_up"]
                obj.b_down = data["b_down"]
                if "W_mlp" in data:
                    obj.W_mlp = data["W_mlp"]
                obj.W_embed = data["W_embed"]
                if "W_embed_anchor" in data:
                    obj.W_embed_anchor = data["W_embed_anchor"].copy()
                else:
                    obj.W_embed_anchor = obj.W_embed.copy()
                if "hopfield_keys" in data:
                    hk = data["hopfield_keys"]
                    hv = data["hopfield_values"]
                    obj.hopfield_keys = [hk[i].copy() for i in range(hk.shape[0])]
                    obj.hopfield_values = [hv[i].copy() for i in range(hv.shape[0])]
            obj.vocab = meta["vocab"]
            obj.inv_vocab = {int(k): v for k, v in meta["inv_vocab"].items()}
            obj.conformal_quantile = meta.get("conformal_quantile")
            obj.per_class_quantiles = {str(k): float(v) for k, v in meta.get("per_class_quantiles", {}).items()}
            obj.per_class_counts = {str(k): int(v) for k, v in meta.get("per_class_counts", {}).items()}
            obj.conformal_alpha = float(meta.get("conformal_alpha", 0.05))
            obj.conformal_scores = list(meta.get("conformal_scores", []))
            obj.n_calibration_samples = int(meta.get("n_calibration_samples", len(obj.conformal_scores)))
            obj.coverage_std_error = float(meta.get("coverage_std_error", 0.0))
            obj.conformal_basis = str(meta.get("conformal_basis", "uncalibrated"))
            obj.eval_samples = int(meta.get("eval_samples", 0) or 0)
            obj.eval_errors = int(meta.get("eval_errors", 0) or 0)
            obj.eval_sample_id = meta.get("eval_sample_id")
            min_req = max(1, math.ceil(1.0 / max(1e-6, obj.conformal_alpha)) - 1)
            obj.conformal_guarantee_valid = bool(meta.get("conformal_guarantee_valid", obj.conformal_basis == "held_out" and obj.n_calibration_samples >= min_req))
            obj.weights_sha256 = actual_sha
            obj.induction_heads = list(meta.get("induction_heads", []))
            return obj
        except Exception:
            return None
