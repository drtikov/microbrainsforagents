"""
Aetheris Mixture-of-Depths (MoD) & Dynamic Early-Exit Router.
Inspired by Apple Research & DeepMind 2025-2026 MoD frameworks.
Triage input token and query complexity at Layer 1: simple, high-confidence
queries exit at Layer 1 (one matrix multiply); complex or uncertain queries
propagate through Layers 2-4. Latency is measured per call and reported in
`latency_ms`; no speedup percentage is claimed because none has been measured.
The exit gate is an UNTRAINED random projection (status
"untrained_heuristic_gate") — treat its routing decisions as heuristic.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class MixtureOfDepthsRouter:
    """Evaluates task representation at Layer 1 and dynamically decides early exit vs deep execution."""

    _instance: Optional[MixtureOfDepthsRouter] = None

    def __init__(self, latent_dim: int = 64, exit_threshold: float = 0.85, seed: int = 42):
        self.latent_dim = latent_dim
        self.exit_threshold = exit_threshold
        self.seed = seed
        self.status = "untrained_heuristic_gate"
        
        # Layer 1 parameters (random projection baseline)
        rng = np.random.RandomState(seed)
        self.W_layer1 = rng.randn(latent_dim, latent_dim).astype(np.float32) / math.sqrt(latent_dim)
        self.b_layer1 = np.zeros(latent_dim, dtype=np.float32)

        # Early-exit router head
        self.W_gate = rng.randn(latent_dim, 1).astype(np.float32) / math.sqrt(latent_dim)
        self.b_gate = 0.5  # Bias toward early exit on simple tasks

        # Deep continuation layers (Layers 2, 3, 4)
        self.W_deep = [
            rng.randn(latent_dim, latent_dim).astype(np.float32) / math.sqrt(latent_dim)
            for _ in range(3)
        ]

        # Differentiable Logic Gate Network (DiffLogic / LUT-Net) parameters
        self.n_logic_gates = latent_dim
        self.conn_a = rng.randint(0, latent_dim, size=self.n_logic_gates).astype(np.int32)
        self.conn_b = ((self.conn_a + rng.randint(1, latent_dim, size=self.n_logic_gates)) % latent_dim).astype(np.int32)
        # Softmax logits over the 16 2-input Boolean truth tables per gate
        self.logic_gate_logits = rng.randn(self.n_logic_gates, 16).astype(np.float32) * 0.25

    @classmethod
    def get_instance(cls) -> MixtureOfDepthsRouter:
        if cls._instance is None:
            cls._instance = MixtureOfDepthsRouter()
        return cls._instance

    def forward(self, input_vector: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Executes dynamic compute forward pass:
        1. h1 = GeLU(W_layer1 @ x + b1)
        2. confidence = Sigmoid(W_gate^T @ h1 + b_gate)
        3. If confidence >= exit_threshold: Early Exit at Layer 1
        4. Else: propagate through Layers 2, 3, 4
        """
        t0 = time.perf_counter()
        x = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(x) < self.latent_dim:
            x = np.pad(x, (0, self.latent_dim - len(x)))
        else:
            x = x[:self.latent_dim]

        # Layer 1 transformation
        pre_h1 = np.dot(self.W_layer1, x) + self.b_layer1
        # Fast GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        h1 = 0.5 * pre_h1 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (pre_h1 + 0.044715 * (pre_h1 ** 3))))

        # Compute exit confidence gate
        gate_logit = float(np.dot(h1, self.W_gate).item()) + self.b_gate
        confidence = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, gate_logit))))

        if confidence >= self.exit_threshold:
            # Early exit triggered
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return {
                "exit_layer": 1,
                "early_exited": True,
                "confidence_score": round(confidence, 4),
                "output_vector": h1.tolist(),
                "compute_ratio": 0.25,  # 1 out of 4 layers executed
                "latency_ms": elapsed_ms,
                "gate_status": self.status,
            }

        # Deep execution through layers 2, 3, 4
        h = h1
        for idx, W_l in enumerate(self.W_deep):
            h = np.maximum(0.0, np.dot(W_l, h) + 0.1 * h)  # Residual ReLU block

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "exit_layer": 4,
            "early_exited": False,
            "confidence_score": round(confidence, 4),
            "output_vector": h.tolist(),
            "compute_ratio": 1.0,
            "latency_ms": elapsed_ms,
            "gate_status": self.status,
        }

    def forward_difflogic_bitplane(
        self, batch_inputs: Union[List[List[float]], np.ndarray], force_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Bitplane Differentiable Logic Gate Tree Evaluation on GT 710 OpenCL GPU.
        Packs batches of 32 binary feature vectors into single uint32 words so each GPU
        instruction evaluates 32 parallel samples across the 16 Boolean logic gates.
        """
        from .opencl_microbrain_accelerator import get_opencl_accelerator

        x_mat = np.asarray(batch_inputs, dtype=np.float32)
        if x_mat.ndim == 1:
            x_mat = x_mat[None, :]
        batch_size, raw_dim = x_mat.shape
        if raw_dim < self.latent_dim:
            x_mat = np.pad(x_mat, ((0, 0), (0, self.latent_dim - raw_dim)))
        else:
            x_mat = x_mat[:, : self.latent_dim]

        # Pack sign bits (x >= 0) across batch dimension into uint32 bitplanes (32 samples per word)
        n_words = max(1, int(math.ceil(batch_size / 32.0)))
        padded_batch = n_words * 32
        bits_bool = np.zeros((self.latent_dim, padded_batch), dtype=np.uint8)
        bits_bool[:, :batch_size] = (x_mat.T >= 0.0).astype(np.uint8)
        x_bits = np.packbits(bits_bool, axis=1).view(np.uint32).reshape(self.latent_dim, n_words)

        # Collapse softmax distribution to discrete argmax Boolean gate IDs (0..15)
        gate_ids = np.argmax(self.logic_gate_logits, axis=-1).astype(np.int32)

        acc = get_opencl_accelerator()
        y_bits, backend = acc.difflogic_bitplane_forward(x_bits, self.conn_a, self.conn_b, gate_ids, force_gpu=force_gpu)

        # Unpack gate outputs per sample to compute Layer-1 logic vote confidence
        y_unpacked = np.unpackbits(y_bits.view(np.uint8), axis=1)[:, :batch_size]  # (n_gates, batch_size)
        sample_confidences = np.mean(y_unpacked, axis=0)
        early_exits = (sample_confidences >= self.exit_threshold).tolist()

        return {
            "status": "difflogic_bitplane_evaluated",
            "backend": backend,
            "batch_size": batch_size,
            "packed_uint32_words": n_words,
            "active_gate_ids": gate_ids[:8].tolist(),
            "mean_confidence": round(float(np.mean(sample_confidences)), 4),
            "early_exit_flags": early_exits,
            "early_exit_count": int(sum(early_exits)),
        }

    def adapt_difflogic_gates(
        self,
        a_prob: np.ndarray,
        b_prob: np.ndarray,
        target_prob: np.ndarray,
        lr: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Differentiable relaxation update over the 16 two-input Boolean truth tables:
        Updates `self.logic_gate_logits` so each gate converges to the target Boolean operator.
        """
        a = np.clip(np.asarray(a_prob, dtype=np.float32).flatten()[: self.n_logic_gates], 0.0, 1.0)
        b = np.clip(np.asarray(b_prob, dtype=np.float32).flatten()[: self.n_logic_gates], 0.0, 1.0)
        t = np.clip(np.asarray(target_prob, dtype=np.float32).flatten()[: self.n_logic_gates], 0.0, 1.0)

        # Continuous De Morgan probabilistic relaxations of all 16 Boolean operators
        ab = a * b
        a_not_b = a * (1.0 - b)
        not_a_b = (1.0 - a) * b
        nor = (1.0 - a) * (1.0 - b)
        ops = np.stack([
            np.zeros_like(a),          # 0: FALSE
            ab,                        # 1: AND
            a_not_b,                   # 2: A AND NOT B
            a,                         # 3: A
            not_a_b,                   # 4: NOT A AND B
            b,                         # 5: B
            a_not_b + not_a_b,         # 6: XOR
            1.0 - nor,                 # 7: OR
            nor,                       # 8: NOR
            ab + nor,                  # 9: XNOR
            1.0 - b,                   # 10: NOT B
            1.0 - not_a_b,             # 11: B IMPLIES A
            1.0 - a,                   # 12: NOT A
            1.0 - a_not_b,             # 13: A IMPLIES B
            1.0 - ab,                  # 14: NAND
            np.ones_like(a),           # 15: TRUE
        ], axis=-1)  # (n_gates, 16)

        logits = self.logic_gate_logits - np.max(self.logic_gate_logits, axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= (np.sum(probs, axis=-1, keepdims=True) + 1e-12)

        pred = np.sum(probs * ops, axis=-1)
        err = pred - t
        loss = float(np.mean(err ** 2))

        # Softmax Jacobian gradient step with respect to gate logits
        grad_probs = 2.0 * err[:, None] * ops
        expected_grad = np.sum(probs * grad_probs, axis=-1, keepdims=True)
        grad_logits = probs * (grad_probs - expected_grad)
        self.logic_gate_logits -= float(lr) * grad_logits

        return {
            "loss": round(loss, 6),
            "dominant_gates": np.argmax(self.logic_gate_logits, axis=-1)[:8].tolist(),
        }

    def train_exit_gate(
        self,
        inputs: Union[List[List[float]], np.ndarray],
        target_exit_labels: Union[List[float], np.ndarray],
        lr: float = 0.25,
        epochs: int = 40,
    ) -> Dict[str, Any]:
        """
        Trains the Layer-1 early-exit confidence gate (W_gate, b_gate) via
        binary cross-entropy gradient descent over frozen GeLU Layer-1 features.
        Once fitted, updates self.status to "calibrated_exit_gate".
        """
        x_raw = np.asarray(inputs, dtype=np.float32)
        if x_raw.ndim == 1:
            x_raw = x_raw[None, :]
        else:
            x_raw = x_raw.reshape(x_raw.shape[0], -1)
        n = x_raw.shape[0]
        if n == 0:
            return {"status": self.status, "error": "empty_inputs", "samples": 0}

        if x_raw.shape[1] < self.latent_dim:
            x_mat = np.pad(x_raw, ((0, 0), (0, self.latent_dim - x_raw.shape[1])))
        else:
            x_mat = x_raw[:, : self.latent_dim]

        y_true = np.clip(np.asarray(target_exit_labels, dtype=np.float32).flatten()[:n], 0.0, 1.0).reshape(n, 1)

        pre_h1 = np.dot(x_mat, self.W_layer1.T) + self.b_layer1[None, :]
        h1 = 0.5 * pre_h1 * (
            1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (pre_h1 + 0.044715 * (pre_h1 ** 3)))
        )

        w = self.W_gate.astype(np.float64).copy()
        b = float(self.b_gate)
        h1_64 = h1.astype(np.float64)
        y_64 = y_true.astype(np.float64)

        def _bce(logits: np.ndarray) -> Tuple[float, np.ndarray]:
            clipped = np.clip(logits, -20.0, 20.0)
            probs = 1.0 / (1.0 + np.exp(-clipped))
            eps = 1e-7
            bce = -float(np.mean(y_64 * np.log(probs + eps) + (1.0 - y_64) * np.log(1.0 - probs + eps)))
            return bce, probs

        pre_loss, _ = _bce(h1_64 @ w + b)

        l2 = 1e-4
        for _ in range(max(1, int(epochs))):
            logits = h1_64 @ w + b
            _, probs = _bce(logits)
            err = (probs - y_64) / float(n)
            grad_w = h1_64.T @ err + l2 * w
            grad_b = float(np.sum(err))
            w -= float(lr) * grad_w
            b -= float(lr) * grad_b

        post_loss, _ = _bce(h1_64 @ w + b)
        self.W_gate = w.astype(np.float32)
        self.b_gate = float(b)
        self.status = "calibrated_exit_gate"

        return {
            "status": self.status,
            "pre_bce_loss": round(pre_loss, 6),
            "post_bce_loss": round(post_loss, 6),
            "loss_reduced": bool(post_loss <= pre_loss),
            "epochs": int(epochs),
            "samples": int(n),
        }


