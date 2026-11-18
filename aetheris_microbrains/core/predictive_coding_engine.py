"""
Aetheris Hierarchical Predictive Coding Engine.
Inspired by Karl Friston, Rao & Ballard, and Hinton's Forward-Forward local learning.
Eliminates backpropagation and global gradient locking:
Each layer computes local prediction errors: e_l = r_l - hat{r}_l
Synaptic weights update strictly locally via e_l @ r_{l+1}^T.
Allows lock-free, multi-threaded continuous online adaptation on CPU.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class PredictiveCodingLayer:
    """A single hierarchical layer executing local predictive coding inference and weight adaptation."""

    def __init__(self, in_features: int, out_features: int, lr: float = 0.02):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        rng = np.random.RandomState(42)
        # Top-down generative weights (predicts layer below)
        self.W = rng.randn(in_features, out_features).astype(np.float32) / math.sqrt(out_features)
        self.prediction_error = np.zeros(in_features, dtype=np.float32)
        # Per-unit precision weighting (inverse error variance EMA for iPC)
        self.error_variance_ema = np.ones(in_features, dtype=np.float32)
        self.precision = np.ones(in_features, dtype=np.float32)

    def predict_bottom(self, r_top: np.ndarray) -> np.ndarray:
        """Top-down generative prediction: hat{r}_{bottom} = W @ r_top."""
        return np.dot(self.W, r_top)

    def compute_local_error(self, r_bottom: np.ndarray, r_top: np.ndarray) -> np.ndarray:
        """Local prediction error: epsilon = r_bottom - hat{r}_{bottom}."""
        r_pred = self.predict_bottom(r_top)
        self.prediction_error = r_bottom - r_pred
        return self.prediction_error

    def adapt_local_weights(self, r_top: np.ndarray) -> float:
        """
        Local Hebbian-Predictive update:
        ΔW = lr * (epsilon @ r_top^T)
        No backprop or global gradients required.
        """
        dW = self.lr * np.outer(self.prediction_error, r_top)
        self.W += dW
        return float(np.linalg.norm(dW))

class HierarchicalPredictiveCodingEngine:
    """A 3-layer hierarchical predictive network executing lock-free continuous learning."""

    _instance: Optional[HierarchicalPredictiveCodingEngine] = None

    def __init__(self, layer_dims: Optional[List[int]] = None):
        if layer_dims is None:
            layer_dims = [32, 24, 16]
        self.layer_dims = layer_dims

        # Initialize hierarchical layers (Level 0 <-> Level 1 <-> Level 2)
        self.layers = [
            PredictiveCodingLayer(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ]
        self.training_steps = 0

    @classmethod
    def get_instance(cls) -> HierarchicalPredictiveCodingEngine:
        if cls._instance is None:
            cls._instance = HierarchicalPredictiveCodingEngine()
        return cls._instance

    def forward_and_adapt(self, bottom_input: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Runs a two-phase predictive coding cycle:
        1. Feedforward inference (activations propagate up)
        2. Top-down prediction and local error minimization
        """
        x = np.asarray(bottom_input, dtype=np.float32).flatten()
        if len(x) < self.layer_dims[0]:
            r0 = np.pad(x, (0, self.layer_dims[0] - len(x)))
        else:
            r0 = x[:self.layer_dims[0]]

        # Feedforward pass (representations)
        r1 = np.tanh(np.dot(self.layers[0].W.T, r0))
        r2 = np.tanh(np.dot(self.layers[1].W.T, r1))

        # Top-down prediction error computation
        e1 = self.layers[1].compute_local_error(r1, r2)
        e0 = self.layers[0].compute_local_error(r0, r1)

        # Local weight updates without global adjoints
        delta_norm1 = self.layers[1].adapt_local_weights(r2)
        delta_norm0 = self.layers[0].adapt_local_weights(r1)

        self.training_steps += 1
        total_error_energy = float(np.linalg.norm(e0) + np.linalg.norm(e1))

        return {
            "status": "adapted_locally",
            "step": self.training_steps,
            "total_prediction_error": round(total_error_energy, 4),
            "layer0_error_norm": round(float(np.linalg.norm(e0)), 4),
            "layer1_error_norm": round(float(np.linalg.norm(e1)), 4),
            "top_representation": r2.tolist(),
            "gradient_free": True,
        }

    def relax_and_adapt_pff(
        self,
        positive_inputs: Union[List[float], np.ndarray],
        negative_inputs: Optional[Union[List[float], np.ndarray]] = None,
        relax_steps: int = 20,
        state_lr: float = 0.08,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Predictive Forward-Forward (PFF) & Iterative Variational Free-Energy
        Relaxation in GT 710 GPU SRAM/Registers (`predictive_coding_sram_relax`).
        Relaxes hierarchical latent states (32 <-> 24 <-> 16) for `relax_steps` iterations
        before performing contrastive positive/negative local Hebbian updates.
        """
        from .opencl_microbrain_accelerator import get_opencl_accelerator

        pos_arr = np.asarray(positive_inputs, dtype=np.float32)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr[None, :]
        batch_size, raw_dim = pos_arr.shape
        d0 = self.layer_dims[0]
        if raw_dim < d0:
            pos_arr = np.pad(pos_arr, ((0, 0), (0, d0 - raw_dim)))
        else:
            pos_arr = pos_arr[:, :d0]

        if negative_inputs is None:
            # Generate phase-shuffled / sign-corrupted contrastive negative stream
            rng = np.random.default_rng(1337 + self.training_steps)
            neg_arr = -0.8 * np.roll(pos_arr, shift=5, axis=1) + 0.3 * rng.standard_normal(pos_arr.shape).astype(np.float32)
        else:
            neg_arr = np.asarray(negative_inputs, dtype=np.float32)
            if neg_arr.ndim == 1:
                neg_arr = neg_arr[None, :]
            if neg_arr.shape[1] < d0:
                neg_arr = np.pad(neg_arr, ((0, 0), (0, d0 - neg_arr.shape[1])))
            else:
                neg_arr = neg_arr[:, :d0]

        acc = get_opencl_accelerator()
        pos_res = acc.predictive_coding_relax_and_adapt(
            r0_batch=pos_arr,
            w0=self.layers[0].W,
            w1=self.layers[1].W,
            relax_steps=relax_steps,
            state_lr=state_lr,
            force_gpu=force_gpu,
        )
        neg_res = acc.predictive_coding_relax_and_adapt(
            r0_batch=neg_arr,
            w0=self.layers[0].W,
            w1=self.layers[1].W,
            relax_steps=max(4, relax_steps // 2),
            state_lr=state_lr,
            force_gpu=force_gpu,
        )

        # Contrastive Predictive-Hebbian weight update: reinforce positive errors, suppress negative error correlations
        lr0 = self.layers[0].lr
        lr1 = self.layers[1].lr
        dw0_pos = np.matmul(pos_res["e0"].T, pos_res["r1"]) / float(batch_size)
        dw1_pos = np.matmul(pos_res["e1"].T, pos_res["r2"]) / float(batch_size)
        dw0_neg = np.matmul(neg_res["e0"].T, neg_res["r1"]) / float(batch_size)
        dw1_neg = np.matmul(neg_res["e1"].T, neg_res["r2"]) / float(batch_size)

        self.layers[0].W += lr0 * (dw0_pos - 0.25 * dw0_neg)
        self.layers[1].W += lr1 * (dw1_pos - 0.25 * dw1_neg)
        for layer in self.layers:
            s_max = float(np.linalg.norm(layer.W, ord=2))
            if s_max > 3.0:
                layer.W *= np.float32(3.0 / s_max)
        self.training_steps += 1

        goodness_gap = float((pos_res["goodness_l1"] + pos_res["goodness_l2"]) - 0.5 * (neg_res["goodness_l1"] + neg_res["goodness_l2"]))
        return {
            "status": "pff_sram_relaxed",
            "backend": pos_res["backend"],
            "step": self.training_steps,
            "batch_size": batch_size,
            "relax_steps": relax_steps,
            "initial_free_energy": round(pos_res["initial_free_energy"], 6),
            "final_free_energy": round(pos_res["final_free_energy"], 6),
            "free_energy_reduction_pct": round(
                100.0 * (pos_res["initial_free_energy"] - pos_res["final_free_energy"]) / (pos_res["initial_free_energy"] + 1e-12),
                2,
            ),
            "positive_goodness": round(pos_res["goodness_l1"] + pos_res["goodness_l2"], 4),
            "negative_goodness": round(neg_res["goodness_l1"] + neg_res["goodness_l2"], 4),
            "contrastive_goodness_gap": round(goodness_gap, 4),
            "top_representation": pos_res["r2"][0].tolist(),
        }

    def step_incremental_pc(
        self,
        bottom_input: Union[List[float], np.ndarray],
        micro_steps: int = 6,
        state_lr: float = 0.12,
    ) -> Dict[str, Any]:
        """
        2025-2026 SOTA Precision-Weighted Incremental Predictive Coding (iPC):
        Interleaves local variational state inference and precision-modulated
        synaptic updates simultaneously at every micro-step, removing update-locking.
        """
        x = np.asarray(bottom_input, dtype=np.float32).flatten()
        d0 = self.layer_dims[0]
        if len(x) < d0:
            r0 = np.pad(x, (0, d0 - len(x)))
        else:
            r0 = x[:d0]

        r1 = np.tanh(np.dot(self.layers[0].W.T, r0)).astype(np.float32)
        r2 = np.tanh(np.dot(self.layers[1].W.T, r1)).astype(np.float32)

        pi0 = self.layers[0].precision
        pi1 = self.layers[1].precision

        e0_init = r0 - np.dot(self.layers[0].W, r1)
        e1_init = r1 - np.dot(self.layers[1].W, r2)
        initial_fe = float(
            0.5 * (np.sum(pi0 * (e0_init ** 2)) + np.sum(pi1 * (e1_init ** 2)))
        )

        w_lr0 = min(0.05, self.layers[0].lr)
        w_lr1 = min(0.05, self.layers[1].lr)

        e0 = e0_init
        e1 = e1_init
        for _ in range(max(1, int(micro_steps))):
            pe0 = pi0 * e0
            pe1 = pi1 * e1

            grad_r1 = -np.dot(self.layers[0].W.T, pe0) + pe1
            grad_r2 = -np.dot(self.layers[1].W.T, pe1)
            r1 = np.clip(r1 - float(state_lr) * grad_r1, -3.0, 3.0).astype(np.float32)
            r2 = np.clip(r2 - float(state_lr) * grad_r2, -3.0, 3.0).astype(np.float32)

            e0 = r0 - np.dot(self.layers[0].W, r1)
            e1 = r1 - np.dot(self.layers[1].W, r2)

            # Simultaneous interleaved iPC synaptic update (normalized step for stability)
            scale0 = w_lr0 / (1.0 + float(np.dot(r1, r1)))
            scale1 = w_lr1 / (1.0 + float(np.dot(r2, r2)))
            self.layers[0].W += scale0 * np.outer(pi0 * e0, r1)
            self.layers[1].W += scale1 * np.outer(pi1 * e1, r2)

        e0_final = r0 - np.dot(self.layers[0].W, r1)
        e1_final = r1 - np.dot(self.layers[1].W, r2)
        self.layers[0].prediction_error = e0_final
        self.layers[1].prediction_error = e1_final
        final_fe = float(
            0.5 * (np.sum(pi0 * (e0_final ** 2)) + np.sum(pi1 * (e1_final ** 2)))
        )

        # Update adaptive unit precisions (inverse EMA error variance) for subsequent steps
        for l_idx, err_vec in ((0, e0_final), (1, e1_final)):
            layer = self.layers[l_idx]
            layer.error_variance_ema = (0.9 * layer.error_variance_ema + 0.1 * (err_vec ** 2)).astype(np.float32)
            layer.precision = np.clip(1.0 / (0.1 + layer.error_variance_ema), 0.1, 5.0).astype(np.float32)
            s_max = float(np.linalg.norm(layer.W, ord=2))
            if s_max > 3.0:
                layer.W *= np.float32(3.0 / s_max)

        self.training_steps += 1
        return {
            "status": "ipc_interleaved_adapted",
            "step": self.training_steps,
            "micro_steps": int(micro_steps),
            "initial_free_energy": round(initial_fe, 6),
            "final_free_energy": round(final_fe, 6),
            "free_energy_reduced": bool(final_fe <= initial_fe),
            "mean_precision_l0": round(float(np.mean(self.layers[0].precision)), 4),
            "top_representation": r2.tolist(),
        }

    def generate_top_down(self, top_latent: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Top-down generative synthesis from top-level representation r2 -> r1_hat -> r0_hat
        using the hierarchical generative weights W_l.
        """
        z = np.asarray(top_latent, dtype=np.float32).flatten()
        d2 = self.layer_dims[-1]
        if len(z) < d2:
            r2 = np.pad(z, (0, d2 - len(z)))
        else:
            r2 = z[:d2]

        r1_hat = np.tanh(self.layers[1].predict_bottom(r2)).astype(np.float32)
        r0_hat = self.layers[0].predict_bottom(r1_hat).astype(np.float32)
        return {
            "status": "top_down_generated",
            "r2": r2.tolist(),
            "r1_hat": r1_hat.tolist(),
            "r0_hat": r0_hat.tolist(),
            "generated_norm": round(float(np.linalg.norm(r0_hat)), 6),
        }


PredictiveCodingEngine = HierarchicalPredictiveCodingEngine

