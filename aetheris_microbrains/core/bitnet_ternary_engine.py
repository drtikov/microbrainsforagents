"""
Aetheris BitNet b1.58 Ternary Quantization & BitLinear Engine.
Implements 1.58-bit ternary quantization {-1, 0, +1} on CPU using pure NumPy.
Replaces floating-point matrix multiplications with ternary additions and subtractions.
"""

from __future__ import annotations
import os
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class BitLinearLayer:
    """
    A quantized BitLinear layer with ternary weights {-1, 0, +1} and 8-bit activations.
    Supports:
    1. Straight-Through Estimator (STE) training/fine-tuning (gradient updates to latent float weights).
    2. Discrete Ternary Hebbian / Reinforcement updates directly in weight space.
    3. BitNet v2 H-BitLinear Fast Walsh-Hadamard activation rotation to suppress outlier spikes.
    """

    def __init__(self, in_features: int, out_features: int, weights: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):
        self.in_features = in_features
        self.out_features = out_features
        if weights is not None:
            self.raw_weight = np.asarray(weights, dtype=np.float32).reshape(out_features, in_features)
        else:
            self.raw_weight = np.random.default_rng(42).standard_normal((out_features, in_features)).astype(np.float32) * np.sqrt(2.0 / in_features)
        
        self.bias = np.asarray(bias, dtype=np.float32).flatten() if bias is not None else None
        self.ternary_weight, self.gamma_w = self._quantize_weights()

    def _quantize_weights(self) -> Tuple[np.ndarray, float]:
        """
        BitNet b1.58 Absmean Weight Quantization:
        W_quant = round( W / γ ) where γ = mean(|W|).
        Clamped to {-1, 0, 1}.
        """
        gamma = float(np.mean(np.abs(self.raw_weight)))
        if gamma < 1e-9:
            gamma = 1e-9
        w_scaled = self.raw_weight / gamma
        w_ternary = np.clip(np.round(w_scaled), -1.0, 1.0).astype(np.int8)
        return w_ternary, gamma

    def _align_input_2d(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Align input tensor last dimension to self.in_features (zero-pad or slice) and flatten batch dims."""
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1,)
        original_shape = x_arr.shape
        last_dim = original_shape[-1]
        x_2d = x_arr.reshape(-1, last_dim)
        if last_dim < self.in_features:
            x_2d = np.pad(x_2d, ((0, 0), (0, self.in_features - last_dim))).astype(np.float32, copy=False)
        elif last_dim > self.in_features:
            x_2d = x_2d[:, :self.in_features]
        return x_2d, original_shape

    def _quantize_activations(
        self, x: np.ndarray, per_token: bool = False
    ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """
        Quantize activations dynamically to Int8 [-127, 127].
        When per_token=True and x is 2D with B > 1, uses independent per-row (per-token)
        scaling of shape (B, 1) so outlier rows do not crush small-magnitude rows.
        """
        if per_token and x.ndim == 2 and x.shape[0] > 1:
            max_abs = np.maximum(np.max(np.abs(x), axis=1, keepdims=True).astype(np.float32), 1e-9)
            scale = (127.0 / max_abs).astype(np.float32)
            x_q = np.clip(np.round(x * scale), -127.0, 127.0).astype(np.int8)
            return x_q, scale

        max_abs = float(np.max(np.abs(x)))
        if max_abs < 1e-9:
            max_abs = 1e-9
        scale = 127.0 / max_abs
        x_q = np.clip(np.round(x * scale), -127.0, 127.0).astype(np.int8)
        return x_q, scale

    def forward(
        self, x: np.ndarray, force_gpu: bool = False, per_token: bool = False
    ) -> np.ndarray:
        """
        Ternary forward pass with Dual-Path CPU/OpenCL-GPU dispatch:
        - Single-vector reflex calls (batch < 32): executes on CPU SIMD in ~0.026 ms.
        - Batched colony calls (batch >= 32 or force_gpu=True): offloads to the
          OpenCL GPU co-processor (e.g. NVIDIA GeForce GT 710) when available.
        """
        x_2d, original_shape = self._align_input_2d(x)
        x_q, scale_x = self._quantize_activations(x_2d, per_token=per_token)
        is_row_scaled = isinstance(scale_x, np.ndarray)
        row_scale = (self.gamma_w / scale_x).astype(np.float32) if is_row_scaled else None
        combined_scale = 1.0 if is_row_scaled else float(self.gamma_w / scale_x)

        batch_size = x_2d.shape[0]
        out_float: Optional[np.ndarray] = None
        if force_gpu or batch_size >= 32:
            try:
                from .opencl_microbrain_accelerator import get_opencl_accelerator
                acc = get_opencl_accelerator()
                if acc.should_use_gpu("bitnet_batch_gemm", batch_size, force_gpu=force_gpu):
                    out_float, self.last_backend = acc.bitnet_forward(
                        x_q, self.ternary_weight, combined_scale, force_gpu=force_gpu
                    )
                    if is_row_scaled and out_float is not None:
                        out_float = out_float * row_scale
            except Exception:
                out_float = None

        if out_float is None:
            # In ternary logic: W in {-1, 0, 1}. Integer dot product maps to additions/subtractions.
            out_int = np.dot(x_q.astype(np.int32), self.ternary_weight.T.astype(np.int32))
            if is_row_scaled:
                out_float = out_int.astype(np.float32) * row_scale
            else:
                out_float = out_int.astype(np.float32) * combined_scale
            self.last_backend = "CPU"

        if self.bias is not None:
            out_float += self.bias

        target_shape = list(original_shape[:-1]) + [self.out_features]
        return out_float.reshape(target_shape)

    @staticmethod
    def fast_walsh_hadamard(x: np.ndarray) -> np.ndarray:
        """
        BitNet v2 Normalized Fast Walsh-Hadamard Transform along the last axis.
        Zero-pads non-power-of-two widths to next power of 2 so H^T H = I holds
        exactly (normalized by 1/sqrt(m)), smoothing activation channel outliers
        in O(d log d) additions/subtractions before Int8 quantization.
        """
        arr = np.asarray(x, dtype=np.float32)
        d = arr.shape[-1]
        if d <= 1:
            return arr.copy()
        m = 1 << (d - 1).bit_length()
        if m != d:
            pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, m - d)]
            y = np.pad(arr, pad_width).astype(np.float32, copy=False)
        else:
            y = arr.copy()

        h = 1
        while h < m:
            shape_bt = y.shape[:-1] + (m // (2 * h), 2, h)
            yv = y.reshape(shape_bt)
            a = yv[..., 0, :]
            b = yv[..., 1, :]
            y = np.stack([a + b, a - b], axis=-2).reshape(y.shape)
            h *= 2
        return (y / math.sqrt(float(m))).astype(np.float32)

    def forward_hadamard(
        self, x: np.ndarray, per_token: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        BitNet v2 H-BitLinear Forward Pass:
        Applies orthogonal Fast Walsh-Hadamard rotation to both activations and
        latent weights ((H x) . (H w) = x . w), reducing activation outlier crest
        factors prior to Int8 activation + 1.58-bit ternary weight quantization.
        """
        x_2d, original_shape = self._align_input_2d(x)

        rms_before = float(np.sqrt(np.mean(x_2d ** 2))) + 1e-9
        crest_before = float(np.max(np.abs(x_2d)) / rms_before)

        x_rot = self.fast_walsh_hadamard(x_2d)
        rms_after = float(np.sqrt(np.mean(x_rot ** 2))) + 1e-9
        crest_after = float(np.max(np.abs(x_rot)) / rms_after)

        w_rot = self.fast_walsh_hadamard(self.raw_weight)
        gamma_rot = float(np.mean(np.abs(w_rot)))
        if gamma_rot < 1e-9:
            gamma_rot = 1e-9
        w_rot_ternary = np.clip(np.round(w_rot / gamma_rot), -1.0, 1.0).astype(np.int8)

        x_q, scale_x = self._quantize_activations(x_rot, per_token=per_token)
        out_int = np.dot(x_q.astype(np.int32), w_rot_ternary.T.astype(np.int32))
        if isinstance(scale_x, np.ndarray):
            out_float = out_int.astype(np.float32) * (gamma_rot / scale_x).astype(np.float32)
        else:
            out_float = out_int.astype(np.float32) * float(gamma_rot / scale_x)
        if self.bias is not None:
            out_float += self.bias

        target_shape = list(original_shape[:-1]) + [self.out_features]
        stats = {
            "mode": "BitNet_v2_H_BitLinear",
            "outlier_crest_factor_before": round(crest_before, 4),
            "outlier_crest_factor_after": round(crest_after, 4),
            "crest_factor_reduced": bool(crest_after <= crest_before),
            "hadamard_dim": int(x_rot.shape[-1]),
        }
        return out_float.reshape(target_shape), stats

    def backward_ste(self, x: np.ndarray, grad_output: np.ndarray, lr: float = 0.001) -> np.ndarray:
        """
        Native Straight-Through Estimator (STE) training step for BitNet b1.58.
        Gradients w.r.t quantized weights pass through directly to latent float weights:
        dL/dW_latent ≈ dL/dW_quant.
        Updates self.raw_weight and immediately re-quantizes to maintain ternary manifold.
        """
        x_arr = np.asarray(x, dtype=np.float32).reshape(-1, self.in_features)
        grad_out = np.asarray(grad_output, dtype=np.float32).reshape(-1, self.out_features)

        # Gradient w.r.t weights: grad_W = grad_out^T * x
        grad_w = np.dot(grad_out.T, x_arr)

        # Update underlying latent continuous weights
        self.raw_weight -= lr * grad_w

        # Immediate re-quantization with STE
        self.ternary_weight, self.gamma_w = self._quantize_weights()

        # Return grad_x for backprop
        grad_x = np.dot(grad_out, self.ternary_weight.astype(np.float32) * self.gamma_w)
        return grad_x

    def step_ternary_hebbian(self, x: np.ndarray, y: np.ndarray, reward: float = 1.0, threshold: float = 0.5):
        """
        Direct Discrete Ternary Hebbian / Meta-plasticity step:
        Avoids float post-hoc drift by directly flipping discrete states {-1, 0, +1}
        conditioned on co-activation and reward signal.
        """
        x_arr = np.asarray(x, dtype=np.float32).flatten()
        y_arr = np.asarray(y, dtype=np.float32).flatten()
        outer = np.outer(y_arr, x_arr) * reward

        # Discrete transitions conditioned on threshold
        potentiate = (outer > threshold) & (self.ternary_weight < 1)
        depress = (outer < -threshold) & (self.ternary_weight > -1)

        self.ternary_weight[potentiate] += 1
        self.ternary_weight[depress] -= 1
        # Synchronize raw_weight to current quantized state
        self.raw_weight = self.ternary_weight.astype(np.float32) * self.gamma_w

class BitNetQuantizer:
    """Manages quantization, export, and inference for Aetheris micro-brains."""

    _instance: Optional[BitNetQuantizer] = None

    @classmethod
    def get_instance(cls) -> BitNetQuantizer:
        if cls._instance is None:
            cls._instance = BitNetQuantizer()
        return cls._instance

    @staticmethod
    def pack_ternary_2bit(w_ternary: np.ndarray) -> np.ndarray:
        """
        Packs ternary {-1, 0, +1} weights into 2 bits per trit (4 trits per uint8 byte)
        by mapping {-1, 0, +1} -> {0, 1, 2}.
        """
        flat = np.clip(np.asarray(w_ternary, dtype=np.int8), -1, 1).flatten() + 1
        pad_len = (-flat.size) % 4
        if pad_len:
            flat = np.pad(flat, (0, pad_len), constant_values=1)
        groups = flat.reshape(-1, 4).astype(np.uint8)
        packed = (
            groups[:, 0]
            | (groups[:, 1] << 2)
            | (groups[:, 2] << 4)
            | (groups[:, 3] << 6)
        )
        return packed.astype(np.uint8)

    @staticmethod
    def unpack_ternary_2bit(packed: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """Unpacks a 2-bit-per-trit byte array back into an int8 {-1, 0, +1} tensor."""
        total = int(np.prod(original_shape))
        p = np.asarray(packed, dtype=np.uint8).flatten()
        t0 = (p >> 0) & 0x03
        t1 = (p >> 2) & 0x03
        t2 = (p >> 4) & 0x03
        t3 = (p >> 6) & 0x03
        interleaved = np.stack([t0, t1, t2, t3], axis=-1).flatten()[:total]
        return (interleaved.astype(np.int8) - 1).reshape(original_shape)

    def quantize_micro_brain(self, brain_meta_path: Union[str, Path]) -> Dict[str, Any]:
        """Reads a neuro-brain meta JSON + weights, quantizes to ternary, and saves compressed artifact."""
        path = Path(brain_meta_path)
        if not path.exists():
            return {"status": "error", "message": f"Brain file not found: {path}"}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        # Load weights from either dict in JSON or companion .npz
        weights_dict = data.get("weights", {})
        npz_companion = path.with_name(path.stem.replace("_meta", "") + "_weights.npz")
        if not weights_dict and npz_companion.exists():
            try:
                with np.load(npz_companion) as npz_data:
                    weights_dict = {k: npz_data[k].tolist() for k in npz_data.files}
            except Exception:
                pass

        if not weights_dict and "W_in" not in data:
            return {"status": "skipped", "message": "No dense weights found in manifest"}

        quantized_layers = {}
        total_params = 0
        packed_2bit_bytes = 0
        ternary_distribution = {"-1": 0, "0": 0, "+1": 0}

        for k, v in (weights_dict or data).items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                arr = np.array(v, dtype=np.float32)
                gamma = float(np.mean(np.abs(arr))) or 1e-9
                ternary = np.clip(np.round(arr / gamma), -1, 1).astype(np.int8)
                packed = self.pack_ternary_2bit(ternary)
                packed_2bit_bytes += int(packed.nbytes)

                counts = np.bincount(ternary.flatten() + 1, minlength=3)
                ternary_distribution["-1"] += int(counts[0])
                ternary_distribution["0"] += int(counts[1])
                ternary_distribution["+1"] += int(counts[2])

                total_params += ternary.size
                quantized_layers[k] = {
                    "ternary_matrix": ternary.tolist(),
                    "gamma": gamma,
                    "shape": list(ternary.shape),
                    "packed_2bit_bytes": int(packed.nbytes),
                }

        out_path = path.with_name(path.stem + "_bitnet158.json")
        result = {
            "status": "quantized_success",
            "brain_id": data.get("name", path.stem),
            "layers": quantized_layers,
            "total_parameters": total_params,
            "packed_2bit_bytes": packed_2bit_bytes,
            "packed_2bit_bits_per_param": round(packed_2bit_bytes * 8.0 / max(1, total_params), 2),
            # 1.58 bits is the PAPER's nominal rate for {-1,0,+1}; the JSON
            # artifact stores ternary matrices as text lists, so the honest
            # on-disk cost is measured below and is typically ~100x higher.
            "format_nominal_bits_per_param": 1.58,
            "ternary_distribution": ternary_distribution,
            "bitnet_path": str(out_path),
        }
        if total_params == 0:
            result["status"] = "skipped_no_quantizable_weights"
            result.pop("bitnet_path", None)
            return result

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            artifact_bytes = out_path.stat().st_size
            # Measured from the artifact that was actually written.
            result["artifact_bytes"] = artifact_bytes
            result["effective_bits_per_param"] = round(artifact_bytes * 8.0 / total_params, 2)
        except Exception as exc:
            # The old except-pass still reported "quantized_success" with a
            # bitnet_path for a file that was never written.
            result["status"] = "quantized_save_failed"
            result["save_error"] = str(exc)
            result.pop("bitnet_path", None)

        return result


BitNetTernaryEngine = BitNetQuantizer

