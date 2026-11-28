"""Edge runtime for BitNet-style ternary linears: 2-bit packing, npz artifacts, ONNX export.

P: 2026-11-18 — robotics extension. Complements ``core/bitnet_ternary_engine.py``:
that module quantizes and computes; this module makes ternary layers PORTABLE
and cheap to STORE, which is what an onboard robot runtime needs:

1. ``PackedTernaryLinear`` — a BitNet b1.58 linear with weights stored 2 bits
   per trit (4 trits/byte, same layout as BitNetQuantizer.pack_ternary_2bit),
   plus gamma scale and optional bias. matvec/forward paths replicate the
   BitLinearLayer CPU op order exactly, so the round trip is verified by an
   EXACT integer-parity test (same int32 dot, same scale application).
2. ``save_packed_npz`` / ``load_packed_npz`` — compressed binary artifacts
   (2-bit packed + scale + bias), written atomically via ``core.atomic``.
   This addresses the ~100x JSON-embedding bloat admitted at
   bitnet_ternary_engine's honest on-disk disclosure: a float32 (N,M) weight
   matrix becomes ceil(N*M/4) packed bytes + scalars.
3. ``export_packed_linear_onnx`` — OPTIONAL (requires the ``onnx`` package;
   raises with an install hint when absent, never silently skips). Emits a
   runnable constant-weight MatMul graph. HONESTY NOTE: exported weights are
   DEQUANTIZED (ternary * gamma, float32) — the graph runs on any ONNX
   runtime, but it is not an integer-only graph. Integer-only ONNX
   (MatMulInteger) export is future work; do not claim MCU bitnet parity
   from this export alone.

No latency constants are declared anywhere; ``bench_receipt`` returns MEASURED
numbers for the caller's machine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .atomic import atomic_replace
from .bitnet_ternary_engine import BitLinearLayer, BitNetQuantizer


def _build_trit_4_lut() -> np.ndarray:
    """Precompute 256x4 LUT mapping packed uint8 bytes to {-1, 0, +1} int32 trits."""
    lut = np.zeros((256, 4), dtype=np.int32)
    decode = np.array([-1, 0, 1, 0], dtype=np.int32)
    for b in range(256):
        lut[b, 0] = decode[(b >> 0) & 0x03]
        lut[b, 1] = decode[(b >> 2) & 0x03]
        lut[b, 2] = decode[(b >> 4) & 0x03]
        lut[b, 3] = decode[(b >> 6) & 0x03]
    return lut


_TRIT_4_LUT = _build_trit_4_lut()
_TRIT_4_LUT_T = np.ascontiguousarray(_TRIT_4_LUT.T)


class PackedTernaryLinear:
    """Ternary linear layer with 2-bit-packed storage and exact-parity compute paths."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
    ):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if weights is not None:
            raw = np.asarray(weights, dtype=np.float32).reshape(self.out_features, self.in_features)
        else:
            raw = (np.random.RandomState(1337).randn(self.out_features, self.in_features) * np.sqrt(2.0 / self.in_features)).astype(np.float32)
        if gamma is None:
            gamma = float(np.mean(np.abs(raw)))
            if gamma < 1e-9:
                gamma = 1e-9
        self.gamma = float(gamma)
        self.ternary = np.clip(np.round(raw / self.gamma), -1.0, 1.0).astype(np.int8)
        self.packed = BitNetQuantizer.pack_ternary_2bit(self.ternary)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32).flatten()

    @classmethod
    def from_bitlinear(cls, layer: BitLinearLayer) -> "PackedTernaryLinear":
        """Reuse an existing BitLinearLayer's quantization verbatim."""
        return cls(layer.in_features, layer.out_features, weights=layer.raw_weight, bias=layer.bias, gamma=float(layer.gamma_w))

    def decode_int8(self) -> np.ndarray:
        return BitNetQuantizer.unpack_ternary_2bit(self.packed, self.ternary.shape)

    def _align_x(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
        original_shape = x_arr.shape
        x_2d = x_arr.reshape(-1, original_shape[-1])
        if x_2d.shape[1] < self.in_features:
            x_2d = np.pad(x_2d, ((0, 0), (0, self.in_features - x_2d.shape[1]))).astype(np.float32, copy=False)
        elif x_2d.shape[1] > self.in_features:
            x_2d = x_2d[:, : self.in_features]
        return x_2d, original_shape

    def matvec_quantized(self, x_2d: np.ndarray) -> np.ndarray:
        """EXACT replica of BitLinearLayer CPU path: int8 activations x int8 ternary, int32 dot."""
        max_abs = float(np.max(np.abs(x_2d)))
        if max_abs < 1e-9:
            max_abs = 1e-9
        scale = 127.0 / max_abs
        x_q = np.clip(np.round(x_2d * scale), -127.0, 127.0).astype(np.int8)
        combined_scale = float(self.gamma / scale)
        T = self.decode_int8()
        out_int = np.dot(x_q.astype(np.int32), T.T.astype(np.int32))
        return out_int.astype(np.float32) * combined_scale

    def matvec_packed_lut(self, x_2d: np.ndarray) -> np.ndarray:
        """
        Quantized int8 activation dot product directly from 2-bit ``self.packed``
        using a precomputed 256-entry 4-trit LUT and per-group activation sums,
        without materializing an uncompressed ``(out_features, in_features)`` weight
        matrix. Matches ``matvec_quantized`` bit-for-bit.
        """
        x_arr = np.asarray(x_2d, dtype=np.float32)
        max_abs = float(np.max(np.abs(x_arr)))
        if max_abs < 1e-9:
            max_abs = 1e-9
        scale = 127.0 / max_abs
        x_q = np.clip(np.round(x_arr * scale), -127.0, 127.0).astype(np.int8)
        combined_scale = float(self.gamma / scale)

        batch = int(x_q.shape[0])
        n_groups = (self.in_features + 3) // 4
        if n_groups == 0 or self.out_features == 0:
            return np.zeros((batch, self.out_features), dtype=np.float32)

        pad_cols = n_groups * 4 - self.in_features
        if pad_cols:
            x_pad = np.pad(x_q, ((0, 0), (0, pad_cols)), constant_values=0).astype(np.int32, copy=False)
        else:
            x_pad = x_q.astype(np.int32, copy=False)
        x_groups = x_pad.reshape(batch, n_groups, 4)
        # group_sums[b, g, byte_val] = dot(x_q[b, 4g:4g+4], 4_trits(byte_val))
        group_sums = x_groups @ _TRIT_4_LUT_T

        out_int = np.zeros((batch, self.out_features), dtype=np.int32)
        rem = self.in_features & 3
        if rem == 0:
            packed_2d = self.packed.reshape(self.out_features, n_groups)
            for g in range(n_groups):
                out_int += group_sums[:, g, packed_2d[:, g]]
        else:
            row_starts = np.arange(self.out_features, dtype=np.int64) * self.in_features
            byte_base = row_starts >> 2
            bit_shift = ((row_starts & 3) * 2).astype(np.uint8)
            r_shift = np.where(bit_shift == 0, np.uint8(0), np.uint8(8) - bit_shift)
            packed_len = int(self.packed.size)
            last_mask = np.uint8((1 << (2 * rem)) - 1)
            last_fill = np.uint8(0x55 & (~int(last_mask) & 0xFF))
            for g in range(n_groups):
                idx0 = byte_base + g
                lo = self.packed[idx0]
                has_next = (idx0 + 1) < packed_len
                idx1 = np.where(has_next, idx0 + 1, idx0)
                hi = np.where(has_next, self.packed[idx1], np.uint8(0x55))
                shifted = ((lo >> bit_shift) | (hi << r_shift)).astype(np.uint8)
                aligned = np.where(bit_shift == 0, lo, shifted)
                if g == n_groups - 1:
                    aligned = (aligned & last_mask) | last_fill
                out_int += group_sums[:, g, aligned]

        return out_int.astype(np.float32) * combined_scale

    def forward(self, x: np.ndarray, quantize_activations: bool = True) -> np.ndarray:
        """Batched forward. quantize_activations=False runs the float path (T*gamma) @ x."""
        x_2d, original_shape = self._align_x(x)
        if quantize_activations:
            out = self.matvec_quantized(x_2d)
        else:
            T = self.decode_int8().astype(np.float32)
            out = x_2d @ (T.T * self.gamma)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(list(original_shape[:-1]) + [self.out_features])

    # ---------------- artifacts ----------------

    def save_packed_npz(self, path: Union[str, Path], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Atomically write the compressed 2-bit artifact; returns honest size receipts."""
        path = Path(path)
        tmp = path.with_name(path.name + f".tmp{time.time_ns()}")
        payload: Dict[str, Any] = {
            "packed": self.packed,
            "shape": np.asarray(self.ternary.shape, dtype=np.int64),
            "gamma": np.float64(self.gamma),
            "format_version": np.int64(1),
        }
        if self.bias is not None:
            payload["bias"] = self.bias
        with open(tmp, "wb") as fh:  # file-object form: numpy would append ".npz" to the name otherwise
            np.savez_compressed(fh, **payload)
        atomic_replace(tmp, path)
        packed_bytes = int(self.packed.nbytes)
        float_bytes = int(self.out_features * self.in_features * 4)
        return {
            "status": "saved",
            "path": str(path),
            "packed_bytes": packed_bytes,
            "float32_equivalent_bytes": float_bytes,
            "storage_ratio_vs_float32": round(float_bytes / max(packed_bytes, 1), 1),
            "gamma": self.gamma,
            "meta": meta or {},
        }

    @classmethod
    def load_packed_npz(cls, path: Union[str, Path]) -> "PackedTernaryLinear":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            shape = tuple(int(v) for v in z["shape"])
            gamma = float(z["gamma"])
            bias = z["bias"] if "bias" in z.files else None
            layer = cls.__new__(cls)
            layer.in_features = int(shape[1])
            layer.out_features = int(shape[0])
            layer.gamma = gamma
            layer.packed = z["packed"].astype(np.uint8)
            layer.ternary = BitNetQuantizer.unpack_ternary_2bit(layer.packed, shape)
            layer.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
            return layer

    def stats(self) -> Dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "gamma": self.gamma,
            "packed_bytes": int(self.packed.nbytes),
            "nominal_bits_per_param": 2.0,
            "on_disk_bits_per_param": round(8.0 * self.packed.nbytes / max(self.ternary.size, 1), 2),
            "has_bias": self.bias is not None,
        }


def bench_receipt(layer: PackedTernaryLinear, batch: int = 1, repeats: int = 32) -> Dict[str, Any]:
    """MEASURED latency receipts for the caller's machine (no claimed constants)."""
    x = np.random.default_rng(42).standard_normal((batch, layer.in_features)).astype(np.float32)
    layer.forward(x[:1])  # warm
    t0 = time.perf_counter()
    for _ in range(repeats):
        layer.forward(x, quantize_activations=True)
    q_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    t0 = time.perf_counter()
    for _ in range(repeats):
        layer.forward(x, quantize_activations=False)
    f_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    return {
        "batch": batch,
        "repeats": repeats,
        "quantized_mean_ms": round(q_ms, 4),
        "float_mean_ms": round(f_ms, 4),
        "measured_on": "caller_machine",
    }


def export_packed_linear_onnx(
    layer: PackedTernaryLinear,
    onnx_path: Union[str, Path],
    opset: int = 17,
) -> Dict[str, Any]:
    """
    Export a runnable ONNX graph. Requires the ``onnx`` package (raises with an
    install hint if missing). Weights are exported DEQUANTIZED (ternary*gamma,
    float32 constant) — see module docstring honesty note.
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError(
            "export_packed_linear_onnx requires the 'onnx' package (pip install onnx). "
            "No graph was written."
        ) from exc

    T = layer.decode_int8().astype(np.float32) * np.float32(layer.gamma)
    W_init = numpy_helper.from_array(T.reshape(layer.out_features, layer.in_features), name="W_dequant")
    nodes = [helper.make_node("MatMul", ["X", "W_dequant"], ["Y_mm"])]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, layer.in_features])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, layer.out_features])]
    initializers = [W_init]
    if layer.bias is not None:
        b_init = numpy_helper.from_array(layer.bias.reshape(1, -1), name="B")
        nodes.append(helper.make_node("Add", ["Y_mm", "B"], ["Y"]))
        initializers.append(b_init)
    else:
        nodes[-1].output[0] = "Y"
    graph = helper.make_graph(nodes, "packed_ternary_linear", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version, 8)  # broad runtime compatibility
    onnx.checker.check_model(model)
    onnx_path = Path(onnx_path)
    tmp = onnx_path.with_name(onnx_path.name + f".tmp{time.time_ns()}")
    tmp.write_bytes(model.SerializeToString())
    atomic_replace(tmp, onnx_path)
    return {
        "status": "exported",
        "path": str(onnx_path),
        "bytes": int(onnx_path.stat().st_size),
        "weights_mode": "dequantized_float32_constant",
        "opset": opset,
        "honesty_note": "integer-only MatMulInteger export is future work; not MCU-bitnet-parity",
    }


def export_packed_linear_onnx_int8(
    layer: PackedTernaryLinear,
    onnx_path: Union[str, Path],
    opset: int = 17,
) -> Dict[str, Any]:
    """
    Export a genuine integer-quantized ONNX graph using
    ``DynamicQuantizeLinear -> MatMulInteger -> Cast(FLOAT) -> Mul(scale) -> [Add(bias)]``.
    Requires the ``onnx`` package (raises RuntimeError with an install hint if missing).
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError(
            "export_packed_linear_onnx_int8 requires the 'onnx' package (pip install onnx). "
            "No graph was written."
        ) from exc

    w_int8 = layer.decode_int8().T.astype(np.int8)  # (in_features, out_features)
    w_init = numpy_helper.from_array(w_int8, name="W_int8")
    w_zp_init = numpy_helper.from_array(np.array(0, dtype=np.int8), name="W_zp")
    w_scale_init = numpy_helper.from_array(np.array(layer.gamma, dtype=np.float32), name="W_scale")

    scaled_out_name = "Y_scaled" if layer.bias is not None else "Y"
    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["X"], ["X_q", "x_scale", "x_zp"]),
        helper.make_node("MatMulInteger", ["X_q", "W_int8", "x_zp", "W_zp"], ["Y_int32"]),
        helper.make_node("Cast", ["Y_int32"], ["Y_f32"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["x_scale", "W_scale"], ["combined_scale"]),
        helper.make_node("Mul", ["Y_f32", "combined_scale"], [scaled_out_name]),
    ]
    initializers = [w_init, w_zp_init, w_scale_init]
    if layer.bias is not None:
        b_init = numpy_helper.from_array(layer.bias.reshape(1, -1).astype(np.float32), name="B")
        nodes.append(helper.make_node("Add", [scaled_out_name, "B"], ["Y"]))
        initializers.append(b_init)

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, layer.in_features])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, layer.out_features])]
    graph = helper.make_graph(nodes, "packed_ternary_linear_int8", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version, 8)
    onnx.checker.check_model(model)

    onnx_path = Path(onnx_path)
    tmp = onnx_path.with_name(onnx_path.name + f".tmp{time.time_ns()}")
    tmp.write_bytes(model.SerializeToString())
    atomic_replace(tmp, onnx_path)
    return {
        "status": "exported",
        "path": str(onnx_path),
        "bytes": int(onnx_path.stat().st_size),
        "weights_mode": "int8_matmul_integer",
        "opset": opset,
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "gamma": layer.gamma,
    }


def pack_fleet_artifact_note(meta_json: Dict[str, Any]) -> Dict[str, Any]:
    """Report (honest) what a fleet meta JSON would weigh vs its packed npz twin."""
    json_bytes = len(json.dumps(meta_json).encode("utf-8"))
    return {"meta_json_bytes": json_bytes}
