"""
Aetheris Online Hebbian & Plastic Adaptation Engine.
Provides real-time local weight adaptation (three-factor / eligibility traces)
upon successful tool verification without full backpropagation.
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .cbf_spectral_barrier import CBFSpectralBarrierEngine


class OnlineHebbianAdaptor:
    """Adapts micro-brain weights on-the-fly via generalized Hebbian plasticity."""

    _instance: Optional[OnlineHebbianAdaptor] = None

    def __init__(self, memory_dir: str = "memory/store/neuro_brains"):
        _p = Path(memory_dir)
        self.memory_dir = _p if (_p.exists() or memory_dir != "memory/store/neuro_brains") else (Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains")
        self.learning_rate = 0.005
        self.decay_rate = 0.001

    @classmethod
    def get_instance(cls) -> OnlineHebbianAdaptor:
        if cls._instance is None:
            cls._instance = OnlineHebbianAdaptor()
        return cls._instance

    def apply_plastic_update(
        self,
        brain_name: str,
        input_activation: Union[List[float], np.ndarray],
        output_feedback: Union[List[float], np.ndarray],
        reinforcement_signal: float = 1.0, # Third factor (reward / pass)
        spectral_limit: float = 2.5,
        use_cbf_spectral_barrier: bool = True,
    ) -> Dict[str, Any]:
        """
        Three-factor Hebbian update rule:
        ΔW = η * R * (y * x^T - γ * y^2 * W)  [Oja's rule with reinforcement modulation]
        """
        if float(spectral_limit) <= 0.0:
            raise ValueError("spectral_limit must be positive")
        limit = float(spectral_limit)

        brain_file = self.memory_dir / f"{brain_name}_meta.json"
        if not brain_file.exists() and not brain_name.startswith("neuro_"):
            brain_file = self.memory_dir / f"neuro_{brain_name}_meta.json"
        if not brain_file.exists():
            brain_file = self.memory_dir / f"{brain_name}.json"
        if not brain_file.exists():
            return {"status": "skipped", "reason": f"Brain {brain_name} not found"}

        try:
            with open(brain_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        x = np.asarray(input_activation, dtype=np.float32).flatten()
        y = np.asarray(output_feedback, dtype=np.float32).flatten()
        R = float(reinforcement_signal)
        cbf_engine = CBFSpectralBarrierEngine.get_instance() if use_cbf_spectral_barrier else None
        cbf_projected_layers: List[str] = []

        def _oja_update(W: np.ndarray, layer_id: str) -> np.ndarray:
            out_dim, in_dim = W.shape
            x_proj = x[:in_dim] if len(x) >= in_dim else np.pad(x, (0, in_dim - len(x)))
            y_proj = y[:out_dim] if len(y) >= out_dim else np.pad(y, (0, out_dim - len(y)))
            # Per-neuron (per-row) Oja normalization prevents single active rows from distorting the full matrix
            y_row_sq = np.minimum(y_proj ** 2, 10.0)[:, np.newaxis]
            dW = R * np.outer(y_proj, x_proj) - self.decay_rate * (y_row_sq * W)
            step_delta = (self.learning_rate * dW).astype(np.float64)
            if use_cbf_spectral_barrier and cbf_engine is not None:
                w64 = W.astype(np.float64)
                safe_delta, was_projected = cbf_engine.enforce_safe_update(
                    w64,
                    step_delta,
                    custom_limit=limit,
                )
                W_new = (w64 + safe_delta).astype(np.float32)
                s_vals = np.linalg.svd(W_new.astype(np.float64), compute_uv=False)
                s_max = float(s_vals[0]) if len(s_vals) > 0 else 0.0
                if s_max > limit:
                    W_new = cbf_engine.project_matrix_to_safety_manifold(
                        W_new.astype(np.float64),
                        max_spectral_norm=limit * (1.0 - 1e-7),
                    ).astype(np.float32)
                    was_projected = True
                if was_projected:
                    cbf_projected_layers.append(layer_id)
                return W_new

            W_new = (W + self.learning_rate * dW).astype(np.float32)
            s_max = float(np.linalg.norm(W_new, ord=2))
            if s_max > limit:
                W_new = (W_new * (limit / s_max)).astype(np.float32)
            return W_new

        updated_layers: List[str] = []
        weights = data.get("weights", {})
        if weights:
            for layer_name, w_list in weights.items():
                if isinstance(w_list, list) and len(w_list) > 0 and isinstance(w_list[0], list):
                    W = np.array(w_list, dtype=np.float32)
                    weights[layer_name] = _oja_update(W, layer_name).tolist()
                    updated_layers.append(layer_name)
            data["weights"] = weights
        else:
            # AnalyticalMicroTransformer / BrainNursery stores weights in a companion .npz
            stem = brain_file.stem[:-5] if brain_file.stem.endswith("_meta") else brain_file.stem
            npz_path = brain_file.with_name(f"{stem}_weights.npz")
            if not npz_path.exists():
                return {"status": "skipped", "reason": "No weights to update"}
            try:
                with np.load(npz_path) as z:
                    arrays = {k: z[k].copy() for k in z.files}
            except Exception as e:
                return {"status": "error", "message": str(e)}

            target_matrices = ("W_mlp", "W_O", "W_up", "W_down")
            for layer_name in target_matrices:
                if layer_name in arrays and arrays[layer_name].ndim == 2:
                    arrays[layer_name] = _oja_update(arrays[layer_name].astype(np.float32), layer_name)
                    updated_layers.append(layer_name)

            if not updated_layers:
                return {"status": "skipped", "reason": "No 2D plastic layers found in .npz"}

            try:
                np.savez_compressed(npz_path, **arrays)
                from .brain_nursery import AutonomousBrainNursery
                if AutonomousBrainNursery._instance is not None:
                    active = AutonomousBrainNursery._instance.active_brains.get(stem)
                    if active is not None:
                        for layer_name in updated_layers:
                            if hasattr(active, layer_name):
                                setattr(active, layer_name, arrays[layer_name])
            except Exception as e:
                return {"status": "error", "message": str(e)}

        data["last_hebbian_update"] = time.time()
        data["plasticity_cycles"] = data.get("plasticity_cycles", 0) + 1

        try:
            with open(brain_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        return {
            "status": "success",
            "brain_name": brain_name,
            "updated_layers": updated_layers,
            "cbf_projected_layers": cbf_projected_layers,
            "spectral_limit": limit,
            "use_cbf_spectral_barrier": bool(use_cbf_spectral_barrier),
            "reinforcement_applied": R,
            "cycle": data["plasticity_cycles"],
        }

