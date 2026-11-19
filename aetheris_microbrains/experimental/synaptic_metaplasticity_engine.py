"""
Aetheris Synaptic Metaplasticity Engine.
Inspired by Genesis and activity-dependent metaplasticity in neuromorphic computing (2025-2026).
Solves catastrophic forgetting: tracks synaptic plasticity states P_{ij} in [0, 1].
Core reasoning pathways consolidate into rigid synapses (low plasticity),
while exploratory pathways remain fluid (high plasticity).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class SynapticMetaplasticityEngine:
    """Manages dual-state synaptic plasticity to protect consolidated knowledge from catastrophic overwrite."""

    _instance: Optional[SynapticMetaplasticityEngine] = None

    def __init__(self, dim: int = 64, consolidation_threshold: float = 5.0, decay_fluidity: float = 0.995):
        self.dim = dim
        self.consolidation_threshold = consolidation_threshold
        self.decay_fluidity = decay_fluidity

        # Weight matrix W and Metaplasticity matrix P (fluidity state in [0.05, 1.0])
        self.W = np.eye(dim, dtype=np.float32) * 0.1
        self.P = np.ones((dim, dim), dtype=np.float32) # Initially all synapses are fluid (P=1.0)
        self.activation_history_count = np.zeros((dim, dim), dtype=np.float32)

    @classmethod
    def get_instance(cls) -> SynapticMetaplasticityEngine:
        if cls._instance is None:
            cls._instance = SynapticMetaplasticityEngine()
        return cls._instance

    def update_synapses_metaplastic(
        self,
        input_activation: Union[List[float], np.ndarray],
        output_activation: Union[List[float], np.ndarray],
        reward_signal: float = 1.0
    ) -> Dict[str, Any]:
        """
        Metaplastic update rule:
        1. Plasticity modulation: Effective_LR = lr * P_{ij}
        2. ΔW_{ij} = Effective_LR * R * (y_i * x_j)
        3. Metaplastic consolidation: P_{ij} decreases as synapse is repeatedly validated
        """
        x = np.asarray(input_activation, dtype=np.float32).flatten()
        y = np.asarray(output_activation, dtype=np.float32).flatten()

        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))
        else:
            x = x[:self.dim]

        if len(y) < self.dim:
            y = np.pad(y, (0, self.dim - len(y)))
        else:
            y = y[:self.dim]

        # Activity correlation
        activity = np.outer(y, x)
        
        # Base learning rate modulated by individual synaptic fluidity P
        lr_base = 0.05
        effective_lr = lr_base * self.P
        
        # Weight update modulated by CBF Spectral Barrier
        dW = effective_lr * float(reward_signal) * activity
        try:
            from ..core.cbf_spectral_barrier import CBFSpectralBarrierEngine
            cbf = CBFSpectralBarrierEngine.get_instance()
            dW, was_projected = cbf.enforce_safe_update(self.W, dW)
        except Exception:
            was_projected = False

        self.W = (self.W + dW).astype(np.float32)
        s_max = float(np.linalg.norm(self.W, ord=2))
        if s_max > 3.0:
            self.W = (self.W * (3.0 / s_max)).astype(np.float32)

        # Prediction-error driven reconsolidation vs activity-dependent stabilization
        if float(reward_signal) >= 0.0:
            self.activation_history_count += np.abs(activity)
            rigid_mask = self.activation_history_count > self.consolidation_threshold
            self.P[rigid_mask] = np.maximum(0.05, self.P[rigid_mask] * self.decay_fluidity)
            reconsolidated_count = 0
        else:
            active_mask = np.abs(activity) > 1e-4
            err_mag = min(abs(float(reward_signal)), 2.0)
            self.P[active_mask] = np.minimum(
                1.0,
                self.P[active_mask] + 0.20 * err_mag * np.clip(np.abs(activity[active_mask]), 0.1, 1.0),
            ).astype(np.float32)
            self.activation_history_count[active_mask] *= 0.70
            reconsolidated_count = int(np.sum(active_mask))

        rigid_count = int(np.sum(self.P <= 0.2))
        fluid_count = int(np.sum(self.P > 0.8))

        return {
            "status": "metaplastic_adapted",
            "weight_norm": round(float(np.linalg.norm(self.W)), 4),
            "rigid_consolidated_synapses": rigid_count,
            "fluid_adaptable_synapses": fluid_count,
            "destabilized_synapses": reconsolidated_count,
            "catastrophic_forgetting_protection": "ACTIVE",
            "mean_fluidity": round(float(np.mean(self.P)), 4),
        }

