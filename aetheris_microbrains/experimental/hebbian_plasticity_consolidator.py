"""
Hebbian Synaptic Plasticity Consolidator
=======================================
Inspired by NeoAura's `spine_evolution/echo_v10900_sovereign_metabolic.py`
and biological long-term potentiation (LTP).

Reinforces successfully co-activated routing pathways and cognitive tools
using zero-gradient differentiable outer-product updates:
    W_hebbian = gamma * W_hebbian + eta * (z @ z.T)
Allows Aetheris to form persistent associative memory circuits without backpropagation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

STORE_DIR = Path(__file__).resolve().parents[1] / "memory" / "store"
HEBBIAN_WEIGHTS_FILE = STORE_DIR / "wsp_hebbian_traces.npz"
HEBBIAN_METADATA_FILE = STORE_DIR / "wsp_hebbian_meta.json"


class HebbianPlasticityConsolidator:
    """Consolidates successful tool co-occurrences and cognitive traces into associative routing weights."""

    _instance: Optional[HebbianPlasticityConsolidator] = None

    def __init__(self, dim: int = 64, eta: float = 0.05, gamma: float = 0.98):
        self.dim = dim
        self.eta = eta
        self.gamma = gamma
        self.W_hebbian = np.zeros((dim, dim), dtype=np.float32)
        self.tool_index_map: Dict[str, int] = {}
        self.total_consolidations = 0
        self._load()

    @classmethod
    def get_instance(cls) -> HebbianPlasticityConsolidator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_or_assign_index(self, tool_name: str) -> int:
        if tool_name not in self.tool_index_map:
            idx = len(self.tool_index_map) % self.dim
            self.tool_index_map[tool_name] = idx
            return idx
        return self.tool_index_map[tool_name]

    def consolidate_success(self, active_tools: List[str], outcome_quality: float = 1.0) -> Dict[str, Any]:
        """
        Record a successful co-activation of tools.
        Forms an outer-product update on the active tool embedding subspace:
            z_active = sum_i e_{tool_i}
            W_hebbian <- gamma * W_hebbian + eta * (z @ z^T) * quality
        """
        if not active_tools:
            return {"status": "skipped", "reason": "empty active tools"}

        z = np.zeros((self.dim,), dtype=np.float32)
        indices = [self._get_or_assign_index(t) for t in active_tools]
        for idx in indices:
            z[idx] += 1.0

        norm = np.linalg.norm(z) + 1e-8
        z_unit = (z / norm).reshape(-1, 1)

        outer = np.matmul(z_unit, z_unit.T) * float(outcome_quality)
        self.W_hebbian = self.gamma * self.W_hebbian + self.eta * outer

        # SVD stabilization: clamp spectral norm to prevent runaway trace explosion
        s_vals = np.linalg.svd(self.W_hebbian, compute_uv=False)
        top_sigma = float(s_vals[0])
        if top_sigma > 2.0:
            self.W_hebbian = self.W_hebbian * (2.0 / top_sigma)

        self.total_consolidations += 1
        self._save()

        return {
            "status": "consolidated",
            "active_tools": active_tools,
            "indices": indices,
            "spectral_norm": min(top_sigma, 2.0),
            "total_consolidations": self.total_consolidations,
        }

    def predict_associative_boost(self, candidate_tool: str, context_tools: List[str]) -> float:
        """
        Predict the associative compatibility / synergy between a candidate tool
        and the current operational context using the Hebbian synaptic matrix:
            boost = z_candidate^T @ W_hebbian @ z_context
        """
        if not context_tools or candidate_tool not in self.tool_index_map:
            return 0.0

        cand_idx = self.tool_index_map[candidate_tool]
        cand_vec = np.zeros((self.dim,), dtype=np.float32)
        cand_vec[cand_idx] = 1.0

        ctx_vec = np.zeros((self.dim,), dtype=np.float32)
        for t in context_tools:
            if t in self.tool_index_map:
                ctx_vec[self.tool_index_map[t]] += 1.0

        ctx_norm = np.linalg.norm(ctx_vec)
        if ctx_norm < 1e-8:
            return 0.0
        ctx_vec /= ctx_norm

        boost = float(np.dot(cand_vec, np.dot(self.W_hebbian, ctx_vec)))
        return float(round(boost, 4))

    def _save(self):
        try:
            STORE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(HEBBIAN_WEIGHTS_FILE, W_hebbian=self.W_hebbian)
            meta = {
                "updated_at": time.time(),
                "tool_index_map": self.tool_index_map,
                "total_consolidations": self.total_consolidations,
                "dim": self.dim,
            }
            with open(HEBBIAN_METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            if HEBBIAN_WEIGHTS_FILE.exists():
                with np.load(HEBBIAN_WEIGHTS_FILE) as data:
                    loaded_w = data["W_hebbian"].copy()
                if loaded_w.shape == (self.dim, self.dim):
                    self.W_hebbian = loaded_w.astype(np.float32)
            if HEBBIAN_METADATA_FILE.exists():
                with open(HEBBIAN_METADATA_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.tool_index_map = meta.get("tool_index_map", {})
                    self.total_consolidations = meta.get("total_consolidations", 0)
        except Exception:
            pass

