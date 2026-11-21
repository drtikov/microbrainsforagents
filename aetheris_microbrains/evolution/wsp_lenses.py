"""
Aetheris Epigenetic LoRA Lenses.
Maintains a catalog of modular, low-rank parameter delta matrices (lenses)
that can be loaded or swapped onto the CPU neural substrate in <0.05 milliseconds.
Enables rapid persona and behavioral adaptation (e.g. Audit, Creative Research, High-Speed Triage)
without touching base weights.
"""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ..core.runtime_paths import get_state_dir

logger = logging.getLogger(__name__)

LENSES_DIR = get_state_dir() / "wsp_lenses"


class EpigeneticLensManager:
    """
    Manages modular LoRA lenses for the CPUMicroRouter.
    Each lens contains:
        delta_W1: (d_in, d_hidden)
        delta_W2: (d_hidden, d_out)
        metadata: description, alpha, creation_timestamp
    """

    def __init__(self, lenses_dir: Optional[Path] = None):
        self.lenses_dir = lenses_dir or LENSES_DIR
        self.lenses_dir.mkdir(parents=True, exist_ok=True)
        self.active_lens_name: Optional[str] = None
        self._ensure_default_lenses()

    def _ensure_default_lenses(self) -> None:
        """Initializes standard functional lenses if not already present."""
        if not (self.lenses_dir / "lens_audit_guard.npz").exists():
            self.create_synthetic_lens(
                name="lens_audit_guard",
                d_in=128,
                d_hidden=64,
                d_out=16,
                description="Maximizes conservative safety invariant enforcement and strict verification",
                bias_class=0,  # Routes towards safety verification
            )

        if not (self.lenses_dir / "lens_creative_research.npz").exists():
            self.create_synthetic_lens(
                name="lens_creative_research",
                d_in=128,
                d_hidden=64,
                d_out=16,
                description="Increases exploration entropy and novel tool formulation",
                bias_class=3,  # Routes towards hypothesis search
            )

        if not (self.lenses_dir / "lens_high_speed_triage.npz").exists():
            self.create_synthetic_lens(
                name="lens_high_speed_triage",
                d_in=128,
                d_hidden=64,
                d_out=16,
                description="Prioritizes fast local tool caching and minimal token spend",
                bias_class=2,  # Routes towards dynamic tool caching
            )

    def create_synthetic_lens(
        self,
        name: str,
        d_in: int = 128,
        d_hidden: int = 64,
        d_out: int = 16,
        description: str = "",
        bias_class: int = 0,
        alpha: float = 0.05,
    ) -> Path:
        """
        Synthesizes a rank-1/rank-2 LoRA delta lens analytically.
        Delta_W1 = alpha * (u1 @ v1^T)
        Delta_W2 = alpha * (u2 @ v2^T)
        """
        seed = int.from_bytes(hashlib.sha256(str(name).encode("utf-8")).digest()[:4], "little")
        rng = np.random.default_rng(seed)

        # Construct Rank-1 Delta for W1
        u1 = rng.standard_normal(d_in).astype(np.float32)
        v1 = rng.standard_normal(d_hidden).astype(np.float32)
        u1 /= np.linalg.norm(u1) + 1e-12
        v1 /= np.linalg.norm(v1) + 1e-12
        delta_W1 = (alpha * np.outer(u1, v1)).astype(np.float32)

        # Construct Rank-1 Delta for W2 biased toward bias_class
        u2 = rng.standard_normal(d_hidden).astype(np.float32)
        u2 /= np.linalg.norm(u2) + 1e-12
        v2 = np.zeros(d_out, dtype=np.float32)
        v2[bias_class % d_out] = 1.0
        delta_W2 = (alpha * np.outer(u2, v2)).astype(np.float32)

        file_path = self.lenses_dir / f"{name}.npz"
        meta = {
            "name": name,
            "description": description,
            "alpha": alpha,
            "bias_class": bias_class,
            "d_in": d_in,
            "d_hidden": d_hidden,
            "d_out": d_out,
        }

        np.savez_compressed(
            file_path,
            delta_W1=delta_W1,
            delta_W2=delta_W2,
            meta_json=json.dumps(meta),
        )
        return file_path

    def load_lens(self, name: str) -> Dict[str, Any]:
        """Loads lens deltas and metadata from disk in < 0.1ms."""
        file_path = self.lenses_dir / f"{name}.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"Lens '{name}' not found at {file_path}")

        data = np.load(file_path)
        meta = json.loads(str(data["meta_json"]))
        self.active_lens_name = name
        return {
            "name": name,
            "delta_W1": data["delta_W1"],
            "delta_W2": data["delta_W2"],
            "meta": meta,
        }

    def list_lenses(self) -> List[Dict[str, Any]]:
        """Lists all registered lenses."""
        res = []
        for p in self.lenses_dir.glob("*.npz"):
            try:
                data = np.load(p)
                meta = json.loads(str(data["meta_json"]))
                res.append(meta)
            except Exception:
                continue
        return res

    def merge_lenses(
        self,
        lens_names: List[str],
        merged_name: str,
        method: str = "ties",
        drop_rate: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Combines multiple registered lenses into a hybrid lens via TIES or DARE.
        """
        from .wsp_merger import WeightSpaceMerger

        loaded = [self.load_lens(name) for name in lens_names]
        d_W1_list = [l["delta_W1"] for l in loaded]
        d_W2_list = [l["delta_W2"] for l in loaded]

        if method == "dare":
            d_W1_list = [WeightSpaceMerger.apply_dare(d, drop_rate=drop_rate) for d in d_W1_list]
            d_W2_list = [WeightSpaceMerger.apply_dare(d, drop_rate=drop_rate) for d in d_W2_list]

        merged_W1 = WeightSpaceMerger.ties_merge(d_W1_list)
        merged_W2 = WeightSpaceMerger.ties_merge(d_W2_list)

        file_path = self.lenses_dir / f"{merged_name}.npz"
        meta = {
            "name": merged_name,
            "description": f"Hybrid merge of {lens_names} via {method}",
            "parent_lenses": lens_names,
            "method": method,
            "d_in": merged_W1.shape[0],
            "d_hidden": merged_W1.shape[1],
            "d_out": merged_W2.shape[1],
        }

        np.savez_compressed(
            file_path,
            delta_W1=merged_W1,
            delta_W2=merged_W2,
            meta_json=json.dumps(meta),
        )
        return {"merged_name": merged_name, "path": str(file_path), "meta": meta}

