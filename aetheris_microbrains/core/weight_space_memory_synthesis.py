"""
Aetheris Weight-Space Memory Synthesis Engine.
Compresses historical failure and success journals directly into neural weight matrices,
enabling instant sub-millisecond associative recall without parsing large JSONL logs.
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class WeightSpaceMemorySynthesizer:
    """Distills episodic journals into an associative Hopfield/Auto-associative memory matrix."""

    _instance: Optional[WeightSpaceMemorySynthesizer] = None

    def __init__(self, memory_dir: str = "memory/store/neuro_brains", latent_dim: int = 64):
        _p = Path(memory_dir)
        self.memory_dir = _p if (_p.exists() or memory_dir != "memory/store/neuro_brains") else (Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains")
        self.latent_dim = latent_dim
        self.memory_brain_path = self.memory_dir / "neuro_weight_space_memory_meta.json"
        self._init_memory_matrix()

    @classmethod
    def get_instance(cls) -> WeightSpaceMemorySynthesizer:
        if cls._instance is None:
            cls._instance = WeightSpaceMemorySynthesizer()
        return cls._instance

    def _init_memory_matrix(self):
        if not self.memory_brain_path.exists():
            W = np.zeros((self.latent_dim, self.latent_dim), dtype=np.float32)
            meta = {
                "brain_id": "neuro_weight_space_memory",
                "latent_dim": self.latent_dim,
                "d_model": self.latent_dim,
                "vocab_size": self.latent_dim,
                "stored_patterns": 0,
                "conformal_basis": "uncalibrated",
                "calibration_q_hat": None,
                "conformal_quantile": None,
                "domain_name": "weight_space_memory",
                "is_vsa": True,
                "health": "healthy",
                "weights": {"associative_memory": W.tolist()},
            }
            try:
                self.memory_dir.mkdir(parents=True, exist_ok=True)
                with open(self.memory_brain_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

    def _embed_text(self, text: str) -> np.ndarray:
        """Deterministic text embedding into d=64 sphere."""
        vec = np.zeros(self.latent_dim, dtype=np.float32)
        tokens = text.lower().split()
        for idx, tok in enumerate(tokens):
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            pos = h % self.latent_dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[pos] += sign / (1.0 + (idx * 0.05))
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def distill_journal_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compresses a batch of journal entries or failure records into the outer product memory:
        W = W + sum( xi * xi^T ) / N
        """
        if not records:
            return {"status": "skipped", "message": "No records to distill"}

        try:
            with open(self.memory_brain_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        W = np.array(data.get("weights", {}).get("associative_memory", []), dtype=np.float32)
        if W.shape != (self.latent_dim, self.latent_dim):
            W = np.zeros((self.latent_dim, self.latent_dim), dtype=np.float32)

        patterns = list(data.get("patterns", []))
        count = 0
        for r in records:
            text = f"{r.get('goal', '')} {r.get('error', '')} {r.get('rule', '')} {r.get('summary', '')}"
            clean_text = text.strip()
            if len(clean_text) > 5:
                xi = self._embed_text(clean_text)
                # Outer product update (Hebbian associative encoding)
                W += 0.05 * np.outer(xi, xi)
                patterns.append({"vec": xi.tolist(), "summary": clean_text[:120]})
                count += 1

        # Keep bounded FIFO window of pattern exemplars for Modern Hopfield energy recall
        if len(patterns) > 128:
            patterns = patterns[-128:]

        # Trace normalization to prevent spectral blowup
        trace = float(np.trace(W))
        if trace > 10.0:
            W = W * (10.0 / trace)

        data["weights"]["associative_memory"] = W.tolist()
        data["patterns"] = patterns
        data["stored_patterns"] = data.get("stored_patterns", 0) + count
        data["last_distillation"] = os.path.getmtime(self.memory_brain_path) if self.memory_brain_path.exists() else 0
        data["d_model"] = self.latent_dim
        data["latent_dim"] = self.latent_dim
        data["vocab_size"] = len(patterns) or self.latent_dim
        data["conformal_basis"] = "uncalibrated"
        data["calibration_q_hat"] = None
        data["conformal_quantile"] = None
        data["domain_name"] = "weight_space_memory"
        data["is_vsa"] = True
        data["health"] = "healthy"

        with open(self.memory_brain_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return {
            "status": "distilled_success",
            "distilled_records_count": count,
            "total_stored_patterns": data["stored_patterns"],
            "spectral_energy": float(np.linalg.norm(W)),
        }

    def query_associative_memory(self, query_text: str, beta: float = 8.0) -> Dict[str, Any]:
        """Recalls the associative resonance and Modern Continuous Hopfield LogSumExp energy."""
        try:
            with open(self.memory_brain_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"resonance": 0.0, "status": "error"}

        W = np.array(data.get("weights", {}).get("associative_memory", []), dtype=np.float32)
        if W.shape != (self.latent_dim, self.latent_dim):
            W = np.zeros((self.latent_dim, self.latent_dim), dtype=np.float32)
        q = self._embed_text(query_text)

        # Energy recall: E = q^T * W * q
        resonance = float(np.dot(q, np.dot(W, q)))

        # Modern Continuous Hopfield LogSumExp Energy over exemplar bank
        hopfield_energy = 0.0
        top_sim = 0.0
        matched_summary = ""
        patterns = data.get("patterns", [])
        if patterns:
            valid_vecs = []
            valid_summaries = []
            for p in patterns:
                v = np.asarray(p.get("vec", []), dtype=np.float32)
                if v.shape == (self.latent_dim,):
                    valid_vecs.append(v)
                    valid_summaries.append(str(p.get("summary", "")))
            if valid_vecs:
                X = np.stack(valid_vecs, axis=0)
                sims = np.dot(X, q)
                scaled = float(beta) * sims
                max_s = float(np.max(scaled))
                lse = (max_s + float(np.log(np.sum(np.exp(scaled - max_s)) + 1e-12))) / max(float(beta), 1e-6)
                hopfield_energy = -lse + 0.5 * float(np.dot(q, q))
                top_idx = int(np.argmax(sims))
                top_sim = float(sims[top_idx])
                matched_summary = valid_summaries[top_idx]

        return {
            "query": query_text[:80],
            "associative_resonance": round(resonance, 4),
            "is_known_pattern": bool(resonance > 0.45 or top_sim > 0.65),
            "stored_patterns": data.get("stored_patterns", 0),
            "modern_hopfield_energy": round(hopfield_energy, 4),
            "top_pattern_similarity": round(top_sim, 4),
            "matched_record_summary": matched_summary,
        }

