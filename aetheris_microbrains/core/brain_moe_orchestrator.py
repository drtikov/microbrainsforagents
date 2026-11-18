"""
Pillar 1: Micro-MoE Federation & Sparse Gating Orchestrator
===========================================================
Inspired by 2025-2026 SOTA research:
- MoEUT (Mixture-of-Experts Universal Transformers)
- MoE++ / Zero-Computation Gating (ICLR 2025)
- Edge0 Streaming MoE

Unifies all autonomous analytical micro-transformers ("neuro-brains")
stored in `memory/store/neuro_brains/` into an integrated Mixture of Experts.
Uses sub-millisecond sparse gating (Top-1 / Top-2 softmax routing) on CPU
to dispatch incoming tasks to the optimal specialist neuro-brain.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer
from .brain_nursery import AutonomousBrainNursery, STORE_DIR

logger = logging.getLogger(__name__)


class MicroMoEOrchestrator:
    """Sparse Gating Router that unifies grown micro-transformers into a cohesive MoE colony."""

    _instance: Optional[MicroMoEOrchestrator] = None

    def __init__(self, d_router: int = 64):
        self.d_router = d_router
        self.nursery = AutonomousBrainNursery.get_instance()
        # Router projection matrix W_gate: R^{d_router x n_experts}
        self.W_gate = np.zeros((d_router, 16), dtype=np.float32)
        self.expert_names: List[str] = []
        self.expert_load_counts: Dict[str, int] = {}
        self.expert_bias: Dict[str, float] = {}
        self._sync_experts()

    @classmethod
    def get_instance(cls) -> MicroMoEOrchestrator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _embed_tokens_deterministic(self, tokens: List[str]) -> np.ndarray:
        """Deterministically embeds tokens & underscores into d_router using SHA-256."""
        vec = np.zeros(self.d_router, dtype=np.float32)
        expanded: List[str] = []
        for tok in tokens:
            t = str(tok).lower().strip()
            if not t or t in ("<pad>", "<unk>", "<eos>"):
                continue
            expanded.append(t)
            if "_" in t:
                expanded.extend(part for part in t.split("_") if len(part) >= 2)
        for idx, tok in enumerate(expanded):
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "little")
            rng = np.random.default_rng(seed)
            weight = 1.0 / (1.0 + 0.05 * idx)
            vec += weight * rng.standard_normal(self.d_router).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        return vec

    def _get_expert_signature_tokens(self, exp_name: str) -> List[str]:
        clean_domain = exp_name.replace("neuro_", "")
        tokens = [exp_name, clean_domain] + [p for p in clean_domain.split("_") if p]
        brain_obj = self.nursery.active_brains.get(exp_name)
        if brain_obj is not None and hasattr(brain_obj, "vocab"):
            tokens.extend(list(brain_obj.vocab.keys())[:32])
        else:
            from . import brain_nursery as nursery_mod
            meta_path = nursery_mod.STORE_DIR / f"{exp_name}_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    vocab = meta.get("vocab", {})
                    if isinstance(vocab, dict):
                        tokens.extend(list(vocab.keys())[:32])
                except Exception:
                    pass
        return tokens

    def _sync_experts(self):
        """Discovers all active neuro-brains and configures deterministic routing slots."""
        brains = self.nursery.list_grown_brains()
        self.expert_names = [b["name"] for b in brains]
        n_exp = max(len(self.expert_names), 4)

        if self.W_gate.shape[1] < n_exp:
            new_gate = np.zeros((self.d_router, n_exp), dtype=np.float32)
            new_gate[:, : self.W_gate.shape[1]] = self.W_gate
            self.W_gate = new_gate

        # Calibrate gating columns from domain name + learned vocabulary tokens
        for idx, exp_name in enumerate(self.expert_names):
            v = self._embed_tokens_deterministic(self._get_expert_signature_tokens(exp_name))
            self.W_gate[:, idx] = v * 2.0

    def route_query(self, query_tokens: List[str], top_k: int = 1, balance_load: bool = True) -> Dict[str, Any]:
        """
        Computes sparse softmax gating scores with DeepSeek-V3 auxiliary-loss-free
        load-balancing bias (decoupled from unbiased softmax routing weights):
            probs = Softmax(x_query @ W_gate)
            select via TopK(x_query @ W_gate + expert_bias)
        Routes to Top-1 (or Top-K) specialized neuro-brains and gathers inferences.
        """
        self._sync_experts()
        if not self.expert_names:
            return {"success": False, "error": "No grown neuro-brains available in nursery"}

        q_unit = self._embed_tokens_deterministic(query_tokens)

        # Compute unbiased gating logits & softmax weights across active experts
        active_gate = self.W_gate[:, : len(self.expert_names)]
        logits = np.matmul(q_unit, active_gate)

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        # Auxiliary-loss-free bias shifts Top-K selection without distorting routing_prob
        if balance_load:
            bias_vec = np.array(
                [self.expert_bias.get(name, 0.0) for name in self.expert_names],
                dtype=np.float32,
            )
            sel_logits = logits + bias_vec
        else:
            sel_logits = logits

        k = min(top_k, len(self.expert_names))
        top_indices = np.argsort(sel_logits)[-k:][::-1]

        dispatched_results = []
        for idx in top_indices:
            exp_name = self.expert_names[idx]
            clean_domain = exp_name.replace("neuro_", "")
            res = self.nursery.infer(clean_domain, query_tokens)
            dispatched_results.append({
                "expert_name": exp_name,
                "routing_prob": float(round(float(probs[idx]), 4)),
                "inference": res,
            })

        def _dispatch_score(r: Dict[str, Any]) -> float:
            inf = r["inference"]
            tok = inf.get("predicted_token")
            tok_mult = 0.2 if tok in (None, "<unk>", "<pad>") else 1.0
            conf = float(inf.get("confidence", 0.0))
            return float(r["routing_prob"]) * (0.5 + 0.5 * conf) * tok_mult

        best = max(dispatched_results, key=_dispatch_score) if len(dispatched_results) > 1 else dispatched_results[0]

        if balance_load and self.expert_names:
            chosen_name = best["expert_name"]
            self.expert_load_counts[chosen_name] = self.expert_load_counts.get(chosen_name, 0) + 1
            loads = [self.expert_load_counts.get(n, 0) for n in self.expert_names]
            mean_load = float(sum(loads)) / float(len(loads))
            for name, load_i in zip(self.expert_names, loads):
                cur_b = self.expert_bias.get(name, 0.0)
                if load_i > mean_load:
                    self.expert_bias[name] = max(-0.20, cur_b - 0.01)
                elif load_i < mean_load:
                    self.expert_bias[name] = min(0.20, cur_b + 0.01)

        return {
            "success": True,
            "chosen_expert": best["expert_name"],
            "routing_confidence": best["routing_prob"],
            "primary_prediction": best["inference"].get("predicted_token"),
            "all_dispatches": dispatched_results,
            "active_experts_count": len(self.expert_names),
        }

    def route_query_hdc(self, query_tokens: List[str], top_k: int = 1) -> Dict[str, Any]:
        """
        Hyperdimensional Computing (VSA) 10,240-Bit Associative Memory Router.
        Encodes each expert's domain + vocabulary tokens into a 10,240-bit bundled
        hypervector (`320 x uint32`) and dispatches hardware bitwise XOR + popcount
        Hamming similarity search on the GT 710 OpenCL GPU.
        """
        from .opencl_microbrain_accelerator import OpenCLMicroBrainAccelerator

        self._sync_experts()
        if not self.expert_names:
            return {"success": False, "error": "No grown neuro-brains available in nursery"}

        accel = OpenCLMicroBrainAccelerator.get_instance()
        brains = self.nursery.list_grown_brains()

        # Build 10,240-bit bundled hypervectors for each expert brain
        expert_hvs = [
            accel.encode_text_hypervector(self._get_expert_signature_tokens(b["name"]))
            for b in brains
        ]

        memory_bank = np.stack(expert_hvs, axis=0).astype(np.uint32)
        query_hv = accel.encode_text_hypervector(query_tokens)

        sims, backend = accel.hdc_search(query_hv, memory_bank)
        k = min(top_k, len(self.expert_names))
        top_indices = np.argsort(sims)[-k:][::-1]

        dispatched_results = []
        for exp_idx in top_indices:
            exp_name = self.expert_names[int(exp_idx)]
            clean_domain = exp_name.replace("neuro_", "")
            res = self.nursery.infer(clean_domain, query_tokens)
            sim_val = float(sims[int(exp_idx)])
            ham_dist = int(round((1.0 - sim_val) * accel.HDC_BITS))
            dispatched_results.append({
                "expert_name": exp_name,
                "hdc_similarity": float(round(sim_val, 4)),
                "hamming_distance": ham_dist,
                "inference": res,
            })

        best = dispatched_results[0]
        return {
            "success": True,
            "chosen_expert": best["expert_name"],
            "hdc_similarity": best["hdc_similarity"],
            "hamming_distance": best["hamming_distance"],
            "primary_prediction": best["inference"].get("predicted_token"),
            "all_dispatches": dispatched_results,
            "active_experts_count": len(self.expert_names),
            "backend": backend,
            "device": accel.device_name if backend == "OpenCL_GPU" else "CPU",
        }


def consult_microbrain(domain: str, tokens: List[str]) -> Dict[str, Any]:
    """Direct domain inference + split-conformal check through AutonomousBrainNursery."""
    nursery = AutonomousBrainNursery.get_instance()
    brain = nursery._get_or_load_or_grow(domain)
    fwd = brain.forward(tokens)
    conf = brain.infer_conformal(tokens)
    pred = fwd.get("predicted_token") or conf.get("predicted_token")
    return {
        **fwd,
        "prediction": pred,
        "prediction_set": conf.get("prediction_set", []),
        "abstained": conf.get("abstained", False),
        "conformal_guarantee_valid": conf.get("conformal_guarantee_valid", False),
        "success": True,
    }


