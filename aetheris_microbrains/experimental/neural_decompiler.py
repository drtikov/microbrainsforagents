"""
Part 2B: Neural Decompiler (Weight-Space to Symbolic Rule Inversion)
===================================================================
Inspired by 2025-2026 decompiler research (Inverting Sub-Symbolic Weights to Logic):
Translates learned synaptic connections in micro-transformer weight matrices
BACK into human-readable Python logic rules and scientific hypotheses.
Completes the bidirectional loop: Code -> Weights -> Self-Evolution -> Decompiled Code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.analytical_micro_transformer import AnalyticalMicroTransformer
from ..core.brain_nursery import AutonomousBrainNursery

logger = logging.getLogger(__name__)


class NeuralDecompiler:
    """Reverse-engineers neural parameter matrices into symbolic Python IF-THEN rules."""

    _instance: Optional[NeuralDecompiler] = None

    def __init__(self):
        self.nursery = AutonomousBrainNursery.get_instance()

    @classmethod
    def get_instance(cls) -> NeuralDecompiler:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def decompile_brain_to_python(self, domain_name: str, activation_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Extracts symbolic associations from W_up and W_down:
        For each neuron j in W_up / W_down:
        - Finds input token k with maximum projection in W_up[:, j].
        - Finds output token v with maximum projection in W_down[j, :].
        - Emits decompiled Python code implementing the learned knowledge.
        """
        clean_name = f"neuro_{domain_name.replace('neuro_', '')}"
        brain: Optional[AnalyticalMicroTransformer] = self.nursery.active_brains.get(clean_name)
        if not brain:
            return {"success": False, "error": f"Brain {clean_name} not found"}

        vocab_tokens = list(brain.vocab.keys())
        discovered_rules: List[Tuple[str, str, float]] = []

        # Analyze each hidden neuron
        for neuron_idx in range(brain.d_mlp):
            w_in = brain.W_up[:, neuron_idx]
            w_out = brain.W_down[neuron_idx, :]

            # Find best input trigger token
            in_scores = []
            for tok in vocab_tokens[3:]:  # skip special tokens
                idx = brain.vocab[tok]
                score = float(np.dot(w_in, brain.W_embed[idx]))
                in_scores.append((tok, score))

            # Find best output target token
            out_scores = []
            for tok in vocab_tokens[3:]:
                idx = brain.vocab[tok]
                score = float(np.dot(w_out, brain.W_embed[idx]))
                out_scores.append((tok, score))

            if in_scores and out_scores:
                in_scores.sort(key=lambda x: x[1], reverse=True)
                out_scores.sort(key=lambda x: x[1], reverse=True)
                best_in, in_strength = in_scores[0]
                best_out, out_strength = out_scores[0]

                confidence = in_strength * out_strength
                if in_strength > activation_threshold and out_strength > activation_threshold and best_in != best_out:
                    discovered_rules.append((best_in, best_out, round(confidence, 3)))

        # Deduplicate rules
        unique_rules = {}
        for trig, targ, conf in discovered_rules:
            if trig not in unique_rules or conf > unique_rules[trig][1]:
                unique_rules[trig] = (targ, conf)

        # Synthesize Python function representation
        lines = [
            f"# Auto-Decompiled Neural Logic from {clean_name}",
            "def neural_router(intent: str) -> str:",
        ]
        for trig, (targ, conf) in unique_rules.items():
            lines.append(f"    if intent == '{trig}':  # confidence: {conf}")
            lines.append(f"        return '{targ}'")
        lines.append("    return 'fallback_default'")

        decompiled_python = "\n".join(lines)
        return {
            "success": True,
            "brain_name": clean_name,
            "rules_count": len(unique_rules),
            "decompiled_rules": unique_rules,
            "synthesized_code": decompiled_python,
        }

