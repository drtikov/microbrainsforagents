"""
Aetheris Counterfactual Dreaming Engine (REM Sleep Simulation).
Inspired by 2025-2026 cognitive architectures (MACD, SleepGate, and Anthropic SCM).
Generates synthetic 'what-if' perturbations (input noise, corrupted types, edge cases)
and stress-tests active micro-brains in an offline memory sandbox during sleep.
Strengthens synaptic resilience without modifying production files on disk.
"""

from __future__ import annotations
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .synaptic_metaplasticity_engine import SynapticMetaplasticityEngine

class CounterfactualDreamEngine:
    """Generates synthetic counterfactual scenarios to harden micro-brains during offline sleep."""

    _instance: Optional[CounterfactualDreamEngine] = None

    def __init__(self, brains_dir: str = "memory/store/neuro_brains"):
        _p = Path(brains_dir)
        self.brains_dir = _p if (_p.exists() or brains_dir != "memory/store/neuro_brains") else (Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains")
        self.metaplasticity = SynapticMetaplasticityEngine.get_instance()
        self.dream_cycles_completed = 0

    @classmethod
    def get_instance(cls) -> CounterfactualDreamEngine:
        if cls._instance is None:
            cls._instance = CounterfactualDreamEngine()
        return cls._instance

    def _perturb_token_sequence(self, tokens: List[str], mode: str) -> List[str]:
        """Counterfactually distort a token sequence.

        P: 2026-11-10 - replaces the old vector-space helper whose output
        fed ``np.tanh(stimulus * 1.5)`` instead of any brain.  Perturbations
        operate on the brain's actual input tokens: unknown-token noise,
        token dropout, attention-order swap.
        """
        seq = list(tokens)
        if not seq:
            return seq
        rng = np.random.RandomState(1234 + len(seq))
        if mode == "gaussian_noise":
            # embedding-space noise analogue: inject <unk> tokens
            k = max(1, len(seq) // 4)
            for pos in rng.choice(len(seq), size=k, replace=False):
                seq[int(pos)] = "<unk>"
        elif mode == "dropout_mask":
            kept = [t for t, m in zip(seq, rng.rand(len(seq)) > 0.3) if m]
            seq = kept or seq
        elif mode == "sign_inversion":
            # perturb attention order: swap two positions
            if len(seq) >= 2:
                i, j = rng.choice(len(seq), size=2, replace=False)
                seq[int(i)], seq[int(j)] = seq[int(j)], seq[int(i)]
        return seq

    def run_dream_session(self, max_scenarios: int = 5) -> Dict[str, Any]:
        """
        Executes a REM dreaming pass:
        1. Loads each micro-brain's REAL weights from disk.
        2. Builds the dream stimulus from the brain's own vocabulary and
           perturbs it counterfactually.
        3. Runs the brain's real forward pass on clean vs perturbed input
           and measures whether the prediction survives.
        4. Consolidates via metaplasticity on the brain's real activation
           vectors (mean input embedding -> predicted-token embedding).

        P: 2026-11-10 - honesty rewrite.  The old body never touched the
        brain weights: the stimulus was ``rng.randn(64)``, the "response"
        was ``np.tanh(stimulus * 1.5)`` -- whose normalized dot with the
        stimulus is always high, so every brain was reported stable -- and
        the report claimed ``dream_entropy_reduced = 0.15 * n`` plus a
        constant ``"HARDENED"`` label.  Unloadable brains are now skipped
        (and counted), stability is a real prediction-survival criterion,
        and the consolidation count is what actually ran.
        """
        scenarios_evaluated = 0
        stabilized_brain_names: List[str] = []
        skipped_brains: List[str] = []
        confidence_deltas: List[float] = []
        consolidation_count = 0
        perturbation_modes = ["gaussian_noise", "dropout_mask", "sign_inversion"]

        meta_files = [
            m for m in sorted(self.brains_dir.glob("*_meta.json"))
            if "_bitnet" not in m.stem
        ]
        if not meta_files:
            return {"status": "skipped", "reason": "No micro-brains found on disk"}

        from ..core.analytical_micro_transformer import AnalyticalMicroTransformer

        for meta_path in meta_files[:max_scenarios]:
            brain_name = meta_path.stem.replace("_meta", "")
            brain = AnalyticalMicroTransformer.load(brain_name, self.brains_dir)
            if brain is None:
                skipped_brains.append(brain_name)
                continue

            # Dream stimulus from the brain's OWN vocabulary: the tokens it
            # was compiled to route on -- not random noise.
            content_tokens = [
                tok for tok in brain.vocab
                if tok not in ("<pad>", "<unk>", "<eos>")
            ]
            if not content_tokens:
                skipped_brains.append(brain_name)
                continue
            seq = (content_tokens * 3)[
                : max(2, min(brain.max_seq_len, len(content_tokens) * 2))
            ]

            base_pred = brain.forward(seq)
            base_conf = float(base_pred.get("confidence", 0.0))

            mode = perturbation_modes[scenarios_evaluated % len(perturbation_modes)]
            perturbed_seq = self._perturb_token_sequence(seq, mode)
            pert_pred = brain.forward(perturbed_seq)
            pert_conf = float(pert_pred.get("confidence", 0.0))

            scenarios_evaluated += 1
            confidence_deltas.append(base_conf - pert_conf)

            # Resilience: same prediction under perturbation, without a
            # collapsing confidence margin.
            is_stable = (
                pert_pred.get("predicted_token") == base_pred.get("predicted_token")
                and pert_conf >= 0.5 * base_conf
            )
            if is_stable:
                stabilized_brain_names.append(brain_name)

            # Metaplastic consolidation on REAL activation vectors.
            try:
                idxs = [brain.vocab.get(t, 1) for t in seq]
                input_vec = brain.W_embed[np.asarray(idxs, dtype=int)].mean(axis=0)
                out_idx = int(brain.vocab.get(str(pert_pred.get("predicted_token")), 1))
                output_vec = brain.W_embed[out_idx]
                reward = 1.0 if is_stable else -0.2
                self.metaplasticity.update_synapses_metaplastic(
                    input_activation=input_vec,
                    output_activation=output_vec,
                    reward_signal=reward,
                )
                consolidation_count += 1
            except Exception:
                pass

        self.dream_cycles_completed += 1

        mean_delta = (
            round(sum(confidence_deltas) / len(confidence_deltas), 4)
            if confidence_deltas else 0.0
        )
        return {
            "status": "rem_dreaming_completed",
            "cycle_number": self.dream_cycles_completed,
            "scenarios_evaluated": scenarios_evaluated,
            "brains_skipped_unloadable": skipped_brains,
            "stabilized_brains": stabilized_brain_names,
            "metaplastic_consolidation": (
                f"applied:{consolidation_count}" if consolidation_count else "skipped"
            ),
            "mean_confidence_delta": mean_delta,
        }

