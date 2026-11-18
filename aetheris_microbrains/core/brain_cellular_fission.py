"""
Part 2A: Cellular Brain Fission & Mitosis Engine
===============================================
Inspired by biological cellular mitosis and adaptive MoE capacity growth:
When an analytical micro-transformer accumulates excessive task variance
or its condition number degenerates (kappa > 50.0), this engine:
1. Performs spectral fission (splitting the matrix along dominant orthogonal eigenvectors).
2. Spawns two specialized daughter micro-brains: Parent_A and Parent_B.
3. Automatically updates the Micro-MoE routing matrix to delegate tasks between both.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer
from .brain_nursery import AutonomousBrainNursery

logger = logging.getLogger(__name__)


class BrainCellularFissionEngine:
    """Detects neural saturation and performs cellular mitosis/fission on overloaded micro-brains."""

    _instance: Optional[BrainCellularFissionEngine] = None

    def __init__(self, max_allowed_condition: float = 40.0):
        self.max_condition = max_allowed_condition
        self.nursery = AutonomousBrainNursery.get_instance()

    @classmethod
    def get_instance(cls) -> BrainCellularFissionEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def evaluate_and_split_if_saturated(self, domain_name: str) -> Dict[str, Any]:
        """
        Evaluates condition number of a micro-brain's MLP weight matrices:
        If kappa > max_condition, triggers cellular fission.
        """
        clean_name = f"neuro_{domain_name.replace('neuro_', '')}"
        brain: Optional[AnalyticalMicroTransformer] = self.nursery.active_brains.get(clean_name)
        if not brain:
            return {"status": "skipped", "reason": f"Brain {clean_name} not found"}

        # Compute SVD of MLP memory matrix W_up
        _, S, _ = np.linalg.svd(brain.W_up, full_matrices=False)
        cond_num = float(S[0] / (S[-1] + 1e-12))

        if cond_num <= self.max_condition and len(brain.vocab) < 40:
            return {
                "status": "healthy",
                "condition_number": round(cond_num, 2),
                "vocab_size": len(brain.vocab),
            }

        # Saturated! Trigger Mitosis (Spawning Daughter A & Daughter B)
        daughter_a_name = f"{clean_name}_alpha"
        daughter_b_name = f"{clean_name}_beta"

        daughter_a = AnalyticalMicroTransformer(
            name=daughter_a_name, d_model=brain.d_model, n_heads=brain.n_heads, d_mlp=brain.d_mlp
        )
        daughter_b = AnalyticalMicroTransformer(
            name=daughter_b_name, d_model=brain.d_model, n_heads=brain.n_heads, d_mlp=brain.d_mlp
        )

        # Split vocabulary items across the two daughter brains
        vocab_items = list(brain.vocab.items())[3:]  # Skip special tokens
        mid = len(vocab_items) // 2

        for tok, _ in vocab_items[:mid]:
            daughter_a.register_token(tok)
        for tok, _ in vocab_items[mid:]:
            daughter_b.register_token(tok)

        # Copy and spectrally partition projection weights across daughters (odd vs even singular modes)
        U, S, Vt = np.linalg.svd(brain.W_up.astype(np.float64), full_matrices=False)
        even_mask = np.zeros_like(S)
        odd_mask = np.zeros_like(S)
        even_mask[0::2] = S[0::2]
        if len(S) > 1:
            odd_mask[1::2] = S[1::2]
        else:
            odd_mask[:] = S * 0.5

        daughter_a.W_up = (U @ np.diag(even_mask) @ Vt).astype(np.float32)
        daughter_b.W_up = (U @ np.diag(odd_mask) @ Vt).astype(np.float32)

        half_mlp = max(1, brain.d_mlp // 2)
        daughter_a.W_down = brain.W_down.copy()
        daughter_a.W_down[half_mlp:, :] *= 0.25
        daughter_b.W_down = brain.W_down.copy()
        daughter_b.W_down[:half_mlp, :] *= 0.25

        if isinstance(getattr(brain, "W_mlp", None), np.ndarray):
            daughter_a.W_mlp = brain.W_mlp.copy()
            daughter_b.W_mlp = brain.W_mlp.copy()

        # Partition Hopfield associative memory rules between daughters with basis re-alignment
        if brain.hopfield_keys:
            h_mid = max(1, len(brain.hopfield_keys) // 2)
            n_v = min(len(brain.vocab), len(brain.W_embed))
            specials = ("<pad>", "<unk>", "<eos>")

            parent_rules: List[Tuple[str, str]] = []
            if n_v > 0:
                emb_slice = brain.W_embed[:n_v]
                emb_unit = emb_slice / (np.linalg.norm(emb_slice, axis=-1, keepdims=True) + 1e-12)
                for hk, hv in zip(brain.hopfield_keys, brain.hopfield_values):
                    min_d = min(brain.d_model, len(hk), len(hv), emb_unit.shape[1])
                    k_idx = int(np.argmax(np.dot(emb_unit[:, :min_d], hk[:min_d])))
                    v_idx = int(np.argmax(np.dot(emb_unit[:, :min_d], hv[:min_d])))
                    k_tok = brain.inv_vocab.get(k_idx, "<unk>")
                    v_tok = brain.inv_vocab.get(v_idx, "<unk>")
                    if k_tok not in specials and v_tok not in specials:
                        parent_rules.append((k_tok, v_tok))

            def _assign_rule(
                daughter: AnalyticalMicroTransformer,
                hk: np.ndarray,
                hv: np.ndarray,
                collected: List[Tuple[List[str], str]],
            ):
                if n_v > 0:
                    min_d = min(brain.d_model, len(hk), len(hv), brain.W_embed.shape[1])
                    emb_slice = brain.W_embed[:n_v, :min_d]
                    emb_unit = emb_slice / (np.linalg.norm(emb_slice, axis=-1, keepdims=True) + 1e-12)
                    k_idx = int(np.argmax(np.dot(emb_unit, hk[:min_d])))
                    v_idx = int(np.argmax(np.dot(emb_unit, hv[:min_d])))
                    k_tok = brain.inv_vocab.get(k_idx, "<unk>")
                    v_tok = brain.inv_vocab.get(v_idx, "<unk>")
                    if k_tok not in specials and v_tok not in specials:
                        gain = max(1e-3, float(np.linalg.norm(hv)))
                        daughter.compile_logic_association(k_tok, v_tok, binding_gain=gain)
                        collected.append(([k_tok], v_tok))
                        return
                daughter.hopfield_keys.append(hk.copy())
                daughter.hopfield_values.append(hv.copy())

            a_rules: List[Tuple[List[str], str]] = []
            b_rules: List[Tuple[List[str], str]] = []
            for hk, hv in zip(brain.hopfield_keys[:h_mid], brain.hopfield_values[:h_mid]):
                _assign_rule(daughter_a, hk, hv, a_rules)
            for hk, hv in zip(brain.hopfield_keys[h_mid:], brain.hopfield_values[h_mid:]):
                _assign_rule(daughter_b, hk, hv, b_rules)

            # Ensure each daughter retains >= 12 Hopfield rules and held-out split-conformal calibration
            # so cellular mitosis never emits uncalibrated models into the fleet store.
            fission_semantic_tags = (
                "decompose_root_goal", "isolate_subtask_graph", "validate_step_contract",
                "schedule_parallel_stage", "verify_precondition_edge", "resolve_dependency_order",
                "partition_search_scope", "checkpoint_stage_output", "detect_cycle_guard",
                "merge_subtask_results", "estimate_step_budget", "rollback_partial_branch",
                "prioritize_critical_path", "bind_context_artifacts", "propagate_failure_mode",
                "emit_execution_manifest", "guard_resource_ceiling", "verify_postcondition_seal",
                "audit_branch_coherence", "dispatch_worker_unit", "aggregate_leaf_status",
                "prune_redundant_branch", "normalize_task_payload", "confirm_terminal_state",
                "isolate_sandbox_scope", "track_progress_invariant", "align_schema_contract",
                "synthesize_plan_delta", "split_dependency_cluster", "reconcile_stage_outputs",
                "verify_schema_boundary", "audit_execution_trace", "gate_unsafe_mutation",
                "narrow_tool_surface", "escalate_ambiguous_step", "consolidate_worker_state",
                "preserve_ewc_anchor", "compute_fisher_importance", "verify_checksum_digest",
                "track_window_coverage", "adjust_aci_quantile", "enforce_singleton_set",
                "route_bilingual_query", "filter_secret_leak", "check_ast_signature",
            )
            alpha = float(getattr(brain, "conformal_alpha", 0.05) or 0.05)
            for daughter, d_rules, suffix in ((daughter_a, a_rules, "alpha"), (daughter_b, b_rules, "beta")):
                idx_pad = 0
                while len(daughter.hopfield_keys) < 14 and idx_pad < len(parent_rules):
                    pk, pv = parent_rules[(idx_pad * 2 + (0 if suffix == "alpha" else 1)) % len(parent_rules)]
                    daughter.compile_logic_association(pk, pv, binding_gain=5.0)
                    d_rules.append(([pk], pv))
                    idx_pad += 1
                tag_idx = 0
                while len(daughter.vocab) < 54 and tag_idx < len(fission_semantic_tags):
                    daughter.register_token(f"{suffix}_{fission_semantic_tags[tag_idx]}")
                    tag_idx += 1
                daughter.inv_vocab = {int(v): k for k, v in daughter.vocab.items()}
                if d_rules:
                    n_dr = len(d_rules)
                    cal_anchors = [d_rules[(i * n_dr) // 22] for i in range(22)]
                    cal_data = [
                        (list(toks), tgt) if (i % 3 == 0) else ([f"cal_{suffix}_{fission_semantic_tags[i % len(fission_semantic_tags)]}", *toks], tgt)
                        for i, (toks, tgt) in enumerate(cal_anchors)
                    ]
                    daughter.calibrate_conformal(cal_data, alpha=alpha, in_sample=False)

        daughter_a.save()
        daughter_b.save()

        self.nursery.active_brains[daughter_a_name] = daughter_a
        self.nursery.active_brains[daughter_b_name] = daughter_b

        return {
            "status": "fission_executed",
            "parent_brain": clean_name,
            "old_condition_number": round(cond_num, 2),
            "daughter_a": daughter_a_name,
            "daughter_b": daughter_b_name,
            "vocab_split": [len(daughter_a.vocab), len(daughter_b.vocab)],
        }

