"""Counterfactual Simulation Advisory Pipeline (`simulate_edit_impact`).

Evaluates proposed code patches in read-only advisory mode before host execution:
  1. `CriticalEnsemble` consensus check (`evaluate_critical_action`), returning early
     on constitutional/safety rejections or non-blast abstentions.
  2. Layer 1 structural AST, secret-leak, interface-hazard, and complexity inspection
     (`analyze_code_diff`) + diff scale blast-radius scoring.
  3. Causal SCM `do`-intervention rollout (`CausalWorldModel.counterfactual_query`).

Note: This module is purely advisory and does NOT apply patches to disk, run test
suites, or mutate/rollback the working tree; execution and rollback remain the
responsibility of the caller/host harness.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cascade_ensemble import get_cascade_ensemble_router
from .causal_world_model import get_causal_world_model
from .critical_ensemble_abstention import get_critical_ensemble
from .layered_decision_engine import analyze_code_diff

log = logging.getLogger(__name__)


@dataclass
class SimulationOutcome:
    """Outcome of 3-step counterfactual simulation."""

    allowed: bool
    status: str  # "APPROVED_FOR_PATCH", "VETOED_CONSTITUTIONAL", "VETOED_HIGH_RISK_BLAST", "ABSTAIN_ESCALATE_LLM"
    blast_radius_score: float
    regression_probability: float
    simulated_failure_modes: List[str]
    veto_reason: Optional[str] = None
    step_predictions: List[str] = field(default_factory=list)


class CounterfactualSimulationPipeline:
    """Orchestrates hypothetical forward-branch simulation before applying code patches."""

    def __init__(self, repo_dir: Optional[Path] = None):
        default_root = Path(__file__).resolve().parents[2]
        self.repo_dir = Path(repo_dir) if repo_dir is not None else (
            default_root if (default_root / "aetheris_microbrains").exists() else Path.cwd()
        )
        self.ensemble = get_critical_ensemble()
        self.cascade_router = get_cascade_ensemble_router()

    def simulate_edit_impact(
        self,
        file_path: str,
        diff_text: str,
        cascade_depth: int = 3,
    ) -> SimulationOutcome:
        """Run 3-step counterfactual simulation for a proposed patch."""
        # 1. Check critical ensemble consensus
        consensus = self.ensemble.evaluate_critical_action(
            action_name="apply_patch",
            target_path=file_path,
            payload_diff=diff_text,
        )

        # Defer early only when abstention is due to missing models/uncertainty or non-blast disagreement;
        # when `patch_quality_predictor` alone raised a blast/interface flag, run full 3-step rollout.
        deny_names = {v.estimator_name for v in consensus.votes if v.verdict == "DENY"}
        abstain_names = {v.estimator_name for v in consensus.votes if v.verdict == "ABSTAIN"}
        only_patch_blast_deny = (deny_names == {"patch_quality_predictor"} and not abstain_names)

        if consensus.abstention_triggered and not only_patch_blast_deny:
            var_thresh = max(0.0, float(getattr(self.ensemble, "variance_threshold", 0.0625)))
            abst_modes: List[str] = []
            if abstain_names:
                abst_modes.append(f"Estimator abstention: {', '.join(sorted(abstain_names))}")
            if consensus.estimator_variance > var_thresh + 1e-12:
                abst_modes.append(
                    f"Estimator disagreement variance {consensus.estimator_variance:.4f} exceeds "
                    f"{var_thresh:.4f} (std <= {math.sqrt(var_thresh):.2f}) threshold"
                )
            if not abst_modes:
                abst_modes.append(consensus.rationale or "Critical ensemble abstention triggered")
            return SimulationOutcome(
                allowed=False,
                status="ABSTAIN_ESCALATE_LLM",
                blast_radius_score=0.5,
                regression_probability=0.5,
                simulated_failure_modes=abst_modes,
                veto_reason="Critical ensemble abstention triggered. Requires LLM verification.",
            )

        if consensus.status == "CONSENSUS_REJECTED":
            return SimulationOutcome(
                allowed=False,
                status="VETOED_CONSTITUTIONAL",
                blast_radius_score=0.9,
                regression_probability=0.95,
                simulated_failure_modes=["Constitutional or boundary violation"],
                veto_reason=consensus.rationale,
            )

        # 2. Counterfactual 3-step forward rollout
        step_predictions = []
        failure_modes = []

        # Step 1: Layer 1 Structural AST & Security Analysis
        diff_analysis = analyze_code_diff(file_path, diff_text)
        step_1_ok = bool(diff_analysis["syntax_valid"] and diff_analysis["security_clean"])
        step_predictions.append(f"Step 1 (Structural AST Parse [{diff_analysis['language']}]): {'PASS' if step_1_ok else 'FAIL'}")
        if not step_1_ok:
            if not diff_analysis["syntax_valid"]:
                failure_modes.append("Step 1: Malformed syntax in proposed diff")
            else:
                failure_modes.append(f"Step 1: Security risk detected in AST ({', '.join(diff_analysis['findings'])})")

        # Step 2: AST Complexity, Interface/Signature Diff & Blast Radius
        line_count = len(diff_text.splitlines())
        interface_hazards = list(diff_analysis.get("interface_hazards", []))
        cyclomatic = int(diff_analysis.get("cyclomatic_complexity", 0) or 0)
        is_static_data = bool(diff_analysis.get("is_static_data_or_doc", False))
        if is_static_data and line_count <= 480 and not interface_hazards and diff_analysis.get("security_clean", True):
            blast_score = min(0.35, line_count / 1200.0)
        else:
            blast_score = min(1.0, (line_count / 600.0) + 0.45 * len(interface_hazards))
            if cyclomatic >= 25:
                blast_score = min(1.0, blast_score + min(0.65, cyclomatic * 0.02))
        step_2_ok = (blast_score < 0.80) and (len(interface_hazards) == 0)
        step_predictions.append(
            f"Step 2 (Interface & Complexity [cc={cyclomatic}]): {'PASS' if step_2_ok else ('WARN_BREAKING_INTERFACE' if interface_hazards else 'WARN_HIGH_BLAST')}"
        )
        if not step_2_ok:
            if interface_hazards:
                failure_modes.append(f"Step 2: Breaking API/interface change ({', '.join(interface_hazards)})")
            elif cyclomatic >= 25:
                failure_modes.append(f"Step 2: High cyclomatic complexity delta ({cyclomatic} branches) risks regression")
            else:
                failure_modes.append(f"Step 2: Massive diff blast radius ({line_count} lines) risks regression")

        # Step 3: Causal SCM do-Intervention Rollout
        scm = get_causal_world_model(self.repo_dir)
        causal_res = scm.counterfactual_query(
            observed_action="apply_patch",
            observed_target=file_path,
            observed_outcome_broken=not (step_1_ok and step_2_ok),
            counterfactual_do={
                "action_type": "apply_patch",
                "target_path": file_path,
                "preconditions_satisfied": True,
                "ast_validity": step_1_ok,
                "downstream_blast_radius": max(len(interface_hazards) * 2, 4 if line_count > 480 else 0),
                "lock_contention": False,
            },
            proposed_code=None,
        )
        step_3_ok = bool(causal_res.is_safe and step_1_ok and step_2_ok)
        reg_prob = float(causal_res.predicted_outcomes.get("regression_probability", 0.95 if not step_3_ok else blast_score * 0.4))
        step_predictions.append(
            f"Step 3 (Causal SCM Rollout [p_reg={reg_prob:.2f}]): {'PASS' if step_3_ok else 'FAIL'}"
        )
        if not step_3_ok:
            failure_modes.extend(causal_res.causal_mechanisms_triggered or ["Step 3: Causal intervention predicts regression"])

        overall_pass = step_1_ok and step_2_ok and step_3_ok
        regression_prob = reg_prob if not overall_pass else min(0.5, max(blast_score * 0.4, reg_prob * 0.5))

        return SimulationOutcome(
            allowed=overall_pass,
            status="APPROVED_FOR_PATCH" if overall_pass else "VETOED_HIGH_RISK_BLAST",
            blast_radius_score=round(blast_score, 3),
            regression_probability=round(regression_prob, 3),
            simulated_failure_modes=list(dict.fromkeys(failure_modes)),
            veto_reason=failure_modes[0] if failure_modes else None,
            step_predictions=step_predictions,
        )


_simulation_pipeline_singleton: Optional[CounterfactualSimulationPipeline] = None


def get_counterfactual_pipeline() -> CounterfactualSimulationPipeline:
    """Singleton getter for CounterfactualSimulationPipeline."""
    global _simulation_pipeline_singleton
    if _simulation_pipeline_singleton is None:
        _simulation_pipeline_singleton = CounterfactualSimulationPipeline()
    return _simulation_pipeline_singleton
