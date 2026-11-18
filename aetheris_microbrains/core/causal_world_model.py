"""Causal World Model: Structural Causal Model (SCM) & Do-Calculus for Ouroboros.

Implements SOTA causal reasoning for autonomous agent actions based on Judea Pearl's
Causal Hierarchy:
  - Level 1: Associational / Observational: P(outcome | action)
  - Level 2: Interventional: P(outcome | do(action)) via graph surgery
  - Level 3: Counterfactual: P(outcome_{do(action')} | evidence) via Abduction-Action-Prediction

Grounds code edits, tool calls, and git operations in a Directed Acyclic Graph (DAG)
of mechanisms, replacing blind Monte Carlo simulations with causal inference.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


@dataclass
class CausalVariable:
    """A node in the Structural Causal Model DAG."""

    name: str
    description: str
    is_endogenous: bool = True  # True if determined within system, False if exogenous noise
    parents: List[str] = field(default_factory=list)
    state: Any = None
    domain_type: str = "binary"  # "binary", "continuous", "categorical"


@dataclass
class CausalIntervention:
    """An interventional do(X=x) operation severing incoming causal edges."""

    target_var: str
    value: Any


@dataclass
class CausalSimulationResult:
    """Outcome of an interventional or counterfactual query on the Causal World Model."""

    query_type: str  # "observational", "interventional", "counterfactual"
    interventions: Dict[str, Any]
    predicted_outcomes: Dict[str, Any]
    risk_score: float  # 0.0 (safe) to 1.0 (dangerous)
    is_safe: bool
    causal_mechanisms_triggered: List[str]
    suggested_guards: List[str]
    epistemic_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "interventions": self.interventions,
            "predicted_outcomes": self.predicted_outcomes,
            "risk_score": round(self.risk_score, 3),
            "is_safe": self.is_safe,
            "causal_mechanisms_triggered": self.causal_mechanisms_triggered,
            "suggested_guards": self.suggested_guards,
            "epistemic_confidence": round(self.epistemic_confidence, 3),
        }


class StructuralCausalModel:
    """Structural Causal Model (SCM) representing causal mechanisms in code operations.

    Structural equations:
      X_i = f_i(PA_i, U_i)
    Under intervention do(X_j = x*):
      The equation for X_j is replaced by X_j = x*, and edges PA_j -> X_j are severed.
    """

    def __init__(self, repo_dir: Optional[Path] = None):
        self.repo_dir = repo_dir or Path(".")
        self.variables: Dict[str, CausalVariable] = {}
        self.mechanisms: Dict[str, Callable[[Dict[str, Any], Any], Any]] = {}
        self._init_code_world_graph()

    def _init_code_world_graph(self) -> None:
        """Initialize the causal DAG representing code edits and agent actions."""
        # 1. Action & Precondition nodes
        self.variables["action_type"] = CausalVariable(
            "action_type", "Type of tool or operation executed", is_endogenous=True
        )
        self.variables["target_path"] = CausalVariable(
            "target_path", "Path of the target file being touched", is_endogenous=True
        )
        self.variables["recent_tools"] = CausalVariable(
            "recent_tools", "Sequence of recently invoked tools", is_endogenous=True
        )

        # 2. Intermediate causal states
        self.variables["preconditions_satisfied"] = CausalVariable(
            "preconditions_satisfied",
            "Whether read-before-write or lock pre-conditions hold",
            parents=["action_type", "recent_tools"],
        )
        self.variables["ast_validity"] = CausalVariable(
            "ast_validity",
            "Syntactic validity of proposed python code",
            parents=["action_type", "target_path"],
        )
        self.variables["ancestral_core_impact"] = CausalVariable(
            "ancestral_core_impact",
            "Whether edit touches foundational system files",
            parents=["target_path"],
        )
        self.variables["downstream_blast_radius"] = CausalVariable(
            "downstream_blast_radius",
            "Number of dependent modules importing target",
            parents=["target_path"],
        )
        self.variables["lock_contention"] = CausalVariable(
            "lock_contention",
            "Git index lock or file access conflict risk",
            parents=["action_type", "target_path"],
        )

        # 3. Outcome nodes
        self.variables["regression_probability"] = CausalVariable(
            "regression_probability",
            "Probability of introducing downstream functional regression",
            parents=["ast_validity", "ancestral_core_impact", "downstream_blast_radius"],
        )
        self.variables["operational_safety"] = CausalVariable(
            "operational_safety",
            "Overall safety of executing the proposed action",
            parents=["preconditions_satisfied", "regression_probability", "lock_contention"],
        )

    def _eval_preconditions(self, parents: Dict[str, Any]) -> bool:
        action = parents.get("action_type", "")
        recent = parents.get("recent_tools") or []
        if action in ("repo_write", "str_replace_editor", "verified_patch"):
            return len(recent) > 0 and recent[-1] == "repo_read"
        return True

    def _eval_ast(self, target_path: str, code: Optional[str]) -> bool:
        if not code:
            return True
        from .layered_decision_engine import analyze_code_diff
        res = analyze_code_diff(target_path, str(code))
        return bool(res["syntax_valid"] and res["security_clean"])

    def _eval_ancestral(self, target_path: str) -> bool:
        from .layered_decision_engine import is_protected_path

        clean = str(target_path).replace("\\", "/")
        core_modules = (
            "ouroboros/agent.py",
            "ouroboros/loop.py",
            "ouroboros/context.py",
            "ouroboros/llm.py",
            "ouroboros/causal_world_model.py",
            "core/analytical_micro_transformer.py",
            "core/layered_decision_engine.py",
            "core/critical_ensemble_abstention.py",
        )
        return is_protected_path(clean) or any(c in clean for c in core_modules)

    def _eval_blast_radius(self, target_path: str) -> int:
        if not str(target_path).endswith(".py"):
            return 0
        stem = Path(target_path).stem
        if not stem or not self.repo_dir.is_dir():
            return 0
        import re as _re

        skip_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache", "node_modules"}
        preferred = [
            d for d in (self.repo_dir / "ouroboros", self.repo_dir / "aetheris_microbrains", self.repo_dir / "src", self.repo_dir / "core", self.repo_dir / "tools")
            if d.is_dir()
        ]
        scan_roots = preferred if preferred else [self.repo_dir]
        import_re = _re.compile(
            rf"\b(?:import\s+{_re.escape(stem)}\b|from\s+(?:[A-Za-z0-9_.]+\.)?{_re.escape(stem)}\s+import\b|from\s+[A-Za-z0-9_.]+\s+import\s+[^#\n]*\b{_re.escape(stem)}\b)"
        )
        count = 0
        scanned = 0
        for root in scan_roots:
            for py_file in root.rglob("*.py"):
                if any(part in skip_dirs for part in py_file.parts):
                    continue
                if py_file.stem == stem:
                    continue
                scanned += 1
                if scanned > 400:
                    return count
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    if import_re.search(content):
                        count += 1
                        if count >= 10:
                            return count
                except Exception:
                    continue
        return count

    def do(self, interventions: Dict[str, Any]) -> StructuralCausalModel:
        """Apply Pearl's do-operator: sever incoming edges to intervened variables."""
        mutated = StructuralCausalModel(self.repo_dir)
        for var_name, var in mutated.variables.items():
            if var_name in interventions:
                # Graph surgery: sever parents
                var.parents = []
                var.state = interventions[var_name]
        return mutated

    def evaluate(
        self,
        action: str,
        target: str,
        proposed_code: Optional[str] = None,
        recent_tools: Optional[List[str]] = None,
        interventions: Optional[Dict[str, Any]] = None,
    ) -> CausalSimulationResult:
        """Evaluate the causal DAG under observational or interventional conditions."""
        interventions = interventions or {}
        model = self.do(interventions) if interventions else self

        # Populate endogenous states
        act = interventions.get("action_type", action)
        tgt = interventions.get("target_path", target)
        recent = interventions.get("recent_tools", recent_tools or [])

        # Causal mechanisms
        precond_ok = (
            interventions["preconditions_satisfied"]
            if "preconditions_satisfied" in interventions
            else self._eval_preconditions({"action_type": act, "recent_tools": recent})
        )

        ast_ok = (
            interventions["ast_validity"]
            if "ast_validity" in interventions
            else self._eval_ast(tgt, proposed_code)
        )

        ancestral = (
            interventions["ancestral_core_impact"]
            if "ancestral_core_impact" in interventions
            else self._eval_ancestral(tgt)
        )

        blast_radius = (
            interventions["downstream_blast_radius"]
            if "downstream_blast_radius" in interventions
            else self._eval_blast_radius(tgt)
        )

        lock_risk = (
            interventions["lock_contention"]
            if "lock_contention" in interventions
            else (act == "git_commit" and ("index.lock" in str(tgt) or ".git" in str(tgt)))
        )

        # Structural Equation for Regression Probability:
        # P(reg) = base + 0.5 * (not ast_ok) + 0.3 * ancestral + 0.1 * min(blast_radius, 4)
        reg_prob = 0.05
        if not ast_ok:
            reg_prob += 0.55
        if ancestral:
            reg_prob += 0.30
        if blast_radius >= 3:
            reg_prob += min(0.25, blast_radius * 0.05)
        reg_prob = min(0.99, max(0.01, reg_prob))

        # Structural Equation for Safety
        hazards: List[str] = []
        guards: List[str] = []
        risk_score = 0.0

        if not precond_ok:
            hazards.append("precondition_breach: file write without preceding repo_read")
            guards.append(f"Execute repo_read('{tgt}') first.")
            risk_score += 0.35

        if not ast_ok:
            hazards.append("syntax_violation: proposed code fails AST parse")
            guards.append("Fix Python syntax errors before saving to repository.")
            risk_score += 0.50

        if ancestral:
            hazards.append("ancestral_site_hazard: touches foundational core architecture")
            guards.append("Run verification tests in isolated worktree before committing.")
            risk_score += 0.25

        if blast_radius >= 4:
            hazards.append(f"high_blast_radius: {blast_radius} downstream modules depend on this file")
            guards.append("Verify dependent imports in downstream modules.")
            risk_score += 0.20

        if lock_risk:
            hazards.append("lock_contention: potential git lock or concurrent file race")
            guards.append("Wait for repository lock release.")
            risk_score += 0.30

        # Consult Neural Microbrain using structural input features BEFORE finalizing verdict
        struct_tokens: List[str] = [
            "ast_structure_valid" if ast_ok else "ast_signature_break",
            "policy_reversible_write" if precond_ok else "policy_irreversible_block",
            "sacred_boundary_protected" if ancestral else "resilience_buffer_adequate",
            "rollback_cost_expensive" if blast_radius >= 4 else "rollback_cost_negligible",
        ]
        neural_conf = round(max(0.0, 1.0 - min(1.0, risk_score)), 4)
        neural_veto = False
        try:
            from .brain_moe_orchestrator import consult_microbrain

            res = consult_microbrain("counterfactual_reasoner", struct_tokens)
            if isinstance(res, dict) and res.get("success", True):
                if res.get("confidence") is not None:
                    neural_conf = float(res["confidence"])
                pred_tok = str(res.get("predicted_token") or "").lower()
                if not res.get("abstained", False) and any(
                    w in pred_tok for w in ("abort", "veto", "breaking", "reject", "unsafe")
                ):
                    neural_veto = True
                    hazards.append(f"neural_counterfactual_veto:{pred_tok}")
                    risk_score += 0.25
        except Exception as exc:
            hazards.append(f"neural_module_unavailable:{type(exc).__name__}")

        risk_score = min(1.0, max(0.0, risk_score))
        is_safe = (risk_score < 0.45) and ast_ok and precond_ok and (not neural_veto)

        return CausalSimulationResult(
            query_type="interventional" if interventions else "observational",
            interventions=interventions,
            predicted_outcomes={
                "preconditions_satisfied": precond_ok,
                "ast_validity": ast_ok,
                "ancestral_core_impact": ancestral,
                "downstream_blast_radius": blast_radius,
                "regression_probability": round(reg_prob, 3),
            },
            risk_score=risk_score,
            is_safe=is_safe,
            causal_mechanisms_triggered=hazards,
            suggested_guards=guards,
            epistemic_confidence=neural_conf,
        )

    def counterfactual_query(
        self,
        observed_action: str,
        observed_target: str,
        observed_outcome_broken: bool,
        counterfactual_do: Dict[str, Any],
        proposed_code: Optional[str] = None,
    ) -> CausalSimulationResult:
        """Run a 3-step Pearl Counterfactual: Abduction -> Action -> Prediction.

        Q: "Given that action A on target T produced a broken outcome,
            what would have happened if we had done do(X=x') instead?"
        """
        # Step 1: Abduction (infer exogenous noise / unobserved confounder)
        # If observed outcome was broken but ast was valid, unobserved context noise is high.
        inferred_environmental_hazard = observed_outcome_broken

        # Step 2: Action (apply do-intervention)
        intervened_model = self.do(counterfactual_do)

        # Step 3: Prediction (evaluate under intervened model with abducted background context)
        result = intervened_model.evaluate(
            action=counterfactual_do.get("action_type", observed_action),
            target=counterfactual_do.get("target_path", observed_target),
            proposed_code=proposed_code,
            interventions=counterfactual_do,
        )

        result.query_type = "counterfactual"
        if inferred_environmental_hazard and not counterfactual_do.get("preconditions_satisfied", False):
            result.risk_score = min(1.0, result.risk_score + 0.15)
            result.is_safe = result.risk_score < 0.45

        return result


# Global singleton instance
_CAUSAL_MODEL: Optional[StructuralCausalModel] = None

# Backward compatibility alias
CausalWorldModel = StructuralCausalModel


def get_causal_world_model(repo_dir: Optional[Path] = None) -> StructuralCausalModel:
    """Retrieve or initialize the active Causal World Model."""
    global _CAUSAL_MODEL
    if _CAUSAL_MODEL is None or (repo_dir and _CAUSAL_MODEL.repo_dir != repo_dir):
        _CAUSAL_MODEL = StructuralCausalModel(repo_dir)
    return _CAUSAL_MODEL
