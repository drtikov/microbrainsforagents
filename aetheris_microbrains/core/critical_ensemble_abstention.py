"""Critical Ensemble with Selective Prediction & Conformal Abstention.

Provides high-assurance consensus gating for high-stakes actions:
1. Queries 4 independent microbrains evaluating an action:
   - `neuro_constitutional_guardian`: Evaluates sovereignty, sacred sites, and boundary violations.
   - `neuro_code_edit_verifier`: Real AST parse tokenization & structural verification.
   - `neuro_patch_quality_predictor`: Diff scale, blast radius, and regression probability estimation.
   - `neuro_tool_failure_classifier`: Action boundary, permission, and Suki-class failure risk.
2. Weight-aggregates opinions using model-predicted conformal nonconformity bounds.
3. Decisions are strictly determined by neural inference outputs (predicted token, confidence, conformal abstention).
4. Computes empirical disagreement variance among estimators:
      Var(P) = (1/K) * Sum_i (p_i - mean_p)^2,   std = sqrt(Var(P))
5. Selective Abstention & Rejection Rules:
      - Direct Rejection (CONSENSUS_REJECTED): Certified hard invariant violations (sacred sites, malformed AST, destructive commands).
      - Selective Abstention (ABSTAIN_DELEGATE_TO_LLM): Ambiguous prediction set (|C_alpha| != 1), missing models, or estimator disagreement variance > variance_threshold (default 0.0625, i.e. std > 0.25).
      - Consensus Approved (CONSENSUS_APPROVED): Low disagreement (Var(P) <= variance_threshold, default 0.0625 / std <= 0.25), zero abstentions, mean probability >= 0.75.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR
from .layered_decision_engine import (
    analyze_code_diff,
    analyze_shell_command,
    is_protected_path,
)

log = logging.getLogger(__name__)

# Semantic Token Categories for Neural Conformal Gating
GUARDIAN_DENY_TOKENS: Set[str] = {
    "violation_severity_blocked",
    "violation_severity_fatal_quarantine",
    "veto_unconditional",
    "action_repo_write_checked",
    "provenance_signature_invalid",
    "sacred_site_aboriginal_core",
    "sacred_site_bible_md",
    "sacred_site_spirit_of_place",
    "sacred_site_songlines",
    "sacred_site_dreamtime",
    "sacred_site_integrity_seal",
}
GUARDIAN_ALLOW_TOKENS: Set[str] = {
    "violation_severity_none",
    "ancestral_law_honored",
    "boundary_containment_active",
    "provenance_signature_valid",
    "kinship_dreamkeeper_guard",
    "kinship_pathfinder_guidance",
    "kinship_explorer_discovery",
}

VERIFIER_DENY_TOKENS: Set[str] = {
    "verifier_verdict_reject_syntax",
    "verifier_verdict_reject_safety",
    "syntax_tree_malformed",
    "secret_leak_pattern_found",
    "side_effect_subprocess_spawn",
    "sacred_path_target_detected",
}
VERIFIER_ALLOW_TOKENS: Set[str] = {
    "verifier_verdict_accept",
    "conformal_safety_guarantee_met",
    "precondition_enforced",
    "syntax_tree_valid",
    "secret_leak_clean",
    "sacred_path_safe",
    "diff_hunk_clean_insertion",
}

PREDICTOR_DENY_TOKENS: Set[str] = {
    "patch_quality_rejected_high_risk",
    "breaking_change_signature_detected",
    "regression_probability_high",
    "ast_syntax_malformed",
}
PREDICTOR_ALLOW_TOKENS: Set[str] = {
    "patch_quality_grade_a",
    "patch_quality_grade_b",
    "patch_quality_grade_c",
    "patch_quality_certified",
    "backward_compatibility_preserved",
    "regression_probability_low",
    "atomic_hunk_isolated",
    "diff_lines_sub_ten",
}

CLASSIFIER_DENY_TOKENS: Set[str] = {
    "failure_fatal_circuit_breaker",
    "failure_severity_system_critical",
    "suki_class_boundary",
    "suki_class_syntax",
    "suki_class_precondition",
    "suki_class_resource",
    "suki_class_sacred_site_mutation",
}
CLASSIFIER_ALLOW_TOKENS: Set[str] = {
    "action_inspect_stack_trace",
    "failure_severity_benign",
    "failure_recoverable_by_retry",
    "clean_success",
}


@dataclass
class EnsembleVote:
    """An individual estimator's opinion in the critical ensemble."""

    estimator_name: str
    verdict: str  # "ALLOW", "DENY", "ABSTAIN"
    confidence: float
    nonconformity_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticalConsensusResult:
    """Aggregated decision from the critical ensemble."""

    status: str  # "CONSENSUS_APPROVED", "CONSENSUS_REJECTED", "ABSTAIN_DELEGATE_TO_LLM"
    consensus_score: float
    estimator_variance: float
    abstention_triggered: bool
    votes: List[EnsembleVote] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "consensus_score": round(self.consensus_score, 4),
            "estimator_variance": round(self.estimator_variance, 4),
            "abstention_triggered": self.abstention_triggered,
            "votes": [
                {
                    "name": v.estimator_name,
                    "verdict": v.verdict,
                    "confidence": round(v.confidence, 4),
                    "nonconformity": round(v.nonconformity_score, 4),
                    "details": v.details,
                }
                for v in self.votes
            ],
            "rationale": self.rationale,
        }


class CriticalEnsemble:
    """Ensemble of trained microbrains with selective abstention gating."""

    def __init__(self, store_dir: Optional[Path] = None, variance_threshold: float = 0.0625):
        self.store_dir = store_dir or STORE_DIR
        self.variance_threshold = variance_threshold
        self._models_cache: Dict[str, AnalyticalMicroTransformer] = {}

    def _get_model(self, model_name: str) -> Optional[AnalyticalMicroTransformer]:
        """Lazy load and cache microbrain model."""
        if model_name in self._models_cache:
            return self._models_cache[model_name]
        m = AnalyticalMicroTransformer.load(model_name, self.store_dir)
        if m is not None:
            self._models_cache[model_name] = m
        return m

    @staticmethod
    def _unique_tokens(tokens: List[str], anchor: str) -> List[str]:
        """Deduplicate context tokens while guaranteeing the readout anchor stays at `[-1]`."""
        seen: Set[str] = {anchor}
        prefix: List[str] = []
        for tok in tokens:
            if tok and tok not in seen:
                seen.add(tok)
                prefix.append(tok)
        prefix.append(anchor)
        return prefix

    def _build_guardian_tokens(self, path_str: str, action_name: str, is_sacred: bool) -> List[str]:
        norm = path_str.replace("\\", "/").lower()
        feats: List[str] = []
        if "bible" in norm:
            feats.append("sacred_site_bible_md")
        if "seal" in norm or "manifest" in norm:
            feats.append("sacred_site_integrity_seal")
        if "constitution" in norm or "aboriginal" in norm:
            feats.append("sacred_site_aboriginal_core")
        if any(s in norm for s in (".env", ".git/hooks", ".git/config", ".ssh", "id_rsa", "id_ed25519", "credentials", "secrets")):
            feats.extend(["protected_charter_path", "unauthorized_seal_edit"])
        if is_sacred or "kinship" in norm or "totem" in norm:
            feats.append("sacred_site_aboriginal_core")
        feats.append(
            "action_shell_exec_checked"
            if action_name in ("run_command", "execute_shell")
            else ("action_config_mutate_checked" if norm.endswith((".json", ".yaml", ".yml", ".toml")) else "clean_tool_call")
        )
        feats.extend(["kinship_dreamkeeper_guard", "boundary_containment_active"])
        anchor = "action_repo_write_checked" if is_sacred else "ancestral_law_honored"
        return self._unique_tokens(feats, anchor)

    def _build_verifier_tokens(
        self,
        diff_analysis: Dict[str, Any],
        has_secret_leak: bool,
        vocab: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        syntax_clean = bool(diff_analysis.get("syntax_valid", True))
        security_clean = bool(diff_analysis.get("security_clean", True))
        interface_hazards = diff_analysis.get("interface_hazards") or []
        feats: List[str] = []
        for tag in diff_analysis.get("ast_nodes") or []:
            if tag not in ("ast_node_try_except", "ast_node_function_def") and (vocab is None or tag in vocab):
                feats.append(tag)
        if diff_analysis.get("language") not in ("python", "empty", ""):
            feats.append("non_python_ts_json")
        if diff_analysis.get("context_ast_parsed"):
            feats.extend(["hunk_verified", "indented_block_valid"])
        feats.append("ast_signature_break" if interface_hazards else "ast_interface_stable")
        feats.append("syntax_tree_malformed" if not syntax_clean else "ast_structure_valid")

        if not syntax_clean:
            anchor = "ast_node_try_except"
        elif has_secret_leak:
            anchor = "secret_leak_pattern_found"
        elif not security_clean:
            anchor = "side_effect_subprocess_spawn"
        else:
            anchor = "ast_node_function_def"
        return self._unique_tokens(feats, anchor)

    def _build_predictor_tokens(
        self,
        path_str: str,
        diff_lines: int,
        diff_analysis: Dict[str, Any],
        interface_hazards: List[str],
    ) -> List[str]:
        syntax_valid = bool(diff_analysis.get("syntax_valid", True))
        security_clean = bool(diff_analysis.get("security_clean", True))
        is_static_data = bool(diff_analysis.get("is_static_data_or_doc", False))
        cyclomatic = int(diff_analysis.get("cyclomatic_complexity", 0) or 0)
        norm = path_str.lower()
        feats: List[str] = [
            "ast_syntax_verified_clean" if syntax_valid else "ast_syntax_malformed",
            "touches_tests_only" if "test" in norm else ("touches_documentation_only" if norm.endswith(".md") else "touches_tools_only"),
            "ast_signature_break" if interface_hazards else "ast_interface_stable",
        ]
        if is_static_data and diff_lines <= 480 and not interface_hazards and security_clean:
            feats.extend(["data_table_static_values", "diff_lines_sub_two_hundred", "regression_probability_low"])
            anchor = "backward_compatibility_preserved"
        elif cyclomatic >= 25:
            feats.extend(["cyclomatic_delta_severe", "regression_probability_high"])
            anchor = "breaking_change_signature_detected"
        elif diff_lines > 500:
            feats.extend(["diff_lines_massive_five_hundred", "regression_probability_high"])
            anchor = "breaking_change_signature_detected"
        elif diff_lines > 50:
            feats.extend(["diff_lines_sub_two_hundred", "regression_probability_low"])
            anchor = "breaking_change_signature_detected" if interface_hazards else "backward_compatibility_preserved"
        else:
            feats.extend(["diff_lines_sub_ten", "atomic_hunk_isolated", "regression_probability_low"])
            anchor = "breaking_change_signature_detected" if interface_hazards else "backward_compatibility_preserved"
        return self._unique_tokens(feats, anchor)

    def _build_classifier_tokens(
        self,
        shell_res: Optional[Dict[str, Any]],
        payload_diff: Optional[str],
        syntax_clean: bool,
        security_clean: bool,
    ) -> List[str]:
        feats: List[str] = []
        is_destructive_shell = bool(shell_res and not shell_res.get("is_safe", True))
        if shell_res is not None:
            feats.append(
                "shell_destructive_block"
                if is_destructive_shell
                else ("shell_allowlist_pass" if shell_res.get("allowlisted") else "shell_unrecognized_escalate")
            )
            finding_str = " ".join(shell_res.get("findings") or []).lower()
            if "rm_rf" in finding_str:
                feats.append("destructive_rm_rf")
            if "remote_code_execution" in finding_str:
                feats.append("curl_pipe_sh_rce")
            if "raw_disk_write" in finding_str:
                feats.append("destructive_dd_disk")
            if "disk_format" in finding_str:
                feats.append("format_disk_cmd")
            if "chmod_777" in finding_str:
                feats.append("chmod_777_root")
            if is_destructive_shell:
                feats.extend(["suki_class_boundary", "failure_severity_system_critical"])
            else:
                feats.extend(["safe_ls_echo_pwd", "failure_severity_benign"])
        elif payload_diff:
            feats.append("suki_class_syntax" if not syntax_clean else "ast_structure_valid")
            feats.append("ast_obfuscated_call" if not security_clean else "ast_interface_stable")
            if syntax_clean and security_clean:
                feats.append("failure_severity_benign")
            else:
                feats.append("failure_severity_system_critical")
        else:
            feats.extend(["benign_file_open", "ast_structure_valid", "failure_severity_benign"])

        anchor = (
            "failure_fatal_circuit_breaker"
            if (is_destructive_shell or (payload_diff and (not syntax_clean or not security_clean)))
            else "action_inspect_stack_trace"
        )
        return self._unique_tokens(feats, anchor)

    def evaluate_critical_action(
        self,
        action_name: str,
        target_path: Optional[str] = None,
        payload_diff: Optional[str] = None,
    ) -> CriticalConsensusResult:
        """Collect votes across 4 independent neural estimators via conformal inference."""
        votes: List[EnsembleVote] = []
        path_str = str(target_path or "")

        # =========================================================================
        # Estimator 1: Constitutional Guardian
        # =========================================================================
        guardian = self._get_model("neuro_constitutional_guardian")
        is_sacred = is_protected_path(path_str)
        g_tokens = self._build_guardian_tokens(path_str, action_name, is_sacred)

        if guardian is None:
            g_verdict = "ABSTAIN"
            g_conf = 0.50
            g_nonconf = 1.0
            g_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            g_res = guardian.infer_conformal(g_tokens, alpha=0.05)
            g_pred = g_res.get("predicted_token", "")
            g_abstained = bool(g_res.get("abstained", False))
            g_conf = float(g_res.get("confidence", 0.50))
            if g_abstained:
                g_verdict = "ABSTAIN"
            elif g_pred in GUARDIAN_DENY_TOKENS:
                g_verdict = "DENY"
            elif g_pred in GUARDIAN_ALLOW_TOKENS:
                g_verdict = "ALLOW"
            else:
                g_verdict = "ABSTAIN"
            g_nonconf = round(1.0 - g_conf, 4) if g_verdict == "ALLOW" else round(g_conf, 4)
            g_details = {
                "is_sacred": is_sacred,
                "tokens": g_tokens,
                "predicted_token": g_pred,
                "prediction_set": g_res.get("prediction_set", []),
                "abstained": g_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="constitutional_guardian",
            verdict=g_verdict,
            confidence=g_conf,
            nonconformity_score=g_nonconf,
            details=g_details,
        ))

        # =========================================================================
        # Estimator 2: Code Edit Verifier (Layer 1 Language-Aware Structural AST)
        # =========================================================================
        verifier = self._get_model("neuro_code_edit_verifier")
        diff_analysis = analyze_code_diff(path_str, payload_diff)
        syntax_clean = bool(diff_analysis["syntax_valid"])
        security_clean = bool(diff_analysis["security_clean"])
        diff_findings = diff_analysis.get("findings", [])
        interface_hazards = diff_analysis.get("interface_hazards", [])
        has_secret_leak = "secret_leak_pattern_found" in diff_findings
        ast_tokens = self._build_verifier_tokens(
            diff_analysis,
            has_secret_leak=has_secret_leak,
            vocab=verifier.vocab if verifier is not None else None,
        )

        if verifier is None:
            v_verdict = "ABSTAIN"
            v_conf = 0.50
            v_nonconf = 1.0
            v_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            v_res = verifier.infer_conformal(ast_tokens, alpha=0.05)
            v_pred = v_res.get("predicted_token", "")
            v_abstained = bool(v_res.get("abstained", False))
            v_conf = float(v_res.get("confidence", 0.50))
            if v_abstained:
                v_verdict = "ABSTAIN"
            elif v_pred in VERIFIER_DENY_TOKENS:
                v_verdict = "DENY"
            elif v_pred in VERIFIER_ALLOW_TOKENS:
                v_verdict = "ALLOW"
            else:
                v_verdict = "ABSTAIN"
            v_nonconf = round(1.0 - v_conf, 4) if v_verdict == "ALLOW" else round(v_conf, 4)
            v_details = {
                "syntax_clean": syntax_clean,
                "security_clean": security_clean,
                "patched_file_ast_verified": bool(diff_analysis.get("patched_file_ast_verified", False)),
                "tokens": ast_tokens,
                "predicted_token": v_pred,
                "prediction_set": v_res.get("prediction_set", []),
                "abstained": v_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="code_edit_verifier",
            verdict=v_verdict,
            confidence=v_conf,
            nonconformity_score=v_nonconf,
            details=v_details,
        ))

        # =========================================================================
        # Estimator 3: Patch Quality Predictor
        # =========================================================================
        predictor = self._get_model("neuro_patch_quality_predictor")
        diff_lines = len(payload_diff.splitlines()) if payload_diff else 0
        p_tokens = self._build_predictor_tokens(path_str, diff_lines, diff_analysis, interface_hazards)

        if predictor is None:
            p_verdict = "ABSTAIN"
            p_conf = 0.50
            p_nonconf = 1.0
            p_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            p_res = predictor.infer_conformal(p_tokens, alpha=0.05)
            p_pred = p_res.get("predicted_token", "")
            p_abstained = bool(p_res.get("abstained", False))
            p_conf = float(p_res.get("confidence", 0.50))
            if p_abstained:
                p_verdict = "ABSTAIN"
            elif p_pred in PREDICTOR_DENY_TOKENS:
                p_verdict = "DENY"
            elif p_pred in PREDICTOR_ALLOW_TOKENS:
                p_verdict = "ALLOW"
            else:
                p_verdict = "ABSTAIN"
            p_nonconf = round(1.0 - p_conf, 4) if p_verdict == "ALLOW" else round(p_conf, 4)
            p_details = {
                "diff_lines": diff_lines,
                "interface_hazards": interface_hazards,
                "tokens": p_tokens,
                "predicted_token": p_pred,
                "prediction_set": p_res.get("prediction_set", []),
                "abstained": p_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="patch_quality_predictor",
            verdict=p_verdict,
            confidence=p_conf,
            nonconformity_score=p_nonconf,
            details=p_details,
        ))

        # =========================================================================
        # Estimator 4: Tool Failure Classifier (Layer 1 Shell Structural Parser)
        # =========================================================================
        classifier = self._get_model("neuro_tool_failure_classifier")
        shell_res: Optional[Dict[str, Any]] = None
        if action_name in ["run_command", "execute_shell"] and path_str:
            shell_res = analyze_shell_command(path_str)
            is_destructive_shell = not shell_res["is_safe"]
        else:
            is_destructive_shell = False
        tf_tokens = self._build_classifier_tokens(
            shell_res=shell_res,
            payload_diff=payload_diff,
            syntax_clean=syntax_clean,
            security_clean=security_clean,
        )

        if classifier is None:
            tf_verdict = "ABSTAIN"
            tf_conf = 0.50
            tf_nonconf = 1.0
            tf_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            tf_res = classifier.infer_conformal(tf_tokens, alpha=0.05)
            tf_pred = tf_res.get("predicted_token", "")
            tf_abstained = bool(tf_res.get("abstained", False))
            tf_conf = float(tf_res.get("confidence", 0.50))
            if tf_abstained:
                tf_verdict = "ABSTAIN"
            elif tf_pred in CLASSIFIER_DENY_TOKENS:
                tf_verdict = "DENY"
            elif tf_pred in CLASSIFIER_ALLOW_TOKENS:
                tf_verdict = "ALLOW"
            else:
                tf_verdict = "ABSTAIN"
            tf_nonconf = round(1.0 - tf_conf, 4) if tf_verdict == "ALLOW" else round(tf_conf, 4)
            tf_details = {
                "is_destructive": is_destructive_shell,
                "tokens": tf_tokens,
                "predicted_token": tf_pred,
                "prediction_set": tf_res.get("prediction_set", []),
                "abstained": tf_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="tool_failure_classifier",
            verdict=tf_verdict,
            confidence=tf_conf,
            nonconformity_score=tf_nonconf,
            details=tf_details,
        ))

        # =========================================================================
        # Aggregate Votes, Measure Estimator Variance, and Decide
        # =========================================================================
        allow_probs = [
            v.confidence if v.verdict == "ALLOW" else (1.0 - v.confidence if v.verdict == "DENY" else 0.50)
            for v in votes
        ]
        mean_prob = float(np.mean(allow_probs))
        variance = float(np.var(allow_probs))
        std_dev = float(np.std(allow_probs))

        deny_votes = [v for v in votes if v.verdict == "DENY"]
        abstain_votes = [v for v in votes if v.verdict == "ABSTAIN"]
        allow_votes = [v for v in votes if v.verdict == "ALLOW"]

        # Hard invariant gate: certified rejection by constitutional, syntax, or destructive boundary models
        hard_veto_estimators = {"constitutional_guardian", "code_edit_verifier", "tool_failure_classifier"}
        hard_veto = any(v.estimator_name in hard_veto_estimators and v.confidence >= 0.85 for v in deny_votes)
        var_threshold = max(0.0, float(self.variance_threshold))
        max_std = math.sqrt(var_threshold)

        if hard_veto or len(deny_votes) >= 2 or mean_prob < 0.40:
            status = "CONSENSUS_REJECTED"
            abstention = False
            rationale = (
                f"Consensus rejected: Certified invariant violation detected "
                f"({len(deny_votes)} DENY votes: {[v.estimator_name for v in deny_votes]}, mean_prob={mean_prob:.3f}). Direct block."
            )
        elif len(deny_votes) == 0 and len(abstain_votes) == 0 and variance <= (var_threshold + 1e-12) and mean_prob >= 0.75:
            # Strict unanimous consensus required for autonomous approval
            status = "CONSENSUS_APPROVED"
            abstention = False
            rationale = (
                f"Ensemble unanimous consensus approved (all {len(allow_votes)} models voted ALLOW, "
                f"mean confidence {mean_prob:.3f}, var={variance:.4f} (threshold={var_threshold:.4f}, std={std_dev:.3f} <= {max_std:.2f}))."
            )
        else:
            status = "ABSTAIN_DELEGATE_TO_LLM"
            abstention = True
            rationale = (
                f"Selective Abstention triggered: Non-unanimous agreement, estimator disagreement, or uncertainty "
                f"(var={variance:.4f} (threshold={var_threshold:.4f}, std={std_dev:.3f} <= {max_std:.2f}), "
                f"mean={mean_prob:.3f}, deny_models={len(deny_votes)}, abstained_models={len(abstain_votes)}). "
                f"Escalating boundary case to Cascade Router / LLM."
            )

        return CriticalConsensusResult(
            status=status,
            consensus_score=mean_prob,
            estimator_variance=variance,
            abstention_triggered=abstention,
            votes=votes,
            rationale=rationale,
        )

    def evaluate_with_cascade(
        self,
        action_name: str,
        target_path: Optional[str] = None,
        payload_diff: Optional[str] = None,
        llm_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Full lifecycle evaluation: Critical Microbrain Ensemble -> Cascade / External LLM escalation.

        Never fabricates `AUTHORIZED` or `llm_invoked=True` when the ensemble abstained
        and no `llm_callback` was supplied.
        """
        consensus = self.evaluate_critical_action(action_name, target_path, payload_diff)
        if consensus.status == "CONSENSUS_APPROVED":
            return {
                "final_verdict": "AUTHORIZED",
                "tier": "MICROBRAIN_CONSENSUS",
                "consensus_result": consensus,
                "llm_invoked": False,
            }
        elif consensus.status == "CONSENSUS_REJECTED":
            return {
                "final_verdict": "BLOCKED",
                "tier": "MICROBRAIN_REJECTION",
                "consensus_result": consensus,
                "llm_invoked": False,
            }
        else:
            from .cascade_ensemble import get_cascade_ensemble_router
            router = get_cascade_ensemble_router()
            cascade_res = router.evaluate_action_pipeline(
                task_prompt=f"Escalated action: {action_name} on {target_path}",
                target_tool=action_name,
                tool_args={"path": target_path, "diff": payload_diff, "cmd": target_path},
            )
            if cascade_res.verdict in ("BLOCKED_CONSTITUTIONAL", "BLOCKED_PRECONDITION_SUKI"):
                return {
                    "final_verdict": cascade_res.verdict,
                    "tier": "CASCADE_REJECTION",
                    "consensus_result": consensus,
                    "cascade_verdict": cascade_res,
                    "llm_invoked": False,
                }
            if llm_callback is not None:
                llm_out = llm_callback({
                    "action_name": action_name,
                    "target_path": target_path,
                    "consensus_result": consensus.to_dict(),
                    "cascade_verdict": cascade_res.to_dict(),
                })
                final_v = llm_out.get("verdict", "ESCALATED_TO_LLM") if isinstance(llm_out, dict) else str(llm_out)
                return {
                    "final_verdict": final_v,
                    "tier": "LLM_CASCADE_ESCALATION",
                    "consensus_result": consensus,
                    "cascade_verdict": cascade_res,
                    "llm_response": llm_out,
                    "llm_invoked": True,
                }
            return {
                "final_verdict": "ESCALATED_TO_LLM",
                "tier": "LLM_CASCADE_ESCALATION",
                "consensus_result": consensus,
                "cascade_verdict": cascade_res,
                "llm_invoked": False,
                "requires_external_llm": True,
            }


_critical_ensemble_singleton: Optional[CriticalEnsemble] = None


def get_critical_ensemble() -> CriticalEnsemble:
    """Singleton getter for CriticalEnsemble."""
    global _critical_ensemble_singleton
    if _critical_ensemble_singleton is None:
        _critical_ensemble_singleton = CriticalEnsemble()
    return _critical_ensemble_singleton
