"""Cascade Ensemble & Multi-Tier Routing for Critical Decision Gating.

Implements a four-tier cascading decision pipeline across microbrains:
  Tier 1: Intent Triage (`fast_intent_classifier`, <1ms classification)
  Tier 2: Tool Selection (`tool_routing_v2`, d_model=128 high-capacity selection)
  Tier 3: Constitutional Gate (`constitutional_guardian` + `code_edit_verifier`)
  Tier 4: Speculative Pre-Action Check (`counterfactual_reasoner` + `tool_failure_classifier`)

Aggregation:
  - Combines conformal quantile thresholds and non-conformity scores across tiers.
  - If any safety or Suki-class failure exceeds threshold, halts execution and
    routes to a safe fallback instead of executing an ungrounded mutation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR
from .critical_ensemble_abstention import (
    CLASSIFIER_DENY_TOKENS,
    CriticalEnsemble,
    GUARDIAN_DENY_TOKENS,
)
from .layered_decision_engine import (
    analyze_code_diff,
    analyze_shell_command,
    is_protected_path,
)
from .tool_routing_moe import ToolRoutingMoE, get_tool_routing_moe

log = logging.getLogger(__name__)


@dataclass
class CascadeTierResult:
    """Outcome of an individual tier within the cascade ensemble."""

    tier_name: str
    brain_used: str
    decision: str
    confidence: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeEnsembleVerdict:
    """Final aggregated verdict of the cascade ensemble pipeline."""

    authorized: bool
    verdict: str  # "AUTHORIZED", "BLOCKED_CONSTITUTIONAL", "BLOCKED_PRECONDITION_SUKI", "DEGRADED_FALLBACK"
    primary_intent: str
    target_tool: str
    cumulative_confidence: float
    tier_results: List[CascadeTierResult] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    escalation_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "authorized": self.authorized,
            "verdict": self.verdict,
            "primary_intent": self.primary_intent,
            "target_tool": self.target_tool,
            "cumulative_confidence": round(self.cumulative_confidence, 4),
            "tier_results": [
                {
                    "tier": r.tier_name,
                    "brain": r.brain_used,
                    "decision": r.decision,
                    "confidence": round(r.confidence, 4),
                    "passed": r.passed,
                }
                for r in self.tier_results
            ],
            "reasons": self.reasons,
        }
        if self.escalation_payload is not None:
            out["escalation_payload"] = self.escalation_payload
        return out


class CascadeEnsembleRouter:
    """Orchestrates multi-brain cascading verification before critical action dispatch."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or STORE_DIR
        self._brains_cache: Dict[str, Any] = {}

    def _get_brain(self, name: str) -> Optional[Any]:
        if name in self._brains_cache:
            return self._brains_cache[name]

        brain = AnalyticalMicroTransformer.load(name, self.store_dir)
        if brain is not None:
            self._brains_cache[name] = brain
        return brain

    def evaluate_action_pipeline(
        self,
        task_prompt: str,
        target_tool: str,
        tool_args: Optional[Dict[str, Any]] = None,
        context_tags: Optional[List[str]] = None,
    ) -> CascadeEnsembleVerdict:
        """Run complete 4-tier cascading evaluation backed by conformal & structural signals."""
        tool_args = tool_args or {}
        context_tags = context_tags or []
        tiers: List[CascadeTierResult] = []
        reasons: List[str] = []
        feature_builder = CriticalEnsemble(store_dir=self.store_dir)

        standard_cli_tools = {
            "ls", "echo", "pwd", "git", "clang-format", "cat", "head", "tail", "wc",
            "grep", "rg", "find", "black", "ruff", "mypy", "pytest", "apply_patch", "delete_file",
        }
        known_moe_tools = {t for ts in ToolRoutingMoE.EXPERT_TOOLS.values() for t in ts}
        known_tools = known_moe_tools | standard_cli_tools
        clean_tool = str(target_tool or "").strip().lower()
        tool_head = clean_tool.split()[0] if clean_tool else ""

        cmd = str(tool_args.get("cmd") or tool_args.get("command") or tool_args.get("CommandLine") or "")
        if not cmd and target_tool and any(ch in target_tool for ch in (" ", "/", "|", ";")):
            cmd = target_tool

        # --- Tier 1: Intent Triage (No synthetic ["read"] substitution on OOD) ---
        fast_intent = self._get_brain("neuro_fast_intent_classifier")
        prompt_tokens = re.findall(r"(?u)\b\w{2,}\b", str(task_prompt or "").lower())
        known_prompt_tokens = [t for t in prompt_tokens if fast_intent and t in fast_intent.vocab]
        inf1: Dict[str, Any] = {}
        tier1_details: Dict[str, Any] = {"model_invoked": False}
        if fast_intent is not None:
            inf1 = fast_intent.infer_conformal(prompt_tokens or ["<pad>"], alpha=0.05)
            tier1_details = {
                "model_invoked": True,
                "predicted_token": inf1.get("predicted_token"),
                "prediction_set": inf1.get("prediction_set", []),
                "abstained": inf1.get("abstained"),
            }

        moe_decision = get_tool_routing_moe().route(task_prompt, context_tags=context_tags)
        if known_prompt_tokens and not inf1.get("abstained") and inf1.get("predicted_token") not in ("<abstain>", "<unk>", "<eos>"):
            intent_decision = str(inf1["predicted_token"])
            intent_conf = float(inf1.get("confidence", 0.90))
            tier1_passed = intent_conf >= 0.70
        elif moe_decision.conformal_passed:
            intent_decision = moe_decision.selected_tool
            intent_conf = max(0.85, float(moe_decision.confidence))
            tier1_passed = True
        elif tool_head in known_tools:
            intent_decision = "execute"
            intent_conf = 0.88
            tier1_passed = True
        else:
            intent_decision = "<abstain>"
            intent_conf = 0.55
            tier1_passed = False
            reasons.append("Tier 1 intent triage abstained on OOD prompt with unrecognized tool")

        tiers.append(CascadeTierResult(
            tier_name="Tier1_IntentTriage",
            brain_used="neuro_fast_intent_classifier",
            decision=intent_decision,
            confidence=intent_conf,
            passed=tier1_passed,
            details=tier1_details,
        ))

        # --- Tier 2: Tool Routing Validation (No synthetic ["tool_repo_read"] substitution) ---
        routing_brain = self._get_brain("neuro_tool_routing_v2")
        tool_tok = f"tool_{clean_tool}"
        tier2_details: Dict[str, Any] = {"model_invoked": False}
        inf2: Dict[str, Any] = {}
        if routing_brain is not None:
            inf2 = routing_brain.infer_conformal([tool_tok], alpha=0.05)
            tier2_details = {
                "model_invoked": True,
                "predicted_token": inf2.get("predicted_token"),
                "prediction_set": inf2.get("prediction_set", []),
                "abstained": inf2.get("abstained"),
            }

        shell_pre = analyze_shell_command(cmd) if cmd else None
        if routing_brain is not None and tool_tok in routing_brain.vocab:
            routing_conf = max(0.85, float(inf2.get("confidence", 0.92))) if not inf2.get("abstained") else float(inf2.get("confidence", 0.60))
            routing_match = (not bool(inf2.get("abstained", False))) or routing_conf >= 0.80
        elif clean_tool in known_tools or tool_head in known_tools or (shell_pre is not None and shell_pre.get("allowlisted", False)):
            routing_match = True
            routing_conf = 0.92
        else:
            routing_match = False
            routing_conf = 0.55
            reasons.append(f"Tier 2 tool routing failed: '{target_tool}' is unregistered/unknown")

        tiers.append(CascadeTierResult(
            tier_name="Tier2_ToolRouting",
            brain_used="neuro_tool_routing_v2",
            decision=target_tool,
            confidence=routing_conf,
            passed=routing_match,
            details=tier2_details,
        ))

        # --- Tier 3: Constitutional & Protected Path Guard ---
        const_brain = self._get_brain("neuro_constitutional_guardian")
        target_path_str = str(tool_args.get("path") or tool_args.get("file_path") or "")
        is_sacred = is_protected_path(target_path_str)
        is_write_op = clean_tool in {"write_file", "edit_file", "repo_write", "apply_patch", "delete_file", "str_replace_editor"} or not clean_tool
        tier3_details: Dict[str, Any] = {"is_sacred": is_sacred, "model_invoked": False}

        const_passed = True
        const_conf = 0.95
        const_decision = "verdict_pass"
        if const_brain is not None:
            g_probe = feature_builder._build_guardian_tokens(target_path_str, clean_tool, is_sacred)
            inf3 = const_brain.infer_conformal(g_probe, alpha=0.05)
            pred3 = str(inf3.get("predicted_token") or "")
            const_conf = float(inf3.get("confidence", 0.95))
            tier3_details.update({
                "model_invoked": True,
                "tokens": g_probe,
                "predicted_token": pred3,
                "prediction_set": inf3.get("prediction_set", []),
                "abstained": inf3.get("abstained"),
            })
            if (is_sacred and is_write_op) or (not inf3.get("abstained") and pred3 in GUARDIAN_DENY_TOKENS):
                const_passed = False
                const_decision = pred3 or "verdict_block"
                const_conf = max(0.95, const_conf)
                reasons.append(f"Sacred Site / Constitutional block: '{target_path_str}' triggers protected boundary guard")
            elif inf3.get("abstained"):
                const_passed = False
                const_decision = "<abstain>"
                const_conf = min(const_conf, 0.60)
                reasons.append("Tier 3 constitutional guardian abstained on boundary probe")
            else:
                const_decision = pred3 or "verdict_pass"
        elif is_sacred and is_write_op:
            const_passed = False
            const_decision = "verdict_block"
            const_conf = 0.99
            reasons.append(f"Sacred Site / Constitutional block: '{target_path_str}' is a protected asset path")

        tiers.append(CascadeTierResult(
            tier_name="Tier3_ConstitutionalGate",
            brain_used="neuro_constitutional_guardian",
            decision=const_decision,
            confidence=const_conf,
            passed=const_passed,
            details=tier3_details,
        ))

        # --- Tier 4: Speculative Failure Check (Layer 1 Shell & AST Complexity Parsers) ---
        suki_brain = self._get_brain("neuro_tool_failure_classifier")
        suki_passed = True
        suki_conf = 0.95
        suki_class = "clean_success"

        shell_res: Optional[Dict[str, Any]] = None
        if clean_tool in {"execute_shell", "run_command"} or cmd:
            shell_res = analyze_shell_command(cmd)
            if not shell_res["is_safe"]:
                suki_passed = False
                suki_class = "suki_class_c_boundary"
                reasons.append(
                    f"Structural shell parser blocked destructive command ({', '.join(shell_res['findings'])})"
                )
            elif cmd and not shell_res.get("allowlisted", False):
                suki_conf = 0.65
                suki_class = "suki_unrecognized_shell_escalate"
                reasons.append(
                    f"Unrecognized shell command requires confirmation ({shell_res.get('unrecognized_commands')})"
                )

        diff_payload = tool_args.get("diff") or tool_args.get("payload_diff")
        syntax_clean = True
        security_clean = True
        if suki_passed and diff_payload:
            diff_res = analyze_code_diff(target_path_str, diff_payload)
            syntax_clean = bool(diff_res["syntax_valid"])
            security_clean = bool(diff_res["security_clean"])
            if not syntax_clean or not security_clean:
                suki_passed = False
                suki_class = "suki_class_ast_violation"
                reasons.append(
                    f"Structural AST parser blocked unsafe patch ({', '.join(diff_res['findings'])})"
                )
            elif diff_res.get("interface_hazards"):
                suki_passed = False
                suki_class = "suki_class_interface_hazard"
                reasons.append(
                    f"AST interface diff detected breaking change ({', '.join(diff_res['interface_hazards'])})"
                )
            elif diff_res.get("line_count", 0) > 480 and not diff_res.get("is_static_data_or_doc", False):
                suki_conf = 0.60
                suki_class = "suki_high_blast_radius"
                reasons.append(
                    f"Diff exceeds safe blast-radius line threshold ({diff_res['line_count']} > 480 lines)"
                )

        tier4_details: Dict[str, Any] = {"model_invoked": False}
        if suki_brain is not None:
            s_probe = feature_builder._build_classifier_tokens(
                shell_res=shell_res,
                payload_diff=diff_payload,
                syntax_clean=syntax_clean,
                security_clean=security_clean,
            )
            inf4 = suki_brain.infer_conformal(s_probe, alpha=0.05)
            pred4 = str(inf4.get("predicted_token") or "")
            conf4 = float(inf4.get("confidence", 0.95))
            tier4_details = {
                "model_invoked": True,
                "tokens": s_probe,
                "predicted_token": pred4,
                "prediction_set": inf4.get("prediction_set", []),
                "abstained": inf4.get("abstained"),
            }
            if not suki_passed or (not inf4.get("abstained") and pred4 in CLASSIFIER_DENY_TOKENS):
                suki_passed = False
                suki_conf = max(0.85, conf4)
                if suki_class == "clean_success":
                    suki_class = pred4 or "failure_fatal_circuit_breaker"
            elif inf4.get("abstained"):
                suki_passed = False
                suki_conf = min(suki_conf, 0.60)
                suki_class = "<abstain>"
            else:
                suki_conf = min(suki_conf, conf4)

        tiers.append(CascadeTierResult(
            tier_name="Tier4_SukiFailureCheck",
            brain_used="neuro_tool_failure_classifier",
            decision=suki_class,
            confidence=suki_conf,
            passed=suki_passed,
            details=tier4_details,
        ))

        # Evaluate weakest active verification tier confidence across pipeline stages
        active_confs = [t.confidence for t in tiers]
        bottleneck_conf = min(active_confs) if active_confs else 0.50

        # Final Aggregation
        all_passed = all(t.passed for t in tiers)
        escalation_payload: Optional[Dict[str, Any]] = None
        if not const_passed:
            verdict_str = "BLOCKED_CONSTITUTIONAL"
        elif not suki_passed:
            verdict_str = "BLOCKED_PRECONDITION_SUKI"
        elif (not all_passed) or bottleneck_conf < 0.70:
            verdict_str = "DEGRADED_FALLBACK"
            all_passed = False
            if bottleneck_conf < 0.70:
                reasons.append(f"Weakest-link tier confidence {bottleneck_conf:.3f} below gate threshold 0.70")
            else:
                failed_tiers = [t.tier_name for t in tiers if not t.passed]
                reasons.append(f"Cascade tier verification failed: {failed_tiers}")
            escalation_payload = {
                "target_tool": target_tool,
                "tool_args": tool_args,
                "task_prompt": task_prompt,
                "bottleneck_tier": min(tiers, key=lambda t: t.confidence).tier_name,
                "bottleneck_confidence": bottleneck_conf,
                "reasons": list(reasons),
                "delegate_to": "LLM_CASCADE_ROUTER",
            }
        else:
            verdict_str = "AUTHORIZED"

        return CascadeEnsembleVerdict(
            authorized=all_passed,
            verdict=verdict_str,
            primary_intent=intent_decision,
            target_tool=target_tool,
            cumulative_confidence=bottleneck_conf,
            tier_results=tiers,
            reasons=reasons,
            escalation_payload=escalation_payload,
        )


_cascade_router_singleton: Optional[CascadeEnsembleRouter] = None


def get_cascade_ensemble_router() -> CascadeEnsembleRouter:
    """Singleton getter for CascadeEnsembleRouter."""
    global _cascade_router_singleton
    if _cascade_router_singleton is None:
        _cascade_router_singleton = CascadeEnsembleRouter()
    return _cascade_router_singleton
