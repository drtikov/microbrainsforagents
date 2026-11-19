"""
Dynamic Tool: Neuro-Route — fast CPU-side routing over the neuro-brain population
=================================================================================
Reads (never grows) the grown micro-transformer population:

- route:            MicroMoEOrchestrator softmax gating -> nursery.infer on the
                    chosen expert (real forward pass on stored weights).
- plan_chain:       ZeroShotToolChainer -> HierarchicalMoECortex + HD-VSA
                    pipeline plan for compound goals (plans only, executes nothing).
- associative_recall: WeightSpaceMemorySynthesizer Hopfield read side — the
                    memory that evolution/extract.py distills into on auto-deploy.

Successful routes feed MorphogeneticSynaptogenesis so the connectome maintained
by the ConcentricSpiralDAG synaptogenesis node is driven by real usage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

schema = {
    "name": "neuro_route",
    "description": "Fast CPU routing over the grown neuro-brain population: MoE-gated micro-transformer inference, compound-goal pipeline planning (HD-VSA + hierarchical MoE), and associative recall over distilled deployment memory. Read-only: never grows or mutates brains.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["route", "plan_chain", "associative_recall", "conformal_route", "resident_hdc_search", "ensemble_route"],
                "description": "Routing action to perform."
            },
            "tokens": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Input tokens for the route action (e.g. an intent like 'analyze_stars')."
            },
            "query_text": {
                "type": "string",
                "description": "Free text query for plan_chain / associative_recall."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of experts to dispatch in the route action.",
                "default": 1
            },
            "alpha": {
                "type": "number",
                "description": "Miscoverage tolerance level alpha for conformal_route.",
                "default": 0.10
            }
        },
        "required": ["action"]
    }
}


def main(
    action: str = "route",
    tokens: Optional[List[str]] = None,
    query_text: str = "",
    top_k: int = 1,
    **kwargs: Any
) -> Dict[str, Any]:
    if action == "route":
        from ..core.brain_moe_orchestrator import MicroMoEOrchestrator
        route_tokens = tokens or ([query_text] if query_text else [])
        if not route_tokens:
            return {"success": False, "error": "route requires tokens or query_text"}
        result = MicroMoEOrchestrator.get_instance().route_query(route_tokens, top_k=int(top_k))
        if not result.get("success", "chosen_expert" in result):
            return result
        # Feed the real interaction into the spiral-maintained connectome:
        # uncertain routes (high prediction error) are exactly the ones the
        # morphogenetic engine sprouts exploratory axons for.
        try:
            from ..core.morphogenetic_synaptogenesis import MorphogeneticSynaptogenesis
            expert = result.get("chosen_expert")
            confidence = float(result.get("routing_confidence", 0.0))
            is_route_success = bool(result.get("success", True))
            if expert:
                MorphogeneticSynaptogenesis.get_instance().register_interaction(
                    source_brain="dispatcher",
                    target_brain=str(expert),
                    prediction_error=max(0.0, 1.0 - confidence),
                    success=is_route_success,
                )
        except Exception:
            pass
        result["action"] = "route"
        result.setdefault("success", True)
        return result

    elif action == "plan_chain":
        from ..core.zero_shot_tool_chainer import ZeroShotToolChainer
        if not query_text:
            return {"success": False, "error": "plan_chain requires query_text"}
        result = ZeroShotToolChainer.get_instance().decompose_and_chain(query_text)
        result["action"] = "plan_chain"
        result.setdefault("success", bool(result.get("pipeline")))
        return result

    elif action == "associative_recall":
        from ..core.weight_space_memory_synthesis import WeightSpaceMemorySynthesizer
        if not query_text:
            return {"success": False, "error": "associative_recall requires query_text"}
        result = WeightSpaceMemorySynthesizer.get_instance().query_associative_memory(query_text)
        result["action"] = "associative_recall"
        result.setdefault("success", True)
        return result

    elif action == "conformal_route":
        from ..core.brain_moe_orchestrator import MicroMoEOrchestrator
        from ..core.brain_nursery import AutonomousBrainNursery
        route_tokens = tokens or ([query_text] if query_text else [])
        if not route_tokens:
            return {"success": False, "error": "conformal_route requires tokens or query_text"}
        moe_res = MicroMoEOrchestrator.get_instance().route_query(route_tokens, top_k=1)
        if not moe_res.get("success", "chosen_expert" in moe_res):
            return moe_res
        chosen = moe_res.get("chosen_expert") or "general_domain"
        # strip neuro_ prefix if present
        domain = chosen.replace("neuro_", "")
        alpha = float(kwargs["alpha"]) if "alpha" in kwargs else None
        res = AutonomousBrainNursery.get_instance().infer_brain_conformal(domain, route_tokens, alpha=alpha)
        res["action"] = "conformal_route"
        res["chosen_expert"] = chosen
        res.setdefault("success", True)
        return res

    elif action == "resident_hdc_search":
        from ..core.evolution_surrogate_prefilter import get_experience_memory
        if not query_text:
            return {"success": False, "error": "resident_hdc_search requires query_text"}
        res = get_experience_memory().search(query_text, top_k=int(top_k))
        res["action"] = "resident_hdc_search"
        res.setdefault("success", True)
        return res

    elif action == "ensemble_route":
        from collections import Counter
        from ..core.brain_moe_orchestrator import MicroMoEOrchestrator
        from ..core.brain_nursery import AutonomousBrainNursery
        route_tokens = tokens or ([query_text] if query_text else [])
        if not route_tokens:
            return {"success": False, "error": "ensemble_route requires tokens or query_text"}
        
        # Top-3 consensus with min_agreement threshold (default 0.70)
        moe_res = MicroMoEOrchestrator.get_instance().route_query(route_tokens, top_k=3)
        dispatch_plan = moe_res.get("dispatch_plan") or [
            {"expert": d.get("expert_name") or d.get("expert"), "score": d.get("routing_prob", 0.0)}
            for d in moe_res.get("all_dispatches", [])
        ]
        if not dispatch_plan and "chosen_expert" in moe_res:
            dispatch_plan = [{"expert": moe_res["chosen_expert"], "score": moe_res.get("routing_confidence", 1.0)}]

        nursery = AutonomousBrainNursery.get_instance()
        predictions: List[str] = []
        confidences: List[float] = []
        experts_consulted: List[str] = []

        for item in dispatch_plan[:3]:
            exp_name = item.get("expert") or item.get("name")
            if not exp_name:
                continue
            dom = exp_name.replace("neuro_", "")
            try:
                inf = nursery.infer_brain_conformal(dom, route_tokens)
                token_cov = float(inf.get("token_coverage", 1.0))
                if token_cov <= 0.0 or inf.get("reason") == "ood_coverage_breach":
                    continue
                pred = inf.get("predicted_token") or inf.get("top_candidate")
                conf = inf.get("confidence", 0.5)
                if pred and pred not in ("<pad>", "<unk>", "<eos>", "<abstain>"):
                    predictions.append(pred)
                    confidences.append(float(conf))
                    experts_consulted.append(dom)
            except Exception:
                continue

        min_agreement = float(kwargs.get("min_agreement", 0.70))

        if not predictions:
            return {
                "success": True,
                "action": "ensemble_route",
                "consensus_token": None,
                "agreement_ratio": 0.0,
                "escalate_to_llm": True,
                "fallback": True,
                "reason": "no valid expert predictions",
                "consulted_experts": experts_consulted,
            }

        counts = Counter(predictions)
        top_token, top_count = counts.most_common(1)[0]
        agreement_ratio = top_count / len(predictions)
        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        if agreement_ratio >= min_agreement:
            return {
                "success": True,
                "action": "ensemble_route",
                "consensus_token": top_token,
                "agreement_ratio": round(agreement_ratio, 3),
                "escalate_to_llm": False,
                "fallback": False,
                "consulted_experts": experts_consulted,
                "predictions": predictions,
                "mean_confidence": round(mean_conf, 4),
            }
        else:
            return {
                "success": True,
                "action": "ensemble_route",
                "consensus_token": top_token,
                "agreement_ratio": round(agreement_ratio, 3),
                "escalate_to_llm": True,
                "fallback": True,
                "reason": f"agreement {agreement_ratio:.2f} < threshold {min_agreement:.2f}",
                "consulted_experts": experts_consulted,
                "predictions": predictions,
                "mean_confidence": round(mean_conf, 4),
            }

    return {"success": False, "error": f"Unknown action: {action}"}
