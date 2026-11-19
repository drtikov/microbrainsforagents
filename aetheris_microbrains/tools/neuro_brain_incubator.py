"""
Dynamic Tool: Neuro-Brain Incubator & Fast Inference Engine
===========================================================
Provides dynamic execution tools for Aetheris to grow, compile, inspect,
and execute autonomous analytical micro-transformers without GPU or gradient descent.

Also exposes the deliberate brain-lifecycle operations (merge / fission /
plasticity / hebbian adaptation / speculative drafting) so the neuro-organ
population can be actively curated instead of only grown.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from ..core.brain_nursery import AutonomousBrainNursery

schema = {
    "name": "neuro_brain_incubator",
    "description": "Grows, compiles, and executes autonomous analytical micro-transformers ('neuro-organs') in CPU RAM zero-shot, without NVIDIA GPUs or gradient descent. Also curates the population: merge, fission, plasticity feedback, hebbian adaptation, speculative tool drafting.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "grow_brain", "infer", "list_brains", "compile_rule",
                    "merge_slerp", "merge_ties", "fission_check",
                    "plasticity_feedback", "hebbian_update", "speculative_draft",
                    "evolve_eggroll_es", "evolve_dns", "conformal_calibrate", "conformal_infer",
                ],
                "description": "Action to perform in the neuro-brain nursery."
            },
            "domain_name": {
                "type": "string",
                "description": "Name of the cognitive domain (e.g., 'routing_orchestrator', 'math_solver', 'safety_auditor')."
            },
            "rules": {
                "type": "object",
                "description": "Dictionary of trigger -> output symbolic rules to compile into transformer MLP weights."
            },
            "input_tokens": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Sequence of tokens to feed into the micro-transformer forward pass."
            },
            "trigger_token": {
                "type": "string",
                "description": "Single trigger token for compile_rule / plasticity_feedback actions."
            },
            "target_output": {
                "type": "string",
                "description": "Target response token for compile_rule action."
            },
            "brain_name_a": {
                "type": "string",
                "description": "First source brain for merge_slerp (nursery name without the neuro_ prefix)."
            },
            "brain_name_b": {
                "type": "string",
                "description": "Second source brain for merge_slerp."
            },
            "brain_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Source brain list for merge_ties."
            },
            "fused_brain_name": {
                "type": "string",
                "description": "Name of the fused brain produced by merge_slerp / merge_ties."
            },
            "interpolation_weight": {
                "type": "number",
                "description": "SLERP interpolation weight (0.0 = brain_a, 1.0 = brain_b).",
                "default": 0.5
            },
            "predicted_token": {
                "type": "string",
                "description": "Token the brain predicted, for plasticity_feedback."
            },
            "success": {
                "type": "boolean",
                "description": "Whether the executed prediction succeeded, for plasticity_feedback.",
                "default": True
            },
            "quality_score": {
                "type": "number",
                "description": "Outcome quality in [0, 1], for plasticity_feedback.",
                "default": 1.0
            },
            "brain_name": {
                "type": "string",
                "description": "Target brain for hebbian_update."
            },
            "input_activation": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Input activation vector for hebbian_update."
            },
            "output_feedback": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Output feedback vector for hebbian_update."
            },
            "reinforcement_signal": {
                "type": "number",
                "description": "Reinforcement scalar for hebbian_update.",
                "default": 1.0
            },
            "intention_keyword": {
                "type": "string",
                "description": "Intention keyword for speculative_draft."
            },
            "context_hints": {
                "type": "object",
                "description": "Optional context hints for speculative_draft."
            },
            "pop_size": {
                "type": "integer",
                "description": "Population size for evolve_eggroll_es / evolve_dns.",
                "default": 256
            },
            "seed": {
                "type": "integer",
                "description": "RNG seed for evolve_eggroll_es / evolve_dns.",
                "default": 42
            },
            "x_samples": {
                "type": "array",
                "description": "Input feature batch for evolve_eggroll_es / evolve_dns."
            },
            "y_targets": {
                "type": "array",
                "description": "Target batch for evolve_eggroll_es / evolve_dns."
            },
            "calibration_data": {
                "type": "array",
                "description": "List of [input_tokens, target_token] pairs for conformal_calibrate."
            },
            "alpha": {
                "type": "number",
                "description": "Miscoverage tolerance level alpha in (0, 1) for conformal calibration/inference.",
                "default": 0.10
            }
        },
        "required": ["action"]
    }
}


def run(
    action: str = "list_brains",
    domain_name: str = "general_domain",
    rules: Optional[Dict[str, str]] = None,
    input_tokens: Optional[List[str]] = None,
    trigger_token: Optional[str] = None,
    target_output: Optional[str] = None,
    brain_name_a: Optional[str] = None,
    brain_name_b: Optional[str] = None,
    brain_names: Optional[List[str]] = None,
    fused_brain_name: Optional[str] = None,
    interpolation_weight: float = 0.5,
    predicted_token: Optional[str] = None,
    success: bool = True,
    quality_score: float = 1.0,
    brain_name: Optional[str] = None,
    input_activation: Optional[List[float]] = None,
    output_feedback: Optional[List[float]] = None,
    reinforcement_signal: float = 1.0,
    intention_keyword: Optional[str] = None,
    context_hints: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    nursery = AutonomousBrainNursery.get_instance()

    if action == "list_brains":
        brains = nursery.list_grown_brains()
        return {
            "success": True,
            "action": "list_brains",
            "count": len(brains),
            "brains": brains,
        }

    elif action == "grow_brain":
        d_model = int(kwargs.get("d_model", 32))
        n_heads = int(kwargs.get("n_heads", 2))
        d_mlp = int(kwargs["d_mlp"]) if "d_mlp" in kwargs else None
        max_seq_len = int(kwargs["max_seq_len"]) if "max_seq_len" in kwargs else None
        brain = nursery.grow_brain_for_domain(
            domain_name=domain_name,
            rules=rules,
            d_model=d_model,
            n_heads=n_heads,
            d_mlp=d_mlp,
            max_seq_len=max_seq_len,
        )
        return {
            "success": True,
            "action": "grow_brain",
            "brain_name": brain.name,
            "d_model": brain.d_model,
            "n_heads": brain.n_heads,
            "d_mlp": brain.d_mlp,
            "max_seq_len": brain.max_seq_len,
            "vocab_size": len(brain.vocab),
            "storage_path": f"memory/store/neuro_brains/{brain.name}_weights.npz",
        }

    elif action == "compile_rule":
        if not trigger_token or not target_output:
            return {"success": False, "error": "trigger_token and target_output are required"}
        brain = nursery.grow_brain_for_domain(domain_name=domain_name)
        brain.compile_logic_association(trigger_token, target_output)
        brain.save()
        return {
            "success": True,
            "action": "compile_rule",
            "brain_name": brain.name,
            "trigger": trigger_token,
            "target": target_output,
        }

    elif action == "infer":
        if input_tokens:
            tokens = input_tokens
        elif trigger_token:
            tokens = [trigger_token]
        else:
            tokens = ["<pad>"]
        out = nursery.infer(domain_name=domain_name, input_tokens=tokens)
        return {
            "success": True,
            "action": "infer",
            "domain_name": domain_name,
            "input_tokens": tokens,
            "result": out,
        }

    elif action == "merge_slerp":
        from ..core.brain_merger import NeuroBrainMerger
        if not brain_name_a or not brain_name_b or not fused_brain_name:
            return {"success": False, "error": "brain_name_a, brain_name_b and fused_brain_name are required"}
        fused = NeuroBrainMerger.get_instance().merge_two_brains_slerp(
            brain_name_a, brain_name_b, fused_brain_name,
            interpolation_weight=float(interpolation_weight),
        )
        if fused is None:
            return {"success": False, "error": f"source brains not found in nursery: {brain_name_a}, {brain_name_b}"}
        # Register fused model weights in WeightSpaceVSA
        # Register fused model weights in WeightSpaceVSA
        try:
            from ..core.weight_space_vsa import get_weight_space_vsa
            vsa = get_weight_space_vsa()
            vsa.store_brain_weights(fused.name, fused.weights)
            vsa.save()
        except Exception:
            pass
        return {
            "success": True,
            "action": "merge_slerp",
            "fused_brain_name": fused.name,
            "vocab_size": len(fused.vocab),
            "storage_path": f"memory/store/neuro_brains/{fused.name}_weights.npz",
        }

    elif action == "merge_ties":
        from ..core.brain_merger import NeuroBrainMerger
        if not brain_names or not fused_brain_name:
            return {"success": False, "error": "brain_names and fused_brain_name are required"}
        fused = NeuroBrainMerger.get_instance().merge_brains_ties(brain_names, fused_brain_name)
        if fused is None:
            return {"success": False, "error": f"source brains not found in nursery: {brain_names}"}
        # Register fused model weights in WeightSpaceVSA
        # Register fused model weights in WeightSpaceVSA
        try:
            from ..core.weight_space_vsa import get_weight_space_vsa
            vsa = get_weight_space_vsa()
            vsa.store_brain_weights(fused.name, fused.weights)
            vsa.save()
        except Exception:
            pass
        return {
            "success": True,
            "action": "merge_ties",
            "fused_brain_name": fused.name,
            "source_count": len(brain_names),
            "storage_path": f"memory/store/neuro_brains/{fused.name}_weights.npz",
        }

    elif action == "fission_check":
        from ..core.brain_cellular_fission import BrainCellularFissionEngine
        result = BrainCellularFissionEngine.get_instance().evaluate_and_split_if_saturated(domain_name)
        result["success"] = result.get("status") in ("healthy", "fission_executed")
        result["action"] = "fission_check"
        # Nearest neighbor domain search in weight space
        try:
            from ..core.weight_space_vsa import get_weight_space_vsa
            vsa = get_weight_space_vsa()
            result["vsa_nearest_domains"] = vsa.find_nearest_domains(domain_name, top_k=3)
        except Exception:
            result["vsa_nearest_domains"] = []
        return result

    elif action == "plasticity_feedback":
        from ..core.brain_plasticity_updater import BrainPlasticityUpdater
        if not trigger_token or not predicted_token:
            return {"success": False, "error": "trigger_token and predicted_token are required"}
        result = BrainPlasticityUpdater.get_instance().process_execution_feedback(
            domain_name, trigger_token, predicted_token,
            success=bool(success), quality_score=float(quality_score),
        )
        result["success"] = result.get("status") == "updated"
        result["action"] = "plasticity_feedback"
        return result

    elif action == "hebbian_update":
        from ..core.online_hebbian_adaptor import OnlineHebbianAdaptor
        if not brain_name or input_activation is None or output_feedback is None:
            return {"success": False, "error": "brain_name, input_activation and output_feedback are required"}
        result = OnlineHebbianAdaptor.get_instance().apply_plastic_update(
            brain_name, input_activation, output_feedback,
            reinforcement_signal=float(reinforcement_signal),
        )
        result["success"] = result.get("status") == "success"
        result["action"] = "hebbian_update"
        return result

    elif action == "speculative_draft":
        from ..core.brain_speculative_drafter import AgentSpeculativeDrafter
        if not intention_keyword:
            return {"success": False, "error": "intention_keyword is required"}
        result = AgentSpeculativeDrafter.get_instance().draft_tool_invocation(
            intention_keyword, context_hints=context_hints,
        )
        result["success"] = bool(result.get("tool"))
        result["action"] = "speculative_draft"
        return result

    elif action == "evolve_eggroll_es":
        import numpy as np
        pop_size = int(kwargs.get("pop_size", 256))
        seed = int(kwargs.get("seed", 42))
        x_in = np.array(kwargs.get("x_samples", np.eye(32)), dtype=np.float32)
        y_tgt = np.array(kwargs.get("y_targets", np.eye(32)), dtype=np.float32)
        res = nursery.evolve_population_eggroll_es(
            domain_name=domain_name,
            x_samples=x_in,
            y_targets=y_tgt,
            pop_size=pop_size,
            seed=seed,
        )
        res["success"] = True
        res["action"] = "evolve_eggroll_es"
        return res

    elif action == "evolve_dns":
        import numpy as np
        pop_size = int(kwargs.get("pop_size", 256))
        seed = int(kwargs.get("seed", 42))
        x_in = np.array(kwargs.get("x_samples", np.eye(32)), dtype=np.float32)
        y_tgt = np.array(kwargs.get("y_targets", np.eye(32)), dtype=np.float32)
        res = nursery.evolve_population_dns(
            domain_name=domain_name,
            x_samples=x_in,
            y_targets=y_tgt,
            pop_size=pop_size,
            seed=seed,
        )
        res["success"] = True
        res["action"] = "evolve_dns"
        return res

    elif action == "conformal_calibrate":
        cal_data = kwargs.get("calibration_data") or []
        alpha = float(kwargs.get("alpha", 0.10))
        res = nursery.calibrate_brain_conformal(domain_name=domain_name, calibration_data=cal_data, alpha=alpha)
        res["success"] = res.get("status") == "calibrated"
        res["action"] = "conformal_calibrate"
        return res

    elif action == "conformal_infer":
        tokens = input_tokens or []
        alpha = float(kwargs["alpha"]) if "alpha" in kwargs else None
        res = nursery.infer_brain_conformal(domain_name=domain_name, input_tokens=tokens, alpha=alpha)
        res["success"] = not res.get("abstained", False)
        res["action"] = "conformal_infer"
        return res

    return {"success": False, "error": f"Unknown action: {action}"}


def main(**kwargs: Any) -> Dict[str, Any]:
    """Dispatcher entry point (laws-gate conforming: accepts **kwargs)."""
    return run(**kwargs)
