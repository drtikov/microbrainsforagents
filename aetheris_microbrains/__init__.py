"""
Aetheris Microbrains Standalone SDK
===================================
Self-contained, portable suite of lightweight pure-NumPy analytical microbrains,
layered structural AST/shell safety parsers, split-conformal abstention gates,
OpenCL acceleration, Weight-Space Programming (WSP) engine, Vector Symbolic
Architecture (VSA), deterministic Mixture of Experts (MoE), and cascade gates.

Clean Python package: does NOT mutate `sys.path` or shadow host modules.
Uses PEP 562 lazy imports so lightweight entrypoints (such as `hook.py`) only
load the stdlib decision engine without pulling in OpenCL or all ~30 modules.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__version__ = "0.7.4"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Hook
    "evaluate_hook_payload": (".hook", "evaluate_hook_payload"),
    # Layered decision engine (stdlib-only)
    "PolicyConfig": (".core.layered_decision_engine", "PolicyConfig"),
    "analyze_code_diff": (".core.layered_decision_engine", "analyze_code_diff"),
    "analyze_shell_command": (".core.layered_decision_engine", "analyze_shell_command"),
    "decide": (".core.layered_decision_engine", "decide"),
    "diagnose_patch": (".core.layered_decision_engine", "diagnose_patch"),
    "evaluate_layer0_policy": (".core.layered_decision_engine", "evaluate_layer0_policy"),
    "get_recent_decision_journal": (".core.layered_decision_engine", "get_recent_decision_journal"),
    "is_protected_path": (".core.layered_decision_engine", "is_protected_path"),
    "recalibrate_from_decision_journal": (".core.layered_decision_engine", "recalibrate_from_decision_journal"),
    "record_decision_journal": (".core.layered_decision_engine", "record_decision_journal"),
    # OpenCL & Core microbrains
    "OpenCLMicroBrainAccelerator": (".core.opencl_microbrain_accelerator", "OpenCLMicroBrainAccelerator"),
    "get_opencl_accelerator": (".core.opencl_microbrain_accelerator", "get_opencl_accelerator"),
    "DendriticTreeMicroBrain": (".core.dendritic_tree_microbrain", "DendriticTreeMicroBrain"),
    "KolmogorovArnoldMicroBrain": (".core.kolmogorov_arnold_microbrain", "KolmogorovArnoldMicroBrain"),
    "LinearStateSpaceMicroBrain": (".core.linear_state_space_microbrain", "LinearStateSpaceMicroBrain"),
    "MicroBrainHypernetwork": (".core.microbrain_hypernetwork", "MicroBrainHypernetwork"),
    "AnalyticalMicroTransformer": (".core.analytical_micro_transformer", "AnalyticalMicroTransformer"),
    "STORE_DIR": (".core.analytical_micro_transformer", "STORE_DIR"),
    "AutonomousBrainNursery": (".core.brain_nursery", "AutonomousBrainNursery"),
    "BrainNursery": (".core.brain_nursery", "BrainNursery"),
    "get_brain_nursery": (".core.brain_nursery", "get_brain_nursery"),
    "BitLinearLayer": (".core.bitnet_ternary_engine", "BitLinearLayer"),
    "BitNetQuantizer": (".core.bitnet_ternary_engine", "BitNetQuantizer"),
    "BitNetTernaryEngine": (".core.bitnet_ternary_engine", "BitNetTernaryEngine"),
    "LiquidStateReservoir": (".core.liquid_state_reservoir", "LiquidStateReservoir"),
    "HierarchicalPredictiveCodingEngine": (".core.predictive_coding_engine", "HierarchicalPredictiveCodingEngine"),
    "PredictiveCodingLayer": (".core.predictive_coding_engine", "PredictiveCodingLayer"),
    "PredictiveCodingEngine": (".core.predictive_coding_engine", "PredictiveCodingEngine"),
    "HyperdimensionalVSAEngine": (".core.hyperdimensional_vsa_engine", "HyperdimensionalVSAEngine"),
    "WSPNeuralSurgeon": (".core.wsp_neural_surgeon", "WSPNeuralSurgeon"),
    # Meta-cognitive
    "StructuralCausalModel": (".core.causal_world_model", "StructuralCausalModel"),
    "CausalWorldModel": (".core.causal_world_model", "CausalWorldModel"),
    "CausalVariable": (".core.causal_world_model", "CausalVariable"),
    "CausalIntervention": (".core.causal_world_model", "CausalIntervention"),
    "CausalSimulationResult": (".core.causal_world_model", "CausalSimulationResult"),
    "get_causal_world_model": (".core.causal_world_model", "get_causal_world_model"),
    "CounterfactualResult": (".core.counterfactual_reasoner", "CounterfactualResult"),
    "simulate_counterfactual": (".core.counterfactual_reasoner", "simulate_counterfactual"),
    "IntentPersistenceTracker": (".core.intent_persistence_tracker", "IntentPersistenceTracker"),
    "IntentDriftAssessment": (".core.intent_persistence_tracker", "IntentDriftAssessment"),
    "MetacognitiveActionVerdict": (".core.metacognitive_calibrator", "MetacognitiveActionVerdict"),
    "CalibratedConfidence": (".core.metacognitive_calibrator", "CalibratedConfidence"),
    "calibrate_action_confidence": (".core.metacognitive_calibrator", "calibrate_action_confidence"),
    "SkillLifecycleState": (".core.skill_lifecycle_manager", "SkillLifecycleState"),
    "SkillTelemetry": (".core.skill_lifecycle_manager", "SkillTelemetry"),
    "FleetHygieneReport": (".core.skill_lifecycle_manager", "FleetHygieneReport"),
    "SkillLifecycleManager": (".core.skill_lifecycle_manager", "SkillLifecycleManager"),
    "get_skill_lifecycle_manager": (".core.skill_lifecycle_manager", "get_skill_lifecycle_manager"),
    # SOTA Ouroboros Neural Extensions
    "WeightSpaceVSA": (".core.weight_space_vsa", "WeightSpaceVSA"),
    "VSAResult": (".core.weight_space_vsa", "VSAResult"),
    "get_weight_space_vsa": (".core.weight_space_vsa", "get_weight_space_vsa"),
    "ToolRoutingMoE": (".core.tool_routing_moe", "ToolRoutingMoE"),
    "MoERoutingDecision": (".core.tool_routing_moe", "MoERoutingDecision"),
    "SessionToolSurface": (".core.tool_routing_moe", "SessionToolSurface"),
    "get_tool_routing_moe": (".core.tool_routing_moe", "get_tool_routing_moe"),
    "narrow_tool_surface": (".core.tool_routing_moe", "narrow_tool_surface"),
    "rank_tools": (".core.tool_routing_moe", "rank_tools"),
    "CriticalEnsemble": (".core.critical_ensemble_abstention", "CriticalEnsemble"),
    "CriticalConsensusResult": (".core.critical_ensemble_abstention", "CriticalConsensusResult"),
    "EnsembleVote": (".core.critical_ensemble_abstention", "EnsembleVote"),
    "get_critical_ensemble": (".core.critical_ensemble_abstention", "get_critical_ensemble"),
    "CascadeEnsembleRouter": (".core.cascade_ensemble", "CascadeEnsembleRouter"),
    "CascadeTierResult": (".core.cascade_ensemble", "CascadeTierResult"),
    "CascadeEnsembleVerdict": (".core.cascade_ensemble", "CascadeEnsembleVerdict"),
    "get_cascade_ensemble_router": (".core.cascade_ensemble", "get_cascade_ensemble_router"),
    "ContinualLearningPipeline": (".core.continual_learning_pipeline", "ContinualLearningPipeline"),
    "ContinualUpdateResult": (".core.continual_learning_pipeline", "ContinualUpdateResult"),
    "PlasticityCycleScheduler": (".core.plasticity_cycle", "PlasticityCycleScheduler"),
    "PlasticityCycleReport": (".core.plasticity_cycle", "PlasticityCycleReport"),
    "get_plasticity_scheduler": (".core.plasticity_cycle", "get_plasticity_scheduler"),
    "MicrobrainTelemetry": (".core.microbrain_telemetry", "MicrobrainTelemetry"),
    "TelemetryEvent": (".core.microbrain_telemetry", "TelemetryEvent"),
    "DashboardReport": (".core.microbrain_telemetry", "DashboardReport"),
    "get_microbrain_telemetry": (".core.microbrain_telemetry", "get_microbrain_telemetry"),
    "CounterfactualSimulationPipeline": (".core.counterfactual_simulation_pipeline", "CounterfactualSimulationPipeline"),
    "SimulationOutcome": (".core.counterfactual_simulation_pipeline", "SimulationOutcome"),
    "get_counterfactual_pipeline": (".core.counterfactual_simulation_pipeline", "get_counterfactual_pipeline"),
    # Tools
    "microbrain_consult": (".tools.microbrain", "microbrain_consult"),
    "microbrain_status": (".tools.microbrain", "microbrain_status"),
    "consult_domain_hint": (".tools.microbrain", "consult_domain_hint"),
    "run_microbrain_lab": (".tools.microbrain_lab", "main"),
    "run_neuro_incubator": (".tools.neuro_brain_incubator", "run"),
    "run_neuro_route": (".tools.neuro_route", "main"),
    "run_wsp_surgeon": (".tools.wsp_tensor_surgeon", "run"),
    "ConformalCalibrationReport": (".tools.conformal_collector", "ConformalCalibrationReport"),
    "run_conformal_calibration_pipeline": (".tools.conformal_collector", "run_conformal_calibration_pipeline"),
    # Robotics extension layer
    "BarrierConstraint": (".core.state_cbf_filter", "BarrierConstraint"),
    "HalfPlaneBarrier": (".core.state_cbf_filter", "HalfPlaneBarrier"),
    "SphereBarrier": (".core.state_cbf_filter", "SphereBarrier"),
    "ControlAffineDynamics": (".core.state_cbf_filter", "ControlAffineDynamics"),
    "LinearDynamics": (".core.state_cbf_filter", "LinearDynamics"),
    "StateCBFSafetyFilter": (".core.state_cbf_filter", "StateCBFSafetyFilter"),
    "SensorimotorLoop": (".core.sensorimotor_loop", "SensorimotorLoop"),
    "ReservoirReflexLayer": (".core.sensorimotor_loop", "ReservoirReflexLayer"),
    "export_reservoir_nir": (".core.nir_export", "export_reservoir_nir"),
    "validate_nir_dialect_graph": (".core.nir_export", "validate_nir_dialect_graph"),
    "GenerativeModel": (".core.pymdp_efe", "GenerativeModel"),
    "EFPAgent": (".core.pymdp_efe", "EFPAgent"),
    "PackedTernaryLinear": (".core.ternary_edge_runtime", "PackedTernaryLinear"),
    "bench_receipt": (".core.ternary_edge_runtime", "bench_receipt"),
    "export_packed_linear_onnx": (".core.ternary_edge_runtime", "export_packed_linear_onnx"),
    "SkillLibrary": (".core.vla_gating", "SkillLibrary"),
    "ConformalActionGate": (".core.vla_gating", "ConformalActionGate"),
    "VLActionGate": (".core.vla_gating", "VLActionGate"),
}

__all__ = ["__version__", *list(_LAZY_EXPORTS.keys())]


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = spec
    mod = importlib.import_module(mod_name, __name__)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
