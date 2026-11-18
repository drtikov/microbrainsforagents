"""
Aetheris Microbrains Standalone SDK
===================================
Self-contained, portable suite of lightweight pure-NumPy analytical microbrains,
layered structural AST/shell safety parsers, split-conformal abstention gates,
OpenCL acceleration, Weight-Space Programming (WSP) engine, Vector Symbolic
Architecture (VSA), deterministic Mixture of Experts (MoE), and cascade gates.

Clean Python package: does NOT mutate `sys.path` or shadow host modules.
"""

from __future__ import annotations

from .core.layered_decision_engine import (
    PolicyConfig,
    analyze_code_diff,
    analyze_shell_command,
    decide,
    evaluate_layer0_policy,
    get_recent_decision_journal,
    is_protected_path,
    recalibrate_from_decision_journal,
    record_decision_journal,
)
from .core.opencl_microbrain_accelerator import (
    OpenCLMicroBrainAccelerator,
    get_opencl_accelerator,
)
from .core.dendritic_tree_microbrain import DendriticTreeMicroBrain
from .core.kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain
from .core.linear_state_space_microbrain import LinearStateSpaceMicroBrain
from .core.microbrain_hypernetwork import MicroBrainHypernetwork
from .core.analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR
from .core.brain_nursery import (
    AutonomousBrainNursery,
    BrainNursery,
    get_brain_nursery,
)
from .core.bitnet_ternary_engine import (
    BitLinearLayer,
    BitNetQuantizer,
    BitNetTernaryEngine,
)
from .core.liquid_state_reservoir import LiquidStateReservoir
from .core.predictive_coding_engine import (
    HierarchicalPredictiveCodingEngine,
    PredictiveCodingLayer,
    PredictiveCodingEngine,
)
from .core.hyperdimensional_vsa_engine import HyperdimensionalVSAEngine
from .core.wsp_neural_surgeon import WSPNeuralSurgeon

# Meta-cognitive components (ported & adapted from Ouroboros)
from .core.causal_world_model import (
    StructuralCausalModel,
    CausalWorldModel,
    CausalVariable,
    CausalIntervention,
    CausalSimulationResult,
    get_causal_world_model,
)
from .core.counterfactual_reasoner import (
    CounterfactualResult,
    simulate_counterfactual,
)
from .core.intent_persistence_tracker import (
    IntentPersistenceTracker,
    IntentDriftAssessment,
)
from .core.metacognitive_calibrator import (
    MetacognitiveActionVerdict,
    CalibratedConfidence,
    calibrate_action_confidence,
)
from .core.skill_lifecycle_manager import (
    SkillLifecycleState,
    SkillTelemetry,
    FleetHygieneReport,
    SkillLifecycleManager,
    get_skill_lifecycle_manager,
)

# SOTA Ouroboros Neural Extensions
from .core.weight_space_vsa import (
    WeightSpaceVSA,
    VSAResult,
    get_weight_space_vsa,
)
from .core.tool_routing_moe import (
    ToolRoutingMoE,
    MoERoutingDecision,
    get_tool_routing_moe,
    narrow_tool_surface,
)
from .core.critical_ensemble_abstention import (
    CriticalEnsemble,
    CriticalConsensusResult,
    EnsembleVote,
    get_critical_ensemble,
)
from .core.cascade_ensemble import (
    CascadeEnsembleRouter,
    CascadeTierResult,
    CascadeEnsembleVerdict,
    get_cascade_ensemble_router,
)
from .core.continual_learning_pipeline import (
    ContinualLearningPipeline,
    ContinualUpdateResult,
)
from .core.plasticity_cycle import (
    PlasticityCycleScheduler,
    PlasticityCycleReport,
    get_plasticity_scheduler,
)
from .core.microbrain_telemetry import (
    MicrobrainTelemetry,
    TelemetryEvent,
    DashboardReport,
    get_microbrain_telemetry,
)
from .core.counterfactual_simulation_pipeline import (
    CounterfactualSimulationPipeline,
    SimulationOutcome,
    get_counterfactual_pipeline,
)

from .tools.microbrain import (
    microbrain_consult,
    microbrain_status,
    consult_domain_hint,
)
from .tools.microbrain_lab import main as run_microbrain_lab
from .tools.neuro_brain_incubator import run as run_neuro_incubator
from .tools.neuro_route import main as run_neuro_route
from .tools.wsp_tensor_surgeon import run as run_wsp_surgeon
from .tools.conformal_collector import (
    ConformalCalibrationReport,
    run_conformal_calibration_pipeline,
)

__all__ = [
    "decide",
    "PolicyConfig",
    "analyze_code_diff",
    "analyze_shell_command",
    "evaluate_layer0_policy",
    "get_recent_decision_journal",
    "is_protected_path",
    "recalibrate_from_decision_journal",
    "record_decision_journal",
    "narrow_tool_surface",
    "OpenCLMicroBrainAccelerator",
    "get_opencl_accelerator",
    "DendriticTreeMicroBrain",
    "KolmogorovArnoldMicroBrain",
    "LinearStateSpaceMicroBrain",
    "MicroBrainHypernetwork",
    "AnalyticalMicroTransformer",
    "STORE_DIR",
    "AutonomousBrainNursery",
    "BrainNursery",
    "get_brain_nursery",
    "BitLinearLayer",
    "BitNetQuantizer",
    "BitNetTernaryEngine",
    "LiquidStateReservoir",
    "HierarchicalPredictiveCodingEngine",
    "PredictiveCodingLayer",
    "PredictiveCodingEngine",
    "HyperdimensionalVSAEngine",
    "WSPNeuralSurgeon",
    # Meta-cognitive
    "StructuralCausalModel",
    "CausalWorldModel",
    "CausalVariable",
    "CausalIntervention",
    "CausalSimulationResult",
    "get_causal_world_model",
    "CounterfactualResult",
    "simulate_counterfactual",
    "IntentPersistenceTracker",
    "IntentDriftAssessment",
    "MetacognitiveActionVerdict",
    "CalibratedConfidence",
    "calibrate_action_confidence",
    "SkillLifecycleState",
    "SkillTelemetry",
    "FleetHygieneReport",
    "SkillLifecycleManager",
    "get_skill_lifecycle_manager",
    # SOTA Ouroboros Neural Extensions
    "WeightSpaceVSA",
    "VSAResult",
    "get_weight_space_vsa",
    "ToolRoutingMoE",
    "MoERoutingDecision",
    "get_tool_routing_moe",
    "CriticalEnsemble",
    "CriticalConsensusResult",
    "EnsembleVote",
    "get_critical_ensemble",
    "CascadeEnsembleRouter",
    "CascadeTierResult",
    "CascadeEnsembleVerdict",
    "get_cascade_ensemble_router",
    "ContinualLearningPipeline",
    "ContinualUpdateResult",
    "PlasticityCycleScheduler",
    "PlasticityCycleReport",
    "get_plasticity_scheduler",
    "MicrobrainTelemetry",
    "TelemetryEvent",
    "DashboardReport",
    "get_microbrain_telemetry",
    "CounterfactualSimulationPipeline",
    "SimulationOutcome",
    "get_counterfactual_pipeline",
    # Tools
    "microbrain_consult",
    "microbrain_status",
    "consult_domain_hint",
    "run_microbrain_lab",
    "run_neuro_incubator",
    "run_neuro_route",
    "run_wsp_surgeon",
    "ConformalCalibrationReport",
    "run_conformal_calibration_pipeline",
]
