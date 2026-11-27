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
from importlib import metadata as _importlib_metadata
from typing import Any, Dict, Tuple

try:
    __version__ = _importlib_metadata.version("aetheris-microbrains")
except _importlib_metadata.PackageNotFoundError:
    __version__ = "0.7.6"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Hook
    "evaluate_hook_payload": (".hook", "evaluate_hook_payload"),
    "evaluate_post_tool_use_payload": (".hook", "evaluate_post_tool_use_payload"),
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
    "redact_secrets": (".core.layered_decision_engine", "redact_secrets"),
    "scrub_tool_output": (".core.secrets", "scrub_tool_output"),
    "scrub_secrets": (".core.secrets", "scrub_tool_output"),
    # Audit & inspection
    "get_journal_entries": (".audit", "get_journal_entries"),
    "format_entry": (".audit", "format_entry"),
    "build_friction_digest": (".audit", "build_friction_digest"),
    "FrictionDigestBuilder": (".audit", "FrictionDigestBuilder"),
    # Policy governance, rule vitality, & directive vault
    "RuleVitalityRecord": (".core.policy_governance", "RuleVitalityRecord"),
    "RuleVitalityLedger": (".core.policy_governance", "RuleVitalityLedger"),
    "GovernanceLedger": (".core.policy_governance", "GovernanceLedger"),
    "verify_policy_patch": (".core.policy_governance", "verify_policy_patch"),
    "synthesize_policy_candidates": (".core.policy_governance", "synthesize_policy_candidates"),
    "PendingRuleQueue": (".core.policy_governance", "PendingRuleQueue"),
    "evaluate_policy_candidate": (".core.policy_governance", "evaluate_policy_candidate"),
    "DirectiveCard": (".core.directive_vault", "DirectiveCard"),
    "DirectiveVault": (".core.directive_vault", "DirectiveVault"),
    # Context hygiene, economics, envelopes, chronicle, & tier ladder
    "HygieneConfig": (".core.context_hygiene", "HygieneConfig"),
    "HygieneVerdict": (".core.context_hygiene", "HygieneVerdict"),
    "FileSnapshotLedger": (".core.context_hygiene", "FileSnapshotLedger"),
    "SymbolMapExtractor": (".core.context_hygiene", "SymbolMapExtractor"),
    "ShellReadInspector": (".core.context_hygiene", "ShellReadInspector"),
    "ContextHygieneGate": (".core.context_hygiene", "ContextHygieneGate"),
    "TokenSavingsLedger": (".core.economics_ledger", "TokenSavingsLedger"),
    "estimate_tokens": (".core.economics_ledger", "estimate_tokens"),
    "record_savings_event": (".core.economics_ledger", "record_savings_event"),
    "summarize_savings": (".core.economics_ledger", "summarize_savings"),
    "format_statusline": (".core.economics_ledger", "format_statusline"),
    "HandshakeEnvelope": (".core.envelope_parser", "HandshakeEnvelope"),
    "parse_handback_envelope": (".core.envelope_parser", "parse_handback_envelope"),
    "validate_handback_envelope": (".core.envelope_parser", "validate_handback_envelope"),
    "ChronicleCompactor": (".core.session_rollup", "ChronicleCompactor"),
    "BootstrapAssembler": (".core.session_rollup", "BootstrapAssembler"),
    "write_chronicle_entry": (".core.session_rollup", "write_chronicle_entry"),
    "compact_past_days": (".core.session_rollup", "compact_past_days"),
    "assemble_bootstrap_context": (".core.session_rollup", "assemble_bootstrap_context"),
    "TierRoute": (".core.tier_ladder", "TierRoute"),
    "TierLadderResolver": (".core.tier_ladder", "TierLadderResolver"),
    "extract_depth_hint": (".core.tier_ladder", "extract_depth_hint"),
    "PolicySliceRenderer": (".doctor", "PolicySliceRenderer"),
    "slice_markdown_by_role": (".doctor", "slice_markdown_by_role"),
    "check_markdown_budget": (".doctor", "check_markdown_budget"),
    # MCP security (Wave 3)
    "detect_prompt_injection": (".core.mcp_security", "detect_prompt_injection"),
    "scan_mcp_tools": (".core.mcp_security", "scan_mcp_tools"),
    "MCPManifestPinner": (".core.mcp_security", "MCPManifestPinner"),
    "SessionSecurityState": (".core.mcp_security", "SessionSecurityState"),
    # MCP risk tiers & doctor
    "mcp_tool_tier": (".core.mcp_tiers", "mcp_tool_tier"),
    "save_mcp_pin": (".core.mcp_tiers", "save_mcp_pin"),
    "save_mcp_manifest_pin": (".core.mcp_tiers", "save_mcp_manifest_pin"),
    "verify_mcp_manifest": (".core.mcp_tiers", "verify_mcp_manifest"),
    "remove_mcp_pin": (".core.mcp_tiers", "remove_mcp_pin"),
    "run_doctor": (".doctor", "run_doctor"),
    "ConformalSemanticGuard": (".core.semantic_guard", "ConformalSemanticGuard"),
    "detect_semantic_injection": (".core.semantic_guard", "detect_semantic_injection"),
    "shannon_entropy": (".core.secrets", "shannon_entropy"),
    "detect_high_entropy_secrets": (".core.secrets", "detect_high_entropy_secrets"),
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
    "ToolSchemaIndex": (".core.tool_routing_moe", "ToolSchemaIndex"),
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
    # Proposals 1-4: Multi-language AST, Deferred Exec Guard, SecretBroker, OS Sandbox, Policy-as-Code
    "analyze_multilang_code": (".core.multilang_ast", "analyze_multilang_code"),
    "analyze_deferred_execution": (".core.deferred_exec_guard", "analyze_deferred_execution"),
    "SecretSpec": (".core.secret_broker", "SecretSpec"),
    "SecretBroker": (".core.secret_broker", "SecretBroker"),
    "DomainAllowlistProxy": (".core.os_sandbox", "DomainAllowlistProxy"),
    "build_os_sandbox_command": (".core.os_sandbox", "build_os_sandbox_command"),
    "find_aetheris_gate_binary": (".core.os_sandbox", "find_aetheris_gate_binary"),
    "CedarStatement": (".core.policy_as_code", "CedarStatement"),
    "parse_cedar_policies": (".core.policy_as_code", "parse_cedar_policies"),
    "evaluate_cedar_policies": (".core.policy_as_code", "evaluate_cedar_policies"),
    "cedar_to_policy_config": (".core.policy_as_code", "cedar_to_policy_config"),
    "export_policy_to_cedar": (".core.policy_as_code", "export_policy_to_cedar"),
    "parse_rego_policy": (".core.policy_as_code", "parse_rego_policy"),
    "evaluate_rego_policy": (".core.policy_as_code", "evaluate_rego_policy"),
    "rego_to_policy_config": (".core.policy_as_code", "rego_to_policy_config"),
    "export_policy_to_rego": (".core.policy_as_code", "export_policy_to_rego"),
    # Proposals 5-7: OTel GenAI, Audit Web UI, Packaging Manifest, PR Gate
    "journal_entry_to_otel_span": (".core.otel_exporter", "journal_entry_to_otel_span"),
    "build_otlp_trace_payload": (".core.otel_exporter", "build_otlp_trace_payload"),
    "export_otlp_file": (".core.otel_exporter", "export_otlp_file"),
    "push_otlp_http": (".core.otel_exporter", "push_otlp_http"),
    "AuditUIServer": (".core.audit_ui", "AuditUIServer"),
    "build_dashboard_snapshot": (".core.audit_ui", "build_dashboard_snapshot"),
    "CORE_SECURITY_MODULES": (".core.packaging_manifest", "CORE_SECURITY_MODULES"),
    "LAB_RESEARCH_MODULES": (".core.packaging_manifest", "LAB_RESEARCH_MODULES"),
    "verify_core_isolation": (".core.packaging_manifest", "verify_core_isolation"),
    "scan_pr_diff": (".pr_gate", "scan_pr_diff"),
    "scan_pr_paths": (".pr_gate", "scan_pr_paths"),
    "to_sarif_report": (".pr_gate", "to_sarif_report"),
    "format_markdown_report": (".pr_gate", "format_markdown_report"),
    # Next-generation platform modules (v0.8 proposals)
    "HostSpec": (".hosts.specs", "HostSpec"),
    "HOST_SPECS": (".hosts.specs", "HOST_SPECS"),
    "compile_hook_config_from_spec": (".hosts.specs", "compile_hook_config_from_spec"),
    "validate_host_specs_parity": (".hosts.specs", "validate_host_specs_parity"),
    "ASTDiffCache": (".core.ast_cache", "ASTDiffCache"),
    "get_default_ast_cache": (".core.ast_cache", "get_default_ast_cache"),
    "WorkspaceTransactionManager": (".core.workspace_tx", "WorkspaceTransactionManager"),
    "run_speculative_transaction": (".core.workspace_tx", "run_speculative_transaction"),
    "CapabilityCaveat": (".core.capability_tokens", "CapabilityCaveat"),
    "CapabilityToken": (".core.capability_tokens", "CapabilityToken"),
    "mint_root_capability": (".core.capability_tokens", "mint_root_capability"),
    "attenuate_capability": (".core.capability_tokens", "attenuate_capability"),
    "verify_capability_and_authorize": (".core.capability_tokens", "verify_capability_and_authorize"),
    "propagate_subagent_taint": (".core.capability_tokens", "propagate_subagent_taint"),
    "generate_repair_hint": (".core.repair_hints", "generate_repair_hint"),
    "calibrate_thresholds_from_journal": (".core.repair_hints", "calibrate_thresholds_from_journal"),
    "create_signed_fleet_bundle": (".core.fleet_bundle", "create_signed_fleet_bundle"),
    "verify_signed_fleet_bundle": (".core.fleet_bundle", "verify_signed_fleet_bundle"),
    "build_fleet_attestation": (".core.fleet_bundle", "build_fleet_attestation"),
    "run_cross_harness_ab_suite": (".core.benchmark_harness", "run_cross_harness_ab_suite"),
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
