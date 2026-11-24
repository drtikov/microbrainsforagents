"""Core vs. Lab Package Boundary Manifest & Isolation Verifier (stdlib-only).

Separates the zero-NumPy production security core (`aetheris-core` surface)
from experimental neuromorphic / robotics / tensor lab modules (`aetheris-lab`)
so security engineers can audit the minimal production surface cleanly.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

CORE_SECURITY_MODULES: Tuple[str, ...] = (
    "__init__",
    "hook",
    "hook_daemon",
    "audit",
    "doctor",
    "verify",
    "mcp_proxy",
    "pr_gate",
    "cli",
    "install_antigravity",
    "install_dsh",
    "install_openclaw",
    "core/audit_ui",
    "core/context_hygiene",
    "core/deferred_exec_guard",
    "core/directive_vault",
    "core/economics_ledger",
    "core/envelope_parser",
    "core/layered_decision_engine",
    "core/lifecycle_engine",
    "core/mcp_security",
    "core/mcp_tiers",
    "core/multilang_ast",
    "core/os_sandbox",
    "core/otel_exporter",
    "core/packaging_manifest",
    "core/policy_as_code",
    "core/policy_governance",
    "core/repo_context",
    "core/runtime_paths",
    "core/secret_broker",
    "core/secrets",
    "core/session_rollup",
    "core/tier_ladder",
    "core/trajectory_engine",
)

LAB_RESEARCH_MODULES: Tuple[str, ...] = (
    "core/active_inference",
    "core/analytical_micro_transformer",
    "core/autotelic_void_explorer",
    "core/bitnet_ternary_engine",
    "core/brain_cellular_fission",
    "core/brain_merger",
    "core/brain_moe_orchestrator",
    "core/brain_nursery",
    "core/brain_plasticity_updater",
    "core/brain_speculative_drafter",
    "core/cascade_ensemble",
    "core/causal_world_model",
    "core/cbf_spectral_barrier",
    "core/continual_learning_pipeline",
    "core/counterfactual_reasoner",
    "core/counterfactual_simulation_pipeline",
    "core/critical_ensemble_abstention",
    "core/dendritic_tree_microbrain",
    "core/evolution_surrogate_prefilter",
    "core/hardware_router",
    "core/hierarchical_moe_cortex",
    "core/hyperdimensional_vsa_engine",
    "core/intent_persistence_tracker",
    "core/kolmogorov_arnold_microbrain",
    "core/linear_state_space_microbrain",
    "core/liquid_state_reservoir",
    "core/metacognitive_calibrator",
    "core/microbrain_hypernetwork",
    "core/mixture_of_depths_router",
    "core/mone_fast_weight_memory",
    "core/morphogenetic_synaptogenesis",
    "core/nir_export",
    "core/online_hebbian_adaptor",
    "core/opencl_microbrain_accelerator",
    "core/plasticity_cycle",
    "core/predictive_coding_engine",
    "core/pymdp_efe",
    "core/sensorimotor_loop",
    "core/skill_lifecycle_manager",
    "core/state_cbf_filter",
    "core/ternary_edge_runtime",
    "core/tool_routing_moe",
    "core/topological_persistence_merger",
    "core/vla_gating",
    "core/vsa_algebraic_attention",
    "core/weight_space_memory_synthesis",
    "core/weight_space_vsa",
    "core/wsp_neural_surgeon",
    "core/zero_shot_tool_chainer",
)


def verify_core_isolation(pkg_root: Path | None = None) -> Dict[str, Any]:
    """Verify every `CORE_SECURITY_MODULES` file exists and has zero NumPy imports."""
    root = Path(pkg_root) if pkg_root is not None else Path(__file__).resolve().parents[1]
    missing: List[str] = []
    numpy_violations: List[str] = []

    for mod in CORE_SECURITY_MODULES:
        fpath = root / f"{mod}.py"
        if not fpath.is_file():
            missing.append(mod)
            continue
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8", errors="ignore"), filename=str(fpath))
        except SyntaxError as exc:
            numpy_violations.append(f"{mod}:SyntaxError:{exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0].lower() == "numpy":
                        numpy_violations.append(mod)
                        break
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0].lower() == "numpy":
                    numpy_violations.append(mod)
                    break

    return {
        "ok": (len(missing) == 0 and len(numpy_violations) == 0),
        "package_root": str(root),
        "core_count": len(CORE_SECURITY_MODULES),
        "lab_count": len(LAB_RESEARCH_MODULES),
        "missing_modules": sorted(set(missing)),
        "numpy_violations": sorted(set(numpy_violations)),
    }
