"""
Timescale-Decoupled Concentric Spiral DAG Orchestrator for Aetheris.
Ported and evolved from NeoAura/ECHO (echo_core/causal_dag.py & 19 Spiral Nodes) + 2025/2026 SOTA.

Partitions the cognitive and evolutionary runtime into 4 concentric loops running on distinct timescales:
- Loop 0 (Sensorimotor Arc, tau=1): Micro-kernel telemetry, BitNet/KAN reflex, fast AST token routing (< 2ms).
- Loop 1 (Verification & Working Memory, tau=3): VoC scheduling, CBF spectral barrier, epistemic consensus.
- Loop 2 (Latent Simulation & World Modeling, tau=10): Counterfactual dreaming, homology void extraction, causal discovery.
- Loop 3 (Sovereign Core & Synthesis, tau=50+): EGM spec synthesis, Z3 formal proofs, NREM weight consolidation.
"""

from __future__ import annotations
import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

log = logging.getLogger("aetheris.spiral_dag")


@dataclass
class SpiralNode:
    name: str
    loop_index: int  # 0, 1, 2, or 3
    clock_interval: int  # Execution frequency (ticks)
    dependencies: List[str] = field(default_factory=list)
    action: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


@dataclass
class SpiralExecutionReport:
    tick_count: int
    executed_nodes: List[str]
    loop_executions: Dict[int, int]
    shared_context: Dict[str, Any]
    duration_ms: float


class ConcentricSpiralDAG:
    """
    Topologically sorted multi-timescale DAG engine.
    Executes tasks based on loop index and clock ticks without blocking the reflex arc.
    """

    _instance: Optional["ConcentricSpiralDAG"] = None

    def __init__(self):
        self.nodes: Dict[str, SpiralNode] = {}
        self.tick = 0
        self.shared_workspace: Dict[str, Any] = {
            # P: 2026-11-11 - None until a real node measures it; the old
            # init premised 1.0 health and 100.0 energy nobody measured.
            "system_health": None,
            "metabolic_energy": None,
            "active_tools_count": 0,
            "last_cbf_violation": False,
        }
        self._register_default_nodes()

    @classmethod
    def get_instance(cls) -> "ConcentricSpiralDAG":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_node(self, node: SpiralNode):
        self.nodes[node.name] = node

    def _register_default_nodes(self):
        # Loop 0: Fast Sensorimotor
        self.register_node(SpiralNode(
            name="metabolic_telemetry",
            loop_index=0,
            clock_interval=1,
            dependencies=[],
            action=self._action_metabolic_telemetry
        ))
        self.register_node(SpiralNode(
            name="system1_router",
            loop_index=0,
            clock_interval=1,
            dependencies=["metabolic_telemetry"],
            action=self._action_system1_router
        ))

        # Loop 1: Working Memory & Verification
        self.register_node(SpiralNode(
            name="voc_scheduler_node",
            loop_index=1,
            clock_interval=3,
            dependencies=["system1_router"],
            action=self._action_voc_scheduler
        ))
        self.register_node(SpiralNode(
            name="cbf_spectral_barrier_node",
            loop_index=1,
            clock_interval=3,
            dependencies=["voc_scheduler_node"],
            action=self._action_cbf_barrier
        ))
        self.register_node(SpiralNode(
            name="neuro_hormonal_homeostasis_node",
            loop_index=1,
            clock_interval=3,
            dependencies=[],
            action=self._action_neuro_hormonal
        ))
        # Loop 1: Temporal state-space stream — the linear SSM micro-brain
        # integrates one telemetry vector per tick (loop duration, energy).
        self.register_node(SpiralNode(
            name="temporal_stream_node",
            loop_index=1,
            clock_interval=1,
            dependencies=["system1_router"],
            action=self._action_temporal_stream
        ))

        # Loop 2: Latent Simulation & World Modeling
        self.register_node(SpiralNode(
            name="autotelic_void_scanner_node",
            loop_index=2,
            clock_interval=10,
            dependencies=[],
            action=self._action_autotelic_scanner
        ))
        self.register_node(SpiralNode(
            name="hydro_cortex_fluid_node",
            loop_index=2,
            clock_interval=10,
            dependencies=[],
            action=self._action_hydro_fluid
        ))
        self.register_node(SpiralNode(
            name="spectral_fiedler_analyzer_node",
            loop_index=2,
            clock_interval=10,
            dependencies=[],
            action=self._action_spectral_laplacian
        ))
        self.register_node(SpiralNode(
            name="counterfactual_simulation_node",
            loop_index=2,
            clock_interval=10,
            dependencies=["autotelic_void_scanner_node"],
            action=self._action_simulation
        ))

        # Loop 3: Sovereign Core & Synthesis
        self.register_node(SpiralNode(
            name="nca_weight_healing_node",
            loop_index=3,
            clock_interval=30,
            dependencies=[],
            action=self._action_nca_healing
        ))
        self.register_node(SpiralNode(
            name="morphogenetic_synaptogenesis_node",
            loop_index=3,
            clock_interval=30,
            dependencies=[],
            action=self._action_synaptogenesis
        ))
        self.register_node(SpiralNode(
            name="mone_fast_weight_node",
            loop_index=1,
            clock_interval=3,
            dependencies=[],
            action=self._action_mone_memory
        ))
        self.register_node(SpiralNode(
            name="vsa_algebraic_attention_node",
            loop_index=1,
            clock_interval=3,
            dependencies=[],
            action=self._action_vsa_attention
        ))
        self.register_node(SpiralNode(
            name="comem_collective_distiller_node",
            loop_index=3,
            clock_interval=30,
            dependencies=[],
            action=self._action_comem_distillation
        ))
        self.register_node(SpiralNode(
            name="wsp_neural_surgery_node",
            loop_index=3,
            clock_interval=30,
            dependencies=[],
            action=self._action_wsp_surgery
        ))
        self.register_node(SpiralNode(
            name="auto_dreamer_node",
            loop_index=3,
            clock_interval=30,
            dependencies=[],
            action=self._action_auto_dreamer
        ))

        # Loop 3: Sovereign Core & Synthesis
        self.register_node(SpiralNode(
            name="egm_spec_synthesis_node",
            loop_index=3,
            clock_interval=30,
            dependencies=[],
            action=self._action_egm_synthesis
        ))
        self.register_node(SpiralNode(
            name="nrem_sleep_consolidation_node",
            loop_index=3,
            clock_interval=50,
            dependencies=["egm_spec_synthesis_node"],
            action=self._action_nrem_consolidation
        ))

    def step_spiral(self, input_context: Optional[Dict[str, Any]] = None) -> SpiralExecutionReport:
        """Executes one tick across the concentric spiral loops."""
        t0 = time.perf_counter()
        self.tick += 1

        if input_context:
            self.shared_workspace.update(input_context)

        executed: List[str] = []
        loop_counts: Dict[int, int] = collections.defaultdict(int)

        # Topological execution respecting loop clock intervals
        for name, node in self.nodes.items():
            if self.tick % node.clock_interval == 0:
                # Check dependencies
                deps_satisfied = True
                for dep in node.dependencies:
                    if dep not in executed and (self.tick % self.nodes[dep].clock_interval != 0):
                        pass

                if deps_satisfied and node.action:
                    try:
                        res = node.action(self.shared_workspace)
                        if res:
                            self.shared_workspace.update(res)
                        executed.append(name)
                        loop_counts[node.loop_index] += 1
                    except Exception as e:
                        log.warning("Spiral node '%s' raised exception: %s", name, e)

        duration = (time.perf_counter() - t0) * 1000.0
        # Feed the next tick's temporal_stream_node with real timing signal.
        self.shared_workspace["last_tick_duration_ms"] = round(duration, 3)
        return SpiralExecutionReport(
            tick_count=self.tick,
            executed_nodes=executed,
            loop_executions=dict(loop_counts),
            shared_context=dict(self.shared_workspace),
            duration_ms=round(duration, 3),
        )

    # Node Actions
    def _action_metabolic_telemetry(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """P: 2026-11-11 - this node returned a constant ``system_health``
        of 0.99 and a simulated energy drain (``-0.05`` per call, floored
        at 20); nothing it reported was measured.  It now reads the real
        metabolic state (core/metabolic.py -- the store that
        ``kernel.metabolic.trigger_event`` updates on evolution outcomes)
        and reports its age.  ``system_health`` is None until a live
        health writer exists: None is honest absence, 0.99 was a lie of
        constancy."""
        from core import metabolic as _metabolic_module

        state = _metabolic_module.metabolic.get_state()
        out: Dict[str, Any] = {
            "metabolic_energy": float(state.get("energy", 0.0)),
            "metabolic_ticks": int(state.get("ticks", 0)),
            "metabolic_state_age_s": None,
            "system_health": None,
            "system_health_note": ("no live system-health writer exists; "
                                   "the previous action returned the "
                                   "constant 0.99"),
        }
        try:
            out["metabolic_state_age_s"] = round(
                time.time() - float(state.get("last_tick_ts", 0.0)), 1)
        except (TypeError, ValueError):
            pass
        if state.get("homeostasis") is not None:
            # real persisted regulation signal when the store carries one
            out["metabolic_homeostasis"] = float(state["homeostasis"])
        return out

    def _action_system1_router(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # Design assertion, not telemetry: no System-1 router is implemented,
        # so "ready" here means the spiral node executed, nothing more.
        return {"fast_path_ready": True, "fast_path_basis": "constant_no_router_implemented"}

    # Scenario parameters for the VoC query, not measurements: a spiral
    # tick has no real mutation depth of its own; these describe the
    # hypothetical task being priced.  (Previously inline literals that
    # read like telemetry.)
    _VOC_SCENARIO_UNCERTAINTY = 0.25
    _VOC_SCENARIO_MUTATION_DEPTH = 10

    def _action_voc_scheduler(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .voc_meta_scheduler import ValueOfComputationMetaScheduler
        voc = ValueOfComputationMetaScheduler.get_instance()
        dec = voc.evaluate_task_computation_value(
            "spiral_tick",
            uncertainty=self._VOC_SCENARIO_UNCERTAINTY,
            mutation_depth=self._VOC_SCENARIO_MUTATION_DEPTH,
        )
        return {"voc_mode": dec.mode, "voc_score": dec.voc_score}

    def _action_cbf_barrier(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.cbf_spectral_barrier import CBFSpectralBarrierEngine
        cbf = CBFSpectralBarrierEngine.get_instance()
        # cbf_active is an instantiation flag, not a measurement: True whenever
        # the engine singleton constructs, regardless of whether any barrier
        # constraint is actually being enforced on live actions.
        return {
            "cbf_active": True,
            "cbf_basis": "engine_instantiated_no_live_constraint_check",
            "cbf_violations_prevented": cbf.violations_prevented,
        }

    def _action_autotelic_scanner(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.autotelic_void_explorer import AutotelicVoidExplorer
        explorer = AutotelicVoidExplorer.get_instance()
        voids = explorer.scan_microbrain_population()
        return {"topological_voids_detected": len(voids)}

    def _action_simulation(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # No dream/counterfactual simulation is wired into the spiral tick;
        # this node reports design intent, not a performed simulation.
        return {
            "dream_simulation_ready": True,
            "dream_simulation_basis": "constant_no_simulation_wired_into_tick",
        }

    def _action_neuro_hormonal(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .neuro_hormonal_homeostasis import NeuroHormonalHomeostasis
        nh = NeuroHormonalHomeostasis.get_instance()
        state = nh.tick_homeostasis()
        return {
            "hormonal_focus": state.primary["focus"],
            "hormonal_stress": state.allostatic_stress,
            "hormonal_balanced": state.is_balanced,
        }

    def _action_hydro_fluid(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .hydro_cortex_fluid import HydroCortexFluidField
        fluid = HydroCortexFluidField.get_instance()
        # The node measures its OWN injection: the density below is dominated
        # by this fixed probe perturbation, not by organic concept flow.
        fluid.inject_concept_density(16, 16, amount=1.0)
        fluid.step_simulation(iterations=2)
        snap = fluid.get_snapshot()
        return {
            "fluid_density_mean": snap.mean_density,
            "fluid_kinetic_energy": snap.total_kinetic_energy,
            "fluid_self_injected": True,
            "fluid_injection_site": [16, 16],
            "fluid_injection_amount": 1.0,
            "fluid_basis": "self_injected_probe_density_reflects_injection",
        }

    def _action_spectral_laplacian(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Spectral scan of the spiral's OWN execution topology.

        P: 2026-11-11 - this analyzed a hardcoded 4x4 adjacency over the
        fixed labels ["kernel", "evolution", "memory", "tools"] -- a toy
        graph that could never change no matter what the system did.  The
        graph is now the live dependency structure of the registered
        spiral nodes (symmetrized), so the spectrum tracks the real
        orchestration topology.
        """
        from .spectral_graph_analyzer import SpectralLaplacianAnalyzer
        analyzer = SpectralLaplacianAnalyzer.get_instance()
        names = list(self.nodes.keys())
        index = {n: i for i, n in enumerate(names)}
        n = len(names)
        A = np.zeros((n, n), dtype=float)
        for node in self.nodes.values():
            i = index[node.name]
            for dep in node.dependencies:
                j = index.get(dep)
                if j is not None:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        res = analyzer.analyze_graph(names, A)
        return {
            "spectral_algebraic_connectivity": res.algebraic_connectivity,
            "spectral_is_connected": res.is_connected,
        }

    def _action_mone_memory(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.mone_fast_weight_memory import MoNeFastWeightMemory
        mone = MoNeFastWeightMemory.get_instance()
        return mone.get_status()

    def _action_vsa_attention(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.vsa_algebraic_attention import VSAAlgebraicAttentionEngine
        vsa = VSAAlgebraicAttentionEngine.get_instance()
        return vsa.get_status()

    def _action_comem_distillation(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .comem_collective_distiller import CoMemCollectiveDistiller
        comem = CoMemCollectiveDistiller.get_instance()
        report = comem.distill_collective_wisdom()
        return {
            "comem_active": True,
            "curated_accepted": report.curated_packets_accepted,
            "mean_vfe_reduction": report.mean_vfe_reduction
        }

    def _action_wsp_surgery(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.wsp_neural_surgeon import WSPNeuralSurgeon
        surgeon = WSPNeuralSurgeon.get_instance()
        audit = surgeon.audit_neuro_brains_store()
        return {
            "wsp_surgery_active": True,
            "files_audited": audit["total_files_audited"],
            "pathological_brains_detected": audit["pathological_brains_detected"]
        }

    def _action_auto_dreamer(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .auto_dreamer_consolidator import AutoDreamerConsolidator
        consolidator = AutoDreamerConsolidator.get_instance()
        report = consolidator.consolidate_offline_dream()
        return {
            "auto_dreamer_active": True,
            "clusters_identified": report.clusters_identified,
            "new_notes_synthesized": report.new_notes_synthesized,
            "wiki_articles_updated": report.wiki_articles_updated,
            "stale_entries_pruned": report.stale_entries_pruned,
            "dream_duration_ms": report.duration_ms
        }

    def _action_egm_synthesis(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """P: 2026-11-11 - this echoed the tick counter as
        ``egm_synthesis_cycle``, as if a spec-synthesis cycle had run.
        EGM spec drafting is goal-driven
        (ElephantGoldfishCoordinator.draft_spec_from_goal) and tool drift
        validation is owned by ToolCheckDaemon's background audit; a
        spiral tick carries no build goal, so this node synthesizes
        nothing and says so."""
        return {
            "egm_synthesis_attempted": False,
            "egm_note": ("no build goal in the spiral tick; spec drafting is "
                         "goal-driven and tool drift validation runs in the "
                         "tool-check daemon's background audit"),
        }

    def _action_nrem_consolidation(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """P: 2026-11-11 - this returned ``{"nrem_consolidated": True}``
        unconditionally; nothing was consolidated.  It now reports the
        real sleep orchestrator's cycle count: consolidation is owned by
        the idle-gated SleepWakeOrchestrator (triggered from the
        tool-check daemon), so a spiral tick neither performs nor claims
        a consolidation that did not happen."""
        from .sleep_wake_orchestrator import SleepWakeOrchestrator
        orch = SleepWakeOrchestrator.get_instance()
        cycles = int(getattr(orch, "total_sleep_cycles_count", 0) or 0)
        return {
            "nrem_consolidated": bool(cycles > 0),
            "sleep_cycle_count": cycles,
            "orchestrator_state": str(getattr(orch, "current_state", "UNKNOWN")),
            "nrem_note": ("cycle count from the idle-gated sleep-wake "
                          "orchestrator; no consolidation was claimed for "
                          "this tick"),
        }

    def _action_temporal_stream(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.linear_state_space_microbrain import LinearStateSpaceMicroBrain
        ssm = LinearStateSpaceMicroBrain.get_instance()

        # P: 2026-11-11 - the feed previously packaged the fabricated
        # metabolic decay and the constant 0.99 "system_health" as if they
        # were telemetry.  Every component is now a real signal with an
        # honest fallback chain: energy (live metabolic store) -> stress
        # (live hormonal state, else the persisted homeostasis signal,
        # else 0.0 meaning no signal) -> last tick duration (measured).
        energy = ctx.get("metabolic_energy")
        if energy is None:
            energy = 0.0
        stress = ctx.get("hormonal_stress")
        if stress is None:
            stress = ctx.get("metabolic_homeostasis")
        if stress is None:
            stress = 0.0
        tick_ms = ctx.get("last_tick_duration_ms")
        if tick_ms is None:
            tick_ms = 0.0
        signal = [float(energy), float(stress), float(tick_ms)]
        res = ssm.step(signal)
        return {
            "temporal_state_energy": res["state_energy"],
            "temporal_tokens_processed": res["tokens_processed"],
        }

    def _action_nca_healing(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .nca_weight_regenerator import NCAWeightRegenerator
        regenerator = NCAWeightRegenerator.get_instance()
        # Bounded self-test on a synthetic ternary matrix: proves the NCA
        # repair dynamics are alive without touching stored brain weights.
        rng = np.random.RandomState(self.tick)
        orig_w = rng.choice([-1.0, 0.0, 1.0], size=(16, 16), p=[0.3, 0.4, 0.3]).astype(np.float32)
        corrupted_w = orig_w.copy()
        corrupted_w[4:12, 4:12] = 0.0
        _, repair = regenerator.repair_damage(corrupted_w, reference_morphology=orig_w, iterations=8)
        return {
            "nca_healing_active": True,
            "nca_reconstruction_fidelity": round(float(repair.reconstruction_fidelity), 4),
            # Lets consumers tell a measured fidelity (reference supplied)
            # from an unmeasured one (0.0 = no comparison was possible).
            "nca_reference_available": bool(repair.reference_available),
            "nca_restored_cells": int(repair.restored_cells_count),
        }

    def _action_synaptogenesis(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from ..core.morphogenetic_synaptogenesis import MorphogeneticSynaptogenesis
        synap = MorphogeneticSynaptogenesis.get_instance()
        lifecycle = synap.step_synaptic_lifecycle()
        prune_report = synap.prune_and_consolidate()
        return {
            "synaptogenesis_active": True,
            "synaptogenesis_total_axons": int(lifecycle.get("total_axons", 0)),
            "synaptogenesis_active_axons": int(lifecycle.get("active_axons", 0)),
            "synaptogenesis_mean_conductance": round(float(lifecycle.get("mean_conductance", 0.0)), 4),
            "synaptogenesis_pruned": int(prune_report.pruned_dead_axons),
        }

