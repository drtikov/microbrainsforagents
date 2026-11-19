"""
Aetheris Sleep-Wake Orchestrator (NREM & REM Biphasic Scheduler).
Inspired by Karpathy autoresearch sleep patterns and biological NREM/REM sleep cycles (2025-2026).
Coordinates autonomous transitions between:
- WAKE Phase: active tool deployment, test execution, sensory intake.
- NREM Phase: SVD synaptic downscaling, obsolete memory pruning, Hopfield trace consolidation.
- REM Phase: counterfactual simulation dreaming via CounterfactualDreamEngine.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import numpy as np

from .sleep_dreaming_engine import AutonomousSleepDreamingEngine
from .counterfactual_dream_engine import CounterfactualDreamEngine
from .astrocyte_homeostatic_field import AstrocyteHomeostaticField

class SleepWakeOrchestrator:
    """Manages biphasic cognitive states (WAKE vs SLEEP) for Aetheris micro-brain swarms."""

    _instance: Optional[SleepWakeOrchestrator] = None

    def __init__(self, idle_sleep_threshold_s: float = 120.0):
        self.idle_sleep_threshold_s = idle_sleep_threshold_s
        self.current_state: str = "WAKE"  # WAKE, NREM_SLEEP, REM_DREAMING
        self.last_activity_ts: float = time.time()
        self.total_sleep_cycles_count: int = 0

        self.nrem_engine = AutonomousSleepDreamingEngine.get_instance()
        self.rem_engine = CounterfactualDreamEngine.get_instance()
        self.astrocyte = AstrocyteHomeostaticField.get_instance()

    @classmethod
    def get_instance(cls) -> SleepWakeOrchestrator:
        if cls._instance is None:
            cls._instance = SleepWakeOrchestrator()
        return cls._instance

    def notify_agent_activity(self, activity_tag: str = "wake_step") -> Dict[str, Any]:
        """Notifies the orchestrator that the agent is actively working in WAKE phase."""
        self.last_activity_ts = time.time()
        if self.current_state != "WAKE":
            self.current_state = "WAKE"
            # Signal astrocyte equilibrium on wakeup
            self.astrocyte.register_neural_event(is_failure=False, execution_latency_s=0.01)

        return {
            "status": "wake_active",
            "state": self.current_state,
            "activity_tag": activity_tag,
            "timestamp": self.last_activity_ts,
        }

    def evaluate_and_transition_sleep_cycle(self, force_sleep: bool = False) -> Dict[str, Any]:
        """
        Evaluates idle duration. If idle threshold is breached, executes full biphasic sleep:
        1. NREM Sleep: SVD trace smoothing and synaptic downscaling.
        2. REM Sleep: counterfactual dreaming and stress test hardening.
        3. Wakeup: Restores nominal execution state with refreshed cognitive capacity.
        """
        now = time.time()
        idle_duration = now - self.last_activity_ts

        if not force_sleep and idle_duration < self.idle_sleep_threshold_s:
            return {
                "status": "wake_sustained",
                "current_state": self.current_state,
                "idle_duration_s": round(idle_duration, 1),
                "threshold_s": self.idle_sleep_threshold_s,
            }

        # Enter NREM Sleep
        self.current_state = "NREM_SLEEP"
        nrem_report = self.nrem_engine.execute_sleep_consolidation_cycle()

        # Transition to REM Dreaming
        self.current_state = "REM_DREAMING"
        rem_report = self.rem_engine.run_dream_session(max_scenarios=5)

        # Complete sleep and wake up refreshed
        self.total_sleep_cycles_count += 1
        self.current_state = "WAKE"
        self.last_activity_ts = time.time()

        return {
            "status": "sleep_cycle_completed",
            "cycle_number": self.total_sleep_cycles_count,
            "nrem_report": {
                "smoothed_brains": nrem_report.get("smoothed_brains_count", 0),
                "replayed_lessons": nrem_report.get("replayed_lessons", 0),
            },
            "rem_report": {
                "scenarios_evaluated": rem_report.get("scenarios_evaluated", 0),
                "stabilized_brains_count": len(rem_report.get("stabilized_brains", [])),
                # P: 2026-11-10 - dream_entropy_reduced was fabricated
                # (0.15 * n); the engine now reports real measurements.
                "brains_skipped_unloadable": len(rem_report.get("brains_skipped_unloadable", [])),
                "mean_confidence_delta": rem_report.get("mean_confidence_delta", 0.0),
            },
            "post_sleep_state": "WAKE_REFRESHED",
        }

