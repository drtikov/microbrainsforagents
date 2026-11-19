"""
Value of Computation (VoC) Meta-Scheduler for Aetheris.
Ported and evolved from NeoAura/ECHO (echo_core/metacognitive_system2.py & Loop 1 VoC Node) + 2025/2026 SOTA.

Dynamically arbitrates cognitive mode between:
- System-1 (Reflex Arc): Fast O(1) BitNet micro-brain / cached heuristic (< 1ms).
- System-2 (Deliberation): Deep formal verification (Z3 SMT, EGM Goldfish Audit, AST Fuzzing).

Arbitration rule:
VoC = E[Delta_Utility] - lambda * Compute_Cost
If VoC <= threshold: System-1 fast exit.
If VoC > threshold: Elevate to System-2 Deliberation.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("aetheris.voc_scheduler")


@dataclass
class VoCDecision:
    mode: str  # "SYSTEM_1_REFLEX" or "SYSTEM_2_DELIBERATION"
    voc_score: float
    expected_utility_gain: float
    compute_cost: float
    risk_level: str
    reasons: List[str]


class ValueOfComputationMetaScheduler:
    """
    Resource-rational meta-scheduler balancing compute budget vs epistemic risk.
    """

    _instance: Optional["ValueOfComputationMetaScheduler"] = None

    def __init__(
        self,
        cost_lambda: float = 0.35,
        voc_threshold: float = 0.12,
        max_budget_ms: float = 500.0,
    ):
        self.cost_lambda = float(cost_lambda)
        self.voc_threshold = float(voc_threshold)
        self.max_budget_ms = float(max_budget_ms)
        self.history: List[VoCDecision] = []

    @classmethod
    def get_instance(cls) -> "ValueOfComputationMetaScheduler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def evaluate_task_computation_value(
        self,
        task_name: str,
        uncertainty: float,         # Epistemic uncertainty in [0.0, 1.0]
        mutation_depth: int,        # Lines of code or AST node changes
        is_security_sensitive: bool = False,
        current_load: float = 0.20,  # Host resource pressure [0.0, 1.0]
    ) -> VoCDecision:
        reasons: List[str] = []

        # 1. Expected utility gain from deliberation
        # Deep deliberation yields highest returns when uncertainty and complexity are high
        expected_gain = 0.60 * float(uncertainty) + 0.30 * min(1.0, mutation_depth / 80.0)

        if is_security_sensitive:
            expected_gain += 0.40
            reasons.append("Security-sensitive task escalates deliberation utility")

        # 2. Compute cost calculation
        # Deliberation cost scales with load and complexity
        raw_cost = 0.20 + 0.50 * current_load + 0.30 * min(1.0, mutation_depth / 100.0)
        weighted_cost = self.cost_lambda * raw_cost

        # 3. Value of Computation
        voc_score = round(expected_gain - weighted_cost, 3)

        # Risk classification
        if uncertainty > 0.65 or is_security_sensitive:
            risk = "HIGH"
        elif uncertainty > 0.35:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Hard override: Security-critical tasks always get System-2
        if is_security_sensitive or voc_score > self.voc_threshold:
            mode = "SYSTEM_2_DELIBERATION"
            reasons.append(f"VoC ({voc_score:.3f}) > threshold ({self.voc_threshold:.3f})")
        else:
            mode = "SYSTEM_1_REFLEX"
            reasons.append(f"VoC ({voc_score:.3f}) <= threshold ({self.voc_threshold:.3f}); bypass deep verification")

        decision = VoCDecision(
            mode=mode,
            voc_score=voc_score,
            expected_utility_gain=round(expected_gain, 3),
            compute_cost=round(weighted_cost, 3),
            risk_level=risk,
            reasons=reasons,
        )

        self.history.append(decision)
        if len(self.history) > 200:
            self.history.pop(0)

        return decision

