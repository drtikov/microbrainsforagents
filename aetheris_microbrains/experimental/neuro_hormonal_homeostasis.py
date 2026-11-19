"""
Neuro-Hormonal Homeostatic Temperament Engine for Aetheris.
Ported and evolved from Aura-Alien VFS (E:/auraalienvfs/homeostaticCycles.ts) + 2025/2026 SOTA.

Manages persistent biological signals:
- Primary signals: Focus, Curiosity, Resilience, Arousal.
- Hormonal signals: Dopamine (Reward/Plasticity), Cortisol (Stress/Friction), Serotonin (Consensus), Acetylcholine (Attention).
- Exponential homeostatic decay (decay_rate = 0.98): prevents permanent panic or mania.
- Temperament baselines: Shifts equilibrium according to current cognitive posture
  (e.g. 'EXPLORER', 'ARCHITECT', 'SENTRY', 'HEALER').
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("aetheris.hormonal_homeostasis")


@dataclass
class HomeostaticState:
    primary: Dict[str, float]
    hormonal: Dict[str, float]
    temperament: str
    allostatic_stress: float
    is_balanced: bool


class NeuroHormonalHomeostasis:
    """
    Biological allostasis regulator balancing exploration, stress, and focus.
    """

    _instance: Optional["NeuroHormonalHomeostasis"] = None

    TEMPERAMENTS: Dict[str, Dict[str, float]] = {
        "EXPLORER": {"focus_bias": -0.1, "curiosity_drive": 0.85, "stress_tolerance": 0.70},
        "ARCHITECT": {"focus_bias": 0.35, "curiosity_drive": 0.40, "stress_tolerance": 0.80},
        "SENTRY": {"focus_bias": 0.50, "curiosity_drive": 0.15, "stress_tolerance": 0.90},
        "HEALER": {"focus_bias": 0.10, "curiosity_drive": 0.30, "stress_tolerance": 0.60},
    }

    def __init__(self, default_temperament: str = "ARCHITECT"):
        self.temperament_name = default_temperament
        self.decay_rate = 0.98  # 2% decay per tick toward baseline
        self.primary = {
            "focus": 0.50,
            "curiosity": 0.50,
            "resilience": 0.80,
            "arousal": 0.40,
        }
        self.hormonal = {
            "dopamine": 0.30,     # Motivation / reward expectation
            "cortisol": 0.10,     # Friction / failure stress
            "serotonin": 0.50,    # Colony harmony / satisfaction
            "acetylcholine": 0.40 # Learning rate / sensory precision
        }

    @classmethod
    def get_instance(cls) -> "NeuroHormonalHomeostasis":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_temperament(self, name: str):
        if name.upper() in self.TEMPERAMENTS:
            self.temperament_name = name.upper()

    def record_stimulus(
        self,
        event_type: str,  # 'SUCCESS', 'FAILURE', 'ANOMALY', 'NOVELTY', 'IDLE'
        magnitude: float = 0.20
    ):
        """Modulates hormones based on cognitive feedback."""
        mag = float(np.clip(magnitude, 0.0, 1.0)) if "np" in globals() else min(1.0, max(0.0, float(magnitude)))

        if event_type == "SUCCESS":
            self.hormonal["dopamine"] = min(1.0, self.hormonal["dopamine"] + mag * 0.4)
            self.hormonal["serotonin"] = min(1.0, self.hormonal["serotonin"] + mag * 0.2)
            self.hormonal["cortisol"] = max(0.02, self.hormonal["cortisol"] - mag * 0.3)
        elif event_type == "FAILURE":
            self.hormonal["cortisol"] = min(1.0, self.hormonal["cortisol"] + mag * 0.5)
            self.hormonal["dopamine"] = max(0.05, self.hormonal["dopamine"] - mag * 0.2)
            self.hormonal["acetylcholine"] = min(1.0, self.hormonal["acetylcholine"] + mag * 0.3)
        elif event_type == "NOVELTY":
            self.primary["curiosity"] = min(1.0, self.primary["curiosity"] + mag * 0.35)
            self.hormonal["dopamine"] = min(1.0, self.hormonal["dopamine"] + mag * 0.25)
        elif event_type == "ANOMALY":
            self.primary["arousal"] = min(1.0, self.primary["arousal"] + mag * 0.4)
            self.hormonal["acetylcholine"] = min(1.0, self.hormonal["acetylcholine"] + mag * 0.4)

    def tick_homeostasis(self) -> HomeostaticState:
        """Executes exponential decay toward temperament equilibrium."""
        temp = self.TEMPERAMENTS.get(self.temperament_name, self.TEMPERAMENTS["ARCHITECT"])

        # Target baselines
        target_focus = max(0.0, min(1.0, 0.5 + temp["focus_bias"]))
        target_curiosity = max(0.0, min(1.0, temp["curiosity_drive"]))

        # Gravitate primary signals
        self.primary["focus"] = self.primary["focus"] * self.decay_rate + target_focus * (1.0 - self.decay_rate)
        self.primary["curiosity"] = self.primary["curiosity"] * self.decay_rate + target_curiosity * (1.0 - self.decay_rate)
        self.primary["arousal"] = self.primary["arousal"] * self.decay_rate + 0.30 * (1.0 - self.decay_rate)

        # Gravitate hormonal signals toward healthy resting baselines
        baselines = {"dopamine": 0.25, "cortisol": 0.08, "serotonin": 0.45, "acetylcholine": 0.35}
        for h, val in self.hormonal.items():
            base = baselines.get(h, 0.20)
            self.hormonal[h] = val * self.decay_rate + base * (1.0 - self.decay_rate)

        # Compute allostatic stress
        stress = self.hormonal["cortisol"] + (1.0 - self.primary["resilience"]) * 0.5
        is_balanced = stress < temp["stress_tolerance"]

        return HomeostaticState(
            primary={k: round(v, 4) for k, v in self.primary.items()},
            hormonal={k: round(v, 4) for k, v in self.hormonal.items()},
            temperament=self.temperament_name,
            allostatic_stress=round(stress, 4),
            is_balanced=is_balanced,
        )

