"""
Aetheris Astrocyte Homeostatic Regulation Field.
Inspired by neuro-glial computing and non-neuronal chemical modulation (2025-2026).
Simulates diffuse slow-acting astrocyte fields that sense failure velocity and metabolic pressure,
releasing chemical inhibitory neuromodulators to cool down chaotic exploration and stabilize the agent.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional

class AstrocyteHomeostaticField:
    """A slow-acting diffuse homeostatic field regulating neural excitation and exploration."""

    _instance: Optional[AstrocyteHomeostaticField] = None

    def __init__(self, clearance_rate: float = 0.05):
        # Diffuse chemical concentration reservoirs [0.0, 1.0]
        self.glutamate_excitation: float = 0.2   # Neuronal excitation / drive
        self.gaba_inhibition: float = 0.1        # Inhibitory calming transmitter
        self.calcium_wave: float = 0.0           # Internal astrocyte stress signal
        self.clearance_rate = clearance_rate
        self.last_update_ts = time.time()

    @classmethod
    def get_instance(cls) -> AstrocyteHomeostaticField:
        if cls._instance is None:
            cls._instance = AstrocyteHomeostaticField()
        return cls._instance

    def register_neural_event(self, is_failure: bool, execution_latency_s: float) -> Dict[str, Any]:
        """
        Sensory input from micro-brains and tools:
        - Failures or latency spikes trigger calcium wave in astrocytes.
        - Astrocytes respond by releasing GABA-like inhibitory tone to prevent loop hysteria.
        """
        now = time.time()
        dt = max(0.01, min(10.0, now - self.last_update_ts))
        self.last_update_ts = now

        # Natural chemical uptake / clearance
        self.glutamate_excitation = max(0.1, self.glutamate_excitation - self.clearance_rate * dt)
        self.gaba_inhibition = max(0.05, self.gaba_inhibition - self.clearance_rate * dt)
        self.calcium_wave = max(0.0, self.calcium_wave - (self.clearance_rate * 1.5) * dt)

        if is_failure:
            # Failure spike raises astrocyte calcium
            self.calcium_wave = min(1.0, self.calcium_wave + 0.35)
            # Astrocytes dump inhibitory modulator to cool runaway loop
            self.gaba_inhibition = min(1.0, self.gaba_inhibition + 0.25)
        else:
            # Success maintains healthy excitation
            self.glutamate_excitation = min(1.0, self.glutamate_excitation + 0.1)

        # High latency also stresses the glial field
        if execution_latency_s > 2.0:
            self.calcium_wave = min(1.0, self.calcium_wave + 0.2)

        # Regulate cognitive state
        if self.gaba_inhibition > 0.6 or self.calcium_wave > 0.7:
            state = "GLIAL_INHIBITION_ACTIVE" # Cool down, explore alternative branches
            recommended_temp = 0.2
            backoff_delay = 1.5
        elif self.glutamate_excitation > 0.5 and self.calcium_wave < 0.3:
            state = "NOMINAL_HIGH_EXCITEMENT" # Fast path, exploit known tools
            recommended_temp = 0.0
            backoff_delay = 0.0
        else:
            state = "HOMEOSTATIC_EQUILIBRIUM"
            recommended_temp = 0.1
            backoff_delay = 0.2

        return {
            "status": "astrocyte_modulated",
            "state": state,
            "calcium_wave_stress": round(self.calcium_wave, 3),
            "inhibition_tone": round(self.gaba_inhibition, 3),
            "excitation_drive": round(self.glutamate_excitation, 3),
            "recommended_exploration_temp": recommended_temp,
            "recommended_backoff_s": backoff_delay,
        }

