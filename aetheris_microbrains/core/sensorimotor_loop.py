"""Fixed-dt sensorimotor loop contract and multi-rate scheduler for robotics.

P: 2026-11-18 — robotics extension. Every pre-existing loop in this library is
driven by tool-call events with wall-clock dt gaps; a robot needs a fixed
timestep contract. This module provides:

1. ``SensorimotorLoop`` — the CONTRACT object a real-time clock (RTOS timer,
   ros2 control cycle, game loop) drives via ``tick(observation)``. The caller
   owns the clock; the loop owns the layer schedule and honest timing receipts
   (measured wall time per layer, overrun counters against the tick budget).
   Three timescales, following the standard robotics loop structure
   (planner ~1 Hz, cognitive/policy 10-50 Hz, reflex every tick):
     - reflex    : every tick
     - cognitive : every ``cognitive_every`` ticks (default 20 -> 50 Hz @ 1 kHz)
     - planner   : every ``planner_every`` ticks   (default 1000 -> 1 Hz @ 1 kHz)
   Layer results are cached between their runs, so a slow planner never blocks
   the reflex path.

2. ``ReservoirReflexLayer`` — a reflex policy backed by LiquidStateReservoir
   with an ONLINE-adapted readout (FORCE/RLS), i.e. a wired consumer of the
   online readout extension. Prediction is emitted pre-update (honest online
   protocol); adaptation happens only when a target action is supplied.

All numbers in results are MEASURED receipts, never claimed constants.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .liquid_state_reservoir import LiquidStateReservoir


class SensorimotorLoop:
    """Multi-rate tick contract. The consumer's real-time clock calls tick()."""

    def __init__(
        self,
        dt: float,
        reflex_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        cognitive_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        planner_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cognitive_every: int = 20,
        planner_every: int = 1000,
    ):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.budget_ms = self.dt * 1000.0
        self.reflex_fn = reflex_fn
        self.cognitive_fn = cognitive_fn
        self.planner_fn = planner_fn
        self.cognitive_every = max(1, int(cognitive_every))
        self.planner_every = max(1, int(planner_every))

        self.tick_count = 0
        self.cognitive_cache: Dict[str, Any] = {}
        self.planner_cache: Dict[str, Any] = {}
        self.overruns = {"reflex": 0, "cognitive": 0, "planner": 0}
        self.last_tick_latency_ms: Optional[float] = None

    def reset(self) -> None:
        self.tick_count = 0
        self.cognitive_cache = {}
        self.planner_cache = {}
        self.overruns = {"reflex": 0, "cognitive": 0, "planner": 0}
        self.last_tick_latency_ms = None

    def tick(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """One control cycle. Runs the scheduled layers, caches slow layers."""
        t_start = time.perf_counter()
        self.tick_count += 1

        reflex_out: Dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            reflex_out = self.reflex_fn(observation) or {}
            error = None
        except Exception as exc:  # reflex path must never kill the loop
            error = f"{type(exc).__name__}: {exc}"
            reflex_out = {}
        reflex_ms = (time.perf_counter() - t0) * 1000.0
        if reflex_ms > self.budget_ms:
            self.overruns["reflex"] += 1

        run_cognitive = self.cognitive_fn is not None and self.tick_count % self.cognitive_every == 0
        if run_cognitive:
            t0 = time.perf_counter()
            try:
                self.cognitive_cache = self.cognitive_fn(observation) or {}
            except Exception as exc:
                self.cognitive_cache = {"cognitive_error": f"{type(exc).__name__}: {exc}"}
            if (time.perf_counter() - t0) * 1000.0 > self.budget_ms:
                self.overruns["cognitive"] += 1

        run_planner = self.planner_fn is not None and self.tick_count % self.planner_every == 0
        if run_planner:
            t0 = time.perf_counter()
            try:
                self.planner_cache = self.planner_fn(observation) or {}
            except Exception as exc:
                self.planner_cache = {"planner_error": f"{type(exc).__name__}: {exc}"}
            if (time.perf_counter() - t0) * 1000.0 > self.budget_ms:
                self.overruns["planner"] += 1

        self.last_tick_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "tick": self.tick_count,
            "action": reflex_out,
            "reflex_error": error,
            "cognitive": self.cognitive_cache,
            "planner": self.planner_cache,
            "layers_ran": {
                "reflex": True,
                "cognitive": run_cognitive,
                "planner": run_planner,
            },
            "timing_ms": {
                "reflex": round(reflex_ms, 4),
                "tick_total": round(self.last_tick_latency_ms, 4),
                "budget": round(self.budget_ms, 4),
                "reflex_overruns_total": self.overruns["reflex"],
                "cognitive_overruns_total": self.overruns["cognitive"],
                "planner_overruns_total": self.overruns["planner"],
            },
        }


class ReservoirReflexLayer:
    """Reflex policy: observation vector -> reservoir liquid -> online readout -> action."""

    def __init__(
        self,
        reservoir: Optional[LiquidStateReservoir] = None,
        action_dim: int = 2,
        method: str = "force",
    ):
        self.reservoir = reservoir or LiquidStateReservoir.get_instance()
        self.action_dim = int(action_dim)
        init = self.reservoir.init_online_readout(self.action_dim, method=method)
        if init.get("status") != "online_readout_ready":
            raise RuntimeError(f"online readout init failed: {init}")
        self.last_action: Optional[np.ndarray] = None

    def __call__(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Suitable as SensorimotorLoop.reflex_fn. observation['vector'] is required."""
        obs_vec = observation.get("vector")
        if obs_vec is None:
            return {"action": [0.0] * self.action_dim, "reflex_reason": "no_vector"}
        target = observation.get("target_action")
        out = self.reservoir.online_step(
            obs_vec,
            target=target,
            event_tag=str(observation.get("tag", "reflex")),
        )
        pred = out.get("readout_prediction")
        action = [float(v) for v in pred[: self.action_dim]] if pred else [0.0] * self.action_dim
        self.last_action = np.asarray(action, dtype=np.float64)
        return {
            "action": action,
            "step_count": out.get("step_count"),
            "reservoir_energy": out.get("reservoir_energy"),
            "online": out.get("online"),
            "online_update": out.get("online_update"),
        }

    def adapt(self, target_action: Sequence[float]) -> Dict[str, Any]:
        """Explicit adaptation against the current liquid state."""
        return self.reservoir.update_readout_online(target_action)
