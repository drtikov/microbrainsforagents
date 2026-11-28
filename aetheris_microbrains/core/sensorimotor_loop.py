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

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .cbf_spectral_barrier import CBFSpectralBarrierEngine
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
        spectral_limit: Optional[float] = None,
        plant_l2_gain: Optional[float] = None,
    ):
        if spectral_limit is not None and float(spectral_limit) <= 0.0:
            raise ValueError("spectral_limit must be positive")
        if plant_l2_gain is not None and float(plant_l2_gain) <= 0.0:
            raise ValueError("plant_l2_gain must be positive")

        self.reservoir = reservoir or LiquidStateReservoir.get_instance()
        self.action_dim = int(action_dim)
        self.spectral_limit = None if spectral_limit is None else float(spectral_limit)
        self.plant_l2_gain = None if plant_l2_gain is None else float(plant_l2_gain)

        if self.plant_l2_gain is not None:
            small_gain_bound = 0.99 / self.plant_l2_gain
            self.effective_spectral_limit: Optional[float] = (
                min(self.spectral_limit, small_gain_bound)
                if self.spectral_limit is not None
                else small_gain_bound
            )
        else:
            self.effective_spectral_limit = self.spectral_limit

        init = self.reservoir.init_online_readout(self.action_dim, method=method)
        if init.get("status") != "online_readout_ready":
            raise RuntimeError(f"online readout init failed: {init}")
        self.last_action: Optional[np.ndarray] = None

    def _enforce_readout_spectral_cbf(
        self,
        w_prev: np.ndarray,
        target: Optional[Sequence[float]] = None,
        update_receipt: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.effective_spectral_limit is None or self.reservoir.W_readout is None:
            return None

        limit = float(self.effective_spectral_limit)
        engine = CBFSpectralBarrierEngine.get_instance()
        w_curr = self.reservoir.W_readout.astype(np.float64)
        dw_raw = w_curr - w_prev
        safe_dw, was_projected = engine.enforce_safe_update(
            w_prev,
            dw_raw,
            custom_limit=limit,
        )
        w_safe = (w_prev + safe_dw).astype(np.float32)

        # Guard against float32 quantization rounding above effective_spectral_limit
        s_vals = np.linalg.svd(w_safe.astype(np.float64), compute_uv=False)
        sigma_max = float(s_vals[0]) if len(s_vals) > 0 else 0.0
        if sigma_max > limit:
            w_safe = engine.project_matrix_to_safety_manifold(
                w_safe.astype(np.float64),
                max_spectral_norm=limit * (1.0 - 1e-7),
            ).astype(np.float32)
            s_vals = np.linalg.svd(w_safe.astype(np.float64), compute_uv=False)
            sigma_max = float(s_vals[0]) if len(s_vals) > 0 else 0.0
            was_projected = True

        self.reservoir.W_readout = w_safe
        if was_projected and update_receipt is not None and target is not None:
            y_vec = np.asarray(target, dtype=np.float64).flatten()
            s_vec = self.reservoir.state.astype(np.float64).flatten()
            if y_vec.shape[0] == w_safe.shape[1] and s_vec.shape[0] == w_safe.shape[0]:
                post_pred = w_safe.astype(np.float64).T @ s_vec
                post_mse = float(np.mean((y_vec - post_pred) ** 2))
                update_receipt["post_mse"] = round(post_mse, 8)

        cbf_info = {
            "enforced": True,
            "projected": bool(was_projected),
            "sigma_max": round(sigma_max, 8),
            "effective_limit": limit,
        }
        if update_receipt is not None and isinstance(update_receipt, dict):
            update_receipt["spectral_cbf"] = cbf_info
        return cbf_info

    def __call__(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Suitable as SensorimotorLoop.reflex_fn. observation['vector'] is required."""
        obs_vec = observation.get("vector")
        if obs_vec is None:
            return {"action": [0.0] * self.action_dim, "reflex_reason": "no_vector"}
        target = observation.get("target_action")
        w_prev = (
            self.reservoir.W_readout.astype(np.float64).copy()
            if (target is not None and self.effective_spectral_limit is not None and self.reservoir.W_readout is not None)
            else None
        )
        out = self.reservoir.online_step(
            obs_vec,
            target=target,
            event_tag=str(observation.get("tag", "reflex")),
        )
        if w_prev is not None:
            self._enforce_readout_spectral_cbf(
                w_prev,
                target=target,
                update_receipt=out.get("online_update"),
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
        w_prev = (
            self.reservoir.W_readout.astype(np.float64).copy()
            if (self.effective_spectral_limit is not None and self.reservoir.W_readout is not None)
            else None
        )
        res = self.reservoir.update_readout_online(target_action)
        if w_prev is not None and res.get("status", "").startswith("updated"):
            self._enforce_readout_spectral_cbf(w_prev, target=target_action, update_receipt=res)
        return res


def _blend_action_value(
    reflex_val: Any,
    backup_val: Any,
    decay: float,
) -> Any:
    """Linearly blend reflex action toward backup action using authority ``decay`` in [0, 1]."""
    alpha = max(0.0, min(1.0, float(decay)))
    if alpha >= 1.0:
        return reflex_val
    if alpha <= 0.0:
        return backup_val
    a_ref = np.asarray(reflex_val, dtype=np.float64)
    a_bak = np.asarray(backup_val, dtype=np.float64)
    if a_ref.shape != a_bak.shape:
        if a_ref.ndim == 1 and a_bak.ndim == 1:
            n = min(a_ref.shape[0], a_bak.shape[0])
            a_ref = a_ref[:n]
            a_bak = a_bak[:n]
        else:
            return backup_val
    blended = alpha * a_ref + (1.0 - alpha) * a_bak
    if isinstance(reflex_val, np.ndarray):
        return blended.astype(reflex_val.dtype, copy=False)
    if isinstance(reflex_val, (int, float)) and np.ndim(blended) == 0:
        return float(blended)
    if isinstance(reflex_val, tuple):
        return tuple(float(v) for v in blended.tolist())
    return blended.tolist()


class AsyncMultiRateSensorimotorLoop:
    """
    Asynchronous multi-rate sensorimotor loop with atomic snapshot double-buffering
    and a deterministic dead-man watchdog.

    ``tick(observation)`` NEVER invokes ``cognitive_fn`` or ``planner_fn`` inline:
    background worker threads run those slower layers on their own schedules and
    atomically swap stamped snapshots ``(seq, timestamp, payload)``. On every tick,
    the reflex path runs immediately, reads the latest snapshots, and applies the
    dead-man TTL watchdog (linear authority decay from 1.0 -> 0.0 toward ``backup_fn``).
    """

    def __init__(
        self,
        dt: float,
        reflex_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        cognitive_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        planner_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        backup_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cognitive_period_ms: Optional[float] = None,
        planner_period_ms: Optional[float] = None,
        cognitive_every: int = 20,
        planner_every: int = 1000,
        cognitive_ttl_ms: float = 250.0,
        planner_ttl_ms: float = 2000.0,
        watchdog_ramp_ms: Optional[float] = None,
        auto_start: bool = True,
    ):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.budget_ms = self.dt * 1000.0
        self.reflex_fn = reflex_fn
        self.cognitive_fn = cognitive_fn
        self.planner_fn = planner_fn
        self.backup_fn = backup_fn

        self.cognitive_every = max(1, int(cognitive_every))
        self.planner_every = max(1, int(planner_every))
        self.cognitive_period_ms = (
            float(cognitive_period_ms)
            if cognitive_period_ms is not None
            else self.cognitive_every * self.budget_ms
        )
        self.planner_period_ms = (
            float(planner_period_ms)
            if planner_period_ms is not None
            else self.planner_every * self.budget_ms
        )
        self.cognitive_ttl_ms = float(cognitive_ttl_ms)
        self.planner_ttl_ms = float(planner_ttl_ms)
        self.watchdog_ramp_ms = (
            float(watchdog_ramp_ms) if watchdog_ramp_ms is not None else max(self.cognitive_ttl_ms, 1e-6)
        )
        self.auto_start = bool(auto_start)

        self.tick_count = 0
        self.overruns = {"reflex": 0, "cognitive": 0, "planner": 0}
        self.last_tick_latency_ms: Optional[float] = None

        # Lockless-style atomic snapshot references (immutable tuples swapped on worker completion)
        self._cog_snapshot: Optional[Tuple[int, float, Dict[str, Any]]] = None
        self._plan_snapshot: Optional[Tuple[int, float, Dict[str, Any]]] = None
        self._cog_error: Optional[str] = None
        self._plan_error: Optional[str] = None
        self._last_seen_cog_seq = 0
        self._last_seen_plan_seq = 0

        self._latest_obs: Optional[Dict[str, Any]] = None
        self._start_ts: Optional[float] = None
        self._running = False
        self._stop_event = threading.Event()
        self._obs_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lifecycle_lock = threading.Lock()

    def start(self) -> "AsyncMultiRateSensorimotorLoop":
        with self._lifecycle_lock:
            if self._running:
                return self
            self._stop_event.clear()
            self._obs_event.clear()
            if self._start_ts is None:
                self._start_ts = time.perf_counter()
            self._threads = []
            if self.cognitive_fn is not None:
                t_cog = threading.Thread(
                    target=self._worker_loop,
                    args=("cognitive", self.cognitive_fn, self.cognitive_period_ms / 1000.0),
                    daemon=True,
                    name="aetheris-async-cognitive",
                )
                t_cog.start()
                self._threads.append(t_cog)
            if self.planner_fn is not None:
                t_plan = threading.Thread(
                    target=self._worker_loop,
                    args=("planner", self.planner_fn, self.planner_period_ms / 1000.0),
                    daemon=True,
                    name="aetheris-async-planner",
                )
                t_plan.start()
                self._threads.append(t_plan)
            self._running = True
        return self

    def stop(self, timeout: float = 1.0) -> None:
        with self._lifecycle_lock:
            if not self._running and not self._threads:
                return
            self._stop_event.set()
            self._obs_event.set()
            threads = list(self._threads)
            self._threads = []
            self._running = False
        for t in threads:
            if t.is_alive():
                t.join(timeout=timeout)

    def __enter__(self) -> "AsyncMultiRateSensorimotorLoop":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop(timeout=0.05)
        except Exception:
            pass

    def reset(self) -> None:
        self.tick_count = 0
        self.overruns = {"reflex": 0, "cognitive": 0, "planner": 0}
        self.last_tick_latency_ms = None
        self._cog_snapshot = None
        self._plan_snapshot = None
        self._cog_error = None
        self._plan_error = None
        self._last_seen_cog_seq = 0
        self._last_seen_plan_seq = 0
        self._latest_obs = None
        self._start_ts = time.perf_counter()

    def _worker_loop(
        self,
        kind: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        period_s: float,
    ) -> None:
        seq = 0
        while not self._stop_event.is_set():
            obs = self._latest_obs
            if obs is None:
                self._obs_event.wait(timeout=0.02)
                continue
            t0 = time.perf_counter()
            try:
                out = fn(obs) or {}
                done_ts = time.perf_counter()
                seq += 1
                snap = (seq, done_ts, dict(out))
                if kind == "cognitive":
                    self._cog_snapshot = snap
                    self._cog_error = None
                else:
                    self._plan_snapshot = snap
                    self._plan_error = None
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                if kind == "cognitive":
                    self._cog_error = err_msg
                else:
                    self._plan_error = err_msg
            elapsed_s = time.perf_counter() - t0
            if elapsed_s * 1000.0 > self.budget_ms:
                self.overruns[kind] += 1
            wait_s = max(0.0, period_s - elapsed_s)
            if wait_s > 0.0:
                self._stop_event.wait(timeout=wait_s)
            else:
                # Yield briefly to avoid starving reflex thread on tight loops
                self._stop_event.wait(timeout=0.0)

    def _compute_ttl_status(
        self,
        has_fn: bool,
        snap: Optional[Tuple[int, float, Dict[str, Any]]],
        ttl_ms: float,
        now: float,
    ) -> Tuple[Optional[float], bool, float]:
        if not has_fn:
            return None, False, 1.0
        ref_ts = snap[1] if snap is not None else (self._start_ts if self._start_ts is not None else now)
        age_ms = max(0.0, (now - ref_ts) * 1000.0)
        if age_ms <= ttl_ms:
            return age_ms, False, 1.0
        ramp = max(self.watchdog_ramp_ms, 1e-6)
        decay = max(0.0, min(1.0, 1.0 - (age_ms - ttl_ms) / ramp))
        return age_ms, True, decay

    def tick(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one real-time control cycle.

        Runs ``reflex_fn(observation)`` immediately, reads the latest stamped
        cognitive/planner snapshots without blocking, evaluates dead-man TTL
        watchdogs, and blends toward ``backup_fn(observation)`` if expired.
        """
        t_start = time.perf_counter()
        if self._start_ts is None:
            self._start_ts = t_start
        if self.auto_start and not self._running and (self.cognitive_fn is not None or self.planner_fn is not None):
            self.start()

        self.tick_count += 1
        # Publish latest observation atomically for background workers
        self._latest_obs = dict(observation) if isinstance(observation, dict) else observation
        if self._running:
            self._obs_event.set()

        # 1. Reflex layer (inline, deterministic)
        t0 = time.perf_counter()
        try:
            reflex_out = dict(self.reflex_fn(observation) or {})
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reflex_out = {}
        reflex_ms = (time.perf_counter() - t0) * 1000.0
        if reflex_ms > self.budget_ms:
            self.overruns["reflex"] += 1

        # 2. Read latest stamped cognitive & planner snapshots (zero blocking)
        now = time.perf_counter()
        cog_snap = self._cog_snapshot
        plan_snap = self._plan_snapshot

        cog_updated = False
        if cog_snap is not None and cog_snap[0] != self._last_seen_cog_seq:
            self._last_seen_cog_seq = cog_snap[0]
            cog_updated = True
        plan_updated = False
        if plan_snap is not None and plan_snap[0] != self._last_seen_plan_seq:
            self._last_seen_plan_seq = plan_snap[0]
            plan_updated = True

        cog_data = dict(cog_snap[2]) if cog_snap is not None else {}
        if self._cog_error is not None:
            cog_data["cognitive_error"] = self._cog_error
        plan_data = dict(plan_snap[2]) if plan_snap is not None else {}
        if self._plan_error is not None:
            plan_data["planner_error"] = self._plan_error

        # 3. Dead-man TTL watchdog evaluation
        cog_age_ms, cog_expired, cog_decay = self._compute_ttl_status(
            self.cognitive_fn is not None, cog_snap, self.cognitive_ttl_ms, now
        )
        plan_age_ms, plan_expired, plan_decay = self._compute_ttl_status(
            self.planner_fn is not None, plan_snap, self.planner_ttl_ms, now
        )
        watchdog_expired = cog_expired or plan_expired
        watchdog_active = watchdog_expired
        decay = min(cog_decay, plan_decay)

        # 4. Blend action toward backup_fn when watchdog expires (or reflex errored)
        final_out = dict(reflex_out)
        backup_used = False
        if self.backup_fn is not None and (watchdog_active or error is not None):
            effective_decay = 0.0 if error is not None else decay
            try:
                backup_out = dict(self.backup_fn(observation) or {})
                backup_used = True
                if error is not None:
                    final_out = backup_out
                elif "action" in final_out and "action" in backup_out:
                    final_out["action"] = _blend_action_value(
                        final_out["action"], backup_out["action"], effective_decay
                    )
                elif "action" not in final_out and "action" in backup_out:
                    final_out["action"] = backup_out["action"]
            except Exception as exc:
                final_out["backup_error"] = f"{type(exc).__name__}: {exc}"

        tick_total_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_tick_latency_ms = tick_total_ms
        cog_age_rounded = None if cog_age_ms is None else round(cog_age_ms, 4)
        plan_age_rounded = None if plan_age_ms is None else round(plan_age_ms, 4)
        reflex_rounded = round(reflex_ms, 4)

        return {
            "tick": self.tick_count,
            "action": final_out,
            "reflex_error": error,
            "cognitive": cog_data,
            "planner": plan_data,
            "reflex_ms": reflex_rounded,
            "cognitive_age_ms": cog_age_rounded,
            "planner_age_ms": plan_age_rounded,
            "watchdog_active": watchdog_active,
            "watchdog_expired": watchdog_expired,
            "cognitive_expired": cog_expired,
            "planner_expired": plan_expired,
            "decay": round(decay, 6),
            "watchdog_decay": round(decay, 6),
            "backup_used": backup_used,
            "layers_ran": {
                "reflex": True,
                "cognitive": cog_updated,
                "planner": plan_updated,
            },
            "timing_ms": {
                "reflex": reflex_rounded,
                "tick_total": round(tick_total_ms, 4),
                "budget": round(self.budget_ms, 4),
                "cognitive_age_ms": cog_age_rounded,
                "planner_age_ms": plan_age_rounded,
                "reflex_overruns_total": self.overruns["reflex"],
                "cognitive_overruns_total": self.overruns["cognitive"],
                "planner_overruns_total": self.overruns["planner"],
            },
        }
