"""Pure-stdlib 1 kHz flight data recorder, counterfactual physics gate, and log replay.

Provides:
- ``FlightDataRecorder``: Fixed-capacity ``collections.deque`` ring buffer storing
  SHA-256 hash-chained 1 kHz telemetry frames in RAM with zero disk I/O during
  ``record(...)`` and atomic JSONL dumps via ``trigger_dump(...)``.
- ``verify_flight_log``: Constant-structure verification of dumped JSONL flight
  logs to detect tampering, missing frames, or modified values.
- ``CounterfactualPhysicsGate``: Deterministic pure-stdlib forward kinematic and
  point-mass rollout pre-check over candidate waypoint plans against sphere and
  box obstacles before execution.
- ``replay_flight_log``: Hash-chain verification and counterfactual replay of a
  dumped flight log under stricter ``hard_limits`` and ``conformal_margin``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from aetheris_microbrains.core.atomic import atomic_replace


_GENESIS_HASH = "0" * 64


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _normalize_json_value(value: Any) -> Any:
    """Convert nested containers into deterministic JSON-safe finite primitives."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be serialized to flight log")
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    raise TypeError(f"unsupported value type in flight frame: {type(value).__name__}")


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_non_finite(k) or _contains_non_finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _compute_frame_hash(frame_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(frame_without_hash)).hexdigest()


class FlightDataRecorder:
    """Fixed-capacity in-memory 1 kHz ring buffer with SHA-256 hash chaining."""

    def __init__(self, capacity: int = 10_000) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._frames: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._prev_hash: str = _GENESIS_HASH
        self._total_recorded: int = 0

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def total_recorded(self) -> int:
        return self._total_recorded

    @property
    def last_hash(self) -> str:
        return self._prev_hash

    @property
    def frames(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._frames)

    def record(
        self,
        *,
        tick: int,
        timestamp_ns: int,
        state: Any,
        u_nominal: Any,
        u_filtered: Any,
        min_barrier_slack: float,
        aci_q_hat: float,
        token_id: Optional[str],
        status: str,
    ) -> Dict[str, Any]:
        """Append a single 1 kHz frame to the RAM ring buffer with zero disk I/O."""
        slack_f = float(min_barrier_slack)
        q_hat_f = float(aci_q_hat)
        if not math.isfinite(slack_f) or not math.isfinite(q_hat_f):
            raise ValueError("min_barrier_slack and aci_q_hat must be finite")

        body: Dict[str, Any] = {
            "tick": int(tick),
            "timestamp_ns": int(timestamp_ns),
            "state": _normalize_json_value(state),
            "u_nominal": _normalize_json_value(u_nominal),
            "u_filtered": _normalize_json_value(u_filtered),
            "min_barrier_slack": slack_f,
            "aci_q_hat": q_hat_f,
            "token_id": str(token_id) if token_id is not None else None,
            "status": str(status),
            "prev_hash": self._prev_hash,
        }
        frame_hash = _compute_frame_hash(body)
        frame = dict(body)
        frame["frame_hash"] = frame_hash
        self._prev_hash = frame_hash
        self._total_recorded += 1
        self._frames.append(frame)
        return frame

    def trigger_dump(self, path: Union[str, Path], reason: str = "anomaly") -> Dict[str, Any]:
        """Atomically write the chained ring-buffer frames and summary header as JSONL."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")

        snapshot = list(self._frames)
        first_tick = snapshot[0]["tick"] if snapshot else None
        last_tick = snapshot[-1]["tick"] if snapshot else None
        anchor_prev_hash = snapshot[0]["prev_hash"] if snapshot else _GENESIS_HASH
        head_hash = snapshot[-1]["frame_hash"] if snapshot else _GENESIS_HASH

        header = {
            "record_type": "flight_log_header",
            "schema_version": 1,
            "reason": str(reason),
            "dumped_at_ns": time.time_ns(),
            "capacity": self.capacity,
            "total_recorded": self._total_recorded,
            "frame_count": len(snapshot),
            "first_tick": first_tick,
            "last_tick": last_tick,
            "anchor_prev_hash": anchor_prev_hash,
            "head_hash": head_hash,
        }

        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
            for frame in snapshot:
                fh.write(json.dumps(frame, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        atomic_replace(str(tmp), str(target))

        return {
            "path": str(target),
            "reason": str(reason),
            "frame_count": len(snapshot),
            "first_tick": first_tick,
            "last_tick": last_tick,
            "head_hash": head_hash,
        }


def _load_flight_log(path: Union[str, Path]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    target = Path(path)
    if not target.exists():
        return None, [], "file_not_found"
    try:
        raw_lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return None, [], f"read_error:{exc}"
    if not raw_lines:
        return None, [], "empty_flight_log"

    try:
        header = json.loads(raw_lines[0])
    except json.JSONDecodeError:
        return None, [], "invalid_header_json"

    if not isinstance(header, Mapping) or header.get("record_type") != "flight_log_header":
        return None, [], "missing_flight_log_header"

    frames: List[Dict[str, Any]] = []
    for idx, line in enumerate(raw_lines[1:], start=1):
        if not line.strip():
            return dict(header), frames, f"blank_line_at_{idx}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return dict(header), frames, f"invalid_frame_json_at_{idx}"
        if not isinstance(obj, Mapping):
            return dict(header), frames, f"invalid_frame_type_at_{idx}"
        frames.append(dict(obj))
    return dict(header), frames, None


_REQUIRED_FRAME_KEYS = (
    "tick",
    "timestamp_ns",
    "state",
    "u_nominal",
    "u_filtered",
    "min_barrier_slack",
    "aci_q_hat",
    "token_id",
    "status",
    "prev_hash",
    "frame_hash",
)


def verify_flight_log(path: Union[str, Path]) -> Dict[str, Any]:
    """Verify SHA-256 hash chain integrity and completeness of a dumped JSONL flight log."""
    header, frames, parse_error = _load_flight_log(path)
    if parse_error is not None or header is None:
        return {
            "verified": False,
            "valid": False,
            "reason": parse_error or "invalid_log",
            "frame_count": len(frames),
            "first_tick": None,
            "last_tick": None,
            "head_hash": None,
        }

    expected_count = header.get("frame_count")
    if expected_count != len(frames):
        return {
            "verified": False,
            "valid": False,
            "reason": "frame_count_mismatch",
            "frame_count": len(frames),
            "first_tick": header.get("first_tick"),
            "last_tick": header.get("last_tick"),
            "head_hash": header.get("head_hash"),
        }

    if not frames:
        if header.get("head_hash") != _GENESIS_HASH or header.get("anchor_prev_hash") != _GENESIS_HASH:
            return {
                "verified": False,
                "valid": False,
                "reason": "empty_log_hash_mismatch",
                "frame_count": 0,
                "first_tick": None,
                "last_tick": None,
                "head_hash": header.get("head_hash"),
            }
        return {
            "verified": True,
            "valid": True,
            "reason": "verified",
            "frame_count": 0,
            "first_tick": None,
            "last_tick": None,
            "head_hash": _GENESIS_HASH,
        }

    if frames[0].get("prev_hash") != header.get("anchor_prev_hash"):
        return {
            "verified": False,
            "valid": False,
            "reason": "anchor_prev_hash_mismatch",
            "frame_count": len(frames),
            "first_tick": frames[0].get("tick"),
            "last_tick": frames[-1].get("tick"),
            "head_hash": frames[-1].get("frame_hash"),
        }

    if frames[0].get("tick") != header.get("first_tick") or frames[-1].get("tick") != header.get("last_tick"):
        return {
            "verified": False,
            "valid": False,
            "reason": "tick_boundary_mismatch",
            "frame_count": len(frames),
            "first_tick": frames[0].get("tick"),
            "last_tick": frames[-1].get("tick"),
            "head_hash": frames[-1].get("frame_hash"),
        }

    prev_hash = str(header["anchor_prev_hash"])
    prev_tick: Optional[int] = None
    for idx, frame in enumerate(frames):
        for key in _REQUIRED_FRAME_KEYS:
            if key not in frame:
                return {
                    "verified": False,
                    "valid": False,
                    "reason": f"missing_field_{key}_at_{idx}",
                    "frame_count": len(frames),
                    "first_tick": frames[0].get("tick"),
                    "last_tick": frames[-1].get("tick"),
                    "head_hash": frames[-1].get("frame_hash"),
                }

        if frame["prev_hash"] != prev_hash:
            return {
                "verified": False,
                "valid": False,
                "reason": f"hash_chain_broken_at_{idx}",
                "frame_count": len(frames),
                "first_tick": frames[0].get("tick"),
                "last_tick": frames[-1].get("tick"),
                "head_hash": frames[-1].get("frame_hash"),
            }

        if prev_tick is not None and int(frame["tick"]) <= prev_tick:
            return {
                "verified": False,
                "valid": False,
                "reason": f"non_monotonic_tick_at_{idx}",
                "frame_count": len(frames),
                "first_tick": frames[0].get("tick"),
                "last_tick": frames[-1].get("tick"),
                "head_hash": frames[-1].get("frame_hash"),
            }

        body = {k: frame[k] for k in frame if k != "frame_hash"}
        try:
            expected_hash = _compute_frame_hash(body)
        except (TypeError, ValueError):
            return {
                "verified": False,
                "valid": False,
                "reason": f"non_canonical_frame_at_{idx}",
                "frame_count": len(frames),
                "first_tick": frames[0].get("tick"),
                "last_tick": frames[-1].get("tick"),
                "head_hash": frames[-1].get("frame_hash"),
            }

        if frame["frame_hash"] != expected_hash:
            return {
                "verified": False,
                "valid": False,
                "reason": f"frame_hash_tampered_at_{idx}",
                "frame_count": len(frames),
                "first_tick": frames[0].get("tick"),
                "last_tick": frames[-1].get("tick"),
                "head_hash": frames[-1].get("frame_hash"),
            }

        prev_hash = str(frame["frame_hash"])
        prev_tick = int(frame["tick"])

    if prev_hash != header.get("head_hash"):
        return {
            "verified": False,
            "valid": False,
            "reason": "head_hash_mismatch",
            "frame_count": len(frames),
            "first_tick": frames[0].get("tick"),
            "last_tick": frames[-1].get("tick"),
            "head_hash": prev_hash,
        }

    return {
        "verified": True,
        "valid": True,
        "reason": "verified",
        "frame_count": len(frames),
        "first_tick": frames[0]["tick"],
        "last_tick": frames[-1]["tick"],
        "head_hash": prev_hash,
    }


def _signed_distance_to_obstacle(point: Sequence[float], obstacle: Mapping[str, Any]) -> float:
    """Compute Euclidean signed distance (>0 outside, <=0 inside/touching) for sphere/box."""
    obs_type = str(obstacle.get("type") or obstacle.get("shape") or "").strip().lower()
    if not obs_type:
        if "radius" in obstacle and "center" in obstacle:
            obs_type = "sphere"
        elif "min" in obstacle and "max" in obstacle:
            obs_type = "box"
        elif "center" in obstacle and ("half_extents" in obstacle or "size" in obstacle):
            obs_type = "box"

    if obs_type == "sphere":
        center = [float(c) for c in obstacle["center"]]
        radius = float(obstacle["radius"])
        if len(center) != len(point):
            raise ValueError("obstacle dimension mismatch")
        dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(point, center))) - radius
        return dist

    if obs_type == "box":
        if "min" in obstacle and "max" in obstacle:
            bmin = [float(v) for v in obstacle["min"]]
            bmax = [float(v) for v in obstacle["max"]]
        else:
            center = [float(c) for c in obstacle["center"]]
            if "half_extents" in obstacle:
                half = [float(h) for h in obstacle["half_extents"]]
            else:
                half = [float(s) * 0.5 for s in obstacle["size"]]
            bmin = [c - h for c, h in zip(center, half)]
            bmax = [c + h for c, h in zip(center, half)]
        if len(bmin) != len(point) or len(bmax) != len(point):
            raise ValueError("obstacle dimension mismatch")
        deltas = [max(lo - p, 0.0, p - hi) for p, lo, hi in zip(point, bmin, bmax)]
        outside_dist = math.sqrt(sum(d * d for d in deltas))
        if outside_dist > 0.0:
            return outside_dist
        # Inside box: negative distance to nearest face
        penetration = min(min(p - lo, hi - p) for p, lo, hi in zip(point, bmin, bmax))
        return -penetration

    raise ValueError(f"unsupported obstacle type: {obs_type!r}")


class CounterfactualPhysicsGate:
    """Pure-stdlib deterministic forward kinematic / point-mass rollout gate."""

    def __init__(
        self,
        *,
        max_linear_vel_mps: Optional[float] = None,
        max_contact_force_n: Optional[float] = None,
        max_kinetic_energy_j: Optional[float] = None,
        mass_kg: float = 1.0,
        dt_s: float = 0.01,
        safety_margin_m: float = 0.0,
        conformal_margin: float = 0.0,
    ) -> None:
        self.max_linear_vel_mps = max_linear_vel_mps
        self.max_contact_force_n = max_contact_force_n
        self.max_kinetic_energy_j = max_kinetic_energy_j
        self.mass_kg = float(mass_kg)
        self.dt_s = float(dt_s)
        self.safety_margin_m = float(safety_margin_m)
        self.conformal_margin = float(conformal_margin)

    def evaluate_plan(
        self,
        waypoints: Sequence[Sequence[float]],
        obstacles: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        initial_position: Optional[Sequence[float]] = None,
        target_velocities: Optional[Sequence[float]] = None,
        contact_forces: Optional[Sequence[float]] = None,
        dt_s: Optional[float] = None,
        safety_margin_m: Optional[float] = None,
        conformal_margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Roll out candidate waypoint plan deterministically and check safety bounds."""
        if _contains_non_finite(waypoints) or _contains_non_finite(obstacles) or _contains_non_finite(initial_position):
            return {
                "allowed": False,
                "decision": "deny",
                "reason": "non_finite_value",
                "min_clearance_m": None,
                "peak_linear_vel_mps": 0.0,
                "peak_kinetic_energy_j": 0.0,
                "peak_contact_force_n": 0.0,
                "violations": [{"kind": "non_finite_value"}],
                "steps_simulated": 0,
            }

        step_dt = float(self.dt_s if dt_s is None else dt_s)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")

        eff_margin = float(self.safety_margin_m if safety_margin_m is None else safety_margin_m) + float(
            self.conformal_margin if conformal_margin is None else conformal_margin
        )
        obs_list = list(obstacles or ())

        points: List[Tuple[float, ...]] = []
        if initial_position is not None:
            points.append(tuple(float(v) for v in initial_position))
        for wp in waypoints:
            points.append(tuple(float(v) for v in wp))

        if not points:
            return {
                "allowed": False,
                "decision": "deny",
                "reason": "empty_waypoint_plan",
                "min_clearance_m": None,
                "peak_linear_vel_mps": 0.0,
                "peak_kinetic_energy_j": 0.0,
                "peak_contact_force_n": 0.0,
                "violations": [{"kind": "empty_waypoint_plan"}],
                "steps_simulated": 0,
            }

        dim = len(points[0])
        if any(len(p) != dim for p in points):
            return {
                "allowed": False,
                "decision": "deny",
                "reason": "waypoint_dimension_mismatch",
                "min_clearance_m": None,
                "peak_linear_vel_mps": 0.0,
                "peak_kinetic_energy_j": 0.0,
                "peak_contact_force_n": 0.0,
                "violations": [{"kind": "waypoint_dimension_mismatch"}],
                "steps_simulated": 0,
            }

        violations: List[Dict[str, Any]] = []
        min_clearance: Optional[float] = None
        peak_vel = 0.0
        peak_ke = 0.0
        peak_force = 0.0
        steps_simulated = 0

        def _check_point(pt: Tuple[float, ...], step_idx: int) -> None:
            nonlocal min_clearance
            for obs_idx, obs in enumerate(obs_list):
                dist = _signed_distance_to_obstacle(pt, obs)
                min_clearance = dist if min_clearance is None else min(min_clearance, dist)
                if dist <= 0.0 or dist < eff_margin:
                    violations.append(
                        {
                            "kind": "obstacle_collision",
                            "step": step_idx,
                            "obstacle_index": obs_idx,
                            "obstacle_name": obs.get("name", f"obstacle_{obs_idx}"),
                            "clearance_m": dist,
                            "required_margin_m": eff_margin,
                            "position": list(pt),
                        }
                    )

        try:
            _check_point(points[0], 0)
            steps_simulated += 1

            for seg_idx in range(len(points) - 1):
                p0 = points[seg_idx]
                p1 = points[seg_idx + 1]
                seg_dist = math.sqrt(sum((b - a) ** 2 for a, b in zip(p0, p1)))

                if target_velocities is not None and seg_idx < len(target_velocities):
                    seg_vel = abs(float(target_velocities[seg_idx]))
                else:
                    seg_vel = seg_dist / step_dt
                if not math.isfinite(seg_vel):
                    raise ValueError("non_finite_velocity")
                peak_vel = max(peak_vel, seg_vel)

                seg_ke = 0.5 * self.mass_kg * (seg_vel ** 2)
                peak_ke = max(peak_ke, seg_ke)

                if self.max_linear_vel_mps is not None and seg_vel > self.max_linear_vel_mps:
                    violations.append(
                        {
                            "kind": "linear_vel_exceeded",
                            "segment": seg_idx,
                            "linear_vel_mps": seg_vel,
                            "limit": self.max_linear_vel_mps,
                        }
                    )
                if self.max_kinetic_energy_j is not None and seg_ke > self.max_kinetic_energy_j:
                    violations.append(
                        {
                            "kind": "kinetic_energy_exceeded",
                            "segment": seg_idx,
                            "kinetic_energy_j": seg_ke,
                            "limit": self.max_kinetic_energy_j,
                        }
                    )

                if contact_forces is not None and seg_idx < len(contact_forces):
                    force_val = abs(float(contact_forces[seg_idx]))
                    if not math.isfinite(force_val):
                        raise ValueError("non_finite_force")
                    peak_force = max(peak_force, force_val)
                    if self.max_contact_force_n is not None and force_val > self.max_contact_force_n:
                        violations.append(
                            {
                                "kind": "contact_force_exceeded",
                                "segment": seg_idx,
                                "contact_force_n": force_val,
                                "limit": self.max_contact_force_n,
                            }
                        )

                substeps = max(1, min(200, int(math.ceil(seg_dist / max(0.02, eff_margin or 0.05)))))
                for s in range(1, substeps + 1):
                    alpha = s / substeps
                    interp = tuple(a + alpha * (b - a) for a, b in zip(p0, p1))
                    _check_point(interp, steps_simulated)
                    steps_simulated += 1
        except ValueError as exc:
            return {
                "allowed": False,
                "decision": "deny",
                "reason": str(exc),
                "min_clearance_m": min_clearance,
                "peak_linear_vel_mps": peak_vel,
                "peak_kinetic_energy_j": peak_ke,
                "peak_contact_force_n": peak_force,
                "violations": violations + [{"kind": str(exc)}],
                "steps_simulated": steps_simulated,
            }

        allowed = len(violations) == 0
        reason = "authorized" if allowed else str(violations[0]["kind"])
        return {
            "allowed": allowed,
            "decision": "allow" if allowed else "deny",
            "reason": reason,
            "min_clearance_m": min_clearance,
            "peak_linear_vel_mps": peak_vel,
            "peak_kinetic_energy_j": peak_ke,
            "peak_contact_force_n": peak_force,
            "violations": violations,
            "steps_simulated": steps_simulated,
        }


def _extract_frame_kinematics(frame: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract position, linear speed, contact force, and kinetic energy from a frame."""
    state = frame.get("state")
    u_filt = frame.get("u_filtered")

    position: Optional[Tuple[float, ...]] = None
    linear_vel_mps: Optional[float] = None
    contact_force_n: Optional[float] = None
    kinetic_energy_j: Optional[float] = None

    if isinstance(state, Mapping):
        raw_pos = state.get("position") or state.get("pos")
        if isinstance(raw_pos, (list, tuple)):
            position = tuple(float(v) for v in raw_pos)
        if state.get("linear_vel_mps") is not None:
            linear_vel_mps = abs(float(state["linear_vel_mps"]))
        elif isinstance(state.get("velocity"), (list, tuple)):
            linear_vel_mps = math.sqrt(sum(float(v) ** 2 for v in state["velocity"]))
        if state.get("contact_force_n") is not None:
            contact_force_n = abs(float(state["contact_force_n"]))
        if state.get("kinetic_energy_j") is not None:
            kinetic_energy_j = float(state["kinetic_energy_j"])

    if isinstance(u_filt, Mapping):
        if linear_vel_mps is None and u_filt.get("linear_vel_mps") is not None:
            linear_vel_mps = abs(float(u_filt["linear_vel_mps"]))
        if contact_force_n is None and u_filt.get("contact_force_n") is not None:
            contact_force_n = abs(float(u_filt["contact_force_n"]))
        if kinetic_energy_j is None and u_filt.get("kinetic_energy_j") is not None:
            kinetic_energy_j = float(u_filt["kinetic_energy_j"])
    elif isinstance(u_filt, (list, tuple)) and u_filt and linear_vel_mps is None:
        linear_vel_mps = math.sqrt(sum(float(v) ** 2 for v in u_filt))

    return {
        "position": position,
        "linear_vel_mps": linear_vel_mps,
        "contact_force_n": contact_force_n,
        "kinetic_energy_j": kinetic_energy_j,
    }


def replay_flight_log(
    log_path: Union[str, Path],
    hard_limits: Optional[Mapping[str, Any]] = None,
    conformal_margin: float = 0.0,
) -> Dict[str, Any]:
    """Verify hash-chain integrity and counterfactually replay a dumped JSONL flight log."""
    verification = verify_flight_log(log_path)
    margin = float(conformal_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("conformal_margin must be finite and non-negative")

    if not verification["verified"]:
        return {
            "verified_integrity": False,
            "integrity_reason": verification["reason"],
            "total_frames": verification["frame_count"],
            "would_intervene": 0,
            "violations": [],
            "conformal_margin": margin,
        }

    _, frames, _ = _load_flight_log(log_path)
    limits = dict(hard_limits or {})
    violations: List[Dict[str, Any]] = []
    intervene_ticks = set()

    min_slack_req = float(limits.get("min_barrier_slack", 0.0))
    max_vel = limits.get("max_linear_vel_mps")
    max_ke = limits.get("max_kinetic_energy_j")
    max_force = limits.get("max_contact_force_n")
    max_q_hat = limits.get("max_aci_q_hat")
    polytope = limits.get("geofence_polytope")
    obstacles = limits.get("obstacles")

    for frame in frames:
        tick = int(frame["tick"])
        eff_slack = float(frame["min_barrier_slack"]) - margin
        if eff_slack < min_slack_req:
            intervene_ticks.add(tick)
            violations.append(
                {
                    "tick": tick,
                    "kind": "barrier_slack_under_conformal_margin",
                    "effective_slack": eff_slack,
                    "required_slack": min_slack_req,
                }
            )

        if max_q_hat is not None and float(frame["aci_q_hat"]) > float(max_q_hat):
            intervene_ticks.add(tick)
            violations.append(
                {
                    "tick": tick,
                    "kind": "aci_q_hat_exceeded",
                    "aci_q_hat": float(frame["aci_q_hat"]),
                    "limit": float(max_q_hat),
                }
            )

        kin = _extract_frame_kinematics(frame)
        if max_vel is not None and kin["linear_vel_mps"] is not None:
            if kin["linear_vel_mps"] > float(max_vel):
                intervene_ticks.add(tick)
                violations.append(
                    {
                        "tick": tick,
                        "kind": "linear_vel_exceeded",
                        "linear_vel_mps": kin["linear_vel_mps"],
                        "limit": float(max_vel),
                    }
                )

        if max_ke is not None and kin["kinetic_energy_j"] is not None:
            if kin["kinetic_energy_j"] > float(max_ke):
                intervene_ticks.add(tick)
                violations.append(
                    {
                        "tick": tick,
                        "kind": "kinetic_energy_exceeded",
                        "kinetic_energy_j": kin["kinetic_energy_j"],
                        "limit": float(max_ke),
                    }
                )

        if max_force is not None and kin["contact_force_n"] is not None:
            if kin["contact_force_n"] > float(max_force):
                intervene_ticks.add(tick)
                violations.append(
                    {
                        "tick": tick,
                        "kind": "contact_force_exceeded",
                        "contact_force_n": kin["contact_force_n"],
                        "limit": float(max_force),
                    }
                )

        pos = kin["position"]
        if pos is not None and polytope is not None:
            for normal, offset in polytope:
                norm_vec = [float(a) for a in normal]
                if len(norm_vec) != len(pos):
                    intervene_ticks.add(tick)
                    violations.append({"tick": tick, "kind": "geofence_dimension_mismatch"})
                    break
                a_norm = math.sqrt(sum(a * a for a in norm_vec))
                dot = sum(a * p for a, p in zip(norm_vec, pos))
                if dot > float(offset) - margin * a_norm + 1e-12:
                    intervene_ticks.add(tick)
                    violations.append(
                        {
                            "tick": tick,
                            "kind": "geofence_violation",
                            "dot": dot,
                            "shrunk_offset": float(offset) - margin * a_norm,
                        }
                    )
                    break

        if pos is not None and obstacles is not None:
            for obs_idx, obs in enumerate(obstacles):
                dist = _signed_distance_to_obstacle(pos, obs)
                if dist <= 0.0 or dist < margin:
                    intervene_ticks.add(tick)
                    violations.append(
                        {
                            "tick": tick,
                            "kind": "obstacle_clearance_violation",
                            "obstacle_index": obs_idx,
                            "clearance_m": dist,
                            "conformal_margin": margin,
                        }
                    )

    return {
        "verified_integrity": True,
        "integrity_reason": verification["reason"],
        "total_frames": len(frames),
        "would_intervene": len(intervene_ticks),
        "violations": violations,
        "conformal_margin": margin,
    }


__all__ = [
    "CounterfactualPhysicsGate",
    "FlightDataRecorder",
    "replay_flight_log",
    "verify_flight_log",
]
