"""Pure-stdlib embodied threat guard for sensor spoofing, mission envelopes, and visual prompt injection.

Provides:
- ``CrossModalSensorVerifier``: Cross-modal timestamp monotonicity, staleness/skew,
  and encoder/IMU vs. visual/SLAM/GPS kinematic innovation gating with automatic
  ``SimplexBackupController`` quarantine recommendation on spoof/teleport detection.
- ``SignedMissionEnvelope``: Immutable HMAC-SHA256 signed mission intent contract
  with constant-time signature verification (``mint_mission_envelope``,
  ``verify_mission_envelope``).
- ``VisualPromptInjectionGuard``: Scene OCR / VLM caption / QR adversarial override
  scanner (including hidden Unicode/homoglyph smuggling) combined with
  latent-hijack-proof physical invariants (``unauthorized_midair_payload_drop`` and
  ``goal_hijack_outside_mission_envelope``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .mcp_security import detect_prompt_injection


_UNICODE_TAG_RE = re.compile(r"[\U000E0000-\U000E007F]")
_BIDI_OVERRIDE_RE = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u00AD\u2060\u180E]")
_VARIATION_SELECTOR_RE = re.compile(r"[\uFE00-\uFE0F\U000E0100-\U000E01EF]")
_HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "ё": "e", "о": "o", "р": "p", "с": "c", "х": "x", "і": "i", "у": "y", "к": "k",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I",
    "α": "a", "ε": "e", "ο": "o", "ρ": "p", "ν": "v", "κ": "k", "ι": "i", "τ": "t",
}
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)
_HOMOGLYPH_CHARS = frozenset(_HOMOGLYPH_MAP.keys())

_PHYSICAL_OVERRIDE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "physical_ignore_safety",
        re.compile(
            r"(?i)\b(?:ignore|bypass|disable|disregard|override|suspend)\s+"
            r"(?:all\s+)?(?:the\s+)?(?:safety|geofence|limits?|barriers?|constraints?|"
            r"guardrails?|envelope|interlocks?|instructions|rules|directives)\b"
        ),
    ),
    (
        "physical_system_override",
        re.compile(
            r"(?i)\b(?:system|admin|operator|root|supervisor|priority|security)\s+override\b|"
            r"\boverride\s+(?:mode|protocol|safety|mission)\b"
        ),
    ),
    (
        "physical_drop_payload",
        re.compile(
            r"(?i)\b(?:drop|release|jettison|dump|throw|unlatch)\s+"
            r"(?:the\s+|all\s+)?(?:payload|package|parcel|cargo|object|item|load|gripper)\b"
        ),
    ),
    (
        "physical_emergency_calibration",
        re.compile(
            r"(?i)\bemergency\s+(?:calibration|recalibration|override|unlock|release|bypass|diagnostic)\b|"
            r"\bcalibration\s+override\b"
        ),
    ),
    (
        "physical_move_to_coordinates",
        re.compile(
            r"(?i)\b(?:move|navigate|go|fly|drive|proceed|teleport|divert|reroute)\s+"
            r"(?:immediately\s+|now\s+)?to\s+(?:coordinates?|gps|pose|waypoint|xyz)\b"
        ),
    ),
)

_RELEASE_TOKENS = frozenset(
    {
        "gripper_open",
        "open_gripper",
        "release",
        "release_payload",
        "payload_release",
        "drop_payload",
        "payload_drop",
        "unlatch",
        "jettison",
    }
)


def _contains_non_finite(value: Any) -> bool:
    """Recursively return True if ``value`` contains any NaN or Inf number."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_non_finite(k) or _contains_non_finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _to_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{field_name} must be finite")
    return num


def _secret_to_bytes(secret: Union[bytes, str]) -> bytes:
    if isinstance(secret, (bytes, bytearray)):
        raw = bytes(secret)
    elif isinstance(secret, str):
        raw = secret.encode("utf-8")
    else:
        raise TypeError(f"secret must be bytes or str, got {type(secret).__name__}")
    if not raw:
        raise ValueError("secret must not be empty")
    return raw


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _extract_vec(raw: Any, field_name: str = "vector") -> Optional[Tuple[float, ...]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        if any(k in raw for k in ("x", "y", "z")):
            coords = [
                _to_finite_float(raw[axis], f"{field_name}.{axis}")
                for axis in ("x", "y", "z")
                if axis in raw and raw[axis] is not None
            ]
            return tuple(coords) if coords else None
        if any(k in raw for k in ("vx", "vy", "vz")):
            coords = [
                _to_finite_float(raw[axis], f"{field_name}.{axis}")
                for axis in ("vx", "vy", "vz")
                if axis in raw and raw[axis] is not None
            ]
            return tuple(coords) if coords else None
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return tuple(_to_finite_float(v, field_name) for v in raw)
    raise TypeError(f"{field_name} must be a sequence or coordinate mapping")


def _coerce_box(
    box: Sequence[Sequence[float]],
    field_name: str = "box",
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if not isinstance(box, (list, tuple)) or len(box) != 2:
        raise ValueError(f"{field_name} must be a (min_xyz, max_xyz) pair")
    min_raw, max_raw = box
    min_vec = _extract_vec(min_raw, f"{field_name}.min_xyz")
    max_vec = _extract_vec(max_raw, f"{field_name}.max_xyz")
    if min_vec is None or max_vec is None or len(min_vec) != len(max_vec):
        raise ValueError(f"{field_name} bounds must be non-empty equal-dimension vectors")
    for lo, hi in zip(min_vec, max_vec):
        if lo > hi:
            raise ValueError(f"{field_name} min_xyz ({lo}) must not exceed max_xyz ({hi})")
    return (min_vec, max_vec)


def _point_in_box(
    point: Sequence[float],
    box: Tuple[Tuple[float, ...], Tuple[float, ...]],
    tol: float = 1e-12,
) -> bool:
    min_vec, max_vec = box
    if len(point) != len(min_vec):
        return False
    return all((lo - tol) <= float(p) <= (hi + tol) for p, lo, hi in zip(point, min_vec, max_vec))


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


@dataclass(frozen=True)
class _VerifiedSensorState:
    timestamp_s: float
    depth_timestamp_s: Optional[float]
    proprioceptive_timestamp_s: Optional[float]
    position: Tuple[float, ...]
    encoder_vel: Tuple[float, ...]
    tick: Optional[int] = None


class CrossModalSensorVerifier:
    """Cross-modal sensor timestamp, skew, and kinematic innovation gate.

    Tracks the most recent verified ``(timestamp_s, position, encoder_vel)`` state
    per sensor/robot stream and vetoes non-finite readings, frozen frames,
    stale/skewed clocks, and visual/GPS/AprilTag position teleports inconsistent
    with proprioceptive dead-reckoning.
    """

    def __init__(
        self,
        *,
        max_staleness_s: float = 0.5,
        max_skew_s: float = 0.1,
        max_innovation_m: float = 0.25,
        innovation_vel_scale: float = 0.35,
    ) -> None:
        self.max_staleness_s = _to_finite_float(max_staleness_s, "max_staleness_s")
        self.max_skew_s = _to_finite_float(max_skew_s, "max_skew_s")
        self.max_innovation_m = _to_finite_float(max_innovation_m, "max_innovation_m")
        self.innovation_vel_scale = _to_finite_float(innovation_vel_scale, "innovation_vel_scale")
        if (
            self.max_staleness_s < 0.0
            or self.max_skew_s < 0.0
            or self.max_innovation_m < 0.0
            or self.innovation_vel_scale < 0.0
        ):
            raise ValueError("CrossModalSensorVerifier thresholds must be non-negative")
        self._streams: Dict[str, _VerifiedSensorState] = {}

    def reset_stream(self, stream_id: Optional[str] = None) -> None:
        """Reset one stream or all tracked sensor streams."""
        if stream_id is None:
            self._streams.clear()
        else:
            self._streams.pop(str(stream_id), None)

    def get_verified_state(self, stream_id: str = "default") -> Optional[Dict[str, Any]]:
        """Return a snapshot of the last verified state for ``stream_id``, if any."""
        st = self._streams.get(str(stream_id))
        if st is None:
            return None
        return {
            "timestamp_s": st.timestamp_s,
            "depth_timestamp_s": st.depth_timestamp_s,
            "proprioceptive_timestamp_s": st.proprioceptive_timestamp_s,
            "position": st.position,
            "encoder_vel": st.encoder_vel,
            "tick": st.tick,
        }

    @staticmethod
    def _veto(
        reason: str,
        stream_id: str,
        *,
        innovation_m: Optional[float] = None,
        innovation_bound_m: Optional[float] = None,
        predicted_position: Optional[Tuple[float, ...]] = None,
        reported_position: Optional[Tuple[float, ...]] = None,
        dt_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "verified": False,
            "authorized": False,
            "decision": "deny",
            "reason": reason,
            "stream_id": stream_id,
            "quarantine": True,
            "recommended_controller": "SimplexBackupController",
            "recommended_fallback": "SimplexBackupController",
            "innovation_m": innovation_m,
            "innovation_bound_m": innovation_bound_m,
            "predicted_position": predicted_position,
            "reported_position": reported_position,
            "dt_s": dt_s,
        }

    def verify(
        self,
        observation: Optional[Mapping[str, Any]] = None,
        *,
        visual_pos: Optional[Union[Sequence[float], Mapping[str, Any]]] = None,
        encoder_vel: Optional[Union[Sequence[float], Mapping[str, Any]]] = None,
        sensor_timestamp_s: Optional[float] = None,
        visual_timestamp_s: Optional[float] = None,
        depth_timestamp_s: Optional[float] = None,
        proprioceptive_timestamp_s: Optional[float] = None,
        now_s: Optional[float] = None,
        stream_id: str = "default",
        tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Verify a multi-modal sensor observation and update stream state if valid."""
        obs: Mapping[str, Any] = observation if isinstance(observation, Mapping) else {}
        if observation is not None and not isinstance(observation, Mapping):
            return self._veto("invalid_sensor_observation", str(stream_id))

        eff_stream = str(obs.get("stream_id") if obs.get("stream_id") is not None else stream_id).strip() or "default"

        # 1. Fail closed immediately on any non-finite input value.
        if (
            _contains_non_finite(obs)
            or _contains_non_finite(visual_pos)
            or _contains_non_finite(encoder_vel)
            or _contains_non_finite(
                (sensor_timestamp_s, visual_timestamp_s, depth_timestamp_s, proprioceptive_timestamp_s, now_s, tick)
            )
        ):
            return self._veto("sensor_non_finite", eff_stream)

        raw_depth_ts = depth_timestamp_s if depth_timestamp_s is not None else obs.get("depth_timestamp_s")
        raw_vis_ts = (
            visual_timestamp_s
            if visual_timestamp_s is not None
            else (
                sensor_timestamp_s
                if sensor_timestamp_s is not None
                else _first_present(
                    obs,
                    "visual_timestamp_s",
                    "sensor_timestamp_s",
                    "timestamp_s",
                    "vis_ts",
                    "depth_timestamp_s",
                )
            )
        )
        raw_prop_ts = (
            proprioceptive_timestamp_s
            if proprioceptive_timestamp_s is not None
            else _first_present(
                obs,
                "proprioceptive_timestamp_s",
                "prop_ts",
                "encoder_timestamp_s",
                "imu_timestamp_s",
            )
        )
        raw_now = now_s if now_s is not None else _first_present(obs, "now_s", "wall_timestamp_s")
        raw_tick = tick if tick is not None else obs.get("tick")

        try:
            depth_ts = _to_finite_float(raw_depth_ts, "depth_timestamp_s") if raw_depth_ts is not None else None
            vis_ts = _to_finite_float(raw_vis_ts, "sensor_timestamp_s") if raw_vis_ts is not None else depth_ts
            prop_ts = (
                _to_finite_float(raw_prop_ts, "proprioceptive_timestamp_s") if raw_prop_ts is not None else None
            )
            now_val = _to_finite_float(raw_now, "now_s") if raw_now is not None else None
            tick_val = int(raw_tick) if raw_tick is not None and not isinstance(raw_tick, bool) else None
        except (TypeError, ValueError):
            return self._veto("sensor_non_finite", eff_stream)

        if vis_ts is None:
            vis_ts = prop_ts
        if vis_ts is None:
            return self._veto("missing_sensor_timestamp", eff_stream)

        raw_pos = (
            visual_pos
            if visual_pos is not None
            else _first_present(
                obs,
                "visual_pos",
                "position",
                "slam_pos",
                "gps_pos",
                "apriltag_pos",
                "reported_position",
            )
        )
        raw_vel = (
            encoder_vel
            if encoder_vel is not None
            else _first_present(
                obs,
                "encoder_vel",
                "proprioceptive_vel",
                "imu_vel",
                "velocity",
            )
        )
        try:
            pos = _extract_vec(raw_pos, "visual_pos")
            vel = _extract_vec(raw_vel, "encoder_vel")
        except (TypeError, ValueError):
            return self._veto("invalid_sensor_vector", eff_stream)

        if pos is None:
            return self._veto("missing_sensor_position", eff_stream)
        if vel is not None and len(vel) != len(pos):
            return self._veto("sensor_dimension_mismatch", eff_stream, reported_position=pos)

        prev = self._streams.get(eff_stream)

        # 2. Monotonicity / frozen sensor frame check across consecutive ticks.
        if prev is not None:
            if tick_val is not None and prev.tick is not None and tick_val <= prev.tick:
                return self._veto("sensor_frozen_frame_veto", eff_stream, reported_position=pos)
            if vis_ts <= prev.timestamp_s:
                return self._veto("sensor_frozen_frame_veto", eff_stream, reported_position=pos)
            if depth_ts is not None and prev.depth_timestamp_s is not None and depth_ts <= prev.depth_timestamp_s:
                return self._veto("sensor_frozen_frame_veto", eff_stream, reported_position=pos)
            if (
                prop_ts is not None
                and prev.proprioceptive_timestamp_s is not None
                and prop_ts <= prev.proprioceptive_timestamp_s
            ):
                return self._veto("sensor_frozen_frame_veto", eff_stream, reported_position=pos)

        # 3. Inter-modality timestamp skew and wall-clock staleness check.
        if now_val is not None:
            if abs(now_val - vis_ts) > self.max_staleness_s:
                return self._veto("sensor_staleness_or_skew_veto", eff_stream, reported_position=pos)
            if depth_ts is not None and abs(now_val - depth_ts) > self.max_staleness_s:
                return self._veto("sensor_staleness_or_skew_veto", eff_stream, reported_position=pos)
            if prop_ts is not None and abs(now_val - prop_ts) > self.max_staleness_s:
                return self._veto("sensor_staleness_or_skew_veto", eff_stream, reported_position=pos)

        if prop_ts is not None and abs(vis_ts - prop_ts) > self.max_skew_s:
            return self._veto("sensor_staleness_or_skew_veto", eff_stream, reported_position=pos)
        if depth_ts is not None and prop_ts is not None and abs(depth_ts - prop_ts) > self.max_skew_s:
            return self._veto("sensor_staleness_or_skew_veto", eff_stream, reported_position=pos)
        if depth_ts is not None and abs(vis_ts - depth_ts) > self.max_skew_s:
            return self._veto("sensor_staleness_or_skew_veto", eff_stream, reported_position=pos)

        # 4. Cross-modal kinematic innovation gate vs. encoder/IMU dead-reckoning.
        eff_vel = vel if vel is not None else (prev.encoder_vel if prev is not None else tuple(0.0 for _ in pos))
        discrepancy: Optional[float] = 0.0
        bound: Optional[float] = self.max_innovation_m
        predicted_pos: Tuple[float, ...] = pos
        dt_s: Optional[float] = None

        if prev is not None:
            if len(pos) != len(prev.position) or len(eff_vel) != len(pos):
                return self._veto("sensor_dimension_mismatch", eff_stream, reported_position=pos)
            dt_s = vis_ts - prev.timestamp_s
            predicted_pos = tuple(p + v * dt_s for p, v in zip(prev.position, eff_vel))
            discrepancy = math.sqrt(sum((r - pred) ** 2 for r, pred in zip(pos, predicted_pos)))
            speed = math.sqrt(sum(v * v for v in eff_vel))
            bound = self.max_innovation_m + self.innovation_vel_scale * speed * dt_s
            if discrepancy > bound:
                return self._veto(
                    "sensor_teleport_or_spoof_detected",
                    eff_stream,
                    innovation_m=discrepancy,
                    innovation_bound_m=bound,
                    predicted_position=predicted_pos,
                    reported_position=pos,
                    dt_s=dt_s,
                )

        self._streams[eff_stream] = _VerifiedSensorState(
            timestamp_s=vis_ts,
            depth_timestamp_s=depth_ts,
            proprioceptive_timestamp_s=prop_ts,
            position=pos,
            encoder_vel=eff_vel,
            tick=tick_val,
        )
        return {
            "verified": True,
            "authorized": True,
            "decision": "allow",
            "reason": "verified",
            "stream_id": eff_stream,
            "quarantine": False,
            "recommended_controller": None,
            "recommended_fallback": None,
            "innovation_m": discrepancy,
            "innovation_bound_m": bound,
            "predicted_position": predicted_pos,
            "reported_position": pos,
            "dt_s": dt_s,
        }

    verify_observation = verify


@dataclass(frozen=True)
class SignedMissionEnvelope:
    """Immutable HMAC-SHA256 signed mission intent contract."""

    mission_id: str
    issued_at: float
    expires_at: float
    goal_bounds_box: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    allowed_release_boxes: Tuple[Tuple[Tuple[float, ...], Tuple[float, ...]], ...] = ()
    forbid_payload_drop_outside_zones: bool = True
    max_speed_mps: Optional[float] = None
    pinned_actuator_whitelist: Optional[Tuple[str, ...]] = None
    signature_hex: str = ""

    def __post_init__(self) -> None:
        clean_id = str(self.mission_id or "").strip()
        if not clean_id:
            raise ValueError("mission_id must not be empty")
        object.__setattr__(self, "mission_id", clean_id)
        object.__setattr__(self, "issued_at", _to_finite_float(self.issued_at, "issued_at"))
        object.__setattr__(self, "expires_at", _to_finite_float(self.expires_at, "expires_at"))
        if self.expires_at < self.issued_at:
            raise ValueError("expires_at must be >= issued_at")

        if self.goal_bounds_box is not None:
            object.__setattr__(self, "goal_bounds_box", _coerce_box(self.goal_bounds_box, "goal_bounds_box"))

        coerced_releases = tuple(
            _coerce_box(b, f"allowed_release_boxes[{idx}]")
            for idx, b in enumerate(self.allowed_release_boxes or ())
        )
        object.__setattr__(self, "allowed_release_boxes", coerced_releases)
        object.__setattr__(self, "forbid_payload_drop_outside_zones", bool(self.forbid_payload_drop_outside_zones))

        if self.max_speed_mps is not None:
            spd = _to_finite_float(self.max_speed_mps, "max_speed_mps")
            if spd < 0.0:
                raise ValueError("max_speed_mps must be non-negative")
            object.__setattr__(self, "max_speed_mps", spd)

        if self.pinned_actuator_whitelist is not None:
            if isinstance(self.pinned_actuator_whitelist, (str, bytes)):
                raise TypeError("pinned_actuator_whitelist must be an iterable of strings, not str")
            acts = tuple(str(a).strip() for a in self.pinned_actuator_whitelist if str(a).strip())
            object.__setattr__(self, "pinned_actuator_whitelist", acts)

        object.__setattr__(self, "signature_hex", str(self.signature_hex or "").strip().lower())

    def payload_dict(self) -> Dict[str, Any]:
        """Return canonical unsigned payload dictionary covered by HMAC-SHA256."""
        return {
            "mission_id": self.mission_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "goal_bounds_box": (
                [list(self.goal_bounds_box[0]), list(self.goal_bounds_box[1])]
                if self.goal_bounds_box is not None
                else None
            ),
            "allowed_release_boxes": [
                [list(lo), list(hi)] for lo, hi in self.allowed_release_boxes
            ],
            "forbid_payload_drop_outside_zones": self.forbid_payload_drop_outside_zones,
            "max_speed_mps": self.max_speed_mps,
            "pinned_actuator_whitelist": (
                list(self.pinned_actuator_whitelist)
                if self.pinned_actuator_whitelist is not None
                else None
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mission envelope including ``signature_hex``."""
        out = self.payload_dict()
        out["signature_hex"] = self.signature_hex
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignedMissionEnvelope":
        """Deserialize a ``SignedMissionEnvelope`` from a dictionary."""
        return cls(
            mission_id=str(data["mission_id"]),
            issued_at=float(data["issued_at"]),
            expires_at=float(data["expires_at"]),
            goal_bounds_box=data.get("goal_bounds_box"),
            allowed_release_boxes=tuple(data.get("allowed_release_boxes") or ()),
            forbid_payload_drop_outside_zones=bool(data.get("forbid_payload_drop_outside_zones", True)),
            max_speed_mps=data.get("max_speed_mps"),
            pinned_actuator_whitelist=(
                tuple(data["pinned_actuator_whitelist"])
                if data.get("pinned_actuator_whitelist") is not None
                else None
            ),
            signature_hex=str(data.get("signature_hex") or ""),
        )


def mint_mission_envelope(
    secret: Union[bytes, str],
    mission_id: Optional[str] = None,
    *,
    issued_at: Optional[float] = None,
    expires_at: Optional[float] = None,
    ttl_s: float = 3600.0,
    goal_bounds_box: Optional[Sequence[Sequence[float]]] = None,
    allowed_release_boxes: Sequence[Sequence[Sequence[float]]] = (),
    forbid_payload_drop_outside_zones: bool = True,
    max_speed_mps: Optional[float] = None,
    pinned_actuator_whitelist: Optional[Iterable[str]] = None,
) -> SignedMissionEnvelope:
    """Mint an immutable HMAC-SHA256 signed ``SignedMissionEnvelope``."""
    secret_bytes = _secret_to_bytes(secret)
    clean_id = str(mission_id).strip() if mission_id is not None else f"mission-{secrets.token_hex(8)}"
    issued = float(time.time() if issued_at is None else issued_at)
    expires = float(issued + float(ttl_s) if expires_at is None else expires_at)

    unsigned = SignedMissionEnvelope(
        mission_id=clean_id,
        issued_at=issued,
        expires_at=expires,
        goal_bounds_box=goal_bounds_box,
        allowed_release_boxes=tuple(allowed_release_boxes or ()),
        forbid_payload_drop_outside_zones=forbid_payload_drop_outside_zones,
        max_speed_mps=max_speed_mps,
        pinned_actuator_whitelist=tuple(pinned_actuator_whitelist) if pinned_actuator_whitelist is not None else None,
        signature_hex="",
    )
    sig_hex = hmac.new(secret_bytes, _canonical_json_bytes(unsigned.payload_dict()), hashlib.sha256).hexdigest()
    return SignedMissionEnvelope(
        mission_id=unsigned.mission_id,
        issued_at=unsigned.issued_at,
        expires_at=unsigned.expires_at,
        goal_bounds_box=unsigned.goal_bounds_box,
        allowed_release_boxes=unsigned.allowed_release_boxes,
        forbid_payload_drop_outside_zones=unsigned.forbid_payload_drop_outside_zones,
        max_speed_mps=unsigned.max_speed_mps,
        pinned_actuator_whitelist=unsigned.pinned_actuator_whitelist,
        signature_hex=sig_hex,
    )


def verify_mission_envelope(
    envelope: Union[SignedMissionEnvelope, Mapping[str, Any]],
    secret: Union[bytes, str],
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify ``SignedMissionEnvelope`` authenticity and freshness in constant time."""
    if isinstance(envelope, Mapping):
        try:
            env = SignedMissionEnvelope.from_dict(envelope)
        except (KeyError, TypeError, ValueError):
            return {
                "verified": False,
                "valid": False,
                "decision": "deny",
                "reason": "invalid_mission_envelope",
                "mission_id": None,
            }
    elif isinstance(envelope, SignedMissionEnvelope):
        env = envelope
    else:
        return {
            "verified": False,
            "valid": False,
            "decision": "deny",
            "reason": "invalid_mission_envelope",
            "mission_id": None,
        }

    try:
        secret_bytes = _secret_to_bytes(secret)
    except (TypeError, ValueError):
        return {
            "verified": False,
            "valid": False,
            "decision": "deny",
            "reason": "invalid_mission_secret",
            "mission_id": env.mission_id,
        }

    try:
        provided_sig = bytes.fromhex(env.signature_hex)
    except ValueError:
        return {
            "verified": False,
            "valid": False,
            "decision": "deny",
            "reason": "invalid_mission_signature",
            "mission_id": env.mission_id,
        }

    if len(provided_sig) != 32:
        return {
            "verified": False,
            "valid": False,
            "decision": "deny",
            "reason": "invalid_mission_signature",
            "mission_id": env.mission_id,
        }

    expected_sig = hmac.new(secret_bytes, _canonical_json_bytes(env.payload_dict()), hashlib.sha256).digest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        return {
            "verified": False,
            "valid": False,
            "decision": "deny",
            "reason": "invalid_mission_signature",
            "mission_id": env.mission_id,
        }

    current_time = float(time.time() if now is None else now)
    if not math.isfinite(current_time):
        return {
            "verified": False,
            "valid": False,
            "decision": "deny",
            "reason": "invalid_timestamp",
            "mission_id": env.mission_id,
        }

    if current_time > env.expires_at:
        return {
            "verified": False,
            "valid": False,
            "decision": "deny",
            "reason": "mission_envelope_expired",
            "mission_id": env.mission_id,
        }

    return {
        "verified": True,
        "valid": True,
        "decision": "allow",
        "reason": "verified",
        "mission_id": env.mission_id,
    }


def _canonicalize_visual_text(raw: str) -> str:
    stripped = _UNICODE_TAG_RE.sub("", raw)
    stripped = _BIDI_OVERRIDE_RE.sub("", stripped)
    stripped = _VARIATION_SELECTOR_RE.sub("", stripped)
    stripped = _ZERO_WIDTH_RE.sub("", stripped)
    norm = unicodedata.normalize("NFKC", stripped)
    return norm.translate(_HOMOGLYPH_TABLE)


def _flatten_scene_texts(scene_text: Any) -> List[str]:
    if scene_text is None:
        return []
    if isinstance(scene_text, str):
        return [scene_text] if scene_text else []
    if isinstance(scene_text, Mapping):
        out: List[str] = []
        for v in scene_text.values():
            out.extend(_flatten_scene_texts(v))
        return out
    if isinstance(scene_text, (list, tuple, set)):
        out = []
        for item in scene_text:
            out.extend(_flatten_scene_texts(item))
        return out
    return [str(scene_text)]


class VisualPromptInjectionGuard:
    """Visual prompt injection scanner and latent-hijack-proof mission invariant guard."""

    def __init__(
        self,
        *,
        open_threshold: float = 0.05,
        payload_threshold_n: float = 2.0,
    ) -> None:
        self.open_threshold = _to_finite_float(open_threshold, "open_threshold")
        self.payload_threshold_n = _to_finite_float(payload_threshold_n, "payload_threshold_n")
        if self.payload_threshold_n < 0.0:
            raise ValueError("payload_threshold_n must be non-negative")

    def scan_scene_text(self, scene_text: Any) -> Dict[str, Any]:
        """Scan OCR, VLM captions, and QR strings for physical override directives."""
        fragments = _flatten_scene_texts(scene_text)
        if not fragments:
            return {
                "safe": True,
                "injection_detected": False,
                "findings": [],
            }

        combined_raw = "\n".join(fragments)
        combined_canon = _canonicalize_visual_text(combined_raw)
        findings: List[str] = []

        has_hidden_unicode = bool(
            _UNICODE_TAG_RE.search(combined_raw)
            or _BIDI_OVERRIDE_RE.search(combined_raw)
            or _ZERO_WIDTH_RE.search(combined_raw)
            or _VARIATION_SELECTOR_RE.search(combined_raw)
        )
        has_mixed_homoglyph = any(ch in _HOMOGLYPH_CHARS for ch in combined_raw) and any(
            "a" <= ch.lower() <= "z" for ch in combined_raw
        )

        if _UNICODE_TAG_RE.search(combined_raw):
            findings.append("unicode_tag_smuggling")
        if _BIDI_OVERRIDE_RE.search(combined_raw):
            findings.append("unicode_bidi_override")
        if len(_ZERO_WIDTH_RE.findall(combined_raw)) >= 2:
            findings.append("zero_width_instruction_smuggling")

        for code, pat in _PHYSICAL_OVERRIDE_PATTERNS:
            if pat.search(combined_canon) or pat.search(combined_raw):
                findings.append(code)
                if has_hidden_unicode or has_mixed_homoglyph:
                    findings.append("homoglyph_or_zero_width_obfuscation")

        generic = detect_prompt_injection(combined_raw)
        if generic["injection_detected"]:
            findings.extend(generic["findings"])

        dedup = list(dict.fromkeys(findings))
        return {
            "safe": len(dedup) == 0,
            "injection_detected": len(dedup) > 0,
            "findings": dedup,
        }

    def _is_gripper_release_command(
        self,
        payload: Mapping[str, Any],
        normalized: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        norm = normalized or {}
        topic_or_action = str(
            norm.get("topic_or_action")
            or _first_present(payload, "topic_or_action", "action", "topic", "policy_action", "task", "msg_type")
            or ""
        ).strip().lower()
        actuator = str(norm.get("actuator") or _first_present(payload, "actuator", "gripper_id") or "").strip().lower()

        norm_token = re.sub(r"[^a-z0-9]+", "_", topic_or_action).strip("_")
        if norm_token in _RELEASE_TOKENS or any(tok in norm_token.split("_") for tok in ("release", "unlatch", "jettison")):
            return True
        if "gripper_open" in norm_token or "open_gripper" in norm_token or "drop_payload" in norm_token:
            return True

        for flag_key in ("gripper_open", "release", "open_gripper", "release_payload"):
            if payload.get(flag_key) is True:
                return True

        cmd_obj = payload.get("command")
        if isinstance(cmd_obj, str):
            cmd_norm = re.sub(r"[^a-z0-9]+", "_", cmd_obj.lower()).strip("_")
            if cmd_norm in _RELEASE_TOKENS or cmd_norm == "open":
                return True
        cmd_map: Mapping[str, Any] = cmd_obj if isinstance(cmd_obj, Mapping) else {}

        for state_src in (payload, cmd_map):
            for key in ("gripper_state", "gripper_action", "gripper_cmd", "state"):
                val = state_src.get(key)
                if isinstance(val, str) and val.strip().lower() in ("open", "release", "released", "drop", "unlatch"):
                    return True

        is_gripper_ctx = (
            "gripper" in topic_or_action
            or "gripper" in actuator
            or "latch" in actuator
            or "hand" in actuator
            or bool(cmd_map)
            or "gripper_position" in payload
            or "jaw_width" in payload
        )
        if is_gripper_ctx:
            for pos_src in (cmd_map, payload):
                for pos_key in ("gripper_position", "jaw_width", "width", "position"):
                    raw_p = pos_src.get(pos_key)
                    if isinstance(raw_p, (int, float)) and not isinstance(raw_p, bool):
                        if float(raw_p) > self.open_threshold:
                            return True

            for eff_src in (cmd_map, payload):
                for eff_key in ("max_effort", "gripper_force", "effort", "force"):
                    if eff_key in eff_src and eff_src[eff_key] is not None:
                        raw_e = eff_src[eff_key]
                        if isinstance(raw_e, (int, float)) and not isinstance(raw_e, bool) and float(raw_e) == 0.0:
                            return True

        return False

    @staticmethod
    def _is_joint_space_payload(payload: Mapping[str, Any]) -> bool:
        if "joint_names" in payload:
            return True
        traj = payload.get("trajectory")
        return isinstance(traj, Mapping) and "joint_names" in traj

    def evaluate(
        self,
        payload: Mapping[str, Any],
        *,
        normalized: Optional[Mapping[str, Any]] = None,
        mission_envelope: Optional[Union[SignedMissionEnvelope, Mapping[str, Any]]] = None,
        mission_secret: Optional[Union[bytes, str]] = None,
        scene_text: Any = None,
        load_cell_n: Optional[float] = None,
        carrying_payload: Optional[bool] = None,
        current_position: Optional[Union[Sequence[float], Mapping[str, Any]]] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate scene text and enforce latent-hijack behavioral invariants."""
        if _contains_non_finite((load_cell_n, current_position)):
            return {
                "authorized": False,
                "decision": "deny",
                "reason": "non_finite_value",
                "findings": ["non_finite_value"],
                "clamp_hints": {},
            }

        env: Optional[SignedMissionEnvelope] = None
        if mission_envelope is not None:
            if mission_secret is not None:
                ver = verify_mission_envelope(mission_envelope, mission_secret, now=now)
                if not ver["verified"]:
                    return {
                        "authorized": False,
                        "decision": "deny",
                        "reason": ver["reason"],
                        "findings": [ver["reason"]],
                        "clamp_hints": {},
                    }
                env = (
                    mission_envelope
                    if isinstance(mission_envelope, SignedMissionEnvelope)
                    else SignedMissionEnvelope.from_dict(mission_envelope)
                )
            elif isinstance(mission_envelope, SignedMissionEnvelope):
                if len(mission_envelope.signature_hex) != 64:
                    return {
                        "authorized": False,
                        "decision": "deny",
                        "reason": "invalid_mission_signature",
                        "findings": ["invalid_mission_signature"],
                        "clamp_hints": {},
                    }
                if now is not None and float(now) > mission_envelope.expires_at:
                    return {
                        "authorized": False,
                        "decision": "deny",
                        "reason": "mission_envelope_expired",
                        "findings": ["mission_envelope_expired"],
                        "clamp_hints": {},
                    }
                env = mission_envelope
            else:
                return {
                    "authorized": False,
                    "decision": "deny",
                    "reason": "missing_mission_secret",
                    "findings": ["missing_mission_secret"],
                    "clamp_hints": {},
                }

        eff_scene = scene_text if scene_text is not None else payload.get("scene_text")
        scan = self.scan_scene_text(eff_scene)
        if scan["injection_detected"]:
            return {
                "authorized": False,
                "decision": "deny",
                "reason": "visual_prompt_injection_detected",
                "findings": scan["findings"],
                "clamp_hints": {"quarantine_visual_stream": True},
            }

        eff_load_n = load_cell_n
        if eff_load_n is None:
            raw_load = _first_present(payload, "carrying_payload_n", "load_cell_n")
            if isinstance(raw_load, (int, float)) and not isinstance(raw_load, bool):
                eff_load_n = float(raw_load)

        eff_carrying = bool(carrying_payload) if carrying_payload is not None else bool(payload.get("carrying_payload", False))
        if eff_load_n is not None and float(eff_load_n) > self.payload_threshold_n:
            eff_carrying = True

        norm = normalized or {}
        target_pos = _extract_vec(norm.get("target_position"), "target_position") if norm.get("target_position") is not None else None
        curr_pos = _extract_vec(current_position, "current_position") if current_position is not None else None
        if curr_pos is None and payload.get("current_position") is not None:
            curr_pos = _extract_vec(payload.get("current_position"), "current_position")

        # Behavioral Invariant 1: Latent-hijack-proof midair / out-of-zone payload drop block.
        if eff_carrying and self._is_gripper_release_command(payload, norm):
            enforce_drop_zone = True if env is None else env.forbid_payload_drop_outside_zones
            if enforce_drop_zone:
                release_boxes = env.allowed_release_boxes if env is not None else ()
                positions_to_verify = [p for p in (curr_pos, target_pos) if p is not None]
                if not release_boxes or not positions_to_verify:
                    return {
                        "authorized": False,
                        "decision": "deny",
                        "reason": "unauthorized_midair_payload_drop",
                        "findings": ["unauthorized_midair_payload_drop"],
                        "clamp_hints": {"keep_gripper_closed": True},
                    }
                for pos in positions_to_verify:
                    if not any(_point_in_box(pos, box) for box in release_boxes):
                        return {
                            "authorized": False,
                            "decision": "deny",
                            "reason": "unauthorized_midair_payload_drop",
                            "findings": ["unauthorized_midair_payload_drop"],
                            "clamp_hints": {"keep_gripper_closed": True},
                        }

        # Behavioral Invariant 2: Goal bounds & speed/actuator enforcement under SignedMissionEnvelope.
        if env is not None:
            if env.goal_bounds_box is not None:
                box_dim = len(env.goal_bounds_box[0])
                if target_pos is not None:
                    if len(target_pos) != box_dim or not _point_in_box(target_pos, env.goal_bounds_box):
                        return {
                            "authorized": False,
                            "decision": "deny",
                            "reason": "goal_hijack_outside_mission_envelope",
                            "findings": ["goal_hijack_outside_mission_envelope"],
                            "clamp_hints": {
                                "goal_bounds_box": [list(env.goal_bounds_box[0]), list(env.goal_bounds_box[1])]
                            },
                        }

                raw_wps = norm.get("waypoints") or ()
                if raw_wps and not self._is_joint_space_payload(payload):
                    for wp in raw_wps:
                        wp_vec = _extract_vec(wp, "waypoint")
                        if wp_vec is not None and len(wp_vec) == box_dim:
                            if not _point_in_box(wp_vec, env.goal_bounds_box):
                                return {
                                    "authorized": False,
                                    "decision": "deny",
                                    "reason": "goal_hijack_outside_mission_envelope",
                                    "findings": ["goal_hijack_outside_mission_envelope"],
                                    "clamp_hints": {
                                        "goal_bounds_box": [
                                            list(env.goal_bounds_box[0]),
                                            list(env.goal_bounds_box[1]),
                                        ]
                                    },
                                }

            if env.max_speed_mps is not None and norm.get("linear_vel_mps") is not None:
                vel_mps = float(norm["linear_vel_mps"])
                if vel_mps > env.max_speed_mps:
                    return {
                        "authorized": False,
                        "decision": "deny",
                        "reason": "mission_envelope_speed_exceeded",
                        "findings": ["mission_envelope_speed_exceeded"],
                        "clamp_hints": {"max_linear_vel_mps": env.max_speed_mps},
                    }

            if env.pinned_actuator_whitelist is not None and norm.get("actuator") is not None:
                act = str(norm["actuator"]).strip()
                if act not in env.pinned_actuator_whitelist:
                    return {
                        "authorized": False,
                        "decision": "deny",
                        "reason": f"mission_envelope_actuator_not_allowed:{act}",
                        "findings": [f"mission_envelope_actuator_not_allowed:{act}"],
                        "clamp_hints": {"allowed_actuators": list(env.pinned_actuator_whitelist)},
                    }

        return {
            "authorized": True,
            "decision": "allow",
            "reason": "authorized",
            "findings": [],
            "clamp_hints": {},
        }


__all__ = [
    "CrossModalSensorVerifier",
    "SignedMissionEnvelope",
    "VisualPromptInjectionGuard",
    "mint_mission_envelope",
    "verify_mission_envelope",
]
