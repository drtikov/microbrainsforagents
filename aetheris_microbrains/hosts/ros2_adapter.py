"""Pure-stdlib embodied action firewall for ROS 2, MoveIt2, LeRobot, and MAVLink.

Normalizes heterogeneous embodied robot control payloads into a unified
kinematic/dynamic action descriptor, rejects non-finite (NaN/Inf) inputs
immediately, enforces deterministic hard safety limits, and integrates with
Macaroon-style physical capability tokens via ``authorize_physical_action``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from aetheris_microbrains.core.capability_tokens import (
    CapabilityToken,
    authorize_physical_action,
)
from aetheris_microbrains.core.embodied_threat_guard import (
    CrossModalSensorVerifier,
    SignedMissionEnvelope,
    VisualPromptInjectionGuard,
)


SUPPORTED_PROTOCOLS = frozenset({"ros2", "moveit2", "lerobot", "mavlink", "unitree"})


def _contains_non_finite(value: Any) -> bool:
    """Recursively check whether ``value`` contains any NaN or Inf numeric value."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_non_finite(k) or _contains_non_finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _to_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{field_name} must be finite")
    return num


def _extract_vec(raw: Any) -> Optional[Tuple[float, ...]]:
    """Extract a numeric vector from either ``{x, y, z}`` mapping or sequence."""
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        coords: List[float] = []
        for axis in ("x", "y", "z"):
            if axis in raw:
                coords.append(_to_finite_float(raw[axis], axis))
        if coords:
            return tuple(coords)
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return tuple(_to_finite_float(v, "vector_element") for v in raw)
    return None


def _vec_norm(vec: Optional[Tuple[float, ...]]) -> Optional[float]:
    if vec is None:
        return None
    return math.sqrt(sum(v * v for v in vec))


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _extract_position_from_mapping(mapping: Mapping[str, Any]) -> Optional[Tuple[float, ...]]:
    direct = _first_present(
        mapping,
        "target_position",
        "end_effector_pos",
        "eef_position",
        "xyz",
    )
    vec = _extract_vec(direct)
    if vec is not None:
        return vec

    for pose_key in ("pose", "goal_pose", "target_pose", "pose_goal"):
        pose = mapping.get(pose_key)
        if isinstance(pose, Mapping):
            pos = _extract_vec(_first_present(pose, "position", "pos"))
            if pos is not None:
                return pos
            pos = _extract_vec(pose)
            if pos is not None:
                return pos

    pos_val = mapping.get("position")
    if isinstance(pos_val, (Mapping, list, tuple)):
        return _extract_vec(pos_val)
    return None


def _extract_waypoints(raw_waypoints: Any) -> Tuple[Tuple[float, ...], ...]:
    if raw_waypoints is None:
        return ()
    if not isinstance(raw_waypoints, (list, tuple)):
        raise ValueError("waypoints must be a list or tuple")
    out: List[Tuple[float, ...]] = []
    for item in raw_waypoints:
        if isinstance(item, Mapping):
            vec = _extract_position_from_mapping(item)
            if vec is None:
                vec = _extract_vec(_first_present(item, "positions", "point", "target_position"))
            if vec is not None:
                out.append(vec)
        else:
            vec = _extract_vec(item)
            if vec is not None:
                out.append(vec)
    return tuple(out)


class EmbodiedActionFirewall:
    """Protocol-agnostic firewall normalizing and gating embodied robot commands."""

    def __init__(self) -> None:
        self._visual_guard = VisualPromptInjectionGuard()

    def normalize(self, payload: Mapping[str, Any], protocol: str = "ros2") -> Dict[str, Any]:
        """Normalize an embodied action payload into a canonical kinematics dict."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if _contains_non_finite(payload):
            raise ValueError("non_finite_value")

        proto = str(protocol or "").strip().lower()
        if proto not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported_protocol:{proto}")

        if proto == "ros2":
            return self._normalize_ros2(payload)
        if proto == "moveit2":
            return self._normalize_moveit2(payload)
        if proto == "lerobot":
            return self._normalize_lerobot(payload)
        if proto == "unitree":
            return self._normalize_unitree(payload)
        return self._normalize_mavlink(payload)

    def _finalize(
        self,
        *,
        protocol: str,
        topic_or_action: str,
        actuator: Optional[str],
        target_position: Optional[Tuple[float, ...]],
        linear_vel_mps: Optional[float],
        contact_force_n: Optional[float],
        kinetic_energy_j: Optional[float],
        waypoints: Tuple[Tuple[float, ...], ...],
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if kinetic_energy_j is None and linear_vel_mps is not None and payload.get("mass_kg") is not None:
            mass_kg = _to_finite_float(payload["mass_kg"], "mass_kg")
            if mass_kg < 0.0:
                raise ValueError("mass_kg must be non-negative")
            kinetic_energy_j = 0.5 * mass_kg * (linear_vel_mps ** 2)
        return {
            "protocol": protocol,
            "topic_or_action": str(topic_or_action),
            "actuator": str(actuator).strip() if actuator is not None else None,
            "target_position": target_position,
            "linear_vel_mps": linear_vel_mps,
            "contact_force_n": contact_force_n,
            "kinetic_energy_j": kinetic_energy_j,
            "waypoints": waypoints,
        }

    def _normalize_ros2(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        topic_or_action = str(
            _first_present(payload, "topic_or_action", "topic", "action", "message_type") or ""
        ).strip()

        # Unwrap optional twist envelope for /cmd_vel
        twist_src = payload.get("twist") if isinstance(payload.get("twist"), Mapping) else payload

        linear_vel = _first_present(payload, "linear_vel_mps", "velocity", "speed")
        linear_vel_mps: Optional[float] = (
            abs(_to_finite_float(linear_vel, "linear_vel_mps")) if linear_vel is not None else None
        )
        if linear_vel_mps is None and isinstance(twist_src, Mapping) and "linear" in twist_src:
            lin_vec = _extract_vec(twist_src["linear"])
            linear_vel_mps = _vec_norm(lin_vec)

        contact_force = _first_present(payload, "contact_force_n", "max_effort", "force", "effort")
        contact_force_n: Optional[float] = (
            abs(_to_finite_float(contact_force, "contact_force_n")) if contact_force is not None else None
        )

        ke_raw = payload.get("kinetic_energy_j")
        kinetic_energy_j = _to_finite_float(ke_raw, "kinetic_energy_j") if ke_raw is not None else None

        target_position = _extract_position_from_mapping(payload)
        waypoints = _extract_waypoints(payload.get("waypoints"))

        # Classify specific ROS 2 messages cleanly
        lower_topic = topic_or_action.lower()
        if (
            lower_topic in ("/cmd_vel", "cmd_vel", "twist")
            or (not topic_or_action and isinstance(twist_src, Mapping) and "linear" in twist_src)
        ):
            topic_or_action = topic_or_action or "/cmd_vel"
            actuator = _first_present(payload, "actuator") or "mobile_base"
        elif (
            "gripper" in lower_topic
            or isinstance(payload.get("command"), Mapping)
            and any(k in payload["command"] for k in ("max_effort", "position"))
        ):
            topic_or_action = topic_or_action or "GripperCommand"
            cmd = payload.get("command") if isinstance(payload.get("command"), Mapping) else payload
            if contact_force_n is None:
                eff = _first_present(cmd, "max_effort", "effort", "force", "contact_force_n")
                if eff is not None:
                    contact_force_n = abs(_to_finite_float(eff, "max_effort"))
            actuator = _first_present(payload, "actuator", "gripper_id") or "gripper"
        elif (
            "followjointtrajectory" in lower_topic
            or "trajectory" in payload
            or "points" in payload
            or "joint_names" in payload
        ):
            topic_or_action = topic_or_action or "FollowJointTrajectory"
            traj = payload.get("trajectory") if isinstance(payload.get("trajectory"), Mapping) else payload
            points = traj.get("points") if isinstance(traj.get("points"), (list, tuple)) else ()
            traj_waypoints: List[Tuple[float, ...]] = []
            peak_vel = linear_vel_mps
            peak_effort = contact_force_n
            for pt in points:
                if not isinstance(pt, Mapping):
                    continue
                pos_vec = _extract_vec(pt.get("positions"))
                if pos_vec is not None:
                    traj_waypoints.append(pos_vec)
                vel_vec = _extract_vec(pt.get("velocities"))
                if vel_vec is not None:
                    v_norm = _vec_norm(vel_vec)
                    if v_norm is not None:
                        peak_vel = v_norm if peak_vel is None else max(peak_vel, v_norm)
                eff_vec = _extract_vec(_first_present(pt, "effort", "efforts"))
                if eff_vec is not None:
                    max_e = max(abs(e) for e in eff_vec)
                    peak_effort = max_e if peak_effort is None else max(peak_effort, max_e)
            if traj_waypoints and not waypoints:
                waypoints = tuple(traj_waypoints)
            linear_vel_mps = peak_vel
            contact_force_n = peak_effort
            actuator = _first_present(payload, "actuator", "group_name") or "arm"
        elif "navigatetopose" in lower_topic or target_position is not None:
            topic_or_action = topic_or_action or "NavigateToPose"
            actuator = _first_present(payload, "actuator") or "mobile_base"
        else:
            topic_or_action = topic_or_action or "ros2_action"
            actuator = _first_present(payload, "actuator") or "robot"

        return self._finalize(
            protocol="ros2",
            topic_or_action=topic_or_action,
            actuator=actuator,
            target_position=target_position,
            linear_vel_mps=linear_vel_mps,
            contact_force_n=contact_force_n,
            kinetic_energy_j=kinetic_energy_j,
            waypoints=waypoints,
            payload=payload,
        )

    def _normalize_unitree(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        topic_or_action = str(
            _first_present(payload, "topic_or_action", "rpc", "method", "action", "topic") or "unitree_cmd"
        ).strip()
        actuator = _first_present(payload, "actuator", "robot_model") or (
            "dex3_hand" if "dex3" in topic_or_action.lower() else "unitree_body"
        )
        vx = payload.get("vx")
        vy = payload.get("vy")
        linear_vel_mps: Optional[float] = None
        if vx is not None or vy is not None:
            vx_f = _to_finite_float(vx or 0.0, "vx")
            vy_f = _to_finite_float(vy or 0.0, "vy")
            linear_vel_mps = math.hypot(vx_f, vy_f)
        elif payload.get("linear_vel_mps") is not None:
            linear_vel_mps = abs(_to_finite_float(payload["linear_vel_mps"], "linear_vel_mps"))

        contact_force = _first_present(payload, "contact_force_n", "max_effort", "tau")
        contact_force_n: Optional[float] = None
        if isinstance(contact_force, (list, tuple)):
            contact_force_n = max((abs(_to_finite_float(v, "tau")) for v in contact_force), default=0.0)
        elif contact_force is not None:
            contact_force_n = abs(_to_finite_float(contact_force, "contact_force_n"))

        motor_cmds = payload.get("motor_cmd")
        if isinstance(motor_cmds, (list, tuple)) and motor_cmds:
            max_tau = 0.0
            for mc in motor_cmds:
                if isinstance(mc, Mapping) and mc.get("tau") is not None:
                    max_tau = max(max_tau, abs(_to_finite_float(mc["tau"], "motor_cmd.tau")))
            if contact_force_n is None:
                contact_force_n = max_tau

        ke_raw = payload.get("kinetic_energy_j")
        kinetic_energy_j = _to_finite_float(ke_raw, "kinetic_energy_j") if ke_raw is not None else None
        target_position = _extract_position_from_mapping(payload)
        waypoints = _extract_waypoints(payload.get("waypoints"))
        return self._finalize(
            protocol="unitree",
            topic_or_action=topic_or_action,
            actuator=actuator,
            target_position=target_position,
            linear_vel_mps=linear_vel_mps,
            contact_force_n=contact_force_n,
            kinetic_energy_j=kinetic_energy_j,
            waypoints=waypoints,
            payload=payload,
        )

    def _normalize_moveit2(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        topic_or_action = str(
            _first_present(payload, "topic_or_action", "action", "topic") or "MoveGroup"
        ).strip()
        actuator = _first_present(payload, "actuator", "group_name", "planning_group") or "arm"
        target_position = _extract_position_from_mapping(payload)
        waypoints = _extract_waypoints(payload.get("waypoints"))
        if target_position is None and waypoints and len(waypoints[-1]) in (2, 3):
            target_position = waypoints[-1]

        vel_raw = _first_present(payload, "linear_vel_mps", "max_linear_vel_mps", "velocity", "speed")
        linear_vel_mps = abs(_to_finite_float(vel_raw, "linear_vel_mps")) if vel_raw is not None else None
        if linear_vel_mps is None and payload.get("max_velocity_scaling_factor") is not None:
            scale = _to_finite_float(payload["max_velocity_scaling_factor"], "max_velocity_scaling_factor")
            ref_vel = _to_finite_float(payload.get("reference_vel_mps", 1.0), "reference_vel_mps")
            linear_vel_mps = abs(scale * ref_vel)

        force_raw = _first_present(payload, "contact_force_n", "max_effort", "max_contact_force_n", "force")
        contact_force_n = abs(_to_finite_float(force_raw, "contact_force_n")) if force_raw is not None else None
        ke_raw = payload.get("kinetic_energy_j")
        kinetic_energy_j = _to_finite_float(ke_raw, "kinetic_energy_j") if ke_raw is not None else None

        return self._finalize(
            protocol="moveit2",
            topic_or_action=topic_or_action,
            actuator=actuator,
            target_position=target_position,
            linear_vel_mps=linear_vel_mps,
            contact_force_n=contact_force_n,
            kinetic_energy_j=kinetic_energy_j,
            waypoints=waypoints,
            payload=payload,
        )

    def _normalize_lerobot(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        topic_or_action = str(
            _first_present(payload, "topic_or_action", "policy_action", "task") or "lerobot_action"
        ).strip()
        actuator = _first_present(payload, "actuator", "arm", "robot_part") or "arm"

        raw_action = payload.get("action")
        action_map: Mapping[str, Any] = raw_action if isinstance(raw_action, Mapping) else {}

        target_position = _extract_position_from_mapping(payload)
        if target_position is None and action_map:
            target_position = _extract_position_from_mapping(action_map)

        waypoints = _extract_waypoints(_first_present(payload, "waypoints") or action_map.get("waypoints"))
        if not waypoints and isinstance(raw_action, (list, tuple)):
            action_vec = _extract_vec(raw_action)
            if action_vec is not None:
                waypoints = (action_vec,)
                if target_position is None:
                    space = str(payload.get("action_space", "")).strip().lower()
                    if len(action_vec) in (2, 3) or (space in ("cartesian", "eef", "xyz") and len(action_vec) >= 3):
                        target_position = action_vec[:3]

        vel_raw = _first_present(payload, "linear_vel_mps", "velocity", "speed")
        if vel_raw is None and action_map:
            vel_raw = _first_present(action_map, "linear_vel_mps", "velocity", "speed")
        if vel_raw is not None:
            if isinstance(vel_raw, (Mapping, list, tuple)):
                linear_vel_mps = _vec_norm(_extract_vec(vel_raw))
            else:
                linear_vel_mps = abs(_to_finite_float(vel_raw, "linear_vel_mps"))
        else:
            linear_vel_mps = None

        force_raw = _first_present(payload, "contact_force_n", "gripper_force", "force", "effort")
        if force_raw is None and action_map:
            force_raw = _first_present(action_map, "contact_force_n", "gripper_force", "force", "effort")
        contact_force_n = abs(_to_finite_float(force_raw, "contact_force_n")) if force_raw is not None else None

        ke_raw = _first_present(payload, "kinetic_energy_j")
        if ke_raw is None and action_map:
            ke_raw = action_map.get("kinetic_energy_j")
        kinetic_energy_j = _to_finite_float(ke_raw, "kinetic_energy_j") if ke_raw is not None else None

        return self._finalize(
            protocol="lerobot",
            topic_or_action=topic_or_action,
            actuator=actuator,
            target_position=target_position,
            linear_vel_mps=linear_vel_mps,
            contact_force_n=contact_force_n,
            kinetic_energy_j=kinetic_energy_j,
            waypoints=waypoints,
            payload=payload,
        )

    def _normalize_mavlink(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        topic_or_action = str(
            _first_present(payload, "msg_type", "message_type", "topic_or_action", "command")
            or "SET_POSITION_TARGET_LOCAL_NED"
        ).strip()
        actuator = _first_present(payload, "actuator", "component") or "flight_controller"

        target_position = _extract_position_from_mapping(payload)
        if target_position is None and any(k in payload for k in ("x", "y", "z")):
            coords: List[float] = []
            for axis in ("x", "y", "z"):
                if axis in payload and payload[axis] is not None:
                    coords.append(_to_finite_float(payload[axis], axis))
            if coords:
                target_position = tuple(coords)

        vel_raw = _first_present(payload, "linear_vel_mps", "velocity", "speed")
        if vel_raw is not None:
            linear_vel_mps: Optional[float] = abs(_to_finite_float(vel_raw, "linear_vel_mps"))
        elif any(k in payload for k in ("vx", "vy", "vz")):
            v_coords = [
                _to_finite_float(payload[k], k) if payload.get(k) is not None else 0.0
                for k in ("vx", "vy", "vz")
            ]
            linear_vel_mps = math.sqrt(sum(v * v for v in v_coords))
        else:
            linear_vel_mps = None

        force_raw = _first_present(payload, "contact_force_n", "thrust_n", "force")
        contact_force_n = abs(_to_finite_float(force_raw, "contact_force_n")) if force_raw is not None else None
        ke_raw = payload.get("kinetic_energy_j")
        kinetic_energy_j = _to_finite_float(ke_raw, "kinetic_energy_j") if ke_raw is not None else None
        waypoints = _extract_waypoints(payload.get("waypoints"))

        return self._finalize(
            protocol="mavlink",
            topic_or_action=topic_or_action,
            actuator=actuator,
            target_position=target_position,
            linear_vel_mps=linear_vel_mps,
            contact_force_n=contact_force_n,
            kinetic_energy_j=kinetic_energy_j,
            waypoints=waypoints,
            payload=payload,
        )

    def evaluate_embodied_action(
        self,
        payload: Mapping[str, Any],
        protocol: str = "ros2",
        capability_token: Optional[CapabilityToken] = None,
        root_secret: Optional[Union[bytes, str]] = None,
        hard_limits: Optional[Mapping[str, Any]] = None,
        mission_envelope: Optional[Union[SignedMissionEnvelope, Mapping[str, Any]]] = None,
        mission_secret: Optional[Union[bytes, str]] = None,
        sensor_observation: Optional[Mapping[str, Any]] = None,
        sensor_verifier: Optional[CrossModalSensorVerifier] = None,
        scene_text: Optional[Any] = None,
        carrying_payload_n: Optional[float] = None,
        current_position: Optional[Union[Sequence[float], Mapping[str, Any]]] = None,
        allow_acrobatic_stunts: bool = False,
        base_speed_mps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate, normalize, and authorize an embodied action payload."""
        if not isinstance(payload, Mapping):
            return {
                "authorized": False,
                "decision": "deny",
                "reason": "invalid_payload_type",
                "normalized": None,
                "clamp_hints": {},
            }

        if (
            _contains_non_finite(payload)
            or _contains_non_finite(carrying_payload_n)
            or _contains_non_finite(current_position)
        ):
            return {
                "authorized": False,
                "decision": "deny",
                "reason": "non_finite_value",
                "normalized": None,
                "clamp_hints": {},
            }

        try:
            normalized = self.normalize(payload, protocol=protocol)
        except (TypeError, ValueError) as exc:
            return {
                "authorized": False,
                "decision": "deny",
                "reason": str(exc) or "normalization_failed",
                "normalized": None,
                "clamp_hints": {},
            }

        clamp_hints: Dict[str, Any] = {}

        if normalized["protocol"] == "unitree":
            act_low = str(normalized["topic_or_action"]).strip().lower()
            if any(stunt in act_low for stunt in ("frontflip", "backflip", "jumpyaw", "zerotorque")) and not allow_acrobatic_stunts:
                return self._deny("unitree_acrobatic_stunt_blocked", normalized, clamp_hints)
            eff_base_speed = base_speed_mps if base_speed_mps is not None else payload.get("base_speed_mps")
            if ("standdown" in act_low or "zerotorque" in act_low) and eff_base_speed is not None:
                if abs(_to_finite_float(eff_base_speed, "base_speed_mps")) > 0.1:
                    return self._deny("unitree_mid_motion_stand_down_blocked", normalized, clamp_hints)

        eff_current_pos = current_position
        if sensor_observation is not None:
            verifier = sensor_verifier if sensor_verifier is not None else CrossModalSensorVerifier()
            sensor_res = verifier.verify_observation(sensor_observation)
            if not sensor_res["verified"]:
                clamp_hints["quarantine"] = True
                clamp_hints["recommended_controller"] = sensor_res.get(
                    "recommended_controller", "SimplexBackupController"
                )
                return {
                    "authorized": False,
                    "decision": "deny",
                    "reason": sensor_res["reason"],
                    "normalized": normalized,
                    "clamp_hints": clamp_hints,
                    "sensor_verification": sensor_res,
                }
            if eff_current_pos is None and sensor_res.get("reported_position") is not None:
                eff_current_pos = sensor_res["reported_position"]

        if hard_limits is not None:
            if not isinstance(hard_limits, Mapping) or _contains_non_finite(hard_limits):
                return {
                    "authorized": False,
                    "decision": "deny",
                    "reason": "invalid_hard_limits",
                    "normalized": normalized,
                    "clamp_hints": clamp_hints,
                }

            limit_vel = hard_limits.get("max_linear_vel_mps")
            if limit_vel is not None:
                max_v = _to_finite_float(limit_vel, "max_linear_vel_mps")
                if max_v < 0.0:
                    return self._deny("invalid_hard_limits", normalized, clamp_hints)
                if normalized["linear_vel_mps"] is not None and normalized["linear_vel_mps"] > max_v:
                    clamp_hints["max_linear_vel_mps"] = max_v
                    if normalized["linear_vel_mps"] > 0.0:
                        clamp_hints["velocity_scale"] = max_v / normalized["linear_vel_mps"]
                    return self._deny("hard_limit_linear_vel_exceeded", normalized, clamp_hints)

            limit_ke = hard_limits.get("max_kinetic_energy_j")
            if limit_ke is not None:
                max_ke = _to_finite_float(limit_ke, "max_kinetic_energy_j")
                if max_ke < 0.0:
                    return self._deny("invalid_hard_limits", normalized, clamp_hints)
                if normalized["kinetic_energy_j"] is not None and normalized["kinetic_energy_j"] > max_ke:
                    clamp_hints["max_kinetic_energy_j"] = max_ke
                    return self._deny("hard_limit_kinetic_energy_exceeded", normalized, clamp_hints)

            limit_force = hard_limits.get("max_contact_force_n")
            if limit_force is not None:
                max_f = _to_finite_float(limit_force, "max_contact_force_n")
                if max_f < 0.0:
                    return self._deny("invalid_hard_limits", normalized, clamp_hints)
                if normalized["contact_force_n"] is not None and normalized["contact_force_n"] > max_f:
                    clamp_hints["max_contact_force_n"] = max_f
                    return self._deny("hard_limit_contact_force_exceeded", normalized, clamp_hints)

            allowed_actuators = hard_limits.get("allowed_actuators")
            if allowed_actuators is not None and normalized["actuator"] is not None:
                allowed_set = {str(a).strip() for a in allowed_actuators}
                if normalized["actuator"] not in allowed_set:
                    clamp_hints["allowed_actuators"] = sorted(allowed_set)
                    return self._deny(
                        f"hard_limit_actuator_not_allowed:{normalized['actuator']}",
                        normalized,
                        clamp_hints,
                    )

            polytope = hard_limits.get("geofence_polytope")
            if polytope is not None and normalized["target_position"] is not None:
                pos = normalized["target_position"]
                for normal, offset in polytope:
                    if len(normal) != len(pos):
                        return self._deny("hard_limit_geofence_dimension_mismatch", normalized, clamp_hints)
                    dot = sum(float(a) * float(p) for a, p in zip(normal, pos))
                    if dot > float(offset) + 1e-12:
                        clamp_hints["geofence_violation"] = True
                        return self._deny("hard_limit_geofence_violation", normalized, clamp_hints)

        eff_scene_text = scene_text
        if eff_scene_text is None and isinstance(sensor_observation, Mapping):
            eff_scene_text = sensor_observation.get("scene_text")
        eff_load_n = carrying_payload_n
        if eff_load_n is None and isinstance(sensor_observation, Mapping):
            raw_obs_load = _first_present(sensor_observation, "carrying_payload_n", "load_cell_n")
            if isinstance(raw_obs_load, (int, float)) and not isinstance(raw_obs_load, bool):
                eff_load_n = float(raw_obs_load)

        if (
            mission_envelope is not None
            or eff_scene_text is not None
            or eff_load_n is not None
            or payload.get("scene_text") is not None
            or payload.get("carrying_payload") is not None
            or payload.get("carrying_payload_n") is not None
            or payload.get("load_cell_n") is not None
        ):
            obs_now = (
                sensor_observation.get("now_s")
                if isinstance(sensor_observation, Mapping) and isinstance(sensor_observation.get("now_s"), (int, float))
                else None
            )
            guard_res = self._visual_guard.evaluate(
                payload,
                normalized=normalized,
                mission_envelope=mission_envelope,
                mission_secret=mission_secret,
                scene_text=eff_scene_text,
                load_cell_n=eff_load_n,
                current_position=eff_current_pos,
                now=obs_now,
            )
            if not guard_res["authorized"]:
                clamp_hints.update(guard_res.get("clamp_hints") or {})
                return {
                    "authorized": False,
                    "decision": "deny",
                    "reason": guard_res["reason"],
                    "normalized": normalized,
                    "clamp_hints": clamp_hints,
                    "threat_guard": guard_res,
                }

        token_auth: Optional[Dict[str, Any]] = None
        if capability_token is not None:
            if root_secret is None:
                return self._deny("missing_root_secret", normalized, clamp_hints)
            token_auth = authorize_physical_action(
                capability_token,
                root_secret,
                position=normalized["target_position"],
                linear_vel_mps=normalized["linear_vel_mps"],
                kinetic_energy_j=normalized["kinetic_energy_j"],
                contact_force_n=normalized["contact_force_n"],
                actuator=normalized["actuator"],
            )
            if not token_auth["authorized"]:
                limits = token_auth.get("effective_limits") or {}
                if "max_linear_vel_mps" in limits and limits["max_linear_vel_mps"] is not None:
                    clamp_hints.setdefault("max_linear_vel_mps", limits["max_linear_vel_mps"])
                if "max_contact_force_n" in limits and limits["max_contact_force_n"] is not None:
                    clamp_hints.setdefault("max_contact_force_n", limits["max_contact_force_n"])
                if "max_kinetic_energy_j" in limits and limits["max_kinetic_energy_j"] is not None:
                    clamp_hints.setdefault("max_kinetic_energy_j", limits["max_kinetic_energy_j"])
                if "geofence" in str(token_auth.get("reason", "")):
                    clamp_hints["geofence_violation"] = True
                return {
                    "authorized": False,
                    "decision": "deny",
                    "reason": token_auth["reason"],
                    "normalized": normalized,
                    "clamp_hints": clamp_hints,
                    "token_auth": token_auth,
                }

        return {
            "authorized": True,
            "decision": "allow",
            "reason": "authorized",
            "normalized": normalized,
            "clamp_hints": clamp_hints,
            "token_auth": token_auth,
        }

    @staticmethod
    def _deny(reason: str, normalized: Dict[str, Any], clamp_hints: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "authorized": False,
            "decision": "deny",
            "reason": reason,
            "normalized": normalized,
            "clamp_hints": clamp_hints,
        }


_DEFAULT_FIREWALL = EmbodiedActionFirewall()


def evaluate_embodied_action(
    payload: Mapping[str, Any],
    protocol: str = "ros2",
    capability_token: Optional[CapabilityToken] = None,
    root_secret: Optional[Union[bytes, str]] = None,
    hard_limits: Optional[Mapping[str, Any]] = None,
    mission_envelope: Optional[Union[SignedMissionEnvelope, Mapping[str, Any]]] = None,
    mission_secret: Optional[Union[bytes, str]] = None,
    sensor_observation: Optional[Mapping[str, Any]] = None,
    sensor_verifier: Optional[CrossModalSensorVerifier] = None,
    scene_text: Optional[Any] = None,
    carrying_payload_n: Optional[float] = None,
    current_position: Optional[Union[Sequence[float], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Module-level convenience wrapper around ``EmbodiedActionFirewall.evaluate_embodied_action``."""
    return _DEFAULT_FIREWALL.evaluate_embodied_action(
        payload,
        protocol=protocol,
        capability_token=capability_token,
        root_secret=root_secret,
        hard_limits=hard_limits,
        mission_envelope=mission_envelope,
        mission_secret=mission_secret,
        sensor_observation=sensor_observation,
        sensor_verifier=sensor_verifier,
        scene_text=scene_text,
        carrying_payload_n=carrying_payload_n,
        current_position=current_position,
    )


__all__ = [
    "EmbodiedActionFirewall",
    "SUPPORTED_PROTOCOLS",
    "evaluate_embodied_action",
]
