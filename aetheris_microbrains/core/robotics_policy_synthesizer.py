"""Pure-stdlib safety policy synthesizer from black-box robotics flight logs.

Inspects hash-chained JSONL flight logs recorded by ``FlightDataRecorder`` and
synthesizes:
1. ``synthesized_rules``: ``PhysicalSafetyIR``-compatible rule dictionaries
   (``speed_limit``, ``force_limit``, and ``avoid`` obstacle spheres clustered
   around near-miss / violation coordinates).
2. ``recommended_hard_limits``: tightened kinematic/dynamic hard limits
   (``max_linear_vel_mps``, ``max_contact_force_n``, ``max_kinetic_energy_j``,
   ``min_barrier_slack``).
3. ``attenuated_caveat``: a ``CapabilityCaveat``-compatible physical caveat dict.
4. ``signed_bundle``: signed ``FleetPolicyBundle`` (via ``create_signed_fleet_bundle``
   when ``signing_key`` is provided, or an unsigned policy wrapper otherwise).
5. ``counterfactual_verification``: automatic ``replay_flight_log(...)`` run under
   the synthesized limits and conformal margin proving before-vs-after interception.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .capability_tokens import CapabilityCaveat
from .fleet_bundle import create_signed_fleet_bundle
from .flight_recorder import (
    _extract_frame_kinematics,
    _load_flight_log,
    replay_flight_log,
    verify_flight_log,
)

_INCIDENT_STATUS_TOKENS = (
    "clamp",
    "unsafe",
    "violat",
    "brake",
    "braking",
    "escalate",
    "deny",
    "anomaly",
    "near_miss",
    "collision",
    "fault",
)


def _is_incident_status(status: str) -> bool:
    low = str(status or "").strip().lower()
    return any(tok in low for tok in _INCIDENT_STATUS_TOKENS)


def _extract_full_kinematics(frame: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract kinematics from ``state``, ``u_filtered``, and ``u_nominal``."""
    kin = _extract_frame_kinematics(frame)
    pos = kin["position"]
    vel = kin["linear_vel_mps"]
    force = kin["contact_force_n"]
    ke = kin["kinetic_energy_j"]

    state = frame.get("state")
    if pos is None and isinstance(state, (list, tuple)) and len(state) >= 2:
        try:
            coords = tuple(float(v) for v in state[:3])
            if all(math.isfinite(c) for c in coords):
                pos = coords
        except (TypeError, ValueError):
            pass

    for vec_key in ("u_filtered", "u_nominal"):
        vec = frame.get(vec_key)
        if isinstance(vec, Mapping):
            if vel is None and vec.get("linear_vel_mps") is not None:
                vel = abs(float(vec["linear_vel_mps"]))
            if force is None and vec.get("contact_force_n") is not None:
                force = abs(float(vec["contact_force_n"]))
            if ke is None and vec.get("kinetic_energy_j") is not None:
                ke = float(vec["kinetic_energy_j"])
        elif isinstance(vec, (list, tuple)) and vec and vel is None:
            try:
                vel = math.sqrt(sum(float(v) ** 2 for v in vec))
            except (TypeError, ValueError):
                pass

    if ke is None and vel is not None:
        mass = 1.0
        if isinstance(state, Mapping) and state.get("mass_kg") is not None:
            mass = max(0.0, float(state["mass_kg"]))
        ke = 0.5 * mass * (vel ** 2)

    return {
        "position": pos,
        "linear_vel_mps": vel,
        "contact_force_n": force,
        "kinetic_energy_j": ke,
    }


def _cluster_violation_positions(
    points: Sequence[Tuple[float, ...]],
    cluster_radius: float,
    extra_margin: float,
) -> List[Dict[str, Any]]:
    """Cluster near-miss/violation coordinates into bounding obstacle spheres."""
    if not points:
        return []

    clusters: List[List[Tuple[float, ...]]] = []
    centroids: List[List[float]] = []

    for pt in points:
        dim = len(pt)
        best_idx = -1
        best_dist = float("inf")
        for idx, cent in enumerate(centroids):
            if len(cent) != dim:
                continue
            dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(pt, cent)))
            if dist <= cluster_radius and dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx >= 0:
            pts = clusters[best_idx]
            pts.append(pt)
            n = float(len(pts))
            centroids[best_idx] = [sum(p[d] for p in pts) / n for d in range(dim)]
        else:
            clusters.append([pt])
            centroids.append([float(v) for v in pt])

    spheres: List[Dict[str, Any]] = []
    for idx, (pts, cent) in enumerate(zip(clusters, centroids)):
        max_r = 0.0
        for pt in pts:
            d = math.sqrt(sum((p - c) ** 2 for p, c in zip(pt, cent)))
            if d > max_r:
                max_r = d
        radius = round(max(0.05, float(extra_margin), max_r + float(extra_margin)), 6)
        spheres.append(
            {
                "cluster_id": idx,
                "center": [round(c, 6) for c in cent],
                "radius": radius,
                "sample_count": len(pts),
            }
        )
    return spheres


def _tighten_upper_limit(
    all_vals: Sequence[float],
    incident_vals: Sequence[float],
    nominal_vals: Sequence[float],
    *,
    default_val: float,
    floor_val: float,
    extra_margin: float,
) -> float:
    """Compute a tightened upper limit separating nominal operation from incident spikes."""
    if not all_vals:
        return round(default_val, 6)

    scale = max(0.55, 1.0 - min(0.35, float(extra_margin)))
    if incident_vals and nominal_vals:
        nom_max = max(nominal_vals)
        spikes = [v for v in incident_vals if v > nom_max + 1e-9]
        if spikes:
            min_spike = min(spikes)
            # Place limit cleanly between nominal peak and lowest incident spike
            tight = nom_max + 0.35 * (min_spike - nom_max)
            return round(max(floor_val, min(tight, min_spike * 0.95)), 6)
        inc_min = min(incident_vals)
        return round(max(floor_val, inc_min * scale), 6)

    if incident_vals:
        inc_min = min(incident_vals)
        return round(max(floor_val, inc_min * scale), 6)

    return round(max(floor_val, max(all_vals) * scale), 6)


def synthesize_safety_patch_from_flight_log(
    log_path: Union[str, Path],
    org_id: str = "local-org",
    signing_key: Optional[Union[bytes, str]] = None,
    extra_safety_margin: float = 0.05,
) -> Dict[str, Any]:
    """Synthesize PhysicalSafetyIR rules, hard limits, caveat, and signed bundle from a flight log."""
    margin_f = float(extra_safety_margin)
    if not math.isfinite(margin_f) or margin_f < 0.0:
        raise ValueError("extra_safety_margin must be finite and non-negative")

    verification = verify_flight_log(log_path)
    if not verification.get("verified"):
        return {
            "verified_integrity": False,
            "integrity_reason": verification.get("reason", "invalid_flight_log"),
            "synthesized_rules": [],
            "recommended_hard_limits": {},
            "recommended_conformal_margin": margin_f,
            "attenuated_caveat": {},
            "signed_bundle": None,
            "counterfactual_verification": {
                "verified_integrity": False,
                "integrity_reason": verification.get("reason", "invalid_flight_log"),
                "would_intervene": 0,
            },
        }

    header, frames, _ = _load_flight_log(log_path)
    if not frames:
        raise ValueError("Cannot synthesize safety patch from an empty flight log")

    slacks: List[float] = []
    q_hats: List[float] = []
    all_vels: List[float] = []
    all_forces: List[float] = []
    all_kes: List[float] = []

    frame_records: List[Dict[str, Any]] = []
    for frame in frames:
        slack = float(frame["min_barrier_slack"])
        q_hat = float(frame["aci_q_hat"])
        status = str(frame.get("status") or "")
        kin = _extract_full_kinematics(frame)

        slacks.append(slack)
        q_hats.append(q_hat)
        if kin["linear_vel_mps"] is not None:
            all_vels.append(float(kin["linear_vel_mps"]))
        if kin["contact_force_n"] is not None:
            all_forces.append(float(kin["contact_force_n"]))
        if kin["kinetic_energy_j"] is not None:
            all_kes.append(float(kin["kinetic_energy_j"]))

        frame_records.append(
            {
                "tick": int(frame["tick"]),
                "slack": slack,
                "q_hat": q_hat,
                "status": status,
                "kin": kin,
            }
        )

    med_vel = sorted(all_vels)[len(all_vels) // 2] if all_vels else 0.0
    med_force = sorted(all_forces)[len(all_forces) // 2] if all_forces else 0.0
    med_q = sorted(q_hats)[len(q_hats) // 2] if q_hats else 0.0

    incident_records: List[Dict[str, Any]] = []
    nominal_records: List[Dict[str, Any]] = []

    for rec in frame_records:
        vel = rec["kin"]["linear_vel_mps"]
        force = rec["kin"]["contact_force_n"]
        near_miss_slack = rec["slack"] <= max(margin_f, 0.02)
        high_q = rec["q_hat"] > max(0.15, 1.8 * med_q) if med_q > 0.0 else rec["q_hat"] > 0.15
        vel_spike = vel is not None and med_vel > 0.0 and vel > 1.5 * med_vel
        force_spike = force is not None and med_force > 0.0 and force > 1.5 * med_force

        if near_miss_slack or high_q or vel_spike or force_spike or _is_incident_status(rec["status"]):
            incident_records.append(rec)
        else:
            nominal_records.append(rec)

    inc_vels = [r["kin"]["linear_vel_mps"] for r in incident_records if r["kin"]["linear_vel_mps"] is not None]
    nom_vels = [r["kin"]["linear_vel_mps"] for r in nominal_records if r["kin"]["linear_vel_mps"] is not None]
    inc_forces = [r["kin"]["contact_force_n"] for r in incident_records if r["kin"]["contact_force_n"] is not None]
    nom_forces = [r["kin"]["contact_force_n"] for r in nominal_records if r["kin"]["contact_force_n"] is not None]
    inc_kes = [r["kin"]["kinetic_energy_j"] for r in incident_records if r["kin"]["kinetic_energy_j"] is not None]
    nom_kes = [r["kin"]["kinetic_energy_j"] for r in nominal_records if r["kin"]["kinetic_energy_j"] is not None]

    max_vel_limit = _tighten_upper_limit(
        all_vels,
        inc_vels,
        nom_vels,
        default_val=0.5,
        floor_val=0.05,
        extra_margin=margin_f,
    )
    max_force_limit = _tighten_upper_limit(
        all_forces,
        inc_forces,
        nom_forces,
        default_val=25.0,
        floor_val=0.5,
        extra_margin=margin_f,
    )
    max_ke_limit = _tighten_upper_limit(
        all_kes,
        inc_kes,
        nom_kes,
        default_val=max(0.5, 0.5 * (max_vel_limit ** 2)),
        floor_val=0.05,
        extra_margin=margin_f,
    )

    peak_q_hat = max(q_hats) if q_hats else 0.0
    if incident_records:
        min_inc_slack = min(r["slack"] for r in incident_records)
        nom_slacks = [r["slack"] for r in nominal_records]
        if nom_slacks and min(nom_slacks) > max(0.0, min_inc_slack):
            # Target effective slack above incident slack while preserving nominal frames when possible
            target_total_slack = max(
                margin_f,
                min_inc_slack + max(margin_f, 0.02),
            )
            if peak_q_hat > 0.0:
                recommended_margin = round(max(margin_f, min(peak_q_hat, target_total_slack)), 6)
            else:
                recommended_margin = round(margin_f, 6)
            min_slack_limit = round(max(0.0, target_total_slack - recommended_margin + 0.01), 6)
        else:
            recommended_margin = round(max(margin_f, peak_q_hat), 6)
            min_slack_limit = round(max(0.0, margin_f), 6)
    else:
        recommended_margin = round(max(margin_f, peak_q_hat), 6)
        min_slack_limit = round(max(0.0, margin_f), 6)

    violation_positions: List[Tuple[float, ...]] = [
        r["kin"]["position"]
        for r in incident_records
        if r["kin"]["position"] is not None
    ]
    obstacle_clusters = _cluster_violation_positions(
        violation_positions,
        cluster_radius=max(0.2, 3.0 * margin_f),
        extra_margin=margin_f,
    )

    recommended_hard_limits: Dict[str, Any] = {
        "max_linear_vel_mps": max_vel_limit,
        "max_contact_force_n": max_force_limit,
        "max_kinetic_energy_j": max_ke_limit,
        "min_barrier_slack": min_slack_limit,
    }
    if obstacle_clusters:
        recommended_hard_limits["obstacles"] = [
            {
                "type": "sphere",
                "name": f"incident_zone_{c['cluster_id']}",
                "center": c["center"],
                "radius": c["radius"],
            }
            for c in obstacle_clusters
        ]

    synthesized_rules: List[Dict[str, Any]] = [
        {
            "rule_type": "speed_limit",
            "entity": None,
            "params": {"max_speed_mps": max_vel_limit},
            "alpha": 0.5,
            "hard": True,
            "raw_spec": f"speed_limit(max_speed_mps={max_vel_limit})",
        },
        {
            "rule_type": "force_limit",
            "entity": None,
            "params": {"max_force_n": max_force_limit},
            "alpha": 0.5,
            "hard": True,
            "raw_spec": f"force_limit(max_force_n={max_force_limit})",
        },
    ]
    for c in obstacle_clusters:
        synthesized_rules.append(
            {
                "rule_type": "avoid",
                "entity": None,
                "params": {
                    "center": c["center"],
                    "radius": c["radius"],
                    "label": f"incident_zone_{c['cluster_id']}",
                    "sample_count": c["sample_count"],
                },
                "alpha": 0.5,
                "hard": True,
                "raw_spec": f"avoid(center={c['center']}, radius={c['radius']})",
            }
        )

    caveat = CapabilityCaveat(
        max_linear_vel_mps=max_vel_limit,
        max_contact_force_n=max_force_limit,
        max_kinetic_energy_j=max_ke_limit,
    )
    attenuated_caveat = caveat.to_dict()

    before_replay = replay_flight_log(log_path, hard_limits={}, conformal_margin=0.0)
    after_replay = replay_flight_log(
        log_path,
        hard_limits=recommended_hard_limits,
        conformal_margin=recommended_margin,
    )
    counterfactual_verification: Dict[str, Any] = dict(after_replay)
    counterfactual_verification["before_would_intervene"] = int(before_replay.get("would_intervene", 0))
    counterfactual_verification["after_would_intervene"] = int(after_replay.get("would_intervene", 0))
    counterfactual_verification["interception_delta"] = (
        counterfactual_verification["after_would_intervene"]
        - counterfactual_verification["before_would_intervene"]
    )

    policy_payload: Dict[str, Any] = {
        "kind": "robotics_safety_patch",
        "source_log_head_hash": verification.get("head_hash"),
        "source_reason": (header or {}).get("reason"),
        "incident_frame_count": len(incident_records),
        "total_frame_count": len(frames),
        "recommended_hard_limits": recommended_hard_limits,
        "recommended_conformal_margin": recommended_margin,
        "synthesized_rules": synthesized_rules,
        "attenuated_caveat": attenuated_caveat,
    }

    if signing_key is not None:
        signed_bundle: Dict[str, Any] = create_signed_fleet_bundle(
            org_id=org_id,
            version=1,
            policy=policy_payload,
            signing_key=signing_key,
        )
    else:
        signed_bundle = {
            "schema_version": 1,
            "org_id": str(org_id),
            "version": 1,
            "signed": False,
            "policy": policy_payload,
        }

    return {
        "verified_integrity": True,
        "source_log_head_hash": verification.get("head_hash"),
        "incident_frame_count": len(incident_records),
        "total_frame_count": len(frames),
        "recommended_conformal_margin": recommended_margin,
        "synthesized_rules": synthesized_rules,
        "recommended_hard_limits": recommended_hard_limits,
        "attenuated_caveat": attenuated_caveat,
        "signed_bundle": signed_bundle,
        "counterfactual_verification": counterfactual_verification,
    }


__all__ = ["synthesize_safety_patch_from_flight_log"]
