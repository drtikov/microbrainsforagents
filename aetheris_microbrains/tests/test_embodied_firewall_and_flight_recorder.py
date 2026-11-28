"""Unit tests for physical capability attenuation, EmbodiedActionFirewall, and FlightDataRecorder."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from pathlib import Path

from aetheris_microbrains.core.capability_tokens import (
    CapabilityCaveat,
    CapabilityToken,
    attenuate_capability,
    authorize_physical_action,
    mint_root_capability,
)
from aetheris_microbrains.core.flight_recorder import (
    CounterfactualPhysicsGate,
    FlightDataRecorder,
    replay_flight_log,
    verify_flight_log,
)
from aetheris_microbrains.hosts.ros2_adapter import (
    EmbodiedActionFirewall,
    evaluate_embodied_action,
)


ROOT_SECRET = b"aetheris-embodied-root-hmac-key-2026"


def test_capability_caveat_physical_attenuation_and_geofence():
    parent_caveat = CapabilityCaveat(
        max_linear_vel_mps=1.0,
        max_kinetic_energy_j=25.0,
        max_contact_force_n=40.0,
        allowed_actuators=("left_gripper", "right_gripper"),
        expires_at=2000.0,
    )
    parent_tok = mint_root_capability(ROOT_SECRET, subject="kitchen-coordinator", caveat=parent_caveat, now=1000.0)

    # Kitchen counter safe polytope: 0.0 <= x <= 0.8, -0.5 <= y <= 0.5, 0.0 <= z <= 1.2
    # Stovetop sits at x >= 1.0 (e.g., (1.2, 0.1, 0.85))
    kitchen_counter_polytope = (
        ((1.0, 0.0, 0.0), 0.8),
        ((-1.0, 0.0, 0.0), 0.0),
        ((0.0, 1.0, 0.0), 0.5),
        ((0.0, -1.0, 0.0), 0.5),
        ((0.0, 0.0, 1.0), 1.2),
        ((0.0, 0.0, -1.0), 0.0),
    )

    child_tok = attenuate_capability(
        parent_tok,
        CapabilityCaveat(
            max_linear_vel_mps=0.25,
            allowed_actuators=("right_gripper",),
            geofence_polytope=kitchen_counter_polytope,
        ),
    )

    # Parent token retains its original 1.0 m/s and left_gripper authority
    parent_ok = authorize_physical_action(
        parent_tok,
        ROOT_SECRET,
        position=(1.2, 0.1, 0.85),
        linear_vel_mps=0.9,
        actuator="left_gripper",
        now=1100.0,
    )
    assert parent_ok["authorized"] is True
    assert parent_ok["decision"] == "allow"

    # Child token succeeds inside counter geofence with right_gripper at <= 0.25 m/s
    child_ok = authorize_physical_action(
        child_tok,
        ROOT_SECRET,
        position=(0.45, 0.15, 0.85),
        linear_vel_mps=0.2,
        kinetic_energy_j=4.0,
        contact_force_n=12.0,
        actuator="right_gripper",
        now=1100.0,
    )
    assert child_ok["authorized"] is True
    assert child_ok["decision"] == "allow"
    assert child_ok["effective_limits"]["max_linear_vel_mps"] == 0.25
    assert child_ok["effective_limits"]["allowed_actuators"] == ["right_gripper"]

    # Rejects stovetop coordinate outside counter polytope
    stovetop_res = authorize_physical_action(
        child_tok,
        ROOT_SECRET,
        position=(1.2, 0.1, 0.85),
        linear_vel_mps=0.2,
        actuator="right_gripper",
        now=1100.0,
    )
    assert stovetop_res["authorized"] is False
    assert stovetop_res["decision"] == "deny"
    assert stovetop_res["reason"] == "geofence_violation"

    # Rejects excess speed (> 0.25 m/s even though parent allowed 1.0 m/s)
    speed_res = authorize_physical_action(
        child_tok,
        ROOT_SECRET,
        position=(0.45, 0.15, 0.85),
        linear_vel_mps=0.6,
        actuator="right_gripper",
        now=1100.0,
    )
    assert speed_res["authorized"] is False
    assert speed_res["reason"] == "linear_vel_exceeded"

    # Rejects left_gripper attenuated away in child
    actuator_res = authorize_physical_action(
        child_tok,
        ROOT_SECRET,
        position=(0.45, 0.15, 0.85),
        linear_vel_mps=0.2,
        actuator="left_gripper",
        now=1100.0,
    )
    assert actuator_res["authorized"] is False
    assert actuator_res["reason"] == "actuator_not_allowed:left_gripper"

    # Serialization round-trip preserves physical caveat HMAC integrity
    restored = CapabilityToken.from_dict(child_tok.to_dict())
    assert authorize_physical_action(
        restored,
        ROOT_SECRET,
        position=(0.45, 0.15, 0.85),
        linear_vel_mps=0.2,
        actuator="right_gripper",
        now=1100.0,
    )["authorized"] is True

    # Tampering with child geofence fails HMAC check
    tampered = replace(
        child_tok,
        caveats=(
            parent_caveat,
            CapabilityCaveat(max_linear_vel_mps=0.25, allowed_actuators=("right_gripper",)),
        ),
    )
    tampered_res = authorize_physical_action(
        tampered,
        ROOT_SECRET,
        position=(1.2, 0.1, 0.85),
        linear_vel_mps=0.2,
        actuator="right_gripper",
        now=1100.0,
    )
    assert tampered_res["authorized"] is False
    assert tampered_res["reason"] == "invalid_hmac_signature"

    # Non-physical caveats omit physical keys in to_dict() for backward compatibility
    plain_dict = CapabilityCaveat(allowed_tools=("read_file",)).to_dict()
    for key in (
        "max_linear_vel_mps",
        "max_kinetic_energy_j",
        "max_contact_force_n",
        "allowed_actuators",
        "geofence_polytope",
    ):
        assert key not in plain_dict


def test_embodied_action_firewall_protocols_and_fail_closed_nan_inf():
    fw = EmbodiedActionFirewall()
    tok = mint_root_capability(
        ROOT_SECRET,
        subject="field-robot",
        caveat=CapabilityCaveat(
            max_linear_vel_mps=0.8,
            max_contact_force_n=25.0,
            max_kinetic_energy_j=30.0,
            allowed_actuators=("mobile_base", "right_gripper", "arm", "flight_controller"),
            geofence_polytope=(((1.0, 0.0, 0.0), 5.0), ((-1.0, 0.0, 0.0), 5.0)),
        ),
    )
    hard_limits = {"max_linear_vel_mps": 1.0, "max_contact_force_n": 30.0}

    # 1. ROS 2 /cmd_vel
    cmd_vel = {
        "topic": "/cmd_vel",
        "linear": {"x": 0.3, "y": 0.4, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.2},
    }
    res_vel = fw.evaluate_embodied_action(cmd_vel, protocol="ros2", capability_token=tok, root_secret=ROOT_SECRET, hard_limits=hard_limits)
    assert res_vel["decision"] == "allow"
    assert abs(res_vel["normalized"]["linear_vel_mps"] - 0.5) < 1e-9
    assert res_vel["normalized"]["topic_or_action"] == "/cmd_vel"

    # 2. ROS 2 GripperCommand
    gripper_cmd = {
        "action": "GripperCommand",
        "actuator": "right_gripper",
        "command": {"position": 0.04, "max_effort": 15.0},
    }
    res_grip = evaluate_embodied_action(gripper_cmd, protocol="ros2", capability_token=tok, root_secret=ROOT_SECRET)
    assert res_grip["decision"] == "allow"
    assert res_grip["normalized"]["contact_force_n"] == 15.0
    assert res_grip["normalized"]["actuator"] == "right_gripper"

    # 3. ROS 2 FollowJointTrajectory (exceeds token max_contact_force_n=25)
    traj_cmd = {
        "action": "FollowJointTrajectory",
        "actuator": "arm",
        "trajectory": {
            "joint_names": ["j1", "j2"],
            "points": [
                {"positions": [0.1, 0.2], "velocities": [0.2, 0.2], "effort": [10.0, 12.0]},
                {"positions": [0.3, 0.4], "velocities": [0.3, 0.3], "effort": [18.0, 28.0]},
            ],
        },
    }
    res_traj = fw.evaluate_embodied_action(traj_cmd, protocol="ros2", capability_token=tok, root_secret=ROOT_SECRET)
    assert res_traj["decision"] == "deny"
    assert res_traj["reason"] == "contact_force_exceeded"
    assert res_traj["clamp_hints"]["max_contact_force_n"] == 25.0

    # 4. ROS 2 NavigateToPose
    nav_cmd = {
        "action": "NavigateToPose",
        "pose": {"position": {"x": 2.0, "y": 1.0, "z": 0.0}},
        "linear_vel_mps": 0.6,
    }
    res_nav = fw.evaluate_embodied_action(nav_cmd, protocol="ros2", capability_token=tok, root_secret=ROOT_SECRET)
    assert res_nav["decision"] == "allow"
    assert res_nav["normalized"]["target_position"] == (2.0, 1.0, 0.0)

    # 5. MoveIt2
    moveit_cmd = {
        "group_name": "arm",
        "pose_goal": {"position": [1.0, 0.5, 0.8]},
        "max_velocity_scaling_factor": 0.5,
        "reference_vel_mps": 1.2,
    }
    res_moveit = fw.evaluate_embodied_action(moveit_cmd, protocol="moveit2", capability_token=tok, root_secret=ROOT_SECRET)
    assert res_moveit["decision"] == "allow"
    assert abs(res_moveit["normalized"]["linear_vel_mps"] - 0.6) < 1e-9

    # 6. LeRobot
    lerobot_cmd = {
        "action": [0.4, 0.2, 0.7],
        "action_space": "cartesian",
        "actuator": "right_gripper",
        "linear_vel_mps": 0.3,
        "gripper_force": 10.0,
    }
    res_lerobot = fw.evaluate_embodied_action(lerobot_cmd, protocol="lerobot", capability_token=tok, root_secret=ROOT_SECRET)
    assert res_lerobot["decision"] == "allow"
    assert res_lerobot["normalized"]["target_position"] == (0.4, 0.2, 0.7)

    # 7. MAVLink SET_POSITION_TARGET_LOCAL_NED (exceeds hard_limits max_linear_vel_mps)
    mav_cmd = {
        "msg_type": "SET_POSITION_TARGET_LOCAL_NED",
        "x": 1.0,
        "y": 2.0,
        "z": -3.0,
        "vx": 1.2,
        "vy": 0.9,
        "vz": 0.0,
    }
    res_mav = fw.evaluate_embodied_action(mav_cmd, protocol="mavlink", capability_token=tok, root_secret=ROOT_SECRET, hard_limits=hard_limits)
    assert res_mav["decision"] == "deny"
    assert res_mav["reason"] == "hard_limit_linear_vel_exceeded"
    assert res_mav["clamp_hints"]["max_linear_vel_mps"] == 1.0

    # 8. NaN / Inf fail-closed rejection across payloads
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        bad_payload = {"topic": "/cmd_vel", "linear": {"x": bad_val, "y": 0.0, "z": 0.0}}
        res_bad = fw.evaluate_embodied_action(bad_payload, protocol="ros2", capability_token=tok, root_secret=ROOT_SECRET)
        assert res_bad["authorized"] is False
        assert res_bad["decision"] == "deny"
        assert res_bad["reason"] == "non_finite_value"


def test_flight_recorder_ring_hash_chain_tamper_and_replay(tmp_path):
    recorder = FlightDataRecorder(capacity=50)
    base_ns = 1_700_000_000_000_000_000

    # Record 120 frames at 1 kHz (1 ms = 1_000_000 ns) -> ring wraps to latest 50 frames (ticks 70..119)
    for tick in range(120):
        pos_x = 0.01 * tick
        vel = 0.25 if tick < 110 else 0.95
        slack = 0.12 if tick < 115 else 0.02
        recorder.record(
            tick=tick,
            timestamp_ns=base_ns + tick * 1_000_000,
            state={"position": [pos_x, 0.0, 0.5], "linear_vel_mps": vel},
            u_nominal=[vel, 0.0, 0.0],
            u_filtered={"linear_vel_mps": vel, "contact_force_n": 10.0},
            min_barrier_slack=slack,
            aci_q_hat=0.05,
            token_id="tok-kitchen-01",
            status="ok" if tick < 115 else "clamped",
        )

    assert len(recorder) == 50
    assert recorder.total_recorded == 120
    assert recorder.frames[0]["tick"] == 70
    assert recorder.frames[-1]["tick"] == 119

    log_path = tmp_path / "blackbox.jsonl"
    dump_info = recorder.trigger_dump(log_path, reason="velocity_spike")
    assert dump_info["frame_count"] == 50
    assert dump_info["first_tick"] == 70
    assert dump_info["last_tick"] == 119

    # Intact SHA-256 chain passes verification
    verified = verify_flight_log(log_path)
    assert verified["verified"] is True
    assert verified["frame_count"] == 50

    # Counterfactual replay with stricter hard_limits and conformal_margin detects interventions
    replay_res = replay_flight_log(
        log_path,
        hard_limits={"max_linear_vel_mps": 0.5, "min_barrier_slack": 0.0},
        conformal_margin=0.05,
    )
    assert replay_res["verified_integrity"] is True
    assert replay_res["total_frames"] == 50
    # Ticks 110..119 violate velocity 0.95 > 0.5; ticks 115..119 also violate slack (0.02 - 0.05 < 0)
    assert replay_res["would_intervene"] == 10
    assert len(replay_res["violations"]) == 15

    # Tamper detection 1: mutate a single frame value
    tampered_value_path = tmp_path / "tampered_value.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    frame_obj = json.loads(lines[10])
    frame_obj["min_barrier_slack"] = 0.99
    lines[10] = json.dumps(frame_obj, sort_keys=True, separators=(",", ":"))
    tampered_value_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_flight_log(tampered_value_path)["verified"] is False
    assert replay_flight_log(tampered_value_path)["verified_integrity"] is False

    # Tamper detection 2: delete a middle frame
    tampered_missing_path = tmp_path / "tampered_missing.jsonl"
    raw_lines = log_path.read_text(encoding="utf-8").splitlines()
    tampered_missing_path.write_text("\n".join(raw_lines[:5] + raw_lines[6:]) + "\n", encoding="utf-8")
    assert verify_flight_log(tampered_missing_path)["verified"] is False


def test_counterfactual_physics_gate_rollout():
    gate = CounterfactualPhysicsGate(
        max_linear_vel_mps=0.8,
        max_contact_force_n=20.0,
        max_kinetic_energy_j=10.0,
        mass_kg=2.0,
        dt_s=0.5,
        safety_margin_m=0.1,
    )
    obstacles = [
        {"type": "sphere", "name": "hot_kettle", "center": [0.5, 0.0, 0.5], "radius": 0.15},
        {"type": "box", "name": "stovetop", "min": [0.9, -0.3, 0.0], "max": [1.4, 0.3, 0.9]},
    ]

    # Safe detour around obstacles
    safe_plan = gate.evaluate_plan(
        waypoints=[[0.0, 0.4, 0.5], [0.4, 0.4, 0.5], [0.7, 0.4, 0.5]],
        obstacles=obstacles,
        initial_position=[0.0, 0.2, 0.5],
        target_velocities=[0.4, 0.4, 0.4],
        contact_forces=[5.0, 5.0, 5.0],
    )
    assert safe_plan["allowed"] is True
    assert safe_plan["decision"] == "allow"
    assert safe_plan["min_clearance_m"] is not None and safe_plan["min_clearance_m"] >= 0.1

    # Segment passes directly through the hot_kettle sphere obstacle even though endpoints are outside
    collision_plan = gate.evaluate_plan(
        waypoints=[[0.8, 0.0, 0.5]],
        obstacles=obstacles,
        initial_position=[0.0, 0.0, 0.5],
        target_velocities=[0.5],
    )
    assert collision_plan["allowed"] is False
    assert collision_plan["decision"] == "deny"
    assert collision_plan["reason"] == "obstacle_collision"

    # Box obstacle violation
    box_plan = gate.evaluate_plan(
        waypoints=[[1.0, 0.0, 0.5]],
        obstacles=obstacles,
        initial_position=[0.75, 0.0, 0.5],
        target_velocities=[0.3],
    )
    assert box_plan["allowed"] is False
    assert box_plan["reason"] == "obstacle_collision"


def test_core_security_modules_are_pure_stdlib_and_no_forbidden_strings():
    repo_root = Path(__file__).resolve().parents[2]
    pure_stdlib_files = [
        repo_root / "aetheris_microbrains" / "core" / "capability_tokens.py",
        repo_root / "aetheris_microbrains" / "hosts" / "ros2_adapter.py",
        repo_root / "aetheris_microbrains" / "core" / "flight_recorder.py",
    ]
    for fpath in pure_stdlib_files:
        src = fpath.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "numpy", f"numpy imported in {fpath}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] != "numpy", f"numpy imported in {fpath}"

    # Verify no forbidden placeholder labels exist in any owned file
    forbidden_pat = re.compile(r"\bpro" r"posal\s*\d+\b", re.IGNORECASE)
    owned_files = pure_stdlib_files + [Path(__file__).resolve()]
    for fpath in owned_files:
        src = fpath.read_text(encoding="utf-8")
        assert forbidden_pat.search(src) is None, f"forbidden label found in {fpath}"
