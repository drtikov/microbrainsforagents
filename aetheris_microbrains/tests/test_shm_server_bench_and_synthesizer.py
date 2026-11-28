"""Tests for SharedMemorySensorimotorBus, robotics_policy_synthesizer, robotics_benchmark_harness, and /v1/robotics/* routes."""

from __future__ import annotations

import ast
import multiprocessing
import re
import struct
import threading
import time
from pathlib import Path

import pytest

from aetheris_microbrains.core.capability_tokens import CapabilityCaveat
from aetheris_microbrains.core.fleet_bundle import verify_signed_fleet_bundle
from aetheris_microbrains.core.flight_recorder import FlightDataRecorder
from aetheris_microbrains.core.robotics_benchmark_harness import run_robotics_benchmark_suite
from aetheris_microbrains.core.robotics_policy_synthesizer import (
    synthesize_safety_patch_from_flight_log,
)
from aetheris_microbrains.core.semantic_cbf_compiler import PhysicalSafetyIR
from aetheris_microbrains.core.shm_sensorimotor_bus import (
    _HEADER_SIZE,
    _SLOT_SEQ_STRUCT,
    SharedMemorySensorimotorBus,
)


def _child_shm_writer(bus_name: str, state: list, u_nom: list, u_filt: list) -> None:
    """Child process entrypoint for cross-process Seqlock write verification."""
    with SharedMemorySensorimotorBus(name=bus_name, create=False) as writer:
        writer.publish_frame(
            state=state,
            u_nominal=u_nom,
            u_filtered=u_filt,
            min_barrier_slack=0.125,
            tank_energy=1.75,
            watchdog_scale=0.92,
            timestamp_ns=1_700_000_123_456_789,
        )


def test_shm_sensorimotor_bus_dual_handle_cross_process_seqlock_and_unlink():
    bus_name = None
    writer = SharedMemorySensorimotorBus(state_dim=4, control_dim=2, capacity=4, create=True)
    try:
        bus_name = writer.name
        reader = SharedMemorySensorimotorBus(name=bus_name, create=False)
        try:
            assert reader.state_dim == 4
            assert reader.control_dim == 2
            assert reader.capacity == 4
            assert reader.read_latest() is None

            pub = writer.publish_frame(
                state=[0.1, -0.2, 0.3, -0.4],
                u_nominal=[1.5, -0.5],
                u_filtered=[0.8, -0.5],
                min_barrier_slack=0.042,
                tank_energy=0.875,
                watchdog_scale=0.75,
                timestamp_ns=1_700_000_000_000_001,
            )
            assert (pub["seq"] & 1) == 0

            got = reader.read_latest()
            assert got is not None
            assert got["seq"] == pub["seq"]
            assert (got["seq"] & 1) == 0
            assert got["timestamp_ns"] == 1_700_000_000_000_001
            assert got["state"] == pytest.approx([0.1, -0.2, 0.3, -0.4], abs=1e-12)
            assert got["u_nominal"] == pytest.approx([1.5, -0.5], abs=1e-12)
            assert got["u_filtered"] == pytest.approx([0.8, -0.5], abs=1e-12)
            assert got["min_barrier_slack"] == pytest.approx(0.042, abs=1e-12)
            assert got["tank_energy"] == pytest.approx(0.875, abs=1e-12)
            assert got["watchdog_scale"] == pytest.approx(0.75, abs=1e-12)

            # Verify Seqlock retry: mark active slot seq odd, release in background thread
            slot_off = _HEADER_SIZE + got["slot_index"] * writer.slot_size
            orig_seq = _SLOT_SEQ_STRUCT.unpack_from(writer._shm.buf, slot_off)[0]
            _SLOT_SEQ_STRUCT.pack_into(writer._shm.buf, slot_off, orig_seq + 1)

            def _finish_write():
                time.sleep(0.02)
                struct.pack_into("<Q", writer._shm.buf, slot_off, orig_seq + 2)

            t = threading.Thread(target=_finish_write)
            t.start()
            retried = reader.read_latest(max_retries=500_000)
            t.join()
            assert retried is not None
            assert retried["seq"] == orig_seq + 2
            assert (retried["seq"] & 1) == 0

            # Real cross-process publish via spawn context
            ctx = multiprocessing.get_context("spawn")
            proc = ctx.Process(
                target=_child_shm_writer,
                args=(bus_name, [1.25, 2.5, -3.75, 4.0], [2.2, -1.1], [0.9, -0.6]),
            )
            proc.start()
            proc.join(timeout=15.0)
            assert proc.exitcode == 0

            cross = reader.read_latest()
            assert cross is not None
            assert (cross["seq"] & 1) == 0
            assert cross["timestamp_ns"] == 1_700_000_123_456_789
            assert cross["state"] == pytest.approx([1.25, 2.5, -3.75, 4.0], abs=1e-12)
            assert cross["u_nominal"] == pytest.approx([2.2, -1.1], abs=1e-12)
            assert cross["u_filtered"] == pytest.approx([0.9, -0.6], abs=1e-12)
            assert cross["min_barrier_slack"] == pytest.approx(0.125, abs=1e-12)
            assert cross["tank_energy"] == pytest.approx(1.75, abs=1e-12)
            assert cross["watchdog_scale"] == pytest.approx(0.92, abs=1e-12)
        finally:
            reader.close()
    finally:
        writer.close()
        writer.unlink()

    if bus_name is not None:
        with pytest.raises(FileNotFoundError):
            SharedMemorySensorimotorBus(name=bus_name, create=False)


def test_synthesize_safety_patch_from_flight_log(tmp_path: Path):
    recorder = FlightDataRecorder(capacity=64)
    base_ns = 1_700_000_000_000_000_000

    # 30 nominal frames + 10 incident frames with speed spike, force spike, and negative/near-zero slack
    for tick in range(40):
        is_incident = tick >= 30
        pos = [0.1 * tick, 0.0, 0.5] if not is_incident else [1.45 + 0.02 * (tick - 30), -0.1, 0.5]
        vel = 0.25 if not is_incident else 1.25
        force = 8.0 if not is_incident else 48.0
        slack = 0.14 if not is_incident else (-0.02 if tick >= 35 else 0.01)
        q_hat = 0.04 if not is_incident else 0.09
        recorder.record(
            tick=tick,
            timestamp_ns=base_ns + tick * 1_000_000,
            state={
                "position": pos,
                "linear_vel_mps": vel,
                "contact_force_n": force,
                "kinetic_energy_j": 0.5 * 2.0 * (vel ** 2),
            },
            u_nominal=[vel, 0.0, 0.0],
            u_filtered={"linear_vel_mps": vel, "contact_force_n": force},
            min_barrier_slack=slack,
            aci_q_hat=q_hat,
            token_id="tok-bench-01",
            status="ok" if not is_incident else "unsafe_no_solution",
        )

    log_path = tmp_path / "incident_blackbox.jsonl"
    recorder.trigger_dump(log_path, reason="collision_near_miss")

    signing_key = b"aetheris-fleet-patch-signing-key-2026"
    patch = synthesize_safety_patch_from_flight_log(
        log_path,
        org_id="robotics-org",
        signing_key=signing_key,
        extra_safety_margin=0.05,
    )

    assert patch["verified_integrity"] is True
    assert patch["incident_frame_count"] == 10
    assert patch["total_frame_count"] == 40

    limits = patch["recommended_hard_limits"]
    assert 0.25 <= limits["max_linear_vel_mps"] < 1.25
    assert 8.0 <= limits["max_contact_force_n"] < 48.0
    assert limits["max_kinetic_energy_j"] < 0.5 * 2.0 * (1.25 ** 2)
    assert "min_barrier_slack" in limits

    rules = patch["synthesized_rules"]
    rule_types = [r["rule_type"] for r in rules]
    assert "speed_limit" in rule_types
    assert "force_limit" in rule_types
    assert "avoid" in rule_types

    # Compatible with PhysicalSafetyIR and CapabilityCaveat
    ir = PhysicalSafetyIR.from_dict({"rules": rules})
    assert len(ir.rules) == len(rules)
    caveat = CapabilityCaveat.from_dict(patch["attenuated_caveat"])
    assert caveat.max_linear_vel_mps == pytest.approx(limits["max_linear_vel_mps"])
    assert caveat.max_contact_force_n == pytest.approx(limits["max_contact_force_n"])

    # Verify HMAC-SHA256 signed FleetPolicyBundle
    bundle = patch["signed_bundle"]
    bundle_check = verify_signed_fleet_bundle(bundle, signing_key=signing_key)
    assert bundle_check["valid"] is True
    assert bundle_check["org_id"] == "robotics-org"

    # Counterfactual replay proves interception after patch synthesis
    cf = patch["counterfactual_verification"]
    assert cf["verified_integrity"] is True
    assert cf["would_intervene"] > 0
    assert cf["after_would_intervene"] > cf["before_would_intervene"]


def test_run_robotics_benchmark_suite_six_scenarios_zero_violations():
    report = run_robotics_benchmark_suite(use_native_rust=False)
    expected_scenarios = (
        "urdf_arm_moving_human",
        "stiff_wall_energy_tank",
        "payload_drop_l1_cbf",
        "swarm_3d_antipodal_swap",
        "visual_injection_midair_drop",
        "actuator_thermal_derating",
    )
    scenarios = report["scenarios"]
    assert set(scenarios.keys()) == set(expected_scenarios)

    for name in expected_scenarios:
        sc = scenarios[name]
        assert sc["passed"] is True, f"{name} failed: {sc}"
        assert sc["violations"] == 0
        assert sc["violation_rate_pct"] == 0.0
        assert sc["interventions"] > 0
        assert sc["latency_p50_us"] > 0.0
        assert sc["latency_p95_us"] >= sc["latency_p50_us"]
        assert sc["latency_p99_us"] >= sc["latency_p95_us"]

    summary = report["summary"]
    assert summary["total_scenarios"] == 6
    assert summary["passed_scenarios"] == 6
    assert summary["all_passed"] is True
    assert summary["total_violations"] == 0
    assert summary["overall_violation_rate_pct"] == 0.0


def test_microbrain_server_v1_robotics_routes(monkeypatch):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    from aetheris_microbrains.core.microbrain_server import app

    monkeypatch.delenv("AETHERIS_SERVER_TOKEN", raising=False)

    with TestClient(app) as client:
        # 1. GET /v1/robotics/status
        status_resp = client.get("/v1/robotics/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["ready"] is True
        assert status_data["status"] == "ready"
        assert set(status_data["supported_protocols"]) == {"ros2", "moveit2", "lerobot", "mavlink"}
        assert status_data["modules"]["shm_sensorimotor_bus"] is True
        assert status_data["modules"]["robotics_policy_synthesizer"] is True
        assert status_data["modules"]["robotics_benchmark_harness"] is True

        # 2. POST /v1/robotics/cbf-filter (linear halfplane + sphere barrier clamping)
        cbf_resp = client.post(
            "/v1/robotics/cbf-filter",
            json={
                "state": [0.95, 0.0],
                "nominal_action": [2.0, 0.0],
                "u_min": [-3.0, -3.0],
                "u_max": [3.0, 3.0],
                "barriers": [
                    {"type": "linear", "a": [-1.0, 0.0], "b": -1.0, "alpha": 1.0},
                    {"type": "sphere", "center": [2.0, 0.0], "radius": 0.5, "alpha": 0.5},
                ],
                "conformal_margin": 0.01,
            },
        )
        assert cbf_resp.status_code == 200
        cbf_data = cbf_resp.json()
        assert cbf_data["verified_safe"] is True
        assert cbf_data["n_barriers"] == 2
        assert cbf_data["u_filtered"][0] <= 0.05 + 1e-4

        # 3. POST /v1/robotics/vla-gate (calibrated safe step)
        vla_resp = client.post(
            "/v1/robotics/vla-gate",
            json={
                "scene": [0.1, 0.2, 0.3],
                "state": [0.0, 0.0],
                "nominal_action": [0.3, 0.0],
                "risk_score": 0.1,
                "calibration_scores": [0.05, 0.1, 0.15, 0.2, 0.25],
                "barriers": [{"type": "sphere", "center": [2.0, 0.0], "radius": 0.5, "alpha": 0.5}],
            },
        )
        assert vla_resp.status_code == 200
        vla_data = vla_resp.json()
        assert vla_data["decision"] in ("local_reflex", "call_vla")
        assert vla_data["safety"]["verified_safe"] is True

        # 4. POST /v1/robotics/embodied-firewall (allow nominal + block midair drop)
        fw_ok = client.post(
            "/v1/robotics/embodied-firewall",
            json={
                "protocol": "ros2",
                "payload": {
                    "topic": "/cmd_vel",
                    "linear": {"x": 0.3, "y": 0.4, "z": 0.0},
                },
                "hard_limits": {"max_linear_vel_mps": 0.8},
            },
        )
        assert fw_ok.status_code == 200
        assert fw_ok.json()["authorized"] is True
        assert fw_ok.json()["decision"] == "allow"

        fw_block = client.post(
            "/v1/robotics/embodied-firewall",
            json={
                "protocol": "lerobot",
                "payload": {
                    "policy_action": "open_gripper",
                    "action": [0.5, 0.0, 0.8],
                    "action_space": "cartesian",
                    "gripper_open": True,
                },
                "carrying_payload_n": 10.0,
                "current_position": [0.5, 0.0, 0.8],
            },
        )
        assert fw_block.status_code == 200
        assert fw_block.json()["authorized"] is False
        assert fw_block.json()["decision"] == "deny"
        assert fw_block.json()["reason"] == "unauthorized_midair_payload_drop"

        # 5. Optional bearer auth enforcement
        monkeypatch.setenv("AETHERIS_SERVER_TOKEN", "test-robotics-secret-token-12345")
        assert client.get("/v1/robotics/status").status_code == 401
        auth_ok = client.get(
            "/v1/robotics/status",
            headers={"Authorization": "Bearer test-robotics-secret-token-12345"},
        )
        assert auth_ok.status_code == 200


def test_pure_stdlib_modules_no_numpy_and_no_forbidden_strings():
    repo_root = Path(__file__).resolve().parents[2]
    pure_stdlib_files = [
        repo_root / "aetheris_microbrains" / "core" / "shm_sensorimotor_bus.py",
        repo_root / "aetheris_microbrains" / "core" / "robotics_policy_synthesizer.py",
    ]
    for fpath in pure_stdlib_files:
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0].lower() != "numpy", f"numpy imported in {fpath}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0].lower() != "numpy", f"numpy imported in {fpath}"

    forbidden_pat = re.compile(r"\bpro" r"posal\b", re.IGNORECASE)
    owned_files = pure_stdlib_files + [
        repo_root / "aetheris_microbrains" / "core" / "robotics_benchmark_harness.py",
        repo_root / "aetheris_microbrains" / "core" / "microbrain_server.py",
        Path(__file__).resolve(),
    ]
    for fpath in owned_files:
        assert forbidden_pat.search(fpath.read_text(encoding="utf-8")) is None, f"forbidden word in {fpath}"
