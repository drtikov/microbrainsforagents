"""Tests for Counterfactual Trace Replay Simulator & Offline Journal-to-Policy Synthesizer."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from aetheris_microbrains.core.fleet_bundle import verify_signed_fleet_bundle
from aetheris_microbrains.core.trace_replay import (
    load_journal_entries,
    replay_decision_journal,
)


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_journal_entries_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    journal = _write_jsonl(
        tmp_path / "decisions.jsonl",
        [
            json.dumps({"tool": "execute_shell", "cmd": "git status", "decision": "allow"}),
            "",
            "   ",
            "{not valid json",
            json.dumps(["a", "list", "not", "a", "dict"]),
            json.dumps({"tool": "read_file", "path": "README.md", "decision": "allow"}),
        ],
    )

    entries = load_journal_entries(journal)
    assert len(entries) == 2
    assert entries[0]["tool"] == "execute_shell"
    assert entries[1]["tool"] == "read_file"

    summary = replay_decision_journal(journal, target_profile="strict")
    assert summary["total_entries"] == 2
    assert summary["replayed_entries"] == 2
    assert summary["skipped_lines"] == 2


def test_replay_decision_journal_dev_to_strict_flips(tmp_path: Path) -> None:
    # Under `dev`, `npm run build` is allowed; under `strict`, it escalates.
    # Conversely, `git status` and `pytest -q` stay allowed under both profiles.
    journal = _write_jsonl(
        tmp_path / "dev_journal.jsonl",
        [
            json.dumps({
                "tool": "execute_shell",
                "cmd": "git status",
                "decision": "allow",
                "reason_code": "layer1_structural_verified",
            }),
            json.dumps({
                "tool": "execute_shell",
                "cmd": "npm run build",
                "decision": "allow",
                "reason_code": "layer1_structural_verified",
            }),
            json.dumps({
                "raw_tool": "Write",
                "tool": "write_file",
                "path": "src/example.py",
                "diff": "x = 1\n",
                "decision": "escalate",
                "reason_code": "manual_review",
            }),
            json.dumps({
                "event_type": "hook_error",
                "decision": "error_passthrough",
                "reason_code": "hook_internal_error",
            }),
        ],
    )

    summary = replay_decision_journal(journal, target_profile="strict")

    assert summary["total_entries"] == 4
    assert summary["replayed_entries"] == 3
    assert summary["skipped_lines"] == 0
    assert summary["original_counts"]["allow"] == 2
    assert summary["original_counts"]["escalate"] == 1
    assert summary["replayed_counts"]["allow"] == 2
    assert summary["replayed_counts"]["escalate"] == 1
    assert summary["newly_blocked"] == 1
    assert summary["newly_allowed"] == 1
    assert summary["latency_mean_ms"] >= 0.0
    assert summary["latency_p95_ms"] >= summary["latency_mean_ms"] or summary["latency_p95_ms"] >= 0.0

    flips = summary["decision_flips"]
    assert len(flips) == 2

    npm_flip = next(f for f in flips if f["index"] == 1)
    assert npm_flip["tool"] == "execute_shell"
    assert npm_flip["original_decision"] == "allow"
    assert npm_flip["replayed_decision"] == "escalate"
    assert npm_flip["reason_code"]

    write_flip = next(f for f in flips if f["index"] == 2)
    assert write_flip["original_decision"] == "escalate"
    assert write_flip["replayed_decision"] == "allow"


def test_replay_includes_clopper_pearson_calibration(tmp_path: Path) -> None:
    lines = [
        json.dumps({
            "tool": "execute_shell",
            "cmd": "npm run build",
            "decision": "allow",
            "verified_correct": False,
        })
        for _ in range(4)
    ] + [
        json.dumps({
            "tool": "execute_shell",
            "cmd": "git status",
            "decision": "allow",
            "verified_correct": True,
        })
        for _ in range(2)
    ]
    journal = _write_jsonl(tmp_path / "cal_journal.jsonl", lines)

    summary = replay_decision_journal(journal, target_profile="strict")
    cal = summary["calibration"]

    assert isinstance(cal, dict)
    assert cal["calibrated_sample_count"] == 6
    assert 0.05 <= cal["recommended_threshold"] <= 0.95
    assert 0.0 <= cal["escalate_upper_bound"] <= 1.0
    assert 0.0 <= cal["block_upper_bound"] <= 1.0
    assert cal["labeled_sample_count"] == 6
    assert cal["false_positive_count"] == 4
    assert cal["fdr_upper_bound"] is not None
    assert cal["fdr_upper_bound"] > 0.5


def test_synthesize_policy_and_signed_fleet_bundle_verification(tmp_path: Path) -> None:
    journal = _write_jsonl(
        tmp_path / "synth_journal.jsonl",
        [
            json.dumps({"tool": "execute_shell", "cmd": "git status", "decision": "allow"}),
            json.dumps({"tool": "execute_shell", "cmd": "git status --short", "decision": "allow"}),
            json.dumps({"tool": "execute_shell", "cmd": "pytest -q tests/", "decision": "allow"}),
            json.dumps({"tool": "execute_shell", "cmd": "pytest -k smoke", "decision": "allow"}),
            # Dangerous / chained commands must NOT be synthesized even if labeled allow
            json.dumps({"tool": "execute_shell", "cmd": "rm -rf /tmp/foo", "decision": "allow"}),
            json.dumps({"tool": "execute_shell", "cmd": "git status && rm -rf /tmp/x", "decision": "allow"}),
        ],
    )

    hmac_key = b"super-secret-fleet-signing-key-2026"
    summary = replay_decision_journal(
        journal,
        target_profile="strict",
        synthesize_policy=True,
        org_id="acme-sec",
        bundle_version=7,
        hmac_key=hmac_key,
    )

    policy = summary["synthesized_policy"]
    assert isinstance(policy, dict)
    prefixes = [item["prefix"] for item in policy["recommended_command_prefixes"]]
    assert "git status" in prefixes
    assert "pytest" in prefixes
    assert all("rm" not in p for p in prefixes)
    assert len(policy["custom_rules"]) >= 2
    assert all(rule["action"] == "allow" for rule in policy["custom_rules"])

    bundle = summary["signed_bundle"]
    assert isinstance(bundle, dict)
    ver = verify_signed_fleet_bundle(bundle, signing_key=hmac_key, min_version=7)
    assert ver["valid"] is True
    assert ver["reason"] == "ok"
    assert ver["org_id"] == "acme-sec"
    assert ver["version"] == 7

    bad_ver = verify_signed_fleet_bundle(bundle, signing_key=b"wrong-key", min_version=7)
    assert bad_ver["valid"] is False
    assert bad_ver["reason"] == "invalid_signature"


def test_trace_replay_zero_numpy_pure_stdlib_imports() -> None:
    src_path = Path(__file__).resolve().parents[1] / "core" / "trace_replay.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported_roots.add(node.module.split(".")[0])

    forbidden = {"numpy", "scipy", "pandas", "torch", "sklearn", "yaml", "requests"}
    assert imported_roots.isdisjoint(forbidden), f"Forbidden third-party imports found: {imported_roots & forbidden}"
