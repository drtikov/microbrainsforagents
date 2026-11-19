"""Tests for PolicyConfig.from_file, audit CLI, and payload preview logging."""

import json
from pathlib import Path
import pytest
from aetheris_microbrains import (
    PolicyConfig,
    decide,
    get_journal_entries,
    format_entry,
)
from aetheris_microbrains.audit import main as audit_main


def test_policy_config_from_json(tmp_path: Path):
    policy_file = tmp_path / "test_policy.json"
    data = {
        "require_approval_for_external": True,
        "strict_ast_allowlist": True,
        "allowed_ast_imports": ["math", "json"],
        "protected_patterns": [".env*", "production.yaml"],
    }
    policy_file.write_text(json.dumps(data), encoding="utf-8")

    cfg = PolicyConfig.from_file(policy_file)
    assert cfg.require_approval_for_external is True
    assert cfg.strict_ast_allowlist is True
    assert "math" in cfg.allowed_ast_imports
    assert "json" in cfg.allowed_ast_imports
    assert "production.yaml" in cfg.protected_patterns


def test_policy_config_from_yaml(tmp_path: Path):
    policy_file = tmp_path / "test_policy.yaml"
    yaml_content = """# Custom test policy
require_approval_for_external: true
strict_ast_allowlist: false
max_diff_lines: 350
allowed_ast_imports:
  - math
  - sys
protected_patterns:
  - ".env*"
  - "deploy/prod.json"
"""
    policy_file.write_text(yaml_content, encoding="utf-8")

    cfg = PolicyConfig.from_file(policy_file)
    assert cfg.require_approval_for_external is True
    assert cfg.strict_ast_allowlist is False
    assert cfg.max_diff_lines == 350
    assert "math" in cfg.allowed_ast_imports
    assert "deploy/prod.json" in cfg.protected_patterns


def test_decision_journal_payload_preview_and_audit(tmp_path: Path):
    journal_file = tmp_path / "decision_journal.jsonl"
    payload = {
        "tool": "execute_shell",
        "cmd": "echo 'hello world'",
    }
    ctx = {
        "log_decisions": True,
        "journal_path": str(journal_file),
    }

    res = decide(payload, context=ctx)
    assert res["decision"] == "allow"
    assert "payload_preview" in res
    assert "echo 'hello world'" in res["payload_preview"]
    assert res["journal_written"] is True

    # Test reading with get_journal_entries
    entries = get_journal_entries(journal_path=journal_file)
    assert len(entries) == 1
    assert entries[0]["reason_code"] == "layer1_structural_verified"
    assert "echo 'hello world'" in entries[0]["payload_preview"]

    # Test formatting
    formatted = format_entry(entries[0])
    assert "[ALLOW]" in formatted
    assert "Reason:" in formatted
    assert "echo 'hello world'" in formatted

    # Test CLI audit_main
    exit_code = audit_main(["--journal-path", str(journal_file), "--last"])
    assert exit_code == 0

    exit_code_summary = audit_main(["--journal-path", str(journal_file), "--summary"])
    assert exit_code_summary == 0
