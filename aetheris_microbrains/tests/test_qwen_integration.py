"""End-to-end tests for native Qwen Code support, .qwen/ protection, installer merge, and self-test."""

from __future__ import annotations

import json
from pathlib import Path

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.layered_decision_engine import PolicyConfig, decide, is_protected_path
from aetheris_microbrains.hook import (
    evaluate_hook_payload,
    evaluate_post_tool_use_payload,
)


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_qwen_control_plane_protected_by_default():
    pol = PolicyConfig(profile="dev")
    assert is_protected_path(".qwen/settings.json", pol) is True
    assert is_protected_path(".qwen/settings.local.json", pol) is True
    assert is_protected_path(".qwen/hooks/guard.sh", pol) is True

    write_v = decide({"tool": "write_file", "path": ".qwen/settings.json", "diff": "{}"}, policy=pol)
    assert write_v["decision"] == "deny"

    read_v = decide({"tool": "read_file", "path": ".qwen/settings.json"}, policy=pol)
    assert read_v["decision"] == "escalate"


def test_qwen_pre_and_post_hook_contract():
    allow_v = evaluate_hook_payload(
        {"tool_name": "run_shell_command", "tool_input": {"command": "git status"}},
        host="qwen",
        profile="dev",
        log_decisions=False,
    )
    assert allow_v["host"] == "qwen"
    assert allow_v["host_contract"] == "qwen_pretooluse_v1"
    assert allow_v["decision"] == "allow"
    assert allow_v["exit_code"] == 0

    deny_v = evaluate_hook_payload(
        {"tool_name": "run_shell_command", "tool_input": {"command": "rm -rf /"}},
        host="qwen",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["decision"] == "deny"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["permissionDecision"] == "deny"

    ask_v = evaluate_hook_payload(
        {"tool_name": "read_file", "tool_input": {"file_path": ".env"}},
        host="qwen",
        profile="dev",
        log_decisions=False,
    )
    assert ask_v["decision"] == "escalate"
    assert ask_v["hookSpecificOutput"]["permissionDecision"] == "ask"

    bad_py = evaluate_hook_payload(
        {"tool_name": "write_file", "tool_input": {"file_path": "app.py", "content": "def broken(:\n"}},
        host="qwen",
        profile="dev",
        log_decisions=False,
    )
    assert bad_py["decision"] == "deny"
    assert bad_py["exit_code"] == 2

    bad_edit = evaluate_hook_payload(
        {"tool_name": "edit", "tool_input": {"file_path": "app.py", "old_string": "x = 1", "new_string": "import os\nos.system('id')"}},
        host="qwen",
        profile="dev",
        log_decisions=False,
    )
    assert bad_edit["decision"] == "deny"
    assert bad_edit["exit_code"] == 2

    for benign_name in ("list_directory", "find_files"):
        v = evaluate_hook_payload(
            {"tool_name": benign_name, "tool_input": {"path": "aetheris_microbrains"}},
            host="qwen",
            profile="dev",
            log_decisions=False,
        )
        assert v["decision"] == "allow", benign_name

    # PostToolUse DLP secret scrubbing emits additionalContext without unsupported updatedToolOutput
    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {"tool_name": "run_shell_command", "tool_input": {"command": "env"}, "tool_output": f"API_KEY={fake_tok}"},
        host="qwen",
        replace_tool_output=True,
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert fake_tok not in post_v["redacted_output"]
    assert "SECURITY WARNING" in post_v["hookSpecificOutput"]["additionalContext"]
    assert "updatedToolOutput" not in post_v["hookSpecificOutput"]


def test_qwen_installer_merge_detection_and_self_test(tmp_path: Path, capsys):
    qwen_dir = tmp_path / ".qwen"
    qwen_dir.mkdir(parents=True)
    assert "qwen" in detect_installed_hosts(tmp_path)

    # Workspace install writes machine-local .qwen/settings.local.json and preserves existing hooks & settings
    shared_settings = qwen_dir / "settings.json"
    shared_settings.write_text(json.dumps({"shared": True}), encoding="utf-8")

    local_settings = qwen_dir / "settings.local.json"
    local_settings.write_text(
        json.dumps({
            "theme": "dark",
            "model": "qwen-coder",
            "hooks": {
                "PreToolUse": [
                    {"matcher": "run_shell_command", "hooks": [{"type": "command", "command": "echo custom-pre"}]}
                ],
                "Stop": [{"hooks": [{"type": "command", "command": "echo custom-stop"}]}],
            },
        }),
        encoding="utf-8",
    )

    assert cli_main(["install", "qwen", "--workspace", str(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()
    assert len(json.loads(local_settings.read_text(encoding="utf-8"))["hooks"]["PreToolUse"]) == 1

    # Run install twice to verify idempotency (no duplicate Aetheris entries)
    assert cli_main(["install", "qwen", "--workspace", str(tmp_path)]) == 0
    assert cli_main(["install", "qwen", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()

    assert json.loads(shared_settings.read_text(encoding="utf-8")) == {"shared": True}
    merged = json.loads(local_settings.read_text(encoding="utf-8"))
    assert merged["theme"] == "dark"
    assert merged["model"] == "qwen-coder"
    assert merged["hooks"]["Stop"][0]["hooks"][0]["command"] == "echo custom-stop"
    assert len(merged["hooks"]["PreToolUse"]) == 2
    assert merged["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "echo custom-pre"

    pre_entry = merged["hooks"]["PreToolUse"][1]
    post_entry = merged["hooks"]["PostToolUse"][0]
    assert "run_shell_command" in pre_entry["matcher"]
    assert "run_shell_command" in post_entry["matcher"]

    pre_hook = pre_entry["hooks"][0]
    post_hook = post_entry["hooks"][0]
    assert "--host qwen" in pre_hook["command"]
    assert "--session-state auto" in pre_hook["command"]
    assert pre_hook["timeout"] == 10
    assert "onFailure" not in pre_hook
    assert "--host qwen" in post_hook["command"]
    assert "--session-state auto" in post_hook["command"]
    assert "--replace-tool-output" not in post_hook["command"]
    assert post_hook["timeout"] == 10

    assert cli_main(["install", "qwen", "--self-test"]) == 0
    out = capsys.readouterr().out
    assert "QWEN SELF-TEST PASSED" in out
