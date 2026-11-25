"""End-to-end tests for native Qoder support, .qoder/ protection, installer merge, and --host cursor fix."""

from __future__ import annotations

import io
import json
from pathlib import Path

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.layered_decision_engine import PolicyConfig, decide, is_protected_path
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hook import (
    evaluate_hook_payload,
    evaluate_post_tool_use_payload,
    main as hook_main,
)


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_qoder_control_plane_protected_by_default():
    pol = PolicyConfig(profile="dev")
    assert is_protected_path(".qoder/settings.json", pol) is True
    assert is_protected_path(".qoder/settings.local.json", pol) is True

    write_v = decide({"tool": "write_file", "path": ".qoder/settings.json", "diff": "{}"}, policy=pol)
    assert write_v["decision"] == "deny"

    read_v = decide({"tool": "read_file", "path": ".qoder/settings.json"}, policy=pol)
    assert read_v["decision"] == "escalate"


def test_qoder_pre_and_post_hook_contract():
    allow_v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        host="qoder",
        profile="dev",
        log_decisions=False,
    )
    assert allow_v["host"] == "qoder"
    assert allow_v["host_contract"] == "qoder_pretooluse_v1"
    assert allow_v["decision"] == "allow"
    assert allow_v["exit_code"] == 0

    deny_v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        host="qoder",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["decision"] == "deny"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["permissionDecision"] == "deny"

    ask_v = evaluate_hook_payload(
        {"tool_name": "Read", "tool_input": {"file_path": ".env"}},
        host="qoder",
        profile="dev",
        log_decisions=False,
    )
    assert ask_v["decision"] == "escalate"
    assert ask_v["hookSpecificOutput"]["permissionDecision"] == "ask"

    bad_py = evaluate_hook_payload(
        {"tool_name": "Write", "tool_input": {"file_path": "app.py", "content": "def broken(:\n"}},
        host="qoder",
        profile="dev",
        log_decisions=False,
    )
    assert bad_py["decision"] == "deny"
    assert bad_py["exit_code"] == 2

    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {"tool_name": "Bash", "tool_input": {"command": "env"}, "tool_output": f"API_KEY={fake_tok}"},
        host="qoder",
        replace_tool_output=True,
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert fake_tok not in post_v["redacted_output"]
    assert fake_tok not in str(post_v["hookSpecificOutput"].get("updatedToolOutput", ""))


def test_cursor_host_flag_accepted_by_cli_and_hook(monkeypatch, capsys):
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git status"}})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    rc = hook_main(["--host", "cursor", "--profile", "dev"])
    assert rc == 0
    capsys.readouterr()

    res = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        host="cursor",
        profile="dev",
        log_decisions=False,
    )
    assert res["host"] == "cursor"
    assert res["host_contract"] == "cursor_hook_v1"
    assert res["decision"] == "deny"


def test_qoder_installer_merge_detection_and_self_test(tmp_path, capsys):
    qoder_dir = tmp_path / ".qoder"
    qoder_dir.mkdir(parents=True)
    assert "qoder" in detect_installed_hosts(tmp_path)

    # Workspace install writes machine-local .qoder/settings.local.json and preserves existing hooks & settings
    shared_settings = qoder_dir / "settings.json"
    shared_settings.write_text(json.dumps({"shared": True}), encoding="utf-8")

    local_settings = qoder_dir / "settings.local.json"
    local_settings.write_text(
        json.dumps({
            "theme": "dark",
            "model": "qoder-default",
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo custom-pre"}]}
                ],
                "Stop": [{"hooks": [{"type": "command", "command": "echo custom-stop"}]}],
            },
        }),
        encoding="utf-8",
    )

    assert cli_main(["install", "qoder", "--workspace", str(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()
    assert len(json.loads(local_settings.read_text(encoding="utf-8"))["hooks"]["PreToolUse"]) == 1

    # Run install twice to verify idempotency (no duplicate Aetheris entries)
    assert cli_main(["install", "qoder", "--workspace", str(tmp_path)]) == 0
    assert cli_main(["install", "qoder", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()

    assert json.loads(shared_settings.read_text(encoding="utf-8")) == {"shared": True}
    merged = json.loads(local_settings.read_text(encoding="utf-8"))
    assert merged["theme"] == "dark"
    assert merged["model"] == "qoder-default"
    assert merged["hooks"]["Stop"][0]["hooks"][0]["command"] == "echo custom-stop"
    assert len(merged["hooks"]["PreToolUse"]) == 2
    assert merged["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "echo custom-pre"

    pre_hook = merged["hooks"]["PreToolUse"][1]["hooks"][0]
    post_hook = merged["hooks"]["PostToolUse"][0]["hooks"][0]
    assert "--host qoder" in pre_hook["command"]
    assert "--session-state auto" in pre_hook["command"]
    assert pre_hook["timeout"] == 10
    assert "onFailure" not in pre_hook
    assert "--host qoder" in post_hook["command"]
    assert "--session-state auto" in post_hook["command"]
    assert "--replace-tool-output" in post_hook["command"]
    assert post_hook["timeout"] == 10

    assert cli_main(["install", "qoder", "--self-test"]) == 0
    out = capsys.readouterr().out
    assert "QODER SELF-TEST PASSED" in out

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []
