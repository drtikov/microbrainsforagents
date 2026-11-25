"""End-to-end tests for Devin native `devin_hook_v1` adapter, `.devin/hooks.v1.json` installer, and auto host detection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.core.rust_shim import build_hook_command
from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload
from aetheris_microbrains.hosts.registry import resolve_host

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_devin_pretool_payloads_and_fixed_tool_mapping():
    deny_v = evaluate_hook_payload(
        {"hook_event_name": "PreToolUse", "tool_name": "exec", "tool_input": {"command": "rm -rf /"}},
        host="devin",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["host"] == "devin"
    assert deny_v["host_contract"] == "devin_hook_v1"
    assert deny_v["decision"] == "deny"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["decision"] == "block"

    # Safe shell and READ_ONLY Devin tools (specifically get_output must NOT false-escalate as execute_shell)
    for payload in (
        {"hook_event_name": "PreToolUse", "tool_name": "exec", "tool_input": {"command": "git status"}},
        {"hook_event_name": "PreToolUse", "tool_name": "get_output", "tool_input": {"shell_id": "1"}},
        {"hook_event_name": "PreToolUse", "tool_name": "notebook_read", "tool_input": {"path": "nb.ipynb"}},
        {"hook_event_name": "PreToolUse", "tool_name": "run_subagent", "tool_input": {"prompt": "inspect"}},
        {"hook_event_name": "PreToolUse", "tool_name": "todo_write", "tool_input": {"todos": []}},
    ):
        v = evaluate_hook_payload(payload, host="devin", profile="dev", log_decisions=False)
        assert v["decision"] == "allow", payload
        assert v["exit_code"] == 0, payload
        assert v["hookSpecificOutput"] == {}, payload

    # Escalate defers to Devin's native permission prompt (empty stdout, exit 0) unless auto_approve_escalations is set
    esc_v = evaluate_hook_payload(
        {"hook_event_name": "PreToolUse", "tool_name": "exec", "tool_input": {"command": "curl https://example.com"}},
        host="devin",
        profile="guardrails",
        log_decisions=False,
    )
    assert esc_v["decision"] == "escalate"
    assert esc_v["exit_code"] == 0
    assert esc_v["hookSpecificOutput"] == {}

    esc_auto = evaluate_hook_payload(
        {"hook_event_name": "PreToolUse", "tool_name": "exec", "tool_input": {"command": "curl https://example.com"}},
        host="devin",
        profile="guardrails",
        auto_approve_escalations=True,
        log_decisions=False,
    )
    assert esc_auto["exit_code"] == 0
    assert esc_auto["hookSpecificOutput"]["decision"] == "approve"

    # Control-plane protection on .devin/hooks.v1.json and .devin/config.json
    for cfg_path in (".devin/hooks.v1.json", ".devin/config.json"):
        cfg_v = evaluate_hook_payload(
            {"hook_event_name": "PreToolUse", "tool_name": "write", "tool_input": {"file_path": cfg_path, "content": "{}"}},
            host="devin",
            profile="dev",
            log_decisions=False,
        )
        assert cfg_v["exit_code"] == 2
        assert cfg_v["hookSpecificOutput"]["decision"] == "block"


def test_devin_post_tool_structured_response_and_host_precedence(monkeypatch):
    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "exec",
            "tool_input": {"command": "echo check"},
            "tool_response": {"success": True, "output": f"API_KEY={fake_tok}", "error": None},
            "prompt_id": "prompt-devin-1",
        },
        host="devin",
        replace_tool_output=True,
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert post_v["host"] == "devin"
    assert post_v["host_action"] == "warn_additional_context"
    assert post_v["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "SECURITY WARNING" in post_v["hookSpecificOutput"]["additionalContext"]

    # DEVIN_PROJECT_DIR resolves --host auto -> devin, NEVER overriding explicit --host claude
    monkeypatch.setenv("DEVIN_PROJECT_DIR", "/tmp/my-devin-proj")
    assert resolve_host("auto", default="claude") == "devin"
    assert resolve_host("claude", default="claude") == "claude"
    monkeypatch.delenv("DEVIN_PROJECT_DIR", raising=False)
    assert resolve_host("auto", default="claude") == "claude"


def test_devin_cli_subprocess_installer_and_claude_overlap_warning(tmp_path, monkeypatch, capsys):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    )
    env["DEVIN_PROJECT_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--host", "auto"],
        input=json.dumps({"hook_event_name": "PreToolUse", "tool_name": "exec", "tool_input": {"command": "rm -rf /"}}),
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    out = json.loads(proc.stdout.strip())
    assert out["decision"] == "block"

    devin_dir = tmp_path / ".devin"
    devin_dir.mkdir(parents=True)
    assert "devin" in detect_installed_hosts(tmp_path)

    # Overlap with .claude/settings.local.json emits double-evaluation warning
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.local.json").write_text(
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "python -m aetheris_microbrains.hook"}]}]}}),
        encoding="utf-8",
    )

    assert cli_main(["install", "devin", "--workspace", str(tmp_path), "--profile", "guardrails"]) == 0
    inst_out = capsys.readouterr().out
    assert "auto-loads .claude/ hooks" in inst_out

    hooks_file = devin_dir / "hooks.v1.json"
    assert hooks_file.is_file()
    cfg = json.loads(hooks_file.read_text(encoding="utf-8"))
    for ev_name in ("PreToolUse", "PostToolUse"):
        entry = cfg["hooks"][ev_name][0]
        assert "exec" in entry["matcher"]
        cmd_hook = entry["hooks"][0]
        assert cmd_hook["type"] == "command"
        assert cmd_hook["timeout"] == 10
        assert "onFailure" not in cmd_hook
        assert "async" not in cmd_hook

    _py_exe, py_argv = build_hook_command("devin", event_args=[], extra_args=[], use_rust_shim=False)
    assert py_argv[:3] == ["-I", "-m", "aetheris_microbrains.hook"]

    assert cli_main(["install", "devin", "--self-test"]) == 0
    assert "DEVIN SELF-TEST PASSED" in capsys.readouterr().out

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []
