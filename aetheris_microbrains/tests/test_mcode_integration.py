"""End-to-end tests for MiniMax Code (`mcode_pretooluse_v1`), BOM stdin handling, installer, and host parity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload
from aetheris_microbrains.hosts.mcode import MCODE_TOOL_MATCHER
from aetheris_microbrains.hosts.registry import HOST_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_mcode_native_probes_and_todowrite_fix():
    deny_v = evaluate_hook_payload(
        {"hook_event_name": "PreToolUse", "session_id": "s1", "cwd": ".", "tool_name": "bash", "tool_input": {"command": "rm -rf /"}},
        host="mcode",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["host"] == "mcode"
    assert deny_v["host_contract"] == "mcode_pretooluse_v1"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["permissionDecision"] == "deny"

    for payload in (
        {"tool_name": "bash", "tool_input": {"command": "git status"}},
        {"tool_name": "todowrite", "tool_input": {"todos": []}},
        {"tool_name": "read", "tool_input": {"file_path": "README.md"}},
    ):
        v = evaluate_hook_payload(payload, host="mcode", profile="dev", log_decisions=False)
        assert v["decision"] == "allow", payload
        assert v["exit_code"] == 0, payload

    ask_v = evaluate_hook_payload(
        {"tool_name": "read", "tool_input": {"file_path": ".env"}},
        host="mcode",
        profile="dev",
        log_decisions=False,
    )
    assert ask_v["decision"] == "escalate"
    assert ask_v["exit_code"] == 0
    assert ask_v["hookSpecificOutput"]["permissionDecision"] == "ask"

    ast_v = evaluate_hook_payload(
        {"tool_name": "edit", "tool_input": {"file_path": "app.py", "old_string": "x = 1", "new_string": "import os\nos.system('rm -rf /')"}},
        host="mcode",
        profile="dev",
        log_decisions=False,
    )
    assert ast_v["exit_code"] == 2
    assert ast_v["hookSpecificOutput"]["permissionDecision"] == "deny"

    cfg_v = evaluate_hook_payload(
        {"tool_name": "write", "tool_input": {"file_path": ".minimax/hooks.json", "content": "{}"}},
        host="mcode",
        profile="dev",
        log_decisions=False,
    )
    assert cfg_v["exit_code"] == 2

    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "read",
            "tool_input": {"file_path": ".env"},
            "tool_response": f"API_KEY={fake_tok}",
        },
        host="mcode",
        replace_tool_output=True,
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert "SECURITY WARNING" in post_v["hookSpecificOutput"]["additionalContext"]
    assert "updatedToolOutput" not in post_v["hookSpecificOutput"]


def test_mcode_bom_stdin_silent_allow_and_installer(tmp_path, monkeypatch, capsys):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    )
    # UTF-8 BOM prefix on stdin (common with Windows PowerShell) must NOT fail JSON parsing
    bom_allow = "\ufeff" + json.dumps({"tool_name": "bash", "tool_input": {"command": "git status"}})
    p_allow = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--host", "mcode"],
        input=bom_allow,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=20,
        check=False,
    )
    assert p_allow.returncode == 0, p_allow.stderr
    assert p_allow.stdout.strip() == ""  # silent pass-through on allow without --auto-allow

    bom_deny = "\ufeff" + json.dumps({"tool_name": "bash", "tool_input": {"command": "rm -rf /"}})
    p_deny = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--host", "mcode"],
        input=bom_deny,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=20,
        check=False,
    )
    assert p_deny.returncode == 2
    out = json.loads(p_deny.stdout.strip())
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    # Project install
    assert cli_main(["install", "mcode", "--workspace", str(tmp_path), "--profile", "guardrails", "--no-rust-shim"]) == 0
    assert "Settings -> Hooks -> Project" in capsys.readouterr().out
    assert "mcode" in detect_installed_hosts(tmp_path)

    proj_cfg = json.loads((tmp_path / ".minimax" / "hooks.json").read_text(encoding="utf-8"))
    for ev in ("PreToolUse", "PostToolUse"):
        entry = proj_cfg["hooks"][ev][0]
        assert entry["matcher"] == MCODE_TOOL_MATCHER
        assert "*" not in entry["matcher"]
        h = entry["hooks"][0]
        assert h["timeout"] == 10
        assert "commandWindows" in h
        assert "-I -m aetheris_microbrains.hook --host mcode" in h["command"]
        assert "--replace-tool-output" not in h["command"]

    # User install writes top-level enabled: true under MINIMAX_DATA_DIR
    mdata = tmp_path / "mdata"
    monkeypatch.setenv("MINIMAX_DATA_DIR", str(mdata))
    assert cli_main(["install", "mcode", "--profile", "guardrails", "--no-rust-shim"]) == 0
    capsys.readouterr()
    user_cfg = json.loads((mdata / "hooks.json").read_text(encoding="utf-8"))
    assert user_cfg["enabled"] is True

    assert cli_main(["install", "mcode", "--self-test"]) == 0
    assert "MCODE SELF-TEST PASSED" in capsys.readouterr().out

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []


@pytest.mark.parametrize("host", sorted(HOST_REGISTRY))
def test_all_registered_hosts_verdict_parity(host: str):
    deny = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        host=host,
        log_decisions=False,
    )
    allow = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        host=host,
        log_decisions=False,
    )
    assert deny["decision"] == "deny"
    assert allow["decision"] == "allow"
    assert allow["exit_code"] == 0
    if host not in {"openclaw", "antigravity"}:
        assert deny["exit_code"] == 2
