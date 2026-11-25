"""End-to-end tests for Cline native `cline_hook_v1` adapter, `.clinerules/hooks/` installer, and SEARCH/REPLACE AST check."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_cline_pretool_schemas_search_replace_and_control_plane():
    # Schema 1: nested preToolUse {toolName, parameters}
    deny_v = evaluate_hook_payload(
        {"preToolUse": {"toolName": "execute_command", "parameters": {"command": "rm -rf /"}}, "taskId": "task-1"},
        host="cline",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["host"] == "cline"
    assert deny_v["host_contract"] == "cline_hook_v1"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["cancel"] is True
    assert deny_v["hookSpecificOutput"]["errorMessage"]

    # Schema 2: {tool: {name, parameters}} safe shell + READ_ONLY attempt_completion
    for payload in (
        {"tool": {"name": "run_commands", "parameters": {"command": "git status"}}},
        {"preToolUse": {"toolName": "read_file", "parameters": {"path": "src/app.py"}}},
        {"preToolUse": {"toolName": "attempt_completion", "parameters": {"result": "done"}}},
    ):
        v = evaluate_hook_payload(payload, host="cline", profile="dev", log_decisions=False)
        assert v["decision"] == "allow", payload
        assert v["exit_code"] == 0, payload
        assert v["hookSpecificOutput"] == {"cancel": False}, payload

    # Cline SEARCH/REPLACE diff: safe Python edit allowed; eval() blocked with cancel: True
    safe_edit = evaluate_hook_payload(
        {
            "preToolUse": {
                "toolName": "replace_in_file",
                "parameters": {
                    "path": "src/app.py",
                    "diff": "------- SEARCH\nx = 1\n=======\nx = 2\n+++++++ REPLACE",
                },
            }
        },
        host="cline",
        profile="dev",
        log_decisions=False,
    )
    assert safe_edit["decision"] == "allow"
    assert safe_edit["hookSpecificOutput"] == {"cancel": False}

    bad_edit = evaluate_hook_payload(
        {
            "preToolUse": {
                "toolName": "replace_in_file",
                "parameters": {
                    "path": "src/app.py",
                    "diff": "------- SEARCH\nx = 1\n=======\neval(user_input)\n+++++++ REPLACE",
                },
            }
        },
        host="cline",
        profile="dev",
        log_decisions=False,
    )
    assert bad_edit["decision"] == "deny"
    assert bad_edit["exit_code"] == 2
    assert bad_edit["hookSpecificOutput"]["cancel"] is True

    # Layer 0 protects .clinerules/hooks/ from self-tampering
    cfg_v = evaluate_hook_payload(
        {
            "preToolUse": {
                "toolName": "write_to_file",
                "parameters": {"path": ".clinerules/hooks/PreToolUse", "content": "exit 0"},
            }
        },
        host="cline",
        profile="dev",
        log_decisions=False,
    )
    assert cfg_v["exit_code"] == 2
    assert cfg_v["hookSpecificOutput"]["cancel"] is True

    # MCP tool normalization works transparently
    mcp_v = evaluate_hook_payload(
        {
            "preToolUse": {
                "toolName": "use_mcp_tool",
                "parameters": {"server_name": "github", "tool_name": "get_issue", "arguments": {"id": 1}},
            }
        },
        host="cline",
        profile="dev",
        log_decisions=False,
    )
    assert mcp_v["hookSpecificOutput"]["cancel"] is False


def test_cline_post_tool_dlp_and_cli_subprocess_and_installer(tmp_path, capsys):
    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {
            "postToolUse": {
                "toolName": "execute_command",
                "parameters": {"command": "echo check"},
                "result": f"SECRET={fake_tok}",
                "success": True,
            },
            "taskId": "task-cline-2",
        },
        host="cline",
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert post_v["host"] == "cline"
    assert post_v["hookSpecificOutput"]["cancel"] is False
    assert "SECURITY WARNING" in post_v["hookSpecificOutput"]["contextModification"]

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    )
    proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--host", "cline"],
        input=json.dumps({"preToolUse": {"toolName": "execute_command", "parameters": {"command": "rm -rf /"}}}),
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip())
    assert out["cancel"] is True
    assert "destructive_rm_rf" in out["errorMessage"]

    (tmp_path / ".clinerules").mkdir()
    assert "cline" in detect_installed_hosts(tmp_path)

    assert cli_main(["install", "cline", "--workspace", str(tmp_path), "--profile", "guardrails", "--no-rust-shim"]) == 0
    capsys.readouterr()
    hooks_dir = tmp_path / ".clinerules" / "hooks"
    for fname in ("PreToolUse", "PostToolUse", "PreToolUse.ps1", "PostToolUse.ps1"):
        fpath = hooks_dir / fname
        assert fpath.is_file(), f"Missing {fpath}"
        txt = fpath.read_text(encoding="utf-8")
        assert "-I -m aetheris_microbrains.hook --host cline" in txt

    assert cli_main(["install", "cline", "--self-test"]) == 0
    assert "CLINE SELF-TEST PASSED" in capsys.readouterr().out

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []
