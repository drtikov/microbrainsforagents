"""End-to-end tests for Cursor native `cursor_hook_v2` adapter, full hooks.json installer, and self-test."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload
from aetheris_microbrains.hosts.cursor import format_pre_response, infer_event, to_kernel_payload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_cursor_shell_and_read_payloads():
    raw_deny = {"command": "rm -rf /", "cwd": ".", "sandbox": False}
    deny_v = evaluate_hook_payload(
        to_kernel_payload(raw_deny, infer_event(raw_deny, "before-shell")),
        host="cursor",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["host"] == "cursor"
    assert deny_v["host_contract"] == "cursor_hook_v2"
    assert deny_v["decision"] == "deny"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["permission"] == "deny"

    raw_allow = {"command": "git status", "cwd": "."}
    allow_v = evaluate_hook_payload(
        {"_cursor_event": "beforeShellExecution", **raw_allow},
        host="cursor",
        profile="dev",
        log_decisions=False,
    )
    assert allow_v["decision"] == "allow"
    assert allow_v["exit_code"] == 0
    assert allow_v["hookSpecificOutput"]["permission"] == "allow"

    ask_v = evaluate_hook_payload(
        {"_cursor_event": "beforeShellExecution", "command": "custom_unlisted_cursor_cmd --flag", "cwd": "."},
        host="cursor",
        profile="dev",
        log_decisions=False,
    )
    assert ask_v["decision"] == "escalate"
    assert ask_v["exit_code"] == 0
    assert ask_v["hookSpecificOutput"]["permission"] == "ask"

    safe_read = evaluate_hook_payload(
        {"_cursor_event": "beforeReadFile", "file_path": "src/app.py", "content": "print('hi')"},
        host="cursor",
        profile="dev",
        log_decisions=False,
    )
    assert safe_read["decision"] == "allow"
    assert safe_read["exit_code"] == 0

    env_read = evaluate_hook_payload(
        {"_cursor_event": "beforeReadFile", "file_path": ".env", "content": "SECRET=1"},
        host="cursor",
        profile="dev",
        log_decisions=False,
    )
    assert env_read["exit_code"] == 2
    assert env_read["hookSpecificOutput"]["permission"] == "deny"


def test_cursor_pretool_escalate_becomes_deny_and_shadow_allows():
    body, code = format_pre_response(event="preToolUse", decision="escalate", reason="novel command")
    assert body["permission"] == "deny"
    assert code == 2

    pre_esc = evaluate_hook_payload(
        {"_cursor_event": "preToolUse", "tool_name": "Shell", "tool_input": {"command": "custom_unlisted_cursor_cmd"}},
        host="cursor",
        profile="dev",
        log_decisions=False,
    )
    assert pre_esc["exit_code"] == 2
    assert pre_esc["hookSpecificOutput"]["permission"] == "deny"

    sh_v = evaluate_hook_payload(
        {"_cursor_event": "beforeShellExecution", "command": "rm -rf /", "cwd": "."},
        host="cursor",
        shadow_mode=True,
        log_decisions=False,
    )
    assert sh_v["exit_code"] == 0
    assert sh_v["hookSpecificOutput"]["permission"] == "allow"


def test_cursor_mcp_and_conversation_id_scoping():
    raw_mcp = {
        "hook_event_name": "beforeMCPExecution",
        "tool_name": "github:create_issue",
        "tool_input": json.dumps({"title": "test"}),
        "conversation_id": "conv-cursor-42",
    }
    kernel_p = to_kernel_payload(raw_mcp, infer_event(raw_mcp, "before-mcp"))
    assert kernel_p["session_id"] == "conv-cursor-42"
    assert kernel_p["tool_name"] == "MCP:github:create_issue"
    assert kernel_p["tool_input"] == {"title": "test"}

    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {
            "_cursor_event": "afterMCPExecution",
            "tool_name": "MCP:env_dump",
            "result_json": f"KEY={fake_tok}",
            "conversation_id": "conv-cursor-42",
        },
        host="cursor",
        replace_tool_output=True,
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert "additional_context" in post_v["hookSpecificOutput"]
    assert fake_tok not in post_v["hookSpecificOutput"]["updated_mcp_tool_output"]


def test_cursor_cli_subprocess_and_installer(tmp_path, monkeypatch, capsys):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    )
    proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--host", "cursor", "--event", "before-shell"],
        input=json.dumps({"command": "rm -rf /", "cwd": "."}),
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    out = json.loads(proc.stdout.strip())
    assert out["permission"] == "deny"
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir(parents=True)
    assert "cursor" in detect_installed_hosts(tmp_path)

    fake_gate = tmp_path / ("aetheris-gate.exe" if os.name == "nt" else "aetheris-gate")
    fake_gate.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("AETHERIS_GATE_BIN", str(fake_gate))

    assert cli_main(["install", "cursor", "--workspace", str(tmp_path), "--use-rust-shim"]) == 0
    capsys.readouterr()

    cfg = json.loads((cursor_dir / "hooks.json").read_text(encoding="utf-8"))
    pre_events = ("preToolUse", "beforeShellExecution", "beforeMCPExecution", "beforeReadFile")
    post_events = ("postToolUse", "afterShellExecution", "afterMCPExecution", "afterFileEdit")
    for ev_name in pre_events + post_events:
        assert ev_name in cfg["hooks"]
        entry = cfg["hooks"][ev_name][0]
        assert entry["timeout"] == 10
        assert entry["failClosed"] is (ev_name in pre_events)
        assert str(fake_gate) in entry["command"]
        if ev_name in post_events:
            assert "--replace-tool-output" in entry["command"]

    for scoped_ev in ("preToolUse", "postToolUse"):
        matcher = cfg["hooks"][scoped_ev][0].get("matcher", "")
        assert "StrReplace" in matcher and "Write" in matcher
        assert "Shell" not in matcher and "MCP" not in matcher

    after_edit_proc = subprocess.run(
        [sys.executable, "-m", "aetheris_microbrains.hook", "--host", "cursor", "--event", "after-file-edit"],
        input=json.dumps({"file_path": "src/app.py", "edits": [{"old_string": "x = 1", "new_string": "x = 2"}]}),
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=False,
    )
    assert after_edit_proc.returncode == 0, after_edit_proc.stderr

    assert cli_main(["install", "cursor", "--self-test"]) == 0
    assert "CURSOR SELF-TEST PASSED" in capsys.readouterr().out

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []


def test_cursor_projects_worktrees_and_native_tools():
    # Workspace/worktree paths under ~/.cursor/projects and ~/.cursor/worktrees must be allowed
    for p in (
        {"_cursor_event": "preToolUse", "tool_name": "Read", "tool_input": {"path": r"C:\Users\me\.cursor\projects\myrepo\src\app.py"}},
        {"_cursor_event": "beforeReadFile", "file_path": r"C:\Users\me\.cursor\projects\myrepo\terminals\1.txt", "content": "ok"},
        {"_cursor_event": "preToolUse", "tool_name": "Read", "tool_input": {"path": "~/.cursor/worktrees/myrepo/wt1/main.py"}},
    ):
        v = evaluate_hook_payload(p, host="cursor", profile="dev", log_decisions=False)
        assert v["decision"] == "allow", p
        assert v["exit_code"] == 0, p
        assert v["hookSpecificOutput"]["permission"] == "allow", p

    # Control-plane & secret paths (outside or nested inside .cursor/projects) must stay denied
    for p in (
        {"_cursor_event": "preToolUse", "tool_name": "Read", "tool_input": {"path": ".cursor/hooks.json"}},
        {"_cursor_event": "beforeReadFile", "file_path": r"C:\Users\me\.cursor\projects\myrepo\.cursor\hooks.json", "content": "{}"},
        {"_cursor_event": "beforeReadFile", "file_path": r"C:\Users\me\.cursor\projects\myrepo\.env", "content": "SECRET=1"},
        {"_cursor_event": "beforeReadFile", "file_path": r"C:\Users\me\.cursor\projects\..\hooks.json", "content": "{}"},
    ):
        v = evaluate_hook_payload(p, host="cursor", profile="dev", log_decisions=False)
        assert v["exit_code"] == 2, p
        assert v["hookSpecificOutput"]["permission"] == "deny", p

    # Safe native Cursor preToolUse tools
    for p in (
        {
            "_cursor_event": "preToolUse",
            "tool_name": "StrReplace",
            "tool_input": {"path": "src/app.py", "old_string": "x = 1", "new_string": "x = 2"},
        },
        {
            "_cursor_event": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": "src/app.py", "contents": "x = 2\n"},
        },
        {
            "_cursor_event": "preToolUse",
            "tool_name": "LS",
            "tool_input": {"target_directory": "src"},
        },
        {
            "_cursor_event": "preToolUse",
            "tool_name": "SemanticSearch",
            "tool_input": {"query": "auth"},
        },
    ):
        v = evaluate_hook_payload(p, host="cursor", profile="dev", log_decisions=False)
        assert v["decision"] == "allow", p
        assert v["exit_code"] == 0, p

    # Malicious AST or protected path edits via StrReplace / Write(contents) remain denied
    for p in (
        {
            "_cursor_event": "preToolUse",
            "tool_name": "StrReplace",
            "tool_input": {"path": "src/app.py", "old_string": "x = 1", "new_string": "eval(user_input)"},
        },
        {
            "_cursor_event": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": ".env", "contents": "SECRET=1\n"},
        },
        {
            "_cursor_event": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": ".cursor/hooks.json", "contents": "{}\n"},
        },
    ):
        v = evaluate_hook_payload(p, host="cursor", profile="dev", log_decisions=False)
        assert v["decision"] == "deny", p
        assert v["exit_code"] == 2, p
