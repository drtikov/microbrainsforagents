"""End-to-end tests for ZCode marketplace plugin, zcode_gate.py, --host zcode adapter, .zcode/ protection, and installer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from aetheris_microbrains import install_zcode
from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.layered_decision_engine import PolicyConfig, decide, is_protected_path
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
ZCODE_PLUGIN_DIR = REPO_ROOT / ".zcode-plugin"
GATE_SCRIPT = ZCODE_PLUGIN_DIR / "hooks" / "zcode_gate.py"


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_zcode_manifests_and_bundle_valid():
    mkt = json.loads((REPO_ROOT / "marketplace.json").read_text(encoding="utf-8"))
    assert mkt["name"] == "aetheris-microbrains"
    assert mkt["plugins"][0]["source"] == "."
    assert mkt["plugins"][0]["version"] == "0.7.6"

    plugin = json.loads((ZCODE_PLUGIN_DIR / "plugin.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "aetheris-microbrains"
    assert plugin["version"] == "0.7.6"

    hooks_doc = json.loads((ZCODE_PLUGIN_DIR / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    for event_name in ("PreToolUse", "PostToolUse"):
        groups = hooks_doc["hooks"][event_name]
        assert len(groups) == 1
        hook = groups[0]["hooks"][0]
        assert hook["type"] == "process"
        assert "onFailure" not in hook
        assert "async" not in hook

    assert GATE_SCRIPT.is_file()
    assert (ZCODE_PLUGIN_DIR / "skills" / "microbrains" / "SKILL.md").is_file()
    for cmd_name in ("audit.md", "doctor.md", "preflight.md"):
        assert (ZCODE_PLUGIN_DIR / "commands" / "mb" / cmd_name).is_file()


def test_zcode_control_plane_protected_by_default():
    pol = PolicyConfig(profile="dev")
    for p in (
        ".zcode/config.json",
        ".zcode/cli/config.json",
        ".zcode-plugin/plugin.json",
        ".zcode-plugin/hooks/hooks.json",
    ):
        assert is_protected_path(p, pol) is True
        assert decide({"tool": "write_file", "path": p, "diff": "{}"}, policy=pol)["decision"] == "deny"
        assert decide({"tool": "read_file", "path": p}, policy=pol)["decision"] == "escalate"


def test_zcode_pre_and_post_hook_contract():
    allow_v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        host="zcode",
        profile="dev",
        log_decisions=False,
    )
    assert allow_v["host"] == "zcode"
    assert allow_v["host_contract"] == "zcode_pretooluse_v1"
    assert allow_v["decision"] == "allow"
    assert allow_v["exit_code"] == 0

    deny_v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        host="zcode",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["decision"] == "deny"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["permissionDecision"] == "deny"

    esc_v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "custom_unlisted_zcode_cmd --flag"}},
        host="zcode",
        profile="dev",
        log_decisions=False,
    )
    assert esc_v["decision"] == "escalate"
    assert esc_v["exit_code"] == 0
    assert esc_v["hookSpecificOutput"]["permissionDecision"] == "ask"

    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {"tool_name": "Bash", "tool_input": {"command": "env"}, "tool_response": f"KEY={fake_tok}"},
        host="zcode",
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True


def test_zcode_gate_wrapper_subprocess_probes(tmp_path):
    env = dict(os.environ)
    env["ZCODE_PLUGIN_ROOT"] = str(REPO_ROOT)
    env["AETHERIS_STATE_DIR"] = str(tmp_path)

    try:
        deny_proc = subprocess.run(
            [sys.executable, "-I", str(GATE_SCRIPT), "--session-state", "auto"],
            input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}),
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
            check=False,
        )
        assert deny_proc.returncode == 2, deny_proc.stderr
        deny_json = json.loads(deny_proc.stdout.strip())
        assert deny_json["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
        assert deny_json["hookSpecificOutput"]["permissionDecision"] == "deny"

        allow_proc = subprocess.run(
            [sys.executable, "-I", str(GATE_SCRIPT), "--session-state", "auto"],
            input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "git status"}}),
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
            check=False,
        )
        assert allow_proc.returncode == 0, allow_proc.stderr
        assert allow_proc.stdout.strip() == ""
    finally:
        state_file = tmp_path / "hook_daemon.json"
        if state_file.is_file():
            try:
                import socket as _sock
                info = json.loads(state_file.read_text(encoding="utf-8"))
                port = int(info.get("port") or 0)
                auth = str(info.get("auth") or "")
                if port and auth:
                    with _sock.create_connection(("127.0.0.1", port), timeout=1.0) as s:
                        s.sendall((json.dumps({"auth": auth, "cmd": "shutdown"}) + "\n").encode("utf-8"))
            except Exception:
                pass


def test_zcode_installer_and_cli_integration(tmp_path, capsys):
    zcode_dir = tmp_path / ".zcode"
    zcode_dir.mkdir(parents=True)
    assert "zcode" in detect_installed_hosts(tmp_path)

    cfg_file = zcode_dir / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "enabled": False,
                    "events": {
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": [{"type": "process", "command": "echo", "args": ["user"]}]}
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    # Dry-run should not overwrite config
    before_text = cfg_file.read_text(encoding="utf-8")
    assert cli_main(["install", "zcode", "--workspace", str(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()
    assert cfg_file.read_text(encoding="utf-8") == before_text

    # Workspace install + idempotent re-run
    assert cli_main(["install", "zcode", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli_main(["install", "zcode", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()

    merged = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert merged["theme"] == "dark"
    assert merged["hooks"]["enabled"] is True
    pre_groups = merged["hooks"]["events"]["PreToolUse"]
    assert len(pre_groups) == 2
    assert pre_groups[0]["hooks"][0]["command"] == "echo"

    # Remove leaves unrelated user keys and hooks intact
    assert install_zcode.main(["--project-dir", str(tmp_path), "--remove"]) == 0
    capsys.readouterr()
    cleaned = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert cleaned["theme"] == "dark"
    assert len(cleaned["hooks"]["events"]["PreToolUse"]) == 1
    assert cleaned["hooks"]["events"]["PreToolUse"][0]["hooks"][0]["command"] == "echo"
    assert "PostToolUse" not in cleaned["hooks"]["events"]

    assert cli_main(["install", "zcode", "--self-test"]) == 0
    assert "ZCODE SELF-TEST PASSED" in capsys.readouterr().out

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []
