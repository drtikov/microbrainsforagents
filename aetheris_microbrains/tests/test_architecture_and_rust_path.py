"""Tests for HostAdapter registry, modular core security facades, and Rust `aetheris-gate` fast-path wiring."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from aetheris_microbrains import install_openhands, install_zcode
from aetheris_microbrains.cli import main as cli_main
from aetheris_microbrains.core import ast_inspector, ifc_journal, policy_paths, rust_shim, shell_inspector
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hosts import (
    HOST_REGISTRY,
    HostAdapter,
    get_host_adapter,
    list_hosts,
    register_host,
    resolve_host,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_GATE_RELEASE = REPO_ROOT / "rust" / "aetheris-gate" / "target" / "release" / (
    "aetheris-gate.exe" if os.name == "nt" else "aetheris-gate"
)


def test_host_adapter_registry_and_custom_registration():
    hosts = list_hosts()
    for expected in ("claude", "codex", "zcode", "openhands", "pi", "omp", "kilocode", "opencode"):
        assert expected in hosts

    assert resolve_host("zcode") == "zcode"
    assert resolve_host("unknown_future_host", default="claude") == "claude"
    assert get_host_adapter("openhands").contract == "openhands_hook_v1"

    custom = HostAdapter(name="test_custom_agent", contract="custom_v1", family="claude_compat")
    register_host(custom)
    try:
        assert resolve_host("test_custom_agent") == "test_custom_agent"
        assert get_host_adapter("test_custom_agent").contract == "custom_v1"
    finally:
        HOST_REGISTRY.pop("test_custom_agent", None)


def test_modular_core_facades_and_isolation():
    pol = policy_paths.PolicyConfig(profile="dev")
    assert policy_paths.is_protected_path(".zcode/config.json", pol) is True
    assert policy_paths.is_protected_path(".openhands/hooks.json", pol) is True

    sh_bad = shell_inspector.analyze_shell_command("rm -rf /", policy=pol)
    assert sh_bad["is_safe"] is False
    assert sh_bad["findings"]

    ast_bad = ast_inspector.analyze_code_diff(diff_text="def broken(:\n", file_path="bad.py", policy=pol)
    assert ast_bad["syntax_valid"] is False

    assert callable(ifc_journal.record_decision_journal)
    assert callable(ifc_journal.get_recent_decision_journal)

    iso = verify_core_isolation()
    assert iso["ok"] is True, iso
    assert iso["missing_modules"] == []
    assert iso["numpy_violations"] == []


def test_use_rust_shim_across_claude_codex_zcode_openhands(tmp_path, monkeypatch, capsys):
    fake_gate = tmp_path / ("aetheris-gate.exe" if os.name == "nt" else "aetheris-gate")
    fake_gate.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("AETHERIS_GATE_BIN", str(fake_gate))

    assert rust_shim.find_aetheris_gate() == str(fake_gate)

    # Claude & Codex via unified CLI
    assert cli_main(["install", "claude", "--workspace", str(tmp_path), "--use-rust-shim"]) == 0
    assert cli_main(["install", "codex", "--workspace", str(tmp_path), "--use-rust-shim"]) == 0
    claude_cfg = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    codex_cfg = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert "--auto-daemon" in claude_cfg["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert str(fake_gate) in codex_cfg["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    # ZCode via unified CLI & remove
    assert cli_main(["install", "zcode", "--workspace", str(tmp_path), "--use-rust-shim"]) == 0
    zcode_cfg = json.loads((tmp_path / ".zcode" / "config.json").read_text(encoding="utf-8"))
    z_hook = zcode_cfg["hooks"]["events"]["PreToolUse"][0]["hooks"][0]
    assert z_hook["command"] == str(fake_gate)
    assert z_hook["args"][0] == "--auto-daemon"
    assert "--host" in z_hook["args"] and "zcode" in z_hook["args"]
    assert install_zcode.main(["--project-dir", str(tmp_path), "--remove"]) == 0
    assert "events" not in json.loads((tmp_path / ".zcode" / "config.json").read_text(encoding="utf-8")).get("hooks", {})

    # OpenHands via unified CLI & remove
    assert cli_main(["install", "openhands", "--workspace", str(tmp_path), "--use-rust-shim"]) == 0
    oh_cfg = json.loads((tmp_path / ".openhands" / "hooks.json").read_text(encoding="utf-8"))
    oh_cmd = oh_cfg["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert str(fake_gate) in oh_cmd and "--auto-daemon" in oh_cmd and "--host openhands" in oh_cmd
    assert install_openhands.main(["--project-dir", str(tmp_path), "--remove"]) == 0
    capsys.readouterr()


def test_rust_aetheris_gate_release_binary_live_probes():
    if not RUST_GATE_RELEASE.is_file():
        return

    st = subprocess.run([str(RUST_GATE_RELEASE), "--self-test"], capture_output=True, text=True, check=False)
    assert st.returncode == 0, st.stderr
    assert json.loads(st.stdout.strip())["ok"] is True

    z_deny = subprocess.run(
        [str(RUST_GATE_RELEASE), "--host", "zcode"],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert z_deny.returncode == 2
    assert json.loads(z_deny.stdout.strip())["hookSpecificOutput"]["permissionDecision"] == "deny"

    oh_deny = subprocess.run(
        [str(RUST_GATE_RELEASE), "--host", "openhands"],
        input=json.dumps(
            {
                "tool_name": "file_editor",
                "tool_input": {"command": "create", "path": ".openhands/hooks.json", "file_text": "{}"},
            }
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert oh_deny.returncode == 2
    assert json.loads(oh_deny.stdout.strip())["hookSpecificOutput"]["permissionDecision"] == "deny"
