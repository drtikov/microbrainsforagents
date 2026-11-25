"""End-to-end tests for native OpenCode ESM plugin, --host opencode adapter, .opencode/ protection, and installer."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.layered_decision_engine import PolicyConfig, decide, is_protected_path
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload
from aetheris_microbrains.hook_daemon import serve_http

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_FILE = REPO_ROOT / "opencode-aetheris-microbrains" / "aetheris-microbrains.js"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_opencode_control_plane_protected_by_default():
    pol = PolicyConfig(profile="dev")
    for p in ("opencode.json", "opencode.jsonc", ".opencode/opencode.json", ".opencode/plugins/aetheris-microbrains.js"):
        assert is_protected_path(p, pol) is True
        assert decide({"tool": "write_file", "path": p, "diff": "{}"}, policy=pol)["decision"] == "deny"
        assert decide({"tool": "read_file", "path": p}, policy=pol)["decision"] == "escalate"


def test_opencode_pre_and_post_hook_contract():
    allow_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "git status"}},
        host="opencode",
        profile="dev",
        log_decisions=False,
    )
    assert allow_v["host"] == "opencode"
    assert allow_v["host_contract"] == "opencode_plugin_v1"
    assert allow_v["decision"] == "allow"
    assert allow_v["hookSpecificOutput"] == {}

    deny_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "rm -rf /"}},
        host="opencode",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["decision"] == "deny"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["block"] is True

    esc_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "custom_unlisted_opencode_cmd --flag"}},
        host="opencode",
        profile="dev",
        log_decisions=False,
    )
    assert esc_v["decision"] == "escalate"
    assert esc_v["hookSpecificOutput"]["escalate"] is True

    wrap_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "ls -la"}},
        host="opencode",
        sandbox_wrap="srt",
        log_decisions=False,
    )
    assert wrap_v["hookSpecificOutput"]["args"]["command"].startswith("srt -- ")

    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {"toolName": "bash", "params": {"command": "env"}, "output": f"KEY={fake_tok}"},
        host="opencode",
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert fake_tok not in post_v["hookSpecificOutput"]["redacted_output"]


def test_opencode_http_daemon_header_and_installer(tmp_path, capsys):
    opencode_dir = tmp_path / ".opencode"
    opencode_dir.mkdir(parents=True)
    assert "opencode" in detect_installed_hosts(tmp_path)

    assert cli_main(["install", "opencode", "--workspace", str(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()
    installed = opencode_dir / "plugins" / "aetheris-microbrains.js"
    assert not installed.exists()

    assert cli_main(["install", "opencode", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    assert installed.is_file()

    assert cli_main(["install", "opencode", "--self-test"]) == 0
    assert "OPENCODE SELF-TEST PASSED" in capsys.readouterr().out

    port = _free_port()
    token = "test-opencode-token"
    t = threading.Thread(
        target=serve_http,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "state_path": tmp_path / "hook_daemon.json",
            "flags": {"host": "opencode", "profile": "dev"},
            "token": token,
        },
        daemon=True,
    )
    t.start()
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.05)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/hook/pre",
        data=json.dumps({"toolName": "bash", "params": {"command": "rm -rf /"}}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}", "X-Aetheris-Host": "opencode"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    assert body["block"] is True

    shut_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/shutdown",
        data=b"{}",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(shut_req, timeout=2.0) as resp:
        assert json.loads(resp.read().decode("utf-8"))["stopping"] is True
    t.join(timeout=3.0)

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []


def test_opencode_esm_plugin_smoke_with_node():
    node = shutil.which("node")
    if not node:
        return
    fake_tok = _make_fixture_token(("sk", "live", "ABCDEF1234567890abcdef"))
    script = (
        f"import plugin from {json.dumps(PLUGIN_FILE.as_uri())};"
        "const hooks = await plugin({ config: { daemonUrl: 'http://127.0.0.1:1', timeoutMs: 250, failClosed: true } });"
        "if (typeof hooks['tool.execute.before'] !== 'function' || typeof hooks['tool.execute.after'] !== 'function') process.exit(2);"
        f"const afterOut = {{ output: {json.dumps('KEY=' + fake_tok)} }};"
        "await hooks['tool.execute.after']({ tool: 'bash' }, afterOut);"
        "if (!afterOut.output.includes('[REDACTED_BY_AETHERIS]')) process.exit(3);"
        f"if (afterOut.output.includes({json.dumps(fake_tok)})) process.exit(4);"
        "let blocked = false;"
        "try { await hooks['tool.execute.before']({ tool: 'bash' }, { args: { command: 'rm -rf /' } }); } catch (e) { blocked = String(e.message).includes('fail-closed'); }"
        "if (!blocked) process.exit(5);"
        "console.log(JSON.stringify({ ok: true }));"
    )
    proc = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert json.loads(proc.stdout.strip())["ok"] is True
