"""End-to-end tests for native Pi Coding Agent ESM extension, --host pi adapter, .pi/ protection, and installer."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
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
EXTENSION_FILE = REPO_ROOT / "pi-aetheris-microbrains" / "index.js"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def _start_test_daemon(tmp_path: Path, token: str):
    port = _free_port()
    t = threading.Thread(
        target=serve_http,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "state_path": tmp_path / "hook_daemon.json",
            "flags": {"host": "pi", "profile": "dev"},
            "token": token,
        },
        daemon=True,
    )
    t.start()
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as resp:
                if resp.status == 200:
                    return port, t
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("Pi test HTTP daemon did not become ready")


def _stop_test_daemon(port: int, token: str, t: threading.Thread) -> None:
    shut_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/shutdown",
        data=b"{}",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(shut_req, timeout=2.0) as resp:
        assert json.loads(resp.read().decode("utf-8"))["stopping"] is True
    t.join(timeout=3.0)


def test_pi_control_plane_protected_by_default():
    pol = PolicyConfig(profile="dev")
    for p in (
        ".pi/settings.json",
        ".pi/extensions/aetheris-microbrains/index.js",
        ".pi/agent/settings.json",
        ".pi/agent/extensions/aetheris-microbrains/index.js",
    ):
        assert is_protected_path(p, pol) is True
        assert decide({"tool": "write_file", "path": p, "diff": "{}"}, policy=pol)["decision"] == "deny"
        assert decide({"tool": "read_file", "path": p}, policy=pol)["decision"] == "escalate"


def test_pi_pre_and_post_hook_contract():
    allow_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "git status"}},
        host="pi",
        profile="dev",
        log_decisions=False,
    )
    assert allow_v["host"] == "pi"
    assert allow_v["host_contract"] == "pi_extension_v1"
    assert allow_v["decision"] == "allow"
    assert allow_v["hookSpecificOutput"] == {}

    deny_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "rm -rf /"}},
        host="pi",
        profile="dev",
        log_decisions=False,
    )
    assert deny_v["decision"] == "deny"
    assert deny_v["exit_code"] == 2
    assert deny_v["hookSpecificOutput"]["block"] is True

    esc_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "custom_unlisted_pi_cmd --flag"}},
        host="pi",
        profile="dev",
        log_decisions=False,
    )
    assert esc_v["decision"] == "escalate"
    assert esc_v["hookSpecificOutput"]["escalate"] is True

    wrap_v = evaluate_hook_payload(
        {"toolName": "bash", "params": {"command": "ls -la"}},
        host="pi",
        sandbox_wrap="srt",
        log_decisions=False,
    )
    assert wrap_v["hookSpecificOutput"]["args"]["command"].startswith("srt -- ")

    fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
    post_v = evaluate_post_tool_use_payload(
        {"toolName": "bash", "params": {"command": "env"}, "output": f"KEY={fake_tok}"},
        host="pi",
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert fake_tok not in post_v["hookSpecificOutput"]["redacted_output"]


def test_pi_http_daemon_header_and_installer(tmp_path, capsys):
    pi_dir = tmp_path / ".pi"
    pi_dir.mkdir(parents=True)
    assert "pi" in detect_installed_hosts(tmp_path)

    assert cli_main(["install", "pi", "--workspace", str(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()
    installed = pi_dir / "extensions" / "aetheris-microbrains" / "index.js"
    assert not installed.exists()

    assert cli_main(["install", "pi", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    assert installed.is_file()

    assert cli_main(["install", "pi", "--self-test"]) == 0
    assert "PI SELF-TEST PASSED" in capsys.readouterr().out

    token = "test-pi-token"
    port, t = _start_test_daemon(tmp_path, token)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/hook/pre",
            data=json.dumps({"toolName": "bash", "params": {"command": "rm -rf /"}}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}", "X-Aetheris-Host": "pi"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["block"] is True
    finally:
        _stop_test_daemon(port, token, t)

    iso = verify_core_isolation()
    assert iso["ok"] is True
    assert iso["numpy_violations"] == []


def test_pi_esm_extension_smoke_with_node(tmp_path):
    node = shutil.which("node")
    if not node:
        return
    fake_tok = _make_fixture_token(("sk", "live", "ABCDEF1234567890abcdef"))

    # Part 1: Fail-closed when daemon is unreachable + session_start warning + stats persistence + tool_result DLP
    offline_script = (
        f"import ext from {json.dumps(EXTENSION_FILE.as_uri())};"
        "const handlers = {}; const commands = {}; const entries = [];"
        "const mockPi = {"
        "  on(ev, fn) { handlers[ev] = fn; },"
        "  registerCommand(name, def) { commands[name] = def; },"
        "  appendEntry(type, data) { entries.push({ customType: type, data }); },"
        "};"
        "ext(mockPi);"
        "if (typeof handlers.tool_call !== 'function' || typeof handlers.tool_result !== 'function' || typeof handlers.session_start !== 'function' || typeof handlers.session_shutdown !== 'function' || typeof commands.aetheris?.handler !== 'function' || typeof commands['aetheris-config']?.handler !== 'function') process.exit(2);"
        "const warnNotes = [];"
        "const staleBranch = [{ type: 'custom', customType: 'aetheris-stats', data: { checked: 1, blocked: 0, approved: 0, redacted: 0 } }, { type: 'custom', customType: 'aetheris-stats', data: { checked: 3, blocked: 1, approved: 1, redacted: 2 } }];"
        "const decoyEntries = [{ type: 'custom', customType: 'aetheris-stats', data: { checked: 99, blocked: 99, approved: 99, redacted: 99 } }];"
        "const ctxOff = { cwd: process.cwd(), sessionManager: { getSessionId: () => 'pi-sess-1', getBranch: () => staleBranch, getEntries: () => decoyEntries }, ui: { notify(msg, level) { warnNotes.push({ msg, level }); } } };"
        "await handlers.session_start({ reason: 'startup' }, ctxOff);"
        "if (!warnNotes.some((n) => n.level === 'warning' && n.msg.includes('unreachable'))) process.exit(6);"
        f"const resEvent = {{ toolName: 'bash', content: [{{ type: 'text', text: {json.dumps('KEY=' + fake_tok)} }}], details: {{ raw: {json.dumps(fake_tok)} }} }};"
        "await handlers.tool_result(resEvent, ctxOff);"
        "if (!resEvent.content[0].text.includes('[REDACTED_BY_AETHERIS]')) process.exit(3);"
        f"if (resEvent.content[0].text.includes({json.dumps(fake_tok)}) || resEvent.details.raw.includes({json.dumps(fake_tok)})) process.exit(4);"
        "const fc = await handlers.tool_call({ toolName: 'bash', toolCallId: 'c1', input: { command: 'rm -rf /' } }, ctxOff);"
        "if (!fc || fc.block !== true || !String(fc.reason).includes('fail-closed')) process.exit(5);"
        "if (entries.length !== 1 || entries[0].data.checked !== 4 || entries[0].data.blocked !== 2 || 'token' in entries[0].data || 'daemonUrl' in entries[0].data) process.exit(8);"
        "await handlers.session_shutdown({}, ctxOff);"
        "if (entries.length !== 2 || entries[1].customType !== 'aetheris-stats' || entries[1].data.checked !== 4 || entries[1].data.blocked !== 2 || entries[1].data.redacted < 4) process.exit(7);"
        "console.log(JSON.stringify({ ok: true }));"
    )
    env = dict(os.environ)
    env["AETHERIS_DAEMON_URL"] = "http://127.0.0.1:1"
    env["AETHERIS_TIMEOUT_MS"] = "250"
    env["AETHERIS_FAIL_CLOSED"] = "1"
    env["AETHERIS_AUTO_START"] = "0"
    proc_off = subprocess.run([node, "--input-type=module", "-e", offline_script], capture_output=True, text=True, env=env, check=False)
    assert proc_off.returncode == 0, proc_off.stderr or proc_off.stdout
    assert json.loads(proc_off.stdout.strip())["ok"] is True

    # Part 2: Live HTTP daemon — settingsManager override, periodic persist, allow/block/escalate, and /aetheris + /aetheris-config commands
    token = "test-pi-node-token"
    port, t = _start_test_daemon(tmp_path, token)
    try:
        online_script = (
            f"import {{ createAetherisPiExtension }} from {json.dumps(EXTENSION_FILE.as_uri())};"
            "const handlers = {}; const commands = {}; const entries = [];"
            "const mockPi = { on(ev, fn) { handlers[ev] = fn; }, registerCommand(name, def) { commands[name] = def; }, appendEntry(type, data) { entries.push({ customType: type, data }); } };"
            "createAetherisPiExtension()(mockPi);"
            f"const settingsCtx = {{ cwd: process.cwd(), sessionManager: {{ getSessionId: () => 'pi-sess-2', getBranch: () => [{{ type: 'custom', customType: 'aetheris-stats', data: {{ checked: 9, blocked: 0, approved: 0, redacted: 0 }} }}] }}, settingsManager: {{ get(k) {{ return k === 'aetheris' ? {{ daemonUrl: 'http://127.0.0.1:{port}', token: {json.dumps(token)}, timeoutMs: 2000 }} : undefined; }} }} }};"
            "await handlers.session_start({ reason: 'reload' }, settingsCtx);"
            "const allowRes = await handlers.tool_call({ toolName: 'bash', toolCallId: 'c1', input: { command: 'git status' } }, settingsCtx);"
            "if (allowRes !== undefined) process.exit(10);"
            "if (entries.length !== 1 || entries[0].data.checked !== 10 || 'token' in entries[0].data || 'daemonUrl' in entries[0].data) process.exit(16);"
            "const blockRes = await handlers.tool_call({ toolName: 'bash', toolCallId: 'c2', input: { command: 'rm -rf /' } }, settingsCtx);"
            "if (!blockRes || blockRes.block !== true) process.exit(11);"
            "const shadowCtx = { ...settingsCtx, settingsManager: { get(k) { return k === 'aetheris' ? { ...settingsCtx.settingsManager.get('aetheris'), shadow: true } : undefined; } } };"
            "const shadowRes = await handlers.tool_call({ toolName: 'bash', toolCallId: 'c2b', input: { command: 'rm -rf /' } }, shadowCtx);"
            "if (shadowRes !== undefined) process.exit(15);"
            "let confirmCalls = 0;"
            "const escApprove = await handlers.tool_call({ toolName: 'bash', toolCallId: 'c3', input: { command: 'custom_unlisted_pi_cmd --flag' } }, { ...settingsCtx, ui: { async confirm() { confirmCalls += 1; return true; } } });"
            "if (escApprove !== undefined || confirmCalls !== 1) process.exit(12);"
            "const escDecline = await handlers.tool_call({ toolName: 'bash', toolCallId: 'c4', input: { command: 'custom_unlisted_pi_cmd --other' } }, { ...settingsCtx, ui: { async confirm() { confirmCalls += 1; return false; } } });"
            "if (!escDecline || escDecline.block !== true || !String(escDecline.reason).includes('declined') || confirmCalls !== 2) process.exit(13);"
            "let notifiedMsg = ''; let notifiedLevel = '';"
            "await commands.aetheris.handler('', { ...settingsCtx, ui: { notify(msg, level) { notifiedMsg = msg; notifiedLevel = level; } } });"
            "if (!notifiedMsg.includes('daemon=UP') || notifiedLevel !== 'info') process.exit(14);"
            "let cfgMsg = ''; let cfgLevel = '';"
            "await commands['aetheris-config'].handler('', { ...settingsCtx, ui: { notify(msg, level) { cfgMsg = msg; cfgLevel = level; } } });"
            f"if (cfgLevel !== 'info' || !cfgMsg.includes('daemonUrl: http://127.0.0.1:{port}') || !cfgMsg.includes('token: ***') || cfgMsg.includes({json.dumps(token)}) || !cfgMsg.includes('Session Stats:')) process.exit(17);"
            "console.log(JSON.stringify({ ok: true, notifiedMsg, cfgMsg }));"
        )
        proc_on = subprocess.run([node, "--input-type=module", "-e", online_script], capture_output=True, text=True, check=False)
        assert proc_on.returncode == 0, proc_on.stderr or proc_on.stdout
        data = json.loads(proc_on.stdout.strip())
        assert data["ok"] is True
        assert "daemon=UP" in data["notifiedMsg"]
    finally:
        _stop_test_daemon(port, token, t)

    # Part 3: Loopback daemon auto-start without manual daemon + clean session_shutdown
    auto_port = _free_port()
    auto_env = dict(os.environ)
    auto_env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{auto_env['PYTHONPATH']}" if auto_env.get("PYTHONPATH") else str(REPO_ROOT)
    )
    auto_env["AETHERIS_DAEMON_URL"] = f"http://127.0.0.1:{auto_port}"
    auto_env["AETHERIS_DAEMON_TOKEN"] = ""
    auto_env["AETHERIS_AUTO_START"] = "1"
    auto_env["AETHERIS_PYTHON"] = sys.executable
    auto_script = (
        f"import ext from {json.dumps(EXTENSION_FILE.as_uri())};"
        "const handlers = {}; const commands = {};"
        "const mockPi = { on(ev, fn) { handlers[ev] = fn; }, registerCommand(name, def) { commands[name] = def; } };"
        "ext(mockPi);"
        "const ctxAuto = { cwd: process.cwd(), sessionManager: { getSessionId: () => 'pi-auto-1' } };"
        "const b = await handlers.tool_call({ toolName: 'bash', toolCallId: 'a1', input: { command: 'rm -rf /' } }, ctxAuto);"
        "if (!b || b.block !== true || String(b.reason).includes('fail-closed')) process.exit(20);"
        "const a = await handlers.tool_call({ toolName: 'bash', toolCallId: 'a2', input: { command: 'git status' } }, ctxAuto);"
        "if (a !== undefined) process.exit(21);"
        "let upMsg = '';"
        "await commands.aetheris.handler('', { ui: { notify(m) { upMsg = m; } } });"
        "if (!upMsg.includes('daemon=UP')) process.exit(22);"
        "await handlers.session_shutdown({}, ctxAuto);"
        "console.log(JSON.stringify({ ok: true }));"
    )
    proc_auto = subprocess.run([node, "--input-type=module", "-e", auto_script], capture_output=True, text=True, env=auto_env, check=False)
    assert proc_auto.returncode == 0, proc_auto.stderr or proc_auto.stdout
    assert json.loads(proc_auto.stdout.strip())["ok"] is True
