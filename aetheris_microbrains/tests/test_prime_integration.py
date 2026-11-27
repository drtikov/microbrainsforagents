"""End-to-end tests for Prime Agent (`prime_pretooluse_v1`) host adapter, sync/async wrappers, and CLI self-test."""

from __future__ import annotations

import asyncio
import dataclasses
import types
from pathlib import Path

import pytest

from aetheris_microbrains.cli import detect_installed_hosts, main as cli_main
from aetheris_microbrains.core.layered_decision_engine import PolicyConfig, decide, is_protected_path
from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload
from aetheris_microbrains.hosts.registry import HOST_REGISTRY
from aetheris_microbrains.prime_gate import (
    PrimeBlockedResult,
    PrimeMicrobrainsBlockedError,
    PrimeMicrobrainsEscalatedError,
    PrimeMicrobrainsGate,
    install_prime_gate,
    uninstall_prime_gate,
)


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def test_prime_control_plane_protected_and_host_verdicts():
    assert "prime" in HOST_REGISTRY
    assert HOST_REGISTRY["prime"].contract == "prime_pretooluse_v1"

    pol = PolicyConfig(profile="dev")
    assert is_protected_path(".prime/agent/settings.json", pol) is True
    assert is_protected_path(".prime/config.json", pol) is True

    write_v = decide({"tool": "write_file", "path": ".prime/agent/settings.json", "diff": "{}"}, policy=pol)
    assert write_v["decision"] == "deny"

    deny_sh = evaluate_hook_payload(
        {"tool_name": "bash", "tool_input": {"command": "rm -rf /"}},
        host="prime",
        profile="dev",
        log_decisions=False,
    )
    assert deny_sh["host"] == "prime"
    assert deny_sh["host_contract"] == "prime_pretooluse_v1"
    assert deny_sh["decision"] == "deny"
    assert deny_sh["exit_code"] == 2

    allow_sh = evaluate_hook_payload(
        {"tool_name": "bash", "tool_input": {"cmd": ["git", "status"]}},
        host="prime",
        profile="dev",
        log_decisions=False,
    )
    assert allow_sh["decision"] == "allow"
    assert allow_sh["exit_code"] == 0

    bad_edit = evaluate_hook_payload(
        {
            "tool_name": "edit.run",
            "tool_input": {"path": "app.py", "old_str": "x = 1", "new_str": "import os\nos.system('id')"},
        },
        host="prime",
        profile="dev",
        log_decisions=False,
    )
    assert bad_edit["decision"] == "deny"
    assert bad_edit["exit_code"] == 2

    fake_tok = _make_fixture_token(("sk", "live", "PRIMEHOSTCHECK1234567890"))
    post_v = evaluate_post_tool_use_payload(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "bash",
            "tool_input": {"command": "env"},
            "tool_output": f"API_KEY={fake_tok}",
        },
        host="prime",
        replace_tool_output=True,
        log_decisions=False,
    )
    assert post_v["secrets_found"] is True
    assert fake_tok not in str(post_v["redacted_output"])


def test_prime_sync_and_async_wrappers_install_idempotency():
    fake_tok = _make_fixture_token(("sk", "live", "PRIMEWRAPPER1234567890ab"))
    ran: list[str] = []

    def _sync_bash(command: str, persistent: bool = False) -> str:
        ran.append(f"sync_bash:{command}")
        if "leak" in command:
            return f"SECRET={fake_tok}"
        return f"ran:{command}"

    async def _async_bash(command: str, persistent: bool = False) -> str:
        ran.append(f"async_bash:{command}")
        return f"aran:{command}"

    async def _async_edit(path: str, old_str: str, new_str: str) -> str:
        ran.append(f"async_edit:{path}")
        return f"edited:{path}"

    async def _async_mcp(connection_id: str, tool: str, arguments=None) -> str:
        ran.append(f"async_mcp:{connection_id}:{tool}")
        return f"mcp:{connection_id}:{tool}:{fake_tok}"

    rlm_mod = types.SimpleNamespace(bash=_sync_bash)
    edit_mod = types.SimpleNamespace(run=_async_edit)
    mcp_mod = types.SimpleNamespace(call_tool=_async_mcp)

    # Install twice -> idempotent, never double-wraps
    gate = install_prime_gate(rlm_mod=rlm_mod, edit_mod=edit_mod, mcp_mod=mcp_mod, profile="dev")
    install_prime_gate(rlm_mod=rlm_mod, edit_mod=edit_mod, mcp_mod=mcp_mod, profile="dev")

    with pytest.raises(PrimeMicrobrainsBlockedError):
        rlm_mod.bash("rm -rf /")
    assert ran == []

    assert rlm_mod.bash("git status") == "ran:git status"
    assert ran == ["sync_bash:git status"]

    scrubbed = rlm_mod.bash("echo leak")
    assert fake_tok not in scrubbed

    with pytest.raises(PrimeMicrobrainsBlockedError):
        asyncio.run(edit_mod.run("app.py", "x = 1", "import os\nos.system('id')"))

    assert asyncio.run(edit_mod.run("app.py", "x = 1", "x = 2")) == "edited:app.py"

    # Untrusted MCP server escalates & raises PrimeMicrobrainsEscalatedError when block_on_escalate=True
    with pytest.raises(PrimeMicrobrainsEscalatedError):
        asyncio.run(mcp_mod.call_tool("untrusted", "run", {"q": "hi"}))

    # Async bash wrapper + allow-through MCP scrub when block_on_escalate=False
    async_gate = PrimeMicrobrainsGate(profile="dev", block_on_escalate=False)
    wrapped_async_bash = async_gate.wrap_bash(_async_bash)
    with pytest.raises(PrimeMicrobrainsBlockedError):
        asyncio.run(wrapped_async_bash("rm -rf /"))
    assert asyncio.run(wrapped_async_bash("git status")) == "aran:git status"

    wrapped_mcp_allow = async_gate.wrap_mcp_call(_async_mcp)
    mcp_out = asyncio.run(wrapped_mcp_allow("srv", "fetch", {"id": 1}))
    assert fake_tok not in mcp_out

    uninstall_prime_gate(rlm_mod=rlm_mod, edit_mod=edit_mod, mcp_mod=mcp_mod)
    assert rlm_mod.bash is _sync_bash
    assert edit_mod.run is _async_edit
    assert mcp_mod.call_tool is _async_mcp


def test_prime_scrub_post_fails_closed_on_immutable_and_leaky_objects():
    fake_tok = _make_fixture_token(("sk", "live", "PRIMEFAILCLOSED12345678"))

    class _ReadOnlyResult:
        @property
        def text(self) -> str:
            return f"API_KEY={fake_tok}"

        def __str__(self) -> str:
            return f"RO({fake_tok})"

    @dataclasses.dataclass(frozen=True)
    class _LeakyResult:
        text: str
        raw: str

    gate = PrimeMicrobrainsGate(profile="dev")
    ro_out, _ = gate.scrub_post("bash", {"command": "env"}, _ReadOnlyResult())
    assert isinstance(ro_out, PrimeBlockedResult)
    assert ro_out.ok is False and ro_out.blocked is True
    assert fake_tok not in str(ro_out)

    dc_out, _ = gate.scrub_post("bash", {"command": "env"}, _LeakyResult(text=fake_tok, raw=fake_tok))
    assert isinstance(dc_out, PrimeBlockedResult)
    assert fake_tok not in str(dc_out)


def test_prime_cli_detect_dry_run_self_test_and_core_isolation(tmp_path: Path, capsys):
    ws = tmp_path / "prime_ws"
    (ws / ".prime" / "agent").mkdir(parents=True)
    assert "prime" in detect_installed_hosts(workspace=ws)

    assert cli_main(["install", "prime", "--dry-run"]) == 0
    assert "install_prime_gate" in capsys.readouterr().out

    assert cli_main(["install", "prime", "--self-test"]) == 0
    assert "PRIME SELF-TEST PASSED" in capsys.readouterr().out

    report = verify_core_isolation()
    assert report["ok"] is True, report
