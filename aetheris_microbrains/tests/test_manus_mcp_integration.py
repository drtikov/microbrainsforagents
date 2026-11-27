"""Integration tests for Manus custom MCP profile, Bearer auth, path confinement, and Streamable HTTP round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aetheris_microbrains.manus_auth import BearerAuthMiddleware
from aetheris_microbrains.mcp_server import (
    MANUS_STATELESS_TOOLS,
    build_mcp_server,
    build_streamable_http_app,
    main as mcp_main,
)


def _make_fixture_token(parts: tuple[str, ...]) -> str:
    return "_".join(parts)


def _extract_tool_json(rpc_response: dict) -> dict:
    result = rpc_response.get("result") or {}
    content = result.get("content") or []
    assert content and isinstance(content[0], dict), rpc_response
    return json.loads(content[0]["text"])


def test_bearer_auth_middleware_validation_and_rejection():
    async def _dummy_app(scope, receive, send):
        return None

    for bad_token in ("", "non-ascii-\u00e9", "has whitespace"):
        with pytest.raises(ValueError):
            BearerAuthMiddleware(_dummy_app, token=bad_token)

    _server, app = build_streamable_http_app(
        profile="manus",
        bearer_token="secret-manus-token-123",
        protect_health=False,
        json_response=True,
    )
    with TestClient(app, base_url="http://127.0.0.1:8080") as client:
        # /health is public by default
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["profile"] == "manus"

        # /mcp and /mcp/subpath fail closed without valid Bearer token
        for target in ("/mcp", "/mcp/subpath"):
            assert client.post(target, json={}).status_code == 401
            assert client.post(target, json={}, headers={"Authorization": "Basic dXNlcjpwYXNz"}).status_code == 401
            assert client.post(target, json={}, headers={"Authorization": "Bearer "}).status_code == 401
            r_wrong = client.post(target, json={}, headers={"Authorization": "Bearer wrong-token"})
            assert r_wrong.status_code == 401
            assert r_wrong.json() == {"error": "unauthorized"}

    _server2, protected_app = build_streamable_http_app(
        profile="manus",
        bearer_token="secret-manus-token-123",
        protect_health=True,
        json_response=True,
    )
    with TestClient(protected_app) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer secret-manus-token-123"}).status_code == 200


def test_manus_streamable_http_roundtrip_allowlist_and_preflight(tmp_path: Path):
    token = "manus-test-bearer-xyz789"
    _server, app = build_streamable_http_app(
        profile="manus",
        bearer_token=token,
        json_response=True,
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
    }

    with TestClient(app, base_url="http://127.0.0.1:8080") as client:
        # 1. MCP initialize
        init_res = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "manus-test", "version": "1.0"},
                },
            },
        )
        assert init_res.status_code == 200, init_res.text
        sess_id = init_res.headers.get("mcp-session-id")
        if sess_id:
            headers["Mcp-Session-Id"] = sess_id

        # Notify initialized
        notif = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert notif.status_code in (200, 202, 204), notif.text

        # 2. tools/list: strict 5-tool advisory allowlist only
        list_res = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert list_res.status_code == 200, list_res.text
        tool_names = sorted(t["name"] for t in list_res.json()["result"]["tools"])
        assert tool_names == sorted(MANUS_STATELESS_TOOLS)
        for forbidden in ("microbrain_audit_explain", "microbrain_directives", "microbrain_verify"):
            assert forbidden not in tool_names

        # 3. tools/call microbrain_health: omits host filesystem paths
        health_res = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "microbrain_health", "arguments": {}},
            },
        )
        assert health_res.status_code == 200
        health_data = _extract_tool_json(health_res.json())
        assert health_data["status"] == "ok"
        assert health_data["profile"] == "manus"
        assert health_data["advisory_only"] is True
        assert "state_dir" not in health_data
        assert "journal_path" not in health_data

        # 4. tools/call microbrain_preflight_check: denies `rm -rf /` & ignores caller session_state_path
        evil_state = tmp_path / "should_not_create_session.json"
        pre_res = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "microbrain_preflight_check",
                    "arguments": {
                        "tool_name": "Bash",
                        "command": "rm -rf /",
                        "session_state_path": str(evil_state),
                    },
                },
            },
        )
        assert pre_res.status_code == 200
        pre_data = _extract_tool_json(pre_res.json())
        assert pre_data["decision"] == "deny"
        assert pre_data["exit_code"] == 2
        assert not evil_state.exists()

        # 5. tools/call microbrain_scrub_and_scan: redacts constructed secret token
        fake_tok = _make_fixture_token(("sk", "live", "1234567890ABCDEFGHijklmn"))
        scrub_res = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "microbrain_scrub_and_scan",
                    "arguments": {"text": f"API_KEY={fake_tok}"},
                },
            },
        )
        assert scrub_res.status_code == 200
        scrub_data = _extract_tool_json(scrub_res.json())
        assert scrub_data["secrets_found"] is True
        assert fake_tok not in scrub_data["redacted_text"]


def test_manus_verify_project_root_confinement_and_cli_fail_closed(tmp_path: Path, monkeypatch, capsys):
    # Without allowed_project_root, microbrain_verify fails closed even when local tools are opted in
    srv_no_root = build_mcp_server(profile="manus", enable_local_tools=True)
    verify_fn_no_root = srv_no_root._tool_manager._tools["microbrain_verify"].fn
    res_no_root = json.loads(verify_fn_no_root(project_dir=str(tmp_path)))
    assert res_no_root["exit_code"] == 2
    assert "allowed_project_root" in res_no_root["error"]

    # With allowed_project_root set, escaping paths fail closed
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    srv_confined = build_mcp_server(
        profile="manus",
        enable_local_tools=True,
        allowed_project_root=str(root),
    )
    verify_fn = srv_confined._tool_manager._tools["microbrain_verify"].fn

    esc_rel = json.loads(verify_fn(project_dir="../outside"))
    assert esc_rel["exit_code"] == 2
    assert "escapes" in esc_rel["error"]

    esc_abs = json.loads(verify_fn(project_dir=str(outside)))
    assert esc_abs["exit_code"] == 2
    assert "escapes" in esc_abs["error"]

    esc_pol = json.loads(verify_fn(project_dir=".", policy=str(outside / "policy.json")))
    assert esc_pol["exit_code"] == 2
    assert "escapes" in esc_pol["error"]

    # CLI --check --profile manus reports 5 tools and passes
    assert mcp_main(["--check", "--profile", "manus"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["profile"] == "manus"
    assert report["tool_count"] == 5

    # Remote HTTP fails closed without Bearer token, even with --allow-unauthenticated-local on 0.0.0.0
    monkeypatch.delenv("AETHERIS_MCP_BEARER_TOKEN", raising=False)
    assert mcp_main(["--http", "--profile", "manus", "--host", "0.0.0.0"]) == 1
    assert mcp_main(["--http", "--profile", "manus", "--host", "0.0.0.0", "--allow-unauthenticated-local"]) == 1
