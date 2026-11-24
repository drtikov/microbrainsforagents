"""Quick health and MCP handshake check for the local Microbrain HTTP server.

Supports both server builds:

  - `aetheris-mcp --http` (SDK-based MCP server): exposes both GET /health and
    POST /mcp (plus the `microbrain_health` MCP tool).
  - `python -m aetheris_microbrains.core.microbrain_server` (FastAPI core):
    exposes GET /health + /microbrain/* only, no POST /mcp.

Exit code 0 when the MCP initialize handshake succeeds (full check; a
best-effort `microbrain_health` tool call is attempted when the server issues
a session id), or when only the core /health endpoint responds (reported as a
health-only pass). Exit code 1 when neither endpoint responds.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

MCP_INIT_BODY = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "check_microbrain", "version": "1.0"},
    },
}).encode("utf-8")

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def check_health(base_url: str) -> bool:
    """Probe GET /health. Returns True when healthy."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        if "tool_count" in health:
            detail = f"tools={health.get('tool_count')}"
        else:
            detail = f"brains={len(health.get('brains_loaded', []))}"
        print(f"[ok] GET {base_url}/health -> {health.get('status')} ({detail})")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"[--] GET {base_url}/health -> 404 (no HTTP /health endpoint on this server)")
            return False
        print(f"[FAIL] GET {base_url}/health failed: HTTP {exc.code}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[FAIL] GET {base_url}/health failed: {exc}", file=sys.stderr)
        return False


def mcp_request(base_url: str, body: bytes, session_id: str = "") -> tuple[int, str, str]:
    """POST one JSON-RPC message to /mcp; return (status, raw body, session id)."""
    headers = dict(MCP_HEADERS)
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(f"{base_url}/mcp", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace"), resp.headers.get("mcp-session-id", "")


def check_mcp_handshake(base_url: str) -> bool:
    """Probe the MCP initialize handshake plus a best-effort health tool call."""
    try:
        status, raw, session_id = mcp_request(base_url, MCP_INIT_BODY)
    except Exception as exc:
        print(f"[--] POST {base_url}/mcp failed: {exc} (core server build; MCP runs via `aetheris-mcp --http`)")
        return False

    if "aetheris-microbrains" not in raw:
        print(f"[FAIL] POST {base_url}/mcp unexpected body: {raw[:200]}", file=sys.stderr)
        return False
    print(f"[ok] POST {base_url}/mcp -> HTTP {status} (MCP initialize handshake OK)")

    if not session_id:
        return True

    # Best-effort: complete the session and call the health tool (informational only).
    try:
        initialized = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode("utf-8")
        mcp_request(base_url, initialized, session_id)
        call = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "microbrain_health", "arguments": {}},
        }).encode("utf-8")
        _, raw, _ = mcp_request(base_url, call, session_id)
        if "microbrain" in raw:
            print("[ok] MCP tools/call microbrain_health -> OK")
        else:
            print(f"[warn] MCP microbrain_health call returned unexpected body: {raw[:120]}")
    except Exception as exc:
        print(f"[warn] MCP health tool call skipped: {exc}")
    return True


def check_server(base_url: str = "http://127.0.0.1:8080") -> int:
    """Check /health and /mcp on base_url, supporting both full MCP and core-only builds."""
    health_ok = check_health(base_url)
    mcp_ok = check_mcp_handshake(base_url)

    if mcp_ok:
        print("MICROBRAIN SERVER & MCP CHECK PASSED")
        return 0
    if health_ok:
        print("MICROBRAIN HEALTH CHECK PASSED (health only; no MCP endpoint on this port)")
        return 0
    print("MICROBRAIN SERVER CHECK FAILED: neither /health nor /mcp responded", file=sys.stderr)
    return 1


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    raise SystemExit(check_server(target.rstrip("/")))
