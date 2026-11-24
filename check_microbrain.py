"""Quick health and MCP handshake check for the local Microbrain HTTP server."""

from __future__ import annotations

import json
import sys
import urllib.request


def check_server(base_url: str = "http://127.0.0.1:8080") -> int:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        if "tool_count" in health:
            detail = f"tools={health.get('tool_count')}"
        else:
            detail = f"brains={len(health.get('brains_loaded', []))}"
        print(f"[ok] GET {base_url}/health -> {health.get('status')} ({detail})")
    except Exception as exc:
        print(f"[FAIL] GET {base_url}/health failed: {exc}", file=sys.stderr)
        return 1

    init_body = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "check_microbrain", "version": "1.0"},
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/mcp",
        data=init_body,
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        if "aetheris-microbrains" not in raw:
            print(f"[FAIL] POST {base_url}/mcp unexpected body: {raw[:200]}", file=sys.stderr)
            return 1
        print(f"[ok] POST {base_url}/mcp -> HTTP {resp.status} (MCP initialize handshake OK)")
    except Exception as exc:
        print(f"[FAIL] POST {base_url}/mcp failed: {exc}", file=sys.stderr)
        return 1

    print("MICROBRAIN SERVER & MCP CHECK PASSED")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    raise SystemExit(check_server(target.rstrip("/")))
