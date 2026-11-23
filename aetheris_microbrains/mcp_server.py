"""Native MCP Server for Aetheris Microbrains (stdio + Streamable HTTP / SSE).

Exposes Microbrains pre-flight policy checks, secret/injection scanning, MCP supply-chain
scanning, cache-aware tool narrowing, journal inspection, procedural directive vault,
and post-policy verification directly in-process (zero loopback HTTP proxy overhead).

Usage:
    python -m aetheris_microbrains.mcp_server                       # stdio (default; Hermes/Claude)
    python -m aetheris_microbrains.mcp_server --http --port 8080     # Streamable HTTP (/mcp)
    aetheris-mcp                                                     # console entrypoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def build_mcp_server(
    name: str = "aetheris-microbrains",
    *,
    streamable_http_path: str = "/mcp",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> Any:
    """Construct and register tools on an MCP server instance (supports FastMCP on mcp<2 and MCPServer on mcp>=2)."""
    try:
        from mcp.server.fastmcp import FastMCP
        server = FastMCP(
            name,
            host=host,
            port=int(port),
            streamable_http_path=streamable_http_path,
        )
    except ModuleNotFoundError:
        from mcp.server.mcpserver import MCPServer
        server = MCPServer(name)

    @server.tool()
    def microbrain_health() -> str:
        """Check Aetheris Microbrains health, version, state directory, and optional warm daemon status."""
        from . import __version__
        from .core.runtime_paths import get_default_journal_path, get_state_dir
        from .hook_daemon import ping_daemon

        daemon_info = ping_daemon(timeout=0.15)
        return _json_dumps({
            "status": "ok",
            "name": "aetheris-microbrains",
            "version": __version__,
            "state_dir": str(get_state_dir()),
            "journal_path": str(get_default_journal_path()),
            "daemon": daemon_info or {"running": False},
        })

    @server.tool()
    def microbrain_preflight_check(
        tool_name: str,
        command: str = "",
        file_path: str = "",
        diff: str = "",
        old_string: str = "",
        new_string: str = "",
        profile: str = "strict",
        strict_ast: bool = False,
        session_state_path: str = "",
    ) -> str:
        """Pre-flight check a tool call (shell command, file edit, or read) before execution."""
        from .hook import evaluate_hook_payload

        tool_input: Dict[str, Any] = {}
        if command:
            tool_input["command"] = command
        if file_path:
            tool_input["file_path"] = file_path
        if diff:
            tool_input["diff"] = diff
        if old_string:
            tool_input["old_string"] = old_string
        if new_string:
            tool_input["new_string"] = new_string

        verdict = evaluate_hook_payload(
            {"tool_name": tool_name, "tool_input": tool_input},
            strict_ast=bool(strict_ast),
            profile=profile or "strict",
            session_state_path=session_state_path or None,
            host="generic",
        )
        return _json_dumps({
            "decision": verdict.get("decision"),
            "permissionDecision": verdict.get("permissionDecision"),
            "reason_code": verdict.get("reason_code"),
            "label": verdict.get("label"),
            "layer": verdict.get("layer"),
            "latency_ms": verdict.get("latency_ms"),
            "exit_code": verdict.get("exit_code"),
        })

    @server.tool()
    def microbrain_scrub_and_scan(text: str) -> str:
        """Scrub leaked secrets/credentials and detect prompt-injection smuggling in untrusted text."""
        from .core.mcp_security import detect_prompt_injection
        from .core.secrets import scrub_tool_output

        scrubbed = scrub_tool_output(text or "")
        inj = detect_prompt_injection(text or "")
        return _json_dumps({
            "secrets_found": scrubbed.get("secrets_found", False),
            "secret_count": scrubbed.get("secret_count", 0),
            "secret_types": scrubbed.get("secret_types", []),
            "redacted_text": scrubbed.get("redacted_text", text),
            "injection_detected": inj.get("injection_detected", False),
            "injection_severity": inj.get("severity", "none"),
            "injection_findings": inj.get("findings", []),
        })

    @server.tool()
    def microbrain_scan_mcp_tools(tools: List[Dict[str, Any]]) -> str:
        """Audit MCP tool schema manifests for poisoning, tool shadowing, or split-payload smuggling."""
        from .core.mcp_security import scan_mcp_tools

        return _json_dumps(scan_mcp_tools(tools or []))

    @server.tool()
    def microbrain_narrow_tools(
        query: str,
        tools: List[Dict[str, Any]],
        top_k: int = 6,
    ) -> str:
        """Select the highest-relevance subset of tool schemas while preserving namespace stability."""
        from .core.tool_routing_moe import narrow_tool_surface

        return _json_dumps(narrow_tool_surface(query, tools or [], top_k=max(1, int(top_k))))

    @server.tool()
    def microbrain_audit_explain(limit: int = 5, sha_prefix: str = "") -> str:
        """Inspect recent decision journal events and explain why an action was blocked or escalated."""
        from .audit import explain_entry, get_journal_entries

        entries = get_journal_entries(limit=max(1, int(limit)))
        if not entries:
            return _json_dumps({"entries": [], "explanation": "No decision journal entries found."})
        target = entries[-1]
        if sha_prefix:
            for e in reversed(entries):
                if str(e.get("input_sha256") or "").startswith(sha_prefix):
                    target = e
                    break
        return _json_dumps({
            "count": len(entries),
            "latest": target,
            "explanation": explain_entry(target),
        })

    @server.tool()
    def microbrain_directives(
        action: str = "select",
        token_budget: int = 800,
        scope_path: str = "",
        directive_id: str = "",
        guidance: str = "",
        scope_glob: str = "*",
        author: str = "hermes",
    ) -> str:
        """Query approved operational directives or stage a pending directive proposal (never auto-approves)."""
        from .core.directive_vault import DirectiveVault

        vault = DirectiveVault()
        act = str(action or "select").strip().lower()
        if act == "propose":
            if not directive_id or not guidance:
                return _json_dumps({"error": "propose requires directive_id and guidance"})
            card = vault.propose(
                directive_id,
                guidance=guidance,
                scope_glob=scope_glob,
                author=author,
            )
            return _json_dumps({"status": "pending_approval", "directive": card.to_dict()})
        return _json_dumps(vault.select_for_bootstrap(token_budget=int(token_budget), scope_path=scope_path))

    @server.tool()
    def microbrain_verify(
        project_dir: str = "",
        policy: str = "",
        skip_self_test: bool = True,
    ) -> str:
        """Run doctor + optional candidate policy replay verification chain."""
        from pathlib import Path
        from .verify import chain_exit_code, run_chain

        proj = Path(project_dir) if project_dir else Path(os.getcwd())
        results = run_chain(
            project_dir=proj,
            policy=policy,
            skip_self_test=bool(skip_self_test),
            verbose=False,
        )
        sanitized = [
            {k: v for k, v in r.items() if k not in ("doctor_report", "eval_report")}
            for r in results
        ]
        return _json_dumps({
            "exit_code": chain_exit_code(results),
            "results": sanitized,
        })

    return server


def run_mcp_self_check() -> Dict[str, Any]:
    """Verify MCP SDK importability, tool registration, health probe, and preflight verdicts."""
    from .hook import evaluate_hook_payload
    from .core.runtime_paths import get_state_dir

    server = build_mcp_server()
    tools = list(getattr(getattr(server, "_tool_manager", None), "_tools", {}).keys())
    deny_probe = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        host="generic",
    )
    allow_probe = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        host="generic",
    )
    ok = (
        len(tools) >= 8
        and deny_probe.get("decision") == "deny"
        and allow_probe.get("decision") == "allow"
    )
    return {
        "ok": ok,
        "tools_registered": sorted(tools),
        "tool_count": len(tools),
        "state_dir": str(get_state_dir()),
        "probes": {
            "rm_rf_root": deny_probe.get("decision"),
            "git_status": allow_probe.get("decision"),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aetheris-mcp",
        description="Run the Aetheris Microbrains MCP server (stdio or Streamable HTTP).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Alias for `--transport streamable-http`",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP bind port (default: 8080)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify MCP SDK availability, registered tools, and pre-flight probe verdicts then exit",
    )
    args = parser.parse_args(argv)

    if args.check:
        try:
            report = run_mcp_self_check()
        except Exception as exc:
            print(_json_dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
            return 1
        print(_json_dumps(report))
        return 0 if report.get("ok") else 1

    transport = "streamable-http" if args.http else args.transport
    try:
        server = build_mcp_server(host=args.host, port=args.port)
    except ImportError as exc:
        print(
            f"MCP SDK is required to run the MCP server ({exc}). Install with: pip install 'aetheris-microbrains[mcp]'",
            file=sys.stderr,
        )
        return 1

    server.run(transport=transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
