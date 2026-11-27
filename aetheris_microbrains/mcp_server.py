"""Native MCP Server for Aetheris Microbrains (stdio + Streamable HTTP / SSE).

Exposes Microbrains pre-flight policy checks, secret/injection scanning, MCP supply-chain
scanning, cache-aware tool narrowing, journal inspection, procedural directive vault,
and post-policy verification directly in-process (zero loopback HTTP proxy overhead).

Usage:
    python -m aetheris_microbrains.mcp_server                                   # stdio (default; Hermes/Claude)
    python -m aetheris_microbrains.mcp_server --http --port 8080                 # local Streamable HTTP (/mcp)
    python -m aetheris_microbrains.mcp_server --http --profile manus --port 8080 # authenticated remote Manus profile
    aetheris-mcp                                                                 # console entrypoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_VALID_MCP_PROFILES = ("local", "manus", "remote")
MANUS_STATELESS_TOOLS: Tuple[str, ...] = (
    "microbrain_health",
    "microbrain_preflight_check",
    "microbrain_scrub_and_scan",
    "microbrain_scan_mcp_tools",
    "microbrain_narrow_tools",
)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _normalize_profile(profile: str) -> str:
    resolved = str(profile or "local").strip().lower()
    if resolved not in _VALID_MCP_PROFILES:
        raise ValueError(f"Unsupported MCP profile {profile!r}; expected one of {_VALID_MCP_PROFILES}")
    return resolved


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        return candidate.resolve().is_relative_to(root.resolve())
    except Exception:
        return False


def build_mcp_server(
    name: str = "aetheris-microbrains",
    *,
    streamable_http_path: str = "/mcp",
    host: str = "127.0.0.1",
    port: int = 8080,
    profile: str = "local",
    enable_local_tools: bool = False,
    allowed_project_root: Optional[str] = None,
    json_response: bool = False,
) -> Any:
    """Construct and register tools on an MCP server instance (supports FastMCP on mcp<2 and MCPServer on mcp>=2)."""
    resolved_profile = _normalize_profile(profile)
    is_remote = resolved_profile in {"manus", "remote"}
    include_local_tools = (not is_remote) or bool(enable_local_tools)
    confined_root = Path(allowed_project_root).resolve() if allowed_project_root else None

    try:
        from mcp.server.fastmcp import FastMCP
        server = FastMCP(
            name,
            host=host,
            port=int(port),
            streamable_http_path=streamable_http_path,
            json_response=bool(json_response),
        )
    except ModuleNotFoundError:
        from mcp.server.mcpserver import MCPServer
        server = MCPServer(name)

    @server.tool()
    def microbrain_health() -> str:
        """Check Aetheris Microbrains health, profile, and server version."""
        from . import __version__

        if is_remote:
            return _json_dumps({
                "status": "ok",
                "name": "aetheris-microbrains",
                "version": __version__,
                "profile": resolved_profile,
                "advisory_only": True,
            })

        from .core.runtime_paths import get_default_journal_path, get_state_dir
        from .hook_daemon import ping_daemon

        daemon_info = ping_daemon(timeout=0.15)
        return _json_dumps({
            "status": "ok",
            "name": "aetheris-microbrains",
            "version": __version__,
            "profile": resolved_profile,
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

        # Remote callers cannot supply arbitrary filesystem paths for session state.
        effective_session_path = None if is_remote else (session_state_path or None)
        verdict = evaluate_hook_payload(
            {"tool_name": tool_name, "tool_input": tool_input},
            strict_ast=bool(strict_ast),
            profile=profile or "strict",
            session_state_path=effective_session_path,
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

    if include_local_tools:
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
            from .verify import chain_exit_code, run_chain

            if is_remote and confined_root is None:
                return _json_dumps({
                    "exit_code": 2,
                    "error": "microbrain_verify requires allowed_project_root in remote profile",
                })

            if confined_root is not None:
                raw_dir = str(project_dir or "").strip()
                if not raw_dir:
                    proj = confined_root
                else:
                    candidate = Path(raw_dir)
                    proj = candidate.resolve() if candidate.is_absolute() else (confined_root / candidate).resolve()
                if not _is_within_root(proj, confined_root):
                    return _json_dumps({
                        "exit_code": 2,
                        "error": "project_dir escapes allowed_project_root",
                    })
                if policy:
                    p_cand = Path(policy)
                    p_resolved = p_cand.resolve() if p_cand.is_absolute() else (confined_root / p_cand).resolve()
                    if not _is_within_root(p_resolved, confined_root):
                        return _json_dumps({
                            "exit_code": 2,
                            "error": "policy path escapes allowed_project_root",
                        })
                    policy = str(p_resolved)
            else:
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

    if hasattr(server, "custom_route"):
        @server.custom_route("/health", methods=["GET"])
        async def _http_health(_request: Any) -> Any:
            from starlette.responses import JSONResponse
            from . import __version__

            tools_registered = sorted(
                getattr(getattr(server, "_tool_manager", None), "_tools", {}).keys()
            )
            return JSONResponse({
                "status": "ok",
                "server": name,
                "version": __version__,
                "profile": resolved_profile,
                "tools_registered": tools_registered,
                "tool_count": len(tools_registered),
            })

    return server


def build_streamable_http_app(
    name: str = "aetheris-microbrains",
    *,
    streamable_http_path: str = "/mcp",
    host: str = "127.0.0.1",
    port: int = 8080,
    profile: str = "local",
    enable_local_tools: bool = False,
    allowed_project_root: Optional[str] = None,
    bearer_token: Optional[str] = None,
    protect_health: bool = False,
    json_response: bool = False,
) -> Tuple[Any, Any]:
    """Build the FastMCP server and its Streamable HTTP ASGI app (optionally wrapped with BearerAuthMiddleware)."""
    server = build_mcp_server(
        name=name,
        streamable_http_path=streamable_http_path,
        host=host,
        port=port,
        profile=profile,
        enable_local_tools=enable_local_tools,
        allowed_project_root=allowed_project_root,
        json_response=json_response,
    )
    app = server.streamable_http_app()
    if bearer_token:
        from .manus_auth import BearerAuthMiddleware

        app = BearerAuthMiddleware(
            app,
            token=bearer_token,
            path=streamable_http_path,
            protect_health=protect_health,
        )
    return server, app


def run_mcp_self_check(*, profile: str = "local") -> Dict[str, Any]:
    """Verify MCP SDK importability, tool registration, health probe, and preflight verdicts."""
    from .hook import evaluate_hook_payload
    from .core.runtime_paths import get_state_dir

    resolved_profile = _normalize_profile(profile)
    is_remote = resolved_profile in {"manus", "remote"}
    server = build_mcp_server(profile=resolved_profile)
    tools = sorted(getattr(getattr(server, "_tool_manager", None), "_tools", {}).keys())
    deny_probe = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        host="generic",
    )
    allow_probe = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        host="generic",
    )
    expected_tools = list(MANUS_STATELESS_TOOLS)
    tools_ok = (
        (set(tools) == set(expected_tools))
        if is_remote
        else (len(tools) >= 8)
    )
    ok = (
        tools_ok
        and deny_probe.get("decision") == "deny"
        and allow_probe.get("decision") == "allow"
    )
    out: Dict[str, Any] = {
        "ok": ok,
        "profile": resolved_profile,
        "tools_registered": tools,
        "tool_count": len(tools),
        "probes": {
            "rm_rf_root": deny_probe.get("decision"),
            "git_status": allow_probe.get("decision"),
        },
    }
    if not is_remote:
        out["state_dir"] = str(get_state_dir())
    return out


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
        "--profile",
        choices=list(_VALID_MCP_PROFILES),
        default="local",
        help="Server exposure profile: local (all tools) or manus/remote (5-tool advisory allowlist)",
    )
    parser.add_argument(
        "--auth-token-env",
        default="AETHERIS_MCP_BEARER_TOKEN",
        help="Env var holding Bearer token for Streamable HTTP authentication",
    )
    parser.add_argument(
        "--protect-health",
        action="store_true",
        help="Also require Bearer auth on GET /health",
    )
    parser.add_argument(
        "--enable-local-tools",
        action="store_true",
        help="Opt into local-state tools (audit/directives/verify) under manus/remote profile",
    )
    parser.add_argument(
        "--allowed-project-root",
        default="",
        help="Confine microbrain_verify paths inside this directory",
    )
    parser.add_argument(
        "--allow-unauthenticated-local",
        action="store_true",
        help="Allow unauthenticated HTTP startup on loopback only under manus/remote profile",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        help="Return JSON HTTP responses instead of SSE streams where supported",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify MCP SDK availability, registered tools, and pre-flight probe verdicts then exit",
    )
    args = parser.parse_args(argv)

    if args.check:
        try:
            report = run_mcp_self_check(profile=args.profile)
        except Exception as exc:
            print(_json_dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
            return 1
        print(_json_dumps(report))
        return 0 if report.get("ok") else 1

    transport = "streamable-http" if args.http else args.transport
    is_remote_profile = args.profile in {"manus", "remote"}
    from .manus_auth import is_loopback_host, resolve_bearer_token

    bearer_token = resolve_bearer_token(args.auth_token_env)
    if transport in {"streamable-http", "sse"} and is_remote_profile and not bearer_token:
        if not (args.allow_unauthenticated_local and is_loopback_host(args.host)):
            print(
                f"Refusing unauthenticated {args.profile!r} HTTP server on {args.host}:{args.port}. "
                f"Set {args.auth_token_env} (or pass --allow-unauthenticated-local on loopback).",
                file=sys.stderr,
            )
            return 1

    try:
        if transport == "streamable-http" and (bearer_token or is_remote_profile):
            server, app = build_streamable_http_app(
                host=args.host,
                port=args.port,
                profile=args.profile,
                enable_local_tools=bool(args.enable_local_tools),
                allowed_project_root=args.allowed_project_root or None,
                bearer_token=bearer_token,
                protect_health=bool(args.protect_health),
                json_response=bool(args.json_response),
            )
        else:
            server = build_mcp_server(
                host=args.host,
                port=args.port,
                profile=args.profile,
                enable_local_tools=bool(args.enable_local_tools),
                allowed_project_root=args.allowed_project_root or None,
                json_response=bool(args.json_response),
            )
            app = None
    except ImportError as exc:
        print(
            f"MCP SDK is required to run the MCP server ({exc}). Install with: pip install 'aetheris-microbrains[mcp]'",
            file=sys.stderr,
        )
        return 1

    if app is not None and transport == "streamable-http":
        import uvicorn

        uvicorn.run(app, host=args.host, port=int(args.port))
        return 0

    server.run(transport=transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
