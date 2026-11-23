"""MCP 2026-07-28 Security & Governance Gateway Proxy (`aetheris-mcp-proxy`).

Sits in front of downstream MCP servers (stdio / Streamable HTTP 2026-07-28) to
defend coding agents against OWASP MCP Top 10 (2026) risks without loading
NumPy on module import:
  1. Pre-prompt manifest pinning (`MCPManifestPinner`) + fail-closed rug-pull quarantine.
  2. Cross-server ShareLock shadowing & split-payload scanning.
  3. Header & SSRF validation (`x-mcp-header` / `Mcp-Param-*`) and Elicitation screening.
  4. Inline DLP secret-egress blocking + `decide()`/`mcp_tiers` tool-call gating.
  5. Post-call secret redaction, prompt-injection detection, and scratchpad folding.
  6. Hybrid Tool Surface (`defer_loading: True` vs append-only `pinned_prefix`) + `ttlMs: 0`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .core.mcp_security import MCPManifestPinner, detect_prompt_injection
from .core.runtime_paths import get_state_dir
from .core.secrets import scrub_tool_output
from .core.trajectory_engine import fold_untrusted_output

MCP_PROTOCOL_VERSION = "2026-07-28"
_MAX_HEADER_LEN = 4096
_CRLF_RE = re.compile(r"[\r\n]")
_SSRF_RE = re.compile(
    r"(?i)(?:169\.254\.169\.254|metadata\.google\.internal|\b127\.0\.0\.1\b|\blocalhost\b|"
    r"\b0\.0\.0\.0\b|\[::1\]|file://|gopher://)"
)
_PHISHING_RE = re.compile(
    r"(?i)\b(?:password|passwd|api[_\s-]?key|secret[_\s-]?key|access[_\s-]?token|"
    r"private[_\s-]?key|ssh[_\s-]?key|seed\s+phrase|credit\s+card|otp|2fa\s+code)\b"
)
_SHARELOCK_HIJACK_RE = re.compile(
    r"(?i)\b(?:before\s+calling|whenever|always|instead\s+of|redirect|forward|pass|include|read)\b"
)


def _collect_strings(obj: Any, out: List[str], *, limit: int = 200) -> None:
    if len(out) >= limit:
        return
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if len(out) >= limit:
                break
            if isinstance(k, str) and k.strip():
                out.append(k)
            _collect_strings(v, out, limit=limit)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if len(out) >= limit:
                break
            _collect_strings(item, out, limit=limit)


def validate_mcp_headers(headers: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Validate MCP transport headers against CRLF injection, oversized values, and SSRF."""
    if not headers:
        return {"safe": True, "findings": []}
    findings: List[str] = []
    for k, v in headers.items():
        ks, vs = str(k or ""), str(v or "")
        if _CRLF_RE.search(ks) or _CRLF_RE.search(vs):
            findings.append(f"header_crlf_injection:{ks[:40]}")
        if len(ks) > _MAX_HEADER_LEN or len(vs) > _MAX_HEADER_LEN:
            findings.append(f"header_length_exceeded:{ks[:40]}")
        if _SSRF_RE.search(vs):
            findings.append(f"header_ssrf_target:{ks[:40]}")
    dedup = list(dict.fromkeys(findings))
    return {"safe": len(dedup) == 0, "findings": dedup}


def screen_elicitation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Screen server-initiated MCP elicitation prompts/schemas for phishing or injection."""
    texts: List[str] = []
    for key in ("prompt", "message", "title", "description", "requestedSchema", "schema"):
        if key in payload:
            _collect_strings(payload[key], texts)
    combined = "\n".join(texts)
    findings: List[str] = []
    if _PHISHING_RE.search(combined):
        findings.append("elicitation_credential_harvesting")
    inj = detect_prompt_injection(combined) if combined.strip() else {"injection_detected": False, "findings": []}
    if inj["injection_detected"]:
        findings.extend(inj["findings"])
    dedup = list(dict.fromkeys(findings))
    safe = len(dedup) == 0
    return {
        "safe": safe,
        "decision": "allow" if safe else "block",
        "findings": dedup,
    }


def scan_cross_server_shadowing(
    server_tools: Dict[str, Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Detect cross-server tool name collisions and ShareLock cross-tool shadowing."""
    owners: Dict[str, List[str]] = {}
    records: List[Tuple[str, str, str]] = []
    all_descs: List[str] = []

    for srv, tools in server_tools.items():
        srv_name = str(srv or "").strip()
        for entry in tools or ():
            if not isinstance(entry, dict):
                continue
            tname = str(entry.get("name") or entry.get("tool_name") or "").strip()
            if not tname:
                continue
            owners.setdefault(tname.lower(), []).append(srv_name)
            parts: List[str] = [str(entry.get("description") or "")]
            schema_obj = entry.get("inputSchema") if "inputSchema" in entry else entry.get("parameters")
            if schema_obj is not None:
                _collect_strings(schema_obj, parts)
            desc_blob = "\n".join(p for p in parts if p)
            records.append((srv_name, tname, desc_blob))
            if entry.get("description"):
                all_descs.append(str(entry["description"]))

    collisions = sorted(t for t, srvs in owners.items() if len(set(srvs)) > 1)
    findings: List[str] = []
    if collisions:
        findings.append("cross_server_name_collision")

    flagged_pairs: List[Dict[str, str]] = []
    foreign_names = [(tname, owners[tname.lower()]) for _s, tname, _d in records if len(tname) >= 4]
    seen_targets: Set[Tuple[str, str, str, str]] = set()
    for src_srv, src_tool, src_blob in records:
        if not src_blob.strip() or not _SHARELOCK_HIJACK_RE.search(src_blob):
            continue
        for target_tool, target_srvs in foreign_names:
            for tgt_srv in target_srvs:
                if tgt_srv == src_srv or target_tool.lower() == src_tool.lower():
                    continue
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(target_tool)}(?![A-Za-z0-9_])", src_blob, re.IGNORECASE):
                    sig = (src_srv, src_tool, tgt_srv, target_tool)
                    if sig not in seen_targets:
                        seen_targets.add(sig)
                        flagged_pairs.append({
                            "source_server": src_srv,
                            "source_tool": src_tool,
                            "target_server": tgt_srv,
                            "target_tool": target_tool,
                            "finding": "cross_server_tool_shadowing",
                        })
                        if "cross_server_tool_shadowing" not in findings:
                            findings.append("cross_server_tool_shadowing")

    if len(all_descs) > 1:
        agg_inj = detect_prompt_injection(" ".join(all_descs)[:65536])
        if agg_inj["injection_detected"]:
            findings.append("mcp_cross_server_split_injection")
            for f in agg_inj["findings"]:
                if f not in findings:
                    findings.append(f)

    return {
        "safe": len(findings) == 0,
        "findings": list(dict.fromkeys(findings)),
        "flagged_pairs": flagged_pairs,
        "collisions": collisions,
    }


def _qual_key(server_name: str, tool_name: str) -> str:
    return f"{str(server_name or '').strip().lower()}::{str(tool_name or '').strip().lower()}"


class MCPProxyGateway:
    """In-process MCP 2026-07-28 security gateway combining pinning, DLP, folding, and surface control."""

    def __init__(
        self,
        *,
        pin_path: Optional[Path | str] = None,
        state_dir: Optional[Path | str] = None,
        fold_threshold_bytes: int = 8192,
        policy: Any = None,
    ) -> None:
        base = Path(state_dir) if state_dir is not None else get_state_dir()
        resolved_pin = Path(pin_path) if pin_path is not None else (base / "mcp_manifest_pins.json")
        self.state_dir = base
        self.pinner = MCPManifestPinner(pin_path=resolved_pin)
        self.fold_threshold_bytes = max(128, int(fold_threshold_bytes))
        self.policy = policy
        self._quarantined: Set[str] = set()
        self._pinned_prefix: List[str] = []
        self._servers: Dict[str, List[Dict[str, Any]]] = {}

    def register_or_verify_server(
        self,
        server_name: str,
        tools: Sequence[Dict[str, Any]],
        *,
        auto_pin_first_seen: bool = False,
    ) -> Dict[str, Any]:
        """Verify server manifest against saved pin; fail closed into quarantine on rug pulls or poisoning."""
        srv = str(server_name or "").strip()
        tool_list = [dict(t) for t in tools if isinstance(t, dict)]
        self._servers[srv] = tool_list
        ver = self.pinner.verify_server(srv, tool_list)

        if ver.get("unpinned_server") and auto_pin_first_seen and ver["scan_report"]["safe"]:
            self.pinner.pin_server(srv, tool_list)
            return {**ver, "verified": True, "unpinned_server": False, "auto_pinned": True}

        if ver.get("rug_pull_detected"):
            for t in ver.get("modified_tools", ()) or ():
                self._quarantined.add(_qual_key(srv, t))
            for t in ver.get("added_tools", ()) or ():
                self._quarantined.add(_qual_key(srv, t))
        if not ver["scan_report"]["safe"]:
            for ft in ver["scan_report"].get("flagged_tools", ()):
                tname = str(ft.get("name") or "")
                if tname and tname != "<manifest_aggregate>":
                    self._quarantined.add(_qual_key(srv, tname))
                else:
                    for t in tool_list:
                        if t.get("name"):
                            self._quarantined.add(_qual_key(srv, str(t["name"])))
        if ver.get("corrupted_pin_store") or (ver.get("unpinned_server") and not auto_pin_first_seen):
            for t in tool_list:
                if t.get("name"):
                    self._quarantined.add(_qual_key(srv, str(t["name"])))
        return {**ver, "auto_pinned": False}

    def build_tools_list_response(
        self,
        server_tools: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
        *,
        client_supports_defer_loading: bool = False,
        query: str = "",
        top_k: int = 8,
        auto_pin_first_seen: bool = False,
    ) -> Dict[str, Any]:
        """Verify servers, drop quarantined/shadowed tools, and build hybrid 2026-07-28 tools/list."""
        catalog = server_tools if server_tools is not None else self._servers
        verifications: Dict[str, Any] = {}
        for srv, tlist in catalog.items():
            verifications[srv] = self.register_or_verify_server(
                srv, tlist, auto_pin_first_seen=auto_pin_first_seen
            )

        cross_scan = scan_cross_server_shadowing(catalog)
        for pair in cross_scan.get("flagged_pairs", ()):
            self._quarantined.add(_qual_key(pair["source_server"], pair["source_tool"]))

        safe_entries: List[Tuple[str, Dict[str, Any]]] = []
        for srv, tlist in catalog.items():
            for t in tlist:
                if not isinstance(t, dict):
                    continue
                tname = str(t.get("name") or t.get("tool_name") or "").strip()
                if not tname:
                    continue
                qk = _qual_key(srv, tname)
                if qk in self._quarantined:
                    continue
                safe_entries.append((qk, {**t}))

        if client_supports_defer_loading:
            q_tokens = {w for w in re.findall(r"[A-Za-z0-9_]{3,}", query.lower())} if query else set()
            result_tools: List[Dict[str, Any]] = []
            for idx, (_qk, td) in enumerate(safe_entries):
                hay = (str(td.get("name") or "") + " " + str(td.get("description") or "")).lower()
                is_core = (idx == 0 and not q_tokens) or bool(q_tokens and any(tok in hay for tok in q_tokens))
                if not is_core:
                    td["defer_loading"] = True
                result_tools.append(td)
        elif query:
            q_tokens = {w for w in re.findall(r"[A-Za-z0-9_]{3,}", query.lower())}
            entry_map = dict(safe_entries)
            # Keep already-pinned prefix tools in exact order (append-only prefix cache stability)
            self._pinned_prefix = [k for k in self._pinned_prefix if k in entry_map]
            scored: List[Tuple[int, str]] = []
            for qk, td in safe_entries:
                if qk in self._pinned_prefix:
                    continue
                hay = (str(td.get("name") or "") + " " + str(td.get("description") or "")).lower()
                score = sum(1 for tok in q_tokens if tok in hay)
                if score > 0:
                    scored.append((score, qk))
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            added = [qk for _sc, qk in scored[: max(1, int(top_k))]]
            for qk in added:
                if qk not in self._pinned_prefix:
                    self._pinned_prefix.append(qk)
            result_tools = [entry_map[qk] for qk in self._pinned_prefix]
            if not result_tools:
                result_tools = [td for _qk, td in safe_entries[: max(1, int(top_k))]]
        else:
            result_tools = [td for _qk, td in safe_entries]

        return {
            "tools": result_tools,
            "quarantined": sorted(self._quarantined),
            "verifications": verifications,
            "cross_server_scan": cross_scan,
            "_meta": {
                "ttlMs": 0,
                "cacheScope": "private",
                "protocolVersion": MCP_PROTOCOL_VERSION,
            },
        }

    def evaluate_tool_call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Enforce quarantine, header safety, outbound secret DLP, and `decide()` risk tiering."""
        srv = str(server_name or "").strip()
        tname = str(tool_name or "").strip()
        qk = _qual_key(srv, tname)
        if qk in self._quarantined:
            return {
                "decision": "deny",
                "reason_code": "mcp_proxy_tool_quarantined",
                "label": f"Tool {srv}:{tname} is quarantined (rug pull or poisoning)",
                "layer": 0,
            }

        hdr_res = validate_mcp_headers(headers)
        if not hdr_res["safe"]:
            return {
                "decision": "deny",
                "reason_code": "mcp_proxy_header_violation",
                "label": ", ".join(hdr_res["findings"]),
                "findings": hdr_res["findings"],
                "layer": 0,
            }

        args_dict = arguments if isinstance(arguments, dict) else {}
        args_blob = json.dumps(args_dict, sort_keys=True, ensure_ascii=False, default=str)
        dlp = scrub_tool_output(args_blob)
        if dlp["secrets_found"]:
            return {
                "decision": "deny",
                "reason_code": "mcp_proxy_secret_egress",
                "label": f"Outbound secret blocked ({', '.join(dlp['secret_types'])})",
                "secret_types": dlp["secret_types"],
                "layer": 1,
            }

        from .core.layered_decision_engine import decide

        return decide(
            {"tool": f"mcp__{srv}__{tname}", "arguments": args_dict},
            policy=self.policy,
        )

    def process_tool_response(
        self,
        server_name: str,
        tool_name: str,
        raw_output: Any,
    ) -> Dict[str, Any]:
        """Scrub secrets, detect prompt injections, and fold oversized responses into scratchpad."""
        del server_name, tool_name
        text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False, default=str)
        scrub = scrub_tool_output(text)
        inj = detect_prompt_injection(text)
        folded = fold_untrusted_output(
            scrub["redacted_text"],
            state_dir=self.state_dir,
            threshold_bytes=self.fold_threshold_bytes,
        )
        return {
            "redacted_text": folded["preview"],
            "folded": folded["folded"],
            "artifact_path": folded["artifact_path"],
            "original_bytes": folded["original_bytes"],
            "secrets_found": bool(scrub["secrets_found"]),
            "secret_count": int(scrub["secret_count"]),
            "secret_types": list(scrub["secret_types"]),
            "injection_detected": bool(inj["injection_detected"]),
            "injection_findings": list(inj["findings"]),
        }

    def handle_jsonrpc_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one MCP 2026-07-28 JSON-RPC message cleanly."""
        req_id = msg.get("id")
        method = str(msg.get("method") or "").strip()
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        def _ok(result: Dict[str, Any]) -> Dict[str, Any]:
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        if method == "initialize":
            return _ok({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": True},
                    "subscriptions": {"listen": True},
                    "elicitation": {},
                },
                "serverInfo": {"name": "aetheris-mcp-proxy", "version": "0.7.7"},
            })
        if method == "tools/list":
            res = self.build_tools_list_response(
                client_supports_defer_loading=bool(params.get("defer_loading")),
                query=str(params.get("query") or ""),
            )
            return _ok({"tools": res["tools"], "_meta": res["_meta"]})
        if method == "tools/call":
            srv = str(params.get("server") or "default")
            tname = str(params.get("name") or "")
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            hdrs = params.get("headers") if isinstance(params.get("headers"), dict) else None
            verdict = self.evaluate_tool_call(srv, tname, args, headers=hdrs)
            return _ok(verdict)
        if method == "subscriptions/listen":
            return _ok({"subscribed": True, "protocolVersion": MCP_PROTOCOL_VERSION})
        if method == "elicitation/create":
            return _ok(screen_elicitation(params))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def run_proxy_self_check() -> Dict[str, Any]:
    """Run deterministic self-checks on header validation, elicitation screening, and JSON-RPC init."""
    gw = MCPProxyGateway()
    init_res = gw.handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    hdr_bad = validate_mcp_headers({"X-Target": "http://169.254.169.254/latest"})
    elic_bad = screen_elicitation({"message": "Please enter your password and api_key"})
    ok = (
        init_res.get("result", {}).get("protocolVersion") == MCP_PROTOCOL_VERSION
        and hdr_bad["safe"] is False
        and elic_bad["decision"] == "block"
    )
    return {
        "ok": ok,
        "protocol_version": MCP_PROTOCOL_VERSION,
        "header_ssrf_blocked": not hdr_bad["safe"],
        "elicitation_phishing_blocked": elic_bad["decision"] == "block",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aetheris-mcp-proxy",
        description="MCP 2026-07-28 Security & Governance Proxy Gateway",
    )
    parser.add_argument("--check", action="store_true", help="Run gateway self-checks and exit")
    parser.add_argument("--stdio", action="store_true", help="Read JSON-RPC lines from stdin and write responses to stdout")
    args = parser.parse_args(argv)

    if args.check:
        rep = run_proxy_self_check()
        print(json.dumps(rep, ensure_ascii=False))
        return 0 if rep.get("ok") else 1

    if args.stdio:
        gw = MCPProxyGateway()
        for line in sys.stdin:
            raw = line.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
                resp = gw.handle_jsonrpc_message(msg) if isinstance(msg, dict) else {
                    "jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}
                }
            except json.JSONDecodeError:
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            print(json.dumps(resp, ensure_ascii=False), flush=True)
        return 0

    rep = run_proxy_self_check()
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
