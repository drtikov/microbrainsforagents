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
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .core.mcp_security import MCPManifestPinner, detect_prompt_injection
from .core.runtime_paths import get_state_dir
from .core.secret_broker import SecretBroker
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


_EMBODIED_EXACT_TOOLS: Set[str] = {
    "ros2",
    "moveit",
    "moveit2",
    "mavlink",
    "lerobot",
    "publish_cmd_vel",
    "cmd_vel",
    "gripper_command",
    "follow_joint_trajectory",
    "navigate_to_pose",
}
_EMBODIED_SUFFIXES: Tuple[str, ...] = (
    "_publish_cmd_vel",
    "_gripper_command",
    "_follow_joint_trajectory",
    "_navigate_to_pose",
)


def is_embodied_mcp_tool(
    tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True when an MCP tool name/payload targets embodied robot control."""
    raw = str(tool_name or "").strip().lower()
    if raw.startswith("mcp__") and raw.count("__") >= 2:
        raw = raw.split("__", 2)[2]
    if (
        raw in _EMBODIED_EXACT_TOOLS
        or raw.startswith(("ros2_", "ros2-", "moveit", "mavlink_", "mavlink-", "lerobot_", "lerobot-", "unitree_", "unitree-", "sport_client_", "loco_client_", "dex3_", "rt_lowcmd"))
        or raw.endswith(_EMBODIED_SUFFIXES)
    ):
        return True
    if isinstance(arguments, Mapping):
        topic = str(arguments.get("topic") or "").strip().lower()
        if topic in ("/cmd_vel", "cmd_vel") or (
            isinstance(arguments.get("twist"), Mapping) and "linear" in arguments["twist"]
        ):
            return True
    return False


def _infer_embodied_protocol_and_payload(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    raw = str(tool_name or "").strip().lower()
    if raw.startswith("mcp__") and raw.count("__") >= 2:
        raw = raw.split("__", 2)[2]

    explicit_proto = str(arguments.get("protocol") or "").strip().lower()
    if explicit_proto in ("ros2", "moveit2", "lerobot", "mavlink", "unitree"):
        protocol = explicit_proto
    elif raw.startswith(("unitree_", "unitree-", "sport_client_", "loco_client_", "dex3_", "rt_lowcmd")):
        protocol = "unitree"
    elif raw.startswith("moveit"):
        protocol = "moveit2"
    elif raw.startswith(("mavlink_", "mavlink-")) or raw == "mavlink":
        protocol = "mavlink"
    elif raw.startswith(("lerobot_", "lerobot-")) or raw == "lerobot":
        protocol = "lerobot"
    else:
        protocol = "ros2"

    payload = dict(arguments)
    has_action_hint = any(
        payload.get(k) is not None for k in ("topic_or_action", "topic", "action", "message_type", "msg_type")
    )
    if not has_action_hint:
        if "cmd_vel" in raw:
            payload["topic_or_action"] = "/cmd_vel"
        elif "gripper" in raw:
            payload["action"] = "GripperCommand"
        elif "follow_joint_trajectory" in raw:
            payload["action"] = "FollowJointTrajectory"
        elif "navigate_to_pose" in raw:
            payload["action"] = "NavigateToPose"
        elif raw:
            payload["topic_or_action"] = raw
    return protocol, payload


def _sanitize_for_flight_log(value: Any) -> Any:
    """Convert non-finite numbers and exotic objects into JSON-safe primitives for FlightDataRecorder."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(k): _sanitize_for_flight_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_flight_log(v) for v in value]
    return str(value)


class StdioMCPDownstreamClient:
    """Line-delimited JSON-RPC stdio client for a downstream MCP child server."""

    def __init__(self, command: Sequence[str], *, cwd: Optional[Path | str] = None) -> None:
        argv = [str(x) for x in command if str(x)]
        if not argv:
            raise ValueError("downstream stdio command must be non-empty")
        self.command = argv
        self.cwd = str(cwd) if cwd is not None else None
        self._proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(  # aetheris: allow subprocess_call:Popen -- spawn downstream stdio MCP server
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self.cwd,
        )

    def send_rpc(self, req: Dict[str, Any]) -> Dict[str, Any]:
        self.start()
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Downstream MCP server closed stdout")
        resp = json.loads(line)
        if not isinstance(resp, dict):
            raise RuntimeError("Downstream MCP server returned non-object JSON-RPC")
        return resp

    def fetch_tools(self) -> List[Dict[str, Any]]:
        self.send_rpc({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": MCP_PROTOCOL_VERSION}})
        res = self.send_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        tools = res.get("result", {}).get("tools", []) if isinstance(res.get("result"), dict) else []
        return [t for t in tools if isinstance(t, dict)]

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except OSError:
                pass
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None


class MCPProxyGateway:
    """In-process MCP 2026-07-28 security gateway combining pinning, secret vault, DLP, folding, and surface control."""

    def __init__(
        self,
        *,
        pin_path: Optional[Path | str] = None,
        state_dir: Optional[Path | str] = None,
        fold_threshold_bytes: int = 8192,
        policy: Any = None,
        secret_broker: Optional[SecretBroker] = None,
        downstream_transports: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
        block_response_injections: bool = False,
        embodied_mode: bool = False,
        embodied_hard_limits: Optional[Dict[str, Any]] = None,
        mission_envelope: Optional[Any] = None,
        mission_secret: Optional[bytes | str] = None,
        capability_token: Optional[Any] = None,
        root_secret: Optional[bytes | str] = None,
        sensor_verifier: Optional[Any] = None,
        flight_recorder: Optional[Any] = None,
    ) -> None:
        base = Path(state_dir) if state_dir is not None else get_state_dir()
        resolved_pin = Path(pin_path) if pin_path is not None else (base / "mcp_manifest_pins.json")
        self.state_dir = base
        self.pinner = MCPManifestPinner(pin_path=resolved_pin)
        self.fold_threshold_bytes = max(128, int(fold_threshold_bytes))
        self.policy = policy
        self.secret_broker = secret_broker
        self._downstream_transports: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = dict(downstream_transports or {})
        self.block_response_injections = bool(block_response_injections)
        self.embodied_mode = bool(embodied_mode)
        self.embodied_hard_limits = dict(embodied_hard_limits) if embodied_hard_limits is not None else None
        self.mission_envelope = mission_envelope
        self.mission_secret = mission_secret
        self.capability_token = capability_token
        self.root_secret = root_secret
        self.sensor_verifier = sensor_verifier
        self.flight_recorder = flight_recorder
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

    def handle_tools_list_changed(
        self,
        server_name: str,
        updated_tools: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Re-verify an already-pinned server after `notifications/tools/list_changed`; never auto-pins."""
        return self.register_or_verify_server(server_name, updated_tools, auto_pin_first_seen=False)

    def evaluate_tool_call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Enforce quarantine, header safety, secret vault scope, outbound DLP, and `decide()` risk tiering."""
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
        if self.secret_broker is not None:
            brok = self.secret_broker.resolve_for_call(srv, tname, args_dict, headers)
            if not brok.get("ok"):
                return {
                    "decision": "deny",
                    "reason_code": str(brok.get("reason_code") or "mcp_proxy_secret_scope_violation"),
                    "label": str(brok.get("label") or "Secret broker scope violation"),
                    "layer": 0,
                }

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

        if self.embodied_mode or is_embodied_mcp_tool(tname, args_dict):
            return self._evaluate_embodied_tool_call(srv, tname, args_dict)

        from .core.layered_decision_engine import decide

        return decide(
            {"tool": f"mcp__{srv}__{tname}", "arguments": args_dict},
            policy=self.policy,
        )

    def _evaluate_embodied_tool_call(
        self,
        server_name: str,
        tool_name: str,
        args_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        from .hosts.ros2_adapter import evaluate_embodied_action

        protocol, emb_payload = _infer_embodied_protocol_and_payload(tool_name, args_dict)
        sensor_obs = emb_payload.pop("sensor_observation", None)
        scene_text = emb_payload.get("scene_text")
        carrying_n = emb_payload.get("carrying_payload_n", emb_payload.get("load_cell_n"))
        current_pos = emb_payload.get("current_position")

        emb_res = evaluate_embodied_action(
            emb_payload,
            protocol=protocol,
            capability_token=self.capability_token,
            root_secret=self.root_secret,
            hard_limits=self.embodied_hard_limits,
            mission_envelope=self.mission_envelope,
            mission_secret=self.mission_secret,
            sensor_observation=sensor_obs if isinstance(sensor_obs, Mapping) else None,
            sensor_verifier=self.sensor_verifier,
            scene_text=scene_text,
            carrying_payload_n=(
                float(carrying_n)
                if isinstance(carrying_n, (int, float)) and not isinstance(carrying_n, bool)
                else None
            ),
            current_position=current_pos,
        )

        frame_hash: Optional[str] = None
        if self.flight_recorder is not None:
            allowed_ok = emb_res.get("decision") == "allow"
            norm_data = emb_res.get("normalized")
            token_id = getattr(self.capability_token, "token_id", None) or getattr(
                self.mission_envelope, "mission_id", None
            )
            slack = 1.0 if allowed_ok else -1.0
            if allowed_ok and isinstance(norm_data, Mapping) and self.embodied_hard_limits:
                max_v = self.embodied_hard_limits.get("max_linear_vel_mps")
                cur_v = norm_data.get("linear_vel_mps")
                if isinstance(max_v, (int, float)) and isinstance(cur_v, (int, float)) and math.isfinite(cur_v):
                    slack = float(max_v) - float(cur_v)

            recorded = self.flight_recorder.record(
                tick=self.flight_recorder.total_recorded,
                timestamp_ns=time.time_ns(),
                state=_sanitize_for_flight_log(
                    {
                        "server": server_name,
                        "tool": tool_name,
                        "position": current_pos
                        or (norm_data.get("target_position") if isinstance(norm_data, Mapping) else None),
                        "carrying_payload_n": carrying_n,
                    }
                ),
                u_nominal=_sanitize_for_flight_log(args_dict),
                u_filtered=(
                    _sanitize_for_flight_log(norm_data)
                    if allowed_ok
                    else {
                        "blocked": True,
                        "reason": str(emb_res.get("reason") or "denied"),
                        "clamp_hints": _sanitize_for_flight_log(emb_res.get("clamp_hints") or {}),
                    }
                ),
                min_barrier_slack=slack,
                aci_q_hat=0.0,
                token_id=token_id,
                status="ok" if allowed_ok else f"blocked:{emb_res.get('reason')}",
            )
            frame_hash = recorded.get("frame_hash")

        reason = str(emb_res.get("reason") or "denied")
        if emb_res.get("decision") != "allow":
            receipt = {
                "server": server_name,
                "tool": tool_name,
                "protocol": protocol,
                "decision": "deny",
                "reason": reason,
                "clamp_hints": emb_res.get("clamp_hints") or {},
                "flight_frame_hash": frame_hash,
            }
            return {
                "decision": "deny",
                "authorized": False,
                "reason": reason,
                "reason_code": f"mcp_proxy_embodied_veto:{reason}",
                "label": f"Embodied action blocked ({reason})",
                "layer": 1,
                "clamp_hints": emb_res.get("clamp_hints") or {},
                "embodied_verdict": emb_res,
                "block_receipt": receipt,
                "flight_frame_hash": frame_hash,
            }

        norm_info = emb_res.get("normalized") or {}
        return {
            "decision": "allow",
            "authorized": True,
            "reason": "authorized",
            "reason_code": "layer1_embodied_firewall_verified",
            "label": f"embodied:{protocol}:{norm_info.get('topic_or_action', tool_name)}",
            "layer": 1,
            "clamp_hints": emb_res.get("clamp_hints") or {},
            "embodied_verdict": emb_res,
            "flight_frame_hash": frame_hash,
        }

    def execute_proxied_tool_call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        req_id: Any = 1,
    ) -> Dict[str, Any]:
        """Gate, inject brokered secrets out-of-band, forward to downstream, and sanitize response."""
        srv = str(server_name or "").strip()
        verdict = self.evaluate_tool_call(srv, tool_name, arguments, headers=headers)
        if verdict.get("decision") != "allow":
            denied_out: Dict[str, Any] = {
                "allowed": False,
                "verdict": verdict,
                "isError": True,
                "content": [{"type": "text", "text": f"Blocked by Aetheris MCP Proxy: {verdict.get('reason_code')}: {verdict.get('label')}"}],
            }
            if "block_receipt" in verdict:
                denied_out["block_receipt"] = verdict["block_receipt"]
                denied_out["reason"] = verdict.get("reason")
            return denied_out

        transport = self._downstream_transports.get(srv) or self._downstream_transports.get(srv.lower())
        if transport is None:
            return {
                "allowed": False,
                "verdict": verdict,
                "isError": True,
                "reason_code": "mcp_proxy_no_downstream_transport",
                "content": [{"type": "text", "text": f"No downstream transport configured for MCP server '{srv}'"}],
            }

        args_dict = dict(arguments) if isinstance(arguments, dict) else {}
        outbound_args = args_dict
        outbound_headers = dict(headers) if headers is not None else None
        resolved_handles: List[str] = []
        if self.secret_broker is not None:
            brok = self.secret_broker.resolve_for_call(srv, tool_name, args_dict, headers)
            if not brok.get("ok"):
                return {
                    "allowed": False,
                    "verdict": {
                        "decision": "deny",
                        "reason_code": brok.get("reason_code"),
                        "label": brok.get("label"),
                        "layer": 0,
                    },
                    "isError": True,
                }
            outbound_args = brok["arguments"]
            outbound_headers = brok["headers"]
            resolved_handles = list(brok.get("resolved_handles") or [])

        rpc_req: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": outbound_args},
        }
        if outbound_headers is not None:
            rpc_req["headers"] = outbound_headers

        raw_resp = transport(rpc_req)
        raw_payload = raw_resp.get("result", raw_resp) if isinstance(raw_resp, dict) else raw_resp
        scanned = self.process_tool_response(srv, tool_name, raw_payload)
        out: Dict[str, Any] = {
            "allowed": True,
            "verdict": verdict,
            "resolved_handles": resolved_handles,
            **scanned,
        }
        if self.block_response_injections and scanned.get("injection_detected"):
            out["response_quarantined"] = True
            out["isError"] = True
            out["redacted_text"] = (
                f"[QUARANTINED_MCP_RESPONSE: indirect prompt injection blocked ({', '.join(scanned.get('injection_findings', ()))})]"
            )
        return out

    def process_tool_response(
        self,
        server_name: str,
        tool_name: str,
        raw_output: Any,
    ) -> Dict[str, Any]:
        """Scrub secrets (including brokered vault tokens), detect injections, and fold oversized responses."""
        del server_name, tool_name
        text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False, default=str)
        vault_hits = 0
        if self.secret_broker is not None:
            text, vault_hits = self.secret_broker.scrub_known_secrets(text)
        scrub = scrub_tool_output(text)
        inj = detect_prompt_injection(text)
        folded = fold_untrusted_output(
            scrub["redacted_text"],
            state_dir=self.state_dir,
            threshold_bytes=self.fold_threshold_bytes,
        )
        secret_types = list(scrub["secret_types"])
        if vault_hits and "vault_brokered_secret" not in secret_types:
            secret_types.append("vault_brokered_secret")
        return {
            "redacted_text": folded["preview"],
            "folded": folded["folded"],
            "artifact_path": folded["artifact_path"],
            "original_bytes": folded["original_bytes"],
            "secrets_found": bool(scrub["secrets_found"] or vault_hits > 0),
            "secret_count": int(scrub["secret_count"]) + vault_hits,
            "secret_types": sorted(secret_types),
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
                "serverInfo": {"name": "aetheris-mcp-proxy", "version": "0.9.0"},
            })
        if method == "tools/list":
            res = self.build_tools_list_response(
                client_supports_defer_loading=bool(params.get("defer_loading")),
                query=str(params.get("query") or ""),
            )
            return _ok({"tools": res["tools"], "_meta": res["_meta"]})
        if method == "notifications/tools/list_changed":
            srv = str(params.get("server") or "default")
            upd = params.get("tools") if isinstance(params.get("tools"), list) else []
            return _ok(self.handle_tools_list_changed(srv, upd))
        if method == "tools/call":
            srv = str(params.get("server") or "default")
            tname = str(params.get("name") or "")
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            hdrs = params.get("headers") if isinstance(params.get("headers"), dict) else None
            if srv in self._downstream_transports or srv.lower() in self._downstream_transports:
                proxied = self.execute_proxied_tool_call(srv, tname, args, headers=hdrs, req_id=req_id)
                if not proxied.get("allowed") and "block_receipt" in proxied:
                    v = proxied["verdict"]
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32001,
                            "message": f"Blocked by Aetheris Embodied Firewall: {v.get('reason')}",
                            "data": {
                                "decision": "deny",
                                "reason": v.get("reason"),
                                "reason_code": v.get("reason_code"),
                                "block_receipt": proxied["block_receipt"],
                            },
                        },
                    }
                return _ok(proxied)
            verdict = self.evaluate_tool_call(srv, tname, args, headers=hdrs)
            if verdict.get("decision") != "allow" and "block_receipt" in verdict:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32001,
                        "message": f"Blocked by Aetheris Embodied Firewall: {verdict.get('reason')}",
                        "data": {
                            "decision": "deny",
                            "reason": verdict.get("reason"),
                            "reason_code": verdict.get("reason_code"),
                            "block_receipt": verdict["block_receipt"],
                        },
                    },
                }
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
    parser.add_argument(
        "--downstream",
        action="append",
        default=[],
        metavar="SERVER=CMD",
        help="Register downstream stdio MCP server (`name=command args...`) for live proxying",
    )
    parser.add_argument("--block-injections", action="store_true", help="Quarantine downstream responses containing indirect prompt injections")
    parser.add_argument("--embodied-ros2", action="store_true", help="Force embodied physical action firewall gating on all MCP tool calls")
    args = parser.parse_args(argv)

    if args.check:
        rep = run_proxy_self_check()
        print(json.dumps(rep, ensure_ascii=False))
        return 0 if rep.get("ok") else 1

    if args.stdio:
        clients: Dict[str, StdioMCPDownstreamClient] = {}
        transports: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        for spec in args.downstream or ():
            if "=" in str(spec):
                sname, scmd = str(spec).split("=", 1)
                cl = StdioMCPDownstreamClient(shlex.split(scmd))
                clients[sname.strip()] = cl
                transports[sname.strip()] = cl.send_rpc
        gw = MCPProxyGateway(
            secret_broker=SecretBroker(),
            downstream_transports=transports,
            block_response_injections=bool(args.block_injections),
            embodied_mode=bool(args.embodied_ros2),
        )
        for sname, cl in clients.items():
            try:
                gw.register_or_verify_server(sname, cl.fetch_tools(), auto_pin_first_seen=True)
            except Exception:
                pass
        try:
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
        finally:
            for cl in clients.values():
                cl.close()
        return 0

    rep = run_proxy_self_check()
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
