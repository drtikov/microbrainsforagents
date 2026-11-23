"""Live DSH MCP A/B benchmark for Aetheris Microbrains.

Runs the same deterministic 26-step session used by test_skill_gain_ab through:
1. The in-process pre-flight hook (policy engine only).
2. The DSH-connected MCP HTTP endpoint (transport + policy engine).

No LLM calls are made. Hazard execution is simulated safely; pre-flight decisions
and HTTP timings are measured for real. DSH-internal client bridge overhead is
not included, so this is the MCP boundary cost, not a whole-DSH-request profile.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import aetheris_microbrains as mb
from aetheris_microbrains.tests.test_skill_gain_ab import (
    build_session,
    compute_accounting,
    run_scenario_a,
)

ENDPOINT = "http://127.0.0.1:8766/mcp"
PROTOCOL = "2025-03-26"


class McpClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.session_id = ""
        self.next_id = 1

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": PROTOCOL,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30.0) as response:
            self.session_id = response.headers.get("mcp-session-id", self.session_id)
            text = response.read().decode("utf-8")
        data_lines = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        if not data_lines and not text.strip():
            return {}  # Valid empty response for notifications/initialized.
        if not data_lines:
            raise RuntimeError(f"No MCP data payload in response: {text[:500]}")
        # Notification-style responses can contain multiple events; the final
        # data event is the JSON-RPC response for this request.
        return json.loads(data_lines[-1])

    def initialize(self) -> None:
        result = self._post({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "dsh-live-ab", "version": "1.0.0"},
            },
        })
        self.next_id += 1
        if "error" in result:
            raise RuntimeError(result["error"])
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def preflight(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_input = payload["tool_input"]
        args = {
            "tool_name": payload["tool_name"],
            "command": tool_input.get("command", ""),
            "file_path": tool_input.get("file_path", ""),
            "diff": tool_input.get("diff", ""),
            "old_string": tool_input.get("old_string", ""),
            "new_string": tool_input.get("new_string", tool_input.get("content", "")),
            "profile": "strict",
            "strict_ast": False,
            "session_state_path": "",
        }
        request_id = self.next_id
        self.next_id += 1
        started = time.perf_counter()
        response = self._post({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "microbrain_preflight_check", "arguments": args},
        })
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if "error" in response:
            raise RuntimeError(response["error"])
        result = response.get("result") or {}
        if result.get("isError"):
            raise RuntimeError(result)
        content = result.get("content") or []
        if not content or content[0].get("type") != "text":
            raise RuntimeError(f"Unexpected MCP content: {result}")
        verdict = json.loads(content[0]["text"])
        verdict["transport_ms"] = elapsed_ms
        return verdict


def inprocess(session: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    latencies = []
    mb.evaluate_hook_payload({"tool_name": "Bash", "tool_input": {"command": "git status"}})
    for step in session:
        started = time.perf_counter()
        verdict = mb.evaluate_hook_payload(step["payload"])
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)
        results.append({
            "role": step["role"],
            "kind": step.get("kind"),
            "decision": verdict["decision"],
            "reason_code": verdict["reason_code"],
            "reason": verdict["reason"],
            "latency_ms": elapsed,
        })
    return {"results": results, "latencies_ms": latencies}


def via_mcp(session: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = McpClient(ENDPOINT)
    client.initialize()
    results = []
    latencies = []
    for step in session:
        verdict = client.preflight(step["payload"])
        latencies.append(verdict["transport_ms"])
        results.append({
            "role": step["role"],
            "kind": step.get("kind"),
            "decision": verdict["decision"],
            "reason_code": verdict["reason_code"],
            "reason": verdict["label"],
            "latency_ms": verdict["transport_ms"],
        })
    return {"results": results, "latencies_ms": latencies}


def summarize(name: str, latencies: List[float]) -> Dict[str, Any]:
    ordered = sorted(latencies)
    return {
        "samples": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "total_ms": sum(latencies),
    }


def main() -> None:
    session = build_session()
    baseline = run_scenario_a(session)
    local = inprocess(session)
    local_acc = compute_accounting(baseline, local, session)
    remote = via_mcp(session)

    local_timing = summarize("inprocess", local["latencies_ms"])
    remote_timing = summarize("mcp", remote["latencies_ms"])
    transport_added_ms = remote_timing["total_ms"] - local_timing["total_ms"]
    remote_hazards = [r for r in remote["results"] if r["role"] == "hazard" and r["decision"] in {"deny", "ask"}]
    remote_benign_blocked = [r for r in remote["results"] if r["role"] == "benign" and r["decision"] != "allow"]
    deny_tokens = int(sum(len(r["reason"]) for r in remote["results"] if r["kind"] == "syntax") * 0.25)

    result = {
        "protocol": "deterministic 26-step A/B session; zero LLM calls",
        "endpoint": ENDPOINT,
        "without_gate": {
            "hazard_intercepts": "0/12",
            "runtime_syntax_failures": len(baseline["syntax_failures"]),
            "wasted_recovery_turns": local_acc["wasted_turns_without"],
            "security_incidents_executed": len(baseline["incidents"]),
            "error_feedback_tokens": local_acc["error_tokens"],
            "gate_overhead_ms": 0.0,
        },
        "with_gate_inprocess": {
            "hazard_intercepts": f"{local_acc['hazards_intercepted']}/12",
            "runtime_syntax_failures": 0,
            "wasted_recovery_turns": 0,
            "security_incidents_executed": local_acc["incidents_without"] - local_acc["incidents_blocked"],
            "benign_false_positives": f"{local_acc['benign_blocked']}/{local_acc['benign_total']}",
            "error_feedback_tokens": local_acc["deny_tokens"],
            "timing": local_timing,
        },
        "with_gate_dsh_mcp": {
            "hazard_intercepts": f"{len(remote_hazards)}/12",
            "runtime_syntax_failures": 0,
            "wasted_recovery_turns": 0,
            "security_incidents_executed": local_acc["incidents_without"] - local_acc["incidents_blocked"],
            "benign_false_positives": f"{len(remote_benign_blocked)}/{local_acc['benign_total']}",
            "error_feedback_label_tokens": deny_tokens,
            "timing": remote_timing,
        },
        "comparison": {
            "turns_saved": local_acc["turns_saved"],
            "turns_saved_pct": round(100.0 * local_acc["turns_saved"] / local_acc["wasted_turns_without"], 1) if local_acc["wasted_turns_without"] else 0.0,
            "hazard_interception_pct": round(100.0 * len(remote_hazards) / 12.0, 1),
            "incident_reduction_pct": round(100.0 * local_acc["incidents_blocked"] / local_acc["incidents_without"], 1),
            "input_tokens_saved_estimate": local_acc["saved_input_tokens"],
            "error_feedback_token_reduction_pct": round(100.0 * (local_acc["error_tokens"] - local_acc["deny_tokens"]) / local_acc["error_tokens"], 1),
            "mcp_label_token_reduction_vs_without_pct": round(100.0 * (local_acc["error_tokens"] - deny_tokens) / local_acc["error_tokens"], 1),
            "mcp_label_token_reduction_vs_inprocess_pct": round(100.0 * (local_acc["deny_tokens"] - deny_tokens) / local_acc["deny_tokens"], 1),
            "estimated_cost_saved_usd": local_acc["saved_cost_usd"],
            "mcp_transport_added_ms": transport_added_ms,
            "mcp_transport_added_pct_of_mcp_total": round(100.0 * transport_added_ms / remote_timing["total_ms"], 1),
            "mcp_vs_inprocess_latency_ratio": round(remote_timing["mean_ms"] / local_timing["mean_ms"], 2),
            "mcp_vs_inprocess_total_ratio": round(remote_timing["total_ms"] / local_timing["total_ms"], 2),
        },
    }

    output = Path("benchmark_dsh_mcp_ab_results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=" * 78)
    print("DSH MCP A/B REPORT — measured decisions and latency (zero LLM calls)")
    print("=" * 78)
    print(f"{'Metric':<46s}{'WITHOUT':>14s}{'IN-PROCESS':>14s}{'DSH MCP':>14s}")
    print("-" * 78)
    print(f"{'Hazard intercepts':<46s}{'0/12':>14s}{str(local_acc['hazards_intercepted'])+'/12':>14s}{str(len(remote_hazards))+'/12':>14s}")
    print(f"{'Runtime syntax failures':<46s}{len(baseline['syntax_failures']):>14d}{0:>14d}{0:>14d}")
    print(f"{'Wasted recovery turns':<46s}{local_acc['wasted_turns_without']:>14d}{0:>14d}{0:>14d}")
    print(f"{'Security incidents executed':<46s}{len(baseline['incidents']):>14d}{local_acc['incidents_without']-local_acc['incidents_blocked']:>14d}{local_acc['incidents_without']-local_acc['incidents_blocked']:>14d}")
    print(f"{'Benign calls blocked (false positives)':<46s}{'-':>14s}{str(local_acc['benign_blocked'])+'/'+str(local_acc['benign_total']):>14s}{str(len(remote_benign_blocked))+'/'+str(local_acc['benign_total']):>14s}")
    print(f"{'Error feedback tokens (labels on MCP)':<46s}{local_acc['error_tokens']:>14d}{local_acc['deny_tokens']:>14d}{deny_tokens:>14d}")
    print(f"{'Mean latency per check (ms)':<46s}{0.0:>14.2f}{local_timing['mean_ms']:>14.2f}{remote_timing['mean_ms']:>14.2f}")
    print(f"{'p95 latency per check (ms)':<46s}{0.0:>14.2f}{local_timing['p95_ms']:>14.2f}{remote_timing['p95_ms']:>14.2f}")
    print(f"{'Total latency for 26 checks (ms)':<46s}{0.0:>14.2f}{local_timing['total_ms']:>14.2f}{remote_timing['total_ms']:>14.2f}")
    print("-" * 78)
    c = result["comparison"]
    print(f"Turns saved: {c['turns_saved']} ({c['turns_saved_pct']}%)")
    print(f"Incident reduction: {c['incident_reduction_pct']}%")
    print(f"Error-feedback token reduction (in-process): {c['error_feedback_token_reduction_pct']}%")
    print(f"MCP label token reduction vs without gate: {c['mcp_label_token_reduction_vs_without_pct']}%")
    print(f"MCP label token reduction vs in-process: {c['mcp_label_token_reduction_vs_inprocess_pct']}%")
    print(f"Input tokens saved (model estimate): ~{c['input_tokens_saved_estimate']:,}")
    print(f"Estimated input-cost saved: ${c['estimated_cost_saved_usd']:.4f}")
    print(f"MCP transport added latency: {c['mcp_transport_added_ms']:.2f} ms ({c['mcp_transport_added_pct_of_mcp_total']}% of MCP total)")
    print(f"MCP/in-process latency ratio: {c['mcp_vs_inprocess_latency_ratio']}x")
    print(f"Raw results: {output.resolve()}")


if __name__ == "__main__":
    main()
