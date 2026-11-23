"""OpenTelemetry GenAI Semantic Conventions & OTLP/HTTP JSON Trace Exporter (stdlib-only).

Converts Aetheris Microbrains decision journal records into OTLP JSON (`v1/traces`)
resourceSpans using GenAI semantic conventions (`gen_ai.operation.name`, `gen_ai.tool.name`)
plus `aetheris.*` attributes, writing either to disk or over OTLP/HTTP.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _hex_id(seed: str, length_chars: int) -> str:
    if seed:
        return hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:length_chars]
    return secrets.token_hex(length_chars // 2)


def _attr_value(val: Any) -> Dict[str, Any]:
    if isinstance(val, bool):
        return {"boolValue": val}
    if isinstance(val, int) and not isinstance(val, bool):
        return {"intValue": str(val)}
    if isinstance(val, float):
        return {"doubleValue": float(val)}
    return {"stringValue": str(val)}


def _make_attrs(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"key": str(k), "value": _attr_value(v)} for k, v in mapping.items() if v is not None]


def journal_entry_to_otel_span(
    entry: Dict[str, Any],
    *,
    service_name: str = "aetheris-microbrains",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one decision journal entry into an OTLP JSON Span dictionary."""
    del service_name
    is_shadow = bool(entry.get("shadow_mode") or entry.get("shadow_decision") or entry.get("shadow_reason_code"))
    eff_dec = str(
        (entry.get("shadow_decision") if is_shadow else None)
        or entry.get("decision")
        or "unknown"
    ).strip().lower()
    eff_reason = str(
        (entry.get("shadow_reason_code") if is_shadow else None)
        or entry.get("reason_code")
        or "none"
    ).strip()

    tool = str(entry.get("tool") or entry.get("payload_tool") or "unknown").strip() or "unknown"
    ts_sec = float(entry.get("timestamp") or time.time())
    lat_ms = max(0.0, float(entry.get("latency_ms") or 0.0))
    start_ns = int(ts_sec * 1_000_000_000)
    end_ns = max(start_ns + 1, start_ns + int(lat_ms * 1_000_000))

    sess_seed = str(entry.get("session_id") or "")
    sha_seed = str(entry.get("input_sha256") or "")
    resolved_trace_id = trace_id or _hex_id(sess_seed or sha_seed, 32)
    span_id = _hex_id(f"{sha_seed}:{ts_sec}" if sha_seed else "", 16)

    attrs_map: Dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool,
        "aetheris.decision": eff_dec,
        "aetheris.reason_code": eff_reason,
        "aetheris.layer": str(entry.get("layer", "unknown")),
        "aetheris.latency_ms": float(lat_ms),
        "aetheris.shadow_mode": is_shadow,
        "aetheris.host": str(entry.get("host") or "unknown"),
        "aetheris.input_sha256": sha_seed or "none",
    }
    if entry.get("avoided_tokens") is not None:
        attrs_map["aetheris.avoided_tokens"] = int(entry["avoided_tokens"])

    status = (
        {"code": 2, "message": str(entry.get("label") or eff_reason)}
        if eff_dec == "deny"
        else {"code": 1}
    )

    return {
        "traceId": resolved_trace_id,
        "spanId": span_id,
        "name": f"aetheris.gate.{tool}",
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": _make_attrs(attrs_map),
        "status": status,
    }


def build_otlp_trace_payload(
    entries: Sequence[Dict[str, Any]],
    *,
    service_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an OTLP/HTTP JSON `ExportTraceServiceRequest` (`resourceSpans`) payload."""
    svc = str(service_name or os.getenv("OTEL_SERVICE_NAME") or "aetheris-microbrains").strip()
    spans = [
        journal_entry_to_otel_span(e, service_name=svc)
        for e in entries
        if isinstance(e, dict)
    ]
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _make_attrs({
                        "service.name": svc,
                        "telemetry.sdk.name": "aetheris-microbrains-otel",
                        "telemetry.sdk.language": "python",
                        "telemetry.sdk.version": "0.7.7",
                    })
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "aetheris_microbrains.core.otel_exporter",
                            "version": "0.7.7",
                        },
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def export_otlp_file(
    entries: Sequence[Dict[str, Any]],
    output_path: Path | str,
    *,
    service_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically write OTLP/HTTP JSON trace payload to `output_path`."""
    payload = build_otlp_trace_payload(entries, service_name=service_name)
    span_count = len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"])
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return {"written": True, "path": str(target), "span_count": span_count}


def push_otlp_http(
    entries: Sequence[Dict[str, Any]],
    *,
    endpoint: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 3.0,
    service_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Push OTLP GenAI trace spans via OTLP/HTTP JSON POST; never raises on collector failure."""
    raw_ep = str(endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not raw_ep:
        return {"ok": False, "error": "missing_otlp_endpoint", "span_count": 0}
    target_url = raw_ep if raw_ep.rstrip("/").endswith("/v1/traces") else f"{raw_ep.rstrip('/')}/v1/traces"

    payload = build_otlp_trace_payload(entries, service_name=service_name)
    span_count = len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"])
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req_headers: Dict[str, str] = {"Content-Type": "application/json"}
    if headers:
        for k, v in headers.items():
            req_headers[str(k)] = str(v)

    req = urllib.request.Request(target_url, data=body, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            return {
                "ok": 200 <= int(resp.status) < 300,
                "status": int(resp.status),
                "endpoint": target_url,
                "span_count": span_count,
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": int(exc.code), "endpoint": target_url, "error": str(exc), "span_count": span_count}
    except Exception as exc:
        return {"ok": False, "endpoint": target_url, "error": f"{type(exc).__name__}: {exc}", "span_count": span_count}
