"""Local Interactive Audit Web Dashboard (`aetheris-audit --ui`, stdlib-only).

Loopback-only HTTP + SSE dashboard serving real-time journal decisions, token
ROI metrics, OTLP exports, and token-authenticated pending rule approvals.
"""

from __future__ import annotations

import hmac
import html
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlsplit

from .economics_ledger import summarize_savings
from .otel_exporter import build_otlp_trace_payload
from .policy_governance import PendingRuleQueue


def build_dashboard_snapshot(
    *,
    journal_path: Optional[str | Path] = None,
    economics_path: Optional[str | Path] = None,
    pending_path: Optional[str | Path] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Assemble live audit counts, friction hotspots, token ROI, and staged rules."""
    from ..audit import build_friction_digest, get_journal_entries

    entries = get_journal_entries(journal_path=journal_path, limit=max(1, int(limit)))
    counts = {"allow": 0, "deny": 0, "escalate": 0, "other": 0}
    latencies = []
    for e in entries:
        dec = str(e.get("shadow_decision") if (e.get("shadow_mode") and e.get("shadow_decision")) else e.get("decision") or "other").lower()
        if dec in counts:
            counts[dec] += 1
        else:
            counts["other"] += 1
        lat = e.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))

    avg_latency_ms = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    digest = build_friction_digest(entries)
    savings = summarize_savings(ledger_path=economics_path)
    pending = PendingRuleQueue(path=pending_path).list_pending()

    return {
        "generated_at": round(time.time(), 3),
        "total_events": len(entries),
        "decisions": counts,
        "avg_latency_ms": avg_latency_ms,
        "savings": savings,
        "friction": digest,
        "pending_rules": pending,
        "recent_entries": list(reversed(entries[-50:])),
    }


def render_dashboard_html(api_token: str) -> str:
    """Render self-contained dark-mode Audit Web UI HTML with CSRF API token."""
    esc_token = html.escape(api_token, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Aetheris Microbrains — Audit UI</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:#0b1020; color:#e2e8f0; margin:0; padding:24px; }}
  h1 {{ margin:0 0 16px; font-size:20px; color:#f8fafc; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:20px; }}
  .card {{ background:#141b33; border:1px solid #26314f; border-radius:10px; padding:14px; }}
  .kpi {{ font-size:24px; font-weight:700; margin-top:4px; }}
  .muted {{ font-size:12px; color:#94a3b8; }}
  .allow {{ color:#4ade80; }} .deny {{ color:#f87171; }} .escalate {{ color:#fbbf24; }}
  table {{ width:100%; border-collapse:collapse; background:#141b33; border:1px solid #26314f; border-radius:8px; overflow:hidden; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #1f2947; font-size:13px; }}
  th {{ background:#0f162b; color:#94a3b8; }}
  button {{ background:#2563eb; color:#fff; border:0; border-radius:6px; padding:6px 12px; cursor:pointer; }}
  a.btn {{ color:#60a5fa; text-decoration:none; font-size:13px; }}
</style>
</head>
<body>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
    <h1>Aetheris Microbrains Audit &amp; Telemetry</h1>
    <a class="btn" href="/api/otlp" download="aetheris_otlp.json">Export OTLP JSON</a>
  </div>
  <div class="grid">
    <div class="card"><div class="muted">Events Loaded</div><div class="kpi" id="k-total">-</div></div>
    <div class="card"><div class="muted">Denied</div><div class="kpi deny" id="k-deny">-</div></div>
    <div class="card"><div class="muted">Escalated</div><div class="kpi escalate" id="k-esc">-</div></div>
    <div class="card"><div class="muted">Avg Latency (ms)</div><div class="kpi" id="k-lat">-</div></div>
    <div class="card"><div class="muted">Tokens Preserved</div><div class="kpi allow" id="k-tok">-</div></div>
  </div>
  <h3>Pending Policy Rules</h3>
  <div id="pending-box" class="card" style="margin-bottom:20px;">Loading...</div>
  <h3>Recent Decisions</h3>
  <table>
    <thead><tr><th>Decision</th><th>Layer</th><th>Tool</th><th>Reason</th><th>Latency</th><th>Preview</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
<script>
const API_TOKEN = "{esc_token}";
function applySnapshot(s) {{
  document.getElementById("k-total").textContent = s.total_events;
  document.getElementById("k-deny").textContent = s.decisions.deny || 0;
  document.getElementById("k-esc").textContent = s.decisions.escalate || 0;
  document.getElementById("k-lat").textContent = s.avg_latency_ms;
  document.getElementById("k-tok").textContent = (s.savings && s.savings.tokens_preserved) || 0;
  const pbox = document.getElementById("pending-box");
  pbox.textContent = "";
  const pending = s.pending_rules || [];
  if (!pending.length) {{ pbox.textContent = "No pending rules staged."; }}
  pending.forEach((r) => {{
    const row = document.createElement("div");
    row.style.margin = "6px 0";
    const span = document.createElement("span");
    span.textContent = r.id + " — " + (r.reason || "") + " ";
    const btn = document.createElement("button");
    btn.textContent = "Approve";
    btn.onclick = async () => {{
      await fetch("/api/pending-rules/approve", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json", "X-Aetheris-Token": API_TOKEN }},
        body: JSON.stringify({{ rule_id: r.id }})
      }});
      refresh();
    }};
    row.appendChild(span);
    row.appendChild(btn);
    pbox.appendChild(row);
  }});
  const tbody = document.getElementById("rows");
  tbody.textContent = "";
  (s.recent_entries || []).slice(0, 30).forEach((e) => {{
    const tr = document.createElement("tr");
    const d = e.shadow_decision || e.decision || "unknown";
    [d.toUpperCase(), String(e.layer ?? "-"), e.tool || "-", e.reason_code || "-", String(e.latency_ms ?? "-"), (e.payload_preview || "").slice(0, 80)].forEach((txt, i) => {{
      const td = document.createElement("td");
      if (i === 0) td.className = d.toLowerCase();
      td.textContent = txt;
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
}}
async function refresh() {{
  const r = await fetch("/api/snapshot");
  if (r.ok) applySnapshot(await r.json());
}}
refresh();
const es = new EventSource("/events");
es.onmessage = (ev) => {{ try {{ applySnapshot(JSON.parse(ev.data)); }} catch (_) {{}} }};
</script>
</body>
</html>
"""


class AuditUIServer:
    """Thread-safe local loopback web dashboard for inspecting decisions and approving staged rules."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8767,
        journal_path: Optional[str | Path] = None,
        economics_path: Optional[str | Path] = None,
        pending_path: Optional[str | Path] = None,
        policy_out_path: Optional[str | Path] = None,
        sse_interval_sec: float = 2.0,
    ) -> None:
        self.host = str(host or "127.0.0.1")
        self.requested_port = int(port)
        self.journal_path = journal_path
        self.economics_path = economics_path
        self.pending_path = pending_path
        self.policy_out_path = policy_out_path
        self.sse_interval_sec = max(0.1, float(sse_interval_sec))
        self.api_token = secrets.token_hex(16)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("AuditUIServer is not running")
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _snapshot(self) -> Dict[str, Any]:
        return build_dashboard_snapshot(
            journal_path=self.journal_path,
            economics_path=self.economics_path,
            pending_path=self.pending_path,
        )

    def start(self) -> "AuditUIServer":
        if self._httpd is not None:
            return self
        parent = self
        self._stop_event.clear()

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, code: int, obj: Any) -> None:
                self._send(code, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                route = parsed.path
                if route in {"/", "/index.html"}:
                    self._send(200, "text/html; charset=utf-8", render_dashboard_html(parent.api_token).encode("utf-8"))
                    return
                if route == "/api/snapshot":
                    self._send_json(200, parent._snapshot())
                    return
                if route == "/api/otlp":
                    from ..audit import get_journal_entries

                    entries = get_journal_entries(journal_path=parent.journal_path, limit=200)
                    self._send_json(200, build_otlp_trace_payload(entries))
                    return
                if route == "/events":
                    qs = parse_qs(parsed.query)
                    once = qs.get("once", ["0"])[0] == "1"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while not parent._stop_event.is_set():
                            frame = f"data: {json.dumps(parent._snapshot(), ensure_ascii=False)}\n\n".encode("utf-8")
                            self.wfile.write(frame)
                            self.wfile.flush()
                            if once:
                                break
                            parent._stop_event.wait(parent.sse_interval_sec)
                    except OSError:
                        return
                    return
                self._send_json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                route = urlsplit(self.path).path
                if route != "/api/pending-rules/approve":
                    self._send_json(404, {"error": "not_found"})
                    return
                tok = str(self.headers.get("X-Aetheris-Token") or "").strip()
                if not hmac.compare_digest(tok, parent.api_token):
                    self._send_json(403, {"error": "invalid_api_token"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                    body = json.loads(self.rfile.read(min(length, 65536)).decode("utf-8"))
                    rule_id = str(body.get("rule_id") or "").strip()
                    approver = str(body.get("approver") or "audit_ui_operator").strip()
                    rec = PendingRuleQueue(path=parent.pending_path).approve(
                        rule_id,
                        approver=approver,
                        policy_out_path=parent.policy_out_path,
                    )
                    self._send_json(200, {"ok": True, "rule": rec})
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        class _Server(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._httpd = _Server((self.host, self.requested_port), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except OSError:
                pass
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def __enter__(self) -> "AuditUIServer":
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()


def serve_audit_ui(
    *,
    port: int = 8767,
    journal_path: Optional[str | Path] = None,
    economics_path: Optional[str | Path] = None,
    pending_path: Optional[str | Path] = None,
    policy_out_path: Optional[str | Path] = None,
) -> int:
    """Start the local Audit Web UI server until Ctrl+C."""
    srv = AuditUIServer(
        port=port,
        journal_path=journal_path,
        economics_path=economics_path,
        pending_path=pending_path,
        policy_out_path=policy_out_path,
    )
    srv.start()
    print(f"Aetheris Audit Web UI running at {srv.url} (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Audit Web UI stopping.")
    finally:
        srv.stop()
    return 0
