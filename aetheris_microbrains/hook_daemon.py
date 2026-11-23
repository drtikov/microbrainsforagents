"""Local hook daemon (A1) + verdict memoization (A3).

The gate's per-call latency is dominated by interpreter startup and imports
(~50-155ms per hook invocation). This daemon keeps a warm evaluator process:
the hook's CLI queries it over localhost TCP first and falls back to
in-process evaluation on ANY failure (missing state, stale version, refused
connection, timeout) — the daemon is a pure optimization, never a dependency.

Handshake/state:
    `<state_dir>/hook_daemon.json` = {port, pid, auth, version, started_at}.
    Requests carry the auth value; a version mismatch makes clients ignore the
    daemon so upgrades fall back safely.

Memoization (A3) is conservative BY DESIGN:
    only `pre` events with `log_decisions` off, no session state, and
    non-edit payloads are cached (edit payloads depend on on-disk
    file_content; logged calls must keep journal completeness). Cached
    verdicts expire after _CACHE_TTL_SEC so policy/journal drift self-heals.

Usage:
    python -m aetheris_microbrains.hook_daemon            # serve until Ctrl+C
    python -m aetheris_microbrains.hook_daemon --max-requests 1   # one-shot
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import socket
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

DAEMON_VERSION = 1
_DEFAULT_CONNECT_TIMEOUT = 0.2
_DEFAULT_RECV_TIMEOUT = 1.0
_CACHE_MAX_ENTRIES = 1024
_CACHE_TTL_SEC = 5.0
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_EDITISH_TOOLS = {
    "edit", "write", "multiedit", "notebookedit", "str_replace_editor",
    "write_file", "edit_file", "apply_patch", "repo_write", "format_code",
    "delete_file",
}


def get_daemon_state_path(state_dir: Optional[Path] = None) -> Path:
    """Resolve `<state_dir>/hook_daemon.json` (honors an explicit state dir)."""
    if state_dir is not None:
        return Path(state_dir) / "hook_daemon.json"
    from .core.runtime_paths import get_state_dir

    return get_state_dir() / "hook_daemon.json"


def _read_line(sock: socket.socket, max_bytes: int = _MAX_REQUEST_BYTES) -> str:
    """Read one newline-terminated line with a hard size cap."""
    buf = bytearray()
    while len(buf) < max_bytes:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        nl = buf.find(b"\n")
        if nl != -1:
            return buf[:nl].decode("utf-8", "replace")
    raise ValueError("request/response exceeded size cap or closed early")


def _atomic_write_json(target: Path, data: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2) + "\n")
            fh.flush()
        if os.name != "nt":
            try:
                os.chmod(tmp_name, 0o600)
            except OSError:
                pass
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class VerdictCache:
    """Tiny TTL + LRU cache for daemon verdicts (thread-safe for HTTP server workers)."""

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES, ttl_sec: float = _CACHE_TTL_SEC) -> None:
        self._max = max(1, int(max_entries))
        self._ttl = float(ttl_sec)
        self._store: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        expired = [k for k, (ts, _v) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            self._store.pop(k, None)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            item = self._store.get(key)
            if item is None:
                return None
            _ts, verdict = item
            self._store.move_to_end(key)
            return verdict

    def put(self, key: str, verdict: Dict[str, Any]) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            self._store[key] = (now, verdict)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)


def _payload_is_editish(payload: Any) -> bool:
    """Edit/write payloads depend on on-disk file content — never cached."""
    if not isinstance(payload, dict):
        return True
    tool = str(payload.get("tool_name") or payload.get("tool") or "").strip().lower()
    if tool in _EDITISH_TOOLS:
        return True
    tool_input = payload.get("tool_input")
    return isinstance(tool_input, dict) and any(
        bool(tool_input.get(k))
        for k in (
            "file_path", "path", "new_string", "content", "patch", "diff",
            "payload_diff", "notebook_path", "new_source", "edits",
        )
    )


def _resolve_under_cwd(raw_path: Any, caller_cwd: Optional[str]) -> Optional[str]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute() and caller_cwd:
        p = Path(caller_cwd) / p
    return str(p)


def _file_sig(raw_path: Optional[str]) -> Any:
    if not raw_path:
        return None
    try:
        st = Path(raw_path).stat()
        return (int(st.st_mtime_ns), int(st.st_size), int(getattr(st, "st_ino", 0)))
    except OSError:
        return None


def _merge_flags(daemon_flags: Dict[str, Any], req: Dict[str, Any]) -> Dict[str, Any]:
    """Merge daemon CLI defaults with client flags without clobbering daemon policy on unset client defaults."""
    merged = dict(daemon_flags or {})
    req_flags = req.get("flags") if isinstance(req.get("flags"), dict) else {}
    explicit = set(req.get("explicit_flags") or ())
    bool_keys = {
        "strict_ast",
        "log_decisions",
        "shadow",
        "codex_escalate_deny",
        "replace_tool_output",
        "semantic_guard",
        "context_guard",
    }
    for k, v in req_flags.items():
        if v is None:
            continue
        if k in bool_keys:
            if bool(v) or k in explicit or k not in merged:
                merged[k] = bool(v)
        else:
            if k in explicit or k not in merged:
                merged[k] = v
    return merged


def _cache_key(
    payload: Any,
    flags: Dict[str, Any],
    policy_sig: Any = None,
    mcp_sig: Any = None,
    pins_sig: Any = None,
) -> str:
    try:
        canonical = json.dumps(
            {
                "p": payload,
                "f": sorted((str(k), str(v)) for k, v in flags.items()),
                "pol": policy_sig,
                "mcp": mcp_sig,
                "pins": pins_sig,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        canonical = repr((payload, flags, policy_sig, mcp_sig, pins_sig))
    return hashlib.sha256(canonical.encode("utf-8", "ignore")).hexdigest()


_LIFECYCLE_EVENTS = frozenset({
    "session-start",
    "instructions-loaded",
    "user-prompt-submit",
    "post-tool-use-failure",
    "posttoolusefailure",
    "post-tool-batch",
    "posttoolbatch",
    "pre-compact",
    "precompact",
    "stop",
    "subagent-stop",
    "subagentstop",
    "permission-denied",
    "permissiondenied",
})


def evaluate_request(
    request: Dict[str, Any],
    cache: Optional[VerdictCache] = None,
) -> "tuple[Dict[str, Any], bool]":
    """Evaluate one daemon request; returns `(verdict, cache_hit)`.

    Cacheable only when: pre-event, no journal logging (journal completeness
    wins over micro-perf), no session state (cross-turn taint state mutates),
    non-shadow (shadow events are journal entries too), and a non-edit
    payload (on-disk file content may have changed).
    """
    if not isinstance(request, dict) or not isinstance(request.get("payload"), dict):
        return (
            {
                "decision": "escalate",
                "permissionDecision": "ask",
                "reason_code": "malformed_daemon_request",
                "label": "Daemon request missing or malformed 'payload'; failing closed",
                "layer": 0,
                "exit_code": 0,
            },
            False,
        )
    raw_payload = request.get("payload")
    event = str(request.get("event") or "pre").strip().lower()
    flags = request.get("flags") if isinstance(request.get("flags"), dict) else {}

    caller_cwd = str(flags.get("cwd") or "").strip() or None
    payload = raw_payload
    if isinstance(raw_payload, dict) and caller_cwd and not raw_payload.get("cwd"):
        payload = dict(raw_payload)
        payload["cwd"] = caller_cwd

    resolved_policy = _resolve_under_cwd(flags.get("policy_path"), caller_cwd)
    resolved_journal = _resolve_under_cwd(flags.get("journal_path"), caller_cwd)
    from .core.runtime_paths import get_default_session_state_path

    resolved_sess = (
        str(get_default_session_state_path())
        if str(flags.get("session_state_path") or "").strip().lower() == "auto"
        else _resolve_under_cwd(flags.get("session_state_path"), caller_cwd)
    )
    proj_dir = caller_cwd or (str(payload.get("cwd") or "").strip() if isinstance(payload, dict) else "") or None
    mcp_sig = (
        (_file_sig(str(Path(proj_dir) / ".mcp.json")), _file_sig(str(Path(proj_dir) / "mcp.json")))
        if proj_dir
        else None
    )
    from .core.mcp_tiers import get_pin_store_path

    pins_sig = _file_sig(str(get_pin_store_path()))

    cacheable = (
        cache is not None
        and event == "pre"
        and not flags.get("log_decisions")
        and not flags.get("session_state_path")
        and not flags.get("shadow")
        and not flags.get("context_guard")
        and not _payload_is_editish(payload)
    )
    key = (
        _cache_key(payload, flags, policy_sig=_file_sig(resolved_policy), mcp_sig=mcp_sig, pins_sig=pins_sig)
        if cacheable
        else None
    )
    if key is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit, True

    from .hook import evaluate_hook_payload, evaluate_post_tool_use_payload

    host = str(flags.get("host") or "claude")
    ev_norm = event.replace("_", "-")
    if ev_norm in _LIFECYCLE_EVENTS:
        from .core.lifecycle_engine import evaluate_lifecycle_event

        verdict = evaluate_lifecycle_event(
            ev_norm,
            payload,
            host=host,
            log_decisions=bool(flags.get("log_decisions")),
            journal_path=resolved_journal,
            session_state_path=resolved_sess,
        )
        return verdict, False
    if event == "post":
        verdict = evaluate_post_tool_use_payload(
            payload,
            log_decisions=bool(flags.get("log_decisions")),
            shadow_mode=bool(flags.get("shadow")),
            host=host,
            journal_path=resolved_journal,
            session_state_path=resolved_sess,
            policy_path=resolved_policy,
            replace_tool_output=bool(flags.get("replace_tool_output")),
            semantic_guard=bool(flags.get("semantic_guard")),
            context_guard=bool(flags.get("context_guard")),
        )
    else:
        verdict = evaluate_hook_payload(
            payload,
            strict_ast=bool(flags.get("strict_ast")),
            log_decisions=bool(flags.get("log_decisions")),
            shadow_mode=bool(flags.get("shadow")),
            host=host,
            codex_escalate_deny=bool(flags.get("codex_escalate_deny")),
            journal_path=resolved_journal,
            session_state_path=resolved_sess,
            profile=flags.get("profile"),
            policy_path=resolved_policy,
            sandbox_wrap=str(flags.get("sandbox_wrap") or "none"),
            context_guard=bool(flags.get("context_guard")),
        )
    if key is not None:
        cache.put(key, verdict)
    return verdict, False


def query_daemon(
    request: Dict[str, Any],
    state_path: Optional[Path] = None,
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
    recv_timeout: float = _DEFAULT_RECV_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Client side: ask the daemon for a verdict; None on ANY failure so the
    caller falls back to in-process evaluation."""
    p = Path(state_path) if state_path else get_daemon_state_path()
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(info, dict) or int(info.get("version") or 0) != DAEMON_VERSION:
            return None
        if str(info.get("mode") or "tcp").strip().lower() == "http":
            return None  # HTTP daemons are called by Claude Code directly; this TCP client must fall back in-process
        port = int(info.get("port") or 0)
        auth_value = str(info.get("auth") or "")
        if not port or not auth_value:
            return None
        req = dict(request)
        req["auth"] = auth_value
        req.setdefault("id", secrets.token_hex(8))
        with socket.create_connection(("127.0.0.1", port), timeout=connect_timeout) as conn:
            conn.settimeout(recv_timeout)
            conn.sendall((json.dumps(req) + "\n").encode("utf-8"))
            line = _read_line(conn)
        resp = json.loads(line)
        if not isinstance(resp, dict) or resp.get("error"):
            return None
        return resp.get("verdict")
    except Exception:
        return None


def _configure_loopback_socket(srv: socket.socket) -> None:
    """Prevent local port hijacking on Windows (SO_EXCLUSIVEADDRUSE); use SO_REUSEADDR on POSIX."""
    if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def ping_daemon(
    state_path: Optional[Path] = None,
    timeout: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """Probe the running daemon cleanly (supports both TCP and HTTP modes)."""
    p = Path(state_path) if state_path else get_daemon_state_path()
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            return None
        port = int(info.get("port") or 0)
        if not port:
            return None
        mode = str(info.get("mode") or "tcp").lower()
        if mode == "http":
            import urllib.request

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        auth_value = str(info.get("auth") or "")
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            req = {"action": "ping", "auth": auth_value, "id": secrets.token_hex(4)}
            conn.sendall((json.dumps(req) + "\n").encode("utf-8"))
            line = _read_line(conn)
        return json.loads(line)
    except Exception:
        return None


def serve(
    host: str = "127.0.0.1",
    port: int = 0,
    state_path: Optional[Path] = None,
    flags: Optional[Dict[str, Any]] = None,
    max_requests: Optional[int] = None,
) -> int:
    """Run the daemon accept loop until Ctrl+C / max_requests; returns 0."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _configure_loopback_socket(srv)
    srv.bind((host, port))
    srv.listen(4)
    bound_port = int(srv.getsockname()[1])
    auth_value = secrets.token_hex(16)
    target = Path(state_path) if state_path else get_daemon_state_path()
    if target.exists() and target.is_dir():
        target = target / "hook_daemon.json"
    info = {
        "mode": "tcp",
        "port": bound_port,
        "pid": os.getpid(),
        "auth": auth_value,
        "version": DAEMON_VERSION,
        "started_at": time.time(),
    }
    _atomic_write_json(target, info)
    daemon_flags = dict(flags or {})
    cache = VerdictCache()
    print(f"hook daemon listening on {host}:{bound_port} (state: {target}, version {DAEMON_VERSION})")
    served = 0
    try:
        while max_requests is None or served < max_requests:
            try:
                conn, _addr = srv.accept()
            except OSError:
                continue
            req_id: Any = None
            try:
                conn.settimeout(_DEFAULT_RECV_TIMEOUT)
                line = _read_line(conn)
                req = json.loads(line)
                req_id = req.get("id") if isinstance(req, dict) else None
                resp: Dict[str, Any]
                if not isinstance(req, dict) or not hmac.compare_digest(str(req.get("auth") or ""), auth_value):
                    served += 1
                    resp = {"id": req_id, "error": "auth"}
                elif req.get("action") == "ping":
                    resp = {"id": req_id, "ok": True, "version": DAEMON_VERSION, "mode": "tcp"}
                else:
                    served += 1
                    req.pop("auth", None)
                    req["flags"] = _merge_flags(daemon_flags, req)
                    verdict, cached = evaluate_request(req, cache=cache)
                    resp = {"id": req_id, "verdict": verdict, "cached": cached}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except Exception as exc:
                served += 1
                try:
                    conn.sendall((json.dumps({"id": req_id, "error": f"{type(exc).__name__}: {exc}"}) + "\n").encode("utf-8"))
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("hook daemon stopping (Ctrl+C)")
    finally:
        srv.close()
        try:
            if target.exists() and json.loads(target.read_text(encoding="utf-8")).get("pid") == os.getpid():
                target.unlink()
        except Exception:
            pass
    return 0


class _ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _gc_session_states(sessions_dir: Path, max_age_sec: float = 7 * 86400) -> int:
    """Best-effort removal of stale per-session taint files (returns removed count)."""
    removed = 0
    now = time.time()
    try:
        entries = list(Path(sessions_dir).glob("*.json"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if now - entry.stat().st_mtime > max_age_sec:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def serve_http(
    host: str = "127.0.0.1",
    port: int = 8765,
    state_path: Optional[Path] = None,
    flags: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    max_requests: Optional[int] = None,
) -> int:
    """Native Claude Code `type: "http"` hook endpoint (zero per-call Python spawn)."""
    auth_value = str(token or os.getenv("AETHERIS_DAEMON_TOKEN") or "").strip() or secrets.token_hex(16)
    daemon_flags = dict(flags or {})
    http_host_flag = str(daemon_flags.get("host") or "claude").strip().lower()
    if http_host_flag != "claude":
        print(f"hook HTTP daemon requires host='claude' (got '{http_host_flag}'): HTTP hook payloads follow the Claude Code contract")
        return 2
    cache = VerdictCache()
    target = Path(state_path) if state_path else get_daemon_state_path()
    if target.exists() and target.is_dir():
        target = target / "hook_daemon.json"
    try:
        _gc_session_states(target.parent / "sessions")
    except Exception:
        pass
    served_lock = threading.Lock()
    served_count = 0

    class _HookHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # silence stderr noise
            return

        def _send_json(self, code: int, body: Dict[str, Any]) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/health":
                self._send_json(200, {"ok": True, "version": DAEMON_VERSION, "mode": "http"})
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            nonlocal served_count
            path = self.path.split("?")[0]
            if path in {"/v1/hook/pre", "/v1/hook/post"}:
                ev = "post" if path == "/v1/hook/post" else "pre"
            elif path.startswith("/v1/hook/event/"):
                cand_ev = path[len("/v1/hook/event/") :].strip("/").replace("_", "-").lower()
                if cand_ev not in _LIFECYCLE_EVENTS:
                    self._send_json(404, {"error": "unknown_lifecycle_event"})
                    return
                ev = cand_ev
            else:
                self._send_json(404, {"error": "not_found"})
                return
            auth_hdr = self.headers.get("Authorization", "")
            supplied = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else ""
            if not hmac.compare_digest(supplied, auth_value):
                self._send_json(401, {"error": "unauthorized"})
                return
            try:
                cl = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                self._send_json(400, {"error": "invalid_content_length"})
                return
            if cl < 0 or cl > _MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "payload_too_large"})
                return
            raw_body = self.rfile.read(cl)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"error": f"invalid_json:{exc}"})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "expected_json_object"})
                return
            req = {"event": ev, "payload": payload, "flags": dict(daemon_flags)}
            verdict, _cached = evaluate_request(req, cache=cache)
            if daemon_flags.get("shadow"):
                out_body: Dict[str, Any] = {}
            elif verdict.get("hookSpecificOutput") is not None:
                hso = verdict["hookSpecificOutput"]
                if (
                    ev == "pre"
                    and verdict.get("permissionDecision") == "allow"
                    and not daemon_flags.get("auto_allow")
                    and "updatedInput" not in hso
                ):
                    out_body = {}
                elif ev in _LIFECYCLE_EVENTS and isinstance(hso, dict) and "hookEventName" not in hso:
                    out_body = hso
                else:
                    out_body = {"hookSpecificOutput": hso}
            else:
                out_body = {}
            self._send_json(200, out_body)
            with served_lock:
                served_count += 1
                if max_requests is not None and served_count >= max_requests:
                    threading.Thread(target=httpd.shutdown, daemon=True).start()

    try:
        httpd = _ExclusiveThreadingHTTPServer((host, port), _HookHTTPHandler)
    except OSError as exc:
        print(f"hook HTTP daemon failed to bind {host}:{port}: {exc} (port already in use? choose another --port)")
        return 1
    bound_port = int(httpd.server_address[1])
    _atomic_write_json(
        target,
        {
            "mode": "http",
            "port": bound_port,
            "url": f"http://{host}:{bound_port}",
            "pid": os.getpid(),
            "auth": auth_value,
            "version": DAEMON_VERSION,
            "started_at": time.time(),
        },
    )
    print(f"hook HTTP daemon listening on http://{host}:{bound_port} (state: {target}, version {DAEMON_VERSION})")
    try:
        httpd.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        print("hook HTTP daemon stopping (Ctrl+C)")
    finally:
        httpd.server_close()
        try:
            if target.exists() and json.loads(target.read_text(encoding="utf-8")).get("pid") == os.getpid():
                target.unlink()
        except Exception:
            pass
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aetheris_microbrains.hook_daemon",
        description="Warm evaluator daemon (TCP or native Claude Code HTTP hook mode; falls back safely).",
    )
    parser.add_argument("--port", type=int, default=0, help="TCP/HTTP port (0 = ephemeral)")
    parser.add_argument("--http", action="store_true", help="Serve native Claude Code `type: http` hook endpoints (/v1/hook/pre, /v1/hook/post, /health)")
    parser.add_argument("--state-dir", default=None, help="Override the state directory for hook_daemon.json")
    parser.add_argument("--max-requests", type=int, default=None, help="Stop after N requests (testing)")
    parser.add_argument("--log-decisions", action="store_true", help="Log decisions to the journal")
    parser.add_argument("--shadow", action="store_true", help="Shadow mode passthrough")
    parser.add_argument("--strict-ast", action="store_true", help="Strict AST passthrough")
    parser.add_argument("--auto-allow", action="store_true", help="Emit allow verdicts explicitly in HTTP mode")
    parser.add_argument("--profile", default=None, help="Policy profile passthrough")
    parser.add_argument("--policy", default=None, help="Policy file passthrough")
    parser.add_argument("--journal-path", default=None, help="Journal path passthrough")
    parser.add_argument("--session-state", default=None, help="Session state path (or 'auto') passthrough")
    parser.add_argument("--replace-tool-output", action="store_true", help="Emit updatedToolOutput on post-tool redaction")
    parser.add_argument("--semantic-guard", action="store_true", help="Enable semantic prompt-injection classifier on post-tool")
    parser.add_argument(
        "--sandbox-wrap",
        choices=["none", "bwrap", "seatbelt", "auto"],
        default="none",
        help="Optional OS sandbox wrapper (`bwrap`, `seatbelt`, `auto`) emitted via updatedInput on allowed Bash calls",
    )
    parser.add_argument("--context-guard", action="store_true", help="Enable context-hygiene gating passthrough")
    parser.add_argument(
        "--no-session-state",
        dest="no_session_state",
        action="store_true",
        help="Disable cross-turn session taint state (weaker IFC; also enables verdict caching)",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    flags = {
        "log_decisions": bool(args.log_decisions),
        "shadow": bool(args.shadow),
        "strict_ast": bool(args.strict_ast),
        "auto_allow": bool(args.auto_allow),
        "replace_tool_output": bool(args.replace_tool_output),
        "semantic_guard": bool(args.semantic_guard),
        "sandbox_wrap": str(args.sandbox_wrap or "none"),
        "context_guard": bool(args.context_guard),
        "profile": args.profile,
        "policy_path": args.policy,
        "journal_path": args.journal_path,
        "session_state_path": None if args.no_session_state else (args.session_state or "auto"),
    }
    state_path = get_daemon_state_path(Path(args.state_dir)) if args.state_dir else None
    if args.http:
        port_val = args.port if args.port else 8765
        return serve_http(port=port_val, state_path=state_path, flags=flags, max_requests=args.max_requests)
    return serve(port=args.port, state_path=state_path, flags=flags, max_requests=args.max_requests)


if __name__ == "__main__":
    sys.exit(main())
