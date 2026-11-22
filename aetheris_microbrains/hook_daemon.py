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
import time
from collections import OrderedDict
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
    """Tiny TTL + LRU cache for daemon verdicts."""

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES, ttl_sec: float = _CACHE_TTL_SEC) -> None:
        self._max = max(1, int(max_entries))
        self._ttl = float(ttl_sec)
        self._store: "OrderedDict[str, tuple]" = OrderedDict()

    def _purge_expired(self, now: float) -> None:
        while self._store:
            _key, (ts, _v) = next(iter(self._store.items()))
            if now - ts <= self._ttl:
                break
            self._store.popitem(last=False)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        self._purge_expired(now)
        item = self._store.get(key)
        if item is None:
            return None
        ts, verdict = item
        if now - ts > self._ttl:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return verdict

    def put(self, key: str, verdict: Dict[str, Any]) -> None:
        now = time.monotonic()
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
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _cache_key(
    payload: Any,
    flags: Dict[str, Any],
    policy_sig: Any = None,
    mcp_sig: Any = None,
) -> str:
    try:
        canonical = json.dumps(
            {
                "p": payload,
                "f": sorted((str(k), str(v)) for k, v in flags.items()),
                "pol": policy_sig,
                "mcp": mcp_sig,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        canonical = repr((payload, flags, policy_sig, mcp_sig))
    return hashlib.sha256(canonical.encode("utf-8", "ignore")).hexdigest()


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
    resolved_sess = _resolve_under_cwd(flags.get("session_state_path"), caller_cwd)
    proj_dir = caller_cwd or (str(payload.get("cwd") or "").strip() if isinstance(payload, dict) else "") or None
    mcp_sig = _file_sig(str(Path(proj_dir) / ".mcp.json")) if proj_dir else None

    cacheable = (
        cache is not None
        and event == "pre"
        and not flags.get("log_decisions")
        and not flags.get("session_state_path")
        and not flags.get("shadow")
        and not _payload_is_editish(payload)
    )
    key = (
        _cache_key(payload, flags, policy_sig=_file_sig(resolved_policy), mcp_sig=mcp_sig)
        if cacheable
        else None
    )
    if key is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit, True

    from .hook import evaluate_hook_payload, evaluate_post_tool_use_payload

    host = str(flags.get("host") or "claude")
    if event == "post":
        verdict = evaluate_post_tool_use_payload(
            payload,
            log_decisions=bool(flags.get("log_decisions")),
            shadow_mode=bool(flags.get("shadow")),
            host=host,
            journal_path=resolved_journal,
            session_state_path=resolved_sess,
            policy_path=resolved_policy,
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


def serve(
    host: str = "127.0.0.1",
    port: int = 0,
    state_path: Optional[Path] = None,
    flags: Optional[Dict[str, Any]] = None,
    max_requests: Optional[int] = None,
) -> int:
    """Run the daemon accept loop until Ctrl+C / max_requests; returns 0."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    bound_port = int(srv.getsockname()[1])
    auth_value = secrets.token_hex(16)
    target = Path(state_path) if state_path else get_daemon_state_path()
    if target.exists() and target.is_dir():
        target = target / "hook_daemon.json"
    info = {
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
            served += 1
            try:
                conn.settimeout(_DEFAULT_RECV_TIMEOUT)
                line = _read_line(conn)
                req = json.loads(line)
                resp: Dict[str, Any]
                if not isinstance(req, dict) or not hmac.compare_digest(str(req.get("auth") or ""), auth_value):
                    resp = {"id": req.get("id") if isinstance(req, dict) else None, "error": "auth"}
                else:
                    req.pop("auth", None)
                    merged_flags = dict(daemon_flags)
                    merged_flags.update(req.get("flags") or {})
                    req["flags"] = merged_flags
                    verdict, cached = evaluate_request(req, cache=cache)
                    resp = {"id": req.get("id"), "verdict": verdict, "cached": cached}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except Exception as exc:
                try:
                    conn.sendall((json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n").encode("utf-8"))
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


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aetheris_microbrains.hook_daemon",
        description="Warm evaluator daemon for the Aetheris hook (falls back safely).",
    )
    parser.add_argument("--port", type=int, default=0, help="TCP port (0 = ephemeral)")
    parser.add_argument("--state-dir", default=None, help="Override the state directory for hook_daemon.json")
    parser.add_argument("--max-requests", type=int, default=None, help="Stop after N requests (testing)")
    parser.add_argument("--log-decisions", action="store_true", help="Log decisions to the journal")
    parser.add_argument("--shadow", action="store_true", help="Shadow mode passthrough")
    parser.add_argument("--strict-ast", action="store_true", help="Strict AST passthrough")
    parser.add_argument("--profile", default=None, help="Policy profile passthrough")
    parser.add_argument("--policy", default=None, help="Policy file passthrough")
    parser.add_argument("--journal-path", default=None, help="Journal path passthrough")
    parser.add_argument("--session-state", default=None, help="Session state path passthrough")
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
        "profile": args.profile,
        "policy_path": args.policy,
        "journal_path": args.journal_path,
        "session_state_path": args.session_state,
    }
    state_path = get_daemon_state_path(Path(args.state_dir)) if args.state_dir else None
    return serve(port=args.port, state_path=state_path, flags=flags, max_requests=args.max_requests)


if __name__ == "__main__":
    sys.exit(main())
