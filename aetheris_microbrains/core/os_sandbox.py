"""OS-Level Execution Sandboxing & Domain-Allowlisted Network Proxy (stdlib-only).

Provides two defense-in-depth primitives:
  1. `build_os_sandbox_command(...)`: wraps approved shell commands with
     OS isolation backends (`srt`, Linux `bwrap` / `aetheris-gate` Landlock LSM,
     macOS `sandbox-exec` Seatbelt, Windows `aetheris-gate` Job Object + secret scrub).
  2. `DomainAllowlistProxy`: lightweight loopback HTTP / CONNECT proxy enforcing
     host-level domain allowlisting while rejecting raw IPs, loopback, and metadata URLs.
"""

from __future__ import annotations

import ipaddress
import os
import select
import shlex
import shutil
import socket
import socketserver
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from urllib.parse import urlsplit

_BLOCKED_LITERAL_HOSTS = frozenset({
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata",
})


def _is_domain_allowlisted(host: str, allowed_domains: Sequence[str]) -> bool:
    clean = str(host or "").strip().lower().rstrip(".")
    if not clean or clean in _BLOCKED_LITERAL_HOSTS or clean.endswith(".local"):
        return False
    try:
        ipaddress.ip_address(clean.strip("[]"))
        return False  # Never allow raw IP literals via domain allowlist
    except ValueError:
        pass
    for dom in allowed_domains:
        d = str(dom or "").strip().lower().lstrip("*.").rstrip(".")
        if d and (clean == d or clean.endswith("." + d)):
            return True
    return False


def _relay_bidirectional(client: socket.socket, upstream: socket.socket, timeout: float = 5.0) -> None:
    sockets = [client, upstream]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], timeout)
            if not readable:
                break
            for sock in readable:
                data = sock.recv(16384)
                if not data:
                    return
                peer = upstream if sock is client else client
                peer.sendall(data)
    except OSError:
        return


def _extract_host_port(request_line: str, headers: Dict[str, str]) -> Tuple[str, int, str]:
    parts = request_line.split()
    method = parts[0].upper() if parts else ""
    target = parts[1] if len(parts) > 1 else ""
    if method == "CONNECT":
        if ":" in target:
            h, p_str = target.rsplit(":", 1)
            try:
                return h.strip("[]"), int(p_str), method
            except ValueError:
                return h.strip("[]"), 443, method
        return target.strip("[]"), 443, method
    if target.startswith(("http://", "https://")):
        parsed = urlsplit(target)
        default_port = 443 if parsed.scheme == "https" else 80
        return str(parsed.hostname or ""), int(parsed.port or default_port), method
    host_hdr = headers.get("host", "")
    if ":" in host_hdr:
        h, p_str = host_hdr.rsplit(":", 1)
        try:
            return h.strip("[]"), int(p_str), method
        except ValueError:
            return h.strip("[]"), 80, method
    return host_hdr.strip("[]"), 80, method


class _AllowlistProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, allowed_domains: Sequence[str], connect_timeout: float = 5.0) -> None:
        self.allowed_domains = tuple(str(d).strip().lower() for d in allowed_domains if str(d).strip())
        self.connect_timeout = float(connect_timeout)
        super().__init__(("127.0.0.1", 0), _AllowlistProxyHandler)


class _AllowlistProxyHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: _AllowlistProxyServer = self.server  # type: ignore[assignment]
        try:
            raw_line = self.rfile.readline(8192).decode("latin-1", errors="replace").strip()
            if not raw_line:
                return
            headers: Dict[str, str] = {}
            header_lines = []
            for _ in range(100):
                line = self.rfile.readline(8192).decode("latin-1", errors="replace")
                if not line or line in ("\r\n", "\n"):
                    break
                header_lines.append(line)
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            host, port, method = _extract_host_port(raw_line, headers)
            if not _is_domain_allowlisted(host, server.allowed_domains):
                body = b"domain_not_allowlisted"
                self.wfile.write(
                    b"HTTP/1.1 403 Forbidden\r\n"
                    b"Content-Type: text/plain\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode("ascii")
                    + b"X-Aetheris-Reason: domain_not_allowlisted\r\n"
                    b"Connection: close\r\n\r\n"
                    + body
                )
                self.wfile.flush()
                return

            with socket.create_connection((host, port), timeout=server.connect_timeout) as upstream:
                if method == "CONNECT":
                    self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    self.wfile.flush()
                else:
                    parts = raw_line.split(" ", 2)
                    req_target = parts[1] if len(parts) > 1 else "/"
                    if req_target.startswith(("http://", "https://")):
                        sp = urlsplit(req_target)
                        req_target = (sp.path or "/") + (f"?{sp.query}" if sp.query else "")
                    ver = parts[2] if len(parts) > 2 else "HTTP/1.1"
                    fwd_head = f"{method} {req_target} {ver}\r\n" + "".join(
                        ln for ln in header_lines if not ln.lower().startswith("proxy-connection:")
                    ) + "\r\n"
                    upstream.sendall(fwd_head.encode("latin-1"))
                _relay_bidirectional(self.connection, upstream)
        except Exception:
            try:
                self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass


class DomainAllowlistProxy:
    """Loopback HTTP/CONNECT egress proxy restricting outbound traffic to `allowed_domains`."""

    def __init__(self, allowed_domains: Sequence[str], *, connect_timeout: float = 5.0) -> None:
        self.allowed_domains = tuple(str(d).strip().lower() for d in allowed_domains if str(d).strip())
        self.connect_timeout = float(connect_timeout)
        self._srv: Optional[_AllowlistProxyServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        if self._srv is None:
            raise RuntimeError("DomainAllowlistProxy is not running")
        return int(self._srv.server_address[1])

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def env_overrides(self) -> Dict[str, str]:
        url = self.proxy_url
        return {"HTTP_PROXY": url, "HTTPS_PROXY": url, "ALL_PROXY": url, "NO_PROXY": ""}

    def start(self) -> "DomainAllowlistProxy":
        if self._srv is None:
            self._srv = _AllowlistProxyServer(self.allowed_domains, connect_timeout=self.connect_timeout)
            self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._srv is not None:
            try:
                self._srv.shutdown()
                self._srv.server_close()
            except OSError:
                pass
            self._srv = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "DomainAllowlistProxy":
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()


def find_aetheris_gate_binary() -> Optional[str]:
    """Locate the native `aetheris-gate` Rust binary on PATH or in the repo build directory."""
    try:
        on_path = shutil.which("aetheris-gate")
    except (AttributeError, OSError):
        on_path = None
    if on_path:
        return on_path
    exe_name = "aetheris-gate.exe" if os.name == "nt" else "aetheris-gate"
    repo_bin = Path(__file__).resolve().parents[2] / "rust" / "aetheris-gate" / "target" / "release" / exe_name
    if repo_bin.is_file():
        return str(repo_bin)
    return None


def _win_quote(tok: str) -> str:
    s = str(tok)
    if not s or any(ch in s for ch in (' ', '\t', '"', '&', '|', '<', '>')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def build_os_sandbox_command(
    raw_cmd: str,
    sandbox_mode: str = "none",
    project_dir: Optional[str] = None,
    *,
    deny_net: bool = False,
    strip_secrets: bool = True,
    gate_bin: Optional[str] = None,
) -> Optional[str]:
    """Wrap an allowed shell command with OS sandboxing (`srt`, `bwrap`, `seatbelt`, `aetheris-gate`)."""
    mode = str(sandbox_mode or "none").strip().lower()
    if mode in {"", "none", "off", "false"}:
        return None
    if mode in {"bwrap", "seatbelt"} and sys.platform == "win32":
        return None
    resolved_gate: Optional[str] = None
    if mode in {"auto", "aetheris-gate", "landlock", "job"}:
        resolved_gate = gate_bin or find_aetheris_gate_binary()
    if mode == "auto":
        if shutil.which("srt"):
            mode = "srt"
        elif sys.platform.startswith("linux") and shutil.which("bwrap"):
            mode = "bwrap"
        elif sys.platform == "darwin" and shutil.which("sandbox-exec"):
            mode = "seatbelt"
        elif resolved_gate is not None:
            mode = "aetheris-gate"
        else:
            return None

    if mode == "srt":
        return f"srt -- /bin/sh -c {shlex.quote(raw_cmd)}"

    proj = str(Path(project_dir or os.getcwd()).resolve())
    quoted_cmd = shlex.quote(raw_cmd)
    if mode == "bwrap":
        return (
            f"bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp "
            f"--bind {shlex.quote(proj)} {shlex.quote(proj)} --chdir {shlex.quote(proj)} "
            f"-- /bin/sh -c {quoted_cmd}"
        )
    if mode == "seatbelt":
        sbpl_proj = proj.replace("\\", "\\\\").replace('"', '\\"')
        net_clause = "(deny network*)" if deny_net else ""
        profile = (
            "(version 1)(allow default)"
            f"{net_clause}"
            '(deny file-write* (subpath "/"))'
            f'(allow file-write* (subpath "{sbpl_proj}") (subpath "/tmp") (subpath "/private/tmp") (subpath "/private/var/folders"))'
        )
        return f"sandbox-exec -p {shlex.quote(profile)} /bin/sh -c {quoted_cmd}"
    if mode in {"aetheris-gate", "landlock", "job"}:
        if resolved_gate is None:
            return None
        flags = ["--cwd", proj, "--rw", proj]
        if strip_secrets:
            flags.append("--strip-secrets")
        if deny_net:
            flags.append("--deny-net")
        if sys.platform == "win32":
            q_parts = [_win_quote(resolved_gate), "sandbox", *(_win_quote(f) for f in flags), "--", "cmd.exe", "/d", "/s", "/c", _win_quote(raw_cmd)]
            return " ".join(q_parts)
        q_parts = [shlex.quote(resolved_gate), "sandbox", *(shlex.quote(f) for f in flags), "--", "/bin/sh", "-c", quoted_cmd]
        return " ".join(q_parts)
    return None
