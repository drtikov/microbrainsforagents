"""Pure-ASGI Bearer authentication middleware for remote Microbrains MCP deployments."""

from __future__ import annotations

import hmac
import os
from typing import Any, Optional

from starlette.responses import JSONResponse

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def resolve_bearer_token(env_var: str = "AETHERIS_MCP_BEARER_TOKEN") -> Optional[str]:
    """Read an optional Bearer token from ``env_var``, returning None when unset/blank."""
    name = str(env_var or "").strip()
    if not name:
        return None
    raw = os.environ.get(name, "")
    token = raw.strip()
    return token or None


def is_loopback_host(host: str) -> bool:
    """Return True when ``host`` binds only to a local loopback interface."""
    return str(host or "").strip().lower() in _LOOPBACK_HOSTS


class BearerAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on ``path`` (and optionally ``/health``)."""

    def __init__(
        self,
        app: Any,
        *,
        token: str,
        path: str = "/mcp",
        protect_health: bool = False,
    ) -> None:
        if not token or not token.isascii() or any(ch.isspace() for ch in token):
            raise ValueError("A non-empty ASCII bearer token without whitespace is required")
        self.app = app
        self.token = token
        self.path = path.rstrip("/") or "/"
        self.protect_health = bool(protect_health)

    def _is_protected(self, req_path: str) -> bool:
        normalized = req_path.rstrip("/") or "/"
        if self.path == "/":
            return True
        if normalized == self.path or normalized.startswith(self.path + "/"):
            return True
        if self.protect_health and (normalized == "/health" or normalized.startswith("/health/")):
            return True
        return False

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and self._is_protected(str(scope.get("path") or "")):
            headers = dict(scope.get("headers") or ())
            raw_auth = headers.get(b"authorization", b"")
            try:
                value = raw_auth.decode("latin-1").strip()
            except Exception:
                value = ""
            scheme, sep, supplied = value.partition(" ")
            valid = (
                sep == " "
                and scheme.lower() == "bearer"
                and bool(supplied)
                and hmac.compare_digest(supplied.strip(), self.token)
            )
            if not valid:
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
