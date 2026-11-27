"""Capability-token hook enforcement and OS sandbox policy compilation (stdlib-only).

Bridges HMAC-SHA256 chained capability tokens (`aetheris_microbrains.core.capability_tokens`)
with hook-time authorization and OS-level execution sandboxes
(`aetheris_microbrains.core.os_sandbox`).

Key capabilities:
- `extract_capability_token(payload, env=None)`: extract a `CapabilityToken` from a hook
  payload (`payload["capability_token"]`) or the `AETHERIS_CAPABILITY_TOKEN` environment variable.
- `enforce_capability_for_hook(...)`: verify cryptographic integrity and caveat authorization
  for a single hook invocation, failing closed on tampered or malformed tokens.
- `compile_sandbox_policy_from_token(token)`: compute the monotonic intersection across all
  caveats in `token` for `allowed_domains`, `read_globs`, `write_globs`, `allow_shell`, and
  `allowed_secrets` so OS sandboxes (`build_os_sandbox_command`, `DomainAllowlistProxy`) and
  `SecretBroker` enforce the exact capability envelope.
"""

from __future__ import annotations

import fnmatch
import json
import os
import posixpath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from aetheris_microbrains.core.capability_tokens import (
    CapabilityToken,
    verify_capability_and_authorize,
)
from aetheris_microbrains.core.os_sandbox import (
    DomainAllowlistProxy,
    build_os_sandbox_command,
)

CAPABILITY_TOKEN_ENV_VAR = "AETHERIS_CAPABILITY_TOKEN"


def _normalize_glob(pattern: str) -> str:
    raw = str(pattern or "").strip().replace("\\", "/")
    if not raw:
        return ""
    norm = posixpath.normpath(raw)
    if norm.startswith("./"):
        norm = norm[2:]
    return norm


def _normalize_domain(domain: str) -> str:
    return str(domain or "").strip().lower().rstrip(".")


def _has_wildcard(pattern: str) -> bool:
    return any(ch in pattern for ch in ("*", "?", "["))


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _domain_subsumes(wider: str, narrower: str) -> bool:
    """Return True if domain pattern `wider` permits every host permitted by `narrower`."""
    w = _normalize_domain(wider)
    n = _normalize_domain(narrower)
    if not w or not n:
        return False
    if w == n or w == "*":
        return True
    if w.startswith("*."):
        suffix = w[1:]  # e.g. ".github.com"
        if n.startswith("*."):
            n_base = n[2:]
            return n_base.endswith(suffix)
        return n.endswith(suffix) and len(n) > len(suffix)
    if not _has_wildcard(n):
        return fnmatch.fnmatchcase(n, w)
    return False


def _intersect_domains(
    current: Optional[List[str]],
    incoming: Tuple[str, ...],
) -> List[str]:
    norm_in = _dedupe_preserve_order([_normalize_domain(d) for d in incoming if _normalize_domain(d)])
    if current is None:
        return norm_in
    out: List[str] = []
    for a in current:
        for b in norm_in:
            if a == b:
                out.append(a)
            elif _domain_subsumes(a, b):
                out.append(b)
            elif _domain_subsumes(b, a):
                out.append(a)
    return _dedupe_preserve_order(out)


def _match_glob_literal(norm_path: str, pattern: str) -> bool:
    norm_pat = _normalize_glob(pattern)
    if not norm_pat:
        return False
    if fnmatch.fnmatchcase(norm_path, norm_pat):
        return True
    if "/**/" in norm_pat:
        collapsed = norm_pat.replace("/**/", "/")
        if fnmatch.fnmatchcase(norm_path, collapsed):
            return True
    if norm_pat.endswith("/**"):
        prefix = norm_pat[:-3]
        if norm_path == prefix or norm_path.startswith(prefix + "/"):
            return True
    return False


def _glob_subsumes(wider: str, narrower: str) -> bool:
    """Return True if glob pattern `wider` covers every path covered by `narrower`."""
    w = _normalize_glob(wider)
    n = _normalize_glob(narrower)
    if not w or not n:
        return False
    if w == n:
        return True
    if w in ("*", "**"):
        return True
    if w == "/**":
        return n.startswith("/")
    if w.endswith("/**"):
        w_prefix = w[:-3]
        if n.endswith("/**"):
            n_prefix = n[:-3]
            return n_prefix == w_prefix or n_prefix.startswith(w_prefix + "/")
        return n == w_prefix or n.startswith(w_prefix + "/")
    if not _has_wildcard(n):
        return _match_glob_literal(n, w)
    if n.endswith("/**"):
        return False
    return fnmatch.fnmatchcase(n, w)


def _intersect_globs(
    current: Optional[List[str]],
    incoming: Tuple[str, ...],
) -> List[str]:
    norm_in = _dedupe_preserve_order([_normalize_glob(g) for g in incoming if _normalize_glob(g)])
    if current is None:
        return norm_in
    out: List[str] = []
    for a in current:
        for b in norm_in:
            if a == b:
                out.append(a)
            elif _glob_subsumes(a, b):
                out.append(b)
            elif _glob_subsumes(b, a):
                out.append(a)
    return _dedupe_preserve_order(out)


def _pattern_subsumes(wider: str, narrower: str, *, case_insensitive: bool = False) -> bool:
    w = str(wider or "").strip()
    n = str(narrower or "").strip()
    if case_insensitive:
        w = w.lower()
        n = n.lower()
    if not w or not n:
        return False
    if w == n or w == "*":
        return True
    if not _has_wildcard(n):
        return fnmatch.fnmatchcase(n, w)
    if w.endswith("*") and not _has_wildcard(w[:-1]) and n.startswith(w[:-1]):
        return True
    return False


def _intersect_patterns(
    current: Optional[List[str]],
    incoming: Tuple[str, ...],
    *,
    case_insensitive: bool = False,
) -> List[str]:
    cleaned = []
    for item in incoming:
        s = str(item or "").strip()
        if case_insensitive:
            s = s.lower()
        if s:
            cleaned.append(s)
    norm_in = _dedupe_preserve_order(cleaned)
    if current is None:
        return norm_in
    out: List[str] = []
    for a in current:
        for b in norm_in:
            if a == b:
                out.append(a)
            elif _pattern_subsumes(a, b, case_insensitive=case_insensitive):
                out.append(b)
            elif _pattern_subsumes(b, a, case_insensitive=case_insensitive):
                out.append(a)
    return _dedupe_preserve_order(out)


def _raw_token_from_sources(
    payload: Optional[Mapping[str, Any]],
    env: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, Any]:
    """Return ``(present, raw_value)`` from payload or environment."""
    if isinstance(payload, Mapping) and "capability_token" in payload:
        raw = payload.get("capability_token")
        if raw is not None and raw != "":
            return True, raw

    source_env = os.environ if env is None else env
    if isinstance(source_env, Mapping):
        raw_env = source_env.get(CAPABILITY_TOKEN_ENV_VAR)
        if raw_env is not None and raw_env != "":
            return True, raw_env

    return False, None


def _parse_capability_token(raw: Any) -> Optional[CapabilityToken]:
    if isinstance(raw, CapabilityToken):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        raw = parsed
    if isinstance(raw, Mapping):
        try:
            return CapabilityToken.from_dict(raw)
        except Exception:
            return None
    return None


def extract_capability_token(
    payload: Optional[Mapping[str, Any]],
    env: Optional[Mapping[str, Any]] = None,
) -> Optional[CapabilityToken]:
    """Extract a `CapabilityToken` from `payload["capability_token"]` or `AETHERIS_CAPABILITY_TOKEN`.

    Returns ``None`` if no token is present or if the token cannot be parsed.
    """
    present, raw = _raw_token_from_sources(payload, env=env)
    if not present:
        return None
    return _parse_capability_token(raw)


def enforce_capability_for_hook(
    payload: Optional[Mapping[str, Any]],
    *,
    root_secret: Union[bytes, str],
    tool_name: str,
    path: Optional[str] = None,
    is_write: bool = False,
    is_shell: bool = False,
    domain: Optional[str] = None,
    secret_handle: Optional[str] = None,
    env: Optional[Mapping[str, Any]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify and enforce a capability token (if present) for a hook event.

    Returns a dictionary with keys:
    ``enforced``, ``allowed``, ``reason``, ``decision`` (plus optional metadata
    ``subject``, ``chain_depth``, and ``token_id`` when enforced).
    """
    present, raw = _raw_token_from_sources(payload, env=env)
    if not present:
        return {
            "enforced": False,
            "allowed": True,
            "reason": "no_capability_token",
            "decision": "allow",
        }

    token = _parse_capability_token(raw)
    if token is None:
        return {
            "enforced": True,
            "allowed": False,
            "reason": "malformed_capability_token",
            "decision": "deny",
        }

    auth = verify_capability_and_authorize(
        token,
        root_secret,
        tool_name=tool_name,
        path=path,
        is_write=is_write,
        is_shell=is_shell,
        secret_handle=secret_handle,
        domain=domain,
        now=now,
    )
    allowed = bool(auth.get("authorized", False))
    return {
        "enforced": True,
        "allowed": allowed,
        "reason": str(auth.get("reason") or ("authorized" if allowed else "denied")),
        "decision": str(auth.get("decision") or ("allow" if allowed else "deny")),
        "subject": str(auth.get("subject") or ""),
        "chain_depth": int(auth.get("chain_depth") or 0),
        "token_id": token.token_id,
    }


def compile_sandbox_policy_from_token(token: CapabilityToken) -> Dict[str, Any]:
    """Compile the monotonic intersection of all caveats in `token` into an OS sandbox policy.

    The resulting policy dictionary can be consumed directly by:
    - `build_os_sandbox_command` (`allow_shell`, `deny_net`, `strip_secrets`, `write_globs`, `read_globs`)
    - `DomainAllowlistProxy` (`allowed_domains`)
    - `SecretBroker` (`allowed_secrets`)
    """
    if not isinstance(token, CapabilityToken):
        raise TypeError(f"token must be a CapabilityToken, got {type(token).__name__}")

    allowed_domains: Optional[List[str]] = None
    read_globs: Optional[List[str]] = None
    write_globs: Optional[List[str]] = None
    allowed_secrets: Optional[List[str]] = None
    allowed_tools: Optional[List[str]] = None
    allow_shell: bool = True
    expires_at: Optional[float] = None
    subagent_id: Optional[str] = None
    subagent_conflict: bool = False
    effective_depth: Optional[int] = None

    for idx, cav in enumerate(token.caveats):
        if idx > 0 and effective_depth is not None:
            effective_depth = max(-1, effective_depth - 1)

        if cav.depth_remaining is not None:
            effective_depth = (
                cav.depth_remaining
                if effective_depth is None
                else min(effective_depth, cav.depth_remaining)
            )

        if cav.subagent_id is not None:
            if subagent_id is None:
                subagent_id = cav.subagent_id
            elif subagent_id != cav.subagent_id:
                subagent_conflict = True

        if cav.expires_at is not None:
            expires_at = cav.expires_at if expires_at is None else min(expires_at, cav.expires_at)

        if cav.allow_shell is False:
            allow_shell = False

        if cav.allowed_domains is not None:
            allowed_domains = _intersect_domains(allowed_domains, cav.allowed_domains)

        if cav.read_globs is not None:
            read_globs = _intersect_globs(read_globs, cav.read_globs)

        if cav.write_globs is not None:
            write_globs = _intersect_globs(write_globs, cav.write_globs)

        if cav.allowed_secrets is not None:
            allowed_secrets = _intersect_patterns(allowed_secrets, cav.allowed_secrets, case_insensitive=False)

        if cav.allowed_tools is not None:
            allowed_tools = _intersect_patterns(allowed_tools, cav.allowed_tools, case_insensitive=True)

    deny_net = allowed_domains is not None and len(allowed_domains) == 0
    strip_secrets = True

    return {
        "token_id": token.token_id,
        "subject": token.subject,
        "chain_depth": len(token.caveats),
        "allowed_domains": allowed_domains,
        "read_globs": read_globs,
        "write_globs": write_globs,
        "allow_shell": allow_shell,
        "allowed_secrets": allowed_secrets,
        "allowed_tools": allowed_tools,
        "deny_net": deny_net,
        "strip_secrets": strip_secrets,
        "expires_at": expires_at,
        "subagent_id": subagent_id,
        "subagent_conflict": subagent_conflict,
        "depth_remaining": effective_depth,
    }


def build_capability_sandbox_command(
    raw_cmd: str,
    token: CapabilityToken,
    *,
    sandbox_mode: str = "auto",
    project_dir: Optional[str] = None,
    gate_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile `token` into an OS sandbox policy and wrap `raw_cmd` via `build_os_sandbox_command`."""
    policy = compile_sandbox_policy_from_token(token)
    if not policy["allow_shell"]:
        return {
            "allowed": False,
            "reason": "shell_not_allowed",
            "command": None,
            "policy": policy,
        }
    wrapped = build_os_sandbox_command(
        raw_cmd,
        sandbox_mode=sandbox_mode,
        project_dir=project_dir,
        deny_net=bool(policy["deny_net"]),
        strip_secrets=bool(policy["strip_secrets"]),
        gate_bin=gate_bin,
    )
    return {
        "allowed": True,
        "reason": "sandbox_compiled" if wrapped is not None else "sandbox_unavailable",
        "command": wrapped,
        "policy": policy,
    }


def create_capability_domain_proxy(
    token: CapabilityToken,
    *,
    connect_timeout: float = 5.0,
) -> Optional[DomainAllowlistProxy]:
    """Instantiate a `DomainAllowlistProxy` scoped to the token's intersected `allowed_domains`.

    Returns ``None`` when the token imposes no domain allowlist caveat.
    """
    policy = compile_sandbox_policy_from_token(token)
    domains = policy["allowed_domains"]
    if domains is None:
        return None
    return DomainAllowlistProxy(domains, connect_timeout=connect_timeout)
