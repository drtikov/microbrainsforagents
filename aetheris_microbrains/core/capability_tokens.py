"""Macaroon-style HMAC-SHA256 capability tokens and subagent taint propagation.

Pure Python standard-library implementation providing:
- Cryptographically chained capability caveats (`CapabilityCaveat`, `CapabilityToken`)
- Root minting with HMAC-SHA256 (`mint_root_capability`)
- Non-interactive monotonic attenuation without the root secret (`attenuate_capability`)
- Constant-time chain verification & monotonic intersection authorization
  (`verify_capability_and_authorize`)
- Cross-agent session taint propagation (`propagate_subagent_taint`) for
  `SessionSecurityState`
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import math
import posixpath
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


def _coerce_str_tuple(value: Optional[Iterable[str]], field_name: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of strings or None, not {type(value).__name__}")
    return tuple(str(item) for item in value)


def _coerce_geofence_polytope(
    value: Optional[Iterable[Any]],
) -> Optional[Tuple[Tuple[Tuple[float, ...], float], ...]]:
    if value is None:
        return None
    halfspaces: List[Tuple[Tuple[float, ...], float]] = []
    for item in value:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("each geofence_polytope halfspace must be a (normal, offset) pair")
        normal_raw, offset_raw = item
        if isinstance(normal_raw, (str, bytes)) or not isinstance(normal_raw, Iterable):
            raise TypeError("geofence_polytope halfspace normal must be an iterable of numbers")
        normal = tuple(float(coord) for coord in normal_raw)
        if not normal:
            raise ValueError("geofence_polytope halfspace normal must not be empty")
        halfspaces.append((normal, float(offset_raw)))
    return tuple(halfspaces)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _secret_to_bytes(root_secret: Union[bytes, str]) -> bytes:
    if isinstance(root_secret, (bytes, bytearray)):
        raw = bytes(root_secret)
    elif isinstance(root_secret, str):
        raw = root_secret.encode("utf-8")
    else:
        raise TypeError(f"root_secret must be bytes or str, got {type(root_secret).__name__}")
    if not raw:
        raise ValueError("root_secret must not be empty")
    return raw


def _header_bytes(token_id: str, subject: str, issued_at: float) -> bytes:
    return _canonical_json_bytes(
        {
            "token_id": str(token_id),
            "subject": str(subject),
            "issued_at": float(issued_at),
        }
    )


def _chain_hmac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _normalize_path(path: str) -> str:
    raw = str(path).strip().replace("\\", "/")
    if not raw:
        return ""
    norm = posixpath.normpath(raw)
    if norm.startswith("./"):
        norm = norm[2:]
    return norm


def _match_glob(norm_path: str, pattern: str) -> bool:
    norm_pat = _normalize_path(pattern)
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


def _match_pattern(value: str, pattern: str) -> bool:
    pat = str(pattern).strip()
    if not pat:
        return False
    return fnmatch.fnmatchcase(value, pat)


def _match_tool(clean_tool: str, pattern: str) -> bool:
    pat = str(pattern).strip().lower()
    if not pat:
        return False
    return fnmatch.fnmatchcase(clean_tool, pat)


def _normalize_domain(domain: str) -> str:
    return str(domain).strip().lower().rstrip(".")


def _match_domain(clean_domain: str, pattern: str) -> bool:
    pat = _normalize_domain(pattern)
    if not pat or not clean_domain:
        return False
    if pat.startswith("*."):
        suffix = pat[1:]
        return clean_domain.endswith(suffix) and len(clean_domain) > len(suffix)
    return fnmatch.fnmatchcase(clean_domain, pat)


@dataclass(frozen=True)
class CapabilityCaveat:
    """Single immutable attenuation caveat in a capability token chain."""

    allowed_tools: Optional[Tuple[str, ...]] = None
    read_globs: Optional[Tuple[str, ...]] = None
    write_globs: Optional[Tuple[str, ...]] = None
    allow_shell: Optional[bool] = None
    allowed_secrets: Optional[Tuple[str, ...]] = None
    allowed_domains: Optional[Tuple[str, ...]] = None
    expires_at: Optional[float] = None
    subagent_id: Optional[str] = None
    depth_remaining: Optional[int] = None
    max_linear_vel_mps: Optional[float] = None
    max_kinetic_energy_j: Optional[float] = None
    max_contact_force_n: Optional[float] = None
    allowed_actuators: Optional[Tuple[str, ...]] = None
    geofence_polytope: Optional[Tuple[Tuple[Tuple[float, ...], float], ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_tools", _coerce_str_tuple(self.allowed_tools, "allowed_tools"))
        object.__setattr__(self, "read_globs", _coerce_str_tuple(self.read_globs, "read_globs"))
        object.__setattr__(self, "write_globs", _coerce_str_tuple(self.write_globs, "write_globs"))
        object.__setattr__(self, "allowed_secrets", _coerce_str_tuple(self.allowed_secrets, "allowed_secrets"))
        object.__setattr__(self, "allowed_domains", _coerce_str_tuple(self.allowed_domains, "allowed_domains"))
        if self.allow_shell is not None:
            object.__setattr__(self, "allow_shell", bool(self.allow_shell))
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", float(self.expires_at))
        if self.subagent_id is not None:
            object.__setattr__(self, "subagent_id", str(self.subagent_id))
        if self.depth_remaining is not None:
            object.__setattr__(self, "depth_remaining", int(self.depth_remaining))
        if self.max_linear_vel_mps is not None:
            object.__setattr__(self, "max_linear_vel_mps", float(self.max_linear_vel_mps))
        if self.max_kinetic_energy_j is not None:
            object.__setattr__(self, "max_kinetic_energy_j", float(self.max_kinetic_energy_j))
        if self.max_contact_force_n is not None:
            object.__setattr__(self, "max_contact_force_n", float(self.max_contact_force_n))
        object.__setattr__(self, "allowed_actuators", _coerce_str_tuple(self.allowed_actuators, "allowed_actuators"))
        object.__setattr__(self, "geofence_polytope", _coerce_geofence_polytope(self.geofence_polytope))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize caveat into a deterministic JSON-compatible mapping."""
        data: Dict[str, Any] = {
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
            "read_globs": list(self.read_globs) if self.read_globs is not None else None,
            "write_globs": list(self.write_globs) if self.write_globs is not None else None,
            "allow_shell": self.allow_shell,
            "allowed_secrets": list(self.allowed_secrets) if self.allowed_secrets is not None else None,
            "allowed_domains": list(self.allowed_domains) if self.allowed_domains is not None else None,
            "expires_at": self.expires_at,
            "subagent_id": self.subagent_id,
            "depth_remaining": self.depth_remaining,
        }
        # Omit unset physical safety keys so non-physical caveats preserve byte-identical hashes.
        if self.max_linear_vel_mps is not None:
            data["max_linear_vel_mps"] = self.max_linear_vel_mps
        if self.max_kinetic_energy_j is not None:
            data["max_kinetic_energy_j"] = self.max_kinetic_energy_j
        if self.max_contact_force_n is not None:
            data["max_contact_force_n"] = self.max_contact_force_n
        if self.allowed_actuators is not None:
            data["allowed_actuators"] = list(self.allowed_actuators)
        if self.geofence_polytope is not None:
            data["geofence_polytope"] = [[list(normal), offset] for normal, offset in self.geofence_polytope]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityCaveat":
        """Deserialize a caveat from a mapping produced by ``to_dict``."""
        return cls(
            allowed_tools=tuple(data["allowed_tools"]) if data.get("allowed_tools") is not None else None,
            read_globs=tuple(data["read_globs"]) if data.get("read_globs") is not None else None,
            write_globs=tuple(data["write_globs"]) if data.get("write_globs") is not None else None,
            allow_shell=data.get("allow_shell"),
            allowed_secrets=tuple(data["allowed_secrets"]) if data.get("allowed_secrets") is not None else None,
            allowed_domains=tuple(data["allowed_domains"]) if data.get("allowed_domains") is not None else None,
            expires_at=data.get("expires_at"),
            subagent_id=data.get("subagent_id"),
            depth_remaining=data.get("depth_remaining"),
            max_linear_vel_mps=data.get("max_linear_vel_mps"),
            max_kinetic_energy_j=data.get("max_kinetic_energy_j"),
            max_contact_force_n=data.get("max_contact_force_n"),
            allowed_actuators=tuple(data["allowed_actuators"]) if data.get("allowed_actuators") is not None else None,
            geofence_polytope=data.get("geofence_polytope"),
        )


@dataclass
class CapabilityToken:
    """Macaroon-style HMAC-SHA256 chained capability token."""

    token_id: str
    subject: str
    issued_at: float
    caveats: Tuple[CapabilityCaveat, ...]
    signature: str

    def __post_init__(self) -> None:
        self.token_id = str(self.token_id)
        self.subject = str(self.subject)
        self.issued_at = float(self.issued_at)
        self.caveats = tuple(
            cav if isinstance(cav, CapabilityCaveat) else CapabilityCaveat.from_dict(cav)
            for cav in self.caveats
        )
        self.signature = str(self.signature).lower()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize token and its caveat chain to a JSON-compatible dictionary."""
        return {
            "token_id": self.token_id,
            "subject": self.subject,
            "issued_at": self.issued_at,
            "caveats": [cav.to_dict() for cav in self.caveats],
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityToken":
        """Deserialize a token from a mapping produced by ``to_dict``."""
        raw_caveats = data.get("caveats") or ()
        return cls(
            token_id=str(data["token_id"]),
            subject=str(data["subject"]),
            issued_at=float(data["issued_at"]),
            caveats=tuple(CapabilityCaveat.from_dict(c) for c in raw_caveats),
            signature=str(data["signature"]),
        )


def mint_root_capability(
    root_secret: Union[bytes, str],
    *,
    subject: str,
    caveat: Optional[CapabilityCaveat] = None,
    now: Optional[float] = None,
) -> CapabilityToken:
    """Mint a root capability token signed with ``root_secret``."""
    clean_subject = str(subject or "").strip()
    if not clean_subject:
        raise ValueError("subject must not be empty")
    secret_bytes = _secret_to_bytes(root_secret)
    issued_at = float(time.time() if now is None else now)
    token_id = secrets.token_hex(16)

    sig = _chain_hmac(secret_bytes, _header_bytes(token_id, clean_subject, issued_at))
    caveats: Tuple[CapabilityCaveat, ...] = ()
    if caveat is not None:
        if not isinstance(caveat, CapabilityCaveat):
            raise TypeError("caveat must be a CapabilityCaveat or None")
        sig = _chain_hmac(sig, _canonical_json_bytes(caveat.to_dict()))
        caveats = (caveat,)

    return CapabilityToken(
        token_id=token_id,
        subject=clean_subject,
        issued_at=issued_at,
        caveats=caveats,
        signature=sig.hex(),
    )


def attenuate_capability(token: CapabilityToken, caveat: CapabilityCaveat) -> CapabilityToken:
    """Non-interactively attenuate ``token`` with an additional ``caveat``.

    Uses ``new_sig = HMAC-SHA256(bytes.fromhex(token.signature), canonical_caveat_json)``
    without needing ``root_secret``. Verification evaluates all caveats in sequence
    as a monotonic intersection, so appending caveats can only narrow authority
    and stripping/modifying any caveat invalidates the HMAC chain.
    """
    if not isinstance(token, CapabilityToken):
        raise TypeError("token must be a CapabilityToken")
    if not isinstance(caveat, CapabilityCaveat):
        raise TypeError("caveat must be a CapabilityCaveat")
    try:
        prev_sig = bytes.fromhex(token.signature)
    except ValueError as exc:
        raise ValueError("token.signature must be valid hex") from exc
    if len(prev_sig) != 32:
        raise ValueError("token.signature must decode to 32 bytes (SHA-256 HMAC)")

    next_sig = _chain_hmac(prev_sig, _canonical_json_bytes(caveat.to_dict())).hex()
    return CapabilityToken(
        token_id=token.token_id,
        subject=token.subject,
        issued_at=token.issued_at,
        caveats=token.caveats + (caveat,),
        signature=next_sig,
    )


def _auth_result(
    authorized: bool,
    reason: str,
    subject: str,
    chain_depth: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "authorized": authorized,
        "decision": "allow" if authorized else "deny",
        "reason": reason,
        "subject": subject,
        "chain_depth": chain_depth,
    }
    if extra is not None:
        result.update(extra)
    return result


def _verify_hmac_chain(
    token: CapabilityToken,
    root_secret: Union[bytes, str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(token, CapabilityToken):
        return _auth_result(False, "invalid_token_type", "", 0)

    subject = token.subject
    chain_depth = len(token.caveats)

    try:
        secret_bytes = _secret_to_bytes(root_secret)
    except (TypeError, ValueError):
        return _auth_result(False, "invalid_root_secret", subject, chain_depth)

    try:
        provided_sig = bytes.fromhex(token.signature)
    except ValueError:
        return _auth_result(False, "invalid_signature_format", subject, chain_depth)

    if len(provided_sig) != 32:
        return _auth_result(False, "invalid_signature_length", subject, chain_depth)

    expected_sig = _chain_hmac(secret_bytes, _header_bytes(token.token_id, token.subject, token.issued_at))
    for cav in token.caveats:
        expected_sig = _chain_hmac(expected_sig, _canonical_json_bytes(cav.to_dict()))

    if not hmac.compare_digest(expected_sig, provided_sig):
        return _auth_result(False, "invalid_hmac_signature", subject, chain_depth)
    return None


def _advance_chain_metadata(
    cav: CapabilityCaveat,
    idx: int,
    current_time: float,
    effective_depth: Optional[int],
    pinned_subagent: Optional[str],
    effective_expires_at: Optional[float],
    subject: str,
    chain_depth: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str], Optional[float]]:
    if idx > 0 and effective_depth is not None:
        effective_depth -= 1
        if effective_depth < 0:
            return _auth_result(False, "delegation_depth_exceeded", subject, chain_depth), effective_depth, pinned_subagent, effective_expires_at

    if cav.depth_remaining is not None:
        if cav.depth_remaining < 0:
            return _auth_result(False, "delegation_depth_exceeded", subject, chain_depth), effective_depth, pinned_subagent, effective_expires_at
        effective_depth = (
            cav.depth_remaining
            if effective_depth is None
            else min(effective_depth, cav.depth_remaining)
        )

    if cav.subagent_id is not None:
        if not cav.subagent_id.strip():
            return _auth_result(False, "invalid_subagent_id", subject, chain_depth), effective_depth, pinned_subagent, effective_expires_at
        if pinned_subagent is None:
            pinned_subagent = cav.subagent_id
        elif pinned_subagent != cav.subagent_id:
            return _auth_result(False, "subagent_id_conflict", subject, chain_depth), effective_depth, pinned_subagent, effective_expires_at

    if cav.expires_at is not None:
        effective_expires_at = (
            cav.expires_at
            if effective_expires_at is None
            else min(effective_expires_at, cav.expires_at)
        )
        if current_time > effective_expires_at:
            return _auth_result(False, "token_expired", subject, chain_depth), effective_depth, pinned_subagent, effective_expires_at

    return None, effective_depth, pinned_subagent, effective_expires_at


def verify_capability_and_authorize(
    token: CapabilityToken,
    root_secret: Union[bytes, str],
    *,
    tool_name: str,
    path: Optional[str] = None,
    is_write: bool = False,
    is_shell: bool = False,
    secret_handle: Optional[str] = None,
    domain: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify HMAC chain integrity and authorize a requested action against all caveats."""
    hmac_err = _verify_hmac_chain(token, root_secret)
    if hmac_err is not None:
        return hmac_err

    subject = token.subject
    chain_depth = len(token.caveats)

    clean_tool = str(tool_name or "").strip().lower()
    if not clean_tool:
        return _auth_result(False, "missing_tool_name", subject, chain_depth)

    current_time = float(time.time() if now is None else now)
    norm_path = _normalize_path(path) if path is not None else None
    clean_secret = str(secret_handle).strip() if secret_handle is not None else None
    clean_domain = _normalize_domain(domain) if domain is not None else None

    effective_depth: Optional[int] = None
    pinned_subagent: Optional[str] = None
    effective_expires_at: Optional[float] = None

    for idx, cav in enumerate(token.caveats):
        meta_err, effective_depth, pinned_subagent, effective_expires_at = _advance_chain_metadata(
            cav,
            idx,
            current_time,
            effective_depth,
            pinned_subagent,
            effective_expires_at,
            subject,
            chain_depth,
        )
        if meta_err is not None:
            return meta_err

        if cav.allowed_tools is not None:
            if not any(_match_tool(clean_tool, pat) for pat in cav.allowed_tools):
                return _auth_result(False, f"tool_not_allowed:{clean_tool}", subject, chain_depth)

        if is_shell and cav.allow_shell is False:
            return _auth_result(False, "shell_not_allowed", subject, chain_depth)

        if is_write and cav.write_globs is not None:
            if not norm_path:
                return _auth_result(False, "write_path_required", subject, chain_depth)
            if not any(_match_glob(norm_path, pat) for pat in cav.write_globs):
                return _auth_result(False, f"write_path_not_allowed:{norm_path}", subject, chain_depth)

        if not is_write and norm_path is not None and cav.read_globs is not None:
            if not norm_path or not any(_match_glob(norm_path, pat) for pat in cav.read_globs):
                return _auth_result(False, f"read_path_not_allowed:{norm_path}", subject, chain_depth)

        if clean_secret is not None and cav.allowed_secrets is not None:
            if not any(_match_pattern(clean_secret, pat) for pat in cav.allowed_secrets):
                return _auth_result(False, f"secret_not_allowed:{clean_secret}", subject, chain_depth)

        if clean_domain is not None and cav.allowed_domains is not None:
            if not any(_match_domain(clean_domain, pat) for pat in cav.allowed_domains):
                return _auth_result(False, f"domain_not_allowed:{clean_domain}", subject, chain_depth)

    return _auth_result(True, "authorized", subject, chain_depth)


def _validate_non_negative_finite(value: Any, name: str) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    num = float(value)
    if not math.isfinite(num) or num < 0.0:
        return None
    return num


def authorize_physical_action(
    token: CapabilityToken,
    root_secret: Union[bytes, str],
    position: Optional[Iterable[float]] = None,
    linear_vel_mps: Optional[float] = None,
    kinetic_energy_j: Optional[float] = None,
    contact_force_n: Optional[float] = None,
    actuator: Optional[str] = None,
    now: Optional[float] = None,
    waypoints: Optional[Iterable[Iterable[float]]] = None,
) -> Dict[str, Any]:
    """Verify HMAC chain and enforce monotonic intersection of physical safety caveats."""
    hmac_err = _verify_hmac_chain(token, root_secret)
    if hmac_err is not None:
        return hmac_err

    subject = token.subject
    chain_depth = len(token.caveats)
    current_time = float(time.time() if now is None else now)
    if not math.isfinite(current_time):
        return _auth_result(False, "invalid_timestamp", subject, chain_depth)

    req_vel: Optional[float] = None
    if linear_vel_mps is not None:
        req_vel = _validate_non_negative_finite(linear_vel_mps, "linear_vel_mps")
        if req_vel is None:
            return _auth_result(False, "invalid_linear_vel", subject, chain_depth)

    req_ke: Optional[float] = None
    if kinetic_energy_j is not None:
        req_ke = _validate_non_negative_finite(kinetic_energy_j, "kinetic_energy_j")
        if req_ke is None:
            return _auth_result(False, "invalid_kinetic_energy", subject, chain_depth)

    req_force: Optional[float] = None
    if contact_force_n is not None:
        req_force = _validate_non_negative_finite(contact_force_n, "contact_force_n")
        if req_force is None:
            return _auth_result(False, "invalid_contact_force", subject, chain_depth)

    req_pos: Optional[Tuple[float, ...]] = None
    if position is not None:
        if isinstance(position, (str, bytes)) or not isinstance(position, Iterable):
            return _auth_result(False, "invalid_position", subject, chain_depth)
        try:
            req_pos = tuple(float(coord) for coord in position)
        except (TypeError, ValueError):
            return _auth_result(False, "invalid_position", subject, chain_depth)
        if not req_pos or any(not math.isfinite(c) for c in req_pos):
            return _auth_result(False, "invalid_position", subject, chain_depth)

    clean_actuator: Optional[str] = None
    if actuator is not None:
        clean_actuator = str(actuator).strip()
        if not clean_actuator:
            return _auth_result(False, "invalid_actuator", subject, chain_depth)

    effective_depth: Optional[int] = None
    pinned_subagent: Optional[str] = None
    effective_expires_at: Optional[float] = None

    effective_max_vel: Optional[float] = None
    effective_max_ke: Optional[float] = None
    effective_max_force: Optional[float] = None
    effective_actuators: Optional[set[str]] = None
    halfspace_count = 0

    for idx, cav in enumerate(token.caveats):
        meta_err, effective_depth, pinned_subagent, effective_expires_at = _advance_chain_metadata(
            cav,
            idx,
            current_time,
            effective_depth,
            pinned_subagent,
            effective_expires_at,
            subject,
            chain_depth,
        )
        if meta_err is not None:
            return meta_err

        if cav.max_linear_vel_mps is not None:
            bound = _validate_non_negative_finite(cav.max_linear_vel_mps, "max_linear_vel_mps")
            if bound is None:
                return _auth_result(False, "invalid_caveat_linear_vel", subject, chain_depth)
            effective_max_vel = bound if effective_max_vel is None else min(effective_max_vel, bound)

        if cav.max_kinetic_energy_j is not None:
            bound = _validate_non_negative_finite(cav.max_kinetic_energy_j, "max_kinetic_energy_j")
            if bound is None:
                return _auth_result(False, "invalid_caveat_kinetic_energy", subject, chain_depth)
            effective_max_ke = bound if effective_max_ke is None else min(effective_max_ke, bound)

        if cav.max_contact_force_n is not None:
            bound = _validate_non_negative_finite(cav.max_contact_force_n, "max_contact_force_n")
            if bound is None:
                return _auth_result(False, "invalid_caveat_contact_force", subject, chain_depth)
            effective_max_force = bound if effective_max_force is None else min(effective_max_force, bound)

        if cav.allowed_actuators is not None:
            cav_acts = {str(a).strip() for a in cav.allowed_actuators if str(a).strip()}
            effective_actuators = cav_acts if effective_actuators is None else (effective_actuators & cav_acts)

        if cav.geofence_polytope is not None:
            halfspace_count += len(cav.geofence_polytope)

    effective_limits = {
        "max_linear_vel_mps": effective_max_vel,
        "max_kinetic_energy_j": effective_max_ke,
        "max_contact_force_n": effective_max_force,
        "allowed_actuators": sorted(effective_actuators) if effective_actuators is not None else None,
        "geofence_halfspaces": halfspace_count,
    }
    extra = {"effective_limits": effective_limits}

    if effective_max_vel is not None and req_vel is not None and req_vel > effective_max_vel:
        return _auth_result(False, "linear_vel_exceeded", subject, chain_depth, extra)

    if effective_max_ke is not None and req_ke is not None and req_ke > effective_max_ke:
        return _auth_result(False, "kinetic_energy_exceeded", subject, chain_depth, extra)

    if effective_max_force is not None and req_force is not None and req_force > effective_max_force:
        return _auth_result(False, "contact_force_exceeded", subject, chain_depth, extra)

    if effective_actuators is not None and clean_actuator is not None:
        if not any(_match_pattern(clean_actuator, pat) for pat in effective_actuators):
            return _auth_result(False, f"actuator_not_allowed:{clean_actuator}", subject, chain_depth, extra)

    fence_points: List[Tuple[float, ...]] = []
    if req_pos is not None:
        fence_points.append(req_pos)
    if waypoints is not None:
        if isinstance(waypoints, (str, bytes)) or not isinstance(waypoints, Iterable):
            return _auth_result(False, "invalid_position", subject, chain_depth)
        for wp in waypoints:
            if isinstance(wp, (str, bytes)) or not isinstance(wp, Iterable):
                return _auth_result(False, "invalid_position", subject, chain_depth)
            try:
                wp_tup = tuple(float(coord) for coord in wp)
            except (TypeError, ValueError):
                return _auth_result(False, "invalid_position", subject, chain_depth)
            if not wp_tup or any(not math.isfinite(c) for c in wp_tup):
                return _auth_result(False, "invalid_position", subject, chain_depth)
            fence_points.append(wp_tup)

    if fence_points:
        for cav in token.caveats:
            if cav.geofence_polytope is None:
                continue
            for pt in fence_points:
                for normal, offset in cav.geofence_polytope:
                    if not math.isfinite(offset) or any(not math.isfinite(a) for a in normal):
                        return _auth_result(False, "invalid_geofence_polytope", subject, chain_depth, extra)
                    if len(normal) != len(pt):
                        return _auth_result(False, "geofence_dimension_mismatch", subject, chain_depth, extra)
                    dot = sum(a * p for a, p in zip(normal, pt))
                    if dot > offset + 1e-12:
                        return _auth_result(False, "geofence_violation", subject, chain_depth, extra)

    return _auth_result(True, "authorized", subject, chain_depth, extra)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def propagate_subagent_taint(
    parent_state: Any,
    child_state: Any,
    *,
    subagent_label: str = "subagent",
) -> Dict[str, Any]:
    """Merge taint flags, events, and artifacts from ``child_state`` into ``parent_state``.

    Preserves ``integrity`` (e.g. ``"untrusted"``) and ``confidentiality``
    (e.g. ``"secret"``) on propagated artifacts, updates ``parent_state``'s
    lethal-trifecta flags, calls ``parent_state.save()`` if available, and
    returns merged counts and resulting taint status.
    """
    if parent_state is None or child_state is None:
        raise ValueError("parent_state and child_state must not be None")

    label = str(subagent_label or "subagent").strip()[:40] or "subagent"
    event_prefix = f"subagent:{label}:"

    child_untrusted = bool(_get_field(child_state, "untrusted_external_ingested", False))
    child_sensitive = bool(_get_field(child_state, "sensitive_local_read", False))
    child_injection = bool(_get_field(child_state, "prompt_injection_seen", False))
    child_corrupted = bool(_get_field(child_state, "corrupted", False))

    child_events = list(_get_field(child_state, "taint_events", []) or [])
    child_artifacts = list(_get_field(child_state, "taint_artifacts", []) or [])

    for art in child_artifacts:
        if not isinstance(art, Mapping):
            continue
        if art.get("integrity") == "untrusted":
            child_untrusted = True
        if art.get("confidentiality") == "secret":
            child_sensitive = True

    parent_state.untrusted_external_ingested = bool(
        getattr(parent_state, "untrusted_external_ingested", False) or child_untrusted
    )
    parent_state.sensitive_local_read = bool(
        getattr(parent_state, "sensitive_local_read", False) or child_sensitive
    )
    parent_state.prompt_injection_seen = bool(
        getattr(parent_state, "prompt_injection_seen", False) or child_injection
    )
    if child_corrupted:
        parent_state.corrupted = True

    existing_events: List[str] = [str(ev) for ev in (getattr(parent_state, "taint_events", None) or []) if ev]
    seen_events = set(existing_events)
    merged_events_count = 0
    for raw_ev in child_events:
        ev_str = str(raw_ev or "").strip()
        if not ev_str:
            continue
        tagged = ev_str if ev_str.startswith(event_prefix) else f"{event_prefix}{ev_str}"
        if tagged not in seen_events:
            seen_events.add(tagged)
            existing_events.append(tagged)
            merged_events_count += 1
    parent_state.taint_events = existing_events[-20:]

    existing_artifacts: List[Dict[str, Any]] = [
        dict(a) for a in (getattr(parent_state, "taint_artifacts", None) or []) if isinstance(a, Mapping)
    ]
    seen_artifacts = {
        (a.get("source"), a.get("tool"), a.get("integrity"), a.get("confidentiality"))
        for a in existing_artifacts
    }
    merged_artifacts_count = 0
    for raw_art in child_artifacts:
        if not isinstance(raw_art, Mapping):
            continue
        art_copy = dict(raw_art)
        art_copy.setdefault("subagent", label)
        key = (
            art_copy.get("source"),
            art_copy.get("tool"),
            art_copy.get("integrity"),
            art_copy.get("confidentiality"),
        )
        if key not in seen_artifacts:
            seen_artifacts.add(key)
            existing_artifacts.append(art_copy)
            merged_artifacts_count += 1
    parent_state.taint_artifacts = existing_artifacts[-50:]

    save_fn = getattr(parent_state, "save", None)
    if callable(save_fn):
        save_fn()

    return {
        "subagent_label": label,
        "merged_events": merged_events_count,
        "merged_artifacts": merged_artifacts_count,
        "total_events": len(parent_state.taint_events),
        "total_artifacts": len(parent_state.taint_artifacts),
        "untrusted_external_ingested": bool(parent_state.untrusted_external_ingested),
        "sensitive_local_read": bool(parent_state.sensitive_local_read),
        "prompt_injection_seen": bool(parent_state.prompt_injection_seen),
        "corrupted": bool(getattr(parent_state, "corrupted", False)),
    }
