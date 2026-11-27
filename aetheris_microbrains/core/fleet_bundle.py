"""Signed Fleet Policy Bundle & Auditable Attestation (100% pure Python stdlib).

Provides HMAC-SHA256 signed, canonicalized policy + MCP pin bundles with
expiration and anti-rollback version enforcement, plus combined attestation over
the bundle and a hash-chained JSONL decision journal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .runtime_paths import verify_jsonl_hash_chain

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_ALGORITHM = "hmac-sha256"
ATTESTATION_SCHEMA_VERSION = 1


def _canonical_json(obj: Any) -> str:
    """Serialize `obj` deterministically for hashing and HMAC signing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_signing_key(signing_key: bytes | str) -> Optional[bytes]:
    """Normalize a non-empty signing key to UTF-8/raw bytes, or None if invalid."""
    if isinstance(signing_key, (bytes, bytearray)):
        raw = bytes(signing_key)
    elif isinstance(signing_key, str):
        raw = signing_key.encode("utf-8")
    else:
        return None
    return raw if raw else None


def _is_finite_number(val: Any) -> bool:
    """Return True for non-boolean int/float values that are finite."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def _is_hex64(val: Any) -> bool:
    """Return True if `val` is a 64-character lowercase/valid SHA-256 hex digest."""
    if not isinstance(val, str) or len(val) != 64:
        return False
    try:
        int(val, 16)
    except ValueError:
        return False
    return True


def _build_canonical_body(
    *,
    schema_version: int,
    org_id: str,
    version: int,
    issued_at: float,
    expires_at: Optional[float],
    policy: Mapping[str, Any],
    mcp_pins: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct the canonical JSON-normalized unsigned bundle body."""
    return {
        "schema_version": int(schema_version),
        "org_id": str(org_id),
        "version": int(version),
        "issued_at": float(issued_at),
        "expires_at": float(expires_at) if expires_at is not None else None,
        "policy": json.loads(_canonical_json(dict(policy))),
        "mcp_pins": json.loads(_canonical_json(dict(mcp_pins))),
    }


def create_signed_fleet_bundle(
    *,
    org_id: str,
    version: int,
    policy: Mapping[str, Any],
    mcp_pins: Optional[Mapping[str, Any]] = None,
    signing_key: bytes | str,
    ttl_sec: Optional[float] = 86400.0,
    issued_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a canonicalized, HMAC-SHA256 signed fleet policy and MCP pin bundle."""
    clean_org = str(org_id or "").strip()
    if not clean_org:
        raise ValueError("org_id must be a non-empty string")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("version must be a non-negative integer")
    if not isinstance(policy, Mapping):
        raise TypeError("policy must be a mapping")
    if mcp_pins is not None and not isinstance(mcp_pins, Mapping):
        raise TypeError("mcp_pins must be a mapping or None")

    key_bytes = _normalize_signing_key(signing_key)
    if key_bytes is None:
        raise ValueError("signing_key must be non-empty bytes or str")

    issued_ts = float(time.time() if issued_at is None else issued_at)
    if not math.isfinite(issued_ts):
        raise ValueError("issued_at must be finite")

    expires_at: Optional[float] = None
    if ttl_sec is not None:
        ttl_val = float(ttl_sec)
        if not math.isfinite(ttl_val) or ttl_val < 0.0:
            raise ValueError("ttl_sec must be a finite non-negative number or None")
        expires_at = issued_ts + ttl_val

    body = _build_canonical_body(
        schema_version=BUNDLE_SCHEMA_VERSION,
        org_id=clean_org,
        version=version,
        issued_at=issued_ts,
        expires_at=expires_at,
        policy=policy,
        mcp_pins=mcp_pins if mcp_pins is not None else {},
    )
    canonical_body = _canonical_json(body)
    bundle_sha256 = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
    signature = hmac.new(key_bytes, canonical_body.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        **body,
        "algorithm": BUNDLE_ALGORITHM,
        "bundle_sha256": bundle_sha256,
        "signature": signature,
    }


def verify_signed_fleet_bundle(
    bundle: Mapping[str, Any],
    *,
    signing_key: bytes | str,
    min_version: int = 0,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Fail-closed verification of bundle structure, HMAC-SHA256, expiry, and anti-rollback."""
    def _fail(
        reason: str,
        org_val: Optional[str] = None,
        ver_val: Optional[int] = None,
        sha_val: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "valid": False,
            "reason": reason,
            "org_id": org_val,
            "version": ver_val,
            "bundle_sha256": sha_val,
        }

    if not isinstance(bundle, Mapping):
        return _fail("invalid_bundle_type")

    raw_org = bundle.get("org_id")
    org_val = raw_org.strip() if isinstance(raw_org, str) and raw_org.strip() else None
    raw_ver = bundle.get("version")
    ver_val = raw_ver if (isinstance(raw_ver, int) and not isinstance(raw_ver, bool)) else None
    raw_sha = bundle.get("bundle_sha256")
    sha_val = raw_sha if isinstance(raw_sha, str) else None

    key_bytes = _normalize_signing_key(signing_key)
    if key_bytes is None:
        return _fail("invalid_signing_key", org_val, ver_val, sha_val)

    required_keys = (
        "schema_version",
        "org_id",
        "version",
        "issued_at",
        "expires_at",
        "policy",
        "mcp_pins",
        "algorithm",
        "bundle_sha256",
        "signature",
    )
    for k in required_keys:
        if k not in bundle:
            return _fail(f"missing_field:{k}", org_val, ver_val, sha_val)

    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        return _fail("unsupported_schema_version", org_val, ver_val, sha_val)
    if bundle.get("algorithm") != BUNDLE_ALGORITHM:
        return _fail("unsupported_algorithm", org_val, ver_val, sha_val)
    if org_val is None:
        return _fail("invalid_org_id", None, ver_val, sha_val)
    if ver_val is None or ver_val < 0:
        return _fail("invalid_version", org_val, None, sha_val)
    if not _is_finite_number(bundle.get("issued_at")):
        return _fail("invalid_issued_at", org_val, ver_val, sha_val)

    raw_exp = bundle.get("expires_at")
    if raw_exp is not None and not _is_finite_number(raw_exp):
        return _fail("invalid_expires_at", org_val, ver_val, sha_val)

    raw_policy = bundle.get("policy")
    raw_pins = bundle.get("mcp_pins")
    if not isinstance(raw_policy, Mapping) or not isinstance(raw_pins, Mapping):
        return _fail("invalid_payload_mapping", org_val, ver_val, sha_val)

    if not _is_hex64(raw_sha):
        return _fail("invalid_bundle_sha256", org_val, ver_val, sha_val)
    raw_sig = bundle.get("signature")
    if not _is_hex64(raw_sig):
        return _fail("invalid_signature_format", org_val, ver_val, sha_val)

    try:
        body = _build_canonical_body(
            schema_version=BUNDLE_SCHEMA_VERSION,
            org_id=org_val,
            version=ver_val,
            issued_at=float(bundle["issued_at"]),
            expires_at=float(raw_exp) if raw_exp is not None else None,
            policy=raw_policy,
            mcp_pins=raw_pins,
        )
        canonical_body = _canonical_json(body)
    except (TypeError, ValueError):
        return _fail("non_canonical_payload", org_val, ver_val, sha_val)

    expected_sha = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
    expected_sig = hmac.new(key_bytes, canonical_body.encode("utf-8"), hashlib.sha256).hexdigest()

    # Check HMAC signature before hash/expiry/version so unauthenticated modifications fail closed.
    if not hmac.compare_digest(str(raw_sig).lower(), expected_sig):
        return _fail("invalid_signature", org_val, ver_val, sha_val)
    if not hmac.compare_digest(str(raw_sha).lower(), expected_sha):
        return _fail("bundle_sha256_mismatch", org_val, ver_val, sha_val)

    now_ts = float(time.time() if now is None else now)
    if not math.isfinite(now_ts):
        return _fail("invalid_now_timestamp", org_val, ver_val, expected_sha)

    if raw_exp is not None and now_ts > float(raw_exp):
        return _fail("bundle_expired", org_val, ver_val, expected_sha)

    if ver_val < int(min_version):
        return _fail("version_rollback", org_val, ver_val, expected_sha)

    return {
        "valid": True,
        "reason": "ok",
        "org_id": org_val,
        "version": ver_val,
        "bundle_sha256": expected_sha,
    }


def build_fleet_attestation(
    bundle: Mapping[str, Any],
    *,
    journal_path: Optional[Path | str] = None,
    signing_key: bytes | str,
    min_version: int = 0,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify a signed fleet bundle and optional hash-chained journal into an attestation."""
    bundle_ver = verify_signed_fleet_bundle(
        bundle,
        signing_key=signing_key,
        min_version=min_version,
        now=now,
    )

    journal_ver: Optional[Dict[str, Any]] = None
    if journal_path is not None and str(journal_path).strip():
        journal_ver = verify_jsonl_hash_chain(Path(journal_path))

    journal_ok = bool(journal_ver.get("ok")) if journal_ver is not None else True
    overall_valid = bool(bundle_ver.get("valid")) and journal_ok

    if not bundle_ver.get("valid"):
        status_reason = f"bundle:{bundle_ver.get('reason')}"
    elif not journal_ok and journal_ver is not None:
        status_reason = f"journal:{journal_ver.get('reason', 'chain_invalid')}"
    else:
        status_reason = "ok"

    attested_ts = round(float(time.time() if now is None else now), 6)
    core_payload: Dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "valid": overall_valid,
        "reason": status_reason,
        "attested_at": attested_ts,
        "org_id": bundle_ver.get("org_id"),
        "version": bundle_ver.get("version"),
        "bundle_sha256": bundle_ver.get("bundle_sha256"),
        "bundle_verification": bundle_ver,
        "journal_verification": journal_ver,
        "journal_head_hash": journal_ver.get("head_hash") if journal_ver is not None else None,
        "otel_attributes": {
            "aetheris.org_id": bundle_ver.get("org_id") or "unknown",
            "aetheris.bundle_version": bundle_ver.get("version") if bundle_ver.get("version") is not None else -1,
            "aetheris.bundle_sha256": bundle_ver.get("bundle_sha256") or "none",
            "aetheris.journal_head_hash": (journal_ver.get("head_hash") if journal_ver is not None else "none") or "none",
            "aetheris.attestation_valid": overall_valid,
        },
    }
    attestation_sha256 = hashlib.sha256(_canonical_json(core_payload).encode("utf-8")).hexdigest()
    return {
        **core_payload,
        "attestation_sha256": attestation_sha256,
    }
