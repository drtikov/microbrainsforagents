"""Canonical credential/token detection and redaction patterns (stdlib-only).

Shared between `hook.py` and `layered_decision_engine.py` so error redaction,
PostToolUse scrubbing, and diff secret scanning never diverge.
"""

from __future__ import annotations

import base64
import re
from typing import Any, Dict, List, Tuple

_SECRET_PATTERNS: Tuple[Tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_key_pem",
        re.compile(
            r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"
            r"[\s\S]*?"
            r"-----END\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        "private_key_pem",
        re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
    ("openai_project_key", re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{12,}\b"), "[REDACTED_OPENAI_KEY]"),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_ANTHROPIC_KEY]"),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_API_KEY]"),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_GITLAB_TOKEN]"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOGLE_API_KEY]"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[REDACTED_HF_TOKEN]"),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b"), "[REDACTED_NPM_TOKEN]"),
    ("pypi_token", re.compile(r"\bpypi-[A-Za-z0-9_\-]{24,}\b"), "[REDACTED_PYPI_TOKEN]"),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "[REDACTED_STRIPE_KEY]"),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{30,}\b"), "[REDACTED_TELEGRAM_TOKEN]"),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_SENDGRID_KEY]"),
    ("databricks_token", re.compile(r"\bdapi[a-f0-9]{32}\b"), "[REDACTED_DATABRICKS_TOKEN]"),
    (
        "db_connection_uri",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://"
            r"[^/\s:@\"']+:(?!pass(?:word)?@|changeme@|example@|your_)[^/\s@\"']{4,}@[^\s\"']+"
        ),
        "[REDACTED_DB_URI]",
    ),
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED_JWT]"),
    ("bearer_token", re.compile(r"Bearer\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/]{16,}=*"), "Bearer [REDACTED]"),
)

_SECRET_TOKEN_REGEXES: Tuple[re.Pattern[str], ...] = tuple(pat for _stype, pat, _repl in _SECRET_PATTERNS)
_REDACT_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = tuple(
    (pat, repl) for _stype, pat, repl in _SECRET_PATTERNS
) + (
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"Bearer\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
)


_BASE64_BLOB_RE = re.compile(r"(?<![A-Za-z0-9+/_=-])[A-Za-z0-9+/_-]{32,}={0,2}(?![A-Za-z0-9+/_=-])")
_HEX_BLOB_RE = re.compile(r"\b[0-9a-fA-F]{64,}\b")
_ENCODED_SCAN_MAX_BLOBS = 200
_ENCODED_BLOB_MAX_CHARS = 8192

# Credential-named keys carrying opaque values with no recognizable token
# format (`api_key: mQ3xV7pL2wR8...`). Values containing placeholder markers,
# POSIX `PWD=/path` directory assignments, or env references are untouched.
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"password|passwd|pwd|private[_-]?key|credential[s]?|session[_-]?token|refresh[_-]?token)\b"
    r"(\s*[:=]\s*)([\"']?)([A-Za-z0-9_\-./+=]{12,})\3"
)
_SENSITIVE_VALUE_SKIP_MARKERS: Tuple[str, ...] = (
    "example", "placeholder", "changeme", "your_", "dummy", "test", "xxx",
    "<your", "$", "{{", "%(", "os.environ", "getenv", "redacted",
)
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _redact_sensitive_key_values(text: str) -> Tuple[str, int]:
    """Redact opaque values assigned to credential-named keys in tool outputs.

    Catches credential dumps that use no recognizable token format. Runs after
    the token-pattern pass, so values already redacted (bracket placeholders)
    are not re-matched, and quoted/unquoted values both qualify. Skips POSIX
    `PWD=/dir/...` directory variables.
    """
    out = str(text or "")
    count = 0

    def _sub(match: "re.Match[str]") -> str:
        nonlocal count
        key = match.group(1).lower()
        value = match.group(4)
        low = value.lower()
        if any(marker in low for marker in _SENSITIVE_VALUE_SKIP_MARKERS):
            return match.group(0)
        if key == "pwd" and (
            value.startswith(("/", "~/", "./", "../"))
            or _WINDOWS_DRIVE_PATH_RE.match(value)
        ):
            return match.group(0)
        count += 1
        quote = match.group(3)
        return match.group(1) + match.group(2) + quote + "[REDACTED_SENSITIVE_VALUE]" + quote

    out = _SENSITIVE_KEY_RE.sub(_sub, out)
    return out, count


def detect_sensitive_key_values(text: str) -> int:
    """Count credential-named key/value pairs whose values look like secrets."""
    _out, n = _redact_sensitive_key_values(str(text or ""))
    return n


def _decode_base64_candidate(blob: str) -> bytes:
    """Decode standard or URL-safe base64 candidates (with optional stripped `=` padding)."""
    if "-" in blob or "_" in blob:
        padded = blob + ("=" * ((4 - len(blob) % 4) % 4))
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    return base64.b64decode(blob, validate=True)


def _redact_encoded_secret_blobs(text: str) -> Tuple[str, List[str], int]:
    """Decode base64/urlsafe-base64/hex blobs and redact any matching credential patterns.

    Bounded to _ENCODED_SCAN_MAX_BLOBS blobs per call with per-blob size caps so
    pathological outputs cannot turn scrubbing into a decode bomb.
    """
    src = str(text or "")
    hits: List[str] = []
    redacted_count = 0
    out = src
    for blob_re, codec in ((_BASE64_BLOB_RE, "b64"), (_HEX_BLOB_RE, "hex")):
        seen = 0
        for match in blob_re.finditer(src):
            seen += 1
            if seen > _ENCODED_SCAN_MAX_BLOBS:
                break
            blob = match.group(0)
            if len(blob) > _ENCODED_BLOB_MAX_CHARS:
                continue
            try:
                if codec == "b64":
                    decoded = _decode_base64_candidate(blob).decode("utf-8", errors="ignore")
                else:
                    decoded = bytes.fromhex(blob).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if not decoded:
                continue
            matched = next((t for t, pat, _r in _SECRET_PATTERNS if pat.search(decoded)), None)
            if matched is not None:
                if matched not in hits:
                    hits.append(matched)
                out = out.replace(blob, "[REDACTED_ENCODED_" + matched.upper() + "]")
                redacted_count += 1
    return out, hits, redacted_count


def detect_encoded_secrets(text: str) -> List[str]:
    """Return secret types found inside base64/hex-encoded blobs in `text`."""
    _out, hits, _n = _redact_encoded_secret_blobs(text)
    return hits


def redact_secrets(text: str) -> str:
    """Mask known, encoded, and sensitive key/value credential patterns before logging or printing."""
    out = str(text or "")
    for pat, repl in _REDACT_PATTERNS:
        out = pat.sub(repl, out)
    out, _enc_types, _enc_hits = _redact_encoded_secret_blobs(out)
    out, _sens_n = _redact_sensitive_key_values(out)
    return out


def scrub_tool_output(text: Any) -> Dict[str, Any]:
    """Scrub raw credential/token patterns from tool outputs before LLM context ingestion."""
    out = "" if text is None else str(text)
    found_types: List[str] = []
    total_hits = 0
    for secret_type, pat, repl in _SECRET_PATTERNS:
        out, n_subs = pat.subn(repl, out)
        if n_subs > 0:
            total_hits += n_subs
            if secret_type not in found_types:
                found_types.append(secret_type)
    out, enc_types, enc_hits = _redact_encoded_secret_blobs(out)
    if enc_hits:
        total_hits += enc_hits
        for st in enc_types:
            if st not in found_types:
                found_types.append(st)
    out, sens_n = _redact_sensitive_key_values(out)
    if sens_n:
        total_hits += sens_n
        if "sensitive_key_value" not in found_types:
            found_types.append("sensitive_key_value")
    return {
        "redacted_text": out,
        "secrets_found": total_hits > 0,
        "secret_count": total_hits,
        "secret_types": sorted(found_types),
    }
