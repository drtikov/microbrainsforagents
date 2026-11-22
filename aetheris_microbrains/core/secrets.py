"""Canonical credential/token detection and redaction patterns (stdlib-only).

Shared between `hook.py` and `layered_decision_engine.py` so error redaction,
PostToolUse scrubbing, and diff secret scanning never diverge.
"""

from __future__ import annotations

import base64
import math
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
    ("google_oauth_token", re.compile(r"\bya29\.[A-Za-z0-9_\-]{24,}\b"), "[REDACTED_GOOGLE_OAUTH]"),
    (
        "slack_webhook",
        re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{20,}"),
        "[REDACTED_SLACK_WEBHOOK]",
    ),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[REDACTED_HF_TOKEN]"),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b"), "[REDACTED_NPM_TOKEN]"),
    ("pypi_token", re.compile(r"\bpypi-[A-Za-z0-9_\-]{24,}\b"), "[REDACTED_PYPI_TOKEN]"),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "[REDACTED_STRIPE_KEY]"),
    ("xai_api_key", re.compile(r"\bxai-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_XAI_KEY]"),
    ("groq_api_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), "[REDACTED_GROQ_KEY]"),
    ("supabase_token", re.compile(r"\bsbp_[a-f0-9]{32,}\b"), "[REDACTED_SUPABASE_TOKEN]"),
    ("vault_token", re.compile(r"\bhvs\.[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_VAULT_TOKEN]"),
    ("linear_api_key", re.compile(r"\blin_api_[A-Za-z0-9]{20,}\b"), "[REDACTED_LINEAR_KEY]"),
    ("notion_token", re.compile(r"\bntn_[A-Za-z0-9]{30,}\b"), "[REDACTED_NOTION_TOKEN]"),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{30,}\b"), "[REDACTED_TELEGRAM_TOKEN]"),
    (
        "discord_bot_token",
        re.compile(r"\b[MN][A-Za-z0-9_\-]{23,28}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,}\b"),
        "[REDACTED_DISCORD_TOKEN]",
    ),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_SENDGRID_KEY]"),
    ("databricks_token", re.compile(r"\bdapi[a-f0-9]{32}\b"), "[REDACTED_DATABRICKS_TOKEN]"),
    ("twilio_api_key", re.compile(r"\bSK[0-9a-fA-F]{32}\b"), "[REDACTED_TWILIO_KEY]"),
    (
        "gcp_service_account",
        re.compile(r'"private_key_id"\s*:\s*"[a-f0-9]{32,}"'),
        '"private_key_id": "[REDACTED_GCP_KEY_ID]"',
    ),
    (
        "db_connection_uri",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://"
            r"[^/\s:@\"']+:(?!pass(?:word)?@|changeme@|example@|your_)[^/\s@\"']{4,}@[^\s\"']+"
        ),
        "[REDACTED_DB_URI]",
    ),
    (
        "url_basic_auth",
        re.compile(
            r"\bhttps?://"
            r"[^/\s:@\"']+:(?!pass(?:word)?@|changeme@|example@|your_)[^/\s@\"']{4,}@[^\s\"']+"
        ),
        "[REDACTED_URL_CREDENTIALS]",
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
_WRAPPED_B64_BLOCK_RE = re.compile(r"(?m)(?:^[ \t]*[A-Za-z0-9+/_-]{16,80}[ \t]*\r?\n){1,80}[ \t]*[A-Za-z0-9+/_-]{8,80}={0,2}[ \t]*$")
_HEX_BLOB_RE = re.compile(r"\b[0-9a-fA-F]{64,}\b")
_ENCODED_SCAN_MAX_BLOBS = 200
_ENCODED_BLOB_MAX_CHARS = 8192

# Credential-named keys (including JSON `"api_key": "..."`) carrying opaque values with
# no recognizable token format. Values containing placeholder markers, POSIX `PWD=/path`
# directory assignments, or env references are untouched.
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)([\"']?)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"aws[_-]?secret[_-]?access[_-]?key|password|passwd|pwd|private[_-]?key|credential[s]?|"
    r"session[_-]?token|refresh[_-]?token)\b\1(\s*[:=]\s*)([\"']?)([A-Za-z0-9_\-./+=]{12,})\4"
)
_SENSITIVE_VALUE_SKIP_SUBSTR: Tuple[str, ...] = (
    "example", "placeholder", "changeme", "your_",
    "<your", "$", "{{", "%(", "os.environ", "getenv", "redacted",
)
_SENSITIVE_VALUE_WORD_SKIP_RE = re.compile(r"(?i)(?:^|[_\-.\s])(?:test|dummy|xxx)(?:$|[_\-.\s\d])")
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[/\\]")
_HIGH_ENTROPY_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z0-9+/=_\-]{20,}(?![A-Za-z0-9_\-])")


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy (bits per character) for secret triage."""
    s = str(text or "")
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = float(len(s))
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def detect_high_entropy_secrets(
    text: str,
    min_len: int = 20,
    min_entropy: float = 4.3,
) -> List[str]:
    """Identify candidate high-entropy alphanumeric/base64 strings (excludes file paths & hex digests)."""
    out: List[str] = []
    for m in _HIGH_ENTROPY_TOKEN_RE.finditer(str(text or "")):
        tok = m.group(0)
        if len(tok) < min_len or tok.startswith(("REDACTED", "[REDACTED")):
            continue
        if re.fullmatch(r"[a-f0-9]+", tok.lower()):
            continue
        if shannon_entropy(tok) >= float(min_entropy) and len(set(tok)) >= 10:
            out.append(tok)
            if len(out) >= 50:
                break
    return out


def _is_placeholder_value(val: str) -> bool:
    low = str(val or "").lower()
    if any(marker in low for marker in _SENSITIVE_VALUE_SKIP_SUBSTR):
        return True
    if _SENSITIVE_VALUE_WORD_SKIP_RE.search(low):
        return True
    if set(low) <= {"x", "*", "-"}:
        return True
    return False


def _redact_sensitive_key_values(text: str) -> Tuple[str, int]:
    """Redact opaque values assigned to credential-named keys in tool outputs.

    Catches credential dumps (env, INI, YAML, and JSON) that use no recognizable token
    format. Skips placeholders and POSIX `PWD=/dir/...` directory variables.
    """
    out = str(text or "")
    count = 0

    def _sub(match: "re.Match[str]") -> str:
        nonlocal count
        key_quote = match.group(1)
        key = match.group(2)
        sep = match.group(3)
        val_quote = match.group(4)
        value = match.group(5)
        if _is_placeholder_value(value):
            return match.group(0)
        if key.lower() == "pwd" and (
            value.startswith(("/", "~/", "./", "../"))
            or _WINDOWS_DRIVE_PATH_RE.match(value)
        ):
            return match.group(0)
        count += 1
        return f"{key_quote}{key}{key_quote}{sep}{val_quote}[REDACTED_SENSITIVE_VALUE]{val_quote}"

    out = _SENSITIVE_KEY_RE.sub(_sub, out)
    return out, count


def detect_sensitive_key_values(text: str) -> int:
    """Count credential-named key/value pairs whose values look like secrets."""
    _out, n = _redact_sensitive_key_values(str(text or ""))
    return n


def _decode_base64_candidate(blob: str) -> bytes:
    """Decode standard or URL-safe base64 candidates (padding unpadded strings to mod-4)."""
    padded = blob + ("=" * ((4 - len(blob) % 4) % 4))
    if "-" in blob or "_" in blob:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    return base64.b64decode(padded.encode("ascii"), validate=True)


def _match_decoded_secret_type(decoded: str) -> str | None:
    matched = next((t for t, pat, _r in _SECRET_PATTERNS if pat.search(decoded)), None)
    if matched is not None:
        return matched
    if detect_sensitive_key_values(decoded) > 0:
        return "sensitive_key_value"
    return None


def _redact_encoded_secret_blobs(text: str) -> Tuple[str, List[str], int]:
    """Decode wrapped/single-line base64, urlsafe-base64, and hex blobs and redact embedded secrets."""
    src = str(text or "")
    hits: List[str] = []
    redacted_count = 0
    out = src

    # Pass 1: multi-line wrapped base64 blocks (e.g., 76-char GNU `base64` PEM keys)
    wrapped_seen = 0
    for match in _WRAPPED_B64_BLOCK_RE.finditer(src):
        wrapped_seen += 1
        if wrapped_seen > 20:
            break
        block = match.group(0)
        joined = re.sub(r"\s+", "", block)
        if len(joined) > _ENCODED_BLOB_MAX_CHARS or len(joined) < 40:
            continue
        try:
            decoded = _decode_base64_candidate(joined).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if not decoded:
            continue
        matched = _match_decoded_secret_type(decoded)
        if matched is not None:
            if matched not in hits:
                hits.append(matched)
            out = out.replace(block, "[REDACTED_ENCODED_" + matched.upper() + "]")
            redacted_count += 1

    # Pass 2: single-line base64 and hex blobs
    for blob_re, codec in ((_BASE64_BLOB_RE, "b64"), (_HEX_BLOB_RE, "hex")):
        seen = 0
        for match in blob_re.finditer(out):
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
            matched = _match_decoded_secret_type(decoded)
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
