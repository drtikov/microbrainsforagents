"""Canonical credential/token detection and redaction patterns (stdlib-only).

Shared between `hook.py` and `layered_decision_engine.py` so error redaction,
PostToolUse scrubbing, and diff secret scanning never diverge.
"""

from __future__ import annotations

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


def redact_secrets(text: str) -> str:
    """Mask known credential/token patterns before logging or printing errors."""
    out = str(text or "")
    for pat, repl in _REDACT_PATTERNS:
        out = pat.sub(repl, out)
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
    return {
        "redacted_text": out,
        "secrets_found": total_hits > 0,
        "secret_count": total_hits,
        "secret_types": sorted(found_types),
    }
