"""Deterministic structured handback envelope parser and validator (stdlib-only).

Extracts standardized completion headers from subagent or worker replies without
requiring secondary LLM parsing, preventing unverified completion claims.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

VALID_OUTCOMES = frozenset({"completed", "failed", "blocked", "partial", "aborted"})
KNOWN_FIELDS = ("SUMMARY", "ARTIFACT", "VERIFIED_BY", "NEXT_ACTION", "BLOCKER")

_OUTCOME_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?OUTCOME(?:\*\*)?\s*[:=]\s*`?([A-Za-z0-9_-]+)`?",
    re.IGNORECASE,
)
_FIELD_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?([A-Za-z][A-Za-z0-9_]*)(?:\*\*)?\s*[:=]\s*(.*)$"
)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class HandshakeEnvelope:
    """Structured completion envelope parsed from a worker reply."""

    outcome: str
    summary: str = ""
    artifact: str = ""
    verified_by: str = ""
    next_action: str = ""
    blocker: str = ""
    extras: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_handback_envelope(text: str, scan_window: int = 2000) -> Optional[HandshakeEnvelope]:
    """Parse a structured `OUTCOME:` header and known fields from the head of `text`."""
    if not isinstance(text, str) or not text.strip():
        return None

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    head = normalized[: max(1, int(scan_window))]
    m = _OUTCOME_RE.search(head)
    if m is None:
        return None

    outcome = m.group(1).strip().lower()
    tail_lines = normalized[m.end() :].splitlines()
    # Skip remainder of the OUTCOME line itself.
    if tail_lines and normalized[m.end() : m.end() + 1] not in ("", "\n"):
        tail_lines = tail_lines[1:]

    fields: Dict[str, str] = {}
    extras: Dict[str, str] = {}
    current_key: Optional[str] = None
    in_extras = False

    for line in tail_lines:
        if _FENCE_RE.match(line):
            break
        if not line.strip():
            if fields or extras:
                break
            continue

        fm = _FIELD_RE.match(line)
        if fm:
            raw_key = fm.group(1).strip().upper()
            val = fm.group(2).strip()
            current_key = raw_key
            if raw_key in KNOWN_FIELDS:
                in_extras = False
                fields[raw_key] = val
            else:
                in_extras = True
                extras[raw_key.lower()] = val
            continue

        # Indented or continuation prose belongs to the most recently opened key.
        if current_key is not None:
            cont = line.strip()
            if in_extras:
                ek = current_key.lower()
                extras[ek] = f"{extras[ek]} {cont}".strip()
            else:
                fields[current_key] = f"{fields[current_key]} {cont}".strip()
        else:
            break

    return HandshakeEnvelope(
        outcome=outcome,
        summary=fields.get("SUMMARY", ""),
        artifact=fields.get("ARTIFACT", ""),
        verified_by=fields.get("VERIFIED_BY", ""),
        next_action=fields.get("NEXT_ACTION", ""),
        blocker=fields.get("BLOCKER", ""),
        extras=extras,
    )


def validate_handback_envelope(
    text: str,
    *,
    require_verification: bool = True,
    scan_window: int = 2000,
) -> Dict[str, Any]:
    """Validate an agent completion envelope and report missing or invalid fields."""
    env = parse_handback_envelope(text, scan_window=scan_window)
    if env is None:
        return {
            "valid": False,
            "envelope": None,
            "missing_fields": ["OUTCOME"],
            "errors": ["missing required `OUTCOME:` header in response"],
        }

    missing: list[str] = []
    errors: list[str] = []

    if env.outcome not in VALID_OUTCOMES:
        errors.append(
            f"invalid OUTCOME {env.outcome!r}; expected one of {sorted(VALID_OUTCOMES)}"
        )

    if not env.summary.strip():
        missing.append("SUMMARY")
        errors.append("missing `SUMMARY:` field")

    if require_verification and env.outcome == "completed" and not env.verified_by.strip():
        missing.append("VERIFIED_BY")
        errors.append("`OUTCOME: completed` requires non-empty `VERIFIED_BY:` evidence")

    if env.outcome in {"blocked", "failed"} and not (env.blocker.strip() or env.next_action.strip()):
        missing.append("BLOCKER")
        errors.append(f"`OUTCOME: {env.outcome}` requires `BLOCKER:` or `NEXT_ACTION:` context")

    return {
        "valid": len(errors) == 0,
        "envelope": env.to_dict(),
        "missing_fields": missing,
        "errors": errors,
    }
