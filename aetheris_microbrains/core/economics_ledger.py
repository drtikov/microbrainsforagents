"""Token preservation and session economics ledger (stdlib-only).

Tracks avoided context tokens from context-hygiene blocks and local microbrain
offloads, computing estimated cost savings and compact statusline metrics.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runtime_paths import append_jsonl_bounded, get_state_dir

DEFAULT_CHARS_PER_TOKEN = 2.9
DEFAULT_USD_PER_MTOK = 3.0


def estimate_tokens(
    text_or_bytes: int | str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimate token count from raw text or byte length using a calibrated code/JSON ratio."""
    ratio = float(chars_per_token) if chars_per_token and chars_per_token > 0 else DEFAULT_CHARS_PER_TOKEN
    count = len(text_or_bytes) if isinstance(text_or_bytes, str) else max(0, int(text_or_bytes))
    if count <= 0:
        return 0
    return max(1, round(count / ratio))


def get_default_economics_path() -> Path:
    """Resolve `economics_ledger.jsonl` under the per-user state dir (or `AETHERIS_ECONOMICS_PATH`)."""
    override = str(os.getenv("AETHERIS_ECONOMICS_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    return get_state_dir() / "economics_ledger.jsonl"


def _resolve_ledger_path(ledger_path: Optional[Path | str] = None) -> Path:
    explicit = str(ledger_path or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return get_default_economics_path()


def record_savings_event(
    event_kind: str,
    avoided_bytes: int,
    *,
    tool: str = "",
    reason_code: str = "",
    target: str = "",
    session_id: str = "",
    usd_per_mtok: float = DEFAULT_USD_PER_MTOK,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ledger_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Append one bounded token-preservation event to the economics ledger."""
    avoided_b = max(0, int(avoided_bytes))
    tokens = estimate_tokens(avoided_b, chars_per_token=chars_per_token)
    est_usd = round((tokens / 1_000_000.0) * float(usd_per_mtok), 6)
    entry: Dict[str, Any] = {
        "timestamp": round(time.time(), 6),
        "event_kind": str(event_kind or "context_hygiene_block"),
        "tool": str(tool or ""),
        "reason_code": str(reason_code or ""),
        "target": str(target or ""),
        "session_id": str(session_id or ""),
        "avoided_bytes": avoided_b,
        "avoided_tokens": tokens,
        "est_usd_saved": est_usd,
    }
    target_path = _resolve_ledger_path(ledger_path)
    try:
        append_jsonl_bounded(target_path, json.dumps(entry, ensure_ascii=False))
        return {"written": True, "path": str(target_path), "entry": entry, "error": None}
    except Exception as exc:
        return {"written": False, "path": str(target_path), "entry": entry, "error": f"{type(exc).__name__}:{exc}"}


def read_savings_events(
    ledger_path: Optional[Path | str] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read economics ledger events, optionally filtered by `session_id`."""
    p = _resolve_ledger_path(ledger_path)
    if not p.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if session_id and str(rec.get("session_id") or "") != session_id:
                    continue
                out.append(rec)
    except OSError:
        return []
    return out


def summarize_savings(
    ledger_path: Optional[Path | str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate avoided bytes, preserved tokens, USD saved, and breakdowns by reason/tool."""
    events = read_savings_events(ledger_path=ledger_path, session_id=session_id)
    total_bytes = 0
    total_tokens = 0
    total_usd = 0.0
    by_reason: Dict[str, Dict[str, int]] = {}
    by_tool: Dict[str, Dict[str, int]] = {}

    for ev in events:
        b = max(0, int(ev.get("avoided_bytes") or 0))
        t = max(0, int(ev.get("avoided_tokens") or 0))
        u = float(ev.get("est_usd_saved") or 0.0)
        total_bytes += b
        total_tokens += t
        total_usd += u

        rc = str(ev.get("reason_code") or "unknown")
        rc_bucket = by_reason.setdefault(rc, {"events": 0, "tokens": 0, "bytes": 0})
        rc_bucket["events"] += 1
        rc_bucket["tokens"] += t
        rc_bucket["bytes"] += b

        tl = str(ev.get("tool") or "unknown")
        tl_bucket = by_tool.setdefault(tl, {"events": 0, "tokens": 0, "bytes": 0})
        tl_bucket["events"] += 1
        tl_bucket["tokens"] += t
        tl_bucket["bytes"] += b

    return {
        "events": len(events),
        "avoided_bytes": total_bytes,
        "tokens_preserved": total_tokens,
        "est_usd_saved": round(total_usd, 4),
        "by_reason": by_reason,
        "by_tool": by_tool,
    }


def format_statusline(summary: Dict[str, Any], as_json: bool = False) -> str:
    """Format a compact single-line telemetry summary suitable for IDE/CLI status bars."""
    tokens = int(summary.get("tokens_preserved") or 0)
    usd = float(summary.get("est_usd_saved") or 0.0)
    events = int(summary.get("events") or 0)
    if as_json:
        return json.dumps(
            {
                "events": events,
                "tokens_preserved": tokens,
                "est_usd_saved": round(usd, 4),
            },
            ensure_ascii=False,
        )
    tok_label = f"{tokens / 1000.0:.1f}k" if tokens >= 1000 else str(tokens)
    return f"MB saved: {tok_label} tok (${usd:.2f}) | {events} hygiene blocks"


class TokenSavingsLedger:
    """Object wrapper around the economics ledger JSONL path."""

    def __init__(
        self,
        ledger_path: Optional[Path | str] = None,
        *,
        usd_per_mtok: float = DEFAULT_USD_PER_MTOK,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ) -> None:
        self.path = _resolve_ledger_path(ledger_path)
        self.usd_per_mtok = float(usd_per_mtok)
        self.chars_per_token = float(chars_per_token)

    def record(
        self,
        event_kind: str,
        avoided_bytes: int,
        *,
        tool: str = "",
        reason_code: str = "",
        target: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        return record_savings_event(
            event_kind=event_kind,
            avoided_bytes=avoided_bytes,
            tool=tool,
            reason_code=reason_code,
            target=target,
            session_id=session_id,
            usd_per_mtok=self.usd_per_mtok,
            chars_per_token=self.chars_per_token,
            ledger_path=self.path,
        )

    def summarize(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        return summarize_savings(ledger_path=self.path, session_id=session_id)

    def statusline(self, session_id: Optional[str] = None, as_json: bool = False) -> str:
        return format_statusline(self.summarize(session_id=session_id), as_json=as_json)
