"""Procedural Directive Vault with Freshness Decay & Token-Budget Curation (stdlib-only).

Stores reusable cross-harness operational directives proposed from workflow feedback,
promotes entries only upon explicit approval, decays utility over elapsed time/failures,
and packs high-value directives under a strict bootstrap token budget while routing
stale entries to reconfirmation instead of silent injection.
"""

from __future__ import annotations

import fnmatch
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runtime_paths import file_lock, get_state_dir

_DEFAULT_CHARS_PER_TOKEN = 2.9


def _atomic_write_json(target: Path, data: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class DirectiveCard:
    """Single approved or pending procedural directive with decay and provenance."""

    directive_id: str
    scope_glob: str
    trigger: str
    guidance: str
    checks: Tuple[str, ...] = ()
    failure_modes: Tuple[str, ...] = ()
    author: str = "operator"
    approved_by: str = ""
    approved_at: float = 0.0
    last_verified_at: float = 0.0
    clean_outcomes: int = 0
    failed_outcomes: int = 0
    half_life_days: float = 30.0
    min_vitality: float = 0.35
    token_cost: int = 0
    status: str = "pending"  # pending | approved | stale | archived

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checks"] = list(self.checks)
        d["failure_modes"] = list(self.failure_modes)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DirectiveCard:
        guidance = str(data.get("guidance") or "")
        cost = int(data.get("token_cost") or max(1, round(len(guidance) / _DEFAULT_CHARS_PER_TOKEN)))
        return cls(
            directive_id=str(data.get("directive_id") or ""),
            scope_glob=str(data.get("scope_glob") or "*"),
            trigger=str(data.get("trigger") or ""),
            guidance=guidance,
            checks=tuple(str(x) for x in (data.get("checks") or ())),
            failure_modes=tuple(str(x) for x in (data.get("failure_modes") or ())),
            author=str(data.get("author") or "operator"),
            approved_by=str(data.get("approved_by") or ""),
            approved_at=float(data.get("approved_at") or 0.0),
            last_verified_at=float(data.get("last_verified_at") or data.get("approved_at") or 0.0),
            clean_outcomes=max(0, int(data.get("clean_outcomes") or 0)),
            failed_outcomes=max(0, int(data.get("failed_outcomes") or 0)),
            half_life_days=max(0.1, float(data.get("half_life_days") or 30.0)),
            min_vitality=max(0.01, min(0.99, float(data.get("min_vitality") or 0.35))),
            token_cost=max(1, cost),
            status=str(data.get("status") or "pending"),
        )


class DirectiveVault:
    """Cross-harness store for approval-gated directives with net-value-per-token ranking."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path is not None else (get_state_dir() / "directives" / "vault.json")

    def _load_unlocked(self) -> Dict[str, DirectiveCard]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        cards_raw = raw.get("directives") if isinstance(raw.get("directives"), dict) else raw
        out: Dict[str, DirectiveCard] = {}
        for k, v in cards_raw.items():
            if isinstance(v, dict):
                card = DirectiveCard.from_dict({**v, "directive_id": str(v.get("directive_id") or k)})
                if card.directive_id:
                    out[card.directive_id] = card
        return out

    def _save_unlocked(self, cards: Dict[str, DirectiveCard]) -> None:
        _atomic_write_json(
            self.path,
            {
                "version": 1,
                "updated_at": round(time.time(), 6),
                "directives": {k: v.to_dict() for k, v in sorted(cards.items())},
            },
        )

    def get(self, directive_id: str) -> Optional[DirectiveCard]:
        did = str(directive_id or "").strip()
        if not did:
            return None
        with file_lock(self.path):
            return self._load_unlocked().get(did)

    def propose(
        self,
        directive_id: str,
        *,
        guidance: str,
        scope_glob: str = "*",
        trigger: str = "",
        checks: Sequence[str] = (),
        failure_modes: Sequence[str] = (),
        author: str = "operator",
        half_life_days: float = 30.0,
        min_vitality: float = 0.35,
        now: Optional[float] = None,
    ) -> DirectiveCard:
        did = str(directive_id or "").strip()
        if not did:
            raise ValueError("directive_id must be non-empty")
        text = str(guidance or "").strip()
        if not text:
            raise ValueError("guidance must be non-empty")
        token_cost = max(1, round(len(text) / _DEFAULT_CHARS_PER_TOKEN))
        with file_lock(self.path):
            cards = self._load_unlocked()
            card = DirectiveCard(
                directive_id=did,
                scope_glob=str(scope_glob or "*"),
                trigger=str(trigger or ""),
                guidance=text,
                checks=tuple(str(x) for x in checks),
                failure_modes=tuple(str(x) for x in failure_modes),
                author=str(author or "operator"),
                half_life_days=max(0.1, float(half_life_days)),
                min_vitality=max(0.01, min(0.99, float(min_vitality))),
                token_cost=token_cost,
                status="pending",
            )
            cards[did] = card
            self._save_unlocked(cards)
            return card

    def approve(
        self,
        directive_id: str,
        *,
        approver: str,
        now: Optional[float] = None,
    ) -> DirectiveCard:
        did = str(directive_id or "").strip()
        ts = time.time() if now is None else float(now)
        with file_lock(self.path):
            cards = self._load_unlocked()
            card = cards.get(did)
            if card is None:
                raise ValueError(f"Directive {did!r} not found")
            card.approved_by = str(approver or "operator")
            card.approved_at = ts
            card.last_verified_at = ts
            card.status = "approved"
            if card.failed_outcomes >= card.clean_outcomes and card.failed_outcomes > 0:
                card.clean_outcomes = card.failed_outcomes + 1
            cards[did] = card
            self._save_unlocked(cards)
            return card

    def record_outcome(
        self,
        directive_id: str,
        *,
        clean: bool,
        now: Optional[float] = None,
    ) -> DirectiveCard:
        did = str(directive_id or "").strip()
        ts = time.time() if now is None else float(now)
        with file_lock(self.path):
            cards = self._load_unlocked()
            card = cards.get(did)
            if card is None:
                raise ValueError(f"Directive {did!r} not found")
            if clean:
                card.clean_outcomes += 1
            else:
                card.failed_outcomes += 1
            card.last_verified_at = max(card.last_verified_at, ts)
            scored = self.score_card(card, now=ts)
            if scored["disposition"] in {"RECONFIRM", "PRUNE"} and card.status == "approved":
                card.status = "stale"
            elif scored["disposition"] == "KEEP" and card.status == "stale":
                card.status = "approved"
            cards[did] = card
            self._save_unlocked(cards)
            return card

    @staticmethod
    def score_card(card: DirectiveCard, now: Optional[float] = None) -> Dict[str, Any]:
        ts = time.time() if now is None else float(now)
        utility = (1.0 + max(0, int(card.clean_outcomes))) / (
            2.0 + max(0, int(card.clean_outcomes)) + max(0, int(card.failed_outcomes))
        )
        anchor = max(float(card.last_verified_at), float(card.approved_at), 0.0)
        elapsed_days = max(0.0, (ts - anchor) / 86400.0) if anchor > 0 else 0.0
        hl = max(0.1, float(card.half_life_days))
        decay = 2.0 ** (-elapsed_days / hl)
        vitality = round(utility * decay, 6)
        nvpt = round(vitality / float(max(1, int(card.token_cost))), 6)

        if card.status == "archived":
            disposition = "PRUNE"
        elif card.status == "pending":
            disposition = "PENDING"
        elif vitality < float(card.min_vitality) * 0.5 or card.failed_outcomes >= card.clean_outcomes + 2:
            disposition = "PRUNE"
        elif vitality < float(card.min_vitality) or card.status == "stale" or (card.failed_outcomes > 0 and card.failed_outcomes >= card.clean_outcomes):
            disposition = "RECONFIRM"
        else:
            disposition = "KEEP"

        return {
            "utility": round(utility, 6),
            "decay": round(decay, 6),
            "vitality": vitality,
            "net_value_per_token": nvpt,
            "disposition": disposition,
        }

    def select_for_bootstrap(
        self,
        *,
        token_budget: int = 1200,
        scope_path: str = "",
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Rank approved directives by net-value-per-token; separate stale cards for reconfirmation."""
        ts = time.time() if now is None else float(now)
        with file_lock(self.path):
            cards = self._load_unlocked()

        norm_scope = str(scope_path or "").replace("\\", "/").strip()
        keep_candidates: List[Tuple[float, DirectiveCard, Dict[str, Any]]] = []
        stale_for_review: List[Dict[str, Any]] = []

        for card in cards.values():
            if card.status not in {"approved", "stale"}:
                continue
            if norm_scope:
                pat = card.scope_glob.replace("\\", "/").strip() or "*"
                if pat != "*" and not fnmatch.fnmatch(norm_scope, pat):
                    continue
            metrics = self.score_card(card, now=ts)
            if metrics["disposition"] == "KEEP":
                keep_candidates.append((metrics["net_value_per_token"], card, metrics))
            else:
                stale_for_review.append({**card.to_dict(), **metrics})

        keep_candidates.sort(key=lambda item: (-item[0], item[1].directive_id))
        selected: List[Dict[str, Any]] = []
        tokens_used = 0
        budget = max(0, int(token_budget))
        for _score, card, metrics in keep_candidates:
            if tokens_used + card.token_cost <= budget:
                tokens_used += card.token_cost
                selected.append({**card.to_dict(), **metrics})

        return {
            "selected": selected,
            "stale_for_review": sorted(stale_for_review, key=lambda d: d["directive_id"]),
            "tokens_used": tokens_used,
            "token_budget": budget,
        }
