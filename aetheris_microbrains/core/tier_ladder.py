"""Declarative model and reasoning-effort tier ladder resolver (stdlib-only).

Maps (tier, role, depth_hint) tuples to concrete model names and effort budgets,
stripping inline `[depth:<level>]` tags cleanly before prompt dispatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_DEPTH_TAG_RE = re.compile(
    r"\s*\[depth\s*:\s*(fast|low|medium|high|deep|max)\]\s*$",
    re.IGNORECASE,
)

_HINT_TO_EFFORT: Dict[str, str] = {
    "fast": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "deep": "high",
    "max": "max",
}

DEFAULT_LADDER_TABLE: Dict[Tuple[str, str], Tuple[str, str]] = {
    # economy tier: minimizes cost for high-volume leaf tasks
    ("economy", "retriever"): ("fast-retriever", "low"),
    ("economy", "coder"): ("compact-coder", "medium"),
    ("economy", "reviewer"): ("compact-reviewer", "medium"),
    ("economy", "adjudicator"): ("standard-reasoner", "medium"),
    ("economy", "planner"): ("standard-reasoner", "high"),
    # standard tier: balanced quality & latency
    ("standard", "retriever"): ("fast-retriever", "medium"),
    ("standard", "coder"): ("standard-coder", "medium"),
    ("standard", "reviewer"): ("standard-reviewer", "medium"),
    ("standard", "adjudicator"): ("frontier-reasoner", "high"),
    ("standard", "planner"): ("frontier-reasoner", "high"),
    # pro tier: maximum capability for critical verification
    ("pro", "retriever"): ("standard-retriever", "medium"),
    ("pro", "coder"): ("frontier-coder", "high"),
    ("pro", "reviewer"): ("frontier-reviewer", "high"),
    ("pro", "adjudicator"): ("frontier-reasoner", "max"),
    ("pro", "planner"): ("frontier-reasoner", "max"),
    # local tier: on-device / air-gapped execution
    ("local", "retriever"): ("local-small", "low"),
    ("local", "coder"): ("local-coder", "medium"),
    ("local", "reviewer"): ("local-coder", "medium"),
    ("local", "adjudicator"): ("local-reasoner", "high"),
    ("local", "planner"): ("local-reasoner", "high"),
}


@dataclass(frozen=True)
class TierRoute:
    """Resolved model and reasoning budget for a helper dispatch."""

    role: str
    tier: str
    model: str
    effort: str
    hint_applied: str = ""


def extract_depth_hint(text: str) -> Tuple[str, str]:
    """Extract and strip a trailing `[depth:<level>]` tag from `text`."""
    if not isinstance(text, str):
        return "", ""
    m = _DEPTH_TAG_RE.search(text)
    if m is None:
        return "", text
    level = m.group(1).strip().lower()
    cleaned = text[: m.start()].rstrip()
    return level, cleaned


class TierLadderResolver:
    """Resolves helper role, tier, and inline depth hints to model routes."""

    def __init__(
        self,
        table: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    ) -> None:
        self._table: Dict[Tuple[str, str], Tuple[str, str]] = dict(DEFAULT_LADDER_TABLE)
        if table:
            for (t, r), val in table.items():
                self._table[(str(t).lower(), str(r).lower())] = val

    def resolve(
        self,
        role: str,
        *,
        tier: str = "standard",
        description: str = "",
        effort_override: Optional[str] = None,
    ) -> Tuple[TierRoute, str]:
        """Resolve `(TierRoute, cleaned_description)` applying hint or explicit override."""
        norm_role = str(role or "coder").strip().lower()
        norm_tier = str(tier or "standard").strip().lower()

        hint, cleaned_desc = extract_depth_hint(description)
        model, default_effort = self._table.get(
            (norm_tier, norm_role),
            self._table.get(("standard", norm_role), ("standard-coder", "medium")),
        )

        effort = default_effort
        hint_applied = ""
        if effort_override and str(effort_override).strip():
            raw_ov = str(effort_override).strip().lower()
            effort = _HINT_TO_EFFORT.get(raw_ov, raw_ov)
            hint_applied = f"override:{effort}"
        elif hint:
            effort = _HINT_TO_EFFORT.get(hint, default_effort)
            hint_applied = hint

        return (
            TierRoute(
                role=norm_role,
                tier=norm_tier,
                model=model,
                effort=effort,
                hint_applied=hint_applied,
            ),
            cleaned_desc,
        )
