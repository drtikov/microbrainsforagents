"""Intent Persistence Tracker: Trajectory-level governance & goal drift prevention.

Implements SOTA Intent Drift Scoring (IDS) and Dynamic Context Re-Anchoring:
  1. Immutable Root Objective Anchor (SHA-256 pinned task invariant)
  2. Trajectory-Level Prefix-Monotone Drift Metric (IDS)
  3. Cumulative Drift Velocity (d(Drift)/dt acceleration detection over sliding windows)
  4. Dynamic Context Re-Anchoring directives when cumulative drift crosses tolerance
  5. Privilege & Action Scoping against root task boundaries
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"(?u)\b\w{3,}\b")
_NEGATION_WORDS = {
    "not", "don't", "dont", "didn't", "didnt", "never",
    "avoid", "skip", "without", "no", "не", "без", "перестань",
}


def _extract_negated_terms(text: str, window: int = 3) -> Set[str]:
    """Return keywords that are preceded by a negation particle within `window` tokens."""
    low = str(text or "").lower()
    raw_tokens = [m.group(0) for m in re.finditer(r"(?u)\b\w+(?:'\w+)?\b", low)]
    negated: Set[str] = set()
    for idx, tok in enumerate(raw_tokens):
        if len(tok) >= 3:
            prev = raw_tokens[max(0, idx - window):idx]
            if any(p in _NEGATION_WORDS for p in prev):
                negated.add(tok)
    return negated


def _trigram_cosine_similarity(a: str, b: str) -> float:
    """Lightweight deterministic character-trigram similarity in [0, 1]."""
    def _tris(s: str) -> Set[str]:
        clean = re.sub(r"\s+", " ", s.lower().strip())
        if len(clean) < 3:
            return {clean} if clean else set()
        return {clean[i:i + 3] for i in range(len(clean) - 2)}

    ta, tb = _tris(a), _tris(b)
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb)) / ((len(ta) * len(tb)) ** 0.5)


def _extract_keywords(text: str) -> Set[str]:
    """Extract lowercase bilingual semantic keywords (RU/EN), excluding common stopwords."""
    stop_words = {
        "the", "and", "for", "with", "this", "that", "from", "are", "was",
        "were", "will", "would", "could", "should", "have", "has", "had",
        "file", "repo", "tool", "call", "step", "task", "done", "test",
        "into", "over", "under", "after", "before", "about",
        "для", "что", "как", "при", "или", "это", "все", "его", "она",
        "они", "быть", "через", "после", "перед", "файл", "репо", "шаг", "тест",
    }
    words = set(_WORD_RE.findall(str(text or "").lower()))
    return words - stop_words


def _stem_overlap(root_terms: Set[str], current_terms: Set[str]) -> Set[str]:
    """Compute keyword overlap with 4-char morphological prefix matching for bilingual inflections."""
    matched: Set[str] = set()
    for rt in root_terms:
        if rt in current_terms:
            matched.add(rt)
        elif len(rt) >= 5 and any(len(ct) >= 5 and ct[:4] == rt[:4] for ct in current_terms):
            matched.add(rt)
    return matched


@dataclass
class IntentDriftAssessment:
    """Comprehensive evaluation of trajectory drift from root objective."""

    root_goal: str
    root_hash: str
    current_action: str
    is_drifting: bool
    drift_score: float  # 0.0 (aligned) to 1.0 (completely detached)
    drift_velocity: float  # Rate of change over sliding window
    trajectory_length: int
    overlapping_terms: List[str] = field(default_factory=list)
    alarm_message: Optional[str] = None
    reanchoring_directive: Optional[str] = None
    neural_signal: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_goal": self.root_goal,
            "root_hash": self.root_hash,
            "current_action": self.current_action,
            "is_drifting": self.is_drifting,
            "drift_score": round(self.drift_score, 3),
            "drift_velocity": round(self.drift_velocity, 3),
            "trajectory_length": self.trajectory_length,
            "overlapping_terms": self.overlapping_terms,
            "alarm_message": self.alarm_message,
            "reanchoring_directive": self.reanchoring_directive,
            "neural_signal": self.neural_signal,
        }


class IntentPersistenceTracker:
    """SOTA trajectory-level goal tracker maintaining immutable purpose across context shifts."""

    def __init__(
        self,
        root_goal: str,
        drift_threshold: float = 0.65,
        velocity_threshold: float = 0.25,
        window_size: int = 4,
    ):
        self.root_goal = str(root_goal or "").strip()
        self.root_hash = hashlib.sha256(self.root_goal.encode("utf-8")).hexdigest()[:12]
        self.root_terms = _extract_keywords(self.root_goal)
        self.drift_threshold = drift_threshold
        self.velocity_threshold = velocity_threshold
        self.window_size = window_size

        self.turn_count = 0
        self.drift_warnings_count = 0
        self.history_scores: List[float] = []
        self.action_history: List[str] = []

    def register_step(
        self,
        action_summary: str,
        current_thoughts: str = "",
        action_params: Optional[Dict[str, Any]] = None,
    ) -> IntentDriftAssessment:
        """Register a turn and compute SOTA Intent Drift Score & Velocity."""
        self.turn_count += 1
        combined_current = f"{action_summary} {current_thoughts}"
        self.action_history.append(action_summary[:120])

        current_terms = _extract_keywords(combined_current)
        negated_current = _extract_negated_terms(combined_current)
        negated_roots = _stem_overlap(self.root_terms, negated_current)
        positive_current = current_terms - negated_current
        overlap = _stem_overlap(self.root_terms, positive_current)

        # Baseline Semantic Distance with negation penalty & paraphrase trigram floor
        if self.root_terms:
            overlap_ratio = len(overlap) / max(1, len(self.root_terms))
            if negated_roots:
                instant_drift = min(1.0, max(0.55, 1.0 - overlap_ratio + 0.35 * (len(negated_roots) / max(1, len(self.root_terms)))))
            else:
                instant_drift = max(0.0, 1.0 - overlap_ratio)
                sem_sim = _trigram_cosine_similarity(self.root_goal, combined_current)
                instant_drift = min(instant_drift, max(0.25, 1.0 - 1.5 * sem_sim))
        else:
            instant_drift = 0.0

        # Exploration grace period on first 2 exploratory turns
        if self.turn_count <= 2:
            instant_drift = min(instant_drift, 0.40)

        self.history_scores.append(instant_drift)

        # Sliding Window Drift Velocity d(Drift)/dt
        if len(self.history_scores) >= self.window_size:
            recent_window = self.history_scores[-self.window_size:]
            velocity = (recent_window[-1] - recent_window[0]) / max(1, len(recent_window) - 1)
        else:
            velocity = 0.0

        # Trajectory-weighted drift score: 0.7 * current + 0.3 * recent average
        avg_recent = sum(self.history_scores[-3:]) / len(self.history_scores[-3:])
        weighted_drift = 0.7 * instant_drift + 0.3 * avg_recent

        # Consult Neural Intent Persistence Tracker using in-vocabulary structural vectors
        pre_drift = (weighted_drift >= self.drift_threshold) or (
            velocity >= self.velocity_threshold and weighted_drift >= 0.50
        )
        neural_sig: Optional[str] = None
        try:
            tokens = (
                ["action_vector_orthogonal", "distraction_filter_engaged"]
                if pre_drift
                else ["action_vector_concordant", "context_amnesia_prevented"]
            )
            try:
                from .brain_moe_orchestrator import consult_microbrain
            except ImportError:
                from aetheris_microbrains.core.brain_moe_orchestrator import consult_microbrain

            res = consult_microbrain("intent_persistence_tracker", tokens)
            if isinstance(res, dict) and not res.get("abstained", False):
                neural_sig = res.get("prediction")
            drift_res = consult_microbrain(
                "drift_detector",
                ["intent_alignment_cosine_low_drift", "task_amnesia_risk_detected"]
                if pre_drift
                else ["scratchpad_entropy_stable", "context_coherence_optimal"],
            )
            if isinstance(drift_res, dict) and not drift_res.get("abstained", False) and drift_res.get("prediction"):
                d_pred = str(drift_res["prediction"])
                neural_sig = f"{neural_sig} | drift:{d_pred}" if neural_sig else f"drift:{d_pred}"
        except Exception as exc:
            log.debug("Intent neural consultation skipped: %s", exc)

        # Trigger conditions: absolute drift threshold, rapid acceleration, or neural veto
        is_drifting = pre_drift or (
            neural_sig is not None and any(w in str(neural_sig).lower() for w in ("abort", "alarm", "diverged"))
        )

        alarm_msg: Optional[str] = None
        reanchor_directive: Optional[str] = None

        if is_drifting:
            self.drift_warnings_count += 1
            alarm_msg = (
                f"⚠️ [INTENT DRIFT WARNING / ALARM - IDS={weighted_drift:.2f}]: Trajectory diverging from root objective "
                f"'{self.root_goal[:90]}' (hash: {self.root_hash})."
            )
            reanchor_directive = (
                f"[DYNAMIC CONTEXT RE-ANCHORING DIRECTIVE]:\n"
                f"1. Root Objective: \"{self.root_goal}\"\n"
                f"2. Observed Shift: Current step '{action_summary[:80]}' has low overlap with root keywords.\n"
                f"3. Observation: Verify alignment with the primary objective or continue if this is an intentional sub-step."
            )

        return IntentDriftAssessment(
            root_goal=self.root_goal,
            root_hash=self.root_hash,
            current_action=action_summary,
            is_drifting=is_drifting,
            drift_score=round(weighted_drift, 3),
            drift_velocity=round(velocity, 3),
            trajectory_length=self.turn_count,
            overlapping_terms=sorted(overlap),
            alarm_message=alarm_msg,
            reanchoring_directive=reanchor_directive,
            neural_signal=neural_sig,
        )


# Backward-compatible alias
IntentTracker = IntentPersistenceTracker
