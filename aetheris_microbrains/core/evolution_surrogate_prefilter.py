"""
Evolution Attempt Pre-Filter Surrogate & Resident HDC Experience Memory
========================================================================
Wave-3 Measured GT 710 Co-Processor Proposals 4 & 6:
1. Fast on-device evolution surrogate pre-filter trained on real historical
   outcomes from `memory/store/journal.jsonl` and `memory/store/death_database.json`.
   Enforces a strict chronological 80/20 train/test split with held-out ROC-AUC evaluation.
   Honesty Gate: Only advises when held_out_auc > 0.55; otherwise abstains as 'unmeasured'.
2. Large-scale resident Hyperdimensional Computing (HDC / VSA) experience memory
   kept stationary in GT 710 VRAM for sub-millisecond near-duplicate goal detection and
   nearest-failure post-mortem retrieval.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .opencl_microbrain_accelerator import (
    OpenCLMicroBrainAccelerator,
    get_opencl_accelerator,
)

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_PATH = Path("memory/store/journal.jsonl")
DEFAULT_DEATH_DB_PATH = Path("memory/store/death_database.json")
DEFAULT_TOOLS_DIR = Path("execution/tools/dynamic")
PREFILTER_CACHE_PATH = Path("memory/store/evolution_surrogate_cache.json")


class ResidentHDCExperienceMemory:
    """
    Stationary 10,240-bit Hyperdimensional Computing (HDC) codebook resident in GT 710 VRAM.
    Indexes historical journal goals, death-database records, and dynamic tools.
    """

    _instance: Optional[ResidentHDCExperienceMemory] = None

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        death_db_path: Optional[Path] = None,
        tools_dir: Optional[Path] = None,
    ) -> None:
        self.journal_path = journal_path or DEFAULT_JOURNAL_PATH
        self.death_db_path = death_db_path or DEFAULT_DEATH_DB_PATH
        self.tools_dir = tools_dir or DEFAULT_TOOLS_DIR
        self.metadata: List[Dict[str, Any]] = []
        self.codebook_hv: np.ndarray = np.zeros((0, OpenCLMicroBrainAccelerator.HDC_WORDS), dtype=np.uint32)
        self.n_records: int = 0
        self.is_resident: bool = False
        self._build_and_upload_codebook()

    @classmethod
    def get_instance(cls) -> ResidentHDCExperienceMemory:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Anchor texts so the resident codebook is never empty. They carry NO
    # outcome label: the previous "seed" rows invented failed/success flags
    # with fake timestamps 1.0-4.0, and those fabricated labels poisoned the
    # surrogate's training corpus.
    ANCHOR_RECORDS: List[Dict[str, Any]] = [
        {"source": "anchor", "text": "causal manifold surgical repair regression fix", "failed": None, "ts": 0.0},
        {"source": "anchor", "text": "infinite recursive self invocation stack overflow", "failed": None, "ts": 0.0},
        {"source": "anchor", "text": "syntax error unclosed parenthesis AST break", "failed": None, "ts": 0.0},
        {"source": "anchor", "text": "optimize runtime speed minimal diff test pass", "failed": None, "ts": 0.0},
    ]

    @staticmethod
    def _parse_ts(value: Any) -> float:
        """Epoch seconds from a numeric or ISO-8601 timestamp; 0.0 if unparseable.

        The journal stores ISO strings ("2026-11-01T05:15:08.991079+00:00");
        float() on those raised ValueError inside the per-line except, which
        silently dropped EVERY real journal record from the harvest.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        try:
            return float(value)  # numeric strings
        except (TypeError, ValueError):
            try:
                from datetime import datetime

                return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError):
                return 0.0

    def _harvest_records(self) -> List[Dict[str, Any]]:
        """Harvest real historical goals, failure post-mortems, and deployed tools."""
        records: List[Dict[str, Any]] = list(self.ANCHOR_RECORDS)

        # 1. Harvest from journal.jsonl
        if self.journal_path.is_file():
            try:
                with open(self.journal_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                            ev_type = str(ev.get("type") or ev.get("event") or "")
                            data = ev.get("data") or {}
                            goal = str(data.get("goal") or data.get("current_goal") or "")
                            if not goal and "goal" in ev:
                                goal = str(ev["goal"])
                            if goal and len(goal) >= 5:
                                is_fail = any(x in ev_type.lower() for x in ("fail", "block", "error", "abort"))
                                is_succ = any(x in ev_type.lower() for x in ("success", "complete", "promote"))
                                if is_fail or is_succ:
                                    records.append({
                                        "source": "journal",
                                        "text": goal,
                                        "event_type": ev_type,
                                        "failed": bool(is_fail),
                                        "ts": self._parse_ts(ev.get("timestamp")),
                                    })
                        except Exception:
                            continue
            except Exception as exc:
                logger.debug("HDC Experience Memory journal harvest: %s", exc)

        # 2. Harvest from death_database.json
        if self.death_db_path.is_file():
            try:
                data = json.loads(self.death_db_path.read_text(encoding="utf-8", errors="ignore"))
                deaths = data.get("deaths") or data.get("incidents") or []
                if isinstance(deaths, dict):
                    deaths = list(deaths.values())
                for d in deaths:
                    if isinstance(d, dict):
                        text = str(d.get("goal") or d.get("hypothesis") or d.get("error") or d.get("cause") or "")
                        if text and len(text) >= 5:
                            records.append({
                                "source": "death_database",
                                "text": text,
                                "event_type": "death_record",
                                "failed": True,
                                "ts": self._parse_ts(d.get("timestamp")),
                            })
            except Exception as exc:
                logger.debug("HDC Experience Memory death DB harvest: %s", exc)

        # 3. Index deployed dynamic tools for retrieval only — file existence
        # is NOT an attempt outcome, so these stay unlabeled and are excluded
        # from surrogate training (the old failed=False inflated the success
        # class and fabricated the held-out AUC).
        if self.tools_dir.is_dir():
            try:
                for tool_file in self.tools_dir.glob("*.py"):
                    name = tool_file.stem
                    records.append({
                        "source": "tool_catalogue",
                        "text": f"dynamic tool {name.replace('_', ' ')}",
                        "event_type": "deployed_tool",
                        "failed": None,
                        "ts": tool_file.stat().st_mtime,
                    })
            except Exception as exc:
                logger.debug("HDC Experience Memory tool harvest: %s", exc)

        return records

    def _build_and_upload_codebook(self) -> None:
        """Encode harvested records into 10,240-bit hypervectors and upload to GT 710 VRAM."""
        records = self._harvest_records()
        if not records:
            # Unreachable while ANCHOR_RECORDS exists; kept as a guard so an
            # empty harvest can never produce a fabricated corpus.
            records = list(self.ANCHOR_RECORDS)

        # Deduplicate identical text
        unique_records = []
        seen = set()
        for r in records:
            t = r["text"].strip().lower()
            if t not in seen:
                seen.add(t)
                unique_records.append(r)

        self.metadata = unique_records
        self.n_records = len(unique_records)

        # Batch encode into (n_records, 320) uint32 hypervectors
        acc = get_opencl_accelerator()
        vectors = np.empty((self.n_records, OpenCLMicroBrainAccelerator.HDC_WORDS), dtype=np.uint32)
        for idx, rec in enumerate(unique_records):
            vectors[idx] = acc.encode_text_hypervector(rec["text"], bind_positions=False)

        self.codebook_hv = vectors
        reg_res = acc.register_resident_hdc_codebook("experience_memory", vectors, self.metadata)
        self.is_resident = bool(reg_res.get("vram_resident", False))

    def search(self, query_text: str, top_k: int = 5, force_gpu: bool = False) -> Dict[str, Any]:
        """Query the resident HDC experience memory in GT 710 VRAM."""
        acc = get_opencl_accelerator()
        q_hv = acc.encode_text_hypervector(query_text, bind_positions=False)
        return acc.search_resident_hdc_codebook("experience_memory", q_hv, top_k=top_k, force_gpu=force_gpu)

    def detect_near_duplicate(self, goal_text: str, threshold: float = 0.88) -> Dict[str, Any]:
        """Detect whether a new goal is an exact or near-duplicate of a prior attempt."""
        res = self.search(goal_text, top_k=1)
        matches = res.get("top_matches") or []
        if matches:
            best = matches[0]
            sim = float(best.get("similarity", 0.0))
            is_dup = sim >= threshold
            return {
                "is_near_duplicate": is_dup,
                "similarity": round(sim, 4),
                "matched_text": best.get("text", ""),
                "matched_source": best.get("source", ""),
                "matched_failed": best.get("failed", False),
                "backend": res.get("backend", "CPU"),
            }
        return {"is_near_duplicate": False, "similarity": 0.0, "backend": res.get("backend", "CPU")}

    def find_nearest_failure(self, goal_text: str) -> Optional[Dict[str, Any]]:
        """Retrieve the closest historical failure or death record."""
        res = self.search(goal_text, top_k=5)
        matches = res.get("top_matches") or []
        for m in matches:
            if m.get("failed") is True:
                return {
                    "text": m.get("text", ""),
                    "source": m.get("source", ""),
                    "similarity": round(float(m.get("similarity", 0.0)), 4),
                    "backend": res.get("backend", "CPU"),
                }
        return None


class EvolutionSurrogatePrefilter:
    """
    Real Evolution Attempt Pre-Filter Surrogate:
    1. Trains on historical (goal, failed) pairs using an 80/20 chronological time-split.
    2. Evaluates held-out ROC-AUC on the 20% test split.
    3. Honesty Gate: Only advises when held_out_auc > 0.55; otherwise honestly abstains as 'unmeasured'.
    4. Applies split-conformal calibration for calibrated risk sets and abstention.
    """

    _instance: Optional[EvolutionSurrogatePrefilter] = None

    def __init__(self, experience_memory: Optional[ResidentHDCExperienceMemory] = None) -> None:
        self.exp_mem = experience_memory or ResidentHDCExperienceMemory.get_instance()
        self.held_out_auc: float = 0.50
        self.status: str = "uncalibrated"
        self.advising: bool = False
        self.conformal_quantile: float = 0.50
        self.train_samples: int = 0
        self.test_samples: int = 0
        self.weights: Optional[np.ndarray] = None
        self._train_and_calibrate()

    @classmethod
    def get_instance(cls) -> EvolutionSurrogatePrefilter:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute exact Wilcoxon-Mann-Whitney ROC-AUC without external scikit-learn dependency."""
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        if n_pos == 0 or n_neg == 0:
            return 0.50

        # Rank scores with exact average rank for ties (1-based)
        order = np.argsort(y_scores, kind="mergesort")
        sorted_scores = y_scores[order]
        ranks = np.empty(len(y_scores), dtype=np.float64)
        i = 0
        n = len(y_scores)
        while i < n:
            j = i
            while j + 1 < n and abs(sorted_scores[j + 1] - sorted_scores[i]) <= 1e-12:
                j += 1
            avg_rank = 0.5 * ((i + 1) + (j + 1))
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1

        # Sum of ranks for positive class
        sum_pos_ranks = np.sum(ranks[y_true == 1])
        u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
        auc = u_stat / (n_pos * n_neg)
        return float(max(0.0, min(1.0, auc)))

    def _extract_features(self, text_list: List[str], dim: int = 32) -> np.ndarray:
        """Extract deterministic low-dimensional feature projections from text tokens."""
        n = len(text_list)
        feats = np.zeros((n, dim), dtype=np.float32)
        acc = get_opencl_accelerator()
        for idx, text in enumerate(text_list):
            hv = acc.encode_text_hypervector(text, bind_positions=False)
            # Unpack first `dim` 32-bit words into float features
            v = hv[:dim].astype(np.float32) / (2.0**31) - 1.0
            norm = np.linalg.norm(v) + 1e-12
            feats[idx] = v / norm
        return feats

    def _train_and_calibrate(self) -> None:
        """Strict 80/20 chronological time-split training and validation."""
        # Only genuinely labeled outcomes train the surrogate: anchors and the
        # deployed-tool catalogue carry failed=None and are excluded, so the
        # held-out AUC scores real journal/death-database evidence only.
        records = [
            r for r in self.exp_mem.metadata
            if r.get("failed") is not None and "text" in r
        ]
        # Sort chronologically by timestamp
        records.sort(key=lambda r: float(r.get("ts", 0.0) or 0.0))

        if len(records) < 10:
            self.status = "unmeasured"
            self.advising = False
            self.held_out_auc = 0.50
            return

        split_idx = int(len(records) * 0.80)
        train_recs = records[:split_idx]
        test_recs = records[split_idx:]

        self.train_samples = len(train_recs)
        self.test_samples = len(test_recs)

        x_train = self._extract_features([r["text"] for r in train_recs], dim=32)
        y_train = np.array([1.0 if r["failed"] else 0.0 for r in train_recs], dtype=np.float32)

        x_test = self._extract_features([r["text"] for r in test_recs], dim=32)
        y_test = np.array([1.0 if r["failed"] else 0.0 for r in test_recs], dtype=np.float32)

        # Train linear logistic weights via regularized least-squares / ridge projection
        reg = 0.10 * np.eye(32, dtype=np.float32)
        # Target in [-1, +1]
        target_train = y_train * 2.0 - 1.0
        w_sol = np.linalg.solve(np.dot(x_train.T, x_train) + reg, np.dot(x_train.T, target_train))
        self.weights = w_sol.astype(np.float32)

        # Predict on held-out test split
        test_logits = np.dot(x_test, self.weights)
        test_probs = 1.0 / (1.0 + np.exp(np.clip(-test_logits, -30.0, 30.0)))

        # Evaluate held-out ROC-AUC
        self.held_out_auc = self._compute_roc_auc(y_test, test_probs)

        # Split-conformal calibration on held-out split
        alpha = 0.10
        conformal_scores = np.abs(y_test - test_probs)
        q_level = min(1.0, math.ceil((len(y_test) + 1) * (1.0 - alpha)) / float(len(y_test)))
        self.conformal_quantile = float(np.quantile(conformal_scores, q_level))

        # Honesty Gate: Only advise when held_out_auc > 0.55
        if self.held_out_auc > 0.55:
            self.status = "calibrated_active"
            self.advising = True
        else:
            self.status = "unmeasured"
            self.advising = False

    def evaluate_goal_risk(self, goal_text: str) -> Dict[str, Any]:
        """
        Evaluate candidate goal risk:
        1. Checks resident HDC memory for near-duplicate and nearest death records.
        2. Applies calibrated surrogate model with honest abstention if AUC <= 0.55.
        """
        near_dup = self.exp_mem.detect_near_duplicate(goal_text)
        nearest_fail = self.exp_mem.find_nearest_failure(goal_text)

        if not self.advising or self.weights is None:
            return {
                "status": self.status,
                "advising": False,
                "reason": "held-out ROC-AUC <= 0.55 (abstaining honestly from uncalibrated predictions)",
                "held_out_auc": round(self.held_out_auc, 4),
                "risk_score": 0.50,
                "risk_score_basis": "neutral_prior_no_model",
                "abstained": True,
                "near_duplicate": near_dup,
                "nearest_failure": nearest_fail,
            }

        feat = self._extract_features([goal_text], dim=32)[0]
        logit = float(np.dot(feat, self.weights))
        prob_fail = float(1.0 / (1.0 + math.exp(max(-30.0, min(30.0, -logit)))))

        # Conformal interval
        interval = [max(0.0, prob_fail - self.conformal_quantile), min(1.0, prob_fail + self.conformal_quantile)]
        ambiguous = (interval[0] < 0.5 < interval[1])

        return {
            "status": self.status,
            "advising": True,
            "risk_score": round(prob_fail, 4),
            "conformal_interval": [round(x, 4) for x in interval],
            "abstained": ambiguous,
            "held_out_auc": round(self.held_out_auc, 4),
            "near_duplicate": near_dup,
            "nearest_failure": nearest_fail,
        }


def get_experience_memory() -> ResidentHDCExperienceMemory:
    return ResidentHDCExperienceMemory.get_instance()


def get_evolution_surrogate_prefilter() -> EvolutionSurrogatePrefilter:
    return EvolutionSurrogatePrefilter.get_instance()


__all__ = [
    "ResidentHDCExperienceMemory",
    "EvolutionSurrogatePrefilter",
    "get_experience_memory",
    "get_evolution_surrogate_prefilter",
]
