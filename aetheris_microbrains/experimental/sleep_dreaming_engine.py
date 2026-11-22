"""
Part 2B: Autonomous Sleep, Memory Consolidation & Dreaming Engine
================================================================
Inspired by NeoAura `sleep_replay.py` and VICReg representation alignment:
During idle periods (when no external operator queries are active):
1. Samples historical success/failure traces from `memory/store/strategy_bank.jsonl`.
2. Plays back self-generated associative queries through active micro-brains.
3. Performs covariance regularized alignment to eliminate noise modes in weight matrices.
4. Sweeps the population for saturated brains and performs spectral fission
   (mitosis into _alpha/_beta daughters) when a brain degenerates.

Merging duplicate brains is deliberately NOT automatic — it is exposed as a
deliberate operation through the neuro_brain_incubator dynamic tool
(merge_slerp / merge_ties), because auto-merge without duplicate detection
could destroy curated brains.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from ..core.brain_nursery import AutonomousBrainNursery
from ..core.brain_merger import NeuroBrainMerger
from ..core.runtime_paths import get_writable_store_dir

logger = logging.getLogger(__name__)
STORE_DIR = get_writable_store_dir()
STRATEGY_BANK_FILE = STORE_DIR / "strategy_bank.jsonl"


class AutonomousSleepDreamingEngine:
    """Consolidates memory traces, aligns weight spaces, and trims redundancy during idle phases."""

    _instance: Optional[AutonomousSleepDreamingEngine] = None

    def __init__(self):
        self.nursery = AutonomousBrainNursery.get_instance()
        self.merger = NeuroBrainMerger.get_instance()
        self.total_sleep_cycles = 0

    @classmethod
    def get_instance(cls) -> AutonomousSleepDreamingEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def execute_sleep_consolidation_cycle(self) -> Dict[str, Any]:
        """
        Runs one offline sleep & memory replay cycle:
        - Replays historical failure/success lessons.
        - Orthogonalizes active weight matrices to prevent spectral drift.
        """
        replayed_lessons = 0
        records_to_distill: List[Dict[str, Any]] = []
        successful_tools: List[str] = []
        if STRATEGY_BANK_FILE.exists():
            try:
                with open(STRATEGY_BANK_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            records_to_distill.append(record)
                            tool_name = str(record.get("tool_name") or record.get("tool") or "").strip()
                            outcome_ok = record.get("success", record.get("outcome") != "failure")
                            if tool_name and outcome_ok:
                                successful_tools.append(tool_name)
                            replayed_lessons += 1
            except Exception:
                pass

        if records_to_distill:
            try:
                from ..core.weight_space_memory_synthesis import WeightSpaceMemorySynthesizer
                WeightSpaceMemorySynthesizer.get_instance().distill_journal_records(records_to_distill)
            except Exception as exc:
                logger.debug("Weight-space distillation skipped: %s", exc)

        if successful_tools:
            try:
                from .hebbian_plasticity_consolidator import HebbianPlasticityConsolidator
                HebbianPlasticityConsolidator.get_instance().consolidate_success(
                    successful_tools, outcome_quality=1.0
                )
            except Exception as exc:
                logger.debug("Hebbian consolidation skipped: %s", exc)

        # Spectral alignment, de-noising & VICReg covariance regularization
        active_brains = list(self.nursery.active_brains.values())
        smoothed_brains = []
        total_cov_penalty = 0.0
        for brain in active_brains:
            # SVD-based trace smoothing on W_up and W_down
            for attr in ("W_up", "W_down"):
                mat = getattr(brain, attr, None)
                if isinstance(mat, np.ndarray) and mat.ndim == 2 and min(mat.shape) > 0:
                    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
                    if S.size > 0 and S[0] > 1e-9:
                        noise_thresh = S[0] * 0.02
                        S_clean = np.where(S < noise_thresh, 0.0, S)
                        setattr(brain, attr, np.matmul(U * S_clean, Vt).astype(np.float32))

            # VICReg off-diagonal covariance regularization on W_mlp
            w_mlp = getattr(brain, "W_mlp", None)
            if isinstance(w_mlp, np.ndarray) and w_mlp.ndim == 2 and w_mlp.shape[0] > 1:
                d_feat = w_mlp.shape[1]
                w_centered = w_mlp - np.mean(w_mlp, axis=0, keepdims=True)
                cov = (w_centered.T @ w_centered) / float(max(1, w_mlp.shape[0] - 1))
                off_mask = 1.0 - np.eye(d_feat, dtype=np.float32)
                c_off = cov * off_mask
                cov_penalty = float(np.sum(c_off ** 2) / float(max(1, d_feat)))
                total_cov_penalty += cov_penalty
                off_diag_grad = np.matmul(w_centered, c_off) / float(max(1, d_feat))
                brain.W_mlp = (w_mlp - 0.01 * off_diag_grad).astype(np.float32)

            brain.save(STORE_DIR)
            smoothed_brains.append(brain.name)

        # Saturation sweep: spectral fission when a brain degenerates
        # (condition number / vocab ceiling). Healthy brains are left untouched.
        fission_checked = 0
        fission_events = []
        try:
            from ..core.brain_cellular_fission import BrainCellularFissionEngine
            fission = BrainCellularFissionEngine.get_instance()
            for brain_name in list(self.nursery.active_brains.keys()):
                domain = brain_name
                if domain.startswith("neuro_"):
                    domain = domain[len("neuro_"):]
                result = fission.evaluate_and_split_if_saturated(domain)
                fission_checked += 1
                if result.get("status") == "fission_executed":
                    fission_events.append({
                        "source": brain_name,
                        "daughters": [result.get("daughter_a"), result.get("daughter_b")],
                    })
        except Exception as e:
            logger.debug("Fission sweep skipped: %s", e)

        self.total_sleep_cycles += 1
        return {
            "status": "sleep_cycle_completed",
            "cycle_number": self.total_sleep_cycles,
            "replayed_lessons": replayed_lessons,
            "distilled_records": len(records_to_distill),
            "smoothed_brains_count": len(smoothed_brains),
            "smoothed_brains": smoothed_brains,
            "vicreg_covariance_penalty": round(float(total_cov_penalty), 6),
            "fission_checked": fission_checked,
            "fission_events": fission_events,
            "timestamp": time.time(),
        }

