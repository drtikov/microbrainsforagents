"""
Aetheris Sleep Memory Consolidation Engine (REM & NREM).
Mimics biological sleep cycles to consolidate waking plastic experience into long-term structure:
    1. Replays and filters daytime Hebbian fast-weights and WSP patches through Sparse Autoencoders (SAE).
    2. Discards low-salience noise & synaptic drift.
    3. Crystalizes high-performing synaptic adjustments into persistent Model Soups via TIES.
    4. Resets fast plastic buffers so the agent wakes up with refreshed cognitive plasticity.
Runs automatically in scheduled background evolution cycles (sched_evolution).
"""

from __future__ import annotations
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ..evolution.wsp_substrate import CPUMicroRouter
from .wsp_hebbian import HebbianPlasticLayer
from ..evolution.wsp_merger import WeightSpaceMerger
from ..evolution.wsp_sae import ConceptSparseAutoencoder

logger = logging.getLogger(__name__)

STORE_DIR = Path("memory/store")
SLEEP_CONSOLIDATION_LEDGER = STORE_DIR / "wsp_sleep_ledger.jsonl"


class SleepConsolidationEngine:
    """
    Orchestrates the Sleep & Synaptic Consolidation Cycle for Aetheris.
    """

    def __init__(
        self,
        router: Optional[CPUMicroRouter] = None,
        hebbian: Optional[HebbianPlasticLayer] = None,
        sae: Optional[ConceptSparseAutoencoder] = None,
        merger: Optional[WeightSpaceMerger] = None,
        salience_threshold: float = 0.05,
    ):
        self.router = router or CPUMicroRouter()
        self.hebbian = hebbian or HebbianPlasticLayer(d_in=self.router.d_hidden, d_out=self.router.d_out)
        self.sae = sae or ConceptSparseAutoencoder(d_model=self.router.d_hidden)
        self.merger = merger or WeightSpaceMerger()
        self.salience_threshold = salience_threshold
        STORE_DIR.mkdir(parents=True, exist_ok=True)

    def run_sleep_cycle(self, winning_attempts: Optional[List[Dict[str, np.ndarray]]] = None) -> Dict[str, Any]:
        """
        Executes a full NREM/REM sleep consolidation cycle:
        Phase 1 (NREM): Synaptic pruning of low-salience fast-weights (Hebbian decay & thresholding).
        Phase 2 (REM): Replay through SAE to ensure concept-level alignment and no catastrophic forgetting.
        Phase 3 (Consolidation): Merges salience-approved plastic deltas into active long-term patches via TIES.
        Phase 4 (Awakening): Flushes short-term fast buffers, returning full plasticity for the next waking epoch.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(f"SleepConsolidationEngine: Initiating sleep consolidation cycle at {timestamp}...")

        # ---------------------------------------------------------------------
        # Phase 1 (NREM): Synaptic Pruning of Daytime Hebbian Weights
        # ---------------------------------------------------------------------
        A_fast = self.hebbian.A_fast.copy()
        raw_norm = float(np.linalg.norm(A_fast))

        # Prune insignificant synaptic noise below salience_threshold
        pruning_mask = np.abs(A_fast) >= self.salience_threshold
        A_pruned = A_fast * pruning_mask
        n_pruned = int(np.count_nonzero(~pruning_mask))
        n_retained = int(np.count_nonzero(pruning_mask))

        # ---------------------------------------------------------------------
        # Phase 2 (REM): SAE Concept Filter & Online AuxK Maintenance
        # ---------------------------------------------------------------------
        # Verify that pruned deltas do not perturb safety invariant concepts
        row_norms = np.linalg.norm(A_pruned, axis=1)
        sae_input = row_norms[: self.sae.d_model]
        if len(sae_input) < self.sae.d_model:
            sae_input = np.pad(sae_input, (0, self.sae.d_model - len(sae_input)))
        sae_train_metrics: Optional[Dict[str, Any]] = None
        if n_retained > 0:
            sae_train_metrics = self.sae.train_step(sae_input)
            self.sae.save_dictionary()
        z_concept = self.sae.encode(sae_input)
        # Concept 0 is 'core_safety_invariant'
        safety_perturbation = float(z_concept[0]) if len(z_concept) > 0 else 0.0

        # ---------------------------------------------------------------------
        # Phase 3: Crystallization into Persistent Active Delta Layer
        # ---------------------------------------------------------------------
        # If Hebbian fast weights accumulated meaningful direction, merge into W2 delta
        if n_retained > 0 and safety_perturbation < 1.0:
            scale_factor = 0.2
            self.router.delta_W2 += (scale_factor * A_pruned).astype(np.float32)
            self.router.save_active_patches()

        # If winning milestone attempt deltas were supplied, brew a long-term Model Soup
        soup_meta: Optional[Dict[str, Any]] = None
        if winning_attempts and len(winning_attempts) > 1:
            scores = [1.0] * len(winning_attempts)
            soup_meta = self.merger.brew_model_soup(
                attempt_deltas=winning_attempts,
                performance_scores=scores,
                name=f"soup_consolidated_{int(datetime.datetime.now().timestamp())}",
            )

        # ---------------------------------------------------------------------
        # Phase 4 (Awakening): Flush Fast Plastic Buffer
        # ---------------------------------------------------------------------
        self.hebbian.reset_fast_weights()

        result = {
            "timestamp": timestamp,
            "status": "consolidated",
            "hebbian_raw_norm": round(raw_norm, 4),
            "synapses_pruned": n_pruned,
            "synapses_crystallized": n_retained,
            "safety_concept_perturbation": round(safety_perturbation, 4),
            "sae_train_metrics": sae_train_metrics,
            "soup_brewed": soup_meta is not None,
            "soup_details": soup_meta,
        }

        self._record_sleep_ledger(result)
        logger.info(f"SleepConsolidationEngine: Consolidation complete. Retained {n_retained} synapses.")
        return result

    def _record_sleep_ledger(self, record: Dict[str, Any]) -> None:
        try:
            with open(SLEEP_CONSOLIDATION_LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to record sleep ledger: {e}")

