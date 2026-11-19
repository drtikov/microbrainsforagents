"""
Auto-Dreamer Offline Memory Consolidation & Wiki Distiller for Aetheris.
Ported and evolved from Anthropic Research ("Dreaming", 2026) and arXiv:2605.20616 ("Auto-Dreamer: Learning Offline Memory Consolidation").

Executes asynchronous offline memory maintenance during Loop-3 daemon idle cycles:
1. Replays raw attempt logs and failure transcripts.
2. Clusters repeated failures to synthesize concise, actionable "Memory Notes" (Rules & Traps).
3. Compiles memory notes into an interlinked knowledge wiki (memory/store/wiki_knowledge.json).
4. Prunes stale and redundant historical logs, bounding disk footprint and eliminating context pollution.
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger("aetheris.memory.auto_dreamer")

_PKG_STORE_DIR = Path(__file__).resolve().parents[1] / "memory" / "store"
WIKI_KNOWLEDGE_PATH = _PKG_STORE_DIR / "wiki_knowledge.json"
MEMORY_NOTES_PATH = _PKG_STORE_DIR / "memory_notes.jsonl"


@dataclass
class MemoryNote:
    note_id: str
    category: str
    target_component: str
    rule_or_trap: str
    supporting_evidence_count: int
    created_at: float = field(default_factory=time.time)
    # Editorial placeholder, not a measurement: nothing computes salience and
    # no consumer ranks by it — every note gets the same constant (0.85 here,
    # 0.90 for trap notes below). Treat the field as decoration until a real
    # scorer lands.
    salience: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DreamReport:
    raw_entries_scanned: int
    clusters_identified: int
    new_notes_synthesized: int
    wiki_articles_updated: int
    stale_entries_pruned: int
    duration_ms: float


class AutoDreamerConsolidator:
    """
    Offline sleep/consolidation engine distilling raw logs into high-level memory notes and wiki entries.
    """

    _instance: Optional["AutoDreamerConsolidator"] = None

    def __init__(
        self,
        attempt_ledger_path: Optional[str] = None,
        wiki_path: Optional[Path] = None,
        notes_path: Optional[Path] = None,
    ):
        self.ledger_path = Path(attempt_ledger_path) if attempt_ledger_path else (_PKG_STORE_DIR / "attempt_ledger.jsonl")
        self.wiki_path = Path(wiki_path) if wiki_path else WIKI_KNOWLEDGE_PATH
        self.notes_path = Path(notes_path) if notes_path else MEMORY_NOTES_PATH
        self.weight_memory_path = self.notes_path.parent / "neuro_brains" / "neuro_weight_space_memory.json"

    @classmethod
    def get_instance(cls) -> "AutoDreamerConsolidator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def consolidate_offline_dream(self, max_entries_to_scan: int = 150, cluster_threshold: int = 2, lookback_window: Optional[int] = None) -> DreamReport:
        """Alias for run_dream_cycle."""
        if lookback_window is not None:
            max_entries_to_scan = lookback_window
        return self.run_dream_cycle(max_entries_to_scan=max_entries_to_scan, cluster_threshold=cluster_threshold)

    def run_dream_cycle(self, max_entries_to_scan: int = 150, cluster_threshold: int = 2) -> DreamReport:
        """
        Runs one offline dream consolidation cycle.
        """
        t0 = time.perf_counter()
        if not self.ledger_path.exists():
            return DreamReport(0, 0, 0, 0, 0, 0.0)

        # 1. Read recent raw entries
        lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        recent_lines = lines[-max_entries_to_scan:]
        failures = []

        for line in recent_lines:
            try:
                entry = json.loads(line)
                # Check standard error fields across ledger formats
                if entry.get("event") in ("REJECTED", "FAILED"):
                    failures.append(str(entry.get("reason", "")))
                elif entry.get("error_type") or "error" in entry:
                    reason = f"{entry.get('error_type', '')} {entry.get('stdout', '')} {entry.get('error', '')}"
                    failures.append(reason)
            except Exception:
                continue

        # 2. Cluster recurring failure modes
        clusters: Dict[str, int] = {}
        for f in failures:
            low = f.lower()
            if "syntax" in low or "ast" in low:
                clusters["ast_syntax_integrity"] = clusters.get("ast_syntax_integrity", 0) + 1
            elif "coverage" in low or "test does not reference" in low or "entrypoint was never called" in low:
                clusters["test_call_coverage"] = clusters.get("test_call_coverage", 0) + 1
            elif "timeout" in low or "timed out" in low:
                clusters["execution_timeout"] = clusters.get("execution_timeout", 0) + 1
            elif "wrong_lane" in low or "internal path" in low:
                clusters["lane_restriction"] = clusters.get("lane_restriction", 0) + 1
            else:
                clusters["general_execution"] = clusters.get("general_execution", 0) + 1

        # 3. Synthesize concise Memory Notes for clusters with >= cluster_threshold occurrences
        new_notes = []
        for cluster_key, count in clusters.items():
            if count >= cluster_threshold:
                note_id = f"note_{cluster_key}"
                rule_text = self._synthesize_rule_for_cluster(cluster_key)
                note = MemoryNote(
                    note_id=note_id,
                    category="trap_warning",
                    target_component=cluster_key,
                    rule_or_trap=rule_text,
                    supporting_evidence_count=count,
                    salience=0.90,  # same decorative placeholder as the field default
                )
                new_notes.append(note)

        # 4. Save/Append Memory Notes
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        existing_notes_keys = set()
        if self.notes_path.exists():
            for line in self.notes_path.read_text(encoding="utf-8").splitlines():
                try:
                    existing_notes_keys.add(json.loads(line).get("note_id"))
                except Exception:
                    pass

        saved_notes_count = 0
        with open(self.notes_path, "a", encoding="utf-8") as f:
            for note in new_notes:
                if note.note_id not in existing_notes_keys:
                    f.write(json.dumps(note.to_dict(), ensure_ascii=False) + "\n")
                    saved_notes_count += 1

        # 5. Update Knowledge Wiki
        wiki_data = self._load_wiki()
        wiki_articles_updated = 0
        for note in new_notes:
            article_key = note.target_component
            wiki_data[article_key] = {
                "summary": note.rule_or_trap,
                "evidence_observations": note.supporting_evidence_count,
                "last_consolidated": time.time()
            }
            wiki_articles_updated += 1
        self._save_wiki(wiki_data)

        # 5.1 Dual-Stream Grounding: Compile synthesized rules into weight-space memory
        weight_implant_count = 0
        try:
            from ..core.wsp_neural_surgeon import WSPNeuralSurgeon
            surgeon = WSPNeuralSurgeon.get_instance()
            for note in new_notes:
                implant_res = self.ground_rule_into_neural_weights(note, surgeon)
                if implant_res:
                    weight_implant_count += 1
        except Exception as e:
            log.debug("Weight grounding bypassed or failed: %s", e)

        # 6. Bounded log maintenance (Pruning older raw transcripts if > 10,000 lines)
        pruned_count = 0
        if len(lines) > 8000:
            compacted_lines = lines[-5000:]
            pruned_count = len(lines) - len(compacted_lines)
            self.ledger_path.write_text("\n".join(compacted_lines) + "\n", encoding="utf-8")
            log.info("AutoDreamer pruned %d stale log lines to prevent bloat.", pruned_count)

        duration = (time.perf_counter() - t0) * 1000.0

        return DreamReport(
            raw_entries_scanned=len(recent_lines),
            clusters_identified=len(clusters),
            new_notes_synthesized=saved_notes_count,
            wiki_articles_updated=wiki_articles_updated,
            stale_entries_pruned=pruned_count,
            duration_ms=round(duration, 3)
        )

    def _synthesize_rule_for_cluster(self, cluster_key: str) -> str:
        rules = {
            "ast_syntax_integrity": "RULE: Generated implementation must be fully valid Python AST without unclosed parens or indentation drift.",
            "test_call_coverage": "RULE: Unit test must explicitly call the primary entrypoint function of the target module.",
            "execution_timeout": "RULE: Keep test execution and inference bounded; avoid infinite loops in reasoning chains.",
            "lane_restriction": "RULE: Periphery tools must never modify internal engine code in core/ or evolution/ directly.",
            "general_execution": "RULE: Ensure all imported dependencies are present in workspace before triggering execution."
        }
        return rules.get(cluster_key, f"RULE: Avoid recurring issues in {cluster_key}.")

    def _load_wiki(self) -> Dict[str, Any]:
        if not self.wiki_path.exists():
            return {}
        try:
            return json.loads(self.wiki_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_wiki(self, data: Dict[str, Any]) -> None:
        self.wiki_path.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def ground_rule_into_neural_weights(
        self,
        note: MemoryNote,
        surgeon: Optional[Any] = None
    ) -> bool:
        """
        Dual-Stream Grounding (Pillar 3):
        Compiles the textual rule note into a rank-1 weight update inside a dedicated neuro-brain matrix.
        Allows microbrains to recall invariant behavior directly through sub-symbolic inference.
        """
        import hashlib
        import numpy as np

        if surgeon is None:
            from ..core.wsp_neural_surgeon import WSPNeuralSurgeon
            surgeon = WSPNeuralSurgeon.get_instance()

        target_file = self.weight_memory_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        dim = 16
        if target_file.exists():
            try:
                data = json.loads(target_file.read_text(encoding="utf-8"))
                raw_w = data.get("weights", np.eye(dim).tolist())
                if isinstance(raw_w, dict):
                    mat = raw_w.get("rule_grounding") or next(
                        (v for v in raw_w.values() if isinstance(v, list)),
                        np.eye(dim).tolist(),
                    )
                    W = np.array(mat, dtype=np.float64)
                else:
                    W = np.array(raw_w, dtype=np.float64)
                if W.shape != (dim, dim):
                    W = np.eye(dim, dtype=np.float64)
            except Exception:
                W = np.eye(dim, dtype=np.float64)
        else:
            W = np.eye(dim, dtype=np.float64)

        # Deterministic pseudo-embedding for key (target component) and value (salience/rule)
        key_hash = hashlib.sha256(note.target_component.encode("utf-8")).digest()
        key_vec = np.array([(b / 128.0) - 1.0 for b in key_hash[:dim]], dtype=np.float64)
        norm_k = np.linalg.norm(key_vec)
        if norm_k > 1e-6:
            key_vec = key_vec / norm_k

        val_hash = hashlib.sha256(note.rule_or_trap.encode("utf-8")).digest()
        target_val = np.array([(b / 128.0) - 1.0 for b in val_hash[:dim]], dtype=np.float64)
        norm_v = np.linalg.norm(target_val)
        if norm_v > 1e-6:
            target_val = target_val / norm_v

        try:
            W_new, implant_res = surgeon.direct_rank1_implant(W, key_vec, target_val)
            out_data = {
                "brain_id": "neuro_weight_space_memory",
                "weights": {"rule_grounding": W_new.tolist()},
                "last_implanted_note": note.note_id,
                "recalled_similarity": implant_res.key_recalled_similarity,
                "timestamp": time.time()
            }
            target_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
            log.info("Implanted memory note %s into neuro_weight_space_memory (sim=%.3f)", note.note_id, implant_res.key_recalled_similarity)
            return True
        except Exception as e:
            log.warning("Failed to implant note into weights: %s", e)
            return False
