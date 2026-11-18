"""Tool Routing v3: Mixture of Experts (MoE) with Dynamic Top-2 Gating.

Implements a deterministic pure-NumPy MoE router (`d_model = 256`) for tool dispatch:
- 6 Specialized Functional Cluster Experts:
    1. ExpertFileOps (read, write, edit, list, patch, diff)
    2. ExpertExecution (shell, command, pytest, unit testing)
    3. ExpertMemoryKnowledge (blackboard, episodic, semantic, genome, ontology)
    4. ExpertGitGovernance (git status, commit, diff, branch, seal)
    5. ExpertMicrobrainMeta (consult, route, incubator, lab, vsa, telemetry)
    6. ExpertConstitutionalReview (sacred sites, kinship, review gate, safety)
- Analytical Dynamic Top-2 Gating:
    Default gating weights `W_gate` are analytically ridge-fitted on bilingual
    keyword/stem tables (with optional real trajectory fitting via `fit_from_traces()`).
    Computes gating logits over experts, selects top-2 experts, and blends
    candidate weights:
        Y = alpha_1 * Expert_top1(X) + alpha_2 * Expert_top2(X)
- Per-Expert Conformal Quantiles:
    Default initialization uses synthetic lexical self-consistency probes
    (`conformal_basis="synthetic_lexical_probes"`); genuine held-out calibration is
    supported via `fit_from_traces(..., holdout_traces=...)` or `evaluate_topk_holdout()`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MoERoutingDecision:
    """Outcome of a dynamic Top-2 MoE routing consultation."""

    selected_tool: str
    secondary_tool: str
    top_experts: List[Tuple[str, float]]
    confidence: float
    conformal_passed: bool
    per_expert_conformal: Dict[str, float]
    routing_rationale: str
    conformal_basis: str = "synthetic_lexical_probes"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_tool": self.selected_tool,
            "secondary_tool": self.secondary_tool,
            "top_experts": [{"expert": e, "weight": round(w, 4)} for e, w in self.top_experts],
            "confidence": round(self.confidence, 4),
            "conformal_passed": self.conformal_passed,
            "per_expert_conformal": {k: round(v, 4) for k, v in self.per_expert_conformal.items()},
            "routing_rationale": self.routing_rationale,
            "conformal_basis": self.conformal_basis,
        }


class ToolRoutingMoE:
    """Mixture of Experts Router for Autonomous Tools."""

    EXPERTS = [
        "ExpertFileOps",
        "ExpertExecution",
        "ExpertMemoryKnowledge",
        "ExpertGitGovernance",
        "ExpertMicrobrainMeta",
        "ExpertConstitutionalReview",
    ]

    EXPERT_TOOLS: Dict[str, List[str]] = {
        "ExpertFileOps": ["repo_read", "repo_write", "read_file", "write_file", "edit_file", "str_replace_editor", "list_dir"],
        "ExpertExecution": ["run_command", "execute_shell", "run_tests", "pytest_runner", "syntax_validator"],
        "ExpertMemoryKnowledge": ["memory_recall", "blackboard_fetch", "blackboard_post", "distill_memory", "knowledge_search"],
        "ExpertGitGovernance": ["git_status", "git_commit", "git_diff", "git_push", "verify_integrity", "generate_seal"],
        "ExpertMicrobrainMeta": ["microbrain_consult", "neuro_route", "neuro_incubator", "microbrain_lab", "vsa_query"],
        "ExpertConstitutionalReview": ["constitutional_check", "review_gate", "sacred_site_check", "ceremonial_reanchor"],
    }

    EXPERT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "ExpertFileOps": (
            "read", "write", "file", "files", "edit", "patch", "dir", "directory", "list",
            "replace", "repository", "repo", "contents",
            "файл", "файлы", "файла", "файлов", "прочитай", "прочитать", "чтение", "запиши",
            "записать", "напиши", "патч", "редактировать", "отредактируй", "исправь", "исправить",
            "директория", "папка", "репозиторий", "код",
        ),
        "ExpertExecution": (
            "run", "test", "tests", "exec", "execute", "cmd", "command", "shell", "pytest",
            "syntax", "validate",
            "запусти", "запустить", "выполни", "выполнить", "команда", "команду", "тест", "тесты",
            "тестов", "проверь", "синтаксис", "оболочка", "терминал",
        ),
        "ExpertMemoryKnowledge": (
            "memory", "blackboard", "recall", "knowledge", "distill", "fetch", "post", "search",
            "память", "памяти", "вспомни", "вспомнить", "знания", "знаний", "найди", "найти", "поиск",
        ),
        "ExpertGitGovernance": (
            "commit", "git", "status", "seal", "branch", "diff", "push", "integrity", "verify",
            "коммит", "закоммить", "ветка", "ветку", "ветки", "статус", "дифф", "пуш", "отправь", "печать",
        ),
        "ExpertMicrobrainMeta": (
            "brain", "incubator", "route", "microbrain", "consult", "neuro", "lab", "vsa",
            "мозг", "мозги", "микромозг", "консультация", "инкубатор", "маршрут",
        ),
        "ExpertConstitutionalReview": (
            "sacred", "bible", "constitution", "constitutional", "kinship", "safety", "review",
            "gate", "totem", "ceremonial",
            "конституция", "священный", "безопасность", "ревью", "проверка", "тотем",
        ),
    }

    # Prefix stems allow bilingual morphological variants (RU/EN inflections) without diluting OOD gate
    EXPERT_STEMS: Dict[str, Tuple[str, ...]] = {
        "ExpertFileOps": ("прочит", "запиш", "напиш", "редакт", "исправ", "файл", "патч", "папк", "директ", "репозитор", "edit", "patch", "write", "read", "file"),
        "ExpertExecution": ("запуст", "выполн", "тест", "команд", "синтакс", "термин", "pytest", "shell", "exec"),
        "ExpertMemoryKnowledge": ("памят", "вспомн", "знан", "поиск", "найд", "recall", "memory", "blackboard", "search"),
        "ExpertGitGovernance": ("ветк", "коммит", "статус", "дифф", "пуш", "git", "commit", "branch", "diff"),
        "ExpertMicrobrainMeta": ("мозг", "микромозг", "инкубат", "маршрут", "консульт", "microbrain", "neuro", "vsa"),
        "ExpertConstitutionalReview": ("конституц", "священ", "безопасн", "ревью", "sacred", "bible", "kinship", "totem"),
    }

    TOOL_KEYWORDS: Dict[str, Dict[str, Tuple[str, ...]]] = {
        "ExpertFileOps": {
            "repo_write": ("write", "patch", "repo_write", "repository", "create", "запиши", "записать", "напиши", "патч", "создай"),
            "edit_file": ("edit", "edit_file", "modify", "fix", "исправь", "исправить", "отредактируй", "редактировать", "измени"),
            "write_file": ("write_file", "save", "overwrite", "dump", "сохрани", "перезапиши"),
            "str_replace_editor": ("str_replace_editor", "replace", "hunk", "замени"),
            "read_file": ("read_file", "open", "cat", "открой", "прочитай"),
            "list_dir": ("list", "list_dir", "dir", "directory", "ls", "список", "папка", "директория"),
            "repo_read": ("read", "repo_read", "inspect", "view", "contents", "посмотри", "изучи", "чтение"),
        },
        "ExpertExecution": {
            "run_command": ("run_command", "command", "cmd", "команда", "команду", "выполни"),
            "execute_shell": ("execute_shell", "shell", "bash", "exec", "оболочка", "терминал"),
            "run_tests": ("run_tests", "tests", "test", "unit", "тест", "тесты", "тестов", "запусти"),
            "pytest_runner": ("pytest", "pytest_runner"),
            "syntax_validator": ("syntax", "syntax_validator", "ast", "синтаксис"),
        },
        "ExpertMemoryKnowledge": {
            "memory_recall": ("memory", "recall", "memory_recall", "память", "памяти", "вспомни"),
            "blackboard_fetch": ("blackboard_fetch", "fetch", "blackboard"),
            "blackboard_post": ("blackboard_post", "post"),
            "distill_memory": ("distill", "distill_memory", "consolidate"),
            "knowledge_search": ("knowledge", "knowledge_search", "search", "знания", "найди", "поиск"),
        },
        "ExpertGitGovernance": {
            "git_status": ("status", "git_status", "branch", "статус", "ветка", "ветку", "ветки"),
            "git_commit": ("commit", "git_commit", "коммит", "закоммить"),
            "git_diff": ("diff", "git_diff", "дифф"),
            "git_push": ("push", "git_push", "пуш", "отправь"),
            "verify_integrity": ("verify", "integrity", "verify_integrity", "целостность"),
            "generate_seal": ("seal", "generate_seal", "печать"),
        },
        "ExpertMicrobrainMeta": {
            "microbrain_consult": ("consult", "microbrain_consult", "microbrain", "консультация", "микромозг"),
            "neuro_route": ("route", "neuro_route", "neuro", "маршрут"),
            "neuro_incubator": ("incubator", "neuro_incubator", "grow", "инкубатор"),
            "microbrain_lab": ("lab", "microbrain_lab", "experiment"),
            "vsa_query": ("vsa", "vsa_query", "holographic"),
        },
        "ExpertConstitutionalReview": {
            "constitutional_check": ("constitution", "constitutional", "constitutional_check", "конституция"),
            "review_gate": ("review", "gate", "review_gate", "safety", "ревью", "безопасность"),
            "sacred_site_check": ("sacred", "bible", "kinship", "sacred_site_check", "священный"),
            "ceremonial_reanchor": ("ceremonial", "totem", "reanchor", "ceremonial_reanchor", "тотем"),
        },
    }

    def __init__(self, d_model: int = 256, seed: int = 42):
        self.d_model = d_model
        self.rng = np.random.default_rng(seed)
        n_experts = len(self.EXPERTS)

        # Deterministic char-trigram bilingual prototype matrix over expert vocabularies
        self.W_trigram_proto = np.zeros((self.d_model, n_experts), dtype=np.float32)
        for idx, expert_name in enumerate(self.EXPERTS):
            proto_tokens = list(self.EXPERT_KEYWORDS.get(expert_name, ())) + list(self.EXPERT_STEMS.get(expert_name, ()))
            self.W_trigram_proto[:, idx] = self._embed_context(proto_tokens)

        # Analytical ridge-fitted gating network (closed-form L2 normal equations, zero random noise)
        self.W_gate = self._fit_ridge_gate(l2_reg=0.25)
        self.w_gate_fitted = True

        # Calibration provenance & measured error counters (no fabricated bounds without verified episodes)
        self.calibration_basis: str = "synthetic_lexical_probes"
        self.verified_episodes: int = 0
        self.verified_errors: int = 0
        self.calibration_alpha: float = 0.05
        self.recalibrated_from_journal: bool = False
        self.calibration_sample_id: Optional[str] = None
        self.error_metric: str = "unmeasured"
        self.last_topk_eval_report: Optional[Dict[str, Any]] = None

        # Synthetic per-expert self-consistency probes (n = 22 per expert, derived from keywords/stems)
        self.per_expert_quantiles: Dict[str, float] = {}
        self.per_expert_coverage: Dict[str, float] = {}
        self._calibrate_expert_quantiles(alpha=0.05)

    @property
    def conformal_basis(self) -> str:
        """Alias synchronized with `calibration_basis` for uniform inspection."""
        return self.calibration_basis

    @conformal_basis.setter
    def conformal_basis(self, value: str) -> None:
        self.calibration_basis = str(value)

    def calibration_report(self) -> Dict[str, Any]:
        """Return honest calibration provenance and Clopper-Pearson bounds (None unless verified)."""
        upper_bound: Optional[float] = None
        upper_bound_95: Optional[float] = None
        if self.verified_episodes > 0:
            from .analytical_micro_transformer import AnalyticalMicroTransformer

            upper_bound = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(
                self.verified_errors,
                self.verified_episodes,
                alpha=float(self.calibration_alpha),
            )
            upper_bound_95 = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(
                self.verified_errors,
                self.verified_episodes,
                alpha=0.05,
            )

        err_metric = self.error_metric if self.verified_episodes > 0 else "unmeasured"
        if self.verified_episodes > 0 and err_metric == "unmeasured":
            err_metric = "top1_routing_mismatch"

        field_certified = bool(
            self.verified_episodes > 0
            and self.calibration_basis in ("measured_holdout", "recalibrated_journal")
        )

        rep: Dict[str, Any] = {
            "calibration_basis": self.calibration_basis,
            "conformal_basis": self.calibration_basis,
            "field_exchangeability_certified": field_certified,
            "recalibrated": bool(self.recalibrated_from_journal),
            "n_calibration_samples": len(self.per_expert_quantiles) * 22,
            "verified_episodes": int(self.verified_episodes),
            "verified_errors": int(self.verified_errors),
            "error_metric": err_metric,
            "confidence_level": round(1.0 - float(self.calibration_alpha), 4),
            "alpha": float(self.calibration_alpha),
            "sample_id": self.calibration_sample_id,
            "upper_error_bound": upper_bound,
            "upper_error_bound_95": upper_bound_95,
            "statistical_confidence_status": "verified" if self.verified_episodes > 0 else "unknown",
        }
        if self.last_topk_eval_report is not None:
            rep["topk_evaluation"] = dict(self.last_topk_eval_report)
        return rep

    def _fit_ridge_gate(self, l2_reg: float = 0.25) -> np.ndarray:
        """Fit gating projection W_gate analytically via L2-regularized ridge regression."""
        n_experts = len(self.EXPERTS)
        rows: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        for e_idx, expert_name in enumerate(self.EXPERTS):
            y_vec = np.zeros(n_experts, dtype=np.float64)
            y_vec[e_idx] = 1.0
            kws = list(self.EXPERT_KEYWORDS.get(expert_name, ()))
            stems = list(self.EXPERT_STEMS.get(expert_name, ()))
            for i in range(len(kws)):
                sample_toks = [kws[i], kws[(i + 1) % len(kws)]]
                rows.append(self._embed_context(sample_toks))
                targets.append(y_vec)
            for s in stems:
                rows.append(self._embed_context([s]))
                targets.append(y_vec)
            for tool_name, t_kws in self.TOOL_KEYWORDS.get(expert_name, {}).items():
                rows.append(self._embed_context([tool_name, *list(t_kws[:4])]))
                targets.append(y_vec)

        X = np.stack(rows, axis=0).astype(np.float64)
        Y = np.stack(targets, axis=0).astype(np.float64)
        gram = X.T @ X + float(l2_reg) * np.eye(self.d_model, dtype=np.float64)
        W = np.linalg.solve(gram, X.T @ Y)
        return W.astype(np.float32)

    def _parse_trace_rows(
        self,
        traces: Sequence[Dict[str, Any]],
        tool_to_expert: Dict[str, int],
    ) -> List[Tuple[List[str], int, np.ndarray]]:
        """Parse `(query, tool/expert)` trace entries into tokenized vectors and expert indices."""
        parsed: List[Tuple[List[str], int, np.ndarray]] = []
        for tr in traces:
            if not isinstance(tr, dict):
                continue
            q = str(tr.get("query") or tr.get("task_prompt") or "").strip()
            tool = str(tr.get("tool") or tr.get("selected_tool") or tr.get("expected_tool") or "").strip()
            e_idx = tool_to_expert.get(tool)
            if e_idx is None and tr.get("expert") in self.EXPERTS:
                e_idx = self.EXPERTS.index(str(tr["expert"]))
            if not q or e_idx is None:
                continue
            toks = [t for t in re.findall(r"(?u)\b\w+\b", q.lower()) if len(t) > 1]
            if not toks:
                continue
            parsed.append((toks, int(e_idx), self._embed_context(toks)))
        return parsed

    def fit_from_traces(
        self,
        traces: Sequence[Dict[str, Any]],
        l2_reg: float = 0.25,
        blend_with_prior: float = 0.35,
        holdout_traces: Optional[Sequence[Dict[str, Any]]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Fit gating weights W_gate from real agent `(query, tool)` execution traces.

        Solves closed-form L2 ridge regression on recorded agent trajectories and
        calibrates per-expert conformal quantiles from `holdout_traces` when supplied
        (otherwise falls back to `synthetic_lexical_probes`).
        """
        n_experts = len(self.EXPERTS)
        tool_to_expert: Dict[str, int] = {}
        for e_idx, expert_name in enumerate(self.EXPERTS):
            for t_name in self.EXPERT_TOOLS.get(expert_name, ()):
                tool_to_expert[t_name] = e_idx

        parsed_train = self._parse_trace_rows(traces, tool_to_expert)
        if not parsed_train:
            return {"n_traces": 0, "experts_updated": [], "train_top1_acc": 0.0}

        rows: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        experts_seen: set[str] = set()
        for _toks, e_idx, ctx_vec in parsed_train:
            rows.append(ctx_vec)
            y_vec = np.zeros(n_experts, dtype=np.float64)
            y_vec[e_idx] = 1.0
            targets.append(y_vec)
            experts_seen.add(self.EXPERTS[e_idx])

        X = np.stack(rows, axis=0).astype(np.float64)
        Y = np.stack(targets, axis=0).astype(np.float64)
        gram = X.T @ X + float(l2_reg) * np.eye(self.d_model, dtype=np.float64)
        W_traces = np.linalg.solve(gram, X.T @ Y).astype(np.float32)
        blend = max(0.0, min(1.0, float(blend_with_prior)))
        self.W_gate = ((1.0 - blend) * W_traces + blend * self.W_gate).astype(np.float32)

        # Initialize baseline quantiles at caller alpha before optional held-out override
        self._calibrate_expert_quantiles(alpha=alpha)
        holdout_summary: Optional[Dict[str, Any]] = None
        if holdout_traces is not None:
            parsed_ho = self._parse_trace_rows(holdout_traces, tool_to_expert)
            per_expert_scores: Dict[str, List[float]] = {}
            holdout_n = 0
            holdout_errors = 0
            for toks, e_idx, ctx_vec in parsed_ho:
                logits, _ = self._compute_raw_gate_logits(toks, ctx_vec)
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)
                exp_name = self.EXPERTS[e_idx]
                per_expert_scores.setdefault(exp_name, []).append(float(1.0 - probs[e_idx]))
                holdout_n += 1
                if int(np.argmax(probs)) != e_idx:
                    holdout_errors += 1
            if holdout_n > 0:
                for exp_name, scores in per_expert_scores.items():
                    n_e = len(scores)
                    q_level = min(1.0, math.ceil((n_e + 1) * (1.0 - float(alpha))) / float(n_e))
                    q_hat = float(np.quantile(scores, q_level))
                    self.per_expert_quantiles[exp_name] = round(q_hat, 4)
                    covered = sum(1 for sc in scores if sc <= q_hat + 1e-7)
                    self.per_expert_coverage[exp_name] = round(covered / float(n_e), 4)
                self.verified_episodes = holdout_n
                self.verified_errors = holdout_errors
                self.calibration_alpha = float(alpha)
                self.error_metric = "holdout_expert_routing_mismatch"
                self.calibration_sample_id = f"holdout_traces_{holdout_n}"
                self.calibration_basis = "measured_holdout" if holdout_n >= 19 else "partial_holdout_traces"
                holdout_summary = {
                    "n_holdout": holdout_n,
                    "errors": holdout_errors,
                    "experts_calibrated": sorted(per_expert_scores.keys()),
                    "calibration_basis": self.calibration_basis,
                }
            else:
                self.calibration_basis = "synthetic_lexical_probes"
                holdout_summary = {
                    "n_holdout": 0,
                    "errors": 0,
                    "experts_calibrated": [],
                    "calibration_basis": self.calibration_basis,
                }
        else:
            self.calibration_basis = "synthetic_lexical_probes"

        preds = np.argmax(X @ self.W_gate, axis=1)
        truth = np.argmax(Y, axis=1)
        acc = float(np.mean(preds == truth))
        out: Dict[str, Any] = {
            "n_traces": len(rows),
            "experts_updated": sorted(experts_seen),
            "train_top1_acc": round(acc, 4),
            "calibration_basis": self.calibration_basis,
        }
        if holdout_summary is not None:
            out["holdout_calibration"] = holdout_summary
        return out

    def _compute_raw_gate_logits(self, tokens: List[str], ctx_vec: np.ndarray) -> Tuple[np.ndarray, int]:
        """Shared deterministic gating logits computation used by calibration and inference."""
        proto_sims = np.dot(ctx_vec, self.W_trigram_proto)
        logits = 2.0 * np.dot(ctx_vec, self.W_gate) + 1.5 * proto_sims
        signal_hits = 0
        tok_set = set(tokens)
        for idx, expert_name in enumerate(self.EXPERTS):
            kws = self.EXPERT_KEYWORDS.get(expert_name, ())
            stems = self.EXPERT_STEMS.get(expert_name, ())
            exact_hits = sum(1 for w in tok_set if w in kws)
            stem_hits = sum(1 for w in tok_set if w not in kws and any(w.startswith(s) for s in stems))
            hits = exact_hits + stem_hits
            if hits > 0:
                boost = 2.5 if expert_name == "ExpertConstitutionalReview" else 2.0
                logits[idx] += boost + 0.25 * (hits - 1)
                signal_hits += hits
        return logits, signal_hits

    def _calibrate_expert_quantiles(self, alpha: float = 0.05) -> None:
        """Compute per-expert synthetic self-consistency quantiles on 22 keyword/stem probes.

        Note: these probes are synthesized from `EXPERT_KEYWORDS` and `EXPERT_STEMS`,
        NOT held-out field data; `per_expert_coverage` measures in-probe consistency only
        and does not certify external generalization or field exchangeability.
        """
        wrappers = (
            ("please", "now"), ("check", "target"), ("agent", "step"),
            ("нужно", "сделать"), ("быстро", "проверка"), ("task", "context"),
            ("current", "workspace"), ("repository", "mode"), ("routine", "ops"),
            ("local", "verify"), ("module", "helper"),
        )
        other_decoy = ("output", "summary", "branch", "report", "memory", "config")
        for e_idx, expert_name in enumerate(self.EXPERTS):
            kws = list(self.EXPERT_KEYWORDS[expert_name])
            stems = list(self.EXPERT_STEMS[expert_name])
            scores: List[float] = []
            for p_idx in range(22):
                w1 = kws[p_idx % len(kws)]
                w2 = stems[(p_idx * 3 + 1) % len(stems)]
                pad_a, pad_b = wrappers[p_idx % len(wrappers)]
                probe_toks = [pad_a, w1, w2, pad_b]
                if p_idx % 5 == 4:
                    # Include mild cross-domain nuance on 20% of calibration probes
                    probe_toks.append(other_decoy[p_idx % len(other_decoy)])
                ctx_vec = self._embed_context(probe_toks)
                logits, _ = self._compute_raw_gate_logits(probe_toks, ctx_vec)
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)
                scores.append(float(1.0 - probs[e_idx]))

            n = len(scores)
            q_level = min(1.0, math.ceil((n + 1) * (1.0 - float(alpha))) / float(n))
            q_hat = float(np.quantile(scores, q_level))
            self.per_expert_quantiles[expert_name] = round(q_hat, 4)
            covered = sum(1 for s in scores if s <= q_hat + 1e-7)
            self.per_expert_coverage[expert_name] = round(covered / float(n), 4)

    @staticmethod
    def _char_trigrams(token: str) -> List[str]:
        if len(token) < 3:
            return [token]
        padded = f"^{token}$"
        return [padded[i : i + 3] for i in range(len(padded) - 2)]

    def _embed_context(self, context_tokens: List[str]) -> np.ndarray:
        """Derive a deterministic SHA-256 d_model vector combining whole tokens and char 3-grams."""
        vec = np.zeros(self.d_model, dtype=np.float32)
        for tok in context_tokens:
            s_tok = str(tok).lower()
            digest = hashlib.sha256(s_tok.encode("utf-8")).digest()
            h = int.from_bytes(digest[:4], "little") % self.d_model
            vec[h] += 1.0
            for tri in self._char_trigrams(s_tok):
                tdigest = hashlib.sha256(tri.encode("utf-8")).digest()
                th = int.from_bytes(tdigest[:4], "little") % self.d_model
                vec[th] += 0.35
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def _score_candidate_tools(
        self,
        candidates: Sequence[str],
        tokens: List[str],
        expert: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Score a candidate tool list against query tokens and morphological segments."""
        tok_set = set(tokens)
        mutation_verbs = {"write", "patch", "edit", "create", "modify", "save", "overwrite", "replace", "запиши", "напиши", "исправь", "патч", "отредактируй"}
        has_mutation = bool(tok_set & mutation_verbs) or any(any(t.startswith(s) for s in ("запиш", "напиш", "исправ", "редакт", "патч")) for t in tok_set)

        expert_map = self.TOOL_KEYWORDS.get(expert, {}) if expert else {}
        scored: List[Tuple[str, float]] = []

        for tool_name in candidates:
            kws = set(expert_map.get(tool_name, ()))
            if not kws:
                for exp_dict in self.TOOL_KEYWORDS.values():
                    if tool_name in exp_dict:
                        kws = set(exp_dict[tool_name])
                        break

            # Include splits of custom tool names (e.g. `custom_db_query` -> `custom`, `db`, `query`)
            name_parts = {p for p in re.split(r"[^a-zA-Z0-9а-яА-ЯёЁ]+", tool_name.lower()) if len(p) > 1}
            all_kws = kws | name_parts | {tool_name.lower()}

            score = 0.0
            for kw in all_kws:
                if kw in tok_set:
                    score += 3.0 if kw == tool_name.lower() else 1.5
                elif len(kw) >= 4 and any(t.startswith(kw[:4]) or kw.startswith(t[:4]) for t in tok_set if len(t) >= 4):
                    score += 0.75

            if has_mutation and tool_name in ("repo_write", "edit_file", "write_file", "str_replace_editor"):
                score += 1.0

            scored.append((tool_name, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def _select_tool_for_expert(
        self,
        expert: str,
        tokens: List[str],
        available_set: Optional[set] = None,
    ) -> str:
        """Score every permitted tool within the selected expert so all 32 tools (or custom subsets) are reachable."""
        selected, _ = self._select_tool_with_score(expert, tokens, available_set=available_set)
        return selected

    def _select_tool_with_score(
        self,
        expert: str,
        tokens: List[str],
        available_set: Optional[set] = None,
    ) -> Tuple[str, float]:
        tools = list(self.EXPERT_TOOLS[expert])
        if available_set is not None:
            tools = [t for t in tools if t in available_set]
            if not tools:
                tools = sorted(available_set)
        if not tools:
            return ("<none>" if available_set is not None else self.EXPERT_TOOLS[expert][0]), 0.0
        if not tokens:
            return tools[0], 0.0
        ranked = self._score_candidate_tools(tools, tokens, expert=expert)
        return ranked[0][0], float(ranked[0][1])

    def route(
        self,
        task_description: str,
        context_tags: Optional[List[str]] = None,
        available_tools: Optional[Sequence[str]] = None,
    ) -> MoERoutingDecision:
        """Execute deterministic dynamic Top-2 MoE routing constrained to `available_tools` if provided."""
        context_tags = context_tags or []
        tokens = [t for t in re.findall(r"(?u)\b\w+\b", str(task_description or "").lower()) if len(t) > 1]
        tokens.extend(t.lower() for t in context_tags if t)
        ctx_vec = self._embed_context(tokens)

        available_list = list(dict.fromkeys(str(t) for t in available_tools if t)) if available_tools is not None else None
        available_set = set(available_list) if available_list is not None else None

        if available_list is not None and len(available_list) == 0:
            exp1, exp2 = self.EXPERTS[0], self.EXPERTS[1]
            return MoERoutingDecision(
                selected_tool="<none>",
                secondary_tool="<none>",
                top_experts=[(exp1, 0.5), (exp2, 0.5)],
                confidence=0.0,
                conformal_passed=False,
                per_expert_conformal={
                    exp1: self.per_expert_quantiles[exp1],
                    exp2: self.per_expert_quantiles[exp2],
                },
                routing_rationale="No registered tools in live agent catalog; escalating.",
                conformal_basis=self.calibration_basis,
            )

        # Compute analytical ridge-fitted gating + trigram prototype logits
        logits, signal_hits = self._compute_raw_gate_logits(tokens, ctx_vec)
        tok_set = set(tokens)

        if available_set is not None:
            for idx, expert_name in enumerate(self.EXPERTS):
                if not any(t in available_set for t in self.EXPERT_TOOLS[expert_name]):
                    logits[idx] -= 5.0

        # Also count direct lexical matches against custom available_tools names if provided
        if available_set is not None and signal_hits == 0:
            for tname in available_set:
                parts = {p for p in re.split(r"[^a-zA-Z0-9а-яА-ЯёЁ]+", tname.lower()) if len(p) > 1}
                if tok_set & parts:
                    signal_hits += len(tok_set & parts)

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        ranked_indices = sorted(
            range(len(self.EXPERTS)),
            key=lambda i: (-float(probs[i]), self.EXPERTS[i]),
        )
        top1_idx, top2_idx = ranked_indices[0], ranked_indices[1]

        expert_1 = self.EXPERTS[top1_idx]
        expert_2 = self.EXPERTS[top2_idx]

        weight_1 = float(probs[top1_idx] / (probs[top1_idx] + probs[top2_idx]))
        weight_2 = float(probs[top2_idx] / (probs[top1_idx] + probs[top2_idx]))

        tool_1, tool_1_score = self._select_tool_with_score(expert_1, tokens, available_set=available_set)
        tool_2, _ = self._select_tool_with_score(expert_2, tokens, available_set=available_set)
        if available_set is not None and tool_2 == tool_1 and len(available_list or []) > 1:
            ranked_all = self._score_candidate_tools(available_list, tokens)
            for cand, _ in ranked_all:
                if cand != tool_1:
                    tool_2 = cand
                    break

        q1 = self.per_expert_quantiles[expert_1]
        q2 = self.per_expert_quantiles[expert_2]
        nonconformity_top1 = 1.0 - float(probs[top1_idx])
        expert_has_registered = (
            available_set is None
            or any(t in available_set for t in self.EXPERT_TOOLS[expert_1])
            or tool_1_score > 0.0
        )
        registry_grounded = (available_set is None) or (tool_1 in available_set and expert_has_registered and tool_1_score > 0.0)
        conformal_passed = bool(signal_hits > 0 and nonconformity_top1 <= (q1 + 1e-6) and registry_grounded)

        conf = float(probs[top1_idx])
        rationale = f"MoE Top-2: {expert_1} ({weight_1:.1%}) & {expert_2} ({weight_2:.1%}), signals={signal_hits}"

        return MoERoutingDecision(
            selected_tool=tool_1,
            secondary_tool=tool_2,
            top_experts=[(expert_1, weight_1), (expert_2, weight_2)],
            confidence=conf,
            conformal_passed=conformal_passed,
            per_expert_conformal={expert_1: q1, expert_2: q2},
            routing_rationale=rationale,
            conformal_basis=self.calibration_basis,
        )

    route_LockFree = route

    def _semantic_description_scores(
        self,
        task_description: str,
        catalog: Sequence[str],
        tool_descriptions: Optional[Dict[str, str]] = None,
        embed_fn: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Compute semantic similarity between task_description and tool descriptions.

        Uses caller-supplied `embed_fn(texts) -> (N, D)` when provided, otherwise falls
        back to a zero-dependency pure-NumPy char-trigram + lexical cosine projection.
        """
        if not catalog or (not tool_descriptions and embed_fn is None):
            return {}
        texts = [
            f"{name.replace('_', ' ')} {str((tool_descriptions or {}).get(name, '')).strip()}".strip()
            for name in catalog
        ]
        if embed_fn is not None:
            vecs = np.asarray(embed_fn([task_description, *texts]), dtype=np.float64)
            q_vec = vecs[0]
            q_norm = float(np.linalg.norm(q_vec)) + 1e-12
            doc_vecs = vecs[1:]
            doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-12
            sims = (doc_vecs @ q_vec) / (doc_norms * q_norm)
            return {name: float(sims[idx]) for idx, name in enumerate(catalog)}

        q_tokens = [t for t in re.findall(r"(?u)\b\w+\b", str(task_description or "").lower()) if len(t) > 1]
        q_emb = self._embed_context(q_tokens)
        q_norm = float(np.linalg.norm(q_emb)) + 1e-12
        scores: Dict[str, float] = {}
        for name, txt in zip(catalog, texts):
            d_tokens = [t for t in re.findall(r"(?u)\b\w+\b", txt.lower()) if len(t) > 1]
            d_emb = self._embed_context(d_tokens)
            d_norm = float(np.linalg.norm(d_emb)) + 1e-12
            scores[name] = float(np.dot(q_emb, d_emb) / (q_norm * d_norm))
        return scores

    def narrow_tool_surface(
        self,
        task_description: str,
        available_tools: Optional[Sequence[str]] = None,
        top_k: int = 5,
        context_tags: Optional[List[str]] = None,
        tool_descriptions: Optional[Dict[str, str]] = None,
        embed_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """MCP-Zero style hierarchical Top-K tool surface narrowing before LLM invocation.

        Returns Top-K candidate tools when conformal gating succeeds; escalates the full
        catalog unchanged on out-of-distribution / ambiguous queries. Supports optional
        semantic `tool_descriptions` and dense `embed_fn` hybrid re-ranking.
        """
        k = max(1, int(top_k))
        all_catalog = (
            list(dict.fromkeys(str(t) for t in available_tools if t))
            if available_tools is not None
            else [t for tools in self.EXPERT_TOOLS.values() for t in tools]
        )
        decision = self.route(task_description, context_tags=context_tags, available_tools=all_catalog if available_tools is not None else None)
        if not decision.conformal_passed:
            if embed_fn is not None and all_catalog:
                sem = self._semantic_description_scores(
                    task_description,
                    all_catalog,
                    tool_descriptions=tool_descriptions,
                    embed_fn=embed_fn,
                )
                if sem and max(sem.values()) >= 0.70:
                    ranked_sem = sorted(sem.items(), key=lambda x: -x[1])
                    narrowed_sem = [name for name, _ in ranked_sem[:k]]
                    reduction = round(1.0 - (len(narrowed_sem) / max(1, len(all_catalog))), 4)
                    return {
                        "mode": "narrowed",
                        "candidate_tools": narrowed_sem,
                        "selected_tool": narrowed_sem[0],
                        "conformal_passed": False,
                        "confidence": round(max(sem.values()), 4),
                        "reduction_ratio": max(0.0, reduction),
                        "semantic_hybrid": True,
                    }
            return {
                "mode": "escalate_full_catalog",
                "candidate_tools": all_catalog,
                "selected_tool": decision.selected_tool,
                "conformal_passed": False,
                "confidence": round(decision.confidence, 4),
                "reduction_ratio": 0.0,
            }

        tokens = [t for t in re.findall(r"(?u)\b\w+\b", str(task_description or "").lower()) if len(t) > 1]
        if context_tags:
            tokens.extend(t.lower() for t in context_tags if t)

        top_expert = decision.top_experts[0][0]
        ranked = self._score_candidate_tools(all_catalog, tokens, expert=top_expert)
        if tool_descriptions or embed_fn is not None:
            sem = self._semantic_description_scores(
                task_description,
                all_catalog,
                tool_descriptions=tool_descriptions,
                embed_fn=embed_fn,
            )
            ranked = sorted(
                ((name, score + 3.0 * sem.get(name, 0.0)) for name, score in ranked),
                key=lambda x: -x[1],
            )
            narrowed: List[str] = []
            for tool_name, _ in ranked:
                if tool_name not in narrowed and len(narrowed) < k:
                    narrowed.append(tool_name)
        else:
            narrowed = [decision.selected_tool]
            for tool_name, _ in ranked:
                if tool_name not in narrowed and len(narrowed) < k:
                    narrowed.append(tool_name)
            if decision.secondary_tool in all_catalog and decision.secondary_tool not in narrowed and len(narrowed) < k:
                narrowed.append(decision.secondary_tool)

        reduction = round(1.0 - (len(narrowed) / max(1, len(all_catalog))), 4)
        return {
            "mode": "narrowed",
            "candidate_tools": narrowed,
            "selected_tool": narrowed[0] if narrowed else decision.selected_tool,
            "conformal_passed": True,
            "confidence": round(decision.confidence, 4),
            "reduction_ratio": max(0.0, reduction),
            "semantic_hybrid": bool(tool_descriptions or embed_fn is not None),
        }

    def rank_tools(
        self,
        task_description: str,
        available_tools: Optional[Sequence[str]] = None,
        top_k: int = 4,
        context_tags: Optional[List[str]] = None,
        tool_descriptions: Optional[Dict[str, str]] = None,
        embed_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Advisory non-mutating tool ranker (never removes tools or authorizes execution)."""
        all_catalog = (
            list(dict.fromkeys(str(t) for t in available_tools if t))
            if available_tools is not None
            else [t for tools in self.EXPERT_TOOLS.values() for t in tools]
        )
        decision = self.route(
            task_description,
            context_tags=context_tags,
            available_tools=all_catalog if available_tools is not None else None,
        )
        tokens = [t for t in re.findall(r"(?u)\b\w+\b", str(task_description or "").lower()) if len(t) > 1]
        if context_tags:
            tokens.extend(t.lower() for t in context_tags if t)
        top_expert = decision.top_experts[0][0] if decision.top_experts else None
        lexical_ranked = self._score_candidate_tools(all_catalog, tokens, expert=top_expert)
        sem_scores = (
            self._semantic_description_scores(
                task_description,
                all_catalog,
                tool_descriptions=tool_descriptions,
                embed_fn=embed_fn,
            )
            if (tool_descriptions or embed_fn is not None)
            else {}
        )
        combined = sorted(
            ((name, round(float(score) + 3.0 * float(sem_scores.get(name, 0.0)), 4)) for name, score in lexical_ranked),
            key=lambda kv: (-kv[1], kv[0]),
        )
        ranked_tools = [
            {"tool": name, "score": sc, "rank": idx + 1}
            for idx, (name, sc) in enumerate(combined)
        ]
        nar = self.narrow_tool_surface(
            task_description,
            available_tools=all_catalog,
            top_k=top_k,
            context_tags=context_tags,
            tool_descriptions=tool_descriptions,
            embed_fn=embed_fn,
        )
        return {
            "ranked_tools": ranked_tools,
            "recommended_top_k": list(nar["candidate_tools"][: max(1, int(top_k))]) if nar["mode"] == "narrowed" else [r["tool"] for r in ranked_tools[: max(1, int(top_k))]],
            "should_escalate_full_catalog": nar["mode"] != "narrowed",
            "confidence": round(float(decision.confidence), 4),
            "authorizes_execution": False,
            "mutates_catalog": False,
            "calibration_report": self.calibration_report(),
        }

    def evaluate_topk_holdout(
        self,
        traces: Sequence[Dict[str, Any]],
        top_k: int = 4,
        available_tools: Optional[Sequence[str]] = None,
        tool_descriptions: Optional[Dict[str, str]] = None,
        alpha: float = 0.05,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Measure actual Top-K tool retention and Top-1 mismatch rates on labeled holdout traces.

        Supports both single-label (`tool`/`expected_tool`/`selected_tool`) and multi-label
        (`acceptable_tools`) traces and updates `calibration_report()` with genuine measured counts.
        """
        from .analytical_micro_transformer import AnalyticalMicroTransformer

        eff_alpha = float(alpha)
        k = max(1, int(top_k))
        n_eval = 0
        top1_mismatches = 0
        topk_drops = 0
        escalated_count = 0
        narrowed_count = 0
        narrowed_drops = 0

        for tr in traces:
            if not isinstance(tr, dict):
                continue
            q = str(tr.get("query") or tr.get("task_prompt") or "").strip()
            acc_raw = tr.get("acceptable_tools")
            if acc_raw is not None:
                acceptable = {str(t).strip() for t in acc_raw if str(t).strip()}
            else:
                single = str(tr.get("expected_tool") or tr.get("tool") or tr.get("selected_tool") or "").strip()
                acceptable = {single} if single else set()
            if not q or not acceptable:
                continue

            nar = self.narrow_tool_surface(
                q,
                available_tools=available_tools,
                top_k=k,
                tool_descriptions=tool_descriptions,
            )
            cands = list(nar.get("candidate_tools", []))
            top1_tool = str(nar.get("selected_tool") or (cands[0] if cands else ""))
            n_eval += 1
            if top1_tool not in acceptable:
                top1_mismatches += 1
            missed_topk = not any(t in cands for t in acceptable)
            if missed_topk:
                topk_drops += 1
            if nar.get("mode") == "narrowed":
                narrowed_count += 1
                if missed_topk:
                    narrowed_drops += 1
            else:
                escalated_count += 1

        resolved_sid = sample_id or f"holdout_topk_{n_eval}"
        if n_eval == 0:
            self.verified_episodes = 0
            self.verified_errors = 0
            self.error_metric = "unmeasured"
            self.calibration_sample_id = resolved_sid
            report = {
                "n_evaluated": 0,
                "top_k": k,
                "top1_mismatches": 0,
                "top1_mismatch_cp95_upper": None,
                "topk_drops": 0,
                "topk_drop_rate": None,
                "topk_drop_cp95_upper": None,
                "topk_drop_bound": None,
                "escalated_count": 0,
                "escalation_rate": None,
                "narrowed_count": 0,
                "narrowed_drops": 0,
                "narrowed_topk_recall": None,
                "alpha": eff_alpha,
                "confidence_level": round(1.0 - eff_alpha, 4),
                "sample_id": resolved_sid,
            }
            self.last_topk_eval_report = report
            return report

        top1_cp95 = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(top1_mismatches, n_eval, alpha=0.05)
        topk_cp95 = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(topk_drops, n_eval, alpha=0.05)
        topk_alpha_bound = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(topk_drops, n_eval, alpha=eff_alpha)
        narrowed_recall = round((narrowed_count - narrowed_drops) / float(narrowed_count), 4) if narrowed_count > 0 else None

        report = {
            "n_evaluated": n_eval,
            "top_k": k,
            "top1_mismatches": top1_mismatches,
            "top1_mismatch_cp95_upper": top1_cp95,
            "topk_drops": topk_drops,
            "topk_drop_rate": round(topk_drops / float(n_eval), 4),
            "topk_drop_cp95_upper": topk_cp95,
            "topk_drop_bound": topk_alpha_bound,
            "escalated_count": escalated_count,
            "escalation_rate": round(escalated_count / float(n_eval), 4),
            "narrowed_count": narrowed_count,
            "narrowed_drops": narrowed_drops,
            "narrowed_topk_recall": narrowed_recall,
            "alpha": eff_alpha,
            "confidence_level": round(1.0 - eff_alpha, 4),
            "sample_id": resolved_sid,
        }
        self.verified_episodes = n_eval
        self.verified_errors = topk_drops
        self.calibration_alpha = eff_alpha
        self.calibration_basis = "measured_holdout"
        self.recalibrated_from_journal = False
        self.calibration_sample_id = resolved_sid
        self.error_metric = f"topk_tool_drop_k{k}"
        self.last_topk_eval_report = dict(report)
        return report


class SessionToolSurface:
    """Prompt-cache-friendly sticky tool surface manager for multi-turn agent sessions.

    Narrows tools once per session (unless refreshed or monotonically expanded to full on
    failure) so LLM prompt prefixes stay cacheable across turns without prefix invalidation.
    """

    def __init__(
        self,
        available_tools: Sequence[str],
        top_k: int = 4,
        tool_descriptions: Optional[Dict[str, str]] = None,
        embed_fn: Optional[Any] = None,
        moe: Optional[ToolRoutingMoE] = None,
    ):
        self.full_catalog: List[str] = list(dict.fromkeys(str(t) for t in available_tools if t))
        self.top_k: int = max(1, int(top_k))
        self.tool_descriptions: Optional[Dict[str, str]] = tool_descriptions
        self.embed_fn: Optional[Any] = embed_fn
        self.moe: ToolRoutingMoE = moe or get_tool_routing_moe()
        self.active_tools: Optional[List[str]] = None
        self.expanded_to_full: bool = False
        self.turns: int = 0
        self.cache_hits: int = 0
        self.cache_invalidations: int = 0

    def select_for_turn(
        self,
        task_description: str,
        context_tags: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        self.turns += 1
        if self.active_tools is not None and not force_refresh:
            self.cache_hits += 1
            return {
                "mode": "sticky_reuse",
                "candidate_tools": list(self.active_tools),
                "cache_prefix_preserved": True,
                "expanded_to_full": self.expanded_to_full,
                "reduction_ratio": round(max(0.0, 1.0 - (len(self.active_tools) / float(max(1, len(self.full_catalog))))), 4),
            }

        prev = list(self.active_tools) if self.active_tools is not None else None
        nar = self.moe.narrow_tool_surface(
            task_description,
            available_tools=self.full_catalog,
            top_k=self.top_k,
            context_tags=context_tags,
            tool_descriptions=self.tool_descriptions,
            embed_fn=self.embed_fn,
        )
        new_cands = list(nar["candidate_tools"])
        self.active_tools = new_cands
        self.expanded_to_full = len(new_cands) >= len(self.full_catalog)
        if prev is not None and new_cands != prev:
            self.cache_invalidations += 1
        return {
            "mode": str(nar.get("mode", "narrowed")),
            "candidate_tools": list(self.active_tools),
            "cache_prefix_preserved": prev is None or new_cands == prev,
            "expanded_to_full": self.expanded_to_full,
            "reduction_ratio": float(nar.get("reduction_ratio", 0.0)),
        }

    def expand_to_full(self) -> List[str]:
        if self.active_tools != self.full_catalog:
            if self.active_tools is not None:
                self.cache_invalidations += 1
            self.active_tools = list(self.full_catalog)
        self.expanded_to_full = True
        return list(self.active_tools)

    def stats(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "cache_hits": self.cache_hits,
            "cache_invalidations": self.cache_invalidations,
            "active_tool_count": len(self.active_tools) if self.active_tools is not None else len(self.full_catalog),
            "full_tool_count": len(self.full_catalog),
            "expanded_to_full": self.expanded_to_full,
        }


_moe_singleton: Optional[ToolRoutingMoE] = None


def get_tool_routing_moe() -> ToolRoutingMoE:
    """Singleton getter for ToolRoutingMoE."""
    global _moe_singleton
    if _moe_singleton is None:
        _moe_singleton = ToolRoutingMoE(d_model=256)
    return _moe_singleton


def rank_tools(
    task_description: str,
    available_tools: Optional[Sequence[str]] = None,
    top_k: int = 4,
    context_tags: Optional[List[str]] = None,
    tool_descriptions: Optional[Dict[str, str]] = None,
    embed_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convenience entrypoint for non-mutating advisory tool ranking."""
    return get_tool_routing_moe().rank_tools(
        task_description=task_description,
        available_tools=available_tools,
        top_k=top_k,
        context_tags=context_tags,
        tool_descriptions=tool_descriptions,
        embed_fn=embed_fn,
    )


def narrow_tool_surface(
    task_description: str,
    available_tools: Optional[Sequence[str]] = None,
    top_k: int = 5,
    context_tags: Optional[List[str]] = None,
    tool_descriptions: Optional[Dict[str, str]] = None,
    embed_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convenience entrypoint for MCP-Zero Top-K tool catalog narrowing."""
    return get_tool_routing_moe().narrow_tool_surface(
        task_description=task_description,
        available_tools=available_tools,
        top_k=top_k,
        context_tags=context_tags,
        tool_descriptions=tool_descriptions,
        embed_fn=embed_fn,
    )
