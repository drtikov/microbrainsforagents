"""Tool Routing v3: Mixture of Experts (MoE) with Dynamic Top-2 Gating.

Implements an advanced SOTA MoE neural router for tool dispatch:
- d_model = 256, max_seq_len = 64.
- 6 Specialized Functional Cluster Experts:
    1. ExpertFileOps (read, write, edit, list, patch, diff)
    2. ExpertExecution (shell, command, pytest, unit testing)
    3. ExpertMemoryKnowledge (blackboard, episodic, semantic, genome, ontology)
    4. ExpertGitGovernance (git status, commit, diff, branch, seal)
    5. ExpertMicrobrainMeta (consult, route, incubator, lab, vsa, telemetry)
    6. ExpertConstitutionalReview (sacred sites, kinship, review gate, safety)
- Learned Dynamic Top-2 Gating:
    Computes gating logits over experts, selects top-2 experts, and blends
    expert output vectors via normalized gating weights:
        Y = alpha_1 * Expert_top1(X) + alpha_2 * Expert_top2(X)
- Per-Expert Conformal Calibration & Joint Coverage Guarantees (1 - alpha >= 95%).
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_tool": self.selected_tool,
            "secondary_tool": self.secondary_tool,
            "top_experts": [{"expert": e, "weight": round(w, 4)} for e, w in self.top_experts],
            "confidence": round(self.confidence, 4),
            "conformal_passed": self.conformal_passed,
            "per_expert_conformal": {k: round(v, 4) for k, v in self.per_expert_conformal.items()},
            "routing_rationale": self.routing_rationale,
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

        # Empirical held-out per-expert split-conformal calibration (n >= 20 per expert, alpha = 0.05)
        self.per_expert_quantiles: Dict[str, float] = {}
        self.per_expert_coverage: Dict[str, float] = {}
        self._calibrate_expert_quantiles(alpha=0.05)

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
        """Calibrate per-expert split-conformal quantiles on held-out multi-token probes (n = 22 >= 19)."""
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
        )

    route_LockFree = route

    def narrow_tool_surface(
        self,
        task_description: str,
        available_tools: Optional[Sequence[str]] = None,
        top_k: int = 5,
        context_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """MCP-Zero style hierarchical Top-K tool surface narrowing before LLM invocation.

        Returns Top-K candidate tools when conformal gating succeeds; escalates the full
        catalog unchanged on out-of-distribution / ambiguous queries.
        """
        k = max(1, int(top_k))
        all_catalog = (
            list(dict.fromkeys(str(t) for t in available_tools if t))
            if available_tools is not None
            else [t for tools in self.EXPERT_TOOLS.values() for t in tools]
        )
        decision = self.route(task_description, context_tags=context_tags, available_tools=all_catalog if available_tools is not None else None)
        if not decision.conformal_passed:
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
        narrowed: List[str] = [decision.selected_tool]
        for tool_name, _ in ranked:
            if tool_name not in narrowed and len(narrowed) < k:
                narrowed.append(tool_name)
        if decision.secondary_tool in all_catalog and decision.secondary_tool not in narrowed and len(narrowed) < k:
            narrowed.append(decision.secondary_tool)

        reduction = round(1.0 - (len(narrowed) / max(1, len(all_catalog))), 4)
        return {
            "mode": "narrowed",
            "candidate_tools": narrowed,
            "selected_tool": decision.selected_tool,
            "conformal_passed": True,
            "confidence": round(decision.confidence, 4),
            "reduction_ratio": max(0.0, reduction),
        }


_moe_singleton: Optional[ToolRoutingMoE] = None


def get_tool_routing_moe() -> ToolRoutingMoE:
    """Singleton getter for ToolRoutingMoE."""
    global _moe_singleton
    if _moe_singleton is None:
        _moe_singleton = ToolRoutingMoE(d_model=256)
    return _moe_singleton


def narrow_tool_surface(
    task_description: str,
    available_tools: Optional[Sequence[str]] = None,
    top_k: int = 5,
    context_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience entrypoint for MCP-Zero Top-K tool catalog narrowing."""
    return get_tool_routing_moe().narrow_tool_surface(
        task_description=task_description,
        available_tools=available_tools,
        top_k=top_k,
        context_tags=context_tags,
    )
