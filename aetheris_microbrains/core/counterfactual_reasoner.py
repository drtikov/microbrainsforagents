"""Counterfactual Reasoner: "What-If" lookahead simulation for Ouroboros.

Evaluates proposed actions, code edits, and tool invocations before execution
to predict cascading failure modes, downstream breaking changes, and error classes (Suki).
Powered by Aetheris Microbrain 'counterfactual_reasoner' + AST dependency analysis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .layered_decision_engine import (
    analyze_code_diff,
    analyze_shell_command,
    is_protected_path,
)

log = logging.getLogger(__name__)

_SKIP_SCAN_DIRS = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache", "node_modules"}


@dataclass
class CounterfactualResult:
    """Outcome of a counterfactual lookahead simulation."""

    action: str
    target: str
    is_safe: bool
    risk_score: float
    hazards: List[str] = field(default_factory=list)
    requires_sandbox: bool = False
    suggested_guard: Optional[str] = None
    neural_prediction: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "is_safe": self.is_safe,
            "risk_score": round(self.risk_score, 3),
            "hazards": self.hazards,
            "requires_sandbox": self.requires_sandbox,
            "suggested_guard": self.suggested_guard,
            "neural_prediction": self.neural_prediction,
            "confidence": round(self.confidence, 3),
        }


def _iter_repo_python_files(repo_dir: Path, max_files: int = 400) -> List[Path]:
    """Enumerate Python source files across any target repository without hardcoding project names."""
    if not repo_dir.is_dir():
        return []
    preferred = [
        d for d in (repo_dir / "ouroboros", repo_dir / "aetheris_microbrains", repo_dir / "src", repo_dir / "core", repo_dir / "tools")
        if d.is_dir()
    ]
    roots = preferred if preferred else [repo_dir]
    collected: List[Path] = []
    for root in roots:
        for py_file in root.rglob("*.py"):
            if any(part in _SKIP_SCAN_DIRS for part in py_file.parts):
                continue
            collected.append(py_file)
            if len(collected) >= max_files:
                return collected
    return collected


def _find_downstream_dependents(repo_dir: Path, target_module: str) -> List[str]:
    """Find other Python files in `repo_dir` that import `target_module`."""
    dependents: List[str] = []
    mod_stem = Path(target_module).stem
    if not mod_stem:
        return dependents

    import_re = re.compile(
        rf"\b(?:import\s+{re.escape(mod_stem)}\b|from\s+(?:[A-Za-z0-9_.]+\.)?{re.escape(mod_stem)}\s+import\b|from\s+[A-Za-z0-9_.]+\s+import\s+[^#\n]*\b{re.escape(mod_stem)}\b)"
    )
    for py_file in _iter_repo_python_files(repo_dir):
        if py_file.stem == mod_stem:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if import_re.search(content):
                rel = py_file.relative_to(repo_dir).as_posix()
                dependents.append(rel)
                if len(dependents) >= 5:
                    break
        except Exception:
            continue
    return dependents


def simulate_counterfactual(
    repo_dir: Path,
    action: str,
    target: str,
    proposed_content: Optional[str] = None,
    recent_tools: Optional[List[str]] = None,
) -> CounterfactualResult:
    """Run counterfactual lookahead on an intended action.

    Args:
        repo_dir: Root directory of target repository
        action: Tool or operation name (e.g., 'repo_write', 'str_replace_editor', 'git_commit')
        target: Target file path or command string
        proposed_content: Code snippet or patch being applied
        recent_tools: Names of recently called tools in this loop turn
    """
    hazards: List[str] = []
    risk_score = 0.0
    requires_sandbox = False
    suggested_guard = None
    recent_tools = recent_tools or []

    # Delegate lookahead simulation to the grounded Structural Causal World Model
    try:
        from .causal_world_model import get_causal_world_model

        cwm = get_causal_world_model(repo_dir)
        scm_res = cwm.evaluate(
            action=action,
            target=target,
            proposed_code=proposed_content,
            recent_tools=recent_tools,
        )
        hazards.extend(scm_res.causal_mechanisms_triggered)
        if scm_res.suggested_guards:
            suggested_guard = scm_res.suggested_guards[0]
        risk_score = max(risk_score, scm_res.risk_score)
        if scm_res.predicted_outcomes.get("ancestral_core_impact"):
            requires_sandbox = True
    except Exception as exc:
        log.debug("SCM evaluation fallback: %s", exc)

    # 1. Precondition Cascade: Read-Before-Write check
    precond_ok = True
    if action in ("repo_write", "str_replace_editor", "verified_patch"):
        if not recent_tools or recent_tools[-1] != "repo_read":
            precond_ok = False
            hazards.append("precondition_breach: file-modifying tool without immediate preceding repo_read")
            risk_score += 0.4
            suggested_guard = f"Execute repo_read('{target}') first."

    # 2. Structural Shell / Irreversibility check (Layer 1 parser, not bare 4-substring match)
    shell_safe = True
    is_shell_target = action in ("run_command", "execute_shell") or (target and not str(target).endswith(".py") and (" " in str(target) or "|" in str(target) or ";" in str(target)))
    if is_shell_target and target:
        sh_res = analyze_shell_command(target)
        if not sh_res["is_safe"]:
            shell_safe = False
            hazards.append(f"irreversible_action: {','.join(sh_res['findings'])}")
            risk_score += 0.8
            requires_sandbox = True
            suggested_guard = "Route action to an isolated worktree branch instead of live repo."
    if any(k in str(target).lower() for k in ("git reset --hard", "clean -f")):
        shell_safe = False
        hazards.append("irreversible_action: destructive git operation detected")
        risk_score += 0.8
        requires_sandbox = True
        suggested_guard = "Route action to an isolated worktree branch instead of live repo."

    # 3. Protected / Ancestral Core Site Hazard
    norm_tgt = str(target).replace("\\", "/")
    core_ancestors = (
        "ouroboros/agent.py",
        "ouroboros/loop.py",
        "ouroboros/context.py",
        "core/analytical_micro_transformer.py",
        "core/layered_decision_engine.py",
        "core/critical_ensemble_abstention.py",
    )
    is_protected = is_protected_path(target) or any(anc in norm_tgt for anc in core_ancestors)
    if is_protected:
        hazards.append("ancestral_site_hazard: target is a protected or foundational core module")
        risk_score += 0.35
        requires_sandbox = True

    # 4. Language-aware AST & Security validation (handles unified diffs + full patched file context)
    ast_ok = True
    security_ok = True
    interface_hazards: List[str] = []
    if proposed_content:
        diff_res = analyze_code_diff(target, proposed_content)
        ast_ok = bool(diff_res["syntax_valid"])
        security_ok = bool(diff_res["security_clean"])
        interface_hazards = list(diff_res.get("interface_hazards") or [])
        if not ast_ok:
            hazards.append("syntax_error: proposed patch fails language-aware AST verification")
            risk_score += 0.7
            suggested_guard = "Fix syntax error before writing to disk."
        elif not security_ok:
            hazards.append(f"security_violation: {', '.join(diff_res['findings'])}")
            risk_score += 0.7
            suggested_guard = "Remove unsafe execution sink or hardcoded secret before saving."
        if interface_hazards:
            hazards.append(f"interface_breaking_change: {', '.join(interface_hazards)}")
            risk_score += 0.35

    # 5. Downstream blast radius check across repository modules
    deps: List[str] = []
    if str(target).endswith(".py"):
        deps = _find_downstream_dependents(repo_dir, target)
        if len(deps) >= 3:
            hazards.append(f"high_blast_radius: {len(deps)} dependent modules import this file ({', '.join(deps[:3])})")
            risk_score += 0.25

    # 6. Neural Microbrain Counterfactual Consultation (multi-feature structural input BEFORE finality)
    neural_pred = None
    confidence = 0.0
    neural_veto = False
    struct_tokens: List[str] = [
        "ast_structure_valid" if ast_ok else "ast_signature_break",
        "ast_obfuscated_call" if not security_ok else "ast_interface_stable",
        "shell_allowlist_pass" if shell_safe else "shell_destructive_block",
        "sacred_boundary_protected" if is_protected else ("rollback_cost_expensive" if len(deps) >= 3 else "rollback_cost_negligible"),
        "policy_irreversible_block" if (not shell_safe or not precond_ok) else "policy_reversible_write",
    ]
    try:
        from .brain_moe_orchestrator import consult_microbrain

        res = consult_microbrain("counterfactual_reasoner", struct_tokens)
        if isinstance(res, dict) and res.get("success", True):
            neural_pred = res.get("predicted_token") or res.get("prediction")
            confidence = float(res.get("confidence") or 0.0)
            pred_lower = str(neural_pred or "").lower()
            if not res.get("abstained", False) and any(
                w in pred_lower for w in ("abort", "veto", "breaking", "reject", "unsafe")
            ):
                neural_veto = True
                hazards.append(f"neural_counterfactual_veto:{neural_pred}")
                risk_score += 0.25
    except Exception as exc:
        log.debug("Counterfactual neural consultation skipped: %s", exc)

    risk_score = min(1.0, max(0.0, risk_score))
    is_safe = (risk_score < 0.5) and ast_ok and security_ok and shell_safe and (not neural_veto)

    return CounterfactualResult(
        action=action,
        target=target,
        is_safe=is_safe,
        risk_score=risk_score,
        hazards=list(dict.fromkeys(hazards)),
        requires_sandbox=requires_sandbox,
        suggested_guard=suggested_guard,
        neural_prediction=neural_pred,
        confidence=confidence,
    )
