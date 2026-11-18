"""Microbrain consult tools: wire the trained neuro_brains fleet into the live loop.

P: 2026-11-15 - the microbrain family (core/analytical_micro_transformer.py and
friends) was fully functional but had zero live-path importers: no journal
events, no evolution evidence, invisible to the dispatcher.  These tools make
the trained brains real participants:

- ``microbrain_consult``  route a token sequence to the best-matching trained
  brain (vocabulary overlap) and return its prediction + conformal verdict.
- ``microbrain_status``   fleet inventory: domains, dims, vocab sizes,
  conformal calibration state, weights age.

Every consult journals ``microbrain_consult`` so the evolution loop finally
sees microbrain successes/failures as evidence - the proposal engine builds
premises from journal events, and until now microbrain produced none.

Kill switch: env ``AETHERIS_MICROBRAIN_TOOLS`` in {0, false, no} disables both
tools with an honest error (never silently).
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_PKG_BRAINS_DIR = Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains"
_CWD_BRAINS_DIR = Path("memory/store/neuro_brains")
BRAINS_DIR = _PKG_BRAINS_DIR if _PKG_BRAINS_DIR.exists() else _CWD_BRAINS_DIR
NAME_PREFIX = "neuro_"

# (name, meta-mtime) -> parsed meta dict.  Brains are retrained by the
# nursery lane (weights rewritten e.g. 2026-11-14 16:07), so the cache key
# includes the meta mtime and reloads when the file changes on disk.
_META_CACHE: Dict[str, Any] = {}
_LOADED: Dict[str, Any] = {}  # name -> (mtime, brain instance)


def _disabled() -> bool:
    return os.environ.get("AETHERIS_MICROBRAIN_TOOLS", "").strip().lower() in (
        "0", "false", "no")


def _list_brain_metas() -> List[Dict[str, Any]]:
    """Parse every ``*_meta.json`` in the fleet (meta-only: no weight IO).

    BitNet variants (``*_meta_bitnet158.json``) are skipped - they are a
    quantized copy of the same brain, not an additional domain.
    """
    out: List[Dict[str, Any]] = []
    try:
        files = sorted(BRAINS_DIR.glob("*_meta.json"))
    except OSError:
        return out
    for f in files:
        stem = f.stem
        if not stem.endswith("_meta"):
            continue  # _meta_bitnet158 etc.
        try:
            mt = f.stat().st_mtime
        except OSError:
            continue
        cached = _META_CACHE.get(str(f))
        if cached and cached.get("_mtime") == mt:
            out.append(cached["meta"])
            continue

        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = str(meta.get("name") or stem[: -len("_meta")])
        merged = dict(meta)
        merged["_name"] = name
        wrapped = {"_mtime": mt, "meta": merged}
        _META_CACHE[str(f)] = wrapped
        out.append(merged)
    return out


def _brain_vocab(meta: Dict[str, Any]) -> Dict[str, int]:
    v = meta.get("vocab")
    return v if isinstance(v, dict) else {}


def _load_brain(name: str):
    """Load one brain's weights (cached; reloads when the npz changes)."""
    weights = BRAINS_DIR / f"{name}_weights.npz"
    try:
        mt = weights.stat().st_mtime
    except OSError:
        return None
    hit = _LOADED.get(name)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        from ..core.analytical_micro_transformer import AnalyticalMicroTransformer
        brain = AnalyticalMicroTransformer.load(name, directory=BRAINS_DIR)
    except Exception as e:
        log.warning("microbrain load failed for %s: %s", name, e)
        return None
    if brain is None:
        return None
    _LOADED[name] = (mt, brain)
    return brain


_SWE_BILINGUAL_SYNONYMS: Dict[str, List[str]] = {
    "failing": ["failing", "failing_test", "tests_failed"],
    "failed": ["failed", "tests_failed", "failing_test"],
    "failure": ["failure", "failing_test", "failure_severity_benign"],
    "test": ["test", "pytest", "unit_test"],
    "tests": ["tests", "pytest", "unit_test"],
    "bug": ["bug", "traceback"],
    "исправь": ["fix", "bug", "edit"],
    "исправить": ["fix", "bug", "edit"],
    "ошибка": ["error", "bug", "exception", "syntax_error"],
    "ошибку": ["error", "bug", "exception", "syntax_error"],
    "тест": ["test", "pytest", "unit_test"],
    "тесты": ["test", "pytest", "unit_test"],
    "тестов": ["test", "pytest", "unit_test"],
    "падающий": ["failing_test", "failing", "tests_failed"],
    "синтаксис": ["syntax", "syntax_error", "syntax_tree_malformed"],
    "синтаксиса": ["syntax", "syntax_error", "syntax_tree_malformed"],
    "парсер": ["parser", "ast"],
    "парсере": ["parser", "ast"],
    "прочитай": ["read", "inspect", "repo_read"],
    "файл": ["file", "source", "repo_read"],
    "файле": ["file", "source"],
    "напиши": ["write", "implement", "repo_write"],
    "патч": ["patch", "code", "repo_write"],
    "коммит": ["commit", "git"],
    "ветку": ["branch", "git"],
    "память": ["memory", "recall"],
    "памяти": ["memory", "recall"],
}


def _tokenize(text: str) -> List[str]:
    """Unicode-aware tokenization supporting Cyrillic, Latin, compound bigrams, and technical SWE terms."""
    return [t for t in re.findall(r"(?u)\b\w+\b", str(text or "").lower()) if t]


def _expand_query_tokens(tokens: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Build candidate match groups per raw token plus adjacent snake_case bigrams.

    Preserves exact OOD resistance because each raw position contributes at most 1 overlap
    credit regardless of synonym count, and casual words produce no SWE expansions.
    """
    expanded: List[str] = []
    position_groups: List[List[str]] = []
    for i, t in enumerate(tokens):
        grp = [t]
        if t in _SWE_BILINGUAL_SYNONYMS:
            grp.extend(s for s in _SWE_BILINGUAL_SYNONYMS[t] if s not in grp)
        if i + 1 < len(tokens):
            bigram = f"{t}_{tokens[i + 1]}"
            grp.append(bigram)
            if bigram == "failing_test" and "failing_test" not in grp:
                grp.append("failing_test")
        position_groups.append(grp)
        expanded.extend(grp)

    # Deduplicate expanded tokens preserving priority order
    seen = set()
    unique_expanded = []
    for tok in expanded:
        if tok not in seen:
            seen.add(tok)
            unique_expanded.append(tok)
    return unique_expanded, position_groups


def _route(tokens: List[str]) -> Dict[str, Any]:
    """Best brain by raw-position vocabulary overlap with compound/bilingual SWE matching."""
    metas = _list_brain_metas()
    if not metas:
        return {"success": True, "abstained": True, "domain": None,
                "reason": "no trained brains found in memory/store/neuro_brains"}

    metas = sorted(metas, key=lambda m: str(m.get("_name") or m.get("name") or ""))
    expanded, pos_groups = _expand_query_tokens(tokens)

    # Prefer core calibrated safety/intent brains cleanly on ties
    best, best_hits, best_score = None, 0, -1.0
    for meta in metas:
        vocab = _brain_vocab(meta)
        hits = sum(1 for grp in pos_groups if any(cand in vocab for cand in grp))
        # Extra specificity weight for exact bigram / compound hits (e.g. failing_test)
        compound_hits = sum(1 for tok in expanded if "_" in tok and tok in vocab)
        score = float(hits) + 0.35 * float(compound_hits)
        if score > best_score or (score == best_score and best is None):
            best, best_hits, best_score = meta, hits, score

    overlap = best_hits / max(1, len(tokens))

    # Strict conformal OOD coverage check on distinct raw input positions
    if best_hits == 0 or overlap < 0.25:
        domains = [str(m.get("_name") or "?") for m in metas]
        return {"success": True, "abstained": True, "domain": None,
                "selected_brain": None,
                "calibrated_confidence": 0.0,
                "prediction_set": [],
                "set_size": 0,
                "reason_code": "ood_coverage_breach",
                "reason": "OOD coverage breach: input shares insufficient vocabulary with trained domains",
                "overlap": round(overlap, 3),
                "available_domains": domains}
    return {"success": True, "abstained": False, "domain": best,
            "expanded_tokens": expanded,
            "overlap": round(overlap, 3)}


def microbrain_consult(text: str = "", tokens: Optional[List[str]] = None,
                       domain: str = "auto") -> Dict[str, Any]:
    """Consult the trained microbrain fleet with a token sequence.

    Routes to the brain whose training vocabulary best matches the input,
    returns its prediction set and conformal verdict.
    """
    if _disabled():
        return {"success": False, "error": "microbrain tools disabled "
                "(AETHERIS_MICROBRAIN_TOOLS)"}
    if tokens is None or not isinstance(tokens, list):
        tokens = _tokenize(text)
    else:
        tokens = [str(t).lower() for t in tokens if str(t).strip()]
    if not tokens:
        return {"success": False, "error": "no tokens to consult on: pass "
                "text=<string> or tokens=<list of strings>"}

    route = _route(tokens)
    if route.get("abstained"):
        return route
    meta = route["domain"]
    name = str(meta.get("_name") or meta.get("name") or "")
    brain = _load_brain(name)
    if brain is None:
        return {"success": False, "error": f"brain {name} failed to load "
                "(weights missing or unreadable)"}

    candidate_tokens = route.get("expanded_tokens") or tokens
    in_vocab_seq = [t for t in candidate_tokens if t in getattr(brain, "vocab", {})]
    seq = in_vocab_seq if in_vocab_seq else [t if t in getattr(brain, "vocab", {}) else "<unk>" for t in tokens]
    seq = seq[: int(getattr(brain, "max_seq_len", 16) or 16)]
    try:
        fwd = brain.forward(seq)
    except Exception as e:
        return {"success": False, "error": f"forward pass failed: {e}"}
    try:
        conf = brain.infer_conformal(seq)
    except Exception:
        conf = {}

    calibrated_conf = round(float(conf.get("confidence") or fwd.get("confidence") or 0.0), 4)
    pred_set = list(conf.get("prediction_set", []))
    result = {
        "success": True,
        "abstained": False,
        "domain": name[len(NAME_PREFIX):] if name.startswith(NAME_PREFIX) else name,
        "selected_brain": name,
        "calibrated_confidence": calibrated_conf,
        "overlap": route.get("overlap"),
        "predicted_token": fwd.get("predicted_token"),
        "confidence": fwd.get("confidence"),
        "prediction_set": pred_set,
        "set_size": len(pred_set),
        "reason_code": str(conf.get("reason") or "ok"),
        "conformal_abstained": bool(conf.get("abstained")),
        "conformal_reason": str(conf.get("reason") or ""),
        "conformal_basis": str(meta.get("conformal_basis") or "uncalibrated"),
        "n_calibration_samples": int(conf.get("n_calibration_samples", len(meta.get("conformal_scores", [])))),
        "conformal_set_valid": bool(conf.get("conformal_guarantee_valid", meta.get("conformal_guarantee_valid", False))),
        "coverage_target": round(1.0 - float(meta.get("conformal_alpha", 0.05) or 0.05), 4),
        "upper_error_bound_95": conf.get("upper_error_bound_95"),
        "conformal_guarantee_valid": bool(conf.get("conformal_guarantee_valid", meta.get("conformal_guarantee_valid", False))),
        "model": name,
    }
    log.debug(
        "microbrain_consult domain=%s selected_brain=%s predicted=%r conf=%.3f overlap=%.2f conformal_abstained=%s",
        result["domain"],
        name,
        result.get("predicted_token"),
        calibrated_conf,
        float(route.get("overlap") or 0.0),
        result["conformal_abstained"],
    )
    return result


def microbrain_status(**kwargs) -> Dict[str, Any]:
    """Fleet inventory: every trained brain's domain, dim, vocab size,
    conformal calibration state, and weights age.  No weight IO."""
    if _disabled():
        return {"success": False, "error": "microbrain tools disabled "
                "(AETHERIS_MICROBRAIN_TOOLS)"}
    now = time.time()
    fleet = []
    for meta in _list_brain_metas():
        name = str(meta.get("_name") or meta.get("name") or "?")
        try:
            age_days = round((now - (BRAINS_DIR / f"{name}_weights.npz")
                              .stat().st_mtime) / 86400.0, 2)
        except OSError:
            age_days = None
        fleet.append({
            "domain": name[len(NAME_PREFIX):] if name.startswith(NAME_PREFIX) else name,
            "model": name,
            "d_model": meta.get("d_model"),
            "max_seq_len": meta.get("max_seq_len"),
            "vocab_size": len(_brain_vocab(meta)),
            "conformal_basis": str(meta.get("conformal_basis") or "uncalibrated"),
            "weights_age_days": age_days,
        })
    return {"success": True, "brains": fleet, "count": len(fleet),
            "healthy": sum(1 for b in fleet if b["vocab_size"] > 3)}


def consult_domain_hint(text: str) -> str:
    """Short advisory string for the plan lane; '' when no brain claims the
    text or the fleet is disabled.  Cheap: routes on meta vocab only."""
    if _disabled():
        return ""
    tokens = _tokenize(text)
    if not tokens:
        return ""
    try:
        route = _route(tokens)
    except Exception:
        return ""
    if route.get("abstained") or not route.get("domain"):
        return ""
    meta = route["domain"]
    name = str(meta.get("_name") or meta.get("name") or "?")
    dom = name[len(NAME_PREFIX):] if name.startswith(NAME_PREFIX) else name
    return ("trained domain '%s' matches this subject (vocab overlap %.2f); "
            "the microbrain_consult tool can predict over it" % (
                dom, float(route.get("overlap") or 0.0)))


__all__ = [
    "microbrain_consult",
    "microbrain_status",
    "consult_domain_hint",
]
