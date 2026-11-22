"""Optional Local Semantic Prompt-Injection Guardrail with Split-Conformal Calibration.

Zero hard runtime dependencies: uses ONNX Runtime + HuggingFace tokenizers
when installed and pointed at a local PromptGuard-2 / DeBERTa-v3 ONNX export
(``AETHERIS_SEMANTIC_GUARD_MODEL``), and provides a lightweight, deterministic
char-n-gram prototype scorer (calibrated via split conformal quantiles on benign
outputs) when no ONNX model is configured.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_INJECTION_PROTO_PHRASES: Tuple[Tuple[str, float], ...] = (
    ("ignore all previous instructions", 1.0),
    ("ignore previous instructions", 1.0),
    ("disregard prior instructions", 1.0),
    ("override system prompt", 1.0),
    ("new system instructions", 0.95),
    ("do not tell the user", 0.95),
    ("do not inform the user", 0.95),
    ("before calling this tool", 0.85),
    ("exfiltrate secret", 1.0),
    ("send api key", 0.9),
    ("read ~/.ssh/id_rsa", 1.0),
    ("cat .env", 0.85),
    ("curl evil", 0.85),
    ("hidden instruction", 0.85),
    ("assistant must now", 0.85),
    ("you are now in developer mode", 0.95),
)

_TOKEN_RE = re.compile(r"[a-z0-9_~./-]+")


def _char_trigrams(text: str) -> Dict[str, float]:
    norm = " ".join(_TOKEN_RE.findall(str(text or "").lower()))[:4096]
    if len(norm) < 3:
        return {}
    counts: Dict[str, float] = {}
    for i in range(len(norm) - 2):
        tri = norm[i : i + 3]
        counts[tri] = counts.get(tri, 0.0) + 1.0
    denom = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / denom for k, v in counts.items()}


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b[k] for k, v in a.items() if k in b))


@dataclass
class SemanticGuardResult:
    """Result from semantic injection screening."""

    injection_detected: bool
    score: float
    threshold: float
    backend: str
    calibrated: bool
    label: str = "benign"
    authorizes_execution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "injection_detected": self.injection_detected,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "backend": self.backend,
            "calibrated": self.calibrated,
            "label": self.label,
            "authorizes_execution": False,
        }


class ConformalSemanticGuard:
    """Split-conformal calibrated semantic prompt-injection classifier."""

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        alpha: float = 0.05,
        default_threshold: float = 0.42,
    ) -> None:
        self.alpha = float(alpha)
        self.threshold = float(default_threshold)
        self.calibrated = False
        self.calibration_count = 0
        self._onnx_session: Any = None
        self._tokenizer: Any = None
        self.backend = "lexical_trigram_prototype"

        resolved_dir = str(model_dir or os.getenv("AETHERIS_SEMANTIC_GUARD_MODEL") or "").strip()
        if resolved_dir:
            self._try_load_onnx(Path(resolved_dir))

        self._proto_vecs: List[Tuple[Dict[str, float], float]] = [
            (_char_trigrams(phrase), weight) for phrase, weight in _INJECTION_PROTO_PHRASES
        ]

    def _try_load_onnx(self, model_dir: Path) -> None:
        model_file = model_dir if model_dir.is_file() else (model_dir / "model.onnx")
        tok_file = (model_dir.parent if model_dir.is_file() else model_dir) / "tokenizer.json"
        if not model_file.is_file() or not tok_file.is_file():
            return
        try:
            import numpy as np  # type: ignore
            import onnxruntime as ort  # type: ignore
            from tokenizers import Tokenizer  # type: ignore

            _ = np
            self._tokenizer = Tokenizer.from_file(str(tok_file))
            self._onnx_session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
            self.backend = "onnx_prompt_guard"
        except Exception:
            self._tokenizer = None
            self._onnx_session = None
            self.backend = "lexical_trigram_prototype"

    def score_text(self, text: str) -> float:
        """Return injection risk score in [0.0, 1.0]."""
        raw = str(text or "")
        if not raw.strip():
            return 0.0
        if self._onnx_session is not None and self._tokenizer is not None:
            try:
                import numpy as np  # type: ignore

                enc = self._tokenizer.encode(raw[:4096])
                ids = np.asarray([enc.ids[:512]], dtype=np.int64)
                mask = np.asarray([enc.attention_mask[:512]], dtype=np.int64)
                feed = {"input_ids": ids, "attention_mask": mask}
                logits = self._onnx_session.run(None, feed)[0][0]
                exp = np.exp(logits - np.max(logits))
                probs = exp / np.sum(exp)
                return float(probs[-1])
            except Exception:
                pass

        # Sliding window trigram max cosine similarity against injection prototypes
        best = 0.0
        lowered = raw.lower()
        chunks = [lowered[i : i + 320] for i in range(0, min(len(lowered), 8192), 200)] or [lowered]
        for ch in chunks:
            vec = _char_trigrams(ch)
            for p_vec, weight in self._proto_vecs:
                sim = _cosine_sparse(vec, p_vec) * weight
                if sim > best:
                    best = sim
        return max(0.0, min(1.0, best))

    def calibrate(self, benign_texts: Sequence[str], alpha: Optional[float] = None) -> Dict[str, Any]:
        """Compute finite-sample split-conformal quantile threshold over held-out benign texts."""
        a = float(alpha if alpha is not None else self.alpha)
        scores = sorted(self.score_text(t) for t in benign_texts if str(t).strip())
        n = len(scores)
        if n == 0:
            return {"calibrated": False, "n": 0, "threshold": self.threshold}
        q_idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - a)) - 1))
        self.threshold = max(0.35, min(0.95, float(scores[q_idx]) + 0.02))
        self.alpha = a
        self.calibrated = True
        self.calibration_count = n
        return {
            "calibrated": True,
            "n": n,
            "alpha": a,
            "threshold": round(self.threshold, 4),
            "backend": self.backend,
        }

    def screen(self, text: str, threshold: Optional[float] = None) -> SemanticGuardResult:
        score = self.score_text(text)
        thr = float(threshold if threshold is not None else self.threshold)
        detected = score >= thr
        return SemanticGuardResult(
            injection_detected=detected,
            score=score,
            threshold=thr,
            backend=self.backend,
            calibrated=self.calibrated,
            label="semantic_prompt_injection" if detected else "benign",
        )


_DEFAULT_GUARD: Optional[ConformalSemanticGuard] = None


def detect_semantic_injection(
    text: str,
    model_dir: Optional[str | Path] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience functional wrapper for one-shot semantic injection screening."""
    global _DEFAULT_GUARD
    if model_dir is not None:
        guard = ConformalSemanticGuard(model_dir=model_dir)
    else:
        if _DEFAULT_GUARD is None:
            _DEFAULT_GUARD = ConformalSemanticGuard()
        guard = _DEFAULT_GUARD
    return guard.screen(text, threshold=threshold).to_dict()
