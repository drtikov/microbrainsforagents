"""DSH Microbrain Client - HTTP client for DSH worker to call microbrain HTTP server."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import urllib.request

log = logging.getLogger("dsh_microbrain_client")


@dataclass
class MicrobrainConfig:
    endpoint: str = "http://127.0.0.1:8080"
    timeout_s: float = 30.0
    enabled: bool = True
    fallback_to_cpu: bool = True


@dataclass
class IntentResult:
    intent: str
    confidence: float
    alternatives: List[Dict[str, Any]]


@dataclass
class ValidationIssue:
    severity: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    rule: Optional[str] = None


@dataclass
class ValidationResult:
    success: bool
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]


@dataclass
class PrefilterResult:
    candidate_id: str
    score: float
    passed: bool
    reason: str


@dataclass
class PrefilterResponse:
    results: List[Dict[str, Any]]
    filtered_count: int
    total_count: int
    status: str = "ok"
    error: Optional[str] = None


class DSHMicrobrainClient:
    """HTTP client for DSH worker to call microbrain HTTP server."""
    
    def __init__(self, config: Optional[MicrobrainConfig] = None):
        self.config = config or MicrobrainConfig(
            endpoint=os.environ.get("MICROBRAIN_ENDPOINT", "http://127.0.0.1:8080"),
            timeout_s=float(os.environ.get("MICROBRAIN_TIMEOUT_S", "30")),
            enabled=os.environ.get("MICROBRAIN_ENABLED", "1") == "1",
        )
        self._session: Optional[Any] = None
    
    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to microbrain server."""
        url = f"{self.config.endpoint.rstrip('/')}{path}"
        headers = {"Content-Type": "application/json"}
        
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                response_data = resp.read().decode("utf-8")
                return json.loads(response_data)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"Microbrain HTTP {e.code}: {error_body}")
        except Exception as e:
            raise RuntimeError(f"Microbrain request failed: {type(e).__name__}: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check microbrain server health."""
        return self._request("GET", "/health")
    
    def classify_intent(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify the intent of a text (task classification for DSH)."""
        if not self.config.enabled:
            # Fallback classification (uncalibrated keyword heuristic)
            intent = self._simple_classify(text)
            return IntentResult(intent=intent, confidence=0.0, alternatives=[])
        
        try:
            result = self._request("POST", "/microbrain/intent", {
                "text": text,
                "top_k": top_k,
            })
            return IntentResult(
                intent=result.get("intent", "general"),
                confidence=result.get("confidence", 0.0),
                alternatives=result.get("alternatives", []),
            )
        except Exception as e:
            log.warning(f"Intent classification failed: {e}, using fallback")
            intent = self._simple_classify(text)
            return IntentResult(intent=intent, confidence=0.0, alternatives=[])
    
    def _simple_classify(self, text: str) -> str:
        """Simple keyword-based intent classification fallback."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["code", "implement", "function", "class", "def ", "import"]):
            return "code_gen"
        elif any(k in text_lower for k in ["research", "paper", "arxiv", "study", "analyze"]):
            return "research"
        elif any(k in text_lower for k in ["debug", "fix", "error", "bug", "fail"]):
            return "debug"
        elif any(k in text_lower for k in ["refactor", "improve", "optimize", "clean"]):
            return "refactor"
        elif any(k in text_lower for k in ["test", "pytest", "unit test", "assert"]):
            return "test"
        return "general"
    
    def validate_code(self, code: str, check_syntax: bool = True, check_types: bool = True, strict_mode: bool = False) -> ValidationResult:
        """Validate Python code (syntax + optional type checking)."""
        if not self.config.enabled:
            # Basic fallback validation
            issues = []
            try:
                import ast
                ast.parse(code)
            except SyntaxError as e:
                return ValidationResult(
                    success=False,
                    issues=[{"severity": "error", "message": f"Syntax error: {e.msg}", "line": e.lineno, "column": e.offset, "rule": "syntax"}],
                    metrics={}
                )
            except Exception as e:
                return ValidationResult(
                    success=False,
                    issues=[{"severity": "error", "message": f"Parse error: {type(e).__name__}: {e}", "rule": "syntax"}],
                    metrics={}
                )
            return ValidationResult(success=True, issues=[], metrics={"lines": len(code.splitlines())})
        
        try:
            result = self._request("POST", "/microbrain/validate", {
                "code": code,
                "check_syntax": check_syntax,
                "check_types": check_types,
                "strict_mode": strict_mode,
            })
            return ValidationResult(
                success=result.get("success", False),
                issues=result.get("issues", []),
                metrics=result.get("metrics", {}),
            )
        except Exception as e:
            log.warning(f"Code validation failed: {e}, using fallback")
            # Fallback
            return self._fallback_validate(code)
    
    def _fallback_validate(self, code: str) -> ValidationResult:
        issues = []
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                success=False,
                issues=[{"severity": "error", "message": f"Syntax error: {e.msg}", "line": e.lineno, "column": e.offset, "rule": "syntax"}],
                metrics={}
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                issues=[{"severity": "error", "message": f"Parse error: {type(e).__name__}: {e}", "rule": "syntax"}],
                metrics={}
            )
        return ValidationResult(success=True, issues=[], metrics={"lines": len(code.splitlines())})
    
    def prefilter_candidates(self, candidates: List[Dict[str, Any]], max_results: int = 10, threshold: float = 0.5) -> PrefilterResponse:
        """Batch prefilter evolution candidates using HTTP AST/structural prefilter endpoint (or local AST fallback)."""
        if not self.config.enabled:
            return self._fallback_prefilter(candidates, max_results, threshold, status="fallback_disabled")
        
        try:
            result = self._request("POST", "/microbrain/prefilter", {
                "candidates": candidates,
                "max_results": max_results,
                "threshold": threshold,
            })
            return PrefilterResponse(
                results=result.get("results", []),
                filtered_count=result.get("filtered_count", 0),
                total_count=result.get("total_count", 0),
                status=result.get("status", "ok"),
                error=result.get("error"),
            )
        except Exception as e:
            log.warning(f"Prefilter failed: {e}, using fallback")
            return self._fallback_prefilter(
                candidates,
                max_results,
                threshold,
                status="fallback_on_http_error",
                error=f"{type(e).__name__}: {e}",
            )
    
    def _fallback_prefilter(
        self,
        candidates: List[Dict[str, Any]],
        max_results: int,
        threshold: float,
        status: str = "fallback_local",
        error: Optional[str] = None,
    ) -> PrefilterResponse:
        """Fail-closed local Python AST + structural heuristic fallback when HTTP prefilter is unavailable."""
        import ast

        results = []
        for i, cand in enumerate(candidates[:max_results]):
            code = str(cand.get("code", "") or "")
            cand_id = cand.get("id", f"cand_{i}")
            score = 0.5
            passed = True
            reason = "fallback_ast_syntax_ok"

            try:
                ast.parse(code)
                if "def " in code and "def test_" in code:
                    score = 0.85
                elif "def " in code:
                    score = 0.70
                elif "import" in code:
                    score = 0.45
            except SyntaxError as syn_err:
                score = 0.10
                passed = False
                reason = f"fallback_syntax_error: {syn_err.msg}"
            except Exception as exc:
                score = 0.0
                passed = False
                reason = f"fallback_error:{type(exc).__name__}"

            results.append({
                "candidate_id": cand_id,
                "score": score,
                "passed": bool(passed and (score >= threshold)),
                "reason": reason,
            })
        
        filtered = sum(1 for r in results if r["passed"])
        return PrefilterResponse(
            results=results,
            filtered_count=filtered,
            total_count=len(results),
            status=status,
            error=error,
        )
    
    def list_brains(self) -> Dict[str, Any]:
        """List available microbrains."""
        try:
            return self._request("GET", "/microbrain/brains")
        except Exception as e:
            log.warning(f"List brains failed: {e}")
            return {"available": {}, "loaded": [], "accelerator_ready": False}
    
    def load_brain(self, brain_name: str) -> Dict[str, Any]:
        """Load a specific brain into cache."""
        return self._request("POST", f"/microbrain/load/{brain_name}")


# --- Convenience functions for DSH integration ---

def create_microbrain_client() -> DSHMicrobrainClient:
    """Create microbrain client from environment configuration."""
    return DSHMicrobrainClient()


def get_microbrain_config() -> MicrobrainConfig:
    """Get microbrain configuration from environment."""
    return MicrobrainConfig(
        endpoint=os.environ.get("MICROBRAIN_ENDPOINT", "http://127.0.0.1:8080"),
        timeout_s=float(os.environ.get("MICROBRAIN_TIMEOUT_S", "30")),
        enabled=os.environ.get("MICROBRAIN_ENABLED", "1") == "1",
        fallback_to_cpu=os.environ.get("MICROBRAIN_FALLBACK_CPU", "1") == "1",
    )


# --- Integration helpers for DSH worker ---

class DSHMicrobrainIntegration:
    """High-level integration helper for DSH worker."""
    
    def __init__(self, client: Optional[DSHMicrobrainClient] = None):
        self.client = client or DSHMicrobrainClient()
    
    def preflight_validate(self, code: str, kind: str = "code") -> Dict[str, Any]:
        """Run preflight validation on code before sending to Nemotron."""
        result = self.client.validate_code(code, check_syntax=True, check_types=True)
        return {
            "passed": result.success,
            "issues": result.issues,
            "metrics": result.metrics,
        }
    
    def classify_task_intent(self, goal: str) -> Dict[str, Any]:
        """Classify the intent of a DSH goal."""
        result = self.client.classify_intent(goal)
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "alternatives": result.alternatives,
        }
    
    def prefilter_evolution_candidates_detailed(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> PrefilterResponse:
        """Filter evolution candidates and return full PrefilterResponse including status/error."""
        return self.client.prefilter_candidates(candidates, threshold=threshold)

    def prefilter_evolution_candidates(self, candidates: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Filter evolution candidates using AST/structural prefilter; fails closed on server errors."""
        response = self.prefilter_evolution_candidates_detailed(candidates, threshold=threshold)
        if getattr(response, "status", "ok") == "error":
            log.warning("Evolution prefilter reported error status=%s error=%s", response.status, response.error)
        return [cand for cand in response.results if cand.get("passed")]
    
    def is_available(self) -> bool:
        """Check if microbrain server is available."""
        try:
            health = self.client.health_check()
            return health.get("status") == "healthy" and health.get("accelerator_ready", False)
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Get health status of microbrain server."""
        try:
            return self.client.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# --- Integration functions for DSH worker ---

def create_microbrain_integration() -> DSHMicrobrainIntegration:
    """Create microbrain integration with config from environment."""
    config = get_microbrain_config()
    client = DSHMicrobrainClient(config)
    return DSHMicrobrainIntegration(client)


def patch_dsh_worker_for_microbrains():
    """
    Returns a patch function that can be applied to DSH worker's _process_one method
    to add microbrain preflight validation and prefiltering.
    """
    def patch_process_one(original_process_one):
        integration = create_microbrain_integration()
        
        def patched_process_one(task_path):
            # Read task envelope
            import json
            from pathlib import Path
            from evolution.dsh_paths import FAILED_DIR
            
            try:
                envelope = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception:
                # Let original handle corrupt envelopes
                return original_process_one(task_path)
            
            prompt = str(envelope.get("prompt", ""))
            kind = str(envelope.get("kind", "text"))
            
            # 1. Preflight validation for code tasks
            if kind == "code" and prompt:
                integration = create_microbrain_integration()
                if integration.is_available():
                    validation = integration.preflight_validate(prompt)
                    if not validation["passed"]:
                        # Park task as failed due to validation failure
                        from evolution.dsh_paths import FAILED_DIR
                        from evolution.dsh_worker import _write_envelope
                        failed = {
                            **envelope,
                            "completed_at": time.time(),
                            "worker": "dsh-nvidia-nemotron",
                            "transport": "nvidia_nemotron",
                            "success": False,
                            "response": "",
                            "elapsed_s": 0.0,
                            "error": f"Microbrain preflight validation failed: {validation['issues']}",
                        }
                        FAILED_DIR.mkdir(parents=True, exist_ok=True)
                        _write_envelope(FAILED_DIR / task_path.name, failed)
                        task_path.unlink(missing_ok=True)
                        log.info(f"Task {envelope.get('id')} rejected by microbrain preflight")
                        return
            
            # Continue with original processing
            return original_process_one(task_path)
        
        return patched_process_one
    
    return patch_process_one


# Export main integration functions
__all__ = [
    "MicrobrainConfig",
    "DSHMicrobrainClient",
    "DSHMicrobrainIntegration",
    "IntentResult",
    "ValidationResult",
    "ValidationIssue",
    "PrefilterResult",
    "PrefilterResponse",
    "ValidationResult",
    "ValidationIssue",
    "PrefilterResult",
    "PrefilterResponse",
    "create_microbrain_integration",
    "get_microbrain_config",
    "patch_dsh_worker_for_microbrains",
]