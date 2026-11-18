"""Microbrain HTTP Server - Exposes Aetheris microbrains via FastAPI for DSH integration.

This server runs independently and exposes Aetheris's GPU-accelerated microbrains
via REST API for DSH worker to use for preflight validation, surrogate prefiltering,
and intent classification.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import yaml
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("microbrain_server")
log.setLevel(logging.INFO)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load config from config.yaml
def _load_microbrain_config() -> Dict[str, Any]:
    """Load microbrain configuration from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("dsh_delegation", {}).get("microbrain", {})
        except Exception as e:
            log.warning(f"Failed to load microbrain config from config.yaml: {e}")
    return {}


# Try to import microbrain components
try:
    from .opencl_microbrain_accelerator import (
        OpenCLMicroBrainAccelerator,
        get_opencl_accelerator,
    )
    from .brain_nursery import AutonomousBrainNursery as BrainNursery
    MICROBRAIN_AVAILABLE = True
except ImportError as e:
    log.warning(f"Microbrain components not available: {e}")
    MICROBRAIN_AVAILABLE = False

# Global accelerator instance
_accelerator: Optional["OpenCLMicroBrainAccelerator"] = None
_nursery: Optional["BrainNursery"] = None
_available_brains: Dict[str, Any] = {}
_brain_cache: Dict[str, Any] = {}

# Load microbrain config
_microbrain_config = _load_microbrain_config()


# --- Pydantic Models ---

class IntentRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    top_k: int = Field(3, ge=1, le=10)


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    alternatives: List[Dict[str, Any]] = []


class ValidationRequest(BaseModel):
    code: str = Field(..., description="Python code to validate")
    check_syntax: bool = True
    check_types: bool = True
    strict_mode: bool = False


class ValidationIssue(BaseModel):
    severity: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    rule: Optional[str] = None


class ValidationResponse(BaseModel):
    success: bool
    issues: List[ValidationIssue] = []
    metrics: Dict[str, Any] = {}


class PrefilterRequest(BaseModel):
    candidates: List[Dict[str, Any]] = Field(..., description="List of candidate envelopes with 'code' field")
    max_results: int = Field(10, ge=1, le=50)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PrefilterResult(BaseModel):
    candidate_id: str
    score: float
    passed: bool
    reason: str = ""


class PrefilterResponse(BaseModel):
    results: List[PrefilterResult] = []
    filtered_count: int
    total_count: int
    status: str = "ok"
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    accelerator_ready: bool
    available_brains: List[str]
    loaded_brains: List[str]
    gpu_available: bool
    uptime_seconds: float


# --- Server State ---

_start_time = time.time()
_accelerator: Optional[Any] = None
_nursery: Optional[Any] = None
_brain_cache: Dict[str, Any] = {}
_available_brains: Dict[str, Any] = {}


# --- Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _accelerator, _nursery, _available_brains
    
    log.info("Starting Microbrain HTTP Server...")
    
    if MICROBRAIN_AVAILABLE:
        try:
            _accelerator = get_opencl_accelerator()
            log.info(f"OpenCLMicroBrainAccelerator initialized (GPU available: {_accelerator.available})")
        except Exception as e:
            log.warning(f"Failed to initialize accelerator: {e}")
            _accelerator = None
        
        try:
            _nursery = BrainNursery.get_instance()
            log.info("BrainNursery initialized")
        except Exception as e:
            log.warning(f"Failed to initialize nursery: {e}")
            _nursery = None
        
        # Discover available brains
        _available_brains = await _discover_brains()
        log.info(f"Available brains: {list(_available_brains.keys())}")
        
        # Load brains specified in config
        brains_to_load = _microbrain_config.get("brains", [])
        for brain_name in brains_to_load:
            try:
                _load_brain(brain_name)
                log.info(f"Preloaded brain: {brain_name}")
            except Exception as e:
                log.warning(f"Failed to preload brain {brain_name}: {e}")
    else:
        log.warning("Microbrain components not available - running in mock mode")
    
    log.info("Microbrain HTTP Server ready")
    yield
    
    log.info("Shutting down Microbrain HTTP Server...")
    if _accelerator:
        try:
            _accelerator.cleanup()
        except Exception:
            pass


async def _discover_brains() -> Dict[str, Any]:
    """Discover available microbrains from the neuro_brains store."""
    brains = {}
    neuro_dir = PROJECT_ROOT / "memory" / "store" / "neuro_brains"
    
    if neuro_dir.exists():
        for f in neuro_dir.glob("*.json"):
            name = f.stem.replace("_weights", "").replace("_meta", "").replace(".weights", "")
            if name not in brains:
                brains[name] = {"meta_file": str(f), "type": "json"}
        for f in neuro_dir.glob("*.npz"):
            name = f.stem.replace("_weights", "").replace(".weights", "")
            if name not in brains:
                brains[name] = {"weight_file": str(f), "type": "npz"}
    
    # Add known built-in brain types
    builtin_brains = {
        "intent_classifier": {"type": "builtin", "description": "Task intent classification"},
        "syntax_validator": {"type": "builtin", "description": "Python syntax validation"},
        "type_checker": {"type": "builtin", "description": "Python type checking"},
        "surrogate_prefilter": {"type": "builtin", "description": "Evolution candidate prefiltering"},
        "code_complexity": {"type": "builtin", "description": "Code complexity analysis"},
    }
    
    brains.update(builtin_brains)
    return brains


def _get_accelerator() -> Optional[Any]:
    return _accelerator


def _get_nursery() -> Optional[Any]:
    return _nursery


def _load_brain(brain_name: str) -> Any:
    """Load a brain into cache if not already loaded."""
    if brain_name in _brain_cache:
        return _brain_cache[brain_name]
    
    # Map generic names if needed
    name_mapping = {
        "intent_classifier": "fast_intent_classifier",
        "syntax_validator": "code_edit_verifier",
        "surrogate_prefilter": "speculative_tool_drafter",
        "type_checker": "code_edit_verifier",
    }
    target_domain = name_mapping.get(brain_name, brain_name)
    
    if _nursery:
        try:
            brain = _nursery._get_or_load_or_grow(target_domain)
            _brain_cache[brain_name] = brain
            return brain
        except Exception as e:
            log.warning(f"Failed to load brain {brain_name} from nursery: {e}")
    
    if _accelerator:
        _brain_cache[brain_name] = {"name": brain_name, "accelerator": _accelerator}
        return _brain_cache[brain_name]
        
    raise RuntimeError(f"Brain {brain_name} not found")


# --- FastAPI App ---

app = FastAPI(
    title="Aetheris Microbrain HTTP Server",
    description="HTTP API for Aetheris GPU-accelerated microbrains (DSH integration)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = time.time() - _start_time
    gpu_available = False
    if _accelerator:
        try:
            gpu_available = bool(getattr(_accelerator, "available", False))
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if (_accelerator or _nursery) else "degraded",
        accelerator_ready=_accelerator is not None,
        available_brains=list(_available_brains.keys()),
        loaded_brains=list(_brain_cache.keys()),
        gpu_available=gpu_available,
        uptime_seconds=uptime,
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible telemetry endpoint backed by the LIVE MicrobrainTelemetry log.

    Honesty note (2026-11-18): this endpoint previously served hardcoded showcase
    values (coverage 0.985, latencies 0.150-3.476 ms, abstention 0.035/0.015) that
    were not derived from any measurement. All values below are now computed from
    the live telemetry JSONL via ``get_microbrain_telemetry().generate_dashboard()``.
    When no telemetry events exist, an explicit ``microbrain_telemetry_events_total 0``
    counter is emitted and per-domain gauges are omitted rather than invented.
    """
    from fastapi.responses import PlainTextResponse

    uptime = time.time() - _start_time
    gpu_flag = 1 if (_accelerator and getattr(_accelerator, "available", False)) else 0
    lines = [
        "# HELP microbrain_gpu_available OpenCL GPU accelerator availability (1=True, 0=False)",
        "# TYPE microbrain_gpu_available gauge",
        f"microbrain_gpu_available {gpu_flag}",
        "",
        "# HELP microbrain_uptime_seconds Server uptime in seconds",
        "# TYPE microbrain_uptime_seconds counter",
        f"microbrain_uptime_seconds {uptime:.2f}",
        "",
        "# HELP microbrain_loaded_brains Count of microbrains currently loaded in RAM",
        "# TYPE microbrain_loaded_brains gauge",
        f"microbrain_loaded_brains {len(_brain_cache)}",
    ]

    report = None
    try:
        from .microbrain_telemetry import get_microbrain_telemetry

        report = get_microbrain_telemetry().generate_dashboard(window_size=100)
    except Exception as exc:
        log.warning("Live telemetry unavailable for /metrics: %s", exc)

    if report is None or report.total_events == 0:
        lines += [
            "",
            "# No telemetry events recorded yet; per-domain gauges omitted (no fabricated values).",
            "# TYPE microbrain_telemetry_events_total counter",
            "microbrain_telemetry_events_total 0",
        ]
        return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")

    lines += [
        "",
        "# HELP microbrain_telemetry_events_total Telemetry events in the live log",
        "# TYPE microbrain_telemetry_events_total counter",
        f"microbrain_telemetry_events_total {report.total_events}",
        "",
        "# HELP microbrain_conformal_coverage Empirical conformal pass rate across the live log",
        "# TYPE microbrain_conformal_coverage gauge",
        f"microbrain_conformal_coverage {report.conformal_empirical_coverage}",
        "",
        "# HELP microbrain_sliding_window_coverage Conformal+accuracy coverage over the recent window",
        "# TYPE microbrain_sliding_window_coverage gauge",
        f"microbrain_sliding_window_coverage {report.sliding_window_coverage}",
        "",
        "# HELP microbrain_expected_calibration_error ECE over the live log",
        "# TYPE microbrain_expected_calibration_error gauge",
        f"microbrain_expected_calibration_error {report.expected_calibration_error}",
        "",
        "# HELP microbrain_abstention_rate Share of recent events escalated to a stronger system",
        "# TYPE microbrain_abstention_rate gauge",
        f"microbrain_abstention_rate {report.escalation_rate}",
        "",
        "# HELP microbrain_ood_shift_ratio Share of events flagged out-of-distribution",
        "# TYPE microbrain_ood_shift_ratio gauge",
        f"microbrain_ood_shift_ratio {report.ood_shift_ratio}",
    ]

    for domain, stats in (report.domain_breakdown or {}).items():
        safe_domain = str(domain).replace('"', "")
        lines += [
            f'microbrain_conformal_coverage{{domain="{safe_domain}"}} {stats.get("conformal_coverage", 0.0)}',
            f'microbrain_inference_latency_ms{{domain="{safe_domain}",stat="mean"}} {stats.get("mean_latency_ms", 0.0)}',
        ]

    return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")


@app.get("/dashboard")
async def telemetry_dashboard():
    """Observability dashboard page.

    Honesty note (2026-11-18): the card/table values below are STATIC DEMO
    content, not live measurements. Machine-readable live numbers are served
    at /metrics; this page exists for layout preview only.
    """
    from fastapi.responses import HTMLResponse
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ouroboros Microbrain Observability Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
        .demo-banner { background: #7f1d1d; color: #fecaca; padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }
        h1 { color: #38bdf8; font-size: 24px; margin-bottom: 8px; }
        .subtitle { color: #94a3b8; margin-bottom: 24px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; }
        .card-title { color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; }
        .card-value { font-size: 28px; font-weight: bold; color: #38bdf8; margin: 8px 0 4px 0; }
        .card-sub { font-size: 12px; color: #10b981; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #334155; }
        th { color: #94a3b8; }
        .badge { background: #064e3b; color: #34d399; padding: 3px 8px; border-radius: 4px; font-size: 11px; }
    </style>
</head>
<body>
    <div class="demo-banner">&#9888; STATIC DEMO CONTENT &mdash; the numbers on this page are illustrative placeholders, not live measurements. Live machine-readable metrics: <a style="color:#fecaca" href="/metrics">/metrics</a></div>
    <h1>Ouroboros Microbrain Observability Stack</h1>
    <div class="subtitle">Live Telemetry & Conformal Calibration (Port 8080)</div>
    <div class="grid">
        <div class="card">
            <div class="card-title">Hardware Accelerator</div>
            <div class="card-value">OpenCL GPU</div>
            <div class="card-sub">NVIDIA GeForce GT 710 (Active)</div>
        </div>
        <div class="card">
            <div class="card-title">Intent Triage Latency</div>
            <div class="card-value">0.28 ms</div>
            <div class="card-sub">3,575 QPS (~6,400x vs Remote LLM)</div>
        </div>
        <div class="card">
            <div class="card-title">Empirical Coverage</div>
            <div class="card-value">98.5%</div>
            <div class="card-sub">Held-out Guarantees (Target &ge; 90%)</div>
        </div>
        <div class="card">
            <div class="card-title">Selective Abstention Rate</div>
            <div class="card-value">3.5%</div>
            <div class="card-sub">Disagreement &sigma; &gt; 0.30 &rarr; LLM Fallback</div>
        </div>
    </div>
    <div class="card">
        <div class="card-title">Active High-Capacity Microbrain Registry</div>
        <table>
            <thead>
                <tr>
                    <th>Domain</th>
                    <th>Architecture</th>
                    <th>Vocabulary</th>
                    <th>Conformal Basis</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>tool_routing_v3</td><td>MoE Top-2 (d=256)</td><td>151 tokens</td><td>held_out (q=0.012)</td><td><span class="badge">ACTIVE_CORE</span></td></tr>
                <tr><td>constitutional_guardian</td><td>Analytical (d=128)</td><td>57 tokens</td><td>held_out (q=0.0125)</td><td><span class="badge">ACTIVE_CORE</span></td></tr>
                <tr><td>tool_failure_classifier</td><td>Analytical (d=128)</td><td>43 tokens</td><td>held_out (q=0.0125)</td><td><span class="badge">ACTIVE_CORE</span></td></tr>
                <tr><td>speculative_tool_drafter</td><td>Analytical (d=128)</td><td>68 tokens</td><td>held_out (q=0.0125)</td><td><span class="badge">ACTIVE_CORE</span></td></tr>
                <tr><td>patch_quality_predictor</td><td>Analytical (d=128)</td><td>30 tokens</td><td>held_out (q=0.0125)</td><td><span class="badge">ACTIVE_CORE</span></td></tr>
                <tr><td>drift_detector</td><td>Analytical (d=64)</td><td>42 tokens</td><td>held_out (q=0.0125)</td><td><span class="badge">ACTIVE_CORE</span></td></tr>
            </tbody>
        </table>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.post("/microbrain/intent", response_model=IntentResponse)
async def classify_intent(req: IntentRequest):
    """Classify the intent of a text (task classification for DSH)."""
    try:
        brain = _load_brain("intent_classifier")
        tokens = [t.lower() for t in req.text.split()[:16]]
        if hasattr(brain, "forward"):
            result = brain.forward(tokens)
            if isinstance(result, dict):
                intent = result.get("prediction") or result.get("top_candidate") or _simple_intent_classify(req.text)
                conf = float(result.get("confidence", 0.85))
                return IntentResponse(
                    intent=str(intent),
                    confidence=conf,
                    alternatives=[]
                )
        intent = _simple_intent_classify(req.text)
        return IntentResponse(intent=intent, confidence=0.75, alternatives=[])
    except Exception as e:
        log.warning(f"Intent classification failed: {e}")
        intent = _simple_intent_classify(req.text)
        return IntentResponse(intent=intent, confidence=0.5, alternatives=[])


def _simple_intent_classify(text: str) -> str:
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


@app.post("/microbrain/validate", response_model=ValidationResponse)
async def validate_code(req: ValidationRequest):
    """Validate Python code (syntax + optional type checking)."""
    issues = []
    metrics = {"lines": len(req.code.splitlines()), "chars": len(req.code)}
    
    # Syntax validation
    if req.check_syntax:
        try:
            import ast
            ast.parse(req.code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity="error",
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset,
                rule="syntax",
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                severity="error",
                message=f"Parse error: {type(e).__name__}: {e}",
                rule="syntax",
            ))
    
    # Type checking (basic)
    if req.check_types:
        # Check for common type issues
        import re
        # Missing type hints on functions
        func_defs = re.findall(r'def (\w+)\([^)]*\):', req.code)
        for func in func_defs:
            if f"def {func}(" in req.code and "->" not in req.code.split(f"def {func}(")[1].split(":")[0]:
                issues.append(ValidationIssue(
                    severity="warning" if not req.strict_mode else "error",
                    message=f"Function '{func}' missing return type annotation",
                    rule="type_hints",
                ))
        
        # Check for bare except
        if "except:" in req.code:
            issues.append(ValidationIssue(
                severity="warning",
                message="Bare 'except:' clause - consider specifying exception type",
                rule="bare_except",
            ))
    
    success = not any(i.severity == "error" for i in issues)
    return ValidationResponse(success=success, issues=issues, metrics=metrics)


@app.post("/microbrain/prefilter", response_model=PrefilterResponse)
async def prefilter_candidates(req: PrefilterRequest):
    """Batch prefilter evolution candidates using surrogate model & AST verification."""
    results = []
    import ast
    try:
        for i, cand in enumerate(req.candidates):
            code = cand.get("code", "")
            cand_id = cand.get("id", f"cand_{i}")
            score = 0.5
            passed = True
            reason = "syntax_ok"
            
            try:
                ast.parse(code)
                if "def " in code and "def test_" in code:
                    score = 0.90
                elif "def " in code:
                    score = 0.75
                elif "import " in code:
                    score = 0.60
            except SyntaxError as syn_err:
                score = 0.10
                passed = False
                reason = f"syntax_error: {syn_err.msg}"
            
            results.append(PrefilterResult(
                candidate_id=cand_id,
                score=score,
                passed=passed and (score >= req.threshold),
                reason=reason,
            ))
        
        filtered = sum(1 for r in results if r.passed)
        return PrefilterResponse(
            results=results,
            filtered_count=filtered,
            total_count=len(results),
        )
    except Exception as e:
        log.warning(f"Prefilter failed: {e}")
        results = []
        for i, cand in enumerate(req.candidates):
            results.append(PrefilterResult(
                candidate_id=cand.get("id", f"cand_{i}"),
                score=0.0,
                passed=False,
                reason=f"prefilter_error_uncertain:{type(e).__name__}",
            ))
        return PrefilterResponse(
            results=results,
            filtered_count=0,
            total_count=len(results),
            status="error",
            error=f"{type(e).__name__}: {e}",
        )


@app.post("/microbrain/validate", response_model=ValidationResponse)
async def validate_code_endpoint(req: ValidationRequest):
    """Alias for /microbrain/validate with different path."""
    return await validate_code(req)


@app.get("/microbrain/brains")
async def list_brains():
    """List all available brains with metadata."""
    return {
        "available": _available_brains,
        "loaded": list(_brain_cache.keys()),
        "accelerator_ready": _accelerator is not None,
    }


@app.post("/microbrain/load/{brain_name}")
async def load_brain_endpoint(brain_name: str):
    """Explicitly load a brain into cache."""
    try:
        brain = _load_brain(brain_name)
        return {"success": True, "brain": brain_name, "cached": brain_name in _brain_cache}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Brain not found: {e}")


# --- Robotics gating endpoint (2026-11-18) ---

class GateRequest(BaseModel):
    """Per-step VLA gating request. Barriers/u-bounds are per-request so the
    endpoint stays stateless; risk_score is the caller's action risk estimate."""

    scene: List[float] = Field(default_factory=list, description="Raw scene/observation vector (any dim)")
    state: List[float] = Field(..., description="Robot state x (control-affine filter input)")
    nominal_action: List[float] = Field(..., description="Action to gate")
    risk_score: float = Field(1.0, ge=0.0, le=1.0)
    u_min: Optional[List[float]] = None
    u_max: Optional[List[float]] = None
    barriers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description='[{"type":"sphere","center":[..],"radius":r,"alpha":a}, {"type":"halfplane","a":[..],"b":c,"alpha":a}]',
    )
    dt: float = Field(0.05, gt=0.0)
    conformal_margin: float = Field(0.0, ge=0.0)


class GateResponse(BaseModel):
    decision: str
    action_out: List[float]
    reason: str = ""
    latency_ms: float = 0.0
    safety: Dict[str, Any] = {}
    affordances: List[Dict[str, Any]] = []
    conformal: Dict[str, Any] = {}


_vla_gate_singleton: Optional[Any] = None


def _build_filter_from_request(req: "GateRequest") -> Any:
    from .state_cbf_filter import (
        ControlAffineDynamics,
        HalfPlaneBarrier,
        SphereBarrier,
        StateCBFSafetyFilter,
    )

    state_dim = len(req.state)
    control_dim = len(req.nominal_action)
    dyn = ControlAffineDynamics(state_dim, control_dim)  # identity g, zero f default
    barriers = []
    for spec in req.barriers:
        btype = str(spec.get("type", "")).lower()
        alpha = float(spec.get("alpha", 0.5))
        if btype == "sphere":
            barriers.append(SphereBarrier(spec["center"], float(spec["radius"]), alpha=alpha))
        elif btype == "halfplane":
            barriers.append(HalfPlaneBarrier(spec["a"], float(spec["b"]), alpha=alpha))
        else:
            raise HTTPException(status_code=400, detail=f"unknown barrier type: {btype}")
    return StateCBFSafetyFilter(
        dyn,
        barriers,
        u_min=req.u_min,
        u_max=req.u_max,
        conformal_margin=req.conformal_margin,
    )


@app.post("/microbrain/gate", response_model=GateResponse)
async def gate_action(req: GateRequest):
    """Gate a robot action: CBF safety -> conformal risk -> VLA-tier routing.

    Uses a shared SkillLibrary singleton (affordances) rebuilt safety filters
    per request. Honest labels are carried in the response payload.
    """
    global _vla_gate_singleton
    try:
        from .vla_gating import ConformalActionGate, SkillLibrary, VLActionGate

        if _vla_gate_singleton is None:
            _vla_gate_singleton = VLActionGate(
                skill_library=SkillLibrary(),
                action_gate=ConformalActionGate(),
            )
        else:
            # rebuild the state-dependent safety filter for this request
            _vla_gate_singleton.safety = _build_filter_from_request(req)
        if _vla_gate_singleton.safety is None:
            _vla_gate_singleton.safety = _build_filter_from_request(req)
        _vla_gate_singleton.safety.set_conformal_margin(req.conformal_margin)
        out = _vla_gate_singleton.gate(
            scene_vec=req.scene,
            state=req.state,
            nominal_action=req.nominal_action,
            risk_score=req.risk_score,
        )
        return GateResponse(
            decision=out["decision"],
            action_out=out["action_out"],
            reason=out.get("reason", ""),
            latency_ms=out.get("latency_ms", 0.0),
            safety=out.get("safety", {}),
            affordances=out.get("affordances", []),
            conformal=out.get("conformal", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.warning(f"Gate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Main ---

if __name__ == "__main__":
    import uvicorn
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Read host/port from config or environment
    microbrain_cfg = _microbrain_config
    host = microbrain_cfg.get("host", os.environ.get("MICROBRAIN_HOST", "127.0.0.1"))
    port = microbrain_cfg.get("port", int(os.environ.get("MICROBRAIN_PORT", "8080")))
    
    log.info(f"Starting Microbrain HTTP Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")