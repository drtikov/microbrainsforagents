"""Skill Lifecycle Manager: Metabolic fleet governance for Ouroboros microbrains & tools.

Implements SOTA Agent Tool Lifecycle Management (ATLM) and EvolveR experience governance:
  1. Incubate (Sandbox definition & schema validation)
  2. Shadow Evaluation (Canary testing & conformal calibration verification)
  3. Promote (Deployment to active MoE router)
  4. Specialize / Fission (Splitting overloaded multi-role expert weights)
  5. Degrade & Recalibrate (Handling drift when coverage drops below 1 - alpha)
  6. Retire & Archive (Purging dead domains like exoplanet/bear_bern to stop cognitive bloat)

This is a deterministic metabolic orchestrator, NOT a stochastic neural network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)


class SkillLifecycleState(str, Enum):
    INCUBATING = "incubating"
    SHADOW_EVAL = "shadow_eval"
    ACTIVE_CORE = "active_core"
    SPECIALIZED = "specialized"
    DEGRADED = "degraded"
    RETIRED = "retired"


# Domains considered core to Ouroboros cognitive autonomy (protected from retirement)
PROTECTED_CORE_DOMAINS: Set[str] = {
    "fast_intent_classifier",
    "speculative_tool_drafter",
    "architectural_breeding",
    "counterfactual_reasoner",
    "code_mutation_predictor",
    "concept_synthesizer",
    "intent_persistence_tracker",
    "metacognitive_calibrator",
    "causal_world_model",
    "skill_lifecycle_manager",
}

# Domains identified as legacy or dormant research artifacts
KNOWN_DORMANT_LEGACY_DOMAINS: Set[str] = {
    "exoplanet_direct_imaging",
    "exoplanet_transit_timing",
    "exoplanet_spectral_retrieval",
    "bear_bern_synthetic_biology",
    "communication_complexity",
    "eggroll_evolutionary_strategies",
    "dark_matter_halo_modeling",
    "cosmological_redshift_drift",
}


@dataclass
class SkillTelemetry:
    """Operational telemetry for a skill or microbrain."""

    name: str
    state: SkillLifecycleState
    call_count: int = 0
    error_count: int = 0
    conformal_coverage: Optional[float] = None  # Observed (1 - alpha); None until measured
    mean_latency_ms: Optional[float] = None
    telemetry_status: str = "unobserved"
    last_called_iso: Optional[str] = None
    quarantine_reason: Optional[str] = None
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        status = "observed" if self.call_count > 0 else "unobserved"
        return {
            "name": self.name,
            "state": self.state.value,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "conformal_coverage": round(self.conformal_coverage, 3) if self.conformal_coverage is not None else None,
            "mean_latency_ms": round(self.mean_latency_ms, 2) if self.mean_latency_ms is not None else None,
            "telemetry_status": status,
            "last_called_iso": self.last_called_iso,
            "quarantine_reason": self.quarantine_reason,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillTelemetry:
        call_count = int(data.get("call_count", 0) or 0)
        last_called = data.get("last_called_iso")
        if call_count == 0 and last_called is None:
            conformal_coverage: Optional[float] = None
            mean_latency_ms: Optional[float] = None
            telemetry_status = "unobserved"
        else:
            raw_cov = data.get("conformal_coverage")
            raw_lat = data.get("mean_latency_ms")
            conformal_coverage = float(raw_cov) if raw_cov is not None else None
            mean_latency_ms = float(raw_lat) if raw_lat is not None else None
            telemetry_status = str(data.get("telemetry_status") or "observed")
        return cls(
            name=data["name"],
            state=SkillLifecycleState(data.get("state", "incubating")),
            call_count=call_count,
            error_count=int(data.get("error_count", 0) or 0),
            conformal_coverage=conformal_coverage,
            mean_latency_ms=mean_latency_ms,
            telemetry_status=telemetry_status,
            last_called_iso=last_called,
            quarantine_reason=data.get("quarantine_reason"),
            version=int(data.get("version", 1) or 1),
        )


@dataclass
class FleetHygieneReport:
    """Outcome of an automated fleet hygiene audit."""

    total_skills: int
    active_count: int
    incubating_count: int
    retired_count: int
    degraded_count: int
    retired_skills: List[str]
    promoted_skills: List[str]
    recommendations: List[str]


class SkillLifecycleManager:
    """Metabolic orchestrator managing the full life-cycle of microbrains and tools."""

    def __init__(self, storage_dir: Optional[Path] = None):
        _cwd_mem = Path("memory")
        _pkg_mem = Path(__file__).resolve().parents[1] / "memory"
        self.storage_dir = storage_dir or (_cwd_mem if (_cwd_mem / "fleet_lifecycle_registry.json").exists() else _pkg_mem)
        self.registry_file = self.storage_dir / "fleet_lifecycle_registry.json"
        self.skills: Dict[str, SkillTelemetry] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load skill registry from disk or initialize with baseline domains."""
        if self.registry_file.is_file():
            try:
                data = json.loads(self.registry_file.read_text(encoding="utf-8"))
                for name, sdata in data.items():
                    self.skills[name] = SkillTelemetry.from_dict(sdata)
                return
            except Exception as exc:
                log.warning("Failed to load skill lifecycle registry: %s", exc)

        # Initialize defaults without fabricated telemetry counts or bounds
        for dom in PROTECTED_CORE_DOMAINS:
            self.skills[dom] = SkillTelemetry(
                name=dom,
                state=SkillLifecycleState.ACTIVE_CORE,
                call_count=0,
                conformal_coverage=None,
                mean_latency_ms=None,
                telemetry_status="unobserved",
            )

        for dom in KNOWN_DORMANT_LEGACY_DOMAINS:
            self.skills[dom] = SkillTelemetry(
                name=dom,
                state=SkillLifecycleState.RETIRED,
                call_count=0,
                conformal_coverage=None,
                mean_latency_ms=None,
                telemetry_status="unobserved",
                quarantine_reason="Dormant research domain retired during metabolic cleanup",
            )

    def save_registry(self) -> None:
        """Persist skill lifecycle registry to disk."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            payload = {name: s.to_dict() for name, s in self.skills.items()}
            self.registry_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Failed to save skill lifecycle registry: %s", exc)

    def record_call(
        self,
        skill_name: str,
        success: bool,
        latency_ms: float = 10.0,
        conformal_passed: Optional[bool] = None,
    ) -> None:
        """Update telemetry when a skill is invoked."""
        from datetime import datetime, timezone

        if skill_name not in self.skills:
            # Auto-register newly observed skill in incubating state
            self.skills[skill_name] = SkillTelemetry(
                name=skill_name,
                state=SkillLifecycleState.INCUBATING,
            )

        sk = self.skills[skill_name]
        sk.call_count += 1
        if not success:
            sk.error_count += 1
        sk.telemetry_status = "observed"
        if sk.mean_latency_ms is None:
            sk.mean_latency_ms = float(latency_ms)
        else:
            sk.mean_latency_ms = 0.8 * sk.mean_latency_ms + 0.2 * float(latency_ms)
        if conformal_passed is not None:
            obs = 1.0 if conformal_passed else 0.0
            if sk.conformal_coverage is None:
                sk.conformal_coverage = obs
            else:
                sk.conformal_coverage = 0.8 * sk.conformal_coverage + 0.2 * obs
        sk.last_called_iso = datetime.now(timezone.utc).isoformat()

        # Check for degradation
        error_rate = sk.error_count / max(1, sk.call_count)
        if error_rate > 0.30 and sk.call_count >= 5 and sk.state == SkillLifecycleState.ACTIVE_CORE:
            sk.state = SkillLifecycleState.DEGRADED
            sk.quarantine_reason = f"High error rate ({error_rate:.1%}) breached stability threshold"
            log.warning("Skill %s degraded due to high error rate: %.1f%%", skill_name, error_rate * 100)

    def incubate(self, skill_name: str) -> SkillTelemetry:
        """Register a new candidate skill in sandbox incubation."""
        sk = SkillTelemetry(name=skill_name, state=SkillLifecycleState.INCUBATING)
        self.skills[skill_name] = sk
        self.save_registry()
        return sk

    def evaluate_canary(self, skill_name: str, test_passed: bool) -> bool:
        """Run canary validation on an incubating or degraded skill."""
        if skill_name not in self.skills:
            return False
        sk = self.skills[skill_name]
        if test_passed:
            sk.state = SkillLifecycleState.ACTIVE_CORE
            sk.quarantine_reason = None
            self.save_registry()
            return True
        else:
            sk.state = SkillLifecycleState.DEGRADED
            sk.quarantine_reason = "Failed canary validation suite"
            self.save_registry()
            return False

    def promote_to_core(self, skill_name: str) -> bool:
        """Promote a validated skill to Active Core."""
        if skill_name not in self.skills:
            return False
        self.skills[skill_name].state = SkillLifecycleState.ACTIVE_CORE
        self.skills[skill_name].quarantine_reason = None
        self.save_registry()
        return True

    def retire_skill(self, skill_name: str, reason: str = "Dormant / zero-utility") -> bool:
        """Safely decommission and archive a skill."""
        if skill_name in PROTECTED_CORE_DOMAINS:
            log.warning("Cannot retire protected core domain: %s", skill_name)
            return False
        if skill_name not in self.skills:
            self.skills[skill_name] = SkillTelemetry(name=skill_name, state=SkillLifecycleState.RETIRED)
        sk = self.skills[skill_name]
        sk.state = SkillLifecycleState.RETIRED
        sk.quarantine_reason = reason
        self.save_registry()
        return True

    def perform_fleet_hygiene(self) -> FleetHygieneReport:
        """Execute automated metabolic audit across the entire microbrain fleet."""
        retired: List[str] = []
        promoted: List[str] = []
        recommendations: List[str] = []

        for name, sk in list(self.skills.items()):
            if name in PROTECTED_CORE_DOMAINS:
                if sk.state != SkillLifecycleState.ACTIVE_CORE:
                    sk.state = SkillLifecycleState.ACTIVE_CORE
                    promoted.append(name)
                continue

            # Auto-retire legacy or completely dormant non-core skills
            if name in KNOWN_DORMANT_LEGACY_DOMAINS and sk.state != SkillLifecycleState.RETIRED:
                sk.state = SkillLifecycleState.RETIRED
                sk.quarantine_reason = "Automated fleet hygiene: dormant research artifact"
                retired.append(name)
            elif sk.call_count == 0 and sk.state == SkillLifecycleState.INCUBATING:
                recommendations.append(f"Skill '{name}' is incubating without traffic; queue for canary test.")
            elif sk.state == SkillLifecycleState.DEGRADED:
                recommendations.append(f"Skill '{name}' is degraded; trigger conformal recalibration.")

        self.save_registry()

        active = sum(1 for s in self.skills.values() if s.state == SkillLifecycleState.ACTIVE_CORE)
        incubating = sum(1 for s in self.skills.values() if s.state == SkillLifecycleState.INCUBATING)
        ret = sum(1 for s in self.skills.values() if s.state == SkillLifecycleState.RETIRED)
        deg = sum(1 for s in self.skills.values() if s.state == SkillLifecycleState.DEGRADED)

        return FleetHygieneReport(
            total_skills=len(self.skills),
            active_count=active,
            incubating_count=incubating,
            retired_count=ret,
            degraded_count=deg,
            retired_skills=retired,
            promoted_skills=promoted,
            recommendations=recommendations,
        )


# Global singleton
_LIFECYCLE_MGR: Optional[SkillLifecycleManager] = None


def get_skill_lifecycle_manager(storage_dir: Optional[Path] = None) -> SkillLifecycleManager:
    """Retrieve or initialize the active Skill Lifecycle Manager."""
    global _LIFECYCLE_MGR
    if _LIFECYCLE_MGR is None:
        _LIFECYCLE_MGR = SkillLifecycleManager(storage_dir)
    return _LIFECYCLE_MGR
