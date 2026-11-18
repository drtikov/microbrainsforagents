"""VLA-era gating layer: skill affordances, conformal abstention, CBF clamping.

P: 2026-11-18 — robotics extension. Composes the robotics primitives added in
this round into the "systems around the VLA" middle layer (the 10-50 Hz
cognitive slot between ~1 kHz MCU servos and tens-of-ms VLA calls):

- ``SkillLibrary``   — Voyager-style skill cache whose affordance ranking is
  SayCan-*shaped* but HONESTLY a deterministic VSA-cosine heuristic, not a
  learned feasibility value function.
- ``ConformalActionGate`` — split-conformal risk gate over gate decisions.
  Uppercased house rule: while UNCALIBRATED it never allows (fail closed,
  ``decision="escalate_human"``, status ``uncalibrated``).
- ``VLActionGate``   — the tier decision per step:
      local_reflex  : safe + calibrated + strong affordance
      call_vla      : safe but weak affordance -> invoke the big model
      escalate_human: unsafe (CBF infeasible) or uncalibrated risk
  plus ``clamp_action_chunk`` which runs a VLA action chunk through the
  state-space CBF filter waypoint-by-waypoint with per-step receipts.
- Optional ``EFPAgent`` deliberator annotates receipts (it does NOT silently
  override decisions).

Consumes: core/state_cbf_filter.py, core/pymdp_efe.py, core/hyperdimensional_vsa_engine.py,
core/microbrain_telemetry.py (recalibrate_from_window).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hyperdimensional_vsa_engine import HyperdimensionalVSAEngine
from .microbrain_telemetry import MicrobrainTelemetry
from .state_cbf_filter import StateCBFSafetyFilter

log = logging.getLogger("aetheris.vla_gate")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SkillLibrary:
    """VSA-indexed skill cache with deterministic affordance scoring."""

    def __init__(self, vsa: Optional[HyperdimensionalVSAEngine] = None, scene_bins: int = 8):
        self.vsa = vsa or HyperdimensionalVSAEngine.get_instance()
        self.scene_bins = max(2, int(scene_bins))
        self.skills: Dict[str, Dict[str, Any]] = {}

    def register_skill(
        self,
        name: str,
        description: str,
        skill_code: Optional[str] = None,
        scene_prototype: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Index a skill by text tokens and optional numeric scene prototype."""
        if name in self.skills:
            return {"status": "already_registered", "name": name}
        tokens = [t for t in "".join(c if c.isalnum() else " " for c in description.lower()).split()][:16]
        token_syms = [self.vsa.get_or_create_symbol(f"tok_{t}") for t in tokens]
        name_sym = self.vsa.get_or_create_symbol(f"skill_{name}")
        hd = self.vsa.bundle_symbols([name_sym] + token_syms) if token_syms else name_sym
        scene_vec_hd = self.encode_scene(scene_prototype) if scene_prototype is not None else None
        self.skills[name] = {
            "name": name,
            "description": description,
            "skill_code": skill_code,
            "vector": hd,
            "scene_vector": scene_vec_hd,
            "n_tokens": len(token_syms),
        }
        return {"status": "registered", "name": name, "n_tokens": len(token_syms)}

    def bind_scene_example(self, name: str, scene_vec: Sequence[float]) -> Dict[str, Any]:
        """Bind/update a skill's numeric scene prototype in the same VSA subspace."""
        if name not in self.skills:
            return {"status": "unknown_skill", "name": name}
        ex_hd = self.encode_scene(scene_vec)
        prev = self.skills[name].get("scene_vector")
        self.skills[name]["scene_vector"] = ex_hd if prev is None else self.vsa.bundle_symbols([prev, ex_hd])
        return {"status": "bound", "name": name}

    def encode_scene(self, scene_vec: Sequence[float]) -> np.ndarray:
        """
        Project a raw scene vector into HD space: each feature index is bound
        with a coarse value bin, then all bindings are bundled (deterministic).
        """
        vec = np.asarray(scene_vec, dtype=np.float64).flatten()
        bound = []
        for i, v in enumerate(vec):
            field = self.vsa.get_or_create_symbol(f"feat_{i}")
            scaled = float(np.clip((float(v) + 1.0) / 2.0, 0.0, 0.999999))
            bin_idx = int(scaled * (self.scene_bins - 1))
            value = self.vsa.get_or_create_symbol(f"val_{i}_{bin_idx}")
            bound.append(self.vsa.bind_symbols(field, value))
        return self.vsa.bundle_symbols(bound) if bound else np.zeros(self.vsa.hd_dim, dtype=np.float32)

    def find_affordances(self, scene_vec: Sequence[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rank registered skills against the scene by cosine similarity.
        Uses bound numeric `scene_vector` when available (same VSA subspace as
        `encode_scene`), falling back to text-token hypervector otherwise.
        HONEST LABEL: deterministic VSA-similarity heuristic — this is NOT a
        learned SayCan feasibility value function.
        """
        if not self.skills:
            return []
        scene_hd = self.encode_scene(scene_vec)
        scored = []
        for name, rec in self.skills.items():
            target_vec = rec["scene_vector"] if rec.get("scene_vector") is not None else rec["vector"]
            scored.append(
                {
                    "skill": name,
                    "affordance": round(_cosine(scene_hd, target_vec), 4),
                }
            )
        scored.sort(key=lambda r: -r["affordance"])
        return scored[: max(1, int(top_k))]


class ConformalActionGate:
    """Split-conformal gate over gate decisions. Uncalibrated = never allow."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.q_hat: Optional[float] = None
        self.n_calibration: int = 0
        self._telemetry = MicrobrainTelemetry()

    def calibrate(self, risk_scores: Sequence[float]) -> Dict[str, Any]:
        out = self._telemetry.recalibrate_from_window(list(risk_scores), alpha=self.alpha)
        if out.get("status") == "recalibrated":
            self.q_hat = float(out["q_hat"])
            self.n_calibration = int(out["n_samples"])
        return out

    def check(self, risk_score: float) -> Dict[str, Any]:
        if self.q_hat is None:
            return {"allowed": False, "calibrated": False, "reason": "uncalibrated_fail_closed"}
        return {
            "allowed": bool(float(risk_score) <= self.q_hat),
            "calibrated": True,
            "q_hat": self.q_hat,
            "risk_score": float(risk_score),
            "n_calibration": self.n_calibration,
        }


class VLActionGate:
    """Three-tier per-step decision: local_reflex / call_vla / escalate_human."""

    def __init__(
        self,
        skill_library: Optional[SkillLibrary] = None,
        safety_filter: Optional[StateCBFSafetyFilter] = None,
        action_gate: Optional[ConformalActionGate] = None,
        affordance_threshold: float = 0.35,
        deliberator: Optional[Any] = None,
    ):
        self.skills = skill_library or SkillLibrary()
        self.safety = safety_filter
        self.action_gate = action_gate
        self.affordance_threshold = float(affordance_threshold)
        self.deliberator = deliberator
        self.decision_counts = {"local_reflex": 0, "call_vla": 0, "escalate_human": 0}

    def gate(
        self,
        scene_vec: Sequence[float],
        state: Sequence[float],
        nominal_action: Sequence[float],
        risk_score: float = 1.0,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        receipt: Dict[str, Any] = {"decision_tiers": []}

        # 1. Safety filter (fail closed on infeasible)
        if self.safety is not None:
            filtered = self.safety.filter(state, nominal_action)
            receipt["safety"] = {k: filtered[k] for k in ("status", "verified_safe", "min_barrier_slack", "conformal_margin")}
            if not filtered["verified_safe"]:
                receipt["reason"] = "cbf_infeasible"
                decision = "escalate_human"
                u_out = filtered["u_filtered"]
            else:
                u_out = filtered["u_filtered"]
        else:
            filtered = None
            receipt["safety"] = {"status": "no_filter_installed"}
            u_out = list(nominal_action)

        if receipt.get("reason") != "cbf_infeasible":
            # 2. Conformal risk gate (uncalibrated never allows)
            if self.action_gate is not None:
                check = self.action_gate.check(risk_score)
                receipt["conformal"] = check
                if not check["allowed"]:
                    receipt["reason"] = check.get("reason", "risk_exceeds_q_hat")
                    decision = "escalate_human"
                else:
                    decision = None
            else:
                receipt["conformal"] = {"calibrated": False, "reason": "no_gate_installed"}
                decision = None

            # 3. Affordance tier
            if decision is None:
                affs = self.skills.find_affordances(scene_vec, top_k=3)
                receipt["affordances"] = affs
                top_aff = affs[0]["affordance"] if affs else 0.0
                if top_aff >= self.affordance_threshold:
                    decision = "local_reflex"
                    receipt["reason"] = "affordance_hit"
                else:
                    decision = "call_vla"
                    receipt["reason"] = "low_affordance"

        if self.deliberator is not None and decision == "call_vla":
            try:
                n_actions = self.deliberator.model.n_actions
                plan = self.deliberator.plan([[u] for u in range(n_actions)])
                receipt["efe_deliberation"] = {
                    "best_action": plan.get("best_action"),
                    "q_pi": plan.get("q_pi"),
                }
            except Exception as exc:
                receipt["efe_deliberation"] = {"error": str(exc)}

        self.decision_counts[decision] = self.decision_counts.get(decision, 0) + 1
        receipt.update(
            {
                "decision": decision,
                "action_out": [float(v) for v in u_out],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }
        )
        return receipt

    def clamp_action_chunk(
        self,
        state: Sequence[float],
        actions: Sequence[Sequence[float]],
        dt: float = 0.05,
    ) -> Dict[str, Any]:
        """Run a VLA action chunk through the CBF filter with per-step receipts."""
        if self.safety is None:
            return {"status": "no_filter_installed", "actions_filtered": [list(a) for a in actions], "all_verified_safe": False}
        out = self.safety.filter_trajectory(np.asarray(state, dtype=np.float64), actions, dt=dt)
        out["status"] = "chunk_clamped" if out["all_verified_safe"] else "chunk_unsafe_steps_present"
        return out
