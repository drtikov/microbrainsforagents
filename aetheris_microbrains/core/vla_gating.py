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

import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hyperdimensional_vsa_engine import HyperdimensionalVSAEngine
from .microbrain_telemetry import MicrobrainTelemetry
from .pymdp_efe import (
    EFPAgent,
    MultiModalEFPAgent,
    expected_free_energy,
    expected_free_energy_multimodal,
)
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
    """Three-tier (plus optional active-inference epistemic probe) per-step decision."""

    def __init__(
        self,
        skill_library: Optional[SkillLibrary] = None,
        safety_filter: Optional[StateCBFSafetyFilter] = None,
        action_gate: Optional[ConformalActionGate] = None,
        affordance_threshold: float = 0.35,
        deliberator: Optional[Any] = None,
        enable_epistemic_probe: bool = False,
        epistemic_probes: Optional[Dict[str, Dict[str, Any]]] = None,
        epistemic_entropy_threshold: float = 0.05,
    ):
        self.skills = skill_library or SkillLibrary()
        self.safety = safety_filter
        self.action_gate = action_gate
        self.affordance_threshold = float(affordance_threshold)
        self.deliberator = deliberator
        self.enable_epistemic_probe = bool(enable_epistemic_probe)
        self.epistemic_probes = dict(epistemic_probes) if epistemic_probes is not None else None
        self.epistemic_entropy_threshold = float(epistemic_entropy_threshold)
        self.decision_counts = {
            "local_reflex": 0,
            "call_vla": 0,
            "escalate_human": 0,
            "epistemic_probe": 0,
        }

    def _select_epistemic_probe(
        self,
        state: Sequence[float],
        fallback_action: Sequence[float],
    ) -> Optional[Dict[str, Any]]:
        """Query deliberator for highest-information-gain CBF-safe active perception probe."""
        if self.deliberator is None:
            return None

        candidates: List[Dict[str, Any]] = []
        if self.epistemic_probes is not None:
            if not self.epistemic_probes:
                return None
            for key, spec in self.epistemic_probes.items():
                if not isinstance(spec, dict):
                    continue
                if "policy" in spec:
                    pol = list(spec["policy"])
                    action_idx = pol[0]
                elif "action_idx" in spec:
                    action_idx = int(spec["action_idx"])
                    pol = [action_idx]
                else:
                    try:
                        action_idx = int(key)
                    except (TypeError, ValueError):
                        continue
                    pol = [action_idx]
                candidates.append(
                    {
                        "pol": pol,
                        "action_idx": action_idx,
                        "label": str(spec.get("label", spec.get("name", key))),
                        "raw_vec": spec.get("probe_action", spec.get("action")),
                    }
                )
        else:
            model = getattr(self.deliberator, "model", None)
            n_actions = int(getattr(model, "n_actions", 0))
            if n_actions <= 0:
                return None
            for u in range(n_actions):
                candidates.append(
                    {
                        "pol": [u],
                        "action_idx": u,
                        "label": f"probe_{u}",
                        "raw_vec": None,
                    }
                )

        if not candidates:
            return None

        try:
            plan = self.deliberator.plan([c["pol"] for c in candidates])
        except Exception:
            return None

        G_list = plan.get("G", [0.0] * len(candidates))
        scored: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            g_val = float(G_list[idx]) if idx < len(G_list) else 0.0
            epistemic = 0.0
            novelty = 0.0
            if isinstance(self.deliberator, MultiModalEFPAgent):
                g_val, tr = expected_free_energy_multimodal(
                    self.deliberator.model,
                    self.deliberator.belief_factors,
                    cand["pol"],
                    pA=self.deliberator.pA,
                )
                epistemic = float(sum(s["epistemic"] for s in tr["per_step"]))
                novelty = float(sum(s["novelty"] for s in tr["per_step"]))
            elif isinstance(self.deliberator, EFPAgent):
                g_val, tr = expected_free_energy(
                    self.deliberator.model,
                    self.deliberator.belief,
                    cand["pol"],
                )
                epistemic = float(sum(s["epistemic"] for s in tr["per_step"]))
            else:
                # Duck-typed deliberator fallback
                ig_list = plan.get("info_gains")
                if isinstance(ig_list, (list, tuple)) and idx < len(ig_list):
                    epistemic = float(ig_list[idx])
                elif cand["action_idx"] == plan.get("best_action"):
                    epistemic = float(plan.get("info_gain", 0.0))

            info_gain = epistemic + novelty
            if info_gain <= 0.0 or info_gain < self.epistemic_entropy_threshold:
                continue

            if cand["raw_vec"] is not None:
                if self.safety is not None:
                    pf = self.safety.filter(state, cand["raw_vec"])
                    if not pf.get("verified_safe", False):
                        continue
                    safe_vec = [float(v) for v in pf["u_filtered"]]
                else:
                    safe_vec = [float(v) for v in cand["raw_vec"]]
            else:
                safe_vec = [float(v) for v in fallback_action]

            scored.append(
                {
                    "action_idx": cand["action_idx"],
                    "label": cand["label"],
                    "safe_vec": safe_vec,
                    "info_gain": info_gain,
                    "epistemic": epistemic,
                    "novelty": novelty,
                    "g_val": g_val,
                }
            )

        if not scored:
            return None

        best = max(scored, key=lambda item: (item["info_gain"], -item["g_val"]))
        return {
            "probe_action": best["safe_vec"],
            "probe_label": best["label"],
            "efe_deliberation": {
                "best_action": best["action_idx"],
                "plan_best_action": plan.get("best_action"),
                "q_pi": plan.get("q_pi"),
                "G": plan.get("G"),
                "info_gain": round(best["info_gain"], 6),
                "epistemic": round(best["epistemic"], 6),
                "novelty": round(best["novelty"], 6),
                "belief_entropy": getattr(self.deliberator, "belief_entropy", None),
            },
        }

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
                    cbf_safe = filtered is None or bool(filtered.get("verified_safe", False))
                    if self.enable_epistemic_probe and cbf_safe and self.deliberator is not None:
                        probe = self._select_epistemic_probe(state, u_out)
                        if probe is not None:
                            decision = "epistemic_probe"
                            receipt["reason"] = "epistemic_probe_high_uncertainty"
                            receipt["probe_action"] = probe["probe_action"]
                            receipt["probe_label"] = probe["probe_label"]
                            receipt["efe_deliberation"] = probe["efe_deliberation"]
                            u_out = probe["probe_action"]
                        else:
                            receipt["reason"] = check.get("reason", "risk_exceeds_q_hat")
                            decision = "escalate_human"
                    else:
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

    def smooth_action_chunk(
        self,
        state: Sequence[float],
        actions: Sequence[Sequence[float]],
        dt: float = 0.05,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Smooth and safety-filter an action chunk via ActionChunkSmoother with clamp fallback."""
        try:
            from .action_chunk_smoother import ActionChunkSmoother
        except ImportError:
            ActionChunkSmoother = None  # type: ignore[assignment]

        if ActionChunkSmoother is not None:
            try:
                init_sig = inspect.signature(ActionChunkSmoother.__init__)
                init_params = init_sig.parameters
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()
                )
                init_kwargs: Dict[str, Any] = {}
                if "safety_filter" in init_params or accepts_var_kw:
                    init_kwargs["safety_filter"] = self.safety
                for k, v in kwargs.items():
                    if k in init_params or accepts_var_kw:
                        init_kwargs[k] = v
                smoother = ActionChunkSmoother(**init_kwargs)

                for method_name in ("smooth", "smooth_chunk", "filter_and_smooth"):
                    method = getattr(smoother, method_name, None)
                    if callable(method):
                        m_sig = inspect.signature(method)
                        m_params = m_sig.parameters
                        m_var_kw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in m_params.values()
                        )
                        call_kwargs: Dict[str, Any] = {}
                        if "dt" in m_params or m_var_kw:
                            call_kwargs["dt"] = dt
                        for k, v in kwargs.items():
                            if (k in m_params or m_var_kw) and k not in init_kwargs:
                                call_kwargs[k] = v
                        return method(state, actions, **call_kwargs)
            except Exception:
                pass

        out = self.clamp_action_chunk(state, actions, dt=dt)
        out.setdefault("fallback", "clamp_action_chunk")
        return out
