"""Active Inference Engine — Expected Free Energy (EFE) minimization and homeostatic drive regulation.

P: 2026-11-10 - honesty rewrite (tier-2 audit finding #30).  The old
module shipped two APIs nobody called (a dict-form
``predict_state_update(state, action)`` that was keyword matching over
inline magic constants — e.g. "reckless/bypass actions increase progress
by 0.25" — presented in its docstring as a "pure generative transition
model", and a positional ``run_active_inference_simulation(initial_state,
candidate_actions, ...)`` form), while the pinned consumer contract —
persona-drive simulation over the four named personas plus file-backed
Bayesian persona memory — was a stub.  The live surfaces are now the
ones the kernel and the tests actually pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .atomic import atomic_read, atomic_write

DEFAULT_PERSONA_MEMORY_PATH = "memory/store/persona_memory.json"

# Bounded recent-outcome window per persona (pinned contract).
_RECENT_WINDOW = 10

# Neutral prior used for untried personas.
NEUTRAL_PRIOR = 0.5


def _clamp01(val: Any) -> float:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return NEUTRAL_PRIOR
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Persona profiles and transition priors (declared model parameters)
# ---------------------------------------------------------------------------


PERSONA_PROFILES: Dict[str, Dict[str, Any]] = {
    # Declared archetype parameters for the persona simulation.  These are
    # the generative model's priors about each persona's disposition —
    # declared model parameters, not measurements.  safety_weight /
    # curiosity_weight scale the shared transition priors; risk_bias /
    # curiosity_bias enter the EFE scoring.
    "aura": {
        "description": "cautious steward",
        "safety_weight": 1.2,
        "curiosity_weight": 0.7,
        "risk_bias": 0.2,
        "curiosity_bias": 0.4,
    },
    "zeno": {
        "description": "balanced analyst",
        "safety_weight": 1.0,
        "curiosity_weight": 1.0,
        "risk_bias": 0.5,
        "curiosity_bias": 0.5,
    },
    "iris": {
        "description": "curious explorer",
        "safety_weight": 0.8,
        "curiosity_weight": 1.3,
        "risk_bias": 0.35,
        "curiosity_bias": 0.85,
    },
    "eris": {
        "description": "decisive executor",
        "safety_weight": 0.6,
        "curiosity_weight": 0.7,
        "risk_bias": 0.75,
        "curiosity_bias": 0.3,
    },
}

ACTION_TRANSITION_PRIORS: Dict[str, Dict[str, float]] = {
    # Declared generative-model priors: how each action category is
    # expected to shift the drive/homeostasis state.  Heuristic priors of
    # the model, not measurements.  An action with no prior predicts no
    # shift (no evidence, no claimed effect).
    "logical_breakdown": {"safety": 0.06, "curiosity": 0.04, "integrity": 0.02, "homeostasis": 0.03},
    "provocative_question": {"safety": -0.10, "curiosity": 0.16, "integrity": 0.0, "homeostasis": -0.04},
    "careful_repair": {"safety": 0.12, "curiosity": 0.02, "integrity": 0.10, "homeostasis": 0.06},
    "focused_execution": {"safety": -0.03, "curiosity": -0.02, "integrity": 0.08, "homeostasis": 0.04},
    "generic_action": {"safety": 0.0, "curiosity": 0.0, "integrity": 0.0, "homeostasis": 0.0},
}

_DEFAULT_CATEGORY = "generic_action"


def _action_category(text: str) -> str:
    """Map goal/prompt text to an action category.

    Declared keyword heuristic over an observable property of the task
    text; it is the model's category prior, not a measured intent.
    """
    t = str(text or "").lower()
    if any(k in t for k in ("fix", "repair", "bug", "error", "debug", "broken")):
        return "careful_repair"
    if any(k in t for k in ("how", "why", "analy", "breakdown", "explain", "question")):
        return "logical_breakdown"
    if any(k in t for k in ("build", "implement", "create", "write", "optimi", "deploy")):
        return "focused_execution"
    return _DEFAULT_CATEGORY


def predict_state_update(
    action_name: str,
    persona: str,
    drives: Dict[str, float],
    homeostasis: float,
) -> Tuple[Dict[str, float], float]:
    """Persona-parameterised transition P(drives', homeostasis' | action, persona).

    P: 2026-11-10 - the old ``predict_state_update(state, action)`` was
    keyword matching over inline magic constants ("reckless/bypass
    actions increase progress by 0.25") with a docstring claiming a
    "pure generative transition model", and it had zero callers.  The
    pinned contract is this persona-drive transition: the declared
    category prior (``ACTION_TRANSITION_PRIORS``) is scaled by the
    persona's declared trait weights (``PERSONA_PROFILES``).  Unknown
    actions predict no shift — no prior, no claimed effect.
    """
    priors = ACTION_TRANSITION_PRIORS.get(
        str(action_name or "").strip().lower(), ACTION_TRANSITION_PRIORS[_DEFAULT_CATEGORY]
    )
    profile = PERSONA_PROFILES.get(str(persona or "").strip().lower(), {})
    safety_w = float(profile.get("safety_weight", 1.0))
    curiosity_w = float(profile.get("curiosity_weight", 1.0))

    pred: Dict[str, float] = {}
    for key in ("safety", "curiosity", "integrity"):
        base = _clamp01((drives or {}).get(key, NEUTRAL_PRIOR))
        shift = float(priors.get(key, 0.0))
        if key == "safety":
            shift *= safety_w
        elif key == "curiosity":
            shift *= curiosity_w
        pred[key] = _clamp01(base + shift)
    # Carry over any extra observed drives unchanged.
    for key, val in (drives or {}).items():
        if key not in pred:
            pred[key] = val

    pred_homeostasis = _clamp01(_clamp01(homeostasis) + float(priors.get("homeostasis", 0.0)))
    rounded = {k: round(v, 4) for k, v in pred.items()}
    return rounded, round(pred_homeostasis, 4)


# ---------------------------------------------------------------------------
# Active Inference Cluster
# ---------------------------------------------------------------------------


class ActiveInferenceCluster:
    """Homeostatic drive cluster and Expected Free Energy (EFE) policy selector."""

    DEFAULT_DRIVES = (
        "survival",
        "epistemic_curiosity",
        "task_completion",
        "self_improvement",
    )

    def __init__(self, drive_names: Optional[List[str]] = None) -> None:
        names = drive_names if drive_names is not None else list(self.DEFAULT_DRIVES)
        self.drives: Dict[str, float] = {name: NEUTRAL_PRIOR for name in names}
        self.efe_history: List[List[Dict[str, Any]]] = []

    @staticmethod
    def _clamp(val: float) -> float:
        return max(0.0, min(1.0, float(val)))

    def update_drives_from_system_state(
        self,
        token_usage_pct: float = 0.0,
        recent_failures: int = 0,
        novel_discoveries: int = 0,
        task_progress: float = 0.5,
    ) -> Dict[str, float]:
        # Survival drive rises with resource pressure and recent failures
        survival_pressure = 0.35 * float(token_usage_pct) + 0.12 * min(5, int(recent_failures))
        self.drives["survival"] = self._clamp(0.35 + survival_pressure)

        # Epistemic curiosity rises when discoveries are scarce, drops when satiated
        if int(novel_discoveries) > 0:
            curiosity = max(0.1, 0.5 - 0.08 * int(novel_discoveries))
        else:
            curiosity = min(0.95, self.drives.get("epistemic_curiosity", NEUTRAL_PRIOR) + 0.15)
        self.drives["epistemic_curiosity"] = self._clamp(curiosity)

        # Task completion drive rises when progress is low
        self.drives["task_completion"] = self._clamp(1.0 - 0.6 * float(task_progress))

        # Self improvement drive rises when failures occur or survival is stable
        self.drives["self_improvement"] = self._clamp(
            0.4 + 0.08 * min(5, int(recent_failures))
        )
        return dict(self.drives)

    def evaluate_candidate_policies(
        self, policies: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Compute Expected Free Energy G(pi) for each candidate policy (lower G is better)."""
        if not policies:
            empty = {"name": "noop", "efe": 0.0, "pragmatic_value": 0.0, "epistemic_value": 0.0, "risk_penalty": 0.0}
            return empty, [empty]

        evaluated: List[Dict[str, Any]] = []
        survival_w = self.drives.get("survival", NEUTRAL_PRIOR)
        curiosity_w = self.drives.get("epistemic_curiosity", NEUTRAL_PRIOR)

        for pol in policies:
            name = str(pol.get("name", "unnamed"))
            outcomes = pol.get("expected_outcomes") or {}
            epistemic_raw = float(pol.get("epistemic_value", 0.0))
            risk_raw = float(pol.get("risk", 0.0))

            pragmatic_val = 0.0
            if isinstance(outcomes, dict):
                for drive_name, gain in outcomes.items():
                    w = self.drives.get(drive_name, NEUTRAL_PRIOR)
                    pragmatic_val += w * float(gain)

            epistemic_val = curiosity_w * epistemic_raw
            # Risk penalty scales strongly with survival drive
            risk_penalty = (0.5 + 2.0 * survival_w) * risk_raw
            if survival_w > 0.7 and outcomes.get("survival", 0.0) > 0.5:
                pragmatic_val += survival_w * float(outcomes.get("survival", 0.0))

            efe = round(risk_penalty - pragmatic_val - epistemic_val, 6)
            evaluated.append(
                {
                    "name": name,
                    "efe": efe,
                    "pragmatic_value": round(pragmatic_val, 4),
                    "epistemic_value": round(epistemic_val, 4),
                    "risk_penalty": round(risk_penalty, 4),
                    "policy": pol,
                }
            )

        evaluated.sort(key=lambda item: item["efe"])
        self.efe_history.append(evaluated)
        return evaluated[0], evaluated

    def run_active_inference(
        self, initial_state: Dict[str, Any], goal: str = ""
    ) -> Dict[str, Any]:
        """Run the four-persona active-inference simulation for the current state and goal.

        P: 2026-11-10 - the kernel and test_causal_active_inference both
        pinned this method and it did not exist (AttributeError), so
        kernel.active_persona always fell back to the hardcoded 'aura'
        default.  The simulation predicts each persona's drive shift
        under the declared transition priors, scores it with an EFE
        heuristic, and selects the minimal-EFE persona.
        """
        drives = dict(initial_state.get("drives") or {}) if isinstance(initial_state, dict) else {}
        homeostasis = initial_state.get("homeostasis", 0.7) if isinstance(initial_state, dict) else 0.7
        category = _action_category(str(goal or ""))
        simulations, bbu_log, winner = _simulate_personas(drives, homeostasis, category)
        return {
            "status": "complete",
            "winner": winner,
            "winning_simulation_id": winner["simulation_id"],
            "simulations": simulations,
            "bbu_log": bbu_log,
            "action_category": category,
        }

    def run_monte_carlo_efe_rollout(
        self,
        initial_state: Dict[str, Any],
        goal: str = "",
        n_particles: int = 256,
        horizon: int = 8,
        noise_scale: float = 0.04,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Parallel 1,024-Particle Monte-Carlo Tree Rollout of Expected Free Energy
        G(pi) on the GT 710 OpenCL GPU (`4 personas x 256 particles x 8 horizon steps`).
        Computes Pragmatic Risk, Epistemic Information Gain (particle dispersion), and CVaR_0.10 tail risk.
        """
        import numpy as np
        from .opencl_microbrain_accelerator import get_opencl_accelerator

        drives = dict(initial_state.get("drives") or {}) if isinstance(initial_state, dict) else {}
        homeostasis = _clamp01(initial_state.get("homeostasis", 0.7) if isinstance(initial_state, dict) else 0.7)
        init_vec = np.array(
            [
                _clamp01(drives.get("safety", NEUTRAL_PRIOR)),
                _clamp01(drives.get("curiosity", NEUTRAL_PRIOR)),
                _clamp01(drives.get("integrity", NEUTRAL_PRIOR)),
                homeostasis,
            ],
            dtype=np.float32,
        )

        category = _action_category(str(goal or ""))
        priors = ACTION_TRANSITION_PRIORS.get(category, ACTION_TRANSITION_PRIORS[_DEFAULT_CATEGORY])
        base_shift = np.array(
            [
                float(priors.get("safety", 0.0)),
                float(priors.get("curiosity", 0.0)),
                float(priors.get("integrity", 0.0)),
                float(priors.get("homeostasis", 0.0)),
            ],
            dtype=np.float32,
        )

        personas = list(PERSONA_PROFILES.keys())
        policy_shifts = np.tile(base_shift, (len(personas), 1))
        persona_weights = np.zeros((len(personas), 4), dtype=np.float32)
        belief_counts = np.ones((len(personas), 2), dtype=np.float32)
        pmem = get_persona_memory()

        for idx, p_name in enumerate(personas):
            prof = PERSONA_PROFILES[p_name]
            persona_weights[idx] = [
                float(prof.get("safety_weight", 1.0)),
                float(prof.get("curiosity_weight", 1.0)),
                1.0,
                1.0,
            ]
            # Dirichlet / Beta observation pseudo-counts from Bayesian persona memory
            att = float(pmem._attempts.get(p_name, 0.0))
            succ = float(pmem._successes.get(p_name, 0.0))
            belief_counts[idx, 0] = max(0.5, 1.0 + succ)
            belief_counts[idx, 1] = max(0.5, 1.0 + max(0.0, att - succ))

            # Add persona-specific exploratory/conservative drift when base_shift is zero
            if np.allclose(base_shift, 0.0):
                policy_shifts[idx] = [
                    0.03 * (1.0 - float(prof.get("risk_bias", 0.5))),
                    0.04 * float(prof.get("curiosity_bias", 0.5)),
                    0.02,
                    0.02,
                ]

        target_prefs = np.array([0.85, 0.75, 0.85, 0.80], dtype=np.float32)
        acc = get_opencl_accelerator()
        mc_res = acc.active_inference_mc_rollout(
            init_drives=init_vec,
            policy_shifts=policy_shifts,
            persona_weights=persona_weights,
            target_prefs=target_prefs,
            n_particles=n_particles,
            horizon=horizon,
            noise_scale=noise_scale,
            belief_counts=belief_counts,
            force_gpu=force_gpu,
        )

        ranked = []
        for idx, m in enumerate(mc_res["policy_metrics"]):
            p_name = personas[idx]
            ranked.append({
                "persona": p_name,
                "efe": m["efe"],
                "pragmatic_risk": m["pragmatic_risk"],
                "epistemic_info_gain": m["epistemic_info_gain"],
                "cvar_tail_risk": m["cvar_tail_risk"],
                "mean_final_drives": m["mean_final_drives"],
            })
        ranked.sort(key=lambda item: item["efe"])

        return {
            "status": "mc_efe_complete",
            "backend": mc_res["backend"],
            "device": mc_res["device"],
            "total_particles": mc_res["total_particles"],
            "horizon": horizon,
            "action_category": category,
            "winning_persona": ranked[0]["persona"],
            "winning_efe": ranked[0]["efe"],
            "ranked_personas": ranked,
        }


def _simulate_personas(
    drives: Optional[Dict[str, float]],
    homeostasis: Any,
    action_category: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Shared four-persona simulation machinery.

    EFE heuristic (declared): G = homeostatic disequilibrium + risk
    exposure − epistemic gain, with the persona's recorded performance
    (its Bayesian adaptive weight) modulating the epistemic term.
    """
    merged: Dict[str, float] = {
        "safety": NEUTRAL_PRIOR,
        "curiosity": NEUTRAL_PRIOR,
        "integrity": NEUTRAL_PRIOR,
    }
    for key, val in (drives or {}).items():
        try:
            merged[key] = _clamp01(val)
        except (TypeError, ValueError):
            continue
    homeo = _clamp01(homeostasis if homeostasis is not None else 0.7)

    memory = get_persona_memory()
    simulations: List[Dict[str, Any]] = []
    bbu_log: List[str] = [
        f"Bayesian Belief Updater: observed drives={merged} homeostasis={homeo} "
        f"action_category={action_category}"
    ]

    for persona, profile in PERSONA_PROFILES.items():
        pred_drives, pred_homeo = predict_state_update(action_category, persona, merged, homeo)
        weight = memory.adaptive_weight(persona, base_utility=1.0)
        disequilibrium = (1.0 - pred_homeo) + 0.5 * (1.0 - pred_drives.get("integrity", NEUTRAL_PRIOR))
        risk_exposure = float(profile.get("risk_bias", NEUTRAL_PRIOR)) * max(
            0.0, 0.5 - pred_drives.get("safety", NEUTRAL_PRIOR)
        )
        epistemic_gain = (
            float(profile.get("curiosity_bias", NEUTRAL_PRIOR))
            * pred_drives.get("curiosity", NEUTRAL_PRIOR)
            * weight
        )
        efe = round(disequilibrium + risk_exposure - epistemic_gain, 6)
        simulations.append(
            {
                "simulation_id": persona,
                "persona": persona,
                "description": profile.get("description", ""),
                "pred_drives": pred_drives,
                "pred_homeostasis": pred_homeo,
                "efe": efe,
                "adaptive_weight": round(weight, 4),
                "plan": _persona_plan(persona, profile, action_category),
            }
        )
        bbu_log.append(
            f"Bayesian Belief Updater [{persona}]: predicted drives={pred_drives} "
            f"homeostasis={pred_homeo} adaptive_weight={weight:.3f} -> EFE {efe}"
        )

    # Stable sort: ties resolve in declared profile order (aura, zeno, iris, eris).
    simulations.sort(key=lambda s: s["efe"])
    winner = simulations[0]
    bbu_log.append(
        f"Bayesian Belief Updater selected {winner['simulation_id']} "
        f"(lowest expected free energy {winner['efe']})"
    )
    return simulations, bbu_log, winner


def _persona_plan(persona: str, profile: Dict[str, Any], action_category: str) -> List[str]:
    """The winning persona's declared policy outline (simulation output, not a promise)."""
    return [
        f"{persona} ({profile.get('description', 'persona')}): approach the task as '{action_category}'",
        "predict the drive/homeostasis shift under the declared transition priors",
        "continue only while the predicted EFE stays below the best alternative",
    ]


def run_active_inference_simulation(
    prompt: str = "",
    current_drives: Optional[Dict[str, float]] = None,
    current_homeostasis: Any = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Simulate the four personas over a prompt and select the minimal-EFE one.

    P: 2026-11-10 - the old positional form
    ``run_active_inference_simulation(initial_state, candidate_actions,
    horizon, preferences)`` had zero callers and returned a fabricated
    "best_action" selection over caller-supplied policies.  The pinned
    contract is this prompt/drives form over the four personas.
    """
    category = _action_category(str(prompt or ""))
    simulations, bbu_log, winner = _simulate_personas(current_drives, current_homeostasis, category)
    return {
        "status": "complete",
        "prompt_category": category,
        "simulations": simulations,
        "optimal_persona": winner["simulation_id"],
        "optimal_plan": winner["plan"],
        "bbu_log": bbu_log,
    }


def calculate_utility(
    predicted_state: Dict[str, Any],
    preferences: Optional[Dict[str, float]] = None,
) -> float:
    """Compute pragmatic + epistemic utility of a predicted state."""
    prefs = preferences or {}
    w_progress = float(prefs.get("progress", 1.2))
    w_distance = float(prefs.get("goal_distance", 1.2))
    w_uncertainty = float(prefs.get("uncertainty", 0.8))
    w_risk = float(prefs.get("safety_risk", 2.5))

    progress = float(predicted_state.get("progress", 0.0))
    goal_distance = float(predicted_state.get("goal_distance", 0.5))
    uncertainty = float(predicted_state.get("uncertainty", 0.5))
    safety_risk = float(predicted_state.get("safety_risk", 0.0))

    utility = (
        w_progress * progress
        + w_distance * (1.0 - goal_distance)
        + w_uncertainty * (1.0 - uncertainty)
        - w_risk * safety_risk
    )
    return round(float(utility), 6)


# ---------------------------------------------------------------------------
# Adaptive Persona Memory (Bayesian, file-backed)
# ---------------------------------------------------------------------------


class AdaptivePersonaMemory:
    """File-backed Bayesian persona memory.

    Tracks attempts/successes per persona plus a bounded recent-outcome
    window; ``win_rate``/``momentum`` default to the neutral 0.5 prior
    for untried personas.  State persists as JSON via the repo's
    sharing-violation-safe atomic writer.

    P: 2026-11-10 - this was a 15-line in-memory stub ("Minimal adaptive
    persona memory stub") with no persistence, no decay, and a
    ``get_persona_memory`` that minted a fresh empty instance per call —
    no persona history could ever accumulate.
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = Path(filepath) if filepath is not None else Path(DEFAULT_PERSONA_MEMORY_PATH)
        # Attempts/successes are floats: decay_to_prior scales them toward
        # the neutral prior and entries below the floor are removed.
        self._attempts: Dict[str, float] = {}
        self._successes: Dict[str, float] = {}
        self._recent_outcomes: Dict[str, List[float]] = {}
        self._utilities: Dict[str, float] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        data = atomic_read(self.filepath, default=None)
        if not isinstance(data, dict):
            return
        for attr, key in (
            ("_attempts", "attempts"),
            ("_successes", "successes"),
            ("_utilities", "utilities"),
        ):
            raw = data.get(key, {})
            if isinstance(raw, dict):
                setattr(self, attr, {str(k): float(v) for k, v in raw.items()})
        raw_recent = data.get("recent_outcomes", {})
        if isinstance(raw_recent, dict):
            self._recent_outcomes = {
                str(k): [float(v) for v in vals]
                for k, vals in raw_recent.items()
                if isinstance(vals, list)
            }

    def _save(self) -> None:
        atomic_write(
            self.filepath,
            {
                "attempts": self._attempts,
                "successes": self._successes,
                "recent_outcomes": self._recent_outcomes,
                "utilities": self._utilities,
            },
        )

    # -- Bayesian state ------------------------------------------------------
    def record_outcome(self, persona: str, won: bool, utility: Optional[float] = None) -> None:
        """Record a win/loss for a persona.

        ``utility`` (when supplied) is persisted as the persona's last
        observed utility; the Bayesian state itself is win/loss counts
        and the recent-outcome window (the pinned contract).
        """
        p = str(persona)
        self._attempts[p] = self._attempts.get(p, 0.0) + 1.0
        if won:
            self._successes[p] = self._successes.get(p, 0.0) + 1.0
        window = self._recent_outcomes.setdefault(p, [])
        window.append(1.0 if won else 0.0)
        del window[:-_RECENT_WINDOW]
        if utility is not None:
            try:
                self._utilities[p] = float(utility)
            except (TypeError, ValueError):
                pass
        self._save()

    def win_rate(self, persona: str) -> float:
        attempts = self._attempts.get(str(persona), 0.0)
        if attempts <= 0.0:
            return NEUTRAL_PRIOR
        return _clamp01(self._successes.get(str(persona), 0.0) / attempts)

    def momentum(self, persona: str) -> float:
        recent = self._recent_outcomes.get(str(persona), [])
        if not recent:
            return NEUTRAL_PRIOR
        return _clamp01(sum(recent) / len(recent))

    def adaptive_weight(self, persona: str, base_utility: float = 1.0) -> float:
        """Modulate a utility by the persona's recorded performance.

        weight = base_utility * (0.7 + 0.3 * bonus), bonus in [0, 1] from
        the neutral-prior blend 0.5 * win_rate + 0.3 * momentum.
        """
        bonus = 0.5 * self.win_rate(persona) + 0.3 * self.momentum(persona)
        return float(base_utility) * (0.7 + 0.3 * bonus)

    def decay_to_prior(self, decay_factor: float = 0.85) -> str:
        """Decay every persona's counts toward the neutral prior.

        Entries scaled below the 0.5-attempt floor are removed entirely so
        the persona reads as untried (neutral prior) again.  Returns a
        human-readable summary of what was decayed/removed.
        """
        factor = float(decay_factor)
        if not 0.0 < factor < 1.0:
            raise ValueError(f"decay_factor must be in (0, 1), got {decay_factor}")
        decayed: List[str] = []
        removed: List[str] = []
        for persona in list(self._attempts.keys()):
            old = self._attempts[persona]
            self._attempts[persona] = old * factor
            self._successes[persona] = self._successes.get(persona, 0.0) * factor
            decayed.append(persona)
            if self._attempts[persona] < 0.5:
                # Below the floor the evidence is no longer distinguishable
                # from the prior: drop the entry entirely.
                self._attempts.pop(persona, None)
                self._successes.pop(persona, None)
                self._recent_outcomes.pop(persona, None)
                self._utilities.pop(persona, None)
                removed.append(persona)
        if decayed:
            self._save()
        summary = (
            f"Decayed persona priors (factor {factor}): "
            + ", ".join(f"{p} (attempts now {self._attempts.get(p, 0.0) if p in self._attempts else 'removed'})" for p in decayed)
            if decayed
            else "No persona entries to decay."
        )
        if removed:
            summary += f" Removed below floor: {', '.join(removed)}."
        return summary

    def reset(self) -> None:
        """Clear all persona history and persist the empty state."""
        self._attempts = {}
        self._successes = {}
        self._recent_outcomes = {}
        self._utilities = {}
        self._save()


_PERSONA_MEMORY_SINGLETON: Optional[AdaptivePersonaMemory] = None


def get_persona_memory(persona: str = "default", filepath: Optional[str] = None) -> AdaptivePersonaMemory:
    """Module-level singleton accessor.

    P: 2026-11-10 - the old stub returned a NEW in-memory instance on
    every call, so nothing could accumulate or persist persona history.
    The ``persona`` argument is retained for interface compatibility;
    all personas share one ledger keyed by persona name.
    """
    global _PERSONA_MEMORY_SINGLETON
    if _PERSONA_MEMORY_SINGLETON is None:
        _PERSONA_MEMORY_SINGLETON = AdaptivePersonaMemory(filepath=filepath)
    return _PERSONA_MEMORY_SINGLETON


__all__ = [
    "ActiveInferenceCluster",
    "AdaptivePersonaMemory",
    "get_persona_memory",
    "calculate_utility",
    "predict_state_update",
    "run_active_inference_simulation",
    "PERSONA_PROFILES",
    "ACTION_TRANSITION_PRIORS",
]
