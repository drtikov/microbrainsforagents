"""Discrete categorical Expected-Free-Energy planner (pymdp API shape).

P: 2026-11-18 — robotics extension. ``core/active_inference.py`` is persona
EFE scoring with declared transition priors; this module implements the real
discrete-state FEP planning machinery, using the A/B/C/D(/E) generative-matrix
naming that pymdp made the de-facto standard (JOSS 2022; pymdp 1.0 JAX 2025):

    A : (n_obs, n_states)   observation likelihood  P(o|s), columns sum to 1
    B : (n_states, n_states, n_actions)  transitions P(s'|s,u), slice [:,:,u]
    C : (n_obs,)            preference over observations (logits; softmaxed)
    D : (n_states,)         prior over states, sums to 1
    E : (n_policies,)       prior over policies (optional)

Expected free energy for a policy (Schwartenbeck et al. 2019 decomposition):

    G(pi) = SUM_t [ pragmatic_t - epistemic_t (- novelty_t) ]
    pragmatic_t = - SUM_o q(o_t|pi) ln softmax(C)[o]        (goal-directed)
    epistemic_t = I(o_t; s_t | pi) >= 0                     (salience; = 0
                  when A is uninformative)                   sum over o,s of
                  q(o,s) ln q(o|s)/q(o) )

Lower G(pi) is better; higher mutual information reduces G(pi) so
q(pi) = softmax(-G) prefers informative epistemic policies.

Also provides ``FactoredGenerativeModel`` and ``MultiModalEFPAgent`` for
multi-factor hidden states, multiple observation modalities, and optional
Dirichlet parameter novelty over ``pA``.

Scope honesty: discrete categorical state spaces, fixed horizon policies
supplied as action sequences. This is the bandit/small-MDP tier of active
inference — not a continuous-state FEP controller.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _softmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).flatten()
    v = v - np.max(v)
    e = np.exp(v)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(v.shape, 1.0 / max(len(v), 1))
    return e / s


def _normalize_columns(mat: np.ndarray) -> np.ndarray:
    col_sums = mat.sum(axis=0, keepdims=True)
    col_sums[col_sums <= 0] = 1.0
    return mat / col_sums


class GenerativeModel:
    """A/B/C(/D/E) container with honest structural validation."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        E: Optional[np.ndarray] = None,
    ):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D (n_obs, n_states); got shape {A.shape}")
        if B.ndim != 3 or B.shape[0] != A.shape[1] or B.shape[1] != A.shape[1]:
            raise ValueError(
                f"B must be (n_states, n_states, n_actions) with n_states={A.shape[1]}; got {B.shape}"
            )
        bad_cols = int(np.sum(B.sum(axis=0) <= 0))
        if bad_cols:
            raise ValueError(f"{bad_cols} B transition slices have non-positive mass")
        n_states = A.shape[1]
        if C is None:
            C = np.zeros(A.shape[0], dtype=np.float64)
        C = np.asarray(C, dtype=np.float64).flatten()
        if C.shape[0] != A.shape[0]:
            raise ValueError(f"C length {C.shape[0]} != n_obs {A.shape[0]}")
        if D is None:
            D = np.full(n_states, 1.0 / n_states)
        D = np.asarray(D, dtype=np.float64).flatten()
        if D.shape[0] != n_states:
            raise ValueError(f"D length {D.shape[0]} != n_states {n_states}")
        if abs(float(D.sum()) - 1.0) > 1e-6:
            D = D / max(D.sum(), 1e-12)
        if E is not None:
            E = np.asarray(E, dtype=np.float64).flatten()
            if E.shape[0] < 1 or abs(float(E.sum()) - 1.0) > 1e-6:
                E = _softmax(E)
        self.A = _normalize_columns(A)
        self.B = np.stack([_normalize_columns(B[:, :, u]) for u in range(B.shape[2])], axis=2)
        self.preferences = _softmax(C)
        self.D = D
        self.E = E
        self.n_obs, self.n_states, self.n_actions = A.shape[0], n_states, B.shape[2]


def expected_free_energy(
    model: GenerativeModel,
    belief: np.ndarray,
    policy: Sequence[int],
) -> Tuple[float, Dict[str, Any]]:
    """
    G(policy) for one action sequence starting from ``belief``.
    Returns (G, trace) where the trace carries per-step pragmatic/epistemic
    terms and the predicted observation distributions (all MEASURED numbers).
    """
    belief = np.asarray(belief, dtype=np.float64).flatten()
    if belief.shape[0] != model.n_states:
        raise ValueError(f"belief length {belief.shape[0]} != n_states {model.n_states}")
    g_total = 0.0
    trace: List[Dict[str, Any]] = []
    q_s = belief.copy()
    for t, u in enumerate(policy):
        u = int(u)
        if u < 0 or u >= model.n_actions:
            raise ValueError(f"policy action {u} out of range at t={t}")
        q_s = model.B[:, :, u] @ q_s
        q_o = model.A @ q_s
        log_C = np.log(np.maximum(model.preferences, 1e-12))
        pragmatic = -float(np.dot(q_o, log_C))
        # epistemic value I(o;s) = sum_{o,s} A[o,s] q(s) ln( A[o,s] / q(o) ) >= 0
        outer = model.A * q_s[None, :]  # q(o, s) = P(o|s) q(s)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(model.A) - np.log(np.maximum(q_o[:, None], 1e-12))
            mi_terms = np.where(outer > 0, outer * log_ratio, 0.0)
        epistemic = float(np.sum(mi_terms))
        g_total += pragmatic - epistemic
        trace.append(
            {
                "t": t,
                "action": u,
                "pragmatic": round(pragmatic, 6),
                "epistemic": round(epistemic, 6),
                "predicted_state": q_s.tolist(),
                "predicted_obs": q_o.tolist(),
            }
        )
    return float(g_total), {"per_step": trace}


class EFPAgent:
    """Single-modality EFE agent: belief update + policy posterior + action."""

    def __init__(self, model: GenerativeModel, prior_E: Optional[np.ndarray] = None):
        self.model = model
        if model.E is not None:
            self.prior_E = model.E
        elif prior_E is not None:
            self.prior_E = _softmax(np.asarray(prior_E, dtype=np.float64))
        else:
            self.prior_E = None
        self.belief = model.D.copy()

    @property
    def belief_entropy(self) -> float:
        """Shannon entropy H(q(s)) of the current state belief."""
        b = np.asarray(self.belief, dtype=np.float64).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(b > 0, b * np.log(np.maximum(b, 1e-12)), 0.0)
        return float(-np.sum(terms))

    def update_belief(self, obs_idx: int, action_idx: Optional[int] = None) -> np.ndarray:
        """Predict (through B if action given) then correct (through A)."""
        obs_idx = int(obs_idx)
        if obs_idx < 0 or obs_idx >= self.model.n_obs:
            raise ValueError(f"obs index {obs_idx} out of range")
        prior = self.belief
        if action_idx is not None:
            prior = self.model.B[:, :, int(action_idx)] @ prior
        likelihood = self.model.A[obs_idx, :]
        post = prior * likelihood
        total = float(post.sum())
        self.belief = post / total if total > 0 else prior / max(prior.sum(), 1e-12)
        return self.belief

    def plan(self, policies: Sequence[Sequence[int]]) -> Dict[str, Any]:
        """Posterior over policies q(pi) = softmax(-G) x E; returns ranked plan."""
        if not policies:
            return {"status": "no_policies", "best_policy": None, "q_pi": []}
        Gs = []
        for pol in policies:
            g, trace = expected_free_energy(self.model, self.belief, pol)
            Gs.append(g)
        G = np.asarray(Gs, dtype=np.float64)
        logits = -G if self.prior_E is None else (-G + np.log(np.maximum(self.prior_E, 1e-12)))
        q_pi = _softmax(logits)
        best = int(np.argmax(q_pi))
        return {
            "status": "planned",
            "best_policy": list(policies[best]),
            "best_action": int(policies[best][0]),
            "q_pi": q_pi.tolist(),
            "G": G.tolist(),
            "n_policies": len(policies),
        }


def _joint_belief(q_factors: Sequence[np.ndarray]) -> np.ndarray:
    """Outer product of factorized state beliefs into joint tensor (*num_states)."""
    joint = np.asarray(q_factors[0], dtype=np.float64).ravel()
    for q_f in q_factors[1:]:
        joint = np.multiply.outer(joint, np.asarray(q_f, dtype=np.float64).ravel())
    return joint


class FactoredGenerativeModel:
    """Multi-factor, multi-modality discrete generative model with optional Dirichlet pA."""

    def __init__(
        self,
        A_modalities: Sequence[np.ndarray],
        B_factors: Sequence[np.ndarray],
        C_modalities: Optional[Sequence[np.ndarray]] = None,
        D_factors: Optional[Sequence[np.ndarray]] = None,
        E: Optional[np.ndarray] = None,
        pA: Optional[Sequence[Optional[np.ndarray]]] = None,
        num_states: Optional[Sequence[int]] = None,
    ):
        if not A_modalities:
            raise ValueError("A_modalities must be a non-empty sequence")
        if not B_factors:
            raise ValueError("B_factors must be a non-empty sequence")

        norm_B: List[np.ndarray] = []
        inferred_num_states: List[int] = []
        for f, B_raw in enumerate(B_factors):
            B_f = np.asarray(B_raw, dtype=np.float64)
            if B_f.ndim != 3 or B_f.shape[0] < 1 or B_f.shape[0] != B_f.shape[1] or B_f.shape[2] < 1:
                raise ValueError(
                    f"B_factors[{f}] must have shape (n_states_f, n_states_f, n_actions_f); got {B_f.shape}"
                )
            if np.any(B_f < 0):
                raise ValueError(f"B_factors[{f}] must be non-negative")
            bad_cols = int(np.sum(B_f.sum(axis=0) <= 0))
            if bad_cols:
                raise ValueError(f"B_factors[{f}] has {bad_cols} transition columns with non-positive mass")
            norm_B.append(np.stack([_normalize_columns(B_f[:, :, u]) for u in range(B_f.shape[2])], axis=2))
            inferred_num_states.append(int(B_f.shape[0]))

        if num_states is not None:
            num_states_list = [int(s) for s in num_states]
            if num_states_list != inferred_num_states:
                raise ValueError(
                    f"num_states {num_states_list} does not match B_factors state dims {inferred_num_states}"
                )
        else:
            num_states_list = inferred_num_states

        self.num_states: List[int] = num_states_list
        self.n_factors: int = len(self.num_states)
        self.n_joint_states: int = int(np.prod(self.num_states))
        self.B_factors: List[np.ndarray] = norm_B
        self.num_actions: List[int] = [int(b.shape[2]) for b in self.B_factors]
        self.n_actions: int = int(np.prod(self.num_actions))

        norm_A: List[np.ndarray] = []
        for m, A_raw in enumerate(A_modalities):
            A_m = self._coerce_modality_tensor(A_raw, f"A_modalities[{m}]")
            if np.any(A_m < 0):
                raise ValueError(f"A_modalities[{m}] must be non-negative")
            norm_A.append(_normalize_columns(A_m))

        self.A_modalities: List[np.ndarray] = norm_A
        self.n_modalities: int = len(self.A_modalities)
        self.num_obs: List[int] = [int(a.shape[0]) for a in self.A_modalities]

        if C_modalities is None:
            raw_C = [np.zeros(n_o, dtype=np.float64) for n_o in self.num_obs]
        else:
            if len(C_modalities) != self.n_modalities:
                raise ValueError(f"C_modalities length {len(C_modalities)} != n_modalities {self.n_modalities}")
            raw_C = []
            for m, c_raw in enumerate(C_modalities):
                c_arr = np.asarray(c_raw, dtype=np.float64).flatten()
                if c_arr.shape[0] != self.num_obs[m]:
                    raise ValueError(f"C_modalities[{m}] length {c_arr.shape[0]} != n_obs {self.num_obs[m]}")
                raw_C.append(c_arr)

        self.C_modalities: List[np.ndarray] = raw_C
        self.preferences_modalities: List[np.ndarray] = [_softmax(c) for c in raw_C]
        self.log_preferences_modalities: List[np.ndarray] = [
            np.log(np.maximum(p, 1e-12)) for p in self.preferences_modalities
        ]

        if D_factors is None:
            norm_D = [np.full(n_s, 1.0 / n_s, dtype=np.float64) for n_s in self.num_states]
        else:
            if len(D_factors) != self.n_factors:
                raise ValueError(f"D_factors length {len(D_factors)} != n_factors {self.n_factors}")
            norm_D = []
            for f, d_raw in enumerate(D_factors):
                d_arr = np.asarray(d_raw, dtype=np.float64).flatten()
                if d_arr.shape[0] != self.num_states[f]:
                    raise ValueError(f"D_factors[{f}] length {d_arr.shape[0]} != num_states[{f}] {self.num_states[f]}")
                if np.any(d_arr < 0) or float(d_arr.sum()) <= 0:
                    raise ValueError(f"D_factors[{f}] must be non-negative with positive sum")
                norm_D.append(d_arr / float(d_arr.sum()))
        self.D_factors: List[np.ndarray] = norm_D

        if E is not None:
            E_arr = np.asarray(E, dtype=np.float64).flatten()
            if E_arr.shape[0] < 1 or abs(float(E_arr.sum()) - 1.0) > 1e-6:
                E_arr = _softmax(E_arr)
            self.E: Optional[np.ndarray] = E_arr
        else:
            self.E = None

        if pA is None:
            self.pA: Optional[List[Optional[np.ndarray]]] = None
        else:
            if len(pA) != self.n_modalities:
                raise ValueError(f"pA length {len(pA)} != n_modalities {self.n_modalities}")
            parsed_pA: List[Optional[np.ndarray]] = []
            for m, p_raw in enumerate(pA):
                if p_raw is None:
                    parsed_pA.append(None)
                else:
                    p_m = self._coerce_modality_tensor(p_raw, f"pA[{m}]", expected_obs=self.num_obs[m])
                    if np.any(p_m < 0):
                        raise ValueError(f"pA[{m}] must be non-negative")
                    parsed_pA.append(p_m.copy())
            self.pA = parsed_pA

    def _coerce_modality_tensor(
        self, arr_raw: np.ndarray, name: str, expected_obs: Optional[int] = None
    ) -> np.ndarray:
        arr = np.asarray(arr_raw, dtype=np.float64)
        if arr.ndim < 2 or arr.shape[0] < 1:
            raise ValueError(f"{name} must have at least 2 dims with n_obs >= 1; got {arr.shape}")
        if expected_obs is not None and arr.shape[0] != expected_obs:
            raise ValueError(f"{name} obs dim {arr.shape[0]} != expected {expected_obs}")
        if tuple(arr.shape[1:]) == tuple(self.num_states):
            return arr
        if arr.ndim == 2 and arr.shape[1] == self.n_joint_states:
            return arr.reshape((arr.shape[0], *self.num_states))
        raise ValueError(
            f"{name} shape {arr.shape} incompatible with num_states={self.num_states} "
            f"(joint={self.n_joint_states})"
        )

    def resolve_factor_actions(self, action: Any) -> Tuple[int, ...]:
        """Resolve a single joint action int or per-factor action sequence into factor indices."""
        if isinstance(action, (int, np.integer)):
            u = int(action)
            if u < 0 or u >= self.n_actions:
                raise ValueError(f"joint action {u} out of range [0, {self.n_actions})")
            return tuple(int(x) for x in np.unravel_index(u, tuple(self.num_actions)))
        act_seq = [int(a) for a in action]
        if len(act_seq) != self.n_factors:
            raise ValueError(f"factor action length {len(act_seq)} != n_factors {self.n_factors}")
        for f, u_f in enumerate(act_seq):
            if u_f < 0 or u_f >= self.num_actions[f]:
                raise ValueError(f"action {u_f} out of range for factor {f} (n_actions={self.num_actions[f]})")
        return tuple(act_seq)


def expected_free_energy_multimodal(
    model: FactoredGenerativeModel,
    belief_factors: Sequence[np.ndarray],
    policy: Sequence[Any],
    pA: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute multi-modal, multi-factor Expected Free Energy G(policy):

        G(pi) = SUM_t SUM_m [ pragmatic_{t,m} - epistemic_{t,m} - novelty_{t,m} ]

    where ``pragmatic`` uses log-preferences, ``epistemic`` is I(o_m; s | pi), and
    ``novelty`` is Dirichlet parameter information gain over ``pA`` (0 when absent).
    """
    if len(belief_factors) != model.n_factors:
        raise ValueError(f"belief_factors length {len(belief_factors)} != n_factors {model.n_factors}")
    q_factors: List[np.ndarray] = []
    for f, q_raw in enumerate(belief_factors):
        q_f = np.asarray(q_raw, dtype=np.float64).flatten()
        if q_f.shape[0] != model.num_states[f]:
            raise ValueError(f"belief_factors[{f}] length {q_f.shape[0]} != num_states[{f}] {model.num_states[f]}")
        q_factors.append(q_f.copy())

    active_pA = model.pA if pA is None else pA
    g_total = 0.0
    trace: List[Dict[str, Any]] = []
    mod_totals = [
        {"modality": m, "pragmatic": 0.0, "epistemic": 0.0, "novelty": 0.0}
        for m in range(model.n_modalities)
    ]

    for t, u in enumerate(policy):
        u_factors = model.resolve_factor_actions(u)
        q_factors = [model.B_factors[f][:, :, u_factors[f]] @ q_factors[f] for f in range(model.n_factors)]
        q_s_joint = _joint_belief(q_factors)
        joint_axes = tuple(range(q_s_joint.ndim))

        step_pragmatic = 0.0
        step_epistemic = 0.0
        step_novelty = 0.0
        per_mod_step: List[Dict[str, Any]] = []

        for m in range(model.n_modalities):
            A_m = model.A_modalities[m]
            state_axes = tuple(range(1, A_m.ndim))
            q_o_m = np.tensordot(A_m, q_s_joint, axes=(state_axes, joint_axes))

            pragmatic_m = -float(np.dot(q_o_m, model.log_preferences_modalities[m]))

            with np.errstate(divide="ignore", invalid="ignore"):
                log_A = np.where(A_m > 0, np.log(A_m), 0.0)
                log_qo = np.where(q_o_m > 0, np.log(np.maximum(q_o_m, 1e-12)), 0.0)
            entropy_A_cond = -np.sum(A_m * log_A, axis=0)
            expected_ambiguity = float(np.sum(entropy_A_cond * q_s_joint))
            entropy_qo = -float(np.sum(q_o_m * log_qo))
            epistemic_m = max(0.0, entropy_qo - expected_ambiguity)

            if active_pA is not None and active_pA[m] is not None:
                pA_m = np.maximum(active_pA[m], 1e-12)
                pA_sum = np.sum(pA_m, axis=0, keepdims=True)
                w_m = 0.5 * (1.0 / pA_m - 1.0 / pA_sum)
                expected_w_obs = np.tensordot(w_m, q_s_joint, axes=(state_axes, joint_axes))
                novelty_m = max(0.0, float(np.dot(q_o_m, expected_w_obs)))
            else:
                novelty_m = 0.0

            step_pragmatic += pragmatic_m
            step_epistemic += epistemic_m
            step_novelty += novelty_m
            mod_totals[m]["pragmatic"] += pragmatic_m
            mod_totals[m]["epistemic"] += epistemic_m
            mod_totals[m]["novelty"] += novelty_m
            per_mod_step.append(
                {
                    "modality": m,
                    "pragmatic": round(pragmatic_m, 6),
                    "epistemic": round(epistemic_m, 6),
                    "novelty": round(novelty_m, 6),
                    "predicted_obs": q_o_m.tolist(),
                }
            )

        g_total += step_pragmatic - step_epistemic - step_novelty
        trace.append(
            {
                "t": t,
                "action": list(u_factors),
                "pragmatic": round(step_pragmatic, 6),
                "epistemic": round(step_epistemic, 6),
                "novelty": round(step_novelty, 6),
                "per_modality": per_mod_step,
                "predicted_factors": [qf.tolist() for qf in q_factors],
            }
        )

    rounded_totals = [
        {
            "modality": mt["modality"],
            "pragmatic": round(mt["pragmatic"], 6),
            "epistemic": round(mt["epistemic"], 6),
            "novelty": round(mt["novelty"], 6),
        }
        for mt in mod_totals
    ]
    return float(g_total), {"per_step": trace, "per_modality_totals": rounded_totals}


class MultiModalEFPAgent:
    """Multi-factor, multi-modality EFE agent with optional Dirichlet pA novelty."""

    def __init__(
        self,
        model: FactoredGenerativeModel,
        prior_E: Optional[np.ndarray] = None,
        learn_pA: bool = True,
    ):
        self.model = model
        if model.E is not None:
            self.prior_E = model.E
        elif prior_E is not None:
            self.prior_E = _softmax(np.asarray(prior_E, dtype=np.float64))
        else:
            self.prior_E = None
        self.belief_factors: List[np.ndarray] = [d.copy() for d in model.D_factors]
        self.joint_belief: np.ndarray = _joint_belief(self.belief_factors)
        self.pA: Optional[List[Optional[np.ndarray]]] = (
            [p.copy() if p is not None else None for p in model.pA]
            if model.pA is not None
            else None
        )
        self.learn_pA = bool(learn_pA)

    @property
    def belief_entropy(self) -> float:
        """Joint Shannon entropy H(q(s)) across state factors."""
        jb = np.asarray(self.joint_belief, dtype=np.float64).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(jb > 0, jb * np.log(np.maximum(jb, 1e-12)), 0.0)
        return float(-np.sum(terms))

    def update_belief(
        self,
        observations: Sequence[int],
        action_indices: Optional[Sequence[int]] = None,
        learn_pA: Optional[bool] = None,
        learning_rate: float = 1.0,
    ) -> List[np.ndarray]:
        """Predict factor beliefs via B_factors (if action given) and correct via A_modalities."""
        if len(observations) != self.model.n_modalities:
            raise ValueError(
                f"observations length {len(observations)} != n_modalities {self.model.n_modalities}"
            )
        obs_ints = [int(o) for o in observations]
        for m, obs_m in enumerate(obs_ints):
            if obs_m < 0 or obs_m >= self.model.num_obs[m]:
                raise ValueError(f"observation {obs_m} out of range for modality {m} (n_obs={self.model.num_obs[m]})")

        if action_indices is not None:
            u_factors = self.model.resolve_factor_actions(action_indices)
            priors = [
                self.model.B_factors[f][:, :, u_factors[f]] @ self.belief_factors[f]
                for f in range(self.model.n_factors)
            ]
        else:
            priors = [q_f.copy() for q_f in self.belief_factors]

        joint_prior = _joint_belief(priors)
        lik_joint = np.ones(tuple(self.model.num_states), dtype=np.float64)
        for m, obs_m in enumerate(obs_ints):
            lik_joint = lik_joint * self.model.A_modalities[m][obs_m]

        joint_post = joint_prior * lik_joint
        total = float(np.sum(joint_post))
        if total > 0:
            joint_post = joint_post / total
        else:
            joint_post = joint_prior / max(float(np.sum(joint_prior)), 1e-12)

        self.joint_belief = joint_post
        for f in range(self.model.n_factors):
            other_axes = tuple(i for i in range(self.model.n_factors) if i != f)
            qf = np.sum(joint_post, axis=other_axes) if other_axes else joint_post.copy()
            s = float(np.sum(qf))
            self.belief_factors[f] = qf / s if s > 0 else qf

        do_learn = self.learn_pA if learn_pA is None else bool(learn_pA)
        if do_learn and self.pA is not None:
            lr = float(learning_rate)
            for m, obs_m in enumerate(obs_ints):
                if self.pA[m] is not None:
                    self.pA[m][obs_m] += lr * self.joint_belief

        return [q_f.copy() for q_f in self.belief_factors]

    def plan(self, policies: Sequence[Sequence[Any]]) -> Dict[str, Any]:
        """Compute multi-modal G(pi) and policy posterior q(pi) = softmax(-G) x E."""
        if not policies:
            return {
                "status": "no_policies",
                "best_policy": None,
                "q_pi": [],
                "belief_entropy": round(self.belief_entropy, 6),
                "per_modality": [],
            }
        Gs: List[float] = []
        traces: List[Dict[str, Any]] = []
        for pol in policies:
            g, tr = expected_free_energy_multimodal(
                self.model, self.belief_factors, pol, pA=self.pA
            )
            Gs.append(g)
            traces.append(tr)
        G = np.asarray(Gs, dtype=np.float64)
        logits = -G if self.prior_E is None else (-G + np.log(np.maximum(self.prior_E, 1e-12)))
        q_pi = _softmax(logits)
        best = int(np.argmax(q_pi))
        best_pol = [
            [int(x) for x in step] if isinstance(step, (list, tuple, np.ndarray)) else int(step)
            for step in policies[best]
        ]
        return {
            "status": "planned",
            "best_policy": best_pol,
            "best_action": best_pol[0],
            "q_pi": q_pi.tolist(),
            "G": G.tolist(),
            "n_policies": len(policies),
            "belief_entropy": round(self.belief_entropy, 6),
            "per_modality": traces[best]["per_modality_totals"],
            "per_policy_traces": traces,
        }
