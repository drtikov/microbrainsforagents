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

    G(pi) = SUM_t [ pragmatic_t + epistemic_t ]
    pragmatic_t = - SUM_o q(o_t|pi) ln softmax(C)[o]        (goal-directed)
    epistemic_t = I(o_t; s_t | pi)                          (salience; = 0
                  when A is uninformative)                   sum over o,s of
                  q(o,s) ln q(o|s)/q(o) )

q(o_t|pi) = A @ q(s_t|pi) with beliefs propagated through the policy's own B
slices; posterior over policies q(pi) = softmax(-G) weighted by E.

Scope honesty: single observation modality, fixed horizon policies supplied
as action sequences. This is the bandit/small-MDP tier of active inference —
not a continuous-state FEP controller.
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
        pragmatic = -float(np.dot(q_o, model.preferences))
        # epistemic value I(o;s) = sum_{o,s} A[o,s] q(s) ln( A[o,s] / q(o) )
        outer = model.A * q_s[None, :]  # q(o, s) = P(o|s) q(s)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(model.A) - np.log(np.maximum(q_o[:, None], 1e-12))
            mi_terms = np.where(outer > 0, outer * log_ratio, 0.0)
        epistemic = float(np.sum(mi_terms))
        g_total += pragmatic + epistemic
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
