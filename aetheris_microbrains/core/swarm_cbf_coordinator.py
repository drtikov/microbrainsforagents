"""Decentralized multi-agent Control Barrier Function (CBF) coordinator.

Provides:
  - ``SwarmAgentDescriptor``: state, geometry, actuator box, and capability
    metadata for a 2D or 3D mobile/aerial robot.
  - ``compute_capability_priority`` & ``compute_responsibility_share``:
    deterministic right-of-way priority weights derived from ``CapabilityToken``
    physical caveats (``max_linear_vel_mps``, ``max_kinetic_energy_j``,
    ``max_contact_force_n``, ``depth_remaining``) or explicit ``priority_score``,
    yielding partition-of-unity evasion burdens ``omega_ij + omega_ji = 1``.
  - ``DecentralizedSwarmCBF``: single-integrator (velocity) and
    double-integrator (acceleration / relative-degree-2 HOCBF) decentralized
    pairwise barrier halfspaces without neighbor control communication, combined
    with chirality-consistent orthogonal circulation symmetry-breaking and exact
    Dykstra cyclic projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .capability_tokens import CapabilityToken
from .state_cbf_filter import (
    BarrierConstraint,
    ControlAffineDynamics,
    LinearDynamics,
    StateCBFSafetyFilter,
)

_REF_LINEAR_VEL_MPS: float = 2.0
_REF_KINETIC_ENERGY_J: float = 25.0
_REF_CONTACT_FORCE_N: float = 50.0
_REF_DEPTH_REMAINING: int = 4
_MIN_PRIORITY: float = 1e-3
_MAX_PRIORITY: float = 1e4


def _as_vec(value: Sequence[float], name: str, expected_dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).flatten()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a non-empty finite numeric vector")
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError(f"{name} dim {arr.shape[0]} != expected dim {expected_dim}")
    return arr


@dataclass
class SwarmAgentDescriptor:
    """Kinematic and capability descriptor for an agent in a 2D or 3D swarm."""

    agent_id: str
    position: Union[Sequence[float], np.ndarray]
    velocity: Optional[Union[Sequence[float], np.ndarray]] = None
    radius: float = 0.25
    u_min: Optional[Union[Sequence[float], np.ndarray]] = None
    u_max: Optional[Union[Sequence[float], np.ndarray]] = None
    capability_token: Optional[CapabilityToken] = None
    priority_score: Optional[float] = None

    def __post_init__(self) -> None:
        clean_id = str(self.agent_id).strip()
        if not clean_id:
            raise ValueError("agent_id must be a non-empty string")
        self.agent_id = clean_id

        pos = _as_vec(self.position, "position")
        if pos.shape[0] not in (2, 3):
            raise ValueError(f"position must be 2D or 3D, got dim={pos.shape[0]}")
        self.position = pos
        dim = pos.shape[0]

        if self.velocity is None:
            self.velocity = np.zeros(dim, dtype=np.float64)
        else:
            self.velocity = _as_vec(self.velocity, "velocity", expected_dim=dim)

        rad = float(self.radius)
        if not math.isfinite(rad) or rad < 0.0:
            raise ValueError("radius must be a finite non-negative number")
        self.radius = rad

        if self.u_min is not None:
            self.u_min = _as_vec(self.u_min, "u_min", expected_dim=dim)
        if self.u_max is not None:
            self.u_max = _as_vec(self.u_max, "u_max", expected_dim=dim)
        if self.u_min is not None and self.u_max is not None and np.any(self.u_min > self.u_max):
            raise ValueError("u_min must be elementwise <= u_max")

        if self.capability_token is not None and not isinstance(self.capability_token, CapabilityToken):
            raise TypeError("capability_token must be a CapabilityToken or None")

        if self.priority_score is not None:
            p_val = float(self.priority_score)
            if not math.isfinite(p_val) or p_val <= 0.0:
                raise ValueError("priority_score must be finite and strictly positive")
            self.priority_score = p_val

    @property
    def dim(self) -> int:
        return int(self.position.shape[0])

    def state_vector(self, dynamics_mode: str = "single_integrator") -> np.ndarray:
        if dynamics_mode == "single_integrator":
            return self.position.copy()
        if dynamics_mode == "double_integrator":
            return np.concatenate([self.position, self.velocity])
        raise ValueError(f"Unsupported dynamics_mode: {dynamics_mode!r}")

    def with_kinematics(
        self,
        position: Union[Sequence[float], np.ndarray],
        velocity: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> "SwarmAgentDescriptor":
        """Return a copy of this descriptor with updated position and velocity."""
        return SwarmAgentDescriptor(
            agent_id=self.agent_id,
            position=position,
            velocity=self.velocity if velocity is None else velocity,
            radius=self.radius,
            u_min=None if self.u_min is None else self.u_min.copy(),
            u_max=None if self.u_max is None else self.u_max.copy(),
            capability_token=self.capability_token,
            priority_score=self.priority_score,
        )


def compute_capability_priority(agent: SwarmAgentDescriptor) -> float:
    """Deterministically compute a strictly positive right-of-way priority weight.

    Agents with tighter physical safety caveats on their ``CapabilityToken``
    (lower ``max_linear_vel_mps``, ``max_kinetic_energy_j``,
    ``max_contact_force_n``, or narrower delegation ``depth_remaining``) are
    more dynamically constrained (e.g., carrying fragile liquid or operating in
    close human proximity) and therefore receive a higher priority ``p_i > 0``.

    In pairwise interactions, ego evasion burden is
    ``omega_ij = p_j / (p_i + p_j)``, so higher ``p_i`` yields smaller evasion
    burden ``omega_ij`` while preserving ``omega_ij + omega_ji = 1``.
    """
    if not isinstance(agent, SwarmAgentDescriptor):
        raise TypeError("agent must be a SwarmAgentDescriptor")

    if agent.priority_score is not None:
        return float(np.clip(agent.priority_score, _MIN_PRIORITY, _MAX_PRIORITY))

    token = agent.capability_token
    if token is None or not token.caveats:
        return 1.0

    eff_vel: Optional[float] = None
    eff_ke: Optional[float] = None
    eff_force: Optional[float] = None
    eff_depth: Optional[int] = None

    for idx, cav in enumerate(token.caveats):
        if idx > 0 and eff_depth is not None:
            eff_depth = max(0, eff_depth - 1)
        if cav.depth_remaining is not None:
            d_val = max(0, int(cav.depth_remaining))
            eff_depth = d_val if eff_depth is None else min(eff_depth, d_val)

        if cav.max_linear_vel_mps is not None and math.isfinite(cav.max_linear_vel_mps):
            v_val = max(0.0, float(cav.max_linear_vel_mps))
            eff_vel = v_val if eff_vel is None else min(eff_vel, v_val)

        if cav.max_kinetic_energy_j is not None and math.isfinite(cav.max_kinetic_energy_j):
            ke_val = max(0.0, float(cav.max_kinetic_energy_j))
            eff_ke = ke_val if eff_ke is None else min(eff_ke, ke_val)

        if cav.max_contact_force_n is not None and math.isfinite(cav.max_contact_force_n):
            f_val = max(0.0, float(cav.max_contact_force_n))
            eff_force = f_val if eff_force is None else min(eff_force, f_val)

    priority = 1.0
    if eff_vel is not None:
        priority *= (_REF_LINEAR_VEL_MPS / max(eff_vel, 0.02)) ** 0.5
    if eff_ke is not None:
        priority *= (_REF_KINETIC_ENERGY_J / max(eff_ke, 0.05)) ** 0.65
    if eff_force is not None:
        priority *= (_REF_CONTACT_FORCE_N / max(eff_force, 0.5)) ** 0.5
    if eff_depth is not None:
        priority *= ((1.0 + _REF_DEPTH_REMAINING) / (1.0 + eff_depth)) ** 0.25

    return float(np.clip(priority, _MIN_PRIORITY, _MAX_PRIORITY))


def compute_responsibility_share(ego: SwarmAgentDescriptor, neighbor: SwarmAgentDescriptor) -> float:
    """Compute ego agent's evasion responsibility share omega_ij = p_j / (p_i + p_j) in (0, 1)."""
    p_i = compute_capability_priority(ego)
    p_j = compute_capability_priority(neighbor)
    return float(p_j / (p_i + p_j))


def _partition_rhs(omega_ij: float, joint_rhs: float) -> float:
    """Partition symmetric constraint 2 r_ij^T u_i + 2 r_ji^T u_j >= joint_rhs.

    Satisfies:
      - Exact partition of unity: ``_partition_rhs(w, B) + _partition_rhs(1 - w, B) == B``
        for all ``B`` in R and ``w`` in (0, 1).
      - Monotonic right-of-way: ``d/dw _partition_rhs(w, B) = |B| >= 0``, so a
        higher-priority agent (smaller ``omega_ij``) receives strictly more slack
        when ``B < 0`` and strictly smaller repulsive burden when ``B > 0``.
    """
    if joint_rhs >= 0.0:
        return float(omega_ij * joint_rhs)
    return float((1.0 - omega_ij) * joint_rhs)


class DecentralizedSwarmCBF:
    """Decentralized multi-agent CBF/HOCBF coordinator with vortex deadlock breaking."""

    def __init__(
        self,
        dim: int = 2,
        dynamics_mode: str = "single_integrator",
        sensing_range: float = 6.0,
        safety_margin: float = 0.0,
        conformal_margin: float = 0.0,
        alpha: float = 2.0,
        alpha1: float = 2.0,
        alpha2: float = 3.0,
        default_u_min: Optional[Sequence[float]] = None,
        default_u_max: Optional[Sequence[float]] = None,
        enable_circulation: bool = True,
        circulation_gain: float = 0.9,
        circulation_cone_angle_rad: float = math.radians(55.0),
        tangential_speed_threshold: float = 0.8,
        circulation_range_factor: float = 7.0,
        helicity_axis_3d: Sequence[float] = (0.17, -0.29, 0.94),
        max_iters: int = 200,
        tol: float = 1e-9,
    ):
        if int(dim) not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        if dynamics_mode not in ("single_integrator", "double_integrator"):
            raise ValueError(
                f"dynamics_mode must be 'single_integrator' or 'double_integrator', got {dynamics_mode!r}"
            )
        self.dim = int(dim)
        self.dynamics_mode = dynamics_mode
        self.sensing_range = max(0.0, float(sensing_range))
        self.safety_margin = max(0.0, float(safety_margin))
        self.conformal_margin = max(0.0, float(conformal_margin))
        self.alpha = max(1e-3, float(alpha))
        self.alpha1 = max(1e-3, float(alpha1))
        self.alpha2 = max(1e-3, float(alpha2))

        default_bound = 2.0 if dynamics_mode == "single_integrator" else 8.0
        self.default_u_min = (
            -default_bound * np.ones(self.dim, dtype=np.float64)
            if default_u_min is None
            else _as_vec(default_u_min, "default_u_min", expected_dim=self.dim)
        )
        self.default_u_max = (
            default_bound * np.ones(self.dim, dtype=np.float64)
            if default_u_max is None
            else _as_vec(default_u_max, "default_u_max", expected_dim=self.dim)
        )

        self.enable_circulation = bool(enable_circulation)
        self.circulation_gain = max(0.0, float(circulation_gain))
        self.circulation_cone_angle_rad = float(np.clip(circulation_cone_angle_rad, 1e-3, math.pi * 0.95))
        self._cos_cone = math.cos(self.circulation_cone_angle_rad)
        self.tangential_speed_threshold = max(1e-3, float(tangential_speed_threshold))
        self.circulation_range_factor = max(1.5, float(circulation_range_factor))

        axis_raw = np.asarray(helicity_axis_3d, dtype=np.float64).flatten()
        if axis_raw.shape[0] != 3 or np.linalg.norm(axis_raw) <= 1e-9:
            raise ValueError("helicity_axis_3d must be a non-zero 3D vector")
        self._helicity_axis = axis_raw / float(np.linalg.norm(axis_raw))
        # Deterministic orthogonal fallback axis for the measure-zero collinear line r_hat == +-helicity_axis
        ref_alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(self._helicity_axis, ref_alt))) > 0.9:
            ref_alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ortho = ref_alt - float(np.dot(ref_alt, self._helicity_axis)) * self._helicity_axis
        self._helicity_fallback = ortho / float(np.linalg.norm(ortho))

        self.max_iters = int(max_iters)
        self.tol = float(tol)

        if self.dynamics_mode == "single_integrator":
            self._ego_dynamics: ControlAffineDynamics = LinearDynamics(
                np.zeros((self.dim, self.dim), dtype=np.float64),
                np.eye(self.dim, dtype=np.float64),
            )
        else:
            A_dbl = np.zeros((2 * self.dim, 2 * self.dim), dtype=np.float64)
            A_dbl[: self.dim, self.dim :] = np.eye(self.dim, dtype=np.float64)
            B_dbl = np.vstack(
                [
                    np.zeros((self.dim, self.dim), dtype=np.float64),
                    np.eye(self.dim, dtype=np.float64),
                ]
            )
            self._ego_dynamics = LinearDynamics(A_dbl, B_dbl)

    def set_conformal_margin(self, delta: float) -> None:
        """Set perception/localization uncertainty margin added to safe separation."""
        self.conformal_margin = max(0.0, float(delta))

    def safe_separation(self, ego: SwarmAgentDescriptor, neighbor: SwarmAgentDescriptor) -> float:
        """Minimum certified center-to-center distance D_ij."""
        return float(ego.radius + neighbor.radius + self.safety_margin + self.conformal_margin)

    def responsibility_share(self, ego: SwarmAgentDescriptor, neighbor: SwarmAgentDescriptor) -> float:
        """Return ego evasion share omega_ij = p_j / (p_i + p_j)."""
        return compute_responsibility_share(ego, neighbor)

    def circulation_tangent(self, r_hat: np.ndarray) -> np.ndarray:
        """Compute chirality-consistent unit tangent orthogonal to r_hat.

        Satisfies ``t(-r_hat) == -t(r_hat)`` and ``t(r_hat) . r_hat == 0`` so two
        antiparallel agents always deflect toward opposite sides in world space.
        """
        r_u = np.asarray(r_hat, dtype=np.float64).flatten()
        norm_r = float(np.linalg.norm(r_u))
        if norm_r <= 1e-12:
            out = np.zeros_like(r_u)
            if out.size >= 2:
                out[1] = 1.0
            return out
        r_u = r_u / norm_r

        if r_u.shape[0] == 2:
            # Right-hand 90-degree rotation R_{pi/2} r_hat = [-r_y, r_x]
            return np.array([-r_u[1], r_u[0]], dtype=np.float64)

        cross_vec = np.cross(self._helicity_axis, r_u)
        c_norm = float(np.linalg.norm(cross_vec))
        if c_norm < 1e-4:
            cross_vec = np.cross(self._helicity_fallback, r_u)
            c_norm = float(np.linalg.norm(cross_vec))
        return cross_vec / max(c_norm, 1e-12)

    def pairwise_halfspace(
        self,
        ego: SwarmAgentDescriptor,
        neighbor: SwarmAgentDescriptor,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Assemble decentralized pairwise halfspace ``a_ij^T u_i >= b_ij`` without ``u_j``."""
        r_ij = ego.position - neighbor.position
        dist = float(np.linalg.norm(r_ij))
        d_safe = self.safe_separation(ego, neighbor)
        h_ij = float(np.dot(r_ij, r_ij) - d_safe * d_safe)
        omega_ij = self.responsibility_share(ego, neighbor)

        if dist <= 1e-9:
            # Degenerate exact overlap fallback: deterministic repulsive direction
            fallback_dir = np.zeros(self.dim, dtype=np.float64)
            fallback_dir[0] = 1.0 if ego.agent_id < neighbor.agent_id else -1.0
            a_row = 2.0 * d_safe * fallback_dir
        else:
            a_row = 2.0 * r_ij

        prox_blend = math.exp(-max(0.0, h_ij) / max(3.0 * d_safe * d_safe, 1e-6))
        omega_eff = 0.5 * (1.0 - prox_blend) + omega_ij * prox_blend

        if self.dynamics_mode == "single_integrator":
            joint_rhs = -self.alpha * h_ij
            b_val = _partition_rhs(omega_eff, joint_rhs)
            h_dot = float(2.0 * np.dot(r_ij, ego.velocity - neighbor.velocity))
        else:
            v_ij = ego.velocity - neighbor.velocity
            h_dot = float(2.0 * np.dot(r_ij, v_ij))
            v_rel_sq = float(np.dot(v_ij, v_ij))
            joint_rhs = -2.0 * v_rel_sq - (self.alpha1 + self.alpha2) * h_dot - (self.alpha1 * self.alpha2) * h_ij
            b_val = _partition_rhs(omega_eff, joint_rhs)

        meta = {
            "neighbor_id": neighbor.agent_id,
            "distance": dist,
            "d_safe": d_safe,
            "clearance_margin": dist - d_safe,
            "h_ij": h_ij,
            "h_dot_ij": h_dot,
            "omega_ij": omega_ij,
            "joint_rhs": joint_rhs,
        }
        return a_row, b_val, meta

    def _compute_circulation_bias(
        self,
        ego: SwarmAgentDescriptor,
        neighbors: Sequence[SwarmAgentDescriptor],
        u_nom: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        if not self.enable_circulation or self.circulation_gain <= 0.0:
            return np.zeros(self.dim, dtype=np.float64), False

        bias = np.zeros(self.dim, dtype=np.float64)
        engaged = False
        u_norm = float(np.linalg.norm(u_nom))
        v_norm = float(np.linalg.norm(ego.velocity))
        box_span = float(np.min(0.5 * (u_max - u_min)))
        if self.dynamics_mode == "single_integrator":
            ref_mag = u_norm
        else:
            ref_mag = max(u_norm, 1.6 * v_norm)
        if ref_mag <= 1e-6:
            return np.zeros(self.dim, dtype=np.float64), False

        for nb in neighbors:
            if nb.agent_id == ego.agent_id:
                continue
            r_ij = ego.position - nb.position
            dist = float(np.linalg.norm(r_ij))
            if dist <= 1e-9 or dist > self.sensing_range:
                continue

            d_safe = self.safe_separation(ego, nb)
            circ_range = min(self.sensing_range, self.circulation_range_factor * max(d_safe, 0.2))
            if dist >= circ_range:
                continue

            r_hat = r_ij / dist
            # Detect head-on opposition between grad_{p_i} h_ij (+r_hat) and nominal intent
            opp_cos = 0.0
            if u_norm > 1e-6:
                opp_cos = max(opp_cos, -float(np.dot(r_hat, u_nom / u_norm)))
            if self.dynamics_mode == "double_integrator":
                intent_v = ego.velocity + 0.35 * u_nom
                iv_norm = float(np.linalg.norm(intent_v))
                if iv_norm > 1e-6:
                    opp_cos = max(opp_cos, -float(np.dot(r_hat, intent_v / iv_norm)))
                if v_norm > 1e-6:
                    opp_cos = max(opp_cos, -float(np.dot(r_hat, ego.velocity / v_norm)))

            if opp_cos <= self._cos_cone:
                continue

            w_cone = (opp_cos - self._cos_cone) / max(1.0 - self._cos_cone, 1e-6)

            # Relative velocity and closing rate along -r_hat: only circulate when closing
            if self.dynamics_mode == "single_integrator":
                v_rel_ref = u_nom - nb.velocity
            else:
                v_rel_ref = (ego.velocity + 0.25 * u_nom) - nb.velocity
            closing_speed = -float(np.dot(r_hat, v_rel_ref))
            if closing_speed <= 1e-4:
                continue
            conflict_horizon_dist = d_safe + max(2.2 * d_safe, 2.5 * closing_speed)
            if dist >= min(circ_range, conflict_horizon_dist):
                continue

            v_tan_vec = v_rel_ref - float(np.dot(v_rel_ref, r_hat)) * r_hat
            v_tan = float(np.linalg.norm(v_tan_vec))
            eff_tan_thresh = max(self.tangential_speed_threshold, 0.75 * max(u_norm, v_norm))
            if v_tan >= eff_tan_thresh:
                continue

            w_tan = 1.0 - (v_tan / eff_tan_thresh)
            w_prox = float(np.clip((circ_range - dist) / max(circ_range - d_safe, 1e-6), 0.15, 1.0))
            omega_ij = self.responsibility_share(ego, nb)

            # Weight by 2 * omega_ij: 1.0 for equal agents, ~0 for high-priority fragile agents,
            # ~2.0 for low-priority agile agents yielding right-of-way.
            amp = (2.0 * omega_ij) * self.circulation_gain * ref_mag * w_cone * w_tan * w_prox
            if amp > 1e-9:
                tangent = self.circulation_tangent(r_hat)
                bias = bias + amp * tangent
                engaged = True

        max_bias = 0.85 * box_span
        bias_norm = float(np.linalg.norm(bias))
        if bias_norm > max_bias and bias_norm > 1e-12:
            bias = bias * (max_bias / bias_norm)

        return bias, engaged

    def _solve_dykstra(
        self,
        A: np.ndarray,
        b: np.ndarray,
        u_target: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        m = A.shape[0]
        sets: List[Tuple[str, Any]] = [("halfspace", (A[i], b[i])) for i in range(m)]
        sets.append(("box", None))

        s_list = [np.zeros_like(u_target) for _ in range(len(sets))]
        w = u_target.copy()
        for _ in range(self.max_iters):
            w_cycle_start = w.copy()
            increments_stable = True
            for j, (kind, payload) in enumerate(sets):
                y = w + s_list[j]
                if kind == "halfspace":
                    a_row, bj = payload
                    a_norm2 = float(a_row @ a_row)
                    viol = bj - float(a_row @ y)
                    if viol > 0.0 and a_norm2 > 1e-18:
                        proj = y + (viol / a_norm2) * a_row
                    else:
                        proj = y
                else:
                    proj = np.minimum(np.maximum(y, u_min), u_max)
                s_new = y - proj
                if float(np.max(np.abs(s_new - s_list[j]))) > self.tol:
                    increments_stable = False
                s_list[j] = s_new
                w = proj
            if increments_stable and float(np.max(np.abs(w - w_cycle_start))) <= self.tol:
                break

        residual = float(np.min(A @ w - b)) if A.size else 0.0
        box_ok = bool(np.all(w >= u_min - 1e-6) and np.all(w <= u_max + 1e-6))
        verified = bool(np.all(np.isfinite(w)) and box_ok and (not A.size or residual >= -1e-6))
        return w, residual, verified

    def filter_agent_action(
        self,
        ego: SwarmAgentDescriptor,
        neighbors: Sequence[SwarmAgentDescriptor],
        u_nominal: Sequence[float],
        extra_barriers: Sequence[BarrierConstraint] = (),
    ) -> Dict[str, Any]:
        """Filter ``u_nominal`` for ``ego`` using only local neighbor ``(position, velocity)``."""
        t0 = time.perf_counter()
        if ego.dim != self.dim:
            raise ValueError(f"ego.dim {ego.dim} != coordinator dim {self.dim}")

        u_nom = _as_vec(u_nominal, "u_nominal", expected_dim=self.dim)
        u_min = self.default_u_min if ego.u_min is None else ego.u_min
        u_max = self.default_u_max if ego.u_max is None else ego.u_max

        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []
        hard_flags: List[bool] = []
        shares: Dict[str, float] = {}
        min_h = float("inf")
        min_clearance = float("inf")

        for nb in neighbors:
            if nb.agent_id == ego.agent_id:
                continue
            if nb.dim != self.dim:
                raise ValueError(f"neighbor {nb.agent_id!r} dim {nb.dim} != coordinator dim {self.dim}")
            dist = float(np.linalg.norm(ego.position - nb.position))
            d_safe = self.safe_separation(ego, nb)
            min_clearance = min(min_clearance, dist - d_safe)
            if dist > self.sensing_range:
                continue
            a_ij, b_ij, meta = self.pairwise_halfspace(ego, nb)
            A_rows.append(a_ij)
            b_rows.append(b_ij)
            hard_flags.append(True)
            shares[nb.agent_id] = meta["omega_ij"]
            min_h = min(min_h, meta["h_ij"])

        if extra_barriers:
            ego_filter = StateCBFSafetyFilter(
                self._ego_dynamics,
                list(extra_barriers),
                u_min=u_min,
                u_max=u_max,
                conformal_margin=self.conformal_margin,
            )
            x_ego = ego.state_vector(self.dynamics_mode)
            A_ext, b_ext, _alphas, ext_hard = ego_filter._halfspaces_with_flags(x_ego)
            for row_idx in range(A_ext.shape[0]):
                A_rows.append(A_ext[row_idx])
                b_rows.append(float(b_ext[row_idx]))
                hard_flags.append(bool(ext_hard[row_idx]))

        A = np.vstack(A_rows) if A_rows else np.zeros((0, self.dim), dtype=np.float64)
        b = np.asarray(b_rows, dtype=np.float64) if b_rows else np.zeros(0, dtype=np.float64)

        circ_bias, circ_engaged = self._compute_circulation_bias(ego, neighbors, u_nom, u_min, u_max)
        u_biased = u_nom + circ_bias

        u_res, slack, verified = self._solve_dykstra(A, b, u_biased, u_min, u_max)
        status = "safe"

        if not verified and A.shape[0] > 0:
            hard_mask = np.asarray(hard_flags, dtype=bool)
            if int(np.sum(hard_mask)) < A.shape[0] and int(np.sum(hard_mask)) > 0:
                u_hard, slack_hard, ok_hard = self._solve_dykstra(
                    A[hard_mask], b[hard_mask], u_biased, u_min, u_max
                )
                if ok_hard:
                    u_res, slack, verified = u_hard, slack_hard, True
                    status = "safe_soft_relaxed"

        if not verified:
            # Fail-closed braking + repulsive separation fallback within actuator bounds
            status = "infeasible_braking_fallback"
            repulse = np.zeros(self.dim, dtype=np.float64)
            for nb in neighbors:
                if nb.agent_id == ego.agent_id:
                    continue
                diff = ego.position - nb.position
                dn = float(np.linalg.norm(diff))
                if 1e-9 < dn <= self.sensing_range:
                    repulse += (diff / dn) / max(dn, 0.1)
            if self.dynamics_mode == "single_integrator":
                rn = float(np.linalg.norm(repulse))
                u_fb = np.zeros(self.dim, dtype=np.float64) if rn <= 1e-9 else 0.5 * (repulse / rn)
            else:
                u_fb = -3.0 * ego.velocity + 2.0 * repulse
            u_res = np.minimum(np.maximum(u_fb, u_min), u_max)
            slack = float(np.min(A @ u_res - b)) if A.size else 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": status,
            "verified_safe": verified,
            "agent_id": ego.agent_id,
            "dynamics_mode": self.dynamics_mode,
            "priority": compute_capability_priority(ego),
            "u_nominal": u_nom.tolist(),
            "u_biased": u_biased.tolist(),
            "u_filtered": u_res.tolist(),
            "circulation_engaged": circ_engaged,
            "circulation_bias": circ_bias.tolist(),
            "active_neighbors": len(shares),
            "responsibility_shares": shares,
            "min_pairwise_h": min_h,
            "min_pairwise_clearance_margin": min_clearance,
            "min_barrier_slack": round(slack, 9),
            "n_constraints": int(A.shape[0]),
            "latency_ms": round(latency_ms, 4),
        }

    def _goal_nominal_control(
        self,
        agent: SwarmAgentDescriptor,
        goal: np.ndarray,
        kp: float,
        kd: float,
    ) -> np.ndarray:
        err = goal - agent.position
        u_min = self.default_u_min if agent.u_min is None else agent.u_min
        u_max = self.default_u_max if agent.u_max is None else agent.u_max

        max_speed: Optional[float] = None
        if agent.capability_token is not None:
            for cav in agent.capability_token.caveats:
                if cav.max_linear_vel_mps is not None and math.isfinite(cav.max_linear_vel_mps):
                    v_lim = max(0.0, float(cav.max_linear_vel_mps))
                    max_speed = v_lim if max_speed is None else min(max_speed, v_lim)

        if self.dynamics_mode == "single_integrator":
            u_raw = kp * err
            speed = float(np.linalg.norm(u_raw))
            cap = float(np.min(np.minimum(np.abs(u_min), np.abs(u_max))))
            if max_speed is not None:
                cap = min(cap, max_speed)
            if speed > cap and speed > 1e-12:
                u_raw = u_raw * (cap / speed)
            return np.minimum(np.maximum(u_raw, u_min), u_max)

        # Double integrator: velocity-saturated PD tracking
        v_des = kp * err
        v_des_norm = float(np.linalg.norm(v_des))
        v_cap = max_speed if max_speed is not None else 1.5
        if v_des_norm > v_cap and v_des_norm > 1e-12:
            v_des = v_des * (v_cap / v_des_norm)
        u_raw = kd * (v_des - agent.velocity)
        return np.minimum(np.maximum(u_raw, u_min), u_max)

    def step_swarm_simulation(
        self,
        agents: Sequence[SwarmAgentDescriptor],
        *,
        nominal_controls: Optional[Union[Mapping[str, Sequence[float]], Sequence[Sequence[float]]]] = None,
        goals: Optional[Union[Mapping[str, Sequence[float]], Sequence[Sequence[float]]]] = None,
        nominal_controller_fn: Optional[Callable[[SwarmAgentDescriptor, int], Sequence[float]]] = None,
        dt: float = 0.02,
        n_steps: int = 1,
        kp: float = 1.4,
        kd: float = 3.2,
        goal_tolerance: float = 0.08,
        extra_barriers_by_agent: Optional[Mapping[str, Sequence[BarrierConstraint]]] = None,
    ) -> Dict[str, Any]:
        """Step all swarm agents synchronously without sharing neighbor controls."""
        if not agents:
            raise ValueError("agents must be non-empty")
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f <= 0.0:
            raise ValueError("dt must be positive and finite")
        steps = max(1, int(n_steps))

        current_agents: List[SwarmAgentDescriptor] = [
            ag.with_kinematics(ag.position.copy(), ag.velocity.copy()) for ag in agents
        ]
        n_agents = len(current_agents)

        goals_map: Optional[Dict[str, np.ndarray]] = None
        if goals is not None:
            goals_map = {}
            if isinstance(goals, Mapping):
                for ag in current_agents:
                    goals_map[ag.agent_id] = _as_vec(goals[ag.agent_id], f"goal[{ag.agent_id}]", self.dim)
            else:
                if len(goals) != n_agents:
                    raise ValueError("goals sequence length must match agents length")
                for ag, g_vec in zip(current_agents, goals):
                    goals_map[ag.agent_id] = _as_vec(g_vec, f"goal[{ag.agent_id}]", self.dim)

        trajectories: Dict[str, np.ndarray] = {
            ag.agent_id: np.zeros((steps + 1, self.dim), dtype=np.float64) for ag in current_agents
        }
        for ag in current_agents:
            trajectories[ag.agent_id][0] = ag.position

        min_pairwise_dist = float("inf")
        min_clearance_margin = float("inf")
        all_verified_safe = True
        circulation_steps = 0
        last_receipts: Dict[str, Dict[str, Any]] = {}

        def _update_pairwise_metrics(snapshot: Sequence[SwarmAgentDescriptor]) -> None:
            nonlocal min_pairwise_dist, min_clearance_margin
            for i in range(len(snapshot)):
                for j in range(i + 1, len(snapshot)):
                    d = float(np.linalg.norm(snapshot[i].position - snapshot[j].position))
                    d_req = self.safe_separation(snapshot[i], snapshot[j])
                    min_pairwise_dist = min(min_pairwise_dist, d)
                    min_clearance_margin = min(min_clearance_margin, d - d_req)

        _update_pairwise_metrics(current_agents)

        for step_idx in range(steps):
            frozen_snapshot = [
                ag.with_kinematics(ag.position.copy(), ag.velocity.copy()) for ag in current_agents
            ]
            step_receipts: Dict[str, Dict[str, Any]] = {}
            filtered_u: List[np.ndarray] = []
            any_circ = False

            for idx, ego in enumerate(frozen_snapshot):
                if nominal_controller_fn is not None:
                    u_nom = _as_vec(nominal_controller_fn(ego, step_idx), "u_nominal", self.dim)
                elif nominal_controls is not None:
                    raw_u = (
                        nominal_controls[ego.agent_id]
                        if isinstance(nominal_controls, Mapping)
                        else nominal_controls[idx]
                    )
                    u_nom = _as_vec(raw_u, "u_nominal", self.dim)
                elif goals_map is not None:
                    u_nom = self._goal_nominal_control(ego, goals_map[ego.agent_id], kp=kp, kd=kd)
                else:
                    u_nom = np.zeros(self.dim, dtype=np.float64)

                extra = () if extra_barriers_by_agent is None else extra_barriers_by_agent.get(ego.agent_id, ())
                neighbors = [nb for j, nb in enumerate(frozen_snapshot) if j != idx]
                receipt = self.filter_agent_action(ego, neighbors, u_nom, extra_barriers=extra)
                step_receipts[ego.agent_id] = receipt
                filtered_u.append(np.asarray(receipt["u_filtered"], dtype=np.float64))
                all_verified_safe = all_verified_safe and bool(receipt["verified_safe"])
                any_circ = any_circ or bool(receipt["circulation_engaged"])

            if any_circ:
                circulation_steps += 1

            next_agents: List[SwarmAgentDescriptor] = []
            for ego, u_act in zip(frozen_snapshot, filtered_u):
                if self.dynamics_mode == "single_integrator":
                    new_pos = ego.position + dt_f * u_act
                    new_vel = u_act.copy()
                else:
                    new_pos = ego.position + dt_f * ego.velocity + 0.5 * (dt_f * dt_f) * u_act
                    new_vel = ego.velocity + dt_f * u_act
                updated = ego.with_kinematics(new_pos, new_vel)
                next_agents.append(updated)
                trajectories[updated.agent_id][step_idx + 1] = new_pos

            current_agents = next_agents
            last_receipts = step_receipts
            _update_pairwise_metrics(current_agents)

            if goals_map is not None and steps > 1:
                if all(
                    float(np.linalg.norm(ag.position - goals_map[ag.agent_id])) <= goal_tolerance
                    and (self.dynamics_mode == "single_integrator" or float(np.linalg.norm(ag.velocity)) <= 0.1)
                    for ag in current_agents
                ):
                    for ag in current_agents:
                        trajectories[ag.agent_id] = trajectories[ag.agent_id][: step_idx + 2]
                    break

        out: Dict[str, Any] = {
            "agents": current_agents,
            "per_agent": last_receipts,
            "min_pairwise_distance": min_pairwise_dist,
            "min_pairwise_clearance_margin": min_clearance_margin,
            "all_verified_safe": all_verified_safe,
            "circulation_steps": circulation_steps,
            "trajectories": trajectories,
        }
        if goals_map is not None:
            goal_errors = {
                ag.agent_id: float(np.linalg.norm(ag.position - goals_map[ag.agent_id])) for ag in current_agents
            }
            out["goal_errors"] = goal_errors
            out["all_reached_goals"] = all(err <= goal_tolerance for err in goal_errors.values())
        return out
