"""Port-Hamiltonian energy tank passivity filter and online contact manifold learner.

Provides certified coupled passivity and hybrid force/motion compliance for
contact-rich robotic manipulation:

1. ``EnergyTankPassivityFilter`` — governs variable Cartesian or joint impedance
   commands ``(K_p(t), K_d(t), x_d(t), F_ff(t))`` via a virtual energy tank
   ``E_tank(t) in [E_min, E_max]`` with ``E_min > 0``. Whenever positive stiffness
   rates ``dK_p/dt > 0`` or active feedforward wrenches ``F_ff`` would deplete the
   tank below ``E_min``, an analytical passivity-CBF valve ``gamma_valve in [0, 1]``
   scales back ``Delta K_p^+`` and active ``F_ff`` so coupled passivity holds
   unconditionally against any passive environment of arbitrary stiffness.

2. ``ContactManifoldLearner`` — online exponentially-weighted SVD/covariance
   estimator over active contact samples ``(velocity_3d, force_3d)`` (and optional
   ``position_3d``) when ``||F|| > F_contact_min``. Identifies ``"free_space"``,
   ``"planar_surface"``, ``"prismatic_1d"``, or ``"revolute_hinge"`` and projects
   nominal motion/force commands onto orthogonal compliant subspaces.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple, Union

import numpy as np


def _to_1d(vec: Union[float, Sequence[float], np.ndarray], dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).flatten()
    if dim is not None and arr.shape[0] != dim:
        if arr.shape[0] == 1:
            arr = np.full(dim, float(arr[0]), dtype=np.float64)
        else:
            raise ValueError(f"Expected vector of length {dim}, got shape {arr.shape}")
    return arr


def _gain_to_matrix(
    gain: Union[float, Sequence[float], np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, str]:
    """Convert scalar, 1D diagonal, or 2D matrix gain into (dim, dim) symmetric PSD matrix."""
    raw = np.asarray(gain, dtype=np.float64)
    if raw.ndim == 0:
        val = max(0.0, float(raw))
        return val * np.eye(dim, dtype=np.float64), "scalar"
    if raw.ndim == 1:
        if raw.shape[0] == 1 and dim > 1:
            diag = np.full(dim, max(0.0, float(raw[0])), dtype=np.float64)
        elif raw.shape[0] == dim:
            diag = np.maximum(0.0, raw.astype(np.float64))
        else:
            raise ValueError(f"1D gain length {raw.shape[0]} does not match state dim {dim}")
        return np.diag(diag), "diag"
    if raw.ndim == 2 and raw.shape == (dim, dim):
        sym = 0.5 * (raw + raw.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals_psd = np.maximum(0.0, eigvals)
        mat = (eigvecs * eigvals_psd) @ eigvecs.T
        return 0.5 * (mat + mat.T), "matrix"
    raise ValueError(f"Gain shape {raw.shape} incompatible with dimension {dim}")


def _format_gain_output(mat: np.ndarray, kind: str) -> Union[float, np.ndarray]:
    if kind == "scalar":
        return float(np.mean(np.diag(mat)))
    if kind == "diag":
        return np.diag(mat).copy()
    return mat.copy()


def _split_symmetric_psd_nsd(delta_mat: np.ndarray, is_diagonal: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Split symmetric matrix Delta K into PSD stiffening part Delta K^+ and NSD softening part Delta K^-."""
    if is_diagonal:
        d = np.diag(delta_mat)
        return np.diag(np.maximum(0.0, d)), np.diag(np.minimum(0.0, d))
    sym = 0.5 * (delta_mat + delta_mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    pos = (eigvecs * np.maximum(0.0, eigvals)) @ eigvecs.T
    neg = (eigvecs * np.minimum(0.0, eigvals)) @ eigvecs.T
    return 0.5 * (pos + pos.T), 0.5 * (neg + neg.T)


def _project_psd(mat: np.ndarray, is_diagonal: bool) -> np.ndarray:
    if is_diagonal:
        return np.diag(np.maximum(0.0, np.diag(mat)))
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    proj = (eigvecs * np.maximum(0.0, eigvals)) @ eigvecs.T
    return 0.5 * (proj + proj.T)


class EnergyTankPassivityFilter:
    """Port-Hamiltonian virtual energy tank filter certifying variable impedance passivity.

    Maintains tank energy ``E_tank(t) in [E_min, E_max]`` with ``E_min > 0`` and enforces
    the discrete passivity Control Barrier Function condition:
        E_tank(t + dt) - E_tank(t) >= -alpha_tank * (E_tank(t) - E_min)
    by computing the optimal valve factor ``gamma_valve in [0, 1]`` scaling back positive
    stiffness increments ``Delta K_p^+`` and active feedforward wrenches ``F_ff``.
    """

    def __init__(
        self,
        dt: float = 0.001,
        e_init: float = 1.0,
        e_min: float = 0.05,
        e_max: float = 5.0,
        alpha_tank: float = 0.5,
        dissipation_rate: float = 1.0,
        k_p_init: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if e_min <= 0.0:
            raise ValueError("e_min must be strictly positive (E_min > 0)")
        if e_max < e_min:
            raise ValueError("e_max must be >= e_min")
        if alpha_tank <= 0.0:
            raise ValueError("alpha_tank must be positive")

        self.dt = float(dt)
        self.e_min = float(e_min)
        self.e_max = float(e_max)
        self.e_init = float(np.clip(e_init, self.e_min, self.e_max))
        self.tank_energy: float = self.e_init
        self.alpha_tank = float(alpha_tank)
        self.dissipation_rate = float(np.clip(dissipation_rate, 0.0, 1.0))

        self._k_p_init_spec = k_p_init
        self.K_p_prev: Optional[np.ndarray] = None
        self.last_receipt: Dict[str, Any] = {}
        self.steps: int = 0
        self.interventions: int = 0

    def reset(
        self,
        e_init: Optional[float] = None,
        k_p_init: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ) -> None:
        if e_init is not None:
            self.e_init = float(np.clip(e_init, self.e_min, self.e_max))
        self.tank_energy = self.e_init
        if k_p_init is not None:
            self._k_p_init_spec = k_p_init
        self.K_p_prev = None
        self.last_receipt = {}
        self.steps = 0
        self.interventions = 0

    def filter(
        self,
        x: Union[float, Sequence[float], np.ndarray],
        dx: Union[float, Sequence[float], np.ndarray],
        K_p: Union[float, Sequence[float], np.ndarray],
        K_d: Union[float, Sequence[float], np.ndarray],
        x_d: Union[float, Sequence[float], np.ndarray],
        F_ff: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        dt: Optional[float] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], np.ndarray, float, float, bool]:
        """Filter variable impedance command and step virtual energy tank.

        Returns:
            ``(K_p_safe, K_d_safe, u_wrench_safe, tank_energy, gamma_valve, passivity_certified)``
        """
        step_dt = float(self.dt if dt is None else dt)
        if step_dt <= 0.0:
            raise ValueError("dt must be positive")

        x_vec = _to_1d(x)
        dim = x_vec.shape[0]
        dx_vec = _to_1d(dx, dim=dim)
        xd_vec = _to_1d(x_d, dim=dim)
        f_ff_vec = np.zeros(dim, dtype=np.float64) if F_ff is None else _to_1d(F_ff, dim=dim)

        kp_mat, kp_kind = _gain_to_matrix(K_p, dim)
        kd_mat, kd_kind = _gain_to_matrix(K_d, dim)
        is_diag = kp_kind in ("scalar", "diag")

        if self.K_p_prev is None:
            if self._k_p_init_spec is not None:
                init_mat, _ = _gain_to_matrix(self._k_p_init_spec, dim)
                self.K_p_prev = init_mat
            else:
                self.K_p_prev = kp_mat.copy()
        elif self.K_p_prev.shape != (dim, dim):
            raise ValueError(f"State dimension changed from {self.K_p_prev.shape[0]} to {dim}")

        x_tilde = x_vec - xd_vec

        # Dissipated power from physical/commanded damping: P_diss = dx^T K_d dx >= 0
        p_diss = max(0.0, float(dx_vec @ kd_mat @ dx_vec))

        # Stiffness variation decomposition: Delta K_p = Delta K_p^+ + Delta K_p^-
        delta_kp = kp_mat - self.K_p_prev
        delta_kp_pos, delta_kp_neg = _split_symmetric_psd_nsd(delta_kp, is_diagonal=is_diag)

        p_stiff_pos = max(0.0, 0.5 * float(x_tilde @ delta_kp_pos @ x_tilde) / step_dt)
        p_stiff_neg = min(0.0, 0.5 * float(x_tilde @ delta_kp_neg @ x_tilde) / step_dt)
        p_ff = float(np.dot(dx_vec, f_ff_vec))
        p_inject_raw = 0.5 * float(x_tilde @ delta_kp @ x_tilde) / step_dt + p_ff

        # Discrete passivity CBF rate factor in (0, 1] ensures E_tank(t + dt) >= E_min > 0
        alpha_eff = min(1.0, max(1e-9, self.alpha_tank))
        tank_reserve = max(0.0, self.tank_energy - self.e_min)
        p_max_allow = self.dissipation_rate * p_diss + (alpha_eff * tank_reserve) / step_dt

        p_active = p_stiff_pos + max(0.0, p_ff)
        p_passive = p_stiff_neg + min(0.0, p_ff)

        if p_inject_raw <= p_max_allow + 1e-12:
            if tank_reserve <= 1e-9 and p_diss <= 1e-12 and float(np.linalg.norm(f_ff_vec)) > 0.0 and p_ff == 0.0:
                # Prevent unthrottled static feedforward push when tank is completely depleted at rest
                gamma_valve = 0.0
            else:
                gamma_valve = 1.0
        else:
            if p_active <= 1e-15:
                gamma_valve = 0.0
            else:
                gamma_valve = float(np.clip((p_max_allow - p_passive) / p_active, 0.0, 1.0))

        if gamma_valve < 1.0 - 1e-9:
            self.interventions += 1

        kp_safe_mat = _project_psd(
            self.K_p_prev + delta_kp_neg + gamma_valve * delta_kp_pos,
            is_diagonal=is_diag,
        )
        # Scale active feedforward wrench; preserve purely dissipative braking when dx^T F_ff < 0
        f_ff_safe = (gamma_valve * f_ff_vec) if p_ff >= 0.0 else f_ff_vec.copy()

        p_inject_safe = (
            0.5 * float(x_tilde @ (kp_safe_mat - self.K_p_prev) @ x_tilde) / step_dt
            + float(np.dot(dx_vec, f_ff_safe))
        )

        # Drain tank by positive non-passive power injection; harvest damping dissipation up to E_max
        p_tank_drain = max(0.0, p_inject_safe)
        e_next_unclipped = self.tank_energy + step_dt * (self.dissipation_rate * p_diss - p_tank_drain)

        u_wrench_safe = -(kp_safe_mat @ x_tilde) - (kd_mat @ dx_vec) + f_ff_safe

        finite_ok = bool(
            np.all(np.isfinite(kp_safe_mat))
            and np.all(np.isfinite(kd_mat))
            and np.all(np.isfinite(u_wrench_safe))
            and np.isfinite(e_next_unclipped)
        )
        passivity_certified = bool(finite_ok and e_next_unclipped >= self.e_min - 1e-9)

        self.tank_energy = float(np.clip(e_next_unclipped, self.e_min, self.e_max))
        self.K_p_prev = kp_safe_mat.copy()
        self.steps += 1

        kp_safe_out = _format_gain_output(kp_safe_mat, kp_kind)
        kd_safe_out = _format_gain_output(kd_mat, kd_kind)

        self.last_receipt = {
            "step": self.steps,
            "P_diss": p_diss,
            "P_inject_raw": p_inject_raw,
            "P_inject_safe": p_inject_safe,
            "P_max_allow": p_max_allow,
            "tank_energy": self.tank_energy,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "gamma_valve": gamma_valve,
            "passivity_certified": passivity_certified,
            "F_ff_safe": f_ff_safe.copy(),
            "u_wrench_safe": u_wrench_safe.copy(),
        }
        return kp_safe_out, kd_safe_out, u_wrench_safe, self.tank_energy, gamma_valve, passivity_certified

    step = filter


class ContactManifoldLearner:
    """Online exponentially-weighted covariance / SVD estimator for 3D contact geometry.

    Classifies active contact kinematics into:
      - ``"free_space"``      : no sustained contact above ``f_contact_min``
      - ``"planar_surface"``  : 2D compliant tangent plane + 1D constrained normal (e.g. table wiping)
      - ``"prismatic_1d"``    : 1D compliant axis + 2D normal constraint plane (e.g. drawer/slider)
      - ``"revolute_hinge"``  : 1D circular arc around an estimated 3D hinge axis + lever arm
    """

    def __init__(
        self,
        f_contact_min: float = 1.0,
        ema_decay: float = 0.94,
        min_contact_samples: int = 8,
        free_space_release_steps: int = 10,
        rank_threshold: float = 0.22,
        window_size: int = 64,
    ):
        if f_contact_min < 0.0:
            raise ValueError("f_contact_min must be non-negative")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError("ema_decay must lie in (0, 1)")

        self.f_contact_min = float(f_contact_min)
        self.ema_decay = float(ema_decay)
        self.min_contact_samples = max(2, int(min_contact_samples))
        self.free_space_release_steps = max(1, int(free_space_release_steps))
        self.rank_threshold = float(rank_threshold)
        self.window_size = max(8, int(window_size))

        self.reset()

    def reset(self) -> None:
        self._mode: str = "free_space"
        self._contact_samples: int = 0
        self._non_contact_streak: int = 0
        self._M_v = np.zeros((3, 3), dtype=np.float64)
        self._M_f = np.zeros((3, 3), dtype=np.float64)
        self._f_mean = np.zeros(3, dtype=np.float64)
        self._singular_values = np.zeros(3, dtype=np.float64)
        self._surface_normal: Optional[np.ndarray] = None
        self._tangent_basis: Optional[np.ndarray] = None
        self._motion_axis: Optional[np.ndarray] = None
        self._hinge_axis: Optional[np.ndarray] = None
        self._hinge_pivot: Optional[np.ndarray] = None
        self._lever_arm: Optional[np.ndarray] = None
        self._lever_radius: Optional[float] = None
        self._latest_pos: Optional[np.ndarray] = None
        self._pos_vel_window: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=self.window_size
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def surface_normal(self) -> Optional[np.ndarray]:
        return None if self._surface_normal is None else self._surface_normal.copy()

    @property
    def motion_axis(self) -> Optional[np.ndarray]:
        return None if self._motion_axis is None else self._motion_axis.copy()

    @property
    def hinge_axis(self) -> Optional[np.ndarray]:
        return None if self._hinge_axis is None else self._hinge_axis.copy()

    @property
    def lever_arm(self) -> Optional[np.ndarray]:
        return None if self._lever_arm is None else self._lever_arm.copy()

    def _try_fit_revolute_hinge(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        u3: np.ndarray,
        s1: float,
        s2: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Check if recent (position_3d, velocity_3d) samples trace a rigid revolute arc."""
        if len(self._pos_vel_window) < self.min_contact_samples:
            return None
        if s1 <= 1e-6 or (s2 / s1) < 0.08:
            return None

        # On a planar surface wipe, normal reaction force aligns strongly with plane normal u3.
        # On a revolute hinge (e.g. door handle), bearing reaction forces act radially in-plane.
        f_mean_norm = float(np.linalg.norm(self._f_mean))
        if f_mean_norm > 1e-6 and abs(float(np.dot(self._f_mean, u3))) / f_mean_norm > 0.82:
            return None

        Q = np.column_stack([u1, u2])  # (3, 2) basis for motion plane
        P3 = np.array([item[0] for item in self._pos_vel_window], dtype=np.float64)
        V3 = np.array([item[1] for item in self._pos_vel_window], dtype=np.float64)

        # Out-of-plane position spread along u3 must be small relative to in-plane arc span
        z_coords = P3 @ u3
        P2 = P3 @ Q
        V2 = V3 @ Q
        in_plane_span = float(np.max(np.linalg.norm(P2 - np.mean(P2, axis=0), axis=1)))
        if in_plane_span < 1e-3 or float(np.std(z_coords)) > 0.25 * in_plane_span:
            return None

        v2_norms = np.linalg.norm(V2, axis=1)
        valid = v2_norms > 0.15 * s1
        if int(np.sum(valid)) < self.min_contact_samples:
            return None

        V2_val = V2[valid]
        P2_val = P2[valid]
        # Exact rigid-hinge linear tangency system: v_k^T p_c = v_k^T p_k
        rhs = np.sum(V2_val * P2_val, axis=1)
        gram = V2_val.T @ V2_val
        cond = float(np.linalg.cond(gram))
        if not np.isfinite(cond) or cond > 1e4:
            return None

        try:
            pc_2d = np.linalg.solve(gram, V2_val.T @ rhs)
        except np.linalg.LinAlgError:
            return None

        rel_2d = P2_val - pc_2d[None, :]
        radii = np.linalg.norm(rel_2d, axis=1)
        mean_r = float(np.mean(radii))
        if not (0.01 <= mean_r <= 10.0):
            return None
        rel_radius_std = float(np.std(radii)) / mean_r
        if rel_radius_std > 0.14:
            return None

        cos_tangency = np.abs(np.sum(rel_2d * V2_val, axis=1)) / np.maximum(
            1e-9, radii * v2_norms[valid]
        )
        if float(np.sqrt(np.mean(cos_tangency ** 2))) > 0.16:
            return None

        # Reconstruct 3D pivot & orient hinge axis via right-hand rule r x v
        pc_3d = Q @ pc_2d + float(np.mean(z_coords)) * u3
        r3_all = P3[valid] - pc_3d[None, :]
        r3_in_plane = r3_all - np.outer(r3_all @ u3, u3)
        ang_mom = np.mean(np.cross(r3_in_plane, V3[valid]), axis=0)
        axis = u3.copy()
        if float(np.dot(ang_mom, axis)) < 0.0:
            axis = -axis

        latest_p = P3[-1]
        lever = (latest_p - pc_3d) - float(np.dot(latest_p - pc_3d, axis)) * axis
        return axis, pc_3d, lever, mean_r

    def observe(
        self,
        velocity_3d: Sequence[float],
        force_3d: Sequence[float],
        position_3d: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Ingest one 3D contact sample and update manifold geometry classification."""
        v = _to_1d(velocity_3d, dim=3)
        f = _to_1d(force_3d, dim=3)
        p = None if position_3d is None else _to_1d(position_3d, dim=3)
        if p is not None:
            self._latest_pos = p.copy()

        f_norm = float(np.linalg.norm(f))
        if f_norm <= self.f_contact_min:
            self._non_contact_streak += 1
            if (
                self._non_contact_streak >= self.free_space_release_steps
                or self._contact_samples < self.min_contact_samples
            ):
                self._mode = "free_space"
            return self.summary()

        self._non_contact_streak = 0
        self._contact_samples += 1
        lam = self.ema_decay if self._contact_samples > 1 else 0.0
        one_m_lam = 1.0 - lam

        self._M_v = lam * self._M_v + one_m_lam * np.outer(v, v)
        f_unit = f / f_norm
        self._M_f = lam * self._M_f + one_m_lam * np.outer(f_unit, f_unit)
        self._f_mean = lam * self._f_mean + one_m_lam * f

        if p is not None:
            self._pos_vel_window.append((p.copy(), v.copy(), f.copy()))

        if self._contact_samples < self.min_contact_samples:
            self._mode = "free_space"
            return self.summary()

        # Symmetric SVD / eigendecomposition of velocity second-moment matrix M_v
        eigvals, eigvecs = np.linalg.eigh(0.5 * (self._M_v + self._M_v.T))
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(0.0, eigvals[order])
        U = eigvecs[:, order]  # columns u1, u2, u3
        s = np.sqrt(eigvals)
        self._singular_values = s

        s1, s2, s3 = float(s[0]), float(s[1]), float(s[2])
        u1, u2, u3 = U[:, 0], U[:, 1], U[:, 2]

        if s1 <= 1e-8:
            # Pure static contact: infer surface normal directly from mean force
            f_mean_norm = float(np.linalg.norm(self._f_mean))
            if f_mean_norm > 1e-8:
                n_vec = self._f_mean / f_mean_norm
                self._surface_normal = n_vec
                self._mode = "planar_surface"
            return self.summary()

        r21 = s2 / s1
        r31 = s3 / s1

        # 1. Check revolute hinge when 3D positions are provided and motion rotates in a plane
        if p is not None and r31 < self.rank_threshold:
            hinge_fit = self._try_fit_revolute_hinge(u1, u2, u3, s1, s2)
            if hinge_fit is not None:
                axis, pivot, lever, radius = hinge_fit
                self._mode = "revolute_hinge"
                self._hinge_axis = axis
                self._hinge_pivot = pivot
                self._lever_arm = lever
                self._lever_radius = radius
                lev_norm = float(np.linalg.norm(lever))
                if lev_norm > 1e-8:
                    t_inst = np.cross(axis, lever / lev_norm)
                    t_norm = float(np.linalg.norm(t_inst))
                    self._motion_axis = t_inst / t_norm if t_norm > 1e-8 else u1.copy()
                    self._surface_normal = lever / lev_norm
                else:
                    self._motion_axis = u1.copy()
                return self.summary()

        # 2. Rank-1 collinear velocity -> 1D prismatic constraint (drawer / slider)
        if r21 < self.rank_threshold:
            axis_1d = u1.copy()
            # Orient 1D axis consistently with latest velocity when non-zero
            if float(np.dot(v, axis_1d)) < 0.0 and float(np.linalg.norm(v)) > 1e-6:
                if self._motion_axis is not None and float(np.dot(self._motion_axis, axis_1d)) > 0.0:
                    pass
                elif self._motion_axis is not None:
                    axis_1d = -axis_1d
            elif self._motion_axis is not None and float(np.dot(self._motion_axis, axis_1d)) < 0.0:
                axis_1d = -axis_1d

            self._mode = "prismatic_1d"
            self._motion_axis = axis_1d
            self._hinge_axis = None
            self._hinge_pivot = None
            self._lever_arm = None
            self._lever_radius = None
            # Primary normal direction in the 2D constrained subspace from mean contact force
            f_orth = self._f_mean - float(np.dot(self._f_mean, axis_1d)) * axis_1d
            f_orth_norm = float(np.linalg.norm(f_orth))
            self._surface_normal = (f_orth / f_orth_norm) if f_orth_norm > 1e-8 else u3.copy()
            self._tangent_basis = axis_1d[:, None]
            return self.summary()

        # 3. Rank-2 planar velocity -> 2D planar surface (table wiping / wall sliding)
        if r31 < max(self.rank_threshold, 0.35):
            n_vec = u3.copy()
            if float(np.dot(self._f_mean, n_vec)) < 0.0:
                n_vec = -n_vec
            self._mode = "planar_surface"
            self._surface_normal = n_vec
            self._tangent_basis = np.column_stack([u1, u2])
            self._motion_axis = u1.copy()
            self._hinge_axis = None
            self._hinge_pivot = None
            self._lever_arm = None
            self._lever_radius = None
            return self.summary()

        # 3D unconstrained motion despite drag/force (e.g. viscous fluid) -> free_space
        self._mode = "free_space"
        return self.summary()

    update = observe

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "contact_samples": self._contact_samples,
            "contact_active": self._non_contact_streak == 0 and self._contact_samples > 0,
            "singular_values": self._singular_values.copy(),
            "normal": None if self._surface_normal is None else self._surface_normal.copy(),
            "motion_axis": None if self._motion_axis is None else self._motion_axis.copy(),
            "hinge_axis": None if self._hinge_axis is None else self._hinge_axis.copy(),
            "hinge_pivot": None if self._hinge_pivot is None else self._hinge_pivot.copy(),
            "lever_arm": None if self._lever_arm is None else self._lever_arm.copy(),
            "lever_radius": self._lever_radius,
        }

    def project_compliant_command(
        self,
        v_nom: Sequence[float],
        f_nom_normal: Union[float, Sequence[float], np.ndarray] = 0.0,
        max_normal_force: float = 20.0,
    ) -> Dict[str, Any]:
        """Project nominal 3D velocity onto the compliant manifold and regulate normal contact force."""
        v_in = _to_1d(v_nom, dim=3)
        f_max = max(0.0, float(max_normal_force))
        I3 = np.eye(3, dtype=np.float64)

        if self._mode == "planar_surface" and self._surface_normal is not None:
            n_hat = self._surface_normal / max(1e-12, float(np.linalg.norm(self._surface_normal)))
            P_force = np.outer(n_hat, n_hat)
            P_motion = I3 - P_force
            f_dir = n_hat
        elif self._mode == "prismatic_1d" and self._motion_axis is not None:
            t_hat = self._motion_axis / max(1e-12, float(np.linalg.norm(self._motion_axis)))
            P_motion = np.outer(t_hat, t_hat)
            P_force = I3 - P_motion
            if self._surface_normal is not None:
                n_proj = P_force @ self._surface_normal
                n_norm = float(np.linalg.norm(n_proj))
                f_dir = (n_proj / n_norm) if n_norm > 1e-8 else np.zeros(3, dtype=np.float64)
            else:
                f_dir = np.zeros(3, dtype=np.float64)
        elif (
            self._mode == "revolute_hinge"
            and self._hinge_axis is not None
            and self._lever_arm is not None
            and float(np.linalg.norm(self._lever_arm)) > 1e-8
        ):
            omega_hat = self._hinge_axis / float(np.linalg.norm(self._hinge_axis))
            r_hat = self._lever_arm / float(np.linalg.norm(self._lever_arm))
            t_vec = np.cross(omega_hat, r_hat)
            t_norm = float(np.linalg.norm(t_vec))
            t_hat = (t_vec / t_norm) if t_norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
            P_motion = np.outer(t_hat, t_hat)
            P_force = I3 - P_motion
            f_dir = P_force @ r_hat
            fn = float(np.linalg.norm(f_dir))
            f_dir = (f_dir / fn) if fn > 1e-8 else np.zeros(3, dtype=np.float64)
        else:
            P_motion = I3
            P_force = np.zeros((3, 3), dtype=np.float64)
            f_dir = np.zeros(3, dtype=np.float64)

        v_compliant = P_motion @ v_in

        f_raw = np.asarray(f_nom_normal, dtype=np.float64)
        if f_raw.ndim == 0:
            f_scalar = float(np.clip(float(f_raw), -f_max, f_max))
            f_compliant = f_scalar * f_dir
        else:
            f_vec = _to_1d(f_raw, dim=3)
            f_proj = P_force @ f_vec
            f_proj_norm = float(np.linalg.norm(f_proj))
            if f_proj_norm > f_max and f_proj_norm > 1e-12:
                f_compliant = f_proj * (f_max / f_proj_norm)
            else:
                f_compliant = f_proj
            f_scalar = float(np.linalg.norm(f_compliant))

        orth_err = abs(float(np.dot(v_compliant, f_compliant)))
        return {
            "mode": self._mode,
            "v_compliant": v_compliant,
            "f_normal_cmd": f_compliant,
            "normal_force_scalar": float(np.linalg.norm(f_compliant)),
            "orthogonality_error": orth_err,
            "P_motion": P_motion,
            "P_force": P_force,
            "normal": self.surface_normal,
            "motion_axis": self.motion_axis,
            "hinge_axis": self.hinge_axis,
            "lever_arm": self.lever_arm,
        }
