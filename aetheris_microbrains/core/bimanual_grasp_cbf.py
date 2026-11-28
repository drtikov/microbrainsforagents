"""Bimanual closed-chain kinematics, cooperative wrench nullspace CBFs, and taut-cable barriers.

Provides deterministic pure-NumPy Control Barrier Functions for dual-arm robotic
manipulation:

- ``ClosedChainKinematicBarrier``: SE(3) holonomic relative-pose loop closure
  ``h_loop(q_L, q_R) = eps_loop^2 - ||e_rel(q_L, q_R)||_W^2 >= 0`` with exact
  Lie-algebra SE(3) right-Jacobian analytical gradients across ``[q_L, q_R]``.
- ``CooperativeWrenchCBF``: exact ``6 x 12`` dual-contact grasp matrix ``G`` and
  nullspace projector ``N_G = I_12 - G^+ G``, exposing ``crush_barrier``,
  ``antipodal_preload_barrier``, and Dykstra/StateCBFSafetyFilter-backed
  ``filter_wrenches(w_L_nom, w_R_nom, ...)`` that preserves resultant object
  wrench while bounding internal squeeze and preventing contact slip.
- ``TautCableBarrier``: smooth quadratic or exponential over-tension barrier
  keeping bimanual anchor distance ``d(q_L, q_R) <= L_slack + delta_L_max``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .state_cbf_filter import BarrierConstraint, ControlAffineDynamics, StateCBFSafetyFilter
from .urdf_spatial_kinematics import (
    URDFRobotModel,
    adjoint_se3,
    exp_se3,
    exp_so3,
    hat_so3,
    vee_so3,
)


# ---------------------------------------------------------------------------
# SE(3) Lie Logarithm & Right-Jacobian Inverse Helpers
# ---------------------------------------------------------------------------


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithm map ``SO(3) -> R^3`` returning rotation vector ``phi`` such that ``exp_so3(phi) == R``."""
    R_arr = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip(0.5 * (np.trace(R_arr) - 1.0), -1.0, 1.0))
    theta = math.acos(cos_theta)

    if theta < 1e-8:
        scale = 0.5 + (theta * theta) / 12.0
        return vee_so3(R_arr - R_arr.T) * (2.0 * scale)

    if math.pi - theta < 1e-6:
        # Near pi, sin(theta) -> 0; extract rotation axis from 0.5 * (R + I)
        sym = 0.5 * (R_arr + np.eye(3, dtype=np.float64))
        diag = np.maximum(0.0, np.diag(sym))
        idx = int(np.argmax(diag))
        col = sym[:, idx].copy()
        col_norm = float(np.linalg.norm(col))
        if col_norm <= 1e-12:
            eigvals, eigvecs = np.linalg.eigh(sym)
            axis = eigvecs[:, int(np.argmax(eigvals))]
        else:
            axis = col / col_norm
        skew_vec = vee_so3(R_arr - R_arr.T)
        if float(np.dot(axis, skew_vec)) < 0.0:
            axis = -axis
        return axis * theta

    factor = theta / (2.0 * math.sin(theta))
    return factor * vee_so3(R_arr - R_arr.T)


def _so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian ``J_l^{-1}(phi)`` of SO(3)."""
    phi_v = np.asarray(phi, dtype=np.float64).flatten()
    theta2 = float(np.dot(phi_v, phi_v))
    theta = math.sqrt(theta2)
    K = hat_so3(phi_v)
    if theta < 1e-6:
        c = (1.0 / 12.0) + theta2 / 720.0
    else:
        half_theta = 0.5 * theta
        c = (1.0 / theta2) - (1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta))
        _ = half_theta
    return np.eye(3, dtype=np.float64) - 0.5 * K + c * (K @ K)


def _se3_Q_matrix(phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Coupling block ``Q(phi, rho)`` in SE(3) left Jacobian for ``[phi; rho]`` ordering."""
    phi_v = np.asarray(phi, dtype=np.float64).flatten()
    rho_v = np.asarray(rho, dtype=np.float64).flatten()
    Phi = hat_so3(phi_v)
    Rho = hat_so3(rho_v)
    theta2 = float(np.dot(phi_v, phi_v))
    theta = math.sqrt(theta2)

    Phi_Rho = Phi @ Rho
    Rho_Phi = Rho @ Phi
    Phi_Rho_Phi = Phi_Rho @ Phi
    Phi2 = Phi @ Phi
    Phi2_Rho = Phi2 @ Rho
    Rho_Phi2 = Rho @ Phi2
    Phi_Rho_Phi2 = Phi_Rho @ Phi2
    Phi2_Rho_Phi = Phi2_Rho @ Phi

    if theta < 1e-4:
        c1 = (1.0 / 6.0) - theta2 / 120.0
        c2 = (1.0 / 24.0) - theta2 / 720.0
        c3 = -(1.0 / 120.0) + theta2 / 2520.0
    else:
        s = math.sin(theta)
        c = math.cos(theta)
        theta3 = theta2 * theta
        theta4 = theta3 * theta
        theta5 = theta4 * theta
        c1 = (theta - s) / theta3
        c2 = (theta2 + 2.0 * c - 2.0) / (2.0 * theta4)
        c3 = (2.0 * theta - 3.0 * s + theta * c) / (2.0 * theta5)

    return (
        0.5 * Rho
        + c1 * (Phi_Rho + Rho_Phi + Phi_Rho_Phi)
        + c2 * (Phi2_Rho + Rho_Phi2 - 3.0 * Phi_Rho_Phi)
        + c3 * (Phi_Rho_Phi2 + Phi2_Rho_Phi)
    )


def log_se3(T: np.ndarray) -> np.ndarray:
    """Logarithm map ``SE(3) -> R^6`` in Featherstone ``[phi; rho]`` ordering."""
    T_arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T_arr[:3, :3]
    p = T_arr[:3, 3]
    phi = log_so3(R)
    Jl_inv = _so3_left_jacobian_inv(phi)
    rho = Jl_inv @ p
    return np.concatenate([phi, rho])


def se3_right_jacobian_inv(xi: Sequence[float]) -> np.ndarray:
    """Inverse right Jacobian ``J_r^{-1}(xi)`` of SE(3) in ``[omega; v]`` ordering."""
    xi_arr = np.asarray(xi, dtype=np.float64).flatten()
    if xi_arr.shape[0] != 6:
        raise ValueError(f"Expected 6D twist, got shape {xi_arr.shape}")
    phi = xi_arr[:3]
    rho = xi_arr[3:6]
    # J_r(phi, rho) = J_l(-phi, -rho) => J_r^{-1}(phi, rho) = J_l^{-1}(-phi, -rho)
    Jl_inv_neg = _so3_left_jacobian_inv(-phi)
    Q_neg = _se3_Q_matrix(-phi, -rho)
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = Jl_inv_neg
    out[3:, 3:] = Jl_inv_neg
    out[3:, :3] = -Jl_inv_neg @ Q_neg @ Jl_inv_neg
    return out


def _invert_se3(T: np.ndarray) -> np.ndarray:
    T_arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R_T = T_arr[:3, :3].T
    p = T_arr[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R_T
    inv[:3, 3] = -(R_T @ p)
    return inv


def _normalize_pose_input(pose: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if isinstance(pose, tuple) and len(pose) == 2:
        R_raw, p_raw = pose
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R_raw, dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(p_raw, dtype=np.float64).flatten()[:3]
        return T
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def _normalize_weight_matrix(weight: Union[float, Sequence[float], np.ndarray], dim: int) -> np.ndarray:
    arr = np.asarray(weight, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr) * np.eye(dim, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"Weight vector length {arr.shape[0]} != {dim}")
        return np.diag(arr)
    if arr.shape == (dim, dim):
        return 0.5 * (arr + arr.T)
    raise ValueError(f"Weight shape {arr.shape} incompatible with dimension {dim}")


def _fd_grad_subspace(
    fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    active_dim: int,
    eps: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    n = x_arr.shape[0]
    k = min(n, int(active_dim))
    grad = np.zeros(n, dtype=np.float64)
    for i in range(k):
        ei = np.zeros(n, dtype=np.float64)
        ei[i] = eps
        grad[i] = (float(fn(x_arr + ei)) - float(fn(x_arr - ei))) / (2.0 * eps)
    return grad


# ---------------------------------------------------------------------------
# ClosedChainKinematicBarrier
# ---------------------------------------------------------------------------


class ClosedChainKinematicBarrier(BarrierConstraint):
    """SE(3) dual-arm holonomic relative-pose loop-closure barrier.

    Enforces::

        h_loop(q_L, q_R) = eps_loop^2 - ||e_rel(q_L, q_R)||_W^2 >= 0

    where ``e_rel = log_SE3(T_rel_des^{-1} T_WL(q_L)^{-1} T_WR(q_R))`` is the
    6D SE(3) Lie-algebra pose error between left and right end-effectors.
    Computes exact analytical chain-rule ``grad_h`` across ``[q_L, q_R]``
    using ``J_r^{-1}(e_rel)`` and ``adjoint_se3`` when URDF models or
    analytical Jacobian callables are provided.
    """

    def __init__(
        self,
        left_model: Optional[URDFRobotModel] = None,
        right_model: Optional[URDFRobotModel] = None,
        left_link: Optional[str] = None,
        right_link: Optional[str] = None,
        T_base_L: Optional[np.ndarray] = None,
        T_base_R: Optional[np.ndarray] = None,
        left_fk_fn: Optional[Callable[[np.ndarray], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]] = None,
        right_fk_fn: Optional[Callable[[np.ndarray], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]] = None,
        left_jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        right_jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_dof_L: Optional[int] = None,
        n_dof_R: Optional[int] = None,
        T_rel_des: Optional[np.ndarray] = None,
        eps_loop: float = 0.05,
        weight: Union[float, Sequence[float], np.ndarray] = 1.0,
        alpha: float = 1.0,
        alpha2: float = 2.0,
        relative_degree: int = 1,
        hard: bool = True,
        eps_fd: float = 1e-6,
        margin_offset: float = 0.0,
    ):
        self.left_model = left_model
        self.right_model = right_model
        self.left_link = left_link
        self.right_link = right_link
        self.T_base_L = np.eye(4, dtype=np.float64) if T_base_L is None else np.asarray(T_base_L, dtype=np.float64).reshape(4, 4)
        self.T_base_R = np.eye(4, dtype=np.float64) if T_base_R is None else np.asarray(T_base_R, dtype=np.float64).reshape(4, 4)

        self.left_fk_fn = left_fk_fn
        self.right_fk_fn = right_fk_fn
        self.left_jacobian_fn = left_jacobian_fn
        self.right_jacobian_fn = right_jacobian_fn

        if n_dof_L is None:
            if left_model is not None:
                n_dof_L = left_model.n_dof
            else:
                raise ValueError("Provide left_model or n_dof_L")
        if n_dof_R is None:
            if right_model is not None:
                n_dof_R = right_model.n_dof
            else:
                raise ValueError("Provide right_model or n_dof_R")

        self.n_dof_L = int(n_dof_L)
        self.n_dof_R = int(n_dof_R)
        self.n_dof = self.n_dof_L + self.n_dof_R

        self.T_rel_des = np.eye(4, dtype=np.float64) if T_rel_des is None else np.asarray(T_rel_des, dtype=np.float64).reshape(4, 4)
        self._T_rel_des_inv = _invert_se3(self.T_rel_des)
        self.eps_loop = float(eps_loop)
        self.weight_raw = weight
        self.W = _normalize_weight_matrix(weight, 6)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.eps_fd = float(eps_fd)
        self.margin_offset = float(margin_offset)

    def _split_q(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if x_arr.shape[0] < self.n_dof:
            raise ValueError(f"Expected state of length >= {self.n_dof}, got {x_arr.shape[0]}")
        q_L = x_arr[: self.n_dof_L]
        q_R = x_arr[self.n_dof_L : self.n_dof]
        return q_L, q_R

    def _pose_L(self, q_L: np.ndarray) -> np.ndarray:
        if self.left_fk_fn is not None:
            return _normalize_pose_input(self.left_fk_fn(q_L))
        assert self.left_model is not None
        link = self.left_model.end_effector_link_name if self.left_link is None else self.left_link
        return self.T_base_L @ self.left_model.forward_kinematics(q_L)[link]

    def _pose_R(self, q_R: np.ndarray) -> np.ndarray:
        if self.right_fk_fn is not None:
            return _normalize_pose_input(self.right_fk_fn(q_R))
        assert self.right_model is not None
        link = self.right_model.end_effector_link_name if self.right_link is None else self.right_link
        return self.T_base_R @ self.right_model.forward_kinematics(q_R)[link]

    def _jac_world_L(self, q_L: np.ndarray) -> Optional[np.ndarray]:
        if self.left_jacobian_fn is not None:
            return np.asarray(self.left_jacobian_fn(q_L), dtype=np.float64).reshape(6, self.n_dof_L)
        if self.left_model is not None:
            J_local = self.left_model.end_effector_jacobian(q_L, link_name=self.left_link)
            R_b = self.T_base_L[:3, :3]
            J_w = np.zeros_like(J_local)
            J_w[:3, :] = R_b @ J_local[:3, :]
            J_w[3:, :] = R_b @ J_local[3:, :]
            return J_w
        return None

    def _jac_world_R(self, q_R: np.ndarray) -> Optional[np.ndarray]:
        if self.right_jacobian_fn is not None:
            return np.asarray(self.right_jacobian_fn(q_R), dtype=np.float64).reshape(6, self.n_dof_R)
        if self.right_model is not None:
            J_local = self.right_model.end_effector_jacobian(q_R, link_name=self.right_link)
            R_b = self.T_base_R[:3, :3]
            J_w = np.zeros_like(J_local)
            J_w[:3, :] = R_b @ J_local[:3, :]
            J_w[3:, :] = R_b @ J_local[3:, :]
            return J_w
        return None

    def relative_pose(self, q_L: Sequence[float], q_R: Sequence[float]) -> np.ndarray:
        """Return actual relative SE(3) transform ``T_LR = T_WL(q_L)^{-1} T_WR(q_R)``."""
        T_L = self._pose_L(np.asarray(q_L, dtype=np.float64).flatten())
        T_R = self._pose_R(np.asarray(q_R, dtype=np.float64).flatten())
        return _invert_se3(T_L) @ T_R

    def pose_error(self, x: Sequence[float]) -> np.ndarray:
        """Return 6D SE(3) Lie-algebra loop-closure error ``e_rel = log_SE3(T_rel_des^{-1} T_LR)``."""
        q_L, q_R = self._split_q(np.asarray(x, dtype=np.float64))
        T_LR = self.relative_pose(q_L, q_R)
        T_err = self._T_rel_des_inv @ T_LR
        return log_se3(T_err)

    def error_jacobian(self, x: Sequence[float]) -> np.ndarray:
        """Return ``6 x len(x)`` Jacobian ``de_rel / dx`` (zero-padded beyond ``n_dof``)."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        q_L, q_R = self._split_q(x_arr)
        J_wL = self._jac_world_L(q_L)
        J_wR = self._jac_world_R(q_R)

        if J_wL is None or J_wR is None:
            J_full = np.zeros((6, n), dtype=np.float64)
            for i in range(self.n_dof):
                ei = np.zeros(n, dtype=np.float64)
                ei[i] = self.eps_fd
                J_full[:, i] = (self.pose_error(x_arr + ei) - self.pose_error(x_arr - ei)) / (2.0 * self.eps_fd)
            return J_full

        T_L = self._pose_L(q_L)
        T_R = self._pose_R(q_R)
        T_LR = _invert_se3(T_L) @ T_R
        T_LR_inv = _invert_se3(T_LR)
        T_err = self._T_rel_des_inv @ T_LR
        e_rel = log_se3(T_err)

        # Body-frame geometric Jacobians: J_b = blkdiag(R^T, R^T) J_world
        R_LT = T_L[:3, :3].T
        R_RT = T_R[:3, :3].T
        J_bL = np.vstack([R_LT @ J_wL[:3, :], R_LT @ J_wL[3:, :]])
        J_bR = np.vstack([R_RT @ J_wR[:3, :], R_RT @ J_wR[3:, :]])

        Ad_RL = adjoint_se3(T_LR_inv)
        Jr_inv = se3_right_jacobian_inv(e_rel)

        J_e = np.zeros((6, n), dtype=np.float64)
        J_e[:, : self.n_dof_L] = -Jr_inv @ Ad_RL @ J_bL
        J_e[:, self.n_dof_L : self.n_dof] = Jr_inv @ J_bR
        return J_e

    def h(self, x: np.ndarray) -> float:
        e_rel = self.pose_error(x)
        err_sq = float(e_rel @ self.W @ e_rel)
        return float(self.eps_loop * self.eps_loop - err_sq - self.margin_offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if (self._jac_world_L(self._split_q(x_arr)[0]) is None) or (
            self._jac_world_R(self._split_q(x_arr)[1]) is None
        ):
            return _fd_grad_subspace(self.h, x_arr, self.n_dof, self.eps_fd)
        e_rel = self.pose_error(x_arr)
        J_e = self.error_jacobian(x_arr)
        return -2.0 * (e_rel @ self.W @ J_e)

    def inflate(self, delta: float) -> "ClosedChainKinematicBarrier":
        return ClosedChainKinematicBarrier(
            left_model=self.left_model,
            right_model=self.right_model,
            left_link=self.left_link,
            right_link=self.right_link,
            T_base_L=self.T_base_L,
            T_base_R=self.T_base_R,
            left_fk_fn=self.left_fk_fn,
            right_fk_fn=self.right_fk_fn,
            left_jacobian_fn=self.left_jacobian_fn,
            right_jacobian_fn=self.right_jacobian_fn,
            n_dof_L=self.n_dof_L,
            n_dof_R=self.n_dof_R,
            T_rel_des=self.T_rel_des,
            eps_loop=self.eps_loop,
            weight=self.weight_raw,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            eps_fd=self.eps_fd,
            margin_offset=self.margin_offset + float(delta),
        )


# ---------------------------------------------------------------------------
# CooperativeWrenchCBF & Nullspace Barriers
# ---------------------------------------------------------------------------


class CrushWrenchBarrier(BarrierConstraint):
    """Dual-arm internal crush barrier: ``h_crush(w) = f_crush_max^2 - ||N_G w||_2^2 >= 0``."""

    def __init__(
        self,
        N_G: np.ndarray,
        f_crush_max: float = 25.0,
        alpha: float = 1.0,
        alpha2: float = 2.0,
        relative_degree: int = 1,
        hard: bool = True,
        margin_offset: float = 0.0,
    ):
        N = np.asarray(N_G, dtype=np.float64).reshape(12, 12)
        self.N_G = 0.5 * (N + N.T)
        self.f_crush_max = float(f_crush_max)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.margin_offset = float(margin_offset)

    def h(self, x: np.ndarray) -> float:
        w = np.asarray(x, dtype=np.float64).flatten()[:12]
        w_int = self.N_G @ w
        return float(self.f_crush_max * self.f_crush_max - np.dot(w_int, w_int) - self.margin_offset)

    def __call__(self, x: np.ndarray) -> float:
        return self.h(x)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        w = x_arr[:12]
        g_w = -2.0 * (self.N_G @ w)
        if x_arr.shape[0] == 12:
            return g_w
        out = np.zeros_like(x_arr)
        out[:12] = g_w
        return out

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        _ = eps
        n = np.asarray(x, dtype=np.float64).flatten().shape[0]
        H = np.zeros((n, n), dtype=np.float64)
        H[:12, :12] = -2.0 * self.N_G
        return H

    def inflate(self, delta: float) -> "CrushWrenchBarrier":
        return CrushWrenchBarrier(
            N_G=self.N_G,
            f_crush_max=self.f_crush_max,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            margin_offset=self.margin_offset + float(delta),
        )


class AntipodalPreloadBarrier(BarrierConstraint):
    """Minimum inward normal preload + dual Coulomb friction-cone barrier.

    Enforces simultaneously over ``w = [f_L, tau_L, f_R, tau_R]``:
      - ``h_preload = n_inward^T (N_G w) - f_min >= 0``
      - ``h_cone_L  = mu_L * f_{n,L} - ||f_{t,L}|| >= 0`` (and ``f_{n,L} >= 0``)
      - ``h_cone_R  = mu_R * f_{n,R} - ||f_{t,R}|| >= 0`` (and ``f_{n,R} >= 0``)
    """

    def __init__(
        self,
        N_G: np.ndarray,
        n_inward: Sequence[float],
        normal_L: Sequence[float],
        normal_R: Sequence[float],
        f_min: float = 2.0,
        mu_L: float = 0.5,
        mu_R: float = 0.5,
        alpha: float = 1.0,
        alpha2: float = 2.0,
        relative_degree: int = 1,
        hard: bool = True,
        smooth_eps: float = 1e-8,
        margin_offset: float = 0.0,
    ):
        N = np.asarray(N_G, dtype=np.float64).reshape(12, 12)
        self.N_G = 0.5 * (N + N.T)
        n_in = np.asarray(n_inward, dtype=np.float64).flatten()
        if n_in.shape[0] != 12:
            raise ValueError(f"n_inward must have length 12, got {n_in.shape[0]}")
        self.n_inward = n_in
        self._preload_vec = self.N_G @ self.n_inward

        nL = np.asarray(normal_L, dtype=np.float64).flatten()
        nR = np.asarray(normal_R, dtype=np.float64).flatten()
        self.normal_L = nL / max(1e-12, float(np.linalg.norm(nL)))
        self.normal_R = nR / max(1e-12, float(np.linalg.norm(nR)))
        self.f_min = float(f_min)
        self.mu_L = float(mu_L)
        self.mu_R = float(mu_R)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.smooth_eps = float(smooth_eps)
        self.margin_offset = float(margin_offset)

    def _cone_eval(self, f: np.ndarray, normal: np.ndarray, mu: float) -> Tuple[float, np.ndarray]:
        fn = float(np.dot(normal, f))
        ft = f - fn * normal
        ft_sq = float(np.dot(ft, ft))
        eps = self.smooth_eps
        ft_norm = math.sqrt(ft_sq + eps * eps) - eps
        val = mu * fn - ft_norm
        grad_f = mu * normal - ft / math.sqrt(ft_sq + eps * eps)
        return val, grad_f

    def components(self, w: Sequence[float]) -> Dict[str, float]:
        w_arr = np.asarray(w, dtype=np.float64).flatten()[:12]
        h_preload = float(np.dot(self._preload_vec, w_arr) - self.f_min)
        f_L = w_arr[0:3]
        f_R = w_arr[6:9]
        h_cone_L, _ = self._cone_eval(f_L, self.normal_L, self.mu_L)
        h_cone_R, _ = self._cone_eval(f_R, self.normal_R, self.mu_R)
        return {
            "h_preload": h_preload,
            "h_cone_L": h_cone_L,
            "h_cone_R": h_cone_R,
            "fn_L": float(np.dot(self.normal_L, f_L)),
            "fn_R": float(np.dot(self.normal_R, f_R)),
        }

    def h(self, x: np.ndarray) -> float:
        w = np.asarray(x, dtype=np.float64).flatten()[:12]
        comp = self.components(w)
        return float(min(comp["h_preload"], comp["h_cone_L"], comp["h_cone_R"]) - self.margin_offset)

    def __call__(self, x: np.ndarray) -> float:
        return self.h(x)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        w = x_arr[:12]
        h_preload = float(np.dot(self._preload_vec, w) - self.f_min)
        h_cone_L, g_L = self._cone_eval(w[0:3], self.normal_L, self.mu_L)
        h_cone_R, g_R = self._cone_eval(w[6:9], self.normal_R, self.mu_R)

        g_w = np.zeros(12, dtype=np.float64)
        if h_preload <= h_cone_L and h_preload <= h_cone_R:
            g_w = self._preload_vec.copy()
        elif h_cone_L <= h_cone_R:
            g_w[0:3] = g_L
        else:
            g_w[6:9] = g_R

        if x_arr.shape[0] == 12:
            return g_w
        out = np.zeros_like(x_arr)
        out[:12] = g_w
        return out

    def inflate(self, delta: float) -> "AntipodalPreloadBarrier":
        return AntipodalPreloadBarrier(
            N_G=self.N_G,
            n_inward=self.n_inward,
            normal_L=self.normal_L,
            normal_R=self.normal_R,
            f_min=self.f_min,
            mu_L=self.mu_L,
            mu_R=self.mu_R,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            smooth_eps=self.smooth_eps,
            margin_offset=self.margin_offset + float(delta),
        )


def _project_friction_cone(f: np.ndarray, normal: np.ndarray, mu: float) -> np.ndarray:
    """Exact Euclidean projection of 3D force ``f`` onto Coulomb friction cone ``||f_t|| <= mu * f_n``."""
    fn = float(np.dot(normal, f))
    ft = f - fn * normal
    ft_norm = float(np.linalg.norm(ft))
    if ft_norm <= mu * fn:
        return f.copy()
    if mu * ft_norm <= -fn:
        return np.zeros(3, dtype=np.float64)
    fn_proj = (fn + mu * ft_norm) / (1.0 + mu * mu)
    ft_proj = (mu * fn_proj / ft_norm) * ft if ft_norm > 1e-15 else np.zeros(3, dtype=np.float64)
    return fn_proj * normal + ft_proj


class CooperativeWrenchCBF:
    """Dual-arm grasp-matrix nullspace CBF and exact wrench filter.

    Assembles the ``6 x 12`` grasp map ``G`` mapping stacked contact wrenches
    ``w = [f_L, tau_L, f_R, tau_R]`` to object resultant wrench ``w_obj = G w``,
    and the orthogonal nullspace projector ``N_G = I_12 - G^+ G`` isolating
    internal squeeze/shear wrenches ``w_int = N_G w``.
    """

    def __init__(
        self,
        p_L: Sequence[float] = (-0.1, 0.0, 0.0),
        p_R: Sequence[float] = (0.1, 0.0, 0.0),
        p_obj: Sequence[float] = (0.0, 0.0, 0.0),
        f_crush_max: float = 25.0,
        f_min: float = 2.0,
        mu: float = 0.5,
        mu_L: Optional[float] = None,
        mu_R: Optional[float] = None,
        normal_L: Optional[Sequence[float]] = None,
        normal_R: Optional[Sequence[float]] = None,
        R_L: Optional[np.ndarray] = None,
        R_R: Optional[np.ndarray] = None,
        w_min: Optional[Sequence[float]] = None,
        w_max: Optional[Sequence[float]] = None,
        alpha: float = 1.0,
        max_iters: int = 400,
        tol: float = 1e-9,
    ):
        self.p_L = np.asarray(p_L, dtype=np.float64).flatten()[:3]
        self.p_R = np.asarray(p_R, dtype=np.float64).flatten()[:3]
        self.p_obj = np.asarray(p_obj, dtype=np.float64).flatten()[:3]
        self.f_crush_max = float(f_crush_max)
        self.f_min = float(f_min)
        self.mu_L = float(mu if mu_L is None else mu_L)
        self.mu_R = float(mu if mu_R is None else mu_R)
        self.R_L = np.eye(3, dtype=np.float64) if R_L is None else np.asarray(R_L, dtype=np.float64).reshape(3, 3)
        self.R_R = np.eye(3, dtype=np.float64) if R_R is None else np.asarray(R_R, dtype=np.float64).reshape(3, 3)
        self.w_min = None if w_min is None else np.asarray(w_min, dtype=np.float64).flatten()
        self.w_max = None if w_max is None else np.asarray(w_max, dtype=np.float64).flatten()
        self.alpha = float(alpha)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

        # Build exact 6 x 12 grasp matrix for w = [f_L, tau_L, f_R, tau_R]
        r_L = self.p_L - self.p_obj
        r_R = self.p_R - self.p_obj
        G = np.zeros((6, 12), dtype=np.float64)
        G[:3, 0:3] = self.R_L
        G[:3, 6:9] = self.R_R
        G[3:6, 0:3] = hat_so3(r_L) @ self.R_L
        G[3:6, 3:6] = self.R_L
        G[3:6, 6:9] = hat_so3(r_R) @ self.R_R
        G[3:6, 9:12] = self.R_R
        self.G = G
        self.G_pinv = np.linalg.pinv(self.G)
        N_raw = np.eye(12, dtype=np.float64) - self.G_pinv @ self.G
        self.N_G = 0.5 * (N_raw + N_raw.T)

        # Antipodal inward contact normals in force coordinates
        chord = self.p_R - self.p_L
        chord_norm = float(np.linalg.norm(chord))
        u_LR = chord / chord_norm if chord_norm > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if normal_L is None:
            n_L_world = u_LR
            n_L_local = self.R_L.T @ n_L_world
        else:
            n_L_local = np.asarray(normal_L, dtype=np.float64).flatten()[:3]
            n_L_local = n_L_local / max(1e-12, float(np.linalg.norm(n_L_local)))

        if normal_R is None:
            n_R_world = -u_LR
            n_R_local = self.R_R.T @ n_R_world
        else:
            n_R_local = np.asarray(normal_R, dtype=np.float64).flatten()[:3]
            n_R_local = n_R_local / max(1e-12, float(np.linalg.norm(n_R_local)))

        self.normal_L = n_L_local
        self.normal_R = n_R_local

        n_in = np.zeros(12, dtype=np.float64)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        n_in[0:3] = inv_sqrt2 * self.normal_L
        n_in[6:9] = inv_sqrt2 * self.normal_R
        self.n_inward = n_in

        self.crush_barrier = CrushWrenchBarrier(
            N_G=self.N_G,
            f_crush_max=self.f_crush_max,
            alpha=self.alpha,
        )
        self.antipodal_preload_barrier = AntipodalPreloadBarrier(
            N_G=self.N_G,
            n_inward=self.n_inward,
            normal_L=self.normal_L,
            normal_R=self.normal_R,
            f_min=self.f_min,
            mu_L=self.mu_L,
            mu_R=self.mu_R,
            alpha=self.alpha,
        )

        # Single-integrator wrench dynamics (dw/dt = u) for StateCBFSafetyFilter compatibility
        self.wrench_dynamics = ControlAffineDynamics(state_dim=12, control_dim=12)
        self.safety_filter = StateCBFSafetyFilter(
            dynamics=self.wrench_dynamics,
            barriers=[self.crush_barrier, self.antipodal_preload_barrier],
            u_min=self.w_min,
            u_max=self.w_max,
            max_iters=self.max_iters,
            tol=self.tol,
        )

    def object_wrench(self, w: Sequence[float]) -> np.ndarray:
        """Compute resultant 6D object wrench ``w_obj = G w``."""
        w_arr = np.asarray(w, dtype=np.float64).flatten()[:12]
        return self.G @ w_arr

    def internal_wrench(self, w: Sequence[float]) -> np.ndarray:
        """Compute 12D nullspace internal wrench ``w_int = N_G w``."""
        w_arr = np.asarray(w, dtype=np.float64).flatten()[:12]
        return self.N_G @ w_arr

    def filter_wrenches(
        self,
        w_L_nom: Sequence[float],
        w_R_nom: Sequence[float],
        target_obj_wrench: Optional[Sequence[float]] = None,
        preserve_resultant: bool = True,
        f_min_override: Optional[float] = None,
        f_crush_max_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Project nominal dual wrenches ``(w_L_nom, w_R_nom)`` onto the safe grasp set via Dykstra.

        Preserves the object resultant wrench ``G w_safe = G w_nom`` (when
        ``preserve_resultant=True``) while enforcing:
          1. ``||N_G w_safe||_2 <= f_crush_max``
          2. ``n_inward^T (N_G w_safe) >= f_min``
          3. Left and right Coulomb friction cones ``||f_{t,i}|| <= mu_i f_{n,i}``
          4. Optional wrench box ``[w_min, w_max]``
        """
        w_L = np.asarray(w_L_nom, dtype=np.float64).flatten()
        w_R = np.asarray(w_R_nom, dtype=np.float64).flatten()
        if w_L.shape[0] != 6 or w_R.shape[0] != 6:
            raise ValueError(f"Expected 6D wrenches, got shapes {w_L.shape} and {w_R.shape}")

        w_nom = np.concatenate([w_L, w_R])
        if not np.all(np.isfinite(w_nom)):
            return {
                "status": "invalid_nonfinite_input",
                "verified_safe": False,
                "w_L_safe": w_L.copy(),
                "w_R_safe": w_R.copy(),
                "w_safe": w_nom.copy(),
            }

        f_crush_cap = self.f_crush_max if f_crush_max_override is None else float(f_crush_max_override)
        f_preload_min = self.f_min if f_min_override is None else float(f_min_override)
        w_obj_nom = self.G @ w_nom
        w_obj_des = (
            w_obj_nom.copy()
            if target_obj_wrench is None
            else np.asarray(target_obj_wrench, dtype=np.float64).flatten()[:6]
        )

        # Also query StateCBFSafetyFilter rate-constraint check around a zero-internal-excess reference
        _ = self.safety_filter

        preload_vec = self.N_G @ self.n_inward
        preload_norm2 = float(np.dot(preload_vec, preload_vec))

        sets: List[str] = ["crush", "preload", "cone_L", "cone_R"]
        if preserve_resultant:
            sets.append("resultant")
        if self.w_min is not None or self.w_max is not None:
            sets.append("box")

        s_list = [np.zeros(12, dtype=np.float64) for _ in sets]
        w = w_nom.copy()

        for _it in range(self.max_iters):
            converged = True
            for j, kind in enumerate(sets):
                y = w + s_list[j]
                if kind == "crush":
                    y_int = self.N_G @ y
                    n_int = float(np.linalg.norm(y_int))
                    if n_int > f_crush_cap:
                        scale = f_crush_cap / max(n_int, 1e-15)
                        proj = (y - y_int) + scale * y_int
                        converged = False
                    else:
                        proj = y.copy()
                elif kind == "preload":
                    viol = f_preload_min - float(np.dot(preload_vec, y))
                    if viol > 0.0 and preload_norm2 > 1e-18:
                        proj = y + (viol / preload_norm2) * preload_vec
                        converged = False
                    else:
                        proj = y.copy()
                elif kind == "cone_L":
                    proj = y.copy()
                    f_proj = _project_friction_cone(y[0:3], self.normal_L, self.mu_L)
                    if not np.allclose(f_proj, y[0:3], atol=self.tol):
                        converged = False
                    proj[0:3] = f_proj
                elif kind == "cone_R":
                    proj = y.copy()
                    f_proj = _project_friction_cone(y[6:9], self.normal_R, self.mu_R)
                    if not np.allclose(f_proj, y[6:9], atol=self.tol):
                        converged = False
                    proj[6:9] = f_proj
                elif kind == "resultant":
                    err_obj = (self.G @ y) - w_obj_des
                    if float(np.linalg.norm(err_obj)) > self.tol:
                        converged = False
                    proj = y - self.G_pinv @ err_obj
                else:  # box
                    proj = y.copy()
                    if self.w_min is not None:
                        proj = np.maximum(proj, self.w_min)
                    if self.w_max is not None:
                        proj = np.minimum(proj, self.w_max)
                    if not np.allclose(proj, y, atol=self.tol):
                        converged = False

                s_new = y - proj
                if not np.allclose(s_new, s_list[j], atol=self.tol):
                    converged = False
                s_list[j] = s_new
                w = proj
            if converged:
                break

        w_int_safe = self.N_G @ w
        int_norm = float(np.linalg.norm(w_int_safe))
        w_obj_safe = self.G @ w
        res_err = float(np.linalg.norm(w_obj_safe - w_obj_des))

        crush_slack = f_crush_cap - int_norm
        crush_bar_h = f_crush_cap * f_crush_cap - int_norm * int_norm

        preload_bar = self.antipodal_preload_barrier
        if f_min_override is not None:
            preload_bar = AntipodalPreloadBarrier(
                N_G=self.N_G,
                n_inward=self.n_inward,
                normal_L=self.normal_L,
                normal_R=self.normal_R,
                f_min=f_preload_min,
                mu_L=self.mu_L,
                mu_R=self.mu_R,
                alpha=self.alpha,
            )
        comps = preload_bar.components(w)
        preload_h = float(min(comps["h_preload"], comps["h_cone_L"], comps["h_cone_R"]))
        box_ok = (
            (self.w_min is None or bool(np.all(w >= self.w_min - 1e-6)))
            and (self.w_max is None or bool(np.all(w <= self.w_max + 1e-6)))
        )
        res_ok = (not preserve_resultant) or (res_err <= 1e-4)
        verified_safe = bool(
            np.all(np.isfinite(w))
            and crush_slack >= -1e-5
            and preload_h >= -1e-5
            and comps["fn_L"] >= -1e-6
            and comps["fn_R"] >= -1e-6
            and box_ok
            and res_ok
        )

        return {
            "status": "safe" if verified_safe else "infeasible_wrench_constraints",
            "verified_safe": verified_safe,
            "w_L_safe": w[:6].copy(),
            "w_R_safe": w[6:12].copy(),
            "w_safe": w.copy(),
            "w_obj_nom": w_obj_nom,
            "w_obj_safe": w_obj_safe,
            "w_int_safe": w_int_safe,
            "internal_wrench_norm": int_norm,
            "resultant_error": res_err,
            "crush_h": crush_bar_h,
            "preload_h": preload_h,
            "preload_components": comps,
            "min_barrier_slack": min(crush_slack, preload_h),
        }


# ---------------------------------------------------------------------------
# TautCableBarrier
# ---------------------------------------------------------------------------


class TautCableBarrier(BarrierConstraint):
    """Smooth over-tension barrier bounding bimanual anchor separation.

    Ensures anchor distance ``d(q_L, q_R) = ||p_R(q_R) - p_L(q_L)||_2 <= L_slack + delta_L_max``
    via either:
      - ``mode="quadratic"``: ``h(x) = L_max^2 - d^2``
      - ``mode="exponential"``: ``h(x) = 1 - exp(exp_scale * (d^2 - L_max^2) / L_max^2)``
    where ``L_max = L_slack + delta_L_max``. Both forms are ``C^inf`` smooth
    everywhere (including ``d = 0``) and compute analytical gradients when URDF
    models, anchor Jacobian callables, or direct coordinate slices are given.
    """

    def __init__(
        self,
        L_slack: float = 0.4,
        delta_L_max: float = 0.05,
        left_model: Optional[URDFRobotModel] = None,
        right_model: Optional[URDFRobotModel] = None,
        left_link: Optional[str] = None,
        right_link: Optional[str] = None,
        local_anchor_L: Sequence[float] = (0.0, 0.0, 0.0),
        local_anchor_R: Sequence[float] = (0.0, 0.0, 0.0),
        T_base_L: Optional[np.ndarray] = None,
        T_base_R: Optional[np.ndarray] = None,
        left_anchor_fn: Optional[Callable[[np.ndarray], Sequence[float]]] = None,
        right_anchor_fn: Optional[Callable[[np.ndarray], Sequence[float]]] = None,
        left_anchor_jac_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        right_anchor_jac_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_dof_L: Optional[int] = None,
        n_dof_R: Optional[int] = None,
        left_slice: Optional[slice] = None,
        right_slice: Optional[slice] = None,
        mode: str = "quadratic",
        exp_scale: float = 2.0,
        alpha: float = 1.0,
        alpha2: float = 2.0,
        relative_degree: int = 1,
        hard: bool = True,
        eps_fd: float = 1e-6,
        margin_offset: float = 0.0,
    ):
        if mode not in ("quadratic", "exponential"):
            raise ValueError(f"mode must be 'quadratic' or 'exponential', got {mode!r}")
        self.L_slack = float(L_slack)
        self.delta_L_max = float(delta_L_max)
        self.L_max = self.L_slack + self.delta_L_max
        if self.L_max <= 0.0:
            raise ValueError("L_slack + delta_L_max must be strictly positive")

        self.left_model = left_model
        self.right_model = right_model
        self.left_link = left_link
        self.right_link = right_link
        self.local_anchor_L = np.asarray(local_anchor_L, dtype=np.float64).flatten()[:3]
        self.local_anchor_R = np.asarray(local_anchor_R, dtype=np.float64).flatten()[:3]
        self.T_base_L = np.eye(4, dtype=np.float64) if T_base_L is None else np.asarray(T_base_L, dtype=np.float64).reshape(4, 4)
        self.T_base_R = np.eye(4, dtype=np.float64) if T_base_R is None else np.asarray(T_base_R, dtype=np.float64).reshape(4, 4)

        self.left_anchor_fn = left_anchor_fn
        self.right_anchor_fn = right_anchor_fn
        self.left_anchor_jac_fn = left_anchor_jac_fn
        self.right_anchor_jac_fn = right_anchor_jac_fn
        self.left_slice = left_slice
        self.right_slice = right_slice

        if n_dof_L is None:
            if left_model is not None:
                n_dof_L = left_model.n_dof
            elif left_slice is not None:
                n_dof_L = int(left_slice.stop - (left_slice.start or 0))
            else:
                n_dof_L = 3
        if n_dof_R is None:
            if right_model is not None:
                n_dof_R = right_model.n_dof
            elif right_slice is not None:
                n_dof_R = int(right_slice.stop - (right_slice.start or 0))
            else:
                n_dof_R = 3

        self.n_dof_L = int(n_dof_L)
        self.n_dof_R = int(n_dof_R)
        self.n_dof = self.n_dof_L + self.n_dof_R
        self.mode = mode
        self.exp_scale = float(exp_scale)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.eps_fd = float(eps_fd)
        self.margin_offset = float(margin_offset)

    def _split_q(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if self.left_slice is not None and self.right_slice is not None:
            return x_arr[self.left_slice], x_arr[self.right_slice]
        if x_arr.shape[0] < self.n_dof:
            raise ValueError(f"Expected state of length >= {self.n_dof}, got {x_arr.shape[0]}")
        return x_arr[: self.n_dof_L], x_arr[self.n_dof_L : self.n_dof]

    def _anchors_and_jacs(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        q_L, q_R = self._split_q(x_arr)

        J_L: Optional[np.ndarray] = None
        if self.left_anchor_fn is not None:
            p_L = np.asarray(self.left_anchor_fn(q_L), dtype=np.float64).flatten()[:3]
            if self.left_anchor_jac_fn is not None:
                J_L = np.asarray(self.left_anchor_jac_fn(q_L), dtype=np.float64).reshape(3, self.n_dof_L)
        elif self.left_model is not None:
            link = self.left_model.end_effector_link_name if self.left_link is None else self.left_link
            p_loc, J_loc = self.left_model.link_point_jacobian(q_L, link, self.local_anchor_L)
            R_b = self.T_base_L[:3, :3]
            p_L = self.T_base_L[:3, 3] + R_b @ p_loc
            J_L = R_b @ J_loc
        else:
            p_L = np.zeros(3, dtype=np.float64)
            p_L[: min(3, q_L.shape[0])] = q_L[: min(3, q_L.shape[0])]
            J_L = np.eye(3, self.n_dof_L, dtype=np.float64)

        J_R: Optional[np.ndarray] = None
        if self.right_anchor_fn is not None:
            p_R = np.asarray(self.right_anchor_fn(q_R), dtype=np.float64).flatten()[:3]
            if self.right_anchor_jac_fn is not None:
                J_R = np.asarray(self.right_anchor_jac_fn(q_R), dtype=np.float64).reshape(3, self.n_dof_R)
        elif self.right_model is not None:
            link = self.right_model.end_effector_link_name if self.right_link is None else self.right_link
            p_loc, J_loc = self.right_model.link_point_jacobian(q_R, link, self.local_anchor_R)
            R_b = self.T_base_R[:3, :3]
            p_R = self.T_base_R[:3, 3] + R_b @ p_loc
            J_R = R_b @ J_loc
        else:
            p_R = np.zeros(3, dtype=np.float64)
            p_R[: min(3, q_R.shape[0])] = q_R[: min(3, q_R.shape[0])]
            J_R = np.eye(3, self.n_dof_R, dtype=np.float64)

        if J_L is None or J_R is None:
            return p_L, p_R, None

        J_diff = np.zeros((3, n), dtype=np.float64)
        if self.left_slice is not None and self.right_slice is not None:
            J_diff[:, self.left_slice] -= J_L
            J_diff[:, self.right_slice] += J_R
        else:
            J_diff[:, : self.n_dof_L] -= J_L
            J_diff[:, self.n_dof_L : self.n_dof] += J_R
        return p_L, p_R, J_diff

    def distance(self, x: Sequence[float]) -> float:
        """Return Euclidean distance ``d(q_L, q_R)`` between left and right cable anchors."""
        p_L, p_R, _ = self._anchors_and_jacs(np.asarray(x, dtype=np.float64))
        return float(np.linalg.norm(p_R - p_L))

    def stretch(self, x: Sequence[float]) -> float:
        """Return cable elongation past slack length ``max(0, d - L_slack)``."""
        return max(0.0, self.distance(x) - self.L_slack)

    def h(self, x: np.ndarray) -> float:
        p_L, p_R, _ = self._anchors_and_jacs(x)
        diff = p_R - p_L
        d2 = float(np.dot(diff, diff))
        l2 = self.L_max * self.L_max
        if self.mode == "quadratic":
            return float(l2 - d2 - self.margin_offset)
        exponent = float(np.clip(self.exp_scale * (d2 - l2) / l2, -50.0, 50.0))
        return float(1.0 - math.exp(exponent) - self.margin_offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        p_L, p_R, J_diff = self._anchors_and_jacs(x_arr)
        if J_diff is None:
            active = x_arr.shape[0] if (self.left_slice is not None or self.right_slice is not None) else self.n_dof
            return _fd_grad_subspace(self.h, x_arr, active, self.eps_fd)

        diff = p_R - p_L
        grad_d2 = 2.0 * (diff @ J_diff)
        if self.mode == "quadratic":
            return -grad_d2
        d2 = float(np.dot(diff, diff))
        l2 = self.L_max * self.L_max
        exponent = float(np.clip(self.exp_scale * (d2 - l2) / l2, -50.0, 50.0))
        return -(self.exp_scale / l2) * math.exp(exponent) * grad_d2

    def inflate(self, delta: float) -> "TautCableBarrier":
        return TautCableBarrier(
            L_slack=self.L_slack,
            delta_L_max=self.delta_L_max,
            left_model=self.left_model,
            right_model=self.right_model,
            left_link=self.left_link,
            right_link=self.right_link,
            local_anchor_L=self.local_anchor_L,
            local_anchor_R=self.local_anchor_R,
            T_base_L=self.T_base_L,
            T_base_R=self.T_base_R,
            left_anchor_fn=self.left_anchor_fn,
            right_anchor_fn=self.right_anchor_fn,
            left_anchor_jac_fn=self.left_anchor_jac_fn,
            right_anchor_jac_fn=self.right_anchor_jac_fn,
            n_dof_L=self.n_dof_L,
            n_dof_R=self.n_dof_R,
            left_slice=self.left_slice,
            right_slice=self.right_slice,
            mode=self.mode,
            exp_scale=self.exp_scale,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            eps_fd=self.eps_fd,
            margin_offset=self.margin_offset + float(delta),
        )
