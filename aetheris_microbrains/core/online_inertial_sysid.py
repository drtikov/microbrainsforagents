"""Online physically-consistent inertial & friction system identification (SPD pseudo-inertia)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def project_pseudo_inertia_spd(
    theta_10: Sequence[float],
    min_mass: float = 1e-3,
    min_eig: float = 1e-6,
) -> Dict[str, Any]:
    """Project 10 inertial parameters onto the 4x4 SPD pseudo-inertia cone.

    ``theta_10 = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]``
    Guarantees ``m > 0`` and all rigid-body triangle inequalities ``I1 + I2 >= I3``.
    """
    th = np.asarray(theta_10, dtype=np.float64).flatten()
    if th.shape[0] != 10:
        raise ValueError(f"Expected 10 inertial parameters, got {th.shape[0]}")
    m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz = [float(v) for v in th]

    # Pseudo-inertia J_P = [[Sigma, h], [h^T, m]] where Sigma = 0.5*tr(I)*I_3 - I
    tr_half = 0.5 * (Ixx + Iyy + Izz)
    Sigma = np.array(
        [
            [tr_half - Ixx, -Ixy, -Ixz],
            [-Ixy, tr_half - Iyy, -Iyz],
            [-Ixz, -Iyz, tr_half - Izz],
        ],
        dtype=np.float64,
    )
    h_vec = np.array([hx, hy, hz], dtype=np.float64)
    J_P = np.zeros((4, 4), dtype=np.float64)
    J_P[:3, :3] = Sigma
    J_P[:3, 3] = h_vec
    J_P[3, :3] = h_vec
    J_P[3, 3] = m
    J_P = 0.5 * (J_P + J_P.T)

    eigvals, eigvecs = np.linalg.eigh(J_P)
    eigvals_proj = np.maximum(eigvals, float(min_eig))
    J_spd = (eigvecs * eigvals_proj) @ eigvecs.T
    J_spd = 0.5 * (J_spd + J_spd.T)
    if J_spd[3, 3] < min_mass:
        J_spd[3, 3] = float(min_mass)

    m_out = float(J_spd[3, 3])
    h_out = J_spd[:3, 3]
    com_out = h_out / m_out
    Sigma_out = J_spd[:3, :3]
    I_out = np.trace(Sigma_out) * np.eye(3, dtype=np.float64) - Sigma_out
    I_out = 0.5 * (I_out + I_out.T)
    theta_out = np.array(
        [
            m_out,
            h_out[0], h_out[1], h_out[2],
            I_out[0, 0], I_out[0, 1], I_out[0, 2],
            I_out[1, 1], I_out[1, 2], I_out[2, 2],
        ],
        dtype=np.float64,
    )
    return {
        "theta_10": theta_out,
        "mass": m_out,
        "com": com_out,
        "inertia_3x3": I_out,
    }


def _wrench_regressor_matrix(
    a_lin_ee: Sequence[float],
    omega_ee: Sequence[float],
    alpha_ee: Sequence[float],
    gravity_ee: Sequence[float],
) -> np.ndarray:
    """Build 6x10 linear-in-parameters spatial wrench regressor [tau_ee; f_ee] = Y * theta_10."""
    a = np.asarray(a_lin_ee, dtype=np.float64).flatten()[:3] - np.asarray(gravity_ee, dtype=np.float64).flatten()[:3]
    w = np.asarray(omega_ee, dtype=np.float64).flatten()[:3]
    wd = np.asarray(alpha_ee, dtype=np.float64).flatten()[:3]

    def _hat(v: np.ndarray) -> np.ndarray:
        return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)

    def _sym_dot(v: np.ndarray) -> np.ndarray:
        # v = [vx, vy, vz] -> 3x6 matrix mapping [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return np.array(
            [
                [vx, vy, vz, 0.0, 0.0, 0.0],
                [0.0, vx, 0.0, vy, vz, 0.0],
                [0.0, 0.0, vx, 0.0, vy, vz],
            ],
            dtype=np.float64,
        )

    Y = np.zeros((6, 10), dtype=np.float64)
    # Force f = m * a + (wd_x + w_x w_x) * h
    Y[3:, 0] = a
    Y[3:, 1:4] = _hat(wd) + _hat(w) @ _hat(w)
    # Torque tau = -a_x * h + I * wd + w_x (I * w)
    Y[:3, 1:4] = -_hat(a)
    Y[:3, 4:10] = _sym_dot(wd) + _hat(w) @ _sym_dot(w)
    return Y


class PhysicalPayloadIdentifier:
    """Online forgetting-factor RLS payload & friction estimator with SPD projection."""

    def __init__(
        self,
        forgetting_factor: float = 0.985,
        init_cov: float = 100.0,
        min_mass: float = 1e-3,
    ):
        self.lam = float(np.clip(forgetting_factor, 0.85, 1.0))
        self.min_mass = float(min_mass)
        self.theta_raw = np.zeros(10, dtype=np.float64)
        self.theta_raw[0] = 0.1
        self.P = float(init_cov) * np.eye(10, dtype=np.float64)
        self.last_projected = project_pseudo_inertia_spd(self.theta_raw, min_mass=self.min_mass)
        self.updates = 0

    def update_ee_wrench(
        self,
        a_lin_ee: Sequence[float],
        omega_ee: Sequence[float],
        alpha_ee: Sequence[float],
        gravity_ee: Sequence[float],
        wrench_meas: Sequence[float],
    ) -> Dict[str, Any]:
        Y = _wrench_regressor_matrix(a_lin_ee, omega_ee, alpha_ee, gravity_ee)
        y = np.asarray(wrench_meas, dtype=np.float64).flatten()[:6]
        err = y - Y @ self.theta_raw
        S = self.lam * np.eye(6, dtype=np.float64) + Y @ self.P @ Y.T
        K = np.linalg.solve(S, Y @ self.P).T
        self.theta_raw = self.theta_raw + K @ err
        self.P = (self.P - K @ Y @ self.P) / self.lam
        self.P = 0.5 * (self.P + self.P.T)
        self.updates += 1
        self.last_projected = project_pseudo_inertia_spd(self.theta_raw, min_mass=self.min_mass)
        return self.last_projected
