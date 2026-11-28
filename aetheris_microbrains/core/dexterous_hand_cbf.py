"""Multi-finger dexterous hand force-closure, tactile slip, and I^2*t thermal CBFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .urdf_spatial_kinematics import hat_so3


@dataclass(frozen=True)
class DexterousContact:
    finger_name: str
    position_3d: Tuple[float, float, float]
    normal_3d: Tuple[float, float, float]
    friction_mu: float = 0.6


class DexterousGraspModel:
    """N-finger (e.g. Dex3-1 3-finger or 5-finger) grasp matrix and force-closure analyzer."""

    def __init__(self, contacts: Sequence[DexterousContact], object_com: Sequence[float] = (0.0, 0.0, 0.0)):
        if len(contacts) < 2:
            raise ValueError("DexterousGraspModel requires at least 2 contacts")
        self.contacts = list(contacts)
        self.object_com = np.asarray(object_com, dtype=np.float64).flatten()[:3]

    def grasp_matrix(self) -> np.ndarray:
        """Build 6 x (3*N_f) point-contact-with-friction grasp matrix G."""
        blocks: List[np.ndarray] = []
        for c in self.contacts:
            r = np.asarray(c.position_3d, dtype=np.float64) - self.object_com
            G_i = np.vstack([np.eye(3, dtype=np.float64), hat_so3(r)])
            blocks.append(G_i)
        return np.hstack(blocks)

    def force_closure_margin(self, n_cone_edges: int = 8) -> float:
        """Compute signed Ferrari-Canny force-closure margin (> 0 implies force closure)."""
        G = self.grasp_matrix()
        rays: List[np.ndarray] = []
        for idx, c in enumerate(self.contacts):
            n = np.asarray(c.normal_3d, dtype=np.float64)
            n_norm = float(np.linalg.norm(n))
            n = n / max(n_norm, 1e-12)
            ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t1 = np.cross(n, ref)
            t1 /= max(float(np.linalg.norm(t1)), 1e-12)
            t2 = np.cross(n, t1)
            G_i = G[:, 3 * idx : 3 * (idx + 1)]
            for k in range(n_cone_edges):
                ang = 2.0 * np.pi * k / n_cone_edges
                f_dir = n + c.friction_mu * (np.cos(ang) * t1 + np.sin(ang) * t2)
                f_dir /= max(float(np.linalg.norm(f_dir)), 1e-12)
                rays.append(G_i @ f_dir)
        W = np.column_stack(rays)  # 6 x M
        # Check whether 0 lies strictly inside convex hull of wrench rays via directional support
        # Min over sampled unit directions d of max_j (d^T w_j): > 0 iff 0 is in interior of conv(W)
        rng = np.random.RandomState(42)
        dirs = rng.randn(128, 6)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # Include coordinate axes
        axes = np.vstack([np.eye(6), -np.eye(6), dirs])
        supports = np.max(axes @ W, axis=1)
        return float(np.min(supports))


class DexterousTactileCBF:
    """Per-fingertip tactile slip/fragile squeeze barrier and I^2*t micro-servo thermal CBF."""

    def __init__(
        self,
        n_joints: int = 7,
        tau_cont: float = 0.8,
        i2t_max: float = 4.0,
        f_fragile_max: float = 15.0,
        mu_default: float = 0.6,
    ):
        self.n_joints = int(n_joints)
        self.tau_cont = float(tau_cont)
        self.i2t_max = float(i2t_max)
        self.f_fragile_max = float(f_fragile_max)
        self.mu_default = float(mu_default)
        self.thermal_energy_j2s = np.zeros(self.n_joints, dtype=np.float64)

    def step_tactile_and_thermal(
        self,
        tau_nom: Sequence[float],
        f_normal_meas: Sequence[float],
        f_tan_meas: Sequence[float],
        dt: float = 0.002,
    ) -> Dict[str, Any]:
        tau = np.asarray(tau_nom, dtype=np.float64).flatten()[: self.n_joints].copy()
        fn = np.asarray(f_normal_meas, dtype=np.float64).flatten()
        ft = np.asarray(f_tan_meas, dtype=np.float64).flatten()

        # Incipient slip check: f_normal >= |f_tan| / mu + margin
        f_req = np.abs(ft) / max(self.mu_default, 1e-3) + 0.5
        slip_deficit = np.maximum(0.0, f_req - fn)
        boost = float(np.max(slip_deficit)) if slip_deficit.size else 0.0

        # Fragile object squeeze cap
        excess_squeeze = np.maximum(0.0, fn - self.f_fragile_max)
        squeeze_scale = 1.0 if not excess_squeeze.size or float(np.max(excess_squeeze)) == 0.0 else max(0.2, 1.0 - float(np.max(excess_squeeze)) / self.f_fragile_max)
        tau = tau * squeeze_scale

        # Continuous I^2*t thermal energy accumulation & derating
        self.thermal_energy_j2s = np.maximum(
            0.0,
            self.thermal_energy_j2s + (tau * tau - self.tau_cont * self.tau_cont) * float(dt),
        )
        therm_margin = np.maximum(0.0, self.i2t_max - self.thermal_energy_j2s)
        therm_scale = np.clip(therm_margin / max(0.25 * self.i2t_max, 1e-6), 0.15, 1.0)
        tau_safe = np.sign(tau) * np.minimum(np.abs(tau), self.tau_cont + therm_scale * np.abs(tau))

        return {
            "tau_safe": tau_safe.tolist(),
            "slip_force_boost_n": boost,
            "squeeze_scale": float(squeeze_scale),
            "min_thermal_margin_j2s": float(np.min(therm_margin)),
            "thermal_derated": bool(np.any(therm_scale < 1.0 - 1e-6)),
        }
