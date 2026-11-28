"""Whole-body loco-manipulation momentum coupling and 2.5D terrain foothold CBFs."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .legged_centroidal_cbf import CapturePointDCMBarrier
from .state_cbf_filter import BarrierConstraint
from .urdf_spatial_kinematics import URDFRobotModel


class LocoManipulationBalanceCBF:
    """Couples upper-body arm accelerations into Capture-Point / ZMP balance constraints."""

    def __init__(
        self,
        robot: URDFRobotModel,
        dcm_barrier: CapturePointDCMBarrier,
        loco_indices: Sequence[int],
        arm_indices: Sequence[int],
        gravity: float = 9.81,
    ):
        self.robot = robot
        self.dcm_barrier = dcm_barrier
        self.loco_idx = np.asarray(loco_indices, dtype=np.int64)
        self.arm_idx = np.asarray(arm_indices, dtype=np.int64)
        self.gravity = float(gravity)

    def arm_balance_halfspaces(
        self,
        q: Sequence[float],
        v: Sequence[float],
        qddot_loco: Optional[Sequence[float]] = None,
        z_com_override: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return linear halfspaces A_arm * qddot_arm >= b_arm preserving DCM balance."""
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        v_arr = np.asarray(v, dtype=np.float64).flatten()
        p_com = self.robot.center_of_mass(q_arr)
        J_com = self.robot.com_jacobian(q_arr)
        v_com = J_com @ v_arr

        z_c = max(0.15, float(p_com[2] if z_com_override is None else z_com_override))
        omega_0 = math.sqrt(self.gravity / z_c)
        xi = p_com[:2] + v_com[:2] / omega_0

        J_xy_arm = J_com[:2, self.arm_idx]
        J_xy_loco = J_com[:2, self.loco_idx]
        ddq_loco = np.zeros(len(self.loco_idx), dtype=np.float64) if qddot_loco is None else np.asarray(qddot_loco, dtype=np.float64).flatten()

        # d(xi)/dt = v_com_xy + (1/omega_0) * (J_xy_loco * ddq_loco + J_xy_arm * ddq_arm)
        A_poly = self.dcm_barrier.A
        b_poly = self.dcm_barrier.b
        alpha = float(self.dcm_barrier.alpha)

        A_arm = (A_poly @ J_xy_arm) / omega_0
        drift = v_com[:2] + (J_xy_loco @ ddq_loco) / omega_0
        h_vals = A_poly @ xi - b_poly
        b_arm = -alpha * h_vals - (A_poly @ drift)
        return A_arm, b_arm


class TerrainHeightmap25D:
    """2.5D elevation grid computing slope, roughness, and 2D step-edge clearance."""

    def __init__(
        self,
        elevation: np.ndarray,
        origin_xy: Sequence[float] = (0.0, 0.0),
        resolution: float = 0.05,
        max_slope_rad: float = 0.35,
        max_roughness: float = 0.04,
    ):
        self.elevation = np.asarray(elevation, dtype=np.float64)
        self.origin_xy = np.asarray(origin_xy, dtype=np.float64).flatten()[:2]
        self.resolution = max(1e-4, float(resolution))
        self.max_slope_rad = float(max_slope_rad)
        self.max_roughness = float(max_roughness)

        gx, gy = np.gradient(self.elevation, self.resolution)
        self.slope = np.arctan(np.hypot(gx, gy))

        # Local 3x3 roughness
        pad = np.pad(self.elevation, 1, mode="edge")
        nx, ny = self.elevation.shape
        neigh = np.stack([pad[i:i + nx, j:j + ny] for i in range(3) for j in range(3)], axis=0)
        self.roughness = np.std(neigh, axis=0)
        self.traversable = (self.slope <= self.max_slope_rad) & (self.roughness <= self.max_roughness)

    def world_to_grid(self, xy: Sequence[float]) -> Tuple[int, int]:
        p = np.asarray(xy, dtype=np.float64).flatten()[:2]
        ix = int(np.clip(round((p[0] - self.origin_xy[0]) / self.resolution), 0, self.elevation.shape[0] - 1))
        iy = int(np.clip(round((p[1] - self.origin_xy[1]) / self.resolution), 0, self.elevation.shape[1] - 1))
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> np.ndarray:
        return self.origin_xy + np.array([ix * self.resolution, iy * self.resolution], dtype=np.float64)


class TerrainFootholdCBF(BarrierConstraint):
    """Smooth traversability barrier over 2.5D terrain heightmap."""

    def __init__(self, heightmap: TerrainHeightmap25D, alpha: float = 1.0, offset: float = 0.0):
        self.heightmap = heightmap
        self.alpha = float(alpha)
        self.offset = float(offset)

    def h(self, x: np.ndarray) -> float:
        p = np.asarray(x, dtype=np.float64).flatten()[:2]
        ix, iy = self.heightmap.world_to_grid(p)
        s_margin = math.tan(self.heightmap.max_slope_rad) - math.tan(float(self.heightmap.slope[ix, iy]))
        r_margin = self.heightmap.max_roughness - float(self.heightmap.roughness[ix, iy])
        return float(min(s_margin, r_margin) - self.offset)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        grad = np.zeros_like(x_arr)
        eps = self.heightmap.resolution
        for i in range(min(2, x_arr.shape[0])):
            ei = np.zeros_like(x_arr)
            ei[i] = eps
            grad[i] = (self.h(x_arr + ei) - self.h(x_arr - ei)) / (2.0 * eps)
        return grad

    def inflate(self, delta: float) -> "TerrainFootholdCBF":
        return TerrainFootholdCBF(self.heightmap, alpha=self.alpha, offset=self.offset + float(delta))


class SafeFootholdProjector:
    """Projects unsafe swing-foot landing targets onto nearest traversable terrain cell."""

    def __init__(self, heightmap: TerrainHeightmap25D):
        self.heightmap = heightmap

    def project(self, xy_nom: Sequence[float]) -> Dict[str, Any]:
        p = np.asarray(xy_nom, dtype=np.float64).flatten()[:2]
        ix, iy = self.heightmap.world_to_grid(p)
        if self.heightmap.traversable[ix, iy]:
            return {"xy_safe": p.tolist(), "projected": False, "elevation_m": float(self.heightmap.elevation[ix, iy])}
        valid_idx = np.argwhere(self.heightmap.traversable)
        if valid_idx.size == 0:
            return {"xy_safe": p.tolist(), "projected": False, "elevation_m": float(self.heightmap.elevation[ix, iy]), "no_safe_foothold": True}
        pts = self.heightmap.origin_xy[None, :] + valid_idx * self.heightmap.resolution
        dists = np.linalg.norm(pts - p[None, :], axis=1)
        best = int(np.argmin(dists))
        b_ix, b_iy = int(valid_idx[best, 0]), int(valid_idx[best, 1])
        return {
            "xy_safe": pts[best].tolist(),
            "projected": True,
            "shift_m": float(dists[best]),
            "elevation_m": float(self.heightmap.elevation[b_ix, b_iy]),
        }
