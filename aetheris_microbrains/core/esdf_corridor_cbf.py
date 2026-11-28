"""Voxel ESDF + Safe Polytope Corridor + Topological Homotopy Guidance for CBF Safety Filters.

Provides:
- Exact O(V) Felzenszwalb-Huttenlocher 1D and 3D Euclidean Distance Transform
  (`_edt_1d`, `_compute_3d_esdf_grid`) producing signed distance (positive outside
  occupied voxels, negative inside occupied interiors).
- `VoxelESDFGrid`: continuous trilinear SDF + analytical spatial gradient queries
  directly from raw 3D point clouds.
- `VoxelSDFBarrier(BarrierConstraint)`: drop-in barrier for `StateCBFSafetyFilter`
  enforcing h(x) = d_SDF(p(x)) - r_robot >= 0.
- `inflate_safe_polytope`: iterative closest-point separating tangent-plane
  peeling producing compact convex polytope corridors compatible with both
  `HalfPlaneBarrier` / `StateCBFSafetyFilter` and `CapabilityCaveat.geofence_polytope`.
- `TopologicalHomotopyGuide`: free-space visibility roadmap + `TopologicalPersistenceMerger`
  Betti-1 cycle detection and distinct homotopy waypoint extraction to rescue
  local CBF controllers from non-convex U-shaped deadlocks.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .state_cbf_filter import BarrierConstraint, HalfPlaneBarrier
from .topological_persistence_merger import TopologicalPersistenceMerger


def _edt_1d(f: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Exact O(n) Felzenszwalb-Huttenlocher 1D squared Euclidean distance transform.

    Computes ``d[q] = min_r ((q - r) * spacing)^2 + f[r]`` for ``0 <= q < n``.
    Handles ``np.inf`` entries in ``f`` without NaN propagation.
    """
    arr = np.asarray(f, dtype=np.float64).flatten()
    n = int(arr.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    s_val = float(spacing)
    if s_val <= 0.0:
        raise ValueError("spacing must be positive")
    if not bool(np.any(np.isfinite(arr))):
        return np.full(n, np.inf, dtype=np.float64)

    s2 = s_val * s_val
    scaled_f = (arr / s2).tolist()

    v = [0] * n
    z = [0.0] * (n + 1)
    k = -1
    isfinite = math.isfinite

    for q in range(n):
        fq = scaled_f[q]
        if not isfinite(fq):
            continue
        if k < 0:
            k = 0
            v[0] = q
            z[0] = -math.inf
            z[1] = math.inf
            continue

        s = 0.0
        qq = q * q
        while k >= 0:
            r = v[k]
            s = ((fq + qq) - (scaled_f[r] + r * r)) / (2.0 * (q - r))
            if s <= z[k]:
                k -= 1
            else:
                break
        k += 1
        v[k] = q
        z[k] = -math.inf if k == 0 else s
        z[k + 1] = math.inf

    if k < 0:
        return np.full(n, np.inf, dtype=np.float64)

    d = [0.0] * n
    idx = 0
    for q in range(n):
        while z[idx + 1] < q:
            idx += 1
        r = v[idx]
        diff = float(q - r)
        d[q] = (diff * diff + scaled_f[r]) * s2
    return np.asarray(d, dtype=np.float64)


def _edt_3d_squared(seed_mask: np.ndarray, voxel_sizes: Tuple[float, float, float]) -> np.ndarray:
    """Compute exact 3D squared Euclidean distance to True entries in ``seed_mask``."""
    nx, ny, nz = seed_mask.shape
    dx, dy, dz = voxel_sizes

    grid = np.where(seed_mask, 0.0, np.inf)

    # Pass along X
    for iy in range(ny):
        for iz in range(nz):
            grid[:, iy, iz] = _edt_1d(grid[:, iy, iz], spacing=dx)

    # Pass along Y
    for ix in range(nx):
        for iz in range(nz):
            grid[ix, :, iz] = _edt_1d(grid[ix, :, iz], spacing=dy)

    # Pass along Z
    for ix in range(nx):
        for iy in range(ny):
            grid[ix, iy, :] = _edt_1d(grid[ix, iy, :], spacing=dz)

    return grid


def _coerce_voxel_size(voxel_size: Union[float, Sequence[float]]) -> Tuple[float, float, float]:
    if isinstance(voxel_size, (int, float)):
        v = float(voxel_size)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("voxel_size must be positive and finite")
        return (v, v, v)
    arr = np.asarray(voxel_size, dtype=np.float64).flatten()
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("voxel_size must be a positive scalar or 3-element sequence")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _compute_3d_esdf_grid(
    occupancy_grid: np.ndarray,
    voxel_size: Union[float, Sequence[float]] = 1.0,
) -> np.ndarray:
    """Compute exact 3D signed Euclidean distance field over a boolean occupancy grid.

    Returns positive distances outside occupied voxels and negative signed distances
    inside occupied voxel interiors: ``sdf = dist_to_occupied - dist_to_free``.
    """
    occ = np.asarray(occupancy_grid, dtype=bool)
    if occ.ndim != 3:
        raise ValueError(f"occupancy_grid must be 3D, got shape {occ.shape}")
    if occ.size == 0:
        raise ValueError("occupancy_grid must be non-empty")

    vs = _coerce_voxel_size(voxel_size)
    has_occ = bool(np.any(occ))
    has_free = bool(np.any(~occ))

    if not has_occ:
        diag = float(np.linalg.norm(np.array(occ.shape, dtype=np.float64) * np.array(vs)))
        return np.full(occ.shape, max(diag, 1.0), dtype=np.float64)
    if not has_free:
        diag = float(np.linalg.norm(np.array(occ.shape, dtype=np.float64) * np.array(vs)))
        return np.full(occ.shape, -max(diag, 1.0), dtype=np.float64)

    sq_out = _edt_3d_squared(occ, vs)
    sq_in = _edt_3d_squared(~occ, vs)
    dist_out = np.sqrt(np.maximum(sq_out, 0.0))
    dist_in = np.sqrt(np.maximum(sq_in, 0.0))
    return dist_out - dist_in


class VoxelESDFGrid:
    """Continuous 3D Voxel Signed Distance Field with trilinear value & gradient queries."""

    def __init__(
        self,
        sdf: np.ndarray,
        origin: Sequence[float],
        voxel_size: Union[float, Sequence[float]],
        inflation_radius: float = 0.0,
    ):
        sdf_arr = np.asarray(sdf, dtype=np.float64)
        if sdf_arr.ndim != 3 or sdf_arr.size == 0:
            raise ValueError("sdf must be a non-empty 3D array")
        origin_arr = np.asarray(origin, dtype=np.float64).flatten()
        if origin_arr.shape != (3,) or not np.all(np.isfinite(origin_arr)):
            raise ValueError("origin must be a finite 3D coordinate")

        self.sdf = sdf_arr
        self.origin = origin_arr
        self.voxel_size = np.asarray(_coerce_voxel_size(voxel_size), dtype=np.float64)
        self.grid_shape = sdf_arr.shape
        self.inflation_radius = float(inflation_radius)

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray,
        origin: Sequence[float],
        voxel_size: Union[float, Sequence[float]],
        grid_shape: Sequence[int],
        inflation_radius: float = 0.0,
    ) -> "VoxelESDFGrid":
        """Build a VoxelESDFGrid from a raw point cloud ``P`` of shape ``(N, 3)``."""
        shape_tuple = tuple(int(s) for s in grid_shape)
        if len(shape_tuple) != 3 or any(s <= 0 for s in shape_tuple):
            raise ValueError("grid_shape must have 3 positive integers")

        origin_arr = np.asarray(origin, dtype=np.float64).flatten()
        if origin_arr.shape != (3,) or not np.all(np.isfinite(origin_arr)):
            raise ValueError("origin must be a finite 3D coordinate")

        vs = np.asarray(_coerce_voxel_size(voxel_size), dtype=np.float64)
        occ = np.zeros(shape_tuple, dtype=bool)

        pts = np.asarray(points, dtype=np.float64)
        if pts.size > 0:
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
            finite_mask = np.all(np.isfinite(pts), axis=1)
            pts_valid = pts[finite_mask]
            if pts_valid.shape[0] > 0:
                rel = (pts_valid - origin_arr) / vs
                in_bounds = np.all((rel >= 0.0) & (rel <= np.array(shape_tuple, dtype=np.float64)), axis=1)
                pts_in = rel[in_bounds]
                if pts_in.shape[0] > 0:
                    idx = np.floor(pts_in).astype(np.int64)
                    idx = np.clip(idx, 0, np.array(shape_tuple, dtype=np.int64) - 1)
                    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

        sdf = _compute_3d_esdf_grid(occ, tuple(vs.tolist()))
        inf_r = float(inflation_radius)
        if inf_r != 0.0:
            sdf = sdf - inf_r
        return cls(sdf=sdf, origin=origin_arr, voxel_size=tuple(vs.tolist()), inflation_radius=inf_r)

    def voxel_center(self, ix: int, iy: int, iz: int) -> np.ndarray:
        """World-frame 3D center coordinate of voxel ``(ix, iy, iz)``."""
        return self.origin + (np.array([ix, iy, iz], dtype=np.float64) + 0.5) * self.voxel_size

    def _trilinear_eval(self, pos: np.ndarray) -> Tuple[float, np.ndarray]:
        nx, ny, nz = self.grid_shape
        u = (pos - self.origin) / self.voxel_size - 0.5
        max_u = np.array([max(0, nx - 1), max(0, ny - 1), max(0, nz - 1)], dtype=np.float64)
        u_c = np.clip(u, 0.0, max_u)

        def _axis_idx(val: float, n: int) -> Tuple[int, int, float]:
            if n <= 1:
                return 0, 0, 0.0
            i0 = int(math.floor(val))
            if i0 >= n - 1:
                i0 = n - 2
            elif i0 < 0:
                i0 = 0
            t = float(val - i0)
            return i0, i0 + 1, t

        i0, i1, tx = _axis_idx(float(u_c[0]), nx)
        j0, j1, ty = _axis_idx(float(u_c[1]), ny)
        k0, k1, tz = _axis_idx(float(u_c[2]), nz)

        c000 = self.sdf[i0, j0, k0]
        c100 = self.sdf[i1, j0, k0]
        c010 = self.sdf[i0, j1, k0]
        c110 = self.sdf[i1, j1, k0]
        c001 = self.sdf[i0, j0, k1]
        c101 = self.sdf[i1, j0, k1]
        c011 = self.sdf[i0, j1, k1]
        c111 = self.sdf[i1, j1, k1]

        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        val = (
            c000 * wx0 * wy0 * wz0
            + c100 * wx1 * wy0 * wz0
            + c010 * wx0 * wy1 * wz0
            + c110 * wx1 * wy1 * wz0
            + c001 * wx0 * wy0 * wz1
            + c101 * wx1 * wy0 * wz1
            + c011 * wx0 * wy1 * wz1
            + c111 * wx1 * wy1 * wz1
        )

        df_dx = (
            (c100 - c000) * wy0 * wz0
            + (c110 - c010) * wy1 * wz0
            + (c101 - c001) * wy0 * wz1
            + (c111 - c011) * wy1 * wz1
        ) / self.voxel_size[0]

        df_dy = (
            (c010 - c000) * wx0 * wz0
            + (c110 - c100) * wx1 * wz0
            + (c011 - c001) * wx0 * wz1
            + (c111 - c101) * wx1 * wz1
        ) / self.voxel_size[1]

        df_dz = (
            (c001 - c000) * wx0 * wy0
            + (c101 - c100) * wx1 * wy0
            + (c011 - c010) * wx0 * wy1
            + (c111 - c110) * wx1 * wy1
        ) / self.voxel_size[2]

        grad = np.array([df_dx, df_dy, df_dz], dtype=np.float64)

        # Smooth linear extrapolation beyond the clamped voxel-center bounding box
        pos_c = self.origin + (u_c + 0.5) * self.voxel_size
        delta = pos - pos_c
        if np.any(delta != 0.0):
            dist_out = float(np.linalg.norm(delta))
            if np.all(np.abs(delta) <= 0.5 * self.voxel_size + 1e-12):
                val = float(val + np.dot(grad, delta))
            else:
                # Far outside grid: grow distance smoothly away from clamped boundary
                val = float(val + np.dot(grad, np.clip(delta, -0.5 * self.voxel_size, 0.5 * self.voxel_size)) + dist_out)
                if dist_out > 1e-12:
                    grad = delta / dist_out

        return float(val), grad

    def query_distance_and_grad(self, pos_3d: Sequence[float]) -> Tuple[float, np.ndarray]:
        """Return smooth trilinearly interpolated signed distance and analytical 3D gradient."""
        pos = np.asarray(pos_3d, dtype=np.float64).flatten()
        if pos.shape != (3,):
            raise ValueError(f"pos_3d must have 3 coordinates, got shape {pos.shape}")
        return self._trilinear_eval(pos)

    def query_distance(self, pos_3d: Sequence[float]) -> float:
        """Return smooth trilinearly interpolated signed distance at ``pos_3d``."""
        d, _ = self.query_distance_and_grad(pos_3d)
        return d

    def _query_distances_batch(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized trilinear SDF evaluation for an ``(M, 3)`` array of points."""
        pts_arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        m = pts_arr.shape[0]
        if m == 0:
            return np.zeros(0, dtype=np.float64)

        nx, ny, nz = self.grid_shape
        u = (pts_arr - self.origin) / self.voxel_size - 0.5
        max_u = np.array([max(0, nx - 1), max(0, ny - 1), max(0, nz - 1)], dtype=np.float64)
        u_c = np.clip(u, 0.0, max_u)

        def _axis_vec(col: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if n <= 1:
                zeros = np.zeros(m, dtype=np.int64)
                return zeros, zeros, np.zeros(m, dtype=np.float64)
            i0 = np.clip(np.floor(col).astype(np.int64), 0, n - 2)
            i1 = i0 + 1
            t = col - i0
            return i0, i1, t

        i0, i1, tx = _axis_vec(u_c[:, 0], nx)
        j0, j1, ty = _axis_vec(u_c[:, 1], ny)
        k0, k1, tz = _axis_vec(u_c[:, 2], nz)

        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        val = (
            self.sdf[i0, j0, k0] * wx0 * wy0 * wz0
            + self.sdf[i1, j0, k0] * wx1 * wy0 * wz0
            + self.sdf[i0, j1, k0] * wx0 * wy1 * wz0
            + self.sdf[i1, j1, k0] * wx1 * wy1 * wz0
            + self.sdf[i0, j0, k1] * wx0 * wy0 * wz1
            + self.sdf[i1, j0, k1] * wx1 * wy0 * wz1
            + self.sdf[i0, j1, k1] * wx0 * wy1 * wz1
            + self.sdf[i1, j1, k1] * wx1 * wy1 * wz1
        )
        return val


class VoxelSDFBarrier(BarrierConstraint):
    """Control Barrier Function ``h(x) = d_SDF(p(x)) - r_robot >= 0`` over a VoxelESDFGrid."""

    def __init__(
        self,
        esdf_grid: VoxelESDFGrid,
        robot_radius: float = 0.0,
        alpha: float = 0.5,
        relative_degree: int = 1,
        hard: bool = True,
        alpha2: float = 1.0,
        state_slice: Optional[slice] = None,
        default_z: Optional[float] = None,
    ):
        self.esdf_grid = esdf_grid
        self.robot_radius = float(robot_radius)
        self.alpha = float(alpha)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.alpha2 = float(alpha2)
        self.state_slice: slice = slice(0, 3) if state_slice is None else state_slice
        if default_z is None:
            z_center = float(esdf_grid.origin[2] + 0.5 * esdf_grid.grid_shape[2] * esdf_grid.voxel_size[2])
            self.default_z = z_center
        else:
            self.default_z = float(default_z)

    def _extract_pos_3d(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        sub = x_arr[self.state_slice]
        sub_dim = int(sub.shape[0])
        if sub_dim == 3:
            return sub, 3
        if sub_dim == 2:
            return np.array([sub[0], sub[1], self.default_z], dtype=np.float64), 2
        raise ValueError(f"VoxelSDFBarrier state_slice must yield 2 or 3 coordinates, got {sub_dim}")

    def h(self, x: np.ndarray) -> float:
        pos_3d, _ = self._extract_pos_3d(x)
        d, _ = self.esdf_grid.query_distance_and_grad(pos_3d)
        return float(d - self.robot_radius)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        pos_3d, sub_dim = self._extract_pos_3d(x_arr)
        _, grad_3d = self.esdf_grid.query_distance_and_grad(pos_3d)
        out = np.zeros(x_arr.shape[0], dtype=np.float64)
        out[self.state_slice] = grad_3d[:sub_dim]
        return out

    def inflate(self, delta: float) -> "VoxelSDFBarrier":
        return VoxelSDFBarrier(
            self.esdf_grid,
            robot_radius=self.robot_radius + float(delta),
            alpha=self.alpha,
            relative_degree=self.relative_degree,
            hard=self.hard,
            alpha2=self.alpha2,
            state_slice=self.state_slice,
            default_z=self.default_z,
        )


def inflate_safe_polytope(
    seed_point: Sequence[float],
    obstacle_points: Optional[np.ndarray],
    bounding_min: Optional[Sequence[float]] = None,
    bounding_max: Optional[Sequence[float]] = None,
    max_planes: int = 16,
    safety_margin: float = 0.05,
) -> Dict[str, Any]:
    """Fast IRIS-style convex corridor inflation around ``seed_point``.

    Iteratively peels the closest un-separated obstacle point with a tangent
    halfspace, returning a compact polytope ``A p <= b`` compatible with both
    ``StateCBFSafetyFilter`` (``HalfPlaneBarrier``: ``(-a) . p >= -b``) and
    ``CapabilityCaveat.geofence_polytope`` (``a . p <= b``).
    """
    seed = np.asarray(seed_point, dtype=np.float64).flatten()
    dim = int(seed.shape[0])
    if dim == 0 or not np.all(np.isfinite(seed)):
        raise ValueError("seed_point must be a non-empty finite coordinate vector")
    if max_planes < 1:
        raise ValueError("max_planes must be >= 1")
    margin = max(0.0, float(safety_margin))

    normals: List[np.ndarray] = []
    offsets: List[float] = []
    is_box_plane: List[bool] = []

    if bounding_min is not None:
        bmin = np.asarray(bounding_min, dtype=np.float64).flatten()
        if bmin.shape != (dim,):
            raise ValueError(f"bounding_min must have length {dim}")
        if np.any(seed < bmin):
            raise ValueError("seed_point lies below bounding_min")
        for k in range(dim):
            if len(normals) >= max_planes:
                break
            e = np.zeros(dim, dtype=np.float64)
            e[k] = -1.0
            normals.append(e)
            offsets.append(-float(bmin[k]))
            is_box_plane.append(True)

    if bounding_max is not None:
        bmax = np.asarray(bounding_max, dtype=np.float64).flatten()
        if bmax.shape != (dim,):
            raise ValueError(f"bounding_max must have length {dim}")
        if np.any(seed > bmax):
            raise ValueError("seed_point lies above bounding_max")
        for k in range(dim):
            if len(normals) >= max_planes:
                break
            e = np.zeros(dim, dtype=np.float64)
            e[k] = 1.0
            normals.append(e)
            offsets.append(float(bmax[k]))
            is_box_plane.append(True)

    if obstacle_points is not None:
        obs_raw = np.asarray(obstacle_points, dtype=np.float64)
        if obs_raw.size > 0:
            obs = obs_raw.reshape(-1, dim)
            obs = obs[np.all(np.isfinite(obs), axis=1)]
        else:
            obs = np.zeros((0, dim), dtype=np.float64)
    else:
        obs = np.zeros((0, dim), dtype=np.float64)

    def _remaining_mask(pts: np.ndarray, n_list: List[np.ndarray], b_list: List[float]) -> np.ndarray:
        if pts.shape[0] == 0 or not n_list:
            return np.ones(pts.shape[0], dtype=bool)
        A_cur = np.vstack(n_list)
        b_cur = np.asarray(b_list, dtype=np.float64)
        # Points not yet strictly excluded (still satisfy A_cur @ q <= b_cur)
        return np.all((pts @ A_cur.T) <= (b_cur + 1e-12), axis=1)

    # Iterative closest-point tangent-plane peeling
    active = _remaining_mask(obs, normals, offsets)
    while len(normals) < max_planes and np.any(active):
        rem_pts = obs[active]
        diffs = rem_pts - seed
        dists = np.linalg.norm(diffs, axis=1)
        pos_mask = dists > 1e-12
        if not np.any(pos_mask):
            break
        rem_pts = rem_pts[pos_mask]
        diffs = diffs[pos_mask]
        dists = dists[pos_mask]

        best_idx = int(np.argmin(dists))
        q_star = rem_pts[best_idx]
        dist_star = float(dists[best_idx])
        normal = diffs[best_idx] / dist_star

        eff_margin = min(margin, 0.5 * dist_star)
        if eff_margin < 1e-9:
            eff_margin = 0.5 * dist_star
        b_plane = float(np.dot(normal, q_star) - eff_margin)

        # Deduplicate nearly parallel planes: replace looser parallel plane if present
        replaced = False
        for i, existing_n in enumerate(normals):
            if float(np.dot(existing_n, normal)) > 1.0 - 1e-9:
                if b_plane < offsets[i]:
                    offsets[i] = b_plane
                    is_box_plane[i] = False
                replaced = True
                break
        if not replaced:
            normals.append(normal)
            offsets.append(b_plane)
            is_box_plane.append(False)

        active = _remaining_mask(obs, normals, offsets)

    # Fallback tightening if max_planes was reached with residual un-excluded obstacle points
    if np.any(active) and normals:
        rem_pts = obs[active]
        A_cur = np.vstack(normals)
        b_cur = np.asarray(offsets, dtype=np.float64)
        obs_indices = [i for i, bx in enumerate(is_box_plane) if not bx]
        search_indices = obs_indices if obs_indices else list(range(len(normals)))

        for q in rem_pts:
            if np.any(A_cur @ q > b_cur + 1e-12):
                continue
            dq = q - seed
            dots = A_cur[search_indices] @ dq
            best_local = int(np.argmax(dots))
            best_dot = float(dots[best_local])
            if best_dot <= 1e-12 and obs_indices:
                # Fall back to all planes (box planes span R^dim)
                dots_all = A_cur @ dq
                best_plane = int(np.argmax(dots_all))
                best_dot = float(dots_all[best_plane])
            else:
                best_plane = search_indices[best_local]
            if best_dot > 1e-12:
                eff_m = min(margin, 0.25 * best_dot)
                if eff_m < 1e-9:
                    eff_m = 0.25 * best_dot
                target_b = float(np.dot(A_cur[best_plane], q) - eff_m)
                if target_b < b_cur[best_plane]:
                    b_cur[best_plane] = target_b
        offsets = b_cur.tolist()

    if normals:
        A_mat = np.vstack(normals).astype(np.float64)
        b_vec = np.asarray(offsets, dtype=np.float64)
    else:
        A_mat = np.zeros((0, dim), dtype=np.float64)
        b_vec = np.zeros(0, dtype=np.float64)

    halfplane_barriers = [
        HalfPlaneBarrier(a=-A_mat[i], b=-float(b_vec[i]))
        for i in range(A_mat.shape[0])
    ]
    geofence_polytope: Tuple[Tuple[Tuple[float, ...], float], ...] = tuple(
        (tuple(float(v) for v in A_mat[i].tolist()), float(b_vec[i]))
        for i in range(A_mat.shape[0])
    )

    return {
        "A": A_mat,
        "b": b_vec,
        "halfplane_barriers": halfplane_barriers,
        "geofence_polytope": geofence_polytope,
    }


class TopologicalHomotopyGuide:
    """Free-space visibility roadmap + Betti-1 homotopy guidance for U-trap rescue."""

    def __init__(
        self,
        esdf_grid: VoxelESDFGrid,
        min_clearance: float = 0.1,
        persistence_merger: Optional[TopologicalPersistenceMerger] = None,
    ):
        self.esdf_grid = esdf_grid
        self.min_clearance = float(min_clearance)
        self.merger = (
            persistence_merger
            if persistence_merger is not None
            else TopologicalPersistenceMerger.get_instance()
        )

    def _is_segment_clear(self, p: np.ndarray, q: np.ndarray, clearance: Optional[float] = None) -> bool:
        clr = self.min_clearance if clearance is None else float(clearance)
        dist = float(np.linalg.norm(q - p))
        min_vs = float(np.min(self.esdf_grid.voxel_size))
        n_steps = max(2, int(math.ceil(dist / max(0.5 * min_vs, 1e-3))) + 1)
        ts = np.linspace(0.0, 1.0, n_steps)[:, np.newaxis]
        pts = (1.0 - ts) * p[np.newaxis, :] + ts * q[np.newaxis, :]
        dists = self.esdf_grid._query_distances_batch(pts)
        return bool(np.all(dists >= clr))

    def _build_roadmap(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        stride: Optional[int],
    ) -> Tuple[np.ndarray, List[List[Tuple[int, float]]]]:
        nx, ny, nz = self.esdf_grid.grid_shape
        if stride is None:
            s_xy = max(1, min(nx, ny) // 16)
        else:
            s_xy = max(1, int(stride))
        s_z = max(1, nz // 6) if nz > 6 else 1

        xs = list(range(0, nx, s_xy))
        if (nx - 1) not in xs:
            xs.append(nx - 1)
        ys = list(range(0, ny, s_xy))
        if (ny - 1) not in ys:
            ys.append(ny - 1)
        zs_set = set(range(0, nz, s_z))
        for p in (start, goal):
            iz = int(round((p[2] - self.esdf_grid.origin[2]) / self.esdf_grid.voxel_size[2] - 0.5))
            zs_set.add(int(np.clip(iz, 0, nz - 1)))
        zs = sorted(zs_set)

        cand_pts: List[np.ndarray] = [start.copy(), goal.copy()]
        grid_indices = np.array(
            [[ix, iy, iz] for ix in xs for iy in ys for iz in zs],
            dtype=np.float64,
        )
        world_pts = self.esdf_grid.origin + (grid_indices + 0.5) * self.esdf_grid.voxel_size
        sdfs = self.esdf_grid._query_distances_batch(world_pts)
        free_pts = world_pts[sdfs >= self.min_clearance]

        for pt in free_pts:
            if np.linalg.norm(pt - start) > 1e-4 and np.linalg.norm(pt - goal) > 1e-4:
                cand_pts.append(pt)

        nodes = np.vstack(cand_pts).astype(np.float64)
        n_nodes = nodes.shape[0]
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]

        step_m = max(
            s_xy * float(self.esdf_grid.voxel_size[0]),
            s_xy * float(self.esdf_grid.voxel_size[1]),
        )
        connect_r = max(2.6 * step_m, 3.0 * float(np.max(self.esdf_grid.voxel_size)))

        diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff * diff, axis=-1))

        added_edges: set[Tuple[int, int]] = set()

        def _try_add_edge(u: int, v: int) -> None:
            key = (u, v) if u < v else (v, u)
            if u == v or key in added_edges:
                return
            if self._is_segment_clear(nodes[u], nodes[v]):
                w = float(dist_mat[u, v])
                adj[u].append((v, w))
                adj[v].append((u, w))
                added_edges.add(key)

        # Connect endpoints (start=0, goal=1) generously to nearest nodes
        for ep in (0, 1):
            order = np.argsort(dist_mat[ep])
            for nbr in order[1 : min(n_nodes, 28)]:
                _try_add_edge(ep, int(nbr))

        # Connect sampled roadmap nodes within connect_r (capped per node)
        for u in range(2, n_nodes):
            nbrs = np.where((dist_mat[u] > 1e-9) & (dist_mat[u] <= connect_r))[0]
            if nbrs.shape[0] > 18:
                nbrs = nbrs[np.argsort(dist_mat[u, nbrs])[:18]]
            for v in nbrs:
                _try_add_edge(u, int(v))

        return nodes, adj

    def _dijkstra(
        self,
        nodes: np.ndarray,
        adj: List[List[Tuple[int, float]]],
        start_idx: int,
        goal_idx: int,
        blocked: Optional[set[int]] = None,
        node_penalty: Optional[np.ndarray] = None,
    ) -> Optional[List[int]]:
        n = nodes.shape[0]
        dist = np.full(n, math.inf, dtype=np.float64)
        parent = np.full(n, -1, dtype=np.int64)
        dist[start_idx] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, start_idx)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > dist[u]:
                continue
            if u == goal_idx:
                break
            for v, w in adj[u]:
                if blocked is not None and v in blocked:
                    continue
                extra = float(node_penalty[v]) if node_penalty is not None else 0.0
                nd = d_u + w + extra
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

        if not math.isfinite(dist[goal_idx]):
            return None
        path_idx: List[int] = []
        curr = goal_idx
        while curr != -1:
            path_idx.append(int(curr))
            curr = int(parent[curr])
        path_idx.reverse()
        return path_idx

    def _shortcut_path(self, pts: List[np.ndarray]) -> List[np.ndarray]:
        if len(pts) <= 2:
            return [p.copy() for p in pts]
        out = [pts[0].copy()]
        i = 0
        n = len(pts)
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if self._is_segment_clear(pts[i], pts[j]):
                    break
                j -= 1
            out.append(pts[j].copy())
            i = j
        return out

    @staticmethod
    def _resample_polyline(pts: Sequence[np.ndarray], n_samples: int) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64)
        if arr.shape[0] == 1:
            return np.repeat(arr, n_samples, axis=0)
        segs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(segs)])
        total = float(cum[-1])
        if total <= 1e-12:
            return np.repeat(arr[:1], n_samples, axis=0)
        targets = np.linspace(0.0, total, n_samples)
        out = np.empty((n_samples, arr.shape[1]), dtype=np.float64)
        for d in range(arr.shape[1]):
            out[:, d] = np.interp(targets, cum, arr[:, d])
        return out

    @staticmethod
    def _polyline_length(pts: Sequence[np.ndarray]) -> float:
        if len(pts) < 2:
            return 0.0
        arr = np.asarray(pts, dtype=np.float64)
        return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    def _are_paths_homotopic(self, path_a: Sequence[np.ndarray], path_b: Sequence[np.ndarray]) -> bool:
        n_s = 16
        ra = self._resample_polyline(path_a, n_s)
        rb = self._resample_polyline(path_b, n_s)
        t_vals = np.array([0.2, 0.35, 0.5, 0.65, 0.8], dtype=np.float64)
        test_pts: List[np.ndarray] = []
        for s_idx in range(1, n_s - 1):
            pa = ra[s_idx]
            pb = rb[s_idx]
            for t in t_vals:
                test_pts.append((1.0 - t) * pa + t * pb)
        if not test_pts:
            return True
        dists = self.esdf_grid._query_distances_batch(np.vstack(test_pts))
        return bool(np.all(dists >= self.min_clearance))

    def _compute_paths_betti1(self, paths: Sequence[List[np.ndarray]]) -> Dict[str, Any]:
        if not paths:
            return {"vertices": 0, "edges": 0, "beta_0": 0, "beta_1": 0, "persistence_energy": 0.0}

        lengths = [self._polyline_length(p) for p in paths]
        min_len = max(min(lengths), 1e-3)
        min_vs = float(np.min(self.esdf_grid.voxel_size))
        delta = max(min_vs, min(0.25 * min_len, min_len / 12.0))

        # Resample primary path at spacing delta
        n0 = max(4, int(math.ceil(lengths[0] / delta)) + 1)
        pts_list = [self._resample_polyline(paths[0], n0)]

        # Resample secondary branches slightly denser and filter near-duplicates so branch ends link cleanly
        for k in range(1, len(paths)):
            nk = max(5, int(math.ceil(lengths[k] / (0.8 * delta))) + 1)
            rk = self._resample_polyline(paths[k], nk)
            cur_cloud = np.vstack(pts_list)
            keep: List[np.ndarray] = []
            for pt in rk:
                d_min = float(np.min(np.linalg.norm(cur_cloud - pt, axis=1)))
                if d_min >= 0.45 * delta:
                    keep.append(pt)
            if keep:
                pts_list.append(np.vstack(keep))

        cloud = np.vstack(pts_list)
        return self.merger._compute_rips_persistence_betti1(cloud, threshold=1.3 * delta)

    def compute_homotopy_guidance(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        max_branches: int = 3,
        stride: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract collision-free homotopy branch waypoints and measure Betti-1 cycles."""
        s_pt = np.asarray(start, dtype=np.float64).flatten()
        g_pt = np.asarray(goal, dtype=np.float64).flatten()
        if s_pt.shape != (3,) or g_pt.shape != (3,):
            raise ValueError("start and goal must be 3D coordinates")

        nodes, adj = self._build_roadmap(s_pt, g_pt, stride=stride)
        n_edges = sum(len(nbrs) for nbrs in adj) // 2

        p0_idx = self._dijkstra(nodes, adj, 0, 1)
        if p0_idx is None:
            return {
                "betti_0": 2,
                "betti_1": 0,
                "has_topological_loop": False,
                "candidate_paths": [],
                "waypoints": [],
                "branches": [],
                "roadmap_vertices": int(nodes.shape[0]),
                "roadmap_edges": int(n_edges),
            }

        accepted_paths: List[List[np.ndarray]] = [self._shortcut_path([nodes[i] for i in p0_idx])]

        chord = g_pt - s_pt
        chord_len = float(np.linalg.norm(chord))
        if max_branches > 1 and chord_len > 1e-6:
            u_fwd = chord / chord_len
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(u_fwd, ref))) > 0.9:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            u_left = np.cross(ref, u_fwd)
            u_left /= float(np.linalg.norm(u_left))
            u_up = np.cross(u_fwd, u_left)
            u_up /= float(np.linalg.norm(u_up))

            # Lateral bias penalties against the primary path's excursion sides
            rel_nodes = nodes - s_pt
            p0_arr = self._resample_polyline(accepted_paths[0], 15)
            p0_rel = p0_arr - s_pt
            for axis in (u_left, -u_left, u_up, -u_up):
                if len(accepted_paths) >= max_branches:
                    break
                proj_p0 = p0_rel @ axis
                side_weight = float(np.mean(proj_p0))
                if side_weight < 1e-3:
                    continue
                proj_nodes = rel_nodes @ axis
                penalty = np.where(proj_nodes > 0.05 * chord_len, 20.0 * chord_len, 0.0)
                penalty[0] = 0.0
                penalty[1] = 0.0
                cand_idx = self._dijkstra(nodes, adj, 0, 1, node_penalty=penalty)
                if cand_idx is not None:
                    cand_path = self._shortcut_path([nodes[i] for i in cand_idx])
                    if all(not self._are_paths_homotopic(cand_path, acc) for acc in accepted_paths):
                        accepted_paths.append(cand_path)

            # Mid-path spherical blocking around interior excursion points of accepted paths
            for acc in list(accepted_paths):
                if len(accepted_paths) >= max_branches:
                    break
                r_acc = self._resample_polyline(acc, 13)
                for frac_idx in (6, 8, 4):
                    if len(accepted_paths) >= max_branches:
                        break
                    center = r_acc[frac_idx]
                    for r_frac in (0.30, 0.20, 0.42):
                        if len(accepted_paths) >= max_branches:
                            break
                        block_r = max(2.0 * float(np.max(self.esdf_grid.voxel_size)), r_frac * chord_len)
                        d_center = np.linalg.norm(nodes - center, axis=1)
                        blocked = {
                            int(i)
                            for i in np.where(d_center <= block_r)[0]
                            if i not in (0, 1)
                        }
                        cand_idx = self._dijkstra(nodes, adj, 0, 1, blocked=blocked)
                        if cand_idx is not None:
                            cand_path = self._shortcut_path([nodes[i] for i in cand_idx])
                            if all(not self._are_paths_homotopic(cand_path, prev) for prev in accepted_paths):
                                accepted_paths.append(cand_path)

        topo = self._compute_paths_betti1(accepted_paths)
        branches: List[Dict[str, Any]] = []
        for idx, path in enumerate(accepted_paths):
            dense = self._resample_polyline(path, 25)
            min_sdf = float(np.min(self.esdf_grid._query_distances_batch(dense)))
            branches.append(
                {
                    "branch_id": idx,
                    "waypoints": [pt.tolist() for pt in path],
                    "length": round(self._polyline_length(path), 6),
                    "min_sdf": round(min_sdf, 6),
                }
            )

        return {
            "betti_0": int(topo["beta_0"]),
            "betti_1": int(topo["beta_1"]),
            "has_topological_loop": bool(topo["beta_1"] >= 1),
            "candidate_paths": accepted_paths,
            "waypoints": [pt.tolist() for pt in accepted_paths[0]],
            "branches": branches,
            "roadmap_vertices": int(nodes.shape[0]),
            "roadmap_edges": int(n_edges),
        }
