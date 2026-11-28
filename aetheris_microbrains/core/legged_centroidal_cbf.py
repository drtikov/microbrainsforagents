"""Legged robotics centroidal Control Barrier Functions and GRF friction-pyramid allocator.

Provides pure-NumPy, deterministic safety filters and contact-force allocators for
bipeds and quadrupeds:

1. ``support_polygon_from_feet`` — 2D convex hull halfspaces ``A p - b >= 0`` with
   inward unit normals from stance foot positions (with optional sole expansion).
2. ``CapturePointDCMBarrier`` — 3D LIPM Divergent Component of Motion (Capture Point)
   ``xi = p_com_xy + v_com_xy / omega_0`` barrier against convex support-polygon edges.
3. ``SupportPolygonBalanceCBF`` — ``StateCBFSafetyFilter`` composing
   ``CapturePointDCMBarrier`` over ZMP/CoP control ``u = p_zmp`` with
   ``d(xi)/dt = omega_0 * (xi - p_zmp)`` and physical ZMP polygon containment.
4. ``GRFFrictionPyramidAllocator`` — multi-foot Ground Reaction Force solver using
   exact Dykstra cyclic projection in the wrench-tracking metric subject to
   unilateral normal forces, linearized Coulomb friction pyramids, normal caps,
   and optional rectangular foot CoP bounds.
5. ``SwingFootClearanceBarrier`` — relative-degree-2 vertical ground/obstacle
   clearance HOCBF ``h(x) = z_foot(q) - z_terrain(x_foot, y_foot) - c_min >= 0``
   with analytical URDF FK/Jacobian support.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .state_cbf_filter import (
    BarrierConstraint,
    LinearDynamics,
    StateCBFSafetyFilter,
    _fd_jacobian,
)
from .urdf_spatial_kinematics import URDFRobotModel, hat_so3


# ---------------------------------------------------------------------------
# 2D Convex Hull & Support Polygon Halfspace Geometry
# ---------------------------------------------------------------------------


def _convex_hull_2d_ccw(points: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull vertices in counter-clockwise order via Andrew's monotone chain."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Expected (N, >=2) points array, got shape {pts.shape}")
    pts2 = pts[:, :2]
    if pts2.shape[0] == 0:
        raise ValueError("Cannot construct support polygon from empty point set")

    # Deduplicate points within 1e-12 tolerance
    unique_pts: List[Tuple[float, float]] = []
    for row in sorted((float(p[0]), float(p[1])) for p in pts2):
        if not unique_pts or math.hypot(row[0] - unique_pts[-1][0], row[1] - unique_pts[-1][1]) > 1e-12:
            unique_pts.append(row)

    if len(unique_pts) < 3:
        raise ValueError(
            "Support polygon requires at least 3 non-collinear 2D points "
            "(pass foot_half_size to expand point feet into rectangular soles)"
        )

    def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in unique_pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(unique_pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("Foot positions are collinear; support polygon has zero area")

    hull_arr = np.asarray(hull, dtype=np.float64)
    # Signed shoelace area check
    x_c = hull_arr[:, 0]
    y_c = hull_arr[:, 1]
    area = 0.5 * float(np.sum(x_c * np.roll(y_c, -1) - np.roll(x_c, -1) * y_c))
    if area <= 1e-12:
        raise ValueError("Support polygon has degenerate zero area")
    return hull_arr


def _expand_foot_sole_points(
    foot_positions: Sequence[Sequence[float]],
    foot_half_size: Optional[Union[float, Tuple[float, float]]],
) -> np.ndarray:
    pts = np.asarray(foot_positions, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 2:
        raise ValueError(f"foot_positions must have at least 2 columns (x, y), got shape {pts.shape}")
    pts2 = pts[:, :2]
    if foot_half_size is None:
        return pts2

    if isinstance(foot_half_size, (int, float)):
        hx = hy = float(foot_half_size)
    else:
        arr = np.asarray(foot_half_size, dtype=np.float64).flatten()
        if arr.shape[0] != 2:
            raise ValueError(f"foot_half_size must be a scalar or 2-vector, got shape {arr.shape}")
        hx, hy = float(arr[0]), float(arr[1])
    if hx < 0.0 or hy < 0.0:
        raise ValueError("foot_half_size must be non-negative")
    if hx == 0.0 and hy == 0.0:
        return pts2

    offsets = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    expanded = (pts2[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    return expanded


def support_polygon_vertices(
    foot_positions: Sequence[Sequence[float]],
    foot_half_size: Optional[Union[float, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Return CCW convex-hull vertices ``(K, 2)`` of the 2D stance support polygon."""
    pts = _expand_foot_sole_points(foot_positions, foot_half_size)
    return _convex_hull_2d_ccw(pts)


def support_polygon_from_feet(
    foot_positions: Sequence[Sequence[float]],
    foot_half_size: Optional[Union[float, Tuple[float, float]]] = None,
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build inward-normal 2D halfspaces ``A p - b >= 0`` from stance foot positions.

    Each row ``a_k`` of ``A`` (shape ``(K, 2)``) is an inward unit normal to edge
    ``k`` of the CCW convex hull, and ``b[k] = a_k^T v_k + margin`` so strictly
    interior points satisfy ``A @ p - b > 0``.
    """
    hull = support_polygon_vertices(foot_positions, foot_half_size=foot_half_size)
    k_edges = hull.shape[0]
    A = np.zeros((k_edges, 2), dtype=np.float64)
    b = np.zeros(k_edges, dtype=np.float64)
    margin_f = float(margin)

    for i in range(k_edges):
        v0 = hull[i]
        v1 = hull[(i + 1) % k_edges]
        edge = v1 - v0
        length = float(np.linalg.norm(edge))
        if length <= 1e-15:
            continue
        inward_n = np.array([-edge[1] / length, edge[0] / length], dtype=np.float64)
        A[i] = inward_n
        b[i] = float(np.dot(inward_n, v0)) + margin_f
    return A, b


# ---------------------------------------------------------------------------
# 3D LIPM Capture Point (DCM) Barrier & Support-Polygon Balance CBF
# ---------------------------------------------------------------------------


class CapturePointDCMBarrier(BarrierConstraint):
    """3D LIPM Divergent Component of Motion (Capture Point) support-polygon barrier.

    Models ``xi = p_com_xy + (1 / omega_0) * v_com_xy`` with ``omega_0 = sqrt(g / z_com)``
    constrained inside convex 2D halfspaces ``a_k^T xi - b_k >= 0``.

    Supports:
    - 2D direct DCM state ``x = [xi_x, xi_y]``
    - 4D 2D-LIPM state ``x = [p_x, p_y, v_x, v_y]``
    - >=6D 3D-CoM / full centroidal state ``x = [p_x, p_y, p_z, v_x, v_y, v_z, ...]``
      (or custom ``pos_indices`` / ``vel_indices`` / ``z_index``).
    """

    def __init__(
        self,
        a: Sequence[float],
        b: Union[float, Sequence[float]],
        z_com: Optional[float] = 0.8,
        gravity: float = 9.81,
        alpha: float = 2.0,
        alpha2: float = 4.0,
        relative_degree: int = 1,
        hard: bool = True,
        z_min: float = 0.1,
        pos_indices: Optional[Tuple[int, int]] = None,
        vel_indices: Optional[Tuple[int, int]] = None,
        z_index: Optional[int] = None,
    ):
        a_arr = np.asarray(a, dtype=np.float64)
        if a_arr.ndim == 1:
            if a_arr.shape[0] != 2:
                raise ValueError(f"Halfspace normal 'a' must be 2D, got shape {a_arr.shape}")
            self._multi_edge = False
            self.a = a_arr.copy()
            self.b = float(b)
            self.A = a_arr.reshape(1, 2).copy()
            self.b_vec = np.array([self.b], dtype=np.float64)
        elif a_arr.ndim == 2 and a_arr.shape[1] == 2:
            b_arr = np.asarray(b, dtype=np.float64).flatten()
            if b_arr.shape[0] != a_arr.shape[0]:
                raise ValueError(
                    f"Mismatch between A rows ({a_arr.shape[0]}) and b length ({b_arr.shape[0]})"
                )
            self._multi_edge = True
            self.A = a_arr.copy()
            self.b_vec = b_arr.copy()
            self.a = a_arr[0].copy()
            self.b = float(b_arr[0])
        else:
            raise ValueError(f"Expected 'a' of shape (2,) or (K, 2), got {a_arr.shape}")

        if z_com is not None and float(z_com) <= 0.0:
            raise ValueError("z_com must be strictly positive")
        if float(gravity) <= 0.0:
            raise ValueError("gravity must be strictly positive")
        if float(z_min) <= 0.0:
            raise ValueError("z_min must be strictly positive")

        self.z_com = None if z_com is None else float(z_com)
        self.gravity = float(gravity)
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.z_min = float(z_min)
        self.pos_indices = pos_indices
        self.vel_indices = vel_indices
        self.z_index = z_index

    @classmethod
    def from_feet(
        cls,
        foot_positions: Sequence[Sequence[float]],
        z_com: Optional[float] = 0.8,
        gravity: float = 9.81,
        foot_half_size: Optional[Union[float, Tuple[float, float]]] = None,
        margin: float = 0.0,
        alpha: float = 2.0,
        alpha2: float = 4.0,
        relative_degree: int = 1,
        hard: bool = True,
        z_min: float = 0.1,
        pos_indices: Optional[Tuple[int, int]] = None,
        vel_indices: Optional[Tuple[int, int]] = None,
        z_index: Optional[int] = None,
    ) -> "CapturePointDCMBarrier":
        """Construct a multi-edge ``CapturePointDCMBarrier`` directly from stance foot positions."""
        A, b = support_polygon_from_feet(
            foot_positions, foot_half_size=foot_half_size, margin=margin
        )
        return cls(
            a=A,
            b=b,
            z_com=z_com,
            gravity=gravity,
            alpha=alpha,
            alpha2=alpha2,
            relative_degree=relative_degree,
            hard=hard,
            z_min=z_min,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            z_index=z_index,
        )

    def edge_barriers(self) -> List["CapturePointDCMBarrier"]:
        """Expand into one single-edge ``CapturePointDCMBarrier`` per polygon halfspace."""
        return [
            CapturePointDCMBarrier(
                a=self.A[k],
                b=float(self.b_vec[k]),
                z_com=self.z_com,
                gravity=self.gravity,
                alpha=self.alpha,
                alpha2=self.alpha2,
                relative_degree=self.relative_degree,
                hard=self.hard,
                z_min=self.z_min,
                pos_indices=self.pos_indices,
                vel_indices=self.vel_indices,
                z_index=self.z_index,
            )
            for k in range(self.A.shape[0])
        ]

    def _resolve_layout(
        self, x_arr: np.ndarray
    ) -> Tuple[
        str,
        Optional[Tuple[int, int]],
        Optional[Tuple[int, int]],
        Optional[int],
        float,
        float,
    ]:
        n = x_arr.shape[0]
        if self.pos_indices is not None or self.vel_indices is not None:
            if self.pos_indices is None or self.vel_indices is None:
                raise ValueError("Both pos_indices and vel_indices must be provided together")
            z_idx = self.z_index
            if self.z_com is not None:
                z_val = self.z_com
            elif z_idx is not None:
                z_val = max(self.z_min, float(x_arr[z_idx]))
            else:
                raise ValueError("z_com is None and z_index was not provided")
            return "pos_vel", self.pos_indices, self.vel_indices, z_idx, z_val, math.sqrt(self.gravity / z_val)

        if n == 2:
            z_val = 0.8 if self.z_com is None else self.z_com
            return "dcm", (0, 1), None, None, z_val, math.sqrt(self.gravity / z_val)
        if n == 4:
            z_val = 0.8 if self.z_com is None else self.z_com
            return "pos_vel", (0, 1), (2, 3), None, z_val, math.sqrt(self.gravity / z_val)
        if n >= 6:
            z_idx = 2 if self.z_index is None else self.z_index
            z_val = self.z_com if self.z_com is not None else max(self.z_min, float(x_arr[z_idx]))
            return "pos_vel", (0, 1), (3, 4), (None if self.z_com is not None else z_idx), z_val, math.sqrt(self.gravity / z_val)
        raise ValueError(f"Unsupported state length {n} for CapturePointDCMBarrier (expected 2, 4, or >=6)")

    def omega0(self, x: Optional[Sequence[float]] = None) -> float:
        """Return natural LIPM frequency ``omega_0 = sqrt(g / z_com)``."""
        if self.z_com is not None:
            return math.sqrt(self.gravity / self.z_com)
        if x is None:
            raise ValueError("State x is required to evaluate dynamic omega0 when z_com is None")
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        _, _, _, _, _, w0 = self._resolve_layout(x_arr)
        return w0

    def capture_point(self, x: Sequence[float]) -> np.ndarray:
        """Compute 2D Capture Point (DCM) ``xi = p_com_xy + v_com_xy / omega_0``."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        mode, p_idx, v_idx, _, _, w0 = self._resolve_layout(x_arr)
        assert p_idx is not None
        if mode == "dcm":
            return np.array([x_arr[p_idx[0]], x_arr[p_idx[1]]], dtype=np.float64)
        assert v_idx is not None
        p_xy = np.array([x_arr[p_idx[0]], x_arr[p_idx[1]]], dtype=np.float64)
        v_xy = np.array([x_arr[v_idx[0]], x_arr[v_idx[1]]], dtype=np.float64)
        return p_xy + v_xy / w0

    def h(self, x: np.ndarray) -> float:
        xi = self.capture_point(x)
        if not self._multi_edge:
            return float(np.dot(self.a, xi) - self.b)
        return float(np.min(self.A @ xi - self.b_vec))

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        mode, p_idx, v_idx, z_idx, z_val, w0 = self._resolve_layout(x_arr)
        assert p_idx is not None

        if self._multi_edge:
            xi = self.capture_point(x_arr)
            idx_min = int(np.argmin(self.A @ xi - self.b_vec))
            a_eff = self.A[idx_min]
        else:
            a_eff = self.a

        grad = np.zeros(n, dtype=np.float64)
        if mode == "dcm":
            grad[p_idx[0]] = a_eff[0]
            grad[p_idx[1]] = a_eff[1]
            return grad

        assert v_idx is not None
        grad[p_idx[0]] = a_eff[0]
        grad[p_idx[1]] = a_eff[1]
        grad[v_idx[0]] = a_eff[0] / w0
        grad[v_idx[1]] = a_eff[1] / w0

        if z_idx is not None and float(x_arr[z_idx]) > self.z_min:
            v_xy = np.array([x_arr[v_idx[0]], x_arr[v_idx[1]]], dtype=np.float64)
            grad[z_idx] += float(np.dot(a_eff, v_xy)) / (2.0 * w0 * z_val)
        return grad

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        _ = eps
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        H = np.zeros((n, n), dtype=np.float64)
        mode, _, v_idx, z_idx, z_val, w0 = self._resolve_layout(x_arr)
        if mode == "dcm" or z_idx is None or float(x_arr[z_idx]) <= self.z_min:
            return H

        assert v_idx is not None
        if self._multi_edge:
            xi = self.capture_point(x_arr)
            a_eff = self.A[int(np.argmin(self.A @ xi - self.b_vec))]
        else:
            a_eff = self.a

        cross_coeff = 1.0 / (2.0 * w0 * z_val)
        for dim_k in range(2):
            c_val = a_eff[dim_k] * cross_coeff
            H[z_idx, v_idx[dim_k]] = c_val
            H[v_idx[dim_k], z_idx] = c_val

        v_xy = np.array([x_arr[v_idx[0]], x_arr[v_idx[1]]], dtype=np.float64)
        H[z_idx, z_idx] = -float(np.dot(a_eff, v_xy)) / (4.0 * w0 * z_val * z_val)
        return H

    def inflate(self, delta: float) -> "CapturePointDCMBarrier":
        if self._multi_edge:
            return CapturePointDCMBarrier(
                a=self.A,
                b=self.b_vec + float(delta),
                z_com=self.z_com,
                gravity=self.gravity,
                alpha=self.alpha,
                alpha2=self.alpha2,
                relative_degree=self.relative_degree,
                hard=self.hard,
                z_min=self.z_min,
                pos_indices=self.pos_indices,
                vel_indices=self.vel_indices,
                z_index=self.z_index,
            )
        return CapturePointDCMBarrier(
            a=self.a,
            b=self.b + float(delta),
            z_com=self.z_com,
            gravity=self.gravity,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            z_min=self.z_min,
            pos_indices=self.pos_indices,
            vel_indices=self.vel_indices,
            z_index=self.z_index,
        )


def _build_lipm_dynamics(state_mode: str, omega0: float) -> LinearDynamics:
    """Construct exact control-affine LIPM/DCM dynamics over ZMP control ``u = p_zmp``."""
    mode = state_mode.lower()
    if mode == "dcm":
        A_dyn = omega0 * np.eye(2, dtype=np.float64)
        B_dyn = -omega0 * np.eye(2, dtype=np.float64)
        return LinearDynamics(A_dyn, B_dyn)
    if mode == "lipm_2d":
        w2 = omega0 * omega0
        A_dyn = np.zeros((4, 4), dtype=np.float64)
        A_dyn[0:2, 2:4] = np.eye(2, dtype=np.float64)
        A_dyn[2:4, 0:2] = w2 * np.eye(2, dtype=np.float64)
        B_dyn = np.zeros((4, 2), dtype=np.float64)
        B_dyn[2:4, :] = -w2 * np.eye(2, dtype=np.float64)
        return LinearDynamics(A_dyn, B_dyn)
    if mode == "lipm_3d":
        w2 = omega0 * omega0
        A_dyn = np.zeros((6, 6), dtype=np.float64)
        A_dyn[0:3, 3:6] = np.eye(3, dtype=np.float64)
        A_dyn[3:5, 0:2] = w2 * np.eye(2, dtype=np.float64)
        B_dyn = np.zeros((6, 2), dtype=np.float64)
        B_dyn[3:5, :] = -w2 * np.eye(2, dtype=np.float64)
        return LinearDynamics(A_dyn, B_dyn)
    raise ValueError(f"Unsupported state_mode {state_mode!r}; expected 'dcm', 'lipm_2d', or 'lipm_3d'")


class SupportPolygonBalanceCBF(StateCBFSafetyFilter):
    """Exact-projection CBF safety filter keeping the 3D LIPM Capture Point inside the support polygon.

    Composes one ``CapturePointDCMBarrier`` per support-polygon edge into
    ``StateCBFSafetyFilter`` over ZMP/CoP control ``u = p_zmp in R^2`` under
    ``d(xi)/dt = omega_0 * (xi - p_zmp)``, and optionally enforces physical ZMP
    containment ``A_poly u >= b_poly``.
    """

    def __init__(
        self,
        A_poly: Optional[Sequence[Sequence[float]]] = None,
        b_poly: Optional[Sequence[float]] = None,
        foot_positions: Optional[Sequence[Sequence[float]]] = None,
        foot_half_size: Optional[Union[float, Tuple[float, float]]] = None,
        z_com: float = 0.8,
        gravity: float = 9.81,
        alpha: float = 2.0,
        state_mode: str = "dcm",
        enforce_zmp_in_polygon: bool = True,
        margin: float = 0.0,
        conformal_margin: float = 0.0,
        u_min: Optional[Sequence[float]] = None,
        u_max: Optional[Sequence[float]] = None,
        max_iters: int = 250,
        tol: float = 1e-9,
        backup_controller: Optional[Any] = None,
    ):
        if foot_positions is not None:
            A_arr, b_arr = support_polygon_from_feet(
                foot_positions, foot_half_size=foot_half_size, margin=margin
            )
            if u_min is None or u_max is None:
                verts = support_polygon_vertices(foot_positions, foot_half_size=foot_half_size)
                if u_min is None:
                    u_min = np.min(verts, axis=0)
                if u_max is None:
                    u_max = np.max(verts, axis=0)
        elif A_poly is not None and b_poly is not None:
            A_arr = np.asarray(A_poly, dtype=np.float64)
            b_arr = np.asarray(b_poly, dtype=np.float64).flatten() + float(margin)
            if A_arr.ndim != 2 or A_arr.shape[1] != 2 or b_arr.shape[0] != A_arr.shape[0]:
                raise ValueError("A_poly must have shape (K, 2) matching b_poly length K")
        else:
            raise ValueError("Provide either (A_poly, b_poly) or foot_positions")

        if float(z_com) <= 0.0 or float(gravity) <= 0.0:
            raise ValueError("z_com and gravity must be strictly positive")

        self.z_com = float(z_com)
        self.gravity = float(gravity)
        self.omega0 = math.sqrt(self.gravity / self.z_com)
        self.state_mode = state_mode.lower()
        self.enforce_zmp_in_polygon = bool(enforce_zmp_in_polygon)

        # Normalize rows so b_poly / conformal_margin represent Euclidean meters
        norms = np.linalg.norm(A_arr, axis=1)
        valid_rows = norms > 1e-12
        A_norm = A_arr[valid_rows] / norms[valid_rows, None]
        b_norm = b_arr[valid_rows] / norms[valid_rows]

        self.A_poly = A_norm
        self.b_poly = b_norm
        self._b_poly_zmp = b_norm - float(margin)

        dynamics = _build_lipm_dynamics(self.state_mode, self.omega0)
        barriers = [
            CapturePointDCMBarrier(
                a=self.A_poly[k],
                b=float(self.b_poly[k]),
                z_com=self.z_com,
                gravity=self.gravity,
                alpha=alpha,
                relative_degree=1,
                hard=True,
            )
            for k in range(self.A_poly.shape[0])
        ]

        super().__init__(
            dynamics=dynamics,
            barriers=barriers,
            u_min=u_min,
            u_max=u_max,
            conformal_margin=conformal_margin,
            max_iters=max_iters,
            tol=tol,
            backup_controller=backup_controller,
        )

    @classmethod
    def from_feet(
        cls,
        foot_positions: Sequence[Sequence[float]],
        foot_half_size: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> "SupportPolygonBalanceCBF":
        """Construct ``SupportPolygonBalanceCBF`` directly from stance foot coordinates."""
        return cls(foot_positions=foot_positions, foot_half_size=foot_half_size, **kwargs)

    def _halfspaces_with_flags(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[bool]]:
        A_cbf, b_cbf, alphas, hard_flags = super()._halfspaces_with_flags(x)
        if not self.enforce_zmp_in_polygon or self.A_poly.shape[0] == 0:
            return A_cbf, b_cbf, alphas, hard_flags
        A_all = np.vstack([A_cbf, self.A_poly]) if A_cbf.size else self.A_poly.copy()
        b_all = np.concatenate([b_cbf, self._b_poly_zmp]) if b_cbf.size else self._b_poly_zmp.copy()
        alphas_all = list(alphas) + [0.0] * self.A_poly.shape[0]
        hard_all = list(hard_flags) + [True] * self.A_poly.shape[0]
        return A_all, b_all, alphas_all, hard_all

    def capture_point(self, x: Sequence[float]) -> np.ndarray:
        """Extract 2D Capture Point ``xi`` from state ``x``."""
        return self.barriers[0].capture_point(x)  # type: ignore[attr-defined]

    def polygon_margin(self, point_xy: Sequence[float]) -> float:
        """Signed minimum distance to support-polygon edges (positive strictly inside)."""
        pt = np.asarray(point_xy, dtype=np.float64).flatten()[:2]
        return float(np.min(self.A_poly @ pt - self.b_poly))

    def is_inside_polygon(self, point_xy: Sequence[float], tol: float = 0.0) -> bool:
        """Return True if 2D point lies inside the support polygon within ``tol``."""
        return self.polygon_margin(point_xy) >= -float(tol)

    def step_dynamics(
        self,
        x: Sequence[float],
        p_zmp: Sequence[float],
        dt: float,
        perturbation: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Advance LIPM/DCM state one Euler step under ZMP command and optional push perturbation."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        u_arr = np.asarray(p_zmp, dtype=np.float64).flatten()
        x_next = x_arr + float(dt) * (self.dynamics.f(x_arr) + self.dynamics.g(x_arr) @ u_arr)
        if perturbation is not None:
            x_next = x_next + float(dt) * np.asarray(perturbation, dtype=np.float64).flatten()
        return x_next


# ---------------------------------------------------------------------------
# Multi-Foot Ground Reaction Force (GRF) Friction-Pyramid Allocator
# ---------------------------------------------------------------------------


def _normalize_cop_bounds(
    cop_bounds: Optional[Union[float, Sequence[float]]],
) -> Optional[Tuple[float, float, float, float]]:
    if cop_bounds is None:
        return None
    if isinstance(cop_bounds, (int, float)):
        s = abs(float(cop_bounds))
        return (-s, s, -s, s)
    arr = np.asarray(cop_bounds, dtype=np.float64).flatten()
    if arr.shape[0] == 2:
        lx, ly = abs(float(arr[0])), abs(float(arr[1]))
        return (-lx, lx, -ly, ly)
    if arr.shape[0] == 4:
        x_min, x_max, y_min, y_max = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        if x_min > x_max or y_min > y_max:
            raise ValueError("Invalid cop_bounds: min exceeds max")
        return (x_min, x_max, y_min, y_max)
    raise ValueError(f"cop_bounds must be scalar, 2-tuple, or 4-tuple; got shape {arr.shape}")


class GRFFrictionPyramidAllocator:
    """Multi-foot Ground Reaction Force solver using exact Dykstra cyclic projection.

    Allocates contact forces ``f_c = [f_{c,x}, f_{c,y}, f_{c,z}]^T`` (and optional
    local foot moments ``[tau_{c,x}, tau_{c,y}]`` when ``cop_bounds`` is given)
    across ``N_c`` active stance feet to track a desired centroidal wrench
    ``(F_des, tau_des)`` about ``p_com`` subject to:
      - Unilateral normal force: ``f_{c,z} >= f_min > 0``
      - Maximum normal force:    ``f_{c,z} <= f_max``
      - Coulomb friction cone:   ``|f_{c,x}| <= mu * f_{c,z}``, ``|f_{c,y}| <= mu * f_{c,z}``
      - Optional foot CoP box:   ``x_min f_{c,z} <= -tau_{c,y} <= x_max f_{c,z}``,
                                 ``y_min f_{c,z} <=  tau_{c,x} <= y_max f_{c,z}``.
    """

    def __init__(
        self,
        mu: float = 0.6,
        f_min: float = 5.0,
        f_max: float = 1000.0,
        cop_bounds: Optional[Union[float, Sequence[float]]] = None,
        wrench_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        reg: float = 0.05,
        max_iters: int = 1000,
        max_outer_iters: int = 40,
        tol: float = 1e-8,
    ):
        if float(mu) <= 0.0:
            raise ValueError("Friction coefficient mu must be strictly positive")
        if float(f_min) <= 0.0:
            raise ValueError("Minimum unilateral normal force f_min must be strictly positive")
        if float(f_max) < float(f_min):
            raise ValueError("f_max must be >= f_min")
        if float(reg) <= 0.0:
            raise ValueError("Regularization reg must be strictly positive")

        w_arr = np.asarray(wrench_weights, dtype=np.float64).flatten()
        if w_arr.shape[0] != 6 or np.any(w_arr <= 0.0):
            raise ValueError("wrench_weights must be a 6-vector of strictly positive weights")

        self.mu = float(mu)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.cop_bounds = _normalize_cop_bounds(cop_bounds)
        self.wrench_weights = w_arr
        self.reg = float(reg)
        self.max_iters = int(max_iters)
        self.max_outer_iters = int(max_outer_iters)
        self.tol = float(tol)

    @staticmethod
    def _dykstra_halfspaces(
        A_norm: np.ndarray,
        b_norm: np.ndarray,
        z_target: np.ndarray,
        max_iters: int,
        tol: float,
    ) -> Tuple[np.ndarray, int]:
        """Exact Dykstra cyclic projection onto ``{z : A_norm z >= b_norm}`` with unit-norm rows."""
        m = A_norm.shape[0]
        if m == 0:
            return z_target.copy(), 0
        s = np.zeros((m, z_target.shape[0]), dtype=np.float64)
        w = z_target.copy()
        iters = 0
        for it in range(1, max_iters + 1):
            iters = it
            max_step = 0.0
            for j in range(m):
                y = w + s[j]
                a_j = A_norm[j]
                viol = b_norm[j] - float(np.dot(a_j, y))
                if viol > 0.0:
                    proj = y + viol * a_j
                else:
                    proj = y
                s_new = y - proj
                diff = float(np.max(np.abs(s_new - s[j])))
                if diff > max_step:
                    max_step = diff
                s[j] = s_new
                w = proj
            if max_step <= tol:
                break
        return w, iters

    def allocate(
        self,
        foot_positions: Sequence[Sequence[float]],
        com_position: Sequence[float],
        desired_force: Sequence[float],
        desired_torque: Sequence[float] = (0.0, 0.0, 0.0),
        active_mask: Optional[Sequence[bool]] = None,
        mu: Optional[float] = None,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        cop_bounds: Optional[Union[float, Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """Allocate contact forces across active stance feet via exact Dykstra projection."""
        feet_raw = np.asarray(foot_positions, dtype=np.float64)
        if feet_raw.ndim != 2 or feet_raw.shape[1] not in (2, 3):
            raise ValueError(f"foot_positions must have shape (N_c, 2 or 3), got {feet_raw.shape}")
        n_feet = feet_raw.shape[0]
        if feet_raw.shape[1] == 2:
            feet = np.column_stack([feet_raw, np.zeros(n_feet, dtype=np.float64)])
        else:
            feet = feet_raw

        com = np.asarray(com_position, dtype=np.float64).flatten()
        f_des = np.asarray(desired_force, dtype=np.float64).flatten()
        tau_des = np.asarray(desired_torque, dtype=np.float64).flatten()
        if com.shape[0] != 3 or f_des.shape[0] != 3 or tau_des.shape[0] != 3:
            raise ValueError("com_position, desired_force, and desired_torque must be 3-vectors")

        mu_eff = self.mu if mu is None else float(mu)
        f_min_eff = self.f_min if f_min is None else float(f_min)
        f_max_eff = self.f_max if f_max is None else float(f_max)
        cop_eff = self.cop_bounds if cop_bounds is None else _normalize_cop_bounds(cop_bounds)

        if mu_eff <= 0.0 or f_min_eff <= 0.0 or f_max_eff < f_min_eff:
            raise ValueError("Invalid contact cone parameters (require mu > 0, 0 < f_min <= f_max)")

        if active_mask is None:
            mask = np.ones(n_feet, dtype=bool)
        else:
            mask = np.asarray(active_mask, dtype=bool).flatten()
            if mask.shape[0] != n_feet:
                raise ValueError(f"active_mask length {mask.shape[0]} != n_feet {n_feet}")

        active_indices = np.flatnonzero(mask)
        n_act = int(active_indices.shape[0])
        if n_act == 0:
            w_des = np.concatenate([f_des, tau_des])
            return {
                "forces": np.zeros((n_feet, 3), dtype=np.float64),
                "local_torques": np.zeros((n_feet, 2), dtype=np.float64),
                "net_force": np.zeros(3, dtype=np.float64),
                "net_torque": np.zeros(3, dtype=np.float64),
                "wrench_error": float(np.linalg.norm(w_des)),
                "feasible": False,
                "max_violation": float("inf"),
                "iterations": 0,
            }

        has_cop = cop_eff is not None
        d_foot = 5 if has_cop else 3
        n_vars = n_act * d_foot

        A_grf = np.zeros((6, n_vars), dtype=np.float64)
        ineq_rows: List[np.ndarray] = []
        ineq_rhs: List[float] = []

        for c_idx, foot_i in enumerate(active_indices):
            col0 = c_idx * d_foot
            r_arm = feet[foot_i] - com
            A_grf[0:3, col0 : col0 + 3] = np.eye(3, dtype=np.float64)
            A_grf[3:6, col0 : col0 + 3] = hat_so3(r_arm)
            if has_cop:
                A_grf[3, col0 + 3] = 1.0
                A_grf[4, col0 + 4] = 1.0

            # Normal force unilateral and upper cap
            row_fmin = np.zeros(n_vars, dtype=np.float64)
            row_fmin[col0 + 2] = 1.0
            ineq_rows.append(row_fmin)
            ineq_rhs.append(f_min_eff)

            if math.isfinite(f_max_eff):
                row_fmax = np.zeros(n_vars, dtype=np.float64)
                row_fmax[col0 + 2] = -1.0
                ineq_rows.append(row_fmax)
                ineq_rhs.append(-f_max_eff)

            # Linearized Coulomb friction pyramid: |fx| <= mu * fz, |fy| <= mu * fz
            for axis_k, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)):
                row = np.zeros(n_vars, dtype=np.float64)
                row[col0 + axis_k] = sign
                row[col0 + 2] = mu_eff
                ineq_rows.append(row)
                ineq_rhs.append(0.0)

            # Optional rectangular CoP bounds: x_min <= -tau_y/fz <= x_max, y_min <= tau_x/fz <= y_max
            if cop_eff is not None:
                x_min_c, x_max_c, y_min_c, y_max_c = cop_eff
                r1 = np.zeros(n_vars, dtype=np.float64)
                r1[col0 + 4] = 1.0
                r1[col0 + 2] = x_max_c
                ineq_rows.append(r1)
                ineq_rhs.append(0.0)

                r2 = np.zeros(n_vars, dtype=np.float64)
                r2[col0 + 4] = -1.0
                r2[col0 + 2] = -x_min_c
                ineq_rows.append(r2)
                ineq_rhs.append(0.0)

                r3 = np.zeros(n_vars, dtype=np.float64)
                r3[col0 + 3] = 1.0
                r3[col0 + 2] = -y_min_c
                ineq_rows.append(r3)
                ineq_rhs.append(0.0)

                r4 = np.zeros(n_vars, dtype=np.float64)
                r4[col0 + 3] = -1.0
                r4[col0 + 2] = y_max_c
                ineq_rows.append(r4)
                ineq_rhs.append(0.0)

        C_ineq = np.vstack(ineq_rows)
        d_ineq = np.asarray(ineq_rhs, dtype=np.float64)

        w_des = np.concatenate([f_des, tau_des])
        W = np.diag(self.wrench_weights)
        rho = max(0.02, self.reg)
        Q = (A_grf.T @ W @ A_grf) + rho * np.eye(n_vars, dtype=np.float64)
        Q = 0.5 * (Q + Q.T)

        f_cur = np.zeros(n_vars, dtype=np.float64)
        fz_share = float(np.clip(f_des[2] / n_act, f_min_eff, f_max_eff))
        for c_idx in range(n_act):
            f_cur[c_idx * d_foot + 2] = fz_share

        L = np.linalg.cholesky(Q)
        AtW_w = A_grf.T @ (W @ w_des)

        # Map C_ineq f >= d_ineq into well-conditioned z-space: (L^{-1} C_ineq^T)^T z >= d_ineq
        A_z = np.linalg.solve(L, C_ineq.T).T
        row_norms = np.linalg.norm(A_z, axis=1)
        keep = row_norms > 1e-14
        A_z_norm = A_z[keep] / row_norms[keep, None]
        d_z_norm = d_ineq[keep] / row_norms[keep]

        # Proximal-point outer loop with exact Dykstra cyclic projection in Q-metric
        # eliminates regularization bias while keeping condition number small.
        n_iters = 0
        f_sol = f_cur.copy()
        for _outer in range(self.max_outer_iters):
            rhs = AtW_w + rho * f_cur
            f_unc = np.linalg.solve(Q, rhs)
            z_unc = L.T @ f_unc
            z_sol, it_inner = self._dykstra_halfspaces(
                A_z_norm, d_z_norm, z_unc, max_iters=self.max_iters, tol=self.tol
            )
            n_iters += it_inner
            f_sol = np.linalg.solve(L.T, z_sol)
            step_norm = float(np.linalg.norm(f_sol - f_cur))
            f_cur = f_sol
            if step_norm <= 1e-7:
                break

        violations = d_ineq - (C_ineq @ f_sol)
        max_violation = max(0.0, float(np.max(violations)))
        feasible = bool(np.all(np.isfinite(f_sol)) and max_violation <= 1e-5)

        forces_full = np.zeros((n_feet, 3), dtype=np.float64)
        local_torques_full = np.zeros((n_feet, 2), dtype=np.float64)
        for c_idx, foot_i in enumerate(active_indices):
            col0 = c_idx * d_foot
            forces_full[foot_i] = f_sol[col0 : col0 + 3]
            if has_cop:
                local_torques_full[foot_i] = f_sol[col0 + 3 : col0 + 5]

        achieved_wrench = A_grf @ f_sol
        net_force = achieved_wrench[:3]
        net_torque = achieved_wrench[3:6]
        wrench_error = float(np.linalg.norm(achieved_wrench - w_des))

        return {
            "forces": forces_full,
            "local_torques": local_torques_full,
            "net_force": net_force,
            "net_torque": net_torque,
            "wrench_error": wrench_error,
            "feasible": feasible,
            "max_violation": max_violation,
            "iterations": n_iters,
        }

    solve = allocate


# ---------------------------------------------------------------------------
# Swing-Foot Ground & Obstacle Clearance Barrier (HOCBF)
# ---------------------------------------------------------------------------


class SwingFootClearanceBarrier(BarrierConstraint):
    """Vertical swing-foot ground/obstacle clearance barrier (relative-degree-2 HOCBF).

    Enforces ``h(x) = z_foot(q) - z_terrain(x_foot(q), y_foot(q)) - c_min >= 0``
    using analytical URDF FK/Jacobians (or user-supplied FK/Jacobian callables).
    """

    def __init__(
        self,
        robot: Optional[URDFRobotModel] = None,
        foot_link: Optional[str] = None,
        local_point: Sequence[float] = (0.0, 0.0, 0.0),
        foot_fk_fn: Optional[Callable[[np.ndarray], Sequence[float]]] = None,
        foot_jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        terrain_height_fn: Union[float, Callable[[float, float], float]] = 0.0,
        terrain_grad_fn: Optional[Callable[[float, float], Tuple[float, float]]] = None,
        c_min: float = 0.0,
        n_dof: Optional[int] = None,
        alpha: float = 2.0,
        alpha2: float = 4.0,
        relative_degree: int = 2,
        hard: bool = True,
        eps: float = 1e-5,
    ):
        if robot is None and foot_fk_fn is None:
            raise ValueError("Provide either a URDFRobotModel ('robot') or 'foot_fk_fn'")

        self.robot = robot
        self.foot_link = (
            robot.end_effector_link_name
            if (robot is not None and foot_link is None)
            else foot_link
        )
        self.local_point = np.asarray(local_point, dtype=np.float64).flatten()
        if self.local_point.shape[0] != 3:
            raise ValueError(f"local_point must be 3D, got shape {self.local_point.shape}")
        self.foot_fk_fn = foot_fk_fn
        self.foot_jacobian_fn = foot_jacobian_fn
        self.terrain_height_fn = terrain_height_fn
        self.terrain_grad_fn = terrain_grad_fn
        self.c_min = float(c_min)
        if n_dof is not None:
            self.n_dof: Optional[int] = int(n_dof)
        elif robot is not None:
            self.n_dof = robot.n_dof
        else:
            self.n_dof = None
        self.alpha = float(alpha)
        self.alpha1 = self.alpha
        self.alpha2 = float(alpha2)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.eps = float(eps)

    def _split_q(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if self.n_dof is not None:
            q_len = min(x_arr.shape[0], self.n_dof)
        elif self.relative_degree == 2 and x_arr.shape[0] >= 2 and x_arr.shape[0] % 2 == 0:
            q_len = x_arr.shape[0] // 2
        else:
            q_len = x_arr.shape[0]
        return x_arr[:q_len], q_len

    def _eval_foot_pos(self, q: np.ndarray) -> np.ndarray:
        if self.foot_fk_fn is not None:
            p = np.asarray(self.foot_fk_fn(q), dtype=np.float64).flatten()
        else:
            assert self.robot is not None and self.foot_link is not None
            p, _ = self.robot.link_point_jacobian(q, self.foot_link, self.local_point)
        if p.shape[0] != 3:
            raise ValueError(f"Foot FK must return 3D position, got shape {p.shape}")
        return p

    def _eval_foot_pos_and_jac(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q_len = q.shape[0]
        if self.robot is not None and self.foot_fk_fn is None:
            assert self.foot_link is not None
            return self.robot.link_point_jacobian(q, self.foot_link, self.local_point)
        p = self._eval_foot_pos(q)
        if self.foot_jacobian_fn is not None:
            J = np.asarray(self.foot_jacobian_fn(q), dtype=np.float64).reshape(3, q_len)
        else:
            J = _fd_jacobian(self._eval_foot_pos, q, eps=self.eps)
        return p, J

    def _eval_terrain_height(self, px: float, py: float) -> float:
        if callable(self.terrain_height_fn):
            return float(self.terrain_height_fn(px, py))
        return float(self.terrain_height_fn)

    def _eval_terrain_grad(self, px: float, py: float) -> Tuple[float, float]:
        if not callable(self.terrain_height_fn):
            return 0.0, 0.0
        if self.terrain_grad_fn is not None:
            gx, gy = self.terrain_grad_fn(px, py)
            return float(gx), float(gy)
        h_eps = self.eps
        dz_dx = (
            float(self.terrain_height_fn(px + h_eps, py))
            - float(self.terrain_height_fn(px - h_eps, py))
        ) / (2.0 * h_eps)
        dz_dy = (
            float(self.terrain_height_fn(px, py + h_eps))
            - float(self.terrain_height_fn(px, py - h_eps))
        ) / (2.0 * h_eps)
        return dz_dx, dz_dy

    def h(self, x: np.ndarray) -> float:
        q, _ = self._split_q(x)
        p = self._eval_foot_pos(q)
        z_t = self._eval_terrain_height(float(p[0]), float(p[1]))
        return float(p[2] - z_t - self.c_min)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        q, q_len = self._split_q(x_arr)
        p, J_lin = self._eval_foot_pos_and_jac(q)
        dz_dx, dz_dy = self._eval_terrain_grad(float(p[0]), float(p[1]))
        n_terrain = np.array([-dz_dx, -dz_dy, 1.0], dtype=np.float64)
        grad = np.zeros(n, dtype=np.float64)
        grad[:q_len] = n_terrain @ J_lin
        return grad

    def hess_h(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = x_arr.shape[0]
        q, q_len = self._split_q(x_arr)
        H_q = np.zeros((q_len, q_len), dtype=np.float64)
        for i in range(q_len):
            dq = np.zeros(q_len, dtype=np.float64)
            dq[i] = eps
            gp = self.grad_h(q + dq)[:q_len]
            gm = self.grad_h(q - dq)[:q_len]
            H_q[:, i] = (gp - gm) / (2.0 * eps)
        H_q = 0.5 * (H_q + H_q.T)
        if q_len == n:
            return H_q
        out = np.zeros((n, n), dtype=np.float64)
        out[:q_len, :q_len] = H_q
        return out

    def inflate(self, delta: float) -> "SwingFootClearanceBarrier":
        return SwingFootClearanceBarrier(
            robot=self.robot,
            foot_link=self.foot_link,
            local_point=self.local_point,
            foot_fk_fn=self.foot_fk_fn,
            foot_jacobian_fn=self.foot_jacobian_fn,
            terrain_height_fn=self.terrain_height_fn,
            terrain_grad_fn=self.terrain_grad_fn,
            c_min=self.c_min + float(delta),
            n_dof=self.n_dof,
            alpha=self.alpha,
            alpha2=self.alpha2,
            relative_degree=self.relative_degree,
            hard=self.hard,
            eps=self.eps,
        )
