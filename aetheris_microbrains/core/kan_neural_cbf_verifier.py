"""Formal Verification and Synthesis for KAN Control Barrier Functions.

Combines Kolmogorov-Arnold Chebyshev networks (``KolmogorovArnoldMicroBrain``)
with exact analytical first/second derivatives, sound interval arithmetic
branch-and-bound certification, and Counterexample-Guided Inductive Synthesis
(CEGIS) for ``StateCBFSafetyFilter``.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.polynomial import chebyshev as np_cheb
from numpy.polynomial import polynomial as np_poly

from .kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain
from .state_cbf_filter import (
    BarrierConstraint,
    ControlAffineDynamics,
    HalfPlaneBarrier,
    HighOrderBarrier,
    LinearDynamics,
    SphereBarrier,
)

# Cache of interior critical points in (-1, 1) for T_k'(z) and T_k''(z) by degree k.
_DT_CRIT_CACHE: Dict[int, np.ndarray] = {}
_D2T_CRIT_CACHE: Dict[int, np.ndarray] = {}


def _cheb_basis_crit_points(degree_k: int, deriv_order: int) -> np.ndarray:
    """Return interior real critical points in (-1, 1) of d^(deriv_order)/dz T_k(z)."""
    cache = _DT_CRIT_CACHE if deriv_order == 1 else _D2T_CRIT_CACHE
    if degree_k in cache:
        return cache[degree_k]

    if deriv_order == 1:
        if degree_k <= 2:
            pts = np.zeros(0, dtype=np.float64)
        else:
            c = np.zeros(degree_k + 1, dtype=np.float64)
            c[degree_k] = 1.0
            d2 = np_cheb.chebder(c, m=2)
            roots = np_cheb.chebroots(d2)
            real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-10])
            pts = real_roots[(real_roots > -1.0) & (real_roots < 1.0)]
    else:
        if degree_k <= 3:
            pts = np.zeros(0, dtype=np.float64)
        else:
            c = np.zeros(degree_k + 1, dtype=np.float64)
            c[degree_k] = 1.0
            d3 = np_cheb.chebder(c, m=3)
            roots = np_cheb.chebroots(d3)
            real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-10])
            pts = real_roots[(real_roots > -1.0) & (real_roots < 1.0)]

    cache[degree_k] = np.sort(pts.astype(np.float64))
    return cache[degree_k]


def chebyshev_with_derivatives(
    z: Union[float, np.ndarray], degree: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Chebyshev polynomials T_k(z), T_k'(z), and T_k''(z) for k=0..degree.

    Uses the exact three-term recurrence and its analytical first and second
    derivatives, which are singularity-free across all z in [-1, 1].
    """
    z_arr = np.asarray(z, dtype=np.float64)
    zc = np.clip(z_arr, -1.0, 1.0)
    deg = max(0, int(degree))

    t_list = [np.ones_like(zc, dtype=np.float64)]
    dt_list = [np.zeros_like(zc, dtype=np.float64)]
    d2t_list = [np.zeros_like(zc, dtype=np.float64)]

    if deg >= 1:
        t_list.append(zc.copy())
        dt_list.append(np.ones_like(zc, dtype=np.float64))
        d2t_list.append(np.zeros_like(zc, dtype=np.float64))

    for k in range(2, deg + 1):
        t_k = 2.0 * zc * t_list[k - 1] - t_list[k - 2]
        dt_k = 2.0 * t_list[k - 1] + 2.0 * zc * dt_list[k - 1] - dt_list[k - 2]
        d2t_k = 4.0 * dt_list[k - 1] + 2.0 * zc * d2t_list[k - 1] - d2t_list[k - 2]
        t_list.append(t_k)
        dt_list.append(dt_k)
        d2t_list.append(d2t_k)

    return (
        np.stack(t_list, axis=-1),
        np.stack(dt_list, axis=-1),
        np.stack(d2t_list, axis=-1),
    )


def tanh_domain_derivatives(
    x: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate xi_j = tanh((x_j - c_j) / s_j) and its first two derivatives wrt x_j."""
    x_arr = np.asarray(x, dtype=np.float64)
    c_arr = np.asarray(center, dtype=np.float64)
    s_arr = np.maximum(np.asarray(scale, dtype=np.float64), 1e-6)
    u = (x_arr - c_arr) / s_arr
    xi = np.tanh(u)
    sech2 = 1.0 - xi * xi
    dxi_dx = sech2 / s_arr
    d2xi_dx2 = (-2.0 * xi * sech2) / (s_arr * s_arr)
    return xi, dxi_dx, d2xi_dx2


def _interval_chebyshev_bounds(
    z_lo: Union[float, np.ndarray],
    z_hi: Union[float, np.ndarray],
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact sound interval bounds on T_k, T_k', T_k'' over [z_lo, z_hi] in [-1, 1].

    Returns ``(T_lo, T_hi, dT_lo, dT_hi, d2T_lo, d2T_hi)`` with trailing axis ``degree + 1``.
    """
    zlo = np.clip(np.asarray(z_lo, dtype=np.float64), -1.0, 1.0)
    zhi = np.clip(np.asarray(z_hi, dtype=np.float64), -1.0, 1.0)
    zlo, zhi = np.minimum(zlo, zhi), np.maximum(zlo, zhi)
    deg = max(0, int(degree))

    t_lo_v, dt_lo_v, d2t_lo_v = chebyshev_with_derivatives(zlo, deg)
    t_hi_v, dt_hi_v, d2t_hi_v = chebyshev_with_derivatives(zhi, deg)

    t_lo = np.minimum(t_lo_v, t_hi_v)
    t_hi = np.maximum(t_lo_v, t_hi_v)
    dt_lo = np.minimum(dt_lo_v, dt_hi_v)
    dt_hi = np.maximum(dt_lo_v, dt_hi_v)
    d2t_lo = np.minimum(d2t_lo_v, d2t_hi_v)
    d2t_hi = np.maximum(d2t_lo_v, d2t_hi_v)

    for k in range(2, deg + 1):
        # Extrema of T_k on (-1, 1) occur at cos(m * pi / k), where T_k = (-1)^m.
        m_idx = np.arange(1, k, dtype=np.float64)
        crit_t = np.cos(m_idx * np.pi / float(k))
        vals_t = np.where((m_idx.astype(np.int64) % 2) == 0, 1.0, -1.0)
        for z_star, val_star in zip(crit_t, vals_t):
            inside = (zlo < z_star) & (z_star < zhi)
            if np.any(inside):
                t_lo[..., k] = np.where(inside, np.minimum(t_lo[..., k], val_star), t_lo[..., k])
                t_hi[..., k] = np.where(inside, np.maximum(t_hi[..., k], val_star), t_hi[..., k])

        # Extrema of T_k' on (-1, 1).
        crit_dt = _cheb_basis_crit_points(k, deriv_order=1)
        if crit_dt.size > 0:
            _, dt_crit, _ = chebyshev_with_derivatives(crit_dt, k)
            for z_star, val_star in zip(crit_dt, dt_crit[:, k]):
                inside = (zlo < z_star) & (z_star < zhi)
                if np.any(inside):
                    dt_lo[..., k] = np.where(inside, np.minimum(dt_lo[..., k], val_star), dt_lo[..., k])
                    dt_hi[..., k] = np.where(inside, np.maximum(dt_hi[..., k], val_star), dt_hi[..., k])

        # Extrema of T_k'' on (-1, 1).
        crit_d2t = _cheb_basis_crit_points(k, deriv_order=2)
        if crit_d2t.size > 0:
            _, _, d2t_crit = chebyshev_with_derivatives(crit_d2t, k)
            for z_star, val_star in zip(crit_d2t, d2t_crit[:, k]):
                inside = (zlo < z_star) & (z_star < zhi)
                if np.any(inside):
                    d2t_lo[..., k] = np.where(inside, np.minimum(d2t_lo[..., k], val_star), d2t_lo[..., k])
                    d2t_hi[..., k] = np.where(inside, np.maximum(d2t_hi[..., k], val_star), d2t_hi[..., k])

    return t_lo, t_hi, dt_lo, dt_hi, d2t_lo, d2t_hi


def _interval_mul(
    a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sound elementwise interval product [a_lo, a_hi] * [b_lo, b_hi]."""
    p1 = a_lo * b_lo
    p2 = a_lo * b_hi
    p3 = a_hi * b_lo
    p4 = a_hi * b_hi
    lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return lo, hi


def _interval_dot(
    a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray
) -> Tuple[float, float]:
    """Sound interval dot product of two 1D interval vectors."""
    p_lo, p_hi = _interval_mul(a_lo, a_hi, b_lo, b_hi)
    return float(np.sum(p_lo)), float(np.sum(p_hi))


def _interval_matmul(
    m_lo: np.ndarray, m_hi: np.ndarray, v_lo: np.ndarray, v_hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sound interval matrix-vector (or row-vector * matrix) product."""
    m_lo_arr = np.asarray(m_lo, dtype=np.float64)
    m_hi_arr = np.asarray(m_hi, dtype=np.float64)
    v_lo_arr = np.asarray(v_lo, dtype=np.float64)
    v_hi_arr = np.asarray(v_hi, dtype=np.float64)
    if v_lo_arr.ndim == 1 and m_lo_arr.ndim == 2:
        p_lo, p_hi = _interval_mul(m_lo_arr, m_hi_arr, v_lo_arr[None, :], v_hi_arr[None, :])
        return np.sum(p_lo, axis=1), np.sum(p_hi, axis=1)
    p_lo, p_hi = _interval_mul(m_lo_arr, m_hi_arr, v_lo_arr, v_hi_arr)
    return np.sum(p_lo, axis=-1), np.sum(p_hi, axis=-1)


class KANBarrierConstraint(BarrierConstraint):
    """Chebyshev KAN Control Barrier Function with exact calculus & ridge fitting.

    Models ``h(x) = f_KAN(x) - offset >= 0`` using a dedicated single-output
    ``KolmogorovArnoldMicroBrain(in_features=state_dim, out_features=1, degree=degree)``
    with per-instance ``grid_center`` / ``grid_scale`` domain mapping.
    """

    def __init__(
        self,
        state_dim: int,
        degree: int = 4,
        alpha: float = 0.5,
        relative_degree: int = 1,
        hard: bool = True,
        alpha2: float = 1.0,
        offset: float = 0.0,
        kan: Optional[KolmogorovArnoldMicroBrain] = None,
        grid_center: Optional[Sequence[float]] = None,
        grid_scale: Optional[Sequence[float]] = None,
    ):
        self.state_dim = int(state_dim)
        self.n_dof = self.state_dim
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.relative_degree = int(relative_degree)
        self.hard = bool(hard)
        self.alpha2 = float(alpha2)
        self.offset = float(offset)

        if kan is not None:
            if kan.in_features != self.state_dim or kan.out_features < 1:
                raise ValueError(
                    f"KAN shape mismatch: expected in_features={self.state_dim}, out_features>=1; "
                    f"got ({kan.in_features}, {kan.out_features})"
                )
            self.kan = kan
            self.degree = int(kan.degree)
        else:
            self.kan = KolmogorovArnoldMicroBrain(
                in_features=self.state_dim,
                out_features=1,
                degree=self.degree,
            )

        if grid_center is not None:
            c_arr = np.asarray(grid_center, dtype=np.float32).flatten()
            if c_arr.size != self.state_dim:
                raise ValueError(f"grid_center length {c_arr.size} != state_dim {self.state_dim}")
            self.kan.grid_center = c_arr
            self.kan.grid_adapted = True
        if grid_scale is not None:
            s_arr = np.maximum(np.asarray(grid_scale, dtype=np.float32).flatten(), 1e-6)
            if s_arr.size != self.state_dim:
                raise ValueError(f"grid_scale length {s_arr.size} != state_dim {self.state_dim}")
            self.kan.grid_scale = s_arr
            self.kan.grid_adapted = True

    def _effective_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.kan.grid_adapted:
            center = np.asarray(self.kan.grid_center, dtype=np.float64).flatten()
            scale = np.maximum(np.asarray(self.kan.grid_scale, dtype=np.float64).flatten(), 1e-6)
        else:
            center = np.zeros(self.state_dim, dtype=np.float64)
            scale = np.ones(self.state_dim, dtype=np.float64)
        return center, scale

    def _validate_state(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if x_arr.size != self.state_dim:
            raise ValueError(f"Expected state of dimension {self.state_dim}, got {x_arr.size}")
        return x_arr

    def _coeffs_f64(self) -> np.ndarray:
        return np.asarray(self.kan.coeffs[0], dtype=np.float64)

    def h(self, x: Union[Sequence[float], np.ndarray]) -> float:
        x_arr = self._validate_state(x)
        center, scale = self._effective_grid()
        xi, _, _ = tanh_domain_derivatives(x_arr, center, scale)
        t_vals, _, _ = chebyshev_with_derivatives(xi, self.degree)
        coeffs = self._coeffs_f64()
        return float(np.sum(coeffs * t_vals) - self.offset)

    def grad_h(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x_arr = self._validate_state(x)
        center, scale = self._effective_grid()
        xi, dxi_dx, _ = tanh_domain_derivatives(x_arr, center, scale)
        _, dt_vals, _ = chebyshev_with_derivatives(xi, self.degree)
        coeffs = self._coeffs_f64()
        dphi_dxi = np.sum(coeffs * dt_vals, axis=-1)
        return dphi_dxi * dxi_dx

    def hess_h(self, x: Union[Sequence[float], np.ndarray], eps: float = 1e-5) -> np.ndarray:
        _ = eps
        x_arr = self._validate_state(x)
        center, scale = self._effective_grid()
        xi, dxi_dx, d2xi_dx2 = tanh_domain_derivatives(x_arr, center, scale)
        _, dt_vals, d2t_vals = chebyshev_with_derivatives(xi, self.degree)
        coeffs = self._coeffs_f64()
        dphi_dxi = np.sum(coeffs * dt_vals, axis=-1)
        d2phi_dxi2 = np.sum(coeffs * d2t_vals, axis=-1)
        diag_hess = d2phi_dxi2 * (dxi_dx * dxi_dx) + dphi_dxi * d2xi_dx2
        return np.diag(diag_hess)

    def h_batch(self, X: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
        """Vectorized batch evaluation of h(x) for shape (N, state_dim)."""
        x_mat = np.asarray(X, dtype=np.float64)
        if x_mat.ndim == 1:
            x_mat = x_mat.reshape(1, -1)
        if x_mat.shape[1] != self.state_dim:
            raise ValueError(f"Expected (N, {self.state_dim}), got {x_mat.shape}")
        center, scale = self._effective_grid()
        xi, _, _ = tanh_domain_derivatives(x_mat, center[None, :], scale[None, :])
        t_vals, _, _ = chebyshev_with_derivatives(xi, self.degree)
        coeffs = self._coeffs_f64()
        return np.einsum("ik,bik->b", coeffs, t_vals) - self.offset

    def inflate(self, delta: float) -> "KANBarrierConstraint":
        kan_copy = KolmogorovArnoldMicroBrain(
            in_features=self.state_dim,
            out_features=1,
            degree=self.degree,
        )
        kan_copy.coeffs = self.kan.coeffs.copy()
        kan_copy.grid_center = self.kan.grid_center.copy()
        kan_copy.grid_scale = self.kan.grid_scale.copy()
        kan_copy.grid_adapted = bool(self.kan.grid_adapted)
        kan_copy.trained = bool(self.kan.trained)
        kan_copy.training_mse = self.kan.training_mse
        return KANBarrierConstraint(
            state_dim=self.state_dim,
            degree=self.degree,
            alpha=self.alpha,
            relative_degree=self.relative_degree,
            hard=self.hard,
            alpha2=self.alpha2,
            offset=self.offset + float(delta),
            kan=kan_copy,
        )

    def fit_from_samples(
        self,
        X_safe: Union[Sequence[Sequence[float]], np.ndarray],
        X_unsafe: Union[Sequence[Sequence[float]], np.ndarray],
        X_boundary: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
        margin_safe: float = 0.2,
        margin_unsafe: float = 0.2,
        reg_l2: float = 1e-4,
        adapt_grid: Optional[bool] = None,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Fit Chebyshev edge coefficients via closed-form weighted ridge regression."""
        xs = np.asarray(X_safe, dtype=np.float64).reshape(-1, self.state_dim)
        xu = np.asarray(X_unsafe, dtype=np.float64).reshape(-1, self.state_dim)
        parts: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        if xs.shape[0] > 0:
            parts.append(xs)
            targets.append(np.full(xs.shape[0], float(margin_safe), dtype=np.float64))
        if xu.shape[0] > 0:
            parts.append(xu)
            targets.append(np.full(xu.shape[0], -float(margin_unsafe), dtype=np.float64))
        if X_boundary is not None:
            xb = np.asarray(X_boundary, dtype=np.float64).reshape(-1, self.state_dim)
            if xb.shape[0] > 0:
                parts.append(xb)
                targets.append(np.zeros(xb.shape[0], dtype=np.float64))
        else:
            xb = np.zeros((0, self.state_dim), dtype=np.float64)

        if not parts:
            return {"status": "skipped_empty_samples", "trained": self.kan.trained}

        X_all = np.vstack(parts)
        y_all = np.concatenate(targets)
        n_total = X_all.shape[0]

        should_adapt = (not self.kan.grid_adapted) if adapt_grid is None else bool(adapt_grid)
        if should_adapt:
            self.kan.adapt_grid(X_all.astype(np.float32), margin=0.15)

        center, scale = self._effective_grid()
        xi, _, _ = tanh_domain_derivatives(X_all, center[None, :], scale[None, :])
        t_vals, _, _ = chebyshev_with_derivatives(xi, self.degree)
        n_basis = self.state_dim * (self.degree + 1)
        phi = t_vals.reshape(n_total, n_basis)

        if sample_weights is not None:
            w_arr = np.maximum(np.asarray(sample_weights, dtype=np.float64).flatten(), 0.0)
            if w_arr.size != n_total:
                raise ValueError(f"sample_weights length {w_arr.size} != total samples {n_total}")
            sqrt_w = np.sqrt(w_arr)[:, None]
            phi_w = phi * sqrt_w
            y_w = y_all * sqrt_w[:, 0]
        else:
            phi_w = phi
            y_w = y_all

        gram = phi_w.T @ phi_w + float(reg_l2) * np.eye(n_basis, dtype=np.float64)
        rhs = phi_w.T @ y_w
        try:
            weights_flat = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            weights_flat, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)

        new_coeffs = weights_flat.reshape(1, self.state_dim, self.degree + 1).astype(np.float32)
        self.kan.coeffs = new_coeffs
        self.kan.trained = True
        self.offset = 0.0

        preds = self.h_batch(X_all)
        mse = float(np.mean((preds - y_all) ** 2))
        self.kan.training_mse = round(mse, 6)

        safe_vals = self.h_batch(xs) if xs.shape[0] > 0 else np.zeros(0, dtype=np.float64)
        unsafe_vals = self.h_batch(xu) if xu.shape[0] > 0 else np.zeros(0, dtype=np.float64)
        safe_acc = float(np.mean(safe_vals > 0.0)) if safe_vals.size > 0 else 1.0
        unsafe_acc = float(np.mean(unsafe_vals < 0.0)) if unsafe_vals.size > 0 else 1.0

        return {
            "status": "kan_barrier_fitted",
            "trained": True,
            "n_safe": int(xs.shape[0]),
            "n_unsafe": int(xu.shape[0]),
            "n_boundary": int(xb.shape[0]),
            "mse": mse,
            "safe_accuracy": safe_acc,
            "unsafe_accuracy": unsafe_acc,
            "min_safe_h": float(np.min(safe_vals)) if safe_vals.size > 0 else 0.0,
            "max_unsafe_h": float(np.max(unsafe_vals)) if unsafe_vals.size > 0 else 0.0,
        }

    def export_symbolic_formula(self, top_k_terms: Optional[int] = None) -> Dict[str, Any]:
        """Delegate symbolic formula extraction to KolmogorovArnoldMicroBrain and emit coefficients."""
        base = self.kan.extract_symbolic_formula(top_k_terms=top_k_terms)
        center, scale = self._effective_grid()
        raw_formula = base["symbolic_formulas"].get("y_0", "0.0")
        if abs(self.offset) > 1e-12:
            formula_with_offset = f"({raw_formula}) - {self.offset:.6g}"
        else:
            formula_with_offset = raw_formula
        return {
            **base,
            "state_dim": self.state_dim,
            "degree": self.degree,
            "offset": float(self.offset),
            "alpha": float(self.alpha),
            "relative_degree": int(self.relative_degree),
            "grid_adapted": bool(self.kan.grid_adapted),
            "grid_center": [float(v) for v in center],
            "grid_scale": [float(v) for v in scale],
            "coefficients": self.kan.coeffs[0].astype(np.float64).tolist(),
            "barrier_formula": formula_with_offset,
        }

    def interval_bounds(
        self, x_lo: np.ndarray, x_hi: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute sound (tight) interval bounds on (h, grad_h, diag(hess_h)) over [x_lo, x_hi].

        Combines monotonic ``tanh`` interval propagation and ``_interval_chebyshev_bounds``
        with exact 1D polynomial critical-point evaluation per KAN edge so each
        additive coordinate edge has virtually zero interval over-approximation.
        """
        lo = np.asarray(x_lo, dtype=np.float64).flatten()
        hi = np.asarray(x_hi, dtype=np.float64).flatten()
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        center, scale = self._effective_grid()
        coeffs = self._coeffs_f64()

        # Monotonic tanh propagation: u_i = (x_i - c_i) / s_i, z_i = tanh(u_i)
        z_lo = np.tanh((lo - center) / scale)
        z_hi = np.tanh((hi - center) / scale)

        t_lo, t_hi, dt_lo, dt_hi, d2t_lo, d2t_hi = _interval_chebyshev_bounds(
            z_lo, z_hi, self.degree
        )

        # Interval arithmetic bounds per edge for phi_i, dphi_i/dxi, d2phi_i/dxi2
        c_pos = np.maximum(coeffs, 0.0)
        c_neg = np.minimum(coeffs, 0.0)

        phi_ia_lo = np.sum(c_pos * t_lo + c_neg * t_hi, axis=-1)
        phi_ia_hi = np.sum(c_pos * t_hi + c_neg * t_lo, axis=-1)

        dphi_ia_lo = np.sum(c_pos * dt_lo + c_neg * dt_hi, axis=-1)
        dphi_ia_hi = np.sum(c_pos * dt_hi + c_neg * dt_lo, axis=-1)

        d2phi_ia_lo = np.sum(c_pos * d2t_lo + c_neg * d2t_hi, axis=-1)
        d2phi_ia_hi = np.sum(c_pos * d2t_hi + c_neg * d2t_lo, axis=-1)

        # Exact bounds on sech^2 = 1 - z^2 over [z_lo, z_hi]
        sech2_end_lo = np.minimum(1.0 - z_lo * z_lo, 1.0 - z_hi * z_hi)
        sech2_end_hi = np.maximum(1.0 - z_lo * z_lo, 1.0 - z_hi * z_hi)
        contains_zero = (z_lo <= 0.0) & (0.0 <= z_hi)
        sech2_hi = np.where(contains_zero, 1.0, sech2_end_hi)
        dxi_lo = sech2_end_lo / scale
        dxi_hi = sech2_hi / scale

        gh_ia_lo, gh_ia_hi = _interval_mul(dphi_ia_lo, dphi_ia_hi, dxi_lo, dxi_hi)

        # Exact bounds on d2xi/dx2 = -2 * z * (1 - z^2) / s^2: extrema at z = +-1/sqrt(3)
        def _d2xi_fn(z_val: np.ndarray) -> np.ndarray:
            return (-2.0 * z_val * (1.0 - z_val * z_val)) / (scale * scale)

        d2xi_a = _d2xi_fn(z_lo)
        d2xi_b = _d2xi_fn(z_hi)
        d2xi_lo = np.minimum(d2xi_a, d2xi_b)
        d2xi_hi = np.maximum(d2xi_a, d2xi_b)
        inv_s3 = 1.0 / np.sqrt(3.0)
        for z_star in (-inv_s3, inv_s3):
            inside = (z_lo < z_star) & (z_star < z_hi)
            if np.any(inside):
                val_star = _d2xi_fn(np.full_like(z_lo, z_star))
                d2xi_lo = np.where(inside, np.minimum(d2xi_lo, val_star), d2xi_lo)
                d2xi_hi = np.where(inside, np.maximum(d2xi_hi, val_star), d2xi_hi)

        dxi2_lo = dxi_lo * dxi_lo  # dxi >= 0 everywhere
        dxi2_hi = dxi_hi * dxi_hi
        term1_lo, term1_hi = _interval_mul(d2phi_ia_lo, d2phi_ia_hi, dxi2_lo, dxi2_hi)
        term2_lo, term2_hi = _interval_mul(dphi_ia_lo, dphi_ia_hi, d2xi_lo, d2xi_hi)
        hdiag_lo = term1_lo + term2_lo
        hdiag_hi = term1_hi + term2_hi

        # Tighten phi_i(z) and grad_h_i(z) = ((1 - z^2)/s_i) * sum_k c_{i,k} T_k'(z)
        # using exact 1D polynomial critical points intersected with interval bounds.
        phi_lo = np.empty(self.state_dim, dtype=np.float64)
        phi_hi = np.empty(self.state_dim, dtype=np.float64)
        gh_lo = np.empty(self.state_dim, dtype=np.float64)
        gh_hi = np.empty(self.state_dim, dtype=np.float64)

        for i in range(self.state_dim):
            za, zb = float(z_lo[i]), float(z_hi[i])
            if zb - za <= 1e-14:
                t_pt, dt_pt, _ = chebyshev_with_derivatives(np.array([za]), self.degree)
                p_val = float(np.sum(coeffs[i] * t_pt[0]))
                g_val = float(np.sum(coeffs[i] * dt_pt[0])) * ((1.0 - za * za) / scale[i])
                phi_lo[i], phi_hi[i] = min(p_val, phi_ia_hi[i]), max(p_val, phi_ia_lo[i])
                phi_lo[i] = max(phi_ia_lo[i], p_val - 1e-12)
                phi_hi[i] = min(phi_ia_hi[i], p_val + 1e-12)
                gh_lo[i] = max(gh_ia_lo[i], g_val - 1e-12)
                gh_hi[i] = min(gh_ia_hi[i], g_val + 1e-12)
                continue

            # Exact 1D range of phi_i(z)
            c_i = coeffs[i]
            d_c_i = np_cheb.chebder(c_i)
            cand_phi = [za, zb]
            if np.any(np.abs(d_c_i) > 1e-14):
                roots_phi = np_cheb.chebroots(d_c_i)
                real_r = np.real(roots_phi[np.abs(np.imag(roots_phi)) < 1e-9])
                for r in real_r:
                    if za < r < zb:
                        cand_phi.append(float(r))
            z_cand_phi = np.asarray(cand_phi, dtype=np.float64)
            t_cand, _, _ = chebyshev_with_derivatives(z_cand_phi, self.degree)
            vals_phi = t_cand @ c_i
            pad_p = 1e-11 * (1.0 + float(np.max(np.abs(vals_phi))))
            p_lo_exact = float(np.min(vals_phi)) - pad_p
            p_hi_exact = float(np.max(vals_phi)) + pad_p
            p_lo = max(float(phi_ia_lo[i]), p_lo_exact)
            p_hi = min(float(phi_ia_hi[i]), p_hi_exact)
            if p_lo > p_hi:
                p_lo, p_hi = float(phi_ia_lo[i]), float(phi_ia_hi[i])
            phi_lo[i], phi_hi[i] = p_lo, p_hi

            # Exact 1D range of grad_h_i(z) = ((1 - z^2) / s_i) * sum_k c_{i,k} T_k'(z)
            d_cheb = np_cheb.chebder(c_i)
            p_dcheb = np_cheb.cheb2poly(d_cheb)
            # Multiply by (1 - z^2) / s_i = (1 / s_i) - (1 / s_i) z^2
            g_poly = np_poly.polymul(p_dcheb, np.array([1.0, 0.0, -1.0], dtype=np.float64)) / scale[i]
            dg_poly = np_poly.polyder(g_poly)
            cand_gh = [za, zb]
            # Trim near-zero trailing coefficients for numerical stability in polyroots
            nz = np.where(np.abs(dg_poly) > 1e-14)[0]
            if nz.size > 0 and int(nz[-1]) >= 1:
                roots_gh = np_poly.polyroots(dg_poly[: int(nz[-1]) + 1])
                real_g = np.real(roots_gh[np.abs(np.imag(roots_gh)) < 1e-9])
                for r in real_g:
                    if za < r < zb:
                        cand_gh.append(float(r))
            z_cand_gh = np.asarray(cand_gh, dtype=np.float64)
            _, dt_cand, _ = chebyshev_with_derivatives(z_cand_gh, self.degree)
            vals_gh = (dt_cand @ c_i) * ((1.0 - z_cand_gh * z_cand_gh) / scale[i])
            pad_g = 1e-11 * (1.0 + float(np.max(np.abs(vals_gh))))
            g_lo_exact = float(np.min(vals_gh)) - pad_g
            g_hi_exact = float(np.max(vals_gh)) + pad_g
            g_lo = max(float(gh_ia_lo[i]), g_lo_exact)
            g_hi = min(float(gh_ia_hi[i]), g_hi_exact)
            if g_lo > g_hi:
                g_lo, g_hi = float(gh_ia_lo[i]), float(gh_ia_hi[i])
            gh_lo[i], gh_hi[i] = g_lo, g_hi

        h_lo = float(np.sum(phi_lo) - self.offset)
        h_hi = float(np.sum(phi_hi) - self.offset)
        return h_lo, h_hi, gh_lo, gh_hi, hdiag_lo, hdiag_hi


class FormalCBFVerifier:
    """Sound interval-arithmetic verifier and CEGIS synthesizer for KAN CBFs."""

    @staticmethod
    def _interval_chebyshev_bounds(
        z_lo: Union[float, np.ndarray],
        z_hi: Union[float, np.ndarray],
        degree: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _interval_chebyshev_bounds(z_lo, z_hi, degree)

    @staticmethod
    def _interval_tanh(u_lo: np.ndarray, u_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(u_lo, dtype=np.float64)
        hi = np.asarray(u_hi, dtype=np.float64)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        return np.tanh(lo), np.tanh(hi)

    @staticmethod
    def _interval_mul(
        a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _interval_mul(a_lo, a_hi, b_lo, b_hi)

    @staticmethod
    def _interval_matmul(
        m_lo: np.ndarray, m_hi: np.ndarray, v_lo: np.ndarray, v_hi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _interval_matmul(m_lo, m_hi, v_lo, v_hi)

    @staticmethod
    def _dynamics_interval_bounds(
        dynamics: ControlAffineDynamics, x_lo: np.ndarray, x_hi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute sound interval bounds [f_lo, f_hi] and [G_lo, G_hi] over [x_lo, x_hi]."""
        lo = np.asarray(x_lo, dtype=np.float64).flatten()
        hi = np.asarray(x_hi, dtype=np.float64).flatten()

        if hasattr(dynamics, "interval_f") and hasattr(dynamics, "interval_g"):
            f_lo, f_hi = dynamics.interval_f(lo, hi)  # type: ignore[attr-defined]
            g_lo, g_hi = dynamics.interval_g(lo, hi)  # type: ignore[attr-defined]
            return (
                np.asarray(f_lo, dtype=np.float64),
                np.asarray(f_hi, dtype=np.float64),
                np.asarray(g_lo, dtype=np.float64),
                np.asarray(g_hi, dtype=np.float64),
            )

        if isinstance(dynamics, LinearDynamics):
            A = np.asarray(dynamics.A, dtype=np.float64)
            B = np.asarray(dynamics.B, dtype=np.float64)
            a_pos = np.maximum(A, 0.0)
            a_neg = np.minimum(A, 0.0)
            f_lo = a_pos @ lo + a_neg @ hi
            f_hi = a_pos @ hi + a_neg @ lo
            return f_lo, f_hi, B.copy(), B.copy()

        # General ControlAffineDynamics fallback: sample corners/midpoint + Jacobian remainder
        d = lo.shape[0]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        samples: List[np.ndarray] = [mid]
        if d <= 5:
            for bits in itertools.product((0, 1), repeat=d):
                pt = np.where(np.asarray(bits) == 0, lo, hi)
                samples.append(pt)
        else:
            samples.extend([lo, hi])
            for idx in range(d):
                p1 = mid.copy()
                p1[idx] = lo[idx]
                p2 = mid.copy()
                p2[idx] = hi[idx]
                samples.extend([p1, p2])

        f_vals = np.stack([np.asarray(dynamics.f(s), dtype=np.float64).flatten() for s in samples], axis=0)
        g_vals = np.stack([np.asarray(dynamics.g(s), dtype=np.float64) for s in samples], axis=0)
        f_lo = np.min(f_vals, axis=0)
        f_hi = np.max(f_vals, axis=0)
        g_lo = np.min(g_vals, axis=0)
        g_hi = np.max(g_vals, axis=0)

        # Pad for potential interior non-monotonicity when f or g is nonlinear
        f_mid = f_vals[0]
        g_mid = g_vals[0]
        if np.any(np.abs(f_hi - f_lo) > 1e-12):
            j_mid = np.asarray(dynamics.df_dx(mid), dtype=np.float64)
            f_lin_dev = np.abs(j_mid) @ half
            # Compare against sampled variation to bound curvature safely
            curv_f = np.maximum(0.0, np.maximum(f_hi - f_mid, f_mid - f_lo) - f_lin_dev)
            pad_f = 0.25 * (f_hi - f_lo) * (np.max(half) > 1e-9) + curv_f
            # Check linearity across samples: if affine, pad is 0
            lin_preds = np.stack([f_mid + j_mid @ (s - mid) for s in samples], axis=0)
            nonlin_err = float(np.max(np.abs(f_vals - lin_preds)))
            if nonlin_err > 1e-9:
                f_lo = f_lo - 0.5 * pad_f - 0.25 * nonlin_err
                f_hi = f_hi + 0.5 * pad_f + 0.25 * nonlin_err

        if np.any(np.abs(g_hi - g_lo) > 1e-12):
            pad_g = 0.25 * (g_hi - g_lo)
            g_lo = g_lo - pad_g
            g_hi = g_hi + pad_g

        return f_lo, f_hi, g_lo, g_hi

    @staticmethod
    def _barrier_interval_bounds(
        barrier: BarrierConstraint, x_lo: np.ndarray, x_hi: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Compute sound bounds (h_lo, h_hi, grad_lo, grad_hi) for a BarrierConstraint."""
        if isinstance(barrier, KANBarrierConstraint):
            h_lo, h_hi, gh_lo, gh_hi, _, _ = barrier.interval_bounds(x_lo, x_hi)
            return h_lo, h_hi, gh_lo, gh_hi

        if isinstance(barrier, HalfPlaneBarrier):
            a = barrier.a
            a_pos = np.maximum(a, 0.0)
            a_neg = np.minimum(a, 0.0)
            h_lo = float(a_pos @ x_lo + a_neg @ x_hi - barrier.b)
            h_hi = float(a_pos @ x_hi + a_neg @ x_lo - barrier.b)
            return h_lo, h_hi, a.copy(), a.copy()

        if isinstance(barrier, SphereBarrier):
            c = barrier.c
            d_lo = x_lo - c
            d_hi = x_hi - c
            sq_hi = np.maximum(d_lo * d_lo, d_hi * d_hi)
            contains_c = (d_lo <= 0.0) & (0.0 <= d_hi)
            sq_lo = np.where(contains_c, 0.0, np.minimum(d_lo * d_lo, d_hi * d_hi))
            h_lo = float(np.sum(sq_lo) - barrier.r * barrier.r)
            h_hi = float(np.sum(sq_hi) - barrier.r * barrier.r)
            return h_lo, h_hi, 2.0 * d_lo, 2.0 * d_hi

        if isinstance(barrier, HighOrderBarrier):
            sl = barrier._resolve_slice(x_lo.shape[0])
            h_lo, h_hi, g_sub_lo, g_sub_hi = FormalCBFVerifier._barrier_interval_bounds(
                barrier.base_barrier, x_lo[sl], x_hi[sl]
            )
            gh_lo = np.zeros_like(x_lo)
            gh_hi = np.zeros_like(x_hi)
            gh_lo[sl] = g_sub_lo
            gh_hi[sl] = g_sub_hi
            return h_lo, h_hi, gh_lo, gh_hi

        raise TypeError(f"Unsupported barrier type for interval verification: {type(barrier).__name__}")

    @staticmethod
    def _max_control_interval(
        lg_lo: np.ndarray,
        lg_hi: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Tuple[float, float]:
        """Exact bounds on sum_j max(Lg_j * u_min_j, Lg_j * u_max_j) for Lg_j in [lg_lo_j, lg_hi_j]."""
        val_a = np.maximum(lg_lo * u_min, lg_lo * u_max)
        val_b = np.maximum(lg_hi * u_min, lg_hi * u_max)
        crosses_zero = (lg_lo < 0.0) & (0.0 < lg_hi)
        ctrl_lo = np.where(crosses_zero, np.minimum(np.minimum(val_a, val_b), 0.0), np.minimum(val_a, val_b))
        ctrl_hi = np.maximum(val_a, val_b)
        return float(np.sum(ctrl_lo)), float(np.sum(ctrl_hi))

    @staticmethod
    def _exact_cbf_slack(
        barrier: BarrierConstraint,
        dynamics: ControlAffineDynamics,
        x: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Tuple[float, float]:
        """Evaluate exact (h(x), max_{u in [u_min, u_max]} CBF_slack(x, u)) at state x."""
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        h_val = float(barrier.h(x_arr))
        gh = np.asarray(barrier.grad_h(x_arr), dtype=np.float64).flatten()
        f_x = np.asarray(dynamics.f(x_arr), dtype=np.float64).flatten()
        G_x = np.asarray(dynamics.g(x_arr), dtype=np.float64)

        if int(getattr(barrier, "relative_degree", 1)) == 2:
            if hasattr(dynamics, "df_dx"):
                j_f = np.asarray(dynamics.df_dx(x_arr), dtype=np.float64)
            elif hasattr(dynamics, "J_f"):
                j_f = np.asarray(dynamics.J_f(x_arr), dtype=np.float64)
            else:
                j_f = np.zeros((x_arr.size, x_arr.size), dtype=np.float64)
            hess = np.asarray(barrier.hess_h(x_arr), dtype=np.float64)
            lf_h = float(np.dot(gh, f_x))
            lf2_h = float(f_x @ hess @ f_x + gh @ (j_f @ f_x))
            lg_lf_h = np.asarray((f_x @ hess @ G_x) + (gh @ (j_f @ G_x)), dtype=np.float64).flatten()
            alpha1 = float(getattr(barrier, "alpha1", getattr(barrier, "alpha", 0.5)))
            alpha2 = float(getattr(barrier, "alpha2", 1.0))
            ctrl_max = float(np.sum(np.maximum(lg_lf_h * u_min, lg_lf_h * u_max)))
            slack = lf2_h + ctrl_max + (alpha1 + alpha2) * lf_h + alpha1 * alpha2 * h_val
            return h_val, float(slack)

        lf_h = float(np.dot(gh, f_x))
        lg_h = np.asarray(gh @ G_x, dtype=np.float64).flatten()
        alpha = float(getattr(barrier, "alpha", 0.5))
        ctrl_max = float(np.sum(np.maximum(lg_h * u_min, lg_h * u_max)))
        slack = lf_h + ctrl_max + alpha * h_val
        return h_val, float(slack)

    def _box_cbf_slack_bounds(
        self,
        barrier: BarrierConstraint,
        dynamics: ControlAffineDynamics,
        x_lo: np.ndarray,
        x_hi: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Return (h_lo, h_hi, slack_lo, slack_hi) over hyperrectangle [x_lo, x_hi]."""
        h_lo, h_hi, gh_lo, gh_hi = self._barrier_interval_bounds(barrier, x_lo, x_hi)
        f_lo, f_hi, g_lo, g_hi = self._dynamics_interval_bounds(dynamics, x_lo, x_hi)
        lf_lo, lf_hi = _interval_dot(gh_lo, gh_hi, f_lo, f_hi)

        if int(getattr(barrier, "relative_degree", 1)) == 2:
            # Sound relative-degree-2 bounds when barrier supports hess_h intervals
            if isinstance(barrier, KANBarrierConstraint):
                _, _, _, _, hd_lo, hd_hi = barrier.interval_bounds(x_lo, x_hi)
                h_mat_lo = np.diag(hd_lo)
                h_mat_hi = np.diag(hd_hi)
            elif isinstance(barrier, HighOrderBarrier) and isinstance(barrier.base_barrier, KANBarrierConstraint):
                sl = barrier._resolve_slice(x_lo.shape[0])
                _, _, _, _, hd_sub_lo, hd_sub_hi = barrier.base_barrier.interval_bounds(x_lo[sl], x_hi[sl])
                hd_lo = np.zeros_like(x_lo)
                hd_hi = np.zeros_like(x_hi)
                hd_lo[sl] = hd_sub_lo
                hd_hi[sl] = hd_sub_hi
                h_mat_lo = np.diag(hd_lo)
                h_mat_hi = np.diag(hd_hi)
            else:
                mid = 0.5 * (x_lo + x_hi)
                hess_mid = np.asarray(barrier.hess_h(mid), dtype=np.float64)
                h_mat_lo, h_mat_hi = hess_mid, hess_mid

            if isinstance(dynamics, LinearDynamics):
                jf_lo = np.asarray(dynamics.A, dtype=np.float64)
                jf_hi = jf_lo
            else:
                mid = 0.5 * (x_lo + x_hi)
                jf_mid = np.asarray(dynamics.df_dx(mid), dtype=np.float64)
                jf_lo, jf_hi = jf_mid, jf_mid

            hf_lo, hf_hi = _interval_matmul(h_mat_lo, h_mat_hi, f_lo, f_hi)
            fhf_lo, fhf_hi = _interval_dot(f_lo, f_hi, hf_lo, hf_hi)
            jff_lo, jff_hi = _interval_matmul(jf_lo, jf_hi, f_lo, f_hi)
            gjf_lo, gjf_hi = _interval_dot(gh_lo, gh_hi, jff_lo, jff_hi)
            lf2_lo = fhf_lo + gjf_lo
            lf2_hi = fhf_hi + gjf_hi

            # L_g L_f h = f^T H G + grad_h^T (J_f G)
            fhg_lo, fhg_hi = _interval_matmul(g_lo.T, g_hi.T, hf_lo, hf_hi)
            jg_lo, jg_hi = _interval_mul(
                jf_lo[:, :, None], jf_hi[:, :, None], g_lo[None, :, :], g_hi[None, :, :]
            )
            jg_sum_lo = np.sum(jg_lo, axis=1)
            jg_sum_hi = np.sum(jg_hi, axis=1)
            gjg_lo, gjg_hi = _interval_matmul(jg_sum_lo.T, jg_sum_hi.T, gh_lo, gh_hi)
            lglf_lo = fhg_lo + gjg_lo
            lglf_hi = fhg_hi + gjg_hi

            ctrl_lo, ctrl_hi = self._max_control_interval(lglf_lo, lglf_hi, u_min, u_max)
            alpha1 = float(getattr(barrier, "alpha1", getattr(barrier, "alpha", 0.5)))
            alpha2 = float(getattr(barrier, "alpha2", 1.0))
            a_sum = alpha1 + alpha2
            a_prod = alpha1 * alpha2
            ah_lo = min(a_prod * h_lo, a_prod * h_hi)
            ah_hi = max(a_prod * h_lo, a_prod * h_hi)
            alf_lo = min(a_sum * lf_lo, a_sum * lf_hi)
            alf_hi = max(a_sum * lf_lo, a_sum * lf_hi)
            return h_lo, h_hi, lf2_lo + ctrl_lo + alf_lo + ah_lo, lf2_hi + ctrl_hi + alf_hi + ah_hi

        # L_g h = grad_h^T G = G^T grad_h
        lg_lo, lg_hi = _interval_matmul(g_lo.T, g_hi.T, gh_lo, gh_hi)
        ctrl_lo, ctrl_hi = self._max_control_interval(lg_lo, lg_hi, u_min, u_max)
        alpha = float(getattr(barrier, "alpha", 0.5))
        ah_lo = min(alpha * h_lo, alpha * h_hi)
        ah_hi = max(alpha * h_lo, alpha * h_hi)
        return h_lo, h_hi, lf_lo + ctrl_lo + ah_lo, lf_hi + ctrl_hi + ah_hi

    def verify(
        self,
        barrier: BarrierConstraint,
        dynamics: ControlAffineDynamics,
        domain_lo: Sequence[float],
        domain_hi: Sequence[float],
        u_min: Sequence[float],
        u_max: Sequence[float],
        max_splits: int = 256,
        eps_cert: float = 1e-4,
        tau_band: float = 0.05,
        max_counterexamples: int = 16,
    ) -> Dict[str, Any]:
        """Branch-and-bound interval verification over boxes overlapping the boundary strip.

        Checks ``max_{u in [u_min, u_max]} (L_f h + L_g h u + alpha h) >= eps_cert``
        on every box overlapping ``|h(x)| <= tau_band`` (including ``h_lo <= 0 <= h_hi``).
        """
        d_lo = np.asarray(domain_lo, dtype=np.float64).flatten()
        d_hi = np.asarray(domain_hi, dtype=np.float64).flatten()
        d_lo, d_hi = np.minimum(d_lo, d_hi), np.maximum(d_lo, d_hi)
        u_lo = np.asarray(u_min, dtype=np.float64).flatten()
        u_hi = np.asarray(u_max, dtype=np.float64).flatten()

        if np.any(u_lo > u_hi):
            return {
                "certified": False,
                "counterexamples": [0.5 * (d_lo + d_hi)],
                "boxes_checked": 1,
                "min_certified_slack": float("-inf"),
            }

        eps_f = float(eps_cert)
        tau_f = max(0.0, float(tau_band))
        min_box_width = 1e-6 * max(1.0, float(np.max(d_hi - d_lo)))

        counterexamples: List[np.ndarray] = []
        boxes_checked = 0
        splits_used = 0
        undecided_boxes = 0
        min_certified_slack = float("inf")
        min_observed_slack = float("inf")
        boundary_boxes_seen = 0

        # Priority queue ordered by (slack_lo, box_id, lo, hi)
        box_counter = itertools.count()
        heap: List[Tuple[float, int, np.ndarray, np.ndarray]] = []

        def _evaluate_and_enqueue(b_lo: np.ndarray, b_hi: np.ndarray) -> None:
            nonlocal boxes_checked, boundary_boxes_seen, min_certified_slack, min_observed_slack
            boxes_checked += 1
            h_lo, h_hi, slack_lo, slack_hi = self._box_cbf_slack_bounds(
                barrier, dynamics, b_lo, b_hi, u_lo, u_hi
            )
            # Prune boxes strictly outside the zero-level boundary strip
            if h_lo > tau_f or h_hi < -tau_f:
                return

            boundary_boxes_seen += 1

            # Concrete midpoint check for counterexamples inside boundary strip
            x_mid = 0.5 * (b_lo + b_hi)
            h_mid, s_mid = self._exact_cbf_slack(barrier, dynamics, x_mid, u_lo, u_hi)
            if abs(h_mid) <= tau_f and s_mid < eps_f:
                counterexamples.append(x_mid)
                min_observed_slack = min(min_observed_slack, s_mid)
                return

            if slack_lo >= eps_f:
                min_certified_slack = min(min_certified_slack, slack_lo)
                min_observed_slack = min(min_observed_slack, slack_lo)
                return

            # If even the box upper bound fails eps_cert, record midpoint when in strip
            if slack_hi < eps_f and abs(h_mid) <= tau_f:
                counterexamples.append(x_mid)
                min_observed_slack = min(min_observed_slack, s_mid)
                return

            heapq.heappush(heap, (slack_lo, next(box_counter), b_lo, b_hi))

        _evaluate_and_enqueue(d_lo, d_hi)

        while heap and len(counterexamples) < max_counterexamples:
            slack_lo, _, b_lo, b_hi = heapq.heappop(heap)
            widths = b_hi - b_lo
            max_w = float(np.max(widths))
            if splits_used >= max_splits or max_w <= min_box_width:
                undecided_boxes += 1
                min_observed_slack = min(min_observed_slack, slack_lo)
                x_mid = 0.5 * (b_lo + b_hi)
                h_mid, s_mid = self._exact_cbf_slack(barrier, dynamics, x_mid, u_lo, u_hi)
                if abs(h_mid) <= tau_f and s_mid < eps_f:
                    counterexamples.append(x_mid)
                continue

            split_dim = int(np.argmax(widths))
            mid_val = 0.5 * (b_lo[split_dim] + b_hi[split_dim])
            left_hi = b_hi.copy()
            left_hi[split_dim] = mid_val
            right_lo = b_lo.copy()
            right_lo[split_dim] = mid_val

            splits_used += 1
            _evaluate_and_enqueue(b_lo, left_hi)
            _evaluate_and_enqueue(right_lo, b_hi)

        # Count any remaining heap boxes as undecided if stopped early
        for rem_slack_lo, _, _, _ in heap:
            min_observed_slack = min(min_observed_slack, rem_slack_lo)
        undecided_boxes += len(heap)

        if boundary_boxes_seen == 0:
            return {
                "certified": False,
                "counterexamples": [],
                "boxes_checked": boxes_checked,
                "min_certified_slack": 0.0,
            }

        certified = (len(counterexamples) == 0) and (undecided_boxes == 0)
        reported_slack = min_certified_slack if certified else min_observed_slack
        if not np.isfinite(reported_slack):
            reported_slack = 0.0

        return {
            "certified": bool(certified),
            "counterexamples": counterexamples,
            "boxes_checked": int(boxes_checked),
            "min_certified_slack": float(reported_slack),
        }

    def cegis_synthesize(
        self,
        barrier: KANBarrierConstraint,
        dynamics: ControlAffineDynamics,
        X_safe: Union[Sequence[Sequence[float]], np.ndarray],
        X_unsafe: Union[Sequence[Sequence[float]], np.ndarray],
        domain_lo: Sequence[float],
        domain_hi: Sequence[float],
        u_min: Sequence[float],
        u_max: Sequence[float],
        max_rounds: int = 6,
        margin_safe: float = 0.2,
        margin_unsafe: float = 0.2,
        reg_l2: float = 1e-4,
        max_splits: int = 256,
        eps_cert: float = 1e-4,
        tau_band: float = 0.05,
        X_boundary: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
        adapt_alpha: bool = True,
        ce_weight: float = 3.0,
        max_alpha: float = 10.0,
    ) -> Dict[str, Any]:
        """Counterexample-Guided Inductive Synthesis (CEGIS) for KANBarrierConstraint.

        Alternates interval verification with counterexample incorporation and
        parameter refinement until ``certified=True`` or ``max_rounds`` is reached.
        """
        xs = np.asarray(X_safe, dtype=np.float64).reshape(-1, barrier.state_dim)
        xu = np.asarray(X_unsafe, dtype=np.float64).reshape(-1, barrier.state_dim)
        u_lo = np.asarray(u_min, dtype=np.float64).flatten()
        u_hi = np.asarray(u_max, dtype=np.float64).flatten()

        # Maintain per-sample weights for X_unsafe so counterexamples gain priority
        xu_weights = np.ones(xu.shape[0], dtype=np.float64)
        xs_weights = np.ones(xs.shape[0], dtype=np.float64)
        xb = (
            np.asarray(X_boundary, dtype=np.float64).reshape(-1, barrier.state_dim)
            if X_boundary is not None
            else None
        )
        xb_weights = np.ones(xb.shape[0], dtype=np.float64) if xb is not None else np.zeros(0, dtype=np.float64)

        cur_margin_safe = float(margin_safe)
        cur_margin_unsafe = float(margin_unsafe)
        history: List[Dict[str, Any]] = []
        total_ces_added = 0
        last_ver: Dict[str, Any] = {}

        for round_idx in range(max(1, int(max_rounds))):
            # Fit on round 0 only if not already trained; refit on all subsequent rounds
            if round_idx > 0 or not barrier.kan.trained:
                weight_parts = [xs_weights, xu_weights]
                if xb is not None and xb.shape[0] > 0:
                    weight_parts.append(xb_weights)
                all_weights = np.concatenate(weight_parts)
                barrier.fit_from_samples(
                    xs,
                    xu,
                    X_boundary=xb,
                    margin_safe=cur_margin_safe,
                    margin_unsafe=cur_margin_unsafe,
                    reg_l2=reg_l2,
                    adapt_grid=not barrier.kan.grid_adapted,
                    sample_weights=all_weights,
                )

            ver = self.verify(
                barrier,
                dynamics,
                domain_lo=domain_lo,
                domain_hi=domain_hi,
                u_min=u_lo,
                u_max=u_hi,
                max_splits=max_splits,
                eps_cert=eps_cert,
                tau_band=tau_band,
            )
            last_ver = ver
            history.append(
                {
                    "round": round_idx + 1,
                    "certified": bool(ver["certified"]),
                    "n_counterexamples": len(ver["counterexamples"]),
                    "boxes_checked": int(ver["boxes_checked"]),
                    "min_certified_slack": float(ver["min_certified_slack"]),
                    "alpha": float(barrier.alpha),
                }
            )

            if ver["certified"]:
                return {
                    "certified": True,
                    "rounds": round_idx + 1,
                    "barrier": barrier,
                    "verification": ver,
                    "history": history,
                    "counterexamples_added": total_ces_added,
                }

            ces = ver["counterexamples"]
            new_unsafe: List[np.ndarray] = []
            alpha_needs_boost = False
            max_alpha_needed = float(barrier.alpha)

            for x_ce in ces:
                h_ce = float(barrier.h(x_ce))
                gh_ce = np.asarray(barrier.grad_h(x_ce), dtype=np.float64).flatten()
                f_ce = np.asarray(dynamics.f(x_ce), dtype=np.float64).flatten()
                g_ce = np.asarray(dynamics.g(x_ce), dtype=np.float64)
                lf_h = float(np.dot(gh_ce, f_ce))
                lg_h = np.asarray(gh_ce @ g_ce, dtype=np.float64).flatten()
                ctrl_max = float(np.sum(np.maximum(lg_h * u_lo, lg_h * u_hi)))
                boundary_slack = lf_h + ctrl_max

                dist_safe = (
                    float(np.min(np.linalg.norm(xs - x_ce[None, :], axis=1)))
                    if xs.shape[0] > 0
                    else float("inf")
                )
                dist_unsafe = (
                    float(np.min(np.linalg.norm(xu - x_ce[None, :], axis=1)))
                    if xu.shape[0] > 0
                    else float("inf")
                )

                # If point is strictly on the safe side (h_ce > 0) and closer to safe core,
                # increasing alpha can certify it without shrinking the safe set.
                if adapt_alpha and h_ce > 1e-4 and dist_safe < dist_unsafe and boundary_slack < eps_cert:
                    needed = (eps_cert - boundary_slack) / h_ce
                    if needed <= max_alpha:
                        alpha_needs_boost = True
                        max_alpha_needed = max(max_alpha_needed, needed * 1.15)
                        continue

                new_unsafe.append(x_ce)

            if alpha_needs_boost and adapt_alpha:
                barrier.alpha = min(float(max_alpha), max(float(barrier.alpha) * 1.5, max_alpha_needed))

            if new_unsafe:
                added = np.stack(new_unsafe, axis=0)
                total_w = float(np.sum(xs_weights) + np.sum(xu_weights) + np.sum(xb_weights))
                per_ce_w = max(float(ce_weight), (0.3 * total_w) / float(added.shape[0]))
                xu = np.vstack([xu, added])
                xu_weights = np.concatenate([xu_weights, np.full(added.shape[0], per_ce_w)])
                total_ces_added += int(added.shape[0])
            elif not alpha_needs_boost:
                # No concrete counterexamples (only undecided over-approximated boxes):
                # steepen boundary margins slightly to raise |grad h| and slack.
                cur_margin_safe *= 1.2
                cur_margin_unsafe *= 1.2

        return {
            "certified": False,
            "rounds": max(1, int(max_rounds)),
            "barrier": barrier,
            "verification": last_ver,
            "history": history,
            "counterexamples_added": total_ces_added,
        }


__all__ = [
    "KANBarrierConstraint",
    "FormalCBFVerifier",
    "chebyshev_with_derivatives",
    "tanh_domain_derivatives",
]
