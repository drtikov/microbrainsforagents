"""Tests for KANBarrierConstraint, FormalCBFVerifier, and CEGIS synthesis."""

from __future__ import annotations

import numpy as np

from aetheris_microbrains.core.kan_neural_cbf_verifier import (
    FormalCBFVerifier,
    KANBarrierConstraint,
    chebyshev_with_derivatives,
    tanh_domain_derivatives,
)
from aetheris_microbrains.core.state_cbf_filter import (
    HighOrderBarrier,
    LinearDynamics,
    StateCBFSafetyFilter,
)


def _central_fd_grad(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).flatten()
    g = np.zeros_like(x)
    for i in range(x.size):
        ei = np.zeros_like(x)
        ei[i] = eps
        g[i] = (fn(x + ei) - fn(x - ei)) / (2.0 * eps)
    return g


def _central_fd_hess(grad_fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).flatten()
    n = x.size
    H = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ei = np.zeros_like(x)
        ei[i] = eps
        H[:, i] = (grad_fn(x + ei) - grad_fn(x - ei)) / (2.0 * eps)
    return 0.5 * (H + H.T)


def test_chebyshev_and_tanh_analytical_derivatives():
    rng = np.random.RandomState(7)
    z = rng.uniform(-0.95, 0.95, size=(12,))
    T, dT, d2T = chebyshev_with_derivatives(z, degree=5)
    assert T.shape == (12, 6)

    # Check against central finite differences on T_k and T_k'
    eps = 1e-6
    Tp, dTp, _ = chebyshev_with_derivatives(z + eps, degree=5)
    Tm, dTm, _ = chebyshev_with_derivatives(z - eps, degree=5)
    fd_dT = (Tp - Tm) / (2.0 * eps)
    fd_d2T = (dTp - dTm) / (2.0 * eps)
    assert np.max(np.abs(dT - fd_dT)) < 1e-7
    assert np.max(np.abs(d2T - fd_d2T)) < 1e-6

    x = rng.uniform(-3.0, 3.0, size=(4,))
    center = np.array([0.4, -0.7, 1.1, 0.0])
    scale = np.array([1.5, 2.2, 0.8, 3.0])
    xi, dxi, d2xi = tanh_domain_derivatives(x, center, scale)
    for j in range(4):
        ej = np.zeros(4)
        ej[j] = eps
        xi_p, dxi_p, _ = tanh_domain_derivatives(x + ej, center, scale)
        xi_m, dxi_m, _ = tanh_domain_derivatives(x - ej, center, scale)
        assert abs(dxi[j] - (xi_p[j] - xi_m[j]) / (2.0 * eps)) < 1e-8
        assert abs(d2xi[j] - (dxi_p[j] - dxi_m[j]) / (2.0 * eps)) < 1e-7


def test_kan_barrier_grad_and_hess_match_finite_differences():
    rng = np.random.RandomState(123)
    for grid_adapted in (False, True):
        grid_center = [0.5, -0.4, 0.2] if grid_adapted else None
        grid_scale = [2.1, 1.4, 0.9] if grid_adapted else None
        barrier = KANBarrierConstraint(
            state_dim=3,
            degree=5,
            offset=0.15,
            grid_center=grid_center,
            grid_scale=grid_scale,
        )
        barrier.kan.coeffs = rng.randn(1, 3, 6).astype(np.float32) * 0.4

        for _ in range(10):
            x = rng.uniform(-2.5, 2.5, size=(3,))
            g_exact = barrier.grad_h(x)
            g_fd = _central_fd_grad(barrier.h, x, eps=1e-6)
            assert np.max(np.abs(g_exact - g_fd)) < 1e-5

            H_exact = barrier.hess_h(x)
            H_fd = _central_fd_hess(barrier.grad_h, x, eps=1e-5)
            assert np.max(np.abs(H_exact - H_fd)) < 1e-4

        # Check inflate preserves derivatives and shifts h by -delta
        inflated = barrier.inflate(0.25)
        x_probe = rng.uniform(-1.0, 1.0, size=(3,))
        assert abs(inflated.h(x_probe) - (barrier.h(x_probe) - 0.25)) < 1e-10
        assert np.allclose(inflated.grad_h(x_probe), barrier.grad_h(x_probe))
        assert np.allclose(inflated.hess_h(x_probe), barrier.hess_h(x_probe))


def test_fit_elliptical_and_nonconvex_safe_regions_and_filter_trajectories():
    rng = np.random.RandomState(42)

    # 1. Elliptical safe region: (x0 / 1.5)^2 + (x1 / 0.8)^2 <= 1
    theta = rng.uniform(0.0, 2.0 * np.pi, 240)
    r_in = np.sqrt(rng.uniform(0.0, 0.85 ** 2, 240))
    r_out = rng.uniform(1.2, 2.0, 240)
    X_safe_ell = np.column_stack([1.5 * r_in * np.cos(theta), 0.8 * r_in * np.sin(theta)])
    X_unsafe_ell = np.column_stack([1.5 * r_out * np.cos(theta), 0.8 * r_out * np.sin(theta)])
    theta_b = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    X_boundary_ell = np.column_stack([1.5 * np.cos(theta_b), 0.8 * np.sin(theta_b)])

    ell_barrier = KANBarrierConstraint(state_dim=2, degree=4, alpha=1.5)
    fit_ell = ell_barrier.fit_from_samples(
        X_safe_ell,
        X_unsafe_ell,
        X_boundary=X_boundary_ell,
        margin_safe=0.2,
        margin_unsafe=0.2,
    )
    assert fit_ell["trained"] is True
    assert fit_ell["safe_accuracy"] == 1.0
    assert fit_ell["unsafe_accuracy"] == 1.0
    assert np.all(ell_barrier.h_batch(X_safe_ell) > 0.0)
    assert np.all(ell_barrier.h_batch(X_unsafe_ell) < 0.0)

    sym = ell_barrier.export_symbolic_formula()
    assert sym["trained"] is True
    assert "y_0" in sym["symbolic_formulas"]
    assert len(sym["coefficients"]) == 2

    # 2. Non-convex peanut/dumbbell safe region: 0.65 - (x0^2 - 0.75)^2 - x1^2 >= 0
    pts_nc = rng.uniform(-1.5, 1.5, size=(800, 2))
    peanut_val = 0.65 - (pts_nc[:, 0] ** 2 - 0.75) ** 2 - pts_nc[:, 1] ** 2
    X_safe_nc = pts_nc[peanut_val > 0.15]
    X_unsafe_nc = pts_nc[peanut_val < -0.15]
    assert X_safe_nc.shape[0] > 50 and X_unsafe_nc.shape[0] > 50

    nc_barrier = KANBarrierConstraint(state_dim=2, degree=6, alpha=2.0)
    fit_nc = nc_barrier.fit_from_samples(X_safe_nc, X_unsafe_nc, margin_safe=0.2, margin_unsafe=0.2)
    assert fit_nc["safe_accuracy"] >= 0.95
    assert fit_nc["unsafe_accuracy"] >= 0.95

    # Protect a trajectory inside StateCBFSafetyFilter (relative degree 1)
    dyn_1 = LinearDynamics(A=np.zeros((2, 2)), B=np.eye(2))
    cbf_1 = StateCBFSafetyFilter(
        dynamics=dyn_1,
        barriers=[ell_barrier],
        u_min=[-2.0, -2.0],
        u_max=[2.0, 2.0],
    )
    x = np.array([0.8, 0.2], dtype=np.float64)
    dt = 0.05
    for _ in range(40):
        res = cbf_1.filter(x, u_nominal=[1.8, 1.5])  # aggressive outward nominal action
        assert res["verified_safe"] is True
        u_f = np.asarray(res["u_filtered"], dtype=np.float64)
        x = x + dt * u_f
        assert ell_barrier.h(x) >= -1e-3

    # Also verify seamless integration with relative-degree-2 HighOrderBarrier on 4D double integrator
    A_di = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    B_di = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dyn_2 = LinearDynamics(A=A_di, B=B_di)
    hocbf_barrier = HighOrderBarrier(ell_barrier, alpha1=2.0, alpha2=3.0)
    cbf_2 = StateCBFSafetyFilter(
        dynamics=dyn_2,
        barriers=[hocbf_barrier],
        u_min=[-6.0, -6.0],
        u_max=[6.0, 6.0],
    )
    x4 = np.array([0.5, 0.1, 0.6, 0.2], dtype=np.float64)
    for _ in range(30):
        res2 = cbf_2.filter(x4, u_nominal=[4.0, 3.0])
        assert res2["verified_safe"] is True
        u2 = np.asarray(res2["u_filtered"], dtype=np.float64)
        x4 = x4 + 0.03 * (A_di @ x4 + B_di @ u2)
        assert hocbf_barrier.h(x4) >= -5e-3


def test_interval_arithmetic_soundness():
    rng = np.random.RandomState(99)
    verifier = FormalCBFVerifier()

    # Verify _interval_chebyshev_bounds covers all interior samples
    z_lo = np.array([-0.9, -0.2, 0.1])
    z_hi = np.array([0.8, 0.6, 0.95])
    t_lo, t_hi, dt_lo, dt_hi, d2t_lo, d2t_hi = verifier._interval_chebyshev_bounds(z_lo, z_hi, degree=5)
    for _ in range(50):
        z_s = rng.uniform(z_lo, z_hi)
        t_s, dt_s, d2t_s = chebyshev_with_derivatives(z_s, degree=5)
        assert np.all(t_s >= t_lo - 1e-12) and np.all(t_s <= t_hi + 1e-12)
        assert np.all(dt_s >= dt_lo - 1e-12) and np.all(dt_s <= dt_hi + 1e-12)
        assert np.all(d2t_s >= d2t_lo - 1e-12) and np.all(d2t_s <= d2t_hi + 1e-12)

    # Verify KANBarrierConstraint.interval_bounds covers all box samples
    barrier = KANBarrierConstraint(
        state_dim=2,
        degree=4,
        offset=0.1,
        grid_center=[0.2, -0.3],
        grid_scale=[1.5, 1.2],
    )
    barrier.kan.coeffs = rng.randn(1, 2, 5).astype(np.float32) * 0.5
    box_lo = np.array([-0.8, -0.9])
    box_hi = np.array([1.1, 0.7])
    h_lo, h_hi, gh_lo, gh_hi, hd_lo, hd_hi = barrier.interval_bounds(box_lo, box_hi)
    for _ in range(80):
        pt = rng.uniform(box_lo, box_hi)
        hv = barrier.h(pt)
        ghv = barrier.grad_h(pt)
        hdv = np.diag(barrier.hess_h(pt))
        assert h_lo - 1e-10 <= hv <= h_hi + 1e-10
        assert np.all(ghv >= gh_lo - 1e-10) and np.all(ghv <= gh_hi + 1e-10)
        assert np.all(hdv >= hd_lo - 1e-10) and np.all(hdv <= hd_hi + 1e-10)


def test_formal_verifier_detects_counterexamples_and_cegis_synthesizes_certified_2d():
    verifier = FormalCBFVerifier()

    # 1. Under-actuated / strong outward drift system where boundary at r=1.5 is uncertifiable
    # f(x) = 1.2 * x, g(x) = I_2, u in [-0.6, 0.6]^2 -> at ||x||=1.5, outward drift exceeds control!
    dyn_strong = LinearDynamics(A=1.2 * np.eye(2), B=np.eye(2))
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    r_grid = np.linspace(0.0, 2.2, 24)
    rr, tt = np.meshgrid(r_grid, theta)
    pts = np.column_stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()])
    rad = np.linalg.norm(pts, axis=1)

    # Initial samples place boundary around r ~ 1.5 (where drift=1.8 > control_max <= 0.85)
    X_safe_init = pts[rad <= 1.1]
    X_unsafe_init = pts[rad >= 1.9]

    barrier_fail = KANBarrierConstraint(
        state_dim=2,
        degree=2,
        alpha=0.2,
        grid_center=[0.0, 0.0],
        grid_scale=[2.2, 2.2],
    )
    barrier_fail.fit_from_samples(X_safe_init, X_unsafe_init, adapt_grid=False)
    ver_fail = verifier.verify(
        barrier_fail,
        dyn_strong,
        domain_lo=[-2.2, -2.2],
        domain_hi=[2.2, 2.2],
        u_min=[-0.6, -0.6],
        u_max=[0.6, 0.6],
        max_splits=256,
        eps_cert=1e-4,
    )
    assert ver_fail["certified"] is False
    assert len(ver_fail["counterexamples"]) > 0
    for x_ce in ver_fail["counterexamples"]:
        h_ce, slack_ce = verifier._exact_cbf_slack(
            barrier_fail, dyn_strong, x_ce, np.array([-0.6, -0.6]), np.array([0.6, 0.6])
        )
        assert abs(h_ce) <= 0.05 + 1e-9
        assert slack_ce < 1e-4

    # 2. Also detect insufficient-alpha counterexample on a point with h > 0
    # Construct a barrier where near x=(0.9, 0), h > 0 (h ~ 0.04), drift pushes outward,
    # and alpha=0.05 is too small while u is bounded to [-0.2, 0.2]^2.
    dyn_drift = LinearDynamics(A=0.6 * np.eye(2), B=np.eye(2))
    ver_low_alpha = verifier.verify(
        barrier_fail,
        dyn_drift,
        domain_lo=[-2.2, -2.2],
        domain_hi=[2.2, 2.2],
        u_min=[-0.2, -0.2],
        u_max=[0.2, 0.2],
        max_splits=256,
        eps_cert=1e-4,
    )
    assert ver_low_alpha["certified"] is False
    assert len(ver_low_alpha["counterexamples"]) > 0

    # 3. CEGIS synthesis on a verifiable 2D system with localized unrecoverable outer boundary:
    # f(x) = [0.75 * x_0, 0.25 * x_1], g(x) = I_2, u in [-1.0, 1.0]^2.
    # For |x_0| >= 1.34, drift 0.75 * |x_0| > 1.0 exceeds u_0 authority; for |x_0| <= 1.0, it is controllable.
    # Initial unsafe samples at |x| >= 1.8 allow the first fit to bulge past |x_0| = 1.34,
    # producing counterexamples that CEGIS incorporates until certified=True!
    dyn_verifiable = LinearDynamics(A=np.diag([0.75, 0.25]), B=np.eye(2))
    X_safe_cegis = pts[rad <= 0.7]
    X_unsafe_cegis = pts[rad >= 1.85]

    cegis_barrier = KANBarrierConstraint(
        state_dim=2,
        degree=2,
        alpha=0.8,
        grid_center=[0.0, 0.0],
        grid_scale=[2.0, 2.0],
    )
    cegis_res = verifier.cegis_synthesize(
        cegis_barrier,
        dyn_verifiable,
        X_safe=X_safe_cegis,
        X_unsafe=X_unsafe_cegis,
        domain_lo=[-2.0, -2.0],
        domain_hi=[2.0, 2.0],
        u_min=[-1.0, -1.0],
        u_max=[1.0, 1.0],
        max_rounds=6,
        max_splits=256,
        eps_cert=1e-4,
    )
    assert cegis_res["certified"] is True
    assert cegis_res["verification"]["certified"] is True
    assert cegis_res["verification"]["min_certified_slack"] >= 1e-4
    assert cegis_res["verification"]["boxes_checked"] > 1
    assert cegis_res["rounds"] <= 6
    # Safe core must remain strictly safe after CEGIS tightening
    assert np.all(cegis_barrier.h_batch(X_safe_cegis) > 0.0)
