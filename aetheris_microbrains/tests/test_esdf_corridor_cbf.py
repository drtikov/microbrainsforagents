"""Tests for Voxel ESDF, Safe Polytope Corridors, and Topological Homotopy Guidance."""

from __future__ import annotations

import numpy as np
import pytest

from aetheris_microbrains.core.capability_tokens import (
    CapabilityCaveat,
    authorize_physical_action,
    mint_root_capability,
)
from aetheris_microbrains.core.esdf_corridor_cbf import (
    TopologicalHomotopyGuide,
    VoxelESDFGrid,
    VoxelSDFBarrier,
    _compute_3d_esdf_grid,
    _edt_1d,
    inflate_safe_polytope,
)
from aetheris_microbrains.core.state_cbf_filter import (
    ControlAffineDynamics,
    HighOrderBarrier,
    StateCBFSafetyFilter,
)


def _single_integrator_3d() -> ControlAffineDynamics:
    return ControlAffineDynamics(
        state_dim=3,
        control_dim=3,
        f=lambda x: np.zeros(3, dtype=np.float64),
        g=lambda x: np.eye(3, dtype=np.float64),
    )


def _double_integrator_3d() -> ControlAffineDynamics:
    def f(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        return np.concatenate([x_arr[3:6], np.zeros(3, dtype=np.float64)])

    G = np.vstack([np.zeros((3, 3), dtype=np.float64), np.eye(3, dtype=np.float64)])
    return ControlAffineDynamics(state_dim=6, control_dim=3, f=f, g=lambda x: G)


def test_edt_1d_and_3d_signed_interior() -> None:
    # 1D exact check against brute-force O(n^2)
    f = np.array([np.inf, 2.0, np.inf, np.inf, 0.5, np.inf, np.inf], dtype=np.float64)
    spacing = 0.25
    got = _edt_1d(f, spacing=spacing)
    n = len(f)
    expected = np.empty(n, dtype=np.float64)
    for q in range(n):
        best = np.inf
        for r in range(n):
            if np.isfinite(f[r]):
                best = min(best, ((q - r) * spacing) ** 2 + f[r])
        expected[q] = best
    np.testing.assert_allclose(got, expected, atol=1e-12)

    # 3D signed distance: positive outside, negative inside solid block
    occ = np.zeros((11, 11, 11), dtype=bool)
    occ[3:8, 3:8, 3:8] = True
    sdf = _compute_3d_esdf_grid(occ, voxel_size=0.2)
    assert sdf[5, 5, 5] < -0.5  # Deep interior (3 voxels from free -> -0.6)
    assert np.isclose(sdf[5, 5, 5], -0.6, atol=1e-9)
    assert np.isclose(sdf[3, 5, 5], -0.2, atol=1e-9)  # Boundary occupied voxel
    assert np.isclose(sdf[2, 5, 5], 0.2, atol=1e-9)   # Adjacent free voxel
    assert np.all(sdf[occ] < 0.0)
    assert np.all(sdf[~occ] > 0.0)


def test_voxel_esdf_grid_accuracy_and_trilinear_gradient() -> None:
    # Point cloud forming a solid cluster / obstacle block around (1.5, 1.5, 1.5)
    xs = np.linspace(1.25, 1.75, 6)
    ys = np.linspace(1.25, 1.75, 6)
    zs = np.linspace(1.25, 1.75, 6)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    voxel_size = 0.1
    grid = VoxelESDFGrid.from_point_cloud(
        points=pts,
        origin=(0.0, 0.0, 0.0),
        voxel_size=voxel_size,
        grid_shape=(30, 30, 30),
        inflation_radius=0.0,
    )

    # Check interior is strictly negative
    d_inside, _ = grid.query_distance_and_grad((1.5, 1.5, 1.5))
    assert d_inside < 0.0

    # Query points inside trilinear cell interiors and compare with brute-force NN distance
    queries = np.array(
        [
            [0.53, 1.52, 1.48],
            [2.42, 1.51, 1.50],
            [1.47, 2.63, 1.55],
            [2.33, 2.37, 2.34],
        ],
        dtype=np.float64,
    )
    for q in queries:
        d_sdf, grad = grid.query_distance_and_grad(q)
        diffs = q - pts
        nn_idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
        d_exact = float(np.linalg.norm(diffs[nn_idx]))

        # Voxel quantization bound is <= sqrt(3) * voxel_size
        assert abs(d_sdf - d_exact) <= np.sqrt(3.0) * voxel_size + 1e-6
        assert d_sdf > 0.0

        # Analytical gradient matches central finite differences of trilinear SDF
        eps = 1e-5
        fd_grad = np.zeros(3, dtype=np.float64)
        for k in range(3):
            ei = np.zeros(3, dtype=np.float64)
            ei[k] = eps
            dp, _ = grid.query_distance_and_grad(q + ei)
            dm, _ = grid.query_distance_and_grad(q - ei)
            fd_grad[k] = (dp - dm) / (2.0 * eps)
        np.testing.assert_allclose(grad, fd_grad, atol=1e-6)

        # Gradient points directly away from the nearest obstacle point
        dir_away = diffs[nn_idx] / d_exact
        cos_sim = float(np.dot(grad / np.linalg.norm(grad), dir_away))
        assert cos_sim > 0.95


def test_voxel_sdf_barrier_cbf_navigates_l_wall() -> None:
    # Build an L-shaped wall point cloud in [0, 4] x [0, 4] x [0, 1]
    wall_pts: list[np.ndarray] = []
    # Vertical leg along x=2.0, y in [1.5, 2.7]
    for y in np.linspace(1.5, 2.7, 25):
        for z in np.linspace(0.1, 0.9, 9):
            wall_pts.append(np.array([2.0, y, z]))
    # Horizontal leg along y=2.7, x in [1.2, 2.0]
    for x in np.linspace(1.2, 2.0, 17):
        for z in np.linspace(0.1, 0.9, 9):
            wall_pts.append(np.array([x, 2.7, z]))
    pts = np.vstack(wall_pts)

    grid = VoxelESDFGrid.from_point_cloud(
        points=pts,
        origin=(0.0, 0.0, 0.0),
        voxel_size=0.1,
        grid_shape=(40, 40, 10),
    )

    r_robot = 0.25
    barrier = VoxelSDFBarrier(grid, robot_radius=r_robot, alpha=2.5, state_slice=slice(0, 3))

    # Check inflate(delta) shifts barrier by -delta
    inflated = barrier.inflate(0.15)
    x_probe = np.array([1.0, 1.5, 0.5], dtype=np.float64)
    assert np.isclose(inflated.h(x_probe), barrier.h(x_probe) - 0.15, atol=1e-9)

    # Also verify HighOrderBarrier compatibility on 6D state [p, v]
    ho_barrier = HighOrderBarrier(barrier, alpha1=2.0, alpha2=2.0)
    ho_filter = StateCBFSafetyFilter(
        _double_integrator_3d(),
        [ho_barrier],
        u_min=[-5.0, -5.0, -5.0],
        u_max=[5.0, 5.0, 5.0],
    )
    res_ho = ho_filter.filter(np.array([1.0, 1.5, 0.5, 0.3, 0.0, 0.0]), [0.5, 0.0, 0.0])
    assert res_ho["verified_safe"] is True

    # Closed-loop simulation around the L-wall with StateCBFSafetyFilter
    cbf = StateCBFSafetyFilter(
        _single_integrator_3d(),
        [barrier],
        u_min=[-1.5, -1.5, -1.5],
        u_max=[1.5, 1.5, 1.5],
    )
    x = np.array([1.0, 1.8, 0.5], dtype=np.float64)
    goal = np.array([3.2, 1.8, 0.5], dtype=np.float64)
    dt = 0.05
    min_sdf = np.inf

    for _ in range(250):
        d_cur, _ = grid.query_distance_and_grad(x)
        min_sdf = min(min_sdf, d_cur)
        to_goal = goal - x
        if np.linalg.norm(to_goal) < 0.15:
            break
        # Lateral clearance guidance around the y=1.5 end while upstream of the wall
        lateral_bias = np.array([0.0, -1.0, 0.0]) if x[0] < 2.1 else np.zeros(3)
        u_nom = 1.0 * to_goal + lateral_bias
        out = cbf.filter(x, u_nom)
        assert out["verified_safe"] is True
        u_safe = np.asarray(out["u_filtered"], dtype=np.float64)
        x = x + dt * u_safe

    assert min_sdf >= r_robot - 1e-3
    assert np.linalg.norm(x - goal) < 0.35


def test_inflate_safe_polytope_cbf_and_capability_geofence() -> None:
    rng = np.random.default_rng(42)
    seed = np.array([0.2, -0.1, 0.5], dtype=np.float64)

    # Ring/wall obstacle cloud surrounding seed at distance >= 1.2
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    ring_pts = np.column_stack([
        seed[0] + 1.4 * np.cos(angles),
        seed[1] + 1.4 * np.sin(angles),
        np.full_like(angles, seed[2]),
    ])
    extra_obs = np.array([
        [0.2, -0.1, 1.4],
        [0.2, -0.1, -0.3],
        [1.1, 0.6, 0.5],
        [-0.7, -0.8, 0.4],
    ]) + rng.normal(0.0, 0.01, size=(4, 3))
    obs_pts = np.vstack([ring_pts, extra_obs])

    corridor = inflate_safe_polytope(
        seed_point=seed,
        obstacle_points=obs_pts,
        bounding_min=[-2.0, -2.0, -1.0],
        bounding_max=[2.0, 2.0, 2.0],
        max_planes=16,
        safety_margin=0.05,
    )

    A = corridor["A"]
    b = corridor["b"]
    barriers = corridor["halfplane_barriers"]
    geofence = corridor["geofence_polytope"]

    assert A.ndim == 2 and A.shape[1] == 3
    assert A.shape[0] <= 16
    assert b.shape == (A.shape[0],)
    assert len(barriers) == A.shape[0]
    assert len(geofence) == A.shape[0]

    # Seed point strictly inside polytope and all HalfPlaneBarriers
    assert np.all(A @ seed < b)
    assert all(bar.h(seed) > 0.0 for bar in barriers)

    # Every obstacle point is strictly excluded
    for q in obs_pts:
        assert np.any(A @ q > b), f"Obstacle point {q} was not excluded"
        assert any(bar.h(q) < 0.0 for bar in barriers)

    # Works inside StateCBFSafetyFilter
    cbf = StateCBFSafetyFilter(
        _single_integrator_3d(),
        barriers,
        u_min=[-2.0, -2.0, -2.0],
        u_max=[2.0, 2.0, 2.0],
    )
    filt = cbf.filter(seed, [1.5, 0.5, 0.0])
    assert filt["verified_safe"] is True

    # Works directly with CapabilityCaveat.geofence_polytope and authorize_physical_action
    secret = b"aetheris-esdf-test-secret"
    cav = CapabilityCaveat(
        allowed_actuators=("base_vel",),
        geofence_polytope=geofence,
    )
    token = mint_root_capability(secret, subject="rover-1", caveat=cav, now=1000.0)

    auth_seed = authorize_physical_action(
        token,
        secret,
        position=tuple(seed.tolist()),
        actuator="base_vel",
        now=1001.0,
    )
    assert auth_seed["authorized"] is True
    assert auth_seed["effective_limits"]["geofence_halfspaces"] == len(geofence)

    for q in obs_pts[:10]:
        auth_obs = authorize_physical_action(
            token,
            secret,
            position=tuple(q.tolist()),
            actuator="base_vel",
            now=1001.0,
        )
        assert auth_obs["authorized"] is False
        assert auth_obs["reason"] == "geofence_violation"


def test_topological_homotopy_guide_betti1_and_u_trap_rescue() -> None:
    # 1. Open space -> single straight homotopy class -> beta_1 == 0
    empty_grid = VoxelESDFGrid.from_point_cloud(
        points=np.zeros((0, 3), dtype=np.float64),
        origin=(-1.0, -3.0, 0.0),
        voxel_size=0.2,
        grid_shape=(35, 30, 5),
    )
    empty_guide = TopologicalHomotopyGuide(empty_grid, min_clearance=0.2)
    empty_res = empty_guide.compute_homotopy_guidance(
        start=(0.0, 0.0, 0.5),
        goal=(4.5, 0.0, 0.5),
        max_branches=3,
    )
    assert empty_res["betti_1"] == 0
    assert empty_res["has_topological_loop"] is False
    assert len(empty_res["candidate_paths"]) == 1

    # 2. Obstacle-ring / pillar between start and goal -> beta_1 >= 1 and 2 distinct branches
    ring_angles = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    ring_pts = []
    for z in np.linspace(0.1, 0.9, 5):
        for ang in ring_angles:
            ring_pts.append([2.2 + 0.7 * np.cos(ang), 0.0 + 0.7 * np.sin(ang), z])
    ring_grid = VoxelESDFGrid.from_point_cloud(
        points=np.asarray(ring_pts, dtype=np.float64),
        origin=(-1.0, -3.0, 0.0),
        voxel_size=0.2,
        grid_shape=(35, 30, 5),
    )
    ring_guide = TopologicalHomotopyGuide(ring_grid, min_clearance=0.2)
    ring_res = ring_guide.compute_homotopy_guidance(
        start=(0.2, 0.0, 0.5),
        goal=(4.2, 0.0, 0.5),
        max_branches=3,
    )
    assert ring_res["betti_1"] >= 1
    assert ring_res["has_topological_loop"] is True
    assert len(ring_res["candidate_paths"]) >= 2

    # 3. Non-convex U-trap environment rescuing StateCBFSafetyFilter from deadlock
    u_pts = []
    # Back wall at x = 2.4, y in [-1.2, 1.2]
    for y in np.linspace(-1.2, 1.2, 25):
        for z in np.linspace(0.1, 0.9, 5):
            u_pts.append([2.4, y, z])
    # Top wing at y = 1.2, x in [0.6, 2.4]
    for x in np.linspace(0.6, 2.4, 19):
        for z in np.linspace(0.1, 0.9, 5):
            u_pts.append([x, 1.2, z])
    # Bottom wing at y = -1.2, x in [0.6, 2.4]
    for x in np.linspace(0.6, 2.4, 19):
        for z in np.linspace(0.1, 0.9, 5):
            u_pts.append([x, -1.2, z])

    u_grid = VoxelESDFGrid.from_point_cloud(
        points=np.asarray(u_pts, dtype=np.float64),
        origin=(-1.5, -3.0, 0.0),
        voxel_size=0.2,
        grid_shape=(40, 30, 5),
    )

    r_robot = 0.25
    start = np.array([1.4, 0.0, 0.5], dtype=np.float64)
    goal = np.array([4.2, 0.0, 0.5], dtype=np.float64)
    barrier = VoxelSDFBarrier(u_grid, robot_radius=r_robot, alpha=2.0, state_slice=slice(0, 3))
    cbf = StateCBFSafetyFilter(
        _single_integrator_3d(),
        [barrier],
        u_min=[-1.5, -1.5, -1.5],
        u_max=[1.5, 1.5, 1.5],
    )

    # Confirm naive straight-line attraction stalls in the U-trap
    x_naive = start.copy()
    dt = 0.05
    for _ in range(150):
        out = cbf.filter(x_naive, 1.5 * (goal - x_naive))
        x_naive = x_naive + dt * np.asarray(out["u_filtered"], dtype=np.float64)
    assert np.linalg.norm(x_naive - goal) > 1.5  # Trapped inside U!

    # Now use TopologicalHomotopyGuide to detect loop and extract rescue waypoints
    guide = TopologicalHomotopyGuide(u_grid, min_clearance=r_robot + 0.05)
    guidance = guide.compute_homotopy_guidance(start=start, goal=goal, max_branches=3)
    assert guidance["betti_1"] >= 1
    assert guidance["has_topological_loop"] is True
    assert len(guidance["candidate_paths"]) >= 2

    # Branches go around opposite sides (positive-y and negative-y wings)
    max_ys = [max(pt[1] for pt in path) for path in guidance["candidate_paths"]]
    min_ys = [min(pt[1] for pt in path) for path in guidance["candidate_paths"]]
    assert any(my > 1.3 for my in max_ys)
    assert any(my < -1.3 for my in min_ys)

    # Track the primary homotopy branch through StateCBFSafetyFilter to reach goal safely
    waypoints = [np.asarray(wp, dtype=np.float64) for wp in guidance["waypoints"]]
    x_rescued = start.copy()
    wp_idx = 1 if len(waypoints) > 1 else 0
    min_sdf_rescued = np.inf

    for _ in range(400):
        d_cur, _ = u_grid.query_distance_and_grad(x_rescued)
        min_sdf_rescued = min(min_sdf_rescued, d_cur)
        if wp_idx < len(waypoints) - 1 and np.linalg.norm(x_rescued - waypoints[wp_idx]) < 0.3:
            wp_idx += 1
        target = waypoints[wp_idx]
        if np.linalg.norm(x_rescued - goal) < 0.25:
            break
        u_nom = 2.0 * (target - x_rescued)
        speed = float(np.linalg.norm(u_nom))
        if speed > 1.5:
            u_nom = u_nom * (1.5 / speed)
        out = cbf.filter(x_rescued, u_nom)
        assert out["verified_safe"] is True
        x_rescued = x_rescued + dt * np.asarray(out["u_filtered"], dtype=np.float64)

    assert min_sdf_rescued >= r_robot - 1e-3
    assert np.linalg.norm(x_rescued - goal) < 0.35
