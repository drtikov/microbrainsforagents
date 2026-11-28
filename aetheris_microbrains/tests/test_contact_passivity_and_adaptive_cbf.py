"""Tests for Port-Hamiltonian energy tank passivity, contact manifold learning,
L1 disturbance observer + robust adaptive CBF, and spectral-CBF online plasticity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_AETHER_DIR = _PKG_DIR.parent

for p in [str(_PKG_DIR), str(_AETHER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

from aetheris_microbrains.core.hybrid_contact_passivity import (
    ContactManifoldLearner,
    EnergyTankPassivityFilter,
)
from aetheris_microbrains.core.liquid_state_reservoir import LiquidStateReservoir
from aetheris_microbrains.core.online_hebbian_adaptor import OnlineHebbianAdaptor
from aetheris_microbrains.core.robust_adaptive_cbf import (
    L1DisturbanceObserver,
    RobustAdaptiveCBFFilter,
)
from aetheris_microbrains.core.sensorimotor_loop import (
    ReservoirReflexLayer,
    SensorimotorLoop,
)
from aetheris_microbrains.core.state_cbf_filter import (
    ControlAffineDynamics,
    HalfPlaneBarrier,
    HighOrderBarrier,
    LinearDynamics,
    StateCBFSafetyFilter,
)


# ---------------------------------------------------------------------------
# 1. EnergyTankPassivityFilter — Stiff Wall Contact (K_env = 5000 N/m)
# ---------------------------------------------------------------------------

class TestEnergyTankPassivityFilter:
    def test_stiff_wall_vla_stiffness_pumping_passivity_certified(self):
        """Against a K_env = 5000 N/m stiff wall, rapid VLA stiffness pumping dK_p/dt > 0
        causes unbounded mechanical energy blowup without an energy tank, whereas
        EnergyTankPassivityFilter keeps tank_energy >= E_min and eliminates blowup.
        """
        dt = 0.001
        mass = 1.0
        k_env = 5000.0
        x_d = 0.01  # desired target 1 cm inside the unilateral stiff wall at x >= 0
        k_d = 1.2
        steps = 2500

        def vla_pumped_command(x_pos: float, dx_vel: float) -> tuple[float, float]:
            # Adversarial stiffness pumping: high stiffness when moving toward x_d from afar,
            # soft stiffness on wall rebound -> injects (1/2) x_tilde^2 dK_p > 0 every cycle.
            x_tilde = x_pos - x_d
            approaching = (x_tilde * dx_vel) < 0.0
            k_p_cmd = 2400.0 if approaching else 15.0
            f_ff_cmd = 12.0 * np.sign(dx_vel) if abs(dx_vel) > 1e-3 else 0.0
            return k_p_cmd, float(f_ff_cmd)

        # Rollout A: Unfiltered variable impedance -> unstable contact energy blowup
        x_raw = -0.06
        dx_raw = 0.0
        energies_unfiltered = []
        for _ in range(steps):
            kp_cmd, f_ff_cmd = vla_pumped_command(x_raw, dx_raw)
            u_raw = -kp_cmd * (x_raw - x_d) - k_d * dx_raw + f_ff_cmd
            f_wall = -k_env * max(0.0, x_raw)
            acc = (u_raw + f_wall) / mass
            dx_raw = dx_raw + dt * acc
            x_raw = x_raw + dt * dx_raw
            mech_energy = 0.5 * mass * (dx_raw ** 2) + 0.5 * k_env * (max(0.0, x_raw) ** 2)
            energies_unfiltered.append(mech_energy)

        # Rollout B: EnergyTankPassivityFilter -> unconditionally passive contact
        e_min = 0.05
        e_init = 0.60
        e_max = 1.50
        tank_filter = EnergyTankPassivityFilter(
            dt=dt,
            e_init=e_init,
            e_min=e_min,
            e_max=e_max,
            alpha_tank=0.4,
            dissipation_rate=0.9,
            k_p_init=15.0,
        )

        x_safe = -0.06
        dx_safe = 0.0
        energies_filtered = []
        tank_trace = []
        gammas = []
        for _ in range(steps):
            kp_cmd, f_ff_cmd = vla_pumped_command(x_safe, dx_safe)
            kp_s, kd_s, u_vec, e_tank, gamma, certified = tank_filter.filter(
                x=[x_safe],
                dx=[dx_safe],
                K_p=kp_cmd,
                K_d=k_d,
                x_d=[x_d],
                F_ff=[f_ff_cmd],
            )
            assert certified is True
            assert e_tank >= e_min - 1e-12
            assert e_tank <= e_max + 1e-12
            assert 0.0 <= gamma <= 1.0
            assert kp_s >= 0.0 and kd_s == pytest.approx(k_d)

            f_wall = -k_env * max(0.0, x_safe)
            acc = (float(u_vec[0]) + f_wall) / mass
            dx_safe = dx_safe + dt * acc
            x_safe = x_safe + dt * dx_safe
            mech_energy = 0.5 * mass * (dx_safe ** 2) + 0.5 * k_env * (max(0.0, x_safe) ** 2)
            energies_filtered.append(mech_energy)
            tank_trace.append(e_tank)
            gammas.append(gamma)

        # Unfiltered energy blows up violently; tank-filtered energy stays strictly bounded and decays
        assert max(energies_unfiltered) > 50.0, f"Expected unfiltered blowup, got {max(energies_unfiltered)}"
        assert max(energies_filtered) < 1.2, f"Filtered energy grew unexpectedly: {max(energies_filtered)}"
        assert max(energies_unfiltered) > 40.0 * max(energies_filtered)
        assert energies_filtered[-1] < e_init
        assert min(tank_trace) >= e_min
        assert min(gammas) < 0.5  # valve actively throttled unsafe stiffness/wrench injections
        assert tank_filter.interventions > 10

    def test_3d_matrix_impedance_and_softening_always_permitted(self):
        filt = EnergyTankPassivityFilter(dt=0.002, e_init=0.06, e_min=0.05, e_max=1.0, alpha_tank=0.5)
        K_p_high = np.diag([800.0, 800.0, 800.0])
        K_d_mat = np.diag([10.0, 10.0, 10.0])

        # Initialize K_p_prev at K_p_high with zero jump
        kp0, _, _, e0, g0, cert0 = filt.filter(
            x=[0.1, -0.1, 0.05],
            dx=[0.0, 0.0, 0.0],
            K_p=K_p_high,
            K_d=K_d_mat,
            x_d=[0.0, 0.0, 0.0],
        )
        assert cert0 is True
        assert g0 == pytest.approx(1.0)
        assert np.allclose(kp0, K_p_high)

        # Softening (Delta K_p < 0) injects negative power -> permitted with gamma_valve == 1.0
        K_p_low = np.diag([50.0, 50.0, 50.0])
        kp1, _, _, e1, g1, cert1 = filt.filter(
            x=[0.1, -0.1, 0.05],
            dx=[0.0, 0.0, 0.0],
            K_p=K_p_low,
            K_d=K_d_mat,
            x_d=[0.0, 0.0, 0.0],
        )
        assert cert1 is True
        assert g1 == pytest.approx(1.0)
        assert np.allclose(kp1, K_p_low)
        assert e1 >= filt.e_min

        # Massive stiffening jump from low tank reserve -> throttled cleanly to keep E_tank >= E_min
        kp2, _, u2, e2, g2, cert2 = filt.filter(
            x=[0.2, -0.2, 0.1],
            dx=[0.0, 0.0, 0.0],
            K_p=np.diag([2000.0, 2000.0, 2000.0]),
            K_d=K_d_mat,
            x_d=[0.0, 0.0, 0.0],
            F_ff=[50.0, 0.0, 0.0],
        )
        assert cert2 is True
        assert e2 >= filt.e_min
        assert 0.0 <= g2 < 0.2
        assert np.all(np.diag(kp2) < 500.0)
        assert np.all(np.isfinite(u2))


# ---------------------------------------------------------------------------
# 2. ContactManifoldLearner — 2D Plane, 1D Drawer, Revolute Hinge, Free Space
# ---------------------------------------------------------------------------

class TestContactManifoldLearner:
    def test_recovers_2d_plane_normal_from_noisy_table_wiping(self):
        rng = np.random.RandomState(42)
        learner = ContactManifoldLearner(f_contact_min=2.0, ema_decay=0.92, min_contact_samples=8)

        # Tilted table plane with ground-truth unit normal n_true
        n_raw = np.array([0.15, -0.25, 0.956], dtype=np.float64)
        n_true = n_raw / np.linalg.norm(n_raw)
        # Orthonormal tangent basis (t1, t2) spanning the tilted table
        t1 = np.cross(n_true, np.array([1.0, 0.0, 0.0]))
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n_true, t1)

        # Free-space approach samples below contact threshold must remain free_space
        for _ in range(5):
            info_free = learner.observe([0.1, 0.2, -0.3], [0.2, -0.1, 0.3])
            assert info_free["mode"] == "free_space"

        # Noisy Lissajous wiping velocity on the tilted plane + normal reaction + Coulomb friction
        last_info = None
        for k in range(120):
            v_clean = 0.35 * np.cos(0.17 * k) * t1 + 0.30 * np.sin(0.29 * k) * t2
            v_noisy = v_clean + rng.normal(scale=0.015, size=3)
            f_clean = 14.0 * n_true - 2.0 * (v_clean / max(1e-6, np.linalg.norm(v_clean)))
            f_noisy = f_clean + rng.normal(scale=0.4, size=3)
            last_info = learner.observe(v_noisy, f_noisy)

        assert last_info is not None
        assert learner.mode == "planar_surface"
        n_est = learner.surface_normal
        assert n_est is not None
        alignment = float(np.dot(n_est, n_true))
        assert alignment > 0.99, f"Expected accurate plane normal, got dot={alignment}"

        # Compliant projection strips normal motion and caps normal contact force
        v_nom = 0.4 * t1 - 0.5 * n_true  # tries to dig into the table
        proj = learner.project_compliant_command(v_nom, f_nom_normal=25.0, max_normal_force=12.0)
        assert proj["mode"] == "planar_surface"
        assert abs(float(np.dot(proj["v_compliant"], n_est))) < 1e-9
        assert float(np.dot(proj["v_compliant"], t1)) == pytest.approx(0.4, abs=0.03)
        assert proj["normal_force_scalar"] == pytest.approx(12.0, abs=1e-9)
        assert proj["orthogonality_error"] < 1e-9

    def test_recovers_1d_drawer_axis_from_noisy_stream(self):
        rng = np.random.RandomState(108)
        learner = ContactManifoldLearner(f_contact_min=2.0, ema_decay=0.93, min_contact_samples=8)

        # Arbitrary 3D drawer slide axis
        axis_true = np.array([0.6, -0.8, 0.0], dtype=np.float64)
        lateral = np.array([0.8, 0.6, 0.0], dtype=np.float64)

        for k in range(100):
            speed = 0.32 * (1.0 if (k // 25) % 2 == 0 else -0.8)
            v_noisy = speed * axis_true + rng.normal(scale=0.015, size=3)
            # Off-axis binding reaction + axial slide resistance
            f_noisy = 8.0 * lateral + np.array([0.0, 0.0, 5.0]) - 3.0 * np.sign(speed) * axis_true + rng.normal(scale=0.3, size=3)
            learner.observe(v_noisy, f_noisy)

        assert learner.mode == "prismatic_1d"
        axis_est = learner.motion_axis
        assert axis_est is not None
        cos_sim = abs(float(np.dot(axis_est, axis_true)))
        assert cos_sim > 0.995, f"Expected accurate 1D drawer axis, got |dot|={cos_sim}"

        # Projecting off-axis nominal velocity removes lateral jamming components completely
        v_nom = 0.5 * axis_true + 0.4 * lateral + np.array([0.0, 0.0, 0.3])
        proj = learner.project_compliant_command(v_nom, f_nom_normal=0.0, max_normal_force=10.0)
        assert proj["mode"] == "prismatic_1d"
        v_comp = proj["v_compliant"]
        assert abs(float(np.dot(v_comp, lateral))) < 0.03
        assert abs(float(v_comp[2])) < 0.03
        assert float(np.dot(v_comp, axis_true)) == pytest.approx(0.5, abs=0.02)
        assert proj["normal_force_scalar"] == pytest.approx(0.0, abs=1e-9)

    def test_recovers_revolute_hinge_axis_and_lever_arm(self):
        rng = np.random.RandomState(77)
        learner = ContactManifoldLearner(f_contact_min=1.5, ema_decay=0.92, min_contact_samples=10)

        pivot_true = np.array([0.2, 0.5, 0.0], dtype=np.float64)
        hinge_true = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        radius_true = 0.40
        omega = 1.2

        thetas = np.linspace(0.1, 1.1, 45)
        for th in thetas:
            r_vec = radius_true * np.array([np.cos(th), np.sin(th), 0.0])
            pos = pivot_true + r_vec + rng.normal(scale=0.002, size=3)
            vel = omega * np.cross(hinge_true, r_vec) + rng.normal(scale=0.008, size=3)
            # Radial hinge bearing reaction force in XY plane
            force = -12.0 * (r_vec / radius_true) + rng.normal(scale=0.3, size=3)
            learner.observe(vel, force, position_3d=pos)

        assert learner.mode == "revolute_hinge"
        assert learner.hinge_axis is not None
        assert float(np.dot(learner.hinge_axis, hinge_true)) > 0.98
        assert learner.lever_arm is not None
        assert float(np.linalg.norm(learner.lever_arm)) == pytest.approx(radius_true, abs=0.03)

        proj = learner.project_compliant_command([0.3, 0.3, 0.2], f_nom_normal=15.0, max_normal_force=8.0)
        assert proj["mode"] == "revolute_hinge"
        assert proj["normal_force_scalar"] == pytest.approx(8.0, abs=1e-9)
        assert proj["orthogonality_error"] < 1e-9


# ---------------------------------------------------------------------------
# 3. L1DisturbanceObserver + RobustAdaptiveCBFFilter — Heavy Payload Drop
# ---------------------------------------------------------------------------

class TestRobustAdaptiveCBF:
    def test_payload_drop_relative_degree_1_floor_barrier_safe(self):
        """2D Cartesian velocity-controlled end-effector x = [px, y] under sudden heavy
        downward payload drop d = [0, -8.0]: standard StateCBFSafetyFilter violates
        floor y >= 0, whereas RobustAdaptiveCBFFilter estimates d_hat within ms and
        maintains y(t) >= 0 throughout.
        """
        dt = 0.001
        dyn = ControlAffineDynamics(state_dim=2, control_dim=2)
        floor_barrier = HalfPlaneBarrier(a=[0.0, 1.0], b=0.0, alpha=4.0)  # h(x) = y >= 0
        u_min = [-15.0, -15.0]
        u_max = [15.0, 15.0]
        d_payload = np.array([0.0, -8.0], dtype=np.float64)
        u_nom = [0.5, -1.5]  # nominal downward approach

        # A. Standard uncompensated CBF crashes through the floor y = 0
        std_cbf = StateCBFSafetyFilter(dyn, [floor_barrier], u_min=u_min, u_max=u_max)
        x_std = np.array([0.0, 0.12], dtype=np.float64)
        y_hist_std = []
        for _ in range(300):
            out_s = std_cbf.filter(x_std, u_nom)
            u_s = np.asarray(out_s["u_filtered"], dtype=np.float64)
            x_std = x_std + dt * (dyn.f(x_std) + dyn.g(x_std) @ u_s + d_payload)
            y_hist_std.append(float(x_std[1]))

        assert min(y_hist_std) < -0.5, f"Standard CBF should violate floor under d=[0,-8], got min_y={min(y_hist_std)}"

        # B. RobustAdaptiveCBFFilter estimates d_hat within milliseconds and keeps y(t) >= 0
        observer = L1DisturbanceObserver(
            dyn,
            dt=dt,
            predictor_gain=80.0,
            filter_bandwidth=400.0,
            disturbance_lipschitz=2.0,
            initial_disturbance_bound=2.0,
        )
        rob_cbf = RobustAdaptiveCBFFilter(
            dyn,
            [floor_barrier],
            observer=observer,
            u_min=u_min,
            u_max=u_max,
            dt=dt,
        )

        x_rob = np.array([0.0, 0.12], dtype=np.float64)
        y_hist_rob = [float(x_rob[1])]
        d_hat_at_12ms = None
        last_receipt = None
        for step_idx in range(300):
            out_r = rob_cbf.filter(x_rob, u_nom)
            last_receipt = out_r
            assert out_r["verified_safe"] is True
            assert {"d_hat", "disturbance_bound", "verified_safe", "u_filtered"} <= set(out_r)

            if step_idx >= 1:
                # Certified analytical bound covers true estimation error at every post-drop tick
                est_err = float(np.linalg.norm(d_payload - np.asarray(out_r["d_hat"], dtype=np.float64)))
                assert out_r["disturbance_bound"] >= est_err - 1e-6

            if step_idx == 12:  # 12 ms after drop
                d_hat_at_12ms = np.asarray(out_r["d_hat"], dtype=np.float64)

            u_r = np.asarray(out_r["u_filtered"], dtype=np.float64)
            x_rob = x_rob + dt * (dyn.f(x_rob) + dyn.g(x_rob) @ u_r + d_payload)
            y_hist_rob.append(float(x_rob[1]))

        assert d_hat_at_12ms is not None
        assert np.allclose(d_hat_at_12ms, d_payload, atol=0.15)
        assert last_receipt is not None
        assert np.allclose(last_receipt["d_hat"], d_payload, atol=1e-3)
        assert min(y_hist_rob) >= 0.0, f"RobustAdaptiveCBFFilter violated floor: min_y={min(y_hist_rob)}"

    def test_payload_drop_relative_degree_2_hocbf_floor_barrier_safe(self):
        """Second-order vertical axis x = [y, y_dot] under heavy downward payload
        acceleration drop d = [0, -8.0] with HighOrderBarrier (relative degree 2).
        """
        dt = 0.001
        A = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        B = np.array([[0.0], [1.0]], dtype=np.float64)
        dyn = LinearDynamics(A, B)
        hocbf_floor = HighOrderBarrier(
            HalfPlaneBarrier(a=[1.0, 0.0], b=0.0),
            alpha1=5.0,
            alpha2=6.0,
        )
        d_payload = np.array([0.0, -8.0], dtype=np.float64)
        u_min, u_max = [-25.0], [25.0]

        # Standard HOCBF violates y >= 0 under unmodeled gravity drop d = [0, -8.0]
        std_hocbf = StateCBFSafetyFilter(dyn, [hocbf_floor], u_min=u_min, u_max=u_max)
        x_std = np.array([0.18, -0.4], dtype=np.float64)
        min_y_std = float(x_std[0])
        for _ in range(600):
            res_s = std_hocbf.filter(x_std, [-2.0])
            u_s = np.asarray(res_s["u_filtered"], dtype=np.float64)
            x_std = x_std + dt * (dyn.f(x_std) + dyn.g(x_std) @ u_s + d_payload)
            min_y_std = min(min_y_std, float(x_std[0]))
        assert min_y_std < -0.15

        # RobustAdaptiveCBFFilter (relative degree 2) compensates d = [0, -8.0] and keeps y(t) >= 0
        obs = L1DisturbanceObserver(
            dyn,
            dt=dt,
            predictor_gain=80.0,
            filter_bandwidth=400.0,
            disturbance_lipschitz=2.0,
            initial_disturbance_bound=2.0,
        )
        rob_hocbf = RobustAdaptiveCBFFilter(
            dyn,
            [hocbf_floor],
            observer=obs,
            u_min=u_min,
            u_max=u_max,
            dt=dt,
        )
        x_rob = np.array([0.18, -0.4], dtype=np.float64)
        min_y_rob = float(x_rob[0])
        for step_idx in range(450):
            res_r = rob_hocbf.filter(x_rob, [-2.0])
            assert res_r["verified_safe"] is True
            if step_idx == 15:
                assert np.allclose(res_r["d_hat"], d_payload, atol=0.1)
            u_r = np.asarray(res_r["u_filtered"], dtype=np.float64)
            x_rob = x_rob + dt * (dyn.f(x_rob) + dyn.g(x_rob) @ u_r + d_payload)
            min_y_rob = min(min_y_rob, float(x_rob[0]))

        assert min_y_rob >= 0.0, f"HOCBF RobustAdaptiveCBFFilter violated floor: min_y={min_y_rob}"


# ---------------------------------------------------------------------------
# 4. CBFSpectralBarrierEngine Bridged into Online Plasticity
# ---------------------------------------------------------------------------

class TestSpectralCBFOnlinePlasticity:
    @pytest.mark.parametrize("method", ["force", "rls"])
    def test_reservoir_reflex_layer_small_gain_bound_under_adversarial_spikes(self, method):
        plant_gain = 2.0
        expected_limit = 0.99 / plant_gain  # 0.495

        res_guarded = LiquidStateReservoir(reservoir_dim=32, input_dim=8)
        layer_guarded = ReservoirReflexLayer(
            reservoir=res_guarded,
            action_dim=2,
            method=method,
            plant_l2_gain=plant_gain,
        )
        assert layer_guarded.effective_spectral_limit == pytest.approx(expected_limit)

        res_unguarded = LiquidStateReservoir(reservoir_dim=32, input_dim=8)
        layer_unguarded = ReservoirReflexLayer(
            reservoir=res_unguarded,
            action_dim=2,
            method=method,
        )
        assert layer_unguarded.effective_spectral_limit is None

        loop = SensorimotorLoop(dt=0.005, reflex_fn=layer_guarded)
        rng = np.random.RandomState(2026)

        max_sigma_guarded = 0.0
        max_sigma_unguarded = 0.0
        projected_any = False

        for i in range(60):
            vec = rng.randn(8).astype(np.float32) * 2.5
            spike_target = [400.0 * ((-1.0) ** i), -350.0 * ((-1.0) ** i)]

            tick_out = loop.tick({"vector": vec, "target_action": spike_target})
            upd = tick_out["action"]["online_update"]
            if upd and upd.get("spectral_cbf", {}).get("projected"):
                projected_any = True

            s_g = float(np.linalg.svd(res_guarded.W_readout.astype(np.float64), compute_uv=False)[0])
            max_sigma_guarded = max(max_sigma_guarded, s_g)
            assert s_g <= expected_limit

            # Explicit .adapt() under huge spike must also preserve sigma_max <= 0.99 / 2.0
            adapt_out = layer_guarded.adapt([600.0, -600.0])
            s_g_adapt = float(np.linalg.svd(res_guarded.W_readout.astype(np.float64), compute_uv=False)[0])
            assert s_g_adapt <= expected_limit
            if adapt_out.get("spectral_cbf", {}).get("projected"):
                projected_any = True

            layer_unguarded({"vector": vec, "target_action": spike_target})
            s_u = float(np.linalg.svd(res_unguarded.W_readout.astype(np.float64), compute_uv=False)[0])
            max_sigma_unguarded = max(max_sigma_unguarded, s_u)

        assert projected_any is True
        assert max_sigma_guarded <= expected_limit
        assert max_sigma_unguarded > expected_limit * 5.0

    def test_online_hebbian_adaptor_enforces_svd_spectral_cbf_in_tmp_dir(self, tmp_path: Path):
        """Verify OnlineHebbianAdaptor strictly satisfies sigma_max(W) <= 0.99 / 2.0
        for both JSON and companion .npz weight stores using isolated tmp_path.
        """
        limit = 0.99 / 2.0
        adaptor = OnlineHebbianAdaptor(memory_dir=str(tmp_path))
        adaptor.learning_rate = 0.25  # aggressive plasticity step to stress spectral barrier
        rng = np.random.RandomState(99)

        # 1. JSON weight store brain in tmp_path
        w_init_json = (rng.randn(12, 16) * 0.05).astype(np.float32)
        json_meta = tmp_path / "probe_json_meta.json"
        json_meta.write_text(
            json.dumps({"weights": {"W_readout": w_init_json.tolist()}}),
            encoding="utf-8",
        )

        for step in range(30):
            x_spike = rng.randn(16) * 25.0
            y_spike = rng.randn(12) * 25.0
            out = adaptor.apply_plastic_update(
                "probe_json",
                x_spike,
                y_spike,
                reinforcement_signal=5.0,
                spectral_limit=limit,
                use_cbf_spectral_barrier=True,
            )
            assert out["status"] == "success"
            loaded = json.loads(json_meta.read_text(encoding="utf-8"))
            W_curr = np.asarray(loaded["weights"]["W_readout"], dtype=np.float64)
            sigma = float(np.linalg.svd(W_curr, compute_uv=False)[0])
            assert sigma <= limit + 1e-6, f"JSON brain exceeded limit at step {step}: {sigma} > {limit}"
            assert sigma <= limit

        # 2. Companion .npz weight store brain in tmp_path
        npz_meta = tmp_path / "neuro_probe_npz_meta.json"
        npz_meta.write_text(json.dumps({"name": "neuro_probe_npz"}), encoding="utf-8")
        npz_weights = tmp_path / "neuro_probe_npz_weights.npz"
        np.savez_compressed(
            npz_weights,
            W_mlp=(rng.randn(16, 16) * 0.05).astype(np.float32),
            W_O=(rng.randn(10, 16) * 0.05).astype(np.float32),
        )

        projected_seen = False
        for _ in range(25):
            x_spike = rng.randn(16) * 30.0
            y_spike = rng.randn(16) * 30.0
            out_npz = adaptor.apply_plastic_update(
                "probe_npz",
                x_spike,
                y_spike,
                reinforcement_signal=4.0,
                spectral_limit=limit,
                use_cbf_spectral_barrier=True,
            )
            assert out_npz["status"] == "success"
            if out_npz["cbf_projected_layers"]:
                projected_seen = True
            with np.load(npz_weights) as z:
                for k in ("W_mlp", "W_O"):
                    s_val = float(np.linalg.svd(z[k].astype(np.float64), compute_uv=False)[0])
                    assert s_val <= limit

        assert projected_seen is True
