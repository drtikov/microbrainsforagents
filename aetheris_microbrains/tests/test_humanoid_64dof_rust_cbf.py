"""Tests for 64-DoF / 128-barrier native Rust CBF kernel and Motor-PD C-ABI clamp."""

from __future__ import annotations

import numpy as np

from aetheris_microbrains.core.rust_shim import NativeRTGateLib
from aetheris_microbrains.core.state_cbf_filter import (
    HalfPlaneBarrier,
    LinearDynamics,
    StateCBFSafetyFilter,
)


def test_43dof_humanoid_native_rust_cbf_parity():
    dim = 43
    A = np.zeros((dim, dim), dtype=np.float64)
    B = np.eye(dim, dtype=np.float64)
    dyn = LinearDynamics(A, B)
    barriers = [HalfPlaneBarrier(np.eye(1, dim, i).flatten(), -0.5, alpha=1.0) for i in range(dim)]
    u_min = np.full(dim, -5.0)
    u_max = np.full(dim, 5.0)

    f_py = StateCBFSafetyFilter(dyn, barriers, u_min=u_min, u_max=u_max, use_native_rust=False)
    f_rs = StateCBFSafetyFilter(dyn, barriers, u_min=u_min, u_max=u_max, use_native_rust=True)

    x = np.full(dim, -0.45)
    u_nom = np.full(dim, -3.0)
    out_py = f_py.filter(x, u_nom)
    out_rs = f_rs.filter(x, u_nom)

    assert out_py["verified_safe"] is True
    assert out_rs["verified_safe"] is True
    assert out_rs["solver_backend"] == "native_rust"
    assert np.allclose(out_py["u_filtered"], out_rs["u_filtered"], atol=1e-8)


def test_native_pd_cbf_clamp_43dof():
    lib = NativeRTGateLib.get()
    assert lib is not None
    dim = 43
    q = np.zeros(dim, dtype=np.float64)
    q_dot = np.zeros(dim, dtype=np.float64)
    q_des_nom = np.full(dim, 2.0, dtype=np.float64)
    q_dot_des = np.zeros(dim, dtype=np.float64)
    tau_ff = np.zeros(dim, dtype=np.float64)
    kp = np.full(dim, 60.0, dtype=np.float64)
    kd = np.full(dim, 2.0, dtype=np.float64)
    q_min = np.full(dim, -2.8, dtype=np.float64)
    q_max = np.full(dim, 2.8, dtype=np.float64)
    tau_max = np.full(dim, 30.0, dtype=np.float64)

    res = lib.pd_cbf_clamp(
        a_tau_rows=None,
        b_tau_vec=None,
        q=q,
        q_dot=q_dot,
        q_des_nom=q_des_nom,
        q_dot_des=q_dot_des,
        tau_ff=tau_ff,
        kp=kp,
        kd=kd,
        q_min=q_min,
        q_max=q_max,
        tau_max=tau_max,
    )
    assert res["verified_safe"] is True
    q_safe = np.asarray(res["q_des_safe"], dtype=np.float64)
    tau_applied = kp * (q_safe - q)
    assert np.all(np.abs(tau_applied) <= tau_max + 1e-6)
