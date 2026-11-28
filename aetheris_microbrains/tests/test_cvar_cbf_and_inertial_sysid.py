"""Tests for MultimodalCVaRCBF and online SPD pseudo-inertia system identification."""

from __future__ import annotations

import numpy as np

from aetheris_microbrains.core.cvar_multimodal_cbf import MultimodalCVaRCBF, PedestrianHypothesis
from aetheris_microbrains.core.online_inertial_sysid import PhysicalPayloadIdentifier, _wrench_regressor_matrix, project_pseudo_inertia_spd
from aetheris_microbrains.core.state_cbf_filter import LinearDynamics


def test_cvar_multimodal_cbf_crowd():
    dyn = LinearDynamics(np.zeros((2, 2)), np.eye(2))
    cvar = MultimodalCVaRCBF(dyn, beta=0.15, u_min=[-2.0, -2.0], u_max=[2.0, 2.0])
    hyps = [
        PedestrianHypothesis("straight", 0.80, [1.5, 0.5], [-0.5, 0.0], radius=0.4, alpha=1.0),
        PedestrianHypothesis("swerve", 0.19, [0.8, 0.0], [-0.8, 0.0], radius=0.4, alpha=1.0),
        PedestrianHypothesis("outlier", 0.01, [0.41, 0.0], [-3.0, 0.0], radius=0.4, alpha=1.0),
    ]
    res = cvar.filter([0.0, 0.0], [1.5, 0.0], hyps)
    assert res["verified_safe"] is True
    assert res["cvar_value"] <= 1e-4
    # Protects against swerve without freezing completely
    assert res["u_filtered"][0] < 1.5


def test_pseudo_inertia_spd_and_payload_rls():
    bad_theta = [-2.0, 0.1, 0.2, 0.3, -1.0, 0.0, 0.0, 0.5, 0.0, 10.0]
    proj = project_pseudo_inertia_spd(bad_theta)
    assert proj["mass"] > 0.0
    eig_I = np.linalg.eigvalsh(proj["inertia_3x3"])
    assert np.all(eig_I > 0.0)

    # True payload: 3.5 kg at com=[0.08, -0.04, 0.12]
    m_true = 3.5
    com_true = np.array([0.08, -0.04, 0.12])
    I_true = np.diag([0.05, 0.06, 0.04])
    theta_true = np.array([
        m_true,
        m_true * com_true[0], m_true * com_true[1], m_true * com_true[2],
        I_true[0, 0], 0.0, 0.0, I_true[1, 1], 0.0, I_true[2, 2],
    ])

    ident = PhysicalPayloadIdentifier(forgetting_factor=0.98)
    rng = np.random.RandomState(7)
    for _ in range(250):
        a = rng.randn(3) * 2.0
        w = rng.randn(3) * 1.5
        wd = rng.randn(3) * 2.0
        g = np.array([0.0, 0.0, -9.81])
        Y = _wrench_regressor_matrix(a, w, wd, g)
        wrench = Y @ theta_true + rng.randn(6) * 0.01
        out = ident.update_ee_wrench(a, w, wd, g, wrench)

    assert abs(out["mass"] - m_true) / m_true < 0.05
    assert np.linalg.norm(out["com"] - com_true) < 0.03
