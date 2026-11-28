"""Tests for HierarchicalTaskCBFIK lexicographic priority and Tier-0 safety."""

from __future__ import annotations

import numpy as np

from aetheris_microbrains.core.hierarchical_task_cbf_ik import HierarchicalTaskCBFIK, TaskObjective
from aetheris_microbrains.core.state_cbf_filter import HalfPlaneBarrier, LinearDynamics, StateCBFSafetyFilter


def test_hierarchical_task_priority_and_tier0_safety():
    n_dof = 3
    dyn = LinearDynamics(np.zeros((n_dof, n_dof)), np.eye(n_dof))
    wall = HalfPlaneBarrier([-1.0, 0.0, 0.0], -0.5, alpha=1.0)  # x0 <= 0.5
    filt = StateCBFSafetyFilter(dyn, [wall], u_min=[-2.0] * n_dof, u_max=[2.0] * n_dof)
    ik = HierarchicalTaskCBFIK(n_dof=n_dof, safety_filter=filt)

    # Priority 1: move x0+x1 along [1, 1, 0]
    t1 = TaskObjective(
        name="ee_pos",
        priority=1,
        jacobian_fn=lambda q: np.array([[1.0, 1.0, 0.0]]),
        desired_twist_fn=lambda q: np.array([1.0]),
    )
    # Priority 2: conflicting posture trying to pull x0+x1 negative
    t2 = TaskObjective(
        name="posture",
        priority=2,
        jacobian_fn=lambda q: np.eye(n_dof),
        desired_twist_fn=lambda q: np.array([-0.5, -0.5, 0.2]),
    )

    res = ik.solve_step([0.0, 0.0, 0.0], [t1, t2])
    assert res["verified_safe"] is True
    assert res["task_errors"]["ee_pos"] < 1e-3

    # Near wall x0=0.49, Tier-0 CBF clamps dq_safe[0] <= 0.01
    res_wall = ik.solve_step([0.49, 0.0, 0.0], [t1, t2])
    assert res_wall["verified_safe"] is True
    assert res_wall["dq_safe"][0] <= 0.015
