"""Hierarchical Task-Priority Operational-Space IK with Tier-0 CBF polytopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .state_cbf_filter import BarrierConstraint, StateCBFSafetyFilter


@dataclass
class TaskObjective:
    name: str
    priority: int  # 1 = highest task priority after Tier-0 safety
    jacobian_fn: Callable[[np.ndarray], np.ndarray]
    desired_twist_fn: Callable[[np.ndarray], np.ndarray]
    weight: float = 1.0
    damping: float = 1e-3
    adaptive_damping: bool = True
    singularity_thresh: float = 0.05


def _damped_pinv(J: np.ndarray, damping: float, adaptive: bool, sing_thresh: float) -> np.ndarray:
    if J.size == 0:
        return np.zeros((J.shape[1], J.shape[0]), dtype=np.float64)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    lam2 = float(damping) ** 2
    if adaptive and s.size > 0:
        s_min = float(s[-1])
        if s_min < sing_thresh:
            lam2 += (0.05 * (1.0 - s_min / max(sing_thresh, 1e-9))) ** 2
    s_inv = s / (s * s + lam2)
    return (Vt.T * s_inv[None, :]) @ U.T


class HierarchicalTaskCBFIK:
    """Lexicographic nullspace IK projected onto Tier-0 hard safety CBFs."""

    def __init__(self, n_dof: int, safety_filter: Optional[StateCBFSafetyFilter] = None):
        self.n_dof = int(n_dof)
        self.safety_filter = safety_filter

    def solve_step(
        self,
        x: Sequence[float],
        tasks: Sequence[TaskObjective],
        extra_barriers: Sequence[BarrierConstraint] = (),
    ) -> Dict[str, Any]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        q = x_arr[: self.n_dof]
        sorted_tasks = sorted(tasks, key=lambda t: int(t.priority))

        dq = np.zeros(self.n_dof, dtype=np.float64)
        N = np.eye(self.n_dof, dtype=np.float64)
        task_errors: Dict[str, float] = {}

        for task in sorted_tasks:
            J = np.asarray(task.jacobian_fn(q), dtype=np.float64)
            dx_des = np.asarray(task.desired_twist_fn(q), dtype=np.float64).flatten()
            J_proj = J @ N
            J_pinv = _damped_pinv(J_proj, task.damping, task.adaptive_damping, task.singularity_thresh)
            dq = dq + float(task.weight) * (J_pinv @ (dx_des - J @ dq))
            N = N - J_pinv @ J_proj
            task_errors[task.name] = float(np.linalg.norm(dx_des - J @ dq))

        if self.safety_filter is not None:
            orig_barriers = list(self.safety_filter.barriers)
            if extra_barriers:
                self.safety_filter.barriers = orig_barriers + list(extra_barriers)
            try:
                res = self.safety_filter.filter(x_arr, dq)
            finally:
                self.safety_filter.barriers = orig_barriers
            return {
                "dq_nominal": dq.tolist(),
                "dq_safe": res["u_filtered"],
                "verified_safe": res["verified_safe"],
                "status": res["status"],
                "task_errors": task_errors,
            }

        return {
            "dq_nominal": dq.tolist(),
            "dq_safe": dq.tolist(),
            "verified_safe": True,
            "status": "no_filter",
            "task_errors": task_errors,
        }
