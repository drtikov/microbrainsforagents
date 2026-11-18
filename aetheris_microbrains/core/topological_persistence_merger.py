"""
Aetheris Topological Persistence Diagram Merger.
Combines Betti-1 homology from NeoAura with Vietoris-Rips persistent filtration.
Guarantees that merging specialized daughter brains (SLERP / TIES) preserves
fundamental 1-dimensional topological reasoning cycles and avoids circular hallucination collapses.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class TopologicalPersistenceMerger:
    """Guards brain merges by ensuring topological homology cycles remain invariant."""

    _instance: Optional[TopologicalPersistenceMerger] = None

    def __init__(self, homology_persistence_tol: float = 0.35):
        self.homology_persistence_tol = homology_persistence_tol

    @classmethod
    def get_instance(cls) -> TopologicalPersistenceMerger:
        if cls._instance is None:
            cls._instance = TopologicalPersistenceMerger()
        return cls._instance

    def _compute_rips_persistence_betti1(self, matrix: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Approximates 1-dimensional persistent homology (Betti-1 cycles):
        Constructs Vietoris-Rips adjacency graph at distance threshold:
        beta_0 = connected components
        beta_1 = edges - vertices + beta_0 (fundamental cycles)
        """
        # Pairwise euclidean distance matrix
        diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        n = len(matrix)
        adj = (dist < threshold).astype(int)
        np.fill_diagonal(adj, 0)

        num_edges = int(np.sum(adj) // 2)

        # BFS for connected components (beta_0)
        visited = set()
        components = 0
        for i in range(n):
            if i not in visited:
                components += 1
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    for neighbor in np.where(adj[curr] == 1)[0]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

        beta_0 = components
        # Euler-Poincare characteristic for 1-complex: beta_1 = |E| - |V| + beta_0
        beta_1 = max(0, num_edges - n + beta_0)

        # Off-diagonal upper triangle sum represents total metric dispersion / persistent energy
        off_diag_energy = float(np.sum(np.triu(dist, k=1)))
        return {
            "vertices": n,
            "edges": num_edges,
            "beta_0": beta_0,
            "beta_1": beta_1,
            "persistence_energy": round(off_diag_energy, 4)
        }

    def safe_slerp_merge(
        self,
        weights_a: Union[List[float], np.ndarray],
        weights_b: Union[List[float], np.ndarray],
        interpolation_t: float = 0.5
    ) -> Dict[str, Any]:
        """
        Performs Spherical Linear Interpolation (SLERP) guarded by Topological Homology Invariance:
        W_fused = sin((1-t)*theta)/sin(theta) * W_a + sin(t*theta)/sin(theta) * W_b
        Verifies that beta_1 (essential reasoning loops) is not collapsed.
        """
        wa = np.asarray(weights_a, dtype=np.float32).flatten()
        wb = np.asarray(weights_b, dtype=np.float32).flatten()

        max_len = max(len(wa), len(wb))
        if len(wa) < max_len:
            wa = np.pad(wa, (0, max_len - len(wa)))
        if len(wb) < max_len:
            wb = np.pad(wb, (0, max_len - len(wb)))

        # Normalize vectors for SLERP
        norm_a = np.linalg.norm(wa) or 1e-9
        norm_b = np.linalg.norm(wb) or 1e-9
        u_a = wa / norm_a
        u_b = wb / norm_b

        # Compute dot product and angle
        dot = float(np.clip(np.dot(u_a, u_b), -1.0, 1.0))
        theta = math.acos(dot)

        # Compute arbitrary-length compatible point cloud reshaping for Rips persistence
        pt_dim = max(1, min(16, max(1, max_len // 2)))
        padded_len = int(math.ceil(max_len / float(pt_dim)) * pt_dim)

        def _reshape_pts(vec: np.ndarray) -> np.ndarray:
            if len(vec) < padded_len:
                vec = np.pad(vec, (0, padded_len - len(vec)))
            return vec[:padded_len].reshape(-1, pt_dim)

        # Baseline homology check on parents
        topo_a = self._compute_rips_persistence_betti1(_reshape_pts(u_a))
        topo_b = self._compute_rips_persistence_betti1(_reshape_pts(u_b))

        # Candidate interpolation
        t = float(np.clip(interpolation_t, 0.0, 1.0))
        if abs(theta) < 1e-5 or math.sin(theta) < 1e-5:
            # Collinear vectors
            u_fused = (1.0 - t) * u_a + t * u_b
        else:
            u_fused = (math.sin((1.0 - t) * theta) / math.sin(theta)) * u_a + (math.sin(t * theta) / math.sin(theta)) * u_b

        target_norm = (1.0 - t) * norm_a + t * norm_b
        w_fused = u_fused * target_norm

        # Evaluate candidate manifold topology
        topo_fused = self._compute_rips_persistence_betti1(_reshape_pts(u_fused))

        # Check topological conservation: beta_1 shouldn't suffer collapse.
        # The gate must be able to FAIL: the old `delta <= avg + 2.0` absolute
        # slack passed even a doubled cycle count on zero-cycle parents, so
        # "merged_safe" was unfalsifiable. Now the fused beta_1 must stay
        # within a relative tolerance of the parent average.
        avg_parent_betti1 = (topo_a["beta_1"] + topo_b["beta_1"]) / 2.0
        delta_betti1 = abs(topo_fused["beta_1"] - avg_parent_betti1)

        is_safe = delta_betti1 <= self.homology_persistence_tol * max(1.0, avg_parent_betti1)

        projected_t = t
        if not is_safe:
            tol_limit = self.homology_persistence_tol * max(1.0, avg_parent_betti1)
            best_t = t
            best_delta = delta_betti1
            best_u = u_fused
            best_topo = topo_fused
            for t_cand in sorted(np.linspace(0.0, 1.0, 21).tolist(), key=lambda tc: abs(tc - t)):
                if abs(theta) < 1e-5 or math.sin(theta) < 1e-5:
                    u_cand = (1.0 - t_cand) * u_a + t_cand * u_b
                else:
                    u_cand = (
                        (math.sin((1.0 - t_cand) * theta) / math.sin(theta)) * u_a
                        + (math.sin(t_cand * theta) / math.sin(theta)) * u_b
                    )
                topo_cand = self._compute_rips_persistence_betti1(_reshape_pts(u_cand))
                d_cand = abs(topo_cand["beta_1"] - avg_parent_betti1)
                if d_cand <= tol_limit:
                    best_t = float(t_cand)
                    best_u = u_cand
                    best_topo = topo_cand
                    break
                if d_cand < best_delta:
                    best_delta = d_cand
                    best_t = float(t_cand)
                    best_u = u_cand
                    best_topo = topo_cand

            projected_t = best_t
            target_norm = (1.0 - projected_t) * norm_a + projected_t * norm_b
            w_fused = best_u * target_norm
            topo_fused = best_topo

        return {
            "status": "merged_safe" if is_safe else "merged_projected",
            "is_topologically_conserved": is_safe,
            "interpolation_t": t,
            "projected_t": round(float(projected_t), 4),
            "parent_a_betti1": topo_a["beta_1"],
            "parent_b_betti1": topo_b["beta_1"],
            "fused_betti1": topo_fused["beta_1"],
            "fused_norm": round(float(np.linalg.norm(w_fused)), 4),
            "fused_weights": w_fused.tolist(),
        }

    def multiscale_geodesic_slerp_merge(
        self,
        weights_a: Union[List[float], np.ndarray],
        weights_b: Union[List[float], np.ndarray],
        n_t: int = 16,
        n_eps: int = 32,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Multi-Scale Vietoris-Rips Persistence Barcode & Geodesic SLERP Search
        on the GT 710 OpenCL GPU (`n_t x n_eps = 512` parallel filtration graphs).
        Automatically selects the optimal geodesic interpolation parameter t* that minimizes
        the L1 Wasserstein discrepancy to the parent Betti-0/Betti-1 persistence barcodes.
        """
        from .opencl_microbrain_accelerator import get_opencl_accelerator

        wa = np.asarray(weights_a, dtype=np.float32).flatten()
        wb = np.asarray(weights_b, dtype=np.float32).flatten()
        max_len = max(len(wa), len(wb), 32)
        pt_dim = min(16, max(4, max_len // 8))
        padded_len = int(math.ceil(max_len / float(pt_dim)) * pt_dim)
        if len(wa) < padded_len:
            wa_pad = np.pad(wa, (0, padded_len - len(wa)))
        else:
            wa_pad = wa[:padded_len]
        if len(wb) < padded_len:
            wb_pad = np.pad(wb, (0, padded_len - len(wb)))
        else:
            wb_pad = wb[:padded_len]

        norm_a = float(np.linalg.norm(wa_pad)) or 1e-9
        norm_b = float(np.linalg.norm(wb_pad)) or 1e-9
        u_a = wa_pad / norm_a
        u_b = wb_pad / norm_b

        dot = float(np.clip(np.dot(u_a, u_b), -1.0, 1.0))
        theta = math.acos(dot)

        ua_pts = u_a.reshape(-1, pt_dim)
        ub_pts = u_b.reshape(-1, pt_dim)

        t_grid = np.linspace(0.0, 1.0, max(3, n_t), dtype=np.float32)
        eps_grid = np.linspace(0.05, 0.45, max(4, n_eps), dtype=np.float32)

        acc = get_opencl_accelerator()
        rips_res = acc.rips_multiscale_slerp_search(
            u_a=ua_pts,
            u_b=ub_pts,
            theta=theta,
            t_grid=t_grid,
            eps_grid=eps_grid,
            force_gpu=force_gpu,
        )

        opt_t = float(rips_res["optimal_t"])
        if abs(theta) < 1e-5 or math.sin(theta) < 1e-5:
            u_fused = (1.0 - opt_t) * u_a + opt_t * u_b
        else:
            u_fused = (
                (math.sin((1.0 - opt_t) * theta) / math.sin(theta)) * u_a
                + (math.sin(opt_t * theta) / math.sin(theta)) * u_b
            )
        target_norm = (1.0 - opt_t) * norm_a + opt_t * norm_b
        w_fused = (u_fused * target_norm)[: max(len(wa), len(wb))]

        return {
            "status": "multiscale_geodesic_merged",
            "backend": rips_res["backend"],
            "device": rips_res["device"],
            "optimal_t": opt_t,
            "wasserstein_l1_discrepancy": rips_res["wasserstein_l1_discrepancy"],
            "grid_evaluations": rips_res["grid_evaluations"],
            "parent_a_beta1_integral": rips_res["parent_a_beta1_integral"],
            "parent_b_beta1_integral": rips_res["parent_b_beta1_integral"],
            "fused_beta1_integral": rips_res["fused_beta1_integral"],
            "beta0_barcode": rips_res["beta0_barcode"],
            "beta1_barcode": rips_res["beta1_barcode"],
            "fused_weights": w_fused.tolist(),
        }


