"""
Spectral Graph Laplacian & Fiedler Vector Analyzer for Aetheris.
Ported and evolved from Aura-Alien VFS (E:/auraalienvfs/spectral_laplacian.cpp) + 2025/2026 SOTA.

Calculates:
1. Graph Laplacian L = D - A and normalized Laplacian.
2. Algebraic connectivity lambda_2 (Fiedler value) via shifted power iteration or SVD/eigh.
3. Fiedler vector (eigenvector of lambda_2):
   - Signs of components reveal natural bipartite / spectral graph cuts.
   - Values pinpoint communication bottlenecks and network vulnerabilities.
   - Detects isolated components and tool fragmentation in the micro-brain colony.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger("aetheris.spectral_laplacian")


@dataclass
class SpectralAnalysisResult:
    is_connected: bool
    algebraic_connectivity: float  # lambda_2
    spectral_gap: float
    fiedler_vector: List[float]
    bottleneck_nodes: List[str]
    isolated_nodes: List[str]
    cluster_partition_a: List[str]
    cluster_partition_b: List[str]
    summary: str


class SpectralLaplacianAnalyzer:
    """
    Computes algebraic connectivity and the Fiedler vector for the tool & micro-brain network.
    """

    _instance: Optional["SpectralLaplacianAnalyzer"] = None

    def __init__(self, connectivity_threshold: float = 1e-4):
        self.connectivity_threshold = float(connectivity_threshold)

    @classmethod
    def get_instance(cls) -> "SpectralLaplacianAnalyzer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_graph(self, node_names: List[str], adjacency_matrix: np.ndarray) -> SpectralAnalysisResult:
        n = len(node_names)
        A = np.asarray(adjacency_matrix, dtype=np.float64)

        if n < 2 or A.shape != (n, n):
            return SpectralAnalysisResult(
                is_connected=False,
                algebraic_connectivity=0.0,
                spectral_gap=0.0,
                fiedler_vector=[0.0] * n,
                bottleneck_nodes=[],
                isolated_nodes=list(node_names),
                cluster_partition_a=[],
                cluster_partition_b=[],
                summary="Trivial or empty graph (< 2 nodes).",
            )

        # Ensure symmetry (undirected connectivity)
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)

        # Degree vector and matrix
        degrees = np.sum(A, axis=1)
        isolated = [node_names[i] for i in range(n) if degrees[i] < 1e-8]

        # Laplacian matrix L = D - A
        L = np.diag(degrees) - A

        # Eigen-decomposition of symmetric Laplacian (eigh gives sorted eigenvalues)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except np.linalg.LinAlgError as e:
            return SpectralAnalysisResult(
                is_connected=False,
                algebraic_connectivity=0.0,
                spectral_gap=0.0,
                fiedler_vector=[0.0] * n,
                bottleneck_nodes=[],
                isolated_nodes=isolated,
                cluster_partition_a=[],
                cluster_partition_b=[],
                summary=f"Laplacian eigh failed: {e}",
            )

        # lambda_0 is ~0.0; lambda_1 is lambda_2 in 1-based indexing (algebraic connectivity)
        lambda_2 = float(max(0.0, eigenvalues[1])) if n > 1 else 0.0
        fiedler_vec = eigenvectors[:, 1] if n > 1 else np.zeros(n)

        # Spectral gap lambda_3 - lambda_2 (or lambda_2 - lambda_1)
        spectral_gap = float(eigenvalues[2] - eigenvalues[1]) if n > 2 else float(lambda_2)

        is_connected = lambda_2 > self.connectivity_threshold and len(isolated) == 0

        # Partition based on Fiedler vector signs
        partition_a = [node_names[i] for i in range(n) if fiedler_vec[i] >= 0]
        partition_b = [node_names[i] for i in range(n) if fiedler_vec[i] < 0]

        # Bottleneck nodes: nodes with Fiedler values closest to 0 (transition zone between clusters)
        abs_fiedler = np.abs(fiedler_vec)
        # Exclude completely isolated nodes
        connected_indices = [i for i in range(n) if degrees[i] > 1e-8]
        if connected_indices:
            sorted_indices = sorted(connected_indices, key=lambda idx: abs_fiedler[idx])
            bottlenecks = [node_names[idx] for idx in sorted_indices[:min(3, len(sorted_indices))]]
        else:
            bottlenecks = []

        summary = (
            f"Spectral Graph Analysis: {'CONNECTED' if is_connected else 'FRAGMENTED'} "
            f"(lambda_2={lambda_2:.4f}, spectral_gap={spectral_gap:.4f}, "
            f"isolated={len(isolated)}, bottlenecks={len(bottlenecks)})"
        )

        return SpectralAnalysisResult(
            is_connected=is_connected,
            algebraic_connectivity=round(lambda_2, 5),
            spectral_gap=round(spectral_gap, 5),
            fiedler_vector=[round(float(x), 4) for x in fiedler_vec],
            bottleneck_nodes=bottlenecks,
            isolated_nodes=isolated,
            cluster_partition_a=partition_a,
            cluster_partition_b=partition_b,
            summary=summary,
        )

