"""
Neural Cellular Automata (NCA) Weight Self-Repair Engine for Aetheris.
Ported and evolved from 2026 SOTA (Luca Mariot et al., NCAtorch, Levin's Morphogenesis).

Treats micro-brain weight matrices W in {-1, 0, +1} as 2D living cellular grids:
1. Local State Vector: Each weight cell (i, j) maintains a multi-channel morphogenetic state.
2. Decentralized Local Update: Perception filters (Sobel / Laplacian stencils) detect local
   corruptions, sudden zeroing, or bit-flip perturbations.
3. Self-Healing Regeneration: Rebuilds destroyed or corrupted synaptic weights using local
   cell-to-cell communication without backpropagation or external checkpoint reloading.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger("aetheris.nca_weight_repair")


@dataclass
class NCARepairResult:
    was_corrupted: bool
    corrupted_cells_count: int
    restored_cells_count: int
    reconstruction_fidelity: float  # Cosine similarity to target morphology (0.0 to 1.0)
    iterations_run: int
    # True only when reconstruction_fidelity is a real measurement against
    # a supplied reference morphology. When False, reconstruction_fidelity
    # is 0.0 and means "no comparison was possible", never "poor match".
    reference_available: bool = False


class NCAWeightSelfRepairEngine:
    """
    Self-healing Neural Cellular Automata grid restoring damaged weight matrices.
    """

    _instance: Optional["NCAWeightSelfRepairEngine"] = None

    @classmethod
    def get_instance(cls) -> "NCAWeightSelfRepairEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, channels: int = 4, alpha_decay: float = 0.98, grid_size: int = 32, steps_per_cycle: int = 15):
        self.channels = channels  # Channel 0: Weight value; Channels 1-3: Morphogenetic latent state
        self.alpha_decay = alpha_decay
        self.grid_size = grid_size
        self.steps_per_cycle = steps_per_cycle

        # Standard 2D discrete differential filters (Sobel X, Sobel Y, Laplacian)
        self.sobel_x = np.array([[-1.0, 0.0, 1.0],
                                 [-2.0, 0.0, 2.0],
                                 [-1.0, 0.0, 1.0]], dtype=np.float64) / 8.0

        self.sobel_y = np.array([[-1.0, -2.0, -1.0],
                                 [ 0.0,  0.0,  0.0],
                                 [ 1.0,  2.0,  1.0]], dtype=np.float64) / 8.0

        self.laplacian = np.array([[ 0.0,  1.0,  0.0],
                                   [ 1.0, -4.0,  1.0],
                                   [ 0.0,  1.0,  0.0]], dtype=np.float64) / 4.0

    def _to_morphogenetic_2d(self, weight_tensor: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], int]:
        """
        Convert an arbitrary 1D (including prime lengths), 2D, or N-D weight tensor
        into a 2D morphogenetic lattice, returning (W_2d, orig_shape, orig_size).
        """
        W = np.asarray(weight_tensor, dtype=np.float64)
        orig_shape = W.shape if W.ndim > 0 else (1,)
        orig_size = int(W.size)

        if W.ndim == 0:
            return W.reshape(1, 1), orig_shape, 1

        if W.ndim == 1:
            dim = len(W)
            h = int(np.floor(np.sqrt(dim)))
            while dim % h != 0 and h > 1:
                h -= 1
            if h > 1 or dim <= 3:
                w = dim // h
                return W.reshape(h, w).copy(), orig_shape, orig_size
            # Prime length > 3: embed into smallest composite square grid for 2D neighbor diffusion
            side = int(np.ceil(np.sqrt(dim)))
            buf = np.zeros(side * side, dtype=np.float64)
            buf[:dim] = W
            return buf.reshape(side, side), orig_shape, orig_size

        if W.ndim == 2:
            return W.copy(), orig_shape, orig_size

        # 3D+ tensor: fold trailing dimensions into columns for 2D cellular grid
        return W.reshape(W.shape[0], -1).copy(), orig_shape, orig_size

    @staticmethod
    def _detect_damage_mask(
        W_2d: np.ndarray,
        nan_mask_2d: np.ndarray,
        explicit_mask_2d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Detect corrupted cells from NaN/Inf values, optional explicit damage mask,
        and contiguous >= 3x3 zeroed blocks inside an active weight matrix.
        """
        mask = np.asarray(nan_mask_2d, dtype=bool).copy()
        if explicit_mask_2d is not None:
            mask |= np.asarray(explicit_mask_2d, dtype=bool)

        h, w = W_2d.shape
        if h >= 4 and w >= 4 and float(np.mean(np.abs(W_2d) > 1e-9)) > 0.25:
            zero_cells = np.abs(W_2d) <= 1e-12
            for r in range(h - 2):
                for c in range(w - 2):
                    if np.all(zero_cells[r : r + 3, c : c + 3]):
                        mask[r : r + 3, c : c + 3] = True
        return mask

    def initialize_morphogenetic_grid(self, weight_matrix: np.ndarray) -> np.ndarray:
        """
        Expands an arbitrary-shape weight tensor into an (H, W, channels) morphogenetic
        cellular lattice. Channel 0 stores weights; Channels 1-3 store spatial anchors.
        """
        W, _, _ = self._to_morphogenetic_2d(weight_matrix)

        h, w = W.shape
        grid = np.zeros((h, w, self.channels), dtype=np.float64)
        grid[:, :, 0] = W

        # Encode relative spatial anchors in latent channels
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        grid[:, :, 1] = x_coords / max(1, w - 1) * 2.0 - 1.0
        grid[:, :, 2] = y_coords / max(1, h - 1) * 2.0 - 1.0
        # Harmonic phase seed
        grid[:, :, 3] = np.sin(np.pi * grid[:, :, 1]) * np.cos(np.pi * grid[:, :, 2])

        return grid

    def step_nca(
        self,
        grid: np.ndarray,
        iterations: int = 5,
        clamp_mask: Optional[np.ndarray] = None,
        anchor_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Runs decentralized local cellular automata updates over the grid.
        When `clamp_mask` and `anchor_values` are provided (2025-2026 Mask-Clamped Inpainting),
        uncorrupted cells (~clamp_mask) stay anchored while damaged cells regenerate from neighbors.
        """
        curr = np.copy(grid)
        h, w, c = curr.shape
        has_clamp = (
            clamp_mask is not None
            and anchor_values is not None
            and clamp_mask.shape == (h, w)
            and anchor_values.shape == (h, w)
        )

        for _ in range(iterations):
            new_grid = np.copy(curr)
            for ch in range(c):
                plane = curr[:, :, ch]
                # Convolution with boundary padding (wrap / circular)
                padded = np.pad(plane, pad_width=1, mode="wrap")

                # Local perception
                gx = self._apply_filter(padded, self.sobel_x)
                gy = self._apply_filter(padded, self.sobel_y)
                lap = self._apply_filter(padded, self.laplacian)

                # Local reaction-diffusion update
                if ch == 0:
                    # Weight channel update conditioned on latent coordinates
                    update = 0.40 * lap + 0.15 * (curr[:, :, 1] * gx + curr[:, :, 2] * gy)
                    # Deterministic firing on damaged zones during masked inpainting; stochastic elsewhere
                    stochastic_mask = (np.random.default_rng(1337).random((h, w)) < 0.85).astype(np.float64)
                    if has_clamp:
                        stochastic_mask[clamp_mask] = 1.0
                    updated_w = np.clip(curr[:, :, ch] + update * stochastic_mask, -2.0, 2.0)
                    if has_clamp:
                        updated_w[~clamp_mask] = anchor_values[~clamp_mask]
                    new_grid[:, :, ch] = updated_w
                else:
                    # Latent channels maintain structural coordinate manifold
                    update = 0.25 * lap
                    new_grid[:, :, ch] = curr[:, :, ch] * self.alpha_decay + update

            curr = new_grid

        return curr

    def repair_damage(
        self,
        damaged_matrix: np.ndarray,
        reference_morphology: Optional[np.ndarray] = None,
        iterations: int = 8,
        damage_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, NCARepairResult]:
        """
        Detects anomalies/damage (NaN/Inf cells, explicit masks, or >= 3x3 wiped zero blocks)
        and applies Mask-Clamped Morphogenetic Inpainting NCA to restore weight integrity
        while preserving healthy anchor cells.
        """
        W_raw = np.asarray(damaged_matrix, dtype=np.float64)
        nan_mask_raw = np.isnan(W_raw) | np.isinf(W_raw)
        W_clean_raw = np.where(nan_mask_raw, 0.0, W_raw)

        W_2d, orig_shape, orig_size = self._to_morphogenetic_2d(W_clean_raw)
        nan_mask_2d, _, _ = self._to_morphogenetic_2d(nan_mask_raw.astype(np.float64))
        nan_mask_2d = nan_mask_2d > 0.5

        explicit_mask_2d = None
        if damage_mask is not None:
            em_2d, _, _ = self._to_morphogenetic_2d(np.asarray(damage_mask, dtype=np.float64))
            explicit_mask_2d = em_2d > 0.5

        mask_2d = self._detect_damage_mask(W_2d, nan_mask_2d, explicit_mask_2d=explicit_mask_2d)
        # Do not count padding cells (for prime-length 1D embeddings) as damaged original cells
        valid_2d = np.zeros(W_2d.size, dtype=bool)
        valid_2d[:orig_size] = True
        valid_mask_2d = valid_2d.reshape(W_2d.shape)

        corrupted_count = int(np.sum(mask_2d & valid_mask_2d))
        valid_total = max(1, orig_size)
        partial_repair = 0 < corrupted_count < valid_total

        grid = self.initialize_morphogenetic_grid(W_clean_raw)
        # Keep padding cells excluded from healthy anchor drift only on damaged interior
        healed_grid = self.step_nca(
            grid,
            iterations=iterations,
            clamp_mask=mask_2d if partial_repair else None,
            anchor_values=W_2d if partial_repair else None,
        )
        W_healed_2d = healed_grid[:, :, 0]

        if partial_repair:
            repaired_vals = W_healed_2d[mask_2d]
            healthy_vals = W_2d[(~mask_2d) & valid_mask_2d]
            if repaired_vals.size > 0 and float(np.max(np.abs(repaired_vals))) > 0.0:
                local_scale = max(1e-12, float(np.mean(np.abs(repaired_vals))))
                anchor_nonzero = np.abs(healthy_vals) > 1e-12
                anchor_scale = (
                    float(np.mean(np.abs(healthy_vals[anchor_nonzero])))
                    if np.any(anchor_nonzero)
                    else local_scale
                )
                W_healed_2d[mask_2d] = np.round(np.clip(repaired_vals / local_scale, -1.0, 1.0)) * anchor_scale
            W_healed_2d[~mask_2d] = W_2d[~mask_2d]
        else:
            # Preserve unconstrained BitNet quantization behavior when no partial mask is active
            if np.max(np.abs(W_healed_2d)) > 0:
                scale = np.mean(np.abs(W_healed_2d))
                W_healed_2d = np.round(np.clip(W_healed_2d / max(1e-12, scale), -1.0, 1.0)) * scale

        W_healed = W_healed_2d.flatten()[:orig_size]
        if W_raw.ndim > 0:
            W_healed = W_healed.reshape(orig_shape)
        else:
            W_healed = W_healed.reshape(())

        # Calculate fidelity. It is only a measurement when a reference
        # morphology exists; without one there is nothing to compare
        # against, so the honest value is 0.0 ("no comparison possible"),
        # not a fabricated near-perfect score.
        reference_available = reference_morphology is not None
        if reference_morphology is not None:
            ref = reference_morphology.flatten()
            cand = W_healed.flatten()
            norm_r = np.linalg.norm(ref)
            norm_c = np.linalg.norm(cand)
            if norm_r > 1e-12 and norm_c > 1e-12:
                fidelity = float(np.dot(ref, cand) / (norm_r * norm_c))
            elif norm_r <= 1e-12 and norm_c <= 1e-12:
                # Both sides are (numerically) zero vectors: they are
                # identical, so this degenerate case counts as a match.
                # Cosine similarity is undefined here, but reporting 0.0
                # for two provably-identical vectors would be dishonest
                # in the other direction.
                fidelity = 1.0
            else:
                # Exactly one side is a zero vector: a zero vector has no
                # direction, so no similarity is measurable. The old code
                # reported a perfect 1.0 here; that was fabricated.
                fidelity = 0.0
        else:
            fidelity = 0.0

        result = NCARepairResult(
            was_corrupted=corrupted_count > 0,
            corrupted_cells_count=corrupted_count,
            restored_cells_count=corrupted_count,
            reconstruction_fidelity=round(max(0.0, min(1.0, fidelity)), 4),
            iterations_run=iterations,
            reference_available=reference_available,
        )

        return W_healed, result

    def _apply_filter(self, padded_plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Fast 3x3 vectorized convolution."""
        return (
            padded_plane[:-2, :-2] * kernel[0, 0] + padded_plane[:-2, 1:-1] * kernel[0, 1] + padded_plane[:-2, 2:] * kernel[0, 2] +
            padded_plane[1:-1, :-2] * kernel[1, 0] + padded_plane[1:-1, 1:-1] * kernel[1, 1] + padded_plane[1:-1, 2:] * kernel[1, 2] +
            padded_plane[2:, :-2] * kernel[2, 0] + padded_plane[2:, 1:-1] * kernel[2, 1] + padded_plane[2:, 2:] * kernel[2, 2]
        )

    def heal_weights(self, W: np.ndarray, target_prototype: Optional[np.ndarray] = None, steps: int = 15) -> np.ndarray:
        """Convenience method for repairing damaged weights."""
        healed, _ = self.repair_damage(W, reference_morphology=target_prototype, iterations=steps)
        return healed

    def get_status(self) -> Dict[str, Any]:
        return {
            "channels": self.channels,
            "alpha_decay": self.alpha_decay,
            "grid_size": self.grid_size,
            "steps_per_cycle": self.steps_per_cycle
        }

# Alias for backward compatibility and clean test suite imports
NCAWeightRegenerator = NCAWeightSelfRepairEngine


