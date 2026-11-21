"""
Aetheris Sparse Autoencoder (SAE) Concept Engine.
Implements a lightweight, CPU-native TopK Sparse Autoencoder on NumPy/PyTorch CPU.
Decomposes polysemantic neural activations into monosemantic concept feature dictionaries,
enabling concept-level steering, selective concept ablation/amplification, and
prevention of catastrophic forgetting during continuous adaptation.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ..core.runtime_paths import get_state_dir

logger = logging.getLogger(__name__)

STORE_DIR = get_state_dir() / "wsp_store"
SAE_DICTIONARY_FILE = STORE_DIR / "wsp_sae_dictionary.npz"
CONCEPT_REGISTRY_FILE = STORE_DIR / "wsp_concept_registry.json"


class ConceptSparseAutoencoder:
    """
    TopK Sparse Autoencoder mapping activation x in R^{d_model} to sparse latents z in R^{d_sae}.
    d_sae = d_model * expansion_factor (overcomplete sparse basis).
    Encoder: z = TopK(ReLU((x - b_dec) @ W_enc + b_enc), k=k_active)
    Decoder: x_hat = z @ W_dec + b_dec
    """

    def __init__(
        self,
        d_model: int = 64,
        expansion_factor: int = 4,
        k_active: int = 4,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k_active = k_active
        self.rng = np.random.default_rng(seed)

        self.b_dec = np.zeros(d_model, dtype=np.float32)
        self.b_enc = np.zeros(self.d_sae, dtype=np.float32)

        # Normalized dictionary columns for W_dec
        raw_dec = self.rng.standard_normal((self.d_sae, d_model), dtype=np.float32)
        norms = np.linalg.norm(raw_dec, axis=1, keepdims=True) + 1e-12
        self.W_dec = raw_dec / norms  # (d_sae, d_model)
        self.W_enc = self.W_dec.T.copy()  # Tied initialization (d_model, d_sae)

        # Dead-feature tracking (2025-2026 TopK + AuxK SAE revival)
        self.feature_fire_counts = np.zeros(self.d_sae, dtype=np.int64)
        self.total_steps = 0

        # Concept annotations {concept_idx: label}
        self.concept_labels: Dict[int, str] = {}

        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.load_or_initialize()

    def load_or_initialize(self) -> None:
        if SAE_DICTIONARY_FILE.exists():
            try:
                data = np.load(SAE_DICTIONARY_FILE)
                if data["W_enc"].shape == (self.d_model, self.d_sae) and data["W_dec"].shape == (self.d_sae, self.d_model):
                    self.W_enc = data["W_enc"]
                    self.W_dec = data["W_dec"]
                    self.b_enc = data["b_enc"]
                    self.b_dec = data["b_dec"]
                else:
                    self.save_dictionary()
            except Exception as e:
                logger.warning(f"Failed to load SAE dictionary: {e}")
                self.save_dictionary()
        else:
            self.save_dictionary()

        if CONCEPT_REGISTRY_FILE.exists():
            try:
                with open(CONCEPT_REGISTRY_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.concept_labels = {int(k): str(v) for k, v in raw.items()}
            except Exception as e:
                logger.warning(f"Failed to load concept registry: {e}")
                self._init_default_concept_labels()
        else:
            self._init_default_concept_labels()

    def _init_default_concept_labels(self) -> None:
        default_concepts = {
            0: "core_safety_invariant",
            1: "code_syntax_verification",
            2: "tool_selection_dynamic",
            3: "epistemic_hypothesis_search",
            4: "memory_store_retrieval",
            5: "spectral_loop_prevention",
            6: "test_scaffold_rigor",
            7: "error_recovery_grace",
        }
        self.concept_labels.update(default_concepts)
        self.save_concept_registry()

    def save_dictionary(self) -> None:
        np.savez_compressed(
            SAE_DICTIONARY_FILE,
            W_enc=self.W_enc,
            W_dec=self.W_dec,
            b_enc=self.b_enc,
            b_dec=self.b_dec,
        )

    def save_concept_registry(self) -> None:
        try:
            with open(CONCEPT_REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.concept_labels.items()}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save concept registry: {e}")

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Projects activation x into sparse monosemantic concept activations via TopK.
        x: (batch_size, d_model) or (d_model,)
        Returns sparse z of shape (batch_size, d_sae) or (d_sae,) with exactly k_active nonzeros.
        """
        x_in = np.asarray(x, dtype=np.float32)
        is_1d = (x_in.ndim == 1)
        if is_1d:
            x_in = x_in[np.newaxis, :]

        # Centered input: x - b_dec
        x_cent = x_in - self.b_dec
        # Linear + ReLU
        pre_act = np.maximum(0.0, np.matmul(x_cent, self.W_enc) + self.b_enc)

        # Top-K Sparsification
        batch_size = pre_act.shape[0]
        z_sparse = np.zeros_like(pre_act)
        for i in range(batch_size):
            row = pre_act[i]
            if self.k_active < len(row):
                top_indices = np.argpartition(row, -self.k_active)[-self.k_active:]
                z_sparse[i, top_indices] = row[top_indices]
            else:
                z_sparse[i] = row

        self.feature_fire_counts += np.sum(z_sparse > 0.0, axis=0).astype(np.int64)
        self.total_steps += batch_size

        return z_sparse.squeeze(0) if is_1d else z_sparse

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstructs dense activation x_hat from sparse concept latents: z @ W_dec + b_dec."""
        z_in = np.asarray(z, dtype=np.float32)
        is_1d = (z_in.ndim == 1)
        if is_1d:
            z_in = z_in[np.newaxis, :]

        x_hat = np.matmul(z_in, self.W_dec) + self.b_dec
        return x_hat.squeeze(0) if is_1d else x_hat

    def train_step(self, x: np.ndarray, lr: float = 0.02, auxk_coef: float = 0.1) -> Dict[str, Any]:
        """
        Single CPU optimization step for TopK SAE with AuxK dead-feature revival
        and unit-norm decoder dictionary projection (Gao et al. / Anthropic 2025-2026).
        """
        x_in = np.asarray(x, dtype=np.float32)
        if x_in.ndim == 1:
            x_in = x_in[np.newaxis, :]
        batch_size = max(1, x_in.shape[0])

        # Centering decoder bias slowly toward batch mean reduces activation outlier death
        self.b_dec = (self.b_dec + 0.05 * lr * np.mean(x_in - self.b_dec, axis=0)).astype(np.float32)
        x_cent = x_in - self.b_dec

        z = self.encode(x_in)
        if z.ndim == 1:
            z = z[np.newaxis, :]
        x_hat = self.decode(z)
        if x_hat.ndim == 1:
            x_hat = x_hat[np.newaxis, :]

        err = x_in - x_hat  # (B, d_model), negative gradient wrt x_hat
        recon_mse = float(np.mean(np.square(err)))

        active_mask = (z > 0.0).astype(np.float32)
        grad_W_dec = -np.matmul(z.T, err) / batch_size
        grad_z = -np.matmul(err, self.W_dec.T) * active_mask
        grad_W_enc = np.matmul(x_cent.T, grad_z) / batch_size
        grad_b_enc = np.mean(grad_z, axis=0)

        # AuxK dead-feature revival targeting unexplained residual error
        dead_mask = (self.feature_fire_counts == 0)
        if np.any(dead_mask) and auxk_coef > 0.0:
            dead_indices = np.where(dead_mask)[0]
            k_aux = min(self.k_active, len(dead_indices))
            dead_pre = np.maximum(0.0, np.matmul(x_cent, self.W_enc[:, dead_indices]) + self.b_enc[dead_indices])
            z_aux = np.zeros_like(dead_pre)
            for i in range(batch_size):
                row = dead_pre[i]
                if k_aux < len(row):
                    top_d = np.argpartition(row, -k_aux)[-k_aux:]
                    z_aux[i, top_d] = row[top_d]
                else:
                    z_aux[i] = row

            aux_active = (z_aux > 0.0).astype(np.float32)
            aux_err = err - np.matmul(z_aux, self.W_dec[dead_indices, :])
            grad_W_dec[dead_indices, :] += auxk_coef * (-np.matmul(z_aux.T, aux_err) / batch_size)
            grad_z_aux = -auxk_coef * np.matmul(aux_err, self.W_dec[dead_indices, :].T) * aux_active
            grad_W_enc[:, dead_indices] += np.matmul(x_cent.T, grad_z_aux) / batch_size
            grad_b_enc[dead_indices] += np.mean(grad_z_aux, axis=0)

            # Re-seed silent dead latents that had zero pre-activation toward unexplained residual
            silent_cols = dead_indices[np.sum(aux_active, axis=0) == 0]
            if len(silent_cols) > 0:
                mean_err = np.mean(err, axis=0)
                err_norm = float(np.linalg.norm(mean_err))
                mean_cent = np.mean(x_cent, axis=0)
                cent_norm = float(np.linalg.norm(mean_cent))
                if err_norm > 1e-8 and cent_norm > 1e-8:
                    err_unit = (mean_err / err_norm).astype(np.float32)
                    cent_unit = (mean_cent / cent_norm).astype(np.float32)
                    reseed = silent_cols[:k_aux]
                    blend = min(0.5, float(lr) * 2.0)
                    self.W_dec[reseed, :] = (1.0 - blend) * self.W_dec[reseed, :] + blend * err_unit
                    self.W_enc[:, reseed] = (1.0 - blend) * self.W_enc[:, reseed] + blend * cent_unit[:, np.newaxis]
                    self.b_enc[reseed] += float(lr) * float(auxk_coef)

        # Orthogonal gradient projection for unit-norm decoder dictionary rows
        grad_W_dec -= np.sum(grad_W_dec * self.W_dec, axis=1, keepdims=True) * self.W_dec

        step = float(lr)
        self.W_dec = (self.W_dec - step * grad_W_dec).astype(np.float32)
        self.W_enc = (self.W_enc - step * grad_W_enc).astype(np.float32)
        self.b_enc = (self.b_enc - step * grad_b_enc).astype(np.float32)

        # Re-normalize W_dec rows to unit l2 norm
        norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True) + 1e-12
        self.W_dec = (self.W_dec / norms).astype(np.float32)

        return {
            "recon_mse": round(recon_mse, 6),
            "active_features": int(np.count_nonzero(z)),
            "dead_features_remaining": int(np.sum(self.feature_fire_counts == 0)),
        }

    def steer_concept(self, x: np.ndarray, concept_idx: int, factor: float = 2.0) -> np.ndarray:
        """
        Concept-level steering: encodes x, scales specific concept_idx, and reconstructs x_steered.
        Bypasses polysemantic entanglement without touching unrelated concepts.
        """
        idx = int(concept_idx) % self.d_sae
        z = self.encode(x)
        z_steered = z.copy()
        if z_steered.ndim == 1:
            z_steered[idx] = (z_steered[idx] + 0.5) * factor
        else:
            z_steered[:, idx] = (z_steered[:, idx] + 0.5) * factor

        return self.decode(z_steered)

