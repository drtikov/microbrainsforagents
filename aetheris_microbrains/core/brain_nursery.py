"""
Autonomous Brain Nursery & Neurogenesis Incubator
=================================================
Manages the zero-shot growth, compilation, and registry of autonomous
analytical micro-transformers ("digital neuro-organs") within Aetheris.

When Aetheris encounters a specialized domain (e.g., fast exoplanet scoring,
table formatting, safety auditing, or intent classification), the Nursery:
1. Instantiates a dedicated AnalyticalMicroTransformer.
2. Compiles analytical induction heads and symbolic logic rules into weights.
3. Persists the weights into `memory/store/neuro_brains/`.
4. Exposes the brain for sub-millisecond CPU inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR

logger = logging.getLogger(__name__)


class AutonomousBrainNursery:
    """Orchestrates zero-gradient neural network creation, growth, and execution."""

    _instance: Optional[AutonomousBrainNursery] = None

    def __init__(self, store_dir: Optional[Path] = None):
        self._custom_store_dir: Optional[Path] = Path(store_dir) if store_dir is not None else None
        self.active_brains: Dict[str, AnalyticalMicroTransformer] = {}
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_brains()

    @property
    def store_dir(self) -> Path:
        return self._custom_store_dir if self._custom_store_dir is not None else STORE_DIR

    @classmethod
    def get_instance(cls) -> AutonomousBrainNursery:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_existing_brains(self):
        try:
            for meta_file in self.store_dir.glob("*_meta.json"):
                name = meta_file.name.replace("_meta.json", "")
                if "_bitnet" in name:
                    continue
                brain = AnalyticalMicroTransformer.load(name, self.store_dir, strict_directory=(self._custom_store_dir is not None))
                if brain:
                    self.active_brains[name] = brain
        except Exception as e:
            logger.debug(f"Brain nursery initial load skipped: {e}")

    @staticmethod
    def _normalize_brain_name(domain_name: str) -> str:
        clean = domain_name.lower().strip().replace(" ", "_")
        return clean if clean.startswith("neuro_") else f"neuro_{clean}"

    def _get_or_load_or_grow(self, domain_name: str) -> AnalyticalMicroTransformer:
        brain_name = self._normalize_brain_name(domain_name)
        if brain_name in self.active_brains:
            return self.active_brains[brain_name]
        loaded = AnalyticalMicroTransformer.load(brain_name, self.store_dir, strict_directory=(self._custom_store_dir is not None))
        if loaded is not None:
            self.active_brains[brain_name] = loaded
            return loaded
        return self.grow_brain_for_domain(domain_name)

    def grow_brain_for_domain(
        self,
        domain_name: str,
        rules: Optional[Dict[str, str]] = None,
        d_model: int = 32,
        n_heads: int = 2,
        d_mlp: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> AnalyticalMicroTransformer:
        """
        Grows an analytical micro-transformer brain for a specific domain.
        Compiles symbolic rules into exact weight matrices without backprop,
        preserving matching brains already persisted on disk.
        """
        brain_name = self._normalize_brain_name(domain_name)
        target_dmlp = d_mlp if d_mlp is not None else (d_model * 2)
        target_seq_len = max_seq_len if max_seq_len is not None else 16

        if brain_name in self.active_brains:
            brain = self.active_brains[brain_name]
        else:
            loaded = AnalyticalMicroTransformer.load(brain_name, self.store_dir, strict_directory=(self._custom_store_dir is not None))
            if (
                loaded is not None
                and loaded.d_model == d_model
                and loaded.n_heads == n_heads
                and loaded.d_mlp == target_dmlp
                and loaded.max_seq_len == target_seq_len
            ):
                brain = loaded
            else:
                brain = AnalyticalMicroTransformer(
                    name=brain_name,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_mlp=target_dmlp,
                    max_seq_len=target_seq_len,
                )
                # Compile analytical induction head for in-context sequence continuity
                brain.compile_induction_head(head_idx=0, copy_strength=2.5)

        # Compile domain rules directly into MLP weights
        if rules:
            for trigger, output in rules.items():
                brain.compile_logic_association(trigger, output, binding_gain=3.0)

        brain.save(self.store_dir)
        self.active_brains[brain_name] = brain
        return brain

    def infer(self, domain_name: str, input_tokens: List[str]) -> Dict[str, Any]:
        """Runs sub-millisecond CPU inference on a grown micro-transformer."""
        brain = self._get_or_load_or_grow(domain_name)
        return brain.forward(input_tokens)

    def list_grown_brains(self) -> List[Dict[str, Any]]:
        try:
            for meta_file in self.store_dir.glob("*_meta.json"):
                name = meta_file.name.replace("_meta.json", "")
                if "_bitnet" in name or name in self.active_brains:
                    continue
                loaded = AnalyticalMicroTransformer.load(name, self.store_dir, strict_directory=(self._custom_store_dir is not None))
                if loaded is not None:
                    self.active_brains[name] = loaded
        except Exception:
            pass
        return [
            {
                "name": b.name,
                "d_model": b.d_model,
                "n_heads": b.n_heads,
                "vocab_size": len(b.vocab),
                "max_seq_len": b.max_seq_len,
            }
            for b in self.active_brains.values()
        ]

    def evolve_population_eggroll(
        self,
        domain_name: str,
        x_samples: np.ndarray,
        y_targets: np.ndarray,
        pop_size: int = 256,
        sigma: float = 0.35,
        seed: int = 42,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        EGGROLL Low-Rank Neuroevolution in GPU VRAM:
        Evaluates `pop_size` low-rank perturbed daughter brains in parallel on the
        OpenCL GPU co-processor (or CPU fallback) and updates the domain brain's
        MLP projection with the top-scoring ternary mutation.
        """
        brain = self._get_or_load_or_grow(domain_name)
        brain_name = brain.name

        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()

        w_base = getattr(brain, "W_mlp", None)
        if w_base is None or not isinstance(w_base, np.ndarray):
            w_base = np.eye(brain.d_model, dtype=np.float32)
            # The rollout runs against the identity-matrix substitute when the
            # brain has no usable W_mlp; disclose that in the report.
            report = acc.eggroll_evolve_population(
                w_base=w_base,
                x_samples=x_samples,
                y_targets=y_targets,
                pop_size=pop_size,
                sigma=sigma,
                seed=seed,
                force_gpu=force_gpu,
            )
            best_w = report.pop("best_ternary_weights")
            report["weights_substituted"] = True
            report["brain_name"] = brain_name
            return report

        report = acc.eggroll_evolve_population(
            w_base=w_base,
            x_samples=x_samples,
            y_targets=y_targets,
            pop_size=pop_size,
            sigma=sigma,
            seed=seed,
            force_gpu=force_gpu,
        )
        best_w = report.pop("best_ternary_weights")
        if hasattr(brain, "W_mlp") and isinstance(brain.W_mlp, np.ndarray) and brain.W_mlp.shape == best_w.shape:
            brain.W_mlp = best_w.astype(np.float32)
            brain.save(self.store_dir)
        report["brain_name"] = brain_name
        return report

    def evolve_population_eggroll_es(
        self,
        domain_name: str,
        x_samples: np.ndarray,
        y_targets: np.ndarray,
        pop_size: int = 256,
        rank_r: int = 2,
        sigma: float = 0.25,
        lr: float = 0.20,
        seed: int = 42,
        force_gpu: bool = False,  # measured: GPU slower than CPU at production pop sizes on this box
    ) -> Dict[str, Any]:
        """
        True EGGROLL (arXiv:2511.16652) Evolution Strategies:
        Applies on-device counter-based RNG, rank-r factor perturbation, antithetic mirrored sampling,
        and centered rank-based fitness-shaped ES gradient update in VRAM.
        """
        brain = self._get_or_load_or_grow(domain_name)
        brain_name = brain.name

        identity_fallback = False
        w_base = getattr(brain, "W_mlp", None)
        if w_base is None or not isinstance(w_base, np.ndarray) or w_base.shape != (brain.d_model, brain.d_model):
            w_base = np.eye(brain.d_model, dtype=np.float32)
            identity_fallback = True

        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        step_res = acc.eggroll_es_step(
            w_base=w_base,
            x_samples=x_samples,
            y_targets=y_targets,
            pop_size=pop_size,
            rank_r=rank_r,
            sigma=sigma,
            lr=lr,
            seed=seed,
            force_gpu=force_gpu,
        )
        if identity_fallback:
            # The rollout evolved the identity-matrix substitute, not the
            # brain's actual weights — writing that back would silently
            # overwrite real weights with an evolved placeholder.
            step_res["weights_substituted"] = True
        else:
            brain.W_mlp = step_res["updated_weights"]
            brain.save(self.store_dir)
        step_res["brain_name"] = brain_name
        return step_res

    def evolve_population_dns(
        self,
        domain_name: str,
        x_samples: np.ndarray,
        y_targets: np.ndarray,
        pop_size: int = 256,
        k_nearest: int = 5,
        seed: int = 42,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Dominated Novelty Search (DNS, 2025 SOTA Quality-Diversity):
        Evolves a population using EGGROLL on-device rollouts, extracts 8D behavior descriptors,
        and uses the GPU DNS competition kernel to select the optimal Pareto-novelty candidate brain.
        """
        brain = self._get_or_load_or_grow(domain_name)
        brain_name = brain.name

        identity_fallback = False
        w_base = getattr(brain, "W_mlp", None)
        if w_base is None or not isinstance(w_base, np.ndarray) or w_base.shape != (brain.d_model, brain.d_model):
            w_base = np.eye(brain.d_model, dtype=np.float32)
            identity_fallback = True

        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        es_res = acc.eggroll_es_step(
            w_base=w_base,
            x_samples=x_samples,
            y_targets=y_targets,
            pop_size=pop_size,
            seed=seed,
            force_gpu=force_gpu,
        )
        behaviors = es_res.get("behavior_descriptors")
        if behaviors is None or len(behaviors) != es_res["population_size"]:
            behaviors = np.zeros((es_res["population_size"], 8), dtype=np.float32)

        # Fitness is negative candidate MSE per population member (higher is better).
        # Using candidate-specific MSE ensures Dominated Novelty Search performs real
        # quality-diversity competition rather than collapsing to pure outlier selection.
        cand_mses = es_res.get("candidate_mses")
        if cand_mses is not None and len(cand_mses) == es_res["population_size"]:
            fitness = -cand_mses.astype(np.float32)
        else:
            fitness = np.full(es_res["population_size"], -es_res["post_update_mse"], dtype=np.float32)

        dns_res = acc.dominated_novelty_search(
            fitness=fitness,
            descriptors=behaviors,
            k_nearest=k_nearest,
            force_gpu=force_gpu,
        )
        if not identity_fallback and "updated_weights" in es_res:
            brain.W_mlp = es_res["updated_weights"]
            brain.save(self.store_dir)
        return {
            "status": "dns_evolved",
            "brain_name": brain_name,
            "dns_backend": dns_res["backend"],
            "best_candidate_idx": dns_res["best_candidate_idx"],
            "best_candidate_dns_score": float(dns_res["dns_scores"][dns_res["best_candidate_idx"]]),
            "es_post_mse": es_res["post_update_mse"],
            "pop_size": es_res["population_size"],
            "identity_fallback": identity_fallback,
            "weights_substituted": identity_fallback,
        }

    def calibrate_brain_conformal(
        self,
        domain_name: str,
        calibration_data: List[Tuple[List[str], str]],
        alpha: float = 0.10,
    ) -> Dict[str, Any]:
        """Calibrates split-conformal prediction sets on the domain's micro-brain."""
        brain = self._get_or_load_or_grow(domain_name)
        cal = brain.calibrate_conformal(calibration_data, alpha=alpha)
        brain.save(self.store_dir)
        return cal

    def infer_brain_conformal(
        self,
        domain_name: str,
        input_tokens: List[str],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluates conformal prediction set with principled calibrated abstention."""
        brain = self._get_or_load_or_grow(domain_name)
        return brain.infer_conformal(input_tokens, alpha=alpha)


BrainNursery = AutonomousBrainNursery


def get_brain_nursery() -> AutonomousBrainNursery:
    return AutonomousBrainNursery.get_instance()

