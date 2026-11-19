"""
Dynamic Tool: MicroBrain Lab — experimental neural micro-architectures
======================================================================
Invokable access to the four experimental CPU micro-brain architectures:

- dendritic_forward:     multi-compartment dendritic tree neuron layer.
- ssm_step/ssm_sequence: linear state-space (Mamba-style) recurrent core —
                         O(sequence) time, O(1) state; same instance persists
                         across calls so sequences can be streamed tick-by-tick.
- kan_forward:           Kolmogorov-Arnold network layer (Chebyshev edge basis).
- kan_extract_formula:   decompile the KAN weights into symbolic formulas.
- hypernet_synthesize:   zero-shot fast-weight synthesis from a task description
                         (spectrally normalized + BitNet b1.58 ternary form).

All latency figures are measured at runtime, never asserted literals.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

schema = {
    "name": "microbrain_lab",
    "description": "Run experimental CPU/GPU micro-brain architectures: dendritic tree neurons (+ three-factor plateau adaptation), linear state-space Mamba core (+ Blelloch parallel scan), Kolmogorov-Arnold Chebyshev network (+ symbolic ridge fitting & formula extraction), BitNet v2 H-BitLinear Hadamard rotation, Mixture-of-Depths gate calibration, and zero-shot Titans/Muon hypernetwork adaptation. All latencies measured live.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "dendritic_forward", "dendritic_adapt",
                    "ssm_step", "ssm_sequence", "ssm_parallel_scan",
                    "ssm_mamba3_step", "ssm_mamba3_sequence", "ssm_fit_readout",
                    "kan_forward", "kan_adapt_grid", "kan_fit_symbolic", "kan_extract_formula",
                    "hypernet_synthesize", "hypernet_titans_muon",
                    "bitnet_hadamard", "mod_train_gate",
                    "liquid_cfc_step", "pc_incremental_step", "pc_generate_top_down",
                    "topo_geodesic_merge", "vsa_resonator_factorize",
                    "mone_delta_step", "vsa_attention_query", "wsp_memit_implant",
                ],
                "description": "Micro-architecture operation to run."
            },
            "input_vector": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Numeric input vector for dendritic_forward / dendritic_adapt / ssm_step / kan_forward / bitnet_hadamard."
            },
            "target_vector": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Target vector for dendritic_adapt."
            },
            "sequence": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "List of input vectors for ssm_sequence / ssm_parallel_scan / mod_train_gate / kan_fit_symbolic."
            },
            "x_samples": {
                "type": "array",
                "description": "Input batch for kan_fit_symbolic / hypernet_titans_muon / mod_train_gate."
            },
            "y_targets": {
                "type": "array",
                "description": "Target batch for kan_fit_symbolic / hypernet_titans_muon / mod_train_gate."
            },
            "reset_state": {
                "type": "boolean",
                "description": "Reset the SSM recurrent state before stepping.",
                "default": False
            },
            "task_description": {
                "type": "string",
                "description": "Task text for hypernet_synthesize."
            },
            "out_dim": {
                "type": "integer",
                "description": "Output rows of weight matrix.",
                "default": 64
            },
            "in_dim": {
                "type": "integer",
                "description": "Input columns of weight matrix.",
                "default": 64
            },
            "lr": {
                "type": "number",
                "description": "Learning rate for adaptation actions.",
                "default": 0.05
            },
            "reward": {
                "type": "number",
                "description": "Three-factor plasticity reward multiplier.",
                "default": 1.0
            },
            "epochs": {
                "type": "integer",
                "description": "Optimization epochs for mod_train_gate.",
                "default": 25
            },
            "weights_a": {
                "type": "array",
                "items": {"type": "number"},
                "description": "First flattened weight vector for topo_geodesic_merge."
            },
            "weights_b": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Second flattened weight vector for topo_geodesic_merge."
            },
            "composite_vec": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Superposed HD-VSA composite hypervector for vsa_resonator_factorize."
            },
            "candidate_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate tool names codebook for vsa_resonator_factorize."
            },
            "candidate_roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate role names codebook for vsa_resonator_factorize."
            },
            "micro_steps": {
                "type": "integer",
                "description": "Interleaved micro-steps for pc_incremental_step.",
                "default": 6
            },
            "state_lr": {
                "type": "number",
                "description": "State inference step size for pc_incremental_step.",
                "default": 0.12
            },
            "delta_t": {
                "type": "number",
                "description": "Elapsed continuous time interval for liquid_cfc_step."
            },
            "adapt_domain_grid": {
                "type": "boolean",
                "description": "Auto-adapt KAN domain grid to x_samples range in kan_fit_symbolic.",
                "default": False
            },
            "key_vector": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Key vector for mone_delta_step / vsa_attention_query."
            },
            "value_vector": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Value vector for mone_delta_step / vsa_attention_query."
            },
            "beta_gate": {
                "type": "number",
                "description": "Gated delta-rule learning gate for mone_delta_step.",
                "default": 1.0
            },
            "label": {
                "type": "string",
                "description": "Optional codebook label for vsa_attention_query."
            },
            "key_matrix": {
                "type": "array",
                "description": "Key rows matrix for wsp_memit_implant."
            },
            "value_matrix": {
                "type": "array",
                "description": "Value rows matrix for wsp_memit_implant."
            },
            "weight_matrix": {
                "type": "array",
                "description": "Optional base weight matrix for wsp_memit_implant."
            },
            "top_latent": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Top-level latent vector r2 for pc_generate_top_down."
            }
        },
        "required": ["action"]
    }
}


def main(
    action: str = "kan_extract_formula",
    input_vector: Optional[List[float]] = None,
    target_vector: Optional[List[float]] = None,
    sequence: Optional[List[List[float]]] = None,
    x_samples: Optional[Any] = None,
    y_targets: Optional[Any] = None,
    reset_state: bool = False,
    task_description: str = "generic aetheris tool specialization",
    out_dim: int = 64,
    in_dim: int = 64,
    lr: float = 0.05,
    reward: float = 1.0,
    epochs: int = 25,
    **kwargs: Any
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    if action == "dendritic_forward":
        from ..core.dendritic_tree_microbrain import DendriticTreeMicroBrain
        if input_vector is None:
            return {"success": False, "error": "dendritic_forward requires input_vector"}
        res = DendriticTreeMicroBrain.get_instance().forward(input_vector)
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "dendritic_adapt":
        from ..core.dendritic_tree_microbrain import DendriticTreeMicroBrain
        if input_vector is None or target_vector is None:
            return {"success": False, "error": "dendritic_adapt requires input_vector and target_vector"}
        res = DendriticTreeMicroBrain.get_instance().adapt_dendritic_plateau(
            input_vector, target_vector, reward=float(reward), lr=float(lr),
        )
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action in ("ssm_step", "ssm_sequence", "ssm_parallel_scan", "ssm_mamba3_step", "ssm_mamba3_sequence", "ssm_fit_readout"):
        from ..core.linear_state_space_microbrain import LinearStateSpaceMicroBrain
        ssm = LinearStateSpaceMicroBrain.get_instance()
        if reset_state:
            ssm.reset()
        if action == "ssm_step":
            if input_vector is None:
                return {"success": False, "error": "ssm_step requires input_vector"}
            res = ssm.step(input_vector)
        elif action == "ssm_mamba3_step":
            if input_vector is None:
                return {"success": False, "error": "ssm_mamba3_step requires input_vector"}
            res = ssm.step_mamba3(input_vector)
        elif action == "ssm_mamba3_sequence":
            if not sequence:
                return {"success": False, "error": "ssm_mamba3_sequence requires sequence"}
            res = ssm.process_sequence_mamba3(sequence)
        elif action == "ssm_fit_readout":
            xs = x_samples if x_samples is not None else sequence
            ys = y_targets
            if xs is None or ys is None:
                return {"success": False, "error": "ssm_fit_readout requires sequence/x_samples and y_targets"}
            res = ssm.fit_readout(xs, ys)
        elif action == "ssm_parallel_scan":
            if not sequence:
                return {"success": False, "error": "ssm_parallel_scan requires sequence"}
            res = ssm.process_sequence_parallel_scan(sequence)
            if "all_outputs" in res and hasattr(res["all_outputs"], "tolist"):
                res["all_outputs"] = res["all_outputs"].tolist()
        else:
            if not sequence:
                return {"success": False, "error": "ssm_sequence requires sequence"}
            res = ssm.process_sequence(sequence)
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "kan_adapt_grid":
        from ..core.kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain
        xs = x_samples if x_samples is not None else sequence
        if xs is None:
            return {"success": False, "error": "kan_adapt_grid requires x_samples or sequence"}
        res = KolmogorovArnoldMicroBrain.get_instance().adapt_grid(xs)
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "kan_forward":
        from ..core.kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain
        if input_vector is None:
            return {"success": False, "error": "kan_forward requires input_vector"}
        y = KolmogorovArnoldMicroBrain.get_instance().forward(input_vector)
        return {
            "success": True,
            "action": action,
            "output_vector": y.tolist(),
            "wrapper_latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        }

    elif action == "kan_fit_symbolic":
        from ..core.kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain
        xs = x_samples if x_samples is not None else sequence
        ys = y_targets
        if xs is None or ys is None:
            return {"success": False, "error": "kan_fit_symbolic requires x_samples and y_targets"}
        kan = KolmogorovArnoldMicroBrain.get_instance()
        fit_res = kan.fit_symbolic(xs, ys, adapt_domain_grid=bool(kwargs.get("adapt_domain_grid", False)))
        formula_res = kan.extract_symbolic_formula()
        return {
            "success": True,
            "action": action,
            "fit": fit_res,
            "formula": formula_res,
            "wrapper_latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        }

    elif action == "kan_extract_formula":
        from ..core.kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain
        res = KolmogorovArnoldMicroBrain.get_instance().extract_symbolic_formula()
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "hypernet_synthesize":
        from ..core.microbrain_hypernetwork import MicroBrainHypernetwork
        res = MicroBrainHypernetwork.get_instance().synthesize_weights(
            task_description, out_dim=int(out_dim), in_dim=int(in_dim),
        )
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "hypernet_titans_muon":
        from ..core.microbrain_hypernetwork import MicroBrainHypernetwork
        xs = x_samples if x_samples is not None else sequence
        ys = y_targets
        if xs is None or ys is None:
            return {"success": False, "error": "hypernet_titans_muon requires x_samples and y_targets"}
        res = MicroBrainHypernetwork.get_instance().adapt_titans_muon_ttt(xs, ys, lr=float(lr))
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "bitnet_hadamard":
        import numpy as np
        from ..core.bitnet_ternary_engine import BitLinearLayer, BitNetQuantizer
        vec = input_vector if input_vector is not None else sequence
        if vec is None:
            return {"success": False, "error": "bitnet_hadamard requires input_vector or sequence"}
        arr = np.asarray(vec, dtype=np.float32)
        in_d = arr.shape[-1] if arr.ndim > 1 else arr.size
        out_d = int(kwargs.get("layer_out_dim", min(max(1, out_dim), 64)))
        layer = BitLinearLayer(in_features=in_d, out_features=out_d)
        out_rot, stats = layer.forward_hadamard(arr)
        packed = BitNetQuantizer.pack_ternary_2bit(layer.ternary_weight)
        return {
            "success": True,
            "action": action,
            "output_vector": out_rot.tolist(),
            "stats": stats,
            "packed_2bit_bytes": int(packed.nbytes),
            "wrapper_latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        }

    elif action == "mod_train_gate":
        from ..core.mixture_of_depths_router import MixtureOfDepthsRouter
        xs = x_samples if x_samples is not None else sequence
        ys = y_targets
        if xs is None or ys is None:
            return {"success": False, "error": "mod_train_gate requires x_samples and y_targets"}
        res = MixtureOfDepthsRouter.get_instance().train_exit_gate(xs, ys, lr=float(lr), epochs=int(epochs))
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "liquid_cfc_step":
        from ..core.liquid_state_reservoir import LiquidStateReservoir
        resv = LiquidStateReservoir.get_instance()
        if reset_state:
            resv.reset_state()
        if x_samples is not None and y_targets is not None:
            fit_res = resv.fit_readout(x_samples, y_targets)
            if input_vector is None:
                fit_res["success"] = True
                fit_res["action"] = action
                fit_res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
                return fit_res
        if input_vector is None:
            return {"success": False, "error": "liquid_cfc_step requires input_vector or (x_samples, y_targets)"}
        res = resv.step_cfc_continuous(input_vector, delta_t=kwargs.get("delta_t"))
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "pc_incremental_step":
        from ..core.predictive_coding_engine import HierarchicalPredictiveCodingEngine
        if input_vector is None:
            return {"success": False, "error": "pc_incremental_step requires input_vector"}
        res = HierarchicalPredictiveCodingEngine.get_instance().step_incremental_pc(
            input_vector, micro_steps=int(kwargs.get("micro_steps", 6)), state_lr=float(kwargs.get("state_lr", 0.12)),
        )
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "pc_generate_top_down":
        from ..core.predictive_coding_engine import HierarchicalPredictiveCodingEngine
        top_lat = kwargs.get("top_latent") or input_vector
        if top_lat is None:
            return {"success": False, "error": "pc_generate_top_down requires top_latent or input_vector"}
        res = HierarchicalPredictiveCodingEngine.get_instance().generate_top_down(top_lat)
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "topo_geodesic_merge":
        from ..core.topological_persistence_merger import TopologicalPersistenceMerger
        wa = kwargs.get("weights_a") or input_vector
        wb = kwargs.get("weights_b") or target_vector
        if wa is None or wb is None:
            return {"success": False, "error": "topo_geodesic_merge requires weights_a and weights_b"}
        res = TopologicalPersistenceMerger.get_instance().multiscale_geodesic_slerp_merge(wa, wb)
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "vsa_resonator_factorize":
        from ..core.hyperdimensional_vsa_engine import HyperdimensionalVSAEngine
        comp = kwargs.get("composite_vec") or input_vector
        tools = kwargs.get("candidate_tools") or []
        roles = kwargs.get("candidate_roles") or []
        if comp is None or not tools or not roles:
            return {"success": False, "error": "vsa_resonator_factorize requires composite_vec/input_vector, candidate_tools, candidate_roles"}
        res = HyperdimensionalVSAEngine.get_instance().factorize_resonator(
            comp, tools, roles, top_k_factors=int(kwargs.get("top_k_factors", 1)),
        )
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "mone_delta_step":
        from ..core.mone_fast_weight_memory import MoNeFastWeightMemory
        k_vec = kwargs.get("key_vector") or input_vector
        v_vec = kwargs.get("value_vector") or target_vector
        if k_vec is None or v_vec is None:
            return {"success": False, "error": "mone_delta_step requires key_vector and value_vector"}
        mone = MoNeFastWeightMemory.get_instance()
        if reset_state:
            mone.clear()
        momentum_beta = float(kwargs.get("momentum_beta", 0.0))
        res = mone.ingest_delta_rule(
            k_vec,
            v_vec,
            beta_gate=float(kwargs.get("beta_gate", 1.0)),
            momentum_beta=momentum_beta,
        )
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "vsa_attention_query":
        import numpy as np
        from ..core.vsa_algebraic_attention import VSAAlgebraicAttentionEngine
        k_vec = kwargs.get("key_vector") or input_vector
        if k_vec is None:
            return {"success": False, "error": "vsa_attention_query requires key_vector or input_vector"}
        vsa_attn = VSAAlgebraicAttentionEngine.get_instance()
        if reset_state:
            vsa_attn.clear()
        v_vec = kwargs.get("value_vector") or target_vector
        if v_vec is not None:
            vsa_attn.bind_and_accumulate(np.asarray(k_vec), np.asarray(v_vec), label=kwargs.get("label"))
        res = vsa_attn.query_with_cleanup(np.asarray(k_vec))
        res["success"] = True
        res["action"] = action
        res["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return res

    elif action == "wsp_memit_implant":
        import numpy as np
        from ..core.wsp_neural_surgeon import WSPNeuralSurgeon
        k_mat = kwargs.get("key_matrix") or x_samples
        v_mat = kwargs.get("value_matrix") or y_targets
        if k_mat is None or v_mat is None:
            return {"success": False, "error": "wsp_memit_implant requires key_matrix/x_samples and value_matrix/y_targets"}
        k_arr = np.asarray(k_mat, dtype=np.float64)
        v_arr = np.asarray(v_mat, dtype=np.float64)
        w_base = kwargs.get("weight_matrix")
        if w_base is not None:
            w_arr = np.asarray(w_base, dtype=np.float64)
        else:
            d_in = k_arr.shape[1] if k_arr.ndim > 1 else len(k_arr)
            d_out = v_arr.shape[1] if v_arr.ndim > 1 else len(v_arr)
            w_arr = np.zeros((d_out, d_in), dtype=np.float64)
        w_new, report = WSPNeuralSurgeon.get_instance().direct_multifact_memit_implant(w_arr, k_arr, v_arr)
        report["success"] = True
        report["action"] = action
        report["updated_weights_shape"] = list(w_new.shape)
        report["wrapper_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return report

    return {"success": False, "error": f"Unknown action: {action}"}
