"""NIR-dialect graph export for the liquid-state reservoir.

P: 2026-11-18 — robotics extension. Emits a NIR-*shaped* JSON description of a
LiquidStateReservoir: an input node feeding a leaky-integrator-with-tanh
recurrent layer feeding a linear readout, as nodes/edges/primitives.

HONESTY NOTE: this is a dialect that mirrors the Neuromorphic Intermediate
Representation's graph structure (named nodes carrying primitive parameter
dicts, edges as (source, source_port, target, target_port) tuples). It is NOT
a certified NIR export: the ``nir`` package is not a dependency of this
library and the exact primitive schemas (``nir.Affine``, ``nir.LI``) may
differ from the parameter names chosen here. A round-trip validation test
runs against the real ``nir`` package when it is installed (and is skipped
otherwise). Validate with the nir toolchain before feeding any hardware
backend (Speck / Akida / Loihi-class targets).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

_NIR_DIALECT_VERSION = 1


def export_reservoir_nir(
    reservoir: Any,
    include_readout: bool = True,
) -> Dict[str, Any]:
    """
    Export a LiquidStateReservoir as a NIR-dialect graph dict.

    Structure:
      "input" (input) -> "reservoir" (leaky_integrator_tanh, W_in + W_res) -> ["readout" (affine, W_readout)]
    Continuous-time CfC variants are exported in discrete form with tau derived
    from leak_rate (dt=1 mapping); the export records this in "notes".
    """
    W_res = np.asarray(getattr(reservoir, "W_res"), dtype=np.float64)
    W_in = np.asarray(getattr(reservoir, "W_in"), dtype=np.float64)
    leak = float(getattr(reservoir, "leak_rate", 0.3))

    n, m = int(W_in.shape[0]), int(W_in.shape[1])
    nodes: Dict[str, Dict[str, Any]] = {
        "input": {"primitive": "input", "shape": [m]},
        "reservoir": {
            "primitive": "leaky_integrator_tanh",
            "tau": round(1.0 / max(leak, 1e-6), 6),
            "input_weights": W_in.tolist(),
            "recurrent_weights": W_res.tolist(),
            "shape": [n],
        },
    }
    edges: List[List[Any]] = [
        ["input", "output", "reservoir", "input"],
    ]

    if include_readout and getattr(reservoir, "W_readout", None) is not None:
        W_out = np.asarray(reservoir.W_readout, dtype=np.float64)
        nodes["readout"] = {
            "primitive": "affine",
            "weights": W_out.T.tolist(),  # NIR affine weights are (out, in)
            "bias": [0.0] * int(W_out.shape[1]),
            "shape": [int(W_out.shape[1])],
        }
        edges.append(["reservoir", "output", "readout", "input"])

    graph = {
        "nir_dialect_version": _NIR_DIALECT_VERSION,
        "producer": "aetheris_microbrains.nir_export",
        "notes": [
            "Discrete-export mapping: tau = 1/leak_rate with dt=1.",
            "Readout converted from reservoir convention (N,M) to NIR affine (M,N).",
            "Validate against the nir package before hardware use; see module docstring.",
        ],
        "nodes": nodes,
        "edges": edges,
    }
    return graph


def validate_nir_dialect_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Structural validation of a nir-dialect graph (no nir package needed)."""
    problems: List[str] = []
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    if "input" not in nodes:
        problems.append("missing input node")
    if "reservoir" not in nodes:
        problems.append("missing reservoir node")
    if not isinstance(nodes.get("reservoir", {}).get("recurrent_weights", None), list):
        problems.append("reservoir missing recurrent_weights")
    for src, _sp, dst, _tp in edges:
        if src not in nodes:
            problems.append(f"edge references unknown source node {src}")
        if dst not in nodes:
            problems.append(f"edge references unknown target node {dst}")
    # shape consistency input -> reservoir
    try:
        n = len(nodes["reservoir"]["recurrent_weights"])
        in_width = len(nodes["reservoir"]["input_weights"][0])
        if int(nodes["input"].get("shape", [0])[0]) != in_width:
            problems.append("input node shape != input_weights width")
        if len(nodes["reservoir"]["input_weights"]) != n:
            problems.append("input_weights rows != reservoir dim")
        if "readout" in nodes:
            r_w = nodes["readout"]["weights"]
            if len(r_w) != len(nodes["readout"]["bias"]):
                problems.append("readout bias/weight mismatch")
            if n > 0 and len(r_w[0]) != n:
                problems.append("readout input dim != reservoir dim")
    except (IndexError, TypeError, KeyError) as exc:
        problems.append(f"malformed weights structure: {exc}")
    return {"valid": not problems, "problems": problems, "n_nodes": len(nodes), "n_edges": len(edges)}
