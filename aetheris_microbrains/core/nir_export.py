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

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def export_reservoir_nir_certified(
    reservoir: Any,
    dt: float = 0.001,
    include_readout: bool = True,
    hdf5_path: Optional[Union[str, Path]] = None,
    neuron_type: str = "li",
) -> Dict[str, Any]:
    """
    Export a LiquidStateReservoir using physical time constant ``tau_phys = dt / max(leak_rate, 1e-6)``.

    When the ``nir`` package is installed, constructs a genuine ``nir.NIRGraph``
    (``nir.Input``, ``nir.Affine``, ``nir.LI`` / ``nir.LIF`` / ``nir.CubaLIF``, ``nir.Output``)
    and writes it via ``nir.write(...)`` if ``hdf5_path`` is provided. When ``nir``
    is absent, returns a deterministic certified-schema dict (or raises ``RuntimeError``
    if ``hdf5_path`` was explicitly requested).
    """
    dt_val = float(dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be positive")
    norm_type = neuron_type.strip().lower().replace("_", "")
    if norm_type not in ("li", "lif", "cubalif"):
        raise ValueError(f"Unsupported neuron_type {neuron_type!r}; expected 'li', 'lif', or 'cubalif'")

    W_res = np.asarray(getattr(reservoir, "W_res"), dtype=np.float64)
    W_in = np.asarray(getattr(reservoir, "W_in"), dtype=np.float64)
    leak = float(getattr(reservoir, "leak_rate", 0.3))
    tau_phys = dt_val / max(leak, 1e-6)
    n, m = int(W_in.shape[0]), int(W_in.shape[1])

    has_readout = bool(include_readout and getattr(reservoir, "W_readout", None) is not None)
    W_out = np.asarray(reservoir.W_readout, dtype=np.float64) if has_readout else None
    out_dim = int(W_out.shape[1]) if W_out is not None else n

    try:
        import nir  # type: ignore
    except ImportError:
        nir = None

    if hdf5_path is not None and nir is None:
        raise RuntimeError(
            "export_reservoir_nir_certified with hdf5_path requires the 'nir' package (pip install nir). "
            "No HDF5 file was written."
        )

    schema_nodes: Dict[str, Dict[str, Any]] = {
        "input": {"nir_type": "nir.Input", "input_type": [m]},
        "input_affine": {
            "nir_type": "nir.Affine",
            "weight_shape": [n, m],
            "bias_shape": [n],
        },
        "reservoir": {
            "nir_type": {"li": "nir.LI", "lif": "nir.LIF", "cubalif": "nir.CubaLIF"}[norm_type],
            "tau": tau_phys,
            "tau_phys": tau_phys,
            "shape": [n],
        },
        "recurrent": {
            "nir_type": "nir.Affine",
            "weight_shape": [n, n],
            "bias_shape": [n],
        },
    }
    schema_edges: List[List[str]] = [
        ["input", "input_affine"],
        ["input_affine", "reservoir"],
        ["reservoir", "recurrent"],
        ["recurrent", "reservoir"],
    ]
    if W_out is not None:
        schema_nodes["readout"] = {
            "nir_type": "nir.Affine",
            "weight_shape": [out_dim, n],
            "bias_shape": [out_dim],
        }
        schema_nodes["output"] = {"nir_type": "nir.Output", "output_type": [out_dim]}
        schema_edges.extend([["reservoir", "readout"], ["readout", "output"]])
    else:
        schema_nodes["output"] = {"nir_type": "nir.Output", "output_type": [n]}
        schema_edges.append(["reservoir", "output"])

    hdf5_written = False
    nir_graph_built = False
    if nir is not None:
        tau_vec = np.full(n, tau_phys, dtype=np.float64)
        r_vec = np.ones(n, dtype=np.float64)
        v_leak_vec = np.zeros(n, dtype=np.float64)
        if norm_type == "li":
            res_node = nir.LI(tau=tau_vec, r=r_vec, v_leak=v_leak_vec)
        elif norm_type == "lif":
            res_node = nir.LIF(
                tau=tau_vec,
                r=r_vec,
                v_leak=v_leak_vec,
                v_threshold=np.ones(n, dtype=np.float64),
            )
        else:
            res_node = nir.CubaLIF(
                tau_mem=tau_vec,
                tau_syn=tau_vec * 0.5,
                r=r_vec,
                v_leak=v_leak_vec,
                v_threshold=np.ones(n, dtype=np.float64),
                w_in=np.ones(n, dtype=np.float64),
            )

        nir_nodes: Dict[str, Any] = {
            "input": nir.Input(input_type=np.array([m])),
            "input_affine": nir.Affine(weight=W_in, bias=np.zeros(n, dtype=np.float64)),
            "reservoir": res_node,
            "recurrent": nir.Affine(weight=W_res, bias=np.zeros(n, dtype=np.float64)),
        }
        if W_out is not None:
            nir_nodes["readout"] = nir.Affine(weight=W_out.T, bias=np.zeros(out_dim, dtype=np.float64))
            nir_nodes["output"] = nir.Output(output_type=np.array([out_dim]))
        else:
            nir_nodes["output"] = nir.Output(output_type=np.array([n]))

        nir_edges = [(src, dst) for src, dst in schema_edges]
        nir_graph = nir.NIRGraph(nodes=nir_nodes, edges=nir_edges)
        nir_graph_built = True
        if hdf5_path is not None:
            out_p = Path(hdf5_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            nir.write(str(out_p), nir_graph)
            hdf5_written = True

    return {
        "schema": "nir.NIRGraph.v1",
        "producer": "aetheris_microbrains.nir_export",
        "nir_installed": nir is not None,
        "certified_nir_graph": nir_graph_built,
        "hdf5_written": hdf5_written,
        "hdf5_path": str(hdf5_path) if hdf5_path is not None else None,
        "dt": dt_val,
        "leak_rate": leak,
        "tau_phys": tau_phys,
        "neuron_type": norm_type,
        "input_dim": m,
        "reservoir_dim": n,
        "output_dim": out_dim,
        "nodes": schema_nodes,
        "edges": schema_edges,
    }


class DVSEventSpikeEncoder:
    """
    Encode asynchronous Dynamic Vision Sensor ``(x, y, t_us, polarity)`` packets
    into binned spike rates or spatiotemporal spike tensors compatible with
    ``LiquidStateReservoir`` and neuromorphic inputs.
    """

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        out_dim: Optional[int] = None,
        n_bins: int = 1,
        polarity_mode: str = "signed",
        seed: int = 1337,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if polarity_mode not in ("signed", "two_channel", "unsigned"):
            raise ValueError("polarity_mode must be 'signed', 'two_channel', or 'unsigned'")
        self.width = int(width)
        self.height = int(height)
        self.n_bins = int(n_bins)
        self.polarity_mode = polarity_mode
        self.channels = 2 if polarity_mode == "two_channel" else 1
        self.flat_dim = self.channels * self.height * self.width
        self.out_dim = int(out_dim) if out_dim is not None else self.flat_dim
        self._proj: Optional[np.ndarray] = None
        if self.out_dim != self.flat_dim:
            rng = np.random.RandomState(int(seed))
            scale = 1.0 / np.sqrt(float(self.flat_dim))
            self._proj = (rng.randn(self.flat_dim, self.out_dim) * scale).astype(np.float32)

    @staticmethod
    def _parse_events(events: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if events is None:
            empty = np.zeros(0, dtype=np.float64)
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty, empty
        if isinstance(events, np.ndarray):
            if events.size == 0:
                empty = np.zeros(0, dtype=np.float64)
                return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty, empty
            arr = np.asarray(events, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return (
                arr[:, 0].astype(np.int64),
                arr[:, 1].astype(np.int64),
                arr[:, 2].astype(np.float64),
                arr[:, 3].astype(np.float64),
            )
        xs: List[int] = []
        ys: List[int] = []
        ts: List[float] = []
        ps: List[float] = []
        for ev in events:
            if isinstance(ev, dict):
                xs.append(int(ev["x"]))
                ys.append(int(ev["y"]))
                ts.append(float(ev.get("t_us", ev.get("t", 0.0))))
                ps.append(float(ev.get("polarity", ev.get("p", 1.0))))
            else:
                xs.append(int(ev[0]))
                ys.append(int(ev[1]))
                ts.append(float(ev[2]))
                ps.append(float(ev[3]))
        return (
            np.asarray(xs, dtype=np.int64),
            np.asarray(ys, dtype=np.int64),
            np.asarray(ts, dtype=np.float64),
            np.asarray(ps, dtype=np.float64),
        )

    def encode_events(
        self,
        events: Any,
        n_bins: Optional[int] = None,
        t_start_us: Optional[float] = None,
        t_end_us: Optional[float] = None,
        mode: str = "rate",
        squeeze: bool = True,
    ) -> np.ndarray:
        """
        Encode ``(x, y, t_us, polarity)`` events into spike rates or spatiotemporal tensors.

        Modes:
          - ``"rate"``: event count per bin normalized by bin duration (seconds) -> ``(n_bins, out_dim)`` (or ``(out_dim,)`` when ``n_bins==1 and squeeze``)
          - ``"spatiotemporal"``: raw spatial grid ``(n_bins, channels, height, width)``
          - ``"binary"``: binary spike indicator ``{0.0, 1.0}`` per bin -> ``(n_bins, out_dim)`` (or ``(out_dim,)`` when ``n_bins==1 and squeeze``)
        """
        if mode not in ("rate", "spatiotemporal", "binary"):
            raise ValueError(f"Unsupported mode {mode!r}; expected 'rate', 'spatiotemporal', or 'binary'")
        bins = int(n_bins) if n_bins is not None else self.n_bins
        if bins <= 0:
            raise ValueError("n_bins must be positive")

        xs, ys, ts, ps = self._parse_events(events)
        grid = np.zeros((bins, self.channels, self.height, self.width), dtype=np.float32)

        if xs.size > 0:
            valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
            xs, ys, ts, ps = xs[valid], ys[valid], ts[valid], ps[valid]

        if xs.size > 0:
            t0 = float(t_start_us) if t_start_us is not None else float(np.min(ts))
            t1 = float(t_end_us) if t_end_us is not None else float(np.max(ts))
            span_us = max(t1 - t0, 1.0)
            rel = np.clip((ts - t0) / span_us, 0.0, 1.0 - 1e-12)
            bin_idx = np.clip((rel * bins).astype(np.int64), 0, bins - 1)

            if self.polarity_mode == "unsigned":
                np.add.at(grid, (bin_idx, 0, ys, xs), 1.0)
            elif self.polarity_mode == "signed":
                signs = np.where(ps > 0, 1.0, -1.0)
                np.add.at(grid, (bin_idx, 0, ys, xs), signs.astype(np.float32))
            else:  # two_channel: ch 1 = ON (>0), ch 0 = OFF (<=0)
                ch_idx = np.where(ps > 0, 1, 0)
                np.add.at(grid, (bin_idx, ch_idx, ys, xs), 1.0)

            if mode == "rate":
                bin_dur_s = (span_us / bins) * 1e-6
                if bin_dur_s > 0.0:
                    grid = grid / np.float32(bin_dur_s)

        if mode == "spatiotemporal":
            return grid

        flat = grid.reshape(bins, self.flat_dim)
        if mode == "binary":
            flat = (flat != 0.0).astype(np.float32)

        if self._proj is not None:
            flat = (flat @ self._proj).astype(np.float32)

        if bins == 1 and squeeze:
            return flat[0]
        return flat
