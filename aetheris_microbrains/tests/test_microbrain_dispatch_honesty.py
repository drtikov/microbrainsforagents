"""Discriminating tests for the microbrain dispatch-honesty round (2026-11-12).

Contracts encoded here (all red against the pre-fix bodies):
- the GPU crossover table is a conservative bootstrap, never labeled
  "Empirically measured" without a measurement artifact;
- every kernel dispatch consults should_use_gpu — no inline thresholds, no
  unconditional GPU paths;
- eggroll ES no longer forces the GPU by default (measured: GPU slower at
  production pop sizes on this box);
- autotune refuses ops it cannot benchmark and never invents a crossover
  (the old max(candidate_sizes)*16 fallback) when the GPU never wins;
- the MoE cortex routes only over the real catalog (dynamic tools + nursery
  brains), reports "unrouted" honestly, and times itself with the clock;
- describe_image discloses that no vision model is wired.
"""

import ast
import inspect
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from aetheris_microbrains.core.opencl_microbrain_accelerator import (
    OpenCLMicroBrainAccelerator,
    get_opencl_accelerator,
)

ROOT = Path(__file__).resolve().parents[1]


class TestCrossoverTableHonesty:
    def test_no_fabricated_measurement_label(self):
        src = (ROOT / "core/opencl_microbrain_accelerator.py").read_text(encoding="utf-8")
        assert "Empirically measured" not in src, (
            "crossover table labeled as measured again — only real benchmark "
            "receipts (memory/store/opencl_crossover.json) may claim that"
        )
        assert "DEFAULT_CROSSOVER_BOOTSTRAP" in src

    def test_unknown_op_defaults_to_unmeasured_not_gpu(self):
        acc = get_opencl_accelerator()
        assert acc.get_crossover("op_that_was_never_measured_xyz") == acc.GPU_UNMEASURED
        # Even an enormous problem size must not dispatch an unmeasured op.
        assert acc.should_use_gpu("op_that_was_never_measured_xyz", problem_size=10**9) is False

    def test_only_measured_ops_carry_numeric_bootstrap(self):
        acc = get_opencl_accelerator()
        for op, threshold in acc.DEFAULT_CROSSOVER_BOOTSTRAP.items():
            if op != "hdc_hamming_search":
                assert threshold == acc.GPU_UNMEASURED, (
                    f"{op} bootstrap claims threshold {threshold} without a benchmark receipt"
                )

    def test_hdc_measured_bootstrap_dispatches_by_size(self):
        acc = get_opencl_accelerator()
        # The live table now carries REAL autotune measurements persisted to
        # memory/store/opencl_crossover.json (2026-11-12: hdc searches won at
        # >= 512 on the GT 710), so the test pins the dispatch CONTRACT, not
        # a synthetic bootstrap number: below the crossover GPU is refused,
        # at/above it GPU wins — and GPU_UNMEASURED never dispatches.
        crossover = acc.get_crossover("hdc_hamming_search")
        assert isinstance(crossover, int)
        assert acc.should_use_gpu(
            "hdc_hamming_search", problem_size=crossover - 1, force_gpu=False
        ) is False
        if acc.gpu_available and crossover != acc.GPU_UNMEASURED:
            assert acc.should_use_gpu(
                "hdc_hamming_search", problem_size=crossover, force_gpu=False
            ) is True
        else:
            assert acc.should_use_gpu(
                "hdc_hamming_search", problem_size=10**9, force_gpu=False
            ) is False

    def test_every_kernel_launch_site_is_gated_by_should_use_gpu(self):
        src = (ROOT / "core/opencl_microbrain_accelerator.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        offenders = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                launches_kernel = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Subscript)
                    and isinstance(n.func.value, ast.Attribute)
                    and n.func.value.attr == "_kernels"
                    for n in ast.walk(fn)
                )
                if launches_kernel:
                    gates = any(
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "should_use_gpu"
                        for n in ast.walk(fn)
                    )
                    if not gates:
                        offenders.append(fn.name)
        assert offenders == [], (
            f"kernel methods not gated by should_use_gpu (inline thresholds or "
            f"unconditional GPU paths): {offenders}"
        )

    def test_eggroll_es_no_longer_forces_gpu_by_default(self):
        import aetheris_microbrains.core.brain_nursery as nursery_mod

        acc = get_opencl_accelerator()
        assert inspect.signature(acc.eggroll_es_step).parameters["force_gpu"].default is False
        assert (
            inspect.signature(acc.eggroll_evolve_population).parameters["force_gpu"].default
            is False
        )
        assert (
            inspect.signature(nursery_mod.AutonomousBrainNursery.evolve_population_eggroll_es)
            .parameters["force_gpu"]
            .default
            is False
        )
        assert (
            inspect.signature(nursery_mod.AutonomousBrainNursery.evolve_population_eggroll)
            .parameters["force_gpu"]
            .default
            is False
        )


class TestAutotuneHonesty:
    def test_unknown_op_raises_instead_of_wrong_kernel_benchmark(self):
        acc = get_opencl_accelerator()
        with pytest.raises(ValueError):
            acc.autotune_crossover("kuramoto_hopfield_step", candidate_sizes=[64], persist=False)
        with pytest.raises(ValueError):
            acc.autotune_crossover("totally_unknown_op", candidate_sizes=[64], persist=False)

    def test_gpu_never_winning_yields_no_crossover_not_an_invention(self):
        rows_losing = [
            {"size": 64, "gpu_ms": 2.5, "cpu_ms": 1.6},
            {"size": 512, "gpu_ms": 20.0, "cpu_ms": 9.0},
        ]
        assert OpenCLMicroBrainAccelerator._choose_crossover(rows_losing) is None
        rows_winning = [
            {"size": 64, "gpu_ms": 1.0, "cpu_ms": 2.0},
            {"size": 512, "gpu_ms": 3.0, "cpu_ms": 9.0},
        ]
        assert OpenCLMicroBrainAccelerator._choose_crossover(rows_winning) == 64

    def test_autotune_loss_keeps_cpu_dispatch_and_reports_honestly(self):
        acc = get_opencl_accelerator()
        if not acc.gpu_available:
            pytest.skip("no GPU on this box")
        orig = acc.get_crossover("bitnet_batch_gemm")
        try:
            receipt = acc.autotune_crossover(
                "bitnet_batch_gemm", candidate_sizes=[64, 256], persist=False
            )
            if receipt["gpu_won_in_range"] is False:
                # GT 710 measured 8.4x slower at B=32 (2026-11-12); the honest
                # outcome is CPU dispatch, never max(candidate_sizes)*16 = 4096.
                assert receipt["crossover"] == acc.GPU_UNMEASURED
                assert acc.get_crossover("bitnet_batch_gemm") == acc.GPU_UNMEASURED
            else:
                assert receipt["crossover"] in (64, 256)
        finally:
            acc._crossover_table["bitnet_batch_gemm"] = orig


class TestMoECortexHonesty:
    def _cortex(self, tmp_path, monkeypatch, tool_stems):
        from aetheris_microbrains.core import hierarchical_moe_cortex as hmod

        tools = tmp_path / "tools"
        tools.mkdir(exist_ok=True)
        for stem in tool_stems:
            (tools / f"{stem}.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(hmod, "DYNAMIC_TOOLS_DIR", tools)
        return hmod.HierarchicalMoECortex(brains_dir=str(tmp_path / "no_brains"))

    def test_empty_catalog_is_unrouted_not_routed_success(self, tmp_path, monkeypatch):
        from aetheris_microbrains.core import hierarchical_moe_cortex as hmod

        monkeypatch.setattr(hmod, "DYNAMIC_TOOLS_DIR", tmp_path / "no_tools_dir")
        cortex = hmod.HierarchicalMoECortex(brains_dir=str(tmp_path / "no_brains"))
        res = cortex.route_query("format as markdown table")
        assert res["status"] == "unrouted"
        assert res["reason"] == "catalog_empty"
        assert "target_tool" not in res

    def test_unrelated_query_is_unrouted_with_real_catalog(self, tmp_path, monkeypatch):
        cortex = self._cortex(tmp_path, monkeypatch, ["levenshtein_distance"])
        # Probe verified collision-free: its embedding buckets are disjoint from
        # levenshtein_distance's, so the honest matcher must refuse to route.
        res = cortex.route_query("mmmttv qqqzz")
        assert res["status"] == "unrouted"
        assert res["reason"] == "no_catalog_entry_matched"
        assert "target_tool" not in res

    def test_routes_to_real_dynamic_tool_with_measured_latency(self, tmp_path, monkeypatch):
        cortex = self._cortex(tmp_path, monkeypatch, ["levenshtein_distance"])
        res = cortex.route_query("compute the levenshtein distance between two strings")
        assert res["target_tool"] == "levenshtein_distance"
        assert res["target_source"] == "dynamic_tool"
        assert res["search_complexity"] == "linear_scan"
        assert isinstance(res["routing_latency_ms"], float) and res["routing_latency_ms"] >= 0.0

    def test_route_without_microbrain_weights_fails_honestly(self, tmp_path, monkeypatch):
        # resolved_brain_file=None used to keep status="routed_success" with a
        # target_microbrain nobody could load; the route now fails honestly.
        cortex = self._cortex(tmp_path, monkeypatch, ["levenshtein_distance"])
        res = cortex.route_query("compute the levenshtein distance between two strings")
        assert res["status"] == "unrouted"
        assert res["reason"] == "no_microbrain_weights_for_tool"
        # Candidate fields stay for transparency.
        assert res["target_tool"] == "levenshtein_distance"
        assert res["resolved_brain_file"] is None

    def test_route_with_real_brain_weights_succeeds(self, tmp_path, monkeypatch):
        from aetheris_microbrains.core import hierarchical_moe_cortex as hmod

        tools = tmp_path / "tools"
        tools.mkdir(exist_ok=True)
        (tools / "levenshtein_distance.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(hmod, "DYNAMIC_TOOLS_DIR", tools)
        brains = tmp_path / "brains"
        brains.mkdir()
        (brains / "neuro_levenshtein_distance_meta.json").write_text(
            '{"name": "neuro_levenshtein_distance"}\n', encoding="utf-8"
        )
        cortex = hmod.HierarchicalMoECortex(brains_dir=str(brains))
        res = cortex.route_query("compute the levenshtein distance between two strings")
        assert res["status"] == "routed_success"
        assert res["resolved_brain_file"] is not None
        assert "reason" not in res

    def test_no_phantom_tools_or_literal_latency(self):
        src = (ROOT / "core/hierarchical_moe_cortex.py").read_text(encoding="utf-8")
        for phantom in (
            "analyze_stars",
            "classical_albedo_estimator",
            '"routing_latency_ms": 0.003',
            "O(log_N)_hierarchical",
        ):
            assert phantom not in src, f"hierarchical_moe_cortex still hardcodes {phantom}"


class TestVisionHonesty:
    def test_describe_image_discloses_no_vision_model(self):
        import asyncio

        try:
            from aetheris_microbrains.core.vision import describe_image
        except ImportError:
            from aetheris_microbrains.tools.microbrain import microbrain_consult

            res = microbrain_consult("describe PNG 16x32px pixel bitmap image channels")
            assert res["success"] is True
            assert res["abstained"] is True
            assert res["reason_code"] == "ood_coverage_breach"
            return

        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x10"  # width = 16
            b"\x00\x00\x00\x20"  # height = 32
            b"\x08\x02\x00\x00\x00"
        )
        desc = asyncio.run(describe_image(png_bytes))
        assert "PNG" in desc
        assert "16x32px" in desc
        assert "vision processing complete" not in desc
        assert "no vision model" in desc.lower()
