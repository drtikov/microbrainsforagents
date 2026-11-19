"""LOW-round honesty tripwires (microbrain audit follow-up).

The MEDIUM round (tests/test_microbrain_medium_honesty.py) pinned behavioral
fixes; this suite pins the LOW documentation/telemetry/bookkeeping fixes.
Source tripwires are the house pattern here: they go red if the folklore
docstrings, constant-only telemetry, or tautological test asserts ever
reappear.  Docstring extractions use ast, not raw greps, so only real
docstrings (not matching source substrings elsewhere) satisfy them.
"""

import ast
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]


def _source(rel: str) -> str:
    target = REPO / rel
    if not target.exists():
        exp_target = REPO / "experimental" / Path(rel).name
        if exp_target.exists():
            target = exp_target
        else:
            pytest.skip(f"File not found in standalone SDK: {rel}")
    return target.read_text(encoding="utf-8")


def _module_docstring(rel: str) -> str:
    return ast.get_docstring(ast.parse(_source(rel))) or ""


def _func_docstring(rel: str, func_name: str) -> str:
    tree = ast.parse(_source(rel))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.get_docstring(node) or ""
    return ""


class TestDocstringFolkloreRemoved:
    def test_dendritic_no_parameter_reduction_claim(self):
        doc = _module_docstring("core/dendritic_tree_microbrain.py")
        assert "17x parameter reduction" not in doc
        assert "0.005 ms" not in doc
        # The honest account points at the module's own measured ratio field.
        assert "parameter_ratio_vs_dense_mlp" in doc

    def test_wsp_oracle_no_latency_claim(self):
        doc = _module_docstring("evolution/wsp_epistemic_oracle.py")
        assert "<0.02ms" not in doc
        assert "<0.02" not in doc

    def test_dirichlet_docstring_discloses_approximations(self):
        doc = _func_docstring(
            "core/opencl_microbrain_accelerator.py", "_dirichlet_expected_kl_gain"
        )
        # The +1/n term must be explained as the digamma recurrence...
        assert "psi(x+1) = psi(x) + 1/x" in doc
        # ...and the two real approximations must be disclosed.
        assert "asymptotic" in doc
        assert "clamped" in doc
        assert "Exact" not in doc.split("The closed form")[0]


class TestSpiralTelemetryDisclosed:
    def test_constant_actions_carry_basis_fields(self):
        src = _source("core/concentric_spiral_dag.py")
        assert '"fast_path_basis"' in src
        assert '"cbf_basis"' in src
        assert '"dream_simulation_basis"' in src

    def test_hydro_self_injection_disclosed(self):
        src = _source("core/concentric_spiral_dag.py")
        assert '"fluid_self_injected"' in src
        assert '"fluid_injection_site"' in src
        assert "fluid_basis" in src


class TestDreamerBookkeepingDisclosed:
    def test_salience_marked_as_placeholder(self):
        src = _source("core/auto_dreamer_consolidator.py")
        assert "decorative placeholder" in src or "Editorial placeholder" in src
        assert "no consumer ranks by it" in src

    def test_research_harvester_no_longer_claims_memory_notes(self):
        target = REPO / "evolution/research_harvester.py"
        if target.exists():
            doc = _func_docstring("evolution/research_harvester.py", "derive_dynamic_research_need")
            assert "memory_notes.jsonl is NOT consulted" in doc
        else:
            doc = _module_docstring("evolution/wsp_epistemic_oracle.py")
            assert "per-call latency is not claimed here" in doc
            assert "no GPU path" in doc

    def test_autotelic_no_claims_unwired_consumers(self):
        doc = _module_docstring("core/autotelic_void_explorer.py")
        assert "do NOT read these results" in doc


class TestAcceleratorTestContractStaysDiscriminating:
    """Pin the test-file fixes themselves against regression."""

    def test_no_vacuous_device_assert(self):
        src = _source("tests/test_opencl_microbrain_accelerator.py")
        assert 'or len(evo_res["device"]) > 0' not in src
        assert 'evo_res["device"] == accel.device_name' in src

    def test_no_pcie_tautology_asserts(self):
        src = _source("tests/test_opencl_microbrain_accelerator.py")
        assert 'assert search_res["pcie_codebook_bytes_uploaded"] == 0' not in src
        assert 'assert es_res["pcie_perturbation_bytes_uploaded"] == 0' not in src

    def test_provider_helper_exercised(self):
        src = _source("tests/test_opencl_microbrain_accelerator.py")
        assert "get_best_execution_provider(batch_size=" in src
