"""Discriminating tests for the microbrain MEDIUM-honesty round (2026-11-12).

Contracts encoded here (all red against the pre-fix bodies):
- the surrogate prefilter harvests ISO-timestamp journal records (the old
  float() cast dropped every real record) and trains only on genuinely
  labeled outcomes — anchors and the tool catalogue carry failed=None;
- bitnet reports a MEASURED effective_bits_per_param from the artifact and
  refuses to claim quantized_success over a silently failed save;
- the analytical micro-transformer persists Hopfield rules across save/load,
  discloses conformal calibration provenance, and gives empty inputs zero
  confidence instead of a fabricated perfect score;
- eggroll ES returns per-candidate MSEs (the real fitness channel) and the
  nursery discloses identity-matrix substitution instead of silently
  evolving a substitute brain;
- the KAN formulas are labeled untrained, the MoD router gate is labeled
  untrained with clock-measured latency;
- the hardware report never claims a CUDA execution provider on driver
  presence alone;
- the topological merge gate is falsifiable (relative tolerance) and the
  persistence energy is not np.trace(dist) == 0;
- pipeline-fixed surfaces (incubator/neuro_route success semantics, ollama
  reachability probe, autotelic .npz glob) keep their honest bodies.
"""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class TestSurrogatePrefilterHonesty:
    def _memory(self, tmp_path, journal_events):
        from core.evolution_surrogate_prefilter import ResidentHDCExperienceMemory

        journal = tmp_path / "journal.jsonl"
        journal.write_text(
            "\n".join(json.dumps(e) for e in journal_events) + "\n", encoding="utf-8"
        )
        tools = tmp_path / "tools"
        tools.mkdir()
        (tools / "sample_tool.py").write_text("x = 1\n", encoding="utf-8")
        return ResidentHDCExperienceMemory(
            journal_path=journal,
            death_db_path=tmp_path / "no_death_db.json",
            tools_dir=tools,
        )

    def test_iso_timestamp_journal_records_are_harvested(self, tmp_path):
        events = [
            {"type": "evolution_success", "data": {"goal": "fix the astropy unit parser regression"},
             "timestamp": "2026-11-01T05:15:08.991079+00:00"},
            {"type": "evolution_failed", "data": {"goal": "refactor the dispatcher import cycle"},
             "timestamp": "2026-11-02T10:00:00.000000+00:00"},
        ]
        mem = self._memory(tmp_path, events)
        journal_recs = [r for r in mem.metadata if r.get("source") == "journal"]
        assert len(journal_recs) == 2, (
            "ISO-timestamp journal records were dropped by the harvest again"
        )
        # Parsed to real epoch seconds, not 0.0 placeholders
        recs_by_goal = {r["text"]: r for r in journal_recs}
        assert recs_by_goal["fix the astropy unit parser regression"]["ts"] > 1.5e9
        assert recs_by_goal["fix the astropy unit parser regression"]["failed"] is False
        assert recs_by_goal["refactor the dispatcher import cycle"]["failed"] is True

    def test_anchors_and_tool_catalogue_carry_no_fabricated_labels(self, tmp_path):
        from core.evolution_surrogate_prefilter import ResidentHDCExperienceMemory

        mem = self._memory(tmp_path, [])
        for r in mem.metadata:
            if r.get("source") in ("anchor", "tool_catalogue"):
                assert r.get("failed") is None, (
                    f"{r['source']} record carries a fabricated outcome label"
                )

    def test_training_uses_only_labeled_records(self, tmp_path):
        from core.evolution_surrogate_prefilter import (
            EvolutionSurrogatePrefilter,
            ResidentHDCExperienceMemory,
        )

        events = [
            {"type": "evolution_success", "data": {"goal": f"implement feature number {i} cleanly"},
             "timestamp": f"2026-11-{(i % 10) + 1:02d}T0{i % 9}:00:00"}
            for i in range(14)
        ]
        mem = self._memory(tmp_path, events)
        prefilter = EvolutionSurrogatePrefilter(experience_memory=mem)
        # 14 labeled journal records only — anchors and the tool catalogue
        # must not inflate the training corpus (old body counted them all).
        assert prefilter.train_samples + prefilter.test_samples == 14


class TestBitnetEngineHonesty:
    def test_effective_bits_measured_from_artifact(self, tmp_path):
        from core.bitnet_ternary_engine import BitNetQuantizer

        brain = tmp_path / "probe_meta.json"
        weights = [[0.9, -0.2, 0.1, -0.8], [0.3, 0.7, -0.9, 0.05]]
        brain.write_text(
            json.dumps({"name": "probe", "weights": {"W_mlp": weights}}), encoding="utf-8"
        )
        res = BitNetQuantizer.get_instance().quantize_micro_brain(brain)
        assert res["status"] == "quantized_success"
        # The paper label is disclosed as nominal, never as the measurement.
        assert res["format_nominal_bits_per_param"] == 1.58
        assert res["artifact_bytes"] == Path(res["bitnet_path"]).stat().st_size
        measured = round(res["artifact_bytes"] * 8.0 / res["total_parameters"], 2)
        assert res["effective_bits_per_param"] == measured
        # Text-JSON ternary storage is orders of magnitude above 1.58 bits.
        assert res["effective_bits_per_param"] > 1.58


class TestAnalyticalMicroTransformerHonesty:
    def _brain(self, name="m_probe"):
        from core.analytical_micro_transformer import AnalyticalMicroTransformer

        return AnalyticalMicroTransformer(
            name=name, d_model=16, n_heads=2, d_mlp=32, max_seq_len=8
        )

    def test_empty_input_confidence_is_zero_not_perfect(self):
        brain = self._brain()
        fwd = brain.forward([])
        assert fwd["confidence"] == 0.0
        assert "reason" in fwd
        kur = brain.forward_kuramoto_hopfield([])
        assert kur["confidence"] == 0.0
        assert kur["kuramoto_order_r"] == 0.0

    def test_hopfield_rules_survive_save_load(self, tmp_path):
        brain = self._brain("m_probe_persist")
        brain.compile_logic_association("quasar", "pulsar")
        brain.save(tmp_path)
        reloaded = type(brain).load("m_probe_persist", tmp_path)
        assert reloaded is not None
        assert len(reloaded.hopfield_keys) == 1, (
            "compiled Hopfield rules were lost across save/load again"
        )
        assert len(reloaded.hopfield_values) == 1

    def test_conformal_basis_disclosed(self, tmp_path):
        brain = self._brain("m_probe_conf")
        res = brain.infer_conformal(["token_without_calibration"])
        assert res["calibrated"] is False
        assert res["calibration_basis"] == "uncalibrated_default"
        brain.compile_logic_association("a_tok", "b_tok")
        cal = brain.calibrate_conformal([(["a_tok"], "b_tok")], in_sample=True)
        assert cal["basis"] == "in_sample_auto"
        assert brain.conformal_basis == "in_sample_auto"


class TestEggrollPerCandidateFitness:
    def test_es_step_returns_per_candidate_mses(self):
        from core.opencl_microbrain_accelerator import get_opencl_accelerator

        acc = get_opencl_accelerator()
        rng = np.random.RandomState(7)
        x = rng.randn(4, 8).astype(np.float32)
        y = rng.randn(4, 8).astype(np.float32)
        res = acc.eggroll_es_step(
            w_base=rng.randn(8, 8).astype(np.float32),
            x_samples=x, y_targets=y, pop_size=4, force_gpu=False,
        )
        mses = res.get("candidate_mses")
        assert mses is not None, "eggroll_es_step stopped returning per-candidate MSEs"
        assert len(mses) == res["population_size"]
        # Real per-candidate values vary; the old DNS fitness was a constant.
        assert float(np.std(np.asarray(mses, dtype=np.float32))) > 0.0

    def test_es_path_discloses_identity_substitution_and_skips_writeback(
        self, tmp_path, monkeypatch
    ):
        import core.brain_nursery as nursery_mod
        from core.brain_nursery import AutonomousBrainNursery

        monkeypatch.setattr(nursery_mod, "STORE_DIR", tmp_path)
        # monkeypatch restores the previous singleton at teardown — a bare
        # assignment here leaked a tmp-path nursery into every later suite.
        monkeypatch.setattr(AutonomousBrainNursery, "_instance", None)
        nursery = AutonomousBrainNursery.get_instance()
        nursery.grow_brain_for_domain("identity probe domain")
        brain = nursery.active_brains["neuro_identity_probe_domain"]
        brain.W_mlp = None  # force the identity-matrix fallback path
        rng = np.random.RandomState(11)
        res = nursery.evolve_population_eggroll_es(
            domain_name="identity probe domain",
            x_samples=rng.randn(4, 8).astype(np.float32),
            y_targets=rng.randn(4, 8).astype(np.float32),
            pop_size=4, force_gpu=False,
        )
        assert res.get("weights_substituted") is True, (
            "the ES path evolved the identity substitute without disclosing it"
        )
        assert brain.W_mlp is None, "substitute weights were written back into the brain"


class TestKanAndModHonesty:
    def test_kan_formulas_labeled_untrained(self):
        from core.kolmogorov_arnold_microbrain import KolmogorovArnoldMicroBrain

        res = KolmogorovArnoldMicroBrain.get_instance().extract_symbolic_formula()
        assert res["status"] == "extracted_untrained_basis"
        assert res["trained"] is False

    def test_mod_gate_discloses_untrained_status_and_measures_latency(self):
        from core.mixture_of_depths_router import MixtureOfDepthsRouter

        res = MixtureOfDepthsRouter.get_instance().forward([0.5, -1.0, 2.0])
        assert res["gate_status"] == "untrained_heuristic_gate"
        assert isinstance(res["latency_ms"], (int, float))
        assert "estimated_latency_ms" not in res


class TestHardwareRouterHonesty:
    def test_driver_presence_never_claims_cuda_provider(self, monkeypatch):
        import core.hardware_router as hr

        class _FakeRes:
            returncode = 0
            stdout = "NVIDIA GeForce GT 710, 2048 MiB\n"
            stderr = ""

        monkeypatch.setattr(hr.subprocess, "run", lambda *a, **k: _FakeRes())

        class _FakeAcc:
            gpu_available = False

        import core.opencl_microbrain_accelerator as acc_mod
        monkeypatch.setattr(acc_mod, "get_opencl_accelerator", lambda: _FakeAcc())

        report = hr.get_hardware_report()
        assert report["gpu_detected"] is True
        assert report.get("cuda_driver_present") is True
        assert report["best_provider"] != "CUDA", (
            "best_provider claimed CUDA from driver presence alone again"
        )


class TestTopologicalMergerGate:
    def test_persistence_energy_is_not_trace_zero(self):
        from core.topological_persistence_merger import TopologicalPersistenceMerger

        rng = np.random.RandomState(3)
        points = rng.randn(6, 4).astype(np.float32)
        topo = TopologicalPersistenceMerger.get_instance()._compute_rips_persistence_betti1(points)
        assert topo["persistence_energy"] > 0.0, (
            "persistence_energy collapsed to np.trace(dist) == 0 again"
        )

    def test_merge_gate_is_falsifiable_relative_tolerance(self):
        src = (ROOT / "core/topological_persistence_merger.py").read_text(encoding="utf-8")
        assert "avg_parent_betti1 + 2.0" not in src, (
            "the unfalsifiable +2.0 absolute gate tolerance is back"
        )
        assert "homology_persistence_tol * max(1.0, avg_parent_betti1)" in src


class TestPipelineFixedSurfacesStayHonest:
    """Source tripwires for the concurrency-fixed surfaces (red vs their old bodies)."""

    def test_incubator_success_semantics(self):
        target = ROOT / "tools/neuro_brain_incubator.py"
        if not target.exists():
            target = ROOT / "execution/tools/dynamic/neuro_brain_incubator.py"
        src = target.read_text(encoding="utf-8")
        assert 'res["success"] = not res.get("abstained", False)' in src
        assert 'result["success"] = result.get("status") == "updated"' in src

    def test_neuro_route_connectome_uses_real_route_success(self):
        target = ROOT / "tools/neuro_route.py"
        if not target.exists():
            target = ROOT / "execution/tools/dynamic/neuro_route.py"
        src = target.read_text(encoding="utf-8")
        assert "is_route_success" in src

    def test_harness_ollama_is_probed_not_hardcoded(self):
        target = ROOT / "api/routers/harness.py"
        if target.exists():
            src = target.read_text(encoding="utf-8")
            assert "ollama_reachable" in src
            assert '"is_configured": True,\n        "masked_key": ollama_url' not in src
        else:
            from tools.microbrain import microbrain_status

            src = (ROOT / "tools/microbrain.py").read_text(encoding="utf-8")
            assert 'BRAINS_DIR.glob("*_meta.json")' in src
            status = microbrain_status()
            assert status["success"] is True
            expected_count = len(list((ROOT / "memory/store/neuro_brains").glob("*_meta.json")))
            assert expected_count >= 40
            assert len(status["brains"]) == status["count"] == expected_count

    def test_autotelic_scans_npz_not_only_npy(self):
        src = (ROOT / "core/autotelic_void_explorer.py").read_text(encoding="utf-8")
        assert 'glob("*.npz")' in src, (
            "the explorer is blind to the .npz brain store again (counter pinned at 0)"
        )
        # The SHA-of-filename pseudo-embedding fallback must stay gone.
        assert "sha256(p.stem" not in src
