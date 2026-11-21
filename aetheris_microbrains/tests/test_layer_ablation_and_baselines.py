"""Layer Ablation, Baseline Comparison, Allowlist Shell Gate, MCP-Zero,
Mondrian Conformal, Checksum Integrity, and Staged Plasticity Test Suite.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aetheris_microbrains as mb
from aetheris_microbrains.core.analytical_micro_transformer import AnalyticalMicroTransformer
from aetheris_microbrains.core.layered_decision_engine import (
    analyze_code_diff,
    analyze_shell_command,
    evaluate_layer0_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parents[1]


class TestLayerAblationStudy:
    """Ablate Layer 0 -> Layer 0+1 -> Layer 0+1+2 on obfuscated sinks, allowlist shell & hard benign inputs."""

    OBFUSCATED_CODE_CASES = [
        ("src/sub.py", "__builtins__['eval']('1 + 1')\n"),
        ("src/glob.py", "fn = globals()['eval']\nfn(user_input)\n"),
        ("src/b64.py", "import base64\npayload = base64.b64decode(b'cHJpbnQoMSk=')\nexec(payload)\n"),
        ("src/dynimp.py", "import importlib\nmod = importlib.import_module('os')\n"),
    ]

    DANGEROUS_SHELL_CASES = [
        "rm -f -r ~",
        "sh -c 'rm -rf /'",
        "python -c \"import os; os.system('rm -rf /')\"",
        "ls $(whoami)",
        "echo pwn > /etc/passwd",
    ]

    ALLOWLISTED_SHELL_CASES = [
        "ls -la",
        "pwd",
        "echo 'hello > quoted_is_safe'",
        "git log --format '%h'",
        "clang-format -i main.c",
        "pytest -q",
    ]

    UNRECOGNIZED_SHELL_CASES = [
        "custom_deploy_binary --prod",
        "find . -exec rm -rf {} +",
        "git push --force origin main",
    ]

    def test_ablation_layer0_vs_layer01_vs_full(self):
        # 1. Layer 0 catches constitutional boundaries, misses code/shell sinks
        l0_caught_protected = evaluate_layer0_policy("write_file", "memory/bible.md")["decision"] == "deny"
        l0_misses_obfuscated = all(
            evaluate_layer0_policy("edit_file", path)["decision"] == "allow"
            for path, _ in self.OBFUSCATED_CODE_CASES
        )
        assert l0_caught_protected is True
        assert l0_misses_obfuscated is True

        # 2. Layer 0 + Layer 1 catches all obfuscated code sinks & dangerous shell commands
        for path, code in self.OBFUSCATED_CODE_CASES:
            res = mb.decide({"tool": "edit_file", "path": path, "diff": code})
            assert res["decision"] == "deny", f"Obfuscated diff escaped Layer 1: {path} {code}"
            assert res["layer"] == 1

        for cmd in self.DANGEROUS_SHELL_CASES:
            res = mb.decide({"tool": "execute_shell", "cmd": cmd})
            assert res["decision"] == "deny", f"Dangerous shell escaped Layer 1: {cmd}"
            assert res["layer"] == 1

        # 3. Allowlist-first shell behavior: allowlisted pass, unrecognized escalate (never blind-allow)
        for cmd in self.ALLOWLISTED_SHELL_CASES:
            res = mb.decide({"tool": "execute_shell", "cmd": cmd})
            assert res["decision"] == "allow", f"Allowlisted shell command blocked/escalated: {cmd} -> {res}"

        for cmd in self.UNRECOGNIZED_SHELL_CASES:
            res = mb.decide({"tool": "execute_shell", "cmd": cmd})
            assert res["decision"] in ("escalate", "deny"), f"Unrecognized shell auto-allowed: {cmd} -> {res}"
            if cmd not in ("git push --force origin main", "find . -exec rm -rf {} +"):
                assert res["decision"] == "escalate"
                assert res["reason_code"] == "layer1_unrecognized_shell_command_escalate"

        # Verify Rule-of-3 statistical bound calculation matches n evaluated samples
        n_danger = len(self.OBFUSCATED_CODE_CASES) + len(self.DANGEROUS_SHELL_CASES)
        rule_of_3_upper_bound = 3.0 / float(n_danger)
        assert rule_of_3_upper_bound == pytest.approx(3.0 / 9.0, rel=1e-4)

    def test_cross_line_original_source_alias_context(self):
        original_file = "import os as _os_alias\n\ndef safe_fn():\n    return 1\n"
        hunk_only = "def trigger(cmd):\n    _os_alias.system(cmd)\n"
        # Even with aliased import residing in untouched top of original_source, Layer 1 resolves `_os_alias -> os`
        analysis = analyze_code_diff("src/runner.py", hunk_only, original_source=original_file)
        assert analysis["context_ast_parsed"] is True
        assert analysis["security_clean"] is False
        assert any("os.system" in f for f in analysis["findings"])

        verdict = mb.decide(
            {"tool": "edit_file", "path": "src/runner.py", "diff": hunk_only},
            context={"file_content": original_file},
        )
        assert verdict["decision"] == "deny"


class TestBaselineComparisonBenchmark:
    """Compare Naive EN Keyword Baseline vs. Pure-NumPy Trigram Linear vs. Hybrid Bilingual MoE."""

    BILINGUAL_CASES = [
        ("inspect file contents and write safe patch to repository", {"repo_write", "edit_file", "write_file"}),
        ("show the current git branch and status", {"git_status"}),
        ("run pytest unit tests for validation", {"run_tests", "pytest_runner"}),
        ("прочитай файл и напиши патч для репозитория", {"repo_write", "edit_file", "write_file"}),
        ("покажи текущую ветку git и статус", {"git_status"}),
        ("запусти тесты и проверь синтаксис", {"run_tests", "pytest_runner", "syntax_validator"}),
        ("найди в памяти прошлые знания", {"memory_recall", "knowledge_search"}),
    ]

    OOD_CASES = [
        "summarize what we discussed yesterday",
        "закажи такси до дома",
        "какая завтра погода в берлине",
    ]

    @staticmethod
    def _naive_english_keyword_baseline(query: str) -> str:
        q = query.lower()
        if "write" in q or "patch" in q or "edit" in q:
            return "repo_write"
        if "git" in q or "branch" in q or "status" in q:
            return "git_status"
        if "test" in q or "pytest" in q:
            return "run_tests"
        if "memory" in q or "recall" in q:
            return "memory_recall"
        return "repo_read"

    def test_hybrid_moe_outperforms_english_keyword_baseline_on_bilingual_traces(self):
        router = mb.get_tool_routing_moe()

        en_only_hits = 0
        hybrid_hits = 0
        for query, expected_set in self.BILINGUAL_CASES:
            if self._naive_english_keyword_baseline(query) in expected_set:
                en_only_hits += 1
            res = router.route(query)
            if res.conformal_passed and res.selected_tool in expected_set:
                hybrid_hits += 1

        en_acc = en_only_hits / len(self.BILINGUAL_CASES)
        hybrid_acc = hybrid_hits / len(self.BILINGUAL_CASES)

        # English baseline misses Russian prompts without English anchors; hybrid MoE handles both
        assert hybrid_acc > en_acc
        assert hybrid_acc >= 0.85

        # Genuine OOD prompts still abstain deterministically
        for ood_q in self.OOD_CASES:
            ood_res = router.route(ood_q)
            assert ood_res.conformal_passed is False

    def test_bilingual_determinism_across_hash_seeds(self):
        queries = [q for q, _ in self.BILINGUAL_CASES[:4]]
        script = (
            "import json, sys; sys.path.insert(0, %r); "
            "import aetheris_microbrains as mb; "
            "r = mb.get_tool_routing_moe(); "
            "qs = %r; "
            "print(json.dumps([(q, r.route(q).to_dict()) for q in qs], sort_keys=True))"
        ) % (str(REPO_ROOT), queries)

        outs = []
        for seed in ("1", "77", "4242"):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=dict(os.environ, PYTHONHASHSEED=seed),
                check=True,
            )
            outs.append(proc.stdout.strip())
        assert outs[0] == outs[1] == outs[2]


class TestDynamicToolRegistryAndMCPZero:
    def test_available_tools_constrains_output_strictly(self):
        router = mb.get_tool_routing_moe()
        restricted = ["edit_file", "memory_recall", "web_fetch"]
        res = router.route("inspect file contents and write safe patch to repository", available_tools=restricted)
        assert res.selected_tool in restricted
        assert res.secondary_tool in restricted
        assert res.selected_tool == "edit_file"

        # Custom agent-specific tool catalog never leaks unmounted tools
        custom_only = ["custom_git_save", "custom_memory_fetch"]
        custom_res = router.route("git commit changes to repository", available_tools=custom_only)
        assert custom_res.selected_tool == "custom_git_save"

        verdict = mb.decide({"query": "git commit branch"}, context={"available_tools": custom_only})
        assert verdict["label"] in custom_only

    def test_mcp_zero_narrowing_and_ood_full_fallback(self):
        narrow = mb.narrow_tool_surface("fix bug and write repository patch", top_k=4)
        assert narrow["mode"] == "narrowed"
        assert narrow["conformal_passed"] is True
        assert 1 <= len(narrow["candidate_tools"]) <= 4
        assert narrow["selected_tool"] in narrow["candidate_tools"]
        assert narrow["reduction_ratio"] >= 0.85

        ood = mb.narrow_tool_surface("закажи пиццу домой", top_k=4)
        assert ood["mode"] == "escalate_full_catalog"
        assert ood["conformal_passed"] is False
        assert len(ood["candidate_tools"]) == 32
        assert ood["reduction_ratio"] == 0.0


class TestFailClosedShadowAndJournal:
    def test_fail_closed_by_risk_class(self):
        ext_err = mb.decide(
            {"tool": "execute_shell", "cmd": "ls -la"},
            context={"_test_fail_layer1": True},
        )
        assert ext_err["decision"] == "deny"
        assert ext_err["reason_code"].startswith("layer1_fail_closed")

        ro_err = mb.decide(
            {"tool": "read_file", "path": "README.md"},
            context={"_test_fail_layer1": True},
        )
        assert ro_err["decision"] == "escalate"
        assert ro_err["reason_code"].startswith("layer1_escalate_on_error")

    def test_shadow_mode_and_structured_journal(self, tmp_path: Path):
        journal = tmp_path / "decision_journal.jsonl"
        verdict = mb.decide(
            {"tool": "edit_file", "path": "src/a.py", "diff": "os.system('whoami')\n"},
            context={"shadow_mode": True, "journal_path": str(journal)},
        )
        assert verdict["shadow_mode"] is True
        assert verdict["shadow_decision"] == "deny"
        assert verdict["decision"] == "escalate"
        assert verdict["reason_code"] == "shadow_mode_observation"
        assert len(verdict["input_sha256"]) == 16

        assert journal.exists()
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["input_sha256"] == verdict["input_sha256"]
        assert rows[0]["shadow_decision"] == "deny"


class TestMondrianChecksumsAndStagedContinualLearning:
    def test_mondrian_per_class_quantiles_and_checksum_tamper_rejection(self, tmp_path: Path):
        brains_dir = PKG_DIR / "memory" / "store" / "neuro_brains"
        brain_orig = AnalyticalMicroTransformer.load("neuro_code_edit_verifier", brains_dir)
        assert brain_orig is not None
        assert brain_orig.weights_sha256 is not None
        assert len(brain_orig.per_class_quantiles) >= 2
        assert brain_orig.coverage_std_error > 0.0

        inf = brain_orig.infer_conformal(["syntax_tree_valid", "ast_node_function_def"], alpha=0.05, mondrian=True)
        assert "per_class_quantiles" in inf and len(inf["per_class_quantiles"]) >= 2
        assert inf["coverage_std_error"] > 0.0
        assert inf["weights_sha256"] == brain_orig.weights_sha256

        # Save clean copy in tmp_path, verify load succeeds, then corrupt weights by 1 byte
        brain_orig.save(directory=tmp_path)
        assert AnalyticalMicroTransformer.load("neuro_code_edit_verifier", tmp_path, verify_checksum=True) is not None

        wfile = tmp_path / "neuro_code_edit_verifier_weights.npz"
        raw = bytearray(wfile.read_bytes())
        raw[-1] ^= 0xFF
        wfile.write_bytes(bytes(raw))

        tampered = AnalyticalMicroTransformer.load("neuro_code_edit_verifier", tmp_path, verify_checksum=True)
        assert tampered is None, "Tampered weights NPZ must be rejected when SHA-256 verification is enabled!"

    def test_staged_candidate_verification_and_rollback(self, tmp_path: Path):
        brains_dir = PKG_DIR / "memory" / "store" / "neuro_brains"
        for suffix in ("_weights.npz", "_meta.json"):
            shutil.copy2(brains_dir / f"neuro_code_edit_verifier{suffix}", tmp_path / f"neuro_code_edit_verifier{suffix}")

        orig_meta = json.loads((tmp_path / "neuro_code_edit_verifier_meta.json").read_text(encoding="utf-8"))
        orig_sha = orig_meta["weights_sha256"]

        pipeline = mb.ContinualLearningPipeline(store_dir=tmp_path, audit_log_path=tmp_path / "audit.json")
        res = pipeline.apply_plasticity_feedback("code_edit_verifier", ["new_ast_token_alpha"], promote=False)
        assert res.status == "success"
        assert res.candidate_staged is True
        assert res.promoted is False

        # Production weights in tmp_path remain completely untouched!
        after_stage_meta = json.loads((tmp_path / "neuro_code_edit_verifier_meta.json").read_text(encoding="utf-8"))
        assert after_stage_meta["weights_sha256"] == orig_sha
        assert (tmp_path / "_candidates" / "neuro_code_edit_verifier_weights.npz").exists()

        # Bad holdout fails gate
        bad_holdout = [(["unrelated_missing_tok"], "nonexistent_target")] * 5
        rej = pipeline.verify_and_promote_candidate("code_edit_verifier", bad_holdout, min_coverage=0.95)
        assert rej.promoted is False
        assert rej.status == "rejected_verification"

        # Valid holdout built from existing compiled Hopfield knowledge passes gate & promotes
        cand = AnalyticalMicroTransformer.load("neuro_code_edit_verifier", tmp_path / "_candidates")
        assert cand is not None and len(cand.hopfield_keys) >= 20
        good_holdout = []
        vocab_slice = cand.W_embed[: len(cand.vocab)]
        for k_vec, v_vec in zip(cand.hopfield_keys[:22], cand.hopfield_values[:22]):
            k_tok = cand.inv_vocab[int(np.argmax(np.dot(vocab_slice, k_vec)))]
            v_tok = cand.inv_vocab[int(np.argmax(np.dot(vocab_slice, v_vec)))]
            good_holdout.append(([k_tok], v_tok))

        prom = pipeline.verify_and_promote_candidate("code_edit_verifier", good_holdout, min_coverage=0.95)
        assert prom.promoted is True
        assert prom.status == "promoted"
        assert (tmp_path / "_rollback" / "neuro_code_edit_verifier_weights.npz").exists()

        # Rollback restores exact original SHA-256
        assert pipeline.rollback_domain("code_edit_verifier") is True
        restored = AnalyticalMicroTransformer.load("neuro_code_edit_verifier", tmp_path, verify_checksum=True)
        assert restored is not None
        assert restored.weights_sha256 == orig_sha
