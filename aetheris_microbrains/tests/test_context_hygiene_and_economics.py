"""Comprehensive tests for context hygiene, token economics, handback envelopes,
chronicle rollup, policy role slicing, and tier ladder routing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import aetheris_microbrains as mb
from aetheris_microbrains import audit, doctor
from aetheris_microbrains.core.context_hygiene import (
    ContextHygieneGate,
    FileSnapshotLedger,
    HygieneConfig,
    ShellReadInspector,
    SymbolMapExtractor,
)
from aetheris_microbrains.core.economics_ledger import (
    DEFAULT_CHARS_PER_TOKEN,
    TokenSavingsLedger,
    estimate_tokens,
    format_statusline,
)
from aetheris_microbrains.core.envelope_parser import (
    parse_handback_envelope,
    validate_handback_envelope,
)
from aetheris_microbrains.core.session_rollup import (
    BootstrapAssembler,
    ChronicleCompactor,
)
from aetheris_microbrains.core.tier_ladder import TierLadderResolver, extract_depth_hint


class TestFileSnapshotLedger:
    def test_redundant_read_block_and_escape_valve(self, tmp_path: Path):
        target = tmp_path / "sample.py"
        target.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        ledger = FileSnapshotLedger(ledger_path=tmp_path / "snaps.json")

        # 1st whole read: allowed and recorded
        assert ledger.check_or_record(target, max_consecutive_blocks=3) is None

        # 2nd, 3rd, 4th consecutive unchanged reads: blocked (streak 1, 2, 3)
        for idx in range(1, 4):
            v = ledger.check_or_record(target, max_consecutive_blocks=3)
            assert v is not None
            assert v.allowed is False
            assert v.reason_code == "hygiene_redundant_whole_read"
            assert f"attempt {idx}/3" in v.reason
            assert v.avoided_bytes == target.stat().st_size

        # 5th attempt (exceeding max_consecutive_blocks=3): escape valve resets and allows
        assert ledger.check_or_record(target, max_consecutive_blocks=3) is None

    def test_modification_or_eviction_allows_reread(self, tmp_path: Path):
        target = tmp_path / "mod.py"
        target.write_text("x = 1\n", encoding="utf-8")
        ledger = FileSnapshotLedger(ledger_path=tmp_path / "snaps.json")

        assert ledger.check_or_record(target) is None
        blocked = ledger.check_or_record(target)
        assert blocked is not None and not blocked.allowed

        # Evict simulates PostToolUse edit invalidation
        assert ledger.evict_path(target) is True
        assert ledger.check_or_record(target) is None

        # Modifying size allows immediate re-read without eviction
        target.write_text("x = 1\ny = 2\n", encoding="utf-8")
        assert ledger.check_or_record(target) is None


class TestSymbolMapExtractor:
    def test_python_markdown_ts_outlines_and_row_cap(self, tmp_path: Path):
        py_file = tmp_path / "mod.py"
        lines = []
        for i in range(40):
            lines.append(f"def fn_{i}(a, b):\n    \"\"\"Doc for {i}.\"\"\"\n    return a + b\n")
        py_file.write_text("\n".join(lines), encoding="utf-8")

        rows = SymbolMapExtractor.extract_symbol_map(py_file, max_rows=10)
        assert len(rows) == 11  # 10 rows + truncation note
        assert "def fn_0(a, b)" in rows[0]
        assert "more symbol(s) omitted" in rows[-1]

        md_file = tmp_path / "doc.md"
        md_file.write_text("# Title\n```md\n# Inside fence\n```\n## Section A\n### Sub B\n", encoding="utf-8")
        md_rows = SymbolMapExtractor.extract_symbol_map(md_file)
        assert any("Title" in r for r in md_rows)
        assert any("Section A" in r for r in md_rows)
        assert not any("Inside fence" in r for r in md_rows)

        ts_file = tmp_path / "app.ts"
        ts_file.write_text("export class Engine {\n  run() {}\n}\nexport const handler = async (x) => x;\n", encoding="utf-8")
        ts_rows = SymbolMapExtractor.extract_symbol_map(ts_file)
        assert any("class Engine" in r for r in ts_rows)
        assert any("const handler" in r for r in ts_rows)

    def test_oversized_read_inlines_outline(self, tmp_path: Path):
        big = tmp_path / "big.py"
        big.write_text("class Core:\n    pass\n" + ("# padding\n" * 1000), encoding="utf-8")
        gate = ContextHygieneGate(HygieneConfig(whole_read_char_limit=500))
        v = gate.evaluate_pre_tool("Read", {"file_path": str(big)}, ledger_path=tmp_path / "snap.json")
        assert v.allowed is False
        assert v.reason_code == "hygiene_oversized_whole_read"
        assert "class Core" in v.reason
        assert v.avoided_bytes > 500


class TestShellReadInspector:
    def test_shell_reads_spill_stdin_and_root_search(self, tmp_path: Path):
        big = tmp_path / "large.log"
        big.write_text("line\n" * 2000, encoding="utf-8")
        cfg = HygieneConfig(whole_read_char_limit=1000)

        # Unbounded cat of oversized file -> blocked
        v_cat = ShellReadInspector.inspect_command(f"cat {big.name}", cfg, cwd=str(tmp_path))
        assert v_cat is not None and not v_cat.allowed
        assert v_cat.reason_code == "hygiene_shell_read_oversized"

        # Narrowed via grep pipeline -> allowed
        assert ShellReadInspector.inspect_command(f"cat {big.name} | grep error", cfg, cwd=str(tmp_path)) is None

        # Narrowed with head -n -> allowed
        assert ShellReadInspector.inspect_command(f"head -n 25 {big.name}", cfg, cwd=str(tmp_path)) is None

        # Spilled output whole read -> blocked
        v_spill = ShellReadInspector.inspect_command("cat /tmp/tool-results/toolu_abc123.txt", cfg)
        assert v_spill is not None and v_spill.reason_code == "hygiene_spilled_output_whole_read"

        # Bare python - without pipe/heredoc -> blocked
        v_stdin = ShellReadInspector.inspect_command("python -", cfg)
        assert v_stdin is not None and v_stdin.reason_code == "hygiene_stdin_hang_risk"

        # Piped into python - -> allowed
        assert ShellReadInspector.inspect_command("echo print(1) | python -", cfg) is None

        # Heredoc into python - -> allowed
        assert ShellReadInspector.inspect_command("python - << 'EOF'\nprint(1)\nEOF", cfg) is None

        # Root-origin find -> blocked
        v_root = ShellReadInspector.inspect_command("find / -name foo.py", cfg)
        assert v_root is not None and v_root.reason_code == "hygiene_root_origin_search"

        # Windows drive root recursive grep -> blocked
        v_win = ShellReadInspector.inspect_command("grep -rn secret C:\\", cfg)
        assert v_win is not None and v_win.reason_code == "hygiene_root_origin_search"


class TestHookIntegrationWithContextGuard:
    def test_hook_blocks_redundant_read_logs_savings_and_evicts_on_edit(self, tmp_path: Path):
        target = tmp_path / "module.py"
        target.write_text("VALUE = 42\n", encoding="utf-8")
        econ_path = tmp_path / "economics.jsonl"
        hyg_path = tmp_path / "hygiene.json"
        journal_path = tmp_path / "journal.jsonl"

        read_payload = {
            "session_id": "sess-test-1",
            "tool_name": "Read",
            "tool_input": {"file_path": str(target)},
        }

        # Default (context_guard=False) keeps identical legacy allow behavior on repeated reads
        assert mb.evaluate_hook_payload(read_payload, log_decisions=False)["decision"] == "allow"
        assert mb.evaluate_hook_payload(read_payload, log_decisions=False)["decision"] == "allow"

        # With context_guard=True: 1st read allows, 2nd identical read denies + logs savings
        v1 = mb.evaluate_hook_payload(
            read_payload,
            context_guard=True,
            hygiene_ledger_path=str(hyg_path),
            economics_ledger_path=str(econ_path),
            journal_path=str(journal_path),
            log_decisions=True,
        )
        assert v1["decision"] == "allow"

        v2 = mb.evaluate_hook_payload(
            read_payload,
            context_guard=True,
            hygiene_ledger_path=str(hyg_path),
            economics_ledger_path=str(econ_path),
            journal_path=str(journal_path),
            log_decisions=True,
        )
        assert v2["decision"] == "deny"
        assert v2["reason_code"] == "hygiene_redundant_whole_read"
        assert v2["hygiene_blocked"] is True
        assert v2["avoided_tokens"] > 0

        # Economics ledger recorded the savings
        ledger = TokenSavingsLedger(econ_path)
        summary = ledger.summarize()
        assert summary["events"] == 1
        assert summary["tokens_preserved"] == v2["avoided_tokens"]
        assert "MB saved:" in format_statusline(summary)

        # Security deny MUST take precedence over context hygiene
        sec_deny = mb.evaluate_hook_payload(
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
            context_guard=True,
            hygiene_ledger_path=str(hyg_path),
            economics_ledger_path=str(econ_path),
            log_decisions=False,
        )
        assert sec_deny["decision"] == "deny"
        assert sec_deny.get("hygiene_blocked") is not True

        # PostToolUse Edit evicts snapshot so re-read succeeds cleanly
        mb.evaluate_post_tool_use_payload(
            {
                "session_id": "sess-test-1",
                "tool_name": "Edit",
                "tool_input": {"file_path": str(target), "old_string": "42", "new_string": "43"},
                "tool_response": "ok",
            },
            context_guard=True,
            hygiene_ledger_path=str(hyg_path),
        )
        v3 = mb.evaluate_hook_payload(
            read_payload,
            context_guard=True,
            hygiene_ledger_path=str(hyg_path),
            economics_ledger_path=str(econ_path),
            log_decisions=False,
        )
        assert v3["decision"] == "allow"


class TestAuditDigestAndSavingsCLI:
    def test_audit_digest_savings_statusline_and_explain(self, tmp_path: Path, capsys):
        econ = tmp_path / "econ.jsonl"
        TokenSavingsLedger(econ).record(
            "context_hygiene_block",
            2900,
            tool="Read",
            reason_code="hygiene_oversized_whole_read",
            target="src/big.py",
        )
        assert audit.main(["--savings", "--json", "--economics-path", str(econ)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["events"] == 1
        assert out["tokens_preserved"] == estimate_tokens(2900, DEFAULT_CHARS_PER_TOKEN)

        assert audit.main(["--statusline", "--economics-path", str(econ)]) == 0
        assert "MB saved:" in capsys.readouterr().out

        journal = tmp_path / "journal.jsonl"
        journal.write_text(
            json.dumps({
                "tool": "Read",
                "decision": "deny",
                "reason_code": "hygiene_oversized_whole_read",
                "label": "too big",
                "target": "src/big.py",
                "input_sha256": "abc123",
            }) + "\n",
            encoding="utf-8",
        )
        assert audit.main(["--digest", "--json", "--journal-path", str(journal)]) == 0
        dig = json.loads(capsys.readouterr().out)
        assert dig["total_entries"] == 1
        assert dig["friction_matrix"][0]["recommendation"] == "split_file"

        assert audit.main(["--explain", "--journal-path", str(journal)]) == 0
        exp = capsys.readouterr().out
        assert "structural symbol map" in exp


class TestEnvelopeParser:
    def test_valid_markdown_bold_wrapped_and_missing_verification(self):
        good = (
            "- **OUTCOME**: completed\n"
            "SUMMARY: Fixed race condition in queue lock\n"
            "  across Windows workers.\n"
            "ARTIFACT: aetheris_microbrains/core/context_hygiene.py\n"
            "VERIFIED_BY: pytest aetheris_microbrains/tests/ -q (all pass)\n"
        )
        parsed = parse_handback_envelope(good)
        assert parsed is not None
        assert parsed.outcome == "completed"
        assert "Windows workers" in parsed.summary
        res = validate_handback_envelope(good, require_verification=True)
        assert res["valid"] is True

        vague = "OUTCOME: completed\nSUMMARY: All done!\n"
        res_vague = validate_handback_envelope(vague, require_verification=True)
        assert res_vague["valid"] is False
        assert "VERIFIED_BY" in res_vague["missing_fields"]


class TestSessionRollupAndBootstrap:
    def test_chronicle_write_compact_idempotent_and_bootstrap_budget(self, tmp_path: Path):
        old_ts = time.mktime(time.strptime("2026-11-20 12:00:00", "%Y-%m-%d %H:%M:%S"))
        today_ts = time.mktime(time.strptime("2026-11-22 10:00:00", "%Y-%m-%d %H:%M:%S"))

        p_old = ChronicleCompactor.write_entry(tmp_path, "coder", "progress", "Implemented module A", now=old_ts)
        os.utime(p_old, (old_ts, old_ts))
        ChronicleCompactor.write_entry(tmp_path, "reviewer", "note", "Active today", now=today_ts)

        rep = ChronicleCompactor.compact_past_days(
            tmp_path, today="2026-11-22", min_age_sec=3600.0, now=today_ts
        )
        assert rep["rolled_days"] == ["2026-11-20"]
        assert rep["entries_compacted"] == 1
        assert (tmp_path / "chronicle" / "2026-11-20.md").is_file()
        assert (tmp_path / "chronicle" / "2026-11-22").is_dir()

        # Idempotent second run
        rep2 = ChronicleCompactor.compact_past_days(
            tmp_path, today="2026-11-22", min_age_sec=3600.0, now=today_ts
        )
        assert rep2["rolled_days"] == []

        entries = ChronicleCompactor.read_entries(tmp_path)
        assert len(entries) == 2

        boot = BootstrapAssembler.assemble(
            policy_slice="Follow safety gates.",
            security_summary="taint=clean",
            open_blockers=["Verify CI run"],
            recent_entries=entries,
            token_budget=120,
        )
        assert boot["estimated_tokens"] <= 125
        assert "Follow safety gates." in boot["text"]


class TestPolicySliceAndTierLadder:
    def test_role_slicer_budget_and_tier_ladder(self, tmp_path: Path, capsys):
        doc = (
            "# Skill Guide\n\n"
            "## Shared Rules\nAlways safe.\n\n"
            "## Executor Details\n<!-- scope: executor, reviewer -->\nRun tests first.\n\n"
            "## Searcher Rules\n<!-- scope: retriever -->\nUse grep.\n"
        )
        md_file = tmp_path / "POLICY.md"
        md_file.write_text(doc, encoding="utf-8")

        assert doctor.main(["--slice", "executor", "--slice-file", str(md_file)]) == 0
        sliced = capsys.readouterr().out
        assert "Executor Details" in sliced
        assert "Shared Rules" in sliced
        assert "Searcher Rules" not in sliced

        assert doctor.main(["--check-budget", "--slice-file", str(md_file), "--max-chars", "500"]) == 0
        capsys.readouterr()

        hint, cleaned = extract_depth_hint("Audit module carefully [depth:deep]")
        assert hint == "deep"
        assert cleaned == "Audit module carefully"

        route, out_desc = TierLadderResolver().resolve(
            "retriever", tier="economy", description="Find symbols [depth:fast]"
        )
        assert route.effort == "low"
        assert route.tier == "economy"
        assert out_desc == "Find symbols"
