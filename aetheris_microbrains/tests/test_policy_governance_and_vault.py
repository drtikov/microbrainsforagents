"""Tests for rule vitality decay, oversight load gating, verified policy synthesis,
counterfactual policy replay, governance provenance chains, and the directive vault.
"""

from __future__ import annotations

import json
import time

import pytest

from aetheris_microbrains.audit import explain_entry, format_entry
from aetheris_microbrains.audit import main as audit_main
from aetheris_microbrains.core.directive_vault import DirectiveVault
from aetheris_microbrains.core.layered_decision_engine import PolicyConfig, decide
from aetheris_microbrains.core.mcp_security import SessionSecurityState
from aetheris_microbrains.core.policy_governance import (
    GovernanceLedger,
    PendingRuleQueue,
    RuleVitalityLedger,
    evaluate_policy_candidate,
    synthesize_policy_candidates,
    verify_policy_patch,
)
from aetheris_microbrains.core.session_rollup import assemble_bootstrap_context
from aetheris_microbrains.hook import evaluate_hook_payload


def test_rule_vitality_decay_and_stale_reconfirm(tmp_path):
    """Approved allow rules decay over idle time/failures and downgrade to escalate."""
    ledger = RuleVitalityLedger(path=tmp_path / "vitality.json")
    t0 = 1_700_000_000.0
    ledger.register("allow_make_test", half_life_days=10.0, min_vitality=0.35, now=t0)
    ledger.record_outcome("allow_make_test", clean=True, now=t0)

    fresh = ledger.check_rule("allow_make_test", now=t0 + 86400)
    assert fresh["known"] is True
    assert fresh["stale"] is False
    assert fresh["vitality"] > 0.35

    pol = PolicyConfig(
        custom_rules=(
            {
                "id": "allow_make_test",
                "action": "allow",
                "tool": "execute_shell",
                "cmd_regex": r"^make\s+test(\s|$)",
                "provenance": {"approved_by": "sec-lead", "rationale": "Verified unit runner"},
            },
        )
    )
    ok_res = decide(
        {"tool": "execute_shell", "cmd": "make test"},
        context={"vitality_ledger": ledger, "now": t0 + 86400},
        policy=pol,
    )
    assert ok_res["decision"] == "allow"
    assert ok_res["reason_code"] == "layer0_custom_policy_rule:allow_make_test"
    assert ok_res["rule_provenance"]["approved_by"] == "sec-lead"

    # Advance past multiple half-lives so vitality decays below min_vitality (0.35)
    stale_res = decide(
        {"tool": "execute_shell", "cmd": "make test"},
        context={"vitality_ledger": ledger, "now": t0 + 30 * 86400},
        policy=pol,
    )
    assert stale_res["decision"] == "escalate"
    assert stale_res["reason_code"] == "layer0_stale_rule_reconfirm:allow_make_test"

    # Reconfirming restores allow behavior
    ledger.reconfirm("allow_make_test", now=t0 + 30 * 86400)
    restored = decide(
        {"tool": "execute_shell", "cmd": "make test"},
        context={"vitality_ledger": ledger, "now": t0 + 30 * 86400 + 60},
        policy=pol,
    )
    assert restored["decision"] == "allow"


def test_constitutional_invariants_cannot_be_relaxed_by_custom_allow():
    """Custom allow rules never override Layer-0 protected paths or Layer-1 hard vetoes."""
    permissive_pol = PolicyConfig(
        custom_rules=(
            {"id": "too_broad_shell", "action": "allow", "tool": "execute_shell", "cmd_regex": r"^rm\b"},
            {"id": "too_broad_read", "action": "allow", "tool": "read_file", "path_glob": "*"},
            {"id": "too_broad_edit", "action": "allow", "tool": "edit_file", "path_glob": "src/*"},
        )
    )
    assert decide({"tool": "execute_shell", "cmd": "rm -rf /"}, policy=permissive_pol)["decision"] == "deny"
    assert decide({"tool": "read_file", "path": ".ssh/id_rsa"}, policy=permissive_pol)["decision"] != "allow"
    assert decide(
        {"tool": "edit_file", "path": "src/a.py", "diff": "eval(user_in)\n"},
        policy=permissive_pol,
    )["decision"] == "deny"


def test_session_oversight_load_and_anti_flooding_hook(tmp_path):
    """High-rate escalations in a tainted session trigger oversight flood veto."""
    sess_file = tmp_path / "sess.json"
    sess = SessionSecurityState(sess_file)
    sess.untrusted_external_ingested = True
    sess.save()

    now = time.time()
    for idx in range(7):
        gate = sess.evaluate_oversight_gate(
            "escalate",
            "layer1_unrecognized_shell_command_escalate",
            fingerprint="execute_shell:layer1_unrecognized_shell_command_escalate",
            now=now + idx,
            window_sec=60.0,
            flood_threshold=8,
        )
        assert gate["decision"] == "pass"

    assert gate["oversight"]["high_friction_patterns"]

    # 8th escalation inside the tainted window trips the anti-flooding veto
    hook_res = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "custom_unlisted_tool --flag"}},
        session_state_path=str(sess_file),
    )
    assert hook_res["decision"] == "deny"
    assert hook_res["exit_code"] == 2
    assert hook_res["reason_code"] == "layer0_oversight_flood_veto"


def test_policy_synthesis_verifier_and_pending_queue_workflow(tmp_path):
    """Synthesizer proposes verified narrow rules; operator approval writes provenance + policy."""
    journal = [
        {"decision": "escalate", "tool": "execute_shell", "reason_code": "layer1_unrecognized_shell_command_escalate", "payload_preview": "just test", "input_sha256": f"sha{i}"}
        for i in range(4)
    ] + [
        # Must NOT propose rules for protected path escalations
        {"decision": "escalate", "tool": "read_file", "reason_code": "layer0_protected_path_read_requires_confirmation", "payload_preview": ".env", "input_sha256": f"sec{i}"}
        for i in range(5)
    ]
    cands = synthesize_policy_candidates(journal, min_occurrences=3)
    assert len(cands) == 1
    assert cands[0]["verified"] is True
    assert cands[0]["tool"] == "execute_shell"

    bad_ver = verify_policy_patch({"id": "bad_wild", "action": "allow", "tool": "execute_shell", "cmd_regex": ".*"})
    assert bad_ver["ok"] is False

    q = PendingRuleQueue(path=tmp_path / "pending.json")
    assert q.stage(cands) == 1
    out_policy_file = tmp_path / "approved_policy.json"
    ledger_file = tmp_path / "gov.jsonl"
    vitality_file = tmp_path / "vit.json"

    approved = q.approve(
        cands[0]["id"],
        approver="alice",
        rationale="Verified project test runner",
        policy_out_path=out_policy_file,
        ledger_path=ledger_file,
        vitality_path=vitality_file,
    )
    assert approved["provenance"]["approved_by"] == "alice"
    assert q.list_pending() == []
    assert out_policy_file.is_file()
    loaded = PolicyConfig.from_file(out_policy_file)
    assert decide({"tool": "execute_shell", "cmd": "just test"}, policy=loaded)["decision"] == "allow"


def test_counterfactual_policy_replay_and_audit_cli_eval(tmp_path, capsys):
    """evaluate_policy_candidate and `audit --eval-policy` score safe candidate promotions cleanly."""
    j_path = tmp_path / "journal.jsonl"
    entries = [
        {"tool": "execute_shell", "cmd": "just lint", "decision": "escalate", "reason_code": "layer1_unrecognized_shell_command_escalate"},
        {"tool": "execute_shell", "cmd": "just fmt", "decision": "escalate", "reason_code": "layer1_unrecognized_shell_command_escalate"},
    ]
    j_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    cand_path = tmp_path / "cand_policy.json"
    cand_path.write_text(
        json.dumps({
            "custom_rules": [
                {"id": "allow_just", "action": "allow", "tool": "execute_shell", "cmd_regex": r"^just\s+\w+(\s|$)"}
            ]
        }),
        encoding="utf-8",
    )
    rc = audit_main(["--eval-policy", str(cand_path), "--journal-path", str(j_path), "--json"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["promote_safe"] is True
    assert rep["newly_allowed"] == 2
    assert rep["flip_matrix"].get("escalate->allow") == 2
    assert rep["upper_risk_bound_95"] is not None

    # Empty-prefix / bogus check: confirm direct evaluator detects redteam probes when tampered
    eval_clean = evaluate_policy_candidate(PolicyConfig.from_file(cand_path), journal_entries=entries)
    assert eval_clean["probes_checked"] >= 10
    assert eval_clean["safety_regressions"] == []


def test_governance_ledger_hash_chain_and_audit_provenance(tmp_path):
    """GovernanceLedger detects tampering and audit formatters surface rule provenance."""
    gpath = tmp_path / "governance.jsonl"
    ledger = GovernanceLedger(path=gpath)
    ledger.append("rule_a", "staged", actor="alice", rationale="initial")
    ledger.append("rule_a", "approved", actor="bob", approved_by="bob", rationale="reviewed", evidence_refs=["sha123"])
    assert ledger.verify_chain() == {"valid": True, "count": 2, "broken_at": None}

    lines = gpath.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["rationale"] = "forged"
    lines[0] = json.dumps(tampered)
    gpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert ledger.verify_chain()["valid"] is False

    entry = {
        "decision": "allow",
        "layer": 0,
        "reason_code": "layer0_custom_policy_rule:rule_a",
        "label": "ok",
        "rule_provenance": {"approved_by": "bob", "rationale": "reviewed", "vitality": 0.92},
    }
    assert "Provenance: approved_by=bob vitality=0.92 | reviewed" in format_entry(entry)
    assert "Provenance: approved_by=bob | vitality=0.92 | reviewed" in explain_entry(entry)


def test_directive_vault_nvpt_ranking_and_bootstrap_integration(tmp_path):
    """Directives require approval, pack by net-value-per-token, and stale items ask first."""
    vault = DirectiveVault(path=tmp_path / "vault.json")
    t0 = 1_700_000_000.0

    # Pending directives must never appear in bootstrap output
    vault.propose("unapproved_rule", guidance="Never auto-inject unapproved notes.", author="alice", now=t0)

    # Approve two directives: concise high-utility vs verbose lower-utility vs decayed
    vault.propose("run_targeted_pytest", guidance="Run pytest -q on changed files first.", scope_glob="tests/*", author="alice", half_life_days=30.0, now=t0)
    vault.approve("run_targeted_pytest", approver="bob", now=t0)
    vault.record_outcome("run_targeted_pytest", clean=True, now=t0)
    vault.record_outcome("run_targeted_pytest", clean=True, now=t0)

    vault.propose("verbose_note", guidance=" ".join(["documentation"] * 60), scope_glob="tests/*", author="alice", half_life_days=30.0, now=t0)
    vault.approve("verbose_note", approver="bob", now=t0)

    vault.propose("old_hint", guidance="Check legacy script.", scope_glob="tests/*", author="alice", half_life_days=5.0, now=t0 - 30 * 86400)
    vault.approve("old_hint", approver="bob", now=t0 - 30 * 86400)

    selection = vault.select_for_bootstrap(token_budget=30, scope_path="tests/test_a.py", now=t0 + 60)
    selected_ids = [d["directive_id"] for d in selection["selected"]]
    stale_ids = [d["directive_id"] for d in selection["stale_for_review"]]
    assert selected_ids == ["run_targeted_pytest"]
    assert "old_hint" in stale_ids
    assert "unapproved_rule" not in selected_ids + stale_ids

    boot = assemble_bootstrap_context(
        policy_slice="strict",
        directives=selection["selected"],
        token_budget=500,
    )
    assert "## Active Directives" in boot["text"]
    assert "run_targeted_pytest" in boot["text"]
