"""Regression tests for the 15 technical & security audit fixes (v0.7.7)."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import pytest

from aetheris_microbrains.core.layered_decision_engine import (
    PolicyConfig,
    analyze_shell_command,
    decide,
)
from aetheris_microbrains.core.mcp_tiers import is_writeish_mcp_tool
from aetheris_microbrains.core.policy_governance import (
    PendingRuleQueue,
    RuleVitalityLedger,
    _journal_entry_to_payload,
)
from aetheris_microbrains.core.secrets import scrub_tool_output
from aetheris_microbrains.hook import (
    _build_sandbox_wrapped_command,
    evaluate_hook_payload,
    main as hook_main,
)
from aetheris_microbrains.hook_daemon import VerdictCache, evaluate_request


def test_secrets_base64_after_equals_and_20_byte_keys():
    """P1 #1: Scrubber catches base64/hex secrets right after `=` and 20-byte AWS keys."""
    gh_tok = ("ghp_" + "1234567890abcdef" * 2 + "abcd").encode("utf-8")
    aws_tok = ("AKIA" + "IOSFODNN7EXAMPLE").encode("utf-8")

    b64_gh = base64.b64encode(gh_tok).decode("ascii")
    res_gh = scrub_tool_output(f"token={b64_gh}")
    assert res_gh["secrets_found"] is True
    assert res_gh["secret_count"] >= 1

    b64_aws = base64.b64encode(aws_tok).decode("ascii")
    res_aws_b64 = scrub_tool_output(f"token={b64_aws}")
    assert res_aws_b64["secrets_found"] is True

    hex_aws = aws_tok.hex()
    res_aws_hex = scrub_tool_output(f"token={hex_aws}")
    assert res_aws_hex["secrets_found"] is True


def test_custom_rules_match_bash_alias_through_hook(tmp_path):
    """P1 #2: `custom_rules` with `tool: 'Bash'` fires even after hook normalization."""
    pol_path = tmp_path / "policy.json"
    pol_path.write_text(
        json.dumps(
            {
                "custom_rules": [
                    {
                        "id": "block_prod_ssh",
                        "action": "deny",
                        "tool": "Bash",
                        "cmd_regex": r"ssh\s+prod",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    verdict = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "ssh prod-db"}},
        policy_path=str(pol_path),
        log_decisions=False,
    )
    assert verdict["decision"] == "deny"
    assert verdict["reason_code"] == "layer0_custom_policy_rule:block_prod_ssh"

    # Also works when canonical tool name `execute_shell` is passed directly
    direct_cfg = PolicyConfig.from_file(str(pol_path))
    direct_v = decide({"tool": "execute_shell", "cmd": "ssh prod-db"}, policy=direct_cfg)
    assert direct_v["decision"] == "deny"
    assert direct_v["reason_code"] == "layer0_custom_policy_rule:block_prod_ssh"


def test_verdict_cache_thread_safety_and_daemon_context_guard(tmp_path):
    """P1 #3 & P2 #8: VerdictCache is thread-safe under concurrent access and respects context_guard."""
    cache = VerdictCache(max_entries=32, ttl_sec=5.0)
    errors: list[BaseException] = []

    def _worker(tid: int) -> None:
        try:
            for i in range(150):
                key = f"k-{(tid * 17 + i) % 64}"
                cache.put(key, {"decision": "allow", "tid": tid, "i": i})
                _ = cache.get(key)
        except BaseException as exc:  # pragma: no cover - captured for assertion
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    # Daemon evaluate_request must not serve stale cached verdict when context_guard=True
    big_file = tmp_path / "large.py"
    big_file.write_text("x = 1\n" * 6000, encoding="utf-8")
    req = {
        "event": "pre",
        "payload": {
            "tool_name": "Read",
            "tool_input": {"file_path": str(big_file)},
            "session_id": "sess_cg",
        },
        "flags": {"context_guard": True},
    }
    v, cached = evaluate_request(req, cache=cache)
    assert cached is False
    assert v["decision"] == "deny"
    assert v.get("hygiene_blocked") is True
    # Second request should also NOT return cached hit when context_guard=True
    _v2, cached2 = evaluate_request(req, cache=cache)
    assert cached2 is False


def test_journal_and_verdict_preserve_tool_name(tmp_path):
    """P1 #4: `decide()` records `"tool"` in output/journal so replay preserves `edit_file`."""
    jpath = tmp_path / "decision_journal.jsonl"
    v = evaluate_hook_payload(
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "src/app.py",
                "old_string": "a = 1",
                "new_string": "a = 2",
            },
        },
        log_decisions=True,
        journal_path=str(jpath),
    )
    assert v.get("journal_written") is True
    records = [json.loads(ln) for ln in jpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(records) == 1
    assert records[0].get("tool") == "edit_file"

    replayed = _journal_entry_to_payload(records[0])
    assert replayed is not None
    assert replayed["tool"] == "edit_file"


def test_hook_connects_rule_vitality_ledger(tmp_path, monkeypatch):
    """P2 #5: Stale allow rule in RuleVitalityLedger re-escalates through evaluate_hook_payload."""
    monkeypatch.setenv("AETHERIS_STATE_DIR", str(tmp_path))
    pol_path = tmp_path / "policy.json"
    pol_path.write_text(
        json.dumps(
            {
                "custom_rules": [
                    {
                        "id": "allow_custom_diag",
                        "action": "allow",
                        "tool": "Bash",
                        "cmd_regex": r"^my_diag_tool",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Register the rule 120 days ago so it decays past min_vitality
    stale_ts = time.time() - (120 * 86400.0)
    RuleVitalityLedger().register("allow_custom_diag", now=stale_ts)

    verdict = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "my_diag_tool --status"}},
        policy_path=str(pol_path),
        log_decisions=False,
    )
    assert verdict["decision"] == "escalate"
    assert verdict["reason_code"] == "layer0_stale_rule_reconfirm:allow_custom_diag"


def test_pending_rule_approve_handles_toml_and_yaml(tmp_path, monkeypatch):
    """P2 #6: PendingRuleQueue.approve rejects .toml cleanly and handles .yaml safely."""
    q = PendingRuleQueue(path=tmp_path / "pending.json")
    cand = {
        "id": "allow_pytest_diag",
        "action": "allow",
        "tool": "execute_shell",
        "cmd_regex": r"^pytest\b",
        "reason": "allow safe pytest",
    }

    # .toml rejects without crashing with JSONDecodeError or overwriting TOML as JSON
    q.stage([cand])
    toml_path = tmp_path / "security.toml"
    toml_path.write_text('profile = "strict"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="TOML"):
        q.approve(
            "allow_pytest_diag",
            approver="tester",
            policy_out_path=toml_path,
            ledger_path=tmp_path / "gov.jsonl",
            vitality_path=tmp_path / "vit.json",
        )
    assert toml_path.read_text(encoding="utf-8") == 'profile = "strict"\n'
    # Failed approve must NOT consume the pending rule from the queue
    assert len(q.list_pending()) == 1

    # .yaml either writes valid YAML when PyYAML is present or raises ValueError when absent
    yaml_path = tmp_path / "security.yaml"
    yaml_path.write_text("profile: strict\n", encoding="utf-8")
    try:
        import yaml  # type: ignore

        rec = q.approve(
            "allow_pytest_diag",
            approver="tester",
            policy_out_path=yaml_path,
            ledger_path=tmp_path / "gov.jsonl",
            vitality_path=tmp_path / "vit.json",
        )
        assert rec["id"] == "allow_pytest_diag"
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        assert loaded["profile"] == "strict"
        assert loaded["custom_rules"][0]["id"] == "allow_pytest_diag"
    except ImportError:
        with pytest.raises(ValueError, match="PyYAML"):
            q.approve(
                "allow_pytest_diag",
                approver="tester",
                policy_out_path=yaml_path,
                ledger_path=tmp_path / "gov.jsonl",
                vitality_path=tmp_path / "vit.json",
            )


def test_is_writeish_mcp_tool_avoids_substring_false_positives():
    """P2 #7: Token-based matching avoids false positives on safe read-only MCP tool names."""
    for safe_tool in (
        "get_email",
        "list_posts",
        "get_credit_limit",
        "get_design_tokens",
        "get_copyright",
    ):
        assert is_writeish_mcp_tool(safe_tool) is False, f"{safe_tool} unexpectedly flagged writeish"

    for write_tool in (
        "create_issue",
        "dropTable",
        "send_mail",
        "delete_file",
    ):
        assert is_writeish_mcp_tool(write_tool) is True, f"{write_tool} missed as writeish"


def test_context_guard_shadow_mode_no_fake_savings_and_preserves_shadow_decision(tmp_path):
    """P2 #8: Shadow-mode context_guard must not write fake savings and must set shadow_decision='deny'."""
    big_file = tmp_path / "huge.py"
    big_file.write_text("x = 1\n" * 6000, encoding="utf-8")
    econ_path = tmp_path / "economics.jsonl"
    hyg_path = tmp_path / "hygiene.json"

    res = evaluate_hook_payload(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": str(big_file)},
            "session_id": "sess_shadow_cg",
        },
        shadow_mode=True,
        context_guard=True,
        economics_ledger_path=str(econ_path),
        hygiene_ledger_path=str(hyg_path),
        log_decisions=False,
    )
    assert res["decision"] == "escalate"
    assert res["shadow_decision"] == "deny"
    assert res["shadow_reason_code"] == "hygiene_oversized_whole_read"
    assert not econ_path.exists()


def test_seatbelt_escapes_project_dir_and_allows_private_tmp(monkeypatch):
    """P2 #10: macOS Seatbelt profile includes /private/tmp and escapes project_dir."""
    monkeypatch.setattr(sys, "platform", "darwin")
    wrapped = _build_sandbox_wrapped_command(
        "ls -la",
        sandbox_mode="seatbelt",
        project_dir='/tmp/work"dir',
    )
    assert wrapped is not None
    assert '(subpath "/private/tmp")' in wrapped
    assert '\\"dir' in wrapped


def test_parse_simple_yaml_supports_security_md_custom_rules():
    """P2 #11: Fallback YAML parser handles SECURITY.md `custom_rules:` without PyYAML."""
    sample_yaml = """
# Security policy sample
require_approval_for_external: true
strict_ast_allowlist: true
allowed_ast_imports:
  - math
  - json
custom_rules:
  - id: "block_prod_ssh"
    action: "deny"
    tool: "Bash"
    cmd_regex: "ssh\\\\s+prod"
  - id: "escalate_vault_files"
    action: "escalate"
    path_glob: "*internal_vault*"
profile: "dev"
"""
    parsed = PolicyConfig._parse_simple_yaml(sample_yaml)
    assert parsed["require_approval_for_external"] is True
    assert parsed["allowed_ast_imports"] == ["math", "json"]
    assert parsed["profile"] == "dev"
    assert len(parsed["custom_rules"]) == 2
    assert parsed["custom_rules"][0]["id"] == "block_prod_ssh"
    assert parsed["custom_rules"][0]["action"] == "deny"
    assert parsed["custom_rules"][0]["tool"] == "Bash"
    assert parsed["custom_rules"][1]["path_glob"] == "*internal_vault*"

    cfg = PolicyConfig.from_dict(parsed)
    assert len(cfg.custom_rules) == 2


def test_install_hook_http_daemon_warns_on_ignored_policy_flags(tmp_path, capsys):
    """P2 #12: `install_hook --http-daemon` warns when CLI policy flags are ignored."""
    inst_path = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "microbrains" / "install_hook.py"
    spec = importlib.util.spec_from_file_location("install_hook_mod", inst_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rc = mod.install(
        strict_ast=True,
        shadow=False,
        profile="dev",
        policy="security.toml",
        settings_path=tmp_path / "settings.local.json",
        http_daemon="http://127.0.0.1:8765",
        context_guard=True,
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING: --http-daemon ignores CLI policy flags" in err
    assert "--profile dev" in err
    assert "--context-guard" in err


def test_powershell_analyze_shell_no_re_split_deprecation_warning():
    """P3 #13: PowerShell tokenization uses keyword `maxsplit=1` with zero DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        res = analyze_shell_command("Get-Content README.md")
    assert res["is_safe"] is True
    re_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning) and "maxsplit" in str(w.message)]
    assert not re_warnings


def test_hermes_host_env_var_recognized(monkeypatch, capsys):
    """Bonus #15: AETHERIS_HOOK_HOST=hermes is preserved by `hook.main()`."""
    monkeypatch.setenv("AETHERIS_HOOK_HOST", "hermes")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"tool_name": "terminal", "tool_input": {"command": "rm -rf /"}})),
    )
    rc = hook_main(["--session-state", ""])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "block"
