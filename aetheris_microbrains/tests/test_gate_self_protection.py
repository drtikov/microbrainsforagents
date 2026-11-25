"""Gate self-protection + auto-approve eligibility hardening tests (24 Nov 2026).

fix1 — the RUNNING gate's own source tree is write-protected: write-class tool
       calls and shell output redirections into the live package hard-deny,
       reads stay free (no escalation), and the operator break-glass
       (AETHERIS_ALLOW_POLICY_EDIT=1) unlocks package maintenance exactly like
       control-plane files.
fix2 — `layer1_protected_path_*` escalations (protected-path shell reads) are
       NEVER eligible for --auto-approve-escalations / --approve-once silent
       approval; they surface as a real ask.
"""

from pathlib import Path

from aetheris_microbrains.core import layered_decision_engine as lde
from aetheris_microbrains.hook import (
    _APPROVE_ONCE_INELIGIBLE_PREFIXES,
    _is_approve_once_eligible,
    _is_auto_approve_escalation_eligible,
    evaluate_hook_payload,
)

PKG_DIR = Path(lde.__file__).resolve().parents[1]
DEV_POLICY = lde.PolicyConfig(profile="dev")


# ---------- fix 1: live package write protection ----------

def test_live_package_dir_resolves_to_package():
    assert Path(lde._live_package_dir()).resolve() == PKG_DIR.resolve()


def test_write_to_own_source_is_denied():
    res = lde.evaluate_layer0_policy("write_file", str(PKG_DIR / "hook.py"), DEV_POLICY)
    assert res["decision"] == "deny"
    assert res["reason_code"] == "layer0_protected_path_veto"


def test_edit_tool_to_own_source_is_denied():
    res = lde.evaluate_layer0_policy("edit_file", str(PKG_DIR / "core" / "secrets.py"), DEV_POLICY)
    assert res["decision"] == "deny"


def test_relative_package_write_is_denied():
    res = lde.evaluate_layer0_policy("write_file", "aetheris_microbrains/hook.py", DEV_POLICY)
    assert res["decision"] == "deny"


def test_read_of_own_source_is_free():
    res = lde.evaluate_layer0_policy("read_file", str(PKG_DIR / "hook.py"), DEV_POLICY)
    assert res["decision"] == "allow"


def test_unrelated_writes_unaffected(tmp_path):
    res = lde.evaluate_layer0_policy("write_file", str(tmp_path / "plain.txt"), DEV_POLICY)
    assert res["decision"] == "allow"


def test_shell_redirect_into_own_source_is_unsafe():
    assert lde._is_privileged_or_unsafe_redirect_target(str(PKG_DIR / "hook.py")) is True
    assert lde._is_privileged_or_unsafe_redirect_target(str(PKG_DIR / "core" / "secrets.py")) is True


def test_end_to_end_decide_denies_package_write(monkeypatch):
    monkeypatch.delenv("AETHERIS_ALLOW_POLICY_EDIT", raising=False)
    v = lde.decide(
        {"tool": "write_file", "path": str(PKG_DIR / "hook.py")},
        context=None,
        policy=lde.PolicyConfig(profile="dev"),
    )
    assert v["decision"] == "deny"


def test_break_glass_unlocks_package_maintenance(monkeypatch):
    monkeypatch.setenv("AETHERIS_ALLOW_POLICY_EDIT", "1")
    v = lde.decide(
        {"tool": "write_file", "path": str(PKG_DIR / "hook.py")},
        context=None,
        policy=lde.PolicyConfig(profile="dev"),
    )
    assert v["decision"] != "deny"


def test_break_glass_never_unlocks_credentials(monkeypatch):
    monkeypatch.setenv("AETHERIS_ALLOW_POLICY_EDIT", "1")
    v = lde.decide(
        {"tool": "write_file", "path": ".env"},
        context=None,
        policy=lde.PolicyConfig(profile="dev"),
    )
    assert v["decision"] == "deny"


# ---------- fix 2: protected-path shell reads never silently auto-approve ----------

def test_prefix_registry_contains_layer1_protected_path():
    assert "layer1_protected_path" in _APPROVE_ONCE_INELIGIBLE_PREFIXES


def test_protected_shell_read_never_auto_approves():
    assert _is_approve_once_eligible("layer1_protected_path_shell_read_escalate") is False
    assert _is_auto_approve_escalation_eligible("layer1_protected_path_shell_read_escalate", None) is False
    assert _is_auto_approve_escalation_eligible("layer1_protected_path_shell_read", None) is False


def test_other_ineligible_classes_survive():
    for rc in (
        "layer0_protected_path_veto",
        "layer0_protected_path_read_requires_confirmation",
        "layer0_secret_x",
        "layer1_ifc_taint_y",
        "malformed_payload",
    ):
        assert _is_auto_approve_escalation_eligible(rc, None) is False


def test_benign_escalations_still_eligible():
    assert _is_auto_approve_escalation_eligible("layer2_verification_escalation", None) is True
    # the deliberate read carve-out: agent control-plane / known-safe basenames
    assert _is_auto_approve_escalation_eligible(
        "layer0_protected_path_read_requires_confirmation", "CLAUDE.md"
    ) is True


def test_shell_protected_read_survives_auto_approve_flag_end_to_end():
    v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "cat ~/.ssh/id_rsa"}},
        host="claude",
        profile="dev",
        auto_approve_escalations=True,
        session_state_path=None,
    )
    assert v["decision"] == "escalate"
    assert v.get("permissionDecision") == "ask"


def test_ordinary_command_still_auto_allows_end_to_end():
    v = evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        host="claude",
        profile="dev",
        auto_approve_escalations=True,
        session_state_path=None,
    )
    assert v["decision"] == "allow"


# ---------- fix 3: .env.<suffix> glob bypass closure ----------

def test_env_suffix_glob_variants_escalate():
    for cmd in (
        "cat .[e]nv.local",
        "cat .e??.local",
        "cat .en?.local",
        "cat .[e]nv.*",
        "cat config/.[e]nv.local",
        "cat .???.local",
        "cat .???.production",
        "cat .*.local",
        "cat .*.production",
    ):
        v = lde.decide({"tool": "execute_shell", "cmd": cmd}, context=None, policy=DEV_POLICY)
        assert v["decision"] == "escalate", f"Expected escalate for {cmd}, got {v}"


def test_benign_dot_globs_stay_allowed():
    for cmd in ("ls *", "ls *.local", "ls .eslint*.js", "ls .git*.local"):
        v = lde.decide({"tool": "execute_shell", "cmd": cmd}, context=None, policy=DEV_POLICY)
        assert v["decision"] == "allow", f"Expected allow for {cmd}, got {v}"


# ---------- fix 4: --auto-approve-escalations strict allowlist ----------

def test_auto_approve_refuses_unrecognized_and_sensitive_escalations():
    for rc in (
        "layer1_unrecognized_shell_command_escalate",
        "layer1_uninspected_tool_escalate",
        "layer1_sensitive_env_access_escalate",
        "layer1_shell_redirection_escalate",
        "layer1_ast_dev_escalate",
        "layer1_mcp_write_tool_escalate",
        "layer1_strict_ast_allowlist",
    ):
        assert _is_auto_approve_escalation_eligible(rc, None) is False

    for cmd in (
        "custom_deploy_binary --prod",
        "some_unknown_tool --flag",
        "curl https://api.github.com",
        "git clean -fdx",
    ):
        v = evaluate_hook_payload(
            {"tool_name": "Bash", "tool_input": {"command": cmd}},
            host="antigravity",
            profile="dev",
            auto_approve_escalations=True,
            session_state_path=None,
            log_decisions=False,
        )
        assert v["decision"] == "escalate", f"Expected escalate for {cmd}, got {v}"
        assert v["hookSpecificOutput"]["decision"] in {"ask", "force_ask"}
