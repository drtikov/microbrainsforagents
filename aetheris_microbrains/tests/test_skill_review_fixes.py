"""Regression tests for the skill-review fix set (Nov 2026).

Covers: NotebookEdit AST coverage, harness-native tool classification,
read-only pip allowlist, installer contract (idempotency / matchers / remove),
and decision-journal rotation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from aetheris_microbrains.core.layered_decision_engine import (
    TOOL_RISK_CLASSES,
    _is_command_allowlisted,
    analyze_shell_command,
    decide,
)
from aetheris_microbrains.hook import _normalize_hook_payload, evaluate_hook_payload

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTALLER = _REPO_ROOT / ".claude" / "skills" / "microbrains" / "install_hook.py"

_PRE_MATCHER = "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
_POST_MATCHER = "Bash|Read|Grep|WebFetch|WebSearch|mcp__.*"

# ---------------------------------------------------------------------------
# I1: harness-native local vs external web tools
# ---------------------------------------------------------------------------


def test_harness_native_tools_classified_read_only():
    for name in ("task", "agent", "todowrite"):
        assert TOOL_RISK_CLASSES.get(name) == "READ_ONLY"
    for name in ("webfetch", "websearch"):
        assert TOOL_RISK_CLASSES.get(name) == "EXTERNAL_ACTION"


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "WebSearch", "tool_input": {"query": "test"}},
        {"tool_name": "TodoWrite", "tool_input": {"todos": []}},
        {"tool_name": "Task", "tool_input": {"prompt": "explore repo"}},
        {"tool_name": "Agent", "tool_input": {"prompt": "explore repo"}},
    ],
    ids=["websearch", "todowrite", "task", "agent"],
)
def test_harness_tools_allow_without_prompt(payload):
    verdict = evaluate_hook_payload(payload, log_decisions=False)
    assert verdict["decision"] == "allow"
    assert verdict["reason_code"] == "layer1_structural_verified"


# ---------------------------------------------------------------------------
# I3: NotebookEdit cell source is AST-analyzed instead of blanket-escalated
# ---------------------------------------------------------------------------


def test_notebookedit_normalizes_to_edit_with_diff():
    norm = _normalize_hook_payload(
        {
            "tool_name": "NotebookEdit",
            "tool_input": {
                "notebook_path": "analysis.ipynb",
                "cell_id": "cell-1",
                "new_source": "print(sum(range(10)))",
            },
        }
    )
    assert norm["tool"] == "edit_file"
    assert norm["path"] == "analysis.ipynb"
    assert norm["diff"] == "print(sum(range(10)))"


def test_notebookedit_benign_cell_allows():
    verdict = evaluate_hook_payload(
        {
            "tool_name": "NotebookEdit",
            "tool_input": {
                "notebook_path": "analysis.ipynb",
                "new_source": "print(sum(range(10)))",
            },
        },
        log_decisions=False,
    )
    assert verdict["decision"] == "allow"


def test_notebookedit_dangerous_cell_denies():
    verdict = evaluate_hook_payload(
        {
            "tool_name": "NotebookEdit",
            "tool_input": {
                "notebook_path": "analysis.ipynb",
                "new_source": "import os\nos.system('rm -rf /tmp/x')",
            },
        },
        log_decisions=False,
    )
    assert verdict["decision"] == "deny"
    assert verdict["reason_code"] == "layer1_ast_security_veto"


def test_notebookedit_markdown_cell_escalates_not_denies():
    """Markdown cells must not be Python-linted; prose should ask, never false-deny."""
    verdict = evaluate_hook_payload(
        {
            "tool_name": "NotebookEdit",
            "tool_input": {
                "notebook_path": "analysis.ipynb",
                "cell_type": "markdown",
                "new_source": "# Title\n\nSome **markdown** text with <code>print(x)</code> examples.",
            },
        },
        log_decisions=False,
    )
    assert verdict["decision"] == "escalate"
    assert verdict["reason_code"] == "layer1_uninspected_tool"


def test_notebookedit_raw_cell_escalates_not_denies():
    verdict = evaluate_hook_payload(
        {
            "tool_name": "NotebookEdit",
            "tool_input": {
                "notebook_path": "analysis.ipynb",
                "cell_type": "raw",
                "new_source": "def broken(:",
            },
        },
        log_decisions=False,
    )
    assert verdict["decision"] == "escalate"


def test_notebookedit_code_cell_still_linted_with_explicit_type():
    verdict = evaluate_hook_payload(
        {
            "tool_name": "NotebookEdit",
            "tool_input": {
                "notebook_path": "analysis.ipynb",
                "cell_type": "code",
                "new_source": "import os\nos.system('rm -rf /tmp/x')",
            },
        },
        log_decisions=False,
    )
    assert verdict["decision"] == "deny"


# ---------------------------------------------------------------------------
# I4: read-only pip introspection allowlisted in any profile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", ["pip show pytest", "pip list", "pip --version", "pip3 show requests"])
def test_pip_readonly_allowlisted(cmd):
    res = analyze_shell_command(cmd, policy=None)
    assert res["is_safe"], res["findings"]
    assert res["allowlisted"], res["unrecognized_commands"]


def test_pip_install_escalates_in_strict_profile():
    res = analyze_shell_command("pip install requests", policy=None)
    assert res["is_safe"]
    assert not res["allowlisted"]  # unrecognized -> escalate (ask), not deny


# ---------------------------------------------------------------------------
# I7: installer contract (loaded from the protected .claude/ tree via importlib)
# ---------------------------------------------------------------------------


def _load_installer():
    sys.dont_write_bytecode = True  # keep __pycache__ out of the skill dir
    spec = importlib.util.spec_from_file_location("mb_install_hook_test_mod", _INSTALLER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_installer_detects_aetheris_entries_recursively():
    mod = _load_installer()
    assert mod._is_aetheris_entry({"command": "python -m aetheris_microbrains.hook --log-decisions"})
    wrapper = {
        "matcher": "*",
        "hooks": [{"type": "command", "command": "python -m aetheris_microbrains.hook"}],
    }
    assert mod._is_aetheris_entry(wrapper)
    assert not mod._is_aetheris_entry({"command": "echo hi"})
    assert not mod._is_aetheris_entry({"matcher": "*", "hooks": [{"command": "echo hi"}]})


def test_installer_install_is_idempotent_with_narrow_matchers(tmp_path, monkeypatch):
    mod = _load_installer()
    settings = tmp_path / "settings.local.json"
    monkeypatch.setattr(mod, "SETTINGS_LOCAL_PATH", settings)
    monkeypatch.setattr(mod, "SETTINGS_SHARED_PATH", settings)
    assert mod.install(strict_ast=False, shadow=False, shared=False) == 0
    assert mod.install(strict_ast=False, shadow=False, shared=False) == 0  # idempotent

    cfg = json.loads(settings.read_text(encoding="utf-8"))
    pre = [e for e in cfg["hooks"]["PreToolUse"] if mod._is_aetheris_entry(e)]
    post = [e for e in cfg["hooks"]["PostToolUse"] if mod._is_aetheris_entry(e)]
    assert len(pre) == 1 and len(post) == 1
    assert pre[0]["matcher"] == _PRE_MATCHER
    assert pre[0]["hooks"][0].get("onFailure") == "block"
    assert post[0]["matcher"] == _POST_MATCHER
    post_hook = post[0]["hooks"][0]
    assert "async" not in post_hook
    assert "--log-decisions" in post_hook["command"]


def test_installer_remove_strips_aetheris_only(tmp_path, monkeypatch):
    mod = _load_installer()
    settings = tmp_path / "settings.local.json"
    monkeypatch.setattr(mod, "SETTINGS_LOCAL_PATH", settings)
    monkeypatch.setattr(mod, "SETTINGS_SHARED_PATH", settings)
    mod.install(strict_ast=False, shadow=False, shared=False)
    assert mod.remove(shared=False) == 0
    cfg = json.loads(settings.read_text(encoding="utf-8"))
    assert all(not mod._is_aetheris_entry(e) for e in cfg["hooks"].get("PreToolUse", []))
    assert all(not mod._is_aetheris_entry(e) for e in cfg["hooks"].get("PostToolUse", []))


def test_installer_auto_allow_flag_appended(tmp_path, monkeypatch):
    mod = _load_installer()
    settings = tmp_path / "settings.local.json"
    monkeypatch.setattr(mod, "SETTINGS_LOCAL_PATH", settings)
    monkeypatch.setattr(mod, "SETTINGS_SHARED_PATH", settings)
    assert mod.install(strict_ast=False, shadow=False, shared=False, auto_allow=True) == 0
    cfg = json.loads(settings.read_text(encoding="utf-8"))
    pre_cmd = cfg["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    post_cmd = cfg["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
    assert "--auto-allow" in pre_cmd
    assert "--auto-allow" not in post_cmd
    assert "--shadow" not in pre_cmd


# ---------------------------------------------------------------------------
# Journal rotation via append_jsonl_bounded (10 MiB default, .1 generation)
# ---------------------------------------------------------------------------


def test_journal_rotation(tmp_path):
    from aetheris_microbrains.core.runtime_paths import append_jsonl_bounded

    target = tmp_path / "decision_journal.jsonl"
    append_jsonl_bounded(target, "0123456789", max_bytes=10)  # 11 bytes with newline >= limit
    append_jsonl_bounded(target, "abcdefghij", max_bytes=10)
    assert target.read_text(encoding="utf-8").strip() == "abcdefghij"
    rotated = target.with_name(target.name + ".1")
    assert rotated.exists()
    assert "0123456789" in rotated.read_text(encoding="utf-8")
