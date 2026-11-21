"""Pytest configuration for aetheris_microbrains standalone test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TESTS_DIR.parent
_REPO_ROOT = _PKG_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolate_decision_journal(tmp_path, monkeypatch):
    """Redirect decision-journal writes to a per-test temp file.

    Tests exercising `decide` / `evaluate_hook_payload` journal their verdicts;
    without this they append to the real `memory/decision_journal.jsonl` and
    pollute production audit history.
    """
    from aetheris_microbrains.core import layered_decision_engine
    from aetheris_microbrains import hook as _hook

    tmp_journal = tmp_path / "decision_journal.jsonl"
    monkeypatch.setattr(layered_decision_engine, "DEFAULT_JOURNAL_PATH", tmp_journal)
    monkeypatch.setattr(_hook, "_DEFAULT_HOOK_JOURNAL_PATH", tmp_journal)
    monkeypatch.setenv("AETHERIS_JOURNAL_PATH", str(tmp_journal))
