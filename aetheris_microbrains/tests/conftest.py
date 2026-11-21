"""Pytest configuration for aetheris_microbrains standalone test suite.

Maps legacy top-level test imports (`core.*`, `tools.*`, `evolution.*`) onto the
canonical `aetheris_microbrains.*` modules inside `sys.modules` ONLY during pytest
runs so module globals and monkeypatches share the exact same singletons, while
keeping `aetheris_microbrains/__init__.py` free of `sys.path` mutations.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TESTS_DIR.parent
_REPO_ROOT = _PKG_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical_name: str):
        self.canonical_name = canonical_name

    def create_module(self, spec):
        mod = importlib.import_module(self.canonical_name)
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    _PREFIXES = ("core", "tools", "evolution")

    def find_spec(self, fullname, path, target=None):
        head = fullname.split(".", 1)[0]
        if head not in self._PREFIXES:
            return None
        canonical = f"aetheris_microbrains.{fullname}"
        if importlib.util.find_spec(canonical) is None:
            return None
        is_pkg = "." not in fullname or (_PKG_DIR / fullname.replace(".", "/")).is_dir()
        return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical), is_package=is_pkg)


if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _LegacyAliasFinder())


import os

import pytest


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
