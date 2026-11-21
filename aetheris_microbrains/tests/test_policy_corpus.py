"""Declarative policy corpus regression tests loaded from tests/corpus/*.txt."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from aetheris_microbrains import PolicyConfig, decide

_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def _load_corpus(filename: str) -> List[str]:
    path = _CORPUS_DIR / filename
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


@pytest.mark.parametrize("cmd", _load_corpus("shell_allow.txt"))
def test_corpus_shell_allow_strict_and_dev(cmd: str) -> None:
    assert decide({"tool": "execute_shell", "cmd": cmd}, policy=PolicyConfig(profile="strict"))["decision"] == "allow", cmd
    assert decide({"tool": "execute_shell", "cmd": cmd}, policy=PolicyConfig(profile="dev"))["decision"] == "allow", cmd


@pytest.mark.parametrize("cmd", _load_corpus("shell_deny.txt"))
def test_corpus_shell_deny_strict_and_dev(cmd: str) -> None:
    assert decide({"tool": "execute_shell", "cmd": cmd}, policy=PolicyConfig(profile="strict"))["decision"] == "deny", cmd
    assert decide({"tool": "execute_shell", "cmd": cmd}, policy=PolicyConfig(profile="dev"))["decision"] == "deny", cmd


@pytest.mark.parametrize("cmd", _load_corpus("shell_escalate_strict.txt"))
def test_corpus_shell_escalate_strict(cmd: str) -> None:
    res = decide({"tool": "execute_shell", "cmd": cmd}, policy=PolicyConfig(profile="strict"))
    assert res["decision"] == "escalate", f"Expected escalate for {cmd!r}, got {res}"


@pytest.mark.parametrize("cmd", _load_corpus("shell_escalate_dev.txt"))
def test_corpus_shell_escalate_dev(cmd: str) -> None:
    res = decide({"tool": "execute_shell", "cmd": cmd}, policy=PolicyConfig(profile="dev"))
    assert res["decision"] == "escalate", f"Expected escalate in dev for {cmd!r}, got {res}"
