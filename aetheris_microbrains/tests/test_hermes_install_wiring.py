"""`aetheris install hermes` must write what Hermes reads: config.yaml `hooks:`.

Regression guard for the inert integration: the installer used to create
~/.hermes/aetheris_hooks.json, a file no Hermes code path opens, so hooks were
never registered while the command reported success.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from aetheris_microbrains import cli  # noqa: E402
from aetheris_microbrains.hosts.hermes_config import build_hermes_hooks  # noqa: E402

CLAUDE_BUNDLE = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Bash|Edit|Write|MultiEdit|Read|Grep|mcp__.*",
                "hooks": [
                    {
                        "type": "command",
                        "command": '"/usr/bin/python3" -m aetheris_microbrains.hook --host hermes',
                        "onFailure": "block",
                    }
                ],
            }
        ],
        "PostToolUse": [
            {
                "matcher": "Bash|Edit|Write",
                "hooks": [
                    {
                        "type": "command",
                        "command": '"/usr/bin/python3" -m aetheris_microbrains.hook --host hermes --event post-tool-use',
                    }
                ],
            }
        ],
    }
}


def test_events_and_matcher_are_hermes_shaped():
    block = build_hermes_hooks(CLAUDE_BUNDLE)
    assert set(block) == {"pre_tool_call", "post_tool_call"}
    pre = block["pre_tool_call"][0]
    assert pre["fail_closed"] is True
    import re

    matcher = re.compile(pre["matcher"])
    # Hermes tool names must match; Claude-only names must not.
    assert matcher.search("terminal") and matcher.search("write_file")
    assert not matcher.search("Bash")
    assert "fail_closed" not in block["post_tool_call"][0]


def test_install_writes_config_yaml_and_skips_legacy_json(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  name: keep-me\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cli, "_build_generic_hook_bundle", lambda *a, **k: CLAUDE_BUNDLE)

    rc = cli.main(["install", "hermes", "--profile", "dev"])
    assert rc == 0

    import yaml

    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["name"] == "keep-me"  # untouched keys survive the merge
    assert cfg["hooks"]["pre_tool_call"][0]["matcher"].startswith("terminal|")
    assert "aetheris_microbrains.hook" in cfg["hooks"]["pre_tool_call"][0]["command"]
    # The file no Hermes code reads must not be created any more.
    assert not (home / "aetheris_hooks.json").exists()


def test_install_is_idempotent(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cli, "_build_generic_hook_bundle", lambda *a, **k: CLAUDE_BUNDLE)

    assert cli.main(["install", "hermes"]) == 0
    assert cli.main(["install", "hermes"]) == 0

    import yaml

    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert len(cfg["hooks"]["pre_tool_call"]) == 1
