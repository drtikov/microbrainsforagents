"""Wire Aetheris hooks into the file Hermes actually reads.

Hermes' shell-hook bridge lives in `agent/shell_hooks.py` and registers whatever
sits under the `hooks:` key of `~/.hermes/config.yaml` (or `$HERMES_HOME`).
It has no loader for a host-specific JSON file, so writing
`~/.hermes/aetheris_hooks.json` produced an inert integration: the file was
created, `aetheris install hermes` reported success, and no hook ever fired.

Two contract differences matter:

* event names are snake_case (`pre_tool_call` / `post_tool_call`), not
  `PreToolUse` / `PostToolUse`;
* `matcher` is a regex over Hermes tool names (`terminal`, `patch`,
  `write_file`, ...), not over Claude Code's (`Bash`, `Edit`, `Write`, ...).

This module converts a Claude-shaped hook bundle into the Hermes shape and
merges it non-destructively into config.yaml.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Hermes tool names that can reach the filesystem, a shell, or the network.
# Kept as one alternation so a single hook entry covers them all.
HERMES_TOOL_MATCHER = (
    "terminal|execute_code|patch|write_file|read_file|search_files|"
    "browser_exec|web_search|web_extract|delegate_task|image_generate|"
    "video_generate|skill_manage|mcp_.*"
)

HERMES_EVENTS = {"PreToolUse": "pre_tool_call", "PostToolUse": "post_tool_call"}

_HOOK_MARKER = "aetheris_microbrains.hook"


def hermes_config_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home).expanduser() / "config.yaml"


def _command_from_bundle(event: str, entry: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    hooks = entry.get("hooks") if isinstance(entry, dict) else None
    if not isinstance(hooks, list) or not hooks:
        return None, False
    first = hooks[0] if isinstance(hooks[0], dict) else {}
    command = first.get("command")
    if not isinstance(command, str) or not command.strip():
        return None, False
    fail_closed = event == "PreToolUse" and first.get("onFailure") == "block"
    return command.strip(), fail_closed


def build_hermes_hooks(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Claude/Codex hook bundle -> Hermes `hooks:` block (stdlib only)."""
    out: Dict[str, Any] = {}
    for claude_event, entries in (bundle.get("hooks") or {}).items():
        hermes_event = HERMES_EVENTS.get(claude_event)
        if hermes_event is None or not isinstance(entries, list):
            continue
        specs = []
        for entry in entries:
            command, fail_closed = _command_from_bundle(claude_event, entry)
            if command is None:
                continue
            spec: Dict[str, Any] = {"command": command, "matcher": HERMES_TOOL_MATCHER, "timeout": 60}
            if fail_closed:
                spec["fail_closed"] = True
            specs.append(spec)
        if specs:
            out[hermes_event] = specs
    return out


def _load_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError(
            "Hermes stores hooks in config.yaml, so this installer needs PyYAML: "
            "pip install 'aetheris-microbrains[hermes]'"
        ) from exc
    return yaml


def merge_into_config(config: Dict[str, Any], hooks_block: Dict[str, Any]) -> Dict[str, Any]:
    """Insert our specs, dropping only our own previous entries (idempotent)."""
    merged = dict(config or {})
    hooks = merged.get("hooks")
    hooks = dict(hooks) if isinstance(hooks, dict) else {}
    for event, specs in hooks_block.items():
        existing = hooks.get(event)
        existing = list(existing) if isinstance(existing, list) else []
        kept = [
            item
            for item in existing
            if not (isinstance(item, dict) and _HOOK_MARKER in str(item.get("command", "")))
        ]
        hooks[event] = kept + specs
    merged["hooks"] = hooks
    return merged


def install_hermes_hooks(
    bundle: Dict[str, Any],
    config_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Merge the bundle into an existing config.yaml without touching other keys."""
    yaml = _load_yaml()
    target = Path(config_path) if config_path else hermes_config_path()
    config: Dict[str, Any] = {}
    if target.is_file():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        config = loaded if isinstance(loaded, dict) else {}
    hooks_block = build_hermes_hooks(bundle)
    merged = merge_into_config(config, hooks_block)
    if dry_run:
        return target, merged
    text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True, default_flow_style=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target, merged
