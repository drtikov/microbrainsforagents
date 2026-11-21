"""OS-standard runtime state path helpers and bounded JSONL logging (stdlib-only).

Keeps mutable runtime telemetry (`decision_journal.jsonl`,
`continual_learning_audit.json`) outside the Python package directory so
installed site-packages remain read-only safe and git working trees stay clean
when running tests or examples.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_JOURNAL_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB before .1 rotation


def get_state_dir() -> Path:
    """Resolve the writable user/runtime state directory for Aetheris Microbrains.

    Respects `AETHERIS_STATE_DIR` first; otherwise uses XDG / OS standard
    per-user state directories (`%LOCALAPPDATA%` on Windows, Application Support
    on macOS, `$XDG_STATE_HOME` / `~/.local/state` on Linux/POSIX).
    """
    override = str(os.getenv("AETHERIS_STATE_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = str(os.getenv("LOCALAPPDATA") or "").strip()
        root = Path(base).expanduser() if base else (Path.home() / "AppData" / "Local")
        return root / "aetheris-microbrains"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "aetheris-microbrains"
    xdg_state = str(os.getenv("XDG_STATE_HOME") or "").strip()
    root = Path(xdg_state).expanduser() if xdg_state else (Path.home() / ".local" / "state")
    return root / "aetheris-microbrains"


def get_default_journal_path() -> Path:
    """Return the default path for `decision_journal.jsonl` outside the package tree."""
    env_path = str(os.getenv("AETHERIS_JOURNAL_PATH") or "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return get_state_dir() / "decision_journal.jsonl"


def get_default_audit_log_path() -> Path:
    """Return the default path for `continual_learning_audit.json` outside the package tree."""
    return get_state_dir() / "continual_learning_audit.json"


def append_jsonl_bounded(
    target: Path,
    line: str,
    max_bytes: int = DEFAULT_JOURNAL_MAX_BYTES,
) -> None:
    """Atomically append one JSONL line with 0o600 permissions and size rotation."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if max_bytes > 0 and target.exists():
        try:
            if target.stat().st_size >= max_bytes:
                rotated = target.with_name(target.name + ".1")
                os.replace(str(target), str(rotated))
        except OSError:
            pass

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(str(target), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
    except Exception:
        # os.fdopen closes fd once constructed, but guard early constructor errors.
        raise
