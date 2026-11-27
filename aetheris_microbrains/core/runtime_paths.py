"""OS-standard runtime state path helpers and bounded JSONL logging (stdlib-only).

Keeps mutable runtime telemetry (`decision_journal.jsonl`,
`continual_learning_audit.json`) outside the Python package directory so
installed site-packages remain read-only safe and git working trees stay clean
when running tests or examples.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None

DEFAULT_JOURNAL_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB before .1 rotation
JOURNAL_GENESIS_HASH = "0" * 64


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


def get_writable_store_dir() -> Path:
    """Return a writable store directory under the per-user state dir (never inside site-packages)."""
    return get_state_dir() / "store"


def get_default_session_state_path() -> Path:
    """Return the default path for `SessionSecurityState` outside the package tree."""
    env_path = str(os.getenv("AETHERIS_SESSION_STATE_PATH") or "").strip()
    if env_path and env_path.lower() != "auto":
        return Path(env_path).expanduser()
    return get_state_dir() / "sessions" / "session.json"


def get_default_audit_log_path() -> Path:
    """Return the default path for `continual_learning_audit.json` outside the package tree."""
    return get_state_dir() / "continual_learning_audit.json"


@contextmanager
def file_lock(target: Path) -> Iterator[None]:
    """Cross-platform advisory file lock serializing atomic reads/writes/rotations."""
    if _fcntl is None and _msvcrt is None:
        yield
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        if _fcntl is not None:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
        else:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write("0")
                fh.flush()
            fh.seek(0)
            _msvcrt.locking(fh.fileno(), _msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                _msvcrt.locking(fh.fileno(), _msvcrt.LK_UNLCK, 1)
    finally:
        fh.close()


_journal_lock = file_lock


def compute_journal_entry_hash(entry: Dict[str, Any], prev_hash: str) -> str:
    """Compute canonical SHA-256 chain digest over an audit entry and its predecessor hash."""
    body = {k: v for k, v in entry.items() if k not in ("entry_hash", "prev_hash")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(f"{prev_hash}:{canonical}".encode("utf-8")).hexdigest()


def _read_tail_prev_hash(target: Path) -> str:
    """Read the last entry's `entry_hash` from `target`, returning genesis when empty or legacy."""
    if not target.is_file():
        return JOURNAL_GENESIS_HASH
    try:
        size = target.stat().st_size
        if size == 0:
            return JOURNAL_GENESIS_HASH
        with open(target, "rb") as fh:
            if size > 65536:
                fh.seek(-65536, os.SEEK_END)
                fh.readline()
            tail_lines = [ln.strip() for ln in fh.read().decode("utf-8", errors="ignore").splitlines() if ln.strip()]
        if not tail_lines:
            return JOURNAL_GENESIS_HASH
        last_obj = json.loads(tail_lines[-1])
        if isinstance(last_obj, dict) and isinstance(last_obj.get("entry_hash"), str) and last_obj["entry_hash"]:
            return str(last_obj["entry_hash"])
    except Exception:
        pass
    return JOURNAL_GENESIS_HASH


def verify_jsonl_hash_chain(path: Path) -> Dict[str, Any]:
    """Verify SHA-256 hash-chain integrity across a JSONL journal file."""
    target = Path(path).expanduser()
    if not target.is_file():
        return {
            "ok": True,
            "path": str(target),
            "total_entries": 0,
            "chained_entries": 0,
            "legacy_entries": 0,
            "head_hash": JOURNAL_GENESIS_HASH,
        }

    total = 0
    chained = 0
    legacy = 0
    expected_prev = JOURNAL_GENESIS_HASH
    saw_chained = False

    with open(target, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            total += 1
            try:
                obj = json.loads(stripped)
            except Exception as exc:
                return {
                    "ok": False,
                    "path": str(target),
                    "total_entries": total,
                    "chained_entries": chained,
                    "legacy_entries": legacy,
                    "broken_at_line": line_no,
                    "reason": f"invalid_json:{type(exc).__name__}",
                }
            if not isinstance(obj, dict):
                return {
                    "ok": False,
                    "path": str(target),
                    "total_entries": total,
                    "chained_entries": chained,
                    "legacy_entries": legacy,
                    "broken_at_line": line_no,
                    "reason": "non_object_entry",
                }

            has_prev = "prev_hash" in obj
            has_entry = "entry_hash" in obj
            if not has_prev and not has_entry:
                if saw_chained:
                    return {
                        "ok": False,
                        "path": str(target),
                        "total_entries": total,
                        "chained_entries": chained,
                        "legacy_entries": legacy,
                        "broken_at_line": line_no,
                        "reason": "missing_chain_fields_after_chained_entry",
                    }
                legacy += 1
                continue

            if not (has_prev and has_entry):
                return {
                    "ok": False,
                    "path": str(target),
                    "total_entries": total,
                    "chained_entries": chained,
                    "legacy_entries": legacy,
                    "broken_at_line": line_no,
                    "reason": "incomplete_chain_fields",
                }

            saw_chained = True
            chained += 1
            actual_prev = str(obj.get("prev_hash") or "")
            actual_hash = str(obj.get("entry_hash") or "")
            if actual_prev != expected_prev:
                return {
                    "ok": False,
                    "path": str(target),
                    "total_entries": total,
                    "chained_entries": chained,
                    "legacy_entries": legacy,
                    "broken_at_line": line_no,
                    "reason": "prev_hash_mismatch",
                }
            computed = compute_journal_entry_hash(obj, expected_prev)
            if actual_hash != computed:
                return {
                    "ok": False,
                    "path": str(target),
                    "total_entries": total,
                    "chained_entries": chained,
                    "legacy_entries": legacy,
                    "broken_at_line": line_no,
                    "reason": "entry_hash_mismatch",
                }
            expected_prev = actual_hash

    return {
        "ok": True,
        "path": str(target),
        "total_entries": total,
        "chained_entries": chained,
        "legacy_entries": legacy,
        "head_hash": expected_prev,
    }


def append_jsonl_bounded(
    target: Path,
    line: str,
    max_bytes: int = DEFAULT_JOURNAL_MAX_BYTES,
    *,
    hash_chain: bool = True,
) -> None:
    """Atomically append one JSONL line with 0o600 permissions, hash chaining, and lock-serialized rotation."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(target):
        if max_bytes > 0 and target.exists():
            try:
                if target.stat().st_size >= max_bytes:
                    rotated = target.with_name(target.name + ".1")
                    os.replace(str(target), str(rotated))
            except OSError:
                pass

        out_line = line
        if hash_chain:
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "entry_hash" not in parsed:
                    prev = _read_tail_prev_hash(target)
                    parsed["prev_hash"] = prev
                    parsed["entry_hash"] = compute_journal_entry_hash(parsed, prev)
                    out_line = json.dumps(parsed, ensure_ascii=False, default=str)
            except Exception:
                pass

        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(target), flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(out_line.rstrip("\n") + "\n")
        except Exception:
            raise
