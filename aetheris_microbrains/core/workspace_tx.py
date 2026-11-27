"""Speculative workspace transaction manager with atomic rollback (stdlib-only).

Provides file-level snapshotting, path-escape confinement, atomic restoration,
newly-created-file cleanup, and automatic rollback on exceptions, failed
validators, or secret leakage detected via ``scrub_tool_output``.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from aetheris_microbrains.core.atomic import atomic_replace
from aetheris_microbrains.core.secrets import scrub_tool_output

DEFAULT_SKIP_DIRS: Tuple[str, ...] = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "target",
)


class WorkspaceTransactionError(RuntimeError):
    """Raised when a workspace transaction cannot be initialized or violates bounds."""


@dataclass(frozen=True)
class TransactionRecord:
    """Snapshot of a single path captured at transaction start."""

    path: Path
    existed: bool
    content: Optional[bytes] = None
    mtime_ns: Optional[int] = None
    mode: Optional[int] = None
    size: int = 0
    skipped_oversize: bool = False


class WorkspaceTransactionManager:
    """Manage a rollback-capable filesystem transaction confined to ``workspace_root``."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        target_paths: Optional[Sequence[Path | str]] = None,
        max_file_bytes: int = 2 * 1024 * 1024,
        skip_dirs: Sequence[str] = DEFAULT_SKIP_DIRS,
    ) -> None:
        if max_file_bytes <= 0:
            raise ValueError(f"max_file_bytes must be > 0, got {max_file_bytes}")
        self.workspace_root: Path = Path(workspace_root).resolve()
        self._raw_target_paths: Optional[Tuple[Path | str, ...]] = (
            tuple(target_paths) if target_paths is not None else None
        )
        self.max_file_bytes: int = int(max_file_bytes)
        self.skip_dirs: Tuple[str, ...] = tuple(skip_dirs)
        self._skip_dirs_set: Set[str] = set(self.skip_dirs)

        self._snapshots: Dict[Path, TransactionRecord] = {}
        self._initial_files: Set[Path] = set()
        self._initial_stats: Dict[Path, Tuple[int, int]] = {}
        self._initial_dirs: Set[Path] = set()
        self._active: bool = False
        self._committed: bool = False
        self._rolled_back: bool = False
        self._last_rollback_report: Optional[Dict[str, Any]] = None

    def _resolve_within_root(self, raw_path: Path | str) -> Path:
        """Resolve ``raw_path`` and ensure it stays inside ``self.workspace_root``."""
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise WorkspaceTransactionError(
                f"Path escapes workspace root {self.workspace_root}: {raw_path}"
            )
        return resolved

    def _rel_posix(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()

    def _walk_workspace(self) -> Tuple[Set[Path], Set[Path]]:
        """Return ``(files, dirs)`` under ``workspace_root`` excluding ``skip_dirs``."""
        files: Set[Path] = set()
        dirs: Set[Path] = set()
        for dirpath, dirnames, filenames in os.walk(self.workspace_root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in self._skip_dirs_set]
            cur_dir = Path(dirpath)
            for dname in dirnames:
                dpath = cur_dir / dname
                try:
                    d_resolved = dpath.resolve()
                except OSError:
                    continue
                if d_resolved.is_relative_to(self.workspace_root):
                    dirs.add(d_resolved)
            for fname in filenames:
                fpath = cur_dir / fname
                try:
                    if fpath.is_symlink():
                        resolved = fpath.resolve()
                        if not resolved.is_relative_to(self.workspace_root):
                            continue
                    else:
                        resolved = fpath.resolve()
                    if resolved.is_relative_to(self.workspace_root) and resolved.is_file():
                        files.add(resolved)
                except OSError:
                    continue
        return files, dirs

    def _snapshot_file(self, resolved: Path) -> TransactionRecord:
        """Capture existence, stat metadata, and bytes (up to ``max_file_bytes``) for ``resolved``."""
        if not resolved.exists():
            return TransactionRecord(path=resolved, existed=False)
        st = resolved.stat()
        size = int(st.st_size)
        if size > self.max_file_bytes:
            return TransactionRecord(
                path=resolved,
                existed=True,
                content=None,
                mtime_ns=int(st.st_mtime_ns),
                mode=int(st.st_mode),
                size=size,
                skipped_oversize=True,
            )
        data = resolved.read_bytes()
        return TransactionRecord(
            path=resolved,
            existed=True,
            content=data,
            mtime_ns=int(st.st_mtime_ns),
            mode=int(st.st_mode),
            size=size,
            skipped_oversize=False,
        )

    def _atomic_write_bytes(self, target: Path, data: bytes) -> None:
        """Write ``data`` to ``target`` via a sibling temp file and ``atomic_replace``."""
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".wtx_",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            atomic_replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def begin(self) -> None:
        """Snapshot workspace state and start the transaction."""
        if self._active:
            raise WorkspaceTransactionError("Transaction is already active")
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            raise WorkspaceTransactionError(
                f"Workspace root is not an existing directory: {self.workspace_root}"
            )

        self._snapshots.clear()
        self._initial_files.clear()
        self._initial_stats.clear()
        self._initial_dirs.clear()
        self._committed = False
        self._rolled_back = False
        self._last_rollback_report = None

        initial_files, initial_dirs = self._walk_workspace()
        self._initial_files = initial_files
        self._initial_dirs = initial_dirs
        for fpath in initial_files:
            try:
                st = fpath.stat()
                self._initial_stats[fpath] = (int(st.st_mtime_ns), int(st.st_size))
            except OSError:
                continue

        if self._raw_target_paths is not None:
            for raw_target in self._raw_target_paths:
                resolved = self._resolve_within_root(raw_target)
                if resolved == self.workspace_root:
                    for fpath in sorted(initial_files):
                        self._snapshots[fpath] = self._snapshot_file(fpath)
                elif resolved.is_dir():
                    for dirpath, dirnames, filenames in os.walk(resolved, followlinks=False):
                        dirnames[:] = [d for d in dirnames if d not in self._skip_dirs_set]
                        for fname in filenames:
                            child = self._resolve_within_root(Path(dirpath) / fname)
                            if child.is_file():
                                self._snapshots[child] = self._snapshot_file(child)
                else:
                    self._snapshots[resolved] = self._snapshot_file(resolved)
        else:
            for fpath in sorted(initial_files):
                self._snapshots[fpath] = self._snapshot_file(fpath)

        self._active = True

    def _changed_files(self) -> List[Path]:
        """Return currently existing files that were modified or created since ``begin()``."""
        if not self._active:
            return []
        current_files, _dirs = self._walk_workspace()
        changed: List[Path] = []
        seen: Set[Path] = set()

        for fpath in sorted(current_files):
            if fpath not in self._initial_files:
                changed.append(fpath)
                seen.add(fpath)
                continue
            if fpath in self._snapshots:
                rec = self._snapshots[fpath]
                if not rec.existed:
                    changed.append(fpath)
                    seen.add(fpath)
                elif rec.content is not None:
                    try:
                        if fpath.read_bytes() != rec.content:
                            changed.append(fpath)
                            seen.add(fpath)
                    except OSError:
                        pass
                continue
            prev_stat = self._initial_stats.get(fpath)
            if prev_stat is not None:
                try:
                    st = fpath.stat()
                    if (int(st.st_mtime_ns), int(st.st_size)) != prev_stat:
                        changed.append(fpath)
                        seen.add(fpath)
                except OSError:
                    pass

        for spath, rec in sorted(self._snapshots.items()):
            if spath in seen or not spath.exists() or not spath.is_file():
                continue
            if not rec.existed or (rec.content is not None and spath.read_bytes() != rec.content):
                changed.append(spath)
                seen.add(spath)

        return changed

    def rollback(self, reason: str = "") -> Dict[str, Any]:
        """Restore modified/deleted files atomically and remove newly created files."""
        if self._rolled_back and self._last_rollback_report is not None:
            return dict(self._last_rollback_report)
        if self._committed:
            raise WorkspaceTransactionError("Cannot rollback: transaction already committed")
        if not self._active:
            raise WorkspaceTransactionError("Cannot rollback: transaction has not begun")

        restored_files: List[str] = []
        deleted_created_files: List[str] = []
        deleted_set: Set[Path] = set()

        for path, rec in sorted(self._snapshots.items()):
            if not rec.existed:
                if path.exists() or path.is_symlink():
                    path.unlink(missing_ok=True)
                    deleted_set.add(path)
                    deleted_created_files.append(self._rel_posix(path))
                continue

            if rec.skipped_oversize or rec.content is None:
                continue

            needs_restore = False
            if not path.exists():
                needs_restore = True
            else:
                try:
                    if path.read_bytes() != rec.content:
                        needs_restore = True
                except OSError:
                    needs_restore = True

            if needs_restore:
                self._atomic_write_bytes(path, rec.content)
                try:
                    if rec.mode is not None:
                        os.chmod(path, stat.S_IMODE(rec.mode))
                    if rec.mtime_ns is not None:
                        os.utime(path, ns=(rec.mtime_ns, rec.mtime_ns))
                except OSError:
                    pass
                restored_files.append(self._rel_posix(path))

        current_files, current_dirs = self._walk_workspace()
        for fpath in sorted(current_files):
            if fpath not in self._initial_files and fpath not in deleted_set:
                fpath.unlink(missing_ok=True)
                deleted_set.add(fpath)
                deleted_created_files.append(self._rel_posix(fpath))

        new_dirs = sorted(
            (d for d in current_dirs if d not in self._initial_dirs),
            key=lambda p: len(p.parts),
            reverse=True,
        )
        for dpath in new_dirs:
            try:
                dpath.rmdir()
            except OSError:
                pass

        report: Dict[str, Any] = {
            "rolled_back": True,
            "reason": reason,
            "restored_files": sorted(restored_files),
            "deleted_created_files": sorted(deleted_created_files),
        }

        self._snapshots.clear()
        self._initial_files.clear()
        self._initial_stats.clear()
        self._initial_dirs.clear()
        self._active = False
        self._rolled_back = True
        self._last_rollback_report = report
        return report

    def commit(self) -> Dict[str, Any]:
        """Clear transaction snapshots and mark the workspace changes committed."""
        if not self._active:
            raise WorkspaceTransactionError("Cannot commit: transaction is not active")
        snapshot_count = len(self._snapshots)
        self._snapshots.clear()
        self._initial_files.clear()
        self._initial_stats.clear()
        self._initial_dirs.clear()
        self._active = False
        self._committed = True
        return {
            "committed": True,
            "rolled_back": False,
            "snapshot_count": snapshot_count,
        }

    def __enter__(self) -> "WorkspaceTransactionManager":
        self.begin()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if not self._active or self._committed or self._rolled_back:
            return False
        if exc_type is not None:
            self.rollback(reason=f"exception:{exc_type.__name__}")
            return False
        self.commit()
        return False


def run_speculative_transaction(
    workspace_root: Path | str,
    fn: Callable[[], Any],
    *,
    target_paths: Optional[Sequence[Path | str]] = None,
    validator: Optional[Callable[[Any], Any]] = None,
    rollback_on_secret: bool = True,
    max_file_bytes: int = 2 * 1024 * 1024,
    skip_dirs: Sequence[str] = DEFAULT_SKIP_DIRS,
    re_raise: bool = False,
) -> Dict[str, Any]:
    """Execute ``fn()`` inside a workspace transaction with validation and secret scanning.

    Automatically rolls back if ``fn()`` raises an exception, if ``validator(result) is False``,
    or if ``rollback_on_secret`` is enabled and ``scrub_tool_output`` detects secrets in
    ``str(result)`` or in any modified or newly created workspace file.
    """
    tx = WorkspaceTransactionManager(
        workspace_root,
        target_paths=target_paths,
        max_file_bytes=max_file_bytes,
        skip_dirs=skip_dirs,
    )
    tx.begin()

    try:
        result = fn()
    except Exception as exc:
        rb = tx.rollback(reason=f"exception:{type(exc).__name__}")
        if re_raise:
            raise
        redacted_err = scrub_tool_output(str(exc))["redacted_text"]
        return {
            "committed": False,
            "rolled_back": True,
            "reason": rb["reason"],
            "error": f"{type(exc).__name__}: {redacted_err}",
            "result": None,
            "restored_files": rb["restored_files"],
            "deleted_created_files": rb["deleted_created_files"],
        }

    if rollback_on_secret:
        if result is not None:
            res_scan = scrub_tool_output(str(result))
            if res_scan["secrets_found"]:
                rb = tx.rollback(reason="secret_in_result")
                return {
                    "committed": False,
                    "rolled_back": True,
                    "reason": rb["reason"],
                    "error": None,
                    "secret_types": res_scan["secret_types"],
                    "result": None,
                    "restored_files": rb["restored_files"],
                    "deleted_created_files": rb["deleted_created_files"],
                }

        for changed_path in tx._changed_files():
            try:
                raw_size = changed_path.stat().st_size
                if raw_size > max_file_bytes:
                    continue
                text = changed_path.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
            file_scan = scrub_tool_output(text)
            if file_scan["secrets_found"]:
                rel_path = tx._rel_posix(changed_path)
                rb = tx.rollback(reason="secret_in_file")
                return {
                    "committed": False,
                    "rolled_back": True,
                    "reason": rb["reason"],
                    "error": None,
                    "secret_file": rel_path,
                    "secret_types": file_scan["secret_types"],
                    "result": None,
                    "restored_files": rb["restored_files"],
                    "deleted_created_files": rb["deleted_created_files"],
                }

    if validator is not None:
        try:
            verdict = validator(result)
        except Exception as exc:
            rb = tx.rollback(reason=f"validator_exception:{type(exc).__name__}")
            if re_raise:
                raise
            redacted_err = scrub_tool_output(str(exc))["redacted_text"]
            return {
                "committed": False,
                "rolled_back": True,
                "reason": rb["reason"],
                "error": f"{type(exc).__name__}: {redacted_err}",
                "result": None,
                "restored_files": rb["restored_files"],
                "deleted_created_files": rb["deleted_created_files"],
            }
        if verdict is False:
            rb = tx.rollback(reason="validation_failed")
            return {
                "committed": False,
                "rolled_back": True,
                "reason": rb["reason"],
                "error": None,
                "result": result,
                "restored_files": rb["restored_files"],
                "deleted_created_files": rb["deleted_created_files"],
            }

    commit_report = tx.commit()
    return {
        "committed": True,
        "rolled_back": False,
        "reason": "",
        "error": None,
        "result": result,
        "snapshot_count": commit_report["snapshot_count"],
    }


__all__ = [
    "DEFAULT_SKIP_DIRS",
    "TransactionRecord",
    "WorkspaceTransactionError",
    "WorkspaceTransactionManager",
    "run_speculative_transaction",
]
