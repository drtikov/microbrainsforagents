"""Cross-hook speculative workspace checkpoints with transactional rollback (stdlib-only).

Provides `HookCheckpointStore`, a cross-process persistent checkpoint store rooted
under `get_state_dir()` (or an explicit `state_dir`). Before a mutating hook/tool
executes, `begin_checkpoint(session_id, paths)` snapshots existing files into
content-addressed blobs and records non-existent paths as newly created targets.
After execution:
- `commit_checkpoint(session_id, checkpoint_id)` discards the checkpoint once verified.
- `rollback_checkpoint(session_id, checkpoint_id, reason=...)` restores modified or
  deleted files from their verified snapshots and removes newly created files, with
  strict path-traversal protection on both checkpoint storage and workspace paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from aetheris_microbrains.core.runtime_paths import file_lock, get_state_dir

_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_BLOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _session_dir_key(session_id: str) -> str:
    clean = str(session_id or "").strip()
    if not clean:
        raise ValueError("session_id must not be empty")
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:32]


def _validate_checkpoint_id(checkpoint_id: str) -> str:
    clean = str(checkpoint_id or "").strip()
    if not clean or not _CHECKPOINT_ID_RE.match(clean):
        raise ValueError(f"invalid_checkpoint_id:{checkpoint_id!r}")
    return clean


def _validate_blob_name(blob_name: str) -> str:
    clean = str(blob_name or "").strip()
    if not clean or clean in (".", "..") or "/" in clean or "\\" in clean or not _BLOB_NAME_RE.match(clean):
        raise ValueError(f"invalid_blob_name:{blob_name!r}")
    return clean


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_target_path(
    raw_path: Union[str, os.PathLike[str]],
    *,
    workspace_root: Optional[Path] = None,
) -> Path:
    text = os.fspath(raw_path)
    if not isinstance(text, str):
        text = text.decode("utf-8", errors="strict")
    if not text or not text.strip():
        raise ValueError("checkpoint path must not be empty")
    if "\x00" in text:
        raise ValueError("path_traversal_detected:null_byte")

    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        base = workspace_root if workspace_root is not None else Path.cwd()
        candidate = base / candidate

    resolved = candidate.resolve()
    if workspace_root is not None:
        resolved_root = workspace_root.resolve()
        if not _is_relative_to(resolved, resolved_root):
            raise ValueError(f"path_traversal_detected:outside_workspace_root:{text}")
    return resolved


def _atomic_write_json(target: Path, data: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    payload = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, default=str)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp_path), str(target))


def _atomic_restore_file(src_blob: Path, dst_path: Path, mode: Optional[int] = None) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(f".{dst_path.name}.restore.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        shutil.copyfile(str(src_blob), str(tmp_path))
        if mode is not None:
            try:
                os.chmod(tmp_path, int(mode))
            except OSError:
                pass
        os.replace(str(tmp_path), str(dst_path))
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


class HookCheckpointStore:
    """Persistent cross-process checkpoint store for speculative hook/tool mutations."""

    def __init__(
        self,
        state_dir: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        workspace_root: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> None:
        base = Path(state_dir).expanduser() if state_dir is not None else get_state_dir()
        self.state_dir: Path = base.resolve()
        self.checkpoints_root: Path = self.state_dir / "hook_checkpoints"
        self.workspace_root: Optional[Path] = (
            Path(workspace_root).expanduser().resolve() if workspace_root is not None else None
        )

    def _session_dir(self, session_id: str) -> Path:
        return self.checkpoints_root / _session_dir_key(session_id)

    def _checkpoint_dir(self, session_id: str, checkpoint_id: str) -> Path:
        clean_ckpt = _validate_checkpoint_id(checkpoint_id)
        session_dir = self._session_dir(session_id).resolve()
        ckpt_dir = (session_dir / clean_ckpt).resolve()
        if not _is_relative_to(ckpt_dir, session_dir) or ckpt_dir == session_dir:
            raise ValueError(f"path_traversal_detected:checkpoint_id:{checkpoint_id!r}")
        return ckpt_dir

    def _lock_target(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "checkpoints.lock"

    def begin_checkpoint(
        self,
        session_id: str,
        paths: Iterable[Union[str, os.PathLike[str]]],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        workspace_root: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> str:
        """Snapshot existing files and record non-existent paths as newly created targets.

        Returns the newly created `checkpoint_id`.
        """
        if isinstance(paths, (str, bytes, os.PathLike)):
            raise TypeError("paths must be an iterable of path-like objects, not a single path")

        clean_session = str(session_id or "").strip()
        if not clean_session:
            raise ValueError("session_id must not be empty")

        effective_root: Optional[Path] = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else self.workspace_root
        )

        resolved_paths: List[Path] = []
        seen_paths = set()
        for raw in paths:
            resolved = _resolve_target_path(raw, workspace_root=effective_root)
            key = str(resolved)
            if key not in seen_paths:
                seen_paths.add(key)
                resolved_paths.append(resolved)

        checkpoint_id = f"ckpt_{int(time.time() * 1000)}_{secrets.token_hex(8)}"
        ckpt_dir = self._checkpoint_dir(clean_session, checkpoint_id)
        blobs_dir = ckpt_dir / "blobs"

        with file_lock(self._lock_target(clean_session)):
            blobs_dir.mkdir(parents=True, exist_ok=True)
            entries: List[Dict[str, Any]] = []
            try:
                for idx, target in enumerate(resolved_paths):
                    if target.exists():
                        if target.is_dir():
                            raise ValueError(f"cannot_checkpoint_directory:{target}")
                        digest = _sha256_file(target)
                        blob_name = f"entry_{idx}_{digest[:16]}.bin"
                        blob_path = (blobs_dir / blob_name).resolve()
                        if not _is_relative_to(blob_path, blobs_dir.resolve()):
                            raise ValueError(f"path_traversal_detected:blob:{blob_name}")
                        shutil.copyfile(str(target), str(blob_path))
                        st = target.stat()
                        entries.append(
                            {
                                "path": str(target),
                                "existed": True,
                                "blob_name": blob_name,
                                "sha256": digest,
                                "size": int(st.st_size),
                                "mode": int(stat.S_IMODE(st.st_mode)),
                            }
                        )
                    else:
                        entries.append(
                            {
                                "path": str(target),
                                "existed": False,
                                "blob_name": None,
                                "sha256": None,
                                "size": 0,
                                "mode": None,
                            }
                        )

                manifest: Dict[str, Any] = {
                    "version": 1,
                    "checkpoint_id": checkpoint_id,
                    "session_id": clean_session,
                    "created_at": time.time(),
                    "status": "active",
                    "workspace_root": str(effective_root) if effective_root is not None else None,
                    "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
                    "entries": entries,
                }
                _atomic_write_json(ckpt_dir / "manifest.json", manifest)
            except Exception:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                raise

        return checkpoint_id

    def get_checkpoint(self, session_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Read a checkpoint manifest if it exists, or return ``None``."""
        clean_session = str(session_id or "").strip()
        if not clean_session:
            raise ValueError("session_id must not be empty")
        ckpt_dir = self._checkpoint_dir(clean_session, checkpoint_id)
        manifest_path = ckpt_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or data.get("session_id") != clean_session:
            return None
        return data

    def list_checkpoints(self, session_id: str) -> List[str]:
        """List active checkpoint IDs for `session_id`."""
        clean_session = str(session_id or "").strip()
        if not clean_session:
            raise ValueError("session_id must not be empty")
        session_dir = self._session_dir(clean_session)
        if not session_dir.is_dir():
            return []
        ids: List[str] = []
        for child in sorted(session_dir.iterdir()):
            if child.is_dir() and (child / "manifest.json").is_file():
                ids.append(child.name)
        return ids

    def commit_checkpoint(self, session_id: str, checkpoint_id: str) -> Dict[str, Any]:
        """Commit a checkpoint, clearing its stored snapshots and manifest."""
        clean_session = str(session_id or "").strip()
        if not clean_session:
            raise ValueError("session_id must not be empty")
        ckpt_dir = self._checkpoint_dir(clean_session, checkpoint_id)
        manifest_path = ckpt_dir / "manifest.json"

        with file_lock(self._lock_target(clean_session)):
            if not manifest_path.is_file():
                return {
                    "ok": False,
                    "status": "not_found",
                    "session_id": clean_session,
                    "checkpoint_id": checkpoint_id,
                }
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if not isinstance(manifest, dict) or manifest.get("session_id") != clean_session:
                return {
                    "ok": False,
                    "status": "session_mismatch",
                    "session_id": clean_session,
                    "checkpoint_id": checkpoint_id,
                }
            paths = [str(entry.get("path", "")) for entry in manifest.get("entries", []) if isinstance(entry, dict)]
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            return {
                "ok": True,
                "status": "committed",
                "session_id": clean_session,
                "checkpoint_id": checkpoint_id,
                "paths": paths,
                "committed_at": time.time(),
            }

    def rollback_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        *,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Restore snapshotted files and delete newly created files, then clear the checkpoint."""
        clean_session = str(session_id or "").strip()
        if not clean_session:
            raise ValueError("session_id must not be empty")
        ckpt_dir = self._checkpoint_dir(clean_session, checkpoint_id)
        manifest_path = ckpt_dir / "manifest.json"
        blobs_dir = (ckpt_dir / "blobs").resolve()

        with file_lock(self._lock_target(clean_session)):
            if not manifest_path.is_file():
                return {
                    "ok": False,
                    "status": "not_found",
                    "session_id": clean_session,
                    "checkpoint_id": checkpoint_id,
                    "reason": str(reason or ""),
                    "restored": [],
                    "removed": [],
                    "errors": ["checkpoint_not_found"],
                }

            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if not isinstance(manifest, dict) or manifest.get("session_id") != clean_session:
                return {
                    "ok": False,
                    "status": "session_mismatch",
                    "session_id": clean_session,
                    "checkpoint_id": checkpoint_id,
                    "reason": str(reason or ""),
                    "restored": [],
                    "removed": [],
                    "errors": ["session_mismatch"],
                }

            manifest_root_raw = manifest.get("workspace_root")
            effective_root: Optional[Path] = self.workspace_root
            if manifest_root_raw:
                effective_root = Path(str(manifest_root_raw)).expanduser().resolve()

            restored: List[str] = []
            removed: List[str] = []
            errors: List[str] = []

            for entry in manifest.get("entries", []):
                if not isinstance(entry, dict):
                    errors.append("invalid_entry")
                    continue
                raw_target = str(entry.get("path") or "")
                if not raw_target or "\x00" in raw_target or ".." in Path(raw_target).parts:
                    errors.append(f"path_traversal_detected:{raw_target}")
                    continue

                try:
                    target = _resolve_target_path(raw_target, workspace_root=effective_root)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue

                existed = bool(entry.get("existed", False))
                if existed:
                    try:
                        blob_name = _validate_blob_name(str(entry.get("blob_name") or ""))
                    except ValueError as exc:
                        errors.append(str(exc))
                        continue
                    blob_path = (blobs_dir / blob_name).resolve()
                    if not _is_relative_to(blob_path, blobs_dir) or not blob_path.is_file():
                        errors.append(f"invalid_or_missing_blob:{blob_name}")
                        continue
                    expected_sha = str(entry.get("sha256") or "")
                    if expected_sha and _sha256_file(blob_path) != expected_sha:
                        errors.append(f"blob_integrity_mismatch:{blob_name}")
                        continue
                    mode = entry.get("mode")
                    _atomic_restore_file(blob_path, target, mode=int(mode) if mode is not None else None)
                    restored.append(str(target))
                else:
                    if target.is_symlink() or target.is_file():
                        target.unlink()
                        removed.append(str(target))
                    elif target.exists():
                        errors.append(f"refused_to_remove_non_file:{target}")

            if not errors:
                shutil.rmtree(ckpt_dir, ignore_errors=True)

            return {
                "ok": len(errors) == 0,
                "status": "rolled_back" if not errors else "partial_rollback",
                "session_id": clean_session,
                "checkpoint_id": checkpoint_id,
                "reason": str(reason or ""),
                "restored": restored,
                "removed": removed,
                "errors": errors,
                "rolled_back_at": time.time(),
            }
