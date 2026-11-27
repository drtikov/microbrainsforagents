"""Split-Horizon Virtual Workspace filesystem abstraction (stdlib-only).

Provides a unified read-only ``WorkspaceProvider`` interface so security gates
can inspect file contents across local working trees, hook payload overlays,
Git object databases, and running Docker containers without third-party
dependencies or shell interpolation.
"""

from __future__ import annotations

import abc
import hashlib
import hmac
import os
import posixpath
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

DEFAULT_MAX_BYTES: int = 512_000

_CONTAINER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9._/@^{}~-]+$")
_SHELL_META_RE = re.compile(r"[;&|$`()<>*?\[\]{}~!'\"\\#\x00-\x1f\x7f]")


@dataclass(frozen=True)
class WorkspaceReadResult:
    """Result of reading a file through a ``WorkspaceProvider``."""

    ok: bool
    content: Optional[str]
    sha256: Optional[str]
    source: str
    error: Optional[str] = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _validate_max_bytes(max_bytes: int) -> Optional[str]:
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        return f"max_bytes must be a positive integer, got {max_bytes!r}"
    return None


def _has_traversal_segment(raw_path: str) -> bool:
    normalized = raw_path.replace("\\", "/")
    return any(part == ".." for part in normalized.split("/"))


def _normalize_overlay_key(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _extract_command_result(result: Any) -> Tuple[int, bytes, bytes]:
    """Normalize a subprocess CompletedProcess or 3-tuple from a runner."""
    if isinstance(result, tuple) and len(result) == 3:
        rc, stdout, stderr = result
    else:
        rc = getattr(result, "returncode", 1)
        stdout = getattr(result, "stdout", b"")
        stderr = getattr(result, "stderr", b"")

    if isinstance(stdout, str):
        stdout_bytes = stdout.encode("utf-8")
    elif stdout is None:
        stdout_bytes = b""
    else:
        stdout_bytes = bytes(stdout)

    if isinstance(stderr, str):
        stderr_bytes = stderr.encode("utf-8")
    elif stderr is None:
        stderr_bytes = b""
    else:
        stderr_bytes = bytes(stderr)

    return int(rc), stdout_bytes, stderr_bytes


class WorkspaceProvider(abc.ABC):
    """Abstract read-only interface for split-horizon workspace file access."""

    @abc.abstractmethod
    def read_text(
        self,
        rel_or_abs_path: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> WorkspaceReadResult:
        """Read UTF-8 text from ``rel_or_abs_path`` within ``max_bytes``."""

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """Return True if ``path`` exists as a readable file in this workspace."""


class LocalWorkspaceFS(WorkspaceProvider):
    """Containment-checked local filesystem reader confined to ``root_dir``."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir: Path = Path(root_dir).resolve()

    def _resolve_contained(self, rel_or_abs_path: str) -> Tuple[Optional[Path], Optional[str]]:
        if not isinstance(rel_or_abs_path, str) or not rel_or_abs_path.strip():
            return None, "path must be a non-empty string"
        if "\x00" in rel_or_abs_path:
            return None, "path contains null byte"

        candidate = Path(rel_or_abs_path)
        if not candidate.is_absolute():
            candidate = self.root_dir / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            return None, f"failed to resolve path: {exc}"

        try:
            resolved.relative_to(self.root_dir)
        except ValueError:
            return None, f"path escapes workspace root {self.root_dir}: {rel_or_abs_path}"

        return resolved, None

    def read_text(
        self,
        rel_or_abs_path: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> WorkspaceReadResult:
        size_err = _validate_max_bytes(max_bytes)
        if size_err is not None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source="local",
                error=size_err,
            )

        resolved, err = self._resolve_contained(rel_or_abs_path)
        if err is not None or resolved is None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source="local",
                error=err or "invalid path",
            )

        if not resolved.exists() or not resolved.is_file():
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source="local",
                error=f"file not found: {rel_or_abs_path}",
            )

        try:
            st_size = resolved.stat().st_size
            if st_size > max_bytes:
                return WorkspaceReadResult(
                    ok=False,
                    content=None,
                    sha256=None,
                    source="local",
                    error=f"file exceeds max_bytes ({st_size} > {max_bytes})",
                )
            with resolved.open("rb") as fh:
                raw = fh.read(max_bytes + 1)
        except OSError as exc:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source="local",
                error=f"failed to read file: {exc}",
            )

        if len(raw) > max_bytes:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source="local",
                error=f"file exceeds max_bytes ({len(raw)} > {max_bytes})",
            )

        return WorkspaceReadResult(
            ok=True,
            content=raw.decode("utf-8", errors="replace"),
            sha256=_sha256_bytes(raw),
            source="local",
        )

    def exists(self, path: str) -> bool:
        resolved, err = self._resolve_contained(path)
        if err is not None or resolved is None:
            return False
        try:
            return resolved.is_file()
        except OSError:
            return False


class PayloadOverlayFS(WorkspaceProvider):
    """In-memory overlay serving hook-supplied ``base_content`` / ``workspace_files``.

    Optionally enforces SHA-256 integrity checks (e.g. ``base_sha256``) and
    delegates cache misses to a ``fallback`` provider.
    """

    def __init__(
        self,
        files_map: Mapping[str, str],
        expected_hashes: Optional[Mapping[str, str]] = None,
        fallback: Optional[WorkspaceProvider] = None,
    ) -> None:
        self._files: Dict[str, str] = {
            _normalize_overlay_key(str(k)): str(v)
            for k, v in files_map.items()
            if isinstance(k, str) and k.strip()
        }
        self._expected_hashes: Dict[str, str] = {}
        if expected_hashes is not None:
            for k, digest in expected_hashes.items():
                if isinstance(k, str) and k.strip() and isinstance(digest, str) and digest.strip():
                    self._expected_hashes[_normalize_overlay_key(k)] = digest.strip().lower()
        self.fallback: Optional[WorkspaceProvider] = fallback

    def _validate_query_path(self, raw_path: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None, "path must be a non-empty string"
        if "\x00" in raw_path:
            return None, "path contains null byte"
        if _has_traversal_segment(raw_path):
            return None, f"path traversal rejected: {raw_path}"
        return _normalize_overlay_key(raw_path), None

    def read_text(
        self,
        rel_or_abs_path: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> WorkspaceReadResult:
        size_err = _validate_max_bytes(max_bytes)
        if size_err is not None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source="payload_overlay",
                error=size_err,
            )

        norm_path, err = self._validate_query_path(rel_or_abs_path)
        if err is not None or norm_path is None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source="payload_overlay",
                error=err or "invalid path",
            )

        if norm_path in self._files:
            content = self._files[norm_path]
            raw = content.encode("utf-8")
            if len(raw) > max_bytes:
                return WorkspaceReadResult(
                    ok=False,
                    content=None,
                    sha256=None,
                    source="payload_overlay",
                    error=f"overlay file exceeds max_bytes ({len(raw)} > {max_bytes})",
                )
            actual_sha256 = _sha256_bytes(raw)
            expected = self._expected_hashes.get(norm_path)
            if expected is not None and not hmac.compare_digest(actual_sha256, expected):
                return WorkspaceReadResult(
                    ok=False,
                    content=None,
                    sha256=None,
                    source="payload_overlay",
                    error=(
                        f"sha256 mismatch for {norm_path}: "
                        f"expected {expected}, got {actual_sha256}"
                    ),
                )
            return WorkspaceReadResult(
                ok=True,
                content=content,
                sha256=actual_sha256,
                source="payload_overlay",
            )

        if self.fallback is not None:
            return self.fallback.read_text(rel_or_abs_path, max_bytes=max_bytes)

        return WorkspaceReadResult(
            ok=False,
            content=None,
            sha256=None,
            source="payload_overlay",
            error=f"path not found in payload overlay: {rel_or_abs_path}",
        )

    def exists(self, path: str) -> bool:
        norm_path, err = self._validate_query_path(path)
        if err is not None or norm_path is None:
            return False
        if norm_path in self._files:
            return True
        if self.fallback is not None:
            return self.fallback.exists(path)
        return False


class GitBlobWorkspaceFS(WorkspaceProvider):
    """Reads file blobs via ``git show <ref>:<rel_path>`` without checkout coupling."""

    def __init__(
        self,
        repo_dir: Path | str,
        ref: str = "HEAD",
        *,
        timeout_s: float = 10.0,
        runner: Optional[Callable[[Sequence[str]], Any]] = None,
    ) -> None:
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("git ref must be a non-empty string")
        cleaned_ref = ref.strip()
        if cleaned_ref.startswith("-") or ".." in cleaned_ref or not _GIT_REF_RE.match(cleaned_ref):
            raise ValueError(f"unsafe git ref: {ref!r}")

        self.repo_dir: Path = Path(repo_dir).resolve()
        self.ref: str = cleaned_ref
        self.timeout_s: float = float(timeout_s)
        self._runner = runner

    def _source_label(self) -> str:
        return f"git:{self.ref}"

    def _normalize_repo_rel_path(self, rel_or_abs_path: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(rel_or_abs_path, str) or not rel_or_abs_path.strip():
            return None, "path must be a non-empty string"
        if "\x00" in rel_or_abs_path or "\n" in rel_or_abs_path or "\r" in rel_or_abs_path:
            return None, "path contains disallowed control characters"
        if _has_traversal_segment(rel_or_abs_path):
            return None, f"path traversal rejected: {rel_or_abs_path}"

        candidate = Path(rel_or_abs_path)
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve()
                rel = resolved.relative_to(self.repo_dir)
            except (OSError, ValueError):
                return None, f"path escapes git repo {self.repo_dir}: {rel_or_abs_path}"
            rel_posix = rel.as_posix()
        else:
            rel_posix = rel_or_abs_path.replace("\\", "/").strip()
            while rel_posix.startswith("./"):
                rel_posix = rel_posix[2:]
            rel_posix = posixpath.normpath(rel_posix)
            if rel_posix in ("", ".") or rel_posix.startswith("/") or rel_posix == ".." or rel_posix.startswith("../"):
                return None, f"invalid relative git path: {rel_or_abs_path}"

        return rel_posix, None

    def _run_git(self, argv: Sequence[str]) -> Tuple[int, bytes, bytes]:
        if self._runner is not None:
            return _extract_command_result(self._runner(argv))
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            check=False,
            timeout=self.timeout_s,
        )
        return _extract_command_result(proc)

    def read_text(
        self,
        rel_or_abs_path: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> WorkspaceReadResult:
        size_err = _validate_max_bytes(max_bytes)
        if size_err is not None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=size_err,
            )

        rel_posix, err = self._normalize_repo_rel_path(rel_or_abs_path)
        if err is not None or rel_posix is None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=err or "invalid path",
            )

        argv = ["git", "-C", str(self.repo_dir), "show", f"{self.ref}:{rel_posix}"]
        try:
            rc, stdout, stderr = self._run_git(argv)
        except (OSError, subprocess.SubprocessError) as exc:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=f"git show failed: {exc}",
            )

        if rc != 0:
            detail = stderr.decode("utf-8", errors="replace").strip() or f"exit code {rc}"
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=f"git show failed for {self.ref}:{rel_posix}: {detail}",
            )

        if len(stdout) > max_bytes:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=f"git blob exceeds max_bytes ({len(stdout)} > {max_bytes})",
            )

        return WorkspaceReadResult(
            ok=True,
            content=stdout.decode("utf-8", errors="replace"),
            sha256=_sha256_bytes(stdout),
            source=self._source_label(),
        )

    def exists(self, path: str) -> bool:
        rel_posix, err = self._normalize_repo_rel_path(path)
        if err is not None or rel_posix is None:
            return False
        argv = ["git", "-C", str(self.repo_dir), "cat-file", "-e", f"{self.ref}:{rel_posix}"]
        try:
            rc, _, _ = self._run_git(argv)
        except (OSError, subprocess.SubprocessError):
            return False
        return rc == 0


class DockerExecWorkspaceFS(WorkspaceProvider):
    """Reads files from a running Docker container via safe argv-only ``docker exec``."""

    def __init__(
        self,
        container_id: str,
        workdir: str = "/workspace",
        runner: Optional[Callable[[Sequence[str]], Any]] = None,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        if not isinstance(container_id, str) or not _CONTAINER_ID_RE.match(container_id):
            raise ValueError(f"invalid docker container_id: {container_id!r}")
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise ValueError(f"docker workdir must be an absolute POSIX path: {workdir!r}")
        if _SHELL_META_RE.search(workdir) or _has_traversal_segment(workdir):
            raise ValueError(f"unsafe docker workdir: {workdir!r}")

        self.container_id: str = container_id
        self.workdir: str = posixpath.normpath(workdir)
        self._runner = runner
        self.timeout_s: float = float(timeout_s)

    def _source_label(self) -> str:
        return f"docker:{self.container_id}"

    def _resolve_container_path(self, rel_or_abs_path: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(rel_or_abs_path, str) or not rel_or_abs_path.strip():
            return None, "path must be a non-empty string"
        if _SHELL_META_RE.search(rel_or_abs_path):
            return None, f"shell metacharacters rejected in path: {rel_or_abs_path!r}"
        if _has_traversal_segment(rel_or_abs_path):
            return None, f"path traversal rejected: {rel_or_abs_path}"

        raw = rel_or_abs_path.strip()
        if raw.startswith("/"):
            target = posixpath.normpath(raw)
        else:
            target = posixpath.normpath(posixpath.join(self.workdir, raw))

        prefix = "/" if self.workdir == "/" else f"{self.workdir.rstrip('/')}/"
        if target != self.workdir and not target.startswith(prefix):
            return None, f"path escapes container workdir {self.workdir}: {rel_or_abs_path}"

        return target, None

    def _run_docker(self, argv: Sequence[str]) -> Tuple[int, bytes, bytes]:
        if self._runner is not None:
            return _extract_command_result(self._runner(argv))
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            check=False,
            timeout=self.timeout_s,
        )
        return _extract_command_result(proc)

    def read_text(
        self,
        rel_or_abs_path: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> WorkspaceReadResult:
        size_err = _validate_max_bytes(max_bytes)
        if size_err is not None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=size_err,
            )

        target_path, err = self._resolve_container_path(rel_or_abs_path)
        if err is not None or target_path is None:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=err or "invalid container path",
            )

        argv = ["docker", "exec", "--", self.container_id, "cat", "--", target_path]
        try:
            rc, stdout, stderr = self._run_docker(argv)
        except (OSError, subprocess.SubprocessError) as exc:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=f"docker exec failed: {exc}",
            )

        if rc != 0:
            detail = stderr.decode("utf-8", errors="replace").strip() or f"exit code {rc}"
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=f"docker exec read failed for {target_path}: {detail}",
            )

        if len(stdout) > max_bytes:
            return WorkspaceReadResult(
                ok=False,
                content=None,
                sha256=None,
                source=self._source_label(),
                error=f"container file exceeds max_bytes ({len(stdout)} > {max_bytes})",
            )

        return WorkspaceReadResult(
            ok=True,
            content=stdout.decode("utf-8", errors="replace"),
            sha256=_sha256_bytes(stdout),
            source=self._source_label(),
        )

    def exists(self, path: str) -> bool:
        target_path, err = self._resolve_container_path(path)
        if err is not None or target_path is None:
            return False
        argv = ["docker", "exec", "--", self.container_id, "test", "-f", target_path]
        try:
            rc, _, _ = self._run_docker(argv)
        except (OSError, subprocess.SubprocessError):
            return False
        return rc == 0


def _extract_payload_target_path(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("file_path", "path", "target_path"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, Mapping):
        for key in ("file_path", "path", "target_path"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def resolve_workspace_provider(
    payload: Mapping[str, Any],
    *,
    cwd: Optional[str] = None,
) -> WorkspaceProvider:
    """Resolve the appropriate ``WorkspaceProvider`` for a hook/gate payload.

    Dispatch priority:
    1. ``PayloadOverlayFS`` when ``base_content`` or ``workspace_files`` is supplied
       (with fallback to Docker / Git / Local).
    2. ``DockerExecWorkspaceFS`` when ``workspace_container`` is set.
    3. ``GitBlobWorkspaceFS`` when ``workspace_git_ref`` or ``git_ref`` is set.
    4. ``LocalWorkspaceFS`` rooted at ``cwd`` / ``payload['cwd']`` / current dir.
    """
    root_dir = (
        cwd
        or (payload.get("cwd") if isinstance(payload.get("cwd"), str) and payload.get("cwd") else None)
        or (
            payload.get("workspace_root")
            if isinstance(payload.get("workspace_root"), str) and payload.get("workspace_root")
            else None
        )
        or os.getcwd()
    )

    container = payload.get("workspace_container")
    git_ref = payload.get("workspace_git_ref") or payload.get("git_ref")

    base_provider: WorkspaceProvider
    if isinstance(container, str) and container.strip():
        workdir = (
            payload.get("workspace_workdir")
            or payload.get("container_workdir")
            or "/workspace"
        )
        base_provider = DockerExecWorkspaceFS(
            container.strip(),
            workdir=str(workdir),
        )
    elif isinstance(git_ref, str) and git_ref.strip():
        base_provider = GitBlobWorkspaceFS(root_dir, ref=git_ref.strip())
    else:
        base_provider = LocalWorkspaceFS(root_dir)

    workspace_files = payload.get("workspace_files")
    base_content = payload.get("base_content")
    has_workspace_files = isinstance(workspace_files, Mapping) and len(workspace_files) > 0
    has_base_content = isinstance(base_content, str)

    if has_workspace_files or has_base_content:
        files_map: Dict[str, str] = {}
        expected_hashes: Dict[str, str] = {}

        if has_workspace_files:
            for k, v in workspace_files.items():
                if isinstance(k, str) and isinstance(v, str):
                    files_map[k] = v

        raw_hashes = payload.get("expected_hashes") or payload.get("workspace_file_hashes")
        if isinstance(raw_hashes, Mapping):
            for k, v in raw_hashes.items():
                if isinstance(k, str) and isinstance(v, str):
                    expected_hashes[k] = v

        if has_base_content:
            target_path = _extract_payload_target_path(payload) or "__base_content__"
            files_map[target_path] = base_content
            base_sha256 = payload.get("base_sha256")
            if isinstance(base_sha256, str) and base_sha256.strip():
                expected_hashes[target_path] = base_sha256.strip()

        return PayloadOverlayFS(
            files_map,
            expected_hashes=expected_hashes or None,
            fallback=base_provider,
        )

    return base_provider
