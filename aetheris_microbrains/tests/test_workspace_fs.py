"""Unit tests for Split-Horizon Virtual Workspace (WorkspaceFS)."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from typing import List, Sequence

import pytest

from aetheris_microbrains.core import workspace_fs
from aetheris_microbrains.core.workspace_fs import (
    DockerExecWorkspaceFS,
    GitBlobWorkspaceFS,
    LocalWorkspaceFS,
    PayloadOverlayFS,
    WorkspaceProvider,
    WorkspaceReadResult,
    resolve_workspace_provider,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. LocalWorkspaceFS
# ---------------------------------------------------------------------------


def test_local_workspace_reads_contained_file_and_checks_exists(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target = src_dir / "app.py"
    content = "print('hello workspace')\n"
    target.write_text(content, encoding="utf-8", newline="\n")

    fs = LocalWorkspaceFS(tmp_path)
    assert isinstance(fs, WorkspaceProvider)
    assert fs.exists("src/app.py") is True
    assert fs.exists(str(target)) is True
    assert fs.exists("src/missing.py") is False
    assert fs.exists("src") is False  # directory, not file

    res = fs.read_text("src/app.py")
    assert isinstance(res, WorkspaceReadResult)
    assert res.ok is True
    assert res.content == content
    assert res.sha256 == _sha256(content)
    assert res.source == "local"
    assert res.error is None


def test_local_workspace_blocks_path_traversal_and_escape(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET", encoding="utf-8")

    fs = LocalWorkspaceFS(root)
    assert fs.exists("../secret.txt") is False
    assert fs.exists(str(secret)) is False

    res_rel = fs.read_text("../secret.txt")
    assert res_rel.ok is False
    assert res_rel.content is None
    assert "escapes workspace root" in (res_rel.error or "")

    res_abs = fs.read_text(str(secret))
    assert res_abs.ok is False
    assert res_abs.content is None
    assert "escapes workspace root" in (res_abs.error or "")


def test_local_workspace_enforces_max_bytes(tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 1024, encoding="utf-8")
    fs = LocalWorkspaceFS(tmp_path)

    res = fs.read_text("big.txt", max_bytes=100)
    assert res.ok is False
    assert res.content is None
    assert "max_bytes" in (res.error or "")


# ---------------------------------------------------------------------------
# 2. PayloadOverlayFS
# ---------------------------------------------------------------------------


def test_payload_overlay_serves_files_verifies_sha256_and_falls_back(tmp_path: Path):
    fallback_file = tmp_path / "on_disk.py"
    fallback_file.write_text("DISK = 1\n", encoding="utf-8", newline="\n")

    overlay_text = "OVERLAY = 2\n"
    overlay_hash = _sha256(overlay_text)

    fs = PayloadOverlayFS(
        {"./src/main.py": overlay_text},
        expected_hashes={"src/main.py": overlay_hash.upper()},
        fallback=LocalWorkspaceFS(tmp_path),
    )

    assert fs.exists("src/main.py") is True
    assert fs.exists("on_disk.py") is True
    assert fs.exists("nope.py") is False

    res = fs.read_text("src/main.py")
    assert res.ok is True
    assert res.content == overlay_text
    assert res.sha256 == overlay_hash
    assert res.source == "payload_overlay"
    assert res.error is None

    fb_res = fs.read_text("on_disk.py")
    assert fb_res.ok is True
    assert fb_res.content == "DISK = 1\n"
    assert fb_res.source == "local"


def test_payload_overlay_rejects_sha256_mismatch_and_traversal():
    fs = PayloadOverlayFS(
        {"src/app.py": "tampered content"},
        expected_hashes={"src/app.py": "0" * 64},
    )

    res = fs.read_text("src/app.py")
    assert res.ok is False
    assert res.content is None
    assert res.sha256 is None
    assert "sha256 mismatch" in (res.error or "")

    trav = fs.read_text("../etc/passwd")
    assert trav.ok is False
    assert "traversal" in (trav.error or "")
    assert fs.exists("../etc/passwd") is False


# ---------------------------------------------------------------------------
# 3. GitBlobWorkspaceFS
# ---------------------------------------------------------------------------


def test_git_blob_workspace_reads_committed_blob_ignoring_dirty_worktree(tmp_path: Path):
    repo = tmp_path / "git_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)

    readme = repo / "README.md"
    committed_text = "# Committed Version\n"
    readme.write_text(committed_text, encoding="utf-8", newline="\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)

    # Mutate working tree without committing — GitBlobWorkspaceFS must still read HEAD blob
    readme.write_text("# Uncommitted Dirty Edit\n", encoding="utf-8")

    fs = GitBlobWorkspaceFS(repo, ref="HEAD")
    assert fs.exists("README.md") is True
    assert fs.exists("DOES_NOT_EXIST.md") is False
    assert fs.exists("../outside.txt") is False

    res = fs.read_text("README.md")
    assert res.ok is True
    assert res.content == committed_text
    assert res.sha256 == _sha256(committed_text)
    assert res.source == "git:HEAD"

    trav = fs.read_text("../outside.txt")
    assert trav.ok is False
    assert "traversal" in (trav.error or "")


def test_git_blob_workspace_rejects_unsafe_ref(tmp_path: Path):
    with pytest.raises(ValueError):
        GitBlobWorkspaceFS(tmp_path, ref="--upload-pack=evil")
    with pytest.raises(ValueError):
        GitBlobWorkspaceFS(tmp_path, ref="HEAD..main")


# ---------------------------------------------------------------------------
# 4. DockerExecWorkspaceFS
# ---------------------------------------------------------------------------


def test_docker_exec_workspace_reads_with_mock_runner():
    recorded_cmds: List[List[str]] = []

    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        cmd = list(argv)
        recorded_cmds.append(cmd)
        if cmd[4] == "test":
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        return subprocess.CompletedProcess(cmd, 0, b"container_val = 42\n", b"")

    fs = DockerExecWorkspaceFS("dev_box_01", workdir="/workspace", runner=fake_runner)
    assert fs.exists("pkg/mod.py") is True
    assert recorded_cmds[-1] == ["docker", "exec", "--", "dev_box_01", "test", "-f", "/workspace/pkg/mod.py"]

    res = fs.read_text("pkg/mod.py")
    assert res.ok is True
    assert res.content == "container_val = 42\n"
    assert res.sha256 == _sha256("container_val = 42\n")
    assert res.source == "docker:dev_box_01"
    assert recorded_cmds[-1] == ["docker", "exec", "--", "dev_box_01", "cat", "--", "/workspace/pkg/mod.py"]


def test_docker_exec_workspace_rejects_traversal_and_shell_metacharacters():
    called = False

    def should_not_run(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess(list(argv), 0, b"", b"")

    fs = DockerExecWorkspaceFS("dev_box_01", workdir="/workspace", runner=should_not_run)

    for bad_path in (
        "../etc/passwd",
        "/etc/passwd",
        "app.py; rm -rf /",
        "app.py | cat /etc/shadow",
        "$(whoami).py",
        "file`id`.py",
    ):
        res = fs.read_text(bad_path)
        assert res.ok is False, f"Expected rejection for {bad_path!r}"
        assert fs.exists(bad_path) is False

    assert called is False

    with pytest.raises(ValueError):
        DockerExecWorkspaceFS("bad;container")
    with pytest.raises(ValueError):
        DockerExecWorkspaceFS("dev_box_01", workdir="relative/dir")


# ---------------------------------------------------------------------------
# 5. resolve_workspace_provider dispatch
# ---------------------------------------------------------------------------


def test_resolve_workspace_provider_dispatch(tmp_path: Path):
    # 1) base_content -> PayloadOverlayFS (with base_sha256 verification)
    content = "x = 1\n"
    p_base = resolve_workspace_provider(
        {
            "file_path": "src/a.py",
            "base_content": content,
            "base_sha256": _sha256(content),
        },
        cwd=str(tmp_path),
    )
    assert isinstance(p_base, PayloadOverlayFS)
    res = p_base.read_text("src/a.py")
    assert res.ok is True
    assert res.content == content

    # 2) workspace_files -> PayloadOverlayFS
    p_files = resolve_workspace_provider(
        {"workspace_files": {"README.md": "hi"}},
        cwd=str(tmp_path),
    )
    assert isinstance(p_files, PayloadOverlayFS)
    assert p_files.read_text("README.md").content == "hi"

    # 3) workspace_container -> DockerExecWorkspaceFS
    p_docker = resolve_workspace_provider(
        {"workspace_container": "sandbox_ctr_1", "workspace_workdir": "/app"},
        cwd=str(tmp_path),
    )
    assert isinstance(p_docker, DockerExecWorkspaceFS)
    assert p_docker.container_id == "sandbox_ctr_1"
    assert p_docker.workdir == "/app"

    # 4) default -> LocalWorkspaceFS
    p_local = resolve_workspace_provider({}, cwd=str(tmp_path))
    assert isinstance(p_local, LocalWorkspaceFS)
    assert p_local.root_dir == tmp_path.resolve()


# ---------------------------------------------------------------------------
# 6. Zero-NumPy stdlib-only isolation check
# ---------------------------------------------------------------------------


def test_workspace_fs_has_zero_numpy_imports():
    src_path = Path(workspace_fs.__file__).resolve()
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0].lower())
    assert "numpy" not in imported
