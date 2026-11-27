"""Tests for speculative workspace transactions and rollback guarantees."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from aetheris_microbrains.core.workspace_tx import (
    TransactionRecord,
    WorkspaceTransactionError,
    WorkspaceTransactionManager,
    run_speculative_transaction,
)


def _make_fake_stripe_token() -> str:
    return "_".join(("sk", "live", "AbCdEfGhIjKlMnOpQrStUvWx123456"))


def test_modified_file_restored_verbatim_on_rollback(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    original_bytes = b"def greet():\r\n    return 'hello'\r\n\x00"
    target.write_bytes(original_bytes)

    tx = WorkspaceTransactionManager(tmp_path)
    tx.begin()
    assert isinstance(tx._snapshots[target.resolve()], TransactionRecord)

    target.write_bytes(b"corrupted content")
    report = tx.rollback(reason="manual_abort")

    assert report["rolled_back"] is True
    assert report["reason"] == "manual_abort"
    assert "src/app.py" in report["restored_files"]
    assert report["deleted_created_files"] == []
    assert target.read_bytes() == original_bytes


def test_deleted_file_restored_verbatim_on_rollback(tmp_path: Path) -> None:
    target = tmp_path / "data" / "config.bin"
    target.parent.mkdir(parents=True)
    original_bytes = bytes(range(256))
    target.write_bytes(original_bytes)

    tx = WorkspaceTransactionManager(tmp_path)
    tx.begin()
    target.unlink()
    assert not target.exists()

    report = tx.rollback(reason="deleted_file")
    assert report["rolled_back"] is True
    assert "data/config.bin" in report["restored_files"]
    assert target.exists()
    assert target.read_bytes() == original_bytes


def test_newly_created_file_removed_on_rollback(tmp_path: Path) -> None:
    existing = tmp_path / "keep.txt"
    existing.write_text("keep", encoding="utf-8")

    tx = WorkspaceTransactionManager(tmp_path, target_paths=["keep.txt"])
    tx.begin()

    created_root = tmp_path / "spurious.txt"
    created_nested = tmp_path / "new_dir" / "nested.txt"
    created_nested.parent.mkdir(parents=True)
    created_root.write_text("temp", encoding="utf-8")
    created_nested.write_text("temp2", encoding="utf-8")

    report = tx.rollback(reason="cleanup")
    assert report["rolled_back"] is True
    assert "spurious.txt" in report["deleted_created_files"]
    assert "new_dir/nested.txt" in report["deleted_created_files"]
    assert not created_root.exists()
    assert not created_nested.exists()
    assert not created_nested.parent.exists()
    assert existing.read_text(encoding="utf-8") == "keep"


def test_context_manager_rollback_on_exception_and_commit_on_success(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_a.write_text("v1", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        with WorkspaceTransactionManager(tmp_path) as tx:
            assert tx is not None
            file_a.write_text("v2", encoding="utf-8")
            (tmp_path / "extra.txt").write_text("gone", encoding="utf-8")
            raise RuntimeError("boom")

    assert file_a.read_text(encoding="utf-8") == "v1"
    assert not (tmp_path / "extra.txt").exists()

    with WorkspaceTransactionManager(tmp_path) as tx:
        file_a.write_text("v3", encoding="utf-8")
        commit_info = tx.commit()
        assert commit_info["committed"] is True

    assert file_a.read_text(encoding="utf-8") == "v3"


def test_speculative_tx_rollback_on_exception(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")

    def failing() -> None:
        target.write_text("x = 999\n", encoding="utf-8")
        (tmp_path / "scratch.txt").write_text("orphan", encoding="utf-8")
        raise ValueError("bad edit")

    outcome = run_speculative_transaction(tmp_path, failing)
    assert outcome["committed"] is False
    assert outcome["rolled_back"] is True
    assert outcome["reason"] == "exception:ValueError"
    assert "bad edit" in outcome["error"]
    assert target.read_text(encoding="utf-8") == "x = 1\n"
    assert not (tmp_path / "scratch.txt").exists()


def test_speculative_tx_rollback_on_failed_validator(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")

    def mutate() -> dict[str, int]:
        target.write_text("x = 2\n", encoding="utf-8")
        return {"exit_code": 1}

    outcome = run_speculative_transaction(
        tmp_path,
        mutate,
        validator=lambda res: res["exit_code"] == 0,
    )
    assert outcome["committed"] is False
    assert outcome["rolled_back"] is True
    assert outcome["reason"] == "validation_failed"
    assert "module.py" in outcome["restored_files"]
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_speculative_tx_rollback_on_leaked_secret_in_file_and_result(tmp_path: Path) -> None:
    target = tmp_path / "settings.py"
    target.write_text("KEY = ''\n", encoding="utf-8")
    token = _make_fake_stripe_token()

    def write_secret_to_file() -> str:
        target.write_text(f"KEY = '{token}'\n", encoding="utf-8")
        return "ok"

    file_outcome = run_speculative_transaction(tmp_path, write_secret_to_file)
    assert file_outcome["committed"] is False
    assert file_outcome["rolled_back"] is True
    assert file_outcome["reason"] == "secret_in_file"
    assert file_outcome["secret_file"] == "settings.py"
    assert "stripe_key" in file_outcome["secret_types"]
    assert target.read_text(encoding="utf-8") == "KEY = ''\n"

    def return_secret_in_result() -> str:
        (tmp_path / "notes.txt").write_text("clean", encoding="utf-8")
        return f"leaked={token}"

    res_outcome = run_speculative_transaction(tmp_path, return_secret_in_result)
    assert res_outcome["committed"] is False
    assert res_outcome["rolled_back"] is True
    assert res_outcome["reason"] == "secret_in_result"
    assert "stripe_key" in res_outcome["secret_types"]
    assert not (tmp_path / "notes.txt").exists()


def test_speculative_tx_clean_commit(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("before", encoding="utf-8")

    def succeed() -> str:
        target.write_text("after", encoding="utf-8")
        (tmp_path / "added.txt").write_text("new", encoding="utf-8")
        return "done"

    outcome = run_speculative_transaction(
        tmp_path,
        succeed,
        validator=lambda r: r == "done",
    )
    assert outcome["committed"] is True
    assert outcome["rolled_back"] is False
    assert outcome["result"] == "done"
    assert target.read_text(encoding="utf-8") == "after"
    assert (tmp_path / "added.txt").read_text(encoding="utf-8") == "new"


def test_path_escape_protection(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    tx_rel_escape = WorkspaceTransactionManager(workspace, target_paths=["../outside.txt"])
    with pytest.raises(WorkspaceTransactionError, match="escapes workspace root"):
        tx_rel_escape.begin()

    tx_abs_escape = WorkspaceTransactionManager(workspace, target_paths=[outside])
    with pytest.raises(WorkspaceTransactionError, match="escapes workspace root"):
        tx_abs_escape.begin()

    assert outside.read_text(encoding="utf-8") == "secret"


def test_zero_numpy_dependency() -> None:
    tx_path = Path(__file__).resolve().parents[1] / "core" / "workspace_tx.py"
    tree = ast.parse(tx_path.read_text(encoding="utf-8"), filename=str(tx_path))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "numpy" not in imported_roots

    repo_root = str(Path(__file__).resolve().parents[2])
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import aetheris_microbrains.core.workspace_tx; "
                "assert 'numpy' not in sys.modules, 'numpy was imported'"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
