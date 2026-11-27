"""Unit tests for capability_enforcer and hook_checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetheris_microbrains.core.capability_enforcer import (
    CAPABILITY_TOKEN_ENV_VAR,
    build_capability_sandbox_command,
    compile_sandbox_policy_from_token,
    create_capability_domain_proxy,
    enforce_capability_for_hook,
    extract_capability_token,
)
from aetheris_microbrains.core.capability_tokens import (
    CapabilityCaveat,
    CapabilityToken,
    attenuate_capability,
    mint_root_capability,
)
from aetheris_microbrains.core.hook_checkpoints import HookCheckpointStore

ROOT_SECRET = "cap-enforcer-unit-test-secret-key"


def _make_root_token(caveat: CapabilityCaveat | None = None) -> CapabilityToken:
    return mint_root_capability(ROOT_SECRET, subject="agent:primary", caveat=caveat, now=1_700_000_000.0)


# ---------------------------------------------------------------------------
# extract_capability_token
# ---------------------------------------------------------------------------


def test_extract_capability_token_from_payload_dict_and_json_and_instance() -> None:
    tok = _make_root_token(CapabilityCaveat(allowed_tools=("read_file",)))

    # From CapabilityToken instance
    assert extract_capability_token({"capability_token": tok}, env={}) == tok

    # From dict representation
    extracted_dict = extract_capability_token({"capability_token": tok.to_dict()}, env={})
    assert extracted_dict == tok

    # From JSON-encoded string in payload
    extracted_str = extract_capability_token({"capability_token": json.dumps(tok.to_dict())}, env={})
    assert extracted_str == tok


def test_extract_capability_token_from_env_and_missing_or_malformed() -> None:
    tok = _make_root_token(CapabilityCaveat(allow_shell=False))
    env = {CAPABILITY_TOKEN_ENV_VAR: json.dumps(tok.to_dict())}

    assert extract_capability_token({}, env=env) == tok
    assert extract_capability_token(None, env=env) == tok
    assert extract_capability_token({}, env={}) is None
    assert extract_capability_token({"capability_token": "{not-valid-json"}, env={}) is None
    assert extract_capability_token({}, env={CAPABILITY_TOKEN_ENV_VAR: "not-json"}) is None


# ---------------------------------------------------------------------------
# enforce_capability_for_hook
# ---------------------------------------------------------------------------


def test_enforce_capability_for_hook_not_enforced_when_absent() -> None:
    res = enforce_capability_for_hook(
        {"tool_input": {"path": "src/app.py"}},
        root_secret=ROOT_SECRET,
        tool_name="write_file",
        path="src/app.py",
        is_write=True,
        env={},
    )
    assert res["enforced"] is False
    assert res["allowed"] is True
    assert res["decision"] == "allow"
    assert res["reason"] == "no_capability_token"


def test_enforce_capability_for_hook_allow_and_deny() -> None:
    tok = _make_root_token(
        CapabilityCaveat(
            allowed_tools=("read_file", "write_file", "http_fetch"),
            read_globs=("src/**",),
            write_globs=("src/safe/**",),
            allow_shell=False,
            allowed_domains=("*.github.com",),
            allowed_secrets=("github_read_*",),
        )
    )
    payload = {"capability_token": tok.to_dict()}

    # Allowed write inside write_globs
    ok_write = enforce_capability_for_hook(
        payload,
        root_secret=ROOT_SECRET,
        tool_name="write_file",
        path="src/safe/module.py",
        is_write=True,
        env={},
    )
    assert ok_write["enforced"] is True
    assert ok_write["allowed"] is True
    assert ok_write["decision"] == "allow"

    # Denied write outside write_globs
    bad_write = enforce_capability_for_hook(
        payload,
        root_secret=ROOT_SECRET,
        tool_name="write_file",
        path="etc/passwd",
        is_write=True,
        env={},
    )
    assert bad_write["enforced"] is True
    assert bad_write["allowed"] is False
    assert bad_write["decision"] == "deny"
    assert "write_path_not_allowed" in bad_write["reason"]

    # Denied shell
    bad_shell = enforce_capability_for_hook(
        payload,
        root_secret=ROOT_SECRET,
        tool_name="read_file",
        is_shell=True,
        env={},
    )
    assert bad_shell["enforced"] is True
    assert bad_shell["allowed"] is False
    assert bad_shell["decision"] == "deny"
    assert bad_shell["reason"] == "shell_not_allowed"

    # Allowed domain + secret vs denied domain + secret
    ok_net = enforce_capability_for_hook(
        payload,
        root_secret=ROOT_SECRET,
        tool_name="http_fetch",
        domain="api.github.com",
        secret_handle="github_read_token",
        env={},
    )
    assert ok_net["allowed"] is True

    bad_domain = enforce_capability_for_hook(
        payload,
        root_secret=ROOT_SECRET,
        tool_name="http_fetch",
        domain="evil.example.com",
        env={},
    )
    assert bad_domain["allowed"] is False
    assert bad_domain["decision"] == "deny"

    bad_secret = enforce_capability_for_hook(
        payload,
        root_secret=ROOT_SECRET,
        tool_name="http_fetch",
        secret_handle="stripe_live_key",
        env={},
    )
    assert bad_secret["allowed"] is False
    assert bad_secret["decision"] == "deny"


def test_enforce_capability_for_hook_tampered_and_malformed_fail_closed() -> None:
    tok = _make_root_token(CapabilityCaveat(allow_shell=False))
    tampered = tok.to_dict()
    tampered["caveats"][0]["allow_shell"] = True  # tamper without re-signing

    res_tampered = enforce_capability_for_hook(
        {"capability_token": tampered},
        root_secret=ROOT_SECRET,
        tool_name="bash",
        is_shell=True,
        env={},
    )
    assert res_tampered["enforced"] is True
    assert res_tampered["allowed"] is False
    assert res_tampered["decision"] == "deny"
    assert res_tampered["reason"] == "invalid_hmac_signature"

    res_malformed = enforce_capability_for_hook(
        {"capability_token": "{broken-json"},
        root_secret=ROOT_SECRET,
        tool_name="read_file",
        env={},
    )
    assert res_malformed["enforced"] is True
    assert res_malformed["allowed"] is False
    assert res_malformed["decision"] == "deny"
    assert res_malformed["reason"] == "malformed_capability_token"


# ---------------------------------------------------------------------------
# compile_sandbox_policy_from_token & sandbox builders
# ---------------------------------------------------------------------------


def test_compile_sandbox_policy_monotonic_narrowing() -> None:
    root = _make_root_token(
        CapabilityCaveat(
            allowed_domains=("*.github.com", "api.openai.com", "pypi.org"),
            read_globs=("src/**", "tests/**", "docs/**"),
            write_globs=("src/**", "tests/**"),
            allow_shell=True,
            allowed_secrets=("github_*", "openai_*"),
            expires_at=2_000_000_000.0,
        )
    )
    child = attenuate_capability(
        root,
        CapabilityCaveat(
            allowed_domains=("api.github.com", "pypi.org", "attacker.com"),
            read_globs=("src/core/**", "secrets/**"),
            write_globs=("src/core/**", "docs/**"),
            allow_shell=False,
            allowed_secrets=("github_read", "aws_admin"),
            expires_at=1_800_000_000.0,
        ),
    )
    # Attempting to widen in grandchild must NOT re-expand anything
    grandchild = attenuate_capability(
        child,
        CapabilityCaveat(
            allowed_domains=("*.github.com", "attacker.com"),
            read_globs=("**",),
            write_globs=("src/core/safe.py", "etc/shadow"),
            allow_shell=True,
            allowed_secrets=("*",),
            expires_at=2_500_000_000.0,
        ),
    )

    policy = compile_sandbox_policy_from_token(grandchild)
    assert policy["allowed_domains"] == ["api.github.com"]
    assert policy["read_globs"] == ["src/core/**"]
    assert policy["write_globs"] == ["src/core/safe.py"]
    assert policy["allow_shell"] is False
    assert policy["allowed_secrets"] == ["github_read"]
    assert policy["deny_net"] is False
    assert policy["strip_secrets"] is True
    assert policy["expires_at"] == 1_800_000_000.0

    # Empty domain intersection must set deny_net=True
    disjoint = attenuate_capability(
        grandchild,
        CapabilityCaveat(allowed_domains=("only.other.org",)),
    )
    disjoint_policy = compile_sandbox_policy_from_token(disjoint)
    assert disjoint_policy["allowed_domains"] == []
    assert disjoint_policy["deny_net"] is True


def test_build_capability_sandbox_command_and_domain_proxy() -> None:
    no_shell = _make_root_token(CapabilityCaveat(allow_shell=False, allowed_domains=("api.github.com",)))
    blocked = build_capability_sandbox_command("echo hi", no_shell, sandbox_mode="srt")
    assert blocked["allowed"] is False
    assert blocked["reason"] == "shell_not_allowed"
    assert blocked["command"] is None

    with_shell = _make_root_token(CapabilityCaveat(allow_shell=True, allowed_domains=("api.github.com",)))
    compiled = build_capability_sandbox_command("echo hi", with_shell, sandbox_mode="srt")
    assert compiled["allowed"] is True
    assert compiled["command"] is not None
    assert "srt" in compiled["command"]

    proxy = create_capability_domain_proxy(with_shell)
    assert proxy is not None
    assert proxy.allowed_domains == ("api.github.com",)

    unrestricted = _make_root_token()
    assert create_capability_domain_proxy(unrestricted) is None


# ---------------------------------------------------------------------------
# HookCheckpointStore
# ---------------------------------------------------------------------------


def test_hook_checkpoint_snapshot_modify_create_and_rollback(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    existing_file = workspace / "config.txt"
    deleted_file = workspace / "keep_me.txt"
    new_file = workspace / "speculative_new.txt"

    existing_file.write_text("original-v1\n", encoding="utf-8")
    deleted_file.write_text("do-not-lose\n", encoding="utf-8")

    store = HookCheckpointStore(state_dir=state_dir, workspace_root=workspace)
    ckpt_id = store.begin_checkpoint(
        "sess-1",
        [existing_file, deleted_file, new_file],
        metadata={"tool": "edit_file"},
    )
    assert ckpt_id in store.list_checkpoints("sess-1")

    # Simulate speculative tool mutations across processes:
    # 1) modify existing file, 2) delete existing file, 3) create new file
    existing_file.write_text("corrupted-v2\n", encoding="utf-8")
    deleted_file.unlink()
    new_file.write_text("unwanted-artifact\n", encoding="utf-8")

    # Use a fresh store instance (simulating separate hook process) to rollback
    store_process_2 = HookCheckpointStore(state_dir=state_dir, workspace_root=workspace)
    rb = store_process_2.rollback_checkpoint("sess-1", ckpt_id, reason="post_tool_taint_detected")

    assert rb["ok"] is True
    assert rb["status"] == "rolled_back"
    assert rb["reason"] == "post_tool_taint_detected"
    assert existing_file.read_text(encoding="utf-8") == "original-v1\n"
    assert deleted_file.read_text(encoding="utf-8") == "do-not-lose\n"
    assert not new_file.exists()
    assert store_process_2.list_checkpoints("sess-1") == []


def test_hook_checkpoint_commit_clears_checkpoint(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")

    store = HookCheckpointStore(state_dir=state_dir, workspace_root=workspace)
    ckpt_id = store.begin_checkpoint("sess-commit", [target])
    target.write_text("x = 2\n", encoding="utf-8")

    commit = store.commit_checkpoint("sess-commit", ckpt_id)
    assert commit["ok"] is True
    assert commit["status"] == "committed"
    assert target.read_text(encoding="utf-8") == "x = 2\n"
    assert store.get_checkpoint("sess-commit", ckpt_id) is None
    assert store.list_checkpoints("sess-commit") == []

    # Second commit reports not_found cleanly
    again = store.commit_checkpoint("sess-commit", ckpt_id)
    assert again["ok"] is False
    assert again["status"] == "not_found"


def test_hook_checkpoint_path_traversal_protection(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("top-secret", encoding="utf-8")

    store = HookCheckpointStore(state_dir=state_dir, workspace_root=workspace)

    # Rejects target outside workspace_root
    with pytest.raises(ValueError, match="path_traversal_detected"):
        store.begin_checkpoint("sess-sec", [outside])

    with pytest.raises(ValueError, match="path_traversal_detected"):
        store.begin_checkpoint("sess-sec", [workspace / ".." / "outside_secret.txt"])

    # Rejects traversal in checkpoint_id
    with pytest.raises(ValueError, match="invalid_checkpoint_id"):
        store.rollback_checkpoint("sess-sec", "../../evil")

    with pytest.raises(ValueError, match="invalid_checkpoint_id"):
        store.commit_checkpoint("sess-sec", "../evil")

    # Rejects tampered manifest attempting to overwrite outside file on rollback
    inside = workspace / "legit.txt"
    inside.write_text("ok", encoding="utf-8")
    ckpt_id = store.begin_checkpoint("sess-sec", [inside])
    manifest_file = store._checkpoint_dir("sess-sec", ckpt_id) / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["entries"][0]["path"] = str(outside)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    rb = store.rollback_checkpoint("sess-sec", ckpt_id, reason="tamper_test")
    assert rb["ok"] is False
    assert any("path_traversal_detected" in err for err in rb["errors"])
    assert outside.read_text(encoding="utf-8") == "top-secret"
