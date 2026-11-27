"""Unit tests for Macaroon-style capability tokens and subagent taint propagation."""

from __future__ import annotations

from dataclasses import replace

from aetheris_microbrains.core.capability_tokens import (
    CapabilityCaveat,
    CapabilityToken,
    attenuate_capability,
    mint_root_capability,
    propagate_subagent_taint,
    verify_capability_and_authorize,
)
from aetheris_microbrains.core.mcp_security import SessionSecurityState


ROOT_SECRET = b"aetheris-swarm-root-hmac-key-2026"


def test_root_mint_and_two_hop_non_interactive_attenuation():
    root_caveat = CapabilityCaveat(
        allowed_tools=("read_file", "grep_search", "web_fetch"),
        read_globs=("docs/**", "src/**"),
        write_globs=(),
        allow_shell=False,
        expires_at=2000.0,
        depth_remaining=2,
    )
    root_tok = mint_root_capability(ROOT_SECRET, subject="orchestrator", caveat=root_caveat, now=1000.0)
    assert root_tok.subject == "orchestrator"
    assert len(root_tok.caveats) == 1

    # Hop 1: parent attenuates WITHOUT root_secret
    hop1_tok = attenuate_capability(
        root_tok,
        CapabilityCaveat(
            allowed_tools=("read_file", "grep_search"),
            read_globs=("src/**",),
            subagent_id="researcher-1",
            depth_remaining=1,
        ),
    )
    assert len(hop1_tok.caveats) == 2
    assert hop1_tok.signature != root_tok.signature

    # Hop 2: researcher-1 delegates further WITHOUT root_secret
    hop2_tok = attenuate_capability(
        hop1_tok,
        CapabilityCaveat(
            allowed_tools=("read_file",),
            read_globs=("src/core/*.py",),
            expires_at=1500.0,
            depth_remaining=0,
        ),
    )
    assert len(hop2_tok.caveats) == 3

    # Allowed within intersected scope
    res_ok = verify_capability_and_authorize(
        hop2_tok,
        ROOT_SECRET,
        tool_name="read_file",
        path="src/core/engine.py",
        now=1200.0,
    )
    assert res_ok == {
        "authorized": True,
        "decision": "allow",
        "reason": "authorized",
        "subject": "orchestrator",
        "chain_depth": 3,
    }

    # Blocked by hop-2 narrowing (outside src/core/*.py, even though hop-1 allowed src/**)
    res_narrow = verify_capability_and_authorize(
        hop2_tok,
        ROOT_SECRET,
        tool_name="read_file",
        path="src/other/file.py",
        now=1200.0,
    )
    assert res_narrow["authorized"] is False
    assert res_narrow["decision"] == "deny"

    # Serialization round-trip preserves exact HMAC validity
    restored = CapabilityToken.from_dict(hop2_tok.to_dict())
    res_roundtrip = verify_capability_and_authorize(
        restored,
        ROOT_SECRET,
        tool_name="read_file",
        path="src/core/engine.py",
        now=1200.0,
    )
    assert res_roundtrip["authorized"] is True


def test_tampering_or_stripping_intermediate_caveat_fails_hmac():
    c0 = CapabilityCaveat(allowed_tools=("read_file", "write_file"), read_globs=("docs/**",), write_globs=("docs/**",))
    c1 = CapabilityCaveat(allowed_tools=("read_file",), write_globs=())
    c2 = CapabilityCaveat(read_globs=("docs/public/*",))

    tok = mint_root_capability(ROOT_SECRET, subject="worker", caveat=c0, now=1000.0)
    tok = attenuate_capability(tok, c1)
    tok = attenuate_capability(tok, c2)

    # Sanity check: valid chain passes read inside docs/public/*
    assert verify_capability_and_authorize(tok, ROOT_SECRET, tool_name="read_file", path="docs/public/a.md")["authorized"]

    # 1. Strip intermediate caveat c1 while keeping final signature -> HMAC fails
    stripped = replace(tok, caveats=(c0, c2))
    res_stripped = verify_capability_and_authorize(stripped, ROOT_SECRET, tool_name="read_file", path="docs/public/a.md")
    assert res_stripped["authorized"] is False
    assert res_stripped["decision"] == "deny"
    assert res_stripped["reason"] == "invalid_hmac_signature"

    # 2. Tamper with an intermediate caveat -> HMAC fails
    tampered_c1 = CapabilityCaveat(allowed_tools=("read_file", "write_file"), write_globs=("docs/**",))
    tampered = replace(tok, caveats=(c0, tampered_c1, c2))
    res_tampered = verify_capability_and_authorize(tampered, ROOT_SECRET, tool_name="write_file", path="docs/public/a.md", is_write=True)
    assert res_tampered["authorized"] is False
    assert res_tampered["reason"] == "invalid_hmac_signature"

    # 3. Attempting to widen by appending a permissive caveat without tampering still cannot widen
    widened = attenuate_capability(
        tok,
        CapabilityCaveat(allowed_tools=("write_file",), write_globs=("*",), allow_shell=True),
    )
    res_widened = verify_capability_and_authorize(
        widened,
        ROOT_SECRET,
        tool_name="write_file",
        path="docs/public/a.md",
        is_write=True,
    )
    assert res_widened["authorized"] is False
    assert res_widened["decision"] == "deny"


def test_expiry_and_depth_enforcement_fail_closed():
    tok = mint_root_capability(
        ROOT_SECRET,
        subject="ephemeral-worker",
        caveat=CapabilityCaveat(allowed_tools=("read_file",), expires_at=1000.0, depth_remaining=1),
        now=900.0,
    )
    # Exactly at expires_at is allowed; strictly past expires_at is denied
    assert verify_capability_and_authorize(tok, ROOT_SECRET, tool_name="read_file", now=1000.0)["authorized"] is True

    expired = verify_capability_and_authorize(tok, ROOT_SECRET, tool_name="read_file", now=1000.001)
    assert expired["authorized"] is False
    assert expired["decision"] == "deny"
    assert expired["reason"] == "token_expired"

    # Attenuation cannot extend expiry past parent caveat
    extended = attenuate_capability(tok, CapabilityCaveat(expires_at=5000.0))
    still_expired = verify_capability_and_authorize(extended, ROOT_SECRET, tool_name="read_file", now=1500.0)
    assert still_expired["authorized"] is False
    assert still_expired["reason"] == "token_expired"

    # Depth remaining=1 allows 1 attenuation, denies 2nd attenuation
    hop1 = attenuate_capability(tok, CapabilityCaveat(read_globs=("src/*",)))
    assert verify_capability_and_authorize(hop1, ROOT_SECRET, tool_name="read_file", path="src/a.py", now=950.0)["authorized"] is True

    hop2 = attenuate_capability(hop1, CapabilityCaveat(read_globs=("src/a.py",)))
    too_deep = verify_capability_and_authorize(hop2, ROOT_SECRET, tool_name="read_file", path="src/a.py", now=950.0)
    assert too_deep["authorized"] is False
    assert too_deep["reason"] == "delegation_depth_exceeded"


def test_read_only_researcher_caveat_enforces_all_boundaries():
    researcher_caveat = CapabilityCaveat(
        allowed_tools=("read_file", "web_fetch", "use_secret"),
        read_globs=("docs/**/*.md", "aetheris_microbrains/core/*.py"),
        write_globs=(),
        allow_shell=False,
        allowed_secrets=("PUBLIC_READ_KEY",),
        allowed_domains=("docs.python.org", "*.arxiv.org"),
    )
    tok = mint_root_capability(ROOT_SECRET, subject="ro-researcher", caveat=researcher_caveat, now=1000.0)

    # Scoped read_globs allowed
    assert verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="read_file",
        path="docs/architecture.md",
    )["authorized"] is True
    assert verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="read_file",
        path="aetheris_microbrains/core/capability_tokens.py",
    )["authorized"] is True

    # Out-of-scope read + traversal blocked
    assert verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="read_file",
        path="docs/../.env",
    )["authorized"] is False

    # Writes blocked (even to paths inside read_globs)
    write_res = verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="read_file",
        path="docs/architecture.md",
        is_write=True,
    )
    assert write_res["authorized"] is False
    assert write_res["decision"] == "deny"

    # Shell execution blocked
    shell_res = verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="read_file",
        is_shell=True,
    )
    assert shell_res["authorized"] is False
    assert shell_res["reason"] == "shell_not_allowed"

    # Secret handles: approved allowed, unapproved denied
    assert verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="use_secret",
        secret_handle="PUBLIC_READ_KEY",
    )["authorized"] is True
    sec_res = verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="use_secret",
        secret_handle="PROD_DB_PASSWORD",
    )
    assert sec_res["authorized"] is False
    assert sec_res["reason"] == "secret_not_allowed:PROD_DB_PASSWORD"

    # Domains: approved exact + wildcard allowed, unapproved/suffix-spoof denied
    assert verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="web_fetch",
        domain="docs.python.org",
    )["authorized"] is True
    assert verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="web_fetch",
        domain="export.arxiv.org",
    )["authorized"] is True
    assert verify_capability_and_authorize(
        tok,
        ROOT_SECRET,
        tool_name="web_fetch",
        domain="evil-arxiv.org",
    )["authorized"] is False


def test_propagate_subagent_taint_merges_untrusted_web_and_secret_taint(tmp_path):
    parent_path = tmp_path / "parent_security.json"
    child_path = tmp_path / "child_security.json"

    parent = SessionSecurityState(state_path=parent_path)
    child = SessionSecurityState(state_path=child_path)

    # Child ingests untrusted web content and reads a protected secret file
    child.record_pre_tool("web_fetch", target_path="https://untrusted.example/page", is_protected=False)
    child.record_post_tool("read_file", secrets_found=True, is_protected_path=True)

    summary = propagate_subagent_taint(parent, child, subagent_label="researcher-1")

    assert summary["subagent_label"] == "researcher-1"
    assert summary["untrusted_external_ingested"] is True
    assert summary["sensitive_local_read"] is True
    assert summary["merged_events"] >= 2
    assert summary["merged_artifacts"] >= 2

    assert parent.untrusted_external_ingested is True
    assert parent.sensitive_local_read is True
    assert any(a.get("integrity") == "untrusted" for a in parent.taint_artifacts)
    assert any(a.get("confidentiality") == "secret" for a in parent.taint_artifacts)

    # Parent must now trigger the lethal-trifecta gate on egress/mutation
    gate = parent.evaluate_taint_gate("execute_shell", is_egress_or_mutation=True)
    assert gate["decision"] == "deny"
    assert gate["reason_code"] == "layer0_session_taint_lethal_trifecta"

    # Persisted to parent_path and reloadable
    reloaded = SessionSecurityState(state_path=parent_path)
    assert reloaded.untrusted_external_ingested is True
    assert reloaded.sensitive_local_read is True
    assert len(reloaded.taint_artifacts) == len(parent.taint_artifacts)
