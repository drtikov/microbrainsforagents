"""Tests for the post-policy-change verification chain (`aetheris_microbrains.verify`)."""

from __future__ import annotations

import json

import pytest

from aetheris_microbrains import verify


def _step(step: str, status: str, detail: str = "") -> dict:
    return {"step": step, "status": status, "detail": detail}


class TestChainExitCode:
    def test_all_ok_is_zero(self):
        results = [_step("self-test", "ok"), _step("doctor", "ok"), _step("shadow-report", "ok")]
        assert verify.chain_exit_code(results) == 0

    def test_skipped_steps_do_not_fail(self):
        results = [_step("self-test", "skipped"), _step("doctor", "ok"), _step("shadow-report", "ok")]
        assert verify.chain_exit_code(results) == 0

    def test_wiring_failure_is_one(self):
        results = [_step("self-test", "ok"), _step("doctor", "FAIL", "settings: no settings file")]
        assert verify.chain_exit_code(results) == 1

    def test_unsafe_policy_is_two(self):
        results = [_step("doctor", "ok"), _step("eval-policy", "FAIL", "promote_safe=False")]
        assert verify.chain_exit_code(results) == 2

    def test_unsafe_policy_outranks_wiring_failure(self):
        results = [_step("doctor", "FAIL", "hooks"), _step("eval-policy", "FAIL", "promote_safe=False")]
        assert verify.chain_exit_code(results) == 2


class TestRenderResults:
    def test_passed_verdict(self):
        out = verify.render_results([_step("doctor", "ok", "13 checks"), _step("shadow-report", "ok")])
        assert "VERIFY PASSED" in out
        assert "[ ok ] doctor" in out

    def test_failed_verdict(self):
        out = verify.render_results([_step("doctor", "FAIL", "settings: bad json")])
        assert "VERIFY FAILED" in out
        assert "[FAIL] doctor" in out
        assert "settings: bad json" in out

    def test_unsafe_policy_verdict_is_specific(self):
        out = verify.render_results([_step("eval-policy", "FAIL", "promote_safe=False")])
        assert "candidate policy not promotion-safe" in out


class TestInstallerSelfTest:
    def test_skips_without_installer(self, tmp_path):
        result = verify.run_installer_self_test(tmp_path)
        assert result["status"] == "skipped"
        assert "installer not found" in result["detail"]

    def test_fails_on_missing_installer_argument_order(self, tmp_path):
        # A directory at the installer path is not a usable installer either.
        fake = tmp_path / ".claude" / "skills" / "microbrains" / "install_hook.py"
        fake.parent.mkdir(parents=True)
        fake.mkdir()
        result = verify.run_installer_self_test(tmp_path)
        assert result["status"] == "skipped"


class TestEvalPolicyStep:
    def test_missing_policy_file_fails_closed(self, tmp_path):
        result = verify.run_eval_policy_step(str(tmp_path / "nope.toml"))
        assert result["status"] == "FAIL"
        assert "nope.toml" in result["detail"]


class TestMainJson:
    @pytest.fixture()
    def _stub_steps(self, monkeypatch):
        """Stub env-dependent steps so the chain is hermetic in CI."""
        monkeypatch.setattr(
            verify, "run_doctor_step", lambda _root: _step("doctor", "ok", "13 checks (0 failed, 0 warning(s))")
        )
        monkeypatch.setattr(
            verify,
            "run_shadow_report_step",
            lambda journal_path="", verbose=True: _step("shadow-report", "ok", "0 shadow event(s)"),
        )

    def test_json_output_skips_reports_and_reports_zero(self, _stub_steps, capsys):
        rc = verify.main(["--json", "--skip-self-test"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["exit_code"] == 0
        steps = [r["step"] for r in payload["results"]]
        assert steps == ["self-test", "doctor", "shadow-report"]
        assert payload["results"][0]["status"] == "skipped"
        for r in payload["results"]:
            assert "doctor_report" not in r
            assert "eval_report" not in r

    def test_doctor_failure_propagates_to_exit_code(self, monkeypatch, capsys):
        monkeypatch.setattr(
            verify, "run_doctor_step", lambda _root: _step("doctor", "FAIL", "settings: no settings file")
        )
        monkeypatch.setattr(
            verify,
            "run_shadow_report_step",
            lambda journal_path="", verbose=True: _step("shadow-report", "ok", "0 shadow event(s)"),
        )
        rc = verify.main(["--json", "--skip-self-test"])
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["exit_code"] == 1
