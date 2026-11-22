"""Post-policy-change verification chain for the Aetheris pre-flight gate.

Chains the four checks an operator should re-run after any policy or
governance change into one command:

    python -m aetheris_microbrains.verify                      # wiring + shadow projection
    python -m aetheris_microbrains.verify --policy cand.toml   # also replay a candidate policy
    python -m aetheris_microbrains.verify --json               # machine-readable summary
    aetheris-verify --skip-self-test                           # console entrypoint

Steps:
  1. installer self-test  - `install_hook.py --self-test` (skipped when the
     installer is absent, e.g. wheel installs without a repo checkout)
  2. doctor               - in-process health check (settings/matchers/policy/
     journal/pins/daemon)
  3. eval-policy          - optional; replays the decision journal and builtin
     constitutional probes against a candidate PolicyConfig (only with --policy)
  4. shadow-report        - informational prompt-fatigue projection from
     shadow-mode journal events (never fails the chain)

Exit codes: 0 all steps passed or skipped, 1 a wiring step failed
(self-test/doctor), 2 the candidate policy is not promotion-safe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

INSTALLER_RELATIVE = Path(".claude") / "skills" / "microbrains" / "install_hook.py"

_OK = "ok"
_WARN = "warn"
_FAIL = "FAIL"
_SKIP = "skipped"


def _reconfigure_stdout() -> None:
    # Windows consoles default to cp1252 and crash on box-drawing characters;
    # force UTF-8 with replacement instead of a hard failure (mirrors audit.py).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def run_installer_self_test(project_dir: Path, timeout: float = 120.0) -> Dict[str, Any]:
    """Run the bundled installer's end-to-end self-test in a subprocess."""
    installer = project_dir / INSTALLER_RELATIVE
    if not installer.is_file():
        return {
            "step": "self-test",
            "status": _SKIP,
            "detail": f"installer not found at {installer} (needs a repo checkout)",
        }
    try:
        proc = subprocess.run(  # aetheris: allow subprocess_call:run -- verify chain spawns the repo installer CLI (read-only --self-test probe)
            [sys.executable, str(installer), "--self-test"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"step": "self-test", "status": _FAIL, "detail": f"timed out after {timeout:.0f}s"}
    except OSError as exc:
        return {"step": "self-test", "status": _FAIL, "detail": f"spawn failed: {exc}"}
    if proc.returncode != 0:
        tail = [ln for ln in (proc.stdout or proc.stderr or "").strip().splitlines() if ln][-3:]
        return {
            "step": "self-test",
            "status": _FAIL,
            "detail": " | ".join(tail) if tail else f"exit code {proc.returncode}",
        }
    return {"step": "self-test", "status": _OK, "detail": "probe verdicts & installed matchers verified"}


def run_doctor_step(project_dir: Path) -> Dict[str, Any]:
    """Run the in-process doctor health check."""
    from .doctor import run_doctor

    report = run_doctor(project_dir=project_dir)
    checks: List[Dict[str, Any]] = report.get("checks", [])
    failed = [c for c in checks if c.get("status") == _FAIL]
    warned = [c for c in checks if c.get("status") == _WARN]
    if failed or not report.get("ok", True):
        detail = "; ".join(f"{c.get('name')}: {c.get('detail')}".strip(": ") for c in failed)
        return {"step": "doctor", "status": _FAIL, "detail": detail or "doctor reported failure", "doctor_report": report}
    return {
        "step": "doctor",
        "status": _OK,
        "detail": f"{len(checks)} checks (0 failed, {len(warned)} warning(s))",
        "doctor_report": report,
    }


def run_eval_policy_step(
    policy_path: str, base_policy: str = "", journal_path: str = ""
) -> Dict[str, Any]:
    """Replay the journal & constitutional probes against a candidate PolicyConfig."""
    from .audit import get_journal_entries
    from .core.layered_decision_engine import PolicyConfig
    from .core.policy_governance import evaluate_policy_candidate

    try:
        cand = PolicyConfig.from_file(policy_path)
    except Exception as exc:
        return {"step": "eval-policy", "status": _FAIL, "detail": f"{policy_path}: {exc}"}
    try:
        base = PolicyConfig.from_file(base_policy) if base_policy else None
    except Exception as exc:
        return {"step": "eval-policy", "status": _FAIL, "detail": f"base policy {base_policy}: {exc}"}
    entries = get_journal_entries(journal_path=journal_path or None, limit=None)
    report = evaluate_policy_candidate(
        cand, base_policy=base, journal_entries=entries, include_builtin_probes=True
    )
    parts = (
        f"promote_safe={report.get('promote_safe')}",
        f"probes={report.get('probes_checked')}",
        f"replayed={report.get('journal_replayed')}",
        f"newly_allowed={report.get('newly_allowed')}",
        f"upper_risk_bound_95={report.get('upper_risk_bound_95')}",
    )
    detail = " | ".join(parts)
    if report.get("safety_regressions"):
        detail += f" | safety_regressions={report['safety_regressions']}"
    return {
        "step": "eval-policy",
        "status": _OK if report.get("promote_safe") else _FAIL,
        "detail": detail,
        "eval_report": report,
    }


def run_shadow_report_step(journal_path: str = "", verbose: bool = True) -> Dict[str, Any]:
    """Summarize shadow-mode events (informational; never fails the chain)."""
    from .audit import get_journal_entries, print_shadow_report

    entries = get_journal_entries(journal_path=journal_path or None, limit=None)
    n_shadow = sum(
        1
        for e in entries
        if e.get("shadow_mode") or e.get("shadow_reason_code") or e.get("shadow_decision")
    )
    if verbose:
        print_shadow_report(entries)
    return {
        "step": "shadow-report",
        "status": _OK,
        "detail": f"{n_shadow} shadow event(s) of {len(entries)} journal entries (informational)",
    }


def run_chain(
    project_dir: Optional[Path] = None,
    policy: str = "",
    base_policy: str = "",
    journal_path: str = "",
    skip_self_test: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Execute the verification chain and return per-step results."""
    root = Path(project_dir) if project_dir else Path.cwd()
    results: List[Dict[str, Any]] = []
    if skip_self_test:
        results.append({"step": "self-test", "status": _SKIP, "detail": "--skip-self-test"})
    else:
        results.append(run_installer_self_test(root))
    results.append(run_doctor_step(root))
    if policy:
        results.append(run_eval_policy_step(policy, base_policy=base_policy, journal_path=journal_path))
    results.append(run_shadow_report_step(journal_path=journal_path, verbose=verbose))
    return results


def chain_exit_code(results: List[Dict[str, Any]]) -> int:
    """Map chain results to an exit code (0 pass, 2 unsafe policy, 1 wiring failure)."""
    if any(r["step"] == "eval-policy" and r["status"] == _FAIL for r in results):
        return 2
    if any(r["status"] == _FAIL for r in results):
        return 1
    return 0


def render_results(results: List[Dict[str, Any]]) -> str:
    """Render the human-readable chain summary (doctor-style box)."""
    lines = ["=" * 62, "AETHERIS MICROBRAINS - POLICY CHANGE VERIFICATION", "=" * 62]
    icon = {_OK: "[ ok ]", _WARN: "[warn]", _FAIL: "[FAIL]", _SKIP: "[skip]"}
    for r in results:
        line = f"{icon.get(r['status'], '[?]')} {r['step']}"
        if r.get("detail"):
            line += f" - {r['detail']}"
        lines.append(line)
    lines.append("-" * 62)
    unsafe = chain_exit_code(results) == 2
    failed = any(r["status"] == _FAIL for r in results)
    verdict = (
        "VERIFY FAILED: candidate policy not promotion-safe"
        if unsafe
        else ("VERIFY FAILED" if failed else "VERIFY PASSED")
    )
    lines.append(f"{verdict}: {len(results)} step(s)")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aetheris-verify",
        description=(
            "Post-policy-change verification chain: installer self-test, doctor, "
            "optional candidate-policy replay, and shadow-mode fatigue projection."
        ),
    )
    parser.add_argument("--project-dir", type=str, default="", help="Project root holding .claude/ (default: cwd)")
    parser.add_argument("--policy", type=str, default="", help="Candidate PolicyConfig file (JSON/TOML/YAML) to replay")
    parser.add_argument("--base-policy", type=str, default="", help="Optional base PolicyConfig for --policy comparison")
    parser.add_argument("--journal-path", type=str, default="", help="Path to decision_journal.jsonl (default: state dir)")
    parser.add_argument("--skip-self-test", action="store_true", help="Skip the installer self-test step")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable summary instead of the box")
    args = parser.parse_args(argv)
    _reconfigure_stdout()

    results = run_chain(
        project_dir=Path(args.project_dir) if args.project_dir else None,
        policy=args.policy,
        base_policy=args.base_policy,
        journal_path=args.journal_path,
        skip_self_test=args.skip_self_test,
        verbose=not args.json,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "results": [
                        {k: v for k, v in r.items() if k not in ("doctor_report", "eval_report")}
                        for r in results
                    ],
                    "exit_code": chain_exit_code(results),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(render_results(results))
    return chain_exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
