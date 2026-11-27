"""Tests for pure-stdlib repair hints and Clopper-Pearson threshold calibration."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from aetheris_microbrains.core.repair_hints import (
    calibrate_thresholds_from_journal,
    clopper_pearson_upper_bound,
    generate_repair_hint,
)


def test_allow_verdict_returns_none():
    payload = {"tool": "Write", "path": "app.py", "content": "x = 1\n"}
    verdict = {"decision": "allow", "reason_code": "layer1_ast_clean"}
    assert generate_repair_hint(payload, verdict) is None


def test_python_syntax_error_exact_line_col_and_snippet():
    broken = "def greet(name):\n    if name\n        return name\n"
    payload = {"tool": "Write", "path": "src/greet.py", "content": broken}
    verdict = {"decision": "deny", "reason_code": "layer1_python_syntax_error"}

    hint = generate_repair_hint(payload, verdict)
    assert hint is not None
    assert hint["category"] == "python_syntax_error"
    assert hint["line"] == 2
    assert hint["col"] == 12
    assert "if name" in hint["snippet"]
    assert "line 2" in hint["hint"]
    assert "col 12" in hint["hint"]
    assert hint["updated_input"] is None


def test_python_syntax_error_in_unified_diff_maps_target_line():
    diff_text = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -10,2 +10,3 @@\n"
        " def ok():\n"
        "+    x = 1 +\n"
        "     return x\n"
    )
    payload = {"tool": "Edit", "path": "mod.py", "diff": diff_text}
    verdict = {"decision": "deny", "reason_code": "layer1_syntax_invalid"}

    hint = generate_repair_hint(payload, verdict)
    assert hint is not None
    assert hint["category"] == "python_syntax_error"
    assert hint["line"] == 11
    assert isinstance(hint["col"], int) and hint["col"] >= 1
    assert "x = 1 +" in hint["snippet"]


def test_git_push_force_rewrite_flag_toggle():
    payload = {"tool": "Bash", "command": "git push --force origin main"}
    verdict = {"decision": "escalate", "reason_code": "layer1_shell_force_push"}

    no_rewrite = generate_repair_hint(payload, verdict, allow_rewrite=False)
    assert no_rewrite is not None
    assert no_rewrite["category"] == "dangerous_shell"
    assert "--force-with-lease" in no_rewrite["hint"]
    assert no_rewrite["updated_input"] is None

    with_rewrite = generate_repair_hint(payload, verdict, allow_rewrite=True)
    assert with_rewrite is not None
    assert with_rewrite["category"] == "dangerous_shell"
    assert with_rewrite["updated_input"] == {"command": "git push --force-with-lease origin main"}

    # Short `-f` flag also rewrites safely; existing `--force-with-lease` does not trigger rewrite
    short_hint = generate_repair_hint(
        {"tool": "Bash", "command": "git push -uf origin feat"},
        verdict,
        allow_rewrite=True,
    )
    assert short_hint is not None
    assert short_hint["updated_input"] == {"command": "git push -u --force-with-lease origin feat"}

    safe_hint = generate_repair_hint(
        {"tool": "Bash", "command": "git push --force-with-lease origin main"},
        {"decision": "escalate", "reason_code": "layer2_verification_escalation"},
        allow_rewrite=True,
    )
    assert safe_hint is not None
    assert safe_hint["category"] != "dangerous_shell"
    assert safe_hint["updated_input"] is None


def test_shell_to_structured_tool_hints():
    for cmd, expected_tool in (
        ("cat README.md", "Read"),
        ("grep -rn TODO src/", "Grep"),
        ("rg pattern aetheris_microbrains/", "Grep"),
        ("find . -name '*.py'", "Glob"),
    ):
        hint = generate_repair_hint(
            {"tool": "Bash", "command": cmd},
            {"decision": "deny", "reason_code": "layer1_shell_over_structured_tool"},
        )
        assert hint is not None
        assert hint["category"] == "shell_over_structured_tool"
        assert hint["suggested_tool"] == expected_tool
        assert expected_tool in hint["hint"]


def test_secret_literal_hint_recommends_secret_handle_and_never_leaks_raw_token():
    raw_token = "sk-proj-abc1234567890XYZ987654"
    payload = {
        "tool": "Write",
        "path": "src/client.py",
        "content": f'OPENAI_KEY = "{raw_token}"\n',
    }
    verdict = {"decision": "deny", "reason_code": "layer0_secret_detected"}

    hint = generate_repair_hint(payload, verdict)
    assert hint is not None
    assert hint["category"] == "secret_leak"
    assert "$SECRET:OPENAI_KEY" in hint["hint"]
    assert "os.environ" in hint["hint"]
    assert raw_token not in hint["hint"]
    assert hint["line"] == 1


def test_protected_path_hints_and_generic_fallback():
    for protected_path in (".env", ".claude/settings.json", ".qwen/config.json", ".prime/state.json", ".git/hooks/pre-commit"):
        hint = generate_repair_hint(
            {"tool": "Write", "path": protected_path, "content": "KEY=val\n"},
            {"decision": "deny", "reason_code": "layer0_protected_path"},
        )
        assert hint is not None
        assert hint["category"] == "protected_path"
        assert len(hint["hint"]) > 20

    generic = generate_repair_hint(
        {"tool": "custom_tool", "target": "something"},
        {"decision": "escalate", "reason_code": "layer2_uncertain", "suggested_action": "confirm_with_operator"},
    )
    assert generic is not None
    assert generic["category"] == "generic"
    assert generic["suggested_action"] == "confirm_with_operator"
    assert "confirm_with_operator" in generic["hint"]


def test_calibrate_thresholds_from_journal_mixed_and_fdr_bounds():
    empty_res = calibrate_thresholds_from_journal([], current_threshold=0.5, target_fdr=0.05)
    assert empty_res["calibrated_sample_count"] == 0
    assert empty_res["recommended_threshold"] == 0.5
    assert empty_res["noisy_reason_codes"] == []

    # Mixed allow / deny / escalate with labeled false positives on a noisy escalation rule
    entries = []
    for _ in range(12):
        entries.append({"decision": "allow", "reason_code": "layer0_allowlisted"})
    for _ in range(3):
        entries.append({"decision": "deny", "reason_code": "layer0_protected_path", "verified_correct": True})
    for _ in range(5):
        entries.append({"decision": "escalate", "reason_code": "layer2_noisy_heuristic", "verified_correct": False})

    res = calibrate_thresholds_from_journal(entries, current_threshold=0.5, target_fdr=0.05)
    assert res["calibrated_sample_count"] == 20
    assert res["allow_count"] == 12
    assert res["block_count"] == 3
    assert res["escalate_count"] == 5
    assert abs(res["allow_rate"] - 0.6) < 1e-6
    assert abs(res["block_rate"] - 0.15) < 1e-6
    assert abs(res["escalate_rate"] - 0.25) < 1e-6
    assert "layer2_noisy_heuristic" in res["noisy_reason_codes"]
    assert "layer0_protected_path" not in res["noisy_reason_codes"]
    assert res["recommended_threshold"] > 0.5
    assert res["fdr_upper_bound"] is not None
    assert res["fdr_upper_bound"] >= res["empirical_fdr"]

    # Unlabeled journal also flags repeated escalations on a noisy rule
    unlabeled = (
        [{"decision": "allow", "reason_code": "layer0_allowlisted"} for _ in range(10)]
        + [{"decision": "deny", "reason_code": "layer0_secret"} for _ in range(2)]
        + [{"decision": "escalate", "reason_code": "layer1_noisy_shell"} for _ in range(4)]
        + [{"decision": "escalate", "reason_code": "layer1_rare_edge"}]
    )
    unlabeled_res = calibrate_thresholds_from_journal(unlabeled, current_threshold=0.5, target_fdr=0.05)
    assert unlabeled_res["calibrated_sample_count"] == 17
    assert "layer1_noisy_shell" in unlabeled_res["noisy_reason_codes"]
    assert "layer1_rare_edge" not in unlabeled_res["noisy_reason_codes"]

    # Exact Clopper-Pearson sanity check for k=0: 1 - alpha^(1/n)
    cp_zero = clopper_pearson_upper_bound(0, 10, alpha=0.05)
    assert abs(cp_zero - (1.0 - 0.05 ** 0.1)) < 1e-9
    assert clopper_pearson_upper_bound(5, 5, alpha=0.05) == 1.0
    assert clopper_pearson_upper_bound(2, 20, alpha=0.05) > 0.1


def test_repair_hints_zero_numpy():
    module_path = Path(__file__).resolve().parents[1] / "core" / "repair_hints.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "numpy" not in imported
    assert "scipy" not in imported

    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import aetheris_microbrains.core.repair_hints; assert 'numpy' not in sys.modules",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
