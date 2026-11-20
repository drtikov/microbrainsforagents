"""Integration tests for Claude Code / Codex PreToolUse & PostToolUse hook.

Validates the full hook pipeline (CLI entrypoint, stdin JSON parsing, payload normalization,
file write content inspection, host contract adapters, exit codes, and journal recording)
to ensure content-level security inspections cannot silently regress.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from aetheris_microbrains.hook import evaluate_hook_payload, evaluate_post_tool_use_payload, main


def _run_hook_cli(payload: Dict[str, Any], args: list[str] | None = None) -> tuple[int, str, str]:
    """Execute hook CLI in-process via main() capturing stdin/stdout/stderr."""
    import io
    stdin_data = json.dumps(payload)
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        code = main(args or [])
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
        return code, out, err
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr


def _run_hook_subprocess(payload: Dict[str, Any], extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run `python -m aetheris_microbrains.hook` in an isolated subprocess."""
    cmd = [sys.executable, "-m", "aetheris_microbrains.hook"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )


class TestHookContentInspectionIntegration:
    """End-to-end tests for hook.py content inspection across tool types and hosts."""

    @pytest.mark.parametrize("tool_name", ["write", "write_file", "repo_write", "edit", "edit_file", "str_replace_editor"])
    def test_hook_blocks_dangerous_write_content_direct(self, tool_name: str):
        """Dangerous Python payloads in write/edit tool calls must be denied by hook."""
        payload = {
            "tool_name": tool_name,
            "tool_input": {
                "file_path": "payload.py",
                "content": "import os\nos.system('rm -rf /')\n",
            },
        }
        res = evaluate_hook_payload(payload, host="claude")
        assert res["decision"] == "deny"
        assert res["exit_code"] == 2
        assert res["permissionDecision"] == "deny"
        assert "dangerous_method:os.system" in res["reason"]

    def test_hook_blocks_subprocess_in_new_string(self):
        """str_replace_editor new_string content inspection."""
        payload = {
            "tool": "str_replace_editor",
            "tool_input": {
                "path": "server.py",
                "old_str": "# placeholder",
                "new_string": "import subprocess\nsubprocess.run(['cat', '/etc/shadow'])\n",
            },
        }
        res = evaluate_hook_payload(payload, host="claude")
        assert res["decision"] == "deny"
        assert res["exit_code"] == 2
        assert "subprocess_call:run" in res["reason"]

    def test_hook_allows_benign_write(self):
        """Normal code files must be cleanly allowed."""
        payload = {
            "tool_name": "write_file",
            "tool_input": {
                "file_path": "math_utils.py",
                "content": "def add(a: int, b: int) -> int:\n    return a + b\n",
            },
        }
        res = evaluate_hook_payload(payload, host="claude")
        assert res["decision"] == "allow"
        assert res["exit_code"] == 0
        assert res["permissionDecision"] == "allow"

    def test_hook_cli_main_claude_contract(self):
        """CLI main() invocation with Claude contract format."""
        payload = {
            "tool_name": "write_file",
            "tool_input": {
                "file_path": "exploit.py",
                "content": "import pickle\npickle.loads(b'cos\\nsystem\\n(S\"id\"\\ntR.')\n",
            },
        }
        code, out, err = _run_hook_cli(payload, ["--host", "claude"])
        assert code == 2
        assert "unsafe_deserialization:pickle.loads" in err or "pickle.loads" in out
        data = json.loads(out)
        hook_out = data.get("hookSpecificOutput", data)
        assert hook_out.get("permissionDecision") == "deny"

    def test_hook_cli_subprocess_end_to_end(self):
        """Full external subprocess test simulating Claude Code calling the hook."""
        payload = {
            "tool_name": "bash",
            "tool_input": {
                "command": "cat payload.sh | sh",
            },
        }
        proc = _run_hook_subprocess(payload, ["--host", "claude"])
        assert proc.returncode == 2
        out_json = json.loads(proc.stdout)
        hook_out = out_json.get("hookSpecificOutput", out_json)
        assert hook_out["permissionDecision"] == "deny"
        assert "remote_code_execution_pipe" in hook_out["permissionDecisionReason"]

    def test_hook_codex_host_contract_adaptation(self):
        """Codex PreToolUse does not support 'ask'; verify hook adaptation."""
        # Benign action -> allow
        allow_payload = {
            "tool_name": "execute_shell",
            "tool_input": {"command": "git status"},
        }
        res_allow = evaluate_hook_payload(allow_payload, host="codex")
        assert res_allow["decision"] == "allow"
        assert res_allow["exit_code"] == 0
        assert res_allow["permissionDecision"] == "allow"

        # Deny action -> deny exit 2
        deny_payload = {
            "tool_name": "write_file",
            "tool_input": {
                "file_path": "app.py",
                "content": "eval('__import__(\"os\").system(\"id\")')\n",
            },
        }
        res_deny = evaluate_hook_payload(deny_payload, host="codex")
        assert res_deny["decision"] == "deny"
        assert res_deny["exit_code"] == 2
        assert res_deny["permissionDecision"] == "deny"

    def test_hook_shadow_mode_never_blocks(self, tmp_path: Path):
        """AETHERIS_HOOK_SHADOW=1 logs to journal and returns 0 exit code even for malicious payload."""
        journal_file = tmp_path / "shadow_journal.jsonl"
        payload = {
            "tool_name": "write_file",
            "tool_input": {
                "file_path": "exploit.py",
                "content": "import os\nos.system('whoami')\n",
            },
        }
        res = evaluate_hook_payload(
            payload,
            shadow_mode=True,
            host="claude",
            journal_path=str(journal_file),
        )
        assert res["exit_code"] == 0
        assert res["permissionDecision"] is None
        assert res["host_action"] == "shadow_observe"

    def test_post_tool_use_secret_scrubbing(self):
        """PostToolUse hook sanitizes raw credentials in output and adds warning context."""
        payload = {
            "tool": "read_file",
            "tool_input": {"path": ".env"},
            "tool_output": "API_KEY=sk-ant-api03-abcdef1234567890abcdef1234567890\n",
        }
        res = evaluate_post_tool_use_payload(payload, host="claude")
        assert res["secrets_found"] is True
        assert "[REDACTED_ANTHROPIC_KEY]" in res["redacted_output"]
        assert res["hookSpecificOutput"] is not None
        assert "SECURITY WARNING" in res["hookSpecificOutput"]["additionalContext"]
