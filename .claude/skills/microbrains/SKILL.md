---
name: microbrains
description: Pre-flight safety gate and economics toolkit for Claude Code. Use when installing/enabling the Microbrains (Aetheris) PreToolUse/PostToolUse hooks, explaining a blocked/denied tool call from the decision journal, pre-flight checking a proposed diff or shell command before applying it, or auditing MCP servers for prompt injection and tool-schema risk. Triggers - "microbrains", "mb", "aetheris", "pre-flight gate", "why was my command blocked", "hook denied", "check this patch", "scan MCP", "decision journal".
---

# Aetheris Pre-Flight Gate Skill

Aetheris Microbrains runs a deterministic, <2ms local pre-flight gate in front of
tool execution: AST lint of diffs, shlex-aware shell analysis, secret-egress veto,
protected-path guard, and fail-closed escalation. It is a static-analysis layer,
NOT an OS sandbox.

## Commands

### 1. Install the hooks into `.claude/settings.local.json`

Run the bundled installer (idempotent, merges without clobbering existing hooks). By default it writes to `.claude/settings.local.json` (gitignored, pinned to your local Python executable); pass `--shared` to write portable `python` hooks into `.claude/settings.json`:

```bash
python .claude/skills/microbrains/install_hook.py              # enforce mode (allow/ask/deny)
python .claude/skills/microbrains/install_hook.py --shadow      # observe-only, never blocks
python .claude/skills/microbrains/install_hook.py --strict-ast  # also allowlist dynamic calls/imports
python .claude/skills/microbrains/install_hook.py --shared      # write portable config to .claude/settings.json
python .claude/skills/microbrains/install_hook.py --remove      # uninstall the hooks
```

Verify with a probe payload (expect `permissionDecision: "deny"` for the rm probe):

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /tmp/aetheris_probe"}}' | python -m aetheris_microbrains.hook
```

If `aetheris_microbrains` is not installed (`python -c "import aetheris_microbrains"` fails),
install it first: `pip install -e .` from this repo (or `pip install aetheris-microbrains`).

### 2. Explain a blocked tool call

When a tool call was denied and the user (or you) needs to know why:

```bash
python -m aetheris_microbrains.audit --last      # most recent decision
python -m aetheris_microbrains.audit --tail 20   # last 20
python -m aetheris_microbrains.audit --summary   # aggregate
```

Translate the `reason_code` + `label` into a one-sentence human explanation and the
concrete fix. Every entry has `input_sha256` and a `payload_preview` for correlation.

### 3. Pre-flight check a diff or command BEFORE applying it

For a proposed Python edit (in-process, no file touched):

```python
import aetheris_microbrains as mb
verdict = mb.evaluate_hook_payload({
    "tool_name": "Edit",
    "tool_input": {"file_path": "src/app.py", "new_string": "<proposed code>"},
})
print(verdict["decision"], verdict["reason_code"], verdict["reason"])
```

For a proposed shell command: same call with `{"tool_name": "Bash",
"tool_input": {"command": "<cmd>"}}`. `allow` -> safe to apply; `ask` -> escalate to
the user with `verdict["reason"]`; `deny` -> fix the code/command first. For ambiguous
diffs pass `file_content=` (full file text) via
`mb.evaluate_hook_payload(payload, file_content=...)` so the AST layer has context;
without it ambiguous cases safely degrade to `ask`.

### 4. Scan MCP servers & scrub outputs

```python
import aetheris_microbrains as mb
report = mb.scan_mcp_tools(<tool_schemas_list>)                 # returns dict: tool-schema risk report
inj = mb.detect_prompt_injection(<tool_output_text>)           # returns dict: injection_detected, severity, findings
scrubbed = mb.scrub_tool_output(<tool_output_text>)            # returns dict: secrets_found, secret_types, redacted_text
```

## Response contract

- `allow` -> proceed normally.
- `ask` (escalate) -> surface `reason` to the user and wait. Fail-closed: silence never grants permission.
- `deny` -> do NOT retry the same call. Read `reason_code`, fix the root cause
  (syntax, secret, protected path, destructive command), then re-submit the fixed version.
- Journal lives at `aetheris_microbrains/memory/decision_journal.jsonl`
  (override with `AETHERIS_JOURNAL_PATH`). Never commit it.
