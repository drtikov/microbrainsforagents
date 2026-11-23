---
name: microbrains
description: Deterministic pre-flight safety gate, AST linter, secret-egress blocker, and forensic journal explainer for OpenAI Codex CLI. Use when configuring `.codex/hooks.json`, explaining blocked commands (`aetheris-audit --explain`), checking patches before applying, or wiring optional Microbrains MCP tools.
---

# Aetheris Microbrains Skill for OpenAI Codex CLI

Aetheris Microbrains runs a deterministic local pre-flight policy gate (`< 2 ms` in-process) in front of Codex tool execution (`Bash`, `shell_command`, `exec_command`, `apply_patch`, `Edit`, `Write`, `fwrite`).

## 1. Quick Setup (Recommended Pilot -> Enforce Flow)

1. Install the package:
   ```bash
   pip install -e .
   ```
2. Start in **shadow (observe-only)** mode to audit verdicts in the journal first:
   ```bash
   python .agents/skills/microbrains/install_codex_hook.py --shadow --profile dev
   ```
3. Review the journal, then switch to active **enforcement**:
   ```bash
   python .agents/skills/microbrains/install_codex_hook.py --profile dev
   # Or install globally for your user profile (~/.codex/hooks.json):
   python .agents/skills/microbrains/install_codex_hook.py --user --profile dev
   ```
4. Verify the Codex hook contract and installation:
   ```bash
   python .agents/skills/microbrains/install_codex_hook.py --self-test
   ```

## 2. How Escalations Work in Codex (`--host codex`)

Unlike Claude Code, Codex `PreToolUse` does **not** use `"permissionDecision": "ask"`:
- **Deny (`decision == "deny"`):** Emits `permissionDecision: "deny"` JSON + `stderr` reason and exits `2` (always blocked before execution).
- **Allow (`decision == "allow"`):** Exits `0` silently (or emits `"allow"` if `--auto-allow` is set).
- **Escalate (`decision == "escalate"`):** Exits `0` with no `"permissionDecision"` field (`defer_to_host_policy`), handing control to Codex's configured approval policy. Pass `--codex-escalate-deny` if you want escalated commands treated as hard denials in automated pipelines.
- **PostToolUse Alerts:** Emits `{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "SECURITY WARNING ..."}}` when secrets or indirect prompt injections are found in tool output.

## 3. Explain Blocked Actions & Run Checks

```bash
python -m aetheris_microbrains.audit --last
python -m aetheris_microbrains.audit --explain
python -m aetheris_microbrains.audit --digest --summary
python -m aetheris_microbrains.doctor --scan-mcp
```

## 4. Optional On-Demand MCP Server

To expose Microbrains tools (`microbrain_preflight_check`, `microbrain_scrub_and_scan`, `microbrain_audit_explain`) inside Codex in addition to automatic hooks:

```bash
pip install -e ".[mcp]"
python -m aetheris_microbrains.mcp_server --check
```
