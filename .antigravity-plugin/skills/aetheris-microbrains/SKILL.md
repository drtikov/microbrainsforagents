---
name: aetheris-microbrains
description: Deterministic pre-flight AST linter, secret-egress gate, context-hygiene guard, and forensic audit journal for Google Antigravity. Use when auditing blocked actions (`aetheris-audit --explain`), toggling `--shadow` or enforce profiles, scanning MCP tools (`aetheris-doctor --scan-mcp`), or calling `microbrain_*` MCP utilities.
---

# Aetheris Microbrains in Google Antigravity

Aetheris Microbrains protects Antigravity tool calls (`run_command`, `edit_file`, `write_file`, `replace_file_content`, `write_to_file`, and `mcp_*`) through native `PreToolUse` and `PostToolUse` hooks plus optional MCP server tools.

## 1. Inspect Recent Decisions & Explain Blocks

```bash
python -m aetheris_microbrains.audit --last
python -m aetheris_microbrains.audit --explain
python -m aetheris_microbrains.audit --digest --summary
```

## 2. Check Health & Scan Configured MCP Servers

```bash
python -m aetheris_microbrains.doctor --scan-mcp
python -m aetheris_microbrains.mcp_server --check
```

## 3. Reconfigure or Verify the Antigravity Plugin

```bash
# Shadow (observe-only) pilot
python -m aetheris_microbrains.install_antigravity --shadow --profile dev

# Active enforcement (global ~/.gemini/config/plugins/aetheris-microbrains/)
python -m aetheris_microbrains.install_antigravity --profile dev

# Verify contract & installed bundle files
python -m aetheris_microbrains.install_antigravity --self-test
```

## 4. Available On-Demand MCP Tools

When `mcp_config.json` is active (requires `pip install -e ".[mcp]"`):
- `microbrain_preflight_check` — dry-run shell commands, file writes, or edits before running them.
- `microbrain_scrub_and_scan` — redact secrets and check untrusted text for indirect prompt injection.
- `microbrain_audit_explain` — inspect recent journal blocks and rule rationales.
- `microbrain_scan_mcp_tools` — detect schema poisoning and rug-pull drift on registered MCP tools.
- `microbrain_narrow_tools` — sticky cache-safe tool surface routing for multi-turn agent loops.
- `microbrain_verify` / `microbrain_health` / `microbrain_directives` — offline verification and diagnostic status.
