---
name: microbrains-gate
description: Pre-flight shell, edit, secret, and MCP safety gate for Kimi Code via Aetheris Microbrains.
---

# Aetheris Microbrains Gate for Kimi Code

This plugin enforces deterministic `PreToolUse` safety checks (syntax AST validation, destructive shell blocking, `.env`/`.ssh`/control-plane protection, and secret egress defense) and `PostToolUse` secret/prompt-injection audit logging.

## Verdict Semantics (`--host kimi`)
- `allow`: Exits `0` silently; preserves Kimi's normal permission flow.
- `deny`: Prints reason to `stderr` and exits `2` to block execution.
- `escalate`: Exits `0` silently (`defer_to_host_policy`), or passes `--codex-escalate-deny` for strict CI blocking.
- `PostToolUse`: Scans outputs for leaked secrets/injection and records findings to the local decision journal without mutating tool output.

## Recommended Rollout
1. Install plugin: `/plugins install https://github.com/drtikov/microbrainsforagents` then `/reload`.
2. Start in `--shadow` (observe) mode, inspect `decision_journal.jsonl`, then run `--profile guardrails`.
3. Optional MCP tools (`pip install "aetheris-microbrains[mcp]"`): configure `~/.kimi-code/mcp.json` with `command = "aetheris-mcp"` and `"deferred": true`.
