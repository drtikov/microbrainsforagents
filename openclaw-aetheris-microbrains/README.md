# Aetheris Microbrains OpenClaw Plugin (`@aetheris/openclaw-microbrains`)

Native OpenClaw Gateway plugin wiring Aetheris Microbrains into OpenClaw's typed plugin lifecycle:

- **`before_tool_call` (priority `100`):** Sub-2ms pre-flight gate (Layer-0 protected paths/IFC taint + Layer-1 polyglot AST & `shlex` shell check). Maps `deny` to `{ block: true, blockReason }`, `escalate` to OpenClaw's native interactive `requireApproval` (`timeoutBehavior: "deny"`), unattended `trigger: "cron"` escalations to `{ block: true }`, and optional `--sandbox-wrap` rewrites to `{ params }`. Uses an internal `AbortController` timeout (`<= 4000 ms`) shorter than OpenClaw's `timeoutMs` (`5000 ms`) so failures fail closed cleanly.
- **`tool_result_persist` (sync):** Redacts leaked API keys, tokens, DB credentials, and PEM private keys before OpenClaw persists tool output into session transcripts.
- **`after_tool_call` (async):** Records post-tool observations, secret/injection warnings, and cross-turn IFC session taint via `/v1/hook/post`.
- **`aetheris-hook-daemon` service:** Automatically starts/stops `python -m aetheris_microbrains.hook_daemon --http --host openclaw` on Gateway lifecycle transitions with a fresh per-session bearer token.

## Quick Setup

1. Install Python package:
   ```bash
   pip install -e .
   ```
2. Install the OpenClaw plugin from the repository root:
   ```bash
   openclaw plugins install --link ./openclaw-aetheris-microbrains
   ```
   Or configure `~/.openclaw/openclaw.json` automatically via:
   ```bash
   aetheris install openclaw --dry-run
   aetheris install openclaw
   aetheris install openclaw --self-test
   ```
3. Restart OpenClaw Gateway:
   ```bash
   openclaw gateway restart
   ```
