# Aetheris Microbrains for Pi Coding Agent

Zero-dependency Level 3 Pi extension connecting Pi's `tool_call` and `tool_result` lifecycle hooks to Aetheris Microbrains' deterministic pre-flight policy gate and DLP secret scrubber.

## Features

- **`tool_call` pre-flight gate**: Evaluates every tool invocation (`bash`, `write`, `edit`, `read`, etc.) against `aetheris-daemon` before execution (`X-Aetheris-Host: pi`).
- **Native Layer 3 escalation**: Uses `ctx.ui.confirm(...)` in TUI and RPC modes when Microbrains returns `escalate: true`, falling back to `blockOnEscalate` in non-interactive runs.
- **`tool_result` synchronous DLP scrubbing**: Redacts API keys, private keys, and connection strings from `event.content` and `event.details` before the LLM sees tool output, with best-effort daemon audit logging.
- **`/aetheris` & `/aetheris-config` commands**: `/aetheris` displays live daemon health and session metrics (`checked`, `blocked`, `approved`, `redacted`); `/aetheris-config` displays effective configuration (with token masked) and current session stats.
- **Settings & Lifecycle Persistence**: Reads optional overrides from `.pi/settings.json` (`"aetheris": { ... }`) via `ctx.settingsManager`, restores the latest branch-aware `aetheris-stats` snapshot on `session_start` (`getBranch()` fallback `getEntries()`), checks daemon health without blocking startup, aborts in-flight requests on `session_shutdown`, and persists stats-only snapshots on blocks, escalations, every 10 checks, and shutdown via `pi.appendEntry("aetheris-stats", ...)`.
- **Fail-closed default**: Blocks tool execution when the daemon is unreachable unless `AETHERIS_FAIL_CLOSED=0` or `AETHERIS_HOOK_SHADOW=1`.

## Quick Setup

```bash
pip install -e .
aetheris install pi --self-test
aetheris install pi --workspace .          # installs .pi/extensions/aetheris-microbrains/index.js
# or omit --workspace to install globally to ~/.pi/agent/extensions/aetheris-microbrains/index.js

# Start the warm HTTP daemon before launching pi
aetheris-daemon --http --host pi --port 8765
```

Inside a Pi session, run `/reload` after installing or run `/aetheris` to verify daemon connectivity.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AETHERIS_DAEMON_URL` | `http://127.0.0.1:8765` | Warm HTTP daemon endpoint |
| `AETHERIS_DAEMON_TOKEN` | *(empty)* | Optional Bearer token for authenticated daemons |
| `AETHERIS_HOOK_SHADOW` | `0` | Set `1` for observe-only shadow mode |
| `AETHERIS_FAIL_CLOSED` | `1` | Set `0` to allow tools when the daemon is unreachable |
| `AETHERIS_BLOCK_ON_ESCALATE` | `1` | Set `0` to allow non-interactive escalations when `ctx.ui.confirm` is unavailable |
| `AETHERIS_TIMEOUT_MS` | `2500` | Per-check HTTP timeout clamped to `200..4500` ms |
