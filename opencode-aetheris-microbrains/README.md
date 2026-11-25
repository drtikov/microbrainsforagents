# Aetheris Microbrains OpenCode Plugin

Native plugin module for **OpenCode** integrating Aetheris Microbrains into `.opencode/plugins/` (workspace) or `~/.config/opencode/plugins/` (global) using OpenCode's `tool.execute.before` and `tool.execute.after` hooks.

## Features

- **`tool.execute.before`**: Sends tool calls (`bash`, `write`, `edit`, `read`, MCP tools) over HTTP/1.1 keep-alive to the local Aetheris warm daemon (`--host opencode`), throwing `Error(blockReason)` to halt blocked or escalated calls (and optionally applying sandbox `output.args` rewrites).
- **`tool.execute.after`**: Synchronously scrubs leaked API keys, DB URIs, and private keys from `output.output` via `scrubSecretsSync`, then reports post-tool observations for cross-turn IFC taint tracking.
- **Control-Plane Protection**: `.opencode/`, `.opencode/plugins/`, `opencode.json`, and `opencode.jsonc` are write-protected by default.

## Quick Setup

```bash
pip install -e .
aetheris install opencode --self-test
aetheris install opencode --workspace .

# Start the local HTTP daemon before launching OpenCode:
aetheris-daemon --http --host opencode --port 8765 --token "$AETHERIS_DAEMON_TOKEN"
```

### Environment Variables

- `AETHERIS_DAEMON_URL` (default: `http://127.0.0.1:8765`)
- `AETHERIS_DAEMON_TOKEN` (optional bearer token matching `aetheris-daemon --token`)
- `AETHERIS_HOOK_SHADOW` (`1` for observe-only shadow mode)
- `AETHERIS_FAIL_CLOSED` (default: `1`; set `0` to allow execution if daemon is stopped)
- `AETHERIS_BLOCK_ON_ESCALATE` (default: `1`; set `0` to pass escalated calls through)
- `AETHERIS_TIMEOUT_MS` (default: `2500`, clamped `200..4500`)
