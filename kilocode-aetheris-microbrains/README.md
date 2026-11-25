# Aetheris Microbrains Kilo Code Plugin

Native ESM plugin for **Kilo Code** installing into `.kilo/plugins/` (workspace) or `~/.config/kilo/plugins/` (global) using Kilo Code's `tool.execute.before` and `tool.execute.after` hooks.

## Features

- **`tool.execute.before`**: Sends tool calls (`bash`, `write`, `edit`, `read`, MCP tools) over HTTP/1.1 keep-alive to the local Aetheris warm daemon (`--host kilocode`), throwing `Error(blockReason)` to halt blocked or escalated calls (and optionally applying sandbox `output.args` rewrites).
- **`tool.execute.after`**: Synchronously scrubs leaked API keys, DB URIs, and private keys from `output.output` via `scrubSecretsSync`, then reports post-tool observations for cross-turn IFC taint tracking.
- **Control-Plane Protection**: `.kilo/`, `.kilo/plugins/`, `.kilo/plugin/`, `kilo.json`, and `kilo.jsonc` are write-protected by default.

## Quick Setup

```bash
pip install -e .
aetheris install kilocode --self-test
aetheris install kilocode --workspace .

# Start the local HTTP daemon before launching Kilo Code:
aetheris-daemon --http --host kilocode --port 8765 --token "$AETHERIS_DAEMON_TOKEN"
```

### Environment Variables

- `AETHERIS_DAEMON_URL` (default: `http://127.0.0.1:8765`)
- `AETHERIS_DAEMON_TOKEN` (optional bearer token matching `aetheris-daemon --token`)
- `AETHERIS_HOOK_SHADOW` (`1` for observe-only shadow mode)
- `AETHERIS_FAIL_CLOSED` (default: `1`; set `0` to allow execution if daemon is stopped)
- `AETHERIS_BLOCK_ON_ESCALATE` (default: `1`; set `0` to pass escalated calls through)
- `AETHERIS_TIMEOUT_MS` (default: `2500`, clamped `200..4500`)
