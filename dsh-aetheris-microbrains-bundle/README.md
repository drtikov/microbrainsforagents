# Aetheris Microbrains MCP bundle

Connects the local `aetheris_microbrains.mcp_server` to DeepSeek Harness through the shipped `@deepseek-ai/dsh-mcp-client` bridge over Streamable HTTP at `http://127.0.0.1:8766/mcp`.

## Quick Setup

1. **Install MCP dependencies** (Python 3.10+):
   ```bash
   pip install -e ".[mcp]"
   ```
2. **Start the local MCP server** from the repository root:
   ```bash
   python -m aetheris_microbrains.mcp_server --http --port 8766
   ```
3. **Load the bundle & verify** in DeepSeek Harness:
   Install or register `./dsh-aetheris-microbrains-bundle` in DSH (`@local/aetheris-microbrains-mcp`), start a new DSH session, and run `mcp__microbrains__microbrain_health`. You can also validate the bundle offline with:
   ```bash
   python -m aetheris_microbrains.install_dsh --self-test
   ```

> **Port distinction:** This MCP Streamable HTTP server listens on port `8766` (`/mcp`). The separate optional REST worker client (`aetheris_microbrains/evolution/dsh_microbrain_client.py`) defaults to port `8080`.

## On-demand only

The bundle exposes tools when the agent explicitly invokes them; it does **not** automatically intercept ordinary DSH tool calls. Install the repository's separate hook if you need automatic pre-tool enforcement.

## Exposed tools

The repository's verified MCP self-check registers eight tools:

- `microbrain_health`
- `microbrain_verify`
- `microbrain_preflight_check`
- `microbrain_scrub_and_scan`
- `microbrain_narrow_tools`
- `microbrain_audit_explain`
- `microbrain_directives`
- `microbrain_scan_mcp_tools`

In a new DSH session these are available under the `microbrains` namespace, for example `mcp__microbrains__microbrain_health`.

## Scope

This is a deterministic local policy and validation layer for tool calls, diffs, shell commands, and tool output. It is not an OS sandbox, container, or credential broker. Keep production credentials outside model context and use OS-level isolation for untrusted execution.
