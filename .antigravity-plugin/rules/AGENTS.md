# Aetheris Microbrains Pre-Flight Guardrails

Aetheris Microbrains enforces deterministic local safety and hygiene checks via `PreToolUse` and `PostToolUse` lifecycle hooks:

- **Pre-execution blocks (`deny`):** Syntax-broken Python edits/writes, destructive shell patterns, secret egress, or protected-path writes (`.env`, `.ssh/`, `.git/hooks/`) are blocked before execution with an explanation in `reason`.
- **High-scrutiny escalations (`force_ask`):** Ambiguous shell wrappers, external build targets, or unusual operations explicitly prompt for user confirmation.
- **Forensic explanations:** If a tool call is blocked and you need full diagnostic context or policy tuning suggestions, run `python -m aetheris_microbrains.audit --last` or `--explain`.
- **On-demand checks:** When the `aetheris-microbrains` MCP server is enabled, prefer `microbrain_preflight_check` before risky multi-step patches or commands, and `microbrain_scan_mcp_tools` when auditing third-party MCP servers.
