---
name: microbrains
description: Pre-flight safety gate and decision-journal toolkit for ZCode. Use when verifying, configuring, or explaining Aetheris Microbrains PreToolUse/PostToolUse hook verdicts, checking a proposed patch or shell command before execution, or auditing MCP servers.
---

# Aetheris Pre-Flight Gate Skill (ZCode)

Aetheris Microbrains runs a deterministic local pre-flight gate (pure Python stdlib security core; <2 ms in-process on small payloads, ~50–155 ms per fresh CLI hook spawn) in front of ZCode tool execution: AST lint of diffs, shlex-aware shell inspection, secret-egress veto, protected-path guard, and fail-closed escalation. It is a static-analysis layer, not an OS container.

## 1. Verify the Gate Is Live

```bash
# Red-team probe: expect permissionDecision "deny" and exit code 2
echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' | python "${ZCODE_PLUGIN_ROOT:-.}/.zcode-plugin/hooks/zcode_gate.py" --session-state auto

# Safe probe: expect empty stdout and exit code 0
echo '{"tool_name":"Bash","tool_input":{"command":"git status"}}' | python "${ZCODE_PLUGIN_ROOT:-.}/.zcode-plugin/hooks/zcode_gate.py" --session-state auto
```

Confirm plugin registration in ZCode under **Settings → Plugin Management → Aetheris Microbrains**.

## 2. Explain a Blocked Tool Call

```bash
python -m aetheris_microbrains.audit --last
python -m aetheris_microbrains.audit --explain
python -m aetheris_microbrains.audit --summary
python -m aetheris_microbrains.audit --savings
```

## 3. Pre-Flight Check Before Execution

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"<PROPOSED COMMAND>"}}' | python -m aetheris_microbrains.hook --host zcode
echo '{"tool_name":"Edit","tool_input":{"file_path":"mod.py","old_string":"","new_string":"<PROPOSED CODE>"}}' | python -m aetheris_microbrains.hook --host zcode
```

## 4. Installer & Doctor

```bash
aetheris install zcode --self-test
aetheris install zcode --workspace .     # pins sys.executable in .zcode/config.json
python -m aetheris_microbrains.doctor
```

## ZCode-Specific Behavior

- **Strict stdout JSON:** Gate `deny` emits `{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny",...}}` + exit code `2`; `allow` emits empty stdout + exit `0`; `escalate` emits `permissionDecision: "ask"` + exit `0` so ZCode prompts the user. Reason text goes to `stderr`.
- **Matchers:** `Bash|Edit|Write|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*` (`ApplyPatch` aliases to `Write`/`Edit`).
- **No `onFailure` field:** If a hook exits with an unexpected non-zero error, ZCode marks the run failed rather than silently proceeding.
- **Layer 0 control-plane protection:** `.zcode/`, `.zcode/config.json`, `.zcode/cli/config.json`, and `.zcode-plugin/` are write-protected by default while enforcing.
- **Opt-in MCP:** `aetheris-mcp` / `aetheris-mcp-proxy` are omitted from `.zcode-plugin/plugin.json` because they require the `[mcp]` extra; add them manually under `.zcode/config.json` `mcp.servers` if desired.
