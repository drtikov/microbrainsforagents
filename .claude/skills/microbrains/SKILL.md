---
name: microbrains
description: Pre-flight safety gate and economics toolkit for Claude Code. Use when installing/enabling the Microbrains (Aetheris) PreToolUse/PostToolUse hooks, explaining a blocked/denied tool call from the decision journal, pre-flight checking a proposed diff or shell command before applying it, or auditing MCP servers for prompt injection and tool-schema risk. Triggers - "microbrains", "mb", "aetheris", "pre-flight gate", "why was my command blocked", "hook denied", "check this patch", "scan MCP", "decision journal".
---

# Aetheris Pre-Flight Gate Skill

Aetheris Microbrains runs a deterministic local pre-flight gate (`<2 ms` in-process;
~50–155 ms per fresh CLI hook process spawn) in front of tool execution: AST lint
of diffs, shlex-aware shell analysis, secret-egress veto, protected-path guard,
and fail-closed escalation. It is a static-analysis layer, NOT an OS sandbox.

## Commands

### 1. Install the hooks into `.claude/settings.local.json`

Run the bundled installer (idempotent, merges without clobbering existing hooks). By default it writes to `.claude/settings.local.json` (gitignored, pinned to your local Python executable with forward slashes), registers `PreToolUse` (`Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*` with fail-closed `"onFailure": "block"` and `--session-state auto`), and registers synchronous `PostToolUse` (`Bash|Read|Grep|WebFetch|WebSearch|mcp__.*`) so secret/injection alerts apply before context ingestion. Pass `--shared` to write portable `python` hooks into `.claude/settings.json`:

```bash
python .claude/skills/microbrains/install_hook.py              # enforce mode (allow/ask/deny) + --session-state auto
python .claude/skills/microbrains/install_hook.py --shadow      # observe-only, never blocks
python .claude/skills/microbrains/install_hook.py --strict-ast  # also allowlist dynamic calls/imports
python .claude/skills/microbrains/install_hook.py --profile dev # developer workflow allowlist
python .claude/skills/microbrains/install_hook.py --policy security.toml --no-session-state
python .claude/skills/microbrains/install_hook.py --http-daemon http://127.0.0.1:8765  # native Claude Code HTTP hook mode
python .claude/skills/microbrains/install_hook.py --project-dir /path/to/repo  # target another project
python .claude/skills/microbrains/install_hook.py --shared      # write portable config to .claude/settings.json
python .claude/skills/microbrains/install_hook.py --auto-allow  # emit allow on safe verdicts (skips host prompts)
python .claude/skills/microbrains/install_hook.py --async-post  # run PostToolUse asynchronously instead of blocking
python .claude/skills/microbrains/install_hook.py --remove      # uninstall the hooks
python .claude/skills/microbrains/install_hook.py --preset solo-dev   # bundles: dev profile + auto-allow + async post (also: ci, paranoid)
python .claude/skills/microbrains/install_hook.py --preset solo-dev --no-auto-allow --no-async-post  # override any preset boolean
python .claude/skills/microbrains/install_hook.py --context-guard    # block redundant/oversized whole reads + hung stdin / root search
python .claude/skills/microbrains/install_hook.py --dry-run    # preview the settings JSON without writing
python .claude/skills/microbrains/install_hook.py --self-test  # end-to-end probe of the installed gate
```

Console entrypoints installed by `pip install -e .`: `aetheris-hook`, `aetheris-doctor`, `aetheris-audit`, `aetheris-daemon`.

Note: `.claude/` control-plane files are protected by Layer 0 while enforce mode is active; operator maintenance of hook config or skills requires running from a user shell or setting `AETHERIS_ALLOW_POLICY_EDIT=1`.

Verify with a probe payload (expect `permissionDecision: "deny"` and exit code `2` for the root rm probe):

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' | python -m aetheris_microbrains.hook
```

If `aetheris_microbrains` is not installed (`python -c "import aetheris_microbrains"` fails),
install it first: `pip install -e .` from this repo (or `pip install aetheris-microbrains`).

### 2. Explain a blocked tool call

When a tool call was denied and the user (or you) needs to know why:

```bash
python -m aetheris_microbrains.audit --last      # most recent decision
python -m aetheris_microbrains.audit --tail 20   # last 20
python -m aetheris_microbrains.audit --summary   # aggregate
python -m aetheris_microbrains.audit --explain   # what/why/fix for the most recent decision
python -m aetheris_microbrains.audit --explain --find <sha-prefix>  # explain a specific decision
python -m aetheris_microbrains.audit --daily     # per-day digest with top reason codes
python -m aetheris_microbrains.audit --shadow-report  # enforce-mode projection + prompt-fatigue estimate (shadow installs)
python -m aetheris_microbrains.audit --digest    # deterministic I/O hotspot & friction triage tables
python -m aetheris_microbrains.audit --savings   # preserved-token & cost savings ledger (--statusline for 1-line bar)
```

Translate the `reason_code` + `label` into a one-sentence human explanation and the
concrete fix. Every entry has `input_sha256` and a `payload_preview` for correlation.

### 3. Pre-flight check a diff or command BEFORE applying it

For a proposed Python edit (in-process, no file touched):

```python
import aetheris_microbrains as mb
verdict = mb.evaluate_hook_payload({
    "tool_name": "Edit",
    "tool_input": {
        "file_path": "src/app.py",
        "old_string": "<existing code>",
        "new_string": "<proposed code>",
    },
})
print(verdict["decision"], verdict["reason_code"], verdict["reason"])
```

For a proposed shell command: same call with `{"tool_name": "Bash",
"tool_input": {"command": "<cmd>"}}`. `allow` -> safe to apply; `ask` -> escalate to
the user with `verdict["reason"]`; `deny` -> fix the code/command first. For ambiguous
diffs pass `file_content=` (full file text) via
`mb.evaluate_hook_payload(payload, file_content=...)` so the AST layer has context;
without it ambiguous cases safely degrade to `ask`.

### 4. Scan MCP servers & scrub outputs

```python
import aetheris_microbrains as mb
report = mb.scan_mcp_tools(<tool_schemas_list>)                 # returns dict: tool-schema risk report
inj = mb.detect_prompt_injection(<tool_output_text>)           # returns dict: injection_detected, severity, findings
scrubbed = mb.scrub_tool_output(<tool_output_text>)            # returns dict: secrets_found, secret_types, redacted_text
```

### 5. Health check & MCP pinning (doctor)

```bash
python -m aetheris_microbrains.doctor            # verify python/package/settings/matchers/policy/journal/pins/daemon
python -m aetheris_microbrains.doctor --json     # machine-readable report
python -m aetheris_microbrains.doctor --pin-mcp <server>    # trust a server, pinned to its current .mcp.json fingerprint
python -m aetheris_microbrains.doctor --unpin-mcp <server>  # remove the pin (tools fail closed to ask again)
python -m aetheris_microbrains.doctor --scan-mcp [--manifest-file tools.json] [--server <name>]  # scan .mcp.json & tool schemas for poisoning/rug-pulls
python -m aetheris_microbrains.doctor --slice <role>   # extract role-scoped sections from instruction markdown
python -m aetheris_microbrains.doctor --check-budget   # verify markdown character budget & placeholder hygiene
```

Pinned MCP servers get tiered structural verdicts: explicit read-only verbs (`get_*`, `list_*`, `read_*`, `search_*`, ...) allow, while mutating or unrecognized tool names still escalate. `.mcp.json` config drift or missing config invalidates the pin automatically (and manifest pins fail closed on modified or newly added unpinned tools).

### 6. Hook daemon (optional latency optimization)

```bash
python -m aetheris_microbrains.hook_daemon       # warm TCP evaluator on localhost; Ctrl+C to stop
python -m aetheris_microbrains.hook_daemon --http --port 8765  # native Claude Code type:"http" hook server (/v1/hook/pre, /v1/hook/post, /health)
```

The hook CLI queries the TCP daemon first and falls back to identical in-process evaluation on any failure - it is a pure optimization, never a dependency. Or use `--http` + `install_hook.py --http-daemon http://127.0.0.1:8765` to bypass per-call Python spawn overhead completely via native Claude Code HTTP hooks (`Authorization: Bearer $AETHERIS_DAEMON_TOKEN`; the installer emits `allowedEnvVars` so the env var is actually interpolated). Verdict memoization (5s TTL) only caches safe requests: pre-events without journal logging, non-shadow, non-edit payloads (journal completeness wins).

### Advanced flags (hook.py)

- `--host claude|codex|generic` - host contract (codex omits unsupported `ask`)
- `--profile strict|dev` - policy profile (or `AETHERIS_PROFILE`)
- `--policy <file>` - JSON/TOML/YAML PolicyConfig override (or `AETHERIS_POLICY_PATH`)
- `--journal-path <file>` / `AETHERIS_JOURNAL_PATH` - decision journal location
- `--session-state auto|<file>` - cross-turn SessionSecurityState IFC taint tracking (`auto` enabled by default in installer)
- `--replace-tool-output` (`AETHERIS_POST_REPLACE_OUTPUT=1`) - emit redacted `updatedToolOutput`/`updatedMCPToolOutput` in `PostToolUse`
- `--semantic-guard` (`AETHERIS_SEMANTIC_GUARD=1`) - enable optional split-conformal semantic prompt-injection classifier in `PostToolUse`
- `--sandbox-wrap auto|bwrap` - wrap approved shell commands in OS sandbox containment (`bwrap` on Linux when installed)
- `--context-guard` (`AETHERIS_CONTEXT_GUARD=1`) - block redundant unchanged whole reads, serve inline outlines on oversized reads, and block hung stdin / root-origin searches
- `--shadow` - observe-only (or `AETHERIS_HOOK_SHADOW=1`); `--auto-allow` - emit allow on safe verdicts

## Response contract

- `allow` -> proceed normally.
- `ask` (escalate) -> surface `reason` to the user and wait. Fail-closed: silence never grants permission.
- `deny` -> do NOT retry the same call. Read `reason_code`, fix the root cause
  (syntax, secret, protected path, destructive command), then re-submit the fixed version.
- Journal lives in the per-user state dir (`%LOCALAPPDATA%/aetheris-microbrains/decision_journal.jsonl`
  on Windows, `~/.local/state/...` on Linux; base dir override `AETHERIS_STATE_DIR`, file override
  `AETHERIS_JOURNAL_PATH`). Rotates to `.1` at 10 MiB. Never commit it.
