# Changelog

All notable changes to `aetheris-microbrains` are documented in this file.

## [Unreleased]

- **16-proposal operator experience set + hardening (Nov 2026):**
  - *Performance:* local hook daemon (`python -m aetheris_microbrains.hook_daemon`, A1) keeps a warm evaluator on localhost TCP with constant-time token-authenticated, `0o600` state (`hook_daemon.json` under `<state_dir>/hook_daemon.json`), client recv timeouts, CWD/host propagation, and version checks; the hook CLI queries it first and falls back to identical in-process evaluation on ANY failure (bypassing the daemon under `AETHERIS_ALLOW_POLICY_EDIT=1` or custom `AETHERIS_STATE_DIR`). Verdict memoization (A3) caches only safe-to-cache requests (pre-event, no journal logging, non-shadow, no session state, non-edit payloads including `apply_patch`) with a 5s TTL / 1024-entry LRU and policy/`.mcp.json` stat-signature keys. Post-scrub byte budget (A2/D4) bounds deep scanning of large tool outputs (4 MiB) with a `scrub_truncated` flag while still scanning bounded head/tail slices for indirect prompt injection.
  - *Policy & MCP:* MCP risk tiers (B1) — pin servers to their `.mcp.json` fingerprint (`doctor --pin-mcp` / `save_mcp_pin`); explicit read-only verbs allow structurally (`layer1_mcp_trusted_readonly`), mutating/unknown tool verbs escalate (`layer1_mcp_write_tool_escalate`), and unpinned/drifted/missing-`.mcp.json` servers fail closed (case-insensitive `.mcp.json` matching + stat-signature pin cache). Operator web-domain allowlist (B2, `allowed_web_domains`) exempts allowlisted `http`/`https` hosts/subdomains (without userinfo) from external-approval prompts at both layer0 and layer1. PowerShell surface (B3) — explicit read-only cmdlet allowlist with argument, `env:`/provider, and protected-path inspection, scriptblock/`$(...)`/backtick inspection, `Invoke-Expression`/`Start-Process`/encoded command prefix vetoes (`-enc`/`-en`/`-ec`), `Remove-Item -Recurse/-Force` (and abbreviated `-rec`/`-for`) vetoes, redirection checks, and unix-like deletion delegation. Opt-in import taint (B4, `PolicyConfig.import_taint_enabled=False` default) — when enabled, resolves diff imports depth-1 to project files (skipping PEP 562 lazy-loader `__init__.py` boilerplate) and escalates tainted imports (`layer1_imports_flagged_module`).
  - *Observability:* `doctor` health check (C1, `python -m aetheris_microbrains.doctor`) verifies Python `>= 3.10`/package/settings/matchers/policy/journal/MCP pins/daemon version+auth with `--pin-mcp`/`--unpin-mcp` management; `audit --explain`/`--find` (C2) gives canned what/why/fix explanations per reason code; `audit --daily` (C3) gives per-day digests; `audit --shadow-report` (C4) reads full journals and resolves `shadow_decision`/`shadow_reason_code` on real `--shadow` entries for accurate enforce-mode projections.
  - *Secrets:* base64/URL-safe base64/hex-encoded secret blobs decoded and redacted (`scrub_tool_output` and `redact_secrets`), plus credential key/value redaction (`api_key: …`, `password = …`) with placeholder-aware skips and POSIX `PWD=/path` exclusion (D1/D2).
  - *Installer:* `--dry-run` (E1), `--preset solo-dev|ci|paranoid` bundles with `--no-*` `BooleanOptionalAction` overrides (E2), `--self-test` end-to-end probe verifying both `PreToolUse` and `PostToolUse` matchers (D3).
  - *Documented by design (E3):* deleting a lock file is unsafe while a writer holds it — the gate denies broad lock-file deletion and `doctor` reports the rotation state instead.
- **Security hardening:** Catch credentials split via Python string concatenation/`join`/f-strings and JS concatenation (`obfuscated_secret_concatenation`), JS/TS dynamic `require()`/`import()` (`js_dynamic_require`), `$IFS` shell obfuscation (`shell_ifs_obfuscation`), `find -exec`/`-delete` destructive targets, and dangerous inline interpreter calls (`perl -e`, `node -e`, `ruby -e`, `php -r`).
- **Hook installer & gate hardening (Nov 2026):** Risk-matched Pre/PostToolUse matchers (`Bash|Edit|Write|MultiEdit|NotebookEdit|Read|WebFetch|WebSearch|mcp__.*`) with fail-closed `"onFailure": "block"` on the Pre hook (requires Claude Code >= 2.1.295), atomic validated settings writes, aetheris-only uninstall preserving unrelated hooks, synchronous PostToolUse scrubbing before context ingestion by default (`--async-post` opts out), cross-platform journal file locking (`fcntl`/`msvcrt`), secret-egress scanning of tool payload arguments (URLs included), `webfetch`/`websearch` gated by `require_approval_for_external` after egress checks, NotebookEdit cell AST linting, read-only `pip` introspection allowlist, extended edit-reconstruction languages (json/sh/js/ts/yaml/toml/Makefile/Dockerfile) with cwd-relative path resolution, and risk-level-prefixed escalation reasons.
- **Claude Code `/microbrains` skill:** Added `.claude/skills/microbrains/SKILL.md` and portable `install_hook.py` (writes local `.claude/settings.local.json` with cmd.exe-compatible double-quote quoting by default, or portable `--shared` `.claude/settings.json`), A/B session benchmark (`test_skill_gain_ab.py`), and Windows UTF-8 audit console fix.
- **Escalation UX & forensics:** Risk-tiered escalation contracts (`LOW`/`MEDIUM`/`HIGH`), AST `structural_context` (scope/line/column/symbol) preserved through vetoes, journal timestamps, and analytical `xargs` pipeline allowlists (`git ls-files | xargs grep`, `jq`, `yq`, `cut`, `tr`, `column`).
- **Repo hygiene & docs:** Added PEP 561 `py.typed` marker, CI `ruff` lint job (`E9,F63,F7,F82`), explicit out-of-scope threat boundaries in `SECURITY.md` and `README.md`, and subprocess cold-start, file-size scaling, and conformal calibration limit benchmarks in `README.md`.

## [0.7.6] - 2026-11-20

- Applied STRIDE and Round 3 security hardening (fail-closed server auth bind guard, timing-safe token digest, safe `AETHERIS_SERVER_MAX_BODY_BYTES` parsing, hook error-text credential redaction).
- Added raw shell executor tracking, direct RCE builtins (`eval`/`exec`/`source`), expanded source/sink pipe analysis, and end-to-end `hook.py` integration tests.
- Added MCP supply-chain security (`scan_mcp_tools`, `detect_prompt_injection`, `MCPManifestPinner`, `SessionSecurityState` cross-turn taint guard).
- Added GitHub Actions CI matrix (Python 3.10–3.12 + `pip-audit`) and Dependabot config.

## [0.7.5] - 2026-11-19

- Added shell wrapper unwrapping (`xargs`, `find -exec`, `env`, `sudo`, `timeout`, `nohup`), secret egress veto across tool arguments, and PostToolUse output scrubbing.

## [0.7.4] - 2026-11-19

- Added redacted `hook_error` shadow journal observability so hook failures are distinguishable from unchecked calls.

## [0.7.3] - 2026-11-19

- Added atomic versioned weight storage, prompt-cache-aware sticky `SessionToolSurface`, Codex hook host contract, and audit telemetry.

## [0.7.2] - 2026-11-18

- Added paired mutation audits, honest conformal calibration provenance fields, and FastAPI server extras.

## [0.7.1] - 2026-11-18

- Added journal secret redaction (`0600` file permissions), escalation timeout contracts, hunk-only status flags, and calibration-basis provenance.

## [0.7.0] - 2026-11-18

- Added optional robotics extension modules (`StateCBFSafetyFilter`, `VLActionGate`, `SensorimotorLoop`, `EFPAgent`, `PackedTernaryLinear`).
