# Changelog

All notable changes to `aetheris-microbrains` are documented in this file.

## [Unreleased]

- **Security hardening:** Catch credentials split via Python string concatenation/`join`/f-strings and JS concatenation (`obfuscated_secret_concatenation`), JS/TS dynamic `require()`/`import()` (`js_dynamic_require`), `$IFS` shell obfuscation (`shell_ifs_obfuscation`), `find -exec`/`-delete` destructive targets, and dangerous inline interpreter calls (`perl -e`, `node -e`, `ruby -e`, `php -r`).
- **Claude Code `/microbrains` skill:** Added `.claude/skills/microbrains/SKILL.md` and portable `install_hook.py` (writes local `.claude/settings.local.json` with `shlex.quote` by default, or portable `--shared` `.claude/settings.json`), A/B session benchmark (`test_skill_gain_ab.py`), and Windows UTF-8 audit console fix.
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
