# MicroBrains

## An Adaptive Cognitive-Control, Security, and Efficiency Layer for Autonomous AI Agents

**Technical White Paper — Version 0.8**

**Author:** Dr. Dmitry Tikov  
**Date:** November 27, 2026  
**Project:** Aetheris Microbrains (`aetheris-microbrains`)  
**Repository:** <https://github.com/drtikov/microbrainsforagents>  
**License:** Apache-2.0  
**Status:** Active engineering release (v0.8) with Rust/WASM kernel, transactional rollback, subagent IAM, and cross-harness A/B telemetry

---

## Abstract

The rapid adoption of autonomous AI coding agents has revealed a structural systems bottleneck: increasing foundation-model capability does not automatically yield reliable, secure, or cost-efficient execution. In practice, autonomous agents expend substantial compute on repeated context ingestion, full tool-catalog serialization, failed shell/edit iterations, syntax errors, and multi-turn error-recovery loops, while running with broad local filesystem and credential permissions.

**Aetheris Microbrains** provides an external, model-independent cognitive-control, safety-gating, and cache-aware tool-routing plane positioned between an autonomous agent and its host execution environment. Rather than replacing or fine-tuning the underlying Large Language Model (LLM), Microbrains operates via sub-millisecond deterministic inspection—now backed by both a **62-module pure-stdlib Python security core** and a **native + `wasm32` Rust kernel (`aetheris-gate`)**—paired with AST diff caching, speculative workspace transactions, cryptographic subagent capability attenuation, deterministic repair hints, and lightweight CPU-only Mixture-of-Experts (MoE) tool narrowing with split-conformal abstention.

This white paper clearly distinguishes **implemented, deterministically tested system capabilities** from **preliminary live-agent research findings**. In the v0.8 cross-harness Stage 1–4 A/B benchmark (`420` calls across 7 hosts: `claude`, `cursor`, `cline`, `qwen`, `mcode`, `ouroboros`, `prime`), Microbrains achieves **100% hazard interception (`TPR = 1.0`)**, **0% false positives (`FPR = 0.0`)**, **1.18 ms p95 latency**, a **73.53% modeled token reduction**, and a **3.78× Engineering Efficiency ratio** ($E = \text{Verified tasks} / \text{Modeled cost}$). Separately, a preliminary two-worktree OpenAI Codex experiment observed a **58.3% reduction in total input tokens** (1,623,024 vs. 675,867) and a **61.1% reduction in uncached input tokens**. Because hook execution was not independently confirmed in that single exploratory trial's event stream, it serves as an exploratory signal rather than causal proof.

---

## 1. The Problem: Computational Inefficiency and Execution Risk in Autonomous Agents

Modern LLMs exhibit strong code generation and planning abilities, yet autonomous execution loops introduce compounding overhead and operational risk:

1. **Multi-Turn Failure Amplification:** A single syntax error, truncated diff, or malformed shell invocation forces execution failure, stack-trace ingestion, and 1–5 additional LLM recovery turns—re-sending the full conversation prefix each time.
2. **Tool-Catalog Context Bloat & Prefix-Cache Invalidation:** Exposing dozens or hundreds of MCP/tool schemas on every turn consumes thousands of input tokens. However, naively re-selecting top-$K$ tools per turn mutates the prompt prefix and **breaks provider prompt caching**, erasing most theoretical savings.
3. **Unchecked Machine, Credential & Multi-Agent Risk:** Agents running with local shell/file permissions can read `.env*` or `.ssh/` secrets, pipe credentials to external hosts, mutate agent hook/settings files to disable guardrails, ingest poisoned MCP tool schemas, or delegate overly broad permissions to spawned subagents.
4. **Latency & Token Cost of LLM-as-a-Judge:** Calling a secondary LLM to judge every tool call adds 500–2,500 ms latency and recursive token expenditure per step.

Crucially, these failures stem from the **surrounding execution architecture**, not merely from foundation-model weights. MicroBrains addresses five core architectural questions:

- Can deterministic local analysis (native Rust/WASM + stdlib Python) intercept syntax, supply-chain, and policy failures in ~1 ms before tools run?
- Can session-sticky tool routing preserve LLM prompt-cache prefixes while shrinking tool-schema bloat?
- Can one declarative `HostSpec` control plane govern 20 heterogeneous coding harnesses (`Claude Code`, `Codex CLI`, `Google Antigravity`, `OpenClaw`, `Qoder`, `Qwen Code`, `Hermes`, `DeepSeek Harness`, `Cursor`, `Devin`, `Cline`, `MiniMax Code`, `Ouroboros`, `Prime Agent`, etc.) without altering foundation models?
- Can speculative edits be made **transactional** (automatic rollback on validation failure or secret leak) and subagent delegations **cryptographically attenuated**?
- Can an external control layer measurably increase *verified successful engineering output per unit of compute*?

---

## 2. System Architecture (v0.8)

MicroBrains sits transparently in the tool-execution critical path via native host hooks, a local warm daemon (`TCP`/`UDS`/`HTTP/1.1`), a native + `wasm32` Rust kernel (`aetheris-gate` / `aetheris_gate.wasm`), and a Streamable HTTP / stdio MCP server (`aetheris-mcp`):

```text
                       USER PROMPT / TASK
                               │
                               ▼
          ┌────────────────────────────────────────┐
          │   HOST AGENT HARNESS (20 HostSpecs)    │
          │   Claude │ Codex │ Antigravity │ Qwen  │
          │   Cursor │ Cline │ Devin │ mcode │ ... │
          │   Ouroboros (SDK) │ Prime Agent (REPL) │
          └────────────────────┬───────────────────┘
                               │ PreToolUse / PostToolUse / before_tool_call
                               ▼
          ┌────────────────────────────────────────┐
          │   HOST ADAPTERS & LOW-LATENCY TRANSPORT│
          │   • rust/aetheris-gate (Native & WASM) │
          │     L0 + L1 Shell Unwrapper + DLP      │
          │   • hosts/specs.py (20-host registry)  │
          │   • hook.py + hook_daemon.py (IPC)     │
          └────────────────────┬───────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────┐
    │   4-LAYER DECISION ENGINE (62 Stdlib Core Modules) │
    │                                                    │
    │  Layer 0 — Constitutional Policy & Asset Gate      │
    │  • DEFAULT_PROTECTED_PATTERNS (.env*, .ssh/,       │
    │    .claude/, .codex/, .cursor/, .qoder/, .qwen/,   │
    │    .devin/, .clinerules/, .minimax/, .prime/)      │
    │  • Policy-as-Code (Cedar / Rego) + Signed Fleet    │
    │    Policy Bundles (fleet_bundle.py)                │
    │                                                    │
    │  Layer 1 — Structural AST, Shell, Taint & Tx Gate  │
    │  • Polyglot AST + SHA-256 LRU ASTDiffCache         │
    │  • Wrapper-aware shell tokenizer (Rust + Python)   │
    │  • Cross-turn IFC taint + Subagent Capability IAM  │
    │    (HMAC-SHA256 Macaroons, capability_tokens.py)   │
    │  • Speculative WorkspaceTransactionManager         │
    │                                                    │
    │  Layer 2 — Conformal Router & Tool Narrowing       │
    │  • Sticky SessionToolSurface (prefix-cache safe)   │
    │  • Pure-NumPy deterministic MoE & OOD abstention   │
    │                                                    │
    │  Layer 3 — Fail-Closed Escalation & Repair Hints   │
    │  • ask / force_ask / requireApproval               │
    │  • Deterministic repair_hint on deny/escalate      │
    │  • Clopper-Pearson threshold calibration           │
    └───────┬──────────────────────────┬─────────────────┘
            │ Allow / Sandbox Rewrite  │ Audit / Scrub
            ▼                          ▼
    ┌───────────────────┐    ┌───────────────────────────┐
    │ EXECUTION BOUNDARY│    │ OBSERVABILITY & GOVERNANCE│
    │ • OS Sandbox      │    │ • JSONL Decision Journal  │
    │   (Job / Landlock │    │ • Fleet Attestation       │
    │    Seatbelt / srt)│    │ • OTel GenAI Exporter     │
    │ • Secret Broker   │    │ • Local Audit Web UI      │
    │   ($SECRET:<ID>)  │    │ • SARIF / CI PR Gate      │
    │ • Auto-Rollback   │    │ • Sync PostToolUse DLP    │
    └───────────────────┘    └───────────────────────────┘
```

---

## 3. Architectural Principles

### 3.1 Model Independence & Strict Core Isolation

MicroBrains enforces strict separation between deterministic security enforcement and optional numerical/research components (verified continuously via `aetheris verify-core` -> `core_isolated=True`, `numpy_violations=[]`):

- **Security-critical core (Layers 0, 1, 3 — 62 modules):** Python standard library only. Pre-flight AST inspection, `ASTDiffCache`, shell tokenization, protected-path policies, IFC taint tracking, capability tokens, workspace transactions, repair hints, fleet bundles, and host hook adapters carry **zero runtime imports of NumPy or third-party ML packages**.
- **Portable Rust & WASM kernel (`rust/aetheris-gate`):** Implements Layer 0 path/host checks, single-pass Layer 1 shell tokenization + wrapper unwrapping (`layer1_shell.rs`), and secret/prompt-injection DLP (`dlp.rs`) in zero-Python Rust, compiled to both a native binary (`aetheris-gate`) and a `wasm32-unknown-unknown` library (`aetheris_gate.wasm` with C-ABI exports `aetheris_wasm_eval_pretool` and `aetheris_wasm_scrub`).
- **Cognitive-routing subsystem (Layer 2 — default package install):** Lightweight pure-NumPy CPU Mixture-of-Experts (`ToolRoutingMoE`), sticky `SessionToolSurface`, and split-conformal abstention gates.
- **Optional research & server modules (49 lab modules + `[mcp]` / `[server]` / `[ast]` / `[guard-ml]`):** Isolated in separate opt-in dependency profiles.

### 3.2 Allowlist-First, Fail-Closed Policy Evaluation
Agent actions are classified deterministically:
- **Permitted Read-Only / Dev Workflow (`allow`):** Verified safe commands under `strict` or `dev` policy profiles.
- **Unrecognized / Ambiguous Operations (`escalate`):** Unknown binaries, sensitive reads, and risky flags pause for human confirmation (`ask`, `force_ask`, or `requireApproval`) rather than failing open. Zero-prompt flags (`--auto-approve-escalations`) are restricted via an explicit allowlist.
- **Unattended Cron Safety:** When `context.trigger == "cron"`, interactive escalations convert automatically to hard blocks.
- **Catastrophic / Exfiltration / Self-Tampering (`deny`):** Destructive commands, syntax-broken diffs, `.env*`/secret writes, obfuscated dot-glob reads (e.g., `.[e]nv.local`), and write attempts against agent control planes or Microbrains' own running package tree hard-deny immediately.

### 3.3 Self-Healing Feedback & Transactional Workspace Safety
Blocking an agent without actionable guidance wastes retry turns. In v0.8:
- **Deterministic `repair_hint` (`repair_hints.py`):** Attached to `deny`/`escalate` verdicts in `evaluate_hook_payload`, classifying the failure category (e.g., `dangerous_shell`, `syntax_error`, `protected_path`) and returning the exact safe remediation so the LLM fixes its call on the next turn.
- **Speculative Workspace Transactions (`workspace_tx.py`):** `WorkspaceTransactionManager` and `run_speculative_transaction` snapshot touched files prior to multi-file edits and roll back modified, deleted, or newly created files automatically if post-edit validation fails or secrets leak.

### 3.4 Cryptographic Subagent Capability Attenuation
When an orchestrator spawns subagents, flat ambient authority is unsafe. `capability_tokens.py` implements Macaroon-style HMAC-SHA256 chained capability tokens (`mint_root_capability`, `attenuate_capability`, `verify_capability_and_authorize`) plus `propagate_subagent_taint`: child agents can **monotonically narrow** allowed paths, tools, and budgets offline without coordinator round-trips, and cannot escalate privileges.

### 3.5 Prompt-Cache Preservation & OS Defense-in-Depth
- **`SessionToolSurface`:** Preserves LLM KV prefix caches via sticky cluster pinning rather than mutating tool catalogs every turn.
- **OS Sandbox & Secret Broker:** Static AST/tokenizer checks pair with `--sandbox-wrap` (Windows Job Objects, Linux Landlock / `bwrap`, macOS Seatbelt, `srt`) and `$SECRET:<NAME>` Secret Broker (`aetheris-mcp-proxy`).
- **Signed Fleet Governance (`fleet_bundle.py`):** Organizations can sign anti-rollback policy bundles and verify hash-chained fleet attestations (`aetheris fleet-verify`).

---

## 4. Implemented Capabilities & Multi-Host Matrix

### 4.1 Declarative 20-Host Registry (`HostSpec`)

All host adapters are unified under `HOST_SPECS` in [specs.py](file:///i:/github%20microbrains/aetheris_microbrains/hosts/specs.py) with automated parity validation (`validate_host_specs_parity`). Key integrated platforms:

| Host Platform | Integration Mechanism | Maturity | Adapter & Installer | Key Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **Claude Code** | `PreToolUse` / `PostToolUse` / Lifecycle hooks (CLI & HTTP) | **Full E2E Tested** | `--host claude`, `install_hook.py` | Silent allow / ask / deny (exit 2), warm HTTP daemon, `updatedToolOutput` scrubbing |
| **Codex CLI** | `.codex/hooks.json`, skill & `.codex-plugin/` | **Full E2E Tested** | `--host codex`, `install_codex_hook.py` | Deferred escalation under auto-approve policy, `--codex-escalate-deny`, MCP support |
| **Google Antigravity** | Native plugin bundle (`plugin.json`, `hooks.json`, MCP, skill) | **Full E2E Tested** | `--host antigravity`, `aetheris install antigravity` | `allow`/`ask`/`force_ask`/`deny` contract, `--approve-once`, `aetheris-gate` shim option |
| **OpenClaw** | Gateway plugin (`@aetheris/openclaw-microbrains`) | **Full E2E Tested** | `--host openclaw`, `aetheris install openclaw` | `before_tool_call`, sync `tool_result_persist` DLP, `mcp_tool_registered` validation, HTTP/1.1 keep-alive |
| **Qoder (Desktop & CLI)** | `PreToolUse` / `PostToolUse` hooks (`~/.qoder/settings.json`, `.qoder/settings.local.json`) | **Full E2E Tested (Contract)** | `--host qoder`, `aetheris install qoder` | Non-destructive hook merge, `.qoder/` protection, `--session-state auto`, `--replace-tool-output` |
| **Qwen Code (`qwen`)** | `PreToolUse` / `PostToolUse` hooks (`~/.qwen/settings.json`, `.qwen/settings.local.json`) | **Hook & Installer Tested** | `--host qwen`, `aetheris install qwen` | Non-destructive merge, `timeout: 10`s, native `run_shell_command`/`edit`/`list_directory`, `additionalContext` DLP, `.qwen/` protection |
| **Hermes Agent** | FastMCP (`stdio` / Streamable HTTP + `GET /health`) + hooks | **Full E2E Tested** | `--host hermes`, `aetheris-mcp` | 8 MCP tools (`microbrain_health`, preflight, scrub, scan, narrow, audit, directives, verify) |
| **DeepSeek Harness** | Cordis MCP bundle | **On-Demand MCP (Advisory)** | `aetheris install dsh` | On-demand Streamable HTTP MCP utilities (`127.0.0.1:8766/mcp`) |
| **Cursor** | `.cursor/hooks.json` (`preToolUse`, `postToolUse`, `beforeShellExecution`, etc.) | **Hook v2 Tested** | `--host cursor`, `aetheris install cursor` | Native `cursor_hook_v2`, `--replace-tool-output` DLP, `--use-rust-shim` |
| **Devin** | `.devin/hooks.v1.json` / `%APPDATA%\devin\config.json` | **Hook & Installer Tested** | `--host devin`, `aetheris install devin` | Native `devin_hook_v1` (`decision: block` + exit 2), `DEVIN_PROJECT_DIR` detection, `.devin/` protection |
| **Cline (VS Code & CLI)** | `.clinerules/hooks/`, `~/.cline/hooks/` (`PreToolUse` / `PostToolUse` + `.ps1`) | **Hook & Installer Tested** | `--host cline`, `aetheris install cline` | Native `cline_hook_v1` (`cancel: true/false`), SEARCH/REPLACE diff AST inspection, `.clinerules/`/`.cline/` protection |
| **MiniMax Code (`mcode`)** | `.minimax/hooks.json` / `$MINIMAX_DATA_DIR/hooks.json` | **Hook & Installer Tested** | `--host mcode`, `aetheris install mcode` | Native `mcode_pretooluse_v1`, `commandWindows`, `PostToolUse` `additionalContext` DLP, `.minimax/` protection |
| **Ouroboros** | In-process `ToolRegistry` (`execute` / `execute_result` / `schemas`) | **SDK & Registry Tested** | `--host ouroboros`, `wrap_ouroboros_registry` | Native `ouroboros_registry_v1` mapping (`run_command`, `apply_patch`, `edit_batch`, `read_file`), reentrancy-safe gate, PostToolUse DLP, MoE narrowing |
| **Prime Agent (`prime`)** | In-process REPL wrappers (`rlm.bash`, `edit.run`, `mcp.call_tool`) | **SDK & Wrapper Tested** | `--host prime`, `install_prime_gate` | Sync/async REPL wrappers, `.prime/` protection, AST + shell pre-flight gate, fail-closed PostToolUse DLP |

### 4.2 Architectural Case Study: Contextual Shell Classification (Windows PowerShell)

In mixed Windows workflows, read-only PowerShell commands (`Get-Date`, `Get-Content README.md`, `Get-ChildItem | Select-Object Name`) risked generic escalations without dedicated PowerShell inspection, while naively allowlisting any `Verb-Noun` pattern would permit destructive cmdlets (`Invoke-Expression`, `Remove-Item -Recurse`, `-enc` payloads). By routing recognized PowerShell invocations through dedicated read-only cmdlet verification, scriptblock/subshell inspection, protected-path checks, and `env:` provider checks (now also mirrored in Rust `layer1_shell.rs`), MicroBrains blocks `.env`/`.ssh` reads while allowing verified safe commands cleanly.

---

## 5. Empirical Benchmarks & Cross-Harness Evaluation

### 5.1 v0.8 Cross-Harness Stage 1–4 A/B Suite (`aetheris benchmark`)

Using [benchmark_harness.py](file:///i:/github%20microbrains/aetheris_microbrains/core/benchmark_harness.py) (`python -m aetheris_microbrains.cli benchmark --trials 10 --json`), we evaluate **420 deterministic tool-call traces** (210 benign + 210 hazardous, 10 trials × 7 hosts) through the live `evaluate_hook_payload` pipeline with 95% Student-$t$ confidence intervals:

| Host | TPR | FPR | p95 Latency (ms) | Modeled Token Savings | Engineering Efficiency Ratio ($E_{\text{gated}}/E_{\text{base}}$) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `claude` | 1.00 | 0.00 | 1.19 ms | 73.54% | **3.78×** |
| `cursor` | 1.00 | 0.00 | 0.86 ms | 73.51% | **3.78×** |
| `cline` | 1.00 | 0.00 | 1.10 ms | 73.47% | **3.77×** |
| `qwen` | 1.00 | 0.00 | 1.29 ms | 73.55% | **3.78×** |
| `mcode` | 1.00 | 0.00 | 0.95 ms | 73.56% | **3.78×** |
| `ouroboros` | 1.00 | 0.00 | 1.20 ms | 73.55% | **3.78×** |
| `prime` | 1.00 | 0.00 | 1.22 ms | 73.56% | **3.78×** |
| **Aggregate (420 calls)** | **1.00** | **0.00** | **1.18 ms** (`0.55 ms` mean, `95% CI [0.47, 0.63]`) | **73.53%** (`2,072,520 → 548,510` modeled tokens) | **3.78×** (`$6.22 → $1.65` modeled cost) |

### 5.2 Reproducible Security, Routing & Latency Suites

All figures below come from deterministic benchmark scripts in the repository (`examples/`, `aetheris_microbrains/tests/test_skill_gain_ab.py`, `BENCHMARK_RESULTS.md`), using documented Sonnet-class input pricing (`$3.00` / 1M tokens) for modeled USD estimates:

| Suite / Scenario | MicroBrains Result | Baseline / Measured Trade-Off |
| :--- | :--- | :--- |
| **25-step schema & safety** (`05_skill_gain_ab_benchmark.py`) | **12,312 tokens** (−81.0%), **0** missed hazards, **0** false blocks | Baseline: 64,800 tokens; keyword rules: 8 missed, 4 false blocks |
| **26-step coding A/B** (`test_skill_gain_ab.py`) | **12/12** hazards caught, **0** syntax/security failures, **~37,055** tokens saved *(modeled)* | **\$0.1112/session** saved *(modeled)*; 146 → **37** error tokens (−74.7%); **16.5 ms** total gate |
| **16-step research A/B** (`11_claude_skills_efficiency_benchmark.py`) | **9,149 tokens** (−76.6%), **0/7** recovery turns, **0/4** incidents | **\$0.0896/session** saved (29,882 tokens); 8.75 ms mean / 107.79 ms p95 |
| **Agent Security & Adversarial** (`10_agent_security_benchmark.py`) | **22/22** attacks caught (0/12 FP); **100/100** adversarial blocked/escalated (0/100 FP) | **0.384 ms** p50 / **2.545 ms** p95; **100%** BFCL Top-1/3 (16/16) & OOD abstention (4/4) |
| **32-Tool Advisor** (`08_frozen_advisor_benchmark.py`) | **100% Recall@4**, **0/120** drops, **56.7%** cache-adj. saved *(session-sticky)* | Keyword advisor: 94.2% Recall@4, 7/120 drops, 46.6% saved *(per-turn narrowing saved only 3.9%)* |
| **Colony Routing** *(Negative / Mixed Result)* | 86.7% strict / 70.0% held-out, **0/50** missed esc., 1,271 µs p95 | Tuned rules: **89.2%** strict / **100%** held-out, 4/50 missed esc., **25.3 µs** p95 |

### 5.3 Preliminary Live Codex Comparative Experiment

An exploratory two-arm evaluation compared **OpenAI Codex CLI (0.172.0)** starting from commit `498e06a` on Python 3.10.6 in isolated Git worktrees under an identical programming task:

| Metric | Control | MicroBrains | Relative Difference |
| :--- | ---: | ---: | ---: |
| Total input tokens | 1,623,024 | 675,867 | **−58.3%** |
| Cached input tokens | 1,544,704 | 645,376 | **−58.2%** |
| Uncached input tokens | 78,320 | 30,491 | **−61.1%** |
| Output tokens | 6,490 | 4,389 | **−32.4%** |
| Reasoning output tokens | 2,021 | 1,183 | **−41.5%** |

**Methodological Limitations:** Single-task sample ($N=1$); hook invocations were not independently confirmed in the recorded event stream. Treated strictly as an exploratory hypothesis-generating signal (H1–H3).

---

## 6. Economic Implications & Engineering Efficiency

To avoid misleading single-dimension "token savings" claims, MicroBrains optimizes **Engineering Efficiency**:

$$E = \frac{\text{Verified successful tasks}}{\text{Total computational cost}}$$

| Claim | Current Evidence Status |
| :--- | :--- |
| Deterministic pre-flight hazard interception & AST linting (`1.18 ms` p95 across 7 hosts) | **Verified** in local, Rust, WASM & CI test suites |
| Prompt-cache preservation via sticky tool narrowing (`56.7%` vs `3.9%` per-turn) | **Measured** in simulation benchmark (`08_frozen_advisor`) |
| 73.53% modeled token reduction & `3.78×` $E$ across 7-host A/B traces | **Measured** in deterministic `benchmark_harness.py` (`420` calls) |
| 58.3% fewer input tokens in single Codex run | **Observed** in preliminary $N=1$ trial |
| Multi-task live LLM SWE-bench $E$ uplift (including `repair_hint` + rollback gains) | **Planned** (Stage 5 live-LLM evaluation) |

---

## 7. Development Roadmap (Implemented vs. Next Frontier)

- **Phase I — Foundation & Security Core *(Implemented)*:** Pure-stdlib security core (`verify-core` -> 62 core modules, 0 NumPy violations), 4-layer structural AST/shell/taint engine, gate self-write protection, allowlist-hardened escalation policies, Cedar/Rego Policy-as-Code, MCP pinning + `$SECRET:<NAME>` Secret Broker, OS sandboxing, OTel GenAI + Audit UI + CI PR gate.
- **Phase II — Multi-Host Interoperability *(Implemented)*:** Turnkey adapters/installers and SDK wrappers across Claude Code, Codex CLI, Google Antigravity, OpenClaw, Qoder, Qwen Code, Hermes, DeepSeek Harness, Cursor, Devin, Cline, MiniMax Code (`mcode`), Ouroboros (`ToolRegistry`), and Prime Agent (`prime`).
- **Phase III — Rust/WASM Kernel, Caching, Transactions & Subagent IAM *(Implemented in v0.8)*:** Single-pass Rust shell tokenizer (`layer1_shell.rs`) + DLP (`dlp.rs`) compiled to native `aetheris-gate` and `aetheris_gate.wasm`, SHA-256 `ASTDiffCache`, declarative 20-host `HostSpec` registry, speculative `WorkspaceTransactionManager` rollback, chained HMAC capability tokens (`capability_tokens.py`), and deterministic `repair_hint` + Clopper-Pearson threshold calibration (`repair_hints.py`).
- **Phase IV — Fleet Attestation & Cross-Harness A/B Suite *(Implemented in v0.8)*:** Signed anti-rollback org policy bundles + hash-chained attestation (`fleet_bundle.py`) and Stage 1–4 multi-host A/B benchmark evaluator (`benchmark_harness.py`).
- **Phase V — Live Multi-Task Validation & Hardware/WASM Runtime Enforcement *(Planned)*:** Full live-LLM SWE-bench A/B trials ($N \ge 10$ per task across Claude, Codex, Qwen, Ouroboros, Prime) measuring real `repair_hint` turn reduction and speculative-rollback token savings, plus in-process WASM execution inside browser/JS/Node agent hosts.

---

## 8. Architectural Comparison with Existing Approaches

| Approach | Primary Mechanism | Cache / Token Impact | Structural Security Scope | Typical Per-Step Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-based harness hooks** | Regex / static allowlists per harness | None (gating only) | Misses wrapped commands & AST/diff semantics unless custom-coded | `< 1 ms` |
| **LLM-as-a-judge guards** | Secondary foundation/guardrail LLM inference | Adds prompt + completion tokens per call | Broad semantic coverage, non-deterministic, vulnerable to prompt injection | `500–2,500 ms` |
| **Offline static linters** (`Bandit`, `Semgrep`) | Repo/file SAST scans (`CI` / batch) | None (post-write feedback loop) | Strong static rules, but lacks live agent tool gating, IFC taint, or credential redaction | `100 ms – seconds` |
| **Per-turn dynamic tool routers** | Embedding/keyword top-$K$ per prompt | Shrinks per-turn catalog, **invalidates KV prefix cache** (3.9% cache-adj. savings) | Routing only (no shell/AST gate) | `0.2–5 ms` |
| **Aetheris Microbrains (v0.8)** | Rust/WASM L0/L1/DLP + 62-mod stdlib AST/IFC + `HostSpec` + Tx rollback + Capability IAM + `SessionToolSurface` | **Sticky cluster pinning** (`56.7%` cache-adj.) + `repair_hint` + early veto (`73.53%` modeled A/B savings) | Pre-flight syntax, secret-egress, dot-glob, MCP pinning, subagent IAM, auto-rollback | `0.55 ms` mean / `1.18 ms` p95 in-process; `<15 ms` Rust shim |

---

## 9. Conclusion

Improving autonomous AI agents requires more than larger foundation models—it requires disciplined execution infrastructure around those models. With v0.8, **Aetheris Microbrains** evolves into a portable, self-healing, multi-agent control plane: combining a native + `wasm32` Rust inspection kernel, a 62-module pure-stdlib Python decision engine, transactional workspace rollback, cryptographic subagent capability attenuation, and signed fleet governance—achieving **100% hazard detection, 0% false positives, 1.18 ms p95 latency, and 3.78× modeled Engineering Efficiency** across 7 heterogeneous agent harnesses.

---

## Appendix A: Reproducibility & Verification Commands

```bash
# 1. Install with dev & MCP extras
pip install -e ".[dev,mcp]"

# 2. Verify pure-stdlib core isolation (62 core modules, 0 numpy violations)
python -m aetheris_microbrains.cli verify-core

# 3. Run unit & integration test suites
python -m pytest aetheris_microbrains/tests -q

# 4. Run Rust & WebAssembly verification
cargo test --manifest-path rust/aetheris-gate/Cargo.toml
cargo build --release --target wasm32-unknown-unknown --lib --manifest-path rust/aetheris-gate/Cargo.toml

# 5. Run cross-harness Stage 1-4 A/B benchmark (7 hosts, 420 calls)
python -m aetheris_microbrains.cli benchmark --trials 10

# 6. Verify host adapters & MCP server
python -m aetheris_microbrains.cli doctor
aetheris-mcp --check
```
