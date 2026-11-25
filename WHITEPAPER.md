# MicroBrains

## An Adaptive Cognitive-Control and Efficiency Layer for Autonomous AI Agents

**Technical White Paper — Version 0.2**

**Author:** Dr. Dmitry Tikov  
**Date:** November 25, 2026  
**Project:** Aetheris Microbrains (`aetheris-microbrains`)  
**Repository:** <https://github.com/drtikov/microbrainsforagents>  
**License:** Apache-2.0  
**Status:** Active engineering release with experimental live-agent efficiency research

---

## Abstract

The rapid adoption of autonomous AI coding agents has revealed a structural systems bottleneck: increasing foundation-model capability does not automatically yield reliable or cost-efficient execution. In practice, autonomous agents expend substantial compute on repeated context ingestion, full tool-catalog serialization, failed shell/edit iterations, syntax errors, and multi-turn error-recovery loops.

**Aetheris Microbrains** investigates an architectural alternative: an external, model-independent cognitive-control, safety-gating, and cache-aware tool-routing layer positioned between the autonomous agent and its host execution environment. Rather than replacing or fine-tuning the underlying Large Language Model (LLM), Microbrains operates via sub-millisecond deterministic inspection (AST parsing, allowlist-first shell tokenization, information-flow taint tracking, secret egress prevention) paired with lightweight CPU-only Mixture-of-Experts (MoE) tool narrowing and split-conformal abstention.

This white paper clearly distinguishes **implemented, deterministically tested system capabilities** from **preliminary live-agent research findings**. Deterministic policy, AST, secret-redaction, and cache-sticky tool-routing benchmarks demonstrate 100% interception across the defined hazard cases in the reported test suites alongside substantial modeled token reductions. Separately, a preliminary two-worktree OpenAI Codex experiment observed a **58.3% reduction in total input tokens** (1,623,024 vs. 675,867) and a **61.1% reduction in uncached input tokens**. However, hook execution was not independently confirmed in the recorded event stream of that single exploratory trial, so it does not establish causal attribution or general cross-task efficiency gains. We propose a unified evaluation metric—**Engineering Efficiency** ($E = \text{Verified successful tasks} / \text{Total computational cost}$)—and a five-stage experimental program to test these hypotheses rigorously across diverse agent platforms.

---

## 1. The Problem: Computational Inefficiency and Execution Risk in Autonomous Agents

Modern LLMs exhibit strong code generation and planning abilities, yet autonomous execution loops introduce compounding overhead and operational risk:

1. **Multi-Turn Failure Amplification:** A single syntax error, truncated diff, or malformed shell invocation forces execution failure, stack-trace ingestion, and 1–5 additional LLM recovery turns—re-sending the full conversation prefix each time.
2. **Tool-Catalog Context Bloat & Prefix-Cache Invalidation:** Exposing dozens or hundreds of MCP/tool schemas on every turn consumes thousands of input tokens. However, naively re-selecting top-$K$ tools per turn mutates the prompt prefix and **breaks provider prompt caching**, erasing most theoretical savings.
3. **Unchecked Machine & Credential Risk:** Agents running with local shell/file permissions can read `.env*` or `.ssh/` secrets, pipe credentials to external hosts, mutate agent hook/settings files to disable guardrails, or ingest poisoned MCP tool schemas containing indirect prompt injection.
4. **Latency & Token Cost of LLM-as-a-Judge:** Calling a secondary LLM to judge every tool call adds 500–2,500 ms latency and recursive token expenditure per step.

Crucially, these failures stem from the **surrounding execution architecture**, not merely from foundation-model weights. This raises five architectural questions that guide MicroBrains:

- Can deterministic local analysis intercept syntax, supply-chain, and policy failures in sub-millisecond time before tools run?
- Can session-sticky tool routing preserve LLM prompt-cache prefixes while still shrinking tool-schema bloat?
- Can one control plane govern heterogeneous coding harnesses (`Claude Code`, `Codex CLI`, `Google Antigravity`, `OpenClaw`, `Qoder`, `Hermes`, `DeepSeek Harness`, `Cursor`) without altering foundation models?
- Can safety mechanisms remain resistant to evasion (shell wrappers, obfuscated globs, self-tampering) and fail closed?
- Can an external control layer measurably increase *verified successful engineering output per unit of compute*?

---

## 2. System Architecture

MicroBrains sits transparently in the tool-execution critical path via native host hooks, a local warm daemon (`TCP`/`UDS`/`HTTP/1.1`), an optional Rust shim (`aetheris-gate`), and a Streamable HTTP / stdio MCP server (`aetheris-mcp`):

```text
                       USER PROMPT / TASK
                               │
                               ▼
          ┌────────────────────────────────────────┐
          │   HOST AGENT HARNESS                   │
          │   Claude Code │ Codex CLI │ Antigravity│
          │   OpenClaw    │ Qoder     │ Hermes/DSH │
          └────────────────────┬───────────────────┘
                               │ PreToolUse / PostToolUse / before_tool_call
                               ▼
          ┌────────────────────────────────────────┐
          │   HOST ADAPTERS & LOW-LATENCY TRANSPORT│
          │   • rust/aetheris-gate (Layer-0 + IPC) │
          │   • hook.py (--host adapters)          │
          │   • hook_daemon.py (TCP/UDS/HTTP/1.1)  │
          └────────────────────┬───────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────┐
    │   4-LAYER DECISION ENGINE (Stdlib L0/L1/L3 +       │
    │                            Optional NumPy L2)      │
    │                                                    │
    │  Layer 0 — Constitutional Policy & Asset Gate      │
    │  • DEFAULT_PROTECTED_PATTERNS (.env*, .ssh/,       │
    │    .claude/, .codex/, .cursor/, .qoder/, live pkg) │
    │  • Policy-as-Code (Cedar / Rego) + Secret scan     │
    │                                                    │
    │  Layer 1 — Structural AST, Shell & Taint Gate      │
    │  • Polyglot AST (Python, JS/TS, Go, Rust, SQL...)  │
    │  • Deferred-exec guard (Makefile, CI, package.json)│
    │  • Wrapper-aware allowlist shlex + glob matcher    │
    │  • Cross-turn IFC taint (SessionSecurityState)     │
    │                                                    │
    │  Layer 2 — Conformal Router & Tool Narrowing       │
    │  • Sticky SessionToolSurface (prefix-cache safe)   │
    │  • Pure-NumPy deterministic MoE & OOD abstention   │
    │                                                    │
    │  Layer 3 — Fail-Closed Supervisor Escalation       │
    │  • ask / force_ask / requireApproval               │
    │  • Auto-deny on unattended cron / timeout          │
    └───────┬──────────────────────────┬─────────────────┘
            │ Allow / Sandbox Rewrite  │ Audit / Scrub
            ▼                          ▼
    ┌───────────────────┐    ┌───────────────────────────┐
    │ EXECUTION BOUNDARY│    │ OBSERVABILITY & GOVERNANCE│
    │ • OS Sandbox      │    │ • JSONL Decision Journal  │
    │   (Job / Landlock │    │ • OTel GenAI Exporter     │
    │    Seatbelt / srt)│    │ • Local Audit Web UI      │
    │ • Secret Broker   │    │ • SARIF / CI PR Gate      │
    │   ($SECRET:<ID>)  │    │ • Sync PostToolUse DLP    │
    └───────────────────┘    └───────────────────────────┘
```

---

## 3. Architectural Principles

### 3.1 Model Independence & Explicit Dependency Boundaries

MicroBrains enforces strict separation between deterministic security enforcement and optional numerical/research components (verified continuously via `aetheris verify-core` -> `core_isolated=True`, `numpy_violations=[]`):

- **Security-critical core (Layers 0, 1, 3 — 35 modules):** Python standard library only. Pre-flight AST inspection, shell tokenization, protected-path policies, IFC taint tracking, audit journaling, and host hook adapters carry **zero runtime imports of NumPy or third-party ML packages** (enabling `<1 ms` warm hook imports without loading numerical stacks).
- **Cognitive-routing subsystem (Layer 2 — default package install):** Lightweight pure-NumPy CPU Mixture-of-Experts (`ToolRoutingMoE`), sticky `SessionToolSurface`, and split-conformal abstention gates. Note that `pyproject.toml` currently includes `numpy>=1.22.0` in base dependencies so Layer-2 routing works out of the box, while the security core remains strictly NumPy-free at runtime.
- **Optional research & server modules (49 lab modules + `[mcp]` / `[server]` / `[ast]` / `[guard-ml]`):** Isolated in separate opt-in dependency profiles so headless security hooks remain minimal and portable across LLM providers.

### 3.2 Allowlist-First, Fail-Closed Policy Evaluation
Agent actions are classified deterministically across five operational classes:
- **Permitted Read-Only / Dev Workflow (`allow`):** Verified safe commands under `strict` or `dev` policy profiles.
- **Unrecognized / Ambiguous Operations (`escalate`):** Unknown binaries, sensitive reads, and risky flags pause for human confirmation (`ask`, `force_ask`, or `requireApproval`) rather than failing open. Zero-prompt flags (`--auto-approve-escalations`) are restricted via an **explicit allowlist** (`_AUTO_APPROVE_ELIGIBLE_REASON_CODES` plus non-credential config-plane reads) so unknown commands or destructive git flags (`git clean -fdx`) are never silently approved.
- **Unattended Cron Safety:** When `context.trigger == "cron"` (e.g., OpenClaw scheduled runs), interactive escalations convert automatically to hard blocks.
- **Catastrophic / Exfiltration / Self-Tampering (`deny`):** Destructive commands, syntax-broken diffs, `.env*`/secret writes, obfuscated dot-glob reads (e.g., `.[e]nv.local`, `.???.local`, `.*.local`), and write attempts against agent settings or Microbrains' own running package tree hard-deny immediately.

### 3.3 Layered Defense Beyond Naive Regex
Regex-only filters are easily bypassed via wrapper chaining (`env`, `sudo`, `xargs`, `timeout`, `find -exec`), Unicode smuggling, or AST-level obfuscation. Layer 1 performs wrapper unwrapping, shell/newline/subshell parsing, full patched-file AST analysis, cross-turn Information Flow Control (IFC) taint tracking, and synchronous `PostToolUse` DLP secret scrubbing.

### 3.4 Prompt-Cache Preservation Before Greedy Pruning
Naive top-$K$ tool narrowing invalidates transformer KV prefix caches across turns. MicroBrains introduces `SessionToolSurface` with sticky cluster pinning so tool definitions stay stable within a session, sacrificing minimal token count per turn in exchange for preserving high cache-hit rates.

### 3.5 Defense-in-Depth Boundary (Not a Standalone OS Sandbox)
Static AST and tokenizer analysis checks syntax and paths before execution; it does not eliminate OS-level symlink/TOCTOU races or runtime dynamic variable indirection. MicroBrains therefore exposes an opt-in OS containment wrapper (`--sandbox-wrap` via Windows Job Objects, Linux Landlock / `bwrap`, macOS Seatbelt, or `srt`) alongside an out-of-band `$SECRET:<NAME>` Secret Broker (`aetheris-mcp-proxy`) so raw credentials never enter model contexts.

---

## 4. Implemented Capabilities & Multi-Host Case Study

### 4.1 Multi-Host Integration Matrix

| Host Platform | Integration Mechanism | Maturity | Adapter & Installer | Key Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **Claude Code** | `PreToolUse` / `PostToolUse` / Lifecycle hooks (CLI & HTTP) | **Full E2E Tested** | `--host claude`, `install_hook.py` | Silent allow / ask / deny (exit 2), warm HTTP daemon, `updatedToolOutput` scrubbing |
| **Codex CLI** | `.codex/hooks.json`, skill & `.codex-plugin/` | **Full E2E Tested** | `--host codex`, `install_codex_hook.py` | Deferred escalation under auto-approve policy, `--codex-escalate-deny`, MCP support |
| **Google Antigravity** | Native plugin bundle (`plugin.json`, `hooks.json`, MCP, skill) | **Full E2E Tested** | `--host antigravity`, `aetheris install antigravity` | `allow`/`ask`/`force_ask`/`deny` contract, `--approve-once`, `aetheris-gate` shim option |
| **OpenClaw** | Gateway plugin (`@aetheris/openclaw-microbrains`) | **Full E2E Tested** | `--host openclaw`, `aetheris install openclaw` | `before_tool_call`, sync `tool_result_persist` DLP, `mcp_tool_registered` validation, HTTP/1.1 keep-alive |
| **Qoder (Desktop & CLI)** | `PreToolUse` / `PostToolUse` hooks (`~/.qoder/settings.json`, `.qoder/settings.local.json`) | **Full E2E Tested (Config/Contract)** | `--host qoder`, `aetheris install qoder` | Non-destructive hook merge, `.qoder/` protection, `--session-state auto`, `--replace-tool-output` |
| **Hermes Agent** | FastMCP (`stdio` / Streamable HTTP + `GET /health`) + hooks | **Full E2E Tested** | `--host hermes`, `aetheris-mcp` | 8 MCP tools (`microbrain_health`, preflight, scrub, scan, narrow, audit, directives, verify) |
| **DeepSeek Harness** | Cordis MCP bundle | **On-Demand MCP (Advisory)** | `aetheris install dsh` | On-demand Streamable HTTP MCP utilities (`127.0.0.1:8766/mcp`; no automatic tool hook interception) |
| **Cursor** | `.cursor/hooks.json` (`beforeShellExecution` / `afterShellExecution`) | **Partial (Shell-Only Hook)** | `--host cursor`, `aetheris install cursor` | Shell execution pre/post CLI hook gating; file-edit/MCP interception and Cursor-live runtime remain unverified |

### 4.2 Architectural Case Study: Contextual Shell Classification (Windows PowerShell)

A concrete design challenge illustrates MicroBrains' approach to command governance. In mixed Windows developer workflows, read-only PowerShell commands such as:

```powershell
Get-Date
Get-Content README.md
Get-ChildItem | Select-Object Name
```

risked falling back to unrecognized generic shell escalations when not integrated with PowerShell-specific inspection paths—creating prompt fatigue without improving security. Conversely, simply allowlisting any `Verb-Noun` token pattern would silently permit state-mutating or destructive cmdlets (`Invoke-Expression`, `Remove-Item -Recurse`, encoded `-enc` payloads).

By routing recognized PowerShell invocations through dedicated read-only cmdlet verification, scriptblock/subshell inspection, protected-path checks, and sensitive environment checks (`env:` providers), MicroBrains preserves escalation on unknown commands and protected reads (`cat .env`, `Get-Content .ssh/id_rsa`) while avoiding unnecessary friction on verified safe commands.

---

## 5. Empirical Benchmarks & Preliminary Live Experiment

### 5.1 Reproducible Local Benchmarks (Verified Suites)

All figures below come from deterministic benchmark scripts in the repository (`examples/`, `aetheris_microbrains/tests/test_skill_gain_ab.py`, `BENCHMARK_RESULTS.md`). We report datasets, sample sizes, modeled pricing assumptions, and negative/trade-off results transparently:

- **Provenance & Modeled Pricing Assumptions:** `05_skill_gain_ab_benchmark.py` (25 synthetic steps), `test_skill_gain_ab.py` (26 deterministic steps: 14 benign + 12 seeded hazards—4 syntax errors, 8 security incidents), and `11_claude_skills_efficiency_benchmark.py` (16-step deterministic research replay) directly measure gate latency and interception/false-positive counts, while USD/token savings are **modeled estimates** using documented Sonnet-class input pricing (`$3.00` / 1M tokens) in each script. `10_agent_security_benchmark.py` evaluates 22 curated attack payloads + 12 benign controls; adversarial test suites evaluate 100 dangerous + 100 safe cases; `06_real_llm_ab_benchmark.py` evaluates 40 tool-routing tasks; `08_frozen_advisor_benchmark.py` evaluates 120 turns over a 32-tool catalog.

**Safety, Interception & Session Economics**

| Suite / Scenario | MicroBrains Result | Baseline Comparison / Gate Cost |
| :--- | :--- | :--- |
| **25-step schema & safety** (`05_skill_gain_ab_benchmark.py`) | **12,312 tokens** (−81.0%), **0** missed hazards, **0** false blocks | Baseline: 64,800 tokens; keyword rules: 8 missed, 4 false blocks |
| **26-step coding A/B** (`test_skill_gain_ab.py`) | **12/12** hazards caught, **0** syntax/security failures, **~37,055** tokens saved *(modeled)* | **\$0.1112/session** saved *(modeled)*; 146 → **37** error tokens (−74.7%); **16.5 ms** total gate |
| **16-step research A/B** (`11_claude_skills_efficiency_benchmark.py`) | **9,149 tokens** (−76.6%), **0/7** recovery turns, **0/4** incidents *(deterministic replay)* | **\$0.0896/session** saved (29,882 tokens); 8.75 ms mean / 107.79 ms p95 |
| **Agent Security & Adversarial** (`10_agent_security_benchmark.py` + `tests/`) | **22/22** attacks caught (0/12 FP); **100/100** adversarial blocked/escalated (0/100 FP); MCP poison/shadow/rug-pull caught | **0.384 ms** p50 / **2.545 ms** p95; **100%** BFCL Top-1/3 (16/16) & OOD abstention (4/4) |

**Tool Routing, Cache-Sticky Narrowing & Latency Trade-Offs**

| Benchmark | MicroBrains | Best Baseline | Measured Trade-Off |
| :--- | :--- | :--- | :--- |
| **Tool Selection Top-4** (`06_real_llm_ab_benchmark.py`) | **97.4% Recall@4**, **2.6%** drop rate, 38.6% net saved, **4.526 ms** p95 | Naive keyword: 76.3% Recall@4, 43.4% net saved, 0.270 ms p95 | **+21.1 pt recall**, −4.8 pt net savings, ~16.8× slower |
| **32-Tool Advisor** (`08_frozen_advisor_benchmark.py`) | **100% Recall@4**, **0/120** drops, **56.7%** cache-adj. saved *(session-sticky)*, **4.979 ms** p95 | Keyword advisor: 94.2% Recall@4, 7/120 drops, 46.6% saved, 0.194 ms p95 | **+5.8 pt recall**, **+10.1 pt** cache-adj. savings, ~25.7× slower *(per-turn narrowing saved only 3.9%)* |
| **Colony Routing** *(Negative / Mixed Result)* | 86.7% strict / 70.0% held-out, **0/50** missed esc., 13/70 false esc., 1,271 µs p95 | Tuned rules: **89.2%** strict / **100%** held-out, 4/50 missed esc., **4/70** false esc., **25.3 µs** p95 | Zero missed escalations, but lower strict accuracy and ~50× slower than tuned rules |
| **Payload-Size Scaling** | Small shell / 10-line diff: **0.14–0.70 ms** p95 | 100-line diff: **5.5 ms** p95; 480-line diff: **31.5 ms** p95 / **90.8 ms** p99 | Still negligible vs. 5–20 s LLM turns, though `<2 ms` applies to small inputs/warm daemon HTTP (`0.7 ms`), not max-size diffs |

**Latency by Execution Mode**

| Execution Mode | Scope Measured | Observed Latency |
| :--- | :--- | :--- |
| **Warm in-process decision engine** | `decide()` / hook library call (small shell/diff) | **0.07–0.70 ms** (`31.5 ms` p95 at 480-line AST diff) |
| **Warm local daemon / IPC** | Transport (`HTTP/1.1` or `TCP`) + SHA256 + evaluation | **~0.7–0.9 ms** (OpenClaw HTTP) / **~8.7 ms** p50 (TCP + disk log) |
| **Cold host subprocess hook** | Full process spawn + evaluation per tool call | **~400 ms** (Python CLI) / **<15 ms** (Rust `aetheris-gate` shim) |

---

### 5.2 Preliminary Live Codex Comparative Experiment

#### Experimental Setup
An exploratory two-arm evaluation compared **OpenAI Codex CLI (0.172.0)** starting from commit `498e06a` on Python 3.10.6 in isolated Git worktrees under an identical programming task (refining Windows PowerShell command classification while preserving security boundaries):
- **Control Configuration:** Codex CLI without MicroBrains hooks.
- **Treatment Configuration:** Codex CLI with MicroBrains hooks and command-inspection modifications enabled.

#### Observed Token Consumption

| Metric | Control | MicroBrains | Relative Difference |
| :--- | ---: | ---: | ---: |
| Total input tokens | 1,623,024 | 675,867 | **−58.3%** |
| Cached input tokens | 1,544,704 | 645,376 | **−58.2%** |
| Uncached input tokens | 78,320 | 30,491 | **−61.1%** |
| Output tokens | 6,490 | 4,389 | **−32.4%** |
| Reasoning output tokens | 2,021 | 1,183 | **−41.5%** |

#### Implementation Comparison & Caveats
Code comparison showed the MicroBrains-assisted run added 61 lines (40 core lines reusing existing PowerShell policy sets + 21 test lines) versus 32 lines in Control (16 core + 16 test using a broader pattern).

**Methodological Limitations (Explicitly Stated):**
1. **Single-Task Sample ($N=1$):** One completed task per arm cannot establish average savings across software engineering tasks.
2. **Event-Stream Attribution:** Hook invocations were not independently logged in the recorded trace stream, and the two arms diverged in implementation trajectory.
3. **Exploratory Signal Only:** These measurements motivate controlled multi-task experiments; they are **not** a statistically validated general speedup claim.

#### Hypotheses Motivated by This Signal
- **H1 (Context Efficiency):** Pre-flight architectural governance can substantially reduce redundant context growth and error-recovery churn in live agent loops.
- **H2 (Operational Efficiency):** Policy-aware classification and sticky tool surfaces reduce wasted tool calls without harming implementation quality.
- **H3 (Model Independence):** Meaningful efficiency improvements are achievable in the harness control plane without retraining foundation models.

---

## 6. Economic Implications & Target Efficiency Metric

If reproducible across larger task suites, reducing input-token expenditure per task shifts capacity constraints for both API-billed teams and rate-limited individual subscriptions (where a 58.3% input reduction corresponds to ~2.4× illustrative input-token capacity—though actual task throughput also depends on output tokens, prompt-cache hit rates, wall-clock runtime, and task pass rates).

To avoid misleading single-dimension "token savings" claims, commercial and scientific evaluations should optimize **Engineering Efficiency**:

$$E = \frac{\text{Verified successful tasks}}{\text{Total computational cost}}$$

| Claim | Current Evidence Status |
| :--- | :--- |
| Deterministic pre-flight hazard interception & AST linting (<2 ms on typical calls) | **Verified** in local & CI test suites |
| Prompt-cache preservation via sticky tool narrowing (`56.7%` vs `3.9%` per-turn) | **Measured** in simulation benchmark (`08_frozen_advisor`) |
| 58.3% fewer input tokens in single Codex run | **Observed** in preliminary $N=1$ trial |
| Cross-task average token savings / 2× verified engineering throughput | **Not yet established** (requires Stage 1–5 evaluation) |

---

## 7. Proposed Multi-Agent Evaluation Framework

To test H1–H3 rigorously without selection bias or safety degradation, we propose evaluating identical agent configurations across fixed task benchmarks while recording:

| Metric Category | Specific Measurements |
| :--- | :--- |
| **Effectiveness & Correctness** | Verified task pass rate, automated test-suite pass rate, code-review quality |
| **Token & Cache Economics** | Total input tokens, cached vs. uncached input tokens, output + reasoning tokens, USD cost |
| **Execution Overhead** | Wall-clock time, hook/daemon latency (p50/p95/p99), tool-call count, recovery turns |
| **Safety & Integrity** | Security regression rate, secret-egress prevention, prompt-injection containment |

**Five Evaluation Stages:**
1. **Stage 1 — Controlled Reproducibility:** Repeat Codex & Claude Code A/B runs ($N \ge 10$ trials per task) with verified hook-execution telemetry.
2. **Stage 2 — Task Diversity:** Benchmark across debugging, feature addition, refactoring, security hardening, and repo-scale navigation (e.g., SWE-bench subsets).
3. **Stage 3 — Cross-Harness Interoperability:** Compare identical tasks across Claude Code, Codex CLI, OpenClaw, Qoder, and Antigravity.
4. **Stage 4 — Statistical Rigor:** Report distributions, medians, confidence intervals, and variance alongside negative results.
5. **Stage 5 — Independent Reproduction:** Publish trace bundles and open harnesses for third-party verification.

---

## 8. Development Roadmap (Implemented vs. Planned)

- **Phase I — Foundation & Security Core *(Largely Implemented)*:** Pure-stdlib 35-module security core (`verify-core`), 4-layer structural AST/shell/taint engine, gate self-write protection, allowlist-hardened escalation policies, Cedar/Rego Policy-as-Code, MCP pinning + `$SECRET:<NAME>` Secret Broker, Rust native shim + OS sandboxing, OTel GenAI + Audit UI + CI PR gate.
- **Phase II — Multi-Host Interoperability *(Implemented & Expanding)*:** Turnkey adapters/installers for Claude Code, Codex CLI, Google Antigravity, OpenClaw (HTTP/1.1 keep-alive + DLP), Qoder Desktop/CLI, Hermes FastMCP, DeepSeek Harness, and Cursor.
- **Phase III — Large-Diff Performance & Engine Modularization *(Planned)*:** Single-pass shell tokenization (eliminating duplicate `shlex` scans), cross-turn AST caching for 400+ line diffs, and decomposing `layered_decision_engine.py` into focused sub-modules (policy, shell, AST, paths, journal).
- **Phase IV — Validated Multi-Agent Efficiency Benchmarks *(In Research)*:** Large-$N$ multi-harness A/B harnesses measuring $E = \text{Verified successful tasks} / \text{Total computational cost}$ with verified hook telemetry and adaptive policy calibration from audited outcomes.

---

## 9. Open Research Questions

1. What fraction of autonomous coding-agent token waste stems from the execution architecture versus foundation-model planning errors?
2. How much redundant reasoning can pre-flight structural feedback prevent before diminishing returns set in?
3. When should lightweight deterministic rules (`<30 µs`) strictly supersede neural/hybrid routing layers?
4. Do token and turn savings from sticky tool surfaces and AST pre-filtering transfer cleanly across heterogeneous model families (Claude, GPT/Codex, Gemini, DeepSeek)?
5. How can execution-time OS containment (`aetheris-gate sandbox`) and static AST pre-flight checks best co-evolve to eliminate residual symlink/indirection gaps?

---

## 10. Collaboration Opportunities

MicroBrains welcomes engineering review and joint empirical evaluation from teams working on:
- Autonomous coding-agent harnesses and IDE agent runtimes.
- Inference economics, KV prompt-cache optimization, and token budgeting.
- AI agent security, MCP supply-chain integrity, and OS sandboxing.
- Standardized, reproducible agent efficiency benchmarks.

---

## 11. Architectural Comparison with Existing Approaches

The table below highlights structural differences between MicroBrains and established categories of agent tooling rather than claiming universal superiority outside benchmarked workloads:

| Approach | Primary Mechanism | Cache / Token Impact | Structural Security Scope | Typical Per-Step Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-based harness hooks** | Regex / static allowlists per harness | None (gating only) | Misses wrapped commands & AST/diff semantics unless custom-coded | `< 1 ms` |
| **LLM-as-a-judge guards** | Secondary foundation/guardrail LLM inference | Adds prompt + completion tokens per call | Broad semantic coverage, non-deterministic, vulnerable to prompt injection | `500–2,500 ms` |
| **Offline static linters** (`Bandit`, `Semgrep`) | Repo/file SAST scans (`CI` / batch) | None (post-write feedback loop) | Strong static rules, but lacks live agent tool gating, IFC taint, or credential redaction | `100 ms – seconds` |
| **Per-turn dynamic tool routers** | Embedding/keyword top-$K$ per prompt | Shrinks per-turn catalog, **invalidates KV prefix cache** (only 3.9% cache-adj. net savings in our benchmark) | Routing only (no shell/AST gate) | `0.2–5 ms` |
| **Aetheris Microbrains** | Pre-flight AST/shlex/IFC gate + sticky `SessionToolSurface` + Rust/HTTP daemon across 8 hosts | **Sticky cluster pinning** preserves prefix caches (`56.7%` cache-adj. savings in 32-tool sim) + early syntax/hazard veto | Pre-flight syntax, secret-egress, dot-glob, MCP pinning, cron fail-closed (complements OS sandboxes) | `0.1–0.7 ms` in-process / `0.7–8.7 ms` daemon (`~400 ms` cold Python / `<15 ms` Rust) |

---

## 12. Conclusion

Improving autonomous AI agents requires more than larger foundation models—it requires disciplined execution infrastructure around those models. Aetheris Microbrains demonstrates that a lightweight, model-independent cognitive-control layer can enforce specified structural policy boundaries, implement tested mechanisms for detecting and blocking defined secret-exfiltration paths, preserve prompt-cache prefixes via sticky tool surfaces, and integrate cleanly across multiple agent ecosystems with sub-millisecond typical overhead. While preliminary live-agent results show compelling token-reduction signals, the project commits to rigorous, multi-task empirical validation centered on **verified engineering output per unit of compute**.

---

## Appendix A: Reproducibility & Verification Commands

```bash
# 1. Install with dev & MCP extras
pip install -e ".[dev,mcp]"

# 2. Verify pure-stdlib core isolation (35 core modules, 0 numpy violations)
aetheris verify-core

# 3. Run unit & integration test suite
pytest aetheris_microbrains/tests -q

# 4. Reproduce core security & routing benchmarks
python aetheris_microbrains/tests/test_skill_gain_ab.py
python examples/10_agent_security_benchmark.py
python examples/08_frozen_advisor_benchmark.py

# 5. Verify MCP server & host adapters
aetheris-mcp --check
aetheris install antigravity --self-test
aetheris install openclaw --self-test
aetheris install qoder --self-test
```
