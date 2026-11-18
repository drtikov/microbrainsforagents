# Aetheris Microbrains

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-purple.svg)](LICENSE)
[![Zero-GPU](https://img.shields.io/badge/Hardware-Pure_NumPy_%7C_OpenCL-success.svg)](https://github.com/)
[![Latency](https://img.shields.io/badge/Prefilter-<2ms-brightgreen.svg)](https://github.com/)

**Aetheris Microbrains** (`v0.7.0`) is a fast, lightweight safety gate and tool-routing coprocessor that sits directly in front of any autonomous AI agent — from coding agents (**Claude Code**, **SWE-agent**, **Codex**) to personal/OS assistants (**OpenClaw**, **Hermes**, or custom LLM loops).

It runs completely locally in pure NumPy on your CPU in **under 2 milliseconds**.

```text
       [ User / Task ]
              │
              ▼
    ┌───────────────────┐
    │     LLM Agent     │
    └─────────┬─────────┘
              │ Tool call / Diff / Shell command
              ▼
   ⚡ [ AETHERIS MICROBRAINS ]  ── (<2 ms local CPU pre-flight gate)
         │
         ├── 🛑 Blocked locally? (Syntax error, leaked secret, rm -rf, protected path)
         │       ↳ Instant reject. Zero cloud tokens wasted. Zero broken runs.
         │
         ├── 🔍 Ambiguous / Uninspected?
         │       ↳ Safely escalates ("ask") to human or full LLM review.
         │
         └── ✅ Verified safe?
                 ↳ Allowed to run.
```

---

## 🚀 What Is It (In Plain English)?

When an AI agent writes code or executes commands, it frequently wastes time and money:
1. **Broken Patches & Retries:** The agent makes a syntax error, leaks an API token, or invokes a destructive shell command (`rm -rf`, `curl | sh`). The runner executes it, tests fail, and the agent burns multiple expensive LLM turns just to read the traceback and retry.
2. **Context Window Bloat:** Passing 30+ tool definitions in every single turn burns thousands of prompt tokens per step and degrades model focus.

**Microbrains solves this at zero LLM cost:**
* **Instant Pre-Flight Safety Shield:** Validates code diffs with real AST parsing, checks for secrets, and parses shell commands *before* anything runs. Broken patches are intercepted in 1.3 ms.
* **Smart Tool Pruning:** Dynamically narrows large tool catalogs down to the Top-4 most relevant tools for the current step, saving **up to 39% of schema tokens** without starving the agent of needed capabilities.
* **Claude Code & Codex Ready:** Includes a standard `PreToolUse` hook CLI (`python -m aetheris_microbrains.hook`) that plugs right into agent workflows with full `hookSpecificOutput` compatibility.

> **What it is NOT:** Microbrains is **not** an LLM that writes code from scratch, and it does not replace an OS sandbox. It is an ultra-fast structural filter and router designed to make your agent faster, cheaper, and safer.

---

## 🤝 Compatible Agent Ecosystems

Microbrains is architecturally independent of any specific LLM provider or orchestration framework. Below is the exact integration status today:

| Category | Agents & Frameworks | Integration Status | What Microbrains Covers (and Does Not Cover) |
| :--- | :--- | :--- | :--- |
| **Claude Code & `PreToolUse` Harnesses** | **Claude Code** (≥ 2.1.295 for `onFailure: "block"`), **Codex** | **Built-in CLI Hook** (`python -m aetheris_microbrains.hook`) | Pre-flight AST/secret/shell filter on individual tool calls (`allow` passes through to user prompts, `ask` escalates, `deny` blocks). Not an OS/network sandbox. |
| **Coding Agents** | **SWE-agent** / **mini-SWE-agent**, **OpenHands**, **Aider**, **Cursor Agent** | **Python API / Custom Adapter** (`mb.decide(...)`) | Catches malformed syntax, leaked secrets, and destructive commands in diffs/commands before test execution. Does not audit full repos or verify test logic. |
| **Personal & Desktop OS Assistants** | **OpenClaw**, **Hermes Agent**, **Goose**, **Computer Use Agents** | **Python API / Custom Adapter** | Protects `.env`, `.git/hooks/`, `.ssh/` paths, blocks destructive shell chains, and escalates uninspected external calls to `"ask"`. Must be paired with OS/process permissions. |
| **Orchestration & MCP Hosts** | **LangGraph**, **CrewAI**, **AutoGen**, **MCP Clients** | **Python API / Custom Adapter** (`mb.narrow_tool_surface(...)`) | Shrinks tool-schema tokens per turn; `ask` is a client-side prompt (not a server-side MCP permission boundary), and schema pruning does not cap total steps or session spend. |

---

## 💡 Key Benefits

### 1. 💰 Lower API Bills & Fewer Wasted Turns
Catching malformed code, invalid syntax, and leaked keys locally in 1.3 ms prevents paying cloud token fees for broken execution cycles and costly model retries.

### 2. 🛡️ Strong Local Protection
Guards sensitive files (`.env*`, `.git/hooks/`, `.ssh/`, `credentials.json`), blocks obfuscated code injection (`eval`, `exec`, `getattr`, base64 execution), and vetoes destructive terminal commands before they hit your machine.

### 3. ⚡ Zero Lag (<2 ms CPU Latency)
Pure NumPy with zero GPU required. The checks run in microseconds to under 2 milliseconds on standard laptop hardware—adding virtually zero overhead to seconds-long LLM calls.

---

## 🔒 Cybersecurity & Defense-in-Depth Options

When AI agents browse the web, read external tickets/PRs, or connect to third-party MCP servers, they face indirect prompt injection, poisoned diffs, credential leaks, and rogue tool calls. Microbrains serves as an **additional pre-flight filter and audit journal** (paired with OS/container/network isolation—never as a standalone sandbox):

* **Sensitive Asset Protection (Layer 0):** Denies writes/modifications to `.env*`, `.git/hooks/`, `.git/config`, `.ssh/`, `id_rsa`, and `credentials.json`, and escalates `READ_ONLY` access on those paths to `"ask"`.
* **Obfuscated Payload & Secret Screening (Layer 1):** Inspects applied Python ASTs and diffs to block `eval`/`exec`, `os.system`, `subprocess(shell=True)`, `base64`-to-`exec`, `getattr` obfuscation, `__builtins__` tampering, and hardcoded API keys (`secret_leak_pattern_found`).
* **Allowlist-First Shell Guard:** Blocks destructive shell chains (`rm -rf`, `curl | sh`, `dd of=/dev/`, `DROP DATABASE`) and escalates unrecognized commands rather than letting them run unchecked.
* **Strict AST Allowlist (`--strict-ast` / `PolicyConfig(strict_ast_allowlist=True)`):** Escalates any Python diff importing modules outside `allowed_ast_imports` or invoking dynamic calls.
* **Uninspected External / MCP Tool Escalation:** Tools without inspected `cmd`/`diff` payloads (and not explicitly `READ_ONLY`) escalate to `"ask"` instead of auto-approving; `PolicyConfig(require_approval_for_external=True)` enforces confirmation on all external actions.
* **Audit & Safe Rollout (`--shadow`, `--log-decisions`):** Shadow mode logs what would be blocked to `memory/decision_journal.jsonl` without disrupting live workflows; hook errors fail closed with exit code `2` (use `"onFailure": "block"` in Claude Code ≥ `2.1.295`).

---

## 🌐 Beyond AI Agents: Other Use Cases & Readiness

Microbrains' deterministic AST/shell rules, pure-NumPy runtime, and split-conformal abstention engine also apply outside agent loops:

| Domain | Status Today | What's Ready vs. What's Needed |
| :--- | :--- | :--- |
| **CI/CD & Pre-Commit Gates** | **Ready out-of-the-box** | `decide()` and `analyze_code_diff()` already check Python ASTs, secrets, breaking API signatures, YAML/JS patterns, and shell commands in `< 2 ms`. |
| **Edge / Offline & Air-Gapped Systems** | **Ready out-of-the-box** | Pure NumPy + Python stdlib, tiny `.npz` weights (`allow_pickle=False`), zero GPU/internet required, and PEP 562 lazy imports. |
| **Ticket & Intent Triage (Support / IT Ops)** | **Engine ready, needs your data** | `ToolRoutingMoE.fit_from_traces()` + conformal abstention work on arbitrary text categories, **but** shipped weights target SWE/DevOps tasks—you must supply your own labeled traces. |
| **MLOps Drift & Auditable Intake** | **Telemetry ready, needs adapter** | `upper_error_bound_95`, `drift_detected`, `recalibrate_from_decision_journal()`, `input_sha256`, `shadow_mode`, and JSONL logging are built, **but** require your domain schemas and compliance integration. |
| **Robotics (High-Level LLM-to-ROS 2)** | **Conceptually compatible, needs adapter & data** | Fits the upper planner layer (ROS 2 skill narrowing via `narrow_tool_surface`, conformal `"ask"` on ambiguous instructions, script AST/shell gating) without using GPU VRAM; **not** for real-time servo/motor loops or hardware safety interlocks. |

---

## 🏗️ 4-Layer Decision Architecture

1. **Layer 0: Policy & Protected Paths**
   * Enforces live `available_tools` registries.
   * Protects sensitive paths (`.env*`, `.git/hooks/`, `.ssh/`, `credentials.json`) without false-positiving on normal application code (`src/env_loader.py`).
   * Supports `require_approval_for_external` and irreversible action guards.
2. **Layer 1: Structural AST, Secret & Allowlist-First Shell Gate**
   * Full-file reconstructed Python AST parsing (`ast.parse` applied over unified diffs), distinguishing harmless static data tables from high cyclomatic complexity changes.
   * Hardcoded API secret detection (`secret_leak_pattern_found`).
   * Allowlist-first `shlex` shell tokenizer (`ls`, `git log`, `pytest` allow; `rm -rf`, `curl | sh`, `dd of=/dev/` deny; unknown commands safely escalate).
   * Uninspected tools without diffs or commands escalate to `"ask"` rather than auto-approving.
3. **Layer 2: Tool Narrowing & Conformal Abstention**
   * Bilingual (EN/RU) SHA-256 char-trigram `ToolRoutingMoE` + `narrow_tool_surface(top_k=4)` shrinks 32+ tool registries before prompt assembly.
   * Calibrated micro-classifiers with empirical conformal prediction sets, exact Clopper-Pearson bounds, and cross-process decision journal drift detection.
4. **Layer 3: Honest Escalation**
   * Structured fallback returning `escalate` (in Claude Code hooks: `permissionDecision: "ask"`) with full context for human or LLM review.

---

## 📦 Installation

```bash
git clone https://github.com/drtikov/microbrainsforagents.git
cd microbrainsforagents
pip install -e .
```

Importing `aetheris_microbrains` is side-effect clean: it does **not** mutate `sys.path` or shadow top-level project modules (`core`, `tools`, `memory`, `evolution`).

*Optional OpenCL acceleration:*
```bash
pip install -e .[opencl]
```

---

## ⚡ Quickstart

### 0. Unified 4-Layer Decision Contract

```python
import aetheris_microbrains as mb

verdict = mb.decide({
    "tool": "edit_file",
    "path": "src/server.py",
    "diff": "def f():\n    os.system(cmd)\n",
})
print(verdict)
# {'decision': 'deny', 'label': 'dangerous_method:os.system', 'layer': 1, ...}
```

### 1. Microbrain Domain Consultation

```python
import aetheris_microbrains as mb

result = mb.microbrain_consult(text="syntax error in AST parser during code edit")
print("Selected Brain:", result["selected_brain"])
print("Confidence:", result["calibrated_confidence"])
print("Cal Samples:", result["n_calibration_samples"], "Valid Guarantee:", result["conformal_guarantee_valid"])
```

### 2. Deterministic Top-2 Mixture-of-Experts (MoE) Tool Routing

```python
import aetheris_microbrains as mb

router = mb.get_tool_routing_moe()
decision = router.route("inspect file contents and write safe patch to repository")

print("Primary Tool:", decision.selected_tool)       # -> repo_write (deterministic across processes)
print("Top-2 Experts:", decision.top_experts)
print("Conformal Passed:", decision.conformal_passed)
```

### 3. Pre-Commit Structural AST & Counterfactual Safety Gate

```python
import aetheris_microbrains as mb

sim = mb.get_counterfactual_pipeline()
outcome = sim.simulate_edit_impact(
    file_path="src/server.py",
    diff_text="--- a/server.py\n+++ b/server.py\n@@ -10,2 +10,3 @@\n+    eval(untrusted_user_input)\n"
)

if not outcome.allowed:
    print("Blocked:", outcome.status, outcome.veto_reason)
```

### 4. Holographic Weight-Space VSA Memory

```python
import aetheris_microbrains as mb

vsa = mb.get_weight_space_vsa()
nearest = vsa.find_nearest_domains("tool_routing_v2", top_k=3)
print("Nearest models in parameter space:", nearest)
```

### 5. Claude Code & Agent Harness PreToolUse Hook Integration

Microbrains integrates directly into **Claude Code**, **Codex**, or any custom agent framework as a pre-tool execution hook.

Add to your `~/.claude/settings.json` or project `.claude/settings.json` hook configuration (note: `onFailure: "block"` requires Claude Code ≥ `2.1.295`):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python -m aetheris_microbrains.hook --strict-ast",
            "onFailure": "block"
          }
        ]
      }
    ]
  }
}
```

Verify inside Claude Code that the hook is registered via `/hooks` (or test that `rm -rf /` is blocked with exit code `2` and `stderr` output).

**Key Hook Modes & CLI Flags:**
* **Prefilter Mode (Default):** On safe operations, the hook exits `0` silently. It does **not** bypass Claude Code's native user confirmation dialogs—it only intercepts dangerous or broken calls before the prompt is shown.
* **`--shadow` (or `AETHERIS_HOOK_SHADOW=1`):** Safe pilot/audit mode—logs what Microbrains *would* deny or escalate to `memory/decision_journal.jsonl` without blocking live execution.
* **`--auto-allow` (or `AETHERIS_HOOK_AUTO_ALLOW=1`):** Explicitly outputs `{"permissionDecision": "allow"}` in `hookSpecificOutput` for trusted environments where verified commands should run without user prompting.
* **Protected Paths Escalation:** Reading sensitive files (`.env`, `.ssh/id_rsa`, `credentials.json`) automatically returns `permissionDecision: "ask"`, while modifications are strictly denied.
* **Fail-Closed on Exceptions & Import Errors:** Unexpected runtime or import errors return exit code `2` with error details in `stderr`, preventing unsafe actions from running if the hook fails.
* **`--log-decisions` (or `AETHERIS_LOG_DECISIONS=1`):** Records decision telemetry to `memory/decision_journal.jsonl` across independent hook processes, enabling cross-process drift detection.

**Live Claude Code Acceptance Checklist (before production use):**
1. **Normal call (`ls -la`):** exits `0` silently and preserves Claude Code's normal user permission prompt.
2. **Escalation (`Read .env` or unknown MCP tool):** `ask` prompt reaches the user.
3. **Denial (`rm -rf /`):** blocked before execution, reason shown via `stderr`.
4. **Hook failure (`onFailure: "block"` on Claude Code ≥ `2.1.295`):** simulated hook crash blocks the tool call rather than failing open.

---

### 6. Allowlist-First Shell Gate, File-Context AST & Shadow Mode

```python
import aetheris_microbrains as mb

# Allowlisted read-only shell commands pass; unrecognized commands escalate;
# destructive chains (`rm -f -r ~`, `sh -c 'rm -rf /'`, `ls $(whoami)`) are denied.
print(mb.decide({"tool": "execute_shell", "cmd": "ls -la"})["decision"])                    # -> "allow"
print(mb.decide({"tool": "execute_shell", "cmd": "custom_deploy_binary --prod"})["decision"]) # -> "escalate"

# Resolve aliases declared in enclosing `file_content` + log shadow-mode decisions
verdict = mb.decide(
    {"tool": "edit_file", "path": "src/runner.py", "diff": "def f(c):\n    _o.system(c)\n"},
    context={
        "file_content": "import os as _o\n",
        "shadow_mode": True,
        "journal_path": "memory/decision_journal.jsonl",
    },
)
print(verdict["decision"], verdict["shadow_decision"]) # -> "escalate" "deny"
```

### 7. Dynamic Tool Registry (`available_tools`) & MCP-Zero Top-K Narrowing

```python
import aetheris_microbrains as mb

router = mb.get_tool_routing_moe()

# Constrain selection strictly to the active agent mode's tool surface (bilingual RU/EN)
res = router.route(
    "прочитай файл и напиши патч для репозитория",
    available_tools=["edit_file", "memory_recall", "web_fetch"],
)
print("Selected within surface:", res.selected_tool) # -> "edit_file"

# MCP-Zero hierarchical Top-K catalog narrowing (escalates full catalog on OOD)
narrowed = mb.narrow_tool_surface("fix bug and write repository patch", top_k=4)
print(narrowed["mode"], narrowed["candidate_tools"], narrowed["reduction_ratio"])
```

### 8. Mondrian Conformal Quantiles, SHA-256 Checksums & Staged Plasticity

```python
import aetheris_microbrains as mb

# Load weights with automatic SHA-256 integrity verification (`verify_checksum=True`)
brain = mb.AnalyticalMicroTransformer.load("neuro_code_edit_verifier")
inf = brain.infer_conformal(["syntax_tree_valid", "ast_node_function_def"], alpha=0.05, mondrian=True)
print("Mondrian Quantiles:", inf["per_class_quantiles"], "Std Err:", inf["coverage_std_error"])

# Staged continual plasticity: writes to `_candidates/`, requires holdout gate, supports `_rollback/`
pipeline = mb.ContinualLearningPipeline()
staged = pipeline.apply_plasticity_feedback("code_edit_verifier", ["new_ast_pattern"], promote=False)
# prom = pipeline.verify_and_promote_candidate("code_edit_verifier", holdout_data, min_coverage=0.95)
# pipeline.rollback_domain("code_edit_verifier")
```

### 8. AST Signature Blast Radius, Sliding-Window Drift & Adaptive Conformal (ACI)

```python
import aetheris_microbrains as mb

# Counterfactual AST interface/signature diffing + CausalWorldModel rollout
sim = mb.get_counterfactual_pipeline()
sig_check = sim.simulate_edit_impact(
    "src/math_ops.py",
    "--- a/src/math_ops.py\n+++ b/src/math_ops.py\n@@ -1,2 +1,2 @@\n-def compute(a, b, c):\n+def compute(a):\n     return a\n",
)
print("Allowed:", sig_check.allowed, sig_check.simulated_failure_modes)

# Sliding-window drift monitoring & Gibbs-Candès Adaptive Conformal Inference (ACI)
tel = mb.get_microbrain_telemetry()
q_next = tel.update_aci_quantile(current_q=0.05, had_error=True, alpha=0.05, gamma=0.02)
report = tel.generate_dashboard(window_size=100)
print("Window coverage:", report.sliding_window_coverage, "Drift:", report.drift_detected)
```

---

## 🎯 Scope, Where It Fits & Alternatives

* **Where Microbrains Fits:** A **single `pip install` (`numpy`-only), zero-GPU pre-flight bundle** for custom/local agents or harnesses that want AST diff checks, secret scanning, allowlist-first `shlex` shell gating, and Top-K tool narrowing in `< 2 ms` without wiring 4 separate external binaries/services.
* **Established Alternatives by Component:**

| Capability | Microbrains Approach | Established Specialized Alternatives |
| --- | --- | --- |
| Python code & diff checks | Full-file `ast.parse` + sink visitor + opt-in `PolicyConfig(strict_ast_allowlist=True)` | Bandit, Semgrep, Meta LlamaFirewall CodeShield |
| Secret detection | Diff literal & token-prefix scanner (`secret_leak_pattern_found`) | `gitleaks`, `trufflehog` |
| Shell allow/deny | Allowlist-first `shlex` parser (`unknown -> escalate`) | Harness permission rules / `PreToolUse` hooks (Claude Code, Codex) — or drop in `python -m aetheris_microbrains.hook` |
| Tool narrowing | `< 1 ms` bilingual SHA-256 char-trigram MoE + optional `embed_fn`/`tool_descriptions` hybrid | Anthropic deferred tool loading, `langgraph-bigtool` |

* **What Microbrains Are NOT For:**
  - Replacing general LLM reasoning on arbitrary natural language tasks (unknown prompts deterministically abstain and escalate to Layer 3).
  - Acting as a standalone OS sandbox: whatever static rules miss gets through unless isolated at the OS/container boundary.
* **License:** **PolyForm Noncommercial 1.0.0** — free for educational, research, and noncommercial use; commercial/work use requires a commercial license.

---

## 🧪 Testing & Reproducible Benchmarks (`v0.6.8`: 17 Suites — 203 Passed, 0 Skipped)

Run the full verification suite (zero monolith runtime skips):

```bash
pytest aetheris_microbrains/tests -q
```

### 1. External Tool-Drop Rate, Baseline Comparison & Live-LLM A/B (`examples/06_real_llm_ab_benchmark.py`)
Evaluates `narrow_tool_surface(top_k=K)` across 40 diverse external SWE/DevOps/BFCL-style tasks using real serialized 32-tool OpenAI JSON schemas (**`2,589` tokens/step**):

```bash
python examples/06_real_llm_ab_benchmark.py
# Optional live-LLM A/B comparison against any OpenAI-compatible / Ollama endpoint:
python examples/06_real_llm_ab_benchmark.py --live --model gpt-4o-mini
```

| Top-K | Recall@K | Needed-Tool Drop Rate | Escalated to Full | OOD Escalated | Gross Saved | Net Saved (After Retries) | Local p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 92.1% | 7.9% (3/38) | 19/38 | 1/2 | 46.7% | 35.1% | 1.132 ms |
| **4** | **97.4%** | **2.6% (1/38)** | **19/38** | **1/2** | **43.5%** | **39.7%** | **0.900 ms** |
| 6 | 97.4% | 2.6% (1/38) | 19/38 | 1/2 | 40.5% | 36.6% | 0.906 ms |
| 8 | 97.4% | 2.6% (1/38) | 19/38 | 1/2 | 37.5% | 33.6% | 0.916 ms |

**Baseline Comparison (Top-4 on 38 External Tasks, including Matched-Escalation Keyword Baseline):**

| Architecture | Recall@4 | Escalated | Dropped | Drop Rate | 95% CP Upper Drop | Net Saved (After Retries) | Local p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Keyword Top-4 | 76.3% | 0/38 | 9/38 | 23.7% | <= 37.7% | 52.1% | 0.740 ms |
| Naive Keyword Top-4 (Matched Esc.) | 97.4% | 19/38 | 1/38 | 2.6% | <= 11.9% | 39.6% | 0.612 ms |
| Embedding + Ridge Top-4 | 73.7% | 0/38 | 10/38 | 26.3% | <= 40.5% | 49.0% | 0.349 ms |
| **Microbrains Top-4 (Hybrid)** | **97.4%** | **19/38** | **1/38** | **2.6%** | **<= 11.9%** | **39.6%** | **6.495 ms** |

### 2. Local 25-Step Regression & Token-Accounting Smoke Test (`examples/05_reproducible_25step_benchmark.py`)
Deterministic offline trajectory checking local CPU gate latency, AST/shell/secret rules, and OOD escalation (no live LLM called):

```bash
python examples/05_reproducible_25step_benchmark.py
```

| Architecture | 25-Step Schema Tokens | Prefill Token Reduction | Dangerous Missed (FN) | Benign Blocked (FP) | Local p95 Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unfiltered 32-Tool Agent | 64,800 | 0.0% | — (runs unfiltered) | 0 | — (not measured) |
| Naive Keyword Baseline | 64,800 | 0.0% | 8 | 4 | 0.002 ms |
| **Aetheris Microbrains v0.6.8** | **12,312** | **81.0%** | **0** | **0** | **0.962 ms** |

### 3. External SWE-bench Lite (300 Real GitHub Issues) Zero-LLM Gate Benchmark (`examples/07_swebench_lite_gate_benchmark.py`)
Evaluates `decide()` and `narrow_tool_surface(top_k=4)` against all **300 real issues and gold patches** in `princeton-nlp/SWE-bench_Lite` (Astropy, Django, Matplotlib, Pytest, scikit-learn, SymPy, etc.):

```bash
python examples/07_swebench_lite_gate_benchmark.py
```

**Patch Gate (300 Real Gold File Diffs):**

| Metric (Denominator: 300 File Diffs / 60 Mutated Diffs) | Result |
| --- | ---: |
| Gold file diffs allowed (`layer1_structural_verified`) | **213 / 300 (71.0%)** |
| Partial hunks escalated without full `original_source` | **82 / 300 (27.3%)** |
| Gold file diff false-deny rate (`AST syntax FP: 0`) | **5 / 300 (1.67%)** |
| Synthetic mutation templates caught (6 patterns: syntax / secrets / `getattr` / `eval` / `shell=True` / `__builtins__`) | **60 / 60 (100.0%)** |
| Diff gate local latency (`p50` / `p95`) | **1.660 ms / 8.773 ms** |

**Tool-Surface Narrowing (300 Issue Prompts):**

| Metric (Denominator: 300 Issues / 127 Narrowed Prompts) | Result |
| --- | ---: |
| Prompts narrowed to Top-4 vs. full-catalog escalation | **127 / 300 narrowed**, **173 / 300 full-catalog** |
| SWE core tool retained across all prompts (including full-catalog) | **297 / 300 (99.0%)** (`36.7%` schema tokens saved) |
| Any SWE core tool retained within narrowed-only prompts (`mode == "narrowed"`) | **124 / 127 (97.6%)** (`edit_file` / `repo_read`: **103 / 127**) |

### 4. Drop-In Claude Code / Codex `PreToolUse` Hook CLI (`v0.7.0`)
Use Microbrains directly inside Claude Code, Codex, or custom agent hooks via stdin/stdout JSON. Output conforms to Claude Code's official `hookSpecificOutput` schema, while fatal denials and internal errors write structured reasons to `stderr` with exit code `2` (fail-closed):

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' | python -m aetheris_microbrains.hook --strict-ast
```

* **Prefilter-by-default:** On safe `allow` verdicts, the hook exits `0` with empty `stdout` so Claude Code's normal user permission prompts still apply (pass `--auto-allow` only if you explicitly want to bypass prompts). Reading protected files (`.env`, `.ssh/id_rsa`, `credentials.json`) escalates to `"ask"`.
* **Fail-closed recommendation:** Set `"onFailure": "block"` in your Claude Code hook configuration so environment/import errors also block safely.

```python
import aetheris_microbrains as mb

moe = mb.get_tool_routing_moe()
moe.fit_from_traces([
    {"query": "read source file config.py", "tool": "repo_read"},
    {"query": "commit staged bugfix changes", "tool": "git_commit"},
])
```

Also run the **de-leaked 120-query RU/EN + 20-query Held-Out Paraphrased SWE Ablation** (`aetheris_microbrains/tests/test_colony_swe_shadow.py`):
* **Standard 120 RU/EN Queries:** Hybrid Microbrain **106/120 (88.3%)** strict accuracy (**56/60 = 93.3%** RU) with **`0/50`** missed escalations vs. **`4/50`** missed on Pure Rules Baseline.
* **Unseen Held-Out Paraphrased Split (20 queries absent from rule substrings):** Hybrid Microbrain **14/20 (70.0%)** accuracy & **`0/6`** missed escalations vs. Pure Rules Baseline **5/20 (25.0%)** & `1/6` missed escalations.

Or target specific verification suites:

```bash
pytest aetheris_microbrains/tests/test_fable_review_fixes.py aetheris_microbrains/tests/test_reviewer_probe_suite.py aetheris_microbrains/tests/test_end_to_end_economics_and_drift.py aetheris_microbrains/tests/test_layer_ablation_and_baselines.py -v
```

### Independent Audit & Hardening Coverage:
* **`test_fable_review_fixes.py` (`test_fable_1`..`23`)**: Pins persistent `W_embed_anchor` & empirical diagonal Fisher EWC regularization, weight-space similarity in `WeightSpaceVSA.find_nearest_domains()`, dynamic caller `alpha`, honest `ESCALATED_TO_LLM` without an `llm_callback`, secret-leak detection, zero hardcoded OOD stems or leaked benchmark identifiers in `test_colony_swe_shadow.py`, `allow_pickle=False`, zero numbered placeholder vocab tokens across all 42 models, ridge-fitted `W_gate` + empirical per-expert quantiles in `ToolRoutingMoE`, non-polluting `save()` CWD hygiene, full-file patched AST verification, multi-feature ensemble tokens, generic non-`ouroboros` blast-radius scanning, live `available_tools` enforcement, Clopper-Pearson upper bounds + nonconformity drift monitoring, grounded `CascadeEnsembleRouter` verdicts (`escalation_payload` on OOD/unknown tools), bilingual RU/EN `IntentPersistenceTracker`, grounded `MetacognitiveCalibrator`, universal `.env`/`.git/hooks`/`.ssh` protection, static data-table vs. cyclomatic complexity scoring, `recalibrate_from_decision_journal()`, zero legacy `Proposal N` comments, and `n_cal = 120` expanded calibration + external Tool-Drop Rate / retry-adjusted benchmark validation.
* **Split-Conformal & Mondrian Coverage Discipline (`conformal_collector.py`)**: Disjoint **Train**, **120 Held-Out Calibration** ($n_{\text{cal}} \ge 1/\alpha - 1 = 19$ for $\alpha = 0.05 \implies 95\%$ nominal coverage, std error $\le 0.0197$), and **40 Held-Out Validation** splits across **all 42 clean-vocab models** (86 shipped store files, 33 distinct empirical `q_hat` quantiles) validated under strict prediction-set membership $Y \in C_\alpha(X)$, with SHA-256 weight checksums, coverage standard error $\sqrt{\alpha(1-\alpha)/(n+2)}$, Clopper-Pearson 95% upper failure bound, sliding-window drift detection, and Gibbs-Candès ACI adaptation.
* **Ablation, Baselines & Economics (`test_layer_ablation_and_baselines.py`, `test_end_to_end_economics_and_drift.py`)**: Layer 0 / 0+1 / 0+1+2 ablation, bilingual RU/EN routing vs. English keyword baseline, MCP-Zero Top-$K$ schema reduction ($\ge 75\%$), `< 2 ms` p95 decision latency, risk-class fail-closed error handling, `shadow_mode` journaling, and `_candidates/` $\to$ `_rollback/` continual learning.
* **Adversarial & Probe Suites (`test_reviewer_probe_suite.py`, `test_adversarial_safety_suite.py`, `test_colony_swe_shadow.py`)**: Diverse obfuscated Python sinks & secret leaks, realistic indented/`except`/`elif`/comment/`.ts`/`.json`/`.yaml`/`.md` diffs, real weight-space surgical causal inversion (no `MagicMock`), allowlist-first `shlex` shell parsing, and 120 bilingual (60 RU / 60 EN) SWE queries (`0/50` missed escalations).

---

## 📄 License & Commercial Inquiries

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

* **Noncommercial Use:** You may freely use, modify, and distribute this software for educational, research, and noncommercial purposes.
* **Commercial Use:** Commercial use requires a commercial license from the author.

For commercial licensing inquiries, please contact:
**Dmitry Tikov** — [drtikov@gmail.com](mailto:drtikov@gmail.com)
