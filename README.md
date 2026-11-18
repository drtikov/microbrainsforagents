# Aetheris Microbrains

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-purple.svg)](LICENSE)
[![Zero-GPU](https://img.shields.io/badge/Hardware-Pure_NumPy_%7C_OpenCL-success.svg)](https://github.com/)
[![Latency](https://img.shields.io/badge/Prefilter-<2ms-brightgreen.svg)](https://github.com/)

**Aetheris Microbrains** is a super-fast, lightweight safety guard and tool router for AI agents (like **Claude Code**, **Codex**, **SWE-agent**, or custom LLM bots). 

Think of it as an **instant pre-flight check or an "airbag" for your AI**: before an agent runs a command or applies code to your computer, Microbrains inspects it in under **2 milliseconds on your CPU**—stopping syntax bugs, leaked passwords, and dangerous commands before they ever happen.

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
         ├── 🛑 Destructive or broken? (Syntax error, leaked API key, rm -rf, .env leak)
         │       ↳ Intercepted immediately. Zero wasted tokens. Zero machine damage.
         │
         ├── ❓ Ambiguous / Risky?
         │       ↳ Escalates ("ask") to the human or LLM supervisor for confirmation.
         │
         └── ✅ Verified safe?
                 ↳ Allowed to run.
```

---

## 🚀 What Is It (In Plain English)?

When AI agents work on your code or execute terminal commands, two common problems happen:

1. **Wasted Time & Money from Broken Actions:**  
   The agent generates code with a syntax typo, leaks a secret token, or runs a destructive shell command (`rm -rf`, `curl | sh`). The computer crashes or tests fail, and the agent spends 3–5 expensive, slow LLM turns just reading errors and trying to fix itself.
2. **Cluttered Context Windows:**  
   Sending 30+ different tool definitions in every prompt wastes thousands of tokens per step, runs up your API bill, and confuses the model.

### How Microbrains Helps:
* 🛡️ **Instant Safety Shield:** It checks code diffs using real Python syntax trees and scans shell commands *before* execution. If a patch has a syntax error or a destructive command, it stops it in **1.3 ms** on your CPU.
* ✂️ **Smart Tool Trimming:** Shrinks huge lists of tools down to the top 4 most relevant tools for the current step, saving **up to ~39% in prompt tokens**.
* 🔌 **Drop-in Ready:** Plugs straight into **Claude Code** and **Codex** using a single hook command (`python -m aetheris_microbrains.hook`).
* ⚡ **Runs on Any Laptop:** Written in pure Python and NumPy with zero GPU or cloud dependencies.

> **What it is NOT:** Microbrains is **not** an LLM itself, and it doesn't write code from scratch. It is a guardrail and router that makes your existing agents faster, safer, and cheaper to run.

---

## 🤝 Compatible Agent Ecosystems

Microbrains works alongside any AI agent or framework:

| Category | Agents & Frameworks | Integration Status | What Microbrains Does |
| :--- | :--- | :--- | :--- |
| **Claude Code & `PreToolUse` Hooks** | **Claude Code** (≥ 2.1.295), **Codex** | **Built-in CLI Hook** (`python -m aetheris_microbrains.hook`) | Pre-flight inspection on every tool call (`allow`, `ask`, or `deny`). Blocks before execution. |
| **Autonomous Coding Agents** | **SWE-agent**, **OpenHands**, **Aider**, **Cursor Agent** | **Python API / Adapter** (`mb.decide(...)`) | Catches syntax errors, leaked credentials, and dangerous shell commands before tests run. |
| **Personal & Desktop OS Assistants** | **OpenClaw**, **Hermes Agent**, **Goose**, **Computer Use** | **Python API / Adapter** | Guards `.env`, `.ssh/`, and `.git/hooks/`, blocks disk formatting, and pauses unknown actions. |
| **Orchestrators & MCP Hosts** | **LangGraph**, **CrewAI**, **AutoGen**, **MCP Clients** | **Python API / Adapter** (`mb.narrow_tool_surface(...)`) | Prunes tool schemas per turn to cut token costs without starving agents of tools. |

---

## 💡 Key Benefits

### 1. 💰 Lower API Bills & No Wasted Loops
Catching broken syntax, missing variables, or leaked keys locally in 1.3 ms stops the agent from executing broken code and spending $0.10–$0.50 per retry turn.

### 2. 🛡️ Protects Your Machine & Secrets
Shields sensitive files (`.env`, `.ssh/`, `credentials.json`), prevents destructive commands (`rm -rf`, `DROP DATABASE`), and catches hidden code execution tricks (`eval`, `base64` decode into `exec`).

### 3. ⏱️ Blazing Fast (<2 ms Latency)
Runs in pure NumPy on standard CPUs. You won't notice a 1.5 millisecond check when your LLM takes 5 to 20 seconds to reply.

### 4. 🧭 Reliable Escalations, Honest Statistics & Audit Trail (`v0.7.1`–`v0.7.3`)
- **Safe Timeout Contracts:** When an action needs human review (`ask`), an explicit timeout is set (default: 300s). If nobody answers, it **fails closed** (`default_on_timeout: "deny"`)—silence never grants permission.
- **Zero-Fabrication Error Bounds & Calibration Provenance:** `decide()`, `calibrate_conformal()`, and `infer_conformal()` emit `upper_error_bound_95: null` and `upper_error_bound_rule_of_3: null` (`statistical_confidence_status: "unknown"`) unless measured on an explicit labeled evaluation set. Fission-derived models honestly declare `conformal_basis: "fission_synthetic_anchors"` with `conformal_guarantee_valid = False` until calibrated on genuine holdouts; `ToolRoutingMoE` explicitly marks synthetic anchors as `"synthetic_lexical_probes"`.
- **Nuanced Shell & AST Inspection:** Differentiates dangerous redirects (`> /etc/passwd`, `> .env`, `> ../escaped`) which are strictly denied from unprivileged redirects (`echo "ok" > out.txt`) which escalate for confirmation, while allowing standard descriptor piping (`2>&1`, `> /dev/null`). Real optional Tree-sitter grammars are dynamically verified before reporting AST validation.
- **Decoupled Advisors & Cache-Safe Sessions:** Standalone non-authorizing entrypoints `diagnose_patch()` (patch diagnostics) and `rank_tools()` / `SessionToolSurface` (sticky per-session Top-K narrowing that avoids LLM prompt-cache invalidation across turns), plus `evaluate_topk_holdout()` for real Top-K recall audits.
- **Clean Fleet Registry & Dynamic Variance:** Fleet lifecycle registry reports `null` and `"unobserved"` for models without runtime telemetry instead of placeholder defaults. Counterfactual simulations dynamically track empirical variance thresholds. Test probe models are safely isolated in `_archive/` leaving 41 pure production models.
- **Atomic Versioned Storage & Live Telemetry:** `AnalyticalMicroTransformer.save()` commits immutable `_versions/{name}/{rev_id}/` snapshots via a single atomic `.manifest.json` pointer swap; `/dashboard` renders live `MicrobrainTelemetry` (with illustrative preview isolated on `/demo/dashboard`), and `decision_journal.jsonl` redacts secrets and surfaces `journal_written` / `journal_error`.
- **Clear File Context Checks:** Distinguishes full-file verified diffs from single partial hunks (`layer1_hunk_only_checked`), so your agent always knows whether it inspected complete file context.

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
| **Robotics (High-Level LLM-to-ROS 2)** | **Middle-layer modules included (`v0.7.0+`), needs ROS 2 adapter & data** | Includes `StateCBFSafetyFilter`, `VLActionGate`, `SensorimotorLoop`, `EFPAgent`, and `PackedTernaryLinear` for the 10–50 Hz planner/VLA layer without GPU VRAM; **not** for real-time 1 kHz+ servo/motor loops or hardware safety interlocks. |

---

## 🤖 Robotics Extension Layer (10–50 Hz VLA / Planner Middleware)

For high-level robot autonomy (between ~1 kHz MCU/RTOS servo loops and tens-of-ms VLA/LLM calls), the SDK includes six pure-NumPy robotics modules:

| Component | What It IS | What It Is NOT |
| :--- | :--- | :--- |
| **`StateCBFSafetyFilter`** (`core/state_cbf_filter.py`) | Exact Euclidean projection onto affine CBF halfspaces $\cap$ control box via Dykstra's algorithm, with conformal barrier inflation and `verified_safe=False` fail-closed on infeasibility. | Not a hard real-time C++/RTOS safety controller or certified hardware E-stop. |
| **`VLActionGate` & `SkillLibrary`** (`core/vla_gating.py`) | Three-tier gate (`local_reflex` / `call_vla` / `escalate_human`) composing CBF feasibility, fail-closed conformal risk (`uncalibrated` never allows), and VSA skill affordances (with `scene_prototype` / `bind_scene_example`). Also wired at `POST /microbrain/gate`. | Not a learned SayCan feasibility value function unless backed by your scene prototypes/traces. |
| **`SensorimotorLoop` & `nir_export`** (`core/sensorimotor_loop.py`, `core/nir_export.py`) | Fixed-dt multi-rate tick scheduler with `ReservoirReflexLayer` (FORCE/RLS online readouts on `LiquidStateReservoir`) and guarded NIR-dialect graph export. | Not an ISO-certified real-time firmware loop or full Neuromorphic Intermediate Representation compiler. |
| **`EFPAgent` & `GenerativeModel`** (`core/pymdp_efe.py`) | Discrete POMDP generative model (`A`/`B`/`C`/`D`) computing Expected Free Energy (pragmatic + epistemic mutual-information terms) and exact Bayes belief updates. | Not a continuous-control MPC solver. |
| **`PackedTernaryLinear`** (`core/ternary_edge_runtime.py`) | 2-bit packed `.npz` ternary weights with atomic saves, exact integer parity with `BitLinearLayer`, latency receipts (`bench_receipt`), and guarded ONNX export. | Not a custom CUDA/NPU kernel. |

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
* **`--shadow` (or `AETHERIS_HOOK_SHADOW=1`):** Safe pilot/audit mode—logs what Microbrains *would* deny or escalate to `memory/decision_journal.jsonl` without blocking live execution. Empty stdin, malformed JSON, non-object payloads, and engine/import errors still return `0` with empty `stdout` **and** append a redacted stdlib-only `event_type: "hook_error"` entry to the journal so hook failures are never indistinguishable from unchecked calls.
* **`--log-decisions` / `--journal-path` (`AETHERIS_LOG_DECISIONS=1`, `AETHERIS_JOURNAL_PATH`):** Persist hook telemetry (including shadow `hook_error` events) to a custom `decision_journal.jsonl` path across independent hook processes for pilot audits and drift detection.
* **`--auto-allow` (or `AETHERIS_HOOK_AUTO_ALLOW=1`):** Explicitly outputs `{"permissionDecision": "allow"}` in `hookSpecificOutput` for trusted environments where verified commands should run without user prompting.
* **Protected Paths Escalation:** Reading sensitive files (`.env`, `.ssh/id_rsa`, `credentials.json`) automatically returns `permissionDecision: "ask"`, while modifications are strictly denied.
* **Fail-Closed on Exceptions & Import Errors (Non-Shadow):** Unexpected runtime or import errors return exit code `2` with redacted details in `stderr`, preventing unsafe actions from running if the hook fails.

**Live Claude Code & Codex Acceptance Checklist (before production use):**
1. **Normal call (`ls -la`):** exits `0` silently and preserves the host's normal user permission prompt.
2. **Escalation (`Read .env` or unknown MCP tool):** on Claude Code, `ask` prompt reaches the user; on **Codex** (`--host codex`), verify a harmless escalated call under **both** Codex approval settings (*prompt-on* vs. *auto-approve*), because exit `0` + empty `stdout` (`defer_to_host_policy`) leaves the decision to Codex's configured policy rather than forcing a prompt under auto-approve (enable `--codex-escalate-deny` if escalations must hard-block).
3. **Denial (`rm -rf /`):** blocked before execution (`exit 2`), redacted reason shown via `stderr`.
4. **Hook failure / `--shadow` error audit (`onFailure: "block"` on Claude Code ≥ `2.1.295`):** non-shadow hook errors fail closed; in `--shadow` pilots, confirm malformed/engine-error payloads record `event_type: "hook_error"` inside `decision_journal.jsonl`.

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

# Decoupled non-authorizing advisors & cache-safe sticky session narrowing (v0.7.3)
ranked = mb.rank_tools("inspect source file settings.py", available_tools=["repo_read", "edit_file", "run_tests"], top_k=2)
patch_diag = mb.diagnose_patch("src/app.py", "def ok():\n    return 1\n")
session = mb.SessionToolSurface(["repo_read", "edit_file", "run_tests", "git_status", "docker_ps"], top_k=3)
turn1 = session.select_for_turn("read config source file")
turn2 = session.select_for_turn("check line 20") # reuses sticky prefix -> cache_prefix_preserved=True
```

### 8. Mondrian Conformal Quantiles, Atomic Versioned Checksums & Staged Plasticity

```python
import aetheris_microbrains as mb

# Load weights with automatic SHA-256 integrity verification (`verify_checksum=True`)
brain = mb.AnalyticalMicroTransformer.load("neuro_code_edit_verifier")
inf = brain.infer_conformal(["syntax_tree_valid", "ast_node_function_def"], alpha=0.05, mondrian=True)
print("Origin:", inf["calibration_data_origin"], "Std Err:", inf["coverage_std_error"])

# Staged continual plasticity: writes to `_candidates/`, requires holdout gate, atomic `_rollback/`
pipeline = mb.ContinualLearningPipeline()
staged = pipeline.apply_plasticity_feedback("code_edit_verifier", ["new_ast_pattern"], promote=False)
# prom = pipeline.verify_and_promote_candidate("code_edit_verifier", holdout_data, min_coverage=0.95)
# pipeline.rollback_domain("code_edit_verifier")
```

### 9. AST Signature Blast Radius, Sliding-Window Drift & Adaptive Conformal (ACI)

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

## 🧪 Testing & Reproducible Benchmarks (`v0.7.3`: 254 Passed, 2 Skipped)

Run the full verification suite (zero failures; 2 optional-dependency skips in `test_robotics_stack.py` when `nir` or `onnx` are not installed):

```bash
pytest aetheris_microbrains/tests -q
```

### 1. External Tool-Drop Rate, Baseline Comparison & Live-LLM A/B (`examples/06_real_llm_ab_benchmark.py`)
Evaluates `narrow_tool_surface(top_k=K)` across **40 handcrafted SWE / DevOps-style queries (38 in-domain, 2 OOD; not an official BFCL split)** using real serialized 32-tool OpenAI JSON schemas (**`2,589` tokens/step**):

```bash
python examples/06_real_llm_ab_benchmark.py
# Optional live-LLM A/B comparison against any OpenAI-compatible / Ollama endpoint:
python examples/06_real_llm_ab_benchmark.py --live --model gpt-4o-mini
```

| Top-K | Recall@K | Needed-Tool Drop Rate | Escalated to Full | OOD Escalated | Gross Saved | Net Saved (After Retries) | Local p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 92.1% | 7.9% (3/38) | 19/38 | 1/2 | 46.7% | 35.1% | 1.864 ms |
| **4** | **97.4%** | **2.6% (1/38)** | **19/38** | **1/2** | **43.5%** | **39.7%** | **1.272 ms** |
| 6 | 97.4% | 2.6% (1/38) | 19/38 | 1/2 | 40.5% | 36.6% | 1.141 ms |
| 8 | 97.4% | 2.6% (1/38) | 19/38 | 1/2 | 37.5% | 33.6% | 1.246 ms |

**Baseline Comparison (Top-4 on 38 In-Domain Handcrafted Queries, with Pre-Frozen Keyword Escalation Rule):**

| Architecture | Recall@4 | Escalated | Dropped | Drop Rate | 95% CP Upper Drop | Net Saved (After Retries) | Local p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Keyword Top-4 | 76.3% | 0/38 | 9/38 | 23.7% | <= 37.7% | 52.1% | 0.195 ms |
| Naive Keyword Top-4 (Frozen Esc.) | 84.2% | 8/38 | 6/38 | 15.8% | <= 28.8% | 45.5% | 5.119 ms |
| Embedding + Ridge Top-4 | 73.7% | 0/38 | 10/38 | 26.3% | <= 40.5% | 49.0% | 0.478 ms |
| **Microbrains Top-4 (Hybrid)** | **97.4%** | **19/38** | **1/38** | **2.6%** | **<= 11.9%** | **39.6%** | **8.435 ms** |

> *Honesty note:* Using a pre-frozen lexical cutoff (`top_score < 2 or margin < 1`) with zero test-set tuning, Keyword reaches 84.2% Recall@4 while Microbrains reaches 97.4% by escalating half the queries to the full catalog (`--live` retries rely strictly on runtime invalid/missing tool calls, never oracle ground-truth labels).

### 2. Local 25-Step Regression & Schema-Token Smoke Test (`examples/05_reproducible_25step_benchmark.py`)
Deterministic offline trajectory checking local CPU gate latency, AST/shell/secret rules, and OOD escalation (measures **tool-schema tokens only**, excluding conversation history and LLM prompt caching; see Section 5 for cache-adjusted multi-turn metrics):

```bash
python examples/05_reproducible_25step_benchmark.py
```

| Architecture | 25-Step Schema Tokens | Schema Token Reduction | Dangerous Missed (FN) | Benign Blocked (FP) | Local p95 Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unfiltered 32-Tool Agent | 64,800 | 0.0% | — (runs unfiltered) | 0 | — (not measured) |
| Naive Keyword Baseline | 64,800 | 0.0% | 8 | 4 | 0.002 ms |
| **Aetheris Microbrains v0.7.3** | **12,312** | **81.0%** | **0** | **0** | **1.977 ms** |

### 3. External SWE-bench Lite (300 Real GitHub Issues) Zero-LLM Gate Benchmark (`examples/07_swebench_lite_gate_benchmark.py`)
Evaluates `decide()` and `narrow_tool_surface(top_k=4)` against all **300 real issues and gold patches** in `princeton-nlp/SWE-bench_Lite`:

```bash
python examples/07_swebench_lite_gate_benchmark.py
```

**Patch Gate (300 Real Gold File Diffs):**

| Metric (Denominator: 300 File Diffs / 60 Top-Level Mutated Pairs) | Result |
| --- | ---: |
| Gold file diffs allowed (`layer1_hunk_only_checked`) | **213 / 300 (71.0%)** |
| Partial hunks escalated without full `original_source` | **82 / 300 (27.3%)** |
| Gold file diff false-deny rate (`AST syntax FP: 0`) | **5 / 300 (1.67%)** |
| Paired causal mutation flips (verifies expected `reason_code` **and** rule-specific `label`) | **58 / 58 non-deny** (`39 / 39` strictly allowed originals; `60 / 60` raw) |
| Diff gate local latency (`p50` / `p95`) | **1.474 ms / 3.968 ms** |

**Tool-Surface Narrowing Smoke-Test (300 Issue Prompts — Coarse `any(5 SWE core tools)` Criterion):**

| Metric (Denominator: 300 Issues / 127 Narrowed Prompts) | Result |
| --- | ---: |
| Prompts narrowed to Top-4 vs. full-catalog escalation | **127 / 300 narrowed**, **173 / 300 full-catalog** |
| Coarse SWE core retention across all 300 prompts vs. Pre-Frozen Keyword smoke | **297 / 300** vs. **286 / 300** (`36.7%` schema tokens saved, `p95 = 14.487 ms`) |
| Coarse SWE core retention within narrowed-only prompts (`mode == "narrowed"`) | **124 / 127** (`edit_file` / `repo_read`: **103 / 127**) |

### 4. Drop-In Claude Code / Codex `PreToolUse` Hook CLI & HTTP Endpoints (`v0.7.0`–`v0.7.3`)
Use Microbrains directly inside Claude Code, Codex, or custom agent hooks via stdin/stdout JSON. Output adapts per host via `--host claude|codex|generic` (Claude Code uses `hookSpecificOutput` `allow`/`ask`/`deny`; `--host codex` omits unsupported `"ask"` on escalations, returning `host_action: "defer_to_host_policy"` with exit `0` and empty `stdout` so Codex applies its configured approval policy—which does **not** force a prompt under auto-approve modes unless `--codex-escalate-deny` is enabled), while fatal denials and non-shadow invalid/non-object payloads write redacted reasons to `stderr` with exit code `2` (fail-closed):

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' | python -m aetheris_microbrains.hook --strict-ast
echo '{"tool_name":"Bash","tool_input":{"command":"custom_cmd"}}' | python -m aetheris_microbrains.hook --host codex
```

* **Prefilter-by-default & Strict Passive Shadow (`--journal-path` / `AETHERIS_JOURNAL_PATH`):** Safe `allow` verdicts exit `0` with empty `stdout` so host confirmation prompts still apply (`--auto-allow` opts out). Under `--shadow`, `evaluate_hook_payload()` and CLI stay passive (`exit 0`, empty `stdout`, `permissionDecision=None` regardless of `--host codex` flags), log `shadow_reason_code` on valid payloads, and use a stdlib-only journal writer to record redacted `event_type: "hook_error"` entries on empty input, malformed JSON, non-object JSON, and engine/import errors (so hook failures are visible alongside `shadow_mode_observation` lines in `decision_journal.jsonl`), warning on `stderr` if journal persistence fails.
* **Live `/dashboard`, Static `/demo/dashboard` & Server Extra (`pip install aetheris-microbrains[server]`):** Installs FastAPI, Uvicorn, and Pydantic. `/dashboard` and `/metrics` render live `MicrobrainTelemetry` data; static preview cards live on `/demo/dashboard`. `/microbrain/prefilter` fails closed (`status="error"`, `passed=False`) and `DSHMicrobrainClient` propagates server error statuses while validating Python syntax via `ast.parse` in its local fallback.
* **Fail-closed recommendation:** Set `"onFailure": "block"` in Claude Code hook configs so import/environment errors block safely.

### 5. Frozen 120-Task Multi-Label Tool Advisor & Session Prompt-Cache Benchmark (`examples/08_frozen_advisor_benchmark.py`)
Evaluates **120 frozen multi-label coding tasks** (6 external repo archetypes × 20 tasks, each permitting multiple valid `acceptable_tools`) across **3-turn sessions** with realistic prompt-prefix caching (`0.1x` cached token cost when the tool prefix matches the previous turn, `1.0x` on invalidation):

```bash
python examples/08_frozen_advisor_benchmark.py
```

| Strategy | Recall@4 | Narrowed Recall | Narrowed | Dropped | 95% CP Upper Drop | Uncached Net Saved | Cache-Adjusted Net Saved | Local p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 32-Tool Catalog (Baseline) | 100.0% | 100.0% | 0/120 | 0/120 | <= 2.5% | 0.0% | 0.0% | 0.001 ms |
| Simple Name+Desc Keyword (Frozen Rule) | 94.2% | 92.9% | 98/120 | 7/120 | <= 10.7% | 79.5% | 48.8% | 0.284 ms |
| Microbrains Top-4 (Per-Turn) | 100.0% | 100.0% | 78/120 | 0/120 | <= 2.5% | 52.9% | **4.1%** | 7.412 ms |
| **Microbrains Top-4 (Session-Sticky)** | **100.0%** | **100.0%** | **78/120** | **0/120** | **<= 2.5%** | **56.8%** | **56.8%** | **5.559 ms** |

**Equal-Escalation-Rate & Risk–Coverage AURC Comparison (`42/120` Matched Escalation Budget):**

| Selector | Escalated | Narrowed | Dropped | Recall@4 | Narrowed Recall | 95% CP Upper Drop | AURC Drop Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple Name+Desc Keyword | 42/120 (nat 22) | 78/120 | 2/120 | 98.3% | 97.4% | <= 5.1% | 0.0228 |
| **Microbrains Top-4** | **42/120 (nat 42)** | **78/120** | **0/120** | **100.0%** | **100.0%** | **<= 2.5%** | **0.0000** |

* **Agreed-Narrowed Error Check:** Even among `73` queries where both Keyword and Microbrains narrowed, `6` had drops in one selector (measured explicitly). When Keyword is forced to escalate its weakest `42/120` queries (matching Microbrains' `42/120` abstention budget), Keyword still drops `2/120` (`AURC = 0.0228`) vs `0/120` (`AURC = 0.0000`) for Microbrains.
* **Prompt-Cache Effect:** Switching active tools every turn (`Per-Turn`) breaks prefix caching (`52.9%` raw → `4.1%` cache-adjusted); pinning tools per session with `SessionToolSurface` preserves `56.8%` net cache-adjusted savings. Decoupled `diagnose_patch()` matched `5/5` patch diagnostic cases with `authorizes_execution=False`.

### 6. Bilingual SWE 5-Way Ablation & Equal-Escalation Comparison (`aetheris_microbrains/tests/test_colony_swe_shadow.py`)
Disjoint held-out conformal calibration (`in_sample=False`), tracking both missed mandatory escalations and benign false escalations / misclassifications:

| Variant | 120 RU/EN Acc (RU) | Missed Esc. (95% CP Upper) | Syn-Bank Held-Out (20) | Zero-Overlap Morph Split (20) | Morph Missed Esc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1.` Pure Rules Baseline | 106/120 = 88.3% (95.0%) | 4/50 (`<= 17.38%`) | 5/20 (25.0%, `1/6` missed) | 10/20 (50.0%) | `0/6` (`<= 39.3%`) |
| `1b.` Rules (Matched Synonym + Invariants) | 107/120 = 89.2% (95.0%) | 4/50 (`<= 17.38%`) | 20/20 (100.0%, `0/6` missed) | 6/20 (30.0%) | `0/6` (`<= 39.3%`) |
| `2.` Pure Neural (Weights + Threshold) | 103/120 = 85.8% (90.0%) | 1/50 (`<= 9.14%`) | 14/20 (70.0%, `0/6` missed) | 8/20 (40.0%) | `0/6` (`<= 39.3%`) |
| `2b.` Pure Neural (Weights Only, No Threshold) | 60/120 = 50.0% (56.7%) | 50/50 (`<= 100.0%`) | 12/20 (60.0%, `6/6` missed) | 8/20 (40.0%) | `6/6` (`<= 100.0%`) |
| **`3.` Microbrain Hybrid (Weights + Conformal Gate)** | **104/120 = 86.7% (91.7%)** | **0/50 (`<= 5.82%`)** | **14/20 (70.0%, `0/6` missed)** | **8/20 (40.0%)** | **`0/6` (`<= 39.3%`)** |

**Matched Escalation-Budget & Selective AURC (`compare_at_equal_escalation_rate()`):**
* **`colony_120` (`63/120` escalated, `57` local):** Matched Rules (`1b`, natural `50/120`) drops from `89.2%` to **`100/120 (83.3%)`** with `3/50` missed escalations (`selective_error_rate_on_local = 0.0702`, `AURC = 0.0801`), vs. **Microbrain Hybrid (`3`)** at **`104/120 (86.7%)`** with `0/50` missed escalations (`selective_error_rate_on_local = 0.0526`, `AURC = 0.0959`).
* **`frozen_morph_20` (zero exact-token overlap on routine items):** Matched Rules (`1b`) naturally escalates all `20/20` (`6/20 = 30.0%`, `AURC = 0.8518`), whereas **Microbrain Hybrid (`3`)** classifies `3` locally at `17/20` escalations (`8/20 = 40.0%`, `0/6` missed, `AURC = 0.6576`).

> *Honesty note:* Matching `1b`'s synonym vocabulary solves the old `20/20` synonym split, isolating why vocabulary expansion helped there; on `colony_120`, Equal-Budget accuracy (`86.7%` vs `83.3%`, `0/50` vs `3/50` missed) favors Hybrid while overall Risk–Coverage `AURC` across confidence rankings slightly favors exact-keyword `1b` (`0.0801` vs `0.0959`); on `FROZEN_UNSEEN_MORPH_QUERIES` (`0/14` exact token overlap), char-3-gram coverage reduces `AURC` (`0.6576` vs `0.8518`), while ablating the conformal threshold (`2b`) causes `50/50` missed escalations. None of these synthetic upper bounds transfer directly as field exchangeability guarantees.

```python
import aetheris_microbrains as mb

moe = mb.get_tool_routing_moe()
moe.fit_from_traces([
    {"query": "read source file config.py", "tool": "repo_read"},
    {"query": "commit staged bugfix changes", "tool": "git_commit"},
])
# Measure genuine Top-K drop rate on labeled holdout trajectories
eval_rep = moe.evaluate_topk_holdout([
    {"query": "read source file config.py", "acceptable_tools": ["repo_read", "grep_search"]},
] * 20, top_k=4, sample_id="holdout_20")
```

Or target specific verification suites:

```bash
pytest aetheris_microbrains/tests/test_fable_review_fixes.py aetheris_microbrains/tests/test_reviewer_probe_suite.py aetheris_microbrains/tests/test_end_to_end_economics_and_drift.py aetheris_microbrains/tests/test_layer_ablation_and_baselines.py -v
```

### Independent Audit & Hardening Coverage:
* **`test_fable_review_fixes.py` (`test_fable_1`..`43`)**: Pins persistent `W_embed_anchor` & empirical diagonal Fisher EWC regularization, weight-space similarity in `WeightSpaceVSA.find_nearest_domains()`, dynamic caller `alpha`, honest `ESCALATED_TO_LLM` without an `llm_callback`, secret-leak detection, zero hardcoded OOD stems or leaked benchmark identifiers in `test_colony_swe_shadow.py`, `allow_pickle=False`, zero numbered placeholder vocab tokens across all active production models (41 clean-vocab domains; test artifacts isolated to `_archive/`), ridge-fitted `W_gate` + empirical per-expert quantiles + `evaluate_topk_holdout()` in `ToolRoutingMoE`, atomic `_versions/` snapshot commits via single-pointer `.manifest.json` (`strict_directory=True` and atomic rollback), persisted `calibration_data_origin`, non-polluting `save()` CWD hygiene, full-file patched AST verification, live `available_tools` enforcement, Clopper-Pearson upper bounds strictly from real measured eval sets (`null` + `unknown` otherwise), `authorizes_execution=False` + decoupled `rank_tools()` / `diagnose_patch()` / `SessionToolSurface`, host-contracted Claude/Codex hooks (`defer_to_host_policy`) + passive `--shadow` stdlib `hook_error` journaling (`--journal-path`), live `/dashboard` vs static `/demo/dashboard` + DSH AST fallback, zero legacy `Proposal N` comments, `n_cal = 120` expanded calibration, `08_frozen_advisor_benchmark.py` (including equal-escalation-rate & AURC comparisons), and audit remediation (`fission_synthetic_anchors`, `synthetic_lexical_probes`, unobserved registry `null` telemetry, duplicate route/tree-sitter/redirection/variance-threshold fixes).
* **Split-Conformal & Mondrian Consistency Discipline (`conformal_collector.py`)**: Disjoint **Train**, **120 Calibration**, and **40 Validation** splits across **all active production models (41 clean-vocab domains; test artifacts isolated to `_archive/`)** (`calibration_data_origin: "synthetic_rule_consistency"` persisted in `*_meta.json` — verifying internal rule consistency rather than field user-distribution exchangeability) validated under strict prediction-set membership $Y \in C_\alpha(X)$, immutable `_versions/` snapshots with single-pointer `.manifest.json` commits, SHA-256 weight/meta checksums, coverage standard error $\sqrt{\alpha(1-\alpha)/(n+2)}$, Clopper-Pearson upper bounds & Rule-of-3 emitted strictly from real measured evaluation sets (`null` otherwise), sliding-window drift detection, and Gibbs-Candès ACI adaptation.
* **Ablation, Baselines & Economics (`test_layer_ablation_and_baselines.py`, `test_end_to_end_economics_and_drift.py`)**: Layer 0 / 0+1 / 0+1+2 ablation, bilingual RU/EN routing vs. English keyword baseline, MCP-Zero Top-$K$ schema reduction ($\ge 75\%$), `< 2 ms` p95 decision latency, risk-class fail-closed error handling, `shadow_mode` journaling, and `_candidates/` $\to$ `_rollback/` continual learning.
* **Adversarial & Probe Suites (`test_reviewer_probe_suite.py`, `test_adversarial_safety_suite.py`, `test_colony_swe_shadow.py`)**: Diverse obfuscated Python sinks & secret leaks, realistic indented/`except`/`elif`/comment/`.ts`/`.json`/`.yaml`/`.md` diffs, real weight-space surgical causal inversion (no `MagicMock`), allowlist-first `shlex` shell parsing, and 120 bilingual (60 RU / 60 EN) SWE queries (`0/50` missed escalations).

---

## 📄 License & Commercial Inquiries

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

* **Noncommercial Use:** You may freely use, modify, and distribute this software for educational, research, and noncommercial purposes.
* **Commercial Use:** Commercial use requires a commercial license from the author.

For commercial licensing inquiries, please contact:
**Dmitry Tikov** — [drtikov@gmail.com](mailto:drtikov@gmail.com)
