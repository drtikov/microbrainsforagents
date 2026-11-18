# Aetheris Microbrains

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-purple.svg)](LICENSE)
[![Zero-GPU](https://img.shields.io/badge/Hardware-Pure_NumPy_%7C_OpenCL-success.svg)](https://github.com/)
[![Latency](https://img.shields.io/badge/Prefilter-<2ms-brightgreen.svg)](https://github.com/)

**Aetheris Microbrains** (`v0.6.8`) is a deterministic, layered local pre-flight safety, full-file AST/shell verification, and registry-enforced tool-routing library for autonomous LLM agents built in pure NumPy (with optional OpenCL acceleration).

> **Architectural Honesty Note (`v0.6.8`):** Microbrains are **not** SGD-trained generative deep LLMs, and they do **not** replace general language reasoning or an OS sandbox. They are a **fast (`< 2 ms` p95 on CPU), deterministic 4-layer hybrid pre-filter and coprocessor** that stands *in front* of cloud LLMs and runtime execution: **Layer 0 universal SWE security paths (`.env`, `.git/hooks/`, `.ssh/`, `credentials.json`), risk classes, and live `available_tools` enforcement**, **Layer 1 full-file patched Python AST reconstruction (`ast.parse` over applied unified diffs), opt-in `strict_ast_allowlist` mode, YAML `!!python/object` & JS `child_process`/`new Function` + optional `tree-sitter` checks, cyclomatic complexity vs. static data-table differentiation, hardcoded secret-leak detection, and allowlist-first `shlex` shell parsing**, **Layer 2 grounded `CascadeEnsembleRouter` (with `escalation_payload` on all block verdicts), negation-aware bilingual `IntentPersistenceTracker`, `MetacognitiveCalibrator` (`q_hat`/`conformal_basis` provenance), negation-aware `active_inference`, trace-trainable `ToolRoutingMoE.fit_from_traces()` + optional semantic `embed_fn`/`tool_descriptions` hybrid, and 42 clean-vocabulary Kuramoto-Hopfield models** ($\ge 25$ samples per Mondrian class, $n_{\text{cal}} = 120\text{--}550$, $1 - \alpha = 95\%$ `coverage_target`, `conformal_set_valid`, exact Clopper-Pearson 95% upper failure bounds, and journal recalibration), **Layer 3 honest LLM / human escalation**, plus a drop-in **`PreToolUse` hook CLI (`python -m aetheris_microbrains.hook`)**. **License:** PolyForm Noncommercial 1.0.0 (commercial use requires a commercial license).

---

## 💡 Why Microbrains? (For Agents, Users, and the Planet)

Modern LLM agents send thousands of redundant tool-schema tokens in every prompt turn just to expose 32+ tools, or execute/retry patches containing syntax errors, hardcoded API keys, or destructive shell commands (`rm -rf`, `curl | sh`). Because standard tool-calling agents select tools and generate arguments in the same LLM pass, **Microbrains' routing layer cuts per-turn input schema prefill tokens (`narrow_tool_surface`), while its Layer 0/1 gate blocks malformed or dangerous actions locally in `< 2 ms` on CPU before execution or wasted retry turns.**

### 🤖 Why Microbrains Are Useful for Autonomous Agents
* **Prompt & Context Hygiene (MCP-Zero Narrowing):** Dynamically shrinks large tool registries (e.g., 32+ tools) down to the Top-$K$ relevant candidates (`narrow_tool_surface(top_k=4)`) in `< 1.2 ms`, achieving **97.4% Recall@4** (`2.6%` needed-tool drop rate) and **39.7% net schema-token savings after retry penalties** on a 40-task external multi-tool benchmark (`examples/06_real_llm_ab_benchmark.py`), while escalating the full catalog on ambiguous or OOD tasks.
* **Universal Protected Paths & Complexity-Aware Blast Radius (`v0.6.3`+):** Layer 0 protects standard SWE secret/key/hook paths (`.env*`, `.git/hooks/`, `.git/config`, `.ssh/`, `id_rsa`, `credentials.json`) without false-positiving on `src/env_loader.py`; Layer 1 distinguishes benign multi-hundred-line static data tables (`is_static_data_or_doc=True`, zero cyclomatic delta) from high-complexity control-flow mutations (`cyclomatic_complexity >= 25`).
* **Journal-Closed Conformal Recalibration & Clean Packaging:** `decide()` emits `upper_error_bound_95` and `drift_detected`, while `recalibrate_from_decision_journal()` closes the feedback loop from audited runs back to empirical quantile thresholds with zero legacy `Proposal N` scaffold comments anywhere in the SDK.
* **Live Agent Registry Enforcement (`available_tools`):** Both `ToolRoutingMoE.route(..., available_tools=...)` and `decide(..., context={"available_tools": ...})` strictly constrain routing and execution to registered agent tools, denying unregistered calls at Layer 0 and escalating registry mismatches at Layer 2.
* **Calibrated Epistemic Humility, Clopper-Pearson Bounds & Live Drift Detection:** Split-conformal and Mondrian prediction sets ($\alpha = 0.05$, $n_{\text{cal}} = 120$ held-out samples, $n_{\text{val}} = 40$) report exact one-sided Clopper-Pearson 95% upper failure bounds (`upper_error_bound_95`) alongside rolling nonconformity/abstention drift alarms (`drift_detected`) on `infer_conformal()`.

### 👤 Benefits for the User
* **Substantially Lower API Bills:** Cutting redundant tool-definition prefill tokens per turn and catching malformed or dangerous actions locally prevents paying cloud token fees for broken steps and execution retries.
* **Instant `< 2 ms` Local Gate:** Safety verification, AST diff checks, allowlist-first shell parsing, and Top-K tool narrowing complete in microseconds to `< 2 ms` on local CPU before hitting remote APIs or local subprocesses.
* **Stronger Local Codebase, Data & Credential Safety:** Blocks obfuscated code sinks (`eval`, `exec`, `getattr`, `__builtins__[...]`, `base64`-to-`exec`), breaking public API removals, protected path writes (`constitution/`, `memory/bible.md`), and leaked secrets *on your machine* before execution or external transmission.
* **Predictable, Portable & Auditable:** Zero GPU requirement, cross-process deterministic SHA-256 routing across any `PYTHONHASHSEED`, clean Python imports (`sys.path` never mutated), `allow_pickle=False` model bundles, SHA-256 weight checksums, `shadow_mode`, and JSONL audit trails.

### 🌍 Ecological & Sustainable AI (Green AI) Impact
* **Orders-of-Magnitude Lower Energy Per Gate:** Running a `< 2 ms` NumPy / AST / `shlex` pre-filter on a local CPU consumes fractions of a milliwatt-hour—**thousands of times less electricity** than waking a cluster of datacenter GPUs for a full transformer forward pass per agent loop step.
* **Cutting Quadratic Attention Compute on Every Agent Turn:** Narrowing 32+ tool definitions down to Top-4 via `narrow_tool_surface()` eliminates **39.7% net prefill schema tokens after retries** on diverse external tasks (`81.0%` on the 25-step local smoke test), shrinking datacenter FLOPs, $\text{CO}_2\text{e}$ carbon footprint, and cooling water consumed over long multi-step autonomous runs.
* **Hardware Longevity & Zero E-Waste Pressure:** Runs efficiently on existing laptops, edge CPUs, and legacy OpenCL GPUs without demanding continuous hardware upgrades just to route tools or validate patches.

---

## 🏗️ 4-Layer Decision Architecture

```text
Agent Input (tool call, diff, shell command, or intent query)
 │
 ├─ Layer 0: Deterministic Policy (code invariants & risk taxonomy)
 │    • Read-only vs. Reversible Write vs. External vs. Irreversible Destructive
 │    • Protected paths (memory/bible.md, constitution/, etc.) ── veto → deny / escalate
 │
 ├─ Layer 1: Structural Language, Secret & Shell Parsers (zero brittle substring hacks)
 │    • Python AST visitor (dedents indented hunks, wraps except/elif fragments, resolves `file_content` enclosing aliases, ignores comments)
 │    • Catches direct eval/exec, subprocess(shell=True), shutil.rmtree, pickle.loads, aliases, subscript (__builtins__/globals), encoded decode-to-exec, and getattr obfuscation
 │    • Hardcoded secret-leak detection (`secret_leak_pattern_found` on OpenAI/GitHub/AWS keys, PEM private keys, literal credential assignments)
 │    • AST interface & signature diffing (`breaking_signature_change`, `public_symbol_removed`, `exception_guard_removed`) + `CausalWorldModel` do-calculus rollout
 │    • JSON / TypeScript / JS / YAML / Markdown diffs parsed without false Python SyntaxErrors
 │    • Allowlist-first shell `shlex` tokenizer (`rm -rf`, `rm -fr`, `DROP DATABASE`, `sh -c`, `python -c`, `dd of=/dev/...`, `mkfs`, `curl | sh` deny; unrecognized cmds escalate; `ls`, `pwd`, `git log --format`, `clang-format` allow)
 │    • Risk-class fail-closed handling + structured decision journaling & `shadow_mode`
 │
 ├─ Layer 2: Analytical Micro-Classifiers & Split-Conformal Abstention
 │    • Deterministic SHA-256 bilingual (RU/EN) char-trigram + lexical routing (`ToolRoutingMoE`), constrained to `available_tools` + MCP-Zero Top-K narrowing (`narrow_tool_surface`, >=75% schema reduction)
 │    • 42-model fleet calibrated at alpha=0.05 on n_cal=120 held-out samples (n_val=40, std error <= 0.0197), Mondrian per-class quantiles, dynamic caller `alpha` recomputation, `W_embed_anchor`, and SHA-256 checksums (`allow_pickle=False`)
 │    • Pickle-safe Weight-Space VSA (`weight_space_vsa.npz`) ranking nearest domains via actual Count-Sketch projected parameter weights
 │    • Empirical Fisher diagonal EWC continual plasticity (`_candidates/` → leak-free holdout verification → `_rollback/`)
 │    • Sliding-window drift telemetry (`drift_detected`) + Gibbs-Candès Adaptive Conformal Inference (`update_aci_quantile`)
 │    • Singleton set (|C_alpha| == 1) → local allow; OOD / ambiguous (|C_alpha| != 1 or token_coverage < 0.50) → escalate
 │
 └─ Layer 3: Honest External LLM / Human Escalation
      • Invokes caller-supplied `llm_callback` when escalated, or explicitly returns `ESCALATED_TO_LLM` (`llm_invoked=False`) rather than auto-authorizing without an LLM
```

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

### 5. Allowlist-First Shell Gate, File-Context AST & Shadow Mode

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

### 6. Dynamic Tool Registry (`available_tools`) & MCP-Zero Top-K Narrowing

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

### 7. Mondrian Conformal Quantiles, SHA-256 Checksums & Staged Plasticity

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

**Baseline Comparison (Top-4 on 38 External Tasks, with 95% Clopper-Pearson Upper Drop Bound):**

| Architecture | Recall@4 | Dropped | Drop Rate | 95% CP Upper Drop | Net Saved (After Retries) | Local p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Keyword Top-4 | 76.3% | 9/38 | 23.7% | <= 37.7% | 52.1% | 1.468 ms |
| Embedding + Ridge Top-4 | 73.7% | 10/38 | 26.3% | <= 40.5% | 49.0% | 0.734 ms |
| **Microbrains Top-4 (Hybrid)** | **97.4%** | **1/38** | **2.6%** | **<= 11.9%** | **39.6%** | **0.941 ms** |

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

| Metric (300 Real SWE-bench Lite Issues) | Result |
| --- | ---: |
| Gold patches allowed (`layer1_structural_verified`) | **213 / 300 (71.0%)** |
| Partial hunks escalated without full `original_source` | **82 / 300 (27.3%)** |
| Gold patch false-deny rate (`AST syntax FP: 0`) | **5 / 300 (1.67%)** |
| Adversarial mutations on real patches caught (syntax/secret/`getattr`) | **60 / 60 (100.0%)** |
| Diff gate local latency (`p50` / `p95`) | **1.312 ms / 4.358 ms** |
| SWE core tool retained in `narrow_tool_surface(top_k=4)` | **297 / 300 (99.0%)** (`36.7%` schema saved) |

### 4. Drop-In `PreToolUse` Hook CLI, Strict AST Allowlist & Trace Training (`v0.6.5`–`v0.6.8`)
Use Microbrains directly inside Claude Code, Codex, or custom agent hooks via stdin/stdout JSON, or train the Layer 2 MoE router directly from recorded agent traces:

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' | python -m aetheris_microbrains.hook --strict-ast
```

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
