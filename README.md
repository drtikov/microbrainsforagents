# Aetheris Microbrains

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-purple.svg)](LICENSE)
[![Zero-GPU](https://img.shields.io/badge/Hardware-Pure_NumPy_%7C_OpenCL-success.svg)](https://github.com/)
[![Latency](https://img.shields.io/badge/Prefilter-<2ms-brightgreen.svg)](https://github.com/)

**Aetheris Microbrains** (`v0.6.1`) is a deterministic, layered local pre-flight safety, code/shell verification, and tool-routing library for autonomous LLM agents built in pure NumPy (with optional OpenCL acceleration).

> **Architectural Honesty Note (`v0.6.1`):** Microbrains are **not** SGD-trained generative deep LLMs, and they do **not** replace general language reasoning. They are a **fast (`< 2 ms` p95 on CPU), deterministic 4-layer hybrid pre-filter and coprocessor** that stands *in front* of expensive cloud LLMs and runtime execution: **Layer 0 policy & risk classes**, **Layer 1 real Python AST, hardcoded secret-leak, and `shlex` shell parsers**, **Layer 2 ridge-fitted `ToolRoutingMoE` & analytically compiled Kuramoto-Hopfield matrices calibrated across all 42 clean-vocabulary models** ($n_{\text{cal}} = 22 \ge 19$, $1 - \alpha = 95\%$, dynamic caller `alpha`, strict prediction-set validation), and **Layer 3 honest LLM / human escalation** (never silently approving abstained calls without an actual `llm_callback`).

---

## 💡 Why Microbrains? (For Agents, Users, and the Planet)

Modern LLM agents invoke cloud foundation models on every tiny step—sending thousands of redundant tool-schema tokens just to pick a file reader, or waiting seconds for a datacenter GPU to generate a patch with a syntax error, hardcoded API key, or destructive shell command (`rm -rf`, `curl | sh`). **Aetheris Microbrains intercept those high-frequency, structural decisions locally in `< 2 ms` on CPU before any API call or tool execution happens.**

### 🤖 Why Microbrains Are Useful for Autonomous Agents
* **Prompt & Context Hygiene (MCP-Zero Narrowing):** Dynamically shrinks large tool registries (e.g., 32+ tools) down to the Top-$K$ relevant candidates (`narrow_tool_surface(top_k=4)`) in `< 1 ms`, reducing tool-schema prompt bloat by **75–87.5%** while falling back cleanly to the full catalog on ambiguous or out-of-distribution (OOD) tasks.
* **Real Pre-Flight Structural Guardrails Before Execution:** Replaces brittle substring hacks with real **Python `ast` visitors** (supporting indented hunks, `except`/`elif` fragments, and enclosing-file `file_content` alias resolution), **secret-leak detection** (`sk-...`, `ghp_...`, `AKIA...`, PEM private keys), **AST signature/interface diffing** (`breaking_signature_change`, `public_symbol_removed`, `exception_guard_removed`), and an **allowlist-first `shlex` shell parser** that passes safe read-only commands (`ls`, `pwd`, `git log --format`, `clang-format`), blocks destructive ones, and escalates unrecognized commands.
* **Calibrated Epistemic Humility:** Rather than forcing confident guesses on unknown queries, **split-conformal and Mondrian prediction sets** ($\alpha = 0.05$, $n_{\text{cal}} = 22$ held-out samples) detect OOD inputs (`token_coverage < 0.50` or `|C_alpha| != 1`) and explicitly abstain/escalate (`ESCALATED_TO_LLM`).

### 👤 Benefits for the User
* **Substantially Lower API Bills:** Cutting 75%+ of repeated tool-definition prompt tokens per turn and catching malformed or dangerous actions locally prevents paying cloud token fees for broken steps and retries.
* **Instant `< 2 ms` Responsiveness:** Routine tool routing, syntax checks, and safe command verification complete in microseconds to `< 2 ms` on local CPU—no 1–5 second network round-trips for obvious decisions.
* **Stronger Local Codebase, Data & Credential Safety:** Blocks obfuscated code sinks (`eval`, `exec`, `getattr`, `__builtins__[...]`, `base64`-to-`exec`), breaking public API removals, protected path writes (`constitution/`, `memory/bible.md`), and leaked secrets *on your machine* before execution or external transmission.
* **Predictable, Portable & Auditable:** Zero GPU requirement, cross-process deterministic SHA-256 routing across any `PYTHONHASHSEED`, clean Python imports (`sys.path` never mutated), `allow_pickle=False` model bundles, SHA-256 weight checksums, `shadow_mode`, and JSONL audit trails.

### 🌍 Ecological & Sustainable AI (Green AI) Impact
* **Orders-of-Magnitude Lower Energy Per Gate:** Running a `< 2 ms` NumPy / AST / `shlex` pre-filter on a local CPU consumes fractions of a milliwatt-hour—**thousands of times less electricity** than waking a cluster of datacenter GPUs for a full transformer forward pass per agent loop step.
* **Cutting Quadratic Attention Compute on Every Agent Turn:** Narrowing 32+ tool definitions down to Top-4 via `narrow_tool_surface()` eliminates ~75–87.5% of redundant tool-schema tokens across every prompt turn, shrinking total datacenter FLOPs, $\text{CO}_2\text{e}$ carbon footprint, and datacenter evaporative cooling water consumed over long multi-step autonomous runs.
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
 │    • 42-model fleet calibrated at alpha=0.05 on n_cal=22 held-out samples (>= 19 required), Mondrian per-class quantiles, dynamic caller `alpha` recomputation, `W_embed_anchor`, and SHA-256 checksums (`allow_pickle=False`)
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

## 🎯 Scope & Design Principles

* **What Microbrains Are Built For:**
  - Sub-millisecond pre-flight structural filtering on diffs (AST syntax + call-graph sinks + signature diffs), shell commands (`shlex`), and policy risk boundaries before hitting remote LLMs.
  - Narrow tool routing across known tool registries with explicit abstention (`ood_coverage_breach` / `conformal_passed=False`) whenever input shares insufficient vocabulary.
* **What Microbrains Are NOT For:**
  - Replacing general LLM reasoning on arbitrary natural language tasks. Unknown or open-ended prompts deterministically abstain and escalate to Layer 3 (LLM).
  - Acting as a standalone OS sandbox: executable confirmation & environment isolation remain enforced at the runtime policy boundary.

---

## 🧪 Testing & Reproducible 25-Step Benchmark (`v0.6.1`: 17 Suites — 174 Passed, 5 OpenCL GPU Skipped)

Run the full verification suite:

```bash
pytest aetheris_microbrains/tests -q
```

Run the standalone reproducible 25-step agent trajectory benchmark (`examples/05_reproducible_25step_benchmark.py`):

```bash
python examples/05_reproducible_25step_benchmark.py
```

Measured output across the 25-step mixed autonomous trajectory (bilingual RU/EN routing, indented/except/comment/non-Python hunks, obfuscated AST/secret/shell attacks, and OOD queries):

| Architecture | 25-Step Schema Tokens | Token Reduction | Dangerous Missed (FN) | Benign Blocked (FP) | Local p95 Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unfiltered 32-Tool Agent | 96,000 | 0.0% | — (unfiltered) | 0 | — (cloud RTT) |
| Naive Keyword Baseline | 96,000 | 0.0% | 8 | 4 | 0.003 ms |
| **Aetheris Microbrains v0.6.1** | **18,240** | **81.0%** | **0** | **0** | **0.908 ms** |

Or target specific verification suites:

```bash
pytest aetheris_microbrains/tests/test_fable_review_fixes.py aetheris_microbrains/tests/test_reviewer_probe_suite.py aetheris_microbrains/tests/test_end_to_end_economics_and_drift.py aetheris_microbrains/tests/test_layer_ablation_and_baselines.py -v
```

### Independent Audit & Hardening Coverage:
* **`test_fable_review_fixes.py`**: Pins persistent `W_embed_anchor` & empirical diagonal Fisher EWC regularization, weight-space similarity in `WeightSpaceVSA.find_nearest_domains()`, dynamic caller `alpha` conformal recomputation, honest `ESCALATED_TO_LLM` reporting without an `llm_callback`, secret-leak detection, zero hardcoded OOD stems in `test_colony_swe_shadow.py`, `allow_pickle=False`, **zero numbered placeholder vocab tokens across all 42 models**, **ridge-fitted `W_gate` + empirical per-expert quantiles in `ToolRoutingMoE`**, non-polluting `save()` CWD hygiene, and verified `examples/01..05`.
* **Split-Conformal & Mondrian Coverage Discipline (`conformal_collector.py`)**: Disjoint **Train**, **22 Held-Out Calibration** ($n_{\text{cal}} \ge 1/\alpha - 1 = 19$ for $\alpha = 0.05 \implies 95\%$ nominal coverage), and **16 Held-Out Validation** splits across **all 42 clean-vocab models** (86 shipped store files, 33 distinct empirical `q_hat` quantiles) validated under strict prediction-set membership $Y \in C_\alpha(X)$, with SHA-256 weight checksums, coverage standard error $\sqrt{\alpha(1-\alpha)/(n+2)}$, Rule-of-3 bound $3/n$, sliding-window drift detection, and Gibbs-Candès ACI adaptation.
* **Ablation, Baselines & Economics (`test_layer_ablation_and_baselines.py`, `test_end_to_end_economics_and_drift.py`)**: Layer 0 / 0+1 / 0+1+2 ablation, bilingual RU/EN routing vs. English keyword baseline, MCP-Zero Top-$K$ schema reduction ($\ge 75\%$), `< 2 ms` p95 decision latency, risk-class fail-closed error handling, `shadow_mode` journaling, and `_candidates/` $\to$ `_rollback/` continual learning.
* **Adversarial & Probe Suites (`test_reviewer_probe_suite.py`, `test_adversarial_safety_suite.py`, `test_colony_swe_shadow.py`)**: Diverse obfuscated Python sinks & secret leaks, realistic indented/`except`/`elif`/comment/`.ts`/`.json`/`.yaml`/`.md` diffs, real weight-space surgical causal inversion (no `MagicMock`), allowlist-first `shlex` shell parsing, and 120 bilingual (60 RU / 60 EN) SWE queries (`0/50` missed escalations).

---

## 📄 License & Commercial Inquiries

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

* **Noncommercial Use:** You may freely use, modify, and distribute this software for educational, research, and noncommercial purposes.
* **Commercial Use:** Commercial use requires a commercial license from the author.

For commercial licensing inquiries, please contact:
**Dmitry Tikov** — [drtikov@gmail.com](mailto:drtikov@gmail.com)
