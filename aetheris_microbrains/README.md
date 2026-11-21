# Aetheris Microbrains (`aetheris_microbrains`)

Standalone, portable suite of pure-NumPy analytical microbrains, a 4-layer structural AST/`shlex` pre-filter (`layered_decision_engine.py`), deterministic SHA-256 MoE tool routing, and held-out split-conformal abstention gates.

> **Architectural Honesty Note (`v0.7.6`):** Microbrains are **not** SGD-trained deep LLMs, and they do **not** replace general reasoning or an OS sandbox. They are a **fast (`< 2 ms` p95 on CPU), deterministic 4-layer hybrid pre-flight safety & tool-routing library** built in pure NumPy combining **Layer 0 universal `.env`/`.git/hooks`/`.ssh` protected paths, risk classes, and live `available_tools` enforcement**, **Layer 1 full-file patched Python AST reconstruction (`_apply_unified_diff_to_source`), opt-in `strict_ast_allowlist` mode, YAML `!!python/object` & JS `child_process`/`new Function` + optional `tree-sitter` checks, cyclomatic complexity vs. static data-table differentiation (`is_static_data_or_doc`), hardcoded secret-leak detection, and `shlex` shell parsing**, **Layer 2 grounded `CascadeEnsembleRouter` (`escalation_payload`), bilingual RU/EN `IntentPersistenceTracker`, `MetacognitiveCalibrator`, trace-trainable `ToolRoutingMoE.fit_from_traces()` + optional `embed_fn`/`tool_descriptions` hybrid, `recalibrate_from_decision_journal()`, and clean-vocabulary Kuramoto-Hopfield models** ($\ge 20$ samples per Mondrian class, $n_{\text{cal}} \ge 120$, `conformal_set_valid`, `coverage_target = 0.95`, Clopper-Pearson 95% upper failure bounds, rolling drift monitor, `W_embed_anchor`, and pickle-safe Count-Sketch VSA), **Layer 3 honest escalation**, plus a drop-in **`PreToolUse` hook CLI (`python -m aetheris_microbrains.hook`)**. **License:** Apache License 2.0.

## Why Microbrains Matter (Agents, Users & Ecology)

- **For Autonomous Agents:** `narrow_tool_surface(top_k=4)` cuts 32-tool OpenAI JSON schemas (**2,589 tokens/step**) with **97.4% Recall@4** (`2.6%` drop rate) and **39.7% net prefill tokens saved after retries** (`81.0%` on the 25-step local smoke test) while escalating full catalogs on OOD inputs; real Python `ast` + `shlex` + secret-leak (`secret_leak_pattern_found`) checks catch broken/unsafe patches and dangerous shell commands before execution; split-conformal gating (`token_coverage < 0.50` / `|C_alpha| != 1`) enforces calibrated abstention instead of guessing.
- **For the User:** Lower cloud API token costs, instant `< 2 ms` local routing/verification without network round-trips, local secret/codebase protection (`rm -rf`, `eval`, obfuscated imports, API signature breaks), and side-effect-clean, SHA-256 checksummed pure-NumPy execution (`allow_pickle=False`).
- **For Ecology (Green AI):** Offloading routine routing, schema filtering, and syntax/safety checks from datacenter GPUs to `< 2 ms` local CPU passes avoids ~75%+ redundant attention tokens and orders of magnitude of electricity, carbon ($\text{CO}_2\text{e}$), and cooling water across iterative agent loops while running comfortably on existing CPUs and older OpenCL GPUs.

## Directory Structure

- `__init__.py` — Clean Python SDK entrypoint (does **not** mutate `sys.path` or shadow host modules) re-exporting the 4-layer decision engine (`decide`, `PolicyConfig`, `analyze_code_diff`, `analyze_shell_command`), core microbrain architectures, GPU/OpenCL acceleration, SOTA neural extensions, and agent tool wrappers.
- `core/` — Neural, structural parsing, and cognitive core modules (63 modules + `__init__.py`), including:
  - **4-Layer Structural Decision Engine**: `layered_decision_engine.py` (`decide()`, `PolicyConfig`, `evaluate_layer0_policy`, `is_protected_path`, `analyze_code_diff` AST visitor + `original_source` context, allowlist-first `analyze_shell_command` `shlex` tokenizer, risk-class fail-closed handling, `shadow_mode`, and JSONL decision journaling)
  - **OpenCL GPU Co-Processor**: `opencl_microbrain_accelerator.py` (PyOpenCL kernels with transparent NumPy CPU fallback)
  - **Microbrain Architectures**: `dendritic_tree_microbrain.py`, `kolmogorov_arnold_microbrain.py`, `linear_state_space_microbrain.py`, `microbrain_hypernetwork.py`, `liquid_state_reservoir.py`, `predictive_coding_engine.py`, `bitnet_ternary_engine.py`, `hyperdimensional_vsa_engine.py`
  - **Analytical Micro-Transformers & Nursery**: `analytical_micro_transformer.py` (Mondrian per-class quantiles, $\sqrt{\alpha(1-\alpha)/(n+2)}$ & $3/n$ bounds, SHA-256 `weights_sha256` verification on load), `brain_nursery.py`, `brain_moe_orchestrator.py`, `brain_merger.py`, `brain_cellular_fission.py`, `brain_plasticity_updater.py`, `brain_speculative_drafter.py`, `wsp_neural_surgeon.py`, `microbrain_server.py`
  - **Meta-Cognitive Layer**:
    - `causal_world_model.py`: Structural Causal Model (SCM) & Pearl $do$-calculus DAG.
    - `counterfactual_reasoner.py`: What-If lookahead pre-execution simulation with AST blast radius analysis.
    - `intent_persistence_tracker.py`: Trajectory-level Intent Drift Scoring (IDS) and dynamic context re-anchoring.
    - `metacognitive_calibrator.py`: Split-conformal action set coverage and epistemic/aleatoric uncertainty calibration.
    - `skill_lifecycle_manager.py`: Autonomous domain incubation, validation, degraded status, and fleet hygiene.
  - **SOTA Neural Extensions**:
    - `weight_space_vsa.py`: 2048-D Vector Symbolic Architecture / Hyperdimensional Computing associative memory over fleet model weights (`find_nearest_domains` ranks by actual Count-Sketch projected weights; `allow_pickle=False`).
    - `tool_routing_moe.py`: Bilingual (RU/EN) SHA-256 Top-2 Mixture of Experts (MoE) router constrained to active `available_tools` plus Top-K tool narrowing (`narrow_tool_surface`) and OOD abstention.
    - `critical_ensemble_abstention.py`: Multi-brain consensus voting integrated with Layer 0/1 AST, secret-leak (`secret_leak_pattern_found`), interface-hazard & path checks, selective conformal abstention, and honest `evaluate_with_cascade(..., llm_callback=None)`.
    - `cascade_ensemble.py`: Four-tier cascading decision pipeline executing `.infer_conformal()` across tier brains alongside Layer 0/1 structural AST/`shlex` checks (`ls`/`pwd` pass; unrecognized shell cmds & massive diffs escalate).
    - `continual_learning_pipeline.py`: Empirical diagonal Fisher EWC plasticity anchored to persisted `W_embed_anchor`, `_candidates/` staging, leak-free holdout verification using strict set coverage $P(Y \in C_\alpha(X))$, and `_rollback/` rollback.
    - `plasticity_cycle.py`: Automated scheduled plasticity candidate staging with measured rule-retention telemetry.
    - `microbrain_telemetry.py`: Nanosecond-precision inference telemetry with 10MB log rotation, ECE calculation, and health dashboard export.
    - `counterfactual_simulation_pipeline.py`: Pre-commit blast-radius gate backed by Layer 1 Python AST & hunk analysis.
- `evolution/` — Weight-Space Programming (WSP) & client modules (13 modules + `__init__.py`):
  - `wsp_engine.py`, `wsp_substrate.py`, `wsp_lenses.py`, `wsp_sae.py`, `wsp_merger.py`, `wsp_hebbian.py`, `wsp_sleep_consolidation.py`, `wsp_liquid_router.py`, `wsp_ab_probe.py`, `wsp_epistemic_oracle.py`, `wsp_sandbox_tracer.py`, `dsh_microbrain_client.py`, `opencl_microbrain_accelerator.py` (compatibility shim)
- `tools/` — Agent tool wrappers (`microbrain.py`, `microbrain_lab.py`, `neuro_brain_incubator.py`, `neuro_route.py`, `wsp_tensor_surgeon.py`, `conformal_collector.py`)
- `memory/` — Fleet registries (`fleet_lifecycle_registry.json`), pickle-safe projected weight-space VSA store (`store/weight_space_vsa.npz`), and analytically compiled, clean-vocabulary (zero numbered placeholder series), `W_embed_anchor`-pinned & SHA-256 verified `store/neuro_brains/` fleet (42 models / 86 files, 33 distinct `q_hat` quantiles), plus `opencl_crossover.json`
- `tests/` — Verification suites (17 `test_*.py` suites + `conftest.py`; **203 passed, 0 skipped**), including `test_fable_review_fixes.py` (`test_fable_1`..`35`), `test_end_to_end_economics_and_drift.py`, `test_layer_ablation_and_baselines.py`, `test_reviewer_probe_suite.py`, `test_adversarial_safety_suite.py`, `test_colony_swe_shadow.py` (with held-out paraphrased split), and `test_ouroboros_sota_extensions.py`

## Exported SDK Symbols

| Category | Symbols |
| --- | --- |
| **4-Layer Decision Engine** | `decide`, `PolicyConfig`, `evaluate_layer0_policy`, `is_protected_path`, `analyze_code_diff`, `analyze_shell_command`, `record_decision_journal`, `get_recent_decision_journal`, `narrow_tool_surface` |
| **GPU / OpenCL Co-Processor** | `OpenCLMicroBrainAccelerator`, `get_opencl_accelerator` |
| **Microbrain Cores** | `DendriticTreeMicroBrain`, `KolmogorovArnoldMicroBrain`, `LinearStateSpaceMicroBrain`, `MicroBrainHypernetwork`, `LiquidStateReservoir`, `HierarchicalPredictiveCodingEngine` (`PredictiveCodingEngine`), `PredictiveCodingLayer`, `BitLinearLayer`, `BitNetQuantizer` (`BitNetTernaryEngine`), `HyperdimensionalVSAEngine` |
| **Nursery & Surgery** | `AnalyticalMicroTransformer`, `STORE_DIR`, `AutonomousBrainNursery` (`BrainNursery`), `get_brain_nursery`, `WSPNeuralSurgeon` |
| **Meta-Cognitive Layer** | `StructuralCausalModel` (`CausalWorldModel`), `get_causal_world_model`, `CausalVariable`, `CausalIntervention`, `CausalSimulationResult`, `CounterfactualResult`, `simulate_counterfactual`, `IntentPersistenceTracker`, `IntentDriftAssessment`, `MetacognitiveActionVerdict`, `CalibratedConfidence`, `calibrate_action_confidence`, `SkillLifecycleState`, `SkillTelemetry`, `FleetHygieneReport`, `SkillLifecycleManager`, `get_skill_lifecycle_manager` |
| **SOTA Neural Extensions** | `WeightSpaceVSA`, `VSAResult`, `get_weight_space_vsa`, `ToolRoutingMoE`, `MoERoutingDecision`, `get_tool_routing_moe`, `CriticalEnsemble`, `CriticalConsensusResult`, `EnsembleVote`, `get_critical_ensemble`, `CascadeEnsembleRouter`, `CascadeTierResult`, `CascadeEnsembleVerdict`, `get_cascade_ensemble_router`, `ContinualLearningPipeline`, `ContinualUpdateResult`, `PlasticityCycleScheduler`, `PlasticityCycleReport`, `get_plasticity_scheduler`, `MicrobrainTelemetry`, `TelemetryEvent`, `DashboardReport`, `get_microbrain_telemetry`, `CounterfactualSimulationPipeline`, `SimulationOutcome`, `get_counterfactual_pipeline` |
| **Agent Tools** | `microbrain_consult`, `microbrain_status`, `consult_domain_hint`, `run_microbrain_lab`, `run_neuro_incubator`, `run_neuro_route`, `run_wsp_surgeon`, `ConformalCalibrationReport`, `run_conformal_calibration_pipeline` |

## Quickstart Examples

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

### 1. Direct Python SDK Import & Split-Conformal Domain Consultation

Install in editable mode (`pip install -e .`) and import cleanly without mutating `sys.path`:

```python
import aetheris_microbrains as mb

# Inspect the pre-compiled & calibrated neuro-brains fleet
status = mb.microbrain_status()
print(f"Loaded domains: {status['count']}")

# Consult the best-matching brain with conformal calibration metadata
reply = mb.microbrain_consult(text="syntax error in AST parser during code edit")
print("Selected Brain:", reply["selected_brain"])
print("Calibrated Confidence:", reply["calibrated_confidence"])
print("Calibration Samples:", reply["n_calibration_samples"], "Valid:", reply["conformal_guarantee_valid"])
```

### 2. Deterministic Top-2 Mixture of Experts (MoE) Tool Routing

```python
import aetheris_microbrains as mb

moe = mb.get_tool_routing_moe()
decision = moe.route("inspect file contents and write safe patch to repository")
print("Selected Tool:", decision.selected_tool)   # -> repo_write (deterministic across runs)
print("Top Experts:", decision.top_experts)
print("Conformal Guarantee:", decision.conformal_passed)
```

### 3. Holographic Weight Space VSA Memory

```python
import aetheris_microbrains as mb

vsa = mb.get_weight_space_vsa()
nearest = vsa.find_nearest_domains("tool_routing_v2", top_k=3)
print("Nearest Domains in Weight Space:", nearest)
```

### 4. Pre-Commit Structural AST & Counterfactual Safety Gate

```python
import aetheris_microbrains as mb

sim = mb.get_counterfactual_pipeline()
outcome = sim.simulate_edit_impact(
    file_path="src/parser.py",
    diff_text="--- a/src/parser.py\n+++ b/src/parser.py\n@@ -1,2 +1,3 @@\n def parse():\n-    pass\n+    return 42\n"
)
print("Allowed:", outcome.allowed)
print("Status:", outcome.status)
print("Step Rollouts:", outcome.step_predictions)
```

### 5. Allowlist Shell Gate, Dynamic `available_tools`, Top-K Tool Narrowing & Shadow Mode

```python
import aetheris_microbrains as mb

# Allowlist-first shell parser
print(mb.decide({"tool": "execute_shell", "cmd": "ls -la"})["decision"])                    # -> "allow"
print(mb.decide({"tool": "execute_shell", "cmd": "custom_deploy_binary --prod"})["decision"]) # -> "escalate"

# Dynamic tool surface restriction + Top-K tool narrowing
narrowed = mb.narrow_tool_surface("fix bug and write repository patch", top_k=4)
print(narrowed["mode"], narrowed["candidate_tools"], narrowed["reduction_ratio"])

# Shadow-mode journaling + enclosing file alias AST resolution
obs = mb.decide(
    {"tool": "edit_file", "path": "src/runner.py", "diff": "def f(c):\n    _o.system(c)\n"},
    context={"file_content": "import os as _o\n", "shadow_mode": True},
)
print(obs["decision"], obs["shadow_decision"], obs["input_sha256"])

# Staged continual learning (`_candidates/` -> verify -> `_rollback/`)
pipeline = mb.ContinualLearningPipeline()
staged = pipeline.apply_plasticity_feedback("code_edit_verifier", ["new_ast_pattern"], promote=False)
print(staged.candidate_staged, staged.promoted)

# Sliding-window drift telemetry & Gibbs-Candès ACI adaptation
tel = mb.get_microbrain_telemetry()
q_next = tel.update_aci_quantile(current_q=0.05, had_error=True, alpha=0.05, gamma=0.02)
dash = tel.generate_dashboard(window_size=100)
print(dash.sliding_window_coverage, dash.drift_detected, q_next)
```

### 6. Running Tests & the FastAPI Microbrain Server

```bash
pytest aetheris_microbrains/tests -q
python -m aetheris_microbrains.core.microbrain_server
```
