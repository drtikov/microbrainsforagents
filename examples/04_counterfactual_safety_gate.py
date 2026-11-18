"""Example 04: Pre-Commit Counterfactual Simulation & Constitutional Veto Gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aetheris_microbrains as mb

def main():
    print("=== Aetheris Microbrains: Counterfactual Pre-Commit Simulation ===")

    sim = mb.get_counterfactual_pipeline()

    # Case 1: Safe clean patch
    safe_diff = "--- a/math.py\n+++ b/math.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n-    pass\n+    return a + b\n"
    print("\n--- Testing Safe Patch ---")
    outcome_safe = sim.simulate_edit_impact("src/math.py", safe_diff)
    print("Allowed:", outcome_safe.allowed)
    print("Status:", outcome_safe.status)
    print("Step Rollouts:", outcome_safe.step_predictions)

    # Case 2: Obfuscated code execution hazard (caught structurally via Python AST visitor)
    obfuscated_diff = (
        "--- a/runner.py\n+++ b/runner.py\n@@ -1,2 +1,3 @@\n"
        " def run_task(cmd):\n"
        "+    return getattr(os, 'sys' + 'tem')(cmd)\n"
    )
    print("\n--- Testing Obfuscated Execution Diff ---")
    outcome_risky = sim.simulate_edit_impact("src/runner.py", obfuscated_diff)
    print("Allowed:", outcome_risky.allowed)
    print("Status:", outcome_risky.status)
    print("Veto Reason:", outcome_risky.veto_reason)

    # Case 3: Breaking public API signature removal (caught by AST interface diff + Causal SCM rollout)
    sig_break_diff = (
        "--- a/api.py\n+++ b/api.py\n@@ -1,2 +1,2 @@\n"
        "-def compute(a, b, c):\n"
        "+def compute(a):\n"
        "     return a * 2\n"
    )
    print("\n--- Testing Breaking Signature Diff ---")
    outcome_sig = sim.simulate_edit_impact("src/api.py", sig_break_diff)
    print("Allowed:", outcome_sig.allowed)
    print("Status:", outcome_sig.status)
    print("Failure Modes:", outcome_sig.simulated_failure_modes)
    print("Step Rollouts:", outcome_sig.step_predictions)

if __name__ == "__main__":
    main()
