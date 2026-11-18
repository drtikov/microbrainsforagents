"""Example 01: Quickstart Microbrain Consultation (<2ms local inference)."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aetheris_microbrains as mb

def main():
    print("=== Aetheris Microbrains: Fast Intent & Consultation ===")

    # Check fleet status
    status = mb.microbrain_status()
    print(f"Active microbrains in fleet: {status.get('count', 0)}")

    # Consult microbrains for a query
    query = "syntax error in AST parser during code edit"
    print(f"\nQuerying: '{query}'")

    result = mb.microbrain_consult(text=query)
    print("Selected Brain Domain:", result.get("selected_brain"))
    print("Predicted Token:", result.get("predicted_token"))
    print("Prediction Set:", result.get("prediction_set"))
    print("Calibrated Confidence:", result.get("calibrated_confidence"))
    print("Calibration Samples:", result.get("n_calibration_samples"), "Valid:", result.get("conformal_guarantee_valid"))

if __name__ == "__main__":
    main()
