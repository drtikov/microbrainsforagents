"""CLI and programmatic audit tool for inspecting Microbrains decisions and escalations.

Usage:
    python -m aetheris_microbrains.audit --last
    python -m aetheris_microbrains.audit --tail 5
    python -m aetheris_microbrains.audit --summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.runtime_paths import get_default_journal_path

_DEFAULT_JOURNAL_PATH = get_default_journal_path()


def get_journal_entries(
    journal_path: Optional[Path | str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """Read recent journal entries from decision_journal.jsonl."""
    raw_path = journal_path or os.getenv("AETHERIS_JOURNAL_PATH") or _DEFAULT_JOURNAL_PATH
    p = Path(raw_path).expanduser().resolve()
    if not p.is_file():
        return []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        out: List[Dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
    except Exception:
        return []


def format_entry(entry: Dict[str, Any]) -> str:
    """Format a single journal entry for clean human terminal inspection."""
    decision = entry.get("decision", "unknown").upper()
    color_code = {
        "ALLOW": "\033[92m",    # Green
        "DENY": "\033[91m",     # Red
        "ESCALATE": "\033[93m", # Yellow
        "WARN": "\033[95m",     # Magenta
    }.get(decision, "\033[0m")
    reset = "\033[0m"

    ts = entry.get("timestamp")
    time_str = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        if isinstance(ts, (int, float))
        else "N/A"
    )
    layer = entry.get("layer", "N/A")
    reason = entry.get("reason_code", "N/A")
    label = entry.get("label", "N/A")
    latency = entry.get("latency_ms", "N/A")
    sha = entry.get("input_sha256", "N/A")
    preview = entry.get("payload_preview") or entry.get("tool") or ""
    conf_bound = entry.get("upper_error_bound_95")

    box = [
        f"{color_code}┌── [{decision}] Decision Event (Layer {layer}) ─────────────────────────{reset}",
        f"│ Time:       {time_str} | Latency: {latency} ms | Hash: {sha}",
        f"│ Reason:     {reason}",
        f"│ Details:    {label}",
    ]
    if conf_bound is not None:
        box.append(f"│ Conformal:  Upper Error Bound 95% = {conf_bound}")
    if preview:
        box.append(f"│ Payload:    {preview}")
    if entry.get("escalation_contract"):
        contract = entry["escalation_contract"]
        box.append(f"│ Escalation: Target={contract.get('recipient')} Timeout={contract.get('timeout_sec')}s")
    box.append(f"{color_code}└─────────────────────────────────────────────────────────────{reset}")
    return "\n".join(box)


def print_summary(entries: List[Dict[str, Any]]) -> None:
    """Print high-level statistics of recent decisions."""
    total = len(entries)
    if total == 0:
        print("No decision journal entries found.")
        return
    decisions: Dict[str, int] = {}
    layers: Dict[Any, int] = {}
    latencies: List[float] = []

    for e in entries:
        d = str(e.get("decision", "unknown")).lower()
        decisions[d] = decisions.get(d, 0) + 1
        lay = e.get("layer", "other")
        layers[lay] = layers.get(lay, 0) + 1
        lat = e.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))

    avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

    print("=" * 60)
    print("AETHERIS MICROBRAINS - AUDIT JOURNAL SUMMARY")
    print("=" * 60)
    print(f"Total Logged Events: {total}")
    print(f"Average Latency:     {avg_lat} ms")
    print("\nDecisions:")
    for d, count in sorted(decisions.items()):
        pct = round((count / total) * 100, 1)
        print(f"  • {d.upper():<10} {count:>5} ({pct:>5.1f}%)")
    print("\nTriggering Layers:")
    for l, count in sorted(layers.items(), key=lambda x: str(x[0])):
        print(f"  • Layer {str(l):<5} {count:>5}")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aetheris_microbrains.audit",
        description="Inspect and explain Microbrains decisions and audit logs for human review.",
    )
    parser.add_argument("--last", action="store_true", help="Show the most recent decision event")
    parser.add_argument("--tail", type=int, default=0, help="Show the last N decision events")
    parser.add_argument("--summary", action="store_true", help="Show aggregate summary of decisions")
    parser.add_argument("--journal-path", type=str, default="", help="Path to decision_journal.jsonl")

    args = parser.parse_args(argv)
    # Windows consoles default to cp1252 and crash on box-drawing/bullet characters
    # in formatted entries; force UTF-8 with replacement instead of a hard failure.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    p = args.journal_path or None
    entries = get_journal_entries(journal_path=p, limit=max(100, args.tail))

    if args.summary:
        print_summary(entries)
        return 0

    if args.last:
        if not entries:
            print("No decision journal entries found.")
            return 0
        print(format_entry(entries[-1]))
        return 0

    if args.tail > 0:
        if not entries:
            print("No decision journal entries found.")
            return 0
        for e in entries[-args.tail:]:
            print(format_entry(e))
        return 0

    # Default action: show last entry or summary
    if not entries:
        print("No decision journal entries found.")
        return 0
    print(format_entry(entries[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
