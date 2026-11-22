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
    journal_path: Optional[Path | str] = None, limit: Optional[int] = 50
) -> List[Dict[str, Any]]:
    """Read recent journal entries from decision_journal.jsonl (`limit=None` reads all)."""
    raw_path = journal_path or os.getenv("AETHERIS_JOURNAL_PATH") or _DEFAULT_JOURNAL_PATH
    p = Path(raw_path).expanduser().resolve()
    if not p.is_file():
        return []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        selected = lines if (limit is None or limit <= 0) else lines[-limit:]
        out: List[Dict[str, Any]] = []
        for line in selected:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _effective_decision_and_reason(entry: Dict[str, Any]) -> tuple[str, str]:
    """Resolve the enforced decision/reason, preferring `shadow_*` fields on shadow events."""
    is_shadow = bool(
        entry.get("shadow_mode")
        or entry.get("shadow_decision")
        or entry.get("shadow_reason_code")
    )
    if is_shadow:
        dec = str(entry.get("shadow_decision") or entry.get("decision") or "unknown").lower()
        reason = str(entry.get("shadow_reason_code") or entry.get("reason_code") or "none")
    else:
        dec = str(entry.get("decision", "unknown")).lower()
        reason = str(entry.get("reason_code") or entry.get("shadow_reason_code") or "none")
    return dec, reason


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


REASON_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "layer0_protected_path_veto": {
        "what": "The target path is a protected control-plane, credential, or constitutional asset.",
        "fix": "Do not edit protected paths through the agent. Operators may use a user shell, or set AETHERIS_ALLOW_POLICY_EDIT=1 for config-plane maintenance (never credential stores).",
    },
    "layer0_protected_path_read_requires_confirmation": {
        "what": "Reading this path would expose protected or credential material.",
        "fix": "Confirm with the user, or read a non-protected path (e.g. a sample/redacted copy).",
    },
    "layer0_session_state_corrupted": {
        "what": "The cross-turn session security state file failed integrity checks.",
        "fix": "Delete the session state file (it regenerates); investigate what corrupted it before continuing sensitive work.",
    },
    "layer1_ast_security_veto": {
        "what": "The proposed code contains a construct the AST layer vetoes (dangerous calls, obfuscated secrets, dynamic execution).",
        "fix": "Rewrite the code to avoid the flagged construct; run `python -m aetheris_microbrains.audit --explain` again after fixing for the specific finding label.",
    },
    "layer1_ast_syntax_error": {
        "what": "The proposed code does not parse as valid Python.",
        "fix": "Fix the syntax error. Note: only complete statements lint cleanly - submit the full function or block, not a fragment.",
    },
    "layer1_shell_parser_veto": {
        "what": "The shell command contains a vetoed construct (destructive rm, command substitution, dangerous inline code, obfuscation).",
        "fix": "Rewrite the command without the flagged construct (e.g. avoid `$(...)`, rm -rf on broad paths, `python -c` with untrusted code).",
    },
    "layer1_secret_egress_veto": {
        "what": "Credential-looking material was found in outbound arguments (command line, URL, query).",
        "fix": "Remove the secret from the call. Pass it via environment/config the tool reads itself, never in URLs or argv.",
    },
    "layer1_uninspected_tool": {
        "what": "This tool has no payload the static layers can inspect (or is MCP/unclassified), so it cannot be verified safe.",
        "fix": "Confirm the call is intended; for MCP tools, pin the server (`mb pin-mcp` / doctor) to enable trusted-tier verdicts.",
    },
    "layer1_unrecognized_shell_command_escalate": {
        "what": "A command in the pipeline is not on the allowlist, so its safety is unknown.",
        "fix": "Use an allowlisted equivalent, or add the command to a policy allowlist if it is genuinely read-only.",
    },
    "layer1_protected_path_shell_read_escalate": {
        "what": "The command reads or traverses a protected path.",
        "fix": "Restrict the command to non-protected paths, or get explicit operator confirmation.",
    },
    "layer1_sensitive_env_access_escalate": {
        "what": "The command inspects or overrides sensitive environment variables.",
        "fix": "Avoid printing/exporting credential-bearing env vars; reference them indirectly instead.",
    },
    "layer1_shell_redirection_escalate": {
        "what": "Output redirection targets a location the gate treats as needing confirmation.",
        "fix": "Redirect to a project-local path, or split the command.",
    },
    "layer1_strict_ast_allowlist_escalate": {
        "what": "Strict-AST mode flagged an unapproved import or dynamic call.",
        "fix": "Use an allowlisted import, or extend allowed_ast_imports in the policy file.",
    },
    "layer1_diff_scale_exceeded": {
        "what": "The diff exceeds max_diff_lines (high blast radius).",
        "fix": "Split the change into smaller reviewable hunks.",
    },
    "layer1_partial_hunk_without_source": {
        "what": "Only a hunk was provided without the file's full source, so context checks are inconclusive.",
        "fix": "Provide the full file (or let the hook auto-load it from disk); file fragments escalate.",
    },
    "layer1_imports_flagged_module": {
        "what": "The proposed code imports a local module whose own source contains vetoed constructs.",
        "fix": "Review the imported module's findings first; refactor or avoid it.",
    },
    "layer2_conformal_abstention_ood": {
        "what": "The routing classifier abstained: the request looks out-of-distribution for its calibration.",
        "fix": "This is advisory routing only; confirm the chosen tool manually.",
    },
    "layer3_default_escalation": {
        "what": "No structural signal was available, so the request escalated by default (fail-closed).",
        "fix": "Provide more concrete context (tool, path, command) so a structural layer can decide.",
    },
}


def explain_entry(entry: Dict[str, Any]) -> str:
    """Render a human explanation for one journal entry: what happened, why, and the fix."""
    eff_dec, reason = _effective_decision_and_reason(entry)
    base = reason.split(":")[0]
    info = REASON_EXPLANATIONS.get(base)
    lines = [format_entry(entry)]
    lines.append("")
    lines.append("EXPLANATION")
    lines.append("-" * 60)
    if entry.get("shadow_mode") and entry.get("shadow_decision"):
        lines.append(f"Shadow: projected enforce verdict={eff_dec} ({reason})")
    if info is None:
        lines.append(f"Reason code `{reason}` has no canned explanation.")
        lines.append(f"Interpretation: decision={eff_dec} label={entry.get('label')}")
    else:
        lines.append(f"Why:   {info['what']}")
        lines.append(f"Label: {entry.get('label', 'N/A')}")
        lines.append(f"Fix:   {info['fix']}")
    sha = entry.get("input_sha256")
    if sha:
        lines.append(f"Correlate: input_sha256={sha}")
    return "\n".join(lines)


def print_daily(entries: List[Dict[str, Any]], days: int = 7) -> None:
    """Print per-UTC-day decision counts and top reason codes for the last `days` days."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        ts = e.get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        day = time.strftime("%Y-%m-%d", time.gmtime(float(ts)))
        b = buckets.setdefault(day, {"total": 0, "decisions": {}, "reasons": {}})
        b["total"] += 1
        d, raw_reason = _effective_decision_and_reason(e)
        b["decisions"][d] = b["decisions"].get(d, 0) + 1
        r = raw_reason.split(":")[0]
        b["reasons"][r] = b["reasons"].get(r, 0) + 1

    print("=" * 60)
    print(f"AETHERIS MICROBRAINS - DAILY DIGEST (last {days} day(s), UTC)")
    print("=" * 60)
    if not buckets:
        print("No timestamped journal entries found.")
        return
    for day in sorted(buckets)[-days:]:
        b = buckets[day]
        decisions = ", ".join(f"{k}={v}" for k, v in sorted(b["decisions"].items()))
        top_reasons = sorted(b["reasons"].items(), key=lambda kv: -kv[1])[:3]
        reasons = ", ".join(f"{k} ({v})" for k, v in top_reasons)
        print(f"\n{day}  total={b['total']}")
        print(f"  decisions: {decisions}")
        print(f"  top reasons: {reasons}")
    print("=" * 60)


def print_shadow_report(entries: List[Dict[str, Any]]) -> None:
    """Aggregate shadow-mode journal entries into a prompt-fatigue / impact estimate.

    Shows what enforcement WOULD have done had the gate been in enforce mode:
    veto/deny rates, escalate (ask) rates that predict operator prompt fatigue,
    and the top shadow reason codes.
    """
    shadow = [e for e in entries if e.get("shadow_mode") or e.get("shadow_reason_code") or e.get("shadow_decision")]
    print("=" * 60)
    print(f"AETHERIS MICROBRAINS - SHADOW REPORT ({len(shadow)} shadow events of {len(entries)} total)")
    print("=" * 60)
    if not shadow:
        print("No shadow-mode events in the loaded journal window.")
        print("Install with `install_hook.py --shadow` to collect them.")
        return

    decisions: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    for e in shadow:
        d, raw_reason = _effective_decision_and_reason(e)
        decisions[d] = decisions.get(d, 0) + 1
        r = raw_reason.split(":")[0]
        reasons[r] = reasons.get(r, 0) + 1

    total = len(shadow)
    escalate_n = decisions.get("escalate", 0)
    deny_n = decisions.get("deny", 0)
    # Distinct UTC days with shadow traffic -> projected asks per active day.
    active_days = len({time.strftime("%Y-%m-%d", time.gmtime(float(e["timestamp"])))
                       for e in shadow if isinstance(e.get("timestamp"), (int, float))})

    print("\nWould-be verdicts (enforce-mode projection):")
    for d, count in sorted(decisions.items()):
        pct = round((count / total) * 100, 1)
        print(f"  • {d.upper():<10} {count:>5} ({pct:>5.1f}%)")
    if active_days:
        print(f"\nPrompt-fatigue estimate: ~{escalate_n / active_days:.1f} ask(s)/active day "
              f"({escalate_n} escalates over {active_days} day(s))")
    print(f"Hard-block impact: {deny_n} call(s) would have been denied "
          f"({round((deny_n / total) * 100, 1) if total else 0}%)")

    print("\nTop shadow reason codes:")
    for r, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  • {r:<48} {count:>5}")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aetheris_microbrains.audit",
        description="Inspect and explain Microbrains decisions and audit logs for human review.",
    )
    parser.add_argument("--last", action="store_true", help="Show the most recent decision event")
    parser.add_argument("--tail", type=int, default=0, help="Show the last N decision events")
    parser.add_argument("--summary", action="store_true", help="Show aggregate summary of decisions")
    parser.add_argument("--explain", action="store_true", help="Explain the most recent decision: what happened, why, and the fix")
    parser.add_argument("--find", type=str, default="", help="Explain the entry matching this input_sha256 prefix (use with --explain)")
    parser.add_argument("--daily", action="store_true", help="Per-day decision digest with top reason codes")
    parser.add_argument("--days", type=int, default=7, help="Days to include for --daily (default 7)")
    parser.add_argument("--shadow-report", action="store_true", help="Enforce-mode projection + prompt-fatigue estimate from shadow events")
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
    read_all = bool(args.summary or args.daily or args.shadow_report or args.find)
    entries = get_journal_entries(journal_path=p, limit=None if read_all else max(100, args.tail))

    if args.summary:
        print_summary(entries)
        return 0

    if args.daily:
        print_daily(entries, days=max(1, args.days))
        return 0

    if args.shadow_report:
        print_shadow_report(entries)
        return 0

    if args.explain:
        target = None
        if args.find:
            for e in reversed(entries):
                sha = str(e.get("input_sha256") or "")
                if sha and sha.startswith(args.find):
                    target = e
                    break
            if target is None:
                print(f"No journal entry matches input_sha256 prefix {args.find!r}.")
                return 1
        else:
            target = entries[-1] if entries else None
        if target is None:
            print("No decision journal entries found.")
            return 0
        print(explain_entry(target))
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
