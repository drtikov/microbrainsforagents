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

_INITIAL_JOURNAL_PATH = get_default_journal_path()
_DEFAULT_JOURNAL_PATH = _INITIAL_JOURNAL_PATH


def get_journal_entries(
    journal_path: Optional[Path | str] = None, limit: Optional[int] = 50
) -> List[Dict[str, Any]]:
    """Read recent journal entries from decision_journal.jsonl (`limit=None` reads all)."""
    default_jp = (
        Path(_DEFAULT_JOURNAL_PATH)
        if _DEFAULT_JOURNAL_PATH != _INITIAL_JOURNAL_PATH
        else get_default_journal_path()
    )
    raw_path = journal_path or os.getenv("AETHERIS_JOURNAL_PATH") or default_jp
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
        box.append(f"│ Risk Bound: Upper Error Bound 95% = {conf_bound}")
    if preview:
        box.append(f"│ Payload:    {preview}")
    if entry.get("escalation_contract"):
        contract = entry["escalation_contract"]
        box.append(f"│ Escalation: Target={contract.get('recipient')} Timeout={contract.get('timeout_sec')}s")
    if isinstance(entry.get("rule_provenance"), dict):
        prov = entry["rule_provenance"]
        by = prov.get("approved_by") or "operator"
        rat = prov.get("rationale") or ""
        vit = prov.get("vitality")
        vit_str = f" vitality={vit}" if vit is not None else ""
        box.append(f"│ Provenance: approved_by={by}{vit_str} | {rat}"[:96])
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
    "layer0_unregistered_tool": {
        "what": "The tool name is not in the configured active `available_tools` catalog.",
        "fix": "Verify the tool name spelling or include it in `available_tools` passed to `decide()`.",
    },
    "layer0_external_requires_confirmation": {
        "what": "Policy (`require_approval_for_external=True`) requires explicit human confirmation for external/network actions.",
        "fix": "Confirm with the user, or add approved domains to `PolicyConfig.allowed_web_domains`.",
    },
    "layer0_irreversible_requires_confirmation": {
        "what": "This operation is classified as IRREVERSIBLE_DESTRUCTIVE (`git_push`, `delete_file`, `drop_db`) and requires confirmation.",
        "fix": "Review blast radius carefully and confirm explicitly with the operator.",
    },
    "layer0_policy_pass": {
        "what": "Layer 0 policy checks passed.",
        "fix": "No action needed.",
    },
    "layer0_custom_policy_rule": {
        "what": "A declarative `PolicyConfig.custom_rules` rule matched and enforced a policy action.",
        "fix": "Adjust the action/arguments to comply with the custom policy rule, or update `custom_rules` in your policy file.",
    },
    "layer0_session_state_corrupted": {
        "what": "The cross-turn session security state file failed integrity checks.",
        "fix": "Delete the session state file (it regenerates); investigate what corrupted it before continuing sensitive work.",
    },
    "layer0_session_taint_lethal_trifecta": {
        "what": "Session taint gate blocked an egress/mutation call after untrusted external content AND sensitive credentials were both accessed in this session.",
        "fix": "Start a clean session (or inspect `provenance_summary()`) so untrusted inputs and sensitive secrets cannot mix.",
    },
    "layer0_session_prompt_injection_taint_escalate": {
        "what": "Indirect prompt-injection markers were detected earlier in this session; subsequent mutations/egress require confirmation.",
        "fix": "Review the originating untrusted tool output in the journal and confirm intentionally before continuing.",
    },
    "layer1_structural_verified": {
        "what": "All Layer 1 structural (AST, shell, secret egress) checks passed.",
        "fix": "No action needed.",
    },
    "layer1_hunk_only_checked": {
        "what": "A partial diff hunk passed structural checks without full on-disk source reconstruction.",
        "fix": "No action needed (or pass `file_content` / enable `require_full_source_for_allow` for stricter full-file verification).",
    },
    "layer1_null_byte_veto": {
        "what": "A NUL byte (`\\x00`) was detected in path, command, or diff arguments (classic C-string truncation attack).",
        "fix": "Remove null bytes and resubmit sanitized structured inputs.",
    },
    "layer1_ast_security_veto": {
        "what": "The proposed code contains a construct the AST layer vetoes (dangerous calls, obfuscated secrets, dynamic execution).",
        "fix": "Rewrite the code to avoid the flagged construct; run `python -m aetheris_microbrains.audit --explain` again after fixing for the specific finding label.",
    },
    "layer1_ast_dev_escalate": {
        "what": "In `dev` profile, a medium-risk process or filesystem construct (`subprocess.run`, `os.remove`, relative `rmtree`) requires human confirmation.",
        "fix": "Confirm with the user, or add a targeted justified inline comment if appropriate for your dev policy.",
    },
    "layer1_ast_suppressed_escalate": {
        "what": "An `# aetheris: allow ...` inline suppression was present on the diff and requires human confirmation in `strict` profile.",
        "fix": "Confirm the suppression or remove the flagged construct instead of suppressing it.",
    },
    "layer1_ast_syntax_error": {
        "what": "The proposed code does not parse as valid Python (or JSON).",
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
    "layer1_mcp_trusted_readonly": {
        "what": "An allowlisted read-only tool on a pinned/trusted MCP server was permitted structurally.",
        "fix": "No action needed.",
    },
    "layer1_mcp_write_tool_escalate": {
        "what": "A pinned/trusted MCP server tool with mutating or unknown verb semantics was escalated for human confirmation.",
        "fix": "Confirm the MCP mutation explicitly if intended.",
    },
    "layer1_uninspected_tool": {
        "what": "This tool has no payload the static layers can inspect (or is an unpinned/drifted MCP server), so it cannot be verified safe.",
        "fix": "Confirm the call is intended; for MCP tools, pin the server (`python -m aetheris_microbrains.doctor --pin-mcp <server>` or `aetheris-doctor --pin-mcp <server>`) to enable trusted-tier verdicts.",
    },
    "layer1_unrecognized_shell_command_escalate": {
        "what": "A command in the pipeline is not on the allowlist, so its safety is unknown.",
        "fix": "Use an allowlisted equivalent, or switch to `--profile dev` / add an explicit policy rule if genuinely safe.",
    },
    "layer1_protected_path_shell_read": {
        "what": "The shell command reads or traverses a protected path.",
        "fix": "Restrict the command to non-protected paths, or get explicit operator confirmation.",
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
    "layer1_strict_ast_": {
        "what": "Strict-AST mode flagged an unapproved import or dynamic call.",
        "fix": "Use an allowlisted import, or extend allowed_ast_imports in the policy file.",
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
    "layer1_error": {
        "what": "An unexpected exception occurred while evaluating Layer 1 rules (fail-closed).",
        "fix": "Inspect stderr / journal error details and re-run after resolving malformed inputs.",
    },
    "layer1_fail_closed": {
        "what": "An internal parser error aborted evaluation of a mutating/external call (failed closed to deny).",
        "fix": "Run `python -m aetheris_microbrains.doctor` and inspect the failing payload format.",
    },
    "layer1_escalate_on_error": {
        "what": "An internal parser error occurred during a non-mutating check and escalated safely.",
        "fix": "Inspect the payload and confirm manually if safe.",
    },
    "layer2_conformal_singleton": {
        "what": "The calibrated routing classifier selected a unique tool with satisfied conformal guarantee.",
        "fix": "No action needed (advisory routing recommendation).",
    },
    "layer2_conformal_abstention_ood": {
        "what": "The routing classifier abstained: the request looks out-of-distribution for its calibration.",
        "fix": "This is advisory routing only; confirm the chosen tool manually.",
    },
    "layer3_default_escalation": {
        "what": "No structural signal was available, so the request escalated by default (fail-closed).",
        "fix": "Provide more concrete context (tool, path, command) so a structural layer can decide.",
    },
    "hygiene_redundant_whole_read": {
        "what": "Context hygiene blocked a repeat whole-file Read on an unmodified file already loaded in this session.",
        "fix": "Read a specific line range with `offset` and `limit`, or edit directly.",
    },
    "hygiene_oversized_whole_read": {
        "what": "Context hygiene intercepted an oversized whole-file Read and returned a structural symbol map instead.",
        "fix": "Pick the target symbol line range from the returned outline and issue a ranged Read.",
    },
    "hygiene_shell_read_oversized": {
        "what": "Context hygiene blocked an unbounded shell file-reader (`cat`/`type`/`Get-Content`) over the size limit.",
        "fix": "Pipe through `grep`/`rg`/`Select-String` or `head -n`/`tail -n`, or use a ranged Read.",
    },
    "hygiene_spilled_output_whole_read": {
        "what": "Context hygiene blocked an unbounded whole-file read of a spilled tool-output artifact.",
        "fix": "Search the spill file with `grep`/`rg` or inspect the tail with `tail -n 40`.",
    },
    "hygiene_stdin_hang_risk": {
        "what": "An interpreter invocation (`python -` or `node -`) had no pipe, redirect, or heredoc feeding stdin.",
        "fix": "Supply a script path, `-c` expression, `<<` heredoc, or upstream pipe.",
    },
    "hygiene_root_origin_search": {
        "what": "A recursive search targeted a filesystem or drive root directly.",
        "fix": "Scope the search command to a repository or target directory path.",
    },
    "layer0_stale_rule_reconfirm": {
        "what": "An approved custom allow rule became stale (expired half-life or recent failures) and downgraded safely to escalate.",
        "fix": "Reconfirm the rule in `RuleVitalityLedger` or review the pending rule queue before continuing.",
    },
    "layer0_oversight_flood_veto": {
        "what": "Repeated escalations exceeded the oversight capacity threshold in a tainted/high-rate session.",
        "fix": "Inspect the session taint state, clear unrelated escalations, or approve a verified narrow rule.",
    },
    "layer0_custom_policy_rule": {
        "what": "Matched an operator-approved custom policy rule.",
        "fix": "Inspect attached provenance (`approved_by`, `rationale`) or modify `custom_rules` in your policy file.",
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
    if isinstance(entry.get("rule_provenance"), dict):
        prov = entry["rule_provenance"]
        lines.append(
            f"Provenance: approved_by={prov.get('approved_by') or 'operator'} | "
            f"vitality={prov.get('vitality', 'n/a')} | {prov.get('rationale') or ''}"
        )
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


class FrictionDigestBuilder:
    """Deterministic two-stage compressor turning decision journals into compact triage tables."""

    @staticmethod
    def recommend_action(reason_code: str, count: int) -> str:
        rc = str(reason_code or "").lower()
        if "oversized_whole_read" in rc or "diff_scale_exceeded" in rc:
            return "split_file"
        if "redundant_whole_read" in rc or "shell_read" in rc:
            return "tune_threshold" if count >= 4 else "benign"
        if "stdin_hang_risk" in rc or "root_origin_search" in rc:
            return "scaffold_helper"
        if "unrecognized_shell_command" in rc or "strict_ast" in rc:
            return "allowlist_rule" if count >= 3 else "benign"
        return "benign"

    @classmethod
    def build(cls, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        hotspots: Dict[str, Dict[str, int]] = {}
        matrix: Dict[tuple[str, str], int] = {}
        stall_signals: Dict[str, int] = {}

        for e in entries:
            dec, raw_reason = _effective_decision_and_reason(e)
            reason = raw_reason.split(":")[0]
            tool = str(e.get("tool") or "unknown")
            target = str(e.get("target") or e.get("path") or "").strip()

            if target:
                hb = hotspots.setdefault(target, {"events": 0, "blocks": 0})
                hb["events"] += 1
                if dec in {"deny", "escalate"}:
                    hb["blocks"] += 1

            if dec in {"deny", "escalate"}:
                key = (reason, tool)
                matrix[key] = matrix.get(key, 0) + 1

            if reason in {
                "hygiene_stdin_hang_risk",
                "hygiene_root_origin_search",
                "hygiene_redundant_whole_read",
                "hygiene_spilled_output_whole_read",
            }:
                stall_signals[reason] = stall_signals.get(reason, 0) + 1

        top_hotspots = sorted(
            ({"target": k, **v} for k, v in hotspots.items()),
            key=lambda row: (-row["blocks"], -row["events"], row["target"]),
        )[:10]

        friction_rows = [
            {
                "reason_code": rc,
                "tool": tl,
                "count": cnt,
                "recommendation": cls.recommend_action(rc, cnt),
            }
            for (rc, tl), cnt in sorted(matrix.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        ][:15]

        return {
            "total_entries": len(entries),
            "io_hotspots": top_hotspots,
            "friction_matrix": friction_rows,
            "stall_signals": stall_signals,
        }


def build_friction_digest(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build deterministic pre-digest tables from journal entries for triage or light adjudication."""
    return FrictionDigestBuilder.build(entries)


def print_friction_digest(entries: List[Dict[str, Any]], as_json: bool = False) -> None:
    """Render deterministic friction digest tables."""
    digest = build_friction_digest(entries)
    if as_json:
        print(json.dumps(digest, indent=2, ensure_ascii=False))
        return
    print("=" * 64)
    print(f"AETHERIS MICROBRAINS - FRICTION DIGEST ({digest['total_entries']} entries)")
    print("=" * 64)
    if not digest["friction_matrix"] and not digest["io_hotspots"]:
        print("No friction or hotspot signals detected.")
        print("=" * 64)
        return
    if digest["friction_matrix"]:
        print("\nGate Friction Matrix:")
        print(f"  {'REASON CODE':<36} {'TOOL':<14} {'CNT':>4}  RECOMMENDATION")
        for row in digest["friction_matrix"]:
            print(
                f"  {row['reason_code']:<36} {row['tool']:<14} {row['count']:>4}  {row['recommendation']}"
            )
    if digest["io_hotspots"]:
        print("\nI/O Hotspots:")
        for h in digest["io_hotspots"]:
            print(f"  - {h['target']} (events={h['events']}, blocks={h['blocks']})")
    if digest["stall_signals"]:
        print("\nStall / Context Signals:")
        for k, v in sorted(digest["stall_signals"].items()):
            print(f"  - {k}: {v}")
    print("=" * 64)


def print_savings_report(
    economics_path: Optional[str] = None,
    session_id: Optional[str] = None,
    *,
    as_json: bool = False,
    statusline_only: bool = False,
) -> None:
    """Print preserved-token economics summary or single-line status bar text."""
    from .core.economics_ledger import format_statusline, summarize_savings

    summary = summarize_savings(ledger_path=economics_path or None, session_id=session_id or None)
    if statusline_only:
        print(format_statusline(summary, as_json=as_json))
        return
    if as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    print("=" * 60)
    print("AETHERIS MICROBRAINS - TOKEN SAVINGS LEDGER")
    print("=" * 60)
    print(f"Hygiene Events:   {summary['events']}")
    print(f"Avoided Bytes:    {summary['avoided_bytes']}")
    print(f"Tokens Preserved: {summary['tokens_preserved']}")
    print(f"Est. USD Saved:   ${summary['est_usd_saved']:.4f}")
    if summary["by_reason"]:
        print("\nBy Reason Code:")
        for rc, bucket in sorted(summary["by_reason"].items(), key=lambda kv: -kv[1]["tokens"]):
            print(f"  - {rc:<36} {bucket['tokens']:>7} tok ({bucket['events']}x)")
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
    parser.add_argument("--digest", action="store_true", help="Compact deterministic friction & I/O hotspot digest tables")
    parser.add_argument("--savings", action="store_true", help="Show token & cost preservation ledger summary")
    parser.add_argument("--statusline", action="store_true", help="Emit compact single-line token savings statusbar")
    parser.add_argument("--json", action="store_true", help="Emit --digest/--savings/--statusline output as JSON")
    parser.add_argument("--economics-path", type=str, default="", help="Path to economics_ledger.jsonl")
    parser.add_argument("--session-id", type=str, default="", help="Optional session_id filter for --savings/--statusline")
    parser.add_argument("--journal-path", type=str, default="", help="Path to decision_journal.jsonl")
    parser.add_argument("--eval-policy", type=str, default="", help="Replay journal & constitutional probes against candidate PolicyConfig file")
    parser.add_argument("--base-policy", type=str, default="", help="Optional base PolicyConfig file for --eval-policy comparison")
    parser.add_argument("--propose-rules", action="store_true", help="Synthesize verified candidate allow rules from recurring journal escalations")
    parser.add_argument("--min-occurrences", type=int, default=3, help="Minimum escalation repetitions for --propose-rules")
    parser.add_argument("--stage", action="store_true", help="Stage verified candidates from --propose-rules into PendingRuleQueue")
    parser.add_argument("--approve-rule", type=str, default="", help="Approve a staged pending rule by ID")
    parser.add_argument("--approver", type=str, default="operator", help="Approver identity recorded for --approve-rule")
    parser.add_argument("--policy-out", type=str, default="", help="Optional JSON policy file to write approved custom rule into")

    args = parser.parse_args(argv)
    # Windows consoles default to cp1252 and crash on box-drawing/bullet characters
    # in formatted entries; force UTF-8 with replacement instead of a hard failure.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    if args.eval_policy:
        from .core.layered_decision_engine import PolicyConfig
        from .core.policy_governance import evaluate_policy_candidate

        cand_pol = PolicyConfig.from_file(args.eval_policy)
        base_pol = PolicyConfig.from_file(args.base_policy) if args.base_policy else None
        j_entries = get_journal_entries(journal_path=args.journal_path or None, limit=None)
        report = evaluate_policy_candidate(
            cand_pol, base_policy=base_pol, journal_entries=j_entries, include_builtin_probes=True
        )
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(f"promote_safe={report['promote_safe']} | probes={report['probes_checked']} | replayed={report['journal_replayed']} | newly_allowed={report['newly_allowed']} | upper_risk_bound_95={report['upper_risk_bound_95']}")
            if report["safety_regressions"]:
                print(f"Safety regressions: {report['safety_regressions']}")
        return 0 if report["promote_safe"] else 2

    if args.propose_rules:
        from .core.layered_decision_engine import PolicyConfig
        from .core.policy_governance import PendingRuleQueue, synthesize_policy_candidates

        base_pol = PolicyConfig.from_file(args.base_policy) if args.base_policy else None
        j_entries = get_journal_entries(journal_path=args.journal_path or None, limit=None)
        cands = synthesize_policy_candidates(
            j_entries, min_occurrences=max(1, args.min_occurrences), base_policy=base_pol
        )
        staged = PendingRuleQueue().stage(cands, verified_only=True) if args.stage else 0
        out_obj = {"candidates": cands, "staged": staged}
        if args.json:
            print(json.dumps(out_obj, indent=2, ensure_ascii=False))
        else:
            print(f"Proposed {len(cands)} candidate rule(s); staged={staged}")
            for c in cands:
                print(f"  - {c['id']} (verified={c['verified']}, support={c['support_count']}): {c['reason']}")
        return 0

    if args.approve_rule:
        from .core.layered_decision_engine import PolicyConfig
        from .core.policy_governance import PendingRuleQueue

        base_pol = PolicyConfig.from_file(args.base_policy) if args.base_policy else None
        approved = PendingRuleQueue().approve(
            args.approve_rule,
            approver=args.approver,
            base_policy=base_pol,
            policy_out_path=args.policy_out or None,
        )
        if args.json:
            print(json.dumps(approved, indent=2, ensure_ascii=False))
        else:
            print(f"Approved rule {approved['id']} (by {args.approver})")
        return 0

    if args.savings or args.statusline:
        print_savings_report(
            economics_path=args.economics_path or None,
            session_id=args.session_id or None,
            as_json=args.json,
            statusline_only=args.statusline,
        )
        return 0

    p = args.journal_path or None
    read_all = bool(args.summary or args.daily or args.shadow_report or args.digest or args.find)
    entries = get_journal_entries(journal_path=p, limit=None if read_all else max(100, args.tail))

    if args.digest:
        print_friction_digest(entries, as_json=args.json)
        return 0

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
