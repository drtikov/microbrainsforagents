"""Counterfactual Trace Replay Simulator & Offline Journal-to-Policy Synthesizer.

Pure Python stdlib only (zero NumPy or third-party dependencies).

Provides:
  - `load_journal_entries(journal_path)`:
      Safely loads JSONL decision journal records, skipping blank and malformed lines.
  - `replay_decision_journal(journal_path, *, target_profile="strict", ...)`:
      Replays recorded tool-call entries through `evaluate_hook_payload(..., log_decisions=False)`
      to measure counterfactual decision flips, latency statistics, Clopper-Pearson
      threshold calibration, and optional offline policy synthesis + HMAC-SHA256
      signed fleet bundle generation.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .fleet_bundle import create_signed_fleet_bundle
from .repair_hints import calibrate_thresholds_from_journal

_SHELL_TOOLS = frozenset({
    "execute_shell",
    "run_command",
    "bash",
    "shell",
    "shell_command",
    "exec",
    "exec_command",
    "execute_bash",
    "run_shell",
    "run_shell_command",
    "terminal",
})

_SAFE_PREFIX_EXES = frozenset({
    "git",
    "pytest",
    "python",
    "python3",
    "ruff",
    "flake8",
    "mypy",
    "black",
    "isort",
    "npm",
    "npx",
    "pnpm",
    "yarn",
    "cargo",
    "go",
    "ls",
    "pwd",
    "echo",
    "rg",
    "grep",
    "cat",
    "head",
    "tail",
    "wc",
    "find",
    "which",
    "node",
    "tsc",
    "eslint",
    "vitest",
    "jest",
    "make",
    "just",
    "uv",
    "poetry",
})

_SAFE_SUBCOMMANDS: Dict[str, frozenset[str]] = {
    "git": frozenset({
        "status",
        "diff",
        "log",
        "show",
        "branch",
        "rev-parse",
        "describe",
        "ls-files",
        "blame",
        "shortlog",
        "tag",
    }),
    "npm": frozenset({"test", "run", "ls", "list", "outdated", "audit"}),
    "pnpm": frozenset({"test", "run", "ls", "list", "outdated", "audit"}),
    "yarn": frozenset({"test", "run", "list", "outdated", "audit"}),
    "cargo": frozenset({"test", "check", "clippy", "fmt", "build", "bench", "doc", "metadata"}),
    "go": frozenset({"test", "vet", "build", "fmt", "list", "version", "env"}),
    "ruff": frozenset({"check", "format", "rule", "version"}),
    "uv": frozenset({"run", "pip", "version", "tree"}),
    "poetry": frozenset({"run", "show", "version", "check"}),
}

_UNSAFE_SHELL_METACHARS = (";", "&&", "||", "|", ">", "<", "`", "$(", "\n", "\r")
_PLAIN_SUBCMD_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,31}$")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_decision(raw: Any) -> str:
    """Normalize decision strings (`ask` -> `escalate`, `block` -> `deny`)."""
    dec = str(raw or "").strip().lower()
    if dec in ("allow", "approved", "pass"):
        return "allow"
    if dec in ("deny", "block", "blocked", "reject"):
        return "deny"
    if dec in ("escalate", "ask", "review"):
        return "escalate"
    return "other"


def _load_journal_with_stats(journal_path: Path | str) -> Tuple[List[Dict[str, Any]], int]:
    """Read a JSONL decision journal, returning `(entries, skipped_lines)`."""
    path = Path(journal_path).expanduser()
    if not path.is_file():
        return [], 0

    entries: List[Dict[str, Any]] = []
    skipped_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                skipped_lines += 1
                continue
            if not isinstance(parsed, dict):
                skipped_lines += 1
                continue
            entries.append(parsed)

    return entries, skipped_lines


def load_journal_entries(journal_path: Path | str) -> List[Dict[str, Any]]:
    """Load valid JSON object entries from a JSONL decision journal, skipping blank/malformed lines."""
    entries, _skipped = _load_journal_with_stats(journal_path)
    return entries


def _reconstruct_hook_payload(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Reconstruct a normalized hook payload from a recorded journal entry."""
    event_type = str(entry.get("event_type") or "").strip().lower()
    if event_type in {"hook_error", "session_summary", "attestation"}:
        return None

    tool_raw = str(
        entry.get("raw_tool")
        or entry.get("tool_name")
        or entry.get("tool")
        or ""
    ).strip()
    if not tool_raw:
        return None

    tool_input: Dict[str, Any] = {}
    raw_ti = entry.get("tool_input")
    if isinstance(raw_ti, Mapping):
        tool_input.update(dict(raw_ti))

    if entry.get("cmd") is not None and not any(
        k in tool_input for k in ("command", "cmd", "CommandLine")
    ):
        tool_input["command"] = entry["cmd"]

    if entry.get("path") is not None and not any(
        k in tool_input for k in ("file_path", "path", "Path", "target_file", "TargetFile")
    ):
        tool_input["file_path"] = entry["path"]

    if entry.get("diff") is not None and not any(
        k in tool_input for k in ("diff", "payload_diff", "patch", "content", "new_string")
    ):
        tool_input["diff"] = entry["diff"]

    payload: Dict[str, Any] = {
        "tool_name": tool_raw,
        "tool": tool_raw,
        "tool_input": tool_input,
    }
    if entry.get("cwd"):
        payload["cwd"] = str(entry["cwd"])
    return payload


def _extract_command_from_entry(entry: Mapping[str, Any]) -> str:
    """Extract shell command string from a journal entry if present."""
    if isinstance(entry.get("cmd"), str) and entry["cmd"].strip():
        return entry["cmd"].strip()
    raw_ti = entry.get("tool_input")
    if isinstance(raw_ti, Mapping):
        for key in ("command", "cmd", "CommandLine"):
            val = raw_ti.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, (list, tuple)) and val:
                return shlex.join([str(x) for x in val if x is not None]).strip()
    return ""


def _extract_safe_command_prefix(cmd: str) -> Optional[str]:
    """Return a normalized safe read-only/dev command prefix, or `None` if unsafe/unrecognized."""
    text = str(cmd or "").strip()
    if not text:
        return None
    if any(meta in text for meta in _UNSAFE_SHELL_METACHARS):
        return None

    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None

    idx = 0
    while idx < len(tokens) and _ENV_ASSIGN_RE.match(tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return None

    raw_exe = tokens[idx].replace("\\", "/").rsplit("/", 1)[-1].lower()
    for ext in (".exe", ".cmd", ".bat"):
        if raw_exe.endswith(ext) and len(raw_exe) > len(ext):
            raw_exe = raw_exe[: -len(ext)]
            break

    if raw_exe not in _SAFE_PREFIX_EXES:
        return None

    rest = tokens[idx + 1 :]
    if raw_exe in ("python", "python3"):
        # Only synthesize module invocations like `python -m pytest` / `python -m unittest`
        if len(rest) >= 2 and rest[0] == "-m" and rest[1].lower() in {"pytest", "unittest", "ruff", "mypy"}:
            return f"{raw_exe} -m {rest[1].lower()}"
        return None

    if raw_exe in _SAFE_SUBCOMMANDS:
        allowed_subs = _SAFE_SUBCOMMANDS[raw_exe]
        for tok in rest:
            if tok.startswith("-"):
                continue
            sub = tok.lower()
            if sub in allowed_subs:
                return f"{raw_exe} {sub}"
            return None
        return None

    if rest and _PLAIN_SUBCMD_RE.match(rest[0]) and not rest[0].startswith("-"):
        sub = rest[0].lower()
        if raw_exe in {"pytest", "ruff", "mypy", "flake8", "black", "isort", "eslint", "vitest", "jest"}:
            return raw_exe
        if sub in {"check", "test", "lint", "build", "status", "version", "list"}:
            return f"{raw_exe} {sub}"

    return raw_exe


def _compute_percentile(values: Sequence[float], pct: float) -> float:
    """Compute nearest-rank percentile over a non-empty sequence of floats."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    rank = max(1, min(len(ordered), int(math.ceil((pct / 100.0) * len(ordered)))))
    return ordered[rank - 1]


def _synthesize_policy_from_entries(
    entries: Sequence[Mapping[str, Any]],
    replayed_decisions: Sequence[str],
    *,
    target_profile: str,
    calibration: Mapping[str, Any],
    journal_path: Path | str,
    min_occurrences: int = 2,
) -> Dict[str, Any]:
    """Synthesize a policy recommendation from recurring safe `allow` command prefixes."""
    prefix_counts: Counter[str] = Counter()
    for entry, replayed_dec in zip(entries, replayed_decisions):
        orig_dec = _normalize_decision(entry.get("decision"))
        if orig_dec != "allow" and replayed_dec != "allow":
            continue
        tool_name = str(
            entry.get("tool") or entry.get("raw_tool") or entry.get("tool_name") or ""
        ).strip().lower()
        if tool_name not in _SHELL_TOOLS:
            continue
        cmd = _extract_command_from_entry(entry)
        prefix = _extract_safe_command_prefix(cmd)
        if prefix:
            prefix_counts[prefix] += 1

    recurring = [
        (prefix, count)
        for prefix, count in sorted(prefix_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if count >= max(1, int(min_occurrences))
    ]
    if not recurring and prefix_counts:
        # Fall back to observed safe prefixes if journal is small (<2 repeats)
        recurring = sorted(prefix_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    custom_rules: List[Dict[str, Any]] = []
    recommended_prefixes: List[Dict[str, Any]] = []
    for prefix, count in recurring:
        slug = _SLUG_RE.sub("_", prefix.lower()).strip("_") or "cmd"
        recommended_prefixes.append({"prefix": prefix, "count": int(count)})
        custom_rules.append({
            "id": f"synth_allow_{slug}",
            "action": "allow",
            "tool": r"^(execute_shell|run_command|bash|shell)$",
            "cmd_regex": rf"^{re.escape(prefix)}(\s|$)",
            "reason": f"Synthesized from {count} observed safe allow journal entries",
        })

    return {
        "profile": target_profile,
        "custom_rules": custom_rules,
        "recommended_command_prefixes": recommended_prefixes,
        "recommended_threshold": calibration.get("recommended_threshold"),
        "noisy_reason_codes": list(calibration.get("noisy_reason_codes") or []),
        "source_journal": str(journal_path),
    }


def replay_decision_journal(
    journal_path: Path | str,
    *,
    target_profile: str = "strict",
    host: Optional[str] = None,
    synthesize_policy: bool = False,
    org_id: str = "local-org",
    bundle_version: int = 1,
    hmac_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Replay a JSONL decision journal under a counterfactual policy profile.

    Args:
        journal_path: Path to the JSONL decision journal file.
        target_profile: Counterfactual policy profile (`strict`, `dev`, or `guardrails`).
        host: Optional host adapter override (defaults to entry's `host` or `"claude"`).
        synthesize_policy: When True, synthesizes a policy recommendation from recurring
            safe read-only/dev command prefixes observed in `allow` entries.
        org_id: Organization identifier used when signing a fleet policy bundle.
        bundle_version: Monotonic integer version for the signed fleet policy bundle.
        hmac_key: Optional HMAC-SHA256 signing key; when provided alongside
            `synthesize_policy=True`, produces a signed fleet bundle via
            `create_signed_fleet_bundle`.

    Returns:
        Structured counterfactual replay summary dictionary.
    """
    from ..hook import evaluate_hook_payload
    from .runtime_paths import verify_jsonl_hash_chain

    chain_integrity = verify_jsonl_hash_chain(journal_path)
    entries, skipped_lines = _load_journal_with_stats(journal_path)

    original_counts: Dict[str, int] = {"allow": 0, "deny": 0, "escalate": 0, "other": 0}
    replayed_counts: Dict[str, int] = {"allow": 0, "deny": 0, "escalate": 0, "other": 0}
    decision_flips: List[Dict[str, Any]] = []
    newly_blocked = 0
    newly_allowed = 0
    latencies_ms: List[float] = []

    replayed_entries_meta: List[Mapping[str, Any]] = []
    replayed_decisions_list: List[str] = []
    calibration_records: List[Dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        payload = _reconstruct_hook_payload(entry)
        if payload is None:
            continue

        resolved_host = str(host or entry.get("host") or "claude").strip() or "claude"
        t0 = time.perf_counter()
        result = evaluate_hook_payload(
            payload,
            profile=target_profile,
            host=resolved_host,
            log_decisions=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        orig_dec = _normalize_decision(entry.get("decision"))
        rep_dec = _normalize_decision(result.get("decision"))

        original_counts[orig_dec] = original_counts.get(orig_dec, 0) + 1
        replayed_counts[rep_dec] = replayed_counts.get(rep_dec, 0) + 1

        replayed_entries_meta.append(entry)
        replayed_decisions_list.append(rep_dec)

        cal_record: Dict[str, Any] = {
            "decision": rep_dec,
            "reason_code": str(result.get("reason_code") or "unknown"),
        }
        for label_key in ("false_positive", "verified_correct", "outcome", "resolution", "human_decision", "feedback"):
            if label_key in entry:
                cal_record[label_key] = entry[label_key]
        calibration_records.append(cal_record)

        if orig_dec != rep_dec:
            tool_label = str(entry.get("tool") or entry.get("raw_tool") or payload["tool_name"])
            decision_flips.append({
                "index": idx,
                "tool": tool_label,
                "original_decision": orig_dec,
                "replayed_decision": rep_dec,
                "reason_code": str(result.get("reason_code") or ""),
            })
            if orig_dec == "allow" and rep_dec in ("deny", "escalate"):
                newly_blocked += 1
            elif orig_dec in ("deny", "escalate") and rep_dec == "allow":
                newly_allowed += 1

    replayed_count = len(latencies_ms)
    latency_mean_ms = round(sum(latencies_ms) / float(replayed_count), 4) if replayed_count else 0.0
    latency_p95_ms = round(_compute_percentile(latencies_ms, 95.0), 4) if replayed_count else 0.0

    calibration = calibrate_thresholds_from_journal(calibration_records)

    synthesized_policy: Optional[Dict[str, Any]] = None
    signed_bundle: Optional[Dict[str, Any]] = None
    if synthesize_policy:
        synthesized_policy = _synthesize_policy_from_entries(
            replayed_entries_meta,
            replayed_decisions_list,
            target_profile=target_profile,
            calibration=calibration,
            journal_path=journal_path,
        )
        if hmac_key is not None:
            signed_bundle = create_signed_fleet_bundle(
                org_id=org_id,
                version=bundle_version,
                policy=synthesized_policy,
                signing_key=hmac_key,
            )

    return {
        "journal_path": str(journal_path),
        "target_profile": target_profile,
        "host": host,
        "chain_integrity": chain_integrity,
        "total_entries": len(entries),
        "replayed_entries": replayed_count,
        "skipped_lines": skipped_lines,
        "original_counts": original_counts,
        "replayed_counts": replayed_counts,
        "decision_flips": decision_flips,
        "newly_blocked": newly_blocked,
        "newly_allowed": newly_allowed,
        "latency_mean_ms": latency_mean_ms,
        "latency_p95_ms": latency_p95_ms,
        "calibration": calibration,
        "synthesized_policy": synthesized_policy,
        "signed_bundle": signed_bundle,
    }


__all__ = [
    "load_journal_entries",
    "replay_decision_journal",
]
