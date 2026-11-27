"""Deterministic PostToolUse structural output compressor (stdlib-only).

Distills verbose tool outputs (test suites, compiler/build logs, repetitive
warnings, and oversized CLI dumps) into compact, high-signal representations
before they enter the agent context window, while preserving failures,
tracebacks, assertions, and summary footers verbatim.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .economics_ledger import estimate_tokens

# ---------------------------------------------------------------------------
# Test / build log detection & line classification patterns
# ---------------------------------------------------------------------------

_PYTEST_DETECT_RE = re.compile(
    r"(?m)(?:^=+ test session starts =+$"
    r"|^collected \d+ items?\b"
    r"|^=+ .*?\b(?:passed|failed|errors?|skipped|xfailed)\b.*? =+$"
    r"|::\S+\s+(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b)"
)

_CARGO_DETECT_RE = re.compile(
    r"(?m)(?:^running \d+ tests?$"
    r"|^test result:\s+(?:ok|FAILED)\."
    r"|^test \S+ \.\.\. (?:ok|FAILED|ignored)$)"
)

_UNITTEST_DETECT_RE = re.compile(
    r"(?m)(?:^Ran \d+ tests? in [\d.]+s$"
    r"|^\S+ \(.+\) \.\.\. (?:ok|FAIL|ERROR|skipped)\b)"
)

_JS_TEST_DETECT_RE = re.compile(
    r"(?m)(?:^(?:PASS|FAIL)\s+\S+"
    r"|^Test Suites:\s+"
    r"|^Test Files\s+\d+"
    r"|^Tests:\s+\d+\s+(?:passed|failed)"
    r"|^\s*[✓✔×✗]\s+\S+)"
)

_FAILURE_BLOCK_START_RE = re.compile(
    r"^(?:"
    r"=+\s*(?:FAILURES|ERRORS|short test summary info)\s*=+$"
    r"|failures:\s*$"
    r"|Traceback \(most recent call last\):\s*$"
    r"|----\s+.+?\s+stdout\s+----\s*$"
    r"|(?:FAIL|ERROR):\s+\S+"
    r"|●\s+\S+"
    r")"
)

_CRITICAL_LINE_RE = re.compile(
    r"(?:"
    r"\bFAILED\b"
    r"|\bERROR\b"
    r"|\bAssertionError\b"
    r"|Traceback \(most recent call last\):"
    r"|panicked at\b"
    r"|thread '.+?' panicked"
    r"|^FAIL(?::|\s+\S)"
    r"|^E(?:\s{2,}|\t)"
    r"|^\s*(?:Expected|Received):"
    r")"
)

_SUMMARY_LINE_RE = re.compile(
    r"^(?:"
    r"=+\s*test session starts\s*=+$"
    r"|collected \d+ items?\b.*"
    r"|=+ .*?\b(?:passed|failed|errors?|skipped|xfailed|warnings?|deselected)\b.*? =+$"
    r"|test result:\s+(?:ok|FAILED)\..*"
    r"|Ran \d+ tests? in [\d.]+s$"
    r"|(?:OK|FAILED \(.+\))$"
    r"|(?:Test Suites|Test Files|Tests|Snapshots|Time|Duration)\b.*"
    r"|\s*\d+\s+(?:passed|failed|skipped|errors?)\b.*"
    r")"
)

_PASSING_LINE_RE = re.compile(
    r"^(?:"
    r".+?::\S+.*\s+(?:PASSED|SKIPPED|XFAIL|XPASS)(?:\s+\[\s*\d+%\])?\s*$"
    r"|(?:[^\s]+\.py\s+)?[.sxfF]{4,}(?:\s+\[\s*\d+%\])?\s*$"
    r"|test\s+\S+\s+\.\.\.\s+(?:ok|ignored)\s*$"
    r"|\s*(?:Compiling|Checking|Downloading|Downloaded|Updating|Fresh|Locking)\s+\S+.*"
    r"|.+?\s+\.\.\.\s+(?:ok|skipped\b.*)$"
    r"|\s*(?:PASS\s+\S+|[✓✔○↓]\s+.+)$"
    r"|\s*(?:npm (?:notice|http)\b.*|added \d+ packages\b.*)$"
    r")"
)

_SESSION_RESET_RE = re.compile(r"^=+\s*test session starts\s*=+$")

_TEST_TOOL_HINTS = ("pytest", "cargo", "unittest", "jest", "vitest", "npm", "test")


def _is_test_or_build_log(text: str, tool_name: str = "") -> bool:
    """Detect whether output looks like pytest, cargo test, unittest, or npm/jest/vitest output."""
    if (
        _PYTEST_DETECT_RE.search(text)
        or _CARGO_DETECT_RE.search(text)
        or _UNITTEST_DETECT_RE.search(text)
        or _JS_TEST_DETECT_RE.search(text)
    ):
        return True
    tool_l = (tool_name or "").lower()
    if any(hint in tool_l for hint in _TEST_TOOL_HINTS):
        passing_hits = sum(1 for ln in text.splitlines() if _PASSING_LINE_RE.match(ln))
        return passing_hits >= 3
    return False


def _distill_test_log(lines: List[str]) -> Tuple[List[str], bool]:
    """Collapse long runs of passing/progress lines while preserving failures and summaries verbatim."""
    out: List[str] = []
    passing_run: List[str] = []
    passing_count = 0
    in_failure_block = False
    changed = False

    def _flush_passing_run() -> None:
        nonlocal passing_run, passing_count, changed
        if not passing_run:
            return
        if passing_count >= 3:
            out.append(f"[... {passing_count} passing/progress lines collapsed ...]")
            changed = True
        else:
            out.extend(passing_run)
        passing_run = []
        passing_count = 0

    for line in lines:
        if _SESSION_RESET_RE.match(line):
            in_failure_block = False

        # Never treat critical/failure/summary lines as passing noise.
        if _FAILURE_BLOCK_START_RE.match(line):
            _flush_passing_run()
            in_failure_block = True
            out.append(line)
            continue

        if _CRITICAL_LINE_RE.search(line) or _SUMMARY_LINE_RE.match(line):
            _flush_passing_run()
            out.append(line)
            continue

        if in_failure_block:
            # Stay verbatim inside failure/traceback/summary sections.
            out.append(line)
            continue

        if _PASSING_LINE_RE.match(line) and "FAILED" not in line and "ERROR" not in line:
            passing_run.append(line)
            passing_count += 1
            continue

        # Absorb isolated blank lines inside an active passing run.
        if not line.strip() and passing_count > 0:
            passing_run.append(line)
            continue

        _flush_passing_run()
        out.append(line)

    _flush_passing_run()
    return out, changed


def _collapse_duplicate_lines(lines: List[str]) -> Tuple[List[str], bool]:
    """Collapse consecutive and high-frequency identical log/warning lines."""
    if not lines:
        return lines, False

    # Pass 1: collapse consecutive identical lines.
    runs: List[Tuple[str, int]] = []
    changed = False
    blank_run = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            blank_run += 1
            if blank_run == 1:
                runs.append(("", 1))
            elif blank_run >= 3:
                changed = True
            continue
        blank_run = 0
        if runs and runs[-1][0] == line:
            prev_text, prev_count = runs[-1]
            runs[-1] = (prev_text, prev_count + 1)
            changed = True
        else:
            runs.append((line, 1))

    # Pass 2: collapse high-frequency non-consecutive identical lines (e.g. alternating warnings).
    total_counts: Counter[str] = Counter()
    distinct_occurrences: Counter[str] = Counter()
    for text, count in runs:
        stripped = text.strip()
        if len(stripped) >= 8 and not stripped.startswith("[... "):
            total_counts[text] += count
            distinct_occurrences[text] += 1

    high_freq = {
        text
        for text, total in total_counts.items()
        if total >= 4 and distinct_occurrences[text] >= 2
    }

    emitted_high_freq: set[str] = set()
    out: List[str] = []

    for text, count in runs:
        if text in high_freq:
            if text in emitted_high_freq:
                changed = True
                continue
            emitted_high_freq.add(text)
            total = total_counts[text]
            out.append(f"{text}  [repeated {total}\u00d7]")
            changed = True
        elif count >= 2:
            out.append(f"{text}  [repeated {count}\u00d7]")
        else:
            out.append(text)

    return out, changed


def _head_tail_slice(text: str, max_chars: int) -> Tuple[str, bool]:
    """Slice oversized text into bounded head + tail preserving newline boundaries."""
    if len(text) <= max_chars:
        return text, False

    marker_reserve = 72
    budget = max(64, max_chars - marker_reserve)
    head_budget = max(32, int(budget * 0.45))
    tail_budget = max(32, budget - head_budget)

    head_cut = head_budget
    nl_head = text.rfind("\n", 0, head_budget)
    if nl_head >= head_budget // 2:
        head_cut = nl_head

    raw_tail_start = max(head_cut + 1, len(text) - tail_budget)
    tail_start = raw_tail_start
    nl_tail = text.find("\n", raw_tail_start)
    if nl_tail != -1 and nl_tail <= len(text) - (tail_budget // 2):
        tail_start = nl_tail + 1

    if tail_start <= head_cut:
        head_cut = min(head_budget, len(text) // 2)
        tail_start = max(head_cut + 1, len(text) - tail_budget)

    omitted = text[head_cut:tail_start]
    omitted_chars = len(omitted)
    omitted_lines = omitted.count("\n")

    head_part = text[:head_cut].rstrip("\r\n")
    tail_part = text[tail_start:].lstrip("\r\n")
    marker = f"\n... [omitted {omitted_chars} chars / {omitted_lines} lines] ...\n"
    sliced = f"{head_part}{marker}{tail_part}"
    return sliced, True


def _persist_raw_output(raw: str, state_dir: Optional[Path | str]) -> Optional[str]:
    """Persist full raw output to a content-addressed scratchpad file under state_dir."""
    if state_dir is None:
        return None
    state_str = str(state_dir).strip()
    if not state_str:
        return None
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
    out_dir = Path(state_str).expanduser() / "compressed_outputs"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{digest}.log"
        if not out_path.exists():
            out_path.write_text(raw, encoding="utf-8")
        return str(out_path)
    except OSError:
        return None


def compress_tool_output(
    text: str,
    *,
    tool_name: str = "",
    min_chars: int = 1200,
    max_output_chars: int = 2400,
    state_dir: Optional[Path | str] = None,
    store_raw: bool = True,
) -> Dict[str, Any]:
    """Deterministically compress verbose tool output while preserving failure diagnostics.

    Pipeline:
    1. Passthrough if ``len(text) < min_chars``.
    2. Test/Build Log Distiller (pytest, cargo test, unittest, npm/jest/vitest).
    3. Duplicate Line Collapser (consecutive and high-frequency identical lines).
    4. Head+Tail Bounded Slicer with optional content-addressed raw artifact pointer.
    """
    raw = text if isinstance(text, str) else str(text or "")
    original_chars = len(raw)
    original_tokens = estimate_tokens(raw)

    def _passthrough() -> Dict[str, Any]:
        return {
            "compressed": False,
            "text": raw,
            "original_chars": original_chars,
            "compressed_chars": original_chars,
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "tokens_saved": 0,
            "savings_pct": 0.0,
            "artifact_path": None,
            "strategy": "passthrough",
        }

    if original_chars < max(0, int(min_chars)):
        return _passthrough()

    bounded_max = max(128, int(max_output_chars))
    lines = raw.splitlines()
    stages: List[str] = []

    # Stage 1: Test / Build Log Distiller
    if _is_test_or_build_log(raw, tool_name=tool_name):
        distilled_lines, test_changed = _distill_test_log(lines)
        if test_changed:
            lines = distilled_lines
            stages.append("test_log_distill")

    # Stage 2: Duplicate Line Collapser
    dedup_lines, dedup_changed = _collapse_duplicate_lines(lines)
    if dedup_changed:
        lines = dedup_lines
        stages.append("dedup")

    distilled = "\n".join(lines)
    needs_slice = len(distilled) > bounded_max

    if not stages and not needs_slice:
        return _passthrough()

    # Optional content-addressed raw output persistence
    artifact_path: Optional[str] = None
    footer = ""
    if store_raw and state_dir is not None:
        artifact_path = _persist_raw_output(raw, state_dir)
        if artifact_path:
            footer = f"\n[Full raw output saved to {artifact_path}]"

    # Stage 3: Head+Tail Bounded Slicer
    effective_max = max(128, bounded_max - len(footer)) if footer else bounded_max
    if len(distilled) > effective_max:
        distilled, sliced = _head_tail_slice(distilled, effective_max)
        if sliced:
            stages.append("head_tail")

    final_text = f"{distilled}{footer}" if footer else distilled
    compressed_chars = len(final_text)

    # Never return a "compressed" output that is larger than or equal to the original.
    if compressed_chars >= original_chars:
        return _passthrough()

    compressed_tokens = estimate_tokens(final_text)
    tokens_saved = max(0, original_tokens - compressed_tokens)
    savings_pct = round((tokens_saved / float(original_tokens)) * 100.0, 2) if original_tokens > 0 else 0.0

    return {
        "compressed": True,
        "text": final_text,
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": tokens_saved,
        "savings_pct": savings_pct,
        "artifact_path": artifact_path,
        "strategy": "+".join(stages) if stages else "head_tail",
    }
