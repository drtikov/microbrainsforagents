"""Behavioral Trajectory Governance, TraceProbe Loop Detection, Monotonic Confinement & FIDES Provenance (stdlib-only).

Implements state-of-the-art 2026 runtime trajectory governance inside pure Python stdlib
(zero NumPy import overhead on hook startup):
  - TraceProbe structural loop detectors (`search_loop`, `reread_churn`,
    `tool_oscillation`, `shell_over_tool`).
  - Loop Engineering finite state machine (`running`, `success`, `no-op`,
    `blocked`, `stalled`, `exhausted`) with non-completion invariant for failed/exhausted budgets.
  - Live `IntentPersistenceTracker` integration with Progent-style monotonic privilege
    confinement upon high goal drift under untrusted external ingestion.
  - FIDES / PACE artifact provenance graph (`confidentiality`, `integrity`) and
    Write-then-Execute gating on untrusted artifacts + large output folding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .runtime_paths import file_lock, get_state_dir

_SHELL_TOOLS = frozenset({"bash", "shell", "execute_shell", "run_command", "terminal"})
_EDIT_TOOLS = frozenset({
    "edit", "edit_file", "write", "write_file", "multiedit", "notebookedit",
    "str_replace_editor", "apply_patch", "repo_write", "patch", "delete_file",
})
_READ_TOOLS = frozenset({
    "read", "read_file", "repo_read", "view_file", "grep", "grep_search",
    "search_code", "glob", "list_dir",
})
_STRUCTURED_READ_EQUIV = frozenset({"read", "grep", "glob", "read_file", "search_code", "list_dir"})
_SHELL_READ_BINS = frozenset({"cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "find"})
_SHELL_SEARCH_READ_BINS = _SHELL_READ_BINS | frozenset({"ls", "dir", "tree", "fd"})
_EXEC_INTERPRETERS = frozenset({
    "python", "python3", "py", "sh", "bash", "zsh", "dash", "node", "ruby",
    "perl", "pwsh", "powershell",
})
_VALIDATION_CMD_RE = re.compile(
    r"(?i)(?:^|\s|&&|\|\||;)(?:python\s+-m\s+(?:pytest|unittest|mypy|ruff|flake8)|pytest|unittest|npm\s+(?:test|run\s+(?:test|lint|check|build))|cargo\s+(?:test|check|clippy)|go\s+test|ruff|mypy|flake8|jest|vitest)\b"
)


def _atomic_write_json(target: Path, data: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _norm_path(raw_path: str) -> str:
    return str(raw_path or "").replace("\\", "/").strip().lower()


def _leading_shell_bin(cmd: str) -> str:
    s = str(cmd or "").strip()
    if not s:
        return ""
    try:
        toks = shlex.split(s, posix=(os.name != "nt"))
    except ValueError:
        toks = s.split()
    for tok in toks:
        clean = tok.strip("'\"").replace("\\", "/")
        if not clean or "=" in clean.split("/", 1)[0]:
            continue
        if clean in {"sudo", "env", "command", "time", "nohup"}:
            continue
        return clean.rsplit("/", 1)[-1].lower().removesuffix(".exe")
    return ""


def fold_untrusted_output(
    text: str,
    *,
    state_dir: Optional[Path | str] = None,
    threshold_bytes: int = 8192,
    preview_chars: int = 600,
) -> Dict[str, Any]:
    """Store oversized untrusted tool output in a scratchpad artifact and return a compact preview."""
    raw = str(text or "")
    raw_bytes = len(raw.encode("utf-8", errors="ignore"))
    if raw_bytes <= max(1, int(threshold_bytes)):
        return {
            "folded": False,
            "artifact_path": None,
            "preview": raw,
            "original_bytes": raw_bytes,
        }
    base = Path(state_dir) if state_dir is not None else get_state_dir()
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    art_path = base / "artifacts" / f"{digest}.txt"
    art_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".art.", suffix=".tmp", dir=str(art_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(raw)
            fh.flush()
        os.replace(tmp_name, art_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    cap = max(80, int(preview_chars))
    preview = (
        f"{raw[:cap]}\n... [FOLDED {raw_bytes} bytes to {art_path} — inspect targeted slices only]"
    )
    return {
        "folded": True,
        "artifact_path": str(art_path),
        "preview": preview,
        "original_bytes": raw_bytes,
    }


class TrajectoryEngine:
    """Thread/process-safe behavioral trajectory, loop detection, confinement & FIDES graph engine."""

    def __init__(
        self,
        session_id: str,
        *,
        state_dir: Optional[Path | str] = None,
        max_actions: int = 100,
        search_loop_threshold: int = 10,
        reread_threshold: int = 3,
    ) -> None:
        sid = str(session_id or "default").strip() or "default"
        digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16]
        base = Path(state_dir) if state_dir is not None else get_state_dir()
        self.session_id = sid
        self.path = base / "sessions" / f"{digest}_trajectory.json"
        self.max_actions = max(1, int(max_actions))
        self.search_loop_threshold = max(2, int(search_loop_threshold))
        self.reread_threshold = max(2, int(reread_threshold))

    def _default_data(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": "running",
            "action_count": 0,
            "consecutive_reads": 0,
            "fruitless_rounds": 0,
            "untrusted_external_ingested": False,
            "monotonic_confinement": False,
            "drift_score": 0.0,
            "root_goal": "",
            "tracker_turn_count": 0,
            "tracker_history_scores": [],
            "events": [],
            "artifacts": {},
            "updated_at": 0.0,
        }

    def _load_unlocked(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._default_data()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged = self._default_data()
                merged.update(raw)
                return merged
        except Exception:
            pass
        return self._default_data()

    def load(self) -> Dict[str, Any]:
        with file_lock(self.path):
            return self._load_unlocked()

    def tag_artifact(
        self,
        path: str,
        *,
        confidentiality: str = "internal",
        integrity: str = "trusted",
        source: str = "",
    ) -> Dict[str, Any]:
        """Assign FIDES/PACE security labels to a file artifact."""
        npath = _norm_path(path)
        if not npath:
            return {}
        tag = {
            "confidentiality": str(confidentiality or "internal"),
            "integrity": str(integrity or "trusted"),
            "source": str(source or ""),
            "ts": round(time.time(), 3),
        }
        with file_lock(self.path):
            data = self._load_unlocked()
            data.setdefault("artifacts", {})[npath] = tag
            data["updated_at"] = round(time.time(), 3)
            _atomic_write_json(self.path, data)
        return tag

    def reset_confinement(self) -> Dict[str, Any]:
        """Explicitly lift monotonic confinement after human approval."""
        with file_lock(self.path):
            data = self._load_unlocked()
            data["monotonic_confinement"] = False
            if data.get("state") == "blocked":
                data["state"] = "running"
            data["updated_at"] = round(time.time(), 3)
            _atomic_write_json(self.path, data)
            return data

    def transition_state(self, target_state: str) -> str:
        """Explicitly transition state while enforcing Loop Engineering completion invariants."""
        req = str(target_state or "").strip().lower()
        with file_lock(self.path):
            data = self._load_unlocked()
            cur = str(data.get("state") or "running")
            if req == "success" and cur in {"exhausted", "blocked"}:
                return cur
            if req in {"running", "success", "no-op", "blocked", "stalled", "exhausted"}:
                data["state"] = req
                data["updated_at"] = round(time.time(), 3)
                _atomic_write_json(self.path, data)
                return req
            return cur

    @staticmethod
    def _has_tool_oscillation(events: List[Dict[str, Any]], norm_path: str) -> bool:
        if not norm_path:
            return False
        seq = [e for e in events if e.get("path") == norm_path]
        cycles = 0
        i = 0
        while i + 2 < len(seq):
            if (
                seq[i].get("kind") == "read"
                and seq[i + 1].get("kind") == "write"
                and seq[i + 1].get("failed")
                and seq[i + 2].get("kind") == "read"
            ):
                cycles += 1
                i += 2
            else:
                i += 1
        return cycles >= 2

    @staticmethod
    def _count_recent_rereads(events: List[Dict[str, Any]], norm_path: str) -> int:
        if not norm_path:
            return 0
        recent = events[-10:]
        count = 0
        for ev in reversed(recent):
            if ev.get("path") == norm_path:
                if ev.get("kind") == "write":
                    break
                if ev.get("kind") == "read":
                    count += 1
        return count

    @staticmethod
    def _detect_untrusted_exec(cmd: str, artifacts: Dict[str, Any]) -> Optional[str]:
        if not cmd or not artifacts:
            return None
        try:
            toks = shlex.split(cmd, posix=(os.name != "nt"))
        except ValueError:
            toks = cmd.split()
        if not toks:
            return None
        first = toks[0].strip("'\"").replace("\\", "/").lower()
        base_first = first.rsplit("/", 1)[-1].removesuffix(".exe")
        candidates: List[str] = []
        if first.startswith("./") or "/" in first:
            candidates.append(first.lstrip("./"))
        if base_first in _EXEC_INTERPRETERS:
            for t in toks[1:]:
                if not t.startswith("-"):
                    candidates.append(_norm_path(t).lstrip("./"))
                    break
        for cand in candidates:
            for art_path, tag in artifacts.items():
                if not isinstance(tag, dict) or tag.get("integrity") != "untrusted":
                    continue
                norm_art = str(art_path).lstrip("./")
                if cand == norm_art or norm_art.endswith("/" + cand) or cand.endswith("/" + norm_art):
                    return str(art_path)
        return None

    def record_step(
        self,
        tool: str,
        *,
        path: str = "",
        cmd: str = "",
        failed: bool = False,
        validated: bool = False,
        untrusted_external: bool = False,
        root_goal: str = "",
        available_tools: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate one agent action against TraceProbe, Loop state machine, drift, and provenance."""
        tool_low = str(tool or "").strip().lower()
        npath = _norm_path(path)
        cmd_str = str(cmd or "").strip()
        leading_bin = _leading_shell_bin(cmd_str) if cmd_str else ""
        is_val_cmd = bool(validated or (tool_low in _SHELL_TOOLS and _VALIDATION_CMD_RE.search(cmd_str)))

        if tool_low in _EDIT_TOOLS:
            kind = "write"
        elif tool_low in _READ_TOOLS:
            kind = "read"
        elif tool_low in _SHELL_TOOLS:
            if leading_bin in _SHELL_SEARCH_READ_BINS and not is_val_cmd:
                kind = "read"
            else:
                kind = "shell"
        else:
            kind = "other"

        findings: List[str] = []
        decision = "allow"
        reason_code = "trajectory_ok"
        ctx_notes: List[str] = []

        with file_lock(self.path):
            data = self._load_unlocked()
            data["action_count"] = int(data.get("action_count") or 0) + 1
            if untrusted_external:
                data["untrusted_external_ingested"] = True

            # Append step event
            events = list(data.get("events") or [])
            events.append({
                "tool": tool_low,
                "kind": kind,
                "path": npath,
                "failed": bool(failed),
                "validated": is_val_cmd,
                "ts": round(time.time(), 3),
            })
            events = events[-60:]
            data["events"] = events

            # 1) TraceProbe detectors
            if kind == "read":
                data["consecutive_reads"] = int(data.get("consecutive_reads") or 0) + 1
            elif kind == "write" or is_val_cmd:
                data["consecutive_reads"] = 0

            if int(data["consecutive_reads"]) >= self.search_loop_threshold:
                findings.append("search_loop")
                ctx_notes.append(
                    f"Search loop detected ({data['consecutive_reads']} consecutive reads/searches without edits or tests)."
                )

            if kind == "read" and npath and self._count_recent_rereads(events, npath) >= self.reread_threshold:
                findings.append("reread_churn")
                ctx_notes.append(f"Re-read churn detected on {path}: read >= {self.reread_threshold} times without intervening edit.")

            if npath and self._has_tool_oscillation(events, npath):
                findings.append("tool_oscillation")
                ctx_notes.append(f"Tool oscillation detected on {path}: repeated failing READ->WRITE->READ cycles.")

            if tool_low in _SHELL_TOOLS and leading_bin in _SHELL_READ_BINS:
                has_struct = (
                    available_tools is None
                    or bool({str(t).strip().lower() for t in available_tools} & _STRUCTURED_READ_EQUIV)
                )
                if has_struct:
                    findings.append("shell_over_tool")
                    ctx_notes.append(
                        f"Prefer structured Read/Grep/Glob tools over shell `{leading_bin}` to preserve output budget."
                    )

            # 2) Loop Engineering state machine (invariant: blocked/exhausted never becomes success)
            cur_state = str(data.get("state") or "running")
            is_fruitless = bool(failed or any(f in findings for f in ("search_loop", "reread_churn", "tool_oscillation")))
            if is_fruitless:
                data["fruitless_rounds"] = int(data.get("fruitless_rounds") or 0) + 1
            elif (kind == "write" and not failed) or is_val_cmd:
                data["fruitless_rounds"] = 0

            if cur_state not in {"blocked", "exhausted"}:
                if int(data["action_count"]) >= self.max_actions:
                    cur_state = "exhausted"
                    decision = "escalate"
                    reason_code = "trajectory_budget_exhausted"
                    ctx_notes.append(f"Action budget exhausted ({data['action_count']}/{self.max_actions}).")
                elif is_val_cmd and not failed:
                    cur_state = "success"
                elif int(data["fruitless_rounds"]) >= 2:
                    cur_state = "stalled"
                else:
                    cur_state = "running"
            data["state"] = cur_state

            # 3) Live IntentPersistenceTracker + Progent monotonic confinement
            goal = str(root_goal or data.get("root_goal") or "").strip()
            drift_score = float(data.get("drift_score") or 0.0)
            if goal:
                data["root_goal"] = goal[:500]
                from .intent_persistence_tracker import (
                    _extract_keywords,
                    _extract_negated_terms,
                    _stem_overlap,
                    _trigram_cosine_similarity,
                )

                root_terms = _extract_keywords(goal)
                summary = f"{tool_low} {path} {cmd_str}".strip()
                cur_terms = _extract_keywords(summary)
                neg_terms = _extract_negated_terms(summary)
                neg_roots = _stem_overlap(root_terms, neg_terms)
                overlap = _stem_overlap(root_terms, cur_terms - neg_terms)
                turn_n = int(data.get("tracker_turn_count") or 0) + 1
                hist = list(data.get("tracker_history_scores") or [])

                if root_terms:
                    ratio = len(overlap) / max(1, len(root_terms))
                    if neg_roots:
                        inst = min(1.0, max(0.55, 1.0 - ratio + 0.35 * (len(neg_roots) / max(1, len(root_terms)))))
                    else:
                        inst = max(0.0, 1.0 - ratio)
                        inst = min(inst, max(0.25, 1.0 - 1.5 * _trigram_cosine_similarity(goal, summary)))
                else:
                    inst = 0.0
                if turn_n <= 2:
                    inst = min(inst, 0.40)
                hist.append(inst)
                hist = hist[-12:]
                drift_score = round(0.7 * inst + 0.3 * (sum(hist[-3:]) / len(hist[-3:])), 4)
                data["tracker_turn_count"] = turn_n
                data["tracker_history_scores"] = hist
                data["drift_score"] = drift_score
                if drift_score >= 0.65 and bool(data.get("untrusted_external_ingested")):
                    data["monotonic_confinement"] = True

            # 4) FIDES Artifact Provenance + Write-then-Execute
            artifacts = dict(data.get("artifacts") or {})
            if kind == "write" and npath and bool(data.get("untrusted_external_ingested")):
                artifacts[npath] = {
                    "confidentiality": "internal",
                    "integrity": "untrusted",
                    "source": "untrusted_context",
                    "ts": round(time.time(), 3),
                }
                data["artifacts"] = artifacts

            untrusted_exec_hit = self._detect_untrusted_exec(cmd_str, artifacts) if kind == "shell" else None
            if untrusted_exec_hit:
                findings.append("untrusted_artifact_exec")
                decision = "escalate"
                reason_code = "trajectory_untrusted_artifact_exec"
                ctx_notes.append(f"Untrusted artifact execution attempt: {untrusted_exec_hit} requires verification.")
            elif bool(data.get("monotonic_confinement")) and (
                kind == "write"
                or tool_low in _SHELL_TOOLS
                or any(k in tool_low for k in ("fetch", "http", "send", "push", "upload"))
            ):
                findings.append("monotonic_confinement")
                if decision == "allow":
                    decision = "escalate"
                    reason_code = "trajectory_monotonic_confinement"
                ctx_notes.append("Monotonic confinement active due to high goal drift under untrusted ingestion.")
            elif any(f in findings for f in ("search_loop", "reread_churn", "tool_oscillation")) and decision == "allow":
                decision = "escalate"
                reason_code = f"trajectory_{findings[0]}"

            data["updated_at"] = round(time.time(), 3)
            _atomic_write_json(self.path, data)

        return {
            "state": cur_state,
            "findings": findings,
            "decision": decision,
            "reason_code": reason_code,
            "additionalContext": " ".join(ctx_notes) if ctx_notes else None,
            "monotonic_confinement": bool(data.get("monotonic_confinement")),
            "drift_score": drift_score,
            "action_count": int(data["action_count"]),
        }
