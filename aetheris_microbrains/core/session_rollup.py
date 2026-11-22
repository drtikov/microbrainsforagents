"""Atomic session chronicle compaction and token-budgeted bootstrap assembly (stdlib-only).

Stores per-turn notes in date-partitioned folders, idempotently compacts completed
past days into single daily archives, and assembles startup context strictly under
a configurable token ceiling.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_DAY_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DAY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")
_DEFAULT_CHARS_PER_TOKEN = 2.9


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _sanitize_slug(raw: str, fallback: str = "note") -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw or "").strip().lower()).strip("-")
    return clean[:40] or fallback


def _chronicle_base(root: Path) -> Path:
    r = Path(root)
    return r if r.name == "chronicle" else (r / "chronicle")


class ChronicleCompactor:
    """Manages atomic chronicle entries and idempotent day-level compaction."""

    @staticmethod
    def write_entry(
        root: Path | str,
        actor: str,
        kind: str,
        body: str,
        *,
        metadata: Optional[Dict[str, str]] = None,
        now: Optional[float] = None,
    ) -> Path:
        ts = time.time() if now is None else float(now)
        lt = time.localtime(ts)
        day_str = time.strftime("%Y-%m-%d", lt)
        hms = time.strftime("%H%M%S", lt)
        day_dir = _chronicle_base(Path(root)) / day_str
        day_dir.mkdir(parents=True, exist_ok=True)

        existing_prefix = f"{hms}-"
        seq = 1 + sum(1 for p in day_dir.iterdir() if p.is_file() and p.name.startswith(existing_prefix))
        safe_actor = _sanitize_slug(actor, "agent")
        safe_kind = _sanitize_slug(kind, "event")
        fname = f"{hms}-{seq:02d}-{safe_actor}-{safe_kind}.md"
        target = day_dir / fname

        meta_lines = [
            f"ts: {time.strftime('%Y-%m-%dT%H:%M:%S', lt)}",
            f"actor: {safe_actor}",
            f"kind: {safe_kind}",
        ]
        if metadata:
            for k, v in sorted(metadata.items()):
                meta_lines.append(f"{k}: {v}")

        header = "---\n" + "\n".join(meta_lines) + "\n---\n\n"
        content = header + str(body or "").rstrip() + "\n"
        _atomic_write_text(target, content)
        return target

    @staticmethod
    def compact_past_days(
        root: Path | str,
        *,
        today: Optional[str] = None,
        min_age_sec: float = 3600.0,
        dry_run: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fold day directories strictly before `today` into `YYYY-MM-DD.md` archives."""
        base = _chronicle_base(Path(root))
        today_str = today or date.today().isoformat()
        cur_ts = time.time() if now is None else float(now)

        if not base.is_dir():
            return {"rolled_days": [], "entries_compacted": 0, "skipped": []}

        rolled_days: List[str] = []
        skipped: List[str] = []
        total_entries = 0

        for child in sorted(base.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            m = _DAY_DIR_RE.match(child.name)
            if not m:
                continue
            day_name = m.group(1)
            if day_name >= today_str:
                skipped.append(day_name)
                continue

            entry_files = sorted(
                [p for p in child.iterdir() if p.is_file() and p.suffix.lower() == ".md"],
                key=lambda p: p.name,
            )
            if not entry_files:
                if not dry_run:
                    try:
                        child.rmdir()
                    except OSError:
                        pass
                continue

            newest_mtime = max(p.stat().st_mtime for p in entry_files)
            if min_age_sec > 0 and (cur_ts - newest_mtime) < float(min_age_sec):
                skipped.append(day_name)
                continue

            archive_path = base / f"{day_name}.md"
            existing_sections: Dict[str, str] = {}
            if archive_path.is_file():
                for sec in ChronicleCompactor._split_archive_sections(
                    archive_path.read_text(encoding="utf-8", errors="replace")
                ):
                    existing_sections[sec["stem"]] = sec["raw"]

            for ef in entry_files:
                stem = ef.stem
                text = ef.read_text(encoding="utf-8", errors="replace")
                if not text.endswith("\n"):
                    text += "\n"
                existing_sections[stem] = text

            ordered_stems = sorted(existing_sections.keys())
            blocks = [f"# Chronicle — {day_name}\n"]
            for stem in ordered_stems:
                blocks.append(f"\n## {stem}\n{existing_sections[stem]}")
            rendered = "".join(blocks)

            rolled_days.append(day_name)
            total_entries += len(entry_files)

            if not dry_run:
                _atomic_write_text(archive_path, rendered)
                for ef in entry_files:
                    ef.unlink(missing_ok=True)
                try:
                    child.rmdir()
                except OSError:
                    pass

        return {
            "rolled_days": rolled_days,
            "entries_compacted": total_entries,
            "skipped": skipped,
        }

    @staticmethod
    def _split_archive_sections(archive_text: str) -> List[Dict[str, str]]:
        sections: List[Dict[str, str]] = []
        current_stem: Optional[str] = None
        buf: List[str] = []
        for line in archive_text.splitlines(keepends=True):
            if line.startswith("## "):
                if current_stem is not None:
                    sections.append({"stem": current_stem, "raw": "".join(buf)})
                current_stem = line[3:].strip()
                buf = []
            elif current_stem is not None:
                buf.append(line)
        if current_stem is not None:
            sections.append({"stem": current_stem, "raw": "".join(buf)})
        return sections

    @staticmethod
    def read_entries(
        root: Path | str,
        *,
        since: Optional[str] = None,
        last: int = 0,
    ) -> List[Dict[str, Any]]:
        """Read rolled and live chronicle entries in chronological order."""
        base = _chronicle_base(Path(root))
        if not base.is_dir():
            return []

        collected: List[Dict[str, Any]] = []
        for child in sorted(base.iterdir(), key=lambda p: p.name):
            if child.is_file():
                fm = _DAY_FILE_RE.match(child.name)
                if not fm:
                    continue
                day_str = fm.group(1)
                if since and day_str < since:
                    continue
                text = child.read_text(encoding="utf-8", errors="replace")
                for sec in ChronicleCompactor._split_archive_sections(text):
                    collected.append({
                        "date": day_str,
                        "stem": sec["stem"],
                        "path": f"{child.name}#{sec['stem']}",
                        "content": sec["raw"].strip(),
                    })
            elif child.is_dir():
                dm = _DAY_DIR_RE.match(child.name)
                if not dm:
                    continue
                day_str = dm.group(1)
                if since and day_str < since:
                    continue
                for ef in sorted(child.iterdir(), key=lambda p: p.name):
                    if ef.is_file() and ef.suffix.lower() == ".md":
                        collected.append({
                            "date": day_str,
                            "stem": ef.stem,
                            "path": f"{day_str}/{ef.name}",
                            "content": ef.read_text(encoding="utf-8", errors="replace").strip(),
                        })

        if last > 0:
            return collected[-last:]
        return collected


def write_chronicle_entry(
    root: Path | str,
    actor: str,
    kind: str,
    body: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
    now: Optional[float] = None,
) -> Path:
    return ChronicleCompactor.write_entry(
        root, actor, kind, body, metadata=metadata, now=now
    )


def compact_past_days(
    root: Path | str,
    *,
    today: Optional[str] = None,
    min_age_sec: float = 3600.0,
    dry_run: bool = False,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    return ChronicleCompactor.compact_past_days(
        root, today=today, min_age_sec=min_age_sec, dry_run=dry_run, now=now
    )


class BootstrapAssembler:
    """Builds a token-bounded session rehydration context from policy, taint, and chronicle."""

    @staticmethod
    def assemble(
        *,
        policy_slice: str = "",
        security_summary: str = "",
        open_blockers: Sequence[str] = (),
        recent_entries: Sequence[Dict[str, Any]] = (),
        token_budget: int = 24_000,
        chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
    ) -> Dict[str, Any]:
        ratio = float(chars_per_token) if chars_per_token > 0 else _DEFAULT_CHARS_PER_TOKEN
        char_cap = max(200, round(int(token_budget) * ratio))

        sections: List[str] = []
        used_chars = 0
        truncated = False

        def _try_add(title: str, body: str) -> bool:
            nonlocal used_chars, truncated
            clean = body.strip()
            if not clean:
                return True
            block = f"## {title}\n{clean}\n"
            if used_chars + len(block) <= char_cap:
                sections.append(block)
                used_chars += len(block)
                return True
            remaining = char_cap - used_chars - len(f"## {title}\n...\n")
            if remaining > 40:
                sections.append(f"## {title}\n{clean[:remaining]}\n...\n")
                used_chars = char_cap
            truncated = True
            return False

        _try_add("Active Policy Slice", policy_slice)
        _try_add("Session Security State", security_summary)
        if open_blockers:
            blocker_text = "\n".join(f"- {b}" for b in open_blockers if str(b).strip())
            _try_add("Open Blockers", blocker_text)

        included_chronicle: List[str] = []
        chronicle_header = "## Recent Chronicle\n"
        if recent_entries and used_chars + len(chronicle_header) < char_cap:
            budget_left = char_cap - used_chars - len(chronicle_header)
            chosen_reversed: List[str] = []
            for entry in reversed(list(recent_entries)):
                label = entry.get("path") or entry.get("stem") or "entry"
                content = str(entry.get("content") or "").strip()
                item = f"### {label}\n{content}\n"
                if len(item) <= budget_left:
                    chosen_reversed.append(item)
                    budget_left -= len(item)
                else:
                    truncated = True
                    break
            included_chronicle = list(reversed(chosen_reversed))
            if included_chronicle:
                sections.append(chronicle_header + "\n".join(included_chronicle))
                used_chars += len(sections[-1])
        elif recent_entries:
            truncated = True

        full_text = "\n".join(sections).strip()
        if len(full_text) > char_cap:
            full_text = full_text[:char_cap]
            truncated = True

        return {
            "text": full_text,
            "estimated_tokens": max(0, round(len(full_text) / ratio)) if full_text else 0,
            "budget_tokens": int(token_budget),
            "included_entries": len(included_chronicle),
            "truncated": truncated,
        }


def assemble_bootstrap_context(
    *,
    policy_slice: str = "",
    security_summary: str = "",
    open_blockers: Sequence[str] = (),
    recent_entries: Sequence[Dict[str, Any]] = (),
    token_budget: int = 24_000,
    chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
) -> Dict[str, Any]:
    return BootstrapAssembler.assemble(
        policy_slice=policy_slice,
        security_summary=security_summary,
        open_blockers=open_blockers,
        recent_entries=recent_entries,
        token_budget=token_budget,
        chars_per_token=chars_per_token,
    )
