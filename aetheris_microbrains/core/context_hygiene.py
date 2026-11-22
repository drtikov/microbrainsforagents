"""Deterministic context-hygiene gate for preventing redundant and oversized tool I/O.

Pure stdlib implementation designed for low-latency PreToolUse/PostToolUse hooks:
- Tracks per-session file read snapshots (mtime_ns + size) with escape-valve recovery.
- Intercepts oversized whole-file reads and serves compact structural symbol maps inline.
- Inspects shell pipelines for un-narrowed large reads, raw spilled-output dumps,
  root-origin recursive searches, and stdin-blocking interpreter invocations.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .runtime_paths import file_lock, get_state_dir

BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".svg",
    ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".sqlite", ".db", ".whl", ".pyc", ".pyd", ".so", ".dll", ".exe",
    ".bin", ".dat", ".npy", ".npz", ".pt", ".pth", ".safetensors", ".onnx",
})

_DEFAULT_CHARS_PER_TOKEN = 2.9
_SPILL_PATH_RE = re.compile(r"(?:^|[\\/])tool-results[\\/][^\\/\\s\"']+\.txt$", re.IGNORECASE)
_WINDOWS_ROOT_RE = re.compile(r"^[A-Za-z]:(?:[\\/]+|\*?)$")
_STDIN_INTERPRETERS = frozenset({"python", "python3", "py", "node"})
_STDIN_EVAL_FLAGS = frozenset({"-c", "-m", "-e", "--eval", "--print", "-p"})
_STDIN_VALUED_FLAGS = frozenset({"-w", "-x", "-q", "-i", "--require", "-r"})

_SHELL_READERS = frozenset({"cat", "type", "get-content", "gc", "sed", "head", "tail"})
_NARROWING_FILTERS = frozenset({
    "grep", "egrep", "fgrep", "rg", "findstr", "select-string", "sls",
    "head", "tail", "wc", "cut", "sort", "uniq", "awk",
})

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_BRACE_DECL_RE = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:public|private|protected|internal|static|abstract|final|sealed|override|virtual|async|unsafe|pub(?:\([^)]*\))?\s+)*"
    r"(?:class|interface|type|enum|struct|record|trait|impl|fn|func|function)\s+[A-Za-z_][\w<>,\s:()]*|"
    r"^(?:export\s+)?const\s+[A-Za-z_]\w*\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>",
)


@dataclass
class HygieneConfig:
    """Configuration for context-hygiene gating."""

    enabled: bool = True
    redundant_read_guard: bool = True
    max_consecutive_blocks: int = 3
    whole_read_char_limit: int = 18_000
    max_outline_rows: int = 60
    block_spilled_whole_read: bool = True
    block_stdin_hang: bool = True
    block_root_origin_search: bool = True


@dataclass
class HygieneVerdict:
    """Outcome of a context-hygiene check."""

    allowed: bool
    reason_code: str
    reason: str
    avoided_bytes: int = 0
    avoided_tokens: int = 0
    target_path: str = ""


def _canonical_path(path: str | Path, cwd: Optional[str] = None) -> str:
    p = Path(path)
    if not p.is_absolute() and cwd:
        p = Path(cwd) / p
    return os.path.normcase(os.path.abspath(str(p)))


def _estimate_tokens(byte_count: int) -> int:
    if byte_count <= 0:
        return 0
    return max(1, round(byte_count / _DEFAULT_CHARS_PER_TOKEN))


def resolve_snapshot_ledger_path(
    session_id: str = "",
    ledger_path: Optional[str | Path] = None,
) -> Path:
    """Resolve the per-session FileSnapshotLedger JSON file path."""
    if ledger_path:
        return Path(ledger_path)
    sess = str(session_id or "default").strip() or "default"
    digest = hashlib.sha256(sess.encode("utf-8")).hexdigest()[:16]
    return get_state_dir() / "hygiene" / f"snapshots_{digest}.json"


class FileSnapshotLedger:
    """Tracks unchanged file reads per session to block redundant whole-file reloads."""

    def __init__(self, ledger_path: Optional[str | Path] = None, session_id: str = "") -> None:
        self.path = resolve_snapshot_ledger_path(session_id=session_id, ledger_path=ledger_path)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.flush()
            if os.name != "nt":
                try:
                    os.chmod(tmp_name, 0o600)
                except OSError:
                    pass
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def check_or_record(
        self,
        file_path: str | Path,
        *,
        max_consecutive_blocks: int = 3,
        cwd: Optional[str] = None,
    ) -> Optional[HygieneVerdict]:
        """Return a block verdict if `file_path` was already read unchanged; else record & return None."""
        canon = _canonical_path(file_path, cwd=cwd)
        if not os.path.isfile(canon):
            return None
        try:
            st = os.stat(canon)
        except OSError:
            return None

        with file_lock(self.path):
            snapshots = self._load()
            rec = snapshots.get(canon)
            if (
                isinstance(rec, dict)
                and rec.get("mtime_ns") == st.st_mtime_ns
                and rec.get("size") == st.st_size
            ):
                streak = int(rec.get("block_streak") or 0) + 1
                if streak > max(1, int(max_consecutive_blocks)):
                    rec["block_streak"] = 0
                    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    snapshots[canon] = rec
                    self._save(snapshots)
                    return None
                rec["block_streak"] = streak
                snapshots[canon] = rec
                self._save(snapshots)
                try:
                    display_path = os.path.relpath(canon, cwd) if cwd else str(file_path)
                except ValueError:
                    display_path = str(file_path)
                return HygieneVerdict(
                    allowed=False,
                    reason_code="hygiene_redundant_whole_read",
                    reason=(
                        f"{display_path} is unchanged since its whole read at {rec.get('ts', 'earlier')} "
                        f"(attempt {streak}/{max_consecutive_blocks}); read a targeted line range "
                        "(offset/limit) or apply edits directly."
                    ),
                    avoided_bytes=int(st.st_size),
                    avoided_tokens=_estimate_tokens(int(st.st_size)),
                    target_path=canon,
                )

            snapshots[canon] = {
                "mtime_ns": int(st.st_mtime_ns),
                "size": int(st.st_size),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "block_streak": 0,
            }
            self._save(snapshots)
            return None

    def evict_path(self, file_path: str | Path, *, cwd: Optional[str] = None) -> bool:
        """Remove `file_path` from the session snapshot ledger after edits/writes."""
        canon = _canonical_path(file_path, cwd=cwd)
        with file_lock(self.path):
            snapshots = self._load()
            if canon not in snapshots:
                return False
            snapshots.pop(canon, None)
            self._save(snapshots)
            return True


class SymbolMapExtractor:
    """Produces compact line-indexed structural outlines for source and text files."""

    @staticmethod
    def extract_symbol_map(file_path: str | Path, max_rows: int = 60) -> List[str]:
        p = Path(file_path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return [f"[unreadable: {exc}]"]

        ext = p.suffix.lower()
        lines = text.splitlines()
        rows: List[str] = []

        if ext == ".py":
            rows = SymbolMapExtractor._extract_python(text, lines)
        elif ext in {".md", ".markdown"}:
            rows = SymbolMapExtractor._extract_markdown(lines)
        elif ext in {".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".cs"}:
            rows = SymbolMapExtractor._extract_brace_lang(lines)

        if not rows:
            rows = SymbolMapExtractor._extract_header_fallback(lines)

        limit = max(5, int(max_rows))
        if len(rows) > limit:
            omitted = len(rows) - limit
            return rows[:limit] + [f"... ({omitted} more symbol(s) omitted; inspect targeted ranges)"]
        return rows

    @staticmethod
    def _extract_python(text: str, lines: List[str]) -> List[str]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            pat = re.compile(r"^(\s*)(async\s+def|def|class)\s+([A-Za-z_]\w*[^\r\n:]*)")
            out: List[str] = []
            for idx, line in enumerate(lines, 1):
                m = pat.match(line)
                if m:
                    indent = "  " * (len(m.group(1)) // 4)
                    out.append(f"L{idx}: {indent}{m.group(2)} {m.group(3).strip()}")
            return out

        out = []

        def _visit(nodes: List[ast.stmt], depth: int) -> None:
            indent = "  " * depth
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
                    doc = ast.get_docstring(node, clean=True) or ""
                    doc_first = doc.splitlines()[0].strip()[:70] if doc else ""
                    suffix = f"  # {doc_first}" if doc_first else ""
                    out.append(f"L{node.lineno}-{end_line}: {indent}class {node.name}{suffix}")
                    _visit(node.body, depth + 1)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
                    kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    args = [a.arg for a in node.args.args]
                    sig = ", ".join(args[:5]) + (", ..." if len(args) > 5 else "")
                    doc = ast.get_docstring(node, clean=True) or ""
                    doc_first = doc.splitlines()[0].strip()[:60] if doc else ""
                    suffix = f"  # {doc_first}" if doc_first else ""
                    out.append(f"L{node.lineno}-{end_line}: {indent}{kw} {node.name}({sig}){suffix}")

        _visit(tree.body, 0)
        return out

    @staticmethod
    def _extract_markdown(lines: List[str]) -> List[str]:
        out: List[str] = []
        in_fence = False
        for idx, line in enumerate(lines, 1):
            if _MD_FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _MD_HEADING_RE.match(line)
            if m and len(m.group(1)) <= 4:
                out.append(f"L{idx}: {m.group(1)} {m.group(2)}")
        return out

    @staticmethod
    def _extract_brace_lang(lines: List[str]) -> List[str]:
        out: List[str] = []
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", "/*", "*", "#")):
                continue
            if _BRACE_DECL_RE.match(stripped):
                indent_spaces = len(line) - len(line.lstrip())
                indent = "  " * min(3, indent_spaces // 2)
                sig = stripped.rstrip("{").strip()[:100]
                out.append(f"L{idx}: {indent}{sig}")
        return out

    @staticmethod
    def _extract_header_fallback(lines: List[str]) -> List[str]:
        out: List[str] = [f"total lines: {len(lines)}"]
        for idx, line in enumerate(lines, 1):
            s = line.strip()
            if s:
                out.append(f"L{idx}: {s[:100]}")
            if len(out) >= 6:
                break
        return out


def _strip_heredocs(cmd: str) -> str:
    """Replace heredoc bodies so multi-line payloads do not confuse segment splitting."""
    if "<<" not in cmd:
        return cmd
    lines = cmd.splitlines(keepends=True)
    out: List[str] = []
    delimiter: Optional[str] = None
    for line in lines:
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            continue
        out.append(line)
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        if m:
            delimiter = m.group(1)
    return "".join(out)


def _split_top_level(text: str, seps: Tuple[str, ...]) -> List[str]:
    """Split shell command text on separators outside quotes."""
    parts: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    escaped = False
    i = 0
    n = len(text)
    sorted_seps = sorted(seps, key=len, reverse=True)
    while i < n:
        ch = text[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote != "'":
            buf.append(ch)
            escaped = True
            i += 1
            continue
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        matched_sep = None
        for sep in sorted_seps:
            if text.startswith(sep, i):
                matched_sep = sep
                break
        if matched_sep is not None:
            parts.append("".join(buf))
            buf = []
            i += len(matched_sep)
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _safe_tokenize(segment: str) -> List[str]:
    try:
        return shlex.split(segment, posix=(os.name != "nt"))
    except ValueError:
        return segment.split()


def _base_cmd_name(tok: str) -> str:
    name = tok.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _is_narrowed_reader(cmd_name: str, args: List[str]) -> bool:
    """Return True if head/tail/sed has an explicit bounding flag."""
    if cmd_name in {"head", "tail"}:
        return any(
            a == "-n" or a == "-c" or a.startswith(("-n", "-c", "--lines", "--bytes"))
            or re.fullmatch(r"-\d+", a)
            for a in args
        )
    if cmd_name == "sed":
        return any(a == "-n" or a.startswith("-n") for a in args)
    return False


def _extract_reader_paths(cmd_name: str, args: List[str]) -> List[str]:
    """Extract target file paths passed to a shell reader command."""
    paths: List[str] = []
    skip_next = False
    valued_flags = {"-n", "-c", "-e", "-f", "--lines", "--bytes", "--expression", "--file", "-tail", "-totalcount"}
    for idx, tok in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if tok in valued_flags:
            skip_next = True
            continue
        if tok.startswith("-"):
            continue
        if cmd_name == "sed" and idx == 0 and any(ch in tok for ch in ("/", "s", "d", "p")):
            # First positional arg to `sed` without -e is usually the script expression.
            continue
        if tok.startswith(">") or tok.startswith("<") or tok.startswith("|"):
            continue
        paths.append(tok.strip("\"'"))
    return paths


def _is_root_origin_token(tok: str) -> bool:
    clean = tok.strip("\"'").strip()
    if clean in {"/", "/*", "\\"}:
        return True
    return bool(_WINDOWS_ROOT_RE.match(clean))


class ShellReadInspector:
    """Analyzes shell commands for un-narrowed whole reads, hung stdin, and root searches."""

    @staticmethod
    def inspect_command(
        command: str,
        config: Optional[HygieneConfig] = None,
        cwd: Optional[str] = None,
    ) -> Optional[HygieneVerdict]:
        cfg = config or HygieneConfig()
        if not isinstance(command, str) or not command.strip():
            return None

        cleaned = _strip_heredocs(command)
        has_heredoc = "<<" in command
        pipelines = _split_top_level(cleaned, ("&&", "||", ";", "\n", "\r"))

        for pipe in pipelines:
            pipe_str = pipe.strip()
            if not pipe_str:
                continue
            stages_raw = _split_top_level(pipe_str, ("|",))
            parsed_stages: List[Tuple[str, List[str], List[str]]] = []
            for s in stages_raw:
                toks = _safe_tokenize(s)
                if not toks:
                    continue
                parsed_stages.append((_base_cmd_name(toks[0]), toks[1:], toks))

            if not parsed_stages:
                continue

            # 1. Hung stdin interpreter guard: python/node - at stage 0 without redirect/heredoc
            if cfg.block_stdin_hang:
                cmd0, args0, toks0 = parsed_stages[0]
                if (
                    cmd0 in _STDIN_INTERPRETERS
                    and not has_heredoc
                    and not any(re.match(r"^\d?<(?!<)", t) for t in toks0)
                ):
                    skip = False
                    for a in args0:
                        if skip:
                            skip = False
                            continue
                        if a == "-":
                            return HygieneVerdict(
                                allowed=False,
                                reason_code="hygiene_stdin_hang_risk",
                                reason=(
                                    f"Invocation `{cmd0} -` has no stdin pipe, heredoc, or input redirect and will "
                                    "hang indefinitely waiting for EOF. Pass a script path, `-c` expression, or `<<` heredoc."
                                ),
                            )
                        if a.lower() in _STDIN_EVAL_FLAGS or not a.startswith("-"):
                            break
                        if a.lower() in _STDIN_VALUED_FLAGS:
                            skip = True

            # 2. Root-origin search guard + 3. Whole-file shell read parity
            for idx, (name, args, _toks) in enumerate(parsed_stages):
                if cfg.block_root_origin_search and name in {"find", "grep", "rg"}:
                    if name == "find":
                        search_roots = [a for a in args if not a.startswith("-") and not a.startswith("(")]
                        if search_roots and _is_root_origin_token(search_roots[0]):
                            return HygieneVerdict(
                                allowed=False,
                                reason_code="hygiene_root_origin_search",
                                reason=(
                                    f"Root-origin search `{name} {search_roots[0]}` walks the entire filesystem. "
                                    "Scope the search to a repository or target directory."
                                ),
                                target_path=search_roots[0],
                            )
                    elif name == "grep":
                        is_rec = any(
                            a in {"-r", "-R", "--recursive", "--dereference-recursive"}
                            or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
                            for a in args
                        )
                        if is_rec:
                            non_flags = [a for a in args if not a.startswith("-")]
                            if len(non_flags) >= 2 and any(_is_root_origin_token(p) for p in non_flags[1:]):
                                return HygieneVerdict(
                                    allowed=False,
                                    reason_code="hygiene_root_origin_search",
                                    reason=(
                                        "Recursive `grep` targets a filesystem/drive root. "
                                        "Restrict the search path to a concrete subdirectory."
                                    ),
                                )
                    elif name == "rg":
                        non_flags = [a for a in args if not a.startswith("-")]
                        if len(non_flags) >= 2 and any(_is_root_origin_token(p) for p in non_flags[1:]):
                            return HygieneVerdict(
                                allowed=False,
                                reason_code="hygiene_root_origin_search",
                                reason=(
                                    "Recursive `rg` targets a filesystem/drive root. "
                                    "Restrict the search path to a concrete subdirectory."
                                ),
                            )

                if name not in _SHELL_READERS:
                    continue
                downstream_narrows = any(dn in _NARROWING_FILTERS for dn, _, _ in parsed_stages[idx + 1 :])
                if downstream_narrows or _is_narrowed_reader(name, args):
                    continue

                paths = _extract_reader_paths(name, args)
                for p in paths:
                    if cfg.block_spilled_whole_read and _SPILL_PATH_RE.search(p.replace("\\", "/")):
                        return HygieneVerdict(
                            allowed=False,
                            reason_code="hygiene_spilled_output_whole_read",
                            reason=(
                                f"Unbounded shell read of spilled tool output `{p}` rejected. "
                                "Use `grep`/`rg` or `tail -n 40` instead of dumping the whole spill file."
                            ),
                            avoided_bytes=cfg.whole_read_char_limit,
                            avoided_tokens=_estimate_tokens(cfg.whole_read_char_limit),
                            target_path=p,
                        )
                    canon = _canonical_path(p, cwd=cwd)
                    ext = os.path.splitext(canon)[1].lower()
                    if ext in BINARY_EXTENSIONS or not os.path.isfile(canon):
                        continue
                    try:
                        size = os.path.getsize(canon)
                    except OSError:
                        continue
                    if cfg.whole_read_char_limit > 0 and size > cfg.whole_read_char_limit:
                        return HygieneVerdict(
                            allowed=False,
                            reason_code="hygiene_shell_read_oversized",
                            reason=(
                                f"Unbounded `{name}` on {p} ({size / 1000.0:.1f}k bytes > "
                                f"{cfg.whole_read_char_limit / 1000.0:.1f}k cap). Pipe through `grep`/`head -n`/`tail -n` "
                                "or use targeted line-range Read."
                            ),
                            avoided_bytes=size,
                            avoided_tokens=_estimate_tokens(size),
                            target_path=canon,
                        )

        return None


_READ_TOOL_NAMES = frozenset({"read", "read_file", "view_file"})
_EDIT_TOOL_NAMES = frozenset({
    "edit", "write", "multiedit", "notebookedit",
    "write_file", "edit_file", "repo_write", "str_replace_editor", "apply_patch", "format_code",
})
_SHELL_TOOL_NAMES = frozenset({"bash", "shell", "execute_shell", "run_command", "terminal"})


def _extract_file_path(tool_input: Dict[str, Any]) -> str:
    for key in ("file_path", "path", "target_file", "notebook_path"):
        val = tool_input.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _is_ranged_read(tool_input: Dict[str, Any]) -> bool:
    for key in ("offset", "limit", "start_line", "end_line", "StartLine", "EndLine"):
        val = tool_input.get(key)
        if val is not None and val != "" and val != 0:
            return True
    return False


class ContextHygieneGate:
    """Coordinated PreToolUse/PostToolUse context-hygiene controller."""

    def __init__(self, config: Optional[HygieneConfig] = None) -> None:
        self.config = config or HygieneConfig()

    def evaluate_pre_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        *,
        session_id: str = "",
        cwd: Optional[str] = None,
        ledger_path: Optional[str | Path] = None,
    ) -> HygieneVerdict:
        if not self.config.enabled:
            return HygieneVerdict(allowed=True, reason_code="hygiene_disabled", reason="hygiene disabled")

        tname = str(tool_name or "").strip().lower()
        inp = tool_input if isinstance(tool_input, dict) else {}

        if tname in _SHELL_TOOL_NAMES:
            cmd = str(inp.get("command") or inp.get("cmd") or "")
            verdict = ShellReadInspector.inspect_command(cmd, self.config, cwd=cwd)
            if verdict is not None:
                return verdict
            return HygieneVerdict(allowed=True, reason_code="hygiene_pass", reason="shell read hygiene passed")

        if tname in _READ_TOOL_NAMES:
            fp = _extract_file_path(inp)
            if not fp:
                return HygieneVerdict(allowed=True, reason_code="hygiene_pass", reason="no path")
            if _is_ranged_read(inp):
                return HygieneVerdict(allowed=True, reason_code="hygiene_pass", reason="ranged read allowed")

            if self.config.block_spilled_whole_read and _SPILL_PATH_RE.search(fp.replace("\\", "/")):
                return HygieneVerdict(
                    allowed=False,
                    reason_code="hygiene_spilled_output_whole_read",
                    reason=(
                        f"Whole Read of spilled tool output `{fp}` denied. "
                        "Use `grep`/`rg` or `tail -n 40` to extract only relevant lines."
                    ),
                    avoided_bytes=self.config.whole_read_char_limit,
                    avoided_tokens=_estimate_tokens(self.config.whole_read_char_limit),
                    target_path=fp,
                )

            canon = _canonical_path(fp, cwd=cwd)
            ext = os.path.splitext(canon)[1].lower()
            if ext in BINARY_EXTENSIONS or not os.path.isfile(canon):
                return HygieneVerdict(allowed=True, reason_code="hygiene_pass", reason="binary or nonexistent skipped")

            try:
                size = os.path.getsize(canon)
            except OSError:
                return HygieneVerdict(allowed=True, reason_code="hygiene_pass", reason="stat failed")

            if self.config.whole_read_char_limit > 0 and size > self.config.whole_read_char_limit:
                outline = SymbolMapExtractor.extract_symbol_map(canon, max_rows=self.config.max_outline_rows)
                outline_block = "\n".join(outline)
                return HygieneVerdict(
                    allowed=False,
                    reason_code="hygiene_oversized_whole_read",
                    reason=(
                        f"{fp} is {size / 1000.0:.1f}k bytes (cap {self.config.whole_read_char_limit / 1000.0:.1f}k). "
                        f"Structural outline:\n{outline_block}\n"
                        "Re-read only the line range you need with offset/limit."
                    ),
                    avoided_bytes=size,
                    avoided_tokens=_estimate_tokens(size),
                    target_path=canon,
                )

            if self.config.redundant_read_guard:
                ledger = FileSnapshotLedger(ledger_path=ledger_path, session_id=session_id)
                snap_verdict = ledger.check_or_record(
                    canon,
                    max_consecutive_blocks=self.config.max_consecutive_blocks,
                    cwd=cwd,
                )
                if snap_verdict is not None:
                    return snap_verdict

        return HygieneVerdict(allowed=True, reason_code="hygiene_pass", reason="context hygiene passed")

    def notify_post_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        *,
        session_id: str = "",
        cwd: Optional[str] = None,
        ledger_path: Optional[str | Path] = None,
    ) -> bool:
        """Evict modified file paths from the session snapshot ledger."""
        tname = str(tool_name or "").strip().lower()
        if tname not in _EDIT_TOOL_NAMES:
            return False
        inp = tool_input if isinstance(tool_input, dict) else {}
        fp = _extract_file_path(inp)
        if not fp:
            return False
        ledger = FileSnapshotLedger(ledger_path=ledger_path, session_id=session_id)
        return ledger.evict_path(fp, cwd=cwd)
