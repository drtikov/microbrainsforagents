"""Repo-Aware Code Graph, PageRank RepoMap, SBFL Traceback Slicer & Build Script Registry (stdlib-only).

Lightweight local code intelligence without heavy Neo4j servers or non-stdlib imports:
  1. `RepoContextResolver`: AST/regex dependency graph with incremental `mtime_ns` cache
     and 15-iteration pure-Python PageRank (`10x` personalization boost on prompt tokens).
  2. `slice_traceback_context`: maps traceback `File "...", line N` frames directly to the
     enclosing AST symbol and 1-hop graph neighbors (SBFL slice).
  3. `BuildScriptRegistry`: verifies standard test/lint/build targets in `Makefile`,
     `justfile`, `package.json`, `pyproject.toml`, and `Cargo.toml` against manifest SHA-256
     integrity (automatically revoking trust if a manifest was modified).
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .runtime_paths import file_lock, get_state_dir

_DEFAULT_CHARS_PER_TOKEN = 2.9
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "target",
})
_CODE_EXTS = frozenset({".py", ".js", ".ts", ".tsx", ".go", ".rs"})
_SAFE_TARGET_KEYWORDS = (
    "test", "tests", "check", "lint", "fmt", "format", "typecheck",
    "mypy", "ruff", "pytest", "build", "clippy", "vitest", "jest",
)
_JS_SYM_RE = re.compile(r"(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_]\w*)")
_JS_IMP_RE = re.compile(r"""(?:import\s+.*?\s+from\s+|require\s*\(\s*)['"]([^'"]+)['"]""")
_GO_SYM_RE = re.compile(r"^func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)|^type\s+([A-Za-z_]\w*)\s+struct", re.MULTILINE)
_RS_SYM_RE = re.compile(r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_]\w*)", re.MULTILINE)
_RS_IMP_RE = re.compile(r"(?:mod|use\s+crate::)\s*([A-Za-z_]\w*)")
_TRACEBACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+([^\s]+))?')
_SHELL_META_RE = re.compile(r"[;&|`$><\r\n]")


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


def _parse_python_file(src: str) -> Dict[str, Any]:
    symbols: List[str] = []
    imports: List[str] = []
    defs: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"symbols": symbols, "imports": imports, "defs": defs}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
            defs.append({
                "name": node.name,
                "lineno": int(node.lineno),
                "end_lineno": int(getattr(node, "end_lineno", node.lineno) or node.lineno),
            })
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs.append({
                            "name": f"{node.name}.{sub.name}",
                            "lineno": int(sub.lineno),
                            "end_lineno": int(getattr(sub, "end_lineno", sub.lineno) or sub.lineno),
                        })
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return {"symbols": symbols[:25], "imports": list(dict.fromkeys(imports))[:30], "defs": defs[:50]}


def _parse_generic_file(src: str, ext: str) -> Dict[str, Any]:
    symbols: List[str] = []
    imports: List[str] = []
    if ext in {".js", ".ts", ".tsx"}:
        symbols = _JS_SYM_RE.findall(src)[:25]
        imports = _JS_IMP_RE.findall(src)[:30]
    elif ext == ".go":
        for m in _GO_SYM_RE.finditer(src):
            name = m.group(1) or m.group(2)
            if name:
                symbols.append(name)
    elif ext == ".rs":
        symbols = _RS_SYM_RE.findall(src)[:25]
        imports = _RS_IMP_RE.findall(src)[:30]
    return {"symbols": symbols[:25], "imports": list(dict.fromkeys(imports))[:30], "defs": []}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BuildScriptRegistry:
    """Integrity-anchored registry of safe build, test, and lint commands discovered from manifests."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.manifest_hashes: Dict[str, str] = {}
        self.verified_commands: List[str] = self._discover_locked()

    @staticmethod
    def _is_safe_target(name: str) -> bool:
        low = str(name or "").strip().lower()
        return any(kw in low for kw in _SAFE_TARGET_KEYWORDS)

    def _discover_locked(self) -> List[str]:
        cmds: List[str] = []
        mk = self.repo_root / "Makefile"
        if mk.is_file():
            self.manifest_hashes["makefile"] = _sha256_file(mk)
            for m in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)\s*:", mk.read_text(encoding="utf-8", errors="ignore")):
                tgt = m.group(1)
                if not tgt.startswith(".") and self._is_safe_target(tgt):
                    cmds.append(f"make {tgt}")

        jf = self.repo_root / "justfile"
        if jf.is_file():
            self.manifest_hashes["justfile"] = _sha256_file(jf)
            for m in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)(?:\s+[^:]*)?:", jf.read_text(encoding="utf-8", errors="ignore")):
                tgt = m.group(1)
                if self._is_safe_target(tgt):
                    cmds.append(f"just {tgt}")

        pkg = self.repo_root / "package.json"
        if pkg.is_file():
            self.manifest_hashes["package.json"] = _sha256_file(pkg)
            try:
                pdata = json.loads(pkg.read_text(encoding="utf-8"))
                scripts = pdata.get("scripts") if isinstance(pdata, dict) else None
                if isinstance(scripts, dict):
                    for sname in scripts:
                        if self._is_safe_target(str(sname)):
                            if sname == "test":
                                cmds.append("npm test")
                            cmds.append(f"npm run {sname}")
            except Exception:
                pass

        pyp = self.repo_root / "pyproject.toml"
        if pyp.is_file():
            self.manifest_hashes["pyproject.toml"] = _sha256_file(pyp)
            text = pyp.read_text(encoding="utf-8", errors="ignore").lower()
            if "[tool.pytest" in text:
                cmds.append("pytest")
            if "[tool.ruff" in text:
                cmds.append("ruff check")
            if "[tool.mypy" in text:
                cmds.append("mypy")

        ct = self.repo_root / "Cargo.toml"
        if ct.is_file():
            self.manifest_hashes["cargo.toml"] = _sha256_file(ct)
            cmds.extend(["cargo test", "cargo check", "cargo clippy", "cargo build"])

        return list(dict.fromkeys(cmds))

    def _manifests_intact(self, modified_paths: Sequence[str] = ()) -> bool:
        mod_names = {
            Path(str(p)).name.lower()
            for p in modified_paths
            if str(p).strip()
        }
        mod_rels = {
            str(p).replace("\\", "/").strip("./").lower()
            for p in modified_paths
            if str(p).strip()
        }
        for name_low, expected_sha in self.manifest_hashes.items():
            if name_low in mod_names or name_low in mod_rels:
                return False
            candidate = None
            for actual_name in ("Makefile", "justfile", "package.json", "pyproject.toml", "Cargo.toml"):
                if actual_name.lower() == name_low:
                    candidate = self.repo_root / actual_name
                    break
            if candidate is None or not candidate.is_file() or _sha256_file(candidate) != expected_sha:
                return False
        return True

    def is_verified_repo_script(self, cmd: str, *, modified_paths: Sequence[str] = ()) -> bool:
        """True iff `cmd` invokes a verified manifest script AND no manifest has been modified."""
        raw = str(cmd or "").strip()
        if not raw or _SHELL_META_RE.search(raw):
            return False
        if not self._manifests_intact(modified_paths=modified_paths):
            return False
        try:
            toks = shlex.split(raw, posix=(os.name != "nt"))
        except ValueError:
            return False
        if not toks:
            return False
        joined = " ".join(toks).lower()
        for vcmd in self.verified_commands:
            vlow = vcmd.lower()
            if joined == vlow or joined.startswith(vlow + " "):
                return True
        return False


class RepoContextResolver:
    """Incremental dependency graph & personalized PageRank RepoMap builder."""

    def __init__(
        self,
        repo_root: Path | str,
        *,
        state_dir: Optional[Path | str] = None,
        max_files: int = 500,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.max_files = max(1, int(max_files))
        base = Path(state_dir) if state_dir is not None else get_state_dir()
        repo_hash = hashlib.sha256(str(self.repo_root).encode("utf-8")).hexdigest()[:16]
        self.cache_path = base / "repo_cache" / f"{repo_hash}.json"

    def _collect_source_files(self) -> List[Path]:
        files: List[Path] = []
        if not self.repo_root.is_dir():
            return files
        for cur_root, dirs, filenames in os.walk(self.repo_root):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
            for fname in sorted(filenames):
                p = Path(cur_root) / fname
                if p.suffix.lower() in _CODE_EXTS:
                    files.append(p)
                    if len(files) >= self.max_files:
                        return files
        return files

    def build_index(self) -> Dict[str, Any]:
        """Build or incrementally refresh AST/import metadata + adjacency edges."""
        cached_files: Dict[str, Any] = {}
        with file_lock(self.cache_path):
            if self.cache_path.is_file():
                try:
                    raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict) and isinstance(raw.get("files"), dict):
                        cached_files = raw["files"]
                except Exception:
                    cached_files = {}

        new_files: Dict[str, Dict[str, Any]] = {}
        for fp in self._collect_source_files():
            rel = fp.relative_to(self.repo_root).as_posix()
            try:
                st = fp.stat()
            except OSError:
                continue
            prev = cached_files.get(rel)
            if (
                isinstance(prev, dict)
                and prev.get("mtime_ns") == int(st.st_mtime_ns)
                and prev.get("size") == int(st.st_size)
            ):
                new_files[rel] = prev
                continue
            if st.st_size > 512 * 1024:
                continue
            src = fp.read_text(encoding="utf-8", errors="ignore")
            meta = _parse_python_file(src) if fp.suffix.lower() == ".py" else _parse_generic_file(src, fp.suffix.lower())
            new_files[rel] = {
                "mtime_ns": int(st.st_mtime_ns),
                "size": int(st.st_size),
                "symbols": meta["symbols"],
                "imports": meta["imports"],
                "defs": meta["defs"],
            }

        # Map stem & dotted module path -> file relpaths for fast edge resolution
        mod_lookup: Dict[str, List[str]] = {}
        for rel in new_files:
            stem = Path(rel).stem
            mod_lookup.setdefault(stem, []).append(rel)
            no_ext = rel.rsplit(".", 1)[0].replace("/", ".")
            mod_lookup.setdefault(no_ext, []).append(rel)
            if rel.endswith("/__init__.py"):
                pkg = rel[: -len("/__init__.py")].replace("/", ".")
                mod_lookup.setdefault(pkg, []).append(rel)

        edges: Dict[str, List[str]] = {rel: [] for rel in new_files}
        reverse_edges: Dict[str, List[str]] = {rel: [] for rel in new_files}
        for rel, meta in new_files.items():
            targets: Set[str] = set()
            for imp in meta.get("imports") or ():
                cand_keys = [imp, imp.lstrip("./"), imp.split("/")[-1].rsplit(".", 1)[0], imp.split(".")[-1]]
                for ck in cand_keys:
                    for tgt in mod_lookup.get(ck, ()):
                        if tgt != rel:
                            targets.add(tgt)
            edges[rel] = sorted(targets)
            for tgt in targets:
                reverse_edges.setdefault(tgt, []).append(rel)

        index_data = {
            "version": 1,
            "repo_root": str(self.repo_root),
            "updated_at": round(time.time(), 3),
            "files": new_files,
            "edges": edges,
            "reverse_edges": {k: sorted(set(v)) for k, v in reverse_edges.items()},
        }
        with file_lock(self.cache_path):
            _atomic_write_json(self.cache_path, index_data)
        return index_data

    @staticmethod
    def _personalized_pagerank(
        nodes: List[str],
        edges: Dict[str, List[str]],
        files: Dict[str, Dict[str, Any]],
        prompt_text: str = "",
        *,
        damping: float = 0.85,
        iterations: int = 15,
    ) -> Dict[str, float]:
        if not nodes:
            return {}
        tokens = {t for t in re.findall(r"[A-Za-z0-9_]{3,}", str(prompt_text or "").lower())}
        p_raw: Dict[str, float] = {}
        for rel in nodes:
            score = 1.0
            if tokens:
                haystack = rel.lower() + " " + " ".join(str(s).lower() for s in files[rel].get("symbols", ()))
                if any(tok in haystack for tok in tokens):
                    score = 10.0
            p_raw[rel] = score
        p_sum = sum(p_raw.values()) or 1.0
        p = {k: v / p_sum for k, v in p_raw.items()}

        ranks = dict(p)
        for _ in range(iterations):
            next_ranks = {k: (1.0 - damping) * p[k] for k in nodes}
            dangling_sum = sum(ranks[k] for k in nodes if not edges.get(k))
            for k in nodes:
                next_ranks[k] += damping * dangling_sum * p[k]
            for src in nodes:
                outs = edges.get(src) or ()
                if outs:
                    share = damping * ranks[src] / float(len(outs))
                    for dst in outs:
                        if dst in next_ranks:
                            next_ranks[dst] += share
            ranks = next_ranks
        return ranks

    def build_orientation_card(
        self,
        prompt_text: str = "",
        *,
        token_budget: int = 600,
    ) -> Dict[str, Any]:
        """Return a token-bounded Repo Orientation Card ranked via PageRank."""
        idx = self.build_index()
        files = idx["files"]
        nodes = sorted(files.keys())
        ranks = self._personalized_pagerank(nodes, idx["edges"], files, prompt_text=prompt_text)
        ranked = sorted(nodes, key=lambda n: (-ranks.get(n, 0.0), n))

        scripts = BuildScriptRegistry(self.repo_root).verified_commands
        char_cap = max(120, round(int(token_budget) * _DEFAULT_CHARS_PER_TOKEN))
        lines: List[str] = ["## Repo Orientation Card"]
        if scripts:
            lines.append("Verified commands: " + ", ".join(f"`{c}`" for c in scripts[:6]))

        top_files: List[Dict[str, Any]] = []
        used = sum(len(ln) + 1 for ln in lines)
        for rel in ranked[:25]:
            syms = files[rel].get("symbols") or []
            sym_str = f" ({', '.join(syms[:5])})" if syms else ""
            row = f"- `{rel}` [PR={ranks[rel]:.3f}]{sym_str}"
            if used + len(row) + 1 > char_cap:
                break
            lines.append(row)
            used += len(row) + 1
            top_files.append({"path": rel, "rank": round(ranks[rel], 4), "symbols": syms[:5]})

        text = "\n".join(lines)
        return {
            "text": text,
            "top_files": top_files,
            "verified_scripts": scripts,
            "estimated_tokens": max(1, round(len(text) / _DEFAULT_CHARS_PER_TOKEN)),
        }


def slice_traceback_context(traceback_text: str, repo_root: Path | str) -> Dict[str, Any]:
    """Map Python traceback frames to enclosing AST symbols and 1-hop graph neighbors (SBFL)."""
    root = Path(repo_root).resolve()
    frames_raw = _TRACEBACK_FRAME_RE.findall(str(traceback_text or ""))
    if not frames_raw or not root.is_dir():
        return {"frames": [], "summary": ""}

    resolver = RepoContextResolver(root)
    idx = resolver.build_index()
    files = idx["files"]
    edges = idx["edges"]
    rev_edges = idx["reverse_edges"]

    frames: List[Dict[str, Any]] = []
    for raw_path, raw_line, func_hint in frames_raw[-6:]:
        try:
            p = Path(raw_path)
            if not p.is_absolute():
                p = root / p
            rel = p.resolve().relative_to(root).as_posix()
        except Exception:
            continue
        meta = files.get(rel)
        if not meta:
            continue
        line_no = int(raw_line)
        enclosing = str(func_hint or "")
        span: Optional[Tuple[int, int]] = None
        for d in meta.get("defs") or ():
            if int(d["lineno"]) <= line_no <= int(d["end_lineno"]):
                enclosing = str(d["name"])
                span = (int(d["lineno"]), int(d["end_lineno"]))
        snippet = ""
        try:
            all_lines = (root / rel).read_text(encoding="utf-8", errors="ignore").splitlines()
            if 1 <= line_no <= len(all_lines):
                snippet = all_lines[line_no - 1].strip()[:120]
        except OSError:
            pass
        frames.append({
            "file": rel,
            "line": line_no,
            "symbol": enclosing or "<module>",
            "span": span,
            "line_snippet": snippet,
            "callees": list(edges.get(rel) or ())[:5],
            "callers": list(rev_edges.get(rel) or ())[:5],
        })

    if not frames:
        return {"frames": [], "summary": ""}

    summary_parts = [
        f"{fr['file']}:{fr['line']} in `{fr['symbol']}`"
        + (f" (neighbors: {', '.join((fr['callers'] + fr['callees'])[:3])})" if (fr["callers"] or fr["callees"]) else "")
        for fr in frames
    ]
    return {"frames": frames, "summary": "SBFL slice -> " + " -> ".join(summary_parts)}
