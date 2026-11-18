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
import hashlib
import json
import math
import os
import re
import shlex
import textwrap
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

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
    "totem",
    "integrity_seal.json",
    "integrity_manifest",
    # Universal SWE security-critical asset paths
    ".env",
    ".git/config",
    ".git/hooks",
    ".ssh/",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.yaml",
    "secrets.json",
)


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
    max_diff_lines: int = 480
    strict_ast_allowlist: bool = False
    allowed_ast_imports: Tuple[str, ...] = DEFAULT_ALLOWED_AST_IMPORTS


DEFAULT_POLICY = PolicyConfig()


def is_protected_path(path_str: Optional[str], policy: Optional[PolicyConfig] = None) -> bool:
    """Return True if path points to a constitutional, credential, or security-critical asset.

    Protects both custom constitutional patterns and standard SWE secrets/hooks (`.env`,
    `.git/hooks`, `.ssh/id_rsa`, `credentials.json`) while avoiding false positives on
    generic code/math paths like `src/math/axioms.py` or `src/env_loader.py`.
    """
    if not path_str:
        return False
    cfg = policy or DEFAULT_POLICY
    normalized = str(path_str).replace("\\", "/").strip().lower()
    basename = Path(normalized).name
    for pat in cfg.protected_patterns:
        if pat == ".env":
            if basename == ".env" or basename.startswith(".env."):
                return True
        elif pat in normalized:
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
        self.dangerous_aliases: Set[str] = set()
        self.module_aliases: Dict[str, str] = {}
        self.decoded_vars: Set[str] = set()

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
                    self.findings.append(f"dangerous_alias:{tgt.id}")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sub_key = self._check_subscript_lookup(node)
        if sub_key is not None:
            flag = f"obfuscated_subscript_builtin:{sub_key}"
            if flag not in self.findings:
                self.findings.append(flag)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # Case 1: Direct or aliased name call (`eval(...)`, `run(...)`)
        if isinstance(func, ast.Name):
            if func.id in self.DANGEROUS_BUILTINS:
                self.findings.append(f"dangerous_builtin:{func.id}")
                if self._uses_decoded_payload(node.args):
                    self.findings.append("obfuscated_encoded_payload")
            elif func.id in self.dangerous_aliases:
                self.findings.append(f"aliased_dangerous_call:{func.id}")
                if self._uses_decoded_payload(node.args):
                    self.findings.append("obfuscated_encoded_payload")
            elif func.id == "getattr":
                self._check_getattr(node)

        # Case 2: Obfuscated dynamic dispatch `getattr(os, 'sys' + 'tem')(...)`
        elif isinstance(func, ast.Call):
            if isinstance(func.func, ast.Name) and func.func.id == "getattr":
                self._check_getattr(func)
                self.findings.append("obfuscated_dynamic_invocation:getattr")
            if self._uses_decoded_payload(node.args):
                self.findings.append("obfuscated_encoded_payload")

        # Case 2b: Direct subscript invocation (`globals()['eval'](...)`, `__builtins__['exec'](...)`)
        elif isinstance(func, ast.Subscript):
            sub_key = self._check_subscript_lookup(func)
            if sub_key is not None:
                self.findings.append(f"obfuscated_dynamic_subscript_call:{sub_key}")
                if self._uses_decoded_payload(node.args):
                    self.findings.append("obfuscated_encoded_payload")

        # Case 3: Attribute call (`os.system`, `subprocess.run`, `shutil.rmtree`, `pickle.loads`, ...)
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            base_name = func.value.id.lower() if isinstance(func.value, ast.Name) else ""
            resolved_mod = self.module_aliases.get(base_name, base_name)

            if attr in {"eval", "exec", "system", "popen"}:
                self.findings.append(f"dangerous_method:{resolved_mod}.{attr}" if resolved_mod else f"dangerous_method:{attr}")
                if self._uses_decoded_payload(node.args):
                    self.findings.append("obfuscated_encoded_payload")
            elif resolved_mod == "importlib" and attr == "import_module":
                self.findings.append("dynamic_import:importlib.import_module")
            elif resolved_mod == "subprocess" or (isinstance(func.value, ast.Name) and func.value.id == "subprocess"):
                if attr in {"run", "Popen", "call", "check_output", "check_call", "getoutput", "getstatusoutput"}:
                    has_shell_true = any(
                        kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                        for kw in node.keywords
                    )
                    self.findings.append("subprocess_shell_true" if has_shell_true else f"subprocess_call:{attr}")
                    if self._uses_decoded_payload(node.args):
                        self.findings.append("obfuscated_encoded_payload")
            elif resolved_mod == "shutil" or (isinstance(func.value, ast.Name) and func.value.id == "shutil"):
                if attr == "rmtree":
                    self.findings.append("destructive_fs_call:shutil.rmtree")
            elif resolved_mod == "os" or (isinstance(func.value, ast.Name) and func.value.id == "os"):
                if attr in {"system", "popen", "remove", "unlink", "rmdir", "execl", "execv", "spawnl"}:
                    self.findings.append(f"dangerous_os_call:os.{attr}")
            elif resolved_mod in {"pickle", "cpickle", "marshal", "shelve"}:
                if attr in {"loads", "load", "open"}:
                    self.findings.append(f"unsafe_deserialization:{resolved_mod}.{attr}")
            elif resolved_mod == "yaml" and attr in {"unsafe_load", "load"}:
                has_safe_loader = any(
                    kw.arg == "Loader" and isinstance(kw.value, (ast.Name, ast.Attribute)) and "Safe" in ast.dump(kw.value)
                    for kw in node.keywords
                )
                if not has_safe_loader:
                    self.findings.append(f"unsafe_yaml_load:yaml.{attr}")

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
            self.findings.append(f"obfuscated_getattr:{target_obj}.{attr_str or '<dynamic>'}")
        elif attr_str in {"system", "popen", "eval", "exec", "__import__", "rmtree"}:
            if len(node.args) >= 2 and isinstance(node.args[1], ast.BinOp):
                self.findings.append(f"obfuscated_getattr_concat:{attr_str}")
            else:
                self.findings.append(f"obfuscated_getattr_sink:{attr_str}")


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


_SECRET_TOKEN_REGEXES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+)?PRIVATE\s+KEY-----"),
)
_SECRET_ASSIGN_RE = re.compile(
    r"""(?i)\b(?:api_key|secret_key|private_key|access_token|auth_token|client_secret|password|passwd)\b\s*[:=]\s*["']([A-Za-z0-9_\-./+=]{12,})["']"""
)
_SECRET_PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "example", "placeholder", "changeme", "your_", "dummy", "test", "xxx", "***", "<your"
)


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


def analyze_code_diff(
    file_path: Optional[str],
    diff_text: Optional[str],
    original_source: Optional[str] = None,
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

    if ext in {".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".md", ".yaml", ".yml", ".toml", ".txt", ".c", ".cpp", ".h", ".rs", ".go"}:
        non_py_findings: List[str] = list(secret_findings)
        for ln in dedented.splitlines():
            stripped = ln.strip()
            if stripped.startswith(("//", "#", "/*", "*")):
                continue
            if ext in {".js", ".ts", ".jsx", ".tsx"}:
                if re.search(r"\beval\s*\(", stripped):
                    non_py_findings.append("js_eval_call")
                if re.search(r"\bnew\s+Function\s*\(|\bchild_process\b|\bexecSync\s*\(|\bspawnSync\s*\(", stripped):
                    non_py_findings.append("js_dangerous_sink")
            if ext in {".yaml", ".yml"}:
                if re.search(r"!!python/(?:object|apply|module|name)\b", stripped):
                    non_py_findings.append("yaml_unsafe_deserialization_tag")

        # Optional tree-sitter syntax verification when tree_sitter is installed
        ts_verified = False
        try:
            import tree_sitter  # type: ignore  # noqa: F401
            ts_verified = True
        except ImportError:
            pass

        non_py_findings = list(dict.fromkeys(non_py_findings))
        is_doc_or_cfg = ext in {".md", ".txt", ".yaml", ".yml", ".toml", ".css", ".html"} and len(non_py_findings) == 0
        return {
            "language": ext.lstrip(".") or "text",
            "syntax_valid": True,
            "security_clean": len(non_py_findings) == 0,
            "findings": non_py_findings,
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": is_doc_or_cfg,
            "context_ast_parsed": ts_verified,
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

    patched_source = _apply_unified_diff_to_source(resolved_source, diff_text) if resolved_source else None
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
            "findings": list(dict.fromkeys(["syntax_tree_malformed"] + line_findings)),
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": False,
            "context_ast_parsed": context_ast_parsed,
            "patched_file_ast_verified": patched_file_ast_verified,
        }

    hunk_findings: List[str] = []
    if tree is not None and tree is not patched_tree:
        visitor = _SecurityASTVisitor()
        if seed_visitor is not None:
            visitor.module_aliases.update(seed_visitor.module_aliases)
            visitor.dangerous_aliases.update(seed_visitor.dangerous_aliases)
            visitor.decoded_vars.update(seed_visitor.decoded_vars)
        visitor.visit(tree)
        hunk_findings.extend(visitor.findings)

    if patched_tree is not None:
        patched_visitor = _SecurityASTVisitor()
        patched_visitor.visit(patched_tree)
        baseline_counts = Counter(seed_visitor.findings) if seed_visitor is not None else Counter()
        delta_counts = Counter(patched_visitor.findings) - baseline_counts
        for finding, cnt in delta_counts.items():
            if cnt > 0:
                hunk_findings.append(finding)

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
    }


def _tokenize_shell_segment(seg: str) -> List[str]:
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.strip().split()


SAFE_SHELL_COMMANDS: Set[str] = {
    "ls", "pwd", "echo", "cat", "head", "tail", "wc", "sort", "uniq",
    "grep", "rg", "find", "which", "where", "whoami", "date", "env",
    "clang-format", "black", "ruff", "mypy", "pytest",
}
SAFE_GIT_SUBCOMMANDS: Set[str] = {"status", "log", "diff", "branch", "show", "rev-parse", "describe"}


def _has_unsafe_shell_metachar(cmd: str) -> Optional[str]:
    """Detect unquoted/expandable shell substitution (`$(...)`, backticks) or output redirection."""
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
            if ch == "`":
                return "shell_backtick_substitution"
            if ch == "$" and i + 1 < n and cmd[i + 1] == "(":
                return "shell_command_substitution"
        i += 1

    # Output redirection (`>` or `>>`) outside single or double quotes
    in_single = False
    in_double = False
    escaped = False
    for ch in cmd:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and not in_single:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if not in_single and not in_double and ch == ">":
            return "shell_output_redirection"
    return None


def _is_command_allowlisted(exe: str, args: List[str]) -> bool:
    """Strict Layer 1 allowlist check for a parsed command executable and its arguments."""
    args_lower = [a.lower() for a in args]
    if exe in SAFE_SHELL_COMMANDS:
        if exe == "find" and any(a in {"-exec", "-execdir", "-ok", "-okdir", "-delete"} for a in args_lower):
            return False
        if exe == "pytest" and "--capture=sys" in args_lower:
            return True
        return True
    if exe == "git":
        if not args_lower:
            return True
        if any(a == "-c" or a.startswith("--exec") or a == "--ext-diff" for a in args_lower):
            return False
        return args_lower[0] in SAFE_GIT_SUBCOMMANDS
    if exe in {"python", "python3"}:
        if args_lower in (["--version"], ["-v"]):
            return True
        if len(args_lower) >= 2 and args_lower[0] == "-m" and args_lower[1] in {"pytest", "unittest", "py_compile"}:
            return True
        return False
    return False


def analyze_shell_command(cmd_str: Optional[str]) -> Dict[str, Any]:
    """Layer 1 structural POSIX/Windows shell parser using shlex tokenization.

    Combines explicit destructive pattern blocking with strict allowlist recognition
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
        }

    cmd = str(cmd_str).strip()
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
        }

    findings: List[str] = []
    meta_finding = _has_unsafe_shell_metachar(cmd)
    if meta_finding:
        findings.append(meta_finding)

    # First inspect pipe sequences for download-and-execute (`curl ... | sh`)
    if "|" in cmd and "||" not in cmd:
        pipes = [_tokenize_shell_segment(p) for p in cmd.split("|")]
        pipe_cmds = [os.path.basename(p[0]).lower() for p in pipes if p]
        if any(c in ("curl", "wget") for c in pipe_cmds[:-1]) and any(
            c in ("sh", "bash", "zsh", "dash", "pwsh", "powershell", "python", "python3")
            for c in pipe_cmds[1:]
        ):
            return {
                "is_safe": False,
                "allowlisted": False,
                "unrecognized_commands": pipe_cmds,
                "findings": ["remote_code_execution_pipe"],
                "commands": pipe_cmds,
                "risk_class": "IRREVERSIBLE_DESTRUCTIVE",
            }

    subcommands = re.split(r"(?:&&|\|\||;|\|)", cmd)
    parsed_cmds: List[str] = []
    unrecognized: List[str] = []

    for sub in subcommands:
        tokens = _tokenize_shell_segment(sub)
        if not tokens:
            continue
        exe = os.path.basename(tokens[0]).lower().removesuffix(".exe").removesuffix(".com")
        parsed_cmds.append(exe)
        args = tokens[1:]
        args_lower = [a.lower() for a in args]

        if not _is_command_allowlisted(exe, args):
            unrecognized.append(exe)

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

        elif exe in {"sh", "bash", "zsh", "dash"}:
            for idx_arg, a in enumerate(args_lower):
                if (a == "-c" or (a.startswith("-") and not a.startswith("--") and "c" in a)) and idx_arg + 1 < len(args):
                    inner_res = analyze_shell_command(args[idx_arg + 1])
                    if not inner_res["is_safe"]:
                        findings.extend(inner_res["findings"])

        elif exe in {"python", "python3"}:
            for idx_arg, a in enumerate(args_lower):
                if a == "-c" and idx_arg + 1 < len(args):
                    inline_res = analyze_code_diff("<inline>.py", args[idx_arg + 1])
                    if not inline_res["syntax_valid"] or not inline_res["security_clean"]:
                        findings.extend(f"unsafe_inline_python:{f}" for f in inline_res["findings"])

    is_safe = len(findings) == 0
    allowlisted = bool(is_safe and parsed_cmds and len(unrecognized) == 0)
    risk_class = "IRREVERSIBLE_DESTRUCTIVE" if not is_safe else (
        "READ_ONLY" if allowlisted else "EXTERNAL_ACTION"
    )
    return {
        "is_safe": is_safe,
        "allowlisted": allowlisted,
        "unrecognized_commands": unrecognized,
        "findings": list(dict.fromkeys(findings)),
        "commands": parsed_cmds,
        "risk_class": risk_class,
    }


def record_decision_journal(entry: Dict[str, Any], journal_path: Optional[Path] = None) -> None:
    """Record structured decision telemetry into ring buffer and optional JSONL journal."""
    _DECISION_RING_BUFFER.append(dict(entry))
    target = journal_path or DEFAULT_JOURNAL_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_recent_decision_journal(limit: int = 50) -> List[Dict[str, Any]]:
    """Return most recent decision journal records from the in-memory ring buffer."""
    if limit <= 0:
        return []
    return list(_DECISION_RING_BUFFER)[-limit:]


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
        tool_name = str(input_data.get("tool") or input_data.get("action") or "")
        target_path = input_data.get("path") or input_data.get("file_path")
        diff_text = input_data.get("diff") or input_data.get("payload_diff")
        cmd_str = input_data.get("cmd") or input_data.get("command") or (target_path if tool_name in ("run_command", "execute_shell") else None)
        query_text = str(input_data.get("query") or input_data.get("task_prompt") or "")
        original_source = input_data.get("file_content") or ctx.get("file_content")
    else:
        tool_name = str(ctx.get("tool") or "")
        target_path = ctx.get("path")
        diff_text = ctx.get("diff")
        cmd_str = ctx.get("cmd")
        query_text = str(input_data or "")
        original_source = ctx.get("file_content")

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
    ) -> Dict[str, Any]:
        latency_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 4)
        recent = list(_DECISION_RING_BUFFER)[-12:]
        if len(recent) < 9 and (journal_path is not None or should_log):
            j_target = journal_path or DEFAULT_JOURNAL_PATH
            disk_recent = _tail_journal_recent(j_target, limit=12)
            if len(disk_recent) > len(recent):
                recent = disk_recent[-12:]
        ood_hits = sum(1 for item in recent if item.get("reason_code") == "layer2_conformal_abstention_ood")
        if reason_code == "layer2_conformal_abstention_ood":
            ood_hits += 1
        denom = max(1, len(recent) + 1)
        drift_detected = bool(denom >= 10 and (ood_hits / float(denom)) > 0.60)
        upper_error_bound_95 = round(float(conformal_bound), 4) if conformal_bound is not None else None

        out: Dict[str, Any] = {
            "decision": decision,
            "label": label,
            "prediction_set": pred_set,
            "reason_code": reason_code,
            "layer": layer,
            "model_version": DEFAULT_MODEL_VERSION,
            "rules_version": RULES_VERSION,
            "latency_ms": latency_ms,
            "input_sha256": input_sha256,
            "shadow_mode": shadow_mode,
            "upper_error_bound_95": upper_error_bound_95,
            "drift_detected": drift_detected,
        }
        if shadow_mode:
            out["shadow_decision"] = decision
            out["shadow_reason_code"] = reason_code
            out["decision"] = "escalate"
            out["reason_code"] = "shadow_mode_observation"

        if should_log:
            record_decision_journal(out, journal_path=journal_path)
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
            return _finish("deny", "blocked_policy", ["deny"], l0["reason_code"], 0)
        if l0["decision"] == "escalate":
            return _finish("escalate", "requires_confirmation", ["escalate", "deny"], l0["reason_code"], 0)

        # --- Layer 1: Structural AST & Allowlist-First Shell Parsers ---
        if cmd_str:
            sh_res = analyze_shell_command(cmd_str)
            if not sh_res["is_safe"]:
                return _finish("deny", sh_res["findings"][0], ["deny"], "layer1_shell_parser_veto", 1)
            if not sh_res.get("allowlisted", False):
                unrec = sh_res.get("unrecognized_commands", [])
                return _finish(
                    "escalate",
                    unrec[0] if unrec else "unrecognized_shell_command",
                    ["allow", "deny"],
                    "layer1_unrecognized_shell_command_escalate",
                    1,
                )

        if diff_text is not None:
            diff_res = analyze_code_diff(target_path, diff_text, original_source=original_source)
            if not diff_res["security_clean"]:
                return _finish("deny", diff_res["findings"][0], ["deny"], "layer1_ast_security_veto", 1)
            if not diff_res["syntax_valid"]:
                if diff_res.get("partial_hunk_needs_source"):
                    return _finish("escalate", "partial_hunk_without_source", ["allow", "deny"], "layer1_partial_hunk_without_source", 1)
                return _finish("deny", "syntax_error", ["deny"], "layer1_ast_syntax_error", 1)
            if diff_res["line_count"] > cfg.max_diff_lines:
                return _finish("escalate", "high_blast_radius", ["allow", "deny"], "layer1_diff_scale_exceeded", 1)
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
                return _finish("allow", tool_name, [tool_name], "layer1_structural_verified", 1)
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
            cal_bound = (
                moe.calibration_report().get("upper_error_bound_95")
                if hasattr(moe, "calibration_report")
                else round(1.0 - (0.05 ** (1.0 / 120.0)), 4)
            )
            if not moe_res.conformal_passed:
                return _finish(
                    "escalate",
                    moe_res.selected_tool,
                    [moe_res.selected_tool, moe_res.secondary_tool],
                    "layer2_conformal_abstention_ood",
                    2,
                    conformal_bound=cal_bound,
                )
            return _finish(
                "allow",
                moe_res.selected_tool,
                [moe_res.selected_tool],
                "layer2_conformal_singleton",
                2,
                conformal_bound=cal_bound,
            )
        except Exception as exc:
            # Fail safe: escalate to Layer 3 LLM rather than returning a fake constant
            return _finish("escalate", "unknown", [], f"layer2_unavailable:{type(exc).__name__}", 3)

    # --- Layer 3: Fallback Escalation ---
    return _finish("escalate", "ambiguous", [], "layer3_default_escalation", 3)


def recalibrate_from_decision_journal(
    journal_entries: Optional[Sequence[Dict[str, Any]]] = None,
    alpha: float = 0.05,
    journal_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Consume logged decision episodes to compute operational telemetry, failure bounds,
    and updated split-conformal quantiles when labeled nonconformity feedback is present.
    """
    entries: List[Dict[str, Any]] = []
    if journal_entries is not None:
        entries = [dict(e) for e in journal_entries if isinstance(e, dict)]
    else:
        target = journal_path or DEFAULT_JOURNAL_PATH
        if _DECISION_RING_BUFFER:
            entries = list(_DECISION_RING_BUFFER)
        elif target.is_file():
            try:
                for ln in target.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if ln.strip():
                        entries.append(json.loads(ln))
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

    bound_n = verified_total if verified_total > 0 else n_total
    upper_bound = AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(
        verified_errors if verified_total > 0 else 0,
        max(1, bound_n),
        alpha=float(alpha),
    )

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

    return {
        "n_episodes": n_total,
        "decision_counts": decision_counts,
        "layer_counts": layer_counts,
        "escalation_rate": escalation_rate,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "verified_episodes": verified_total,
        "verified_errors": verified_errors,
        "upper_error_bound_95": upper_bound,
        "recalibrated_q_hat": recalibrated_q_hat,
        "router_updated": router_updated,
        "alpha": float(alpha),
    }
