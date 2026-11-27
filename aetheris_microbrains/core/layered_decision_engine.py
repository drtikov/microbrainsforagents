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
from functools import lru_cache
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
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union

from .runtime_paths import append_jsonl_bounded, get_default_journal_path
from .secrets import (
    _SECRET_PATTERNS,
    _SECRET_TOKEN_REGEXES,
    redact_secrets,
)

log = logging.getLogger(__name__)

RULES_VERSION = "2.1.0-layered-structural"
DEFAULT_MODEL_VERSION = "analytical-wsp-v2.1"
_INITIAL_DEFAULT_JOURNAL_PATH = get_default_journal_path()
DEFAULT_JOURNAL_PATH = _INITIAL_DEFAULT_JOURNAL_PATH
_DECISION_RING_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=500)


def _effective_default_journal_path() -> Path:
    """Resolve journal path dynamically while still honoring test monkeypatches on DEFAULT_JOURNAL_PATH."""
    if DEFAULT_JOURNAL_PATH != _INITIAL_DEFAULT_JOURNAL_PATH:
        return Path(DEFAULT_JOURNAL_PATH)
    return get_default_journal_path()

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
    # Harness-native external web vs local informational tools:
    # `webfetch`/`websearch` are external network tools so `require_approval_for_external`
    # governs them; `task`/`agent`/`todowrite` remain `READ_ONLY`.
    "webfetch": "EXTERNAL_ACTION",
    "websearch": "EXTERNAL_ACTION",
    "task": "READ_ONLY",
    "agent": "READ_ONLY",
    "todowrite": "READ_ONLY",
}

DEFAULT_PROTECTED_PATTERNS: Tuple[str, ...] = (
    "memory/bible.md",
    "bible.md",
    "constitution/",
    "immutable_axioms",
    "aboriginal",
    "sacred_site",
    "kinship_charter",
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
    ".config/gh/hosts.yml",
    ".config/gcloud/",
    ".azure/",
    ".gnupg/",
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".gem/credentials",
    ".terraform.d/credentials.tfrc.json",
    ".pgpass",
    ".my.cnf",
    ".vault-token",
    ".claude.json",
    ".envrc",
    "authorized_keys",
    "id_ecdsa",
    "id_dsa",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/environment",
    ".claude/",
    ".claude",
    ".cursor/",
    ".cursor",
    ".cursorrules",
    ".windsurfrules",
    ".codex/",
    ".codex",
    ".qoder/",
    ".qoder",
    ".qwen/",
    ".qwen",
    ".prime/",
    ".prime",
    ".prime/agent/",
    ".prime/agent/settings.json",
    ".opencode/",
    ".opencode",
    "opencode.json",
    "opencode.jsonc",
    ".kilo/",
    ".kilo",
    "kilo.json",
    "kilo.jsonc",
    ".pi/",
    ".pi",
    ".pi/settings.json",
    ".pi/extensions/",
    ".pi/agent/settings.json",
    ".pi/agent/extensions/",
    ".omp/",
    ".omp",
    "omp.json",
    "omp.jsonc",
    ".omp/settings.json",
    ".omp/extensions/",
    ".omp/agent/settings.json",
    ".omp/agent/extensions/",
    ".zcode/",
    ".zcode",
    ".zcode/config.json",
    ".zcode/cli/config.json",
    ".zcode-plugin/",
    ".openhands/",
    ".openhands",
    ".openhands/hooks.json",
    ".openhands/skills/",
    ".openhands/microagents/",
    ".devin/",
    ".devin",
    ".devin/hooks.v1.json",
    ".devin/config.json",
    ".clinerules/",
    ".clinerules",
    ".cline/",
    ".cline",
    ".minimax/",
    ".minimax",
    ".minimax/hooks.json",
    ".minimax-plugin/",
    ".kimi-code/",
    ".kimi-code",
    ".kimi-code/config.toml",
    ".kimi-code/mcp.json",
    ".kimi-plugin/",
    ".kimi-plugin",
    "kimi.plugin.json",
    "claude.md",
    "agents.md",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/hooks",
    ".qoder/settings.json",
    ".qoder/settings.local.json",
    ".qwen/settings.json",
    ".qwen/settings.local.json",
    ".opencode/opencode.json",
    ".opencode/opencode.jsonc",
    ".opencode/plugins/",
    ".kilo/kilo.json",
    ".kilo/kilo.jsonc",
    ".kilo/plugins/",
    ".kilo/plugin/",
    ".mcp.json",
    ".cursor/mcp.json",
    ".vscode/tasks.json",
    ".vscode/settings.json",
)

_PROTECTED_EXACT_BASENAMES: Set[str] = {
    ".env",
    ".envrc",
    ".npmrc",
    ".netrc",
    ".pypirc",
    ".pgpass",
    ".my.cnf",
    ".vault-token",
    ".claude.json",
    ".gitconfig",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    ".zprofile",
    "authorized_keys",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "credentials.json",
    "secrets.yaml",
    "secrets.json",
    "hosts.yml",
    "claude.md",
    "agents.md",
    ".cursorrules",
    ".windsurfrules",
    ".gitlab-ci.yml",
    ".pre-commit-config.yaml",
    ".gitmodules",
    "kimi.plugin.json",
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
    extra_protected_patterns: Tuple[str, ...] = ()
    require_approval_for_external: bool = False
    require_approval_for_irreversible: bool = True
    require_full_source_for_allow: bool = False
    escalation_timeout_sec: int = 300
    default_on_escalation_timeout: str = "deny"
    max_diff_lines: int = 480
    strict_ast_allowlist: bool = False
    allowed_ast_imports: Tuple[str, ...] = DEFAULT_ALLOWED_AST_IMPORTS
    allow_inline_suppression: bool = True
    custom_rules: Tuple[Dict[str, Any], ...] = ()
    # Operator web domain allowlist: `webfetch`/`websearch` targets on these
    # domains are allowed even when `require_approval_for_external` is set.
    # Empty by default (behavior unchanged); subdomains match implicitly.
    allowed_web_domains: Tuple[str, ...] = ()
    # Operator-trusted MCP servers (`mcp__<server>__tool`). Trusted/pinned
    # servers get tiered verdicts: read-ish tool names allow structurally,
    # write-ish names still escalate; unpinned servers keep the fail-closed ask.
    trusted_mcp_servers: Tuple[str, ...] = ()
    # Cross-file import taint (depth 1): opt-in (`False` by default so normal
    # local edits importing packages/utilities never false-escalate). When
    # enabled, a clean proposed Python diff importing a local project module
    # whose own source contains AST-vetoed constructs escalates (never denies).
    import_taint_enabled: bool = False
    profile: str = "strict"  # "strict" (default), "dev", or "guardrails" (low-friction block-bad-first)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyConfig:
        """Construct PolicyConfig from a dictionary with strict schema and range validation."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for PolicyConfig, got {type(data).__name__}")
        valid_fields = cls.__dataclass_fields__
        unknown = sorted(str(k) for k in data.keys() if k not in valid_fields)
        if unknown:
            raise ValueError(f"Unknown PolicyConfig field(s): {', '.join(unknown)}")
        kwargs: Dict[str, Any] = {}
        for k, v in data.items():
            if k in (
                "protected_patterns",
                "extra_protected_patterns",
                "allowed_ast_imports",
                "allowed_web_domains",
                "trusted_mcp_servers",
            ):
                if not isinstance(v, (list, tuple, set)):
                    raise ValueError(f"PolicyConfig field {k!r} must be a list/tuple, got {type(v).__name__}")
                kwargs[k] = tuple(str(x) for x in v)
            elif k == "custom_rules":
                if not isinstance(v, (list, tuple)):
                    raise ValueError(f"PolicyConfig field 'custom_rules' must be a list/tuple, got {type(v).__name__}")
                validated_rules: List[Dict[str, Any]] = []
                for idx, item in enumerate(v):
                    if not isinstance(item, dict):
                        raise ValueError(f"PolicyConfig.custom_rules[{idx}] must be a dict")
                    rule_id = str(item.get("id") or "").strip()
                    action = str(item.get("action") or "").strip().lower()
                    if not rule_id:
                        raise ValueError(f"PolicyConfig.custom_rules[{idx}] missing non-empty 'id'")
                    if action not in {"deny", "escalate", "allow"}:
                        raise ValueError(f"PolicyConfig.custom_rules[{idx}] invalid action {action!r}")
                    for rk in ("tool", "cmd_regex"):
                        rv = item.get(rk)
                        if rv is not None:
                            try:
                                re.compile(str(rv))
                            except re.error as exc:
                                raise ValueError(f"PolicyConfig.custom_rules[{idx}].{rk} invalid regex: {exc}") from exc
                    validated_rules.append(dict(item))
                kwargs[k] = tuple(validated_rules)
            elif k == "allow_inline_suppression":
                if not isinstance(v, bool):
                    raise ValueError(f"PolicyConfig.allow_inline_suppression must be a bool, got {type(v).__name__}")
                kwargs[k] = v
            else:
                kwargs[k] = v
        cfg = cls(**kwargs)
        if cfg.profile not in {"strict", "dev", "guardrails"}:
            raise ValueError(
                f"Invalid PolicyConfig.profile: {cfg.profile!r} (expected 'strict', 'dev', or 'guardrails')"
            )
        if cfg.default_on_escalation_timeout not in {"deny", "escalate"}:
            raise ValueError(
                f"Invalid PolicyConfig.default_on_escalation_timeout: {cfg.default_on_escalation_timeout!r}"
            )
        if isinstance(cfg.max_diff_lines, bool) or not isinstance(cfg.max_diff_lines, int) or cfg.max_diff_lines <= 0:
            raise ValueError(f"PolicyConfig.max_diff_lines must be a positive int, got {cfg.max_diff_lines!r}")
        if (
            isinstance(cfg.escalation_timeout_sec, bool)
            or not isinstance(cfg.escalation_timeout_sec, int)
            or cfg.escalation_timeout_sec <= 0
        ):
            raise ValueError(
                f"PolicyConfig.escalation_timeout_sec must be a positive int, got {cfg.escalation_timeout_sec!r}"
            )
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> PolicyConfig:
        """Load PolicyConfig from JSON, TOML, or YAML (PyYAML recommended)."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Policy file not found: {p}")
        raw = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()
        if suffix == ".cedar":
            from .policy_as_code import cedar_to_policy_config

            return cedar_to_policy_config(raw)
        if suffix == ".rego":
            from .policy_as_code import rego_to_policy_config

            return rego_to_policy_config(raw)
        if suffix == ".toml":
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[no-redef]
                except ImportError as exc:
                    raise ImportError(
                        "Reading .toml policy files requires Python 3.11+ (stdlib tomllib) or `pip install tomli`."
                    ) from exc
            loaded = tomllib.loads(raw)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
                loaded = yaml.safe_load(raw)
            except ImportError:
                loaded = cls._parse_simple_yaml(raw)
        else:
            loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected dict in policy file, got {type(loaded).__name__}")
        return cls.from_dict(loaded)

    @staticmethod
    def _strip_yaml_comment(line: str) -> str:
        """Strip `#` comments outside single/double quotes when preceded by whitespace or at column 0."""
        in_single = False
        in_double = False
        escaped = False
        for idx, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if in_double:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_double = False
                continue
            if in_single:
                if ch == "'":
                    in_single = False
                continue
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "#" and (idx == 0 or line[idx - 1] in (" ", "\t")):
                return line[:idx]
        return line

    @staticmethod
    def _unquote_yaml_scalar(val: str) -> str:
        """Strip one matching pair of single or double quotes without corrupting inner characters."""
        s = val.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            inner = s[1:-1]
            if s[0] == "'":
                return inner.replace("''", "'")
            return inner
        return s

    @classmethod
    def _parse_yaml_scalar(cls, v_stripped: str) -> Any:
        if v_stripped.startswith(("&", "*", "|", ">", "{", "[")):
            raise ValueError(
                "Unsupported YAML feature in fallback parser; install PyYAML or use JSON/TOML policy files."
            )
        v = cls._unquote_yaml_scalar(v_stripped)
        is_unquoted = v_stripped == v
        if is_unquoted and v.lower() == "true":
            return True
        if is_unquoted and v.lower() == "false":
            return False
        if is_unquoted and v.isdigit():
            return int(v)
        return v

    @staticmethod
    def _split_yaml_mapping(text: str) -> Optional[Tuple[str, str]]:
        """Split a `key: value` string outside quotes where `:` ends the key."""
        in_single = False
        in_double = False
        escaped = False
        for idx, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if in_double:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_double = False
                continue
            if in_single:
                if ch == "'":
                    in_single = False
                continue
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == ":" and (idx + 1 == len(text) or text[idx + 1] in (" ", "\t")):
                return text[:idx], text[idx + 1 :]
        return None

    @classmethod
    def _parse_simple_yaml(cls, raw: str) -> Dict[str, Any]:
        """Minimal quote-aware YAML parser fallback for flat scalar/list configs and list-of-dicts when PyYAML is absent."""
        res: Dict[str, Any] = {}
        curr_list_key: Optional[str] = None
        curr_item_indent: Optional[int] = None
        for line in raw.splitlines():
            line_str = cls._strip_yaml_comment(line).rstrip()
            if not line_str.strip():
                continue
            indent = len(line_str) - len(line_str.lstrip())
            stripped = line_str.strip()
            if stripped.startswith("- "):
                if not curr_list_key or indent <= 0:
                    raise ValueError(
                        "Unsupported YAML feature in fallback parser; install PyYAML or use JSON/TOML policy files."
                    )
                item_raw = stripped[2:].strip()
                if item_raw.startswith(("&", "*", "|", ">", "{", "[")):
                    raise ValueError(
                        "Unsupported YAML feature in fallback parser; install PyYAML or use JSON/TOML policy files."
                    )
                kv = cls._split_yaml_mapping(item_raw)
                if kv is not None:
                    k_raw, v_raw = kv
                    k = cls._unquote_yaml_scalar(k_raw)
                    v_stripped = v_raw.strip()
                    if not v_stripped:
                        raise ValueError(
                            "Unsupported YAML feature in fallback parser; install PyYAML or use JSON/TOML policy files."
                        )
                    res[curr_list_key].append({k: cls._parse_yaml_scalar(v_stripped)})
                    curr_item_indent = indent
                else:
                    curr_item_indent = None
                    res[curr_list_key].append(cls._unquote_yaml_scalar(item_raw))
                continue
            if (
                curr_item_indent is not None
                and indent > curr_item_indent
                and curr_list_key
                and isinstance(res.get(curr_list_key), list)
                and res[curr_list_key]
                and isinstance(res[curr_list_key][-1], dict)
                and ":" in stripped
            ):
                k_raw, v_raw = stripped.split(":", 1)
                k = cls._unquote_yaml_scalar(k_raw)
                v_stripped = v_raw.strip()
                if not v_stripped:
                    raise ValueError(
                        "Unsupported YAML feature in fallback parser; install PyYAML or use JSON/TOML policy files."
                    )
                res[curr_list_key][-1][k] = cls._parse_yaml_scalar(v_stripped)
                continue
            if ":" in stripped:
                curr_list_key = None
                curr_item_indent = None
                k_raw, v_raw = stripped.split(":", 1)
                k = cls._unquote_yaml_scalar(k_raw)
                v_stripped = v_raw.strip()
                if not v_stripped:
                    curr_list_key = k
                    res[k] = []
                    continue
                res[k] = cls._parse_yaml_scalar(v_stripped)
                continue
            raise ValueError(
                "Unsupported YAML syntax in fallback parser; install PyYAML or use JSON/TOML policy files."
            )
        return res


DEFAULT_POLICY = PolicyConfig()
_DEFAULT_POLICY_PROTECTED_PATTERNS: Tuple[str, ...] = tuple(
    str(p) for p in DEFAULT_POLICY.protected_patterns
) + tuple(str(p) for p in (getattr(DEFAULT_POLICY, "extra_protected_patterns", ()) or ()))


def _policy_protected_patterns(policy: Optional[PolicyConfig]) -> Tuple[str, ...]:
    if policy is None or policy is DEFAULT_POLICY:
        return _DEFAULT_POLICY_PROTECTED_PATTERNS
    extra = getattr(policy, "extra_protected_patterns", ()) or ()
    return tuple(str(p) for p in policy.protected_patterns) + tuple(str(p) for p in extra)


_CURSOR_WORKSPACE_SUBDIRS: Set[str] = {"projects", "worktrees"}


def _has_non_workspace_cursor_segment(segments: List[str], has_traversal: bool) -> bool:
    """True when any ``.cursor`` path segment is not a workspace data dir (``projects``/``worktrees``)."""
    for idx, seg in enumerate(segments):
        if seg != ".cursor":
            continue
        if has_traversal:
            return True
        if idx + 2 >= len(segments) or segments[idx + 1] not in _CURSOR_WORKSPACE_SUBDIRS:
            return True
    return False


@lru_cache(maxsize=4096)
def _is_protected_path_cached(raw: str, all_patterns: Tuple[str, ...]) -> bool:
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
    if any(clean_target == pat.strip("/").lower() for pat in all_patterns):
        return True

    # Glob and wildcard pattern matching (supporting wildcards anywhere in the path, e.g. .e?, .ss*, .s?h)
    if any(ch in normalized for ch in "*?["):
        normalized_stripped = normalized.lstrip("/")
        path_is_targeted_dotglob = (
            normalized_stripped.startswith(".")
            or normalized_stripped.startswith("~/.")
            or basename.startswith(".")
        )
        normalized_is_pure_wildcard = normalized and all(ch in "*?[/" for ch in normalized)

        if path_is_targeted_dotglob and not normalized_is_pure_wildcard:
            # Match against exact basenames (user's path is a targeted wildcard matching a protected name)
            if any(fnmatch.fnmatch(sens, basename) for sens in _PROTECTED_EXACT_BASENAMES):
                return True
            # Match `.env.<suffix>` glob variants that obscure the `.env` prefix (e.g. `.[e]nv.local`, `.e??.local`, `.???.local`, `.*.local`)
            if basename.startswith(".") and "." in basename[1:]:
                env_head = "." + basename[1:].split(".", 1)[0]
                if fnmatch.fnmatch(".env", env_head):
                    return True
            # Match against all protected patterns and directories
            for pat in all_patterns:
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
            if seg and (all(ch in "*?[" for ch in seg) or seg[0] in "*?["):
                continue
            for pat in all_patterns:
                cpat = pat.strip("/").lower()
                if not cpat:
                    continue
                if fnmatch.fnmatch(cpat, subpath) or fnmatch.fnmatch(cpat, seg):
                    return True
            for sens in _PROTECTED_EXACT_BASENAMES:
                if fnmatch.fnmatch(sens, seg):
                    return True

    norm_segments = [seg for seg in norm_path.strip("/").split("/") if seg]
    for pat in all_patterns:
        if pat == ".env":
            continue
        clean_pat = pat.strip("/").lower()
        if not clean_pat:
            continue
        if any(ch in clean_pat for ch in "*?["):
            if (
                fnmatch.fnmatch(clean_target, clean_pat)
                or fnmatch.fnmatch(norm_path, clean_pat)
                or ("/" not in clean_pat and fnmatch.fnmatch(basename, clean_pat))
                or ("/" in clean_pat and fnmatch.fnmatch(clean_target, "*/" + clean_pat))
            ):
                return True
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
        elif clean_pat == ".cursor":
            has_traversal = ".." in raw_segments
            if (
                _has_non_workspace_cursor_segment(norm_segments, has_traversal)
                or _has_non_workspace_cursor_segment(raw_segments, has_traversal)
            ):
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
    return _is_protected_path_cached(raw, _policy_protected_patterns(policy))


_LIVE_PACKAGE_DIR: Optional[str] = None
_LIVE_PACKAGE_DIR_RESOLVED = False


def _live_package_dir() -> Optional[str]:
    """Absolute path of the RUNNING gate's own package tree (resolved once per process)."""
    global _LIVE_PACKAGE_DIR, _LIVE_PACKAGE_DIR_RESOLVED
    if not _LIVE_PACKAGE_DIR_RESOLVED:
        _LIVE_PACKAGE_DIR_RESOLVED = True
        try:
            _LIVE_PACKAGE_DIR = str(Path(__file__).resolve().parents[1])
        except Exception:
            _LIVE_PACKAGE_DIR = None
    return _LIVE_PACKAGE_DIR


@lru_cache(maxsize=4096)
def _is_live_package_path_cached(raw_norm: str, pkg_norm: str, raw_is_abs: bool) -> bool:
    """True if normalized `raw_norm` is (or lives under) the live package dir `pkg_norm`."""
    if not pkg_norm:
        return False
    if raw_norm == pkg_norm:
        return True
    sep = "\\" if "\\" in pkg_norm else "/"
    if raw_is_abs and raw_norm.startswith(pkg_norm.rstrip(sep) + sep):
        return True
    if not raw_is_abs:
        # Relative targets: the package directory name is an import invariant
        # (`import aetheris_microbrains` requires the dir to be named exactly
        # that), so a cwd-relative path into `<pkg_name>/...` is treated as a
        # package write no matter which cwd would resolve it.
        first_seg = raw_norm.replace("\\", "/").strip("/").split("/", 1)[0]
        pkg_name = pkg_norm.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return bool(first_seg) and first_seg == pkg_name
    return False


def _is_live_package_path(path_str: Optional[str]) -> bool:
    """True if the target is (or lives under) the RUNNING gate's own source tree.

    WRITE-protection only: reads of the gate's own source stay ungated so normal
    development (Read/Grep/cat of the package) never escalates. Wire sites:
    `evaluate_layer0_policy` (write-class tool deny), `_is_privileged_or_unsafe_
    redirect_target` (shell `>` veto), and the `AETHERIS_ALLOW_POLICY_EDIT`
    break-glass in `decide()` (operator package maintenance).
    """
    if not path_str:
        return False
    raw = str(path_str).replace("\\", "/").strip()
    if not raw or "\x00" in raw:
        return False
    raw_norm = os.path.normcase(os.path.normpath(raw))
    pkg_norm = os.path.normcase(os.path.normpath(_live_package_dir() or ""))
    return _is_live_package_path_cached(raw_norm, pkg_norm, os.path.isabs(raw))


def evaluate_layer0_policy(
    tool_name: str,
    target_path: Optional[str] = None,
    policy: Optional[PolicyConfig] = None,
    input_data: Any = None,
) -> Dict[str, Any]:
    """Evaluate Layer 0 deterministic policy (risk class + protected boundaries)."""
    cfg = policy or DEFAULT_POLICY
    clean_tool = str(tool_name or "").strip().lower()
    risk_class = TOOL_RISK_CLASSES.get(clean_tool, "EXTERNAL_ACTION")

    is_write = risk_class in ("REVERSIBLE_WRITE", "IRREVERSIBLE_DESTRUCTIVE", "EXTERNAL_ACTION")
    if target_path and (is_write or clean_tool in ("", "apply_patch")) and _is_live_package_path(target_path):
        # The gate's own source tree is a constitutional asset: hard-deny agent
        # writes (reads stay free — source is public, and read-escalation would
        # poison every dev session). Operator maintenance goes through a user
        # shell or the AETHERIS_ALLOW_POLICY_EDIT=1 break-glass in decide().
        return {
            "decision": "deny",
            "risk_class": risk_class,
            "reason_code": "layer0_protected_path_veto",
            "protected_path": target_path,
        }
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

    if (
        risk_class == "EXTERNAL_ACTION"
        and cfg.require_approval_for_external
        # Operator-allowlisted web targets skip the blanket external confirmation
        # here so Layer 1's structural verification (which re-checks the same
        # allowlist) can still produce a verified allow.
        and not (clean_tool in _HARNESS_WEB_TOOLS and _web_target_allowlisted(input_data, cfg))
    ):
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


def _parse_python_with_context_and_source(code_text: str) -> Tuple[Optional[ast.AST], str]:
    """Parse Python code or hunk fragment and return ``(ast_tree, exact_parsed_source)``."""
    if not code_text.strip():
        return ast.parse(""), ""

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
            return ast.parse(cand), cand
        except SyntaxError:
            continue
    return None, ""


def _parse_python_with_context(code_text: str) -> Optional[ast.AST]:
    """Parse Python code or hunk fragment by wrapping in valid block contexts if needed."""
    tree, _src = _parse_python_with_context_and_source(code_text)
    return tree


def _is_benign_relative_path_literal(path_str: Optional[str], policy: Optional[PolicyConfig] = None) -> bool:
    """Return True if literal target is a simple relative path without traversal/wildcards/protected assets."""
    if not path_str or not isinstance(path_str, str):
        return False
    raw = path_str.strip()
    if not raw or raw == ".":
        return False
    if raw.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:", raw):
        return False
    if any(ch in raw for ch in ("*", "?", "[", "$", "\x00")):
        return False
    norm = raw.replace("\\", "/")
    segs = [seg for seg in norm.split("/") if seg]
    if not segs or ".." in segs:
        return False
    if any(seg.lower() == ".git" for seg in segs):
        return False
    return not is_protected_path(raw, policy=policy)


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
    """AST security analyzer catching direct, aliased, subscript, encoded, Path-var, and getattr sinks."""

    DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__"}
    DANGEROUS_ATTRS = {"system", "popen", "rmtree", "loads", "load", "unsafe_load", "spawn", "run_path", "run_module"}
    DANGEROUS_MODULES = {
        "os", "subprocess", "shutil", "pickle", "cpickle", "marshal", "shelve",
        "pty", "builtins", "__builtins__", "importlib",
        "ctypes", "yaml", "runpy", "code", "webbrowser", "operator",
    }
    SUBSCRIPT_SCOPE_NAMES = {"__builtins__", "builtins", "globals", "locals", "vars"}

    def __init__(self, policy: Optional[PolicyConfig] = None) -> None:
        self.policy = policy
        self.findings: List[str] = []
        self.structural_findings: List[Dict[str, Any]] = []
        self.finding_lines: Dict[str, List[int]] = {}
        self.scope_stack: List[str] = []
        self.dangerous_aliases: Set[str] = set()
        self.module_aliases: Dict[str, str] = {}
        self.path_vars: Dict[str, str] = {}
        self.decoded_vars: Set[str] = set()
        self.has_network_sink: bool = False
        self.has_env_secret_read: bool = False

    def _record_finding(self, rule: str, node: ast.AST, target_symbol: str = "") -> None:
        self.findings.append(rule)
        lineno = getattr(node, "lineno", 0)
        end_lineno = getattr(node, "end_lineno", None) or lineno
        self.finding_lines.setdefault(rule, []).append(lineno)
        scope = "::".join(self.scope_stack) if self.scope_stack else "module"
        self.structural_findings.append({
            "rule": rule,
            "scope": scope,
            "lineno": lineno,
            "end_lineno": end_lineno,
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
    def _check_subscript_lookup(
        cls,
        node: ast.Subscript,
        module_aliases: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        val = node.value
        key = _const_str(node.slice)
        aliases = module_aliases or {}
        if isinstance(val, ast.Attribute) and val.attr == "__dict__":
            base = val.value
            base_name = base.id if isinstance(base, ast.Name) else ""
            resolved = aliases.get(base_name, base_name).lower()
            if base_name in cls.SUBSCRIPT_SCOPE_NAMES or resolved in cls.DANGEROUS_MODULES:
                if key is None or key in cls.DANGEROUS_BUILTINS or key in cls.DANGEROUS_ATTRS or key in {"getattr", "import_module", "open", "system", "popen"}:
                    return key or "<dynamic>"
            return None
        if isinstance(val, ast.Attribute) and cls._attr_chain(val) == "sys.modules":
            top_k = key.split(".")[0].lower() if key else None
            if top_k is None or top_k in cls.DANGEROUS_MODULES or top_k in {"sys", "builtins"}:
                return key or "<dynamic>"
            return None
        is_scope = False
        if isinstance(val, ast.Name) and val.id in cls.SUBSCRIPT_SCOPE_NAMES:
            is_scope = True
        elif isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in cls.SUBSCRIPT_SCOPE_NAMES:
            is_scope = True
        if not is_scope:
            return None
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

    _DANGEROUS_IMPORTFROM_NAMES = {
        "system", "popen", "rmtree", "run", "call", "check_output", "check_call",
        "getoutput", "getstatusoutput", "loads", "load", "unsafe_load", "remove", "unlink", "rmdir",
        "import_module", "run_path", "run_module", "execl", "execle", "execlp", "execlpe", "execv", "execve",
        "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
        "spawnve", "spawnvp", "spawnvpe", "posix_spawn", "posix_spawnp", "fork",
        "cdll", "pydll", "oledll", "windll", "libraryloader", "run_path", "run_module", "attrgetter",
    }

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").split(".")[0].lower()
        for alias in node.names:
            name_lower = alias.name.lower()
            as_name = alias.asname or alias.name
            if name_lower in self.DANGEROUS_BUILTINS or (
                mod in self.DANGEROUS_MODULES
                and name_lower in self._DANGEROUS_IMPORTFROM_NAMES
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

    def _extract_path_ctor_literal(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Call) and node.args:
            is_path_ctor = (isinstance(node.func, ast.Name) and node.func.id in {"Path", "PurePath", "PosixPath", "WindowsPath"}) or (
                self._attr_chain(node.func) in {"pathlib.path", "pathlib.purepath", "pathlib.posixpath", "pathlib.windowspath"}
            )
            if is_path_ctor:
                return self._fold_str(node.args[0]) or _const_str(node.args[0])
        return None

    def _check_alias_assignment(self, targets: List[ast.AST], value: ast.AST) -> None:
        path_lit = self._extract_path_ctor_literal(value)
        if path_lit:
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    self.path_vars[tgt.id] = path_lit
        else:
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id in self.path_vars:
                    self.path_vars.pop(tgt.id, None)

        if self._is_decode_call(value):
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    self.decoded_vars.add(tgt.id)

        is_dangerous_rhs = False
        if isinstance(value, ast.Name) and (value.id in self.DANGEROUS_BUILTINS or value.id in self.dangerous_aliases):
            is_dangerous_rhs = True
        elif isinstance(value, ast.Attribute) and value.attr in {"system", "popen", "rmtree", "loads", "load", "unsafe_load", "run", "call", "import_module", "CDLL", "PyDLL", "OleDLL", "WinDLL", "LibraryLoader"}:
            is_dangerous_rhs = True
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == "__import__":
                is_dangerous_rhs = True
            elif value.func.id == "getattr":
                is_dangerous_rhs = self._is_dangerous_getattr_call(value)
        elif isinstance(value, ast.Subscript):
            sub_key = self._check_subscript_lookup(value, self.module_aliases)
            if sub_key is not None:
                is_dangerous_rhs = True
                self.findings.append(f"obfuscated_subscript_builtin:{sub_key}")

        if is_dangerous_rhs:
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    self.dangerous_aliases.add(tgt.id)
                    self._record_finding(f"dangerous_alias:{tgt.id}", tgt, tgt.id)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sub_key = self._check_subscript_lookup(node, self.module_aliases)
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
                if open_path and is_protected_path(open_path, policy=self.policy):
                    self._record_finding("ast_protected_file_access", node, open_path)

        # Case 2: Obfuscated dynamic dispatch `getattr(os, 'sys' + 'tem')(...)`
        elif isinstance(func, ast.Call):
            if isinstance(func.func, ast.Name) and func.func.id == "getattr":
                self._check_getattr(func)
                is_self_or_cls = (
                    bool(func.args)
                    and isinstance(func.args[0], ast.Name)
                    and func.args[0].id in {"self", "cls"}
                )
                if self._is_dangerous_getattr_call(func) or not is_self_or_cls:
                    self._record_finding("obfuscated_dynamic_invocation:getattr", node, "getattr")
            if self._uses_decoded_payload(node.args):
                self._record_finding("obfuscated_encoded_payload", node)

        # Case 2b: Direct subscript invocation (`globals()['eval'](...)`, `__builtins__['exec'](...)`)
        elif isinstance(func, ast.Subscript):
            sub_key = self._check_subscript_lookup(func, self.module_aliases)
            if sub_key is not None:
                self._record_finding(f"obfuscated_dynamic_subscript_call:{sub_key}", node, str(sub_key))
                if self._uses_decoded_payload(node.args):
                    self._record_finding("obfuscated_encoded_payload", node)

        # Case 3: Attribute call (`os.system`, `subprocess.run`, `shutil.rmtree`, `pickle.loads`, ...)
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            chain = self._attr_chain(func)
            raw_base = func.value.id if isinstance(func.value, ast.Name) else ""
            base_name = raw_base.lower()
            resolved_mod = self.module_aliases.get(raw_base, self.module_aliases.get(base_name, base_name))

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

            if attr in {"read_text", "read_bytes", "write_text", "write_bytes", "open", "unlink"}:
                if isinstance(func.value, ast.Call) and func.value.args:
                    p_str = self._fold_str(func.value.args[0]) or _const_str(func.value.args[0])
                    if p_str and is_protected_path(p_str, policy=self.policy):
                        self._record_finding("ast_protected_file_access", node, p_str)
                elif isinstance(func.value, ast.Name) and func.value.id in self.path_vars:
                    p_str = self.path_vars[func.value.id]
                    if p_str and is_protected_path(p_str, policy=self.policy):
                        self._record_finding("ast_protected_file_access", node, p_str)

            if resolved_mod in {"io", "codecs", "os"} and attr == "open" and node.args:
                p_str = self._fold_str(node.args[0]) or _const_str(node.args[0])
                if p_str and is_protected_path(p_str, policy=self.policy):
                    self._record_finding("ast_protected_file_access", node, p_str)
            elif resolved_mod == "fileinput" and attr == "input" and node.args:
                arg0 = node.args[0]
                cand_nodes = list(arg0.elts) if isinstance(arg0, (ast.List, ast.Tuple)) else [arg0]
                for cn in cand_nodes:
                    p_str = self._fold_str(cn) or _const_str(cn)
                    if p_str and is_protected_path(p_str, policy=self.policy):
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
                mod_arg = (self._fold_str(node.args[0]) or _const_str(node.args[0])) if node.args else None
                top_mod = mod_arg.split(".")[0].lower() if mod_arg else ""
                _CRITICAL_DYN_IMPORT_MODS = {
                    "os", "subprocess", "pty", "ctypes", "shutil", "pickle", "cpickle",
                    "marshal", "shelve", "socket", "builtins", "sys", "posix", "nt",
                }
                dyn_rule = (
                    "dynamic_import_critical:importlib.import_module"
                    if top_mod in _CRITICAL_DYN_IMPORT_MODS
                    else "dynamic_import:importlib.import_module"
                )
                self._record_finding(dyn_rule, node, "importlib.import_module")
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
                    if not has_shell_true and node.args:
                        cmd_preview: Optional[str] = None
                        arg0 = node.args[0]
                        folded_str = self._fold_str(arg0)
                        if folded_str is not None:
                            cmd_preview = folded_str
                        elif isinstance(arg0, (ast.List, ast.Tuple)) and arg0.elts:
                            pieces = [self._fold_str(e) for e in arg0.elts]
                            if all(p is not None for p in pieces):
                                cmd_preview = shlex.join([p for p in pieces if p is not None])
                        if cmd_preview:
                            sh_res = analyze_shell_command(cmd_preview, policy=self.policy)
                            if not sh_res.get("is_safe", True) and sh_res.get("findings"):
                                self._record_finding(
                                    f"subprocess_unsafe_command:{sh_res['findings'][0]}",
                                    node,
                                    cmd_preview[:80],
                                )
            elif resolved_mod == "shutil" or (isinstance(func.value, ast.Name) and func.value.id == "shutil"):
                if attr == "rmtree":
                    rm_target = (self._fold_str(node.args[0]) or _const_str(node.args[0])) if node.args else None
                    rm_rule = (
                        "destructive_fs_call_relative:shutil.rmtree"
                        if _is_benign_relative_path_literal(rm_target, policy=self.policy)
                        else "destructive_fs_call:shutil.rmtree"
                    )
                    self._record_finding(rm_rule, node, "shutil.rmtree")
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
            elif resolved_mod == "runpy" and attr in {"run_path", "run_module", "_run_code", "_run_module_code"}:
                self._record_finding(f"dangerous_runpy_call:runpy.{attr}", node, f"runpy.{attr}")

        self.generic_visit(node)

    def _resolve_getattr_target_and_attr(self, node: ast.Call) -> Tuple[str, Optional[str]]:
        target_obj = ""
        if node.args:
            if isinstance(node.args[0], ast.Name):
                raw_name = node.args[0].id
                target_obj = self.module_aliases.get(raw_name, self.module_aliases.get(raw_name.lower(), raw_name.lower()))
            elif isinstance(node.args[0], ast.Call) and isinstance(node.args[0].func, ast.Name):
                raw_name = node.args[0].func.id.lower()
                target_obj = self.module_aliases.get(raw_name, raw_name)
        attr_str = (self._fold_str(node.args[1]) or _const_str(node.args[1])) if len(node.args) >= 2 else None
        return target_obj, attr_str

    def _is_dangerous_getattr_call(self, node: ast.Call) -> bool:
        target_obj, attr_str = self._resolve_getattr_target_and_attr(node)
        if target_obj in self.DANGEROUS_MODULES or target_obj in self.SUBSCRIPT_SCOPE_NAMES:
            return True
        if attr_str in {"system", "popen", "eval", "exec", "__import__", "rmtree"}:
            return True
        return False

    def _check_getattr(self, node: ast.Call) -> None:
        target_obj, attr_str = self._resolve_getattr_target_and_attr(node)
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


_SECRET_ASSIGN_RE = re.compile(
    r"""(?i)\b(?:api_key|secret_key|private_key|access_token|auth_token|client_secret|password|passwd)\b\s*[:=]\s*["']([A-Za-z0-9_\-./+=]{12,})["']"""
)
_SECRET_PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "example", "placeholder", "changeme", "your_", "dummy", "test", "xxx", "***", "<your"
)
_DIFF_CONTENT_KEYS: Set[str] = {"diff", "payload_diff", "new_string", "old_string", "content", "file_content", "new_source", "patch"}
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

_DEV_ESCALATE_ONLY_AST_FINDINGS: Set[str] = {
    "subprocess_call:run",
    "subprocess_call:Popen",
    "subprocess_call:call",
    "subprocess_call:check_output",
    "subprocess_call:check_call",
    "destructive_fs_call_relative:shutil.rmtree",
    "dangerous_os_call:os.remove",
    "dangerous_os_call:os.unlink",
    "dynamic_import:importlib.import_module",
}

_INLINE_SUPPRESS_RE = re.compile(
    r"(?:#|//|--)\s*aetheris:\s*allow\s+([A-Za-z0-9_.:-]+)\s+--\s+(.{3,}?)\s*$"
)
_SUPPRESSIBLE_FINDING_PREFIXES: Tuple[str, ...] = (
    "subprocess_call:",
    "destructive_fs_call_relative:",
    "dangerous_os_call:os.remove",
    "dangerous_os_call:os.unlink",
    "dynamic_import:importlib.import_module",
    "secret_leak_pattern_found",
)


def _is_finding_suppressible(finding: str) -> bool:
    """Only medium-severity / documentation fixture findings accept inline suppression."""
    return any(finding == p or finding.startswith(p) for p in _SUPPRESSIBLE_FINDING_PREFIXES)


def _parse_inline_suppression(line: str) -> Optional[Tuple[str, str]]:
    """Parse `# aetheris: allow <rule> -- <reason>` returning `(rule, reason)` or None."""
    m = _INLINE_SUPPRESS_RE.search(str(line or ""))
    if not m:
        return None
    rule_tok = m.group(1).strip()
    reason = m.group(2).strip()
    if len(reason) < 3:
        return None
    return rule_tok, reason


def _suppression_matches(rule_token: str, finding: str) -> bool:
    return (
        finding == rule_token
        or finding.startswith(rule_token + ":")
        or (rule_token.endswith(":") and finding.startswith(rule_token))
    )


def _partition_suppressed_ast_findings(
    visitor: _SecurityASTVisitor,
    source_text: str,
    allow_suppression: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Split visitor AST findings into active findings and auditable inline suppressions."""
    lines = source_text.splitlines() if source_text else []
    active: List[str] = []
    suppressed: List[Dict[str, Any]] = []
    handled_counts: Counter[str] = Counter()

    for sf in visitor.structural_findings:
        rule = str(sf.get("rule") or "")
        if not rule:
            continue
        handled_counts[rule] += 1
        lineno = int(sf.get("lineno") or 0)
        end_lineno = int(sf.get("end_lineno") or lineno or 0)
        supp_match: Optional[Tuple[int, str]] = None
        if allow_suppression and _is_finding_suppressible(rule) and lineno >= 1 and lines:
            lo = max(1, lineno)
            hi = min(len(lines), max(lineno, end_lineno))
            for ln_no in range(lo, hi + 1):
                parsed = _parse_inline_suppression(lines[ln_no - 1])
                if parsed is not None and _suppression_matches(parsed[0], rule):
                    supp_match = (ln_no, parsed[1])
                    break
        if supp_match is not None:
            suppressed.append({"rule": rule, "line": supp_match[0], "reason": supp_match[1]})
        else:
            active.append(rule)

    total_counts = Counter(visitor.findings)
    for rule, cnt in (total_counts - handled_counts).items():
        for _ in range(max(0, cnt)):
            active.append(rule)

    return active, suppressed


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


def _detect_secret_leaks(
    lines: Sequence[str],
    suppressed_out: Optional[List[Dict[str, Any]]] = None,
    allow_suppression: bool = True,
) -> List[str]:
    """Detect hardcoded API tokens, PEM private keys, and literal credential assignments."""
    for idx, ln in enumerate(lines, start=1):
        stripped = ln.strip()
        if not stripped:
            continue
        matched_secret = any(pat.search(stripped) for pat in _SECRET_TOKEN_REGEXES)
        if not matched_secret:
            match = _SECRET_ASSIGN_RE.search(stripped)
            if match:
                val_lower = match.group(1).lower()
                if not any(marker in val_lower for marker in _SECRET_PLACEHOLDER_MARKERS):
                    matched_secret = True
        if matched_secret:
            supp = _parse_inline_suppression(ln) if allow_suppression else None
            if supp is not None and _suppression_matches(supp[0], "secret_leak_pattern_found"):
                if suppressed_out is not None:
                    suppressed_out.append({
                        "rule": "secret_leak_pattern_found",
                        "line": idx,
                        "reason": supp[1],
                    })
                continue
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
    ".sh": ("tree_sitter_bash", "language"),
    ".bash": ("tree_sitter_bash", "language"),
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


_TREE_SITTER_LANG_CACHE: Dict[Tuple[str, str], Optional[Tuple[Any, Any]]] = {}


def _get_tree_sitter_parser_factory(mod_name: str, fn_name: str) -> Optional[Tuple[Any, Any]]:
    """Cache (Parser_cls, Language) per grammar to avoid per-call importlib overhead.

    Returns ``(Parser, Language)`` so each call instantiates a lightweight thread-local
    ``Parser(lang)`` safely under concurrent HTTP/TCP daemon requests.
    """
    key = (mod_name, fn_name)
    if key in _TREE_SITTER_LANG_CACHE:
        return _TREE_SITTER_LANG_CACHE[key]
    try:
        ts_mod = importlib.import_module("tree_sitter")
        grammar_mod = importlib.import_module(mod_name)
        parser_cls = getattr(ts_mod, "Parser")
        lang_cls = getattr(ts_mod, "Language")
        lang = lang_cls(getattr(grammar_mod, fn_name)())
        entry: Optional[Tuple[Any, Any]] = (parser_cls, lang)
    except Exception:
        entry = None
    _TREE_SITTER_LANG_CACHE[key] = entry
    return entry


def _scan_bash_ast_tree_sitter(cmd: str, policy: Optional[PolicyConfig] = None) -> Dict[str, Any]:
    """Optional SOTA AST shell pass via tree-sitter-bash when installed (no-op when absent)."""
    if not cmd or not cmd.strip():
        return {"available": False, "findings": [], "protected_reads": []}
    factory = _get_tree_sitter_parser_factory("tree_sitter_bash", "language")
    if factory is None:
        return {"available": False, "findings": [], "protected_reads": []}
    try:
        parser_cls, lang = factory
        parser = parser_cls(lang)
        raw = cmd.encode("utf-8", errors="ignore")
        tree = parser.parse(raw)
        if tree is None or not hasattr(tree, "root_node"):
            return {"available": False, "findings": [], "protected_reads": []}
    except Exception:
        return {"available": False, "findings": [], "protected_reads": []}

    findings: List[str] = []
    protected_reads: List[str] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        ntype = getattr(node, "type", "")
        text = raw[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        if ntype in {"command_substitution", "process_substitution"}:
            findings.append("shell_command_substitution" if ntype == "command_substitution" else "shell_output_redirection")
        elif ntype == "file_redirect":
            redir_op = ""
            redir_tgt = ""
            for ch in getattr(node, "children", ()):
                ct = getattr(ch, "type", "")
                ctext = raw[ch.start_byte:ch.end_byte].decode("utf-8", errors="ignore").strip().strip("\"'")
                if ct in {"<", ">", ">>", "&>", ">&"} or ct.endswith(("<", ">")):
                    redir_op = ctext
                elif ct not in {"file_descriptor", "number"} and ctext:
                    redir_tgt = ctext
            if redir_tgt:
                if "<" in redir_op and "<<" not in redir_op:
                    if is_protected_path(redir_tgt, policy=policy):
                        protected_reads.append(redir_tgt)
                elif ">" in redir_op and _is_privileged_or_unsafe_redirect_target(redir_tgt):
                    findings.append("shell_output_redirection")
        elif ntype == "expansion":
            var_m = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)", text)
            if var_m and var_m.group(1) == "IFS":
                findings.append("shell_ifs_obfuscation")
        stack.extend(reversed(getattr(node, "children", ())))
    return {
        "available": True,
        "findings": list(dict.fromkeys(findings)),
        "protected_reads": list(dict.fromkeys(protected_reads)),
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
    factory = _get_tree_sitter_parser_factory(mod_name, fn_name)
    if factory is None:
        return False, None
    try:
        parser_cls, lang = factory
        parser = parser_cls(lang)
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
    policy: Optional[PolicyConfig] = None,
    allow_suppression: bool = True,
) -> Dict[str, Any]:
    """Layer 1 language-aware structural diff analysis with bounded SHA-256 AST LRU caching."""
    if not diff_text and not edit_ops:
        return _analyze_code_diff_impl(
            file_path, diff_text, original_source=original_source, edit_ops=edit_ops, policy=policy, allow_suppression=allow_suppression
        )

    use_cache = not edit_ops and (policy is None or (not policy.custom_rules and not policy.allowed_ast_imports))
    if not use_cache:
        return _analyze_code_diff_impl(
            file_path, diff_text, original_source=original_source, edit_ops=edit_ops, policy=policy, allow_suppression=allow_suppression
        )

    from .ast_cache import get_default_ast_cache

    disk_tag = ""
    if original_source is None and file_path:
        try:
            p = Path(str(file_path).strip())
            if p.is_file():
                disk_tag = str(p.stat().st_mtime_ns)
        except OSError:
            disk_tag = "err"
    pol_fp = (
        f"{policy.profile}:{policy.strict_ast_allowlist}:{policy.allow_inline_suppression}:{disk_tag}"
        if policy is not None
        else f"default:{disk_tag}"
    )
    cache = get_default_ast_cache()
    key = cache.make_cache_key(
        file_path,
        diff_text,
        original_source=original_source,
        allow_suppression=allow_suppression,
        policy_fingerprint=pol_fp,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    res = _analyze_code_diff_impl(
        file_path,
        diff_text,
        original_source=original_source,
        edit_ops=edit_ops,
        policy=policy,
        allow_suppression=allow_suppression,
    )
    cache.put(key, res)
    return res


def _analyze_code_diff_impl(
    file_path: Optional[str],
    diff_text: Optional[str],
    original_source: Optional[str] = None,
    edit_ops: Optional[Sequence[Dict[str, Any]]] = None,
    policy: Optional[PolicyConfig] = None,
    allow_suppression: bool = True,
) -> Dict[str, Any]:
    """Layer 1 language-aware structural diff analysis.

    When `original_source` (or an on-disk Python file at `file_path`) is available and
    `diff_text` can be anchored as a unified/contextual patch, reconstructs the complete
    patched file in memory and verifies full-file `ast.parse` plus delta security and
    signature hazards. Falls back to context-wrapped hunk parsing for standalone diffs.
    """
    if not diff_text and not edit_ops:
        return {
            "language": "empty",
            "syntax_valid": True,
            "security_clean": True,
            "findings": [],
            "finding_severities": {},
            "suppressed_findings": [],
            "interface_hazards": [],
            "line_count": 0,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": True,
            "context_ast_parsed": False,
            "patched_file_ast_verified": False,
        }

    diff_text = diff_text or ""
    path_str = str(file_path or "").strip()
    ext = Path(path_str).suffix.lower() if path_str else ""
    line_count = len(diff_text.splitlines())

    resolved_source = original_source
    if resolved_source is None and path_str and (ext == ".py" or not ext or (edit_ops and ext == ".json")):
        try:
            candidate_path = Path(path_str)
            if candidate_path.is_file() and candidate_path.stat().st_size <= 512 * 1024:
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
    suppressed_findings: List[Dict[str, Any]] = []
    secret_findings = _detect_secret_leaks(
        secret_scan_lines,
        suppressed_out=suppressed_findings,
        allow_suppression=allow_suppression,
    )

    # Non-Python files: skip Python ast.parse and validate format appropriately
    if ext == ".json":
        json_target_text = dedented
        edit_err: Optional[str] = None
        if resolved_source is not None and edit_ops and not has_unified:
            patched_json, edit_err = _apply_edit_ops_to_source(resolved_source, edit_ops)
            if patched_json is not None:
                json_target_text = patched_json
        from .deferred_exec_guard import analyze_deferred_execution

        deferred_json = analyze_deferred_execution(path_str, json_target_text, policy=policy)
        if edit_err is not None:
            json_ok = False
            findings = [edit_err] + secret_findings + deferred_json
        else:
            try:
                json.loads(json_target_text)
                json_ok = True
            except Exception:
                unreconstructed_hunk = has_unified and not (resolved_source is not None and edit_ops)
                json_ok = bool(
                    unreconstructed_hunk
                    and (('"' in dedented and ":" in dedented) or dedented.strip().startswith(("{", "[")))
                )
            findings = ([] if json_ok else ["malformed_json_hunk"]) + secret_findings + deferred_json
        dedup_json_findings = list(dict.fromkeys(findings))
        sec_clean_json = (len(secret_findings) == 0 and len(deferred_json) == 0)
        return {
            "language": "json",
            "syntax_valid": json_ok,
            "security_clean": sec_clean_json,
            "findings": dedup_json_findings,
            "finding_severities": {
                f: ("medium" if f in _DEV_ESCALATE_ONLY_AST_FINDINGS else "critical")
                for f in dedup_json_findings
            },
            "suppressed_findings": list(suppressed_findings),
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": bool(json_ok and sec_clean_json),
            "context_ast_parsed": bool(resolved_source is not None and edit_ops),
            "patched_file_ast_verified": bool(resolved_source is not None and edit_ops and edit_err is None),
        }

    fname_lower = Path(path_str).name.lower() if path_str else ""
    is_shell_shebang = (not ext) and dedented.lstrip().startswith(("#!/bin/sh", "#!/bin/bash", "#!/usr/bin/env bash", "#!/usr/bin/env zsh"))
    if (
        ext in {".sh", ".bash", ".zsh", ".ksh", ".ps1", ".dockerfile"}
        or is_shell_shebang
        or fname_lower in {"makefile", "dockerfile", "gnumakefile", "justfile"}
    ):
        from .deferred_exec_guard import analyze_deferred_execution
        from .multilang_ast import analyze_multilang_code

        sh_diff_findings: List[str] = list(secret_findings)
        backend_name = "shell_parser"
        if fname_lower == "dockerfile" or ext == ".dockerfile":
            df_res = analyze_multilang_code(
                ext,
                fname_lower,
                dedented,
                policy=policy,
                allow_suppression=allow_suppression,
            )
            sh_diff_findings.extend(df_res["findings"])
            backend_name = df_res["ast_parser_backend"]
            lang_name = "dockerfile"
        else:
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
                sh_res = analyze_shell_command(stripped, policy=policy)
                if not sh_res.get("is_safe", True):
                    for f_item in sh_res.get("findings", []):
                        sh_diff_findings.append(f"shell_script_unsafe_cmd:{f_item}")
            sh_diff_findings.extend(analyze_deferred_execution(path_str, dedented, policy=policy))
            lang_name = "shell" if (ext in {".sh", ".bash", ".zsh", ".ksh"} or is_shell_shebang) else (ext.lstrip(".") or fname_lower)
        sh_diff_findings = list(dict.fromkeys(sh_diff_findings))
        return {
            "language": lang_name,
            "syntax_valid": True,
            "security_clean": len(sh_diff_findings) == 0,
            "findings": sh_diff_findings,
            "finding_severities": {
                f: ("medium" if f in _DEV_ESCALATE_ONLY_AST_FINDINGS else "critical")
                for f in sh_diff_findings
            },
            "suppressed_findings": list(suppressed_findings),
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": False,
            "context_ast_parsed": False,
            "ast_parser_backend": backend_name,
            "patched_file_ast_verified": False,
        }

    if (
        ext in {
            ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
            ".css", ".html", ".md", ".yaml", ".yml", ".toml", ".txt",
            ".c", ".cpp", ".h", ".rs", ".go", ".sql", ".rb", ".java", ".php",
            ".tf", ".tfvars", ".hcl", ".pth", ".ini", ".cfg", ".xml", ".svg", ".lock",
        }
        or fname_lower == "procfile"
    ):
        from .deferred_exec_guard import analyze_deferred_execution
        from .multilang_ast import analyze_multilang_code

        ml_res = analyze_multilang_code(
            ext,
            fname_lower,
            dedented,
            policy=policy,
            allow_suppression=allow_suppression,
        )
        suppressed_findings.extend(ml_res.get("suppressed_findings", []))
        deferred_non_py = analyze_deferred_execution(path_str, dedented, policy=policy)
        non_py_findings: List[str] = list(secret_findings) + list(ml_res["findings"]) + list(deferred_non_py)

        # Fallback literal-concat check for generic/markup/other extensions not handled specifically by multilang_ast
        if ext not in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs", ".sql", ".tf", ".tfvars", ".hcl", ".yaml", ".yml"}:
            for ln in dedented.splitlines():
                stripped = ln.strip()
                if stripped.startswith(("//", "#", "/*", "*")):
                    continue
                collapsed = _LITERAL_CONCAT_RE.sub("", stripped)
                if collapsed != stripped and any(p.search(collapsed) for p in _SECRET_TOKEN_REGEXES):
                    non_py_findings.append("obfuscated_secret_concatenation")

        non_py_findings = list(dict.fromkeys(non_py_findings))
        is_doc_or_cfg = ext in {".md", ".txt", ".yaml", ".yml", ".toml", ".css", ".html"} and len(non_py_findings) == 0
        ts_parsed = bool(ml_res.get("tree_sitter_parsed", False))
        ts_syntax_ok = ml_res.get("syntax_valid", True)
        syntax_valid = bool(ts_syntax_ok) if (ts_parsed and ts_syntax_ok is not None and not has_unified) else True
        return {
            "language": ext.lstrip(".") or "text",
            "syntax_valid": syntax_valid,
            "security_clean": len(non_py_findings) == 0,
            "findings": non_py_findings,
            "finding_severities": {
                f: ("medium" if f in _DEV_ESCALATE_ONLY_AST_FINDINGS else "critical")
                for f in non_py_findings
            },
            "suppressed_findings": list(suppressed_findings),
            "interface_hazards": [],
            "line_count": line_count,
            "ast_nodes": [],
            "cyclomatic_complexity": 0,
            "is_static_data_or_doc": is_doc_or_cfg,
            "context_ast_parsed": ts_parsed,
            "ast_parser_backend": ml_res.get("ast_parser_backend", "regex_heuristic"),
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
            seed_visitor = _SecurityASTVisitor(policy=policy)
            seed_visitor.visit(orig_tree)
            if seed_visitor.has_network_sink and (
                seed_visitor.has_env_secret_read
                or any("ast_protected_file_access" in f for f in seed_visitor.findings)
            ):
                seed_visitor.findings.append("ast_secret_exfil_pattern")
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
                    "finding_severities": {edit_err: "critical"},
                    "suppressed_findings": list(suppressed_findings),
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
            syn_findings = list(dict.fromkeys(list(secret_findings) + ["syntax_tree_malformed", "patched_file_syntax_error"]))
            return {
                "language": "python",
                "syntax_valid": False,
                "security_clean": len(secret_findings) == 0,
                "findings": syn_findings,
                "finding_severities": {
                    f: ("medium" if f in _DEV_ESCALATE_ONLY_AST_FINDINGS else "critical")
                    for f in syn_findings
                },
                "suppressed_findings": list(suppressed_findings),
                "interface_hazards": [],
                "line_count": line_count,
                "ast_nodes": [],
                "cyclomatic_complexity": 0,
                "is_static_data_or_doc": False,
                "context_ast_parsed": True,
                "patched_file_ast_verified": True,
            }

    tree, tree_source_text = _parse_python_with_context_and_source(dedented)
    if tree is None and has_unified and hunks:
        # Multi-hunk unified diff without full file: parse each @@ hunk independently
        parsed_bodies: List[ast.stmt] = []
        combined_hunk_lines: List[str] = []
        all_hunks_ok = True
        for h_lines in hunks:
            h_text = textwrap.dedent("\n".join(h_lines)).strip("\n")
            if not h_text.strip():
                continue
            h_tree, h_src = _parse_python_with_context_and_source(h_text)
            if h_tree is None:
                all_hunks_ok = False
                break
            if isinstance(h_tree, ast.Module):
                if combined_hunk_lines:
                    ast.increment_lineno(h_tree, len(combined_hunk_lines))
                parsed_bodies.extend(h_tree.body)
                combined_hunk_lines.extend(h_src.splitlines())
        if all_hunks_ok and parsed_bodies:
            tree = ast.Module(body=parsed_bodies, type_ignores=[])
            tree_source_text = "\n".join(combined_hunk_lines)
    if tree is None and has_unified:
        # Fallback 1: synthetic unified diffs where context lines omit the 1-space diff prefix
        unshifted = "\n".join(
            l[1:] if l.startswith("+") else l
            for l in raw_lines
            if not (l.startswith(_GIT_META_PREFIXES) or l.startswith("-"))
        )
        if unshifted.strip():
            tree, tree_source_text = _parse_python_with_context_and_source(textwrap.dedent(unshifted).strip("\n"))
    if tree is None and has_unified:
        # Fallback 2: parse added '+' lines only when they are self-contained statements (not dangling block headers)
        added_only = "\n".join(l[1:] for l in raw_lines if l.startswith("+") and not l.startswith("+++"))
        added_dedented = textwrap.dedent(added_only).strip("\n")
        if added_dedented.strip() and not added_dedented.rstrip().endswith(":"):
            tree, tree_source_text = _parse_python_with_context_and_source(added_dedented)
    if tree is None and patched_tree is not None:
        tree = patched_tree
        tree_source_text = patched_source or ""
    elif tree is None and resolved_source:
        combined = resolved_source.rstrip("\n") + "\n" + textwrap.indent(dedented, "    ")
        tree, tree_source_text = _parse_python_with_context_and_source(combined)
        if tree is None:
            combined_top = resolved_source.rstrip("\n") + "\n" + dedented
            tree, tree_source_text = _parse_python_with_context_and_source(combined_top)
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
            lt, lsrc = _parse_python_with_context_and_source(s_ln)
            if lt is not None:
                lv = _SecurityASTVisitor(policy=policy)
                lv.visit(lt)
                l_active, _ = _partition_suppressed_ast_findings(lv, lsrc, allow_suppression=allow_suppression)
                line_findings.extend(l_active)
        line_findings = list(dict.fromkeys(line_findings))
        explicit_bad_syntax = any(
            re.match(r"^\s*(?:def|class)\s+[A-Za-z_]\w*\s*\([^)]*:\s*$", ln)
            for ln in added_lines_text
        )
        partial_needs_source = bool(has_unified and not resolved_source and not explicit_bad_syntax and not line_findings)
        malformed_findings = list(dict.fromkeys(line_findings + ["syntax_tree_malformed"]))
        return {
            "language": "python",
            "syntax_valid": False,
            "partial_hunk_needs_source": partial_needs_source,
            "security_clean": len(line_findings) == 0,
            "findings": malformed_findings,
            "finding_severities": {
                f: ("medium" if f in _DEV_ESCALATE_ONLY_AST_FINDINGS else "critical")
                for f in malformed_findings
            },
            "suppressed_findings": list(suppressed_findings),
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
    tree_supp: List[Dict[str, Any]] = []
    if tree is not None and tree is not patched_tree:
        visitor = _SecurityASTVisitor(policy=policy)
        if seed_visitor is not None:
            visitor.module_aliases.update(seed_visitor.module_aliases)
            visitor.dangerous_aliases.update(seed_visitor.dangerous_aliases)
            visitor.path_vars.update(seed_visitor.path_vars)
            visitor.decoded_vars.update(seed_visitor.decoded_vars)
        visitor.visit(tree)
        tree_active, tree_supp = _partition_suppressed_ast_findings(visitor, tree_source_text, allow_suppression=allow_suppression)
        hunk_findings.extend(tree_active)
        structural_findings.extend(visitor.structural_findings)
        if visitor.has_network_sink and (
            visitor.has_env_secret_read or any("ast_protected_file_access" in f for f in tree_active)
        ):
            hunk_findings.append("ast_secret_exfil_pattern")

    if patched_tree is not None:
        patched_visitor = _SecurityASTVisitor(policy=policy)
        patched_visitor.visit(patched_tree)
        if patched_visitor.has_network_sink and (
            patched_visitor.has_env_secret_read
            or any("ast_protected_file_access" in f for f in patched_visitor.findings)
        ):
            patched_visitor.findings.append("ast_secret_exfil_pattern")
        seed_active: List[str] = []
        seed_supp: List[Dict[str, Any]] = []
        if seed_visitor is not None and resolved_source is not None:
            seed_active, seed_supp = _partition_suppressed_ast_findings(seed_visitor, resolved_source, allow_suppression=allow_suppression)
        patched_active, patched_supp = _partition_suppressed_ast_findings(patched_visitor, patched_source or "", allow_suppression=allow_suppression)
        baseline_counts = Counter(seed_active)
        delta_counts = Counter(patched_active) - baseline_counts
        for finding, cnt in delta_counts.items():
            if cnt > 0:
                hunk_findings.append(finding)
        seed_supp_counts = Counter((s["rule"], s["reason"]) for s in seed_supp)
        for s_item in patched_supp:
            key = (s_item["rule"], s_item["reason"])
            if seed_supp_counts[key] > 0:
                seed_supp_counts[key] -= 1
            else:
                suppressed_findings.append(s_item)
        if not structural_findings:
            structural_findings.extend(patched_visitor.structural_findings)
    elif tree_supp:
        suppressed_findings.extend(tree_supp)

    if fname_lower in {"conftest.py", "sitecustomize.py", "usercustomize.py"}:
        from .deferred_exec_guard import analyze_deferred_execution

        hunk_findings.extend(
            analyze_deferred_execution(path_str, patched_source if patched_source is not None else dedented, policy=policy)
        )

    all_findings = list(dict.fromkeys(hunk_findings + secret_findings))
    finding_severities = {
        f: ("medium" if f in _DEV_ESCALATE_ONLY_AST_FINDINGS else "critical")
        for f in all_findings
    }
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
        "finding_severities": finding_severities,
        "suppressed_findings": suppressed_findings,
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
            elif ch == "&" and not cmd.startswith("&>", i) and not (i > 0 and cmd[i - 1] in (">", "<", "|")):
                blocks.append("".join(curr).strip())
                curr = []
                i += 1
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


def _tokenize_shell_segment_checked(seg: str) -> Tuple[List[str], bool]:
    """Tokenize a shell segment with shlex, reporting whether quotes were unbalanced."""
    try:
        return shlex.split(seg, posix=True), False
    except ValueError:
        return seg.strip().split(), True


def _tokenize_shell_segment(seg: str) -> List[str]:
    tokens, _unbalanced = _tokenize_shell_segment_checked(seg)
    return tokens


SAFE_SHELL_COMMANDS: Set[str] = {
    "ls", "pwd", "cd", "echo", "printf", "cat", "head", "tail", "wc", "sort", "uniq",
    "grep", "rg", "find", "which", "where", "whoami", "id", "uname", "hostname", "date",
    "stat", "file", "tree", "dirname", "basename", "realpath", "readlink",
    "diff", "comm", "cmp", "nl", "less", "more", "bat", "fd", "exa", "eza",
    "du", "df", "true", "false", "test", "[",
    "md5sum", "sha256sum", "shasum", "cksum", "strings", "ps", "pgrep", "sleep",
    "rev", "fold", "fmt", "paste", "expand", "unexpand",
    "clang-format", "black", "ruff", "mypy", "pytest",
    "jq", "yq", "column", "cut", "tr",
}
SAFE_DEV_SHELL_COMMANDS: Set[str] = {
    "mkdir", "touch", "cp", "mv", "rm", "rmdir",
    "npm", "pnpm", "yarn", "cargo", "pip", "poetry", "uv",
    "flake8", "isort", "pylint", "pyright", "ty", "bandit", "tox", "nox",
    "node", "bun", "deno", "tsc", "eslint", "prettier", "vitest", "jest",
    "biome", "tsx", "vite", "next", "turbo", "webpack", "esbuild", "rollup", "swc",
    "go", "gofmt", "golangci-lint", "rustc", "rustfmt", "clippy",
    "make", "cmake", "ninja", "just", "bazel", "gradle", "gradlew", "mvn", "mvnw",
    "dotnet", "javac", "java", "gcc", "g++", "clang", "clang++", "zig", "swift",
    "ruby", "bundle", "rspec", "php", "composer", "mix", "elixir",
}
SAFE_GIT_SUBCOMMANDS: Set[str] = {
    "status", "log", "diff", "branch", "show", "rev-parse", "describe", "ls-files",
    "rev-list", "shortlog", "blame", "grep", "ls-tree", "cat-file",
    "for-each-ref", "symbolic-ref", "merge-base", "name-rev", "reflog", "check-ignore",
}
SAFE_DEV_GIT_SUBCOMMANDS: Set[str] = {
    "add", "commit", "checkout", "switch", "stash", "fetch", "pull", "merge", "tag", "clean",
}
_GUARDRAILS_ESCALATE_COMMANDS: Set[str] = {
    # Network / egress
    "curl", "wget", "nc", "ncat", "netcat", "socat", "ssh", "scp", "sftp",
    "rsync", "ftp", "telnet", "iwr", "irm", "invoke-webrequest", "invoke-restmethod",
    # Privilege / system / permissions
    "sudo", "doas", "su", "chmod", "chown", "chgrp", "mount", "umount",
    "systemctl", "service", "crontab", "at", "kill", "killall", "pkill",
    "shutdown", "reboot", "iptables", "ufw", "useradd", "usermod", "passwd",
    # Destructive / recursive delete / raw disk
    "rm", "dd", "shred", "truncate", "wipefs", "fdisk", "parted", "mkfs",
    # Shell / interpreter inline exec & wrappers
    "sh", "bash", "zsh", "dash", "fish", "ksh", "cmd", "powershell", "pwsh",
    "eval", "exec", "xargs", "env", "strace", "unbalanced_shell_quotes", "env_assignment",
    # Non-allowlisted VCS / package / runtime / container / dangerous-flag tool invocations
    "git", "pip", "pip3", "npm", "pnpm", "yarn", "npx", "python", "python3", "node",
    "gh", "docker", "podman", "kubectl",
    "sed", "awk", "gawk", "mawk", "nawk", "find", "rg", "sort", "yq", "clang-format", "mypy",
}

_SAFE_ANALYTICAL_TOOLS: Set[str] = {
    "grep", "rg", "wc", "sort", "uniq", "head", "tail", "cat", "column", "cut", "tr", "jq", "yq", "file",
}
_FILE_READING_SHELL_COMMANDS: Set[str] = {
    "cat", "head", "tail", "less", "more", "bat", "wc", "sort", "uniq",
    "cut", "jq", "yq", "file", "ls", "find", "grep", "egrep", "fgrep", "rg",
}

_SHELL_WRAPPER_COMMANDS: Set[str] = {
    "env", "command", "nohup", "nice", "timeout", "time", "sudo", "doas", "exec", "strace", "xargs",
}
_DANGEROUS_ENV_OVERRIDES: Set[str] = {
    "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    "PYTHONPATH", "PYTHONSTARTUP", "BASH_ENV", "ENV", "NODE_OPTIONS", "PERL5OPT", "RUBYOPT",
}
_SHELL_MUTATING_COMMANDS: Set[str] = {
    "rm", "mv", "cp", "chmod", "chown", "tee",
    "truncate", "shred", "dd", "ln", "touch", "del", "erase", "rmdir", "rd",
}
_SHELL_SECRET_EGRESS_COMMANDS: Set[str] = {
    "echo", "printf", "printenv", "curl", "wget", "nc", "ncat", "netcat",
    "socat", "ssh", "scp", "rsync", "ftp", "sftp", "iwr", "irm",
    "invoke-webrequest", "invoke-restmethod", "write-output", "write-host",
}
_SHELL_NETWORK_EGRESS_COMMANDS: Set[str] = {
    "curl", "wget", "nc", "ncat", "netcat", "socat", "ssh", "scp",
    "rsync", "ftp", "sftp", "iwr", "irm", "invoke-webrequest", "invoke-restmethod",
}
_SENSITIVE_ENV_VAR_RE = re.compile(
    r"(?i)(?:(?:^|_)(?:"
    r"SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|APIKEY|API_KEY|PRIVATE_KEY|"
    r"ACCESS_KEY|ACCESS_KEY_ID|CLIENT_SECRET|AUTH_TOKEN|AUTH_HEADER|BEARER|JWT|SESSION_TOKEN|"
    r"BOT_TOKEN|SIGNING_KEY|ENCRYPTION_KEY|MASTER_KEY|DATABASE_URL|DB_PASSWORD|"
    r"DB_PASS|REDIS_URL|REDIS_PASSWORD|PGPASSWORD|PGPASSFILE|MYSQL_PWD|STRIPE_KEY|"
    r"OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY|AWS_ACCESS_KEY_ID|"
    r"AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|GITHUB_TOKEN|GH_TOKEN|GITLAB_TOKEN|"
    r"HF_TOKEN|NPM_TOKEN|PYPI_TOKEN|VAULT_TOKEN|CI_JOB_TOKEN|API_SECRET|SECRET_KEY"
    r")(?:$|_)|(?:PASSWORD|PASSWD)$)"
)


def _is_inplace_mutating_text_tool(exe: str, args: List[str]) -> bool:
    """Return True only when sed/perl/awk are invoked with in-place write/execution flags."""
    if exe == "sed":
        return any(a == "-i" or a.startswith("-i") or a == "--in-place" or a.startswith("--in-place=") for a in args)
    if exe == "perl":
        return any(
            a == "-i" or a.startswith("-i") or (a.startswith("-") and not a.startswith("--") and "i" in a[1:])
            for a in args
        )
    if exe in {"awk", "gawk", "mawk", "nawk"}:
        for idx, a in enumerate(args):
            a_low = a.lower()
            if a_low.startswith("-iinplace"):
                return True
            if a_low == "-i" and idx + 1 < len(args) and args[idx + 1].lower() == "inplace":
                return True
            if not a.startswith("-") and ("system(" in a or ">" in a):
                return True
        return False
    return False
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
    """Extract unquoted redirections and process substitutions (`>`, `>>`, `2>&1`, `>(...)`, `<(...)`)."""
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

        if not in_single and not in_double and ch == "<":
            if i + 1 < n and cmd[i + 1] == "(":
                redirs.append({"kind": "process_substitution", "target": ""})
                i += 2
                continue
            # Input redirection `< file` or `0< file` (ignore heredoc `<<` / here-string `<<<`)
            if not (i + 1 < n and cmd[i + 1] == "<"):
                j = i + 1
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
                in_target = "".join(tok_chars)
                if in_target:
                    redirs.append({"kind": "input_file", "target": in_target})
                i = j
                continue
            i += 2
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
    if _is_live_package_path(t):
        # The running gate's own source: never a redirect target.
        return True
    return False


def _is_safe_dev_pkg_command(exe: str, args_lower: List[str]) -> bool:
    """Reject untrusted VCS/URL/editable/global installs so they escalate even in dev profile."""
    if exe == "pip":
        for a in args_lower:
            if a in {"-e", "--editable", "--index-url", "-i", "--extra-index-url", "--find-links", "-f", "--trusted-host"}:
                return False
            if a.startswith(("git+", "hg+", "svn+", "bzr+", "http://", "https://", "ssh://", "file://", "--index-url=", "--extra-index-url=", "--find-links=", "--trusted-host=")):
                return False
        return True
    if exe in {"npm", "pnpm", "yarn"}:
        if args_lower and args_lower[0] in {"exec", "dlx", "publish", "login", "adduser", "owner", "token"}:
            return False
        for a in args_lower:
            if a in {"-g", "--global", "global", "--registry"}:
                return False
            if a.startswith(("git+", "http://", "https://", "github:", "file:", "--registry=")):
                return False
        return True
    return True


def _is_command_allowlisted(exe: str, args: List[str], policy: Optional[PolicyConfig] = None) -> bool:
    """Strict Layer 1 allowlist check for a parsed command executable and its arguments."""
    cfg = policy or DEFAULT_POLICY
    is_dev = getattr(cfg, "profile", "strict") in {"dev", "guardrails"}
    args_lower = [a.lower() for a in args]
    if exe == "sed":
        if _is_inplace_mutating_text_tool(exe, args):
            return False
        if any(
            a_low in {"-f", "--file"} or a_low.startswith(("--file=", "-f"))
            for a_low in args_lower
        ):
            return False
        if any(
            not a.startswith("-")
            and (";e" in a_low or bool(re.search(r"/[a-z0-9]*e(?:$|;|\s)", a_low)))
            for a, a_low in zip(args, args_lower)
        ):
            return False
        return True
    if exe in {"awk", "gawk", "mawk", "nawk"}:
        if _is_inplace_mutating_text_tool(exe, args):
            return False
        if any(
            a_low in {"-f", "--file"} or a_low.startswith(("--file=", "-f"))
            for a_low in args_lower
        ):
            return False
        if any(
            not a.startswith("-") and ("getline" in a_low or "|" in a)
            for a, a_low in zip(args, args_lower)
        ):
            return False
        return True
    if exe in SAFE_SHELL_COMMANDS:
        if exe == "find" and any(
            a in {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}
            for a in args_lower
        ):
            return False
        if exe == "rg" and any(
            a in {"--pre", "--pre-glob"} or a.startswith(("--pre=", "--pre-glob="))
            for a in args_lower
        ):
            return False
        if exe == "sort" and any(
            a in {"--compress-program", "-o", "--output", "--files0-from"}
            or a.startswith(("--compress-program=", "--output=", "--files0-from="))
            or (a.startswith("-o") and len(a) > 2)
            for a in args_lower
        ):
            return False
        if exe == "yq" and any(
            a in {"-i", "--inplace"} or a.startswith("--inplace=")
            for a in args_lower
        ):
            return False
        if exe == "clang-format":
            for a in args:
                al = a.lower()
                if a.startswith("-"):
                    if al.startswith("--style=file:") or al == "--assume-filename" or al.startswith("--assume-filename="):
                        val = a.split("=", 1)[1] if "=" in a else ""
                        if val and (
                            val.startswith(("/", "~"))
                            or ".." in val.replace("\\", "/").split("/")
                            or is_protected_path(val)
                        ):
                            return False
                    continue
                if (
                    a.startswith(("/", "~"))
                    or ".." in a.replace("\\", "/").split("/")
                    or is_protected_path(a)
                ):
                    return False
        if exe == "mypy":
            if any(
                a in {"--config-file", "--custom-typeshed-dir", "--shadow-file"}
                or a.startswith(("--config-file=", "--custom-typeshed-dir=", "--shadow-file="))
                for a in args_lower
            ):
                return False
        if exe == "black" and not is_dev:
            return any(a in {"--check", "--diff"} for a in args_lower)
        if exe == "ruff" and not is_dev:
            if any(a in {"--fix", "--unsafe-fixes", "--fix-only"} for a in args_lower):
                return False
            if args_lower and args_lower[0] == "format" and not any(a in {"--check", "--diff"} for a in args_lower):
                return False
        if exe == "pytest":
            if not is_dev:
                if any(
                    a in {"-o", "--override-ini", "-c", "--pyargs", "-p"}
                    or a.startswith(("-o=", "--override-ini=", "-c=", "--pyargs=", "-p="))
                    for a in args_lower
                ):
                    return False
                for a in args:
                    if not a.startswith("-") and (a.startswith(("/", "~")) or ".." in a.replace("\\", "/").split("/")):
                        return False
            return True
        return True
    if is_dev and exe in SAFE_DEV_SHELL_COMMANDS:
        if exe == "rm":
            # In dev profile, rm is only allowed for non-recursive single files
            has_rec = any(a == "--recursive" or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a)) for a in args)
            return not has_rec
        if exe in {"pip", "npm", "pnpm", "yarn"}:
            return _is_safe_dev_pkg_command(exe, args_lower)
        if exe == "node":
            if any(
                a in {"-e", "--eval", "-p", "--print"}
                or a.startswith(("--eval=", "--print="))
                for a in args_lower
            ):
                return False
        return True
    if is_dev and exe == "npx":
        if any(
            a in {"-p", "--package", "-y", "--yes", "-c", "--call"}
            or a.startswith(("--package=", "--call="))
            for a in args_lower
        ):
            return False
        idx = 0
        while idx < len(args_lower) and args_lower[idx] in {"--no-install", "--no", "-q", "--quiet"}:
            idx += 1
        if idx >= len(args_lower):
            return False
        return args_lower[idx] in {
            "vitest", "jest", "tsc", "eslint", "prettier", "biome",
            "tsx", "vite", "next", "turbo", "webpack", "esbuild", "rollup", "swc",
        }
    if exe == "gh":
        if not args_lower:
            return True
        if args_lower[0] in {"status", "search", "--version"}:
            return True
        return tuple(args_lower[:2]) in {
            ("pr", "view"), ("pr", "list"), ("pr", "diff"), ("pr", "checks"), ("pr", "status"),
            ("issue", "view"), ("issue", "list"), ("issue", "status"),
            ("run", "view"), ("run", "list"), ("repo", "view"),
        }
    if exe in {"docker", "podman"}:
        if not args_lower:
            return True
        if args_lower[0] in {"ps", "images", "logs", "inspect", "version", "info", "stats", "top", "port"}:
            return True
        return tuple(args_lower[:2]) in {
            ("network", "ls"), ("volume", "ls"), ("compose", "ps"), ("compose", "logs"),
        }
    if exe == "kubectl":
        if not args_lower:
            return True
        return args_lower[0] in {"get", "describe", "logs", "version", "cluster-info", "api-resources", "explain", "top"}
    if exe == "git":
        if not args_lower:
            return True
        if any(
            low_a in {"-o", "--output", "--open-files-in-pager", "--ext-diff", "--textconv"}
            or low_a.startswith(("--exec", "--output=", "--open-files-in-pager="))
            for low_a in args_lower
        ):
            return False
        idx = 0
        while idx < len(args):
            raw_a = args[idx]
            low_a = args_lower[idx]
            if raw_a == "-c" or (raw_a.startswith("-c") and len(raw_a) > 2) or low_a.startswith("--config-env"):
                return False
            if raw_a == "-C":
                if idx + 1 >= len(args) or is_protected_path(args[idx + 1], cfg):
                    return False
                idx += 2
            elif raw_a.startswith("-C") and len(raw_a) > 2:
                if is_protected_path(raw_a[2:], cfg):
                    return False
                idx += 1
            elif low_a in {"--git-dir", "--work-tree"}:
                if idx + 1 >= len(args) or is_protected_path(args[idx + 1], cfg):
                    return False
                idx += 2
            elif low_a.startswith(("--git-dir=", "--work-tree=")):
                if is_protected_path(raw_a.split("=", 1)[1], cfg):
                    return False
                idx += 1
            elif raw_a in {"-p", "-P"} or low_a in {"--paginate", "--no-pager", "--bare"}:
                idx += 1
            else:
                break
        sub_args = args[idx:]
        sub_args_lower = args_lower[idx:]
        if not sub_args_lower:
            return True
        subcmd = sub_args_lower[0]
        if subcmd == "branch":
            for raw_a in sub_args[1:]:
                low_a = raw_a.lower()
                if low_a in {"-d", "--delete", "-f", "--force", "-m", "--move", "-c", "--copy", "--unset-upstream", "--edit-description"}:
                    return False
                if low_a.startswith("--set-upstream-to"):
                    return False
                if raw_a.startswith("-") and not raw_a.startswith("--") and any(ch in raw_a[1:] for ch in "dDfmMcC"):
                    return False
            if not is_dev:
                list_flags = {"-l", "--list", "--contains", "--no-contains", "--merged", "--no-merged", "--points-at"}
                has_non_flag = any(not a.startswith("-") for a in sub_args[1:])
                has_list_flag = any(a.lower() in list_flags or a.lower().startswith(("--contains=", "--merged=", "--points-at=")) for a in sub_args[1:])
                if has_non_flag and not has_list_flag:
                    return False
            return True
        if subcmd in SAFE_GIT_SUBCOMMANDS:
            return True
        if is_dev and subcmd in SAFE_DEV_GIT_SUBCOMMANDS:
            if subcmd == "clean":
                if any(a.startswith("-") and not a.startswith("--") and any(ch in a for ch in "fdxX") for a in sub_args_lower[1:]):
                    return False
                if any(a in {"--force", "-f", "-d", "-x", "-X"} for a in sub_args_lower[1:]):
                    return False
            elif subcmd == "checkout":
                for low_a in sub_args_lower[1:]:
                    if low_a in {"--", ".", "./", "-f", "--force", "--ours", "--theirs"}:
                        return False
                    if low_a.startswith("-") and not low_a.startswith("--") and "f" in low_a[1:]:
                        return False
            elif subcmd == "switch":
                for raw_a in sub_args[1:]:
                    low_a = raw_a.lower()
                    if low_a in {"-f", "--force", "--discard-changes"} or raw_a == "-C":
                        return False
                    if raw_a.startswith("-") and not raw_a.startswith("--") and ("f" in raw_a[1:] or "C" in raw_a[1:]):
                        return False
            elif subcmd == "stash":
                if len(sub_args_lower) >= 2 and sub_args_lower[1] in {"drop", "clear"}:
                    return False
            elif subcmd == "tag":
                for low_a in sub_args_lower[1:]:
                    if low_a in {"-d", "--delete", "-f", "--force"}:
                        return False
                    if low_a.startswith("-") and not low_a.startswith("--") and ("d" in low_a[1:] or "f" in low_a[1:]):
                        return False
            return True
        return False
    if exe in {"python", "python3"}:
        if args_lower in (["--version"], ["-v"]):
            return True
        scan_idx = 0
        while scan_idx < len(args_lower):
            tok = args_lower[scan_idx]
            if tok == "-w":
                scan_idx += 2
                continue
            if tok.startswith("-w") or re.fullmatch(r"-[ubiesoqvx]+", tok):
                scan_idx += 1
                continue
            break
        if scan_idx + 1 < len(args_lower) and args_lower[scan_idx] == "-m":
            mod_target = args_lower[scan_idx + 1].replace("-", "_")
            if mod_target in {"http.server", "http_server", "pydoc"}:
                return False
            if mod_target in {"pytest", "unittest", "py_compile"}:
                return True
            if mod_target == "aetheris_microbrains.audit":
                # Local package audit CLI (audit --last/--tail/--summary): read-only self-inspection.
                return True
            if mod_target == "pip" or mod_target.startswith("pip."):
                if is_dev and mod_target == "pip":
                    return _is_safe_dev_pkg_command("pip", args_lower[scan_idx + 2 :])
                # Read-only pip introspection is safe in any profile; anything else stays dev-only.
                sub = args_lower[scan_idx + 2] if len(args_lower) > scan_idx + 2 else ""
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
    if exe in {"pip", "pip3"}:
        sub = args_lower[0] if args_lower else ""
        if sub in {"show", "list", "--version"}:
            # Read-only package introspection is safe in any profile.
            return True
        if is_dev:
            return _is_safe_dev_pkg_command("pip", args_lower)
        return False
    return False


def _extract_unquoted_env_vars(cmd: str) -> List[str]:
    """Extract environment variable names expanded outside single quotes (`$VAR`, `${VAR}`, `$env:VAR`, `%VAR%`).

    Properly tracks double quotes so apostrophes inside double quotes (e.g. `"it's $GITHUB_TOKEN"`)
    do not falsely suppress variable extraction.
    """
    found: List[str] = []
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
        if ch == "\\" and not in_single:
            if not in_double or (i + 1 < n and cmd[i + 1] in ('"', "\\", "$", "`", "\n")):
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


def _expand_bash_braces(
    text: str,
    *,
    max_results: int = 64,
    _depth: int = 0,
    _max_depth: int = 4,
) -> List[str]:
    """Expand bash-style brace expansions with bounded recursion and range size."""
    if _depth >= _max_depth:
        return [text]
    m = re.search(r"\{([^{}]+)\}", text)
    if not m:
        return [text]
    options = m.group(1).split(",")
    if len(options) == 1:
        if ".." in options[0]:
            parts = options[0].split("..")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                if abs(end - start) + 1 > max_results:
                    return [text]
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
        for exp in _expand_bash_braces(
            prefix + opt + suffix,
            max_results=max_results,
            _depth=_depth + 1,
            _max_depth=_max_depth,
        ):
            results.append(exp)
            if len(results) >= max_results:
                return results
    return results


def _extract_candidate_paths_from_arg(arg: str) -> List[str]:
    """Extract filesystem path candidates from a shell argument (flags, `<file`, `0<file`, `rev:path`, bash braces)."""
    cleaned = str(arg or "").strip()
    if not cleaned:
        return []
    m_in = re.match(r"^\d*<(?!<)(.+)$", cleaned)
    if m_in:
        cleaned = m_in.group(1).strip()
    elif cleaned.startswith("<"):
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


_CRITICAL_RM_PREFIXES: Tuple[str, ...] = (
    "/etc", "/var", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/root",
    "/sys", "/proc", "/dev", "/opt", "/home", "/users",
    "c:/windows", "c:/program files", "c:/users",
)


def _is_destructive_rm_target(arg: str, policy: Optional[PolicyConfig] = None) -> bool:
    """Return True for root/system/home/repo-metadata/traversal targets that must hard-deny under `rm -rf`."""
    norm = str(arg or "").replace("\\", "/").strip().lower()
    if not norm:
        return False
    exact_danger = {
        "/", "/*", "~", "~/", "$home", "${home}", "*", ".", "./", "./*", "..", "c:", "c:/",
        "/tmp", "/tmp/", "/tmp/*",
        "/var/tmp", "/var/tmp/", "/var/tmp/*",
        "/private/tmp", "/private/tmp/", "/private/tmp/*",
        ".git", ".git/", "./.git", "./.git/",
    }
    if norm in exact_danger or norm.rstrip("/").endswith("/.git"):
        return True
    if norm.startswith(("~/", "$home/", "${home}/")) or ".." in [seg for seg in norm.split("/") if seg]:
        return True
    # Concrete scratch subdirectories inside /tmp/ /var/tmp/ /private/tmp/ escalate instead of hard-denying
    for tmp_pref in ("/tmp/", "/var/tmp/", "/private/tmp/"):
        if norm.startswith(tmp_pref):
            rest = norm[len(tmp_pref) :].strip("/")
            if rest and not any(ch in rest for ch in ("*", "?", "[", "$")) and not is_protected_path(norm, policy=policy):
                return False
            return True
    for pref in _CRITICAL_RM_PREFIXES:
        if norm == pref or norm.startswith(pref + "/"):
            return True
    if norm.startswith("/") and len([seg for seg in norm.split("/") if seg]) <= 1:
        return True
    return is_protected_path(norm, policy=policy)


def _split_shell_subcommands(cmd: str) -> List[str]:
    """Split shell command on separators (&&, ||, |&, ;, |, &, newline) outside quotes."""
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
            elif ch == "&" and not cmd.startswith("&>", i) and not (i > 0 and cmd[i - 1] in (">", "<", "|")):
                parts.append("".join(buf))
                buf = []
                i += 1
            elif ch in (";", "|", "\n"):
                parts.append("".join(buf))
                buf = []
                i += 1
            else:
                buf.append(ch)
                i += 1

    if in_single or in_double:
        # Unbalanced quotes: fail closed by falling back to raw separator splitting (including single `&`).
        return re.split(r"(?:&&|\|\||\|&|;|\||\r?\n|(?<![<>])&(?!>))", cmd)

    if buf:
        parts.append("".join(buf))
    return parts


_PS_SAFE_READONLY_CMDLETS: Set[str] = {
    "get-date", "get-location", "get-childitem", "get-content", "get-item",
    "get-itemproperty", "get-host", "get-process", "get-service", "get-command",
    "get-help", "get-module", "get-alias", "get-history", "test-path",
    "select-object", "select-string", "measure-object", "compare-object",
    "sort-object", "format-table", "format-list", "format-wide", "where-object",
    "write-output", "write-host", "clear-host", "exit",
}
_PS_DANGEROUS_CMDLETS: Dict[str, str] = {
    "invoke-expression": "powershell_invoke_expression",
    "iex": "powershell_invoke_expression",
    "invoke-item": "powershell_invoke_item",
    "ii": "powershell_invoke_item",
    "invoke-command": "powershell_invoke_command",
    "icm": "powershell_invoke_command",
    "start-process": "powershell_start_process",
    "saps": "powershell_start_process",
    "start": "powershell_start_process",
    "add-type": "powershell_add_type",
}
_PS_MUTATING_CMDLETS: Set[str] = {
    "set-content", "sc", "add-content", "ac", "out-file", "clear-content", "clc",
    "remove-item", "ri", "del", "rd", "erase", "rmdir", "rm",
    "move-item", "mi", "mv", "copy-item", "ci", "cp", "rename-item", "rni",
    "new-item", "ni", "set-itemproperty", "new-itemproperty", "remove-itemproperty",
}
_PS_EGRESS_CMDLETS: Set[str] = {
    "invoke-webrequest", "iwr", "invoke-restmethod", "irm", "curl", "wget",
    "write-output", "write-host", "echo",
}
_PS_UNIX_DELEGATED_DESTRUCTIVE: Set[str] = {
    "rm", "dd", "chmod", "mkfs", "del", "erase", "rmdir", "rd",
}
_PS_REDIR_RE = re.compile(r"(?:[1-6*]?>>?|&>>?)\s*(\"[^\"]*\"|'[^']*'|[^\s;|&<>]+)")


_PS_DANGEROUS_DOTNET_RE = re.compile(
    r"(?i)(?:net\.webclient|\.\s*download(?:string|file|data)\s*\(|\.\s*kill\s*\(|"
    r"reflection\.assembly\s*::\s*load|system\.diagnostics\.process\s*::\s*start|"
    r"scriptcontrol|wscript\.shell|shell\.application)"
)


def _extract_ps_nested_blocks(script: str) -> List[Tuple[str, str]]:
    """Extract inner bodies of `{ ... }`, `$( ... )`, and `( ... )` expressions as `(kind, body)`."""
    blocks: List[Tuple[str, str]] = []
    n = len(script)
    i = 0
    in_single = False
    in_double = False
    while i < n:
        ch = script[i]
        if ch == "`" and not in_single and i + 1 < n:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single and (ch in "({" or (ch == "$" and i + 1 < n and script[i + 1] == "(")):
            if ch == "{":
                kind, open_ch, close_ch, start = "scriptblock", "{", "}", i + 1
            elif ch == "$":
                kind, open_ch, close_ch, start = "subexpr", "(", ")", i + 2
            else:
                kind, open_ch, close_ch, start = "paren", "(", ")", i + 1
            depth = 1
            j = start
            sub_single = False
            sub_double = False
            while j < n and depth > 0:
                c = script[j]
                if c == "`" and not sub_single and j + 1 < n:
                    j += 2
                    continue
                if c == "'" and not sub_double:
                    sub_single = not sub_single
                elif c == '"' and not sub_single:
                    sub_double = not sub_double
                elif not sub_single:
                    if c == open_ch:
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            body = script[start:j].strip()
            if body:
                blocks.append((kind, body))
            i = start
            continue
        i += 1
    return blocks


def _tokenize_ps_statement(stmt: str) -> Tuple[List[str], bool]:
    """Tokenize a PowerShell statement keeping Windows backslashes intact."""
    try:
        raw_toks = shlex.split(stmt, posix=False)
        unbalanced = False
    except ValueError:
        raw_toks = stmt.strip().split()
        unbalanced = True
    cleaned: List[str] = []
    for tok in raw_toks:
        if len(tok) >= 2 and ((tok[0] == tok[-1] == '"') or (tok[0] == tok[-1] == "'")):
            cleaned.append(tok[1:-1])
        else:
            cleaned.append(tok)
    return cleaned, unbalanced


def _inspect_ps_script(
    script: str,
    policy: Optional[PolicyConfig],
    findings: List[str],
    unrecognized: List[str],
    protected_path_reads: List[str],
    sensitive_env_vars: List[str],
    state: Dict[str, bool],
    depth: int = 0,
) -> None:
    """Recursively inspect a PowerShell `-Command` script and nested blocks/subexpressions."""
    if depth > 4 or not script.strip():
        return
    scan_text = script
    if "`" in script:
        # PowerShell backtick escapes can disguise cmdlets (e.g. `i`ex` or `Remove-`Item`)
        if "powershell_backtick_escape" not in unrecognized:
            unrecognized.append("powershell_backtick_escape")
        scan_text = script.replace("`", "")

    if "::" in scan_text and "powershell_dotnet_static_call" not in unrecognized:
        unrecognized.append("powershell_dotnet_static_call")

    if _PS_DANGEROUS_DOTNET_RE.search(scan_text):
        findings.append("powershell_dotnet_dangerous_method")

    nested = _extract_ps_nested_blocks(scan_text)
    if nested or any(sym in scan_text for sym in ("{", "}", "$(", "(", ")")):
        if "powershell_scriptblock_or_subexpr" not in unrecognized:
            unrecognized.append("powershell_scriptblock_or_subexpr")
        for _kind, body in nested:
            _inspect_ps_script(
                body,
                policy,
                findings,
                unrecognized,
                protected_path_reads,
                sensitive_env_vars,
                state,
                depth=depth + 1,
            )

    for m in _PS_REDIR_RE.finditer(scan_text):
        r_target = m.group(1).strip().strip("\"'")
        if not r_target or r_target.lower() in {"$null", "/dev/null", "null"}:
            continue
        if is_protected_path(r_target, policy=policy):
            findings.append("powershell_protected_path_mutation")
        elif "powershell_redirection" not in unrecognized:
            unrecognized.append("powershell_redirection")

    # Strip out redirection operators+targets so statement tokenization sees the command
    stmt_text = _PS_REDIR_RE.sub(" ", scan_text)
    for raw_stmt in _split_shell_subcommands(stmt_text):
        stmt = raw_stmt.strip()
        while len(stmt) >= 2 and stmt.startswith("(") and stmt.endswith(")"):
            stmt = stmt[1:-1].strip()
        if not stmt:
            continue
        toks, unbalanced = _tokenize_ps_statement(stmt)
        if unbalanced and "powershell_unbalanced_quotes" not in unrecognized:
            unrecognized.append("powershell_unbalanced_quotes")
        if not toks:
            continue
        # Handle leading PowerShell call (`&`) or dot-source (`.`) operator
        if toks[0] in {"&", "."}:
            if "powershell_call_operator" not in unrecognized:
                unrecognized.append("powershell_call_operator")
            toks = toks[1:]
            if not toks:
                continue
        elif toks[0].startswith(("&", ".")) and len(toks[0]) > 1 and not toks[0].startswith((".", "./", ".\\")):
            if toks[0].startswith("&"):
                if "powershell_call_operator" not in unrecognized:
                    unrecognized.append("powershell_call_operator")
                toks = [toks[0][1:]] + toks[1:]

        first_tok = toks[0].lstrip("(")
        first_raw = re.split(r"[\s({]", first_tok, maxsplit=1)[0].strip().strip("'\"").lower()
        first = os.path.basename(first_raw).removesuffix(".exe").removesuffix(".com")
        stmt_args = toks[1:]
        if not first:
            if "powershell_unparseable_statement" not in unrecognized:
                unrecognized.append("powershell_unparseable_statement")
            continue

        # Inspect every token for PowerShell env/provider drives and protected filesystem paths
        for tok in toks:
            tok_low = tok.lower()
            env_m = re.search(r"(?:^|\$)(env|cert|hkcu|hklm|variable):([A-Za-z0-9_]*)", tok_low)
            if env_m:
                drv, env_key = env_m.group(1), env_m.group(2)
                if drv == "env" and env_key:
                    if _SENSITIVE_ENV_VAR_RE.search(env_key):
                        findings.append("shell_sensitive_env_exposure")
                        if env_key not in sensitive_env_vars:
                            sensitive_env_vars.append(env_key)
                    else:
                        state["sensitive_env_access"] = True
                else:
                    state["sensitive_env_access"] = True

            for cand in _extract_candidate_paths_from_arg(tok):
                cand_norm = cand.replace("\\", "/")
                if _PROC_ENVIRON_RE.search(cand_norm):
                    findings.append("shell_proc_environ_access")
                    break
                if is_protected_path(cand, policy=policy):
                    if first in _PS_MUTATING_CMDLETS:
                        findings.append("shell_protected_path_mutation")
                    elif first in _PS_EGRESS_CMDLETS:
                        findings.append(f"shell_protected_path_egress:{first}")
                    elif cand not in protected_path_reads:
                        protected_path_reads.append(cand)
                    break

        if first in _PS_DANGEROUS_CMDLETS:
            findings.append(_PS_DANGEROUS_CMDLETS[first])
            continue

        if first in {"remove-item", "ri", "rm"}:
            has_rec_or_force = False
            for a in stmt_args:
                alow = a.lower()
                if alow.startswith("-") and not alow.startswith("--") and len(alow) >= 2:
                    if "-recurse".startswith(alow) or "-force".startswith(alow) or "r" in alow[1:]:
                        has_rec_or_force = True
                        break
            targets_danger = any(
                _is_destructive_rm_target(a, policy=policy)
                for a in stmt_args
                if not a.startswith("-")
            )
            if has_rec_or_force or targets_danger:
                findings.append("powershell_remove_item_recursive")
            else:
                unrecognized.append(f"powershell:{first}")
            continue

        if first in _PS_UNIX_DELEGATED_DESTRUCTIVE or first.startswith("mkfs."):
            safe_args = " ".join(tok.replace("\\", "/") for tok in toks)
            sub_res = analyze_shell_command(safe_args, policy=policy)
            if not sub_res.get("is_safe", True):
                findings.extend(sub_res.get("findings", []))
            else:
                unrecognized.append(f"powershell:{first}")
            continue

        if first in _PS_SAFE_READONLY_CMDLETS:
            continue

        unrecognized.append(f"powershell:{first}")


def _analyze_powershell_invocation(
    args: List[str],
    policy: Optional[PolicyConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str], bool]:
    """B3: analyze `powershell`/`pwsh -Command` payloads against an explicit read-only cmdlet surface.

    Returns `(findings, unrecognized, fully_safe)`. `-EncodedCommand` / `/enc` vetoes
    (`powershell_encoded_command`); `-File` and opaque invocations stay
    unrecognized (escalate); scriptblocks/redirections/subexpressions escalate or
    veto if dangerous cmdlets or protected-path writes are inside. Optional `extra`
    receives `protected_path_reads`, `sensitive_env_access`, and `sensitive_env_vars`.
    """
    findings: List[str] = []
    unrecognized: List[str] = []
    protected_reads: List[str] = []
    sensitive_vars: List[str] = []
    state = {"sensitive_env_access": False}
    mode: Optional[str] = None
    script_parts: List[str] = []
    for idx, a in enumerate(args):
        low = a.lower()
        if low.startswith("/") and len(low) >= 2:
            low = "-" + low[1:]
        if low in {"-encodedcommand", "-enc", "-e", "-ec"} or (
            low.startswith("-en") and "-encodedcommand".startswith(low)
        ):
            findings.append("powershell_encoded_command")
            return findings, unrecognized, False
        if low in {"-command", "-c"} or (
            low.startswith("-co") and "-command".startswith(low)
        ):
            mode = "command"
            script_parts = args[idx + 1 :]
            break
        if low in {"-file", "-f"} or (low.startswith("-fi") and "-file".startswith(low)):
            mode = "file"
            break
    if mode != "command" or not script_parts:
        unrecognized.append("powershell_opaque_payload")
        return findings, unrecognized, False

    script = " ".join(script_parts)
    _inspect_ps_script(
        script,
        policy=policy,
        findings=findings,
        unrecognized=unrecognized,
        protected_path_reads=protected_reads,
        sensitive_env_vars=sensitive_vars,
        state=state,
    )
    dedup_findings = list(dict.fromkeys(findings))
    dedup_unrec = list(dict.fromkeys(unrecognized))
    if extra is not None:
        extra["protected_path_reads"] = protected_reads
        extra["sensitive_env_access"] = state["sensitive_env_access"]
        extra["sensitive_env_vars"] = sensitive_vars
    return dedup_findings, dedup_unrec, (not dedup_findings and not dedup_unrec)


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
    pipeline_blocks = _split_command_pipeline_blocks(cmd)

    # Check destructive SQL statements while avoiding false positives on standalone echo/grep/rg
    _SQL_TEXT_PRINT_CMDS = {"echo", "printf", "grep", "egrep", "fgrep", "rg"}
    _SQL_DB_CLIENTS = {"psql", "mysql", "mariadb", "sqlite3", "sqlcmd", "mongosh", "clickhouse-client"}
    if re.search(r"\b(drop\s+database|truncate\s+table)\b", lower_cmd):
        is_destructive_sql = bool(re.match(r"^\s*(?:drop\s+database|truncate\s+table)\b", lower_cmd))
        if not is_destructive_sql:
            for block in pipeline_blocks:
                pipe_segs = _split_unquoted_pipes(block)
                block_exes: List[str] = []
                for ps in pipe_segs:
                    ps_toks = _tokenize_shell_segment(_clean_shell_grouping(ps))
                    ps_prem, _, _, _, _ = _unwrap_shell_prefix(ps_toks)
                    ps_eff = ps_prem or ps_toks
                    if ps_eff:
                        block_exes.append(os.path.basename(ps_eff[0]).lower().removesuffix(".exe").removesuffix(".com"))
                piped_to_db = any(e in _SQL_DB_CLIENTS for e in block_exes[1:])
                for seg in _split_shell_subcommands(block):
                    seg_low = seg.lower()
                    if not re.search(r"\b(drop\s+database|truncate\s+table)\b", seg_low):
                        continue
                    if re.match(r"^\s*(?:drop\s+database|truncate\s+table)\b", seg_low):
                        is_destructive_sql = True
                        break
                    toks = _tokenize_shell_segment(seg)
                    prem, _, _, _, _ = _unwrap_shell_prefix(toks)
                    eff = prem or toks
                    seg_exe = os.path.basename(eff[0]).lower().removesuffix(".exe").removesuffix(".com") if eff else ""
                    if seg_exe not in _SQL_TEXT_PRINT_CMDS or piped_to_db:
                        is_destructive_sql = True
                        break
                if is_destructive_sql:
                    break
        if is_destructive_sql:
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
    protected_path_reads: List[str] = []
    meta_finding = _has_unsafe_shell_metachar(cmd)
    if meta_finding:
        findings.append(meta_finding)

    unprivileged_redirects: List[str] = []
    for redir in _extract_shell_redirections(cmd):
        kind = redir["kind"]
        r_target = redir["target"]
        if kind == "fd_dup":
            continue
        if kind == "input_file":
            if _PROC_ENVIRON_RE.search(r_target.replace("\\", "/")):
                findings.append("shell_proc_environ_access")
            elif is_protected_path(r_target, policy=policy):
                protected_path_reads.append(r_target)
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

    # Optional SOTA tree-sitter-bash AST check when tree_sitter_bash is installed
    ts_bash = _scan_bash_ast_tree_sitter(cmd, policy=policy)
    if ts_bash.get("available"):
        # dedupe-preserving merge: tree-sitter re-flags constructs the shlex pass already caught
        for f_name in ts_bash.get("findings") or []:
            if f_name not in findings:
                findings.append(f_name)
        for p_read in ts_bash.get("protected_reads") or []:
            if p_read not in protected_path_reads:
                protected_path_reads.append(p_read)

    # `$IFS` / `${IFS}` used as a word separator is an obfuscation technique, not routine agent usage
    if re.search(r"\$\{?IFS\b", cmd):
        findings.append("shell_ifs_obfuscation")

    # Parse pipeline blocks & subcommands once and reuse across egress, RCE, and builtin checks
    parsed_blocks = [
        (block, _split_unquoted_pipes(block))
        for block in pipeline_blocks
    ]

    # Inspect pipes for secret/env exfiltration (e.g. cat .env | curl ..., env | base64 | curl ...)
    for _logical_block, pipe_segments in parsed_blocks:
        if len(pipe_segments) > 1:
            has_protected_read = False
            has_env_dump = False
            has_egress_sink = False
            has_network_egress_sink = False
            for p in pipe_segments:
                p = _clean_shell_grouping(p)  # strip subshell/brace-group wrappers
                sub_segments = _split_shell_subcommands(p)
                for sub_seg in sub_segments:
                    for s_redir in _extract_shell_redirections(sub_seg):
                        if s_redir["kind"] == "input_file" and is_protected_path(s_redir["target"], policy=policy):
                            has_protected_read = True
                    ptoks = _tokenize_shell_segment(sub_seg)
                    prem, _pwrap, _pwf, p_bare_env, _pinl = _unwrap_shell_prefix(ptoks)
                    if p_bare_env:
                        has_env_dump = True
                    eff = prem or ptoks
                    if not eff:
                        continue
                    pexe = os.path.basename(eff[0]).lower().removesuffix(".exe").removesuffix(".com")
                    if pexe in {"env", "printenv"}:
                        non_flag = [a for a in eff[1:] if not a.startswith("-") and "=" not in a]
                        if pexe == "printenv" or not non_flag:
                            has_env_dump = True
                    elif pexe in {"export", "declare", "typeset"} and (len(eff) == 1 or any(a in {"-p", "-x"} for a in eff[1:])):
                        has_env_dump = True
                    if pexe in _SHELL_SECRET_EGRESS_COMMANDS:
                        has_egress_sink = True
                    if pexe in _SHELL_NETWORK_EGRESS_COMMANDS:
                        has_network_egress_sink = True
                    for arg_tok in eff[1:]:
                        for cand in _extract_candidate_paths_from_arg(arg_tok):
                            if is_protected_path(cand, policy=policy):
                                has_protected_read = True
            if has_protected_read and has_egress_sink:
                findings.append("shell_piped_protected_path_egress")
            if has_env_dump and has_network_egress_sink:
                findings.append("shell_piped_env_egress")

    # Inspect pipe sequences per pipeline block for RCE patterns:
    # (1) remote download-and-execute: curl|sh, wget|bash
    # (2) local-data-into-executor: echo|sh, base64 -d|sh, cat script|python, etc.
    # (3) xargs acting as an executor of piped input: echo path | xargs rm
    # Uses BOTH raw (pre-unwrap) and unwrapped executables so that wrapper commands
    # (sudo, env, xargs) are retained in the source/sink lists even after unwrapping.
    for _logical_block, pipe_segments in parsed_blocks:
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
    for logical_block, _pipe_segments in parsed_blocks:
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
    sensitive_env_vars: List[str] = []
    sensitive_env_access = False

    for sub in subcommands:
        # Detect unquoted command-name backslash obfuscation (e.g. `r\m -r\f`, `c\url`) before unescaping
        raw_head_parts = sub.strip().split(None, 1)
        if raw_head_parts:
            raw_head = raw_head_parts[0]
            if (
                "\\" in raw_head
                and "/" not in raw_head
                and ":" not in raw_head
                and not raw_head.startswith(("'", '"', "."))
            ):
                unescaped_head = raw_head.replace("\\", "").lower()
                if unescaped_head in (
                    {"rm", "dd", "mkfs", "chmod", "chown", "eval", "exec", "sh", "bash", "sudo"}
                    | _SHELL_SECRET_EGRESS_COMMANDS
                ):
                    findings.append("shell_backslash_obfuscation")

        tokens, unbalanced_quotes = _tokenize_shell_segment_checked(sub)
        if not tokens:
            if unbalanced_quotes:
                unrecognized.append("unbalanced_shell_quotes")
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

        if unbalanced_quotes:
            unrecognized.append("unbalanced_shell_quotes")
        if wrappers_seen or had_inline_env:
            is_safe_analytical_xargs = bool(
                wrappers_seen == ["xargs"]
                and not had_inline_env
                and exe in _SAFE_ANALYTICAL_TOOLS
                and not any(a.startswith(("-I", "-P", "--max-procs", "--replace")) for a in args)
            )
            if not is_safe_analytical_xargs:
                unrecognized.append(wrappers_seen[0] if wrappers_seen else "env_assignment")
        ps_handled = False
        if exe in {"powershell", "pwsh"}:
            ps_handled = True
            ps_extra: Dict[str, Any] = {}
            ps_findings, ps_unrecognized, _ps_safe = _analyze_powershell_invocation(args, policy=policy, extra=ps_extra)
            findings.extend(ps_findings)
            unrecognized.extend(ps_unrecognized)
            for p_read in ps_extra.get("protected_path_reads") or ():
                protected_path_reads.append(p_read)
            if ps_extra.get("sensitive_env_access"):
                sensitive_env_access = True
            for e_var in ps_extra.get("sensitive_env_vars") or ():
                sensitive_env_vars.append(e_var)
        if not ps_handled and not _is_command_allowlisted(exe, args, policy=policy):
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

        is_git_output_write = exe == "git" and any(
            a in {"-o", "--output"} or a.startswith("--output=")
            for a in args_lower
        )

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
                    if exe in _SHELL_MUTATING_COMMANDS or _is_inplace_mutating_text_tool(exe, args) or is_git_output_write:
                        findings.append("shell_protected_path_mutation")
                    elif exe in _SHELL_SECRET_EGRESS_COMMANDS:
                        findings.append(f"shell_protected_path_egress:{exe}")
                    else:
                        protected_path_reads.append(cand)
                    break

        # Detect unquoted shell variable expansions targeting secret/token names or file paths
        for var_name in _extract_unquoted_env_vars(sub):
            if _SENSITIVE_ENV_VAR_RE.search(var_name):
                sensitive_env_vars.append(var_name)
                if exe in _SHELL_SECRET_EGRESS_COMMANDS:
                    findings.append("shell_sensitive_env_exposure")
                else:
                    sensitive_env_access = True
            elif exe in _FILE_READING_SHELL_COMMANDS:
                sensitive_env_access = True

        if exe == "rm":
            has_recursive = any(
                a == "--recursive" or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
                for a in args
            )
            targets_danger = any(
                _is_destructive_rm_target(a, policy=policy)
                for a in args
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
            has_nuke_perm = any(
                a in {"0", "00", "000", "0000", "a-rwx", "a-r", "a-x", "ugo-rwx", "a=0", "a="}
                or (a.startswith(("=", "a-", "ugo-")) and "rwx" in a)
                for a in args_lower
            )
            has_root_or_rec = any(
                a in {"/", "/*", "-r", "--recursive"}
                or (a.startswith("-") and not a.startswith("--") and "r" in a[1:])
                for a in args_lower
            )
            has_critical_target = any(
                _is_destructive_rm_target(a, policy=policy)
                for a in args
                if not a.startswith("-")
            )
            if has_777 and (has_root_or_rec or has_critical_target):
                findings.append("dangerous_chmod_777")
            elif has_nuke_perm and (has_root_or_rec or has_critical_target) and has_critical_target:
                findings.append("destructive_chmod")

        elif exe in {"rg", "ag"}:
            if any(
                a in {"--unrestricted", "--hidden", "-."}
                or a.startswith("--no-ignore")
                or (a.startswith("-") and not a.startswith("--") and "u" in a[1:])
                for a in args_lower
            ):
                protected_path_reads.append("hidden_file_search")

        elif exe in {"grep", "egrep", "fgrep"}:
            has_rec = any(
                a in {"--recursive", "--dereference-recursive"}
                or (a.startswith("-") and not a.startswith("--") and any(ch in a[1:] for ch in "rR"))
                for a in args
            )
            non_flag = [a for a in args if not a.startswith("-")]
            implicit_cwd = len(non_flag) <= 1
            targets_root_or_dot = any(
                t in {".", "./", "~", "~/"} or t.startswith("/") or ".." in t.replace("\\", "/").split("/")
                for t in non_flag[1:]
            )
            if (has_rec and (implicit_cwd or targets_root_or_dot)) or any(a.startswith("--include=.") for a in args_lower):
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

        elif re.fullmatch(r"python(?:\d+(?:\.\d+)*)?w?", exe):
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
                        inline_res = analyze_code_diff(
                            "<inline>.py",
                            inline_code,
                            policy=policy,
                            allow_suppression=bool(getattr(policy, "allow_inline_suppression", True)),
                        )
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

    if any(c in _SHELL_NETWORK_EGRESS_COMMANDS for c in parsed_cmds):
        if protected_path_reads:
            findings.append("shell_staged_protected_path_egress")
        if sensitive_env_access:
            findings.append("shell_staged_env_egress")

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
    if isinstance(sanitized.get("payload_preview"), str):
        sanitized["payload_preview"] = redact_secrets(sanitized["payload_preview"])[:200]
    if isinstance(sanitized.get("escalation_contract"), dict):
        contract_copy = dict(sanitized["escalation_contract"])
        if isinstance(contract_copy.get("risk_assessment"), str):
            contract_copy["risk_assessment"] = redact_secrets(contract_copy["risk_assessment"])
        sanitized["escalation_contract"] = contract_copy
    _DECISION_RING_BUFFER.append(sanitized)
    target = journal_path or _effective_default_journal_path()
    try:
        append_jsonl_bounded(target, json.dumps(sanitized, ensure_ascii=False))
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


_AGENT_CONTROL_PLANE_SEGMENTS: Tuple[str, ...] = (".claude", ".codex", ".cursor", ".qoder", ".qwen", ".prime", ".opencode", ".kilo", ".pi", ".omp", ".zcode", ".openhands", ".devin", ".clinerules", ".cline", ".minimax", ".kimi-code", ".kimi-plugin", ".mcp.json", ".vscode")
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


_VERIFIED_SHELL_TOOLS: Set[str] = {
    "run_command", "execute_shell", "run_tests", "pytest_runner", "git_commit",
}
_VERIFIED_EDIT_WRITE_TOOLS: Set[str] = {
    "write_file", "repo_write", "edit_file", "str_replace_editor", "apply_patch", "format_code",
}
_HARNESS_WEB_TOOLS: Set[str] = {"webfetch", "websearch"}


def _web_target_allowlisted(input_data: Any, cfg: PolicyConfig) -> bool:
    """True when an http/https webfetch target host is on the operator domain allowlist."""
    if not cfg.allowed_web_domains:
        return False
    tool_input = input_data.get("tool_input") if isinstance(input_data, dict) else None
    raw_url = ""
    if isinstance(tool_input, dict):
        raw_url = str(tool_input.get("url") or tool_input.get("query") or "").strip()
    if not raw_url:
        return False
    parsed = urlparse(raw_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    return any(
        host == d.strip().lower().rstrip(".") or host.endswith("." + d.strip().lower().rstrip("."))
        for d in cfg.allowed_web_domains
        if d.strip()
    )


def _mcp_tier_verdict(
    clean_tool: str,
    cfg: PolicyConfig,
    project_dir: Optional[Path] = None,
) -> Optional[Tuple[str, str, list, str, int]]:
    """Tiered verdict for `mcp__` tools; None falls through to the fail-closed ask.

    Read-only tools on operator-trusted/pinned servers allow structurally (in
    `dev` profile, or when `require_approval_for_external` is False); write-ish
    tools on trusted servers still escalate (the name heuristic is not proof of
    safety). Unpinned or drifted servers keep `layer1_uninspected_tool`.
    """
    try:
        from .mcp_tiers import mcp_tool_tier

        res = mcp_tool_tier(
            clean_tool,
            trusted_servers=tuple(cfg.trusted_mcp_servers or ()),
            project_dir=project_dir,
        )
    except Exception:
        return None
    server = str(res.get("server") or "")
    tool_seg = str(res.get("tool") or "")
    if res.get("tier") == "trusted_readonly":
        return ("allow", f"mcp:{server}:{tool_seg}", [clean_tool], "layer1_mcp_trusted_readonly", 1)
    if res.get("tier") == "trusted_writeish":
        return (
            "escalate",
            f"mcp_write_tool:{server}:{tool_seg}",
            ["allow", "deny"],
            "layer1_mcp_write_tool_escalate",
            1,
        )
    return None


_PROJECT_MODULE_CACHE: Dict[str, Tuple[Tuple[int, int], List[str]]] = {}
_IMPORT_TAINT_MAX_FILE_BYTES = 512 * 1024
_IMPORT_TAINT_MAX_MODULES = 8
_BENIGN_PEP562_LAZY_FINDINGS = {
    "dynamic_import:importlib.import_module",
    "obfuscated_subscript_builtin:<dynamic>",
}


def _resolve_module_candidates(module_name: str, base_dirs: List[Path]) -> List[Path]:
    """Map an import name (+ relative dots) onto concrete project file candidates."""
    rel = str(module_name or "").strip()
    prefix_dots = 0
    while rel.startswith("."):
        prefix_dots += 1
        rel = rel[1:]
    parts = [p for p in rel.strip().split(".") if p]
    if not parts:
        return []
    out: List[Path] = []
    for base in base_dirs:
        b = base
        for _ in range(max(0, prefix_dots - 1)):
            b = b.parent
        stem = b.joinpath(*parts)
        out.append(stem.with_name(stem.name + ".py"))
        out.append(stem / "__init__.py")
    return out


def _check_imported_project_modules(
    imported_modules: List[str],
    target_path: str,
    policy: Optional[PolicyConfig] = None,
    project_dir: Optional[Path] = None,
) -> List[str]:
    """Depth-1 cross-file taint: flag locally-resolved imports whose module
    source contains AST-vetoed constructs. Escalate-only by contract; results
    are cached per `(st_mtime_ns, st_size)` so the hot path stays cheap."""
    base_dirs: List[Path] = []
    tp = str(target_path or "").strip()
    if tp:
        try:
            tp_path = Path(tp)
            if not tp_path.is_absolute() and project_dir is not None:
                tp_path = Path(project_dir) / tp_path
            base_dirs.append(tp_path.parent.resolve())
        except OSError:
            base_dirs.append(Path(tp).parent)
    if project_dir is not None:
        try:
            base_dirs.append(Path(project_dir).resolve())
        except OSError:
            base_dirs.append(Path(project_dir))
    try:
        base_dirs.append(Path.cwd())
    except OSError:
        pass

    flagged: List[str] = []
    for mod in list(imported_modules)[:_IMPORT_TAINT_MAX_MODULES]:
        resolved_file = None
        for cand in _resolve_module_candidates(str(mod), base_dirs):
            try:
                if cand.is_file() and cand.stat().st_size <= _IMPORT_TAINT_MAX_FILE_BYTES:
                    resolved_file = cand
                    break
            except OSError:
                continue
        if resolved_file is None:
            continue
        key = str(resolved_file)
        if is_protected_path(key, policy):
            continue
        try:
            st = resolved_file.stat()
            sig = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            continue
        cached = _PROJECT_MODULE_CACHE.get(key)
        if cached is not None and cached[0] == sig:
            module_findings = cached[1]
        else:
            try:
                src = resolved_file.read_text(encoding="utf-8", errors="ignore")
                res = analyze_code_diff(key, src, original_source=src)
                module_findings = [f for f in (res.get("findings") or []) if f]
                if (
                    module_findings
                    and resolved_file.name == "__init__.py"
                    and "def __getattr__(" in src
                ):
                    module_findings = [
                        f for f in module_findings if f not in _BENIGN_PEP562_LAZY_FINDINGS
                    ]
            except Exception:
                module_findings = []
            _PROJECT_MODULE_CACHE[key] = (sig, module_findings)
        if module_findings and str(mod) not in flagged:
            flagged.append(str(mod))
    return flagged


_TOOL_ALIAS_GROUPS: Tuple[frozenset[str], ...] = (
    frozenset({"bash", "shell", "shell_command", "exec_command", "execute_shell", "run_command", "terminal"}),
    frozenset({"edit", "edit_file", "str_replace_editor", "multiedit", "notebookedit", "patch", "apply_patch"}),
    frozenset({"write", "write_file", "repo_write", "fwrite"}),
    frozenset({"read", "read_file", "repo_read", "view_file"}),
    frozenset({"grep", "grep_search", "search_code"}),
    frozenset({"glob", "list_dir"}),
)


def _expand_tool_aliases(clean_tool: str, raw_tool: Optional[str] = None) -> frozenset[str]:
    candidates = {tok for tok in (clean_tool, str(raw_tool or "").strip().lower()) if tok}
    expanded = set(candidates)
    for grp in _TOOL_ALIAS_GROUPS:
        if candidates & grp:
            expanded.update(grp)
    return frozenset(expanded)


def _evaluate_custom_rules(
    cfg: PolicyConfig,
    clean_tool: str,
    target_path: Optional[str],
    cmd_str: Optional[str],
    action_filter: Optional[str] = None,
    vitality_ledger: Any = None,
    now: Optional[float] = None,
    return_rule: bool = False,
    raw_tool: Optional[str] = None,
) -> Optional[Any]:
    """Evaluate AgentSpec-style declarative `PolicyConfig.custom_rules`."""
    rules = getattr(cfg, "custom_rules", ()) or ()
    if not rules:
        return None
    tool_candidates = _expand_tool_aliases(clean_tool, raw_tool)
    norm_path = str(target_path or "").replace("\\", "/").strip()
    norm_cmd = str(cmd_str or "")
    for r in rules:
        if not isinstance(r, dict):
            continue
        action = str(r.get("action") or "").strip().lower()
        if action not in {"deny", "escalate", "allow"}:
            continue
        if action_filter and action != action_filter:
            continue
        tool_pat = r.get("tool")
        if tool_pat is not None:
            pat_str = str(tool_pat)
            if not tool_candidates or not any(re.fullmatch(pat_str, cand, re.IGNORECASE) for cand in tool_candidates):
                continue
        path_glob = r.get("path_glob")
        if path_glob is not None:
            pg = str(path_glob).replace("\\", "/").lower()
            if not norm_path or not fnmatch.fnmatch(norm_path.lower(), pg):
                continue
        cmd_pat = r.get("cmd_regex")
        if cmd_pat is not None:
            if not norm_cmd or not re.search(str(cmd_pat), norm_cmd):
                continue
        rule_id = str(r.get("id") or "custom_rule").strip()
        label = str(r.get("reason") or f"custom_policy_rule:{rule_id}").strip()
        if action == "allow" and vitality_ledger is not None:
            try:
                v_check = vitality_ledger.check_rule(rule_id, now=now)
                if v_check.get("known") and v_check.get("stale"):
                    res = ("escalate", f"stale_custom_rule_reconfirm:{rule_id}", f"layer0_stale_rule_reconfirm:{rule_id}", 0)
                    return (res, r, v_check) if return_rule else res
                res = (action, label, f"layer0_custom_policy_rule:{rule_id}", 0)
                return (res, r, v_check) if return_rule else res
            except Exception:
                res = ("escalate", f"stale_custom_rule_reconfirm:{rule_id}", f"layer0_stale_rule_reconfirm:{rule_id}", 0)
                return (res, r, None) if return_rule else res
        res = (action, label, f"layer0_custom_policy_rule:{rule_id}", 0)
        return (res, r, None) if return_rule else res
    return None


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
        raw_tool = str(input_data.get("raw_tool") or "").strip().lower()
        target_path = input_data.get("path") or input_data.get("file_path")
        diff_text = None
        for _dk in ("diff", "payload_diff", "new_string", "content"):
            if _dk in input_data and input_data[_dk] is not None:
                diff_text = str(input_data[_dk])
                break
        cmd_str = input_data.get("cmd") or input_data.get("command") or (target_path if tool_name in ("run_command", "execute_shell") else None)
        query_text = str(input_data.get("query") or input_data.get("task_prompt") or "")
        original_source = input_data.get("file_content") or ctx.get("file_content")
        edit_ops = input_data.get("edit_ops")
        if edit_ops is None and isinstance(input_data.get("old_string"), str) and input_data.get("old_string") and "new_string" in input_data:
            edit_ops = [{
                "old_string": input_data["old_string"],
                "new_string": str(input_data.get("new_string") or ""),
                "replace_all": bool(input_data.get("replace_all", False)),
            }]
    else:
        tool_name = str(ctx.get("tool") or "")
        raw_tool = str(ctx.get("raw_tool") or "").strip().lower()
        target_path = ctx.get("path")
        diff_text = None
        for _dk in ("diff", "new_string", "content"):
            if _dk in ctx and ctx[_dk] is not None:
                diff_text = str(ctx[_dk])
                break
        cmd_str = ctx.get("cmd")
        query_text = str(input_data or "")
        original_source = ctx.get("file_content")
        edit_ops = None

    shadow_mode = bool(ctx.get("shadow_mode", False))
    available_tools = ctx.get("available_tools")
    journal_path = Path(ctx["journal_path"]) if ctx.get("journal_path") else None
    should_log = bool(ctx.get("log_decisions", journal_path is not None))

    canonical_payload = json.dumps(
        input_data
        if isinstance(input_data, dict)
        else {"tool": tool_name, "path": target_path, "cmd": cmd_str, "diff": diff_text, "query": query_text},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    full_sha256 = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    input_sha256 = full_sha256[:16]

    clean_tool = str(tool_name or "").strip().lower()
    inferred_risk = TOOL_RISK_CLASSES.get(
        clean_tool,
        "EXTERNAL_ACTION" if (cmd_str or clean_tool) else ("REVERSIBLE_WRITE" if diff_text is not None else "READ_ONLY"),
    )

    diff_structural_context: List[Dict[str, Any]] = []
    diff_suppressed_findings: List[Dict[str, Any]] = []
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
        matched_rule_prov: Optional[Dict[str, Any]] = None
        if decision == "escalate" and reason_code in {
            "layer1_unrecognized_shell_command_escalate",
            "layer1_uninspected_tool",
            "layer0_external_requires_confirmation",
            "layer3_default_escalation",
        }:
            custom_allow = _evaluate_custom_rules(
                cfg,
                clean_tool,
                target_path,
                cmd_str,
                action_filter="allow",
                vitality_ledger=ctx.get("vitality_ledger"),
                now=ctx.get("now"),
                return_rule=True,
                raw_tool=raw_tool,
            )
            if custom_allow is not None:
                (c_act, c_lbl, c_code, c_layer), c_rule, c_vcheck = custom_allow
                decision = c_act
                label = c_lbl
                reason_code = c_code
                layer = c_layer
                if decision == "allow":
                    pred_set = [clean_tool or label]
                prov = c_rule.get("provenance") if isinstance(c_rule.get("provenance"), dict) else None
                if prov is not None or c_vcheck is not None:
                    matched_rule_prov = dict(prov or {})
                    if c_vcheck is not None and c_vcheck.get("known"):
                        matched_rule_prov["vitality"] = c_vcheck.get("vitality")

        latency_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 4)
        recent = list(_DECISION_RING_BUFFER)[-12:]
        if len(recent) < 9 and (journal_path is not None or should_log):
            j_target = journal_path or _effective_default_journal_path()
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

        # Context preview snippet for human review in escalations / audits (redact BEFORE truncating)
        payload_preview: Optional[str] = None
        if cmd_str:
            payload_preview = redact_secrets(str(cmd_str))[:200]
        elif target_path:
            payload_preview = redact_secrets(str(target_path))[:200]
        elif diff_text:
            payload_preview = redact_secrets(str(diff_text))[:200]
        elif query_text:
            payload_preview = redact_secrets(str(query_text))[:200]

        out: Dict[str, Any] = {
            "decision": decision,
            "decision_kind": resolved_kind,
            "tool": clean_tool or (str(tool_name).strip() if tool_name else None),
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
            "input_sha256_full": full_sha256,
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
            elif reason_code in (
                "layer1_shell_redirection_escalate",
                "layer1_sensitive_env_access_escalate",
                "layer1_protected_path_shell_read",
                "layer1_protected_path_shell_read_escalate",
            ):
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
            elif reason_code == "layer1_ast_suppressed_escalate":
                risk_lvl = "MEDIUM"
                risk_assess = f"Inline-suppressed AST finding ({label}) requires human review in strict profile."
            elif reason_code.startswith(("strict_ast_", "layer1_strict_ast_")):
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
            if cmd_str and hasattr(t_sess, "record_external_shell"):
                t_sess.record_external_shell(cmd_str)
            if t_prot or t_sess._is_external_tool(t_tool):
                t_sess.record_pre_tool(t_tool, t_path, t_prot)
        if diff_structural_context:
            out["structural_context"] = diff_structural_context
        if diff_suppressed_findings:
            out["suppressed_findings"] = list(diff_suppressed_findings)
        if matched_rule_prov is not None:
            out["rule_provenance"] = matched_rule_prov
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

        l0 = evaluate_layer0_policy(tool_name, target_path, cfg, input_data=input_data)
        if l0["decision"] == "deny":
            # Break-glass: operator-held env var unlocks policy-plane maintenance only
            # (hook config, skill files, and the gate's own source tree). The agent
            # cannot set env vars in the hook process; credential stores (.env, keys,
            # secrets) are never eligible.
            _bp = str(l0.get("protected_path") or "")
            if not (
                os.getenv("AETHERIS_ALLOW_POLICY_EDIT") == "1"
                and (
                    _is_agent_control_plane_path(_bp)
                    or _is_live_package_path(_bp)
                )
            ):
                return _finish("deny", "blocked_policy", ["deny"], l0["reason_code"], 0)

        custom_deny = _evaluate_custom_rules(
            cfg,
            clean_tool,
            target_path,
            cmd_str,
            action_filter="deny",
            raw_tool=raw_tool,
        )
        if custom_deny is not None:
            c_act, c_lbl, c_code, c_layer = custom_deny
            return _finish(c_act, c_lbl, ["deny"], c_code, c_layer)

        # Cross-turn SessionSecurityState taint guard runs BEFORE l0 escalate so
        # corrupted state/lethal trifecta fail closed and protected reads record taint.
        sess_obj = ctx.get("session_state")
        sess_path = ctx.get("session_state_path")
        if sess_obj is not None or sess_path:
            from .mcp_security import SessionSecurityState

            sess = sess_obj if isinstance(sess_obj, SessionSecurityState) else SessionSecurityState(Path(str(sess_path)))
            if sess.corrupted:
                return _finish("deny", "session_state_corrupted", ["deny"], "layer0_session_state_corrupted", 0)
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
                    "webfetch",
                    "web_search",
                    "websearch",
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

        if l0["decision"] == "escalate":
            return _finish("escalate", "requires_confirmation", ["escalate", "deny"], l0["reason_code"], 0)

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
                profile_low = str(cfg.profile or "strict").strip().lower()
                if profile_low == "guardrails":
                    high_risk_unrec = [
                        cmd_tok
                        for cmd_tok in unrec
                        if (
                            str(cmd_tok).lower() in _GUARDRAILS_ESCALATE_COMMANDS
                            or str(cmd_tok).lower().startswith(("mkfs.", "powershell"))
                        )
                    ]
                else:
                    high_risk_unrec = unrec
                if high_risk_unrec or profile_low != "guardrails":
                    return _finish(
                        "escalate",
                        (high_risk_unrec[0] if high_risk_unrec else (unrec[0] if unrec else "unrecognized_shell_command")),
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
            diff_res = analyze_code_diff(
                target_path,
                diff_text,
                original_source=original_source,
                edit_ops=edit_ops,
                policy=cfg,
                allow_suppression=bool(getattr(cfg, "allow_inline_suppression", True)),
            )
            diff_structural_context.extend(diff_res.get("structural_context", []))
            if diff_res.get("suppressed_findings"):
                diff_suppressed_findings.extend(diff_res["suppressed_findings"])
            if not diff_res["security_clean"]:
                if (
                    str(cfg.profile or "strict").strip().lower() in {"dev", "guardrails"}
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
            if diff_suppressed_findings and str(cfg.profile or "strict").strip().lower() not in {"dev", "guardrails"}:
                return _finish(
                    "escalate",
                    f"suppressed:{diff_suppressed_findings[0]['rule']}",
                    ["allow", "deny"],
                    "layer1_ast_suppressed_escalate",
                    1,
                )

            # B4: depth-1 cross-file import taint — a clean diff that pulls in a
            # local module containing vetoed constructs escalates (never denies).
            if (
                cfg.import_taint_enabled
                and diff_res.get("language") == "python"
                and diff_res["syntax_valid"]
                and not diff_res["findings"]
            ):
                proj_dir = Path(str(ctx["project_dir"])) if ctx.get("project_dir") else None
                tainted = _check_imported_project_modules(
                    diff_res.get("imported_modules") or [],
                    str(target_path or ""),
                    policy=cfg,
                    project_dir=proj_dir,
                )
                if tainted:
                    return _finish("escalate", tainted[0], ["allow", "deny"], "layer1_imports_flagged_module", 1)

        custom_esc = _evaluate_custom_rules(cfg, clean_tool, target_path, cmd_str, action_filter="escalate")
        if custom_esc is not None:
            c_act, c_lbl, c_code, c_layer = custom_esc
            return _finish(c_act, c_lbl, ["allow", "deny"], c_code, c_layer)

        # Named executable/unknown tools NEVER fall through to Layer 2 routing advice.
        # Allow at Layer 1 ONLY for verified shell/edit tools with inspected payloads or explicit READ_ONLY tools.
        if tool_name:
            is_verified = (
                (clean_tool in _VERIFIED_SHELL_TOOLS and bool(cmd_str))
                or (clean_tool in _VERIFIED_EDIT_WRITE_TOOLS and diff_text is not None)
                or (TOOL_RISK_CLASSES.get(clean_tool) == "READ_ONLY")
                or (
                    clean_tool in _HARNESS_WEB_TOOLS
                    and (
                        not cfg.require_approval_for_external
                        or _web_target_allowlisted(input_data, cfg)
                    )
                )
            )
            if is_verified:
                ok_code = "layer1_hunk_only_checked" if hunk_only_checked else "layer1_structural_verified"
                return _finish("allow", tool_name, [tool_name], ok_code, 1)
            if clean_tool.startswith("mcp__"):
                proj_dir = Path(str(ctx["project_dir"])) if ctx.get("project_dir") else None
                tier_v = _mcp_tier_verdict(clean_tool, cfg, project_dir=proj_dir)
                if tier_v is not None:
                    return _finish(*tier_v)
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
        target = Path(journal_path) if journal_path is not None else _effective_default_journal_path()
        if journal_path is None and _DECISION_RING_BUFFER:
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
