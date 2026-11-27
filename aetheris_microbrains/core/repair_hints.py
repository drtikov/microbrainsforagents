"""Deterministic repair hints and pure-stdlib Clopper-Pearson threshold calibration.

Provides:
  - `generate_repair_hint(payload, verdict, *, allow_rewrite=False)`:
      Inspects denied/escalated tool-call payloads and verdicts to emit actionable
      repair guidance (Python syntax line/col + snippet, safe `git push --force-with-lease`
      rewrites, structured-tool alternatives for shell commands, `$SECRET:<NAME>`
      out-of-band credential handles, and protected-path explanations).
  - `calibrate_thresholds_from_journal(entries, *, current_threshold=0.5, target_fdr=0.05)`:
      Computes empirical allow/block/escalate rates and exact one-sided Clopper-Pearson
      upper confidence bounds (pure-stdlib `math` only, zero NumPy) to identify noisy
      escalated rules and emit a calibrated threshold recommendation.
"""

from __future__ import annotations

import ast
import math
import posixpath
import re
import shlex
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .secrets import (
    _SECRET_PATTERNS,
    _SENSITIVE_KEY_RE,
    _is_placeholder_value,
    detect_encoded_secrets,
    redact_secrets,
)

_SHELL_TOOLS = frozenset({
    "bash",
    "shell",
    "terminal",
    "run_command",
    "execute_shell",
    "powershell",
    "cmd",
})

_READ_SHELL_EXES = frozenset({
    "cat",
    "head",
    "tail",
    "nl",
    "less",
    "more",
    "get-content",
    "gc",
    "type",
})

_GREP_SHELL_EXES = frozenset({
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ripgrep",
    "ag",
    "ack",
    "select-string",
    "sls",
})

_GLOB_SHELL_EXES = frozenset({
    "find",
    "fd",
    "get-childitem",
    "gci",
    "dir",
})

_SHELL_ENV_WRAPPERS = frozenset({"env", "command", "sudo", "nohup", "time"})

_CONTROL_PLANE_DIRS = frozenset({
    ".claude",
    ".qwen",
    ".prime",
    ".codex",
    ".cursor",
    ".opencode",
    ".kilo",
    ".cline",
    ".clinerules",
    ".devin",
    ".openhands",
    ".pi",
    ".omp",
    ".zcode",
    ".minimax",
})

_ENV_TEMPLATE_BASENAMES = frozenset({
    ".env.example",
    ".env.template",
    ".env.sample",
    ".env.dist",
})

_DANGEROUS_SHELL_RES: Tuple[Tuple[str, re.Pattern[str], str], ...] = (
    (
        "recursive_root_delete",
        re.compile(r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+(?:/\s*$|/\*|~/?\b|\$HOME\b)"),
        "Recursive deletion targeting root or home directory is blocked. Scope the path to a specific project subdirectory.",
    ),
    (
        "pipe_to_shell",
        re.compile(r"\b(?:curl|wget)\b[^|;\n]*\|\s*(?:ba|z|da|k)?sh\b"),
        "Piping remote content directly into a shell interpreter is unsafe. Download the script, inspect it, and verify checksums before execution.",
    ),
    (
        "raw_device_write",
        re.compile(r"\b(?:dd\b[^\n]*\bof=/dev/|mkfs(?:\.[a-z0-9]+)?\b)"),
        "Direct block-device formatting or raw writes are blocked. Verify target devices and perform destructive disk operations manually.",
    ),
)

_SECRET_ENV_DEFAULT: Dict[str, str] = {
    "private_key_pem": "PRIVATE_KEY",
    "openai_project_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "api_key": "API_KEY",
    "github_pat": "GITHUB_TOKEN",
    "github_token": "GITHUB_TOKEN",
    "gitlab_token": "GITLAB_TOKEN",
    "aws_access_key": "AWS_ACCESS_KEY_ID",
    "google_api_key": "GOOGLE_API_KEY",
    "google_oauth_token": "GOOGLE_OAUTH_TOKEN",
    "slack_webhook": "SLACK_WEBHOOK_URL",
    "slack_token": "SLACK_BOT_TOKEN",
    "huggingface_token": "HF_TOKEN",
    "npm_token": "NPM_TOKEN",
    "pypi_token": "PYPI_TOKEN",
    "stripe_key": "STRIPE_API_KEY",
    "xai_api_key": "XAI_API_KEY",
    "groq_api_key": "GROQ_API_KEY",
    "supabase_token": "SUPABASE_ACCESS_TOKEN",
    "vault_token": "VAULT_TOKEN",
    "linear_api_key": "LINEAR_API_KEY",
    "notion_token": "NOTION_TOKEN",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "discord_bot_token": "DISCORD_BOT_TOKEN",
    "sendgrid_key": "SENDGRID_API_KEY",
    "databricks_token": "DATABRICKS_TOKEN",
    "twilio_api_key": "TWILIO_API_KEY",
    "gcp_service_account": "GCP_SERVICE_ACCOUNT_KEY",
    "db_connection_uri": "DATABASE_URL",
    "url_basic_auth": "SERVICE_AUTH_URL",
    "jwt_token": "AUTH_JWT",
    "bearer_token": "API_BEARER_TOKEN",
    "encoded_secret": "ENCODED_SECRET",
    "sensitive_key_value": "SECRET_VALUE",
}

_ASSIGN_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]{1,63})\s*[:=]")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_HUNK_HEADER_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")


def _flatten_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge nested `tool_input` / `input` / `arguments` into a flat lookup dict."""
    merged: Dict[str, Any] = {}
    for nested_key in ("arguments", "input", "tool_input"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            for k, v in nested.items():
                if k not in merged and v is not None:
                    merged[k] = v
    for k, v in payload.items():
        if v is not None:
            merged[k] = v
    return merged


def _extract_code_text(flat: Mapping[str, Any]) -> Tuple[str, bool]:
    """Extract proposed code/diff text and whether it came from a diff field."""
    for key, is_diff in (
        ("content", False),
        ("new_string", False),
        ("code", False),
        ("source", False),
        ("diff", True),
        ("payload_diff", True),
    ):
        val = flat.get(key)
        if isinstance(val, str) and val != "":
            return val, is_diff
    return "", False


def _is_unified_diff(text: str) -> bool:
    """Return True if `text` looks like a unified diff rather than raw file source."""
    stripped = text.lstrip()
    if stripped.startswith(("diff --git ", "--- ", "+++ ", "@@ ")):
        return True
    return any(_HUNK_HEADER_RE.match(ln) for ln in text.splitlines())


def _extract_diff_source_with_linemap(diff_text: str) -> Tuple[str, List[int]]:
    """Reconstruct added/context lines from a unified diff with 1-based target line numbers."""
    out_lines: List[str] = []
    line_map: List[int] = []
    cur_new_line = 1
    in_hunk = False

    for raw in diff_text.splitlines():
        m = _HUNK_HEADER_RE.match(raw)
        if m:
            cur_new_line = int(m.group(1))
            in_hunk = True
            continue
        if raw.startswith(("diff --git ", "index ", "--- ", "+++ ")):
            continue
        if not in_hunk:
            continue
        if raw.startswith("+"):
            out_lines.append(raw[1:])
            line_map.append(cur_new_line)
            cur_new_line += 1
        elif raw.startswith(" "):
            out_lines.append(raw[1:])
            line_map.append(cur_new_line)
            cur_new_line += 1
        elif raw.startswith("-"):
            continue
        elif raw == r"\ No newline at end of file":
            continue

    return "\n".join(out_lines), line_map


def _inspect_python_syntax(code_text: str) -> Optional[Dict[str, Any]]:
    """Return exact 1-based line, col, snippet, and message if `code_text` fails `ast.parse`."""
    if not code_text.strip():
        return None

    line_map: Optional[List[int]] = None
    parse_target = code_text
    if _is_unified_diff(code_text):
        reconstructed, mapped = _extract_diff_source_with_linemap(code_text)
        if reconstructed.strip():
            parse_target = reconstructed
            line_map = mapped

    try:
        ast.parse(parse_target)
        return None
    except SyntaxError as exc:
        raw_lineno = int(exc.lineno or 1)
        col = int(exc.offset) if exc.offset is not None else None
        lines = parse_target.splitlines()
        snippet = ""
        if 1 <= raw_lineno <= len(lines):
            snippet = lines[raw_lineno - 1].rstrip("\r\n")
        elif exc.text:
            snippet = str(exc.text).rstrip("\r\n")
        mapped_lineno = raw_lineno
        if line_map and 1 <= raw_lineno <= len(line_map):
            mapped_lineno = line_map[raw_lineno - 1]
        return {
            "line": mapped_lineno,
            "col": col,
            "snippet": snippet,
            "msg": str(exc.msg or "invalid syntax"),
        }


def _classify_protected_path(path_str: Optional[str]) -> Optional[str]:
    """Classify protected path category (`env`, `control_plane`, `git_hooks`, `credentials`)."""
    if not path_str:
        return None
    raw = str(path_str).strip()
    if not raw or "\x00" in raw:
        return "credentials"
    norm = posixpath.normpath(raw.replace("\\", "/")).lower()
    segments = [seg for seg in norm.split("/") if seg]
    basename = segments[-1] if segments else ""

    if basename == ".env" or (
        basename.startswith(".env.") and basename not in _ENV_TEMPLATE_BASENAMES
    ):
        return "env"
    if ".git" in segments:
        git_idx = segments.index(".git")
        if git_idx + 1 < len(segments) and segments[git_idx + 1] == "hooks":
            return "git_hooks"
    if any(seg in _CONTROL_PLANE_DIRS for seg in segments) or basename == ".mcp.json":
        return "control_plane"
    if any(seg in {".ssh", ".aws"} for seg in segments) or basename in {
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "credentials",
    }:
        return "credentials"
    return None


def _protected_path_hint(path_str: Optional[str], kind: Optional[str]) -> str:
    """Build an explanation and safe alternative for a blocked protected path edit."""
    target = path_str or "<protected-path>"
    if kind == "env":
        return (
            f"Direct edits to `{target}` are blocked because `.env` files hold live runtime credentials. "
            "Update `.env.example` or `.env.template` with placeholder keys and configure real values "
            "out-of-band via environment variables or `$SECRET:<NAME>` handles."
        )
    if kind == "git_hooks":
        return (
            f"Editing `{target}` is blocked because `.git/hooks/` scripts execute automatically during git "
            "operations. Propose changes in `.pre-commit-config.yaml` or a documented setup script for human review."
        )
    if kind == "control_plane":
        return (
            f"Modifying `{target}` is blocked to protect agent control-plane configuration from autonomous "
            "self-modification. Ask the operator to apply configuration changes or update a project template instead."
        )
    return (
        f"Editing protected path `{target}` is blocked by policy (credential or integrity boundary). "
        "Keep secrets out of repository files and use environment/template configuration instead."
    )


def _infer_secret_var_name(line_text: str, secret_type: str) -> str:
    """Infer a clean uppercase environment variable / `$SECRET:<NAME>` identifier."""
    m = _ASSIGN_NAME_RE.search(line_text)
    if m:
        cand = re.sub(r"[^A-Za-z0-9_]+", "_", m.group(1)).strip("_").upper()
        if cand and not cand.isdigit():
            return cand
    return _SECRET_ENV_DEFAULT.get(secret_type, "API_SECRET")


def _detect_secret_location(text: str) -> Optional[Dict[str, Any]]:
    """Locate the first raw secret in `text` and return type, 1-based line/col, and var name."""
    if not text:
        return None
    lines = text.splitlines() or [text]
    for lineno, line_text in enumerate(lines, start=1):
        if "$SECRET:" in line_text or "[REDACTED" in line_text:
            continue
        for secret_type, pat, _repl in _SECRET_PATTERNS:
            m = pat.search(line_text)
            if m:
                return {
                    "secret_type": secret_type,
                    "line": lineno,
                    "col": m.start() + 1,
                    "var_name": _infer_secret_var_name(line_text, secret_type),
                }
        m_sens = _SENSITIVE_KEY_RE.search(line_text)
        if m_sens:
            val = m_sens.group(5)
            key = m_sens.group(2)
            if not _is_placeholder_value(val) and not (
                key.lower() == "pwd" and val.startswith(("/", "~/", "./", "../"))
            ):
                var_name = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_").upper() or "SECRET_VALUE"
                return {
                    "secret_type": "sensitive_key_value",
                    "line": lineno,
                    "col": m_sens.start(5) + 1,
                    "var_name": var_name,
                }

    enc_hits = detect_encoded_secrets(text)
    if enc_hits:
        return {
            "secret_type": enc_hits[0],
            "line": 1,
            "col": 1,
            "var_name": _SECRET_ENV_DEFAULT.get(enc_hits[0], "ENCODED_SECRET"),
        }
    return None


def _split_shell_segments(command: str) -> List[str]:
    """Split a shell command on top-level `&&`, `||`, `;`, `|` outside quotes."""
    segments: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue
        if command.startswith(("&&", "||"), i):
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf = []
            i += 2
            continue
        if ch in (";", "|"):
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _tokenize_segment(segment: str) -> List[str]:
    """Best-effort shlex tokenization with whitespace fallback."""
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _extract_git_push_subcommand(tokens: List[str]) -> Optional[int]:
    """If `tokens` invokes `git ... push`, return the index of `push`; else `None`."""
    if not tokens:
        return None
    idx = 0
    while idx < len(tokens) and (
        _ENV_ASSIGN_RE.match(tokens[idx]) or tokens[idx].lower() in _SHELL_ENV_WRAPPERS
    ):
        idx += 1
    if idx >= len(tokens):
        return None
    exe = tokens[idx].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if exe.endswith(".exe"):
        exe = exe[:-4]
    if exe != "git":
        return None
    i = idx + 1
    git_opts_with_arg = {"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix"}
    while i < len(tokens):
        tok = tokens[i]
        if tok in git_opts_with_arg:
            i += 2
            continue
        if tok.startswith(("--git-dir=", "--work-tree=", "--namespace=", "-c", "-C")):
            i += 1
            continue
        if tok.startswith("-"):
            i += 1
            continue
        return i if tok == "push" else None
    return None


def _is_force_flag(tok: str) -> bool:
    """True for `--force`, `--force=true`, `-f`, or bundled short flags containing `f`."""
    if tok == "--force-with-lease" or tok.startswith("--force-with-lease="):
        return False
    if tok == "--force" or tok.lower().startswith("--force="):
        return True
    if tok.startswith("-") and not tok.startswith("--") and len(tok) >= 2:
        return "f" in tok[1:]
    return False


def _rewrite_git_force_push_segment(segment: str) -> Optional[str]:
    """Rewrite `git push --force`/`-f` to `--force-with-lease` if unsafe force flag present."""
    tokens = _tokenize_segment(segment)
    push_idx = _extract_git_push_subcommand(tokens)
    if push_idx is None:
        return None

    has_unsafe_force = False
    new_tokens: List[str] = list(tokens[: push_idx + 1])
    for tok in tokens[push_idx + 1 :]:
        if tok == "--force-with-lease" or tok.startswith("--force-with-lease="):
            new_tokens.append(tok)
        elif tok == "--force" or tok.lower().startswith("--force="):
            has_unsafe_force = True
            if "--force-with-lease" not in new_tokens:
                new_tokens.append("--force-with-lease")
        elif tok == "-f":
            has_unsafe_force = True
            if "--force-with-lease" not in new_tokens:
                new_tokens.append("--force-with-lease")
        elif tok.startswith("-") and not tok.startswith("--") and "f" in tok[1:]:
            has_unsafe_force = True
            remaining = tok[1:].replace("f", "")
            if remaining:
                new_tokens.append("-" + remaining)
            if "--force-with-lease" not in new_tokens:
                new_tokens.append("--force-with-lease")
        else:
            new_tokens.append(tok)

    if not has_unsafe_force:
        return None
    return shlex.join(new_tokens)


def _detect_git_force_push(command: str) -> Optional[str]:
    """Return safe `--force-with-lease` rewritten command if `command` contains `git push --force`/`-f`."""
    if not command or not command.strip():
        return None
    segments = _split_shell_segments(command)
    if not segments:
        return None

    any_rewritten = False
    rebuilt_parts: List[str] = []
    # Reassemble using original operators where simple, else ` && `
    pos = 0
    for idx, seg in enumerate(segments):
        rewritten = _rewrite_git_force_push_segment(seg)
        if rewritten is not None:
            any_rewritten = True
        chosen = rewritten if rewritten is not None else seg
        if idx > 0:
            # Find operator between previous segment and current segment
            prev_end = command.find(segments[idx - 1], pos)
            if prev_end != -1:
                after_prev = prev_end + len(segments[idx - 1])
                cur_start = command.find(seg, after_prev)
                if cur_start != -1:
                    op = command[after_prev:cur_start]
                    pos = cur_start
                    rebuilt_parts.append(op + chosen)
                    continue
            rebuilt_parts.append(" && " + chosen)
        else:
            rebuilt_parts.append(chosen)
            found = command.find(seg)
            if found != -1:
                pos = found

    return "".join(rebuilt_parts) if any_rewritten else None


def _detect_shell_over_structured_tool(command: str) -> Optional[Tuple[str, str]]:
    """Return `(exe, suggested_tool)` if `command` uses shell read/search/glob instead of structured tool."""
    if not command or not command.strip():
        return None
    # Skip redirections that write files (e.g., `cat a.txt > b.txt` or heredocs)
    if ">" in command or "<<" in command:
        return None
    segments = _split_shell_segments(command)
    if not segments:
        return None
    tokens = _tokenize_segment(segments[0])
    idx = 0
    while idx < len(tokens) and (
        _ENV_ASSIGN_RE.match(tokens[idx]) or tokens[idx].lower() in _SHELL_ENV_WRAPPERS
    ):
        idx += 1
    if idx >= len(tokens):
        return None
    exe = tokens[idx].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if exe.endswith(".exe"):
        exe = exe[:-4]

    if exe in _READ_SHELL_EXES or (exe == "sed" and "-n" in tokens[idx + 1 :]):
        return exe, "Read"
    if exe in _GREP_SHELL_EXES:
        return exe, "Grep"
    if exe in _GLOB_SHELL_EXES:
        return exe, "Glob"
    if exe == "ls" and any(
        t.startswith("-") and "R" in t[1:] or "*" in t or "?" in t for t in tokens[idx + 1 :]
    ):
        return exe, "Glob"
    return None


def generate_repair_hint(
    payload: Mapping[str, Any],
    verdict: Mapping[str, Any],
    *,
    allow_rewrite: bool = False,
) -> Optional[Dict[str, Any]]:
    """Inspect `(payload, verdict)` and emit a structured repair hint for deny/escalate verdicts.

    Returns `None` when `verdict.get("decision") == "allow"` (or not in `("deny", "escalate")`).
    """
    if not isinstance(verdict, Mapping):
        return None
    decision = str(verdict.get("decision") or "").strip().lower()
    if decision == "allow" or decision not in ("deny", "escalate"):
        return None

    flat = _flatten_payload(payload) if isinstance(payload, Mapping) else {}
    tool_name = str(
        flat.get("tool") or flat.get("tool_name") or flat.get("action") or flat.get("raw_tool") or ""
    ).strip()
    target_path = flat.get("path") or flat.get("file_path") or flat.get("target_path") or flat.get("filename")
    target_path_str = str(target_path) if target_path is not None else None
    code_text, _is_diff = _extract_code_text(flat)
    command = flat.get("command") or flat.get("cmd")
    if command is None and tool_name.lower() in _SHELL_TOOLS and target_path_str:
        command = target_path_str
    command_str = str(command) if command is not None else ""

    reason_code = str(verdict.get("reason_code") or "").strip()
    rc_low = reason_code.lower()
    contract = verdict.get("escalation_contract") if isinstance(verdict.get("escalation_contract"), Mapping) else {}
    verdict_action = str(verdict.get("suggested_action") or contract.get("suggested_action") or "").strip()

    # 1. Python syntax errors
    is_py_target = bool(target_path_str and target_path_str.lower().endswith(".py"))
    if "syntax" in rc_low or (is_py_target and code_text):
        syn = _inspect_python_syntax(code_text)
        if syn is not None:
            line = syn["line"]
            col = syn["col"]
            snippet = syn["snippet"]
            col_text = f", col {col}" if col is not None else ""
            snippet_note = f" Offending snippet: `{snippet.strip()}`." if snippet.strip() else ""
            return {
                "category": "python_syntax_error",
                "hint": (
                    f"Python SyntaxError at line {line}{col_text}: {syn['msg']}.{snippet_note} "
                    "Fix the syntax error before writing or applying this patch."
                ),
                "suggested_tool": None,
                "line": line,
                "col": col,
                "snippet": snippet,
                "updated_input": None,
                "suggested_action": "fix_python_syntax",
                "reason_code": reason_code,
            }
        if "syntax" in rc_low:
            v_line = verdict.get("line") if isinstance(verdict.get("line"), int) else None
            v_col = verdict.get("col") if isinstance(verdict.get("col"), int) else None
            return {
                "category": "python_syntax_error",
                "hint": (
                    "Code failed syntax validation"
                    + (f" at line {v_line}" if v_line is not None else "")
                    + (f", col {v_col}" if v_col is not None else "")
                    + ". Parse and validate the full target file with `ast.parse` before retrying."
                ),
                "suggested_tool": None,
                "line": v_line,
                "col": v_col,
                "snippet": None,
                "updated_input": None,
                "suggested_action": "fix_python_syntax",
                "reason_code": reason_code,
            }

    # 2. Protected path edits (check before secret scan unless verdict explicitly specifies secret)
    protected_kind = _classify_protected_path(target_path_str)
    explicit_secret_rc = any(tok in rc_low for tok in ("secret", "credential", "private_key"))
    if ("protected_path" in rc_low or "control_plane" in rc_low or protected_kind is not None) and not explicit_secret_rc:
        return {
            "category": "protected_path",
            "hint": _protected_path_hint(target_path_str, protected_kind),
            "suggested_tool": None,
            "line": None,
            "col": None,
            "updated_input": None,
            "suggested_action": "use_template_or_env_config",
            "reason_code": reason_code,
        }

    # 3. Secret leak in diff / content / command
    secret_hit = _detect_secret_location(code_text) or _detect_secret_location(command_str)
    if explicit_secret_rc or secret_hit is not None:
        s_type = secret_hit["secret_type"] if secret_hit else "credential"
        s_line = secret_hit["line"] if secret_hit else None
        s_col = secret_hit["col"] if secret_hit else None
        var_name = secret_hit["var_name"] if secret_hit else "API_SECRET"
        loc = f" at line {s_line}, col {s_col}" if s_line is not None else ""
        return {
            "category": "secret_leak",
            "hint": (
                f"Potential raw credential ({s_type}) detected{loc}. "
                f"Do not inline secrets in code or commands; use an out-of-band handle such as "
                f"`$SECRET:{var_name}` or `os.environ[\"{var_name}\"]`."
            ),
            "suggested_tool": None,
            "line": s_line,
            "col": s_col,
            "updated_input": None,
            "suggested_action": "replace_with_secret_handle",
            "reason_code": reason_code,
        }

    # 4. Dangerous shell pattern (`git push --force` / `-f` excluding `--force-with-lease`)
    if command_str:
        safe_push_cmd = _detect_git_force_push(command_str)
        if safe_push_cmd is not None:
            return {
                "category": "dangerous_shell",
                "hint": (
                    "Force-pushing (`git push --force` / `-f`) risks overwriting upstream commits. "
                    "Use `git push --force-with-lease` instead so the push aborts if the remote branch changed."
                ),
                "suggested_tool": None,
                "line": None,
                "col": None,
                "updated_input": {"command": safe_push_cmd} if allow_rewrite else None,
                "suggested_action": "use_force_with_lease",
                "reason_code": reason_code,
            }
        for _pat_name, pat, msg in _DANGEROUS_SHELL_RES:
            if pat.search(command_str):
                return {
                    "category": "dangerous_shell",
                    "hint": msg,
                    "suggested_tool": None,
                    "line": None,
                    "col": None,
                    "updated_input": None,
                    "suggested_action": "revise_destructive_shell_command",
                    "reason_code": reason_code,
                }

    # 5. Shell-over-structured-tool (`cat` -> Read, `grep`/`rg` -> Grep, `find` -> Glob)
    if command_str:
        structured = _detect_shell_over_structured_tool(command_str)
        if structured is not None:
            exe, suggested_tool = structured
            return {
                "category": "shell_over_structured_tool",
                "hint": (
                    f"Shell command `{exe}` duplicates a dedicated structured tool. "
                    f"Use `{suggested_tool}` instead for lower context overhead and safer path handling."
                ),
                "suggested_tool": suggested_tool,
                "line": None,
                "col": None,
                "updated_input": None,
                "suggested_action": f"use_{suggested_tool.lower()}_tool",
                "reason_code": reason_code,
            }

    # 6. Generic fallback hint
    fallback_action = verdict_action or (
        "request_human_approval" if decision == "escalate" else "revise_or_split_action"
    )
    label_str = redact_secrets(str(verdict.get("label") or "")).strip()
    details: List[str] = [f"Action was `{decision}`"]
    if reason_code:
        details.append(f"reason `{reason_code}`")
    if label_str:
        details.append(f"({label_str})")
    hint_msg = (
        " ".join(details)
        + f". Suggested action: `{fallback_action}`. Narrow scope, avoid side effects, or request explicit confirmation."
    )
    return {
        "category": "generic",
        "hint": hint_msg,
        "suggested_tool": None,
        "line": verdict.get("line") if isinstance(verdict.get("line"), int) else None,
        "col": verdict.get("col") if isinstance(verdict.get("col"), int) else None,
        "updated_input": None,
        "suggested_action": fallback_action,
        "reason_code": reason_code,
    }


def _betacf(a: float, b: float, x: float) -> float:
    """Evaluate continued fraction for regularized incomplete beta I_x(a, b) via Lentz's method."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Compute regularized incomplete beta function I_x(a, b) using pure stdlib `math`."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt) if log_bt > -745.0 else 0.0
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def clopper_pearson_upper_bound(k_events: int, n_samples: int, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson `(1 - alpha)` upper binomial confidence bound (pure `math`)."""
    n = int(n_samples)
    k = int(k_events)
    a = max(1e-9, min(0.999999, float(alpha)))
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0 - (a ** (1.0 / float(n)))
    if k >= n:
        return 1.0

    # Solve I_p(k + 1, n - k) = 1 - alpha via bisection (monotonically increasing in p).
    target = 1.0 - a
    lo, hi = 0.0, 1.0
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        if _regularized_incomplete_beta(mid, float(k + 1), float(n - k)) < target:
            lo = mid
        else:
            hi = mid
    return hi


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


def _extract_false_positive_label(entry: Mapping[str, Any], decision: str) -> Optional[bool]:
    """Return True if entry is a labeled false positive, False if true positive, or None if unlabeled."""
    if "false_positive" in entry and entry["false_positive"] is not None:
        return bool(entry["false_positive"])
    if "verified_correct" in entry and entry["verified_correct"] is not None:
        return not bool(entry["verified_correct"])
    outcome = str(
        entry.get("outcome") or entry.get("resolution") or entry.get("human_decision") or entry.get("feedback") or ""
    ).strip().lower()
    if not outcome:
        return None
    if outcome in {"false_positive", "fp", "overridden", "benign", "approved", "allow", "allowed"}:
        return decision in ("deny", "escalate")
    if outcome in {"true_positive", "tp", "confirmed", "rejected", "denied", "deny", "blocked"}:
        return False
    return None


def calibrate_thresholds_from_journal(
    entries: Sequence[Mapping[str, Any]],
    *,
    current_threshold: float = 0.5,
    target_fdr: float = 0.05,
) -> Dict[str, Any]:
    """Summarize decision journal telemetry and calibrate threshold using Clopper-Pearson UCB.

    Args:
        entries: Sequence of decision journal dicts (`decision`, `reason_code`, optional
            `verified_correct`/`false_positive`/`outcome`).
        current_threshold: Current decision/risk threshold in `[0.05, 0.95]`.
        target_fdr: Target false discovery / error rate bound `alpha` in `(0, 0.5]`.

    Returns:
        Summary dictionary with `calibrated_sample_count`, `recommended_threshold`,
        `noisy_reason_codes`, empirical `allow_rate`/`block_rate`/`escalate_rate`, and
        Clopper-Pearson upper confidence bounds.
    """
    cur = max(0.05, min(0.95, float(current_threshold)))
    fdr_target = max(1e-4, min(0.5, float(target_fdr)))

    valid_entries = [e for e in entries if isinstance(e, Mapping)]
    n_total = len(valid_entries)

    allow_count = 0
    deny_count = 0
    escalate_count = 0
    labeled_total = 0
    fp_total = 0

    rule_acc: Dict[str, Dict[str, int]] = {}
    for entry in valid_entries:
        dec = _normalize_decision(entry.get("decision"))
        if dec == "allow":
            allow_count += 1
        elif dec == "deny":
            deny_count += 1
        elif dec == "escalate":
            escalate_count += 1

        rc = str(entry.get("reason_code") or "unknown").strip() or "unknown"
        bucket = rule_acc.setdefault(
            rc,
            {"total": 0, "allow": 0, "deny": 0, "escalate": 0, "labeled": 0, "fp": 0},
        )
        bucket["total"] += 1
        if dec in ("allow", "deny", "escalate"):
            bucket[dec] += 1

        fp_label = _extract_false_positive_label(entry, dec)
        if fp_label is not None:
            labeled_total += 1
            bucket["labeled"] += 1
            if fp_label:
                fp_total += 1
                bucket["fp"] += 1

    per_reason_stats: Dict[str, Dict[str, Any]] = {}
    noisy_candidates: List[Tuple[int, float, str]] = []

    for rc, b in sorted(rule_acc.items()):
        r_total = b["total"]
        r_esc = b["escalate"]
        r_labeled = b["labeled"]
        r_fp = b["fp"]
        r_esc_rate = r_esc / float(max(1, r_total))
        r_esc_ub = clopper_pearson_upper_bound(r_esc, r_total, alpha=fdr_target)

        r_fp_rate: Optional[float] = None
        r_fp_ub: Optional[float] = None
        is_noisy = False

        if r_labeled > 0:
            r_fp_rate = r_fp / float(r_labeled)
            r_fp_ub = clopper_pearson_upper_bound(r_fp, r_labeled, alpha=fdr_target)
            if r_fp > 0 and (r_fp_rate > fdr_target or r_fp_ub > max(fdr_target * 2.0, 0.25)):
                is_noisy = True
                noisy_candidates.append((r_fp, r_fp_ub, rc))
        else:
            share_of_journal = r_esc / float(max(1, n_total))
            min_esc = 2 if n_total <= 20 else 3
            if r_esc >= min_esc and r_esc_rate >= 0.5 and share_of_journal > fdr_target:
                is_noisy = True
                noisy_candidates.append((r_esc, r_esc_ub, rc))

        per_reason_stats[rc] = {
            "total": r_total,
            "allow": b["allow"],
            "block": b["deny"],
            "escalate": r_esc,
            "escalate_rate": round(r_esc_rate, 6),
            "escalate_upper_bound": round(r_esc_ub, 6),
            "labeled_count": r_labeled,
            "false_positive_count": r_fp,
            "false_positive_rate": round(r_fp_rate, 6) if r_fp_rate is not None else None,
            "false_positive_upper_bound": round(r_fp_ub, 6) if r_fp_ub is not None else None,
            "noisy": is_noisy,
        }

    noisy_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    noisy_reason_codes = [item[2] for item in noisy_candidates]

    if n_total == 0:
        return {
            "calibrated_sample_count": 0,
            "current_threshold": round(cur, 4),
            "recommended_threshold": round(cur, 4),
            "target_fdr": float(fdr_target),
            "allow_count": 0,
            "block_count": 0,
            "deny_count": 0,
            "escalate_count": 0,
            "allow_rate": 0.0,
            "block_rate": 0.0,
            "deny_rate": 0.0,
            "escalate_rate": 0.0,
            "block_upper_bound": 1.0,
            "escalate_upper_bound": 1.0,
            "labeled_sample_count": 0,
            "false_positive_count": 0,
            "empirical_fdr": None,
            "fdr_upper_bound": None,
            "noisy_reason_codes": [],
            "per_reason_stats": {},
            "confidence_level": round(1.0 - fdr_target, 6),
        }

    allow_rate = allow_count / float(n_total)
    block_rate = deny_count / float(n_total)
    escalate_rate = escalate_count / float(n_total)
    block_ub = clopper_pearson_upper_bound(deny_count, n_total, alpha=fdr_target)
    escalate_ub = clopper_pearson_upper_bound(escalate_count, n_total, alpha=fdr_target)

    empirical_fdr: Optional[float] = None
    fdr_ub: Optional[float] = None
    if labeled_total > 0:
        empirical_fdr = fp_total / float(labeled_total)
        fdr_ub = clopper_pearson_upper_bound(fp_total, labeled_total, alpha=fdr_target)
        if fp_total > 0 and fdr_ub > fdr_target:
            delta = max(0.02, min(0.25, 0.5 * (fdr_ub - fdr_target)))
            recommended = min(0.95, cur + delta)
        elif fp_total == 0 and fdr_ub <= fdr_target and escalate_rate > max(0.25, fdr_target * 3.0):
            recommended = max(0.05, cur - 0.02)
        else:
            recommended = cur
    else:
        if noisy_reason_codes or escalate_ub > max(0.35, fdr_target * 4.0):
            delta = max(0.02, min(0.20, 0.25 * max(0.0, escalate_rate - fdr_target)))
            recommended = min(0.95, cur + delta)
        else:
            recommended = cur

    return {
        "calibrated_sample_count": n_total,
        "current_threshold": round(cur, 4),
        "recommended_threshold": round(max(0.05, min(0.95, recommended)), 4),
        "target_fdr": float(fdr_target),
        "allow_count": allow_count,
        "block_count": deny_count,
        "deny_count": deny_count,
        "escalate_count": escalate_count,
        "allow_rate": round(allow_rate, 6),
        "block_rate": round(block_rate, 6),
        "deny_rate": round(block_rate, 6),
        "escalate_rate": round(escalate_rate, 6),
        "block_upper_bound": round(block_ub, 6),
        "escalate_upper_bound": round(escalate_ub, 6),
        "labeled_sample_count": labeled_total,
        "false_positive_count": fp_total,
        "empirical_fdr": round(empirical_fdr, 6) if empirical_fdr is not None else None,
        "fdr_upper_bound": round(fdr_ub, 6) if fdr_ub is not None else None,
        "noisy_reason_codes": noisy_reason_codes,
        "per_reason_stats": per_reason_stats,
        "confidence_level": round(1.0 - fdr_target, 6),
    }


__all__ = [
    "calibrate_thresholds_from_journal",
    "clopper_pearson_upper_bound",
    "generate_repair_hint",
]
