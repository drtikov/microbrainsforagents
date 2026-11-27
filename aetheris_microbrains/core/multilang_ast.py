"""Multi-language structural AST & lexical analyzer (stdlib-only + optional tree-sitter).

Supports JS/TS, Go, Rust, SQL, Terraform/HCL, YAML, and Dockerfile with
comment/string-aware scanning, literal-concatenation folding, optional tree-sitter
syntax/node checks, and language-appropriate inline suppressions (`//`, `#`, `--`).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .secrets import _SECRET_TOKEN_REGEXES

_LITERAL_CONCAT_RE = re.compile(r"""['"`]\s*\+\s*['"`]""")
_JS_REQUIRE_CALL_RE = re.compile(r"\b(?:require|import)\s*\(\s*([^)]*)\)")
_JS_LITERAL_ARG_RE = re.compile(r"""(['"])[^'"\\]*\1|`[^`$\\]*`""")
_JS_SENSITIVE_ENV_RE = re.compile(
    r"\bprocess\.env(?:\.[A-Za-z0-9_]*(?:SECRET|TOKEN|KEY|PASSWORD|PASSWD|AUTH|CREDENTIAL|AWS_|GITHUB_|OPENAI_|ANTHROPIC_)[A-Za-z0-9_]*|"
    r"\[\s*['\"][A-Za-z0-9_]*(?:SECRET|TOKEN|KEY|PASSWORD|PASSWD|AUTH|CREDENTIAL|AWS_|GITHUB_|OPENAI_|ANTHROPIC_)[A-Za-z0-9_]*['\"]\s*\])"
)
_JS_NET_SINK_RE = re.compile(
    r"\b(?:fetch\s*\(|axios\.(?:post|put|get|request)\s*\(|https?\.(?:request|get)\s*\(|"
    r"new\s+XMLHttpRequest\b|navigator\.sendBeacon\s*\(|net\.connect\s*\()"
)
_JS_EVAL_RE = re.compile(
    r"\beval\s*\(|(?:globalThis|window|self)\s*\[\s*['\"]eval['\"]\s*\]"
)
_JS_DANGEROUS_SINK_RE = re.compile(
    r"\bnew\s+Function\s*\(|\bchild_process\b|\b(?:execSync|spawnSync|execFile|execFileSync)\s*\(|"
    r"\bvm\.runIn(?:New|This)Context\s*\("
)

_GO_SHELL_RE = re.compile(
    r"""\bexec\.Command(?:Context)?\s*\(\s*(?:[^,]+,\s*)?['"`](?:/bin/)?(?:sh|bash|zsh|cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh)['"`]"""
    r"""|\bsyscall\.(?:Exec|ForkExec)\s*\(""",
    re.DOTALL,
)
_GO_PLUGIN_RE = re.compile(r"\bplugin\.Open\s*\(")

_RUST_SHELL_NEW_RE = re.compile(
    r"""(?:std::process::)?Command::new\s*\(\s*['"](?:/bin/)?(?:sh|bash|zsh|cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh)['"]\s*\)"""
)
_RUST_ARG_C_RE = re.compile(r"""\.args?\s*\(\s*\[?\s*['"](?:-c|/c|-Command)['"]""")
_RUST_TRANSMUTE_RE = re.compile(r"\b(?:std::|core::)?mem::transmute(?:_copy)?\b")

_SQL_DESTRUCTIVE_RE = re.compile(
    r"\b(?:DROP\s+(?:DATABASE|SCHEMA|TABLE|IF\s+EXISTS)\b|TRUNCATE\s+(?:TABLE\s+)?[A-Za-z_\"`[])",
    re.IGNORECASE,
)
_SQL_EXEC_RE = re.compile(
    r"\b(?:xp_cmdshell|COPY\s+[\w.\"`\[\]]+\s+FROM\s+PROGRAM|INTO\s+(?:OUTFILE|DUMPFILE)|LOAD_FILE\s*\()",
    re.IGNORECASE,
)

_TF_PROVISIONER_RE = re.compile(r"""provisioner\s+['"](?:local|remote)-exec['"]""")
_TF_EXTERNAL_DATA_RE = re.compile(r"""\bdata\s+['"]external['"]""")
_TF_WILD_ACTION_RE = re.compile(r"""Action\s*[:=]\s*\[?\s*['"]\*['"]""")
_TF_WILD_RESOURCE_RE = re.compile(r"""Resource\s*[:=]\s*\[?\s*['"]\*['"]""")

_DOCKER_REMOTE_ADD_RE = re.compile(r"(?i)^\s*ADD\s+https?://\S+")
_DOCKER_SUID_RE = re.compile(r"\bchmod\s+(?:[ugo]*\+s|47[0-7]{2}|27[0-7]{2})\b")
_YAML_UNSAFE_TAG_RE = re.compile(r"!!python/(?:object|apply|module|name)\b")
_YAML_CMD_RCE_RE = re.compile(
    r"(?im)^\s*(?:-\s*)?(?:script|run|command|cmd|exec)\s*:\s*.*?\b(?:curl|wget)\b[^#\n]*\|\s*(?:env\s+|sudo\s+)?(?:sh|bash|zsh|dash|python\d?|perl)\b"
)

_TS_GRAMMAR_BY_EXT: Dict[str, Tuple[str, str]] = {
    ".js": ("tree_sitter_javascript", "language"),
    ".jsx": ("tree_sitter_javascript", "language"),
    ".mjs": ("tree_sitter_javascript", "language"),
    ".cjs": ("tree_sitter_javascript", "language"),
    ".ts": ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"),
    ".go": ("tree_sitter_go", "language"),
    ".rs": ("tree_sitter_rust", "language"),
    ".sql": ("tree_sitter_sql", "language"),
    ".yaml": ("tree_sitter_yaml", "language"),
    ".yml": ("tree_sitter_yaml", "language"),
    ".sh": ("tree_sitter_bash", "language"),
    ".bash": ("tree_sitter_bash", "language"),
}

_SUPPRESSIBLE_MULTILANG_RULES = frozenset({
    "rust_unsafe_transmute",
    "terraform_exec_provisioner",
    "terraform_external_program",
    "js_dynamic_require",
    "secret_leak_pattern_found",
})


def strip_comments_preserve_lines(code: str, lang: str) -> str:
    """Strip comments for `lang` while preserving newlines and string literals."""
    src = str(code or "")
    if not src:
        return ""
    has_c_line = lang in {"js", "ts", "go", "rust", "tf"}
    has_c_block = lang in {"js", "ts", "go", "rust", "tf", "sql"}
    has_hash_line = lang in {"tf", "yaml", "shell", "dockerfile"}
    has_dash_line = lang == "sql"
    has_backtick_str = lang in {"js", "ts", "go"}

    out: List[str] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        # String literals (preserve verbatim so patterns inside data strings are seen when appropriate,
        # and quotes inside strings never start comments). For SQL, skip comment-only stripping cleanly.
        if ch in ('"', "'") or (has_backtick_str and ch == "`"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                c2 = src[i]
                out.append(c2)
                if c2 == "\\" and quote != "`" and i + 1 < n and lang != "sql":
                    i += 1
                    out.append(src[i])
                    i += 1
                    continue
                if c2 == quote:
                    if lang == "sql" and quote == "'" and i + 1 < n and src[i + 1] == "'":
                        i += 1
                        out.append(src[i])
                        i += 1
                        continue
                    i += 1
                    break
                i += 1
            continue
        if has_c_block and ch == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n:
                if src[i] == "*" and i + 1 < n and src[i + 1] == "/":
                    out.append("  ")
                    i += 2
                    break
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            continue
        if has_c_line and ch == "/" and nxt == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if has_dash_line and ch == "-" and nxt == "-":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if has_hash_line and ch == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _line_of_offset(text: str, pos: int) -> int:
    return text.count("\n", 0, max(0, pos)) + 1


def scan_with_tree_sitter(ext: str, code: str) -> Tuple[bool, Optional[bool], List[Tuple[str, int]]]:
    """Run optional tree-sitter parse + AST walk; returns (parsed, syntax_ok, raw_findings)."""
    spec = _TS_GRAMMAR_BY_EXT.get(ext)
    if spec is None or not code.strip():
        return False, None, []
    try:
        from .layered_decision_engine import _get_tree_sitter_parser_factory
    except Exception:
        return False, None, []

    factory = _get_tree_sitter_parser_factory(*spec)
    if factory is None:
        return False, None, []
    try:
        parser_cls, lang = factory
        parser = parser_cls(lang)
        raw = code.encode("utf-8", errors="ignore")
        tree = parser.parse(raw)
        if tree is None or not hasattr(tree, "root_node"):
            return False, None, []
    except Exception:
        return False, None, []

    syntax_ok = not bool(getattr(tree.root_node, "has_error", False))
    findings: List[Tuple[str, int]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        ntype = getattr(node, "type", "")
        if ntype in {"call_expression", "new_expression", "macro_invocation"}:
            snippet = raw[node.start_byte : min(node.end_byte, node.start_byte + 240)].decode("utf-8", errors="ignore")
            line_no = int(getattr(node, "start_point", (0, 0))[0]) + 1
            collapsed = _LITERAL_CONCAT_RE.sub("", snippet)
            if ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                if _JS_EVAL_RE.search(collapsed):
                    findings.append(("js_eval_call", line_no))
                if _JS_DANGEROUS_SINK_RE.search(collapsed):
                    findings.append(("js_dangerous_sink", line_no))
            elif ext == ".go" and _GO_SHELL_RE.search(collapsed):
                findings.append(("go_shell_exec_sink", line_no))
            elif ext == ".rs" and _RUST_SHELL_NEW_RE.search(collapsed) and _RUST_ARG_C_RE.search(snippet):
                findings.append(("rust_shell_command_sink", line_no))
        stack.extend(reversed(getattr(node, "children", ())))
    return True, syntax_ok, findings


def analyze_multilang_code(
    ext: str,
    fname_lower: str,
    code_text: str,
    *,
    policy: Any = None,
    allow_suppression: bool = True,
) -> Dict[str, Any]:
    """Perform structural & lexical security analysis across JS/TS, Go, Rust, SQL, HCL, YAML, Dockerfile."""
    src = str(code_text or "")
    ext_low = str(ext or "").strip().lower()
    raw_lines = src.splitlines()

    if fname_lower == "dockerfile" or ext_low == ".dockerfile":
        lang = "dockerfile"
    elif ext_low in {".js", ".jsx", ".mjs", ".cjs"}:
        lang = "js"
    elif ext_low in {".ts", ".tsx"}:
        lang = "ts"
    elif ext_low == ".go":
        lang = "go"
    elif ext_low == ".rs":
        lang = "rust"
    elif ext_low == ".sql":
        lang = "sql"
    elif ext_low in {".tf", ".tfvars", ".hcl"}:
        lang = "tf"
    elif ext_low in {".yaml", ".yml"}:
        lang = "yaml"
    else:
        lang = ext_low.lstrip(".") or "text"

    raw_findings: List[Tuple[str, int]] = []

    if lang == "dockerfile":
        from .layered_decision_engine import analyze_shell_command

        joined_lines: List[Tuple[str, int]] = []
        buf = ""
        start_ln = 1
        for idx, ln in enumerate(raw_lines, start=1):
            stripped = ln.rstrip()
            if not buf:
                start_ln = idx
            if stripped.endswith("\\") and not stripped.lstrip().startswith("#"):
                buf += " " + stripped[:-1].strip()
                continue
            full = (buf + " " + stripped).strip() if buf else stripped.strip()
            buf = ""
            if full and not full.startswith("#"):
                joined_lines.append((full, start_ln))
        if buf.strip():
            joined_lines.append((buf.strip(), start_ln))

        for stmt, ln_no in joined_lines:
            if _DOCKER_REMOTE_ADD_RE.match(stmt):
                raw_findings.append(("dockerfile_remote_add", ln_no))
            if stmt.upper().startswith("RUN "):
                cmd_part = stmt[4:].strip()
                if _DOCKER_SUID_RE.search(cmd_part):
                    raw_findings.append(("dockerfile_privileged_suid", ln_no))
                sh_res = analyze_shell_command(cmd_part, policy=policy)
                if not sh_res.get("is_safe", True):
                    for fi in sh_res.get("findings", ()):
                        raw_findings.append((f"shell_script_unsafe_cmd:{fi}", ln_no))
        return {
            "language": "dockerfile",
            "syntax_valid": True,
            "findings": list(dict.fromkeys(r for r, _ in raw_findings)),
            "suppressed_findings": [],
            "ast_parser_backend": "dockerfile_parser",
            "tree_sitter_parsed": False,
        }

    stripped_src = strip_comments_preserve_lines(src, lang)
    collapsed_src = _LITERAL_CONCAT_RE.sub("", stripped_src)

    # Split-literal secret obfuscation detection across all supported code/config files
    for m in _LITERAL_CONCAT_RE.finditer(stripped_src):
        ln_no = _line_of_offset(stripped_src, m.start())
        ln_txt = stripped_src.splitlines()[ln_no - 1] if ln_no - 1 < len(stripped_src.splitlines()) else ""
        folded_ln = _LITERAL_CONCAT_RE.sub("", ln_txt)
        if any(p.search(folded_ln) for p in _SECRET_TOKEN_REGEXES):
            raw_findings.append(("obfuscated_secret_concatenation", ln_no))

    if lang in {"js", "ts"}:
        for m in _JS_EVAL_RE.finditer(collapsed_src):
            raw_findings.append(("js_eval_call", _line_of_offset(collapsed_src, m.start())))
        for m in _JS_DANGEROUS_SINK_RE.finditer(collapsed_src):
            raw_findings.append(("js_dangerous_sink", _line_of_offset(collapsed_src, m.start())))
        for m_req in _JS_REQUIRE_CALL_RE.finditer(stripped_src):
            arg = m_req.group(1).strip()
            if arg and not _JS_LITERAL_ARG_RE.fullmatch(arg):
                raw_findings.append(("js_dynamic_require", _line_of_offset(stripped_src, m_req.start())))
        if _JS_SENSITIVE_ENV_RE.search(stripped_src) and _JS_NET_SINK_RE.search(stripped_src):
            m_env = _JS_SENSITIVE_ENV_RE.search(stripped_src)
            raw_findings.append(("js_secret_exfil_pattern", _line_of_offset(stripped_src, m_env.start() if m_env else 0)))

    elif lang == "go":
        for m in _GO_SHELL_RE.finditer(collapsed_src):
            raw_findings.append(("go_shell_exec_sink", _line_of_offset(collapsed_src, m.start())))
        for m in _GO_PLUGIN_RE.finditer(collapsed_src):
            raw_findings.append(("go_dynamic_plugin_load", _line_of_offset(collapsed_src, m.start())))

    elif lang == "rust":
        if _RUST_SHELL_NEW_RE.search(collapsed_src) and _RUST_ARG_C_RE.search(collapsed_src):
            m = _RUST_SHELL_NEW_RE.search(collapsed_src)
            raw_findings.append(("rust_shell_command_sink", _line_of_offset(collapsed_src, m.start() if m else 0)))
        for m in _RUST_TRANSMUTE_RE.finditer(stripped_src):
            raw_findings.append(("rust_unsafe_transmute", _line_of_offset(stripped_src, m.start())))

    elif lang == "sql":
        for m in _SQL_DESTRUCTIVE_RE.finditer(stripped_src):
            raw_findings.append(("sql_destructive_ddl", _line_of_offset(stripped_src, m.start())))
        for m in _SQL_EXEC_RE.finditer(stripped_src):
            raw_findings.append(("sql_command_or_file_exec", _line_of_offset(stripped_src, m.start())))

    elif lang == "tf":
        for m in _TF_PROVISIONER_RE.finditer(stripped_src):
            raw_findings.append(("terraform_exec_provisioner", _line_of_offset(stripped_src, m.start())))
        for m in _TF_EXTERNAL_DATA_RE.finditer(stripped_src):
            raw_findings.append(("terraform_external_program", _line_of_offset(stripped_src, m.start())))
        if _TF_WILD_ACTION_RE.search(stripped_src) and _TF_WILD_RESOURCE_RE.search(stripped_src):
            m = _TF_WILD_ACTION_RE.search(stripped_src)
            raw_findings.append(("terraform_wildcard_iam", _line_of_offset(stripped_src, m.start() if m else 0)))

    elif lang == "yaml":
        for m in _YAML_UNSAFE_TAG_RE.finditer(stripped_src):
            raw_findings.append(("yaml_unsafe_deserialization_tag", _line_of_offset(stripped_src, m.start())))
        for m in _YAML_CMD_RCE_RE.finditer(stripped_src):
            raw_findings.append(("yaml_rce_pipe", _line_of_offset(stripped_src, m.start())))

    ts_parsed, ts_syntax_ok, ts_findings = scan_with_tree_sitter(ext_low, src)
    raw_findings.extend(ts_findings)

    # Filter auditable inline suppressions (`// aetheris: allow <rule> -- <reason>`)
    from .layered_decision_engine import _parse_inline_suppression, _suppression_matches

    active: List[str] = []
    suppressed: List[Dict[str, Any]] = []
    seen_active: set[str] = set()
    for rule, ln_no in raw_findings:
        supp = None
        if allow_suppression and rule in _SUPPRESSIBLE_MULTILANG_RULES and 1 <= ln_no <= len(raw_lines):
            for probe_idx in (ln_no - 1, ln_no - 2):
                if 0 <= probe_idx < len(raw_lines):
                    parsed = _parse_inline_suppression(raw_lines[probe_idx])
                    if parsed is not None and _suppression_matches(parsed[0], rule):
                        supp = {"rule": rule, "line": probe_idx + 1, "reason": parsed[1]}
                        break
        if supp is not None:
            suppressed.append(supp)
        elif rule not in seen_active:
            seen_active.add(rule)
            active.append(rule)

    return {
        "language": lang,
        "syntax_valid": bool(ts_syntax_ok) if (ts_parsed and ts_syntax_ok is not None) else True,
        "findings": active,
        "suppressed_findings": suppressed,
        "ast_parser_backend": "tree_sitter" if ts_parsed else "multilang_structural",
        "tree_sitter_parsed": bool(ts_parsed),
    }
