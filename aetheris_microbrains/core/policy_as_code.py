"""Policy-as-Code Compilers & Evaluators: AWS Cedar (.cedar) and OPA Rego (.rego).

Stdlib-only bidirectional bridge between enterprise policy-as-code standards and
Aetheris Microbrains' `PolicyConfig.custom_rules`:
  1. AWS Cedar (`forbid`-wins over `permit`): annotations (`@id`, `@reason`),
     scope constraints (`action == Action::"..."`, `action in [...]`, `resource == File::"..."`),
     and `when`/`unless` boolean expressions (`&&`, `||`, `!`, `==`, `!=`, `like`, `contains`, `in`).
  2. OPA Rego v1 (`deny` > `escalate` > `allow`): package headers, `deny contains msg if`,
     `escalate contains msg if`, `allow if`, `startswith`, `endswith`, `contains`,
     `regex.match`, `glob.match`, `in`, and `not` predicates.
  3. Safety invariant: every compiled `allow` / `permit` rule is validated via
     `verify_policy_patch()` before entering `PolicyConfig.custom_rules`.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# AWS Cedar parser, evaluator, compiler & exporter
# ---------------------------------------------------------------------------

_CEDAR_ANN_RE = re.compile(r'@([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*"([^"]*)"\s*\)')
_CEDAR_HEAD_RE = re.compile(r'\b(permit|forbid)\s*\((.*?)\)\s*(.*);$', re.DOTALL)
_CEDAR_ACTION_EQ_RE = re.compile(r'action\s*==\s*Action::"([^"]+)"')
_CEDAR_ACTION_IN_RE = re.compile(r'action\s+in\s+\[([^\]]+)\]')
_CEDAR_ACTION_ITEM_RE = re.compile(r'Action::"([^"]+)"')
_CEDAR_RESOURCE_EQ_RE = re.compile(r'resource\s*==\s*(?:File|Resource|Path)::"([^"]+)"')


@dataclass
class CedarStatement:
    effect: str  # "permit" | "forbid"
    id: str
    reason: str
    actions: Tuple[str, ...]
    resource_glob: Optional[str]
    when_expr: Optional[str]
    unless_expr: Optional[str]


def _split_cedar_statements(text: str) -> List[str]:
    cleaned_lines = []
    for ln in str(text or "").splitlines():
        st = ln.lstrip()
        if st.startswith("//"):
            continue
        cleaned_lines.append(ln)
    src = "\n".join(cleaned_lines)
    stmts: List[str] = []
    buf: List[str] = []
    in_str = False
    esc = False
    for ch in src:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == ";":
            s = "".join(buf).strip()
            if s:
                stmts.append(s)
            buf = []
    if "".join(buf).strip():
        raise ValueError("Unterminated Cedar policy statement (missing ';')")
    return stmts


def _extract_cedar_clauses(tail: str) -> Tuple[Optional[str], Optional[str]]:
    when_expr: Optional[str] = None
    unless_expr: Optional[str] = None
    pos = 0
    s = tail.strip()
    while pos < len(s):
        rest = s[pos:].lstrip()
        if not rest:
            break
        kind = "when" if rest.startswith("when") else ("unless" if rest.startswith("unless") else "")
        if not kind:
            raise ValueError(f"Unexpected Cedar clause near: {rest[:40]!r}")
        brace_idx = rest.find("{")
        if brace_idx == -1:
            raise ValueError(f"Missing '{{' in Cedar {kind} clause")
        depth = 0
        in_str = False
        esc = False
        end_idx = -1
        for idx in range(brace_idx, len(rest)):
            c = rest[idx]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end_idx = idx
                    break
        if end_idx == -1:
            raise ValueError(f"Unclosed '{{' in Cedar {kind} clause")
        inner = rest[brace_idx + 1 : end_idx].strip()
        if kind == "when":
            when_expr = inner
        else:
            unless_expr = inner
        pos = len(s) - len(rest) + end_idx + 1
    return when_expr, unless_expr


def parse_cedar_policies(cedar_text: str) -> List[CedarStatement]:
    """Parse AWS Cedar statements with `@id`/`@reason`, scope, and `when`/`unless` clauses."""
    out: List[CedarStatement] = []
    for idx, raw_stmt in enumerate(_split_cedar_statements(cedar_text), start=1):
        anns = {m.group(1): m.group(2) for m in _CEDAR_ANN_RE.finditer(raw_stmt)}
        body = _CEDAR_ANN_RE.sub("", raw_stmt).strip()
        m_head = _CEDAR_HEAD_RE.match(body)
        if not m_head:
            raise ValueError(f"Invalid Cedar statement syntax near: {body[:60]!r}")
        effect, scope_txt, tail = m_head.group(1), m_head.group(2), m_head.group(3)
        actions: List[str] = []
        m_eq = _CEDAR_ACTION_EQ_RE.search(scope_txt)
        if m_eq:
            actions.append(m_eq.group(1))
        m_in = _CEDAR_ACTION_IN_RE.search(scope_txt)
        if m_in:
            actions.extend(_CEDAR_ACTION_ITEM_RE.findall(m_in.group(1)))
        m_res = _CEDAR_RESOURCE_EQ_RE.search(scope_txt)
        resource_glob = m_res.group(1) if m_res else None
        when_expr, unless_expr = _extract_cedar_clauses(tail)
        stmt_id = anns.get("id") or f"cedar_{effect}_{idx}"
        reason = anns.get("reason") or f"Cedar {effect}: {stmt_id}"
        out.append(
            CedarStatement(
                effect=effect,
                id=stmt_id,
                reason=reason,
                actions=tuple(dict.fromkeys(actions)),
                resource_glob=resource_glob,
                when_expr=when_expr,
                unless_expr=unless_expr,
            )
        )
    return out


_CEDAR_TOK_RE = re.compile(
    r'\s*(?:(\&\&|\|\||==|!=|[!(),\[\]]|\b(?:like|contains|in|true|false)\b)|'
    r'"((?:[^"\\]|\\.)*)"|([A-Za-z_][A-Za-z0-9_.:]*))'
)


def _tokenize_cedar_expr(expr: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    pos = 0
    s = expr.strip()
    while pos < len(s):
        m = _CEDAR_TOK_RE.match(s, pos)
        if not m:
            raise ValueError(f"Unexpected token in Cedar expression at {s[pos:pos+20]!r}")
        pos = m.end()
        if m.group(1) is not None:
            tokens.append(("OP", m.group(1)))
        elif m.group(2) is not None:
            tokens.append(("STR", bytes(m.group(2), "utf-8").decode("unicode_escape")))
        elif m.group(3) is not None:
            tokens.append(("IDENT", m.group(3)))
    return tokens


def _resolve_cedar_attr(ident: str, request: Dict[str, Any]) -> Any:
    ctx = request.get("context") if isinstance(request.get("context"), dict) else request
    res = request.get("resource") if isinstance(request.get("resource"), dict) else request
    act = request.get("action") if isinstance(request.get("action"), dict) else request
    low = ident.lower()
    if low in {"context.cmd", "cmd", "command"}:
        return str(ctx.get("cmd") or request.get("cmd") or request.get("command") or "")
    if low in {"context.path", "resource.path", "path", "file_path"}:
        return str(ctx.get("path") or res.get("path") or request.get("path") or "").replace("\\", "/")
    if low in {"context.tool", "action.name", "tool"}:
        return str(act.get("name") or request.get("tool") or "").lower()
    if low in {"context.profile", "profile"}:
        return str(ctx.get("profile") or request.get("profile") or "")
    if "::" in ident:
        return ident.split("::", 1)[1].strip('"')
    return ctx.get(ident, request.get(ident, ""))


def _eval_cedar_expr(expr: str, request: Dict[str, Any]) -> bool:
    tokens = _tokenize_cedar_expr(expr)
    idx = 0

    def _peek() -> Optional[Tuple[str, str]]:
        return tokens[idx] if idx < len(tokens) else None

    def _next() -> Tuple[str, str]:
        nonlocal idx
        tok = tokens[idx]
        idx += 1
        return tok

    def _parse_atom() -> Any:
        t = _peek()
        if t is None:
            raise ValueError("Unexpected end of Cedar expression")
        if t == ("OP", "["):
            _next()
            arr = []
            while _peek() and _peek() != ("OP", "]"):
                arr.append(_parse_atom())
                if _peek() == ("OP", ","):
                    _next()
            if _peek() != ("OP", "]"):
                raise ValueError("Expected ']' in Cedar list")
            _next()
            return arr
        _next()
        if t[0] == "STR":
            return t[1]
        if t[1] == "true":
            return True
        if t[1] == "false":
            return False
        return _resolve_cedar_attr(t[1], request)

    def _parse_cmp() -> bool:
        if _peek() == ("OP", "("):
            _next()
            val = _parse_or()
            if _peek() != ("OP", ")"):
                raise ValueError("Expected ')' in Cedar expression")
            _next()
            return val
        if _peek() == ("OP", "!"):
            _next()
            return not _parse_cmp()
        lhs = _parse_atom()
        op_tok = _peek()
        if op_tok is None or op_tok[1] in {"&&", "||", ")"}:
            return bool(lhs)
        op = _next()[1]
        rhs = _parse_atom()
        if op == "==":
            return str(lhs) == str(rhs)
        if op == "!=":
            return str(lhs) != str(rhs)
        if op == "like":
            return fnmatch.fnmatch(str(lhs).lower(), str(rhs).lower())
        if op == "contains":
            return str(rhs) in str(lhs)
        if op == "in":
            if isinstance(rhs, list):
                return str(lhs).lower() in {str(x).lower() for x in rhs}
            return str(lhs) in str(rhs)
        raise ValueError(f"Unsupported Cedar operator: {op!r}")

    def _parse_and() -> bool:
        val = _parse_cmp()
        while _peek() == ("OP", "&&"):
            _next()
            rhs = _parse_cmp()
            val = val and rhs
        return val

    def _parse_or() -> bool:
        val = _parse_and()
        while _peek() == ("OP", "||"):
            _next()
            rhs = _parse_and()
            val = val or rhs
        return val

    return bool(_parse_or())


def evaluate_cedar_policies(
    statements: Sequence[CedarStatement],
    request: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate parsed Cedar statements with strict `forbid`-wins-over-`permit` semantics."""
    req_tool = str(
        (request.get("action") or {}).get("name")
        if isinstance(request.get("action"), dict)
        else request.get("tool") or ""
    ).strip().lower()
    req_path = str(
        (request.get("resource") or {}).get("path")
        if isinstance(request.get("resource"), dict)
        else request.get("path") or ""
    ).replace("\\", "/").strip().lower()

    matched_permits: List[CedarStatement] = []
    for stmt in statements:
        if stmt.actions and req_tool not in {a.lower() for a in stmt.actions}:
            continue
        if stmt.resource_glob:
            if not req_path or not fnmatch.fnmatch(req_path, stmt.resource_glob.replace("\\", "/").lower()):
                continue
        when_ok = _eval_cedar_expr(stmt.when_expr, request) if stmt.when_expr else True
        unless_bad = _eval_cedar_expr(stmt.unless_expr, request) if stmt.unless_expr else False
        if not (when_ok and not unless_bad):
            continue
        if stmt.effect == "forbid":
            return {"decision": "deny", "matched_id": stmt.id, "reason": stmt.reason}
        matched_permits.append(stmt)

    if matched_permits:
        first = matched_permits[0]
        return {"decision": "allow", "matched_id": first.id, "reason": first.reason}
    return {"decision": "default", "matched_id": None, "reason": "no_matching_cedar_statement"}


def _cedar_like_to_regex(pat: str, *, anchor_start: bool = False) -> str:
    esc = re.escape(pat).replace(r"\*", ".*").replace(r"\?", ".")
    return f"^{esc}$" if anchor_start else esc


def _compile_cedar_stmt_to_rule(stmt: CedarStatement) -> Dict[str, Any]:
    rule: Dict[str, Any] = {
        "id": stmt.id,
        "action": "deny" if stmt.effect == "forbid" else "allow",
        "reason": stmt.reason,
    }
    if stmt.actions:
        if len(stmt.actions) == 1:
            rule["tool"] = re.escape(stmt.actions[0])
        else:
            rule["tool"] = "(?:" + "|".join(re.escape(a) for a in stmt.actions) + ")"
    if stmt.resource_glob:
        rule["path_glob"] = stmt.resource_glob
    if stmt.when_expr and "||" not in stmt.when_expr and "!" not in stmt.when_expr:
        for clause in stmt.when_expr.split("&&"):
            c = clause.strip().strip("()")
            m_like = re.match(r'^context\.cmd\s+like\s+"([^"]+)"$', c)
            m_eq = re.match(r'^context\.cmd\s*==\s*"([^"]+)"$', c)
            m_con = re.match(r'^context\.cmd\s+contains\s+"([^"]+)"$', c)
            m_pl = re.match(r'^(?:context|resource)\.path\s+(?:like|==)\s+"([^"]+)"$', c)
            if m_like:
                rule["cmd_regex"] = _cedar_like_to_regex(m_like.group(1), anchor_start=True)
            elif m_eq:
                rule["cmd_regex"] = f"^{re.escape(m_eq.group(1))}$"
            elif m_con:
                rule["cmd_regex"] = re.escape(m_con.group(1))
            elif m_pl:
                rule["path_glob"] = m_pl.group(1)
    return rule


def cedar_to_policy_config(cedar_text: str, *, base_policy: Optional[Any] = None) -> Any:
    """Compile AWS Cedar statements into a verified `PolicyConfig` (`forbid` rules ordered first)."""
    from .layered_decision_engine import DEFAULT_POLICY, PolicyConfig
    from .policy_governance import verify_policy_patch

    base = base_policy if isinstance(base_policy, PolicyConfig) else DEFAULT_POLICY
    stmts = parse_cedar_policies(cedar_text)
    forbid_rules: List[Dict[str, Any]] = []
    permit_rules: List[Dict[str, Any]] = []
    for s in stmts:
        compiled = _compile_cedar_stmt_to_rule(s)
        if s.effect == "forbid":
            forbid_rules.append(compiled)
        else:
            permit_rules.append(compiled)

    # Order forbid rules before permit rules so forbid always wins in PolicyConfig evaluation
    ordered = forbid_rules + permit_rules
    intermediate = PolicyConfig.from_dict({
        "protected_patterns": list(base.protected_patterns),
        "extra_protected_patterns": list(base.extra_protected_patterns),
        "custom_rules": ordered,
        "profile": base.profile,
    })
    for pr in permit_rules:
        ver = verify_policy_patch(pr, base_policy=intermediate)
        if not ver["ok"]:
            raise ValueError(f"Unsafe Cedar permit rule {pr.get('id')!r}: {ver['issues']}")
    return intermediate


def export_policy_to_cedar(policy: Any) -> str:
    """Export `PolicyConfig.custom_rules` into AWS Cedar `.cedar` format."""
    lines: List[str] = ["// Exported by Aetheris Microbrains Policy-as-Code", ""]
    for idx, rule in enumerate(getattr(policy, "custom_rules", ()) or (), start=1):
        if not isinstance(rule, dict):
            continue
        act = str(rule.get("action") or "").lower()
        effect = "allow"
        if act == "deny":
            effect = "forbid"
        elif act == "allow":
            effect = "permit"
        else:
            continue
        rid = str(rule.get("id") or f"rule_{idx}").replace('"', "'")
        reason = str(rule.get("reason") or rid).replace('"', "'")
        scope_parts = ["principal"]
        tool_pat = str(rule.get("tool") or "").strip()
        if tool_pat and tool_pat not in {".*", ".+"}:
            clean_tools = re.sub(r"^\(\?:(.*)\)$", r"\1", tool_pat).split("|")
            clean_tools = [t.replace("\\", "") for t in clean_tools if t]
            if len(clean_tools) == 1:
                scope_parts.append(f'action == Action::"{clean_tools[0]}"')
            elif clean_tools:
                joined = ", ".join(f'Action::"{t}"' for t in clean_tools)
                scope_parts.append(f"action in [{joined}]")
        else:
            scope_parts.append("action")
        if rule.get("path_glob"):
            pg = str(rule["path_glob"]).replace('"', "'")
            scope_parts.append(f'resource == File::"{pg}"')
        else:
            scope_parts.append("resource")

        lines.append(f'@id("{rid}")')
        lines.append(f'@reason("{reason}")')
        when_clauses: List[str] = []
        if rule.get("cmd_regex"):
            cr = str(rule["cmd_regex"]).lstrip("^").rstrip("$").replace(".*", "*").replace("\\", "").replace('"', "'")
            when_clauses.append(f'context.cmd like "{cr}"')
        if when_clauses:
            lines.append(f"{effect} ({', '.join(scope_parts)}) when {{ {' && '.join(when_clauses)} }};")
        else:
            lines.append(f"{effect} ({', '.join(scope_parts)});")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# OPA Rego v1 parser, evaluator, compiler & exporter
# ---------------------------------------------------------------------------

_REGO_PKG_RE = re.compile(r"^\s*package\s+([A-Za-z0-9_.]+)")
_REGO_RULE_RE = re.compile(
    r'^\s*(deny|escalate|allow)(?:\s+contains\s+("([^"]*)"|\w+)|\[("([^"]*)"|\w+)\])?(?:\s+if)?\s*\{\s*$'
)


def parse_rego_policy(rego_text: str) -> Dict[str, Any]:
    """Parse OPA Rego `package`, `default allow`, `deny`, `escalate`, and `allow` rule blocks."""
    package = "aetheris.authz"
    default_allow = False
    rules: List[Dict[str, Any]] = []
    lines = str(rego_text or "").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].split("#", 1)[0].strip()
        if not raw or raw.startswith("import "):
            i += 1
            continue
        m_pkg = _REGO_PKG_RE.match(raw)
        if m_pkg:
            package = m_pkg.group(1)
            i += 1
            continue
        if re.match(r"^default\s+allow\s*:?=\s*true\b", raw):
            default_allow = True
            i += 1
            continue
        if re.match(r"^default\s+allow\s*:?=\s*false\b", raw):
            default_allow = False
            i += 1
            continue
        m_rule = _REGO_RULE_RE.match(raw)
        if m_rule:
            kind = m_rule.group(1)
            msg_lit = m_rule.group(3) or m_rule.group(5) or f"{kind}_rule_{len(rules) + 1}"
            body: List[str] = []
            i += 1
            while i < len(lines):
                ln = lines[i].split("#", 1)[0].strip()
                i += 1
                if ln == "}":
                    break
                if ln:
                    body.append(ln.rstrip(";"))
            rules.append({"kind": kind, "message": msg_lit, "conditions": body})
            continue
        i += 1
    return {"package": package, "default_allow": default_allow, "rules": rules}


def _resolve_rego_var(expr: str, inp: Dict[str, Any]) -> str:
    e = expr.strip()
    if e == "input.tool":
        return str(inp.get("tool") or "").lower()
    if e == "input.path":
        return str(inp.get("path") or "").replace("\\", "/")
    if e in {"input.cmd", "input.command"}:
        return str(inp.get("cmd") or inp.get("command") or "")
    if e.startswith("input."):
        return str(inp.get(e[6:], ""))
    return e


def _eval_rego_condition(cond: str, inp: Dict[str, Any]) -> bool:
    c = cond.strip()
    if c.startswith("not "):
        return not _eval_rego_condition(c[4:].strip(), inp)
    if re.match(r'^msg\s*:?=\s*"([^"]*)"$', c):
        return True
    m_eq = re.match(r'^([A-Za-z0-9_.]+)\s*(==|!=)\s*"([^"]*)"$', c)
    if m_eq:
        val = _resolve_rego_var(m_eq.group(1), inp)
        rhs = m_eq.group(3)
        if m_eq.group(1) == "input.tool":
            rhs = rhs.lower()
        return (val == rhs) if m_eq.group(2) == "==" else (val != rhs)
    m_in = re.match(r"^([A-Za-z0-9_.]+)\s+in\s+\[([^\]]*)\]$", c)
    if m_in:
        val = _resolve_rego_var(m_in.group(1), inp)
        items = [s.lower() if m_in.group(1) == "input.tool" else s for s in re.findall(r'"([^"]*)"', m_in.group(2))]
        return val in items
    m_fn = re.match(r'^(startswith|endswith|contains|regex\.match)\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$', c)
    if m_fn:
        fn, a1, a2 = m_fn.group(1), m_fn.group(2).strip(), m_fn.group(3).strip()
        if fn == "startswith":
            return _resolve_rego_var(a1, inp).startswith(a2.strip('"'))
        if fn == "endswith":
            return _resolve_rego_var(a1, inp).endswith(a2.strip('"'))
        if fn == "contains":
            return a2.strip('"') in _resolve_rego_var(a1, inp)
        if fn == "regex.match":
            return bool(re.search(a1.strip('"'), _resolve_rego_var(a2, inp)))
    m_glob = re.match(r'^glob\.match\(\s*"([^"]*)"\s*,\s*\[[^\]]*\]\s*,\s*([A-Za-z0-9_.]+)\s*\)$', c)
    if m_glob:
        return fnmatch.fnmatch(_resolve_rego_var(m_glob.group(2), inp).lower(), m_glob.group(1).lower())
    return False


def evaluate_rego_policy(rego_policy: Any, input_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate OPA Rego policy (`deny` > `escalate` > `allow`)."""
    parsed = parse_rego_policy(rego_policy) if isinstance(rego_policy, str) else rego_policy
    deny_msgs: List[str] = []
    esc_msgs: List[str] = []
    allow_matched = bool(parsed.get("default_allow", False))

    for rule in parsed.get("rules", []):
        conds = rule.get("conditions", [])
        msg = str(rule.get("message") or "")
        for c in conds:
            m_msg = re.match(r'^msg\s*:?=\s*"([^"]*)"$', str(c).strip())
            if m_msg:
                msg = m_msg.group(1)
        if all(_eval_rego_condition(str(c), input_doc) for c in conds):
            if rule["kind"] == "deny":
                deny_msgs.append(msg)
            elif rule["kind"] == "escalate":
                esc_msgs.append(msg)
            elif rule["kind"] == "allow":
                allow_matched = True

    if deny_msgs:
        return {"decision": "deny", "deny_reasons": deny_msgs, "escalate_reasons": esc_msgs}
    if esc_msgs:
        return {"decision": "escalate", "deny_reasons": [], "escalate_reasons": esc_msgs}
    if allow_matched:
        return {"decision": "allow", "deny_reasons": [], "escalate_reasons": []}
    return {"decision": "default", "deny_reasons": [], "escalate_reasons": []}


def rego_to_policy_config(rego_text: str, *, base_policy: Optional[Any] = None) -> Any:
    """Compile OPA Rego policy rules into a verified `PolicyConfig`."""
    from .layered_decision_engine import DEFAULT_POLICY, PolicyConfig
    from .policy_governance import verify_policy_patch

    base = base_policy if isinstance(base_policy, PolicyConfig) else DEFAULT_POLICY
    parsed = parse_rego_policy(rego_text)
    deny_rules: List[Dict[str, Any]] = []
    esc_rules: List[Dict[str, Any]] = []
    allow_rules: List[Dict[str, Any]] = []

    for idx, rule in enumerate(parsed.get("rules", []), start=1):
        msg = str(rule.get("message") or f"rego_{rule['kind']}_{idx}")
        c_rule: Dict[str, Any] = {
            "id": re.sub(r"[^a-z0-9_]+", "_", msg.lower()).strip("_") or f"rego_{rule['kind']}_{idx}",
            "action": rule["kind"],
            "reason": msg,
        }
        for c in rule.get("conditions", []):
            cs = str(c).strip()
            m_msg = re.match(r'^msg\s*:?=\s*"([^"]*)"$', cs)
            if m_msg:
                c_rule["reason"] = m_msg.group(1)
                c_rule["id"] = re.sub(r"[^a-z0-9_]+", "_", m_msg.group(1).lower()).strip("_") or c_rule["id"]
                continue
            m_tool = re.match(r'^input\.tool\s*==\s*"([^"]*)"$', cs)
            m_in = re.match(r"^input\.tool\s+in\s+\[([^\]]*)\]$", cs)
            m_sw = re.match(r'^startswith\(\s*input\.(?:cmd|command)\s*,\s*"([^"]*)"\s*\)$', cs)
            m_cn = re.match(r'^contains\(\s*input\.(?:cmd|command)\s*,\s*"([^"]*)"\s*\)$', cs)
            m_rm = re.match(r'^regex\.match\(\s*"([^"]*)"\s*,\s*input\.(?:cmd|command)\s*\)$', cs)
            m_gm = re.match(r'^glob\.match\(\s*"([^"]*)"\s*,\s*\[[^\]]*\]\s*,\s*input\.path\s*\)$', cs)
            m_peq = re.match(r'^input\.path\s*==\s*"([^"]*)"$', cs)
            if m_tool:
                c_rule["tool"] = re.escape(m_tool.group(1))
            elif m_in:
                toks = re.findall(r'"([^"]*)"', m_in.group(1))
                if toks:
                    c_rule["tool"] = "(?:" + "|".join(re.escape(t) for t in toks) + ")"
            elif m_sw:
                c_rule["cmd_regex"] = f"^{re.escape(m_sw.group(1))}"
            elif m_cn:
                c_rule["cmd_regex"] = re.escape(m_cn.group(1))
            elif m_rm:
                c_rule["cmd_regex"] = m_rm.group(1)
            elif m_gm:
                c_rule["path_glob"] = m_gm.group(1)
            elif m_peq:
                c_rule["path_glob"] = m_peq.group(1)
        if rule["kind"] == "deny":
            deny_rules.append(c_rule)
        elif rule["kind"] == "escalate":
            esc_rules.append(c_rule)
        elif rule["kind"] == "allow":
            allow_rules.append(c_rule)

    cfg = PolicyConfig.from_dict({
        "protected_patterns": list(base.protected_patterns),
        "extra_protected_patterns": list(base.extra_protected_patterns),
        "custom_rules": deny_rules + esc_rules + allow_rules,
        "profile": base.profile,
    })
    for ar in allow_rules:
        ver = verify_policy_patch(ar, base_policy=cfg)
        if not ver["ok"]:
            raise ValueError(f"Unsafe Rego allow rule {ar.get('id')!r}: {ver['issues']}")
    return cfg


def export_policy_to_rego(policy: Any) -> str:
    """Export `PolicyConfig.custom_rules` to OPA Rego v1 syntax."""
    lines: List[str] = [
        "# Exported by Aetheris Microbrains Policy-as-Code",
        "package aetheris.authz",
        "import rego.v1",
        "",
        "default allow := false",
        "",
    ]
    for idx, rule in enumerate(getattr(policy, "custom_rules", ()) or (), start=1):
        if not isinstance(rule, dict):
            continue
        act = str(rule.get("action") or "").lower()
        if act not in {"deny", "escalate", "allow"}:
            continue
        reason = str(rule.get("reason") or rule.get("id") or f"rule_{idx}").replace('"', "'")
        header = f'{act} contains msg if {{' if act in {"deny", "escalate"} else "allow if {"
        lines.append(header)
        if act in {"deny", "escalate"}:
            lines.append(f'    msg := "{reason}"')
        tool_pat = str(rule.get("tool") or "").strip()
        if tool_pat and tool_pat not in {".*", ".+"}:
            clean_tools = re.sub(r"^\(\?:(.*)\)$", r"\1", tool_pat).split("|")
            clean_tools = [t.replace("\\", "") for t in clean_tools if t]
            if len(clean_tools) == 1:
                lines.append(f'    input.tool == "{clean_tools[0]}"')
            elif clean_tools:
                arr = ", ".join(f'"{t}"' for t in clean_tools)
                lines.append(f"    input.tool in [{arr}]")
        if rule.get("path_glob"):
            pg = str(rule["path_glob"]).replace('"', "'")
            lines.append(f'    glob.match("{pg}", ["/"], input.path)')
        if rule.get("cmd_regex"):
            cr = str(rule["cmd_regex"]).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'    regex.match("{cr}", input.cmd)')
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
