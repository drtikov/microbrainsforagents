"""Deferred Execution & Supply-Chain Integrity Guard (stdlib-only).

Detects poisoned lifecycle hooks and delayed execution vectors that coding agents
may plant in build, package, IDE, Python auto-load, or CI/CD configuration files:
  - `package.json` (`preinstall` / `postinstall` / `prepare` / unsafe `scripts`)
  - `.vscode/tasks.json` (`runOn: "folderOpen"`, unsafe shell commands)
  - `devcontainer.json` (`postCreateCommand`, `postStartCommand`, etc.)
  - `Makefile` / `GNUmakefile` / `Justfile` (`$(shell ...)`, multiline `curl | sh`)
  - `conftest.py`, `sitecustomize.py`, `usercustomize.py`, `*.pth`
  - `.github/workflows/*.yml`, `.gitlab-ci.yml`, `action.yml` (`pull_request_target`
    pwn-request, `${{ github.event... }}` script injection, unsafe `run:` commands)
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PKG_LIFECYCLE_HOOKS = frozenset({
    "preinstall",
    "install",
    "postinstall",
    "prepare",
    "preprepare",
    "postprepare",
    "prepublish",
    "prepublishonly",
    "prepack",
    "postpack",
})

_PKG_SUSPICIOUS_LIFECYCLE_RE = re.compile(
    r"(?i)(?:\bcurl\b|\bwget\b|\binvoke-webrequest\b|\biwr\s|\birm\s|\bnc\s|\bnetcat\b|"
    r"\b(?:node|nodejs)\s+-e\b|\bpython\d*\s+-c\b|\b(?:bash|sh|zsh)\s+-c\b|"
    r"\bpowershell(?:\.exe)?\b|\bbase64\b|\beval\s|\|\s*(?:sh|bash|zsh)\b|https?://)"
)

_MAKE_SHELL_MACRO_RE = re.compile(r"\$[({]shell\s+([^)}]+)[)}]", re.IGNORECASE)
_GHA_UNTRUSTED_EXPR_RE = re.compile(
    r"\$\{\{\s*github\.(?:event\.(?:issue\.(?:title|body)|pull_request\.(?:title|body|head\.(?:ref|label|repo\.default_branch))|"
    r"comment\.body|review\.body|review_comment\.body|discussion\.(?:title|body)|head_commit\.message|"
    r"commits\[[^\]]+\]\.message)|head_ref)\s*\}\}"
)
_GHA_HEAD_REF_RE = re.compile(r"github\.event\.pull_request\.head\.(?:sha|ref)|github\.head_ref")


def _join_backslash_continuations(text: str) -> List[str]:
    out: List[str] = []
    buf = ""
    for ln in str(text or "").splitlines():
        stripped = ln.rstrip()
        if stripped.endswith("\\"):
            buf += " " + stripped[:-1].strip()
            continue
        if buf:
            out.append((buf + " " + stripped.strip()).strip())
            buf = ""
        else:
            out.append(ln)
    if buf.strip():
        out.append(buf.strip())
    return out


def _check_shell_cmd(cmd: str, policy: Any = None) -> bool:
    """Return True if shell command passes `analyze_shell_command` safely."""
    from .layered_decision_engine import analyze_shell_command

    res = analyze_shell_command(cmd, policy=policy)
    return bool(res.get("is_safe", True))


def _check_package_json(text: str, policy: Any = None) -> List[str]:
    findings: List[str] = []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return []
    for name, raw_cmd in scripts.items():
        if not isinstance(raw_cmd, str) or not raw_cmd.strip():
            continue
        hook_name = str(name).strip().lower()
        safe = _check_shell_cmd(raw_cmd, policy=policy)
        if hook_name in _PKG_LIFECYCLE_HOOKS:
            if not safe or _PKG_SUSPICIOUS_LIFECYCLE_RE.search(raw_cmd):
                findings.append("deferred_exec_package_json_lifecycle_hook")
        elif not safe:
            findings.append("deferred_exec_package_json_unsafe_script")
    return findings


def _check_vscode_tasks(text: str, policy: Any = None) -> List[str]:
    findings: List[str] = []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else [data]
    for task in tasks:
        if not isinstance(task, dict):
            continue
        run_opts = task.get("runOptions")
        if isinstance(run_opts, dict) and str(run_opts.get("runOn") or "").strip().lower() == "folderopen":
            findings.append("deferred_exec_ide_autorun_task")
        cmds: List[str] = []
        for sub in (task, task.get("windows"), task.get("osx"), task.get("linux")):
            if not isinstance(sub, dict):
                continue
            cmd = sub.get("command")
            args = sub.get("args")
            if isinstance(cmd, str) and cmd.strip():
                if isinstance(args, list) and args:
                    cmds.append(cmd + " " + " ".join(str(a) for a in args))
                else:
                    cmds.append(cmd)
        for c in cmds:
            if not _check_shell_cmd(c, policy=policy):
                findings.append("deferred_exec_ide_unsafe_task")
    return findings


def _check_devcontainer(text: str, policy: Any = None) -> List[str]:
    findings: List[str] = []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    for key in ("initializeCommand", "onCreateCommand", "updateContentCommand", "postCreateCommand", "postStartCommand"):
        val = data.get(key)
        cmd = " ".join(str(x) for x in val) if isinstance(val, list) else (val if isinstance(val, str) else "")
        if cmd.strip() and (not _check_shell_cmd(cmd, policy=policy) or _PKG_SUSPICIOUS_LIFECYCLE_RE.search(cmd)):
            findings.append("deferred_exec_devcontainer_unsafe_hook")
    return findings


def _check_makefile(text: str, policy: Any = None) -> List[str]:
    findings: List[str] = []
    for m in _MAKE_SHELL_MACRO_RE.finditer(text):
        inner = m.group(1).strip()
        if not _check_shell_cmd(inner, policy=policy) or _PKG_SUSPICIOUS_LIFECYCLE_RE.search(inner):
            findings.append("deferred_exec_makefile_shell_macro")
    for ln in _join_backslash_continuations(text):
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[A-Za-z0-9_.\-/]+:\s*(?:[^=]|$)", stripped) and ";" not in stripped:
            continue
        cmd = stripped.lstrip("@-+").strip()
        if cmd and not _check_shell_cmd(cmd, policy=policy):
            findings.append("deferred_exec_makefile_unsafe_recipe")
    return findings


def _check_python_autoload(text: str) -> List[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    net_mods = {"urllib", "requests", "httpx", "socket", "http"}
    exec_mods = {"subprocess", "os", "pty", "ctypes"}
    imported_risky: set[str] = set()
    aliases: Dict[str, str] = {}

    def _walk_top(stmt: ast.AST) -> bool:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if isinstance(stmt, ast.Import):
            for al in stmt.names:
                top = al.name.split(".")[0].lower()
                if top in net_mods or top in exec_mods:
                    aliases[al.asname or top] = top
                    imported_risky.add(top)
            return False
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            top = stmt.module.split(".")[0].lower()
            if top in net_mods or top in exec_mods:
                for al in stmt.names:
                    aliases[al.asname or al.name] = f"{top}.{al.name}"
                    imported_risky.add(top)
            return False
        if isinstance(stmt, ast.Call):
            fn = stmt.func
            if isinstance(fn, ast.Attribute):
                parts: List[str] = []
                cur: ast.AST = fn
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(aliases.get(cur.id, cur.id))
                    chain = ".".join(reversed(parts)).lower()
                    root_mod = chain.split(".", 1)[0]
                    if root_mod in {"subprocess", "requests", "httpx", "urllib"}:
                        return True
                    if chain in {"os.system", "os.popen", "socket.create_connection", "socket.connect"}:
                        return True
            elif isinstance(fn, ast.Name) and fn.id in aliases:
                target = aliases[fn.id].lower()
                if any(target.startswith(p) for p in ("urllib.", "requests.", "httpx.", "subprocess.", "os.system", "os.popen")):
                    return True
        for child in ast.iter_child_nodes(stmt):
            if _walk_top(child):
                return True
        return False

    for top_stmt in tree.body:
        if _walk_top(top_stmt):
            return ["deferred_exec_python_autoload_hook"]
    return []


def _check_ci_workflow(text: str, policy: Any = None) -> List[str]:
    findings: List[str] = []
    if re.search(r"\bpull_request_target\b", text) and "actions/checkout" in text and _GHA_HEAD_REF_RE.search(text):
        findings.append("gha_pwn_request_hazard")

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        m_run = re.match(r"^(\s*(?:-\s*)?)run\s*:\s*(.*)$", ln)
        if m_run:
            base_indent = len(ln) - len(ln.lstrip(" "))
            rest = m_run.group(2).strip()
            run_lines: List[str] = []
            if rest in {"|", "|-", "|+", ">", ">-", ">+"}:
                i += 1
                while i < len(lines):
                    cur = lines[i]
                    if cur.strip() and (len(cur) - len(cur.lstrip(" "))) <= base_indent:
                        break
                    if cur.strip():
                        run_lines.append(cur.strip())
                    i += 1
            else:
                if rest:
                    run_lines.append(rest)
                i += 1
            block = "\n".join(run_lines)
            if _GHA_UNTRUSTED_EXPR_RE.search(block):
                findings.append("gha_script_injection")
            for cmd_ln in run_lines:
                if not _check_shell_cmd(cmd_ln, policy=policy):
                    findings.append("ci_workflow_unsafe_run_cmd")
                    break
            continue
        i += 1
    return findings


def analyze_deferred_execution(
    file_path: Optional[str],
    content_text: str,
    policy: Any = None,
) -> List[str]:
    """Detect deferred execution hazards in build, package, IDE, Python autoload, and CI files."""
    path_str = str(file_path or "").strip()
    src = str(content_text or "")
    if not path_str or not src.strip():
        return []

    norm = path_str.replace("\\", "/").lower()
    fname = Path(norm).name
    findings: List[str] = []

    if fname == "package.json":
        findings.extend(_check_package_json(src, policy=policy))
    elif fname == "tasks.json":
        findings.extend(_check_vscode_tasks(src, policy=policy))
    elif fname == "devcontainer.json":
        findings.extend(_check_devcontainer(src, policy=policy))
    elif fname in {"makefile", "gnumakefile", "justfile"}:
        findings.extend(_check_makefile(src, policy=policy))
    elif fname.endswith(".pth"):
        for ln in src.splitlines():
            st = ln.strip()
            if st and not st.startswith("#") and st.startswith("import "):
                findings.append("deferred_exec_pth_startup_exec")
                break
    elif fname in {"conftest.py", "sitecustomize.py", "usercustomize.py"}:
        findings.extend(_check_python_autoload(src))
    elif (
        ".github/workflows/" in norm
        or fname in {".gitlab-ci.yml", ".gitlab-ci.yaml", "action.yml", "action.yaml"}
    ) and norm.endswith((".yml", ".yaml")):
        findings.extend(_check_ci_workflow(src, policy=policy))

    return list(dict.fromkeys(findings))
