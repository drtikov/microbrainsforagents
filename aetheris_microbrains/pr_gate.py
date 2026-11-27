"""CI/CD Pre-Merge PR Security Gate & SARIF 2.1.0 Reporter (`aetheris-pr-gate`, stdlib-only).

Inspects unified diffs (`git diff` or `--diff-file`) or repository file paths for:
  - Protected path modifications (`.env`, `.ssh/`, `.git/hooks/`, credentials)
  - Python AST & multi-language security sinks (JS/TS, Go, Rust, SQL, Terraform, Dockerfile)
  - Hardcoded / concatenated secrets
  - Deferred-execution supply-chain hazards (`package.json` `postinstall`, GitHub Actions, `conftest.py`, `*.pth`)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core.layered_decision_engine import PolicyConfig, analyze_code_diff, analyze_shell_command, is_protected_path
from .core.mcp_security import detect_prompt_injection, scan_mcp_tools

_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "target", ".pytest_cache"})
_BINARY_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".pdf", ".zip", ".exe", ".dll", ".pyd", ".so", ".npz"})
_SKILL_MD_NAMES = frozenset({"skill.md", "agents.md", "claude.md"})
_MANIFEST_JSON_NAMES = frozenset({".mcp.json", "mcp.json", "hooks.json"})
_HIDDEN_UNICODE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\U000e0000-\U000e007f]")


def split_unified_diff_by_file(diff_text: str) -> List[Tuple[str, str]]:
    """Split a multi-file `git diff` string into `(file_path, diff_chunk)` pairs."""
    lines = str(diff_text or "").splitlines(keepends=True)
    chunks: List[Tuple[str, str]] = []
    cur_path: Optional[str] = None
    cur_lines: List[str] = []

    def _flush() -> None:
        if cur_path and cur_path != "/dev/null" and cur_lines:
            chunks.append((cur_path, "".join(cur_lines)))

    for ln in lines:
        if ln.startswith("diff --git "):
            _flush()
            cur_lines = [ln]
            m = re.match(r"^diff --git\s+a/\S+\s+b/(.+?)\s*$", ln.rstrip("\r\n"))
            cur_path = m.group(1) if m else None
            continue
        if ln.startswith("+++ "):
            raw_p = ln[4:].strip()
            if raw_p == "/dev/null":
                cur_path = "/dev/null"
            elif raw_p.startswith("b/"):
                cur_path = raw_p[2:]
            elif not cur_path:
                cur_path = raw_p
        cur_lines.append(ln)
    _flush()
    return chunks


def _first_hunk_line(chunk: str) -> int:
    m = re.search(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)", chunk, re.MULTILINE)
    if m:
        return max(1, int(m.group(1)))
    return 1


def _extract_scannable_text(content_or_diff: str) -> str:
    raw = str(content_or_diff or "")
    if raw.startswith("diff --git ") or "\ndiff --git " in raw or re.search(r"^@@\s+-\d+", raw, re.MULTILINE):
        added = [ln[1:] for ln in raw.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        return "\n".join(added)
    return raw


def scan_skill_or_mcp_manifest(
    file_path: str,
    content_or_diff: str,
    *,
    line_hint: int = 1,
) -> List[Dict[str, Any]]:
    """Statically audit skill Markdown docs and MCP/hook JSON manifests for smuggling, injection, or supply-chain risks."""
    norm_path = str(file_path or "").replace("\\", "/").strip().lower()
    basename = norm_path.rsplit("/", 1)[-1]
    is_skill_doc = basename in _SKILL_MD_NAMES or ("/skills/" in norm_path and norm_path.endswith(".md"))
    is_manifest_json = basename in _MANIFEST_JSON_NAMES
    if not (is_skill_doc or is_manifest_json):
        return []

    text = _extract_scannable_text(content_or_diff)
    if not text.strip():
        return []

    findings: List[Dict[str, Any]] = []
    if _HIDDEN_UNICODE_RE.search(text):
        findings.append({
            "file": file_path,
            "rule_id": "skill_hidden_unicode_smuggling",
            "severity": "error",
            "decision": "deny",
            "message": f"Hidden zero-width/bidi/tag Unicode characters detected in {file_path}",
            "line": line_hint,
        })

    inj = detect_prompt_injection(text)
    if is_skill_doc and inj.get("injection_detected"):
        inj_names = ", ".join(str(x) for x in (inj.get("findings") or [])[:4]) or "prompt_injection"
        findings.append({
            "file": file_path,
            "rule_id": "skill_prompt_injection",
            "severity": "error",
            "decision": "deny",
            "message": f"Indirect prompt injection or covert exfiltration in {file_path} ({inj_names})",
            "line": line_hint,
        })

    if is_manifest_json:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            tools_list: List[Dict[str, Any]] = []
            if isinstance(parsed.get("tools"), list):
                tools_list.extend(t for t in parsed["tools"] if isinstance(t, dict))
            servers = parsed.get("mcpServers")
            if isinstance(servers, dict):
                for srv_name, srv_cfg in servers.items():
                    if not isinstance(srv_cfg, dict):
                        continue
                    desc = str(srv_cfg.get("description") or "")
                    if desc:
                        tools_list.append({"name": str(srv_name), "description": desc})
                    cmd = str(srv_cfg.get("command") or "").strip()
                    raw_args = srv_cfg.get("args")
                    args_list = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
                    full_cmd = shlex.join([cmd, *args_list]) if cmd else ""
                    if full_cmd:
                        sh_res = analyze_shell_command(full_cmd)
                        sh_findings = sh_res.get("findings") or []
                        if sh_findings or sh_res.get("risk_class") == "IRREVERSIBLE_DESTRUCTIVE":
                            findings.append({
                                "file": file_path,
                                "rule_id": "mcp_manifest_supply_chain_hazard",
                                "severity": "error",
                                "decision": "deny",
                                "message": f"Dangerous MCP server command in {file_path} ({srv_name}): {', '.join(str(x) for x in sh_findings) or sh_res.get('risk_class')}",
                                "line": line_hint,
                            })
            if tools_list:
                mcp_scan = scan_mcp_tools(tools_list)
                if not mcp_scan.get("safe", True) or mcp_scan.get("flagged_tools"):
                    findings.append({
                        "file": file_path,
                        "rule_id": "mcp_manifest_supply_chain_hazard",
                        "severity": "error",
                        "decision": "deny",
                        "message": f"Poisoned or suspicious MCP tool definition in {file_path}: {', '.join(str(x) for x in (mcp_scan.get('findings') or [])[:4])}",
                        "line": line_hint,
                    })
    return findings


def _scan_one(
    file_path: str,
    content_or_diff: str,
    *,
    policy: PolicyConfig,
    original_source: Optional[str] = None,
    line_hint: int = 1,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if is_protected_path(file_path, policy):
        findings.append({
            "file": file_path,
            "rule_id": "layer0_protected_path_veto",
            "severity": "error",
            "decision": "deny",
            "message": f"PR modifies protected path: {file_path}",
            "line": line_hint,
        })

    res = analyze_code_diff(
        file_path,
        content_or_diff,
        original_source=original_source,
        policy=policy,
        allow_suppression=bool(policy.allow_inline_suppression),
    )
    if not res.get("syntax_valid", True) and not res.get("partial_hunk_needs_source"):
        findings.append({
            "file": file_path,
            "rule_id": "layer1_ast_syntax_error",
            "severity": "error",
            "decision": "deny",
            "message": f"Syntax validation failed for {file_path} ({res.get('language', 'unknown')})",
            "line": line_hint,
        })

    sevs = res.get("finding_severities") or {}
    for raw_f in res.get("findings") or ():
        if raw_f == "syntax_tree_malformed":
            continue
        is_med = sevs.get(raw_f) == "medium" and str(policy.profile).lower() == "dev"
        findings.append({
            "file": file_path,
            "rule_id": str(raw_f),
            "severity": "warning" if is_med else "error",
            "decision": "escalate" if is_med else "deny",
            "message": f"{raw_f} detected in {file_path}",
            "line": line_hint,
        })
    findings.extend(scan_skill_or_mcp_manifest(file_path, content_or_diff, line_hint=line_hint))
    return findings


def _summarize_findings(files_scanned: int, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    deny_n = sum(1 for f in findings if f["decision"] == "deny")
    esc_n = sum(1 for f in findings if f["decision"] == "escalate")
    return {
        "passed": deny_n == 0 and esc_n == 0,
        "files_scanned": files_scanned,
        "deny_count": deny_n,
        "escalate_count": esc_n,
        "findings": findings,
    }


def scan_pr_diff(
    diff_text: str,
    *,
    policy: Optional[PolicyConfig] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan a unified `git diff` across all modified files."""
    cfg = policy or PolicyConfig()
    root = Path(repo_root) if repo_root else None
    chunks = split_unified_diff_by_file(diff_text)
    all_findings: List[Dict[str, Any]] = []
    for fpath, chunk in chunks:
        orig_src: Optional[str] = None
        if root is not None and not is_protected_path(fpath, cfg):
            cand = root / fpath
            try:
                if cand.is_file() and cand.stat().st_size <= 512 * 1024:
                    orig_src = cand.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                orig_src = None
        all_findings.extend(
            _scan_one(
                fpath,
                chunk,
                policy=cfg,
                original_source=orig_src,
                line_hint=_first_hunk_line(chunk),
            )
        )
    return _summarize_findings(len(chunks), all_findings)


def scan_pr_paths(
    paths: Sequence[Path | str],
    *,
    policy: Optional[PolicyConfig] = None,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan explicit file or directory paths on disk (for CI checkouts or directories)."""
    cfg = policy or PolicyConfig()
    base = Path(root_dir) if root_dir else Path.cwd()
    files: List[Path] = []
    for raw_p in paths:
        p = Path(raw_p)
        if not p.is_absolute():
            p = base / p
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and not any(part in _SKIP_DIRS for part in child.parts):
                    files.append(child)

    all_findings: List[Dict[str, Any]] = []
    scanned = 0
    for f in files:
        if f.suffix.lower() in _BINARY_EXTS:
            continue
        try:
            if f.stat().st_size > 512 * 1024:
                continue
            rel = str(f.relative_to(base)).replace("\\", "/") if base in f.parents or f == base else str(f).replace("\\", "/")
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        all_findings.extend(_scan_one(rel, text, policy=cfg))
    return _summarize_findings(scanned, all_findings)


def to_sarif_report(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert PR scan result into GitHub-compatible SARIF 2.1.0 JSON."""
    findings = scan_result.get("findings") or []
    rules_map: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    for f in findings:
        rid = str(f.get("rule_id") or "aetheris_finding")
        if rid not in rules_map:
            rules_map[rid] = {
                "id": rid,
                "name": rid,
                "shortDescription": {"text": f"Aetheris Microbrains rule `{rid}`"},
                "defaultConfiguration": {"level": f.get("severity", "error")},
            }
        results.append({
            "ruleId": rid,
            "level": str(f.get("severity") or "error"),
            "message": {"text": str(f.get("message") or rid)},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": str(f.get("file") or "unknown")},
                        "region": {"startLine": max(1, int(f.get("line") or 1))},
                    }
                }
            ],
        })
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Aetheris Microbrains PR Gate",
                        "version": "0.7.7",
                        "informationUri": "https://github.com/drtikov/microbrainsforagents",
                        "rules": list(rules_map.values()),
                    }
                },
                "results": results,
            }
        ],
    }


def format_markdown_report(scan_result: Dict[str, Any]) -> str:
    """Render concise Markdown summary suitable for GitHub PR comments."""
    passed = bool(scan_result.get("passed"))
    badge = "PASS" if passed else "BLOCK"
    lines = [
        f"## Aetheris Microbrains PR Security Gate — `{badge}`",
        "",
        f"- **Files scanned:** {scan_result.get('files_scanned', 0)}",
        f"- **Deny findings:** {scan_result.get('deny_count', 0)}",
        f"- **Escalate findings:** {scan_result.get('escalate_count', 0)}",
    ]
    findings = scan_result.get("findings") or []
    if findings:
        lines.extend(["", "| File | Line | Rule | Decision |", "|---|---:|---|---|"])
        for f in findings[:50]:
            lines.append(f"| `{f['file']}` | {f['line']} | `{f['rule_id']}` | `{f['decision']}` |")
    return "\n".join(lines) + "\n"


def _atomic_write_json(target: Path, data: Dict[str, Any]) -> None:
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aetheris-pr-gate",
        description="Pre-merge PR security gate for AI-authored diffs (outputs text/markdown/json/SARIF 2.1.0).",
    )
    parser.add_argument("--diff-file", default="", help="Path to unified diff file")
    parser.add_argument("--diff-stdin", action="store_true", help="Read unified diff from stdin")
    parser.add_argument("--base", default="", help="Git base ref for `git diff <base>...<head>`")
    parser.add_argument("--head", default="HEAD", help="Git head ref (default: HEAD)")
    parser.add_argument("--paths", nargs="*", default=[], help="Files or directories to scan directly")
    parser.add_argument("--policy", default="", help="Optional PolicyConfig file (.json/.toml/.yaml/.cedar/.rego)")
    parser.add_argument("--profile", choices=["strict", "dev"], default="strict", help="Policy profile")
    parser.add_argument("--format", choices=["text", "markdown", "json", "sarif"], default="text")
    parser.add_argument("--sarif-out", default="", help="Write SARIF 2.1.0 report to this file path")
    parser.add_argument("--fail-on", choices=["deny", "any"], default="any", help="Exit 1 threshold")
    args = parser.parse_args(argv)

    cfg = PolicyConfig.from_file(args.policy) if args.policy else PolicyConfig(profile=args.profile)
    if args.profile and not args.policy:
        cfg.profile = args.profile

    if args.diff_file:
        res = scan_pr_diff(Path(args.diff_file).read_text(encoding="utf-8", errors="ignore"), policy=cfg)
    elif args.diff_stdin:
        res = scan_pr_diff(sys.stdin.read(), policy=cfg)
    elif args.base:
        proc = subprocess.run(  # aetheris: allow subprocess_call:run -- read git diff range for PR gate
            ["git", "diff", f"{args.base}...{args.head}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if proc.returncode != 0:
            print(f"aetheris-pr-gate git diff failed: {proc.stderr.strip()}", file=sys.stderr)
            return 2
        res = scan_pr_diff(proc.stdout, policy=cfg, repo_root=Path.cwd())
    elif args.paths:
        res = scan_pr_paths(args.paths, policy=cfg)
    else:
        parser.print_help()
        return 2

    if args.sarif_out:
        _atomic_write_json(Path(args.sarif_out), to_sarif_report(res))

    if args.format == "json":
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.format == "sarif":
        print(json.dumps(to_sarif_report(res), indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(format_markdown_report(res), end="")
    else:
        print(
            f"PR_GATE passed={res['passed']} files={res['files_scanned']} "
            f"deny={res['deny_count']} escalate={res['escalate_count']}"
        )
        for f in res["findings"]:
            print(f"  [{f['decision'].upper()}] {f['file']}:{f['line']} {f['rule_id']} - {f['message']}")

    if args.fail_on == "deny":
        return 0 if res["deny_count"] == 0 else 1
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
