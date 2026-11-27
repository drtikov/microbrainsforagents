"""Install, verify, or remove the native Aetheris Microbrains plugin for Google Antigravity.

Creates a self-contained Antigravity plugin bundle (`aetheris-microbrains`) containing:
- `plugin.json`: Antigravity plugin manifest (strictly valid top-level fields only)
- `hooks.json`: Named PreToolUse / PostToolUse hooks (`--host antigravity`)
- `mcp_config.json`: Optional on-demand Microbrains MCP tools (`--no-mcp` to omit)
- `rules/AGENTS.md`: Always-on guardrail notes (plain Markdown, no frontmatter)
- `skills/aetheris-microbrains/SKILL.md`: On-demand usage & audit workflow skill

Default target:
- Global:    `~/.gemini/config/plugins/aetheris-microbrains/`
- Workspace: `<workspace>/.agents/plugins/aetheris-microbrains/` (with `--workspace`)

Usage:
    python -m aetheris_microbrains.install_antigravity --shadow --profile dev
    python -m aetheris_microbrains.install_antigravity --profile dev
    python -m aetheris_microbrains.install_antigravity --workspace . --profile dev
    python -m aetheris_microbrains.install_antigravity --self-test
    python -m aetheris_microbrains.install_antigravity --remove
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

PLUGIN_NAME = "aetheris-microbrains"
PLUGIN_VERSION = "0.9.0"

ANTIGRAVITY_TOOL_MATCHER = (
    "run_command|edit_file|write_file|replace_file_content|"
    "multi_replace_file_content|write_to_file|read_file|view_file|"
    "grep_search|find_by_name|list_dir|mcp_.*"
)

_ALLOWED_PLUGIN_KEYS = frozenset({
    "name",
    "displayName",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "suggestedPrompts",
    "logo",
    "disabled",
})

_PY = sys.executable.replace("\\", "/")


def _quote_cmd_token(tok: str) -> str:
    s = str(tok).replace("\\", "/")
    if not s or any(ch in s for ch in (" ", "\t", "\n", '"', "'", "&", "|", ";", "<", ">", "(", ")")):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def find_rust_gate_binary() -> Optional[Path]:
    """Locate the native `aetheris-gate` Rust shim binary if installed or built locally."""
    which_bin = shutil.which("aetheris-gate")
    if which_bin:
        p = Path(which_bin)
        if p.is_file():
            return p
    exe_name = "aetheris-gate.exe" if os.name == "nt" else "aetheris-gate"
    repo_bin = Path(__file__).resolve().parents[1] / "rust" / "aetheris-gate" / "target" / "release" / exe_name
    if repo_bin.is_file():
        return repo_bin
    return None


def _hook_cmd(
    shared: bool,
    event_args: list[str],
    extra: list[str],
    *,
    use_rust_shim: bool = False,
) -> str:
    quoted_extra = [_quote_cmd_token(x) if not x.startswith("--") else x for x in extra]
    if use_rust_shim:
        rust_bin = find_rust_gate_binary()
        if rust_bin is not None:
            parts = [_quote_cmd_token(str(rust_bin)), "--host", "antigravity", *event_args, *quoted_extra]
            return " ".join(parts)
    py_tok = "python" if shared else _quote_cmd_token(_PY)
    parts = [py_tok, "-m", "aetheris_microbrains.hook", "--host", "antigravity", *event_args, *quoted_extra]
    return " ".join(parts)


def _atomic_write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_write_json(target: Path, data: Dict[str, Any]) -> None:
    _atomic_write_text(target, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _resolve_plugin_dir(
    *,
    plugin_dir: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> Path:
    if plugin_dir is not None:
        return Path(plugin_dir).expanduser().resolve()
    if workspace is not None:
        return Path(workspace).expanduser().resolve() / ".agents" / "plugins" / PLUGIN_NAME
    return Path.home() / ".gemini" / "config" / "plugins" / PLUGIN_NAME


def _build_plugin_manifest() -> Dict[str, Any]:
    return {
        "name": PLUGIN_NAME,
        "displayName": "Aetheris Microbrains",
        "version": PLUGIN_VERSION,
        "description": (
            "Deterministic sub-2ms pre-flight AST linter, secret-egress gate, "
            "context-hygiene guard, and forensic audit journal for Google Antigravity."
        ),
        "author": {
            "name": "Dmitry Tikov",
        },
        "homepage": "https://github.com/drtikov/microbrainsforagents",
        "repository": "https://github.com/drtikov/microbrainsforagents",
        "suggestedPrompts": [
            "Audit recent Microbrains decisions with aetheris-audit --last and explain any blocks",
            "Run Microbrains doctor and scan configured MCP tools for schema poisoning or rug pulls",
            "Check a proposed shell command or Python edit through microbrain_preflight_check before running it",
        ],
    }


def _build_hooks_config(
    *,
    strict_ast: bool = False,
    shadow: bool = False,
    shared: bool = False,
    auto_allow: bool = False,
    profile: Optional[str] = None,
    policy: Optional[str] = None,
    session_state: Optional[str] = "auto",
    journal_path: Optional[str] = None,
    context_guard: bool = False,
    approve_once: bool = False,
    auto_approve_escalations: bool = False,
    use_rust_shim: bool = False,
) -> Dict[str, Any]:
    pre_extra = ["--log-decisions"]
    post_extra = ["--log-decisions"]
    if strict_ast:
        pre_extra.append("--strict-ast")
    if shadow:
        pre_extra.append("--shadow")
        post_extra.append("--shadow")
    if auto_allow:
        pre_extra.append("--auto-allow")
    if auto_approve_escalations:
        pre_extra.append("--auto-approve-escalations")
        if not auto_allow:
            pre_extra.append("--auto-allow")
    if approve_once:
        pre_extra.append("--approve-once")
        post_extra.append("--approve-once")
    if context_guard:
        pre_extra.append("--context-guard")
        post_extra.append("--context-guard")
    if profile:
        pre_extra.extend(["--profile", profile])
    if policy:
        pre_extra.extend(["--policy", policy])
        post_extra.extend(["--policy", policy])
    if session_state and str(session_state).lower() != "none":
        pre_extra.extend(["--session-state", session_state])
        post_extra.extend(["--session-state", session_state])
    if journal_path:
        pre_extra.extend(["--journal-path", journal_path])
        post_extra.extend(["--journal-path", journal_path])

    return {
        PLUGIN_NAME: {
            "enabled": True,
            "PreToolUse": [
                {
                    "matcher": ANTIGRAVITY_TOOL_MATCHER,
                    "hooks": [
                        {
                            "type": "command",
                            "command": _hook_cmd(shared, [], pre_extra, use_rust_shim=use_rust_shim),
                            "timeout": 10,
                        }
                    ],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": ANTIGRAVITY_TOOL_MATCHER,
                    "hooks": [
                        {
                            "type": "command",
                            "command": _hook_cmd(shared, ["--event", "post-tool-use"], post_extra, use_rust_shim=use_rust_shim),
                            "timeout": 10,
                        }
                    ],
                }
            ],
        }
    }


def _build_mcp_config(*, shared: bool = False) -> Dict[str, Any]:
    cmd = "python" if shared else sys.executable
    return {
        "mcpServers": {
            PLUGIN_NAME: {
                "command": cmd,
                "args": ["-m", "aetheris_microbrains.mcp_server"],
            }
        }
    }


_RULES_AGENTS_MD = """# Aetheris Microbrains Pre-Flight Guardrails

Aetheris Microbrains enforces deterministic local safety and hygiene checks via `PreToolUse` and `PostToolUse` lifecycle hooks:

- **Pre-execution blocks (`deny`):** Syntax-broken Python edits/writes, destructive shell patterns, secret egress, or protected-path writes (`.env`, `.ssh/`, `.git/hooks/`) are blocked before execution with an explanation in `reason`.
- **High-scrutiny escalations (`force_ask`):** Ambiguous shell wrappers, external build targets, or unusual operations explicitly prompt for user confirmation.
- **Forensic explanations:** If a tool call is blocked and you need full diagnostic context or policy tuning suggestions, run `python -m aetheris_microbrains.audit --last` or `--explain`.
- **On-demand checks:** When the `aetheris-microbrains` MCP server is enabled, prefer `microbrain_preflight_check` before risky multi-step patches or commands, and `microbrain_scan_mcp_tools` when auditing third-party MCP servers.
"""

_SKILL_MD = """---
name: aetheris-microbrains
description: Deterministic pre-flight AST linter, secret-egress gate, context-hygiene guard, and forensic audit journal for Google Antigravity. Use when auditing blocked actions (`aetheris-audit --explain`), toggling `--shadow` or enforce profiles, scanning MCP tools (`aetheris-doctor --scan-mcp`), or calling `microbrain_*` MCP utilities.
---

# Aetheris Microbrains in Google Antigravity

Aetheris Microbrains protects Antigravity tool calls (`run_command`, `edit_file`, `write_file`, `replace_file_content`, `write_to_file`, and `mcp_*`) through native `PreToolUse` and `PostToolUse` hooks plus optional MCP server tools.

## 1. Inspect Recent Decisions & Explain Blocks

```bash
python -m aetheris_microbrains.audit --last
python -m aetheris_microbrains.audit --explain
python -m aetheris_microbrains.audit --digest --summary
```

## 2. Check Health & Scan Configured MCP Servers

```bash
python -m aetheris_microbrains.doctor --scan-mcp
python -m aetheris_microbrains.mcp_server --check
```

## 3. Reconfigure or Verify the Antigravity Plugin

```bash
# Shadow (observe-only) pilot
python -m aetheris_microbrains.install_antigravity --shadow --profile dev

# Active enforcement (global ~/.gemini/config/plugins/aetheris-microbrains/)
python -m aetheris_microbrains.install_antigravity --profile dev

# Verify contract & installed bundle files
python -m aetheris_microbrains.install_antigravity --self-test
```

## 4. Available On-Demand MCP Tools

When `mcp_config.json` is active (requires `pip install -e ".[mcp]"`):
- `microbrain_preflight_check` — dry-run shell commands, file writes, or edits before running them.
- `microbrain_scrub_and_scan` — redact secrets and check untrusted text for indirect prompt injection.
- `microbrain_audit_explain` — inspect recent journal blocks and rule rationales.
- `microbrain_scan_mcp_tools` — detect schema poisoning and rug-pull drift on registered MCP tools.
- `microbrain_narrow_tools` — sticky cache-safe tool surface routing for multi-turn agent loops.
- `microbrain_verify` / `microbrain_health` / `microbrain_directives` — offline verification and diagnostic status.
"""


def install(
    *,
    strict_ast: bool = False,
    shadow: bool = False,
    shared: bool = False,
    auto_allow: bool = False,
    profile: Optional[str] = None,
    policy: Optional[str] = None,
    session_state: Optional[str] = "auto",
    journal_path: Optional[str] = None,
    context_guard: bool = False,
    approve_once: bool = False,
    auto_approve_escalations: bool = False,
    use_rust_shim: bool = False,
    no_mcp: bool = False,
    workspace: Optional[Path] = None,
    plugin_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    target = _resolve_plugin_dir(plugin_dir=plugin_dir, workspace=workspace)
    manifest = _build_plugin_manifest()
    hooks_cfg = _build_hooks_config(
        strict_ast=strict_ast,
        shadow=shadow,
        shared=shared,
        auto_allow=auto_allow,
        profile=profile,
        policy=policy,
        session_state=session_state,
        journal_path=journal_path,
        context_guard=context_guard,
        approve_once=approve_once,
        auto_approve_escalations=auto_approve_escalations,
        use_rust_shim=use_rust_shim,
    )
    mcp_cfg = None if no_mcp else _build_mcp_config(shared=shared)

    if dry_run:
        print(f"DRY RUN - would write Antigravity plugin bundle to {target}:")
        print("--- plugin.json ---")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print("--- hooks.json ---")
        print(json.dumps(hooks_cfg, indent=2, ensure_ascii=False))
        if mcp_cfg is not None:
            print("--- mcp_config.json ---")
            print(json.dumps(mcp_cfg, indent=2, ensure_ascii=False))
        return 0

    _atomic_write_json(target / "plugin.json", manifest)
    _atomic_write_json(target / "hooks.json", hooks_cfg)
    mcp_file = target / "mcp_config.json"
    if mcp_cfg is not None:
        _atomic_write_json(mcp_file, mcp_cfg)
    elif mcp_file.exists():
        mcp_file.unlink()
    _atomic_write_text(target / "rules" / "AGENTS.md", _RULES_AGENTS_MD)
    _atomic_write_text(target / "skills" / PLUGIN_NAME / "SKILL.md", _SKILL_MD)

    mode = "shadow (observe-only)" if shadow else "enforce"
    scope = "workspace" if workspace is not None else "global"
    print(f"Installed Aetheris Antigravity plugin ({mode}, {scope}) -> {target}")
    return 0


def remove(
    *,
    workspace: Optional[Path] = None,
    plugin_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    target = _resolve_plugin_dir(plugin_dir=plugin_dir, workspace=workspace)
    if not target.exists():
        print(f"No Antigravity plugin at {target}; nothing to remove.")
        return 0
    if target.name != PLUGIN_NAME and not (target / "plugin.json").is_file():
        print(f"ERROR: Refusing to remove non-plugin directory {target}.", file=sys.stderr)
        return 1
    if dry_run:
        print(f"DRY RUN - would remove Antigravity plugin directory {target}")
        return 0
    shutil.rmtree(target)
    print(f"Removed Aetheris Antigravity plugin from {target}")
    return 0


def self_test(
    *,
    workspace: Optional[Path] = None,
    plugin_dir: Optional[Path] = None,
) -> int:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    failures: list[str] = []
    try:
        hook_mod = importlib.import_module("aetheris_microbrains.hook")
        deny_cmd = hook_mod.evaluate_hook_payload(
            {"toolCall": {"id": "probe_1", "name": "run_command", "args": {"CommandLine": "rm -rf /"}}},
            host="antigravity",
            log_decisions=False,
        )
        allow_cmd = hook_mod.evaluate_hook_payload(
            {"toolCall": {"id": "probe_2", "name": "run_command", "args": {"CommandLine": "git status"}}},
            host="antigravity",
            log_decisions=False,
        )
        ask_cmd = hook_mod.evaluate_hook_payload(
            {"toolCall": {"id": "probe_3", "name": "run_command", "args": {"CommandLine": "unrecognized_custom_bin --flag"}}},
            host="antigravity",
            log_decisions=False,
        )
        cred_read = hook_mod.evaluate_hook_payload(
            {"toolCall": {"id": "probe_3b", "name": "read_file", "args": {"Path": ".env"}}},
            host="antigravity",
            log_decisions=False,
        )
        bad_py = hook_mod.evaluate_hook_payload(
            {"toolCall": {"id": "probe_4", "name": "write_file", "args": {"Path": "broken.py", "Content": "def oops(:\n"}}},
            host="antigravity",
            log_decisions=False,
        )
        post_res = hook_mod.evaluate_post_tool_use_payload(
            {"toolCall": {"id": "probe_5", "name": "run_command", "args": {"CommandLine": "echo ok"}}, "output": "ok"},
            host="antigravity",
            log_decisions=False,
        )
        if not (
            deny_cmd.get("exit_code") == 0
            and deny_cmd.get("hookSpecificOutput", {}).get("decision") == "deny"
            and allow_cmd.get("exit_code") == 0
            and allow_cmd.get("hookSpecificOutput", {}).get("decision") == "allow"
            and ask_cmd.get("exit_code") == 0
            and ask_cmd.get("hookSpecificOutput", {}).get("decision") == "ask"
            and cred_read.get("exit_code") == 0
            and cred_read.get("hookSpecificOutput", {}).get("decision") == "force_ask"
            and bad_py.get("exit_code") == 0
            and bad_py.get("hookSpecificOutput", {}).get("decision") == "deny"
            and post_res.get("exit_code") == 0
            and post_res.get("hookSpecificOutput") == {}
        ):
            failures.append("antigravity-probe-contract")
    except Exception as exc:
        print(f"[FAIL] Antigravity contract probe failed: {exc}")
        failures.append("probe-error")

    target = _resolve_plugin_dir(plugin_dir=plugin_dir, workspace=workspace)
    if not target.is_dir():
        print(f"[FAIL] missing plugin directory {target}; run install_antigravity first")
        failures.append("not-installed")
    else:
        plugin_file = target / "plugin.json"
        hooks_file = target / "hooks.json"
        rules_file = target / "rules" / "AGENTS.md"
        skill_file = target / "skills" / PLUGIN_NAME / "SKILL.md"
        mcp_file = target / "mcp_config.json"

        try:
            pj = json.loads(plugin_file.read_text(encoding="utf-8"))
            if not isinstance(pj, dict) or pj.get("name") != PLUGIN_NAME:
                failures.append("plugin-name")
            if not set(pj.keys()).issubset(_ALLOWED_PLUGIN_KEYS):
                failures.append("plugin-unknown-keys")
        except Exception:
            failures.append("plugin-json")

        try:
            hj = json.loads(hooks_file.read_text(encoding="utf-8"))
            if not isinstance(hj, dict) or "hooks" in hj or PLUGIN_NAME not in hj:
                failures.append("hooks-schema")
            else:
                named = hj[PLUGIN_NAME]
                pre = named.get("PreToolUse") or []
                post = named.get("PostToolUse") or []
                if len(pre) != 1 or len(post) != 1:
                    failures.append("hooks-entries")
                else:
                    if pre[0].get("matcher") != ANTIGRAVITY_TOOL_MATCHER or post[0].get("matcher") != ANTIGRAVITY_TOOL_MATCHER:
                        failures.append("matcher-mismatch")
                    pre_cmd = pre[0].get("hooks", [{}])[0].get("command", "")
                    post_cmd = post[0].get("hooks", [{}])[0].get("command", "")
                    if "--host antigravity" not in pre_cmd or "--host antigravity" not in post_cmd:
                        failures.append("hooks-host-flag")
        except Exception:
            failures.append("hooks-json")

        if not rules_file.is_file() or rules_file.read_text(encoding="utf-8").lstrip().startswith("---"):
            failures.append("rules-agents-md")
        if not skill_file.is_file() or f"name: {PLUGIN_NAME}" not in skill_file.read_text(encoding="utf-8"):
            failures.append("skill-md")

        if mcp_file.exists():
            try:
                mj = json.loads(mcp_file.read_text(encoding="utf-8"))
                srv = mj.get("mcpServers", {}).get(PLUGIN_NAME, {})
                if not srv.get("command") or not isinstance(srv.get("args"), list):
                    failures.append("mcp-config")
            except Exception:
                failures.append("mcp-json")

    if failures:
        print(f"ANTIGRAVITY SELF-TEST FAILED: {', '.join(failures)}")
        return 1
    print(f"ANTIGRAVITY SELF-TEST PASSED ({target})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict-ast", action="store_true", help="Enable strict AST import/dynamic-call allowlist")
    ap.add_argument("--shadow", action="store_true", help="Observe-only pilot mode: log without blocking")
    ap.add_argument("--auto-allow", action="store_true", help="Emit explicit decision='allow' on safe verdicts")
    ap.add_argument("--approve-once", action="store_true", help="Remember eligible escalated commands/tools after approving once")
    ap.add_argument("--auto-approve-escalations", action="store_true", help="Auto-approve non-destructive escalations without UI confirmation prompts")
    ap.add_argument("--clear-approvals", action="store_true", help="Clear stored --approve-once escalations and exit")
    ap.add_argument("--profile", choices=["strict", "dev"], default=None, help="Policy profile (`strict` or `dev`)")
    ap.add_argument("--policy", default=None, help="Path to JSON/TOML/YAML PolicyConfig file")
    ap.add_argument("--session-state", default="auto", help="SessionSecurityState path or 'auto' (default: auto)")
    ap.add_argument("--no-session-state", action="store_true", help="Disable SessionSecurityState tracking")
    ap.add_argument("--journal-path", default=None, help="Override path for decision_journal.jsonl")
    ap.add_argument("--context-guard", action="store_true", help="Enable context-hygiene gating")
    ap.add_argument("--no-mcp", action="store_true", help="Omit mcp_config.json from the plugin bundle")
    ap.add_argument("--use-rust-shim", action="store_true", help="Use native `aetheris-gate` Rust binary when built/on PATH")
    ap.add_argument("--shared", action="store_true", help="Use portable `python` executable instead of current interpreter path")
    ap.add_argument("--workspace", default=None, help="Install to `<workspace>/.agents/plugins/aetheris-microbrains/`")
    ap.add_argument("--plugin-dir", default=None, help="Explicit target directory for the Antigravity plugin bundle")
    ap.add_argument("--dry-run", action="store_true", help="Preview generated files without writing")
    ap.add_argument("--self-test", action="store_true", help="Verify Antigravity contract and installed plugin bundle")
    ap.add_argument("--remove", action="store_true", help="Remove the installed Antigravity plugin directory")
    args = ap.parse_args(argv)

    ws = Path(args.workspace) if args.workspace else None
    pd = Path(args.plugin_dir) if args.plugin_dir else None

    if args.self_test:
        return self_test(workspace=ws, plugin_dir=pd)
    if args.clear_approvals:
        from .hook import clear_approvals

        cleared = clear_approvals()
        print(f"Cleared {cleared} stored escalation approval(s).")
        return 0
    if args.remove:
        return remove(workspace=ws, plugin_dir=pd, dry_run=args.dry_run)

    sess_arg = None if args.no_session_state else args.session_state
    return install(
        strict_ast=args.strict_ast,
        shadow=args.shadow,
        shared=args.shared,
        auto_allow=args.auto_allow,
        approve_once=args.approve_once,
        auto_approve_escalations=args.auto_approve_escalations,
        use_rust_shim=args.use_rust_shim,
        profile=args.profile,
        policy=args.policy,
        session_state=sess_arg,
        journal_path=args.journal_path,
        context_guard=args.context_guard,
        no_mcp=args.no_mcp,
        workspace=ws,
        plugin_dir=pd,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
