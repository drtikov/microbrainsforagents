#!/usr/bin/env python
"""Idempotent installer for Aetheris PreToolUse/PostToolUse hooks in Claude settings.

Merges the hook entries into `.claude/settings.local.json` (default, gitignored)
or `.claude/settings.json` (`--shared`) without clobbering unrelated configuration.
`--remove` strips only the aetheris hook entries.

Usage:
    python install_hook.py [--strict-ast] [--shadow] [--shared] [--auto-allow]
                           [--profile strict|dev] [--policy FILE]
                           [--session-state FILE] [--journal-path FILE]
                           [--project-dir DIR] [--async-post]
                           [--preset solo-dev|ci|paranoid] [--no-auto-allow]
                           [--no-async-post] [--dry-run] [--self-test] [--remove]

Presets (`--preset`):
    solo-dev   dev profile + auto-allow + async post (low-friction local work)
    ci         strict profile, enforce mode, synchronous post (deterministic CI)
    paranoid   strict profile + strict AST + enforce (maximum scrutiny)
Explicit flags always override the preset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SKILL_DIR.parents[2]
SETTINGS_LOCAL_PATH = PROJECT_DIR / ".claude" / "settings.local.json"
SETTINGS_SHARED_PATH = PROJECT_DIR / ".claude" / "settings.json"

PRE_TOOL_MATCHER = "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
POST_TOOL_MATCHER = "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|WebFetch|WebSearch|mcp__.*"
FULL_LIFECYCLE_SPECS: tuple[tuple[str, str, str | None], ...] = (
    ("SessionStart", "session-start", "startup|resume|compact"),
    ("InstructionsLoaded", "instructions-loaded", None),
    ("UserPromptSubmit", "user-prompt-submit", None),
    ("PostToolUseFailure", "post-tool-use-failure", "Bash|Edit|Write|MultiEdit|NotebookEdit|mcp__.*"),
    ("PostToolBatch", "post-tool-batch", None),
    ("PreCompact", "pre-compact", None),
    ("SubagentStop", "subagent-stop", None),
    ("Stop", "stop", None),
    ("PermissionDenied", "permission-denied", None),
)
_ALL_MANAGED_HOOK_EVENTS: tuple[str, ...] = ("PreToolUse", "PostToolUse") + tuple(
    ev for ev, _slug, _m in FULL_LIFECYCLE_SPECS
)

# Forward slashes + double quotes: hooks run through Git Bash on Windows or
# sh / cmd.exe; single-quoted paths break under cmd.exe and unescaped backslashes
# are eaten as escapes under Git Bash.
_PY = sys.executable.replace("\\", "/")


def _quote_cmd_token(tok: str) -> str:
    """Double-quote a token when it contains spaces or shell metacharacters."""
    s = str(tok).replace("\\", "/")
    if not s or any(ch in s for ch in (' ', '\t', '\n', '"', "'", '&', '|', ';', '<', '>', '(', ')')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _hook_cmd(shared: bool, event_args: list[str], extra: list[str]) -> list[str]:
    py_tok = "python" if shared else _quote_cmd_token(_PY)
    quoted_extra = [_quote_cmd_token(x) if not x.startswith("--") else x for x in extra]
    return [py_tok, "-m", "aetheris_microbrains.hook", *event_args, *quoted_extra]


def _hook_entry(
    cmd: list[str],
    timeout: int = 10,
    is_async: bool = False,
    on_failure: str | None = None,
) -> dict:
    entry: dict = {
        "type": "command",
        "command": " ".join(cmd),
        "timeout": timeout,
    }
    if on_failure is not None:
        entry["onFailure"] = on_failure
    if is_async:
        entry["async"] = True
    return entry


def _http_hook_entry(
    url: str,
    timeout: int = 5,
    is_async: bool = False,
    on_failure: str | None = None,
    token_env: str = "AETHERIS_DAEMON_TOKEN",
) -> dict:
    entry: dict = {
        "type": "http",
        "url": url,
        "timeout": timeout,
        "headers": {"Authorization": f"Bearer ${token_env}"},
        "allowedEnvVars": [token_env],
    }
    if on_failure is not None:
        entry["onFailure"] = on_failure
    if is_async:
        entry["async"] = True
    return entry


def _is_aetheris_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if "aetheris_microbrains.hook" in str(entry.get("command", "")):
        return True
    if entry.get("type") == "http" and "/v1/hook/" in str(entry.get("url", "")):
        return True
    # Matcher wrappers nest the command under "hooks"; recurse to identify wrappers.
    return any(
        _is_aetheris_entry(h) for h in entry.get("hooks", []) if isinstance(h, dict)
    )


def _strip_aetheris_from_event_list(entries: list) -> list:
    """Remove only Aetheris hook entries while preserving unrelated user hooks in shared wrappers."""
    kept: list = []
    for entry in entries:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        inner = entry.get("hooks")
        if isinstance(inner, list):
            filtered = [h for h in inner if not _is_aetheris_entry(h)]
            if filtered:
                updated = dict(entry)
                updated["hooks"] = filtered
                kept.append(updated)
        elif not _is_aetheris_entry(entry):
            kept.append(entry)
    return kept


def _atomic_write_json(target: Path, data: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2) + "\n")
            fh.flush()
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_validated_settings(target: Path) -> tuple[dict | None, int]:
    """Load and validate Claude settings JSON structure; return `(settings_dict, exit_code)`."""
    if not target.exists():
        return {}, 0
    try:
        settings = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: {target} is not valid JSON ({exc}); refusing to overwrite.", file=sys.stderr)
        return None, 1
    if not isinstance(settings, dict):
        print(f"ERROR: {target} top-level JSON must be an object; refusing to overwrite.", file=sys.stderr)
        return None, 1
    if "hooks" in settings:
        hooks = settings["hooks"]
        if not isinstance(hooks, dict):
            print(f"ERROR: {target} 'hooks' field must be an object; refusing to overwrite.", file=sys.stderr)
            return None, 1
        for ev in _ALL_MANAGED_HOOK_EVENTS:
            if ev in hooks and not isinstance(hooks[ev], list):
                print(f"ERROR: {target} 'hooks.{ev}' must be a list; refusing to overwrite.", file=sys.stderr)
                return None, 1
    return settings, 0


def install(
    strict_ast: bool,
    shadow: bool,
    shared: bool = False,
    settings_path: Path | None = None,
    auto_allow: bool = False,
    profile: str | None = None,
    policy: str | None = None,
    session_state: str | None = "auto",
    journal_path: str | None = None,
    async_post: bool = False,
    project_dir: Path | None = None,
    dry_run: bool = False,
    http_daemon: str | None = None,
    context_guard: bool = False,
    full_lifecycle: bool = False,
) -> int:
    if settings_path is not None:
        target = settings_path
    elif project_dir is not None:
        target = Path(project_dir) / ".claude" / ("settings.json" if shared else "settings.local.json")
    else:
        target = SETTINGS_SHARED_PATH if shared else SETTINGS_LOCAL_PATH

    settings, rc = _load_validated_settings(target)
    if rc != 0 or settings is None:
        return rc

    pre_extra = ["--log-decisions"]
    post_extra = ["--log-decisions"]
    if strict_ast:
        pre_extra.append("--strict-ast")
    if shadow:
        pre_extra.append("--shadow")
        post_extra.append("--shadow")
    if auto_allow:
        pre_extra.append("--auto-allow")
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

    hooks = settings.setdefault("hooks", {})
    pre = _strip_aetheris_from_event_list(hooks.get("PreToolUse", []))
    post = _strip_aetheris_from_event_list(hooks.get("PostToolUse", []))
    if http_daemon:
        ignored_flags: list[str] = []
        if strict_ast:
            ignored_flags.append("--strict-ast")
        if auto_allow:
            ignored_flags.append("--auto-allow")
        if profile:
            ignored_flags.append(f"--profile {profile}")
        if policy:
            ignored_flags.append(f"--policy {policy}")
        if context_guard:
            ignored_flags.append("--context-guard")
        if journal_path:
            ignored_flags.append(f"--journal-path {journal_path}")
        if ignored_flags:
            print(
                f"WARNING: --http-daemon ignores CLI policy flags ({', '.join(ignored_flags)}); "
                "pass those flags to `python -m aetheris_microbrains.hook_daemon --http ...` instead.",
                file=sys.stderr,
            )
        base_url = str(http_daemon).rstrip("/")
        pre.append({
            "matcher": PRE_TOOL_MATCHER,
            "hooks": [_http_hook_entry(f"{base_url}/v1/hook/pre", on_failure=None if shadow else "block")],
        })
        post.append({
            "matcher": POST_TOOL_MATCHER,
            "hooks": [_http_hook_entry(f"{base_url}/v1/hook/post", is_async=async_post)],
        })
    else:
        pre.append({
            "matcher": PRE_TOOL_MATCHER,
            "hooks": [_hook_entry(_hook_cmd(shared, [], pre_extra), on_failure=None if shadow else "block")],
        })
        post.append({
            "matcher": POST_TOOL_MATCHER,
            "hooks": [_hook_entry(_hook_cmd(shared, ["--event", "post-tool-use"], post_extra), is_async=async_post)],
        })
    hooks["PreToolUse"] = pre
    hooks["PostToolUse"] = post

    if full_lifecycle:
        lc_extra = ["--log-decisions"]
        if shadow:
            lc_extra.append("--shadow")
        if session_state and str(session_state).lower() != "none":
            lc_extra.extend(["--session-state", session_state])
        if journal_path:
            lc_extra.extend(["--journal-path", journal_path])
        base_url = str(http_daemon).rstrip("/") if http_daemon else ""
        for ev_name, ev_slug, ev_matcher in FULL_LIFECYCLE_SPECS:
            ev_list = _strip_aetheris_from_event_list(hooks.get(ev_name, []))
            inner_entry = (
                _http_hook_entry(f"{base_url}/v1/hook/event/{ev_slug}", timeout=5)
                if http_daemon
                else _hook_entry(_hook_cmd(shared, ["--event", ev_slug], lc_extra), timeout=10)
            )
            wrapper: dict = {"hooks": [inner_entry]}
            if ev_matcher is not None:
                wrapper["matcher"] = ev_matcher
            ev_list.append(wrapper)
            hooks[ev_name] = ev_list

    if dry_run:
        print(f"DRY RUN - would write to {target}:")
        print(json.dumps(settings, indent=2))
        return 0

    _atomic_write_json(target, settings)
    mode = "shadow (observe-only)" if shadow else "enforce"
    ast_mode = "strict-AST" if strict_ast else "default-AST"
    scope = "shared" if shared else "local"
    transport = f"http:{http_daemon}" if http_daemon else "command"
    print(f"Installed Aetheris hooks ({mode}, {ast_mode}, {scope}, {transport}) -> {target}")
    print('Probe:  python -c "import json,aetheris_microbrains as mb; print(mb.evaluate_hook_payload({\'tool_name\':\'Bash\',\'tool_input\':{\'command\':\'rm -rf /\'}}, log_decisions=False)[\'decision\'])"')
    return 0


def remove(
    shared: bool = False,
    settings_path: Path | None = None,
    project_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    if settings_path is not None:
        target = settings_path
    elif project_dir is not None:
        target = Path(project_dir) / ".claude" / ("settings.json" if shared else "settings.local.json")
    else:
        target = SETTINGS_SHARED_PATH if shared else SETTINGS_LOCAL_PATH

    if not target.exists():
        print(f"No {target.name}; nothing to remove.")
        return 0
    settings, rc = _load_validated_settings(target)
    if rc != 0 or settings is None:
        return rc
    hooks = settings.get("hooks", {})
    for event in _ALL_MANAGED_HOOK_EVENTS:
        if event in hooks:
            hooks[event] = _strip_aetheris_from_event_list(hooks[event])
    if dry_run:
        print(f"DRY RUN - would write to {target}:")
        print(json.dumps(settings, indent=2))
        return 0
    _atomic_write_json(target, settings)
    print(f"Removed Aetheris hooks from {target}")
    return 0


def resolve_preset(
    preset: str | None,
    strict_ast: bool | None,
    shadow: bool | None,
    auto_allow: bool | None,
    async_post: bool | None,
    profile: str | None,
) -> tuple[bool, bool, bool, bool, str | None]:
    """Merge a `--preset` bundle with explicitly-passed flags (explicit always wins)."""
    if preset == "solo-dev":
        profile = profile or "dev"
        auto_allow = True if auto_allow is None else auto_allow
        async_post = True if async_post is None else async_post
    elif preset == "ci":
        profile = profile or "strict"
        async_post = False if async_post is None else async_post
    elif preset == "paranoid":
        profile = profile or "strict"
        strict_ast = True if strict_ast is None else strict_ast
    return bool(strict_ast), bool(shadow), bool(auto_allow), bool(async_post), profile


def self_test(project_dir: Path | None = None) -> int:
    """In-process installation health check: imports, probe verdicts, installed settings.

    Static and in-process only: no subprocesses, no writes, no network.
    """
    import importlib

    # `python install_hook.py` puts the skill dir (not the repo root) on sys.path,
    # so add the project root where the aetheris_microbrains package lives.
    repo_root = str(PROJECT_DIR if project_dir is None else Path(project_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    failures: list[str] = []
    print("=" * 60)
    print("AETHERIS MICROBRAINS - INSTALLER SELF-TEST")
    print("=" * 60)

    v = sys.version_info
    ok = v >= (3, 10)
    print(f"[{'ok' if ok else 'FAIL'}] Python {v.major}.{v.minor}.{v.micro} (needs >= 3.10)")
    if not ok:
        failures.append("python-version")

    try:
        hook_mod = importlib.import_module("aetheris_microbrains.hook")
        print("[ok] aetheris_microbrains.hook importable")
    except Exception as exc:
        print(f"[FAIL] aetheris_microbrains.hook import failed: {exc}")
        print("       Fix: `pip install -e .` from the repo (or `pip install aetheris-microbrains`)")
        hook_mod = None
        failures.append("import")

    if hook_mod is not None:
        try:
            deny_v = hook_mod.evaluate_hook_payload(
                {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
                log_decisions=False,
            )
            allow_v = hook_mod.evaluate_hook_payload(
                {"tool_name": "Bash", "tool_input": {"command": "git status"}},
                log_decisions=False,
            )
            ok = deny_v.get("decision") == "deny" and allow_v.get("decision") == "allow"
            print(f"[{'ok' if ok else 'FAIL'}] probe verdicts: rm -> {deny_v.get('decision')}, git status -> {allow_v.get('decision')}")
            if not ok:
                failures.append("probe-verdicts")
        except Exception as exc:
            print(f"[FAIL] in-process probe raised: {exc}")
            failures.append("probe-crash")

    proj = Path(project_dir) if project_dir else PROJECT_DIR
    target = None
    for cand in (proj / ".claude" / "settings.local.json", proj / ".claude" / "settings.json"):
        if cand.is_file():
            target = cand
            break
    if target is None:
        print(f"[FAIL] no settings file under {proj / '.claude'}; run the installer first")
        failures.append("not-installed")
    else:
        settings, rc = _load_validated_settings(target)
        if rc != 0 or settings is None:
            print(f"[FAIL] {target.name} failed JSON/structure validation")
            failures.append("settings-invalid")
        else:
            pre = [e for e in settings.get("hooks", {}).get("PreToolUse", []) if _is_aetheris_entry(e)]
            post = [e for e in settings.get("hooks", {}).get("PostToolUse", []) if _is_aetheris_entry(e)]
            ok = len(pre) == 1 and len(post) == 1
            print(f"[{'ok' if ok else 'FAIL'}] hook entries in {target.name}: pre={len(pre)} post={len(post)}")
            if not ok:
                failures.append("hook-entries")
            if pre:
                matcher_ok = pre[0].get("matcher") == PRE_TOOL_MATCHER
                print(f"[{'ok' if matcher_ok else 'FAIL'}] PreToolUse matcher: {pre[0].get('matcher')!r}")
                if not matcher_ok:
                    failures.append("pre-matcher")
                inner_list = pre[0].get("hooks")
                inner = inner_list[0] if isinstance(inner_list, list) and inner_list else {}
                print(f"[info] PreToolUse onFailure={inner.get('onFailure')!r} "
                      f"(honored by Claude Code >= 2.1.295; older versions ignore it)")
            if post:
                post_matcher_ok = post[0].get("matcher") == POST_TOOL_MATCHER
                print(f"[{'ok' if post_matcher_ok else 'FAIL'}] PostToolUse matcher: {post[0].get('matcher')!r}")
                if not post_matcher_ok:
                    failures.append("post-matcher")

    print("-" * 60)
    if failures:
        print(f"SELF-TEST FAILED: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict-ast", action=argparse.BooleanOptionalAction, default=None, help="Enable strict AST import/dynamic-call allowlist (negate with --no-strict-ast)")
    ap.add_argument("--shadow", action=argparse.BooleanOptionalAction, default=None, help="Observe-only: log decisions, never block (negate with --no-shadow)")
    ap.add_argument("--auto-allow", action=argparse.BooleanOptionalAction, default=None, help="Emit permissionDecision='allow' on safe verdicts (negate with --no-auto-allow)")
    ap.add_argument("--shared", action="store_true", help="Target shared .claude/settings.json with portable `python`")
    ap.add_argument("--profile", choices=["strict", "dev"], default=None, help="Policy profile (`strict` or `dev`)")
    ap.add_argument("--policy", default=None, help="Path to JSON/TOML/YAML PolicyConfig file")
    ap.add_argument("--session-state", default="auto", help="Path (or 'auto') for cross-turn SessionSecurityState JSON (default: auto)")
    ap.add_argument("--no-session-state", action="store_true", help="Disable cross-turn SessionSecurityState tracking")
    ap.add_argument("--http-daemon", default=None, metavar="URL", help="Install native Claude Code `type: http` hooks targeting URL (e.g. http://127.0.0.1:8765)")
    ap.add_argument("--journal-path", default=None, help="Override path for decision_journal.jsonl")
    ap.add_argument("--project-dir", default=None, help="Target project root containing `.claude/` (defaults to repo root)")
    ap.add_argument("--async-post", action=argparse.BooleanOptionalAction, default=None, help="Run PostToolUse asynchronously instead of blocking before output ingestion (negate with --no-async-post)")
    ap.add_argument("--remove", action="store_true", help="Uninstall the Aetheris hooks")
    ap.add_argument("--dry-run", action="store_true", help="Print the settings that would be written; modify nothing")
    ap.add_argument("--preset", choices=["solo-dev", "ci", "paranoid"], default=None, help="Opinionated flag bundle (explicit flags still override)")
    ap.add_argument("--context-guard", action=argparse.BooleanOptionalAction, default=False, help="Enable deterministic context-hygiene gating for redundant/oversized reads (negate with --no-context-guard)")
    ap.add_argument("--full-lifecycle", action="store_true", help="Also register Phase 1 lifecycle hooks (SessionStart, InstructionsLoaded, UserPromptSubmit, PostToolUseFailure, PostToolBatch, PreCompact, SubagentStop, Stop, PermissionDenied)")
    ap.add_argument("--self-test", action="store_true", help="In-process installation checks (imports, probe verdicts, installed settings)")
    args = ap.parse_args(argv)
    proj = Path(args.project_dir) if args.project_dir else None
    if args.self_test:
        return self_test(project_dir=proj)
    if args.remove:
        return remove(shared=args.shared, project_dir=proj, dry_run=args.dry_run)
    strict_ast, shadow, auto_allow, async_post, profile = resolve_preset(
        args.preset, args.strict_ast, args.shadow, args.auto_allow, args.async_post, args.profile
    )
    sess_arg = None if args.no_session_state else args.session_state
    return install(
        strict_ast,
        shadow,
        shared=args.shared,
        auto_allow=auto_allow,
        profile=profile,
        policy=args.policy,
        session_state=sess_arg,
        journal_path=args.journal_path,
        async_post=async_post,
        project_dir=proj,
        dry_run=args.dry_run,
        http_daemon=args.http_daemon,
        context_guard=bool(args.context_guard),
        full_lifecycle=bool(args.full_lifecycle),
    )


if __name__ == "__main__":
    raise SystemExit(main())
