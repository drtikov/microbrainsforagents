"""Installation and runtime health check for Aetheris Microbrains.

Static and in-process only: no subprocesses, no writes (except explicit
`--pin-mcp`/`--unpin-mcp`), one optional localhost TCP probe for the hook
daemon. Exit 0 when every check passes, 1 when any check fails (warnings
do not fail the run).

Usage:
    python -m aetheris_microbrains.doctor
    python -m aetheris_microbrains.doctor --json
    python -m aetheris_microbrains.doctor --pin-mcp <server>    # pins to current .mcp.json config
    python -m aetheris_microbrains.doctor --unpin-mcp <server>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.mcp_security import scan_mcp_tools
from .core.mcp_tiers import (
    get_pin_store_path,
    load_mcp_config,
    load_mcp_pins,
    remove_mcp_pin,
    save_mcp_pin,
    server_pin_fingerprint,
    verify_mcp_manifest,
)
from .core.runtime_paths import get_default_journal_path, get_state_dir
from .core.secrets import redact_secrets

_SETTINGS_LOCAL = Path(".claude") / "settings.local.json"
_SETTINGS_SHARED = Path(".claude") / "settings.json"
_EXPECTED_PRE_MATCHER = "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
_EXPECTED_POST_MATCHER = "Bash|Read|Grep|WebFetch|WebSearch|mcp__.*"


class _Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.checks.append({"status": status, "name": name, "detail": detail})

    def ok(self, name: str, detail: str = "") -> None:
        self.add("ok", name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.add("warn", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.add("FAIL", name, detail)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)

    def render(self) -> str:
        lines: List[str] = ["=" * 62, "AETHERIS MICROBRAINS - DOCTOR", "=" * 62]
        icon = {"ok": "[ ok ]", "warn": "[warn]", "FAIL": "[FAIL]"}
        for c in self.checks:
            line = f"{icon[c['status']]} {c['name']}"
            if c["detail"]:
                line += f" - {c['detail']}"
            lines.append(line)
        lines.append("-" * 62)
        n_fail = sum(1 for c in self.checks if c["status"] == "FAIL")
        n_warn = sum(1 for c in self.checks if c["status"] == "warn")
        lines.append(f"{'DOCTOR FAILED' if self.failed else 'DOCTOR PASSED'}: "
                     f"{n_fail} failed, {n_warn} warning(s), {len(self.checks)} checks")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": not self.failed, "checks": self.checks}


def _check_imports(rep: _Report) -> None:
    v = sys.version_info
    if v >= (3, 10):
        rep.ok("python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        rep.fail("python", f"{v.major}.{v.minor}.{v.micro} (needs >= 3.10)")
    try:
        from . import __version__ as pkg_version
        from . import hook as _hook_mod  # noqa: F401 - the import itself proves the module loads
        from .core import layered_decision_engine as _lde  # noqa: F401

        rep.ok("package", f"aetheris-microbrains {pkg_version}")
        rep.ok("import:hook")
        rep.ok("import:layered_decision_engine")
    except Exception as exc:
        rep.fail("package", f"import failed: {exc} (fix: pip install -e .)")


def _is_aetheris_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if "aetheris_microbrains.hook" in str(entry.get("command", "")):
        return True
    if entry.get("type") == "http" and "/v1/hook/" in str(entry.get("url", "")):
        return True
    return any(
        _is_aetheris_entry(h) for h in entry.get("hooks", []) if isinstance(h, dict)
    )


def _check_settings(rep: _Report, project_dir: Path) -> None:
    claude_dir = project_dir / ".claude"
    target: Optional[Path] = None
    for cand in (claude_dir / _SETTINGS_LOCAL.name, claude_dir / _SETTINGS_SHARED.name):
        if cand.is_file():
            target = cand
            break
    if target is None:
        user_global = Path.home() / ".claude" / "settings.json"
        if user_global.is_file():
            try:
                ug_data = json.loads(user_global.read_text(encoding="utf-8"))
                ug_hooks = ug_data.get("hooks", {}) if isinstance(ug_data, dict) else {}
                ug_has_aetheris = isinstance(ug_hooks, dict) and any(
                    _is_aetheris_entry(e)
                    for ev in ("PreToolUse", "PostToolUse")
                    for e in (ug_hooks.get(ev, []) if isinstance(ug_hooks.get(ev, []), list) else [])
                )
                if ug_has_aetheris:
                    rep.warn(
                        "settings:user_global",
                        f"global hooks found in {user_global}, but no project settings under {claude_dir}",
                    )
            except Exception:
                pass
        rep.fail("settings", f"no settings file under {claude_dir}; run install_hook.py")
        return
    try:
        settings = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("top-level JSON is not an object")
        rep.ok("settings", str(target))
    except Exception as exc:
        rep.fail("settings", f"{target}: invalid JSON ({exc})")
        return

    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        rep.fail("hooks", f"{target}: expected 'hooks' object, got {type(hooks).__name__}")
        return
    for event, expected_matcher in (
        ("PreToolUse", _EXPECTED_PRE_MATCHER),
        ("PostToolUse", _EXPECTED_POST_MATCHER),
    ):
        ev_list = hooks.get(event, [])
        if not isinstance(ev_list, list):
            rep.fail(f"hooks:{event}", f"expected list, got {type(ev_list).__name__}")
            continue
        entries = [e for e in ev_list if _is_aetheris_entry(e)]
        if len(entries) != 1:
            rep.fail(f"hooks:{event}", f"expected exactly 1 aetheris entry, found {len(entries)}")
            continue
        entry = entries[0]
        matcher = str(entry.get("matcher") or "")
        if matcher == expected_matcher:
            rep.ok(f"hooks:{event}", "matcher matches installer default")
        else:
            rep.warn(f"hooks:{event}", f"matcher {matcher!r} differs from installer default")
        inner_list = entry.get("hooks")
        inner = inner_list[0] if isinstance(inner_list, list) and inner_list else {}
        if event == "PreToolUse":
            onf = inner.get("onFailure")
            if onf == "block":
                rep.ok("hooks:PreToolUse.onFailure", "block (honored by Claude Code >= 2.1.295; older versions ignore it)")
            elif onf is None:
                rep.warn("hooks:PreToolUse.onFailure", "absent - hook failures fail open on this Claude Code version")
            else:
                rep.warn("hooks:PreToolUse.onFailure", f"unexpected value {onf!r}")
        if "--shadow" in str(inner.get("command") or ""):
            rep.warn("mode", "hooks installed in shadow mode (observe-only, never blocks)")


def _check_policy(rep: _Report, policy_path: Optional[str]) -> None:
    from .core.layered_decision_engine import PolicyConfig

    resolved = str(policy_path or os.getenv("AETHERIS_POLICY_PATH") or "").strip()
    if not resolved:
        rep.ok("policy", "none configured (defaults; strict profile)")
        return
    try:
        cfg = PolicyConfig.from_file(resolved)
        rep.ok(
            "policy",
            f"{resolved} (profile={cfg.profile}, strict_ast={cfg.strict_ast_allowlist}, "
            f"trusted_mcp={len(tuple(cfg.trusted_mcp_servers or ()))}, "
            f"allowed_web_domains={len(tuple(cfg.allowed_web_domains or ()))})",
        )
    except Exception as exc:
        rep.fail("policy", f"{resolved}: {exc}")


def _check_journal(rep: _Report, state_dir: Optional[Path] = None) -> None:
    env_override = bool(str(os.getenv("AETHERIS_JOURNAL_PATH") or "").strip())
    if state_dir is not None and not env_override:
        journal = Path(state_dir) / "decision_journal.jsonl"
    else:
        journal = get_default_journal_path()
    try:
        stat = journal.stat()
        age_min = max(0.0, (time.time() - stat.st_mtime) / 60.0)
        rep.ok(
            "journal",
            f"{journal} ({stat.st_size / 1024:.1f} KiB, last write {age_min:.0f} min ago"
            f"{', env override' if env_override else ''})",
        )
        rotated = journal.with_name(journal.name + ".1")
        if rotated.exists():
            rep.ok("journal:rotated", f"{rotated.name} present ({rotated.stat().st_size / 1024:.1f} KiB)")
    except OSError:
        rep.warn("journal", f"{journal} not created yet (no decisions logged)")


def _check_mcp_pins(rep: _Report, project_dir: Path, state_dir: Optional[Path] = None) -> None:
    pins_path = get_pin_store_path(state_dir)
    servers_cfg = load_mcp_config(project_dir)
    pins = load_mcp_pins(state_dir)
    if servers_cfg:
        rep.ok("mcp-config", f"{len(servers_cfg)} server(s) in .mcp.json: {', '.join(sorted(servers_cfg))}")
    else:
        rep.ok("mcp-config", "no .mcp.json (no local MCP servers)")
    if not pins:
        rep.warn("mcp-pins", f"none pinned ({pins_path}); MCP tools fail closed to ask")
        return
    stale = []
    missing_cfg = []
    for name, entry in sorted(pins.items()):
        if name not in servers_cfg:
            missing_cfg.append(name)
            continue
        expected = entry.get("fingerprint") if isinstance(entry, dict) else None
        if not expected or expected != server_pin_fingerprint(servers_cfg.get(name)):
            stale.append(name)
    detail = f"{len(pins)} pinned: {', '.join(sorted(pins))}"
    issues: List[str] = []
    if stale:
        issues.append(f"STALE (config drifted, treated as unpinned): {', '.join(stale)}")
    if missing_cfg:
        issues.append(f"MISSING from .mcp.json (treated as unpinned): {', '.join(missing_cfg)}")
    if issues:
        rep.warn("mcp-pins", f"{detail}; " + "; ".join(issues))
    else:
        rep.ok("mcp-pins", detail)


def _check_daemon(rep: _Report, state_dir: Optional[Path] = None) -> None:
    from .hook_daemon import DAEMON_VERSION, ping_daemon

    base = Path(state_dir) if state_dir else get_state_dir()
    daemon_file = base / "hook_daemon.json"
    try:
        info = json.loads(daemon_file.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError("daemon state is not a JSON object")
    except Exception:
        rep.ok("daemon", "not running (hook falls back to in-process evaluation)")
        return
    port = int(info.get("port") or 0)
    raw_ver = info.get("version")
    ver = str(raw_ver if raw_ver is not None else "?")
    has_auth = bool(str(info.get("auth") or "").strip())
    mode = str(info.get("mode") or "tcp").lower()
    pong = ping_daemon(state_path=daemon_file, timeout=0.3) if port else None
    alive = isinstance(pong, dict) and bool(pong.get("ok"))
    if not alive:
        rep.warn("daemon", f"stale daemon file (mode={mode}, version {ver}, port {port} unreachable); hook falls back to in-process")
    elif raw_ver != DAEMON_VERSION or not has_auth:
        rep.warn(
            "daemon",
            f"running on 127.0.0.1:{port} with incompatible version/auth (got version={ver}, expected={DAEMON_VERSION}); "
            "clients fall back to in-process",
        )
    else:
        rep.ok("daemon", f"running ({mode}) on 127.0.0.1:{port} (version {ver})")


def scan_mcp_command(
    project_dir: Path,
    manifest_file: Optional[Path] = None,
    server: Optional[str] = None,
    state_dir: Optional[Path] = None,
) -> int:
    """Audit `.mcp.json` server definitions and optional tool schemas for poisoning/rug-pulls."""
    servers_cfg = load_mcp_config(project_dir)
    issues: List[str] = []
    print("=" * 62)
    print("AETHERIS MICROBRAINS - MCP SUPPLY-CHAIN SCAN")
    print("=" * 62)
    if not servers_cfg:
        print(f"[info] no .mcp.json servers found under {project_dir}")
    for srv_name, srv_cfg in sorted(servers_cfg.items()):
        if server and srv_name != server.strip().lower():
            continue
        cfg_warnings: List[str] = []
        if isinstance(srv_cfg, dict):
            url = str(srv_cfg.get("url") or "").strip().lower()
            if url.startswith("http://") and not url.startswith(("http://127.0.0.1", "http://localhost")):
                cfg_warnings.append("unencrypted non-loopback http:// MCP URL")
            cmd_line = " ".join([str(srv_cfg.get("command") or ""), *[str(a) for a in (srv_cfg.get("args") or [])]])
            if "@latest" in cmd_line or "-y " in cmd_line and "@" not in cmd_line:
                cfg_warnings.append("unpinned npx/package invocation (use pinned @version)")
            env_map = srv_cfg.get("env")
            if isinstance(env_map, dict):
                raw_env = json.dumps(env_map)
                if redact_secrets(raw_env) != raw_env:
                    cfg_warnings.append("raw credential pattern in .mcp.json env block")
        status_tag = "[warn]" if cfg_warnings else "[ ok ]"
        detail = "; ".join(cfg_warnings) if cfg_warnings else "config hygiene clean"
        print(f"{status_tag} server={srv_name} - {detail}")
        if cfg_warnings:
            issues.extend(f"{srv_name}:{w}" for w in cfg_warnings)

    if manifest_file is not None:
        mpath = Path(manifest_file)
        try:
            raw_m = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[FAIL] failed to parse manifest {mpath}: {exc}", file=sys.stderr)
            return 1
        tools = raw_m.get("tools", raw_m) if isinstance(raw_m, dict) else raw_m
        if not isinstance(tools, list):
            print(f"[FAIL] manifest {mpath} must contain a tool list", file=sys.stderr)
            return 1
        scan = scan_mcp_tools(tools)
        if not scan["safe"]:
            print(f"[FAIL] tool poisoning/shadowing detected: {scan['findings']}")
            for ft in scan["flagged_tools"]:
                print(f"       - {ft['name']}: {ft['findings']}")
            return 1
        print(f"[ ok ] manifest scan passed ({scan['tool_count']} tool(s), zero injections)")
        if server:
            v_res = verify_mcp_manifest(server, tools, state_dir=state_dir)
            if v_res.get("rug_pull_detected"):
                print(
                    f"[FAIL] MCP rug pull detected for {server!r}: modified={v_res['modified_tools']}, "
                    f"added={v_res['added_tools']}, removed={v_res['removed_tools']}"
                )
                return 1
            elif v_res.get("verified"):
                print(f"[ ok ] schema pin verified for {server!r}")
    return 1 if any("raw credential" in i or "unencrypted" in i for i in issues) else 0


def pin_command(server: str, unpin: bool, project_dir: Path, state_dir: Optional[Path] = None) -> int:
    """Operator pin management; the only doctor mode that writes to disk."""
    server = str(server or "").strip()
    if not server:
        print("ERROR: --pin-mcp/--unpin-mcp needs a server name.", file=sys.stderr)
        return 1
    if unpin:
        pins = load_mcp_pins(state_dir)
        if server.lower() not in pins:
            print(f"ERROR: MCP server {server!r} is not pinned.", file=sys.stderr)
            return 1
        remove_mcp_pin(server, state_dir=state_dir)
        print(f"Unpinned MCP server {server!r}.")
        return 0
    servers_cfg = load_mcp_config(project_dir)
    if server.lower() not in {k.lower() for k in servers_cfg}:
        print(
            f"ERROR: server {server!r} not found in {project_dir / '.mcp.json'}; "
            "pinning binds to the server's current config fingerprint.",
            file=sys.stderr,
        )
        return 1
    cfg_key = next(k for k in servers_cfg if k.lower() == server.lower())
    save_mcp_pin(server, servers_cfg.get(cfg_key), state_dir=state_dir)
    print(f"Pinned MCP server {server!r} to its current .mcp.json fingerprint.")
    return 0


def run_doctor(
    project_dir: Optional[Path] = None,
    policy_path: Optional[str] = None,
    state_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all health checks and return the structured report."""
    rep = _Report()
    root = Path(project_dir) if project_dir else Path.cwd()
    _check_imports(rep)
    _check_settings(rep, root)
    _check_policy(rep, policy_path)
    _check_journal(rep, state_dir=state_dir)
    _check_mcp_pins(rep, root, state_dir=state_dir)
    _check_daemon(rep, state_dir=state_dir)
    return rep.to_dict()


def _render_report(report: Dict[str, Any]) -> str:
    rep = _Report()
    rep.checks = list(report["checks"])
    return rep.render()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aetheris_microbrains.doctor",
        description="Health-check the Aetheris Microbrains gate installation.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    parser.add_argument("--project-dir", default=None, help="Project root containing .claude/ (defaults to CWD)")
    parser.add_argument("--state-dir", default=None, help="Override state directory (defaults to AETHERIS_STATE_DIR / OS state dir)")
    parser.add_argument("--policy", default=None, help="Policy file to validate (defaults to AETHERIS_POLICY_PATH)")
    parser.add_argument("--pin-mcp", default=None, metavar="SERVER", help="Pin an MCP server to its current .mcp.json config")
    parser.add_argument("--unpin-mcp", default=None, metavar="SERVER", help="Remove an MCP server pin")
    parser.add_argument("--scan-mcp", action="store_true", help="Audit .mcp.json configs and optional --manifest-file schemas")
    parser.add_argument("--manifest-file", default=None, help="JSON MCP tools/list manifest file to scan/verify with --scan-mcp")
    parser.add_argument("--server", default=None, help="Filter --scan-mcp or verify pinned manifest against a specific server name")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    root = Path(args.project_dir) if args.project_dir else Path.cwd()
    sdir = Path(args.state_dir) if args.state_dir else None
    if args.scan_mcp:
        return scan_mcp_command(
            root,
            manifest_file=Path(args.manifest_file) if args.manifest_file else None,
            server=args.server,
            state_dir=sdir,
        )
    if args.pin_mcp or args.unpin_mcp:
        return pin_command(args.pin_mcp or args.unpin_mcp or "", bool(args.unpin_mcp), root, state_dir=sdir)

    report = run_doctor(project_dir=root, policy_path=args.policy, state_dir=sdir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
