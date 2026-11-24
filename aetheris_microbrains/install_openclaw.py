"""Verify, print setup, or merge config for the OpenClaw native plugin bundle (stdlib-only)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

BUNDLE_DIR_NAME = "openclaw-aetheris-microbrains"
PLUGIN_ID = "aetheris-microbrains"
PACKAGE_NAME = "@aetheris/openclaw-microbrains"
DEFAULT_DAEMON_PORT = 8765
_REQUIRED_FILES = (
    "openclaw.plugin.json",
    "package.json",
    "index.mjs",
    "index.ts",
    "README.md",
)


def _resolve_bundle_dir(bundle_dir: Optional[Path] = None) -> Path:
    if bundle_dir is not None:
        return Path(bundle_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / BUNDLE_DIR_NAME


def _default_openclaw_config_path() -> Path:
    return Path.home() / ".openclaw" / "openclaw.json"


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


def build_openclaw_plugin_entry(
    *,
    bundle_dir: Optional[Path] = None,
    profile: str = "dev",
    shadow: bool = False,
    daemon_port: int = DEFAULT_DAEMON_PORT,
    fail_closed: bool = True,
    auto_approve_escalations: bool = False,
    sandbox_wrap: str = "none",
) -> Dict[str, Any]:
    """Return the OpenClaw `plugins.entries['aetheris-microbrains']` configuration block."""
    resolved_bundle = _resolve_bundle_dir(bundle_dir)
    return {
        "enabled": True,
        "path": str(resolved_bundle).replace("\\", "/"),
        "config": {
            "profile": profile,
            "shadow": bool(shadow),
            "daemonUrl": f"http://127.0.0.1:{int(daemon_port)}",
            "daemonPort": int(daemon_port),
            "autoStartDaemon": True,
            "failClosed": bool(fail_closed),
            "timeoutMs": 2500,
            "sandboxWrap": sandbox_wrap,
            "autoApproveEscalations": bool(auto_approve_escalations),
        },
    }


def install_config(
    *,
    config_path: Optional[Path] = None,
    bundle_dir: Optional[Path] = None,
    profile: str = "dev",
    shadow: bool = False,
    daemon_port: int = DEFAULT_DAEMON_PORT,
    fail_closed: bool = True,
    auto_approve_escalations: bool = False,
    sandbox_wrap: str = "none",
    dry_run: bool = False,
) -> int:
    """Idempotently merge the Aetheris plugin entry into `openclaw.json` without clobbering unrelated config."""
    target = Path(config_path).expanduser() if config_path else _default_openclaw_config_path()
    existing: Dict[str, Any] = {}
    if target.is_file():
        loaded = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected top-level JSON object in {target}")
        existing = loaded

    merged = dict(existing)
    plugins = dict(merged.get("plugins") if isinstance(merged.get("plugins"), dict) else {})
    entries = dict(plugins.get("entries") if isinstance(plugins.get("entries"), dict) else {})
    entries[PLUGIN_ID] = build_openclaw_plugin_entry(
        bundle_dir=bundle_dir,
        profile=profile,
        shadow=shadow,
        daemon_port=daemon_port,
        fail_closed=fail_closed,
        auto_approve_escalations=auto_approve_escalations,
        sandbox_wrap=sandbox_wrap,
    )
    plugins["entries"] = entries
    merged["plugins"] = plugins

    if dry_run:
        print(f"DRY RUN - would write OpenClaw config to {target}:")
        print(json.dumps(merged, indent=2, ensure_ascii=False))
        return 0

    _atomic_write_json(target, merged)
    print(f"Installed OpenClaw plugin entry ({PLUGIN_ID}) -> {target}")
    print("Restart OpenClaw Gateway to apply: openclaw gateway restart")
    return 0


def _print_setup(bundle_dir: Optional[Path] = None, daemon_port: int = DEFAULT_DAEMON_PORT) -> int:
    resolved = _resolve_bundle_dir(bundle_dir)
    print("OpenClaw Aetheris Microbrains Plugin Setup:")
    print('  1. pip install -e "."')
    print(f"  2. openclaw plugins install --link {resolved}")
    print(f"     (or run: aetheris install openclaw --daemon-port {int(daemon_port)})")
    print("  3. openclaw gateway restart")
    return 0


def self_test(bundle_dir: Optional[Path] = None) -> int:
    """Validate the OpenClaw bundle files, JSON schemas, clean paths, and `--host openclaw` probe verdicts."""
    target = _resolve_bundle_dir(bundle_dir)
    failures: list[str] = []

    if not target.is_dir():
        failures.append(f"missing-bundle-dir:{target}")
    else:
        for rel in _REQUIRED_FILES:
            fpath = target / rel
            if not fpath.is_file():
                failures.append(f"missing-{rel}")
                continue
            low = fpath.read_text(encoding="utf-8", errors="ignore").lower()
            if "i:\\" in low or "i:/" in low or "c:\\users\\" in low or "c:/users/" in low:
                failures.append(f"hardcoded-local-path:{rel}")

        pkg_file = target / "package.json"
        if pkg_file.is_file():
            try:
                pkg = json.loads(pkg_file.read_text(encoding="utf-8"))
                if pkg.get("name") != PACKAGE_NAME:
                    failures.append("package-name")
                exts = pkg.get("openclaw", {}).get("extensions", [])
                if "./index.mjs" not in exts:
                    failures.append("package-openclaw-extensions")
            except Exception:
                failures.append("package-json-invalid")

        man_file = target / "openclaw.plugin.json"
        if man_file.is_file():
            try:
                man = json.loads(man_file.read_text(encoding="utf-8"))
                if man.get("id") != PLUGIN_ID:
                    failures.append("manifest-id")
                props = man.get("configSchema", {}).get("properties", {})
                if not isinstance(props, dict) or "daemonUrl" not in props or "failClosed" not in props:
                    failures.append("manifest-config-schema")
            except Exception:
                failures.append("manifest-json-invalid")

    try:
        hook_mod = importlib.import_module("aetheris_microbrains.hook")
        deny_v = hook_mod.evaluate_hook_payload(
            {"toolName": "exec", "params": {"command": "rm -rf /"}},
            host="openclaw",
            log_decisions=False,
        )
        allow_v = hook_mod.evaluate_hook_payload(
            {"toolName": "exec", "params": {"command": "git status"}},
            host="openclaw",
            log_decisions=False,
        )
        hso_deny = deny_v.get("hookSpecificOutput") or {}
        if not (deny_v.get("decision") == "deny" and hso_deny.get("block") is True):
            failures.append("openclaw-probe-deny")
        if allow_v.get("decision") != "allow":
            failures.append("openclaw-probe-allow")
    except Exception as exc:
        failures.append(f"hook-probe:{type(exc).__name__}")

    if failures:
        print(f"OPENCLAW SELF-TEST FAILED: {', '.join(failures)}")
        return 1
    print(f"OPENCLAW SELF-TEST PASSED ({target})")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", default=None, help="Override path to openclaw-aetheris-microbrains")
    ap.add_argument("--config-path", default=None, help="Override path to ~/.openclaw/openclaw.json")
    ap.add_argument("--profile", choices=["strict", "dev"], default="dev", help="Policy profile (default: dev)")
    ap.add_argument("--shadow", action="store_true", help="Observe-only shadow mode")
    ap.add_argument("--daemon-port", type=int, default=DEFAULT_DAEMON_PORT, help="Local hook daemon HTTP port")
    ap.add_argument("--no-fail-closed", action="store_true", help="Do not block on hook timeout/error")
    ap.add_argument("--auto-approve-escalations", action="store_true", help="Auto-approve non-credential escalations")
    ap.add_argument(
        "--sandbox-wrap",
        choices=["none", "srt", "bwrap", "seatbelt", "landlock", "job", "aetheris-gate", "auto"],
        default="none",
        help="Optional OS sandbox wrapper for allowed exec commands",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print resulting openclaw.json without writing")
    ap.add_argument("--self-test", action="store_true", help="Validate OpenClaw bundle, schemas, and probe verdicts")
    ap.add_argument("--print-setup", action="store_true", help="Print 3-step OpenClaw installation commands")
    args = ap.parse_args(argv)

    bd = Path(args.bundle_dir) if args.bundle_dir else None
    if args.self_test:
        return self_test(bundle_dir=bd)
    if args.print_setup:
        return _print_setup(bundle_dir=bd, daemon_port=args.daemon_port)
    return install_config(
        config_path=Path(args.config_path) if args.config_path else None,
        bundle_dir=bd,
        profile=args.profile,
        shadow=bool(args.shadow),
        daemon_port=args.daemon_port,
        fail_closed=not bool(args.no_fail_closed),
        auto_approve_escalations=bool(args.auto_approve_escalations),
        sandbox_wrap=args.sandbox_wrap,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
