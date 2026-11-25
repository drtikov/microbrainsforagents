"""Unified `aetheris` (`uvx aetheris-microbrains`) CLI & Multi-Host Installer (stdlib-only).

Supports:
  - `aetheris install <host>` (`antigravity|claude|codex|hermes|cursor|openclaw|opencode|kilocode|pi|omp|qoder|zcode|dsh|auto`)
  - `aetheris audit ...`
  - `aetheris doctor ...`
  - `aetheris mcp-proxy ...`
  - `aetheris pr-gate ...`
  - `aetheris verify-core [--json]`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_HOSTS = ("antigravity", "claude", "codex", "hermes", "cursor", "openclaw", "opencode", "kilocode", "pi", "omp", "qoder", "zcode", "dsh", "auto")


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


def detect_installed_hosts(workspace: Optional[Path] = None) -> List[str]:
    """Detect AI agent harnesses present on this workstation or workspace."""
    home = Path.home()
    ws = Path(workspace) if workspace else Path.cwd()
    detected: List[str] = []
    if (home / ".gemini").exists() or (ws / ".agents").exists():
        detected.append("antigravity")
    if (home / ".claude").exists() or (ws / ".claude").exists():
        detected.append("claude")
    if (home / ".codex").exists() or (ws / ".codex").exists():
        detected.append("codex")
    if (home / ".hermes").exists():
        detected.append("hermes")
    if (home / ".cursor").exists() or (ws / ".cursor").exists():
        detected.append("cursor")
    if (home / ".openclaw").exists():
        detected.append("openclaw")
    if (home / ".qoder").exists() or (ws / ".qoder").exists():
        detected.append("qoder")
    if (
        (home / ".config" / "opencode").exists()
        or (home / ".opencode").exists()
        or (ws / ".opencode").exists()
        or (ws / "opencode.json").exists()
    ):
        detected.append("opencode")
    if (
        (home / ".config" / "kilo").exists()
        or (home / ".kilo").exists()
        or (ws / ".kilo").exists()
        or (ws / "kilo.json").exists()
    ):
        detected.append("kilocode")
    if (
        (home / ".pi").exists()
        or (home / ".pi" / "agent").exists()
        or (ws / ".pi").exists()
    ):
        detected.append("pi")
    if (
        (home / ".omp").exists()
        or (home / ".omp" / "agent").exists()
        or (ws / ".omp").exists()
        or (ws / "omp.json").exists()
    ):
        detected.append("omp")
    if (home / ".zcode").exists() or (ws / ".zcode").exists():
        detected.append("zcode")
    return detected


def _build_generic_hook_bundle(host: str, *, profile: str, shadow: bool, auto_allow: bool) -> Dict[str, Any]:
    py = sys.executable.replace("\\", "/")
    extra = f"--host {host} --profile {profile} --log-decisions --session-state auto"
    if shadow:
        extra += " --shadow"
    if auto_allow:
        extra += " --auto-allow"
    pre_cmd = f'"{py}" -m aetheris_microbrains.hook {extra}'
    post_extra = f"{extra} --replace-tool-output" if host in {"claude", "qoder"} else extra
    post_cmd = f'"{py}" -m aetheris_microbrains.hook --event post-tool-use {post_extra}'
    if host == "cursor":
        return {
            "version": 1,
            "hooks": {
                "beforeShellExecution": [{"command": pre_cmd}],
                "afterShellExecution": [{"command": post_cmd}],
            },
        }
    pre_hook: Dict[str, Any] = {"type": "command", "command": pre_cmd}
    post_hook: Dict[str, Any] = {"type": "command", "command": post_cmd}
    if host == "qoder":
        pre_hook["timeout"] = 10
        post_hook["timeout"] = 10
    else:
        pre_hook["onFailure"] = "block"
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*",
                    "hooks": [pre_hook],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|WebFetch|WebSearch|mcp__.*",
                    "hooks": [post_hook],
                }
            ],
        }
    }


def _merge_hooks_non_destructively(existing_hooks: Any, new_hooks: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new Aetheris hook entries per-event without removing unrelated user hooks."""
    merged: Dict[str, Any] = dict(existing_hooks) if isinstance(existing_hooks, dict) else {}
    for event_name, new_entries in new_hooks.items():
        if not isinstance(new_entries, list):
            merged[event_name] = new_entries
            continue
        cur_list = merged.get(event_name)
        preserved: List[Any] = []
        if isinstance(cur_list, list):
            for entry in cur_list:
                try:
                    serialized = json.dumps(entry, ensure_ascii=False)
                except Exception:
                    serialized = str(entry)
                if "aetheris_microbrains.hook" not in serialized:
                    preserved.append(entry)
        merged[event_name] = [*preserved, *new_entries]
    return merged


def _install_host(args: argparse.Namespace, host: str) -> int:
    ws = Path(args.workspace).expanduser().resolve() if args.workspace else None
    if host == "antigravity":
        from .install_antigravity import install as ag_install, self_test as ag_self_test

        pd = Path(args.plugin_dir) if getattr(args, "plugin_dir", None) else None
        if args.self_test:
            return ag_self_test(workspace=ws, plugin_dir=pd)
        return ag_install(
            shadow=bool(args.shadow),
            auto_allow=bool(args.auto_allow),
            auto_approve_escalations=bool(args.auto_approve_escalations),
            use_rust_shim=bool(args.use_rust_shim),
            profile=args.profile,
            workspace=ws,
            plugin_dir=pd,
            dry_run=bool(args.dry_run),
        )
    if host == "openclaw":
        from .install_openclaw import install_config as oc_install, self_test as oc_self_test

        if args.self_test:
            return oc_self_test()
        return oc_install(
            profile=str(args.profile or "dev"),
            shadow=bool(args.shadow),
            auto_approve_escalations=bool(args.auto_approve_escalations),
            dry_run=bool(args.dry_run),
        )
    if host == "dsh":
        from .install_dsh import _print_setup, self_test as dsh_self_test

        if args.self_test:
            return dsh_self_test()
        return _print_setup()
    if host == "zcode":
        from .install_zcode import install as zcode_install, self_test as zcode_self_test

        if args.self_test:
            return zcode_self_test()
        return zcode_install(
            workspace=ws,
            profile=str(args.profile or "strict"),
            shadow=bool(args.shadow),
            dry_run=bool(args.dry_run),
        )

    if host == "qoder" and args.self_test:
        from .hook import evaluate_hook_payload

        deny_v = evaluate_hook_payload(
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
            host="qoder",
            log_decisions=False,
        )
        allow_v = evaluate_hook_payload(
            {"tool_name": "Bash", "tool_input": {"command": "git status"}},
            host="qoder",
            log_decisions=False,
        )
        qoder_write = evaluate_hook_payload(
            {"tool_name": "Write", "tool_input": {"file_path": ".qoder/settings.json", "content": "{}"}},
            host="qoder",
            log_decisions=False,
        )
        ok = (
            deny_v.get("decision") == "deny"
            and deny_v.get("exit_code") == 2
            and allow_v.get("decision") == "allow"
            and qoder_write.get("decision") == "deny"
        )
        print("QODER SELF-TEST PASSED" if ok else "QODER SELF-TEST FAILED")
        return 0 if ok else 1

    _ESM_PLUGIN_HOSTS = {
        "opencode": (
            "opencode-aetheris-microbrains",
            "aetheris-microbrains.js",
            "opencode.json",
            (".opencode", "plugins", "aetheris-microbrains.js"),
            (".config", "opencode", "plugins", "aetheris-microbrains.js"),
            "OpenCode",
        ),
        "kilocode": (
            "kilocode-aetheris-microbrains",
            "aetheris-microbrains.js",
            "kilo.json",
            (".kilo", "plugins", "aetheris-microbrains.js"),
            (".config", "kilo", "plugins", "aetheris-microbrains.js"),
            "Kilo Code",
        ),
        "pi": (
            "pi-aetheris-microbrains",
            "index.js",
            ".pi/settings.json",
            (".pi", "extensions", "aetheris-microbrains", "index.js"),
            (".pi", "agent", "extensions", "aetheris-microbrains", "index.js"),
            "Pi",
        ),
        "omp": (
            "omp-aetheris-microbrains",
            "index.js",
            ".omp/settings.json",
            (".omp", "extensions", "aetheris-microbrains", "index.js"),
            (".omp", "agent", "extensions", "aetheris-microbrains", "index.js"),
            "OMP",
        ),
    }
    if host in _ESM_PLUGIN_HOSTS:
        bundle_dir, src_name, probe_cfg, ws_rel_parts, global_rel_parts, display_name = _ESM_PLUGIN_HOSTS[host]
        src_plugin = Path(__file__).resolve().parents[1] / bundle_dir / src_name
        if args.self_test:
            from .hook import evaluate_hook_payload

            ok_file = src_plugin.is_file()
            if ok_file:
                low = src_plugin.read_text(encoding="utf-8", errors="ignore").lower()
                if "i:\\" in low or "i:/" in low or "c:\\users\\" in low or "c:/users/" in low:
                    ok_file = False
            deny_v = evaluate_hook_payload(
                {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
                host=host,
                log_decisions=False,
            )
            allow_v = evaluate_hook_payload(
                {"tool_name": "Bash", "tool_input": {"command": "git status"}},
                host=host,
                log_decisions=False,
            )
            cfg_write = evaluate_hook_payload(
                {"tool_name": "Write", "tool_input": {"file_path": probe_cfg, "content": "{}"}},
                host=host,
                log_decisions=False,
            )
            ok = (
                ok_file
                and deny_v.get("decision") == "deny"
                and (deny_v.get("hookSpecificOutput") or {}).get("block") is True
                and allow_v.get("decision") == "allow"
                and cfg_write.get("decision") == "deny"
            )
            print(f"{host.upper()} SELF-TEST PASSED" if ok else f"{host.upper()} SELF-TEST FAILED")
            return 0 if ok else 1

        if not src_plugin.is_file():
            print(f"{display_name} plugin bundle missing: {src_plugin}")
            return 1
        if getattr(args, "plugin_dir", None):
            target = Path(args.plugin_dir) / src_name
        elif ws is not None:
            target = ws / Path(*ws_rel_parts)
        else:
            target = Path.home() / Path(*global_rel_parts)
        if args.dry_run:
            print(f"DRY RUN - would install {display_name} plugin -> {target}")
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".aetheris-microbrains.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(src_plugin.read_text(encoding="utf-8"))
                fh.flush()
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        print(f"Installed {display_name} plugin -> {target}")
        print(f"Start the warm HTTP daemon before launching {display_name}:")
        print(f"  aetheris-daemon --http --host {host} --port 8765")
        return 0

    bundle = _build_generic_hook_bundle(
        host,
        profile=str(args.profile or "dev"),
        shadow=bool(args.shadow),
        auto_allow=bool(args.auto_allow),
    )
    base_dir = ws if ws is not None else Path.home()
    rel_map = {
        "claude": Path(".claude") / "settings.local.json",
        "codex": Path(".codex") / "hooks.json",
        "hermes": Path(".hermes") / "aetheris_hooks.json",
        "cursor": Path(".cursor") / "hooks.json",
        "qoder": Path(".qoder") / ("settings.local.json" if ws is not None else "settings.json"),
    }
    target = Path(args.plugin_dir) if getattr(args, "plugin_dir", None) else (base_dir / rel_map[host])
    merged_bundle = dict(bundle)
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged = dict(existing)
                merged["hooks"] = _merge_hooks_non_destructively(existing.get("hooks"), bundle.get("hooks", {}))
                if "version" in bundle:
                    merged["version"] = bundle["version"]
                merged_bundle = merged
        except Exception:
            pass
    if args.dry_run:
        print(f"DRY RUN - would write {host} hooks config to {target}:")
        print(json.dumps(merged_bundle, indent=2, ensure_ascii=False))
        return 0
    _atomic_write_json(target, merged_bundle)
    print(f"Installed Aetheris Microbrains ({host}) -> {target}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aetheris",
        description="Unified Aetheris Microbrains CLI (`uvx aetheris-microbrains`).",
    )
    sub = parser.add_subparsers(dest="command")

    p_inst = sub.add_parser("install", help="Install or verify hooks/bundles for an AI host")
    p_inst.add_argument("host", choices=list(SUPPORTED_HOSTS), help="Target AI host or 'auto'")
    p_inst.add_argument("--profile", choices=["strict", "dev"], default="dev", help="Policy profile")
    p_inst.add_argument("--shadow", action="store_true", help="Observe-only shadow mode")
    p_inst.add_argument("--auto-allow", action="store_true", help="Emit explicit allow decisions")
    p_inst.add_argument("--auto-approve-escalations", action="store_true", help="Auto-approve non-credential escalations")
    p_inst.add_argument("--use-rust-shim", action="store_true", help="Use native Rust `aetheris-gate` shim when available")
    p_inst.add_argument("--workspace", default=None, help="Target workspace root")
    p_inst.add_argument("--plugin-dir", default=None, help="Override target plugin/config directory")
    p_inst.add_argument("--dry-run", action="store_true", help="Print configuration without writing files")
    p_inst.add_argument("--self-test", action="store_true", help="Run host integration self-test")

    p_vc = sub.add_parser("verify-core", help="Verify stdlib-only core isolation vs lab research modules")
    p_vc.add_argument("--json", action="store_true", help="Output verification report as JSON")

    for passthrough in ("audit", "doctor", "mcp-proxy", "pr-gate"):
        sub.add_parser(passthrough, add_help=False, help=f"Forward arguments to aetheris_microbrains.{passthrough.replace('-', '_')}")

    known, unknown = parser.parse_known_args(argv)
    if known.command == "verify-core":
        from .core.packaging_manifest import verify_core_isolation

        rep = verify_core_isolation()
        if known.json:
            print(json.dumps(rep, indent=2))
        else:
            print(
                f"core_isolated={rep['ok']} | core_modules={rep['core_count']} | "
                f"lab_modules={rep['lab_count']} | numpy_violations={rep['numpy_violations']}"
            )
        return 0 if rep["ok"] else 1

    if known.command == "install":
        if known.host == "auto":
            hosts = detect_installed_hosts(Path(known.workspace) if known.workspace else None)
            if not hosts:
                hosts = ["antigravity"]
            rc = 0
            for h in hosts:
                rc = max(rc, _install_host(known, h))
            return rc
        return _install_host(known, known.host)

    if known.command == "audit":
        from .audit import main as sub_main

        return sub_main(unknown)
    if known.command == "doctor":
        from .doctor import main as sub_main

        return sub_main(unknown)
    if known.command == "mcp-proxy":
        from .mcp_proxy import main as sub_main

        return sub_main(unknown)
    if known.command == "pr-gate":
        from .pr_gate import main as sub_main

        return sub_main(unknown)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
