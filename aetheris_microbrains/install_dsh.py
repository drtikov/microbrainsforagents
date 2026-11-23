"""Verify or print setup instructions for the DeepSeek Harness (DSH) MCP bundle."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

BUNDLE_DIR_NAME = "dsh-aetheris-microbrains-bundle"
PACKAGE_NAME = "@local/aetheris-microbrains-mcp"
DEFAULT_PORT = 8766
_REQUIRED_FILES = (
    "package.json",
    "cordis.patch.yml",
    "README.md",
    "icon.svg",
    "locale/en.json",
    "docs/DETAIL.md",
    "docs/preview.svg",
)


def _resolve_bundle_dir(bundle_dir: Optional[Path] = None) -> Path:
    if bundle_dir is not None:
        return Path(bundle_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / BUNDLE_DIR_NAME


def _check_server(port: int = DEFAULT_PORT, timeout_s: float = 2.0) -> int:
    url = f"http://127.0.0.1:{int(port)}/mcp"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            print(f"DSH MCP SERVER REACHABLE ({url} -> HTTP {resp.status})")
            return 0
    except urllib.error.HTTPError as exc:
        # Streamable HTTP MCP endpoints often return 400/405/406 on bare GET
        print(f"DSH MCP SERVER REACHABLE ({url} -> HTTP {exc.code})")
        return 0
    except Exception as exc:
        print(f"DSH MCP SERVER UNREACHABLE ({url}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _print_setup(port: int = DEFAULT_PORT) -> int:
    print("DeepSeek Harness (DSH) Microbrains MCP Setup:")
    print('  1. pip install -e ".[mcp]"')
    print(f"  2. python -m aetheris_microbrains.mcp_server --http --port {int(port)}")
    print(f"  3. Install/load ./{BUNDLE_DIR_NAME} in DeepSeek Harness and call mcp__microbrains__microbrain_health")
    return 0


def self_test(bundle_dir: Optional[Path] = None) -> int:
    """Validate the DSH Cordis bundle manifest, patch YAML, assets, and mcp_server import."""
    target = _resolve_bundle_dir(bundle_dir)
    failures: list[str] = []

    try:
        importlib.import_module("aetheris_microbrains.mcp_server")
    except Exception as exc:
        failures.append(f"mcp_server-import:{type(exc).__name__}")

    if not target.is_dir():
        failures.append(f"missing-bundle-dir:{target}")
    else:
        for rel in _REQUIRED_FILES:
            if not (target / rel).is_file():
                failures.append(f"missing-{rel}")

        pkg_file = target / "package.json"
        if pkg_file.is_file():
            try:
                pkg = json.loads(pkg_file.read_text(encoding="utf-8"))
                if pkg.get("name") != PACKAGE_NAME:
                    failures.append("package-name")
                if pkg.get("dsh", {}).get("bundle", {}).get("patch") != "./cordis.patch.yml":
                    failures.append("package-dsh-patch")
            except Exception:
                failures.append("package-json-invalid")

        patch_file = target / "cordis.patch.yml"
        if patch_file.is_file():
            patch_txt = patch_file.read_text(encoding="utf-8")
            for needle in (
                "serverName: microbrains",
                "transport: streamable-http",
                f"http://127.0.0.1:{DEFAULT_PORT}/mcp",
                "failOnStartupError: false",
            ):
                if needle not in patch_txt:
                    failures.append(f"cordis-missing:{needle}")

        for rel in _REQUIRED_FILES:
            fpath = target / rel
            if fpath.is_file():
                low = fpath.read_text(encoding="utf-8", errors="ignore").lower()
                if "i:\\" in low or "i:/" in low or "c:\\users\\" in low or "c:/users/" in low:
                    failures.append(f"hardcoded-local-path:{rel}")

    if failures:
        print(f"DSH SELF-TEST FAILED: {', '.join(failures)}")
        return 1
    print(f"DSH SELF-TEST PASSED ({target})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", default=None, help="Override path to dsh-aetheris-microbrains-bundle")
    ap.add_argument("--self-test", action="store_true", help="Validate the DSH bundle and mcp_server importability")
    ap.add_argument("--check-server", action="store_true", help="Check if http://127.0.0.1:<port>/mcp is reachable")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="MCP HTTP port (default: 8766)")
    ap.add_argument("--print-setup", action="store_true", help="Print 3-step DeepSeek Harness setup instructions")
    args = ap.parse_args(argv)

    bd = Path(args.bundle_dir) if args.bundle_dir else None
    if args.print_setup:
        return _print_setup(port=args.port)
    if args.check_server:
        return _check_server(port=args.port)
    return self_test(bundle_dir=bd)


if __name__ == "__main__":
    raise SystemExit(main())
