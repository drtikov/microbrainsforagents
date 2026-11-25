#!/usr/bin/env python3
"""ZCode hook adapter for Aetheris Microbrains.

ZCode runs this file as a PreToolUse/PostToolUse ``process`` hook (plugin
route). It bootstraps the engine — preferring a pip-installed
``aetheris_microbrains`` and falling back to the vendored copy at the plugin
root — then defers to ``aetheris_microbrains.hook.main()``.

Failure posture: if the engine cannot be imported, the wrapper exits 0 with a
stderr note (fail-open). Set ``AETHERIS_ZCODE_FAIL_CLOSED=1`` to exit 2 instead.
Python 3.10+ stdlib is sufficient for the security gate (NumPy is optional).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_engine() -> None:
    root = os.environ.get("ZCODE_PLUGIN_ROOT") or os.environ.get("CLAUDE_PLUGIN_ROOT") or ""
    if not root:
        root = str(Path(__file__).resolve().parents[2])
    plugin_root = Path(root).resolve()
    if (plugin_root / "aetheris_microbrains" / "__init__.py").is_file():
        root_str = str(plugin_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return
    import aetheris_microbrains  # noqa: F401


def _inject_default_host(argv: list[str]) -> list[str]:
    if any(a == "--host" or a.startswith("--host=") for a in argv):
        return argv
    return [*argv, "--host", os.environ.get("AETHERIS_ZCODE_HOST", "zcode")]


def main() -> int:
    argv = sys.argv[1:]
    try:
        _bootstrap_engine()
        from aetheris_microbrains.hook import main as hook_main
    except Exception as exc:
        print(f"[aetheris-microbrains] gate unavailable: {exc!r}", file=sys.stderr)
        if os.environ.get("AETHERIS_ZCODE_FAIL_CLOSED", "") == "1":
            return 2
        return 0
    return int(hook_main(_inject_default_host(argv)))


if __name__ == "__main__":
    sys.exit(main())
