"""Rust `aetheris-gate` fast-path binary locator & command builder (stdlib-only)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def find_aetheris_gate(repo_root: Optional[Path] = None) -> Optional[str]:
    """Locate the native ``aetheris-gate`` binary if installed or built locally."""
    env_bin = str(os.getenv("AETHERIS_GATE_BIN") or "").strip()
    if env_bin:
        cand = Path(env_bin).expanduser()
        if cand.is_file():
            return str(cand)

    on_path = shutil.which("aetheris-gate")
    if on_path:
        return on_path

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    exe_name = "aetheris-gate.exe" if os.name == "nt" else "aetheris-gate"
    for profile_dir in ("release", "debug"):
        local_bin = root / "rust" / "aetheris-gate" / "target" / profile_dir / exe_name
        if local_bin.is_file():
            return str(local_bin)
    return None


def build_hook_command(
    host: str,
    *,
    event_args: List[str],
    extra_args: List[str],
    use_rust_shim: bool = False,
    auto_daemon: bool = True,
    repo_root: Optional[Path] = None,
) -> Tuple[str, List[str]]:
    """Build ``(executable, argv)`` preferring ``aetheris-gate --auto-daemon`` when enabled."""
    if use_rust_shim:
        gate_bin = find_aetheris_gate(repo_root=repo_root)
        if gate_bin:
            args = ["--auto-daemon"] if auto_daemon else []
            args.extend(["--host", host, *event_args, *extra_args])
            return gate_bin, args

    py = sys.executable.replace("\\", "/")
    return py, ["-m", "aetheris_microbrains.hook", "--host", host, *event_args, *extra_args]
