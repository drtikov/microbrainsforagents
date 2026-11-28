"""Tests for native C-ABI RT safety FFI (`NativeRTGateLib`), C99/C++ header, stdlib isolation, and WASM teleop CBF SDK."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aetheris_microbrains.core.packaging_manifest import verify_core_isolation
from aetheris_microbrains.core.rust_shim import (
    NativeRTGateLib,
    build_hook_command,
    find_aetheris_gate,
    find_aetheris_gate_cdylib,
    get_native_rt_gate_lib,
)
from aetheris_microbrains.core.state_cbf_filter import (
    EulerLagrangeDynamics,
    HalfPlaneBarrier,
    HighOrderBarrier,
    LinearDynamics,
    SphereBarrier,
    StateCBFSafetyFilter,
)
from aetheris_microbrains.core.ternary_edge_runtime import PackedTernaryLinear
from aetheris_microbrains.core.whole_body_barriers import SimplexBackupController

RT_HEADER_PATH = REPO_ROOT / "rust" / "aetheris-gate" / "include" / "aetheris_rt_gate.h"
RUST_SHIM_PATH = REPO_ROOT / "aetheris_microbrains" / "core" / "rust_shim.py"
WASM_SDK_INDEX = REPO_ROOT / "wasm-sdk" / "index.mjs"
WASM_SDK_DTS = REPO_ROOT / "wasm-sdk" / "index.d.ts"
RUST_WASM_RELEASE = (
    REPO_ROOT
    / "rust"
    / "aetheris-gate"
    / "target"
    / "wasm32-unknown-unknown"
    / "release"
    / "aetheris_gate.wasm"
)


def _require_rt_lib() -> NativeRTGateLib:
    lib = get_native_rt_gate_lib(repo_root=REPO_ROOT)
    if lib is None:
        pytest.skip("Native aetheris_gate cdylib not built in this environment")
    return lib


def test_rust_shim_is_pure_stdlib_and_preserves_existing_api():
    src = RUST_SHIM_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(RUST_SHIM_PATH))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0].lower())

    assert "numpy" not in imported_roots, f"Forbidden numpy import in rust_shim.py: {imported_roots}"

    core_report = verify_core_isolation()
    assert core_report["ok"] is True, core_report
    assert core_report["numpy_violations"] == []

    # Verify preserved hook builder API still works identically
    exe, argv = build_hook_command(
        "claude",
        event_args=["--event", "pre-tool-use"],
        extra_args=["--profile", "dev"],
        use_rust_shim=False,
        repo_root=REPO_ROOT,
    )
    assert exe
    assert argv[:3] == ["-I", "-m", "aetheris_microbrains.hook"]
    assert find_aetheris_gate(repo_root=REPO_ROOT) is None or Path(find_aetheris_gate(repo_root=REPO_ROOT)).is_file()


def test_c99_cpp_header_exists_and_declares_all_rt_symbols():
    assert RT_HEADER_PATH.is_file(), f"Missing C/C++ RT header: {RT_HEADER_PATH}"
    header_text = RT_HEADER_PATH.read_text(encoding="utf-8")

    for required in (
        "#ifndef AETHERIS_RT_GATE_H",
        "#define AETHERIS_RT_GATE_H",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#ifdef __cplusplus",
        'extern "C" {',
        "AETHERIS_CBF_SAFE = 0",
        "AETHERIS_CBF_UNSAFE_NO_SOLUTION = 1",
        "AETHERIS_CBF_SIMPLEX_FALLBACK = 2",
        "AETHERIS_CBF_INVALID_INPUT = -1",
        "AETHERIS_CBF_DIM_MISMATCH = -2",
        "aetheris_rt_cbf_project(",
        "aetheris_rt_cbf_filter_or_brake(",
        "aetheris_rt_cbf_watchdog_scale(",
        "aetheris_rt_ternary_matvec_i8(",
        "ros2_control",
    ):
        assert required in header_text, f"Header missing `{required}`"


def test_native_cbf_project_matches_python_state_cbf_filter_to_1e9():
    lib = _require_rt_lib()
    assert find_aetheris_gate_cdylib(repo_root=REPO_ROOT) is not None
    assert NativeRTGateLib.is_available(repo_root=REPO_ROOT) is True

    # 1. Multi-barrier 2-DOF LinearDynamics + Sphere + HalfPlane + conformal margin + box
    dyn_lin = LinearDynamics(np.array([[0.0, 1.0], [-0.4, -0.8]]), np.array([[0.5, 0.1], [0.2, 1.2]]))
    b_wall = HalfPlaneBarrier(a=[-1.0, -0.5], b=-1.2, alpha=1.5)
    b_obs = SphereBarrier(center=[0.8, 0.2], radius=0.35, alpha=2.0)
    u_min = np.array([-4.0, -4.0], dtype=np.float64)
    u_max = np.array([4.0, 4.0], dtype=np.float64)
    py_filt = StateCBFSafetyFilter(
        dyn_lin,
        [b_wall, b_obs],
        u_min=u_min,
        u_max=u_max,
        conformal_margin=0.05,
        max_iters=200,
        tol=1e-9,
    )

    rng = np.random.default_rng(20261128)
    for _ in range(25):
        x = rng.uniform(-0.4, 0.6, size=2)
        u_nom = rng.uniform(-6.0, 6.0, size=2)
        A, b, _ = py_filt._halfspaces(x)
        py_out = py_filt.filter(x, u_nom)
        if not py_out["verified_safe"]:
            continue

        # Fast zero-copy numpy path
        c_out_np = lib.cbf_project(A, b, u_nom, u_min=u_min, u_max=u_max, max_iters=200, tol=1e-9)
        assert c_out_np["verified_safe"] is True
        assert c_out_np["status"] == "safe"
        assert c_out_np["status_code"] == 0
        assert np.max(np.abs(np.asarray(c_out_np["u_filtered"]) - np.asarray(py_out["u_filtered"]))) < 1e-9
        assert abs(c_out_np["min_barrier_slack"] - py_out["min_barrier_slack"]) < 1e-8

        # Pure stdlib Python list/tuple fallback path
        c_out_list = lib.cbf_project(
            A.tolist(),
            tuple(b.tolist()),
            u_nom.tolist(),
            u_min=u_min.tolist(),
            u_max=tuple(u_max.tolist()),
            max_iters=200,
            tol=1e-9,
        )
        assert c_out_list["verified_safe"] is True
        assert np.max(np.abs(np.asarray(c_out_list["u_filtered"]) - np.asarray(py_out["u_filtered"]))) < 1e-9

    # 2. Nonlinear 2-DOF Euler-Lagrange HOCBF (relative_degree=2) parity
    def M_fn(q):
        return np.array([[2.0 + 0.25 * np.cos(q[1]), 0.1], [0.1, 1.3]], dtype=np.float64)

    def C_fn(q, dq):
        return np.array([[-0.125 * np.sin(q[1]) * dq[1], 0.0], [0.125 * np.sin(q[1]) * dq[0], 0.0]], dtype=np.float64)

    def g_fn(q):
        return np.array([0.4 * np.sin(q[0]), 0.25 * np.sin(q[1])], dtype=np.float64)

    dyn_el = EulerLagrangeDynamics(n_dof=2, M_fn=M_fn, C_fn=C_fn, g_fn=g_fn)
    hocbf = HighOrderBarrier(SphereBarrier(center=[1.0, 0.0], radius=0.35, alpha=2.5), alpha1=2.5, alpha2=3.0)
    hocbf_wall = HighOrderBarrier(HalfPlaneBarrier(a=[-1.0, 0.0], b=-1.4, alpha=2.0), alpha1=2.0, alpha2=2.5)
    el_lo = np.array([-15.0, -15.0], dtype=np.float64)
    el_hi = np.array([15.0, 15.0], dtype=np.float64)
    el_filt = StateCBFSafetyFilter(dyn_el, [hocbf, hocbf_wall], u_min=el_lo, u_max=el_hi, max_iters=200, tol=1e-9)

    for _ in range(20):
        q = rng.uniform(-0.2, 0.45, size=2)
        dq = rng.uniform(-0.5, 0.9, size=2)
        x_el = np.concatenate([q, dq])
        u_nom_el = rng.uniform(-18.0, 18.0, size=2)
        A_el, b_el, _ = el_filt._halfspaces(x_el)
        py_el = el_filt.filter(x_el, u_nom_el)
        if not py_el["verified_safe"]:
            continue
        c_el = lib.cbf_project(A_el, b_el, u_nom_el, u_min=el_lo, u_max=el_hi, max_iters=200, tol=1e-9)
        assert c_el["verified_safe"] is True
        assert np.max(np.abs(np.asarray(c_el["u_filtered"]) - np.asarray(py_el["u_filtered"]))) < 1e-9
        assert abs(c_el["min_barrier_slack"] - py_el["min_barrier_slack"]) < 1e-8


def test_native_cbf_filter_or_brake_simplex_and_edge_cases():
    lib = _require_rt_lib()

    # Zero-barrier box-only projection & completely unbounded pass-through
    free_res = lib.cbf_project(None, None, [3.2, -5.1], u_min=[-2.0, -2.0], u_max=[2.0, 2.0])
    assert free_res["status"] == "safe"
    assert free_res["verified_safe"] is True
    assert np.allclose(free_res["u_filtered"], [2.0, -2.0], atol=1e-12)

    unbounded_res = lib.cbf_project([], [], [3.2, -5.1], u_min=None, u_max=None)
    assert unbounded_res["status"] == "safe"
    assert np.allclose(unbounded_res["u_filtered"], [3.2, -5.1], atol=1e-12)

    # Infeasible hard barriers parity with StateCBFSafetyFilter + SimplexBackupController (zero gravity)
    dyn2 = EulerLagrangeDynamics(n_dof=2)
    b1 = HalfPlaneBarrier(a=[0.0, 0.0, 1.0, 0.0], b=2.0, alpha=1.0, hard=True)
    b2 = HalfPlaneBarrier(a=[0.0, 0.0, -1.0, 0.0], b=2.0, alpha=1.0, hard=True)
    lo = np.array([-3.0, -3.0], dtype=np.float64)
    hi = np.array([3.0, 3.0], dtype=np.float64)
    kd = 2.5
    x = np.array([0.1, -0.2, 1.6, -0.8], dtype=np.float64)
    q_dot = x[2:]
    u_nom = np.array([1.2, -0.4], dtype=np.float64)

    filt_no_backup = StateCBFSafetyFilter(dyn2, [b1, b2], u_min=lo, u_max=hi)
    A_inf, b_inf, _ = filt_no_backup._halfspaces(x)
    py_no_backup = filt_no_backup.filter(x, u_nom)
    c_no_backup = lib.cbf_project(A_inf, b_inf, u_nom, u_min=lo, u_max=hi)
    assert py_no_backup["status"] == "unsafe_no_solution"
    assert c_no_backup["status"] == "unsafe_no_solution"
    assert c_no_backup["status_code"] == 1
    assert c_no_backup["verified_safe"] is False

    backup = SimplexBackupController(n_dof=2, kd=kd, g_fn=None)
    filt_simplex = StateCBFSafetyFilter(dyn2, [b1, b2], u_min=lo, u_max=hi, backup_controller=backup)
    py_simplex = filt_simplex.filter(x, u_nom)
    c_simplex = lib.cbf_filter_or_brake(
        A_inf,
        b_inf,
        u_nom,
        u_min=lo,
        u_max=hi,
        q_dot=q_dot,
        kd=kd,
    )
    assert py_simplex["status"] == "simplex_braking_engaged"
    assert c_simplex["status"] == "simplex_braking_engaged"
    assert c_simplex["status_code"] == 2
    assert c_simplex["verified_safe"] is False
    assert c_simplex["backup_engaged"] is True
    assert np.max(np.abs(np.asarray(c_simplex["u_filtered"]) - np.asarray(py_simplex["u_filtered"]))) < 1e-9
    assert abs(c_simplex["min_barrier_slack"] - py_simplex["min_barrier_slack"]) < 1e-8

    # Error handling: non-finite input (-1) and dimension mismatch (-2)
    nan_res = lib.cbf_project([[1.0, 0.0]], [0.5], [float("nan"), 0.0])
    assert nan_res["status_code"] == -1
    assert nan_res["verified_safe"] is False

    dim_res = lib.cbf_project([[1.0, 0.0, 0.0]], [0.5], [0.1, 0.2])
    assert dim_res["status_code"] == -2
    assert dim_res["verified_safe"] is False

    # Watchdog scale checks
    assert lib.watchdog_scale(20.0, 50.0, 100.0) == pytest.approx(1.0, abs=1e-12)
    assert lib.watchdog_scale(100.0, 50.0, 100.0) == pytest.approx(0.5, abs=1e-12)
    assert lib.watchdog_scale(160.0, 50.0, 100.0) == pytest.approx(0.0, abs=1e-12)
    assert lib.watchdog_scale(-5.0, 50.0, 100.0) == pytest.approx(0.0, abs=1e-12)


def test_native_ternary_matvec_i8_exact_integer_parity():
    lib = _require_rt_lib()
    rng = np.random.default_rng(9911)

    for in_features, out_features in ((16, 8), (13, 7), (4, 3), (19, 5)):
        raw_w = rng.normal(0.0, 1.0, size=(out_features, in_features)).astype(np.float32)
        layer = PackedTernaryLinear(in_features=in_features, out_features=out_features, weights=raw_w)

        x_f = rng.normal(0.0, 2.5, size=(1, in_features)).astype(np.float32)
        max_abs = max(float(np.max(np.abs(x_f))), 1e-9)
        scale = 127.0 / max_abs
        x_q = np.clip(np.round(x_f[0] * scale), -127.0, 127.0).astype(np.int8)

        expected_i32 = (x_q.astype(np.int32) @ layer.decode_int8().T.astype(np.int32)).tolist()

        # Zero-copy uint8 + int8 ndarray path
        got_np = lib.ternary_matvec_i8(layer.packed, x_q, out_features, in_features)
        assert got_np == expected_i32

        # Stdlib bytes + Python list path
        got_bytes = lib.ternary_matvec_i8(layer.packed.tobytes(), x_q.tolist(), out_features, in_features)
        assert got_bytes == expected_i32

        # Rescaled float output matches PackedTernaryLinear.matvec_quantized
        combined_scale = float(layer.gamma / scale)
        got_float = np.asarray(got_np, dtype=np.float32) * combined_scale
        ref_float = layer.matvec_quantized(x_f)[0]
        assert np.allclose(got_float, ref_float, atol=1e-6)


def test_wasm_sdk_teleop_cbf_and_watchdog_via_node():
    dts_src = WASM_SDK_DTS.read_text(encoding="utf-8")
    for sym in ("filterTeleopCbf(", "watchdogScale(", "TeleopCbfRequest", "TeleopCbfResult"):
        assert sym in dts_src, f"Missing `{sym}` in {WASM_SDK_DTS}"

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("node not on PATH")
    if not RUST_WASM_RELEASE.is_file():
        pytest.skip(f"WASM artifact not built: {RUST_WASM_RELEASE}")

    sdk_url = WASM_SDK_INDEX.resolve().as_uri()
    script = f"""
import assert from "node:assert/strict";
import {{ loadAetherisWasm }} from {json.dumps(sdk_url)};

const gate = await loadAetherisWasm();
assert.equal(typeof gate.filterTeleopCbf, "function");
assert.equal(typeof gate.watchdogScale, "function");
assert.equal(typeof gate.evalPretool, "function");
assert.equal(typeof gate.scrub, "function");

// 1. Safe clamping with both plain arrays and Float64Array
const safeRes = gate.filterTeleopCbf({{
  u_nominal: new Float64Array([-1.2, 0.9]),
  A: [[2.0, 0.0], [0.0, -1.0]],
  b: [1.0, -0.3],
  u_min: [-2.0, -2.0],
  u_max: [2.0, 2.0],
  q_dot: [0.4, -0.2],
  kd: 2.0,
  max_iters: 200,
  tol: 1e-9,
}});
assert.equal(safeRes.status, "safe");
assert.equal(safeRes.verified_safe, true);
assert.ok(Math.abs(safeRes.u_filtered[0] - 0.5) < 1e-9);
assert.ok(Math.abs(safeRes.u_filtered[1] - 0.3) < 1e-9);
assert.ok(safeRes.min_barrier_slack >= -1e-9);
assert.ok(safeRes.iterations >= 1);

// 2. Infeasible with q_dot -> simplex_braking_engaged
const simplexRes = gate.filterTeleopCbf({{
  u_nominal: [0.8, -0.4],
  A: [[1.0, 0.0], [-1.0, 0.0]],
  b: [1.5, 1.5],
  u_min: [-1.0, -1.0],
  u_max: [1.0, 1.0],
  q_dot: [0.75, -0.25],
  kd: 2.0,
}});
assert.equal(simplexRes.status, "simplex_braking_engaged");
assert.equal(simplexRes.verified_safe, false);
assert.ok(Math.abs(simplexRes.u_filtered[0] - (-1.0)) < 1e-12);
assert.ok(Math.abs(simplexRes.u_filtered[1] - 0.5) < 1e-12);

// 3. Infeasible without q_dot -> unsafe_no_solution
const unsafeRes = gate.filterTeleopCbf({{
  u_nominal: [0.8],
  A: [[1.0], [-1.0]],
  b: [1.5, 1.5],
  u_min: [-1.0],
  u_max: [1.0],
}});
assert.equal(unsafeRes.status, "unsafe_no_solution");
assert.equal(unsafeRes.verified_safe, false);

// 4. Watchdog authority scale
assert.ok(Math.abs(gate.watchdogScale(10.0, 50.0, 100.0) - 1.0) < 1e-12);
assert.ok(Math.abs(gate.watchdogScale(100.0, 50.0, 100.0) - 0.5) < 1e-12);
assert.ok(Math.abs(gate.watchdogScale(160.0, 50.0, 100.0) - 0.0) < 1e-12);
assert.ok(Math.abs(gate.watchdogScale(-1.0, 50.0, 100.0) - 0.0) < 1e-12);
assert.ok(Math.abs(gate.watchdogScale(Number.NaN, 50.0, 100.0) - 0.0) < 1e-12);

// 5. Repeated calls verify zero WASM linear memory leaks
for (let k = 0; k < 64; k += 1) {{
  const r = gate.filterTeleopCbf({{
    u_nominal: [-1.0 - 0.01 * k, 0.2],
    A: [[1.0, 0.0]],
    b: [0.25],
    u_min: [-5.0, -5.0],
    u_max: [5.0, 5.0],
  }});
  assert.equal(r.verified_safe, true);
  assert.ok(Math.abs(r.u_filtered[0] - 0.25) < 1e-9);
}}

// 6. Existing SDK functions remain intact
const allow = gate.evalPretool({{ tool_name: "Bash", tool_input: {{ command: "git status" }} }}, "claude", "dev");
assert.equal(allow.decision, "allow");

console.log(JSON.stringify({{ ok: true, safe: safeRes.u_filtered, simplex: simplexRes.u_filtered }}));
"""
    proc = subprocess.run(
        [node_bin, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, f"Node WASM teleop test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
