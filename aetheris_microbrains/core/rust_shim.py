"""Rust `aetheris-gate` fast-path binary locator, command builder, and stdlib ctypes RT FFI shim."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_F64_PTR = ctypes.POINTER(ctypes.c_double)
_U8_PTR = ctypes.POINTER(ctypes.c_uint8)
_I8_PTR = ctypes.POINTER(ctypes.c_int8)
_I32_PTR = ctypes.POINTER(ctypes.c_int32)
_USIZE_PTR = ctypes.POINTER(ctypes.c_size_t)

_NATIVE_BYTEORDERS = ("=", "|", "<" if sys.byteorder == "little" else ">")

_STATUS_CODE_TO_STR: Dict[int, str] = {
    0: "safe",
    1: "unsafe_no_solution",
    2: "simplex_braking_engaged",
    -1: "invalid_input",
    -2: "dim_mismatch",
}

_LOCK = threading.Lock()
_CDLL_CACHE: Dict[str, ctypes.CDLL] = {}
_DEFAULT_INSTANCE: Optional["NativeRTGateLib"] = None


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


def find_aetheris_gate_cdylib(repo_root: Optional[Path] = None) -> Optional[str]:
    """Locate the native ``aetheris_gate`` shared library (``.dll`` / ``.dylib`` / ``.so``)."""
    env_lib = str(os.getenv("AETHERIS_GATE_CDYLIB") or "").strip()
    if env_lib:
        cand = Path(env_lib).expanduser()
        if cand.is_file():
            return str(cand)

    if os.name == "nt":
        lib_name = "aetheris_gate.dll"
    elif sys.platform == "darwin":
        lib_name = "libaetheris_gate.dylib"
    else:
        lib_name = "libaetheris_gate.so"

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    for profile_dir in ("release", "debug"):
        local_lib = root / "rust" / "aetheris-gate" / "target" / profile_dir / lib_name
        if local_lib.is_file():
            return str(local_lib)
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
    """Build ``(executable, argv)`` preferring ``aetheris-gate --auto-daemon`` when enabled; uses ``-I`` for Python to prevent CWD shadowing."""
    if use_rust_shim:
        gate_bin = find_aetheris_gate(repo_root=repo_root)
        if gate_bin:
            args = ["--auto-daemon"] if auto_daemon else []
            args.extend(["--host", host, *event_args, *extra_args])
            return gate_bin, args

    py = sys.executable.replace("\\", "/")
    return py, ["-I", "-m", "aetheris_microbrains.hook", "--host", host, *event_args, *extra_args]


def _is_c_contiguous_dtype(obj: Any, expected_name: str, expected_itemsize: int) -> bool:
    """Duck-typed check for C-contiguous native-endian buffer objects exposing ``.ctypes``."""
    ctypes_attr = getattr(obj, "ctypes", None)
    flags = getattr(obj, "flags", None)
    dtype = getattr(obj, "dtype", None)
    if ctypes_attr is None or flags is None or dtype is None:
        return False
    if not getattr(flags, "c_contiguous", False):
        return False
    if getattr(dtype, "name", None) != expected_name:
        return False
    if int(getattr(dtype, "itemsize", 0)) != expected_itemsize:
        return False
    if getattr(dtype, "byteorder", "=") not in _NATIVE_BYTEORDERS:
        return False
    return hasattr(ctypes_attr, "data_as")


def _as_f64_vec(vec: Any) -> Tuple[Any, int, Any]:
    """Return ``(c_double_ptr, length, keepalive)`` with zero-copy fast path for contiguous float64 arrays."""
    shape = getattr(vec, "shape", None)
    if shape is not None and _is_c_contiguous_dtype(vec, "float64", 8):
        if len(shape) == 1:
            n = int(shape[0])
            ptr = vec.ctypes.data_as(_F64_PTR) if n > 0 else None
            return ptr, n, vec
        if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            n = int(shape[0] * shape[1])
            ptr = vec.ctypes.data_as(_F64_PTR) if n > 0 else None
            return ptr, n, vec

    if shape is not None and len(shape) > 1 and hasattr(vec, "flat"):
        vals = [float(x) for x in vec.flat]
    else:
        vals = [float(x) for x in vec]
    n = len(vals)
    if n == 0:
        return None, 0, None
    buf = (ctypes.c_double * n)(*vals)
    return ctypes.cast(buf, _F64_PTR), n, buf


def _as_f64_matrix(
    a_rows: Any, expected_dim: int, expected_m: Optional[int] = None
) -> Tuple[Any, int, bool, Any]:
    """Return ``(c_double_ptr, m_barriers, dim_ok, keepalive)`` with zero-copy fast path for 2D float64 arrays."""
    if a_rows is None:
        return None, 0, True, None

    shape = getattr(a_rows, "shape", None)
    if shape is not None:
        if len(shape) == 2:
            m, cols = int(shape[0]), int(shape[1])
            if m == 0:
                return None, 0, (cols in (0, expected_dim)), None
            if cols != expected_dim:
                return None, m, False, None
            if _is_c_contiguous_dtype(a_rows, "float64", 8):
                return a_rows.ctypes.data_as(_F64_PTR), m, True, a_rows
            flat_vals = [float(x) for row in a_rows for x in row]
            buf = (ctypes.c_double * len(flat_vals))(*flat_vals)
            return ctypes.cast(buf, _F64_PTR), m, True, buf
        if len(shape) == 1:
            total = int(shape[0])
            if total == 0:
                return None, 0, True, None
            if expected_dim <= 0 or total % expected_dim != 0:
                return None, 0, False, None
            m = total // expected_dim
            if expected_m is not None and m != expected_m:
                return None, m, False, None
            if _is_c_contiguous_dtype(a_rows, "float64", 8):
                return a_rows.ctypes.data_as(_F64_PTR), m, True, a_rows
            flat_vals = [float(x) for x in a_rows]
            buf = (ctypes.c_double * total)(*flat_vals)
            return ctypes.cast(buf, _F64_PTR), m, True, buf
        return None, 0, False, None

    rows_seq = list(a_rows)
    if len(rows_seq) == 0:
        return None, 0, True, None

    first = rows_seq[0]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        total = len(rows_seq)
        if expected_dim <= 0 or total % expected_dim != 0:
            return None, 0, False, None
        m = total // expected_dim
        if expected_m is not None and m != expected_m:
            return None, m, False, None
        vals = [float(x) for x in rows_seq]
        buf = (ctypes.c_double * total)(*vals)
        return ctypes.cast(buf, _F64_PTR), m, True, buf

    m = len(rows_seq)
    flat: List[float] = []
    for row in rows_seq:
        r_list = [float(x) for x in row]
        if len(r_list) != expected_dim:
            return None, m, False, None
        flat.extend(r_list)
    buf = (ctypes.c_double * len(flat))(*flat)
    return ctypes.cast(buf, _F64_PTR), m, True, buf


def _as_u8_buf(packed_bytes: Any) -> Tuple[Any, int, Any]:
    """Return ``(c_uint8_ptr, byte_len, keepalive)`` for bytes, bytearray, memoryview, or uint8 arrays."""
    if _is_c_contiguous_dtype(packed_bytes, "uint8", 1):
        n = int(getattr(packed_bytes, "size", len(packed_bytes)))
        ptr = packed_bytes.ctypes.data_as(_U8_PTR) if n > 0 else (ctypes.c_uint8 * 1)()
        return ctypes.cast(ptr, _U8_PTR), n, packed_bytes

    if isinstance(packed_bytes, bytearray):
        n = len(packed_bytes)
        buf = (ctypes.c_uint8 * max(n, 1)).from_buffer(packed_bytes) if n > 0 else (ctypes.c_uint8 * 1)()
        return ctypes.cast(buf, _U8_PTR), n, (packed_bytes, buf)

    if isinstance(packed_bytes, memoryview):
        raw = packed_bytes.tobytes()
        n = len(raw)
        buf = (ctypes.c_uint8 * max(n, 1)).from_buffer_copy(raw) if n > 0 else (ctypes.c_uint8 * 1)()
        return ctypes.cast(buf, _U8_PTR), n, buf

    if isinstance(packed_bytes, bytes):
        n = len(packed_bytes)
        buf = (ctypes.c_uint8 * max(n, 1)).from_buffer_copy(packed_bytes) if n > 0 else (ctypes.c_uint8 * 1)()
        return ctypes.cast(buf, _U8_PTR), n, buf

    vals = [int(b) & 0xFF for b in packed_bytes]
    n = len(vals)
    buf = (ctypes.c_uint8 * max(n, 1))(*vals) if n > 0 else (ctypes.c_uint8 * 1)()
    return ctypes.cast(buf, _U8_PTR), n, buf


def _as_i8_buf(x_int8: Any) -> Tuple[Any, int, Any]:
    """Return ``(c_int8_ptr, length, keepalive)`` with zero-copy fast path for contiguous int8 arrays."""
    if _is_c_contiguous_dtype(x_int8, "int8", 1):
        n = int(getattr(x_int8, "size", len(x_int8)))
        ptr = x_int8.ctypes.data_as(_I8_PTR) if n > 0 else (ctypes.c_int8 * 1)()
        return ctypes.cast(ptr, _I8_PTR), n, x_int8

    shape = getattr(x_int8, "shape", None)
    if shape is not None and len(shape) > 1 and hasattr(x_int8, "flat"):
        vals = [int(v) for v in x_int8.flat]
    else:
        vals = [int(v) for v in x_int8]
    n = len(vals)
    buf = (ctypes.c_int8 * max(n, 1))(*vals) if n > 0 else (ctypes.c_int8 * 1)()
    return ctypes.cast(buf, _I8_PTR), n, buf


def _configure_cdll_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Bind exact C-ABI ``argtypes`` and ``restype`` declarations on ``lib``."""
    lib.aetheris_rt_cbf_project.argtypes = [
        _F64_PTR,
        _F64_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _F64_PTR,
        _F64_PTR,
        _F64_PTR,
        ctypes.c_size_t,
        ctypes.c_double,
        _F64_PTR,
        _F64_PTR,
        _USIZE_PTR,
    ]
    lib.aetheris_rt_cbf_project.restype = ctypes.c_int32

    lib.aetheris_rt_cbf_filter_or_brake.argtypes = [
        _F64_PTR,
        _F64_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _F64_PTR,
        _F64_PTR,
        _F64_PTR,
        _F64_PTR,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_double,
        _F64_PTR,
        _F64_PTR,
        _USIZE_PTR,
    ]
    lib.aetheris_rt_cbf_filter_or_brake.restype = ctypes.c_int32

    lib.aetheris_rt_cbf_watchdog_scale.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.aetheris_rt_cbf_watchdog_scale.restype = ctypes.c_double

    lib.aetheris_rt_pd_cbf_clamp.argtypes = [
        _F64_PTR, _F64_PTR, ctypes.c_size_t, ctypes.c_size_t,
        _F64_PTR, _F64_PTR, _F64_PTR, _F64_PTR, _F64_PTR,
        _F64_PTR, _F64_PTR, _F64_PTR, _F64_PTR, _F64_PTR,
        _F64_PTR, _F64_PTR, ctypes.c_double, ctypes.c_size_t, ctypes.c_double,
        _F64_PTR, _F64_PTR, _USIZE_PTR,
    ]
    lib.aetheris_rt_pd_cbf_clamp.restype = ctypes.c_int32

    lib.aetheris_rt_ternary_matvec_i8.argtypes = [
        _U8_PTR,
        ctypes.c_size_t,
        _I8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _I32_PTR,
    ]
    lib.aetheris_rt_ternary_matvec_i8.restype = ctypes.c_int32

    return lib


class NativeRTGateLib:
    """Pure-stdlib ``ctypes.CDLL`` wrapper around the native ``aetheris_gate`` real-time C-ABI."""

    def __init__(
        self,
        cdylib_path: Optional[Union[str, Path]] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        resolved = (
            str(Path(cdylib_path).expanduser().resolve())
            if cdylib_path is not None
            else find_aetheris_gate_cdylib(repo_root=repo_root)
        )
        if not resolved or not Path(resolved).is_file():
            raise FileNotFoundError(
                "Could not locate native `aetheris_gate` shared library; build with "
                "`cargo build --release` in `rust/aetheris-gate` or set AETHERIS_GATE_CDYLIB."
            )
        norm_key = str(Path(resolved).resolve())
        with _LOCK:
            cached = _CDLL_CACHE.get(norm_key)
            if cached is None:
                cached = _configure_cdll_signatures(ctypes.CDLL(norm_key))
                _CDLL_CACHE[norm_key] = cached
        self.path: str = norm_key
        self._lib: ctypes.CDLL = cached

    @classmethod
    def get(cls, repo_root: Optional[Path] = None) -> Optional["NativeRTGateLib"]:
        """Return a cached singleton ``NativeRTGateLib`` if available, or ``None``."""
        global _DEFAULT_INSTANCE
        if repo_root is None and _DEFAULT_INSTANCE is not None and os.getenv("AETHERIS_GATE_CDYLIB") is None:
            return _DEFAULT_INSTANCE
        lib_path = find_aetheris_gate_cdylib(repo_root=repo_root)
        if not lib_path:
            return None
        try:
            inst = cls(cdylib_path=lib_path, repo_root=repo_root)
        except OSError:
            return None
        if repo_root is None and os.getenv("AETHERIS_GATE_CDYLIB") is None:
            with _LOCK:
                _DEFAULT_INSTANCE = inst
        return inst

    @classmethod
    def is_available(cls, repo_root: Optional[Path] = None) -> bool:
        """Return ``True`` when the native ``aetheris_gate`` shared library can be loaded."""
        return cls.get(repo_root=repo_root) is not None

    def cbf_project(
        self,
        A_rows: Any,
        b_vec: Any,
        u_nominal: Any,
        u_min: Any = None,
        u_max: Any = None,
        max_iters: int = 200,
        tol: float = 1e-9,
    ) -> Dict[str, Any]:
        """Project ``u_nominal`` onto ``{ A u >= b } ∩ [u_min, u_max]`` via zero-alloc native Dykstra."""
        u_ptr, dim, u_keep = _as_f64_vec(u_nominal)
        if dim == 0:
            return {
                "status": "dim_mismatch",
                "status_code": -2,
                "verified_safe": False,
                "u_filtered": [],
                "min_barrier_slack": -1.0,
                "iterations": 0,
            }

        if b_vec is None:
            b_ptr, b_len, b_keep = None, 0, None
        else:
            b_ptr, b_len, b_keep = _as_f64_vec(b_vec)

        a_ptr, m_barriers, a_dim_ok, a_keep = _as_f64_matrix(
            A_rows, expected_dim=dim, expected_m=(b_len if b_len > 0 else None)
        )
        if not a_dim_ok or (m_barriers != b_len):
            u_fallback = [float(u_ptr[i]) for i in range(dim)]
            return {
                "status": "dim_mismatch",
                "status_code": -2,
                "verified_safe": False,
                "u_filtered": u_fallback,
                "min_barrier_slack": -1.0,
                "iterations": 0,
            }

        if u_min is None:
            lo_ptr, lo_keep = None, None
        else:
            lo_ptr, lo_len, lo_keep = _as_f64_vec(u_min)
            if lo_len != dim:
                return {
                    "status": "dim_mismatch",
                    "status_code": -2,
                    "verified_safe": False,
                    "u_filtered": [float(u_ptr[i]) for i in range(dim)],
                    "min_barrier_slack": -1.0,
                    "iterations": 0,
                }

        if u_max is None:
            hi_ptr, hi_keep = None, None
        else:
            hi_ptr, hi_len, hi_keep = _as_f64_vec(u_max)
            if hi_len != dim:
                return {
                    "status": "dim_mismatch",
                    "status_code": -2,
                    "verified_safe": False,
                    "u_filtered": [float(u_ptr[i]) for i in range(dim)],
                    "min_barrier_slack": -1.0,
                    "iterations": 0,
                }

        u_out = (ctypes.c_double * dim)()
        out_slack = ctypes.c_double(0.0)
        out_iters = ctypes.c_size_t(0)

        rc = int(
            self._lib.aetheris_rt_cbf_project(
                a_ptr,
                b_ptr,
                ctypes.c_size_t(m_barriers),
                ctypes.c_size_t(dim),
                u_ptr,
                lo_ptr,
                hi_ptr,
                ctypes.c_size_t(max(0, int(max_iters))),
                ctypes.c_double(float(tol)),
                ctypes.cast(u_out, _F64_PTR),
                ctypes.byref(out_slack),
                ctypes.byref(out_iters),
            )
        )
        # Ensure zero-copy buffers stay alive across the C-ABI call
        _ = (u_keep, b_keep, a_keep, lo_keep, hi_keep)

        return {
            "status": _STATUS_CODE_TO_STR.get(rc, "invalid_input"),
            "status_code": rc,
            "verified_safe": rc == 0,
            "u_filtered": [float(u_out[i]) for i in range(dim)],
            "min_barrier_slack": float(out_slack.value) if rc in (0, 1, 2) else -1.0,
            "iterations": int(out_iters.value) if rc in (0, 1, 2) else 0,
        }

    def cbf_filter_or_brake(
        self,
        A_rows: Any,
        b_vec: Any,
        u_nominal: Any,
        u_min: Any = None,
        u_max: Any = None,
        q_dot: Any = None,
        kd: float = 2.0,
        max_iters: int = 200,
        tol: float = 1e-9,
    ) -> Dict[str, Any]:
        """Run native Dykstra CBF-QP projection with Simplex damping fallback (``-kd * q_dot``)."""
        u_ptr, dim, u_keep = _as_f64_vec(u_nominal)
        if dim == 0:
            return {
                "status": "dim_mismatch",
                "status_code": -2,
                "verified_safe": False,
                "backup_engaged": False,
                "u_filtered": [],
                "min_barrier_slack": -1.0,
                "iterations": 0,
            }

        if b_vec is None:
            b_ptr, b_len, b_keep = None, 0, None
        else:
            b_ptr, b_len, b_keep = _as_f64_vec(b_vec)

        a_ptr, m_barriers, a_dim_ok, a_keep = _as_f64_matrix(
            A_rows, expected_dim=dim, expected_m=(b_len if b_len > 0 else None)
        )
        if not a_dim_ok or (m_barriers != b_len):
            return {
                "status": "dim_mismatch",
                "status_code": -2,
                "verified_safe": False,
                "backup_engaged": False,
                "u_filtered": [float(u_ptr[i]) for i in range(dim)],
                "min_barrier_slack": -1.0,
                "iterations": 0,
            }

        if u_min is None:
            lo_ptr, lo_keep = None, None
        else:
            lo_ptr, lo_len, lo_keep = _as_f64_vec(u_min)
            if lo_len != dim:
                return {
                    "status": "dim_mismatch",
                    "status_code": -2,
                    "verified_safe": False,
                    "backup_engaged": False,
                    "u_filtered": [float(u_ptr[i]) for i in range(dim)],
                    "min_barrier_slack": -1.0,
                    "iterations": 0,
                }

        if u_max is None:
            hi_ptr, hi_keep = None, None
        else:
            hi_ptr, hi_len, hi_keep = _as_f64_vec(u_max)
            if hi_len != dim:
                return {
                    "status": "dim_mismatch",
                    "status_code": -2,
                    "verified_safe": False,
                    "backup_engaged": False,
                    "u_filtered": [float(u_ptr[i]) for i in range(dim)],
                    "min_barrier_slack": -1.0,
                    "iterations": 0,
                }

        if q_dot is None:
            q_zero = (ctypes.c_double * dim)()
            qd_ptr, qd_keep = ctypes.cast(q_zero, _F64_PTR), q_zero
        else:
            qd_ptr, qd_len, qd_keep = _as_f64_vec(q_dot)
            if qd_len != dim:
                return {
                    "status": "dim_mismatch",
                    "status_code": -2,
                    "verified_safe": False,
                    "backup_engaged": False,
                    "u_filtered": [float(u_ptr[i]) for i in range(dim)],
                    "min_barrier_slack": -1.0,
                    "iterations": 0,
                }

        u_out = (ctypes.c_double * dim)()
        out_slack = ctypes.c_double(0.0)
        out_iters = ctypes.c_size_t(0)

        rc = int(
            self._lib.aetheris_rt_cbf_filter_or_brake(
                a_ptr,
                b_ptr,
                ctypes.c_size_t(m_barriers),
                ctypes.c_size_t(dim),
                u_ptr,
                lo_ptr,
                hi_ptr,
                qd_ptr,
                ctypes.c_double(float(kd)),
                ctypes.c_size_t(max(0, int(max_iters))),
                ctypes.c_double(float(tol)),
                ctypes.cast(u_out, _F64_PTR),
                ctypes.byref(out_slack),
                ctypes.byref(out_iters),
            )
        )
        _ = (u_keep, b_keep, a_keep, lo_keep, hi_keep, qd_keep)

        return {
            "status": _STATUS_CODE_TO_STR.get(rc, "invalid_input"),
            "status_code": rc,
            "verified_safe": rc == 0,
            "backup_engaged": rc == 2,
            "u_filtered": [float(u_out[i]) for i in range(dim)],
            "min_barrier_slack": float(out_slack.value) if rc in (0, 1, 2) else -1.0,
            "iterations": int(out_iters.value) if rc in (0, 1, 2) else 0,
        }

    def pd_cbf_clamp(
        self,
        *,
        a_tau_rows: Any,
        b_tau_vec: Any,
        q: Any,
        q_dot: Any,
        q_des_nom: Any,
        q_dot_des: Any,
        tau_ff: Any,
        kp: Any,
        kd: Any,
        q_min: Any,
        q_max: Any,
        tau_max: Any,
        q_des_prev: Any = None,
        dq_max: Any = None,
        dt: float = 0.002,
        max_iters: int = 200,
        tol: float = 1e-9,
    ) -> Dict[str, Any]:
        q_ptr, dim, q_keep = _as_f64_vec(q)
        if q_ptr is None or dim <= 0:
            return {"status": "dim_mismatch", "status_code": -2, "verified_safe": False, "q_des_safe": []}
        qd_ptr, _, qd_keep = _as_f64_vec(q_dot)
        qn_ptr, _, qn_keep = _as_f64_vec(q_des_nom)
        qdn_ptr, _, qdn_keep = _as_f64_vec(q_dot_des)
        tff_ptr, _, tff_keep = _as_f64_vec(tau_ff)
        kp_ptr, _, kp_keep = _as_f64_vec(kp)
        kd_ptr, _, kd_keep = _as_f64_vec(kd)
        qlo_ptr, _, qlo_keep = _as_f64_vec(q_min)
        qhi_ptr, _, qhi_keep = _as_f64_vec(q_max)
        tmax_ptr, _, tmax_keep = _as_f64_vec(tau_max)
        qp_ptr, _, qp_keep = (None, 0, None) if q_des_prev is None else _as_f64_vec(q_des_prev)
        dqm_ptr, _, dqm_keep = (None, 0, None) if dq_max is None else _as_f64_vec(dq_max)
        b_ptr, m_b, b_keep = (None, 0, None) if b_tau_vec is None else _as_f64_vec(b_tau_vec)
        a_ptr, m_a, a_ok, a_keep = _as_f64_matrix(a_tau_rows, dim, expected_m=m_b if b_ptr is not None else None)
        if not a_ok or m_a != m_b:
            return {"status": "dim_mismatch", "status_code": -2, "verified_safe": False, "q_des_safe": []}

        q_out = (ctypes.c_double * dim)()
        out_slack = ctypes.c_double(0.0)
        out_iters = ctypes.c_size_t(0)
        rc = int(
            self._lib.aetheris_rt_pd_cbf_clamp(
                a_ptr, b_ptr, ctypes.c_size_t(m_a), ctypes.c_size_t(dim),
                q_ptr, qd_ptr, qn_ptr, qdn_ptr, tff_ptr, kp_ptr, kd_ptr,
                qlo_ptr, qhi_ptr, tmax_ptr, qp_ptr, dqm_ptr,
                ctypes.c_double(float(dt)), ctypes.c_size_t(max(0, int(max_iters))), ctypes.c_double(float(tol)),
                ctypes.cast(q_out, _F64_PTR), ctypes.byref(out_slack), ctypes.byref(out_iters),
            )
        )
        _ = (q_keep, qd_keep, qn_keep, qdn_keep, tff_keep, kp_keep, kd_keep, qlo_keep, qhi_keep, tmax_keep, qp_keep, dqm_keep, b_keep, a_keep)
        return {
            "status": _STATUS_CODE_TO_STR.get(rc, "invalid_input"),
            "status_code": rc,
            "verified_safe": rc == 0,
            "q_des_safe": [float(q_out[i]) for i in range(dim)],
            "min_barrier_slack": float(out_slack.value) if rc in (0, 1, 2) else -1.0,
            "iterations": int(out_iters.value) if rc in (0, 1, 2) else 0,
        }

    def watchdog_scale(self, age_ms: float, ttl_ms: float, ramp_ms: float) -> float:
        """Compute dead-man watchdog authority factor in ``[0.0, 1.0]``."""
        return float(
            self._lib.aetheris_rt_cbf_watchdog_scale(
                ctypes.c_double(float(age_ms)),
                ctypes.c_double(float(ttl_ms)),
                ctypes.c_double(float(ramp_ms)),
            )
        )

    def ternary_matvec_i8(
        self,
        packed_bytes: Any,
        x_int8: Any,
        out_features: int,
        in_features: int,
    ) -> List[int]:
        """Compute exact ``int32`` 2-bit packed ternary matrix-vector product via native C-ABI."""
        out_f = int(out_features)
        in_f = int(in_features)
        if out_f < 0 or in_f < 0:
            raise ValueError(f"Negative dimensions not allowed: ({out_f}, {in_f})")

        p_ptr, p_len, p_keep = _as_u8_buf(packed_bytes)
        x_ptr, x_len, x_keep = _as_i8_buf(x_int8)
        if x_len < in_f:
            raise ValueError(f"x_int8 length {x_len} is smaller than in_features={in_f}")

        out_buf = (ctypes.c_int32 * max(out_f, 1))()
        rc = int(
            self._lib.aetheris_rt_ternary_matvec_i8(
                p_ptr,
                ctypes.c_size_t(p_len),
                x_ptr,
                ctypes.c_size_t(in_f),
                ctypes.c_size_t(out_f),
                ctypes.cast(out_buf, _I32_PTR),
            )
        )
        _ = (p_keep, x_keep)
        if rc != 0:
            raise ValueError(
                f"aetheris_rt_ternary_matvec_i8 failed (rc={rc}) for shape=({out_f}, {in_f}), packed_len={p_len}"
            )
        return [int(out_buf[i]) for i in range(out_f)]


def get_native_rt_gate_lib(repo_root: Optional[Path] = None) -> Optional[NativeRTGateLib]:
    """Convenience helper returning the cached ``NativeRTGateLib`` instance if built."""
    return NativeRTGateLib.get(repo_root=repo_root)
