//! Library + WASM/C-ABI entrypoints for `aetheris_gate`.

pub mod dlp;
pub mod json;
pub mod layer0;
pub mod layer1_shell;
pub mod rt_cbf;
pub mod ternary_simd;

#[cfg(not(target_arch = "wasm32"))]
pub mod ipc;
#[cfg(not(target_arch = "wasm32"))]
pub mod sandbox;

use crate::json::{obj, parse_json, to_json_string, JsonValue};
use crate::layer0::{
    extract_tool_and_input, fast_layer0_veto, first_cmd, is_protected_path, is_supported_host,
    resolved_tool_name, CliFlags,
};
use crate::layer1_shell::{analyze_shell_command, ShellDecision};

fn eval_response_json(
    decision: &str,
    reason_code: &str,
    label: &str,
    layer: i32,
    host: &str,
    profile: &str,
) -> String {
    to_json_string(&obj(vec![
        ("decision", JsonValue::Str(decision.to_string())),
        ("reason_code", JsonValue::Str(reason_code.to_string())),
        ("label", JsonValue::Str(label.to_string())),
        ("layer", JsonValue::Number(layer as f64)),
        ("host", JsonValue::Str(host.to_string())),
        ("profile", JsonValue::Str(profile.to_string())),
    ]))
}

/// Evaluate a PreToolUse JSON payload in-process (OS-independent, safe on `wasm32`).
pub fn evaluate_pretool_json(input_json: &str, host: &str, profile: &str) -> String {
    let host_low = host.trim().to_ascii_lowercase();
    let resolved_host = if is_supported_host(&host_low) {
        host_low
    } else {
        "claude".to_string()
    };
    let prof_low = {
        let p = profile.trim().to_ascii_lowercase();
        if p.is_empty() {
            "dev".to_string()
        } else {
            p
        }
    };

    let payload = match parse_json(input_json.trim()) {
        Ok(v) if v.as_object().is_some() => v,
        _ => {
            return eval_response_json(
                "deny",
                "malformed_json",
                "invalid_json_payload",
                0,
                &resolved_host,
                &prof_low,
            );
        }
    };

    let flags = CliFlags {
        host: resolved_host.clone(),
        event: "pre-tool-use".to_string(),
        shadow: false,
        log_decisions: false,
        profile: Some(prof_low.clone()),
        ..CliFlags::default()
    };

    if let Some(veto) = fast_layer0_veto(&payload, &flags) {
        return eval_response_json(
            "deny",
            &veto.reason_code,
            &veto.label,
            veto.layer,
            &resolved_host,
            &prof_low,
        );
    }

    let (raw_tool, inp) = extract_tool_and_input(&payload);
    let tool = resolved_tool_name(&raw_tool, inp);

    if tool == "execute_shell" {
        if let Some(cmd) = first_cmd(inp) {
            let sh = analyze_shell_command(cmd);
            return match sh.decision {
                ShellDecision::Deny => eval_response_json(
                    "deny",
                    &sh.reason_code,
                    &sh.label,
                    1,
                    &resolved_host,
                    &prof_low,
                ),
                ShellDecision::Allow => eval_response_json(
                    "allow",
                    &sh.reason_code,
                    &sh.label,
                    1,
                    &resolved_host,
                    &prof_low,
                ),
                ShellDecision::Escalate => eval_response_json(
                    "escalate",
                    &sh.reason_code,
                    &sh.label,
                    1,
                    &resolved_host,
                    &prof_low,
                ),
            };
        }
    }

    if matches!(tool.as_str(), "read_file" | "search_code" | "list_dir") {
        if let Some(p) = crate::layer0::first_path(inp) {
            if is_protected_path(p) {
                return eval_response_json(
                    "escalate",
                    "layer0_protected_path_read_requires_confirmation",
                    "protected_read",
                    0,
                    &resolved_host,
                    &prof_low,
                );
            }
        }
        return eval_response_json(
            "allow",
            "layer0_readonly_allow",
            "readonly_tool",
            0,
            &resolved_host,
            &prof_low,
        );
    }

    if matches!(tool.as_str(), "write_file" | "edit_file") {
        return eval_response_json(
            "allow",
            "layer0_allow",
            "non_protected_write",
            0,
            &resolved_host,
            &prof_low,
        );
    }

    eval_response_json(
        "escalate",
        "layer0_defer_to_policy",
        "unclassified_tool",
        0,
        &resolved_host,
        &prof_low,
    )
}

/// Scan and redact secrets / detect prompt injection in `text`, returning JSON.
pub fn scrub_text_json(text: &str) -> String {
    dlp::scrub_text_json(text)
}

/// Allocate `len` bytes in linear/heap memory for WASM/C-ABI callers.
#[no_mangle]
pub extern "C" fn aetheris_wasm_alloc(len: usize) -> *mut u8 {
    let cap = len.max(1);
    let mut boxed = vec![0u8; cap].into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    core::mem::forget(boxed);
    ptr
}

/// Free a buffer previously allocated by `aetheris_wasm_alloc`.
#[no_mangle]
pub extern "C" fn aetheris_wasm_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let cap = len.max(1);
    unsafe {
        let _ = Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, cap));
    }
}

unsafe fn read_utf8_cow<'a>(ptr: *const u8, len: usize) -> std::borrow::Cow<'a, str> {
    if ptr.is_null() || len == 0 {
        return std::borrow::Cow::Borrowed("");
    }
    let slice = core::slice::from_raw_parts(ptr, len);
    String::from_utf8_lossy(slice)
}

fn write_output_buf(s: &str, out_len: *mut usize) -> *mut u8 {
    let bytes = s.as_bytes();
    let ptr = aetheris_wasm_alloc(bytes.len());
    if !ptr.is_null() && !bytes.is_empty() {
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        }
    }
    if !out_len.is_null() {
        unsafe {
            *out_len = bytes.len();
        }
    }
    ptr
}

/// WASM/C-ABI PreToolUse evaluator. Caller frees returned pointer via `aetheris_wasm_dealloc(ptr, out_len)`.
#[no_mangle]
pub extern "C" fn aetheris_wasm_eval_pretool(
    input_ptr: *const u8,
    input_len: usize,
    host_ptr: *const u8,
    host_len: usize,
    profile_ptr: *const u8,
    profile_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let input = unsafe { read_utf8_cow(input_ptr, input_len) };
    let host = unsafe { read_utf8_cow(host_ptr, host_len) };
    let profile = unsafe { read_utf8_cow(profile_ptr, profile_len) };
    let res = evaluate_pretool_json(&input, &host, &profile);
    write_output_buf(&res, out_len)
}

/// WASM/C-ABI DLP text scrubber. Caller frees returned pointer via `aetheris_wasm_dealloc(ptr, out_len)`.
#[no_mangle]
pub extern "C" fn aetheris_wasm_scrub(
    text_ptr: *const u8,
    text_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let text = unsafe { read_utf8_cow(text_ptr, text_len) };
    let res = scrub_text_json(&text);
    write_output_buf(&res, out_len)
}

/// WASM/C-ABI zero-alloc Dykstra CBF-QP projection onto `{ A u >= b } ∩ [u_min, u_max]`.
///
/// Status return codes:
/// - `0` = `Safe`
/// - `1` = `UnsafeNoSolution`
/// - `2` = `SimplexFallback`
/// - `-1` = `InvalidInput` / `NullPtr`
/// - `-2` = `DimMismatch`
#[no_mangle]
pub extern "C" fn aetheris_rt_cbf_project(
    a_flat_ptr: *const f64,
    b_ptr: *const f64,
    m_barriers: usize,
    dim: usize,
    u_nom_ptr: *const f64,
    u_min_ptr: *const f64,
    u_max_ptr: *const f64,
    max_iters: usize,
    tol: f64,
    u_out_ptr: *mut f64,
    out_min_slack: *mut f64,
    out_iters: *mut usize,
) -> i32 {
    if u_nom_ptr.is_null() || u_out_ptr.is_null() {
        return rt_cbf::CBF_STATUS_INVALID_INPUT;
    }
    if m_barriers > 0 && (a_flat_ptr.is_null() || b_ptr.is_null()) {
        return rt_cbf::CBF_STATUS_INVALID_INPUT;
    }
    if dim == 0 || dim > rt_cbf::MAX_DIM || m_barriers > rt_cbf::MAX_BARRIERS {
        return rt_cbf::CBF_STATUS_DIM_MISMATCH;
    }
    if !tol.is_finite() || tol < 0.0 {
        return rt_cbf::CBF_STATUS_INVALID_INPUT;
    }

    let a_flat: &[f64] = if m_barriers == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(a_flat_ptr, m_barriers * dim) }
    };
    let b_vec: &[f64] = if m_barriers == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(b_ptr, m_barriers) }
    };
    let u_nom = unsafe { core::slice::from_raw_parts(u_nom_ptr, dim) };
    let u_min = if u_min_ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(u_min_ptr, dim) })
    };
    let u_max = if u_max_ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(u_max_ptr, dim) })
    };

    let res = rt_cbf::dykstra_cbf_project_flat(
        a_flat, b_vec, m_barriers, u_nom, u_min, u_max, max_iters, tol,
    );

    let u_out = unsafe { core::slice::from_raw_parts_mut(u_out_ptr, dim) };
    u_out.copy_from_slice(&res.u_filtered[..dim]);
    if !out_min_slack.is_null() {
        unsafe {
            *out_min_slack = res.min_barrier_slack;
        }
    }
    if !out_iters.is_null() {
        unsafe {
            *out_iters = res.iterations;
        }
    }
    res.status.to_c_code()
}

/// WASM/C-ABI zero-alloc Dykstra CBF-QP filter with Simplex velocity-damping fallback (`-kd * q_dot`).
///
/// Status return codes:
/// - `0` = `Safe`
/// - `1` = `UnsafeNoSolution`
/// - `2` = `SimplexFallback`
/// - `-1` = `InvalidInput` / `NullPtr`
/// - `-2` = `DimMismatch`
#[no_mangle]
pub extern "C" fn aetheris_rt_cbf_filter_or_brake(
    a_flat_ptr: *const f64,
    b_ptr: *const f64,
    m_barriers: usize,
    dim: usize,
    u_nom_ptr: *const f64,
    u_min_ptr: *const f64,
    u_max_ptr: *const f64,
    q_dot_ptr: *const f64,
    kd: f64,
    max_iters: usize,
    tol: f64,
    u_out_ptr: *mut f64,
    out_min_slack: *mut f64,
    out_iters: *mut usize,
) -> i32 {
    if u_nom_ptr.is_null() || u_out_ptr.is_null() || q_dot_ptr.is_null() {
        return rt_cbf::CBF_STATUS_INVALID_INPUT;
    }
    if m_barriers > 0 && (a_flat_ptr.is_null() || b_ptr.is_null()) {
        return rt_cbf::CBF_STATUS_INVALID_INPUT;
    }
    if dim == 0 || dim > rt_cbf::MAX_DIM || m_barriers > rt_cbf::MAX_BARRIERS {
        return rt_cbf::CBF_STATUS_DIM_MISMATCH;
    }
    if !tol.is_finite() || tol < 0.0 {
        return rt_cbf::CBF_STATUS_INVALID_INPUT;
    }

    let a_flat: &[f64] = if m_barriers == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(a_flat_ptr, m_barriers * dim) }
    };
    let b_vec: &[f64] = if m_barriers == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(b_ptr, m_barriers) }
    };
    let u_nom = unsafe { core::slice::from_raw_parts(u_nom_ptr, dim) };
    let u_min = if u_min_ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(u_min_ptr, dim) })
    };
    let u_max = if u_max_ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(u_max_ptr, dim) })
    };
    let q_dot = unsafe { core::slice::from_raw_parts(q_dot_ptr, dim) };

    let res = rt_cbf::rt_cbf_filter_or_brake_flat(
        a_flat, b_vec, m_barriers, u_nom, q_dot, u_min, u_max, kd, max_iters, tol,
    );

    let u_out = unsafe { core::slice::from_raw_parts_mut(u_out_ptr, dim) };
    u_out.copy_from_slice(&res.u_filtered[..dim]);
    if !out_min_slack.is_null() {
        unsafe {
            *out_min_slack = res.min_barrier_slack;
        }
    }
    if !out_iters.is_null() {
        unsafe {
            *out_iters = res.iterations;
        }
    }
    res.status.to_c_code()
}

fn cbf_json_response(
    status: &str,
    verified_safe: bool,
    u_filtered: &[f64],
    min_barrier_slack: f64,
    iterations: usize,
) -> String {
    let u_arr = JsonValue::Array(u_filtered.iter().map(|&v| JsonValue::Number(v)).collect());
    to_json_string(&obj(vec![
        ("status", JsonValue::Str(status.to_string())),
        ("verified_safe", JsonValue::Bool(verified_safe)),
        ("u_filtered", u_arr),
        ("min_barrier_slack", JsonValue::Number(min_barrier_slack)),
        ("iterations", JsonValue::Number(iterations as f64)),
    ]))
}

fn parse_f64_vec_fixed(
    val: &JsonValue,
    out: &mut [f64; rt_cbf::MAX_DIM],
) -> Result<usize, rt_cbf::CbfStatus> {
    match val {
        JsonValue::Array(items) => {
            if items.len() > rt_cbf::MAX_DIM {
                return Err(rt_cbf::CbfStatus::DimMismatch);
            }
            for (idx, item) in items.iter().enumerate() {
                match item {
                    JsonValue::Number(n) if n.is_finite() => out[idx] = *n,
                    _ => return Err(rt_cbf::CbfStatus::InvalidInput),
                }
            }
            Ok(items.len())
        }
        _ => Err(rt_cbf::CbfStatus::InvalidInput),
    }
}

/// Evaluate a real-time teleop CBF-QP JSON request (safe on `wasm32` and native).
pub fn filter_rt_cbf_json(input_json: &str) -> String {
    let payload = match parse_json(input_json.trim()) {
        Ok(v) if v.as_object().is_some() => v,
        _ => return cbf_json_response("malformed_json", false, &[], -1.0, 0),
    };

    let mut u_nom = [0.0_f64; rt_cbf::MAX_DIM];
    let dim = match payload.get("u_nominal") {
        Some(v) => match parse_f64_vec_fixed(v, &mut u_nom) {
            Ok(0) => return cbf_json_response("dim_mismatch", false, &[], -1.0, 0),
            Ok(d) => d,
            Err(st) => return cbf_json_response(st.as_str(), false, &[], -1.0, 0),
        },
        None => return cbf_json_response("invalid_input", false, &[], -1.0, 0),
    };

    let mut hs_buf = [rt_cbf::CbfHalfspace {
        a: [0.0_f64; rt_cbf::MAX_DIM],
        b: 0.0,
    }; rt_cbf::MAX_BARRIERS];
    let mut m_barriers = 0usize;

    match (payload.get("A"), payload.get("b")) {
        (None | Some(JsonValue::Null), None | Some(JsonValue::Null)) => {
            m_barriers = 0;
        }
        (None | Some(JsonValue::Null), Some(JsonValue::Array(b_items))) => {
            if !b_items.is_empty() {
                return cbf_json_response("dim_mismatch", false, &u_nom[..dim], -1.0, 0);
            }
        }
        (None | Some(JsonValue::Null), Some(_)) => {
            return cbf_json_response("invalid_input", false, &u_nom[..dim], -1.0, 0);
        }
        (Some(JsonValue::Array(a_items)), b_opt) => {
            if a_items.is_empty() {
                match b_opt {
                    None | Some(JsonValue::Null) => m_barriers = 0,
                    Some(JsonValue::Array(b_items)) if b_items.is_empty() => m_barriers = 0,
                    Some(JsonValue::Array(_)) => {
                        return cbf_json_response("dim_mismatch", false, &u_nom[..dim], -1.0, 0);
                    }
                    Some(_) => {
                        return cbf_json_response("invalid_input", false, &u_nom[..dim], -1.0, 0);
                    }
                }
            } else {
                let b_items = match b_opt {
                    Some(JsonValue::Array(items)) => items,
                    _ => return cbf_json_response("invalid_input", false, &u_nom[..dim], -1.0, 0),
                };
                if matches!(a_items[0], JsonValue::Array(_)) {
                    let m = a_items.len();
                    if m > rt_cbf::MAX_BARRIERS || b_items.len() != m {
                        return cbf_json_response("dim_mismatch", false, &u_nom[..dim], -1.0, 0);
                    }
                    for j in 0..m {
                        let row = match &a_items[j] {
                            JsonValue::Array(r) => r,
                            _ => {
                                return cbf_json_response(
                                    "invalid_input",
                                    false,
                                    &u_nom[..dim],
                                    -1.0,
                                    0,
                                )
                            }
                        };
                        if row.len() != dim {
                            return cbf_json_response(
                                "dim_mismatch",
                                false,
                                &u_nom[..dim],
                                -1.0,
                                0,
                            );
                        }
                        for c in 0..dim {
                            match row[c] {
                                JsonValue::Number(n) if n.is_finite() => hs_buf[j].a[c] = n,
                                _ => {
                                    return cbf_json_response(
                                        "invalid_input",
                                        false,
                                        &u_nom[..dim],
                                        -1.0,
                                        0,
                                    )
                                }
                            }
                        }
                        match b_items[j] {
                            JsonValue::Number(bn) if bn.is_finite() => hs_buf[j].b = bn,
                            _ => {
                                return cbf_json_response(
                                    "invalid_input",
                                    false,
                                    &u_nom[..dim],
                                    -1.0,
                                    0,
                                )
                            }
                        }
                    }
                    m_barriers = m;
                } else {
                    let m = b_items.len();
                    if m == 0 || m > rt_cbf::MAX_BARRIERS || a_items.len() != m * dim {
                        return cbf_json_response("dim_mismatch", false, &u_nom[..dim], -1.0, 0);
                    }
                    for j in 0..m {
                        for c in 0..dim {
                            match a_items[j * dim + c] {
                                JsonValue::Number(n) if n.is_finite() => hs_buf[j].a[c] = n,
                                _ => {
                                    return cbf_json_response(
                                        "invalid_input",
                                        false,
                                        &u_nom[..dim],
                                        -1.0,
                                        0,
                                    )
                                }
                            }
                        }
                        match b_items[j] {
                            JsonValue::Number(bn) if bn.is_finite() => hs_buf[j].b = bn,
                            _ => {
                                return cbf_json_response(
                                    "invalid_input",
                                    false,
                                    &u_nom[..dim],
                                    -1.0,
                                    0,
                                )
                            }
                        }
                    }
                    m_barriers = m;
                }
            }
        }
        (Some(_), _) => {
            return cbf_json_response("invalid_input", false, &u_nom[..dim], -1.0, 0);
        }
    }

    let mut u_min_buf = [0.0_f64; rt_cbf::MAX_DIM];
    let has_u_min = match payload.get("u_min") {
        None | Some(JsonValue::Null) => false,
        Some(v) => match parse_f64_vec_fixed(v, &mut u_min_buf) {
            Ok(n) if n == dim => true,
            Ok(_) => return cbf_json_response("dim_mismatch", false, &u_nom[..dim], -1.0, 0),
            Err(st) => return cbf_json_response(st.as_str(), false, &u_nom[..dim], -1.0, 0),
        },
    };

    let mut u_max_buf = [0.0_f64; rt_cbf::MAX_DIM];
    let has_u_max = match payload.get("u_max") {
        None | Some(JsonValue::Null) => false,
        Some(v) => match parse_f64_vec_fixed(v, &mut u_max_buf) {
            Ok(n) if n == dim => true,
            Ok(_) => return cbf_json_response("dim_mismatch", false, &u_nom[..dim], -1.0, 0),
            Err(st) => return cbf_json_response(st.as_str(), false, &u_nom[..dim], -1.0, 0),
        },
    };

    let mut q_dot_buf = [0.0_f64; rt_cbf::MAX_DIM];
    let has_q_dot = match payload.get("q_dot") {
        None | Some(JsonValue::Null) => false,
        Some(v) => match parse_f64_vec_fixed(v, &mut q_dot_buf) {
            Ok(n) if n == dim => true,
            Ok(_) => return cbf_json_response("dim_mismatch", false, &u_nom[..dim], -1.0, 0),
            Err(st) => return cbf_json_response(st.as_str(), false, &u_nom[..dim], -1.0, 0),
        },
    };

    let kd = match payload.get("kd") {
        None | Some(JsonValue::Null) => 2.0_f64,
        Some(JsonValue::Number(n)) if n.is_finite() && *n >= 0.0 => *n,
        _ => return cbf_json_response("invalid_input", false, &u_nom[..dim], -1.0, 0),
    };

    let max_iters = match payload.get("max_iters") {
        None | Some(JsonValue::Null) => 200usize,
        Some(JsonValue::Number(n)) if n.is_finite() && *n >= 1.0 && *n <= 100_000.0 => *n as usize,
        _ => return cbf_json_response("invalid_input", false, &u_nom[..dim], -1.0, 0),
    };

    let tol = match payload.get("tol") {
        None | Some(JsonValue::Null) => 1e-9_f64,
        Some(JsonValue::Number(n)) if n.is_finite() && *n >= 0.0 => *n,
        _ => return cbf_json_response("invalid_input", false, &u_nom[..dim], -1.0, 0),
    };

    let u_min_opt = if has_u_min {
        Some(&u_min_buf[..dim])
    } else {
        None
    };
    let u_max_opt = if has_u_max {
        Some(&u_max_buf[..dim])
    } else {
        None
    };

    let res = if has_q_dot {
        rt_cbf::rt_cbf_filter_or_brake(
            &u_nom[..dim],
            &q_dot_buf[..dim],
            &hs_buf[..m_barriers],
            u_min_opt,
            u_max_opt,
            kd,
            max_iters,
            tol,
        )
    } else {
        rt_cbf::dykstra_cbf_project(
            &u_nom[..dim],
            &hs_buf[..m_barriers],
            u_min_opt,
            u_max_opt,
            max_iters,
            tol,
        )
    };

    cbf_json_response(
        res.status.as_str(),
        res.verified_safe,
        &res.u_filtered[..res.dim],
        res.min_barrier_slack,
        res.iterations,
    )
}

/// WASM/C-ABI one-call JSON teleop CBF filter. Caller frees returned pointer via `aetheris_wasm_dealloc(ptr, out_len)`.
#[no_mangle]
pub extern "C" fn aetheris_wasm_rt_cbf_filter(
    input_ptr: *const u8,
    input_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let input = unsafe { read_utf8_cow(input_ptr, input_len) };
    let res = filter_rt_cbf_json(&input);
    write_output_buf(&res, out_len)
}

/// WASM/C-ABI dead-man watchdog authority scaler (returns 0.0..=1.0).
#[no_mangle]
pub extern "C" fn aetheris_rt_cbf_watchdog_scale(
    age_ms: f64,
    ttl_ms: f64,
    ramp_ms: f64,
) -> f64 {
    rt_cbf::watchdog_authority_scale(age_ms, ttl_ms, ramp_ms)
}

/// WASM/C-ABI zero-alloc 2-bit packed ternary matvec. Returns 0 on success, -1 on invalid inputs.
#[no_mangle]
pub extern "C" fn aetheris_rt_ternary_matvec_i8(
    packed_ptr: *const u8,
    packed_len: usize,
    x_ptr: *const i8,
    in_features: usize,
    out_features: usize,
    out_ptr: *mut i32,
) -> i32 {
    if packed_ptr.is_null() || x_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let packed = unsafe { core::slice::from_raw_parts(packed_ptr, packed_len) };
    let x_q = unsafe { core::slice::from_raw_parts(x_ptr, in_features) };
    let out = unsafe { core::slice::from_raw_parts_mut(out_ptr, out_features) };
    if ternary_simd::ternary_packed_matvec_i8(packed, x_q, out_features, in_features, out) {
        0
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_pretool_and_scrub_json_work_and_c_abi_roundtrips() {
        let deny_json = evaluate_pretool_json(
            r#"{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}"#,
            "devin",
            "dev",
        );
        let parsed_deny = parse_json(&deny_json).unwrap();
        assert_eq!(parsed_deny.get("decision").and_then(|v| v.as_str()), Some("deny"));
        assert_eq!(
            parsed_deny.get("reason_code").and_then(|v| v.as_str()),
            Some("layer1_shell_parser_veto")
        );

        let allow_json = evaluate_pretool_json(
            r#"{"tool_name":"Bash","tool_input":{"command":"git status"}}"#,
            "claude",
            "dev",
        );
        let parsed_allow = parse_json(&allow_json).unwrap();
        assert_eq!(parsed_allow.get("decision").and_then(|v| v.as_str()), Some("allow"));

        // Exercise alloc / dealloc + C-ABI entrypoints
        let sample = format!("token={} <IMPORTANT>", ["sk", "live", "test9988776655"].join("_"));
        let in_bytes = sample.as_bytes();
        let in_ptr = aetheris_wasm_alloc(in_bytes.len());
        unsafe {
            core::ptr::copy_nonoverlapping(in_bytes.as_ptr(), in_ptr, in_bytes.len());
        }
        let mut out_len: usize = 0;
        let out_ptr = aetheris_wasm_scrub(in_ptr, in_bytes.len(), &mut out_len as *mut usize);
        let out_str = unsafe {
            let s = core::slice::from_raw_parts(out_ptr, out_len);
            std::str::from_utf8(s).unwrap().to_string()
        };
        aetheris_wasm_dealloc(in_ptr, in_bytes.len());
        aetheris_wasm_dealloc(out_ptr, out_len);

        let parsed_scrub = parse_json(&out_str).unwrap();
        assert_eq!(parsed_scrub.get("secrets_found").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(parsed_scrub.get("injection_detected").and_then(|v| v.as_bool()), Some(true));
    }

    #[test]
    fn c_abi_rt_cbf_project_and_filter_or_brake_cover_all_statuses() {
        // 1. Safe projection: 2D with 2 halfspaces u0 >= 0.4, u1 <= -0.2 (i.e. -u1 >= 0.2)
        let a_flat = [1.0_f64, 0.0, 0.0, -1.0];
        let b_vec = [0.4_f64, 0.2];
        let u_nom = [-0.5_f64, 0.8];
        let u_min = [-2.0_f64, -2.0];
        let u_max = [2.0_f64, 2.0];
        let mut u_out = [0.0_f64; 2];
        let mut slack = -99.0_f64;
        let mut iters = 0usize;

        let rc = aetheris_rt_cbf_project(
            a_flat.as_ptr(),
            b_vec.as_ptr(),
            2,
            2,
            u_nom.as_ptr(),
            u_min.as_ptr(),
            u_max.as_ptr(),
            200,
            1e-10,
            u_out.as_mut_ptr(),
            &mut slack as *mut f64,
            &mut iters as *mut usize,
        );
        assert_eq!(rc, 0);
        assert!((u_out[0] - 0.4).abs() < 1e-9);
        assert!((u_out[1] - (-0.2)).abs() < 1e-9);
        assert!(slack >= -1e-9);
        assert!(iters >= 1);

        // 2. Zero barriers (m_barriers == 0) with null A/b and unbounded u_min/u_max
        let mut u_free = [0.0_f64; 2];
        let rc_zero = aetheris_rt_cbf_project(
            core::ptr::null(),
            core::ptr::null(),
            0,
            2,
            u_nom.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            200,
            1e-9,
            u_free.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        assert_eq!(rc_zero, 0);
        assert_eq!(u_free, u_nom);

        // 3. Infeasible constraints -> 1 (UnsafeNoSolution) on project, 2 (SimplexFallback) on filter_or_brake
        let a_bad = [1.0_f64, -1.0];
        let b_bad = [1.5_f64, 1.5];
        let u_nom_1 = [0.2_f64];
        let q_dot_1 = [0.6_f64];
        let lo_1 = [-1.0_f64];
        let hi_1 = [1.0_f64];
        let mut out_1 = [0.0_f64; 1];
        let mut slack_1 = 0.0_f64;

        let rc_infeas = aetheris_rt_cbf_project(
            a_bad.as_ptr(),
            b_bad.as_ptr(),
            2,
            1,
            u_nom_1.as_ptr(),
            lo_1.as_ptr(),
            hi_1.as_ptr(),
            100,
            1e-9,
            out_1.as_mut_ptr(),
            &mut slack_1 as *mut f64,
            core::ptr::null_mut(),
        );
        assert_eq!(rc_infeas, 1);

        let rc_brake = aetheris_rt_cbf_filter_or_brake(
            a_bad.as_ptr(),
            b_bad.as_ptr(),
            2,
            1,
            u_nom_1.as_ptr(),
            lo_1.as_ptr(),
            hi_1.as_ptr(),
            q_dot_1.as_ptr(),
            2.0,
            100,
            1e-9,
            out_1.as_mut_ptr(),
            &mut slack_1 as *mut f64,
            core::ptr::null_mut(),
        );
        assert_eq!(rc_brake, 2);
        // -2.0 * 0.6 = -1.2 -> clipped to lo_1 = -1.0
        assert!((out_1[0] - (-1.0)).abs() < 1e-12);

        // 4. Null pointer and dimension mismatch error codes (-1 and -2)
        assert_eq!(
            aetheris_rt_cbf_project(
                core::ptr::null(),
                b_bad.as_ptr(),
                1,
                1,
                u_nom_1.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                50,
                1e-9,
                out_1.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
            -1
        );
        assert_eq!(
            aetheris_rt_cbf_project(
                core::ptr::null(),
                core::ptr::null(),
                0,
                0,
                u_nom_1.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                50,
                1e-9,
                out_1.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
            -2
        );
        assert_eq!(
            aetheris_rt_cbf_filter_or_brake(
                core::ptr::null(),
                core::ptr::null(),
                0,
                1,
                u_nom_1.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                2.0,
                50,
                1e-9,
                out_1.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
            -1
        );
    }

    #[test]
    fn wasm_rt_cbf_filter_json_and_c_abi_roundtrip() {
        let req_safe = r#"{"u_nominal":[-1.0,0.5],"A":[[2.0,0.0]],"b":[1.0],"u_min":[-2.0,-2.0],"u_max":[2.0,2.0],"q_dot":[0.4,-0.2],"kd":2.0,"max_iters":200,"tol":1e-9}"#;
        let in_b = req_safe.as_bytes();
        let mut out_len: usize = 0;
        let out_ptr = aetheris_wasm_rt_cbf_filter(in_b.as_ptr(), in_b.len(), &mut out_len as *mut usize);
        let raw_safe = unsafe {
            let s = core::slice::from_raw_parts(out_ptr, out_len);
            std::str::from_utf8(s).unwrap().to_string()
        };
        aetheris_wasm_dealloc(out_ptr, out_len);

        let parsed = parse_json(&raw_safe).unwrap();
        assert_eq!(parsed.get("status").and_then(|v| v.as_str()), Some("safe"));
        assert_eq!(parsed.get("verified_safe").and_then(|v| v.as_bool()), Some(true));
        if let Some(JsonValue::Array(u_arr)) = parsed.get("u_filtered") {
            assert_eq!(u_arr.len(), 2);
            if let JsonValue::Number(u0) = u_arr[0] {
                assert!((u0 - 0.5).abs() < 1e-9);
            } else {
                panic!("expected number");
            }
        } else {
            panic!("expected u_filtered array");
        }

        // Infeasible with q_dot -> simplex_braking_engaged
        let req_simplex = r#"{"u_nominal":[0.5],"A":[[1.0],[-1.0]],"b":[1.2,1.2],"u_min":[-1.0],"u_max":[1.0],"q_dot":[0.3],"kd":2.0}"#;
        let parsed_sim = parse_json(&filter_rt_cbf_json(req_simplex)).unwrap();
        assert_eq!(
            parsed_sim.get("status").and_then(|v| v.as_str()),
            Some("simplex_braking_engaged")
        );
        assert_eq!(parsed_sim.get("verified_safe").and_then(|v| v.as_bool()), Some(false));

        // Infeasible without q_dot -> unsafe_no_solution
        let req_unsafe = r#"{"u_nominal":[0.5],"A":[[1.0],[-1.0]],"b":[1.2,1.2],"u_min":[-1.0],"u_max":[1.0]}"#;
        let parsed_un = parse_json(&filter_rt_cbf_json(req_unsafe)).unwrap();
        assert_eq!(
            parsed_un.get("status").and_then(|v| v.as_str()),
            Some("unsafe_no_solution")
        );
        assert_eq!(parsed_un.get("verified_safe").and_then(|v| v.as_bool()), Some(false));

        // Watchdog scale C-ABI check
        assert!((aetheris_rt_cbf_watchdog_scale(25.0, 50.0, 100.0) - 1.0).abs() < 1e-12);
        assert!((aetheris_rt_cbf_watchdog_scale(100.0, 50.0, 100.0) - 0.5).abs() < 1e-12);
    }
}
