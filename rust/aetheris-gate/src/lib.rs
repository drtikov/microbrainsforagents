//! Library + WASM/C-ABI entrypoints for `aetheris_gate`.

pub mod dlp;
pub mod json;
pub mod layer0;
pub mod layer1_shell;

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
}
