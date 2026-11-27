//! `aetheris-gate`: Native sub-millisecond Layer 0/1 gate, DLP scrubber, warm IPC client, and OS sandbox runner.

use std::env;
use std::io::{self, Read};
use std::process;

use aetheris_gate::dlp::scrub_text_json;
use aetheris_gate::ipc::{
    emit_daemon_verdict, fallback_python_hook, query_warm_daemon, resolve_state_dir,
    spawn_daemon_detached,
};
use aetheris_gate::json::{obj, parse_json, to_json_string, JsonValue};
use aetheris_gate::layer0::{emit_veto, fast_layer0_veto, parse_cli_flags};
use aetheris_gate::sandbox::run_sandbox_command;

fn run_self_test() -> i32 {
    let mut flags = parse_cli_flags(&[
        "--host".to_string(),
        "antigravity".to_string(),
    ]);
    flags.shadow = false;
    flags.log_decisions = false;
    let probe = parse_json(
        r#"{"toolCall":{"name":"write_file","args":{"Path":".env","Content":"K=1"}}}"#,
    )
    .expect("valid json");
    let veto = fast_layer0_veto(&probe, &flags);
    let ok = veto
        .as_ref()
        .is_some_and(|v| v.reason_code == "layer0_protected_path_veto");
    println!(
        "{}",
        to_json_string(&obj(vec![
            ("ok", JsonValue::Bool(ok)),
            ("binary", JsonValue::Str("aetheris-gate".to_string())),
            ("version", JsonValue::Str("0.7.7".to_string())),
            ("layer0_native_veto", JsonValue::Bool(ok)),
            (
                "state_dir",
                JsonValue::Str(resolve_state_dir().to_string_lossy().to_string()),
            ),
        ]))
    );
    if ok { 0 } else { 1 }
}

fn run_scrub_command() -> i32 {
    let mut raw_stdin = Vec::new();
    if let Err(e) = io::stdin().read_to_end(&mut raw_stdin) {
        eprintln!("aetheris-gate scrub: failed to read stdin: {e}");
        return 2;
    }
    let text = String::from_utf8_lossy(&raw_stdin);
    println!("{}", scrub_text_json(&text));
    0
}

fn main() {
    let raw_args: Vec<String> = env::args().skip(1).collect();

    if raw_args.first().map(String::as_str) == Some("--self-test") {
        process::exit(run_self_test());
    }
    if raw_args.first().map(String::as_str) == Some("sandbox") {
        process::exit(run_sandbox_command(&raw_args[1..]));
    }
    if raw_args.first().map(String::as_str) == Some("scrub") {
        process::exit(run_scrub_command());
    }

    let flags = parse_cli_flags(&raw_args);
    let mut raw_stdin = Vec::new();
    if io::stdin().read_to_end(&mut raw_stdin).is_err() {
        process::exit(fallback_python_hook(&raw_stdin, &raw_args));
    }

    let trimmed = String::from_utf8_lossy(&raw_stdin);
    let trimmed = trimmed.trim();
    if trimmed.is_empty() {
        process::exit(fallback_python_hook(&raw_stdin, &raw_args));
    }

    let payload = match parse_json(trimmed) {
        Ok(v) if v.as_object().is_some() => v,
        _ => process::exit(fallback_python_hook(&raw_stdin, &raw_args)),
    };

    // 1. Sub-100us native Rust Layer 0/1 hard veto (skipped when journal logging/shadow needs Python)
    if let Some(veto) = fast_layer0_veto(&payload, &flags) {
        process::exit(emit_veto(&veto, &flags));
    }

    // 2. Warm IPC daemon fast path (< 3 ms, zero Python interpreter spawn)
    let cwd = env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    if env::var("AETHERIS_STATE_DIR").map_or(true, |v| v.trim().is_empty()) {
        if let Some(verdict) = query_warm_daemon(&payload, &flags, &cwd) {
            process::exit(emit_daemon_verdict(&verdict, &flags));
        }
        if flags.auto_daemon && env::var("AETHERIS_ALLOW_POLICY_EDIT").as_deref() != Ok("1") {
            spawn_daemon_detached(&resolve_state_dir());
        }
    } else if let Some(verdict) = query_warm_daemon(&payload, &flags, &cwd) {
        process::exit(emit_daemon_verdict(&verdict, &flags));
    }

    // 3. Cold fallback: delegate losslessly to `python -m aetheris_microbrains.hook`
    process::exit(fallback_python_hook(&raw_stdin, &raw_args));
}
