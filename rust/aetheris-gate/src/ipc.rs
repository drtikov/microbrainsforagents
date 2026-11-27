//! Warm hook daemon IPC client (TCP + Unix domain sockets) with detached auto-spawn and Python fallback.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use crate::json::{obj, parse_json, to_json_string, JsonValue};
use crate::layer0::CliFlags;

pub fn resolve_state_dir() -> PathBuf {
    if let Ok(ov) = env::var("AETHERIS_STATE_DIR") {
        let trimmed = ov.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if cfg!(windows) {
        if let Ok(local) = env::var("LOCALAPPDATA") {
            let trimmed = local.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join("aetheris-microbrains");
            }
        }
        let home = env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
        return PathBuf::from(home)
            .join("AppData")
            .join("Local")
            .join("aetheris-microbrains");
    }
    if cfg!(target_os = "macos") {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("aetheris-microbrains");
    }
    if let Ok(xdg) = env::var("XDG_STATE_HOME") {
        let trimmed = xdg.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("aetheris-microbrains");
        }
    }
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".local")
        .join("state")
        .join("aetheris-microbrains")
}

fn find_python_executable() -> String {
    if let Ok(py) = env::var("AETHERIS_PYTHON") {
        if !py.trim().is_empty() {
            return py;
        }
    }
    "python".to_string()
}

#[cfg(windows)]
fn clear_std_handle_inheritance() {
    extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> isize;
        fn SetHandleInformation(hObject: isize, dwMask: u32, dwFlags: u32) -> i32;
    }
    const HANDLE_FLAG_INHERIT: u32 = 0x0000_0001;
    for std_id in [(-10i32) as u32, (-11i32) as u32, (-12i32) as u32] {
        unsafe {
            let h = GetStdHandle(std_id);
            if h != 0 && h != -1 {
                SetHandleInformation(h, HANDLE_FLAG_INHERIT, 0);
            }
        }
    }
}

pub fn spawn_daemon_detached(state_dir: &Path) {
    let py = find_python_executable();
    let state_arg = state_dir.to_string_lossy().to_string();
    #[cfg(windows)]
    clear_std_handle_inheritance();
    let mut cmd = Command::new(py);
    cmd.args([
        "-m",
        "aetheris_microbrains.hook_daemon",
        "--idle-timeout",
        "900",
        "--state-dir",
        &state_arg,
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    #[cfg(windows)]
    {
        cmd.creation_flags(0x08000000 | 0x00000008); // CREATE_NO_WINDOW | DETACHED_PROCESS
    }
    let _ = cmd.spawn();
}

fn build_daemon_request(
    payload: &JsonValue,
    flags: &CliFlags,
    auth: &str,
    cwd: &str,
) -> JsonValue {
    let daemon_ev = match flags.event.as_str() {
        "post-tool-use" => "post",
        "pre-tool-use" => "pre",
        other => other,
    };
    let mut flag_entries = vec![
        ("host", JsonValue::Str(flags.host.clone())),
        ("cwd", JsonValue::Str(cwd.to_string())),
    ];
    if flags.strict_ast {
        flag_entries.push(("strict_ast", JsonValue::Bool(true)));
    }
    if flags.log_decisions {
        flag_entries.push(("log_decisions", JsonValue::Bool(true)));
    }
    if flags.shadow {
        flag_entries.push(("shadow", JsonValue::Bool(true)));
    }
    if flags.auto_allow {
        flag_entries.push(("auto_allow", JsonValue::Bool(true)));
    }
    if flags.codex_escalate_deny {
        flag_entries.push(("codex_escalate_deny", JsonValue::Bool(true)));
    }
    if flags.replace_tool_output {
        flag_entries.push(("replace_tool_output", JsonValue::Bool(true)));
    }
    if flags.semantic_guard {
        flag_entries.push(("semantic_guard", JsonValue::Bool(true)));
    }
    if flags.context_guard {
        flag_entries.push(("context_guard", JsonValue::Bool(true)));
    }
    if flags.approve_once {
        flag_entries.push(("approve_once", JsonValue::Bool(true)));
    }
    if flags.auto_approve_escalations {
        flag_entries.push(("auto_approve_escalations", JsonValue::Bool(true)));
    }
    if let Some(ap) = &flags.approvals_path {
        flag_entries.push(("approvals_path", JsonValue::Str(ap.clone())));
    }
    if flags.sandbox_wrap != "none" && !flags.sandbox_wrap.is_empty() {
        flag_entries.push(("sandbox_wrap", JsonValue::Str(flags.sandbox_wrap.clone())));
    }
    if let Some(jp) = &flags.journal_path {
        flag_entries.push(("journal_path", JsonValue::Str(jp.clone())));
    }
    if let Some(sp) = &flags.session_state {
        flag_entries.push(("session_state_path", JsonValue::Str(sp.clone())));
    }
    if let Some(prof) = &flags.profile {
        flag_entries.push(("profile", JsonValue::Str(prof.clone())));
    }
    if let Some(pol) = &flags.policy {
        flag_entries.push(("policy_path", JsonValue::Str(pol.clone())));
    }
    let exp_arr = JsonValue::Array(
        flags
            .explicit_flags
            .iter()
            .map(|s| JsonValue::Str(s.clone()))
            .collect(),
    );
    let req_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| format!("{:x}", d.as_nanos()))
        .unwrap_or_else(|_| "1".to_string());

    obj(vec![
        ("id", JsonValue::Str(req_id)),
        ("auth", JsonValue::Str(auth.to_string())),
        ("event", JsonValue::Str(daemon_ev.to_string())),
        ("payload", payload.clone()),
        ("flags", obj(flag_entries)),
        ("explicit_flags", exp_arr),
    ])
}

fn roundtrip_stream<S: Read + Write>(stream: &mut S, req_line: &[u8]) -> Option<JsonValue> {
    stream.write_all(req_line).ok()?;
    stream.flush().ok()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let n = reader.read_line(&mut line).ok()?;
    if n == 0 {
        return None;
    }
    let resp = parse_json(line.trim()).ok()?;
    if resp.get("error").is_some_and(|e| !e.is_null()) {
        return None;
    }
    let verdict = resp.get("verdict")?;
    if verdict.as_object().is_some() {
        Some(verdict.clone())
    } else {
        None
    }
}

pub fn query_warm_daemon(payload: &JsonValue, flags: &CliFlags, cwd: &str) -> Option<JsonValue> {
    if env::var("AETHERIS_ALLOW_POLICY_EDIT").as_deref() == Ok("1") {
        return None;
    }
    let state_dir = resolve_state_dir();
    let state_file = state_dir.join("hook_daemon.json");
    let raw = fs::read_to_string(&state_file).ok()?;
    let info = parse_json(&raw).ok()?;
    if info.get("version").and_then(|v| v.as_i64()) != Some(1) {
        return None;
    }
    let mode = info
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("tcp")
        .to_ascii_lowercase();
    if mode == "http" {
        return None;
    }
    let auth = info.get("auth").and_then(|v| v.as_str()).unwrap_or("");
    if auth.is_empty() {
        return None;
    }
    let req = build_daemon_request(payload, flags, auth, cwd);
    let req_line = format!("{}\n", to_json_string(&req));

    if mode == "uds" {
        #[cfg(unix)]
        {
            use std::os::unix::net::UnixStream;
            let sock_path = info.get("socket_path").and_then(|v| v.as_str())?;
            let mut stream = UnixStream::connect(sock_path).ok()?;
            let _ = stream.set_read_timeout(Some(Duration::from_millis(1500)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
            return roundtrip_stream(&mut stream, req_line.as_bytes());
        }
        #[cfg(not(unix))]
        return None;
    }

    let port = info.get("port").and_then(|v| v.as_i64())? as u16;
    if port == 0 {
        return None;
    }
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().ok()?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(200)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_millis(1500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    roundtrip_stream(&mut stream, req_line.as_bytes())
}

pub fn emit_daemon_verdict(verdict: &JsonValue, flags: &CliFlags) -> i32 {
    if flags.shadow {
        if flags.host == "antigravity" {
            let fallback = if flags.event == "post-tool-use" {
                "{}"
            } else {
                r#"{"decision":"ask"}"#
            };
            println!("{fallback}");
        }
        return 0;
    }

    let exit_code = verdict.get("exit_code").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    if flags.host == "generic" {
        if exit_code == 2 {
            if let Some(r) = verdict.get("reason").and_then(|v| v.as_str()) {
                eprintln!("{r}");
            }
        }
        println!("{}", to_json_string(verdict));
        return exit_code;
    }

    let hso = verdict.get("hookSpecificOutput");
    if flags.event == "post-tool-use" {
        if flags.host == "antigravity" {
            let out = hso.filter(|v| v.as_object().is_some()).cloned().unwrap_or_else(|| obj(vec![]));
            println!("{}", to_json_string(&out));
            return 0;
        }
        if let Some(h) = hso.filter(|v| !v.is_null()) {
            if flags.host == "hermes" {
                println!("{}", to_json_string(h));
            } else {
                println!("{}", to_json_string(&obj(vec![("hookSpecificOutput", h.clone())])));
            }
        }
        return exit_code;
    }

    let perm = verdict
        .get("permissionDecision")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if flags.host == "antigravity" {
        let has_overwrite = hso.and_then(|v| v.get("overwrite")).is_some();
        if perm == "allow" && !flags.auto_allow && !has_overwrite {
            println!(r#"{{"decision":"ask"}}"#);
            return 0;
        }
        let out = hso
            .filter(|v| v.as_object().is_some())
            .cloned()
            .unwrap_or_else(|| obj(vec![("decision", JsonValue::Str("ask".to_string()))]));
        println!("{}", to_json_string(&out));
        return 0;
    }

    if exit_code == 2 {
        if let Some(r) = verdict.get("reason").and_then(|v| v.as_str()) {
            eprintln!("{r}");
        }
        if let Some(h) = hso.filter(|v| !v.is_null()) {
            if flags.host == "hermes" {
                println!("{}", to_json_string(h));
            } else {
                println!("{}", to_json_string(&obj(vec![("hookSpecificOutput", h.clone())])));
            }
        }
        return 2;
    }

    let host_action = verdict
        .get("host_action")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if (flags.host == "codex" || flags.host == "kimi") && host_action == "defer_to_host_policy" {
        return 0;
    }
    let has_updated_input = hso.and_then(|v| v.get("updatedInput")).is_some();
    if perm == "allow" && !flags.auto_allow && !has_updated_input {
        return 0;
    }
    if let Some(h) = hso.filter(|v| !v.is_null()) {
        if flags.host == "hermes" {
            println!("{}", to_json_string(h));
        } else {
            println!("{}", to_json_string(&obj(vec![("hookSpecificOutput", h.clone())])));
        }
    }
    exit_code
}

pub fn fallback_python_hook(raw_stdin: &[u8], raw_args: &[String]) -> i32 {
    let py = find_python_executable();
    let mut cmd = Command::new(py);
    cmd.args(["-m", "aetheris_microbrains.hook"]);
    for a in raw_args {
        if a != "--auto-daemon" {
            cmd.arg(a);
        }
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("aetheris-gate: failed to spawn python fallback: {e}");
            return 2;
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(raw_stdin);
    }
    match child.wait() {
        Ok(st) => st.code().unwrap_or(2),
        Err(_) => 2,
    }
}
