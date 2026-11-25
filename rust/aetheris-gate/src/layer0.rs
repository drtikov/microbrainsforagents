//! Native sub-millisecond Layer 0 veto + host output contract matching `hook.py`.

use std::env;

use crate::json::{obj, to_json_string, JsonValue};

#[derive(Debug, Clone)]
pub struct CliFlags {
    pub host: String,
    pub event: String,
    pub shadow: bool,
    pub auto_allow: bool,
    pub strict_ast: bool,
    pub log_decisions: bool,
    pub codex_escalate_deny: bool,
    pub replace_tool_output: bool,
    pub semantic_guard: bool,
    pub context_guard: bool,
    pub approve_once: bool,
    pub auto_approve_escalations: bool,
    pub auto_daemon: bool,
    pub profile: Option<String>,
    pub policy: Option<String>,
    pub journal_path: Option<String>,
    pub session_state: Option<String>,
    pub approvals_path: Option<String>,
    pub sandbox_wrap: String,
    pub explicit_flags: Vec<String>,
}

impl Default for CliFlags {
    fn default() -> Self {
        let shadow = env::var("AETHERIS_HOOK_SHADOW").as_deref() == Ok("1");
        let raw_host = env::var("AETHERIS_HOOK_HOST")
            .unwrap_or_else(|_| "claude".to_string())
            .trim()
            .to_ascii_lowercase();
        let host = match raw_host.as_str() {
            "claude" | "codex" | "hermes" | "antigravity" | "openclaw" | "opencode" | "kilocode"
            | "pi" | "omp" | "qoder" | "zcode" | "openhands" | "cursor" | "generic" => raw_host,
            _ => "claude".to_string(),
        };
        Self {
            host,
            event: "pre-tool-use".to_string(),
            shadow,
            auto_allow: env::var("AETHERIS_HOOK_AUTO_ALLOW").as_deref() == Ok("1"),
            strict_ast: false,
            log_decisions: env::var("AETHERIS_LOG_DECISIONS").as_deref() == Ok("1"),
            codex_escalate_deny: false,
            replace_tool_output: env::var("AETHERIS_POST_REPLACE_OUTPUT").as_deref() == Ok("1"),
            semantic_guard: env::var("AETHERIS_SEMANTIC_GUARD").as_deref() == Ok("1"),
            context_guard: env::var("AETHERIS_CONTEXT_GUARD").as_deref() == Ok("1"),
            approve_once: env::var("AETHERIS_APPROVE_ONCE").as_deref() == Ok("1"),
            auto_approve_escalations: env::var("AETHERIS_AUTO_APPROVE_ESCALATIONS").as_deref() == Ok("1"),
            auto_daemon: env::var("AETHERIS_AUTO_DAEMON").as_deref() == Ok("1"),
            profile: env::var("AETHERIS_PROFILE")
                .ok()
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty()),
            policy: env::var("AETHERIS_POLICY_PATH")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            journal_path: env::var("AETHERIS_JOURNAL_PATH")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            session_state: env::var("AETHERIS_SESSION_STATE_PATH")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            approvals_path: env::var("AETHERIS_APPROVALS_PATH")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            sandbox_wrap: env::var("AETHERIS_SANDBOX_WRAP")
                .ok()
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "none".to_string()),
            explicit_flags: Vec::new(),
        }
    }
}

pub fn parse_cli_flags(args: &[String]) -> CliFlags {
    let mut cfg = CliFlags::default();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |idx: &mut usize| -> Option<String> {
            if *idx + 1 < args.len() {
                *idx += 1;
                Some(args[*idx].clone())
            } else {
                None
            }
        };
        match a {
            "--host" => {
                if let Some(v) = next(&mut i) {
                    let low = v.trim().to_ascii_lowercase();
                    if matches!(
                        low.as_str(),
                        "claude"
                            | "codex"
                            | "hermes"
                            | "antigravity"
                            | "openclaw"
                            | "opencode"
                            | "kilocode"
                            | "pi"
                            | "omp"
                            | "qoder"
                            | "zcode"
                            | "openhands"
                            | "cursor"
                            | "generic"
                    ) {
                        cfg.host = low;
                    }
                }
            }
            "--event" => {
                if let Some(v) = next(&mut i) {
                    cfg.event = v.trim().to_ascii_lowercase();
                }
            }
            "--shadow" => {
                cfg.shadow = true;
                cfg.explicit_flags.push("shadow".to_string());
            }
            "--auto-allow" => {
                cfg.auto_allow = true;
                cfg.explicit_flags.push("auto_allow".to_string());
            }
            "--strict-ast" => {
                cfg.strict_ast = true;
                cfg.explicit_flags.push("strict_ast".to_string());
            }
            "--log-decisions" => {
                cfg.log_decisions = true;
                cfg.explicit_flags.push("log_decisions".to_string());
            }
            "--codex-escalate-deny" => {
                cfg.codex_escalate_deny = true;
                cfg.explicit_flags.push("codex_escalate_deny".to_string());
            }
            "--replace-tool-output" => {
                cfg.replace_tool_output = true;
                cfg.explicit_flags.push("replace_tool_output".to_string());
            }
            "--semantic-guard" => {
                cfg.semantic_guard = true;
                cfg.explicit_flags.push("semantic_guard".to_string());
            }
            "--context-guard" => {
                cfg.context_guard = true;
                cfg.explicit_flags.push("context_guard".to_string());
            }
            "--approve-once" => {
                cfg.approve_once = true;
                cfg.explicit_flags.push("approve_once".to_string());
            }
            "--auto-approve-escalations" => {
                cfg.auto_approve_escalations = true;
                cfg.explicit_flags.push("auto_approve_escalations".to_string());
            }
            "--auto-daemon" => cfg.auto_daemon = true,
            "--profile" => {
                if let Some(v) = next(&mut i) {
                    cfg.profile = Some(v.trim().to_ascii_lowercase());
                }
            }
            "--policy" => {
                if let Some(v) = next(&mut i) {
                    cfg.policy = Some(v);
                }
            }
            "--journal-path" => {
                if let Some(v) = next(&mut i) {
                    cfg.journal_path = Some(v);
                }
            }
            "--session-state" => {
                if let Some(v) = next(&mut i) {
                    cfg.session_state = Some(v);
                }
            }
            "--approvals-path" => {
                if let Some(v) = next(&mut i) {
                    cfg.approvals_path = Some(v);
                }
            }
            "--sandbox-wrap" => {
                if let Some(v) = next(&mut i) {
                    cfg.sandbox_wrap = v.trim().to_ascii_lowercase();
                    cfg.explicit_flags.push("sandbox_wrap".to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    cfg
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastVeto {
    pub reason_code: String,
    pub label: String,
    pub layer: i32,
}

const PATH_KEYS: &[&str] = &[
    "file_path",
    "notebook_path",
    "path",
    "Path",
    "TargetFile",
    "AbsolutePath",
    "SearchPath",
    "SearchDirectory",
    "DirectoryPath",
];
const CMD_KEYS: &[&str] = &["command", "cmd", "CommandLine"];
const DIFF_KEYS: &[&str] = &[
    "diff",
    "payload_diff",
    "patch",
    "new_string",
    "new_str",
    "file_text",
    "Replacement",
    "ReplacementContent",
    "content",
    "Content",
    "CodeContent",
    "new_source",
];

fn extract_tool_and_input<'a>(payload: &'a JsonValue) -> (String, &'a JsonValue) {
    let tc = payload.get("toolCall");
    let raw_tool = payload
        .get("tool_name")
        .or_else(|| payload.get("tool"))
        .or_else(|| payload.get("name"))
        .or_else(|| tc.and_then(|c| c.get("name")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let inp = payload
        .get("tool_input")
        .filter(|v| v.as_object().is_some())
        .or_else(|| payload.get("input").filter(|v| v.as_object().is_some()))
        .or_else(|| {
            tc.and_then(|c| c.get("args"))
                .filter(|v| v.as_object().is_some())
        })
        .unwrap_or(payload);
    (raw_tool, inp)
}

fn canonical_tool(raw_low: &str) -> &str {
    match raw_low {
        "bash" | "shell" | "shell_command" | "exec_command" | "execute_shell" | "execute_bash"
        | "run_command" | "terminal" => "execute_shell",
        "edit" | "str_replace_editor" | "file_editor" | "edit_file" | "multiedit"
        | "notebookedit" | "patch" | "apply_patch" | "replace_file_content"
        | "multi_replace_file_content" => "edit_file",
        "write" | "write_file" | "repo_write" | "fwrite" | "write_to_file" => "write_file",
        "read" | "read_file" | "repo_read" | "view_file" => "read_file",
        "grep" | "grep_search" | "search_code" | "find_by_name" => "search_code",
        "glob" | "list_dir" => "list_dir",
        _ => raw_low,
    }
}

fn is_control_plane_segment(seg: &str) -> bool {
    matches!(
        seg,
        ".claude"
            | ".codex"
            | ".cursor"
            | ".qoder"
            | ".opencode"
            | ".kilo"
            | ".pi"
            | ".omp"
            | ".zcode"
            | ".zcode-plugin"
            | ".openhands"
            | ".mcp.json"
            | ".vscode"
    )
}

pub fn is_agent_control_plane_path(path: &str) -> bool {
    let raw_norm = path.replace('\\', "/").trim().to_ascii_lowercase();
    if raw_norm.is_empty() || raw_norm.contains('\0') {
        return false;
    }
    let segs: Vec<&str> = raw_norm.split('/').filter(|s| !s.is_empty()).collect();
    if segs.iter().any(|s| *s == "..") {
        return false;
    }
    const CRED_SEGS: &[&str] = &[
        ".env",
        ".ssh",
        ".aws",
        ".git",
        "credentials",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "secrets",
        "integrity",
        "bible.md",
        "axioms",
        ".pem",
        ".key",
    ];
    if CRED_SEGS.iter().any(|c| raw_norm.contains(c)) {
        return false;
    }
    segs.iter().any(|s| is_control_plane_segment(s))
}

pub fn is_protected_path(path: &str) -> bool {
    let norm = path.replace('\\', "/").trim().to_ascii_lowercase();
    if norm.is_empty() {
        return false;
    }
    if norm.contains('\0') {
        return true;
    }
    let segs: Vec<&str> = norm.split('/').filter(|s| !s.is_empty()).collect();
    let basename = segs.last().copied().unwrap_or("");
    if segs.iter().any(|s| *s == ".env" || s.starts_with(".env.")) {
        return true;
    }
    const EXACT_BASENAMES: &[&str] = &[
        ".env",
        ".envrc",
        ".npmrc",
        ".netrc",
        ".pypirc",
        ".pgpass",
        ".my.cnf",
        ".vault-token",
        ".claude.json",
        ".gitconfig",
        ".bashrc",
        ".zshrc",
        ".profile",
        ".bash_profile",
        ".zprofile",
        "authorized_keys",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "credentials.json",
        "secrets.yaml",
        "secrets.json",
        "hosts.yml",
        "bible.md",
        ".cursorrules",
        ".windsurfrules",
        ".gitlab-ci.yml",
        ".pre-commit-config.yaml",
        ".gitmodules",
        "claude.md",
        "agents.md",
    ];
    if EXACT_BASENAMES.contains(&basename) {
        return true;
    }
    if [".pem", ".key", ".p12", ".pfx"]
        .iter()
        .any(|ext| basename.ends_with(ext))
    {
        return true;
    }
    for s in &segs {
        if matches!(
            *s,
            ".ssh"
                | ".gnupg"
                | ".azure"
                | ".claude"
                | ".codex"
                | ".cursor"
                | ".qoder"
                | ".opencode"
                | ".kilo"
                | ".pi"
                | ".omp"
                | ".zcode"
                | ".zcode-plugin"
                | ".openhands"
                | "constitution"
        ) {
            return true;
        }
    }
    const MULTI_SEGS: &[(&str, &str)] = &[
        (".git", "hooks"),
        (".git", "config"),
        (".github", "workflows"),
        (".aws", "credentials"),
        (".aws", "config"),
        (".kube", "config"),
        (".docker", "config.json"),
        (".config", "gcloud"),
        (".cargo", "credentials"),
        (".cargo", "credentials.toml"),
        (".gem", "credentials"),
        (".vscode", "tasks.json"),
        (".vscode", "settings.json"),
    ];
    for w in segs.windows(2) {
        for (a, b) in MULTI_SEGS {
            if w[0] == *a && w[1] == *b {
                return true;
            }
        }
    }
    norm.contains(".mcp.json")
        || norm.contains("integrity_seal.json")
        || norm.contains("immutable_axioms")
}

fn is_catastrophic_shell_cmd(cmd: &str) -> Option<&'static str> {
    let low = cmd.to_ascii_lowercase();
    if low.contains(":(){ :|:& };:") || low.contains(":(){:|:&};:") {
        return Some("fork_bomb");
    }
    // Check catastrophic `rm -rf /` style root wipes across pipeline segments
    for seg in low.split(|c| matches!(c, '\n' | ';' | '|')) {
        let seg_clean = seg.trim();
        let toks: Vec<&str> = seg_clean.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        let mut idx = 0;
        if toks[0] == "sudo" && toks.len() > 1 {
            idx = 1;
        }
        if toks[idx] == "rm" {
            let mut has_r = false;
            let mut has_f = false;
            let mut targets_root = false;
            for t in &toks[idx + 1..] {
                if *t == "--recursive" {
                    has_r = true;
                } else if *t == "--force" {
                    has_f = true;
                } else if t.starts_with('-') && !t.starts_with("--") {
                    if t.contains('r') {
                        has_r = true;
                    }
                    if t.contains('f') {
                        has_f = true;
                    }
                } else if matches!(*t, "/" | "/*" | "~" | "~/*" | "$home" | "\"/\"" | "'/'") {
                    targets_root = true;
                }
            }
            if has_r && has_f && targets_root {
                return Some("destructive_rm_rf");
            }
        }
    }
    None
}

/// Fast sub-100us Layer 0 hard veto. Returns `None` when uncertain or when logging/shadow needs Python.
pub fn fast_layer0_veto(payload: &JsonValue, flags: &CliFlags) -> Option<FastVeto> {
    if flags.event != "pre-tool-use" || flags.shadow || flags.log_decisions {
        return None;
    }
    let (raw_tool, inp) = extract_tool_and_input(payload);
    let raw_low = raw_tool.to_ascii_lowercase();
    let tool = if matches!(raw_low.as_str(), "file_editor" | "str_replace_editor") {
        match inp.get("command").and_then(|v| v.as_str()).map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("view") => "read_file",
            Some("create") => "write_file",
            _ => "edit_file",
        }
    } else {
        canonical_tool(&raw_low)
    };

    let target_path = PATH_KEYS
        .iter()
        .filter_map(|k| inp.get(k).and_then(|v| v.as_str()))
        .find(|s| !s.is_empty());

    let cmd = CMD_KEYS
        .iter()
        .filter_map(|k| inp.get(k).and_then(|v| v.as_str()))
        .find(|s| !s.is_empty());

    if target_path.is_some_and(|p| p.contains('\0')) || cmd.is_some_and(|c| c.contains('\0')) {
        return Some(FastVeto {
            reason_code: "layer1_null_byte_veto".to_string(),
            label: "null_byte_in_input".to_string(),
            layer: 1,
        });
    }
    for k in DIFF_KEYS {
        if let Some(s) = inp.get(k).and_then(|v| v.as_str()) {
            if s.contains('\0') {
                return Some(FastVeto {
                    reason_code: "layer1_null_byte_veto".to_string(),
                    label: "null_byte_in_input".to_string(),
                    layer: 1,
                });
            }
        }
    }

    let is_write = matches!(
        tool,
        "write_file"
            | "edit_file"
            | "delete_file"
            | "execute_shell"
            | "git_commit"
            | "git_push"
            | "apply_patch"
            | "format_code"
    ) || raw_low.is_empty();

    if let Some(p) = target_path {
        if is_write && is_protected_path(p) {
            let allow_pol = env::var("AETHERIS_ALLOW_POLICY_EDIT").as_deref() == Ok("1")
                && is_agent_control_plane_path(p);
            if !allow_pol {
                return Some(FastVeto {
                    reason_code: "layer0_protected_path_veto".to_string(),
                    label: "blocked_policy".to_string(),
                    layer: 0,
                });
            }
        }
    }

    if let Some(c) = cmd {
        if let Some(label) = is_catastrophic_shell_cmd(c) {
            return Some(FastVeto {
                reason_code: "layer1_shell_parser_veto".to_string(),
                label: label.to_string(),
                layer: 1,
            });
        }
    }
    None
}

pub fn emit_veto(veto: &FastVeto, flags: &CliFlags) -> i32 {
    let reason = format!("{}: {}", veto.reason_code, veto.label);
    match flags.host.as_str() {
        "antigravity" => {
            println!(
                "{}",
                to_json_string(&obj(vec![
                    ("decision", JsonValue::Str("deny".to_string())),
                    ("reason", JsonValue::Str(reason)),
                ]))
            );
            0
        }
        "generic" => {
            eprintln!("{reason}");
            println!(
                "{}",
                to_json_string(&obj(vec![
                    ("decision", JsonValue::Str("deny".to_string())),
                    ("reason_code", JsonValue::Str(veto.reason_code.clone())),
                    ("label", JsonValue::Str(veto.label.clone())),
                    ("layer", JsonValue::Number(veto.layer as f64)),
                    ("host", JsonValue::Str("generic".to_string())),
                    ("exit_code", JsonValue::Number(2.0)),
                ]))
            );
            2
        }
        "hermes" => {
            eprintln!("{reason}");
            println!(
                "{}",
                to_json_string(&obj(vec![
                    ("decision", JsonValue::Str("block".to_string())),
                    ("reason", JsonValue::Str(reason)),
                ]))
            );
            2
        }
        _ => {
            eprintln!("{reason}");
            let hso = obj(vec![
                ("hookEventName", JsonValue::Str("PreToolUse".to_string())),
                ("permissionDecision", JsonValue::Str("deny".to_string())),
                ("permissionDecisionReason", JsonValue::Str(reason)),
            ]);
            println!("{}", to_json_string(&obj(vec![("hookSpecificOutput", hso)])));
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    #[test]
    fn layer0_blocks_dotenv_and_rm_rf_root() {
        let flags = CliFlags {
            host: "antigravity".to_string(),
            ..CliFlags::default()
        };
        let env_write = parse_json(
            r#"{"toolCall":{"name":"write_file","args":{"Path":".env","Content":"X=1"}}}"#,
        )
        .unwrap();
        let v = fast_layer0_veto(&env_write, &flags).expect("must veto .env");
        assert_eq!(v.reason_code, "layer0_protected_path_veto");

        let rm_root = parse_json(
            r#"{"toolCall":{"name":"run_command","args":{"CommandLine":"rm -rf /"}}}"#,
        )
        .unwrap();
        let v2 = fast_layer0_veto(&rm_root, &flags).expect("must veto rm -rf /");
        assert_eq!(v2.reason_code, "layer1_shell_parser_veto");

        let safe = parse_json(
            r#"{"toolCall":{"name":"write_file","args":{"Path":"src/app.py","Content":"x = 1\n"}}}"#,
        )
        .unwrap();
        assert!(fast_layer0_veto(&safe, &flags).is_none());
    }

    #[test]
    fn skips_shortcircuit_when_log_decisions_enabled() {
        let flags = CliFlags {
            log_decisions: true,
            ..CliFlags::default()
        };
        let env_write = parse_json(r#"{"tool":"write_file","path":".env"}"#).unwrap();
        assert!(fast_layer0_veto(&env_write, &flags).is_none());
    }
}
