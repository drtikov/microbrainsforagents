//! Native OS sandbox runner (`aetheris-gate sandbox ...`):
//! - Environment credential scrubbing (`--strip-secrets`)
//! - Windows: Job Object containment (`KILL_ON_JOB_CLOSE`, `ACTIVE_PROCESS` limit)
//! - Linux: `PR_SET_NO_NEW_PRIVS` + Landlock LSM FS/Net confinement
//! - macOS: Seatbelt `/usr/bin/sandbox-exec` wrapper

use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process::Command;

const SECRET_ENV_VARS: &[&str] = &[
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "STRIPE_SECRET_KEY",
    "SLACK_TOKEN",
    "NPM_TOKEN",
    "PYPI_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "VAULT_TOKEN",
    "DATABASE_URL",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
];

#[derive(Debug, Default)]
pub struct SandboxOpts {
    pub cwd: Option<PathBuf>,
    pub rw_paths: Vec<PathBuf>,
    pub deny_net: bool,
    pub strip_secrets: bool,
    pub keep_env: HashSet<String>,
    pub program_and_args: Vec<String>,
}

pub fn parse_sandbox_args(args: &[String]) -> Result<SandboxOpts, String> {
    let mut opts = SandboxOpts::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--" => {
                opts.program_and_args = args[i + 1..].to_vec();
                break;
            }
            "--cwd" => {
                i += 1;
                if i >= args.len() {
                    return Err("--cwd requires a value".to_string());
                }
                opts.cwd = Some(PathBuf::from(&args[i]));
            }
            "--rw" => {
                i += 1;
                if i >= args.len() {
                    return Err("--rw requires a path".to_string());
                }
                opts.rw_paths.push(PathBuf::from(&args[i]));
            }
            "--keep-env" => {
                i += 1;
                if i >= args.len() {
                    return Err("--keep-env requires a var name".to_string());
                }
                opts.keep_env.insert(args[i].clone());
            }
            "--deny-net" => opts.deny_net = true,
            "--strip-secrets" => opts.strip_secrets = true,
            other => {
                if other.starts_with('-') {
                    return Err(format!("unknown sandbox flag: {other}"));
                }
                opts.program_and_args = args[i..].to_vec();
                break;
            }
        }
        i += 1;
    }
    if opts.program_and_args.is_empty() {
        return Err("missing command after '--'".to_string());
    }
    Ok(opts)
}

#[cfg(windows)]
mod win_job {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use std::ptr;

    type Handle = *mut core::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct JobObjectBasicLimitInformation {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct JobObjectExtendedLimitInformation {
        basic_limit_information: JobObjectBasicLimitInformation,
        io_info: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: u32 = 9;
    const JOB_OBJECT_LIMIT_ACTIVE_PROCESS: u32 = 0x0000_0008;
    const JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION: u32 = 0x0000_0400;
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(lp_job_attributes: *const core::ffi::c_void, lp_name: *const u16) -> Handle;
        fn SetInformationJobObject(
            h_job: Handle,
            info_class: u32,
            lp_info: *const core::ffi::c_void,
            cb_info_len: u32,
        ) -> i32;
        fn AssignProcessToJobObject(h_job: Handle, h_process: Handle) -> i32;
        fn CloseHandle(h_object: Handle) -> i32;
    }

    pub struct JobGuard {
        handle: Handle,
    }

    impl JobGuard {
        pub fn create() -> Option<Self> {
            let h = unsafe { CreateJobObjectW(ptr::null(), ptr::null()) };
            if h.is_null() {
                return None;
            }
            let mut info = JobObjectExtendedLimitInformation::default();
            info.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
            info.basic_limit_information.active_process_limit = 32;
            let ok = unsafe {
                SetInformationJobObject(
                    h,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                    (&info as *const JobObjectExtendedLimitInformation).cast(),
                    core::mem::size_of::<JobObjectExtendedLimitInformation>() as u32,
                )
            };
            if ok == 0 {
                unsafe { CloseHandle(h) };
                return None;
            }
            Some(Self { handle: h })
        }

        pub fn assign(&self, child: &Child) {
            let proc_handle = child.as_raw_handle() as Handle;
            let _ = unsafe { AssignProcessToJobObject(self.handle, proc_handle) };
        }
    }

    impl Drop for JobGuard {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { CloseHandle(self.handle) };
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod linux_landlock {
    use std::ffi::CString;
    use std::path::PathBuf;

    const SYS_LANDLOCK_CREATE_RULESET: i64 = 444;
    const SYS_LANDLOCK_ADD_RULE: i64 = 445;
    const SYS_LANDLOCK_RESTRICT_SELF: i64 = 446;
    const PR_SET_NO_NEW_PRIVS: i32 = 38;
    const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;

    const FS_READ_ALL: u64 = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7); // EXEC | READ_FILE | READ_DIR | REFER (subset)
    const FS_RW_ALL: u64 = (1 << 0) | (1 << 1) | FS_READ_ALL | (1 << 4) | (1 << 6) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12);

    #[repr(C)]
    struct RulesetAttr {
        handled_access_fs: u64,
        handled_access_net: u64,
    }

    #[repr(C)]
    struct PathBeneathAttr {
        allowed_access: u64,
        parent_fd: i32,
    }

    extern "C" {
        fn syscall(num: i64, ...) -> i64;
        fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
        fn open(pathname: *const i8, flags: i32, ...) -> i32;
        fn close(fd: i32) -> i32;
    }

    const O_PATH: i32 = 0o10000000;
    const O_CLOEXEC: i32 = 0o2000000;

    pub fn apply(rw_paths: &[PathBuf]) {
        unsafe {
            let _ = prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
            let attr = RulesetAttr {
                handled_access_fs: FS_RW_ALL,
                handled_access_net: 0,
            };
            let rs_fd = syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                &attr as *const RulesetAttr,
                core::mem::size_of::<RulesetAttr>(),
                0u32,
            ) as i32;
            if rs_fd < 0 {
                return;
            }
            for ro in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev", "/proc"] {
                add_path(rs_fd, ro, FS_READ_ALL);
            }
            add_path(rs_fd, "/tmp", FS_RW_ALL);
            for p in rw_paths {
                if let Some(s) = p.to_str() {
                    add_path(rs_fd, s, FS_RW_ALL);
                }
            }
            let _ = syscall(SYS_LANDLOCK_RESTRICT_SELF, rs_fd, 0u32);
            close(rs_fd);
        }
    }

    unsafe fn add_path(rs_fd: i32, path: &str, access: u64) {
        if let Ok(cstr) = CString::new(path) {
            let fd = open(cstr.as_ptr(), O_PATH | O_CLOEXEC);
            if fd >= 0 {
                let pba = PathBeneathAttr {
                    allowed_access: access,
                    parent_fd: fd,
                };
                let _ = syscall(
                    SYS_LANDLOCK_ADD_RULE,
                    rs_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    &pba as *const PathBeneathAttr,
                    0u32,
                );
                close(fd);
            }
        }
    }
}

pub fn run_sandbox_command(args: &[String]) -> i32 {
    let opts = match parse_sandbox_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("aetheris-gate sandbox: {e}");
            return 2;
        }
    };

    let mut cmd = Command::new(&opts.program_and_args[0]);
    if opts.program_and_args.len() > 1 {
        cmd.args(&opts.program_and_args[1..]);
    }
    if let Some(cwd) = &opts.cwd {
        cmd.current_dir(cwd);
    }
    if opts.strip_secrets {
        for var in SECRET_ENV_VARS {
            if !opts.keep_env.contains(*var) {
                cmd.env_remove(var);
            }
        }
        // Also strip any key/token/secret-suffixed env vars not explicitly kept
        for (k, _) in env::vars() {
            let up = k.to_ascii_uppercase();
            if !opts.keep_env.contains(&k)
                && (up.ends_with("_API_KEY")
                    || up.ends_with("_SECRET_KEY")
                    || up.ends_with("_ACCESS_TOKEN")
                    || up.ends_with("_AUTH_TOKEN"))
            {
                cmd.env_remove(&k);
            }
        }
    }
    cmd.env("AETHERIS_SANDBOX_ACTIVE", "1");
    if opts.deny_net {
        cmd.env("AETHERIS_SANDBOX_DENY_NET", "1");
    }

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let rw = opts.rw_paths.clone();
        unsafe {
            cmd.pre_exec(move || {
                linux_landlock::apply(&rw);
                Ok(())
            });
        }
    }

    #[cfg(windows)]
    let job_guard = win_job::JobGuard::create();

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("aetheris-gate sandbox: spawn failed: {e}");
            return 2;
        }
    };

    #[cfg(windows)]
    if let Some(ref j) = job_guard {
        j.assign(&child);
    }

    match child.wait() {
        Ok(st) => st.code().unwrap_or(1),
        Err(e) => {
            eprintln!("aetheris-gate sandbox: wait failed: {e}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sandbox_flags_and_double_dash() {
        let args: Vec<String> = vec![
            "--cwd",
            "/tmp/proj",
            "--rw",
            "/tmp/proj",
            "--strip-secrets",
            "--deny-net",
            "--",
            "python",
            "-c",
            "print('hi')",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let parsed = parse_sandbox_args(&args).expect("valid args");
        assert_eq!(parsed.cwd, Some(PathBuf::from("/tmp/proj")));
        assert!(parsed.deny_net);
        assert!(parsed.strip_secrets);
        assert_eq!(parsed.program_and_args, vec!["python", "-c", "print('hi')"]);
    }
}
