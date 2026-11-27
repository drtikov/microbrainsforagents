"""Declarative HostSpec catalog and hook config compiler (100% pure Python stdlib)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

IntegrationKind = Literal["hook", "esm_plugin", "extension", "sdk", "mcp"]

_CLAUDE_TOOL_MATCHER = (
    "Bash|Edit|Write|MultiEdit|NotebookEdit|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
)
_QWEN_TOOL_MATCHER = (
    "run_shell_command|read_file|write_file|edit|grep_search|glob|find_files|list_directory|"
    "web_fetch|web_search|" + _CLAUDE_TOOL_MATCHER
)
_CURSOR_TOOL_MATCHER = (
    "Write|Edit|StrReplace|Delete|Read|Grep|Glob|LS|SemanticSearch|WebFetch|WebSearch|"
    "Task|CallDynamicTool|GetDynamicTools"
)
_DEVIN_TOOL_MATCHER = (
    "^(exec|get_output|write_to_process|kill_shell|edit|write|apply_patch|"
    "notebook_edit|notebook_read|read|grep|glob|webfetch|run_subagent|"
    "todo_write|skill|exit_plan_mode|mcp__.*)$"
)
_MCODE_TOOL_MATCHER = "bash|read|write|edit|grep|glob|web_fetch|web_search|task"
_ZCODE_TOOL_MATCHER = "Bash|Edit|Write|Read|Grep|Glob|WebFetch|WebSearch|mcp__.*"
_OPENHANDS_TOOL_MATCHER = (
    "terminal|file_editor|execute_bash|str_replace_editor|bash|edit|write|read|mcp__.*"
)
_ANTIGRAVITY_TOOL_MATCHER = (
    "run_command|edit_file|write_file|replace_file_content|"
    "multi_replace_file_content|write_to_file|read_file|view_file|"
    "grep_search|find_by_name|list_dir|mcp_.*"
)
_HERMES_TOOL_MATCHER = (
    "terminal|execute_code|patch|write_file|read_file|search_files|"
    "browser_exec|web_search|web_extract|delegate_task|image_generate|"
    "video_generate|skill_manage|mcp_.*"
)


@dataclass(frozen=True)
class HostSpec:
    """Declarative specification for a supported coding agent host."""

    name: str
    contract: str
    family: str
    config_rel_workspace: Optional[str]
    config_rel_user: Optional[str]
    tool_matcher: str
    timeout_sec: int
    supports_http_daemon: bool
    supports_replace_tool_output: bool
    plugin_exit_code_passthrough: bool
    returns_hso_directly_http: bool
    integration_kind: IntegrationKind
    uses_on_failure_block: bool


_SPECS_LIST: tuple[HostSpec, ...] = (
    HostSpec(
        name="antigravity",
        contract="antigravity_pretooluse_v1",
        family="antigravity",
        config_rel_workspace=".agents/plugins/aetheris-microbrains/hooks.json",
        config_rel_user=".gemini/config/plugins/aetheris-microbrains/hooks.json",
        tool_matcher=_ANTIGRAVITY_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=False,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=True,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="claude",
        contract="claude_pretooluse_v1",
        family="claude_compat",
        config_rel_workspace=".claude/settings.local.json",
        config_rel_user=".claude/settings.local.json",
        tool_matcher=_CLAUDE_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=True,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=True,
    ),
    HostSpec(
        name="codex",
        contract="codex_pretooluse_no_ask_v1",
        family="codex",
        config_rel_workspace=".codex/hooks.json",
        config_rel_user=".codex/hooks.json",
        tool_matcher=_CLAUDE_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=False,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=True,
    ),
    HostSpec(
        name="cursor",
        contract="cursor_hook_v2",
        family="cursor",
        config_rel_workspace=".cursor/hooks.json",
        config_rel_user=".cursor/hooks.json",
        tool_matcher=_CURSOR_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=False,
        supports_replace_tool_output=True,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="devin",
        contract="devin_hook_v1",
        family="devin",
        config_rel_workspace=".devin/hooks.v1.json",
        config_rel_user=".config/devin/config.json",
        tool_matcher=_DEVIN_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=False,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="cline",
        contract="cline_hook_v1",
        family="cline",
        config_rel_workspace=".clinerules/hooks",
        config_rel_user=".cline/hooks",
        tool_matcher="",
        timeout_sec=10,
        supports_http_daemon=False,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="mcode",
        contract="mcode_pretooluse_v1",
        family="claude_compat",
        config_rel_workspace=".minimax/hooks.json",
        config_rel_user=".minimax/hooks.json",
        tool_matcher=_MCODE_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="ouroboros",
        contract="ouroboros_registry_v1",
        family="claude_compat",
        config_rel_workspace=None,
        config_rel_user=None,
        tool_matcher="",
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=True,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="sdk",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="prime",
        contract="prime_pretooluse_v1",
        family="claude_compat",
        config_rel_workspace=None,
        config_rel_user=None,
        tool_matcher="",
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=True,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="sdk",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="openclaw",
        contract="openclaw_before_tool_call_v1",
        family="openclaw",
        config_rel_workspace=None,
        config_rel_user=".openclaw/openclaw.json",
        tool_matcher="",
        timeout_sec=3,
        supports_http_daemon=True,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=True,
        integration_kind="esm_plugin",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="opencode",
        contract="opencode_plugin_v1",
        family="plugin_block",
        config_rel_workspace=".opencode/plugins/aetheris-microbrains.js",
        config_rel_user=".config/opencode/plugins/aetheris-microbrains.js",
        tool_matcher="",
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
        integration_kind="esm_plugin",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="kilocode",
        contract="kilocode_plugin_v1",
        family="plugin_block",
        config_rel_workspace=".kilo/plugins/aetheris-microbrains.js",
        config_rel_user=".config/kilo/plugins/aetheris-microbrains.js",
        tool_matcher="",
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
        integration_kind="esm_plugin",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="pi",
        contract="pi_extension_v1",
        family="plugin_block",
        config_rel_workspace=".pi/extensions/aetheris-microbrains/index.js",
        config_rel_user=".pi/agent/extensions/aetheris-microbrains/index.js",
        tool_matcher="",
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
        integration_kind="extension",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="omp",
        contract="omp_extension_v1",
        family="plugin_block",
        config_rel_workspace=".omp/extensions/aetheris-microbrains/index.js",
        config_rel_user=".omp/agent/extensions/aetheris-microbrains/index.js",
        tool_matcher="",
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=True,
        returns_hso_directly_http=True,
        integration_kind="extension",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="qoder",
        contract="qoder_pretooluse_v1",
        family="claude_compat",
        config_rel_workspace=".qoder/settings.local.json",
        config_rel_user=".qoder/settings.json",
        tool_matcher=_CLAUDE_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=True,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="qwen",
        contract="qwen_pretooluse_v1",
        family="claude_compat",
        config_rel_workspace=".qwen/settings.local.json",
        config_rel_user=".qwen/settings.json",
        tool_matcher=_QWEN_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="zcode",
        contract="zcode_pretooluse_v1",
        family="claude_compat",
        config_rel_workspace=".zcode/config.json",
        config_rel_user=".zcode/cli/config.json",
        tool_matcher=_ZCODE_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=True,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="openhands",
        contract="openhands_hook_v1",
        family="claude_compat",
        config_rel_workspace=".openhands/hooks.json",
        config_rel_user=".openhands/hooks.json",
        tool_matcher=_OPENHANDS_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=True,
        supports_replace_tool_output=True,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
    HostSpec(
        name="hermes",
        contract="hermes_shell_hook_v1",
        family="hermes",
        config_rel_workspace=None,
        config_rel_user=".hermes/config.yaml",
        tool_matcher=_HERMES_TOOL_MATCHER,
        timeout_sec=60,
        supports_http_daemon=False,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=True,
    ),
    HostSpec(
        name="generic",
        contract="generic_v1",
        family="generic",
        config_rel_workspace=None,
        config_rel_user=None,
        tool_matcher=_CLAUDE_TOOL_MATCHER,
        timeout_sec=10,
        supports_http_daemon=False,
        supports_replace_tool_output=False,
        plugin_exit_code_passthrough=False,
        returns_hso_directly_http=False,
        integration_kind="hook",
        uses_on_failure_block=False,
    ),
)

HOST_SPECS: Dict[str, HostSpec] = {spec.name: spec for spec in _SPECS_LIST}


def get_host_spec(host: str) -> Optional[HostSpec]:
    """Return the ``HostSpec`` for ``host`` (case-insensitive), or ``None`` if unknown."""
    key = str(host or "").strip().lower()
    return HOST_SPECS.get(key)


def _compile_cursor_config(spec: HostSpec, *, pre_cmd: str, post_cmd: str) -> Dict[str, Any]:
    def _entry(cmd: str, *, fail_closed: bool, matcher: Optional[str] = None) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "command": cmd,
            "timeout": spec.timeout_sec,
            "failClosed": fail_closed,
        }
        if matcher:
            item["matcher"] = matcher
        return item

    hooks: Dict[str, Any] = {
        "preToolUse": [_entry(pre_cmd, fail_closed=True, matcher=spec.tool_matcher)],
        "beforeShellExecution": [_entry(pre_cmd, fail_closed=True)],
        "beforeMCPExecution": [_entry(pre_cmd, fail_closed=True)],
        "beforeReadFile": [_entry(pre_cmd, fail_closed=True)],
    }
    if post_cmd:
        hooks["postToolUse"] = [_entry(post_cmd, fail_closed=False, matcher=spec.tool_matcher)]
        hooks["afterShellExecution"] = [_entry(post_cmd, fail_closed=False)]
        hooks["afterMCPExecution"] = [_entry(post_cmd, fail_closed=False)]
        hooks["afterFileEdit"] = [_entry(post_cmd, fail_closed=False)]
    return {"version": 1, "hooks": hooks}


def _compile_mcode_config(
    spec: HostSpec, *, pre_cmd: str, post_cmd: str, is_workspace: bool
) -> Dict[str, Any]:
    hooks: Dict[str, Any] = {
        "PreToolUse": [
            {
                "matcher": spec.tool_matcher,
                "hooks": [
                    {
                        "type": "command",
                        "name": "Aetheris Microbrains gate",
                        "command": pre_cmd,
                        "commandWindows": pre_cmd,
                        "timeout": spec.timeout_sec,
                    }
                ],
            }
        ]
    }
    if post_cmd:
        hooks["PostToolUse"] = [
            {
                "matcher": spec.tool_matcher,
                "hooks": [
                    {
                        "type": "command",
                        "name": "Aetheris Microbrains post-tool audit",
                        "command": post_cmd,
                        "commandWindows": post_cmd,
                        "timeout": spec.timeout_sec,
                    }
                ],
            }
        ]
    bundle: Dict[str, Any] = {"hooks": hooks}
    if not is_workspace:
        bundle["enabled"] = True
    return bundle


def _compile_zcode_config(spec: HostSpec, *, pre_cmd: str, post_cmd: str) -> Dict[str, Any]:
    events: Dict[str, Any] = {
        "PreToolUse": [
            {
                "matcher": spec.tool_matcher,
                "hooks": [
                    {
                        "type": "process",
                        "command": pre_cmd,
                        "timeoutMs": spec.timeout_sec * 1000,
                        "statusMessage": "Aetheris Microbrains gate",
                    }
                ],
            }
        ]
    }
    if post_cmd:
        events["PostToolUse"] = [
            {
                "matcher": "Bash|Edit|Write|Read|Grep|WebFetch|WebSearch|mcp__.*",
                "hooks": [
                    {
                        "type": "process",
                        "command": post_cmd,
                        "timeoutMs": spec.timeout_sec * 1000,
                        "statusMessage": "Aetheris Microbrains gate",
                    }
                ],
            }
        ]
    return {"hooks": {"enabled": True, "events": events}}


def _compile_hermes_config(spec: HostSpec, *, pre_cmd: str, post_cmd: str) -> Dict[str, Any]:
    pre_entry: Dict[str, Any] = {
        "command": pre_cmd,
        "matcher": spec.tool_matcher,
        "timeout": spec.timeout_sec,
    }
    if spec.uses_on_failure_block:
        pre_entry["fail_closed"] = True
    hooks: Dict[str, Any] = {"pre_tool_call": [pre_entry]}
    if post_cmd:
        hooks["post_tool_call"] = [
            {
                "command": post_cmd,
                "matcher": spec.tool_matcher,
                "timeout": spec.timeout_sec,
            }
        ]
    return {"hooks": hooks}


def _compile_antigravity_config(spec: HostSpec, *, pre_cmd: str, post_cmd: str) -> Dict[str, Any]:
    inner: Dict[str, Any] = {
        "enabled": True,
        "PreToolUse": [
            {
                "matcher": spec.tool_matcher,
                "hooks": [{"type": "command", "command": pre_cmd, "timeout": spec.timeout_sec}],
            }
        ],
    }
    if post_cmd:
        inner["PostToolUse"] = [
            {
                "matcher": spec.tool_matcher,
                "hooks": [{"type": "command", "command": post_cmd, "timeout": spec.timeout_sec}],
            }
        ]
    return {"aetheris-microbrains": inner}


def compile_hook_config_from_spec(
    host: str,
    *,
    pre_cmd: str,
    post_cmd: str = "",
    is_workspace: bool = True,
) -> Dict[str, Any]:
    """Compile a declarative host hook configuration bundle from ``HOST_SPECS``."""
    spec = get_host_spec(host)
    if spec is None:
        raise ValueError(f"Unknown host: {host!r}")

    name = spec.name
    if name == "cursor":
        return _compile_cursor_config(spec, pre_cmd=pre_cmd, post_cmd=post_cmd)
    if name == "mcode":
        return _compile_mcode_config(
            spec, pre_cmd=pre_cmd, post_cmd=post_cmd, is_workspace=is_workspace
        )
    if name == "zcode":
        return _compile_zcode_config(spec, pre_cmd=pre_cmd, post_cmd=post_cmd)
    if name == "hermes":
        return _compile_hermes_config(spec, pre_cmd=pre_cmd, post_cmd=post_cmd)
    if name == "antigravity":
        return _compile_antigravity_config(spec, pre_cmd=pre_cmd, post_cmd=post_cmd)

    if spec.integration_kind != "hook":
        cfg_rel = spec.config_rel_workspace if is_workspace else spec.config_rel_user
        descriptor: Dict[str, Any] = {
            "host": name,
            "integration_kind": spec.integration_kind,
            "config_rel": cfg_rel,
            "pre_cmd": pre_cmd,
        }
        if post_cmd:
            descriptor["post_cmd"] = post_cmd
        return descriptor

    if name == "cline":
        scripts: Dict[str, Any] = {"PreToolUse": pre_cmd}
        if post_cmd:
            scripts["PostToolUse"] = post_cmd
        return scripts

    pre_hook: Dict[str, Any] = {"type": "command", "command": pre_cmd}
    if spec.uses_on_failure_block:
        pre_hook["onFailure"] = "block"
    else:
        pre_hook["timeout"] = spec.timeout_sec
    if name == "openhands":
        pre_hook["async"] = False

    hooks: Dict[str, Any] = {
        "PreToolUse": [{"matcher": spec.tool_matcher, "hooks": [pre_hook]}]
    }
    if post_cmd:
        post_hook: Dict[str, Any] = {"type": "command", "command": post_cmd}
        if not spec.uses_on_failure_block:
            post_hook["timeout"] = spec.timeout_sec
        if name == "openhands":
            post_hook["async"] = False
        hooks["PostToolUse"] = [{"matcher": spec.tool_matcher, "hooks": [post_hook]}]

    return {"hooks": hooks}


def validate_host_specs_parity() -> Dict[str, Any]:
    """Compare ``HOST_SPECS`` against ``HOST_REGISTRY`` for full structural parity."""
    from .registry import HOST_REGISTRY

    mismatches: List[Dict[str, Any]] = []
    spec_names = set(HOST_SPECS.keys())
    reg_names = set(HOST_REGISTRY.keys())

    for missing_in_spec in sorted(reg_names - spec_names):
        mismatches.append({"host": missing_in_spec, "issue": "missing_in_specs"})
    for extra_in_spec in sorted(spec_names - reg_names):
        mismatches.append({"host": extra_in_spec, "issue": "extra_in_specs"})

    for name in sorted(spec_names & reg_names):
        spec = HOST_SPECS[name]
        adapter = HOST_REGISTRY[name]
        for field in (
            "contract",
            "family",
            "supports_http_daemon",
            "supports_replace_tool_output",
        ):
            spec_val = getattr(spec, field)
            reg_val = getattr(adapter, field)
            if spec_val != reg_val:
                mismatches.append(
                    {
                        "host": name,
                        "field": field,
                        "spec_value": spec_val,
                        "registry_value": reg_val,
                    }
                )

    return {
        "ok": len(mismatches) == 0,
        "spec_count": len(HOST_SPECS),
        "registry_count": len(HOST_REGISTRY),
        "mismatches": mismatches,
    }


__all__ = [
    "HOST_SPECS",
    "HostSpec",
    "IntegrationKind",
    "compile_hook_config_from_spec",
    "get_host_spec",
    "validate_host_specs_parity",
]
