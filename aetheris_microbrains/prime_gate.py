"""Drop-in in-process Microbrains security gate for Prime Agent REPL callables (stdlib-only)."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .hosts.prime import format_block_error


class PrimeMicrobrainsError(RuntimeError):
    """Base error raised when a Prime Agent tool call is blocked or escalated."""

    def __init__(self, tool_name: str, verdict: Mapping[str, Any]) -> None:
        self.tool_name = tool_name
        self.verdict = dict(verdict)
        self.decision = str(verdict.get("decision") or "deny")
        self.reason_code = str(verdict.get("reason_code") or "POLICY_BLOCK")
        self.layer = verdict.get("layer", 1)
        self.latency_ms = float(verdict.get("latency_ms") or 0.0)
        super().__init__(format_block_error(tool_name, verdict))


class PrimeMicrobrainsBlockedError(PrimeMicrobrainsError):
    """Raised when Microbrains returns decision == 'deny'."""


class PrimeMicrobrainsEscalatedError(PrimeMicrobrainsError):
    """Raised when Microbrains returns decision == 'escalate' under blocking escalation."""


@dataclasses.dataclass
class PrimeBlockedResult:
    """Fail-closed placeholder returned when PostToolUse secret redaction cannot safely apply."""

    text: str
    error: str
    output: str
    content: str
    ok: bool = False
    is_error: bool = True
    blocked: bool = True
    verdict: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.text


class PrimeMicrobrainsGate:
    """Pre-flight policy check, fail-closed PostToolUse DLP, and sync/async REPL wrappers for Prime Agent."""

    def __init__(
        self,
        *,
        profile: str = "dev",
        policy_path: Optional[str] = None,
        cwd: Optional[str] = None,
        session_id: Optional[str] = None,
        block_on_escalate: bool = True,
        scrub_outputs: bool = True,
        replace_tool_output: bool = True,
        log_decisions: bool = False,
        shadow_mode: bool = False,
        strict_ast: bool = False,
        session_state_path: Optional[str] = None,
        journal_path: Optional[str] = None,
        tool_aliases: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.profile = profile
        self.policy_path = policy_path
        self.cwd = cwd
        self.session_id = session_id or "prime-session"
        self.block_on_escalate = block_on_escalate
        self.scrub_outputs = scrub_outputs
        self.replace_tool_output = replace_tool_output
        self.log_decisions = log_decisions
        self.shadow_mode = shadow_mode
        self.strict_ast = strict_ast
        self.session_state_path = session_state_path
        self.journal_path = journal_path
        self.tool_aliases: Dict[str, str] = (
            {
                str(k).strip().lower(): str(v).strip()
                for k, v in tool_aliases.items()
                if str(k).strip() and str(v).strip()
            }
            if tool_aliases
            else {}
        )
        self.last_verdict: Optional[Dict[str, Any]] = None
        self.last_post_verdict: Optional[Dict[str, Any]] = None

    def _resolve_tool_name(self, name: str) -> str:
        raw = str(name or "").strip()
        return self.tool_aliases.get(raw.lower(), raw)

    def check_pre(self, name: str, args: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a Prime Agent tool call through ``evaluate_hook_payload(..., host='prime')``."""
        from .hook import evaluate_hook_payload

        payload: Dict[str, Any] = {
            "hook_event_name": "PreToolUse",
            "session_id": self.session_id,
            "tool_name": self._resolve_tool_name(name),
            "tool_input": dict(args or {}),
        }
        if self.cwd:
            payload["cwd"] = self.cwd

        verdict = evaluate_hook_payload(
            payload,
            strict_ast=self.strict_ast,
            log_decisions=self.log_decisions,
            shadow_mode=self.shadow_mode,
            host="prime",
            journal_path=self.journal_path,
            session_state_path=self.session_state_path,
            profile=self.profile,
            policy_path=self.policy_path,
        )
        self.last_verdict = verdict
        return verdict

    def should_block(self, verdict: Mapping[str, Any]) -> bool:
        if self.shadow_mode:
            return False
        dec = str(verdict.get("decision") or "").lower()
        if dec == "deny":
            return True
        if self.block_on_escalate and dec == "escalate":
            return True
        return False

    def _raise_if_blocked(self, name: str, verdict: Mapping[str, Any]) -> None:
        if self.shadow_mode:
            return
        dec = str(verdict.get("decision") or "").lower()
        if dec == "deny":
            raise PrimeMicrobrainsBlockedError(name, verdict)
        if self.block_on_escalate and dec == "escalate":
            raise PrimeMicrobrainsEscalatedError(name, verdict)

    def _has_secrets(self, name: str, args: Optional[Mapping[str, Any]], text: str) -> bool:
        from .hook import evaluate_post_tool_use_payload

        check = evaluate_post_tool_use_payload(
            {
                "hook_event_name": "PostToolUse",
                "session_id": self.session_id,
                "tool_name": self._resolve_tool_name(name),
                "tool_input": dict(args or {}),
                "tool_response": text,
            },
            log_decisions=False,
            shadow_mode=False,
            host="prime",
            policy_path=self.policy_path,
            replace_tool_output=False,
        )
        return bool(check.get("secrets_found"))

    def _safe_dlp_fallback(
        self,
        name: str,
        args: Optional[Mapping[str, Any]],
        redacted: Any,
        post_verdict: Dict[str, Any],
    ) -> PrimeBlockedResult:
        suppressed = (
            f"MICROBRAINS_DLP_SUPPRESSED ({name}): "
            "Tool output suppressed because secret redaction could not be applied safely."
        )
        msg = suppressed
        if isinstance(redacted, str) and redacted and not self._has_secrets(name, args, redacted):
            msg = redacted
        return PrimeBlockedResult(
            text=msg,
            error=msg,
            output=msg,
            content=msg,
            verdict=post_verdict,
        )

    def scrub_post(
        self,
        name: str,
        args: Optional[Mapping[str, Any]],
        output: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run PostToolUse secret scrubbing with fail-closed object protection."""
        if not self.scrub_outputs:
            info = {"scrubbed": False, "secrets_found": False, "prompt_injection_detected": False}
            self.last_post_verdict = info
            return output, info

        from .hook import evaluate_post_tool_use_payload

        is_object = not isinstance(output, (str, dict, list, int, float, bool)) and output is not None
        attr_targets: List[str] = []
        raw_text: Any = output
        if is_object:
            attr_targets = [
                attr for attr in ("text", "output", "content") if isinstance(getattr(output, attr, None), str)
            ]
            if len(attr_targets) == 1:
                raw_text = getattr(output, attr_targets[0])
            elif attr_targets:
                raw_text = "\n".join(getattr(output, attr) for attr in attr_targets)
            else:
                raw_text = str(output)

        resolved_name = self._resolve_tool_name(name)
        args_dict = dict(args or {})
        post_verdict = evaluate_post_tool_use_payload(
            {
                "hook_event_name": "PostToolUse",
                "session_id": self.session_id,
                "tool_name": resolved_name,
                "tool_input": args_dict,
                "tool_response": raw_text,
                **({"cwd": self.cwd} if self.cwd else {}),
            },
            log_decisions=self.log_decisions,
            shadow_mode=self.shadow_mode,
            host="prime",
            journal_path=self.journal_path,
            session_state_path=self.session_state_path,
            policy_path=self.policy_path,
            replace_tool_output=self.replace_tool_output,
        )
        self.last_post_verdict = post_verdict

        if self.replace_tool_output and not self.shadow_mode and post_verdict.get("secrets_found"):
            redacted = post_verdict.get("redacted_output")
            if is_object:
                if not attr_targets:
                    return self._safe_dlp_fallback(name, args, redacted, post_verdict), post_verdict
                try:
                    updates: Dict[str, str] = {}
                    if len(attr_targets) == 1:
                        if not isinstance(redacted, str):
                            return self._safe_dlp_fallback(name, args, redacted, post_verdict), post_verdict
                        updates[attr_targets[0]] = redacted
                    else:
                        for attr in attr_targets:
                            sub_res = evaluate_post_tool_use_payload(
                                {
                                    "hook_event_name": "PostToolUse",
                                    "session_id": self.session_id,
                                    "tool_name": resolved_name,
                                    "tool_input": args_dict,
                                    "tool_response": getattr(output, attr),
                                },
                                log_decisions=False,
                                shadow_mode=False,
                                host="prime",
                                policy_path=self.policy_path,
                                replace_tool_output=True,
                            )
                            sub_redacted = sub_res.get("redacted_output")
                            if sub_res.get("secrets_found") and not isinstance(sub_redacted, str):
                                return self._safe_dlp_fallback(name, args, redacted, post_verdict), post_verdict
                            updates[attr] = sub_redacted if isinstance(sub_redacted, str) else getattr(output, attr)

                    if dataclasses.is_dataclass(output):
                        candidate = dataclasses.replace(output, **updates)
                    else:
                        for attr, val in updates.items():
                            setattr(output, attr, val)
                        candidate = output

                    if self._has_secrets(name, args, str(candidate)):
                        return self._safe_dlp_fallback(name, args, redacted, post_verdict), post_verdict
                    return candidate, post_verdict
                except Exception:
                    return self._safe_dlp_fallback(name, args, redacted, post_verdict), post_verdict

            if redacted is None:
                suppressed = (
                    f"MICROBRAINS_DLP_SUPPRESSED ({name}): "
                    "Tool output suppressed because secret redaction could not be applied safely."
                )
                return (
                    suppressed
                    if isinstance(output, str)
                    else self._safe_dlp_fallback(name, args, redacted, post_verdict)
                ), post_verdict
            return redacted, post_verdict
        return output, post_verdict

    def wrap_bash(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a sync or async Prime Agent ``bash`` callable."""
        orig = getattr(fn, "_aetheris_orig", fn)

        def _extract_cmd(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
            cmd = kwargs.get("command") or kwargs.get("cmd")
            if cmd is None and args:
                cmd = args[0]
            return {"command": cmd}

        if inspect.iscoroutinefunction(orig):
            @functools.wraps(orig)
            async def _async_bash(*args: Any, **kwargs: Any) -> Any:
                ti = _extract_cmd(args, kwargs)
                self._raise_if_blocked("bash", self.check_pre("bash", ti))
                res = await orig(*args, **kwargs)
                clean, _ = self.scrub_post("bash", ti, res)
                return clean

            _async_bash._aetheris_orig = orig  # type: ignore[attr-defined]
            return _async_bash

        @functools.wraps(orig)
        def _sync_bash(*args: Any, **kwargs: Any) -> Any:
            ti = _extract_cmd(args, kwargs)
            self._raise_if_blocked("bash", self.check_pre("bash", ti))
            res = orig(*args, **kwargs)
            clean, _ = self.scrub_post("bash", ti, res)
            return clean

        _sync_bash._aetheris_orig = orig  # type: ignore[attr-defined]
        return _sync_bash

    def wrap_edit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a sync or async Prime Agent ``edit.run`` callable."""
        orig = getattr(fn, "_aetheris_orig", fn)

        def _extract_edit(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
            path = kwargs.get("path") or kwargs.get("file_path") or (args[0] if len(args) > 0 else "")
            old_s = kwargs.get("old_str") or kwargs.get("old_string") or (args[1] if len(args) > 1 else "")
            new_s = kwargs.get("new_str") or kwargs.get("new_string") or (args[2] if len(args) > 2 else "")
            return {"path": path, "old_str": old_s, "new_str": new_s}

        if inspect.iscoroutinefunction(orig):
            @functools.wraps(orig)
            async def _async_edit(*args: Any, **kwargs: Any) -> Any:
                ti = _extract_edit(args, kwargs)
                self._raise_if_blocked("edit", self.check_pre("edit", ti))
                res = await orig(*args, **kwargs)
                clean, _ = self.scrub_post("edit", ti, res)
                return clean

            _async_edit._aetheris_orig = orig  # type: ignore[attr-defined]
            return _async_edit

        @functools.wraps(orig)
        def _sync_edit(*args: Any, **kwargs: Any) -> Any:
            ti = _extract_edit(args, kwargs)
            self._raise_if_blocked("edit", self.check_pre("edit", ti))
            res = orig(*args, **kwargs)
            clean, _ = self.scrub_post("edit", ti, res)
            return clean

        _sync_edit._aetheris_orig = orig  # type: ignore[attr-defined]
        return _sync_edit

    def wrap_mcp_call(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a sync or async Prime Agent ``mcp.call_tool`` callable."""
        orig = getattr(fn, "_aetheris_orig", fn)

        def _extract_mcp(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            conn = kwargs.get("connection_id") or (args[0] if len(args) > 0 else "unknown")
            tool = kwargs.get("tool") or kwargs.get("tool_name") or (args[1] if len(args) > 1 else "unknown")
            tool_args = kwargs.get("arguments")
            if tool_args is None and len(args) > 2:
                tool_args = args[2]
            raw_name = f"mcp__{conn}__{tool}"
            ti = dict(tool_args) if isinstance(tool_args, Mapping) else ({"arguments": tool_args} if tool_args is not None else {})
            return raw_name, ti

        if inspect.iscoroutinefunction(orig):
            @functools.wraps(orig)
            async def _async_mcp(*args: Any, **kwargs: Any) -> Any:
                raw_name, ti = _extract_mcp(args, kwargs)
                self._raise_if_blocked(raw_name, self.check_pre(raw_name, ti))
                res = await orig(*args, **kwargs)
                clean, _ = self.scrub_post(raw_name, ti, res)
                return clean

            _async_mcp._aetheris_orig = orig  # type: ignore[attr-defined]
            return _async_mcp

        @functools.wraps(orig)
        def _sync_mcp(*args: Any, **kwargs: Any) -> Any:
            raw_name, ti = _extract_mcp(args, kwargs)
            self._raise_if_blocked(raw_name, self.check_pre(raw_name, ti))
            res = orig(*args, **kwargs)
            clean, _ = self.scrub_post(raw_name, ti, res)
            return clean

        _sync_mcp._aetheris_orig = orig  # type: ignore[attr-defined]
        return _sync_mcp


_PATCHED_TARGETS: List[Tuple[Any, str]] = []


def install_prime_gate(
    *,
    rlm_mod: Any = None,
    edit_mod: Any = None,
    mcp_mod: Any = None,
    gate: Optional[PrimeMicrobrainsGate] = None,
    **gate_kwargs: Any,
) -> PrimeMicrobrainsGate:
    """Idempotently wrap Prime Agent REPL entrypoints (``rlm.bash``, ``edit.run``, ``mcp.call_tool``)."""
    active_gate = gate if gate is not None else PrimeMicrobrainsGate(**gate_kwargs)

    if rlm_mod is None and edit_mod is None and mcp_mod is None:
        for mod_name, target in (("rlm", "rlm"), ("edit", "edit"), ("mcp", "mcp")):
            try:
                loaded = __import__(mod_name)
                if target == "rlm" and hasattr(loaded, "bash"):
                    rlm_mod = loaded
                elif target == "edit" and hasattr(loaded, "run"):
                    edit_mod = loaded
                elif target == "mcp" and hasattr(loaded, "call_tool"):
                    mcp_mod = loaded
            except ImportError:
                pass

    bindings = (
        (rlm_mod, "bash", active_gate.wrap_bash),
        (edit_mod, "run", active_gate.wrap_edit),
        (mcp_mod, "call_tool", active_gate.wrap_mcp_call),
    )
    for mod, attr, wrapper in bindings:
        if mod is None or not hasattr(mod, attr):
            continue
        cur = getattr(mod, attr)
        orig = getattr(cur, "_aetheris_orig", cur)
        setattr(mod, attr, wrapper(orig))
        mod._aetheris_prime_gate = active_gate
        if (mod, attr) not in _PATCHED_TARGETS:
            _PATCHED_TARGETS.append((mod, attr))

    return active_gate


def uninstall_prime_gate(
    *,
    rlm_mod: Any = None,
    edit_mod: Any = None,
    mcp_mod: Any = None,
) -> None:
    """Restore original Prime Agent callables wrapped by :func:`install_prime_gate`."""
    explicit = [
        (mod, attr)
        for mod, attr in ((rlm_mod, "bash"), (edit_mod, "run"), (mcp_mod, "call_tool"))
        if mod is not None and hasattr(mod, attr)
    ]
    targets = explicit if explicit else list(_PATCHED_TARGETS)
    for mod, attr in targets:
        if hasattr(mod, attr):
            cur = getattr(mod, attr)
            orig = getattr(cur, "_aetheris_orig", None)
            if orig is not None:
                setattr(mod, attr, orig)
        if not explicit and (mod, attr) in _PATCHED_TARGETS:
            _PATCHED_TARGETS.remove((mod, attr))
