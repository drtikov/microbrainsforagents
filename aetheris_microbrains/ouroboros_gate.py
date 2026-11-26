"""Drop-in in-process Microbrains security & routing gate for Ouroboros ToolRegistry."""

from __future__ import annotations

import dataclasses
import functools
import threading
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .hosts.ouroboros import (
    extract_schema_description,
    extract_schema_name,
    format_block_error,
)


@dataclasses.dataclass
class OuroborosBlockedResult:
    """Fallback error result returned by wrapped ``execute_result`` when blocked."""

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


class OuroborosMicrobrainsGate:
    """Pre-flight policy enforcement, PostToolUse DLP scrubbing, and MoE schema narrowing for Ouroboros."""

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
        self.session_id = session_id or "ouroboros-session"
        self.block_on_escalate = block_on_escalate
        self.scrub_outputs = scrub_outputs
        self.replace_tool_output = replace_tool_output
        self.log_decisions = log_decisions
        self.shadow_mode = shadow_mode
        self.strict_ast = strict_ast
        self.session_state_path = session_state_path
        self.journal_path = journal_path
        self.tool_aliases: Dict[str, str] = (
            {str(k).strip().lower(): str(v).strip() for k, v in tool_aliases.items() if str(k).strip() and str(v).strip()}
            if tool_aliases
            else {}
        )
        self.last_verdict: Optional[Dict[str, Any]] = None
        self.last_post_verdict: Optional[Dict[str, Any]] = None
        self._tl = threading.local()

    def _resolve_tool_name(self, name: str) -> str:
        raw = str(name or "").strip()
        return self.tool_aliases.get(raw.lower(), raw)

    def check_pre(self, name: str, args: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate an Ouroboros tool call before execution."""
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
            host="ouroboros",
            journal_path=self.journal_path,
            session_state_path=self.session_state_path,
            profile=self.profile,
            policy_path=self.policy_path,
        )
        self.last_verdict = verdict
        return verdict

    def should_block(self, verdict: Mapping[str, Any]) -> bool:
        """Return True if the verdict should prevent tool execution."""
        if self.shadow_mode:
            return False
        dec = str(verdict.get("decision") or "").lower()
        if dec == "deny":
            return True
        if self.block_on_escalate and dec == "escalate":
            return True
        return False

    def scrub_post(
        self,
        name: str,
        args: Optional[Mapping[str, Any]],
        output: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run PostToolUse secret scrubbing and indirect prompt-injection inspection."""
        if not self.scrub_outputs:
            info = {"scrubbed": False, "secrets_found": False, "prompt_injection_detected": False}
            self.last_post_verdict = info
            return output, info

        from .hook import evaluate_post_tool_use_payload

        raw_text: Any = output
        attr_target: Optional[str] = None
        if not isinstance(output, (str, dict, list, int, float, bool)) and output is not None:
            for attr in ("text", "output", "content"):
                val = getattr(output, attr, None)
                if isinstance(val, str):
                    raw_text = val
                    attr_target = attr
                    break

        payload: Dict[str, Any] = {
            "hook_event_name": "PostToolUse",
            "session_id": self.session_id,
            "tool_name": self._resolve_tool_name(name),
            "tool_input": dict(args or {}),
            "tool_response": raw_text,
        }
        if self.cwd:
            payload["cwd"] = self.cwd

        post_verdict = evaluate_post_tool_use_payload(
            payload,
            log_decisions=self.log_decisions,
            shadow_mode=self.shadow_mode,
            host="ouroboros",
            journal_path=self.journal_path,
            session_state_path=self.session_state_path,
            policy_path=self.policy_path,
            replace_tool_output=self.replace_tool_output,
        )
        self.last_post_verdict = post_verdict

        redacted = post_verdict.get("redacted_output")
        if self.replace_tool_output and not self.shadow_mode and redacted is not None and post_verdict.get("secrets_found"):
            if attr_target is not None:
                try:
                    if dataclasses.is_dataclass(output):
                        return dataclasses.replace(output, **{attr_target: redacted}), post_verdict
                    setattr(output, attr_target, redacted)
                except Exception:
                    pass
                return output, post_verdict
            return redacted, post_verdict
        return output, post_verdict

    def narrow_schemas(
        self,
        task_description: str,
        schemas: Sequence[Dict[str, Any]],
        *,
        top_k: int = 5,
        context_tags: Optional[List[str]] = None,
        always_include: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Filter Ouroboros OpenAI-style tool schemas using lazy MCP-Zero MoE routing."""
        schema_list = list(schemas or [])
        if not schema_list:
            return [], {"mode": "empty", "candidate_tools": [], "reduction_ratio": 0.0}

        # Lazy import preserves pure-stdlib core isolation (`verify_core_isolation()`).
        from .core.tool_routing_moe import narrow_tool_surface

        names: List[str] = []
        descriptions: Dict[str, str] = {}
        flat_schemas: List[Dict[str, Any]] = []
        by_name: Dict[str, Dict[str, Any]] = {}

        for schema in schema_list:
            if not isinstance(schema, Mapping):
                continue
            tname = extract_schema_name(schema)
            if not tname:
                continue
            desc = extract_schema_description(schema)
            names.append(tname)
            descriptions[tname] = desc
            flat_schemas.append({"name": tname, "description": desc})
            by_name[tname] = dict(schema)

        if not names:
            return schema_list, {"mode": "passthrough", "candidate_tools": [], "reduction_ratio": 0.0}

        routing = narrow_tool_surface(
            task_description=task_description,
            available_tools=names,
            top_k=max(1, int(top_k)),
            context_tags=context_tags,
            tool_descriptions=descriptions,
            tool_schemas=flat_schemas,
        )
        selected_names = set(routing.get("candidate_tools") or [])
        if always_include:
            selected_names.update(str(t) for t in always_include if t)

        filtered = [schema for schema in schema_list if extract_schema_name(schema) in selected_names]
        if not filtered:
            filtered = schema_list
        return filtered, routing

    def _depth(self) -> int:
        return int(getattr(self._tl, "depth", 0))

    def _enter(self) -> int:
        d = self._depth() + 1
        self._tl.depth = d
        return d

    def _exit(self) -> None:
        self._tl.depth = max(0, self._depth() - 1)

    def wrap_registry(
        self,
        registry: Any,
        *,
        wrap_schemas: bool = False,
        task_getter: Optional[Callable[[], str]] = None,
    ) -> Any:
        """Idempotently wrap ``registry.execute`` and ``registry.execute_result`` in place."""
        if registry is None:
            raise ValueError("registry must not be None")

        registry._aetheris_gate = self
        if getattr(registry, "_aetheris_microbrains_wrapped", False):
            return registry

        orig_execute = getattr(registry, "execute", None)
        orig_execute_result = getattr(registry, "execute_result", None)
        orig_schemas = getattr(registry, "schemas", None)

        if callable(orig_execute):
            @functools.wraps(orig_execute)
            def _guarded_execute(name: str, args: Optional[Dict[str, Any]] = None, *rest: Any, **kwargs: Any) -> Any:
                active_gate: OuroborosMicrobrainsGate = getattr(registry, "_aetheris_gate", self)
                if active_gate._depth() > 0:
                    return orig_execute(name, args, *rest, **kwargs)
                active_gate._enter()
                try:
                    verdict = active_gate.check_pre(name, args)
                    if active_gate.should_block(verdict):
                        return format_block_error(name, verdict)
                    raw_out = orig_execute(name, args, *rest, **kwargs)
                    clean_out, _ = active_gate.scrub_post(name, args, raw_out)
                    return clean_out
                finally:
                    active_gate._exit()

            registry.execute = _guarded_execute

        if callable(orig_execute_result):
            @functools.wraps(orig_execute_result)
            def _guarded_execute_result(name: str, args: Optional[Dict[str, Any]] = None, *rest: Any, **kwargs: Any) -> Any:
                active_gate: OuroborosMicrobrainsGate = getattr(registry, "_aetheris_gate", self)
                if active_gate._depth() > 0:
                    return orig_execute_result(name, args, *rest, **kwargs)
                active_gate._enter()
                try:
                    verdict = active_gate.check_pre(name, args)
                    if active_gate.should_block(verdict):
                        msg = format_block_error(name, verdict)
                        return OuroborosBlockedResult(
                            text=msg,
                            error=msg,
                            output=msg,
                            content=msg,
                            verdict=verdict,
                        )
                    raw_res = orig_execute_result(name, args, *rest, **kwargs)
                    clean_res, _ = active_gate.scrub_post(name, args, raw_res)
                    return clean_res
                finally:
                    active_gate._exit()

            registry.execute_result = _guarded_execute_result

        if wrap_schemas and callable(orig_schemas):
            @functools.wraps(orig_schemas)
            def _guarded_schemas(*s_args: Any, **s_kwargs: Any) -> List[Dict[str, Any]]:
                all_schemas = list(orig_schemas(*s_args, **s_kwargs) or [])
                active_gate: OuroborosMicrobrainsGate = getattr(registry, "_aetheris_gate", self)
                if callable(task_getter):
                    task_desc = str(task_getter() or "").strip()
                    if task_desc:
                        narrowed, _ = active_gate.narrow_schemas(task_desc, all_schemas)
                        return narrowed
                return all_schemas

            registry.schemas = _guarded_schemas

        registry._aetheris_microbrains_wrapped = True
        return registry


def wrap_ouroboros_registry(
    registry: Any,
    *,
    gate: Optional[OuroborosMicrobrainsGate] = None,
    wrap_schemas: bool = False,
    task_getter: Optional[Callable[[], str]] = None,
    **gate_kwargs: Any,
) -> Any:
    """Attach an :class:`OuroborosMicrobrainsGate` to an Ouroboros ``ToolRegistry`` instance."""
    resolved_gate = gate if gate is not None else OuroborosMicrobrainsGate(**gate_kwargs)
    return resolved_gate.wrap_registry(registry, wrap_schemas=wrap_schemas, task_getter=task_getter)
