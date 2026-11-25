"""OpenHands Python SDK SecurityAnalyzer backed by Aetheris Microbrains."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

try:
    from openhands.sdk.security.analyzer import SecurityAnalyzerBase  # type: ignore[import-not-found]
    from openhands.sdk.security.risk import SecurityRisk  # type: ignore[import-not-found]
except Exception:
    class SecurityRisk(str, Enum):  # type: ignore[no-redef]
        """Fallback OpenHands SecurityRisk enum when openhands-sdk is not installed."""

        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        UNKNOWN = "UNKNOWN"

    class SecurityAnalyzerBase:  # type: ignore[no-redef]
        """Fallback base class matching OpenHands SecurityAnalyzerBase."""

        def security_risk(self, action: Any) -> SecurityRisk:
            return SecurityRisk.UNKNOWN


class AetherisSecurityAnalyzer(SecurityAnalyzerBase):
    """Evaluate OpenHands ``ActionEvent`` objects via Aetheris Microbrains."""

    def __init__(
        self,
        *,
        profile: str = "dev",
        policy_path: Optional[str] = None,
        log_decisions: bool = False,
        session_state_path: Optional[str] = None,
    ) -> None:
        self.profile = profile
        self.policy_path = policy_path
        self.log_decisions = log_decisions
        self.session_state_path = session_state_path
        self.last_verdict: Optional[Dict[str, Any]] = None

    @staticmethod
    def _action_to_payload(action: Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            tool_name = str(
                action.get("tool_name")
                or action.get("toolName")
                or action.get("tool")
                or action.get("action")
                or ""
            )
            tool_input = (
                action.get("tool_input")
                if isinstance(action.get("tool_input"), dict)
                else (
                    action.get("arguments")
                    if isinstance(action.get("arguments"), dict)
                    else (action.get("args") if isinstance(action.get("args"), dict) else action)
                )
            )
            return {"tool_name": tool_name, "tool_input": tool_input}

        tool_name = str(
            getattr(action, "tool_name", None)
            or getattr(action, "toolName", None)
            or getattr(action, "tool", None)
            or getattr(action, "action", None)
            or type(action).__name__
        )
        raw_args = (
            getattr(action, "arguments", None)
            or getattr(action, "tool_input", None)
            or getattr(action, "args", None)
        )
        if not isinstance(raw_args, dict):
            extracted: Dict[str, Any] = {}
            for attr in ("command", "path", "file_path", "old_str", "new_str", "old_string", "new_string", "file_text", "content"):
                val = getattr(action, attr, None)
                if val is not None:
                    extracted[attr] = val
            raw_args = extracted
        return {"tool_name": tool_name, "tool_input": raw_args}

    def evaluate_action(self, action: Any) -> Dict[str, Any]:
        """Run ``evaluate_hook_payload(..., host='openhands')`` on an OpenHands action."""
        from .hook import evaluate_hook_payload

        payload = self._action_to_payload(action)
        verdict = evaluate_hook_payload(
            payload,
            host="openhands",
            profile=self.profile,
            policy_path=self.policy_path,
            log_decisions=self.log_decisions,
            session_state_path=self.session_state_path,
        )
        self.last_verdict = verdict
        return verdict

    def security_risk(self, action: Any) -> SecurityRisk:
        """Map Aetheris decision verdicts to OpenHands ``SecurityRisk`` levels."""
        verdict = self.evaluate_action(action)
        dec = str(verdict.get("decision") or "").strip().lower()
        if dec == "allow":
            return SecurityRisk.LOW
        if dec == "escalate":
            return SecurityRisk.MEDIUM
        if dec == "deny":
            return SecurityRisk.HIGH
        return SecurityRisk.UNKNOWN
