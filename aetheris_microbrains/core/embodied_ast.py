"""Embodied Robotics AST & Skill-Code Safety Verifier (pure Python stdlib).

Inspects LLM-generated robot skill source code (Python, ROS 2 ``rclpy``,
and URScript/launch snippets) for physical and cyber-physical hazards before
execution or caching into ``SkillLibrary``:

1. ``unguarded_actuator_loop`` — unbounded ``while`` loops calling actuator
   commands without iteration bounds, sleep/rate pacing, or safety/CBF guards.
2. ``safety_override_forbidden`` — calls or attributes disabling hardware
   safety interlocks, protective stops, watchdogs, or sending raw CAN/Modbus.
3. ``unit_hazard_deg_to_rad`` — degree-scale literals (> 6.5) passed directly
   to radian-expecting joint commands without ``radians``/``deg2rad`` conversion.
4. ``coordinate_frame_mismatch`` — conflicting TF frames in the same motion
   block without a ``transform`` / ``lookup_transform`` guard.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

RULES_CHECKED: Tuple[str, ...] = (
    "unguarded_actuator_loop",
    "safety_override_forbidden",
    "unit_hazard_deg_to_rad",
    "coordinate_frame_mismatch",
)

_ACTUATOR_CALLS: Set[str] = {
    "publish",
    "send_goal",
    "send_goal_async",
    "movej",
    "movel",
    "movep",
    "movec",
    "servoj",
    "speedj",
    "speedl",
    "set_velocity",
    "apply_torque",
    "command_torque",
    "set_torque",
    "set_joint_positions",
    "set_joint_angles",
    "set_joint_targets",
    "PublishJointTrajectory",
    "publish_joint_trajectory",
}

_SLEEP_CALLS: Set[str] = {
    "sleep",
    "spin_once",
    "sync",
    "wait",
    "wait_for_result",
}

_SAFETY_GUARD_TOKENS: Tuple[str, ...] = (
    "cbf",
    "safety_filter",
    "state_cbf",
    "verify_safe",
    "check_safety",
    "safety_guard",
    "clamp_action",
    "emergency_stop",
    "e_stop",
)

_FORBIDDEN_SAFETY_OVERRIDES: Set[str] = {
    "disable_collision_detection",
    "unlock_protective_stop",
    "bypass_safety",
    "set_watchdog_disabled",
    "disable_watchdog",
    "disable_safety",
    "disable_e_stop",
    "bypass_e_stop",
    "override_safety_limits",
    "disable_joint_limits",
    "can_send_raw",
    "modbus_write_coil",
    "modbus_write_register",
    "write_coil_raw",
}

_RADIAN_JOINT_CALLS: Set[str] = {
    "movej",
    "servoj",
    "speedj",
    "set_joint_positions",
    "set_joint_angles",
    "set_joint_targets",
    "PublishJointTrajectory",
    "publish_joint_trajectory",
    "move_joints",
}

_DEG_TO_RAD_CONVERTERS: Set[str] = {
    "radians",
    "deg2rad",
    "d2r",
    "degrees_to_radians",
    "to_radians",
}

_NON_ANGLE_KWARGS: Set[str] = {
    "a",
    "v",
    "t",
    "r",
    "acc",
    "vel",
    "accel",
    "acceleration",
    "velocity",
    "time",
    "duration",
    "timeout",
    "timeout_s",
    "blend",
    "radius",
    "hz",
    "rate",
    "wait",
    "blocking",
}

_SPATIAL_MOTION_CALLS: Set[str] = {
    "movel",
    "movep",
    "movec",
    "move_to_pose",
    "set_pose",
    "set_target_pose",
    "send_pose_goal",
    "publish_pose",
    "go_to_pose",
    "PoseStamped",
    "apply_cartesian_force",
    "set_cartesian_pose",
    "move_cartesian",
    "compute_ik",
}

_KNOWN_TF_FRAMES: Set[str] = {
    "world",
    "map",
    "odom",
    "base_link",
    "base_footprint",
    "base",
    "ee_link",
    "end_effector",
    "tool0",
    "tool_frame",
    "gripper_link",
    "camera_link",
    "camera_color_optical_frame",
    "wrist_3_link",
    "flange",
}

_FRAME_KWARGS: Set[str] = {
    "frame_id",
    "frame",
    "reference_frame",
    "ref_frame",
    "target_frame",
    "source_frame",
    "base_frame",
    "pose_frame",
}

_TRANSFORM_GUARD_TOKENS: Tuple[str, ...] = (
    "lookup_transform",
    "transform_pose",
    "transform_point",
    "transform_vector",
    "do_transform",
    "tf_transform",
    "convert_frame",
    "wait_for_transform",
    "transform",
)

_RAD_DEG_DEGREE_THRESHOLD = 6.5


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _full_call_path(node: ast.Call) -> str:
    parts: List[str] = []
    cur: ast.AST = node.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _snippet_at(lines: Sequence[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _is_pi_rad_conversion(node: ast.AST) -> bool:
    """Detect inline ``deg * pi / 180`` style conversions."""
    if isinstance(node, ast.Call) and _call_name(node).lower() in _DEG_TO_RAD_CONVERTERS:
        return True
    if isinstance(node, ast.BinOp):
        text = ast.unparse(node) if hasattr(ast, "unparse") else ""
        lowered = text.lower()
        if "pi" in lowered and "180" in lowered:
            return True
    return False


def _extract_degree_hazards(
    node: ast.AST,
    env_literals: Optional[Dict[str, List[float]]] = None,
) -> List[float]:
    """Return numeric literals with |v| > 6.5 not wrapped in radian conversion."""
    if _is_pi_rad_conversion(node):
        return []
    if isinstance(node, ast.Call):
        if _call_name(node).lower() in _DEG_TO_RAD_CONVERTERS:
            return []
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        val = float(node.value)
        return [val] if abs(val) > _RAD_DEG_DEGREE_THRESHOLD else []
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)) and not isinstance(inner.value, bool):
            val = -float(inner.value) if isinstance(node.op, ast.USub) else float(inner.value)
            return [val] if abs(val) > _RAD_DEG_DEGREE_THRESHOLD else []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: List[float] = []
        for elt in node.elts:
            out.extend(_extract_degree_hazards(elt, env_literals))
        return out
    if isinstance(node, ast.Name) and env_literals and node.id in env_literals:
        return list(env_literals[node.id])
    return []


def _collect_counter_vars(test: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for n in ast.walk(test):
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name):
            names.add(n.left.id)
    return names


def _has_bounded_loop_exit(while_node: ast.While) -> bool:
    """True when the while-loop has an explicit counter bound or counter-guarded break."""
    cond_counters = _collect_counter_vars(while_node.test)
    mutated_vars: Set[str] = set()
    for node in ast.walk(while_node):
        if node is while_node:
            continue
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            mutated_vars.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and isinstance(node.value, ast.BinOp):
                    mutated_vars.add(tgt.id)
    if cond_counters & mutated_vars:
        return True

    for node in ast.walk(while_node):
        if isinstance(node, ast.If):
            has_break = any(isinstance(sub, ast.Break) for sub in ast.walk(node))
            if has_break and (_collect_counter_vars(node.test) & mutated_vars):
                return True
    return False


def _normalize_frame(raw: str) -> str:
    return raw.strip().lstrip("/").lower()


def _is_frame_literal(s: str, from_frame_kwarg: bool = False) -> bool:
    norm = _normalize_frame(s)
    if not norm:
        return False
    if norm in _KNOWN_TF_FRAMES:
        return True
    if from_frame_kwarg and ("_" in norm or norm in {"world", "map", "odom", "base", "flange"}):
        return True
    return False


def _extract_frames_from_node(node: ast.AST, from_frame_kwarg: bool = False) -> Set[str]:
    frames: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _is_frame_literal(sub.value, from_frame_kwarg=from_frame_kwarg):
                frames.add(_normalize_frame(sub.value))
    return frames


class EmbodiedASTVerifier:
    """Static safety verifier for LLM-generated robotics skill code."""

    def __init__(self, degree_threshold: float = _RAD_DEG_DEGREE_THRESHOLD):
        self.degree_threshold = float(degree_threshold)

    def _make_finding(
        self,
        rule_id: str,
        severity: str,
        decision: str,
        line: int,
        col: int,
        message: str,
        lines: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "rule_id": rule_id,
            "severity": severity,
            "decision": decision,
            "line": int(line),
            "col": int(col),
            "message": message,
            "snippet": _snippet_at(lines, line),
        }

    def _check_python_ast(self, tree: ast.Module, lines: Sequence[str]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []

        # Track simple literal assignments like `q = [0, 90, 0]` for unit hazard checks.
        env_literals: Dict[str, List[float]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                hazards = _extract_degree_hazards(node.value)
                var_name = node.targets[0].id
                if hazards:
                    env_literals[var_name] = hazards
                elif var_name in env_literals:
                    env_literals.pop(var_name, None)

        # Rule 1: unguarded_actuator_loop
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                actuator_calls: List[Tuple[int, int, str]] = []
                has_sleep = False
                has_safety_guard = False
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        cname = _call_name(sub)
                        cpath = _full_call_path(sub).lower()
                        if cname in _ACTUATOR_CALLS:
                            actuator_calls.append((sub.lineno, sub.col_offset, cname))
                        if cname in _SLEEP_CALLS or "sleep" in cname.lower():
                            has_sleep = True
                        if any(tok in cpath for tok in _SAFETY_GUARD_TOKENS):
                            has_safety_guard = True
                    elif isinstance(sub, ast.Name) and any(tok in sub.id.lower() for tok in _SAFETY_GUARD_TOKENS):
                        has_safety_guard = True
                    elif isinstance(sub, ast.Constant) and sub.value == "verified_safe":
                        has_safety_guard = True

                if actuator_calls and not has_sleep and not has_safety_guard and not _has_bounded_loop_exit(node):
                    ln, col, cname = actuator_calls[0]
                    findings.append(
                        self._make_finding(
                            rule_id="unguarded_actuator_loop",
                            severity="high",
                            decision="deny",
                            line=node.lineno,
                            col=node.col_offset,
                            message=(
                                f"Unbounded while-loop invokes actuator '{cname}' (line {ln}) "
                                "without iteration bound, sleep/rate pacing, or safety/CBF guard."
                            ),
                            lines=lines,
                        )
                    )

        # Rule 2: safety_override_forbidden
        seen_override_locs: Set[Tuple[int, str]] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _call_name(node)
                if cname.lower() in _FORBIDDEN_SAFETY_OVERRIDES:
                    key = (node.lineno, cname.lower())
                    if key not in seen_override_locs:
                        seen_override_locs.add(key)
                        findings.append(
                            self._make_finding(
                                rule_id="safety_override_forbidden",
                                severity="critical",
                                decision="deny",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Forbidden safety override or raw fieldbus call '{cname}'.",
                                lines=lines,
                            )
                        )
            elif isinstance(node, ast.Attribute) and node.attr.lower() in _FORBIDDEN_SAFETY_OVERRIDES:
                key = (node.lineno, node.attr.lower())
                if key not in seen_override_locs:
                    seen_override_locs.add(key)
                    findings.append(
                        self._make_finding(
                            rule_id="safety_override_forbidden",
                            severity="critical",
                            decision="deny",
                            line=node.lineno,
                            col=node.col_offset,
                            message=f"Forbidden safety override attribute '{node.attr}'.",
                            lines=lines,
                        )
                    )
            elif isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                    for tgt in node.targets:
                        tname = tgt.id.lower() if isinstance(tgt, ast.Name) else (
                            tgt.attr.lower() if isinstance(tgt, ast.Attribute) else ""
                        )
                        if (
                            tname in {"safety_enabled", "collision_detection", "watchdog_enabled", "protective_stop"}
                            and node.value.value is False
                        ) or (tname in {"bypass_safety", "bypass_cbf"} and node.value.value is True):
                            findings.append(
                                self._make_finding(
                                    rule_id="safety_override_forbidden",
                                    severity="critical",
                                    decision="deny",
                                    line=node.lineno,
                                    col=node.col_offset,
                                    message=f"Safety interlock flag '{tname}' disabled via assignment.",
                                    lines=lines,
                                )
                            )

        # Rule 3: unit_hazard_deg_to_rad
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _call_name(node)
                if cname in _RADIAN_JOINT_CALLS:
                    suspicious: List[float] = []
                    for arg in node.args:
                        suspicious.extend(_extract_degree_hazards(arg, env_literals))
                    for kw in node.keywords:
                        if kw.arg and kw.arg.lower() in _NON_ANGLE_KWARGS:
                            continue
                        suspicious.extend(_extract_degree_hazards(kw.value, env_literals))
                    filtered = [v for v in suspicious if abs(v) > self.degree_threshold]
                    if filtered:
                        findings.append(
                            self._make_finding(
                                rule_id="unit_hazard_deg_to_rad",
                                severity="high",
                                decision="deny",
                                line=node.lineno,
                                col=node.col_offset,
                                message=(
                                    f"Degree-scale literal(s) {filtered} passed to radian-expecting "
                                    f"joint function '{cname}' without math.radians()/deg2rad()."
                                ),
                                lines=lines,
                            )
                        )

        # Rule 4: coordinate_frame_mismatch (checked per function/module block)
        blocks: List[ast.AST] = [tree]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                blocks.append(node)

        reported_mismatch_lines: Set[int] = set()
        for blk in blocks:
            # Skip module-level double-counting when functions exist by checking direct statements if FunctionDef
            blk_nodes = list(ast.walk(blk)) if blk is not tree else [
                n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            all_subnodes: List[ast.AST] = []
            for root_n in blk_nodes:
                all_subnodes.extend(ast.walk(root_n))

            has_transform_guard = False
            frames_seen: Set[str] = set()
            first_spatial_loc: Optional[Tuple[int, int]] = None
            var_frames: Dict[str, Set[str]] = {}

            for n in all_subnodes:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    # Check if RHS constructs a pose or dict with frame info (and not a transform guard)
                    if not (isinstance(n.value, ast.Call) and any(t in _full_call_path(n.value).lower() for t in _TRANSFORM_GUARD_TOKENS)):
                        vframes = _extract_frames_from_node(n.value, from_frame_kwarg=False)
                        for kw in getattr(n.value, "keywords", []):
                            if kw.arg and kw.arg.lower() in _FRAME_KWARGS:
                                vframes.update(_extract_frames_from_node(kw.value, from_frame_kwarg=True))
                        if vframes:
                            var_frames[n.targets[0].id] = vframes

                if isinstance(n, ast.Call):
                    cname = _call_name(n)
                    cpath = _full_call_path(n).lower()
                    if any(tok in cpath for tok in _TRANSFORM_GUARD_TOKENS):
                        has_transform_guard = True
                        continue

                    call_frames: Set[str] = set()
                    is_spatial = cname in _SPATIAL_MOTION_CALLS
                    for kw in n.keywords:
                        if kw.arg and kw.arg.lower() in _FRAME_KWARGS:
                            is_spatial = True
                            call_frames.update(_extract_frames_from_node(kw.value, from_frame_kwarg=True))
                    if is_spatial:
                        for arg in n.args:
                            call_frames.update(_extract_frames_from_node(arg, from_frame_kwarg=False))
                            if isinstance(arg, ast.Name) and arg.id in var_frames:
                                call_frames.update(var_frames[arg.id])
                        for kw in n.keywords:
                            if isinstance(kw.value, ast.Name) and kw.value.id in var_frames:
                                call_frames.update(var_frames[kw.value.id])
                        if call_frames:
                            frames_seen.update(call_frames)
                            if first_spatial_loc is None:
                                first_spatial_loc = (n.lineno, n.col_offset)

            if len(frames_seen) >= 2 and not has_transform_guard and first_spatial_loc is not None:
                ln, col = first_spatial_loc
                if ln not in reported_mismatch_lines:
                    reported_mismatch_lines.add(ln)
                    findings.append(
                        self._make_finding(
                            rule_id="coordinate_frame_mismatch",
                            severity="medium",
                            decision="escalate",
                            line=ln,
                            col=col,
                            message=(
                                f"Conflicting TF coordinate frames {sorted(frames_seen)} used in spatial "
                                "commands without a transform/lookup_transform guard."
                            ),
                            lines=lines,
                        )
                    )

        findings.extend(self._check_unitree_ast(tree, lines))
        return findings

    def _check_unitree_ast(self, tree: ast.AST, lines: List[str]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        src_low = "\n".join(lines).lower()
        uses_lowcmd = ("lowcmd" in src_low) or ("rt/lowcmd" in src_low)
        if uses_lowcmd and "lowstate" not in src_low and "rt/lowstate" not in src_low:
            findings.append(
                self._make_finding(
                    rule_id="unitree_unhomed_lowcmd",
                    severity="high",
                    decision="deny",
                    line=1,
                    col=0,
                    message="Unitree rt/lowcmd used without reading rt/lowstate initial joint positions.",
                    lines=lines,
                )
            )
        if uses_lowcmd and "crc" not in src_low:
            findings.append(
                self._make_finding(
                    rule_id="unitree_missing_crc",
                    severity="high",
                    decision="deny",
                    line=1,
                    col=0,
                    message="Unitree LowCmd published without CRC32 computation.",
                    lines=lines,
                )
            )
        has_while = any(isinstance(n, ast.While) for n in ast.walk(tree))
        has_try_finally = any(isinstance(n, ast.Try) and bool(n.finalbody) for n in ast.walk(tree))
        if uses_lowcmd and has_while and not has_try_finally:
            findings.append(
                self._make_finding(
                    rule_id="unitree_missing_damping_finally",
                    severity="high",
                    decision="deny",
                    line=1,
                    col=0,
                    message="Unitree LowCmd control loop lacks try/finally damping shutdown handler.",
                    lines=lines,
                )
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == "kp":
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)) and float(node.value.value) > 300.0:
                            findings.append(
                                self._make_finding(
                                    rule_id="unitree_excessive_kp_gain",
                                    severity="high",
                                    decision="deny",
                                    line=getattr(node, "lineno", 1),
                                    col=getattr(node, "col_offset", 0),
                                    message=f"Excessive Unitree motor kp={node.value.value} > 300.0.",
                                    lines=lines,
                                )
                            )
        return findings

    def _looks_like_urscript_or_launch(self, code: str) -> str:
        stripped = code.lstrip()
        if stripped.startswith("<") and ("<launch" in stripped or "<node" in stripped):
            return "launch"
        ur_markers = ("movej(", "movel(", "movep(", "servoj(", "unlock_protective_stop", "def ", "end")
        if any(m in code for m in ur_markers) and ("end" in code or "movej(" in code or "movel(" in code):
            return "urscript"
        return "unknown"

    def _check_non_python_fallback(self, code: str, lines: Sequence[str]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []

        # Rule 2: forbidden safety overrides
        override_re = re.compile(
            r"\b(" + "|".join(sorted(_FORBIDDEN_SAFETY_OVERRIDES)) + r")\b",
            re.IGNORECASE,
        )
        for idx, line in enumerate(lines, 1):
            m = override_re.search(line)
            if m:
                findings.append(
                    self._make_finding(
                        rule_id="safety_override_forbidden",
                        severity="critical",
                        decision="deny",
                        line=idx,
                        col=m.start(),
                        message=f"Forbidden safety override or raw fieldbus call '{m.group(1)}'.",
                        lines=lines,
                    )
                )

        # Rule 3: degree literals in radian joint calls
        joint_call_re = re.compile(
            r"\b(" + "|".join(sorted(_RADIAN_JOINT_CALLS)) + r")\s*\(([^)]*)\)"
        )
        deg_conv_re = re.compile(r"\b(radians|deg2rad|d2r|degrees_to_radians)\s*\(", re.IGNORECASE)
        num_re = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)")
        for idx, line in enumerate(lines, 1):
            for jm in joint_call_re.finditer(line):
                args_str = jm.group(2)
                cleaned = deg_conv_re.sub("(", args_str)
                # Strip keyword args like a=1.4, v=1.05
                cleaned = re.sub(r"\b(?:a|v|t|r|acc|vel|time|timeout|hz|rate)\s*=\s*-?\d+(?:\.\d+)?", "", cleaned)
                if deg_conv_re.search(args_str):
                    continue
                vals = [float(x) for x in num_re.findall(cleaned) if abs(float(x)) > self.degree_threshold]
                if vals:
                    findings.append(
                        self._make_finding(
                            rule_id="unit_hazard_deg_to_rad",
                            severity="high",
                            decision="deny",
                            line=idx,
                            col=jm.start(),
                            message=(
                                f"Degree-scale literal(s) {vals} passed to '{jm.group(1)}' "
                                "without radians()/d2r()/deg2rad()."
                            ),
                            lines=lines,
                        )
                    )

        # Rule 1: unguarded while loops
        while_re = re.compile(r"^\s*while\b.*[:\(]")
        act_re = re.compile(r"\b(" + "|".join(sorted(_ACTUATOR_CALLS)) + r")\s*\(")
        sleep_re = re.compile(r"\b(sleep|sync|rate\.sleep|cbf|safety_filter|verify_safe)\b", re.IGNORECASE)
        for idx, line in enumerate(lines, 1):
            if while_re.match(line):
                window = "\n".join(lines[idx - 1 : min(len(lines), idx + 20)])
                if act_re.search(window) and not sleep_re.search(window):
                    findings.append(
                        self._make_finding(
                            rule_id="unguarded_actuator_loop",
                            severity="high",
                            decision="deny",
                            line=idx,
                            col=0,
                            message="Unbounded while-loop invokes actuator command without sleep/sync/CBF guard.",
                            lines=lines,
                        )
                    )

        # Rule 4: conflicting TF frames
        has_transform = any(tok in code.lower() for tok in _TRANSFORM_GUARD_TOKENS)
        if not has_transform:
            frames_found: Set[str] = set()
            first_ln = 1
            for idx, line in enumerate(lines, 1):
                for fr in _KNOWN_TF_FRAMES:
                    if re.search(rf"""['"/]{re.escape(fr)}['"]""", line, re.IGNORECASE):
                        if not frames_found:
                            first_ln = idx
                        frames_found.add(fr)
            if len(frames_found) >= 2:
                findings.append(
                    self._make_finding(
                        rule_id="coordinate_frame_mismatch",
                        severity="medium",
                        decision="escalate",
                        line=first_ln,
                        col=0,
                        message=(
                            f"Conflicting TF coordinate frames {sorted(frames_found)} used without "
                            "a transform/lookup_transform guard."
                        ),
                        lines=lines,
                    )
                )

        return findings

    def verify(
        self,
        code: str,
        filename: str = "<skill>",
        language: str = "auto",
    ) -> Dict[str, Any]:
        """Inspect robot skill source code and return a structured verification receipt."""
        if not isinstance(code, str):
            raise TypeError(f"code must be str, got {type(code).__name__}")
        lines = code.splitlines()
        findings: List[Dict[str, Any]] = []
        detected_lang = language
        syntax_valid = True

        if language in ("auto", "python"):
            try:
                tree = ast.parse(code, filename=filename)
                detected_lang = "python"
                findings.extend(self._check_python_ast(tree, lines))
            except SyntaxError as exc:
                syntax_valid = False
                inferred = self._looks_like_urscript_or_launch(code) if language == "auto" else "python"
                detected_lang = inferred
                if inferred in ("urscript", "launch"):
                    syntax_valid = True
                    findings.extend(self._check_non_python_fallback(code, lines))
                else:
                    findings.extend(self._check_non_python_fallback(code, lines))
                    findings.append(
                        self._make_finding(
                            rule_id="syntax_error",
                            severity="medium",
                            decision="escalate",
                            line=exc.lineno or 1,
                            col=exc.offset or 0,
                            message=f"Could not parse skill source as Python AST: {exc.msg}",
                            lines=lines,
                        )
                    )
        else:
            findings.extend(self._check_non_python_fallback(code, lines))

        if any(f["decision"] == "deny" for f in findings):
            decision = "deny"
        elif findings:
            decision = "escalate"
        else:
            decision = "allow"

        return {
            "decision": decision,
            "verified_safe": decision == "allow",
            "findings": findings,
            "rules_checked": list(RULES_CHECKED),
            "language": detected_lang,
            "syntax_valid": syntax_valid,
        }


__all__ = ["EmbodiedASTVerifier", "RULES_CHECKED"]
