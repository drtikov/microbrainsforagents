"""Tests for SemanticCBFCompiler, TimeVaryingBarrier, and EmbodiedASTVerifier."""

from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_AETHER_DIR = _PKG_DIR.parent
for _p in (str(_PKG_DIR), str(_AETHER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from aetheris_microbrains.core.embodied_ast import RULES_CHECKED, EmbodiedASTVerifier
from aetheris_microbrains.core.semantic_cbf_compiler import (
    NormBoundBarrier,
    PhysicalSafetyIR,
    PhysicalSafetyRule,
    SemanticCBFCompiler,
    TiltConeBarrier,
    TimeVaryingBarrier,
)
from aetheris_microbrains.core.state_cbf_filter import (
    HalfPlaneBarrier,
    LinearDynamics,
    SphereBarrier,
    StateCBFSafetyFilter,
)


# ---------------------------------------------------------------------------
# SemanticCBFCompiler & TimeVaryingBarrier tests
# ---------------------------------------------------------------------------


class TestSemanticCBFCompiler:
    def _sample_scene(self):
        return {
            "entities": {
                "beaker": {"center": [0.5, 0.0, 0.2], "radius": 0.05},
                "goal_pad": {"center": [1.0, 0.0, 0.2], "radius": 0.15},
            }
        }

    def test_parse_rule_strings_all_dsl_forms(self):
        comp = SemanticCBFCompiler()
        r_avoid = comp.parse_rule_string('avoid(entity="beaker", min_dist_m=0.2)')
        assert r_avoid.rule_type == "avoid"
        assert r_avoid.entity == "beaker"
        assert r_avoid.params["min_dist_m"] == pytest.approx(0.2)

        r_upright = comp.parse_rule_string("keep_upright(max_tilt_deg=15)")
        assert r_upright.rule_type == "keep_upright"
        assert r_upright.params["max_tilt_deg"] == pytest.approx(15.0)

        r_speed = comp.parse_rule_string("always(speed_mps<=0.5)")
        assert r_speed.rule_type == "always"
        assert r_speed.params["metric"] == "speed_mps"
        assert r_speed.params["value"] == pytest.approx(0.5)

        r_force = comp.parse_rule_string("always(force_n<=8.0)")
        assert r_force.rule_type == "always"
        assert r_force.params["metric"] == "force_n"
        assert r_force.params["value"] == pytest.approx(8.0)

        r_ev = comp.parse_rule_string("eventually(within_s=5.0, target=[1, 0, 0], radius=0.2)")
        assert r_ev.rule_type == "eventually"
        assert r_ev.params["within_s"] == pytest.approx(5.0)
        assert r_ev.params["target"] == [1, 0, 0]
        assert r_ev.params["radius"] == pytest.approx(0.2)

        r_box = comp.parse_rule_string("keep_in_box(min=[-1, -1, 0], max=[1, 1, 2])")
        assert r_box.rule_type == "keep_in_box"

    def test_ir_serialization_roundtrip(self):
        comp = SemanticCBFCompiler()
        ir = comp.build_ir(
            [
                'avoid(entity="beaker", min_dist_m=0.2)',
                "always(speed_mps<=0.5)",
                "eventually(within_s=5.0, target=[1, 0, 0], radius=0.2)",
            ],
            metadata={"task": "pour"},
        )
        payload = ir.to_json()
        restored = PhysicalSafetyIR.from_json(payload)
        assert len(restored.rules) == 3
        assert restored.metadata == {"task": "pour"}
        assert restored.rules[0].entity == "beaker"
        assert restored.rules[2].rule_type == "eventually"

    def test_unknown_entity_fails_closed(self):
        comp = SemanticCBFCompiler()
        rules = ['avoid(entity="missing_beaker", min_dist_m=0.2)']
        receipt = comp.compile_with_receipt(rules, scene_graph=self._sample_scene())
        assert receipt["verified_compiled"] is False
        assert receipt["status"] == "unresolved_entities"
        assert receipt["unresolved"] == ["missing_beaker"]
        assert receipt["barriers"] == []

        with pytest.raises(ValueError, match="missing_beaker"):
            comp.compile_rules(rules, scene_graph=self._sample_scene())

    def test_compile_and_integrate_with_state_cbf_filter(self):
        comp = SemanticCBFCompiler()
        scene = self._sample_scene()
        barriers = comp.compile_rules(
            [
                'avoid(entity="beaker", min_dist_m=0.2)',
                "always(speed_mps<=0.5)",
                "always(force_n<=8.0)",
                "keep_in_box(min=[-1.0, -1.0, 0.0], max=[2.0, 1.0, 1.5])",
            ],
            scene_graph=scene,
            pos_slice=slice(0, 3),
            vel_slice=slice(3, 6),
            force_slice=slice(6, 9),
            state_dim=9,
        )
        # 1 Sphere + 1 speed NormBound + 1 force NormBound + 6 box HalfPlanes = 9 barriers
        assert len(barriers) == 9
        assert isinstance(barriers[0], SphereBarrier)
        assert isinstance(barriers[1], NormBoundBarrier)
        assert isinstance(barriers[2], NormBoundBarrier)
        assert all(isinstance(b, HalfPlaneBarrier) for b in barriers[3:])

        # Simple integrator over 9D state [pos(3), vel(3), force(3)]
        A = np.zeros((9, 9), dtype=np.float64)
        B = np.eye(9, dtype=np.float64)
        filt = StateCBFSafetyFilter(
            LinearDynamics(A, B),
            barriers,
            u_min=[-2.0] * 9,
            u_max=[2.0] * 9,
        )

        # Safe starting state inside box, outside beaker, below speed/force bounds
        x = np.array([0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        dt = 0.05
        avoid_bar = barriers[0]
        # Drive aggressively toward beaker center [0.5, 0.0, 0.2]
        for _ in range(40):
            u_nom = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            out = filt.filter(x, u_nom)
            assert out["verified_safe"] is True
            assert avoid_bar.h(x) >= -1e-6
            x = x + dt * np.asarray(out["u_filtered"], dtype=np.float64)
        # Beaker exclusion radius is 0.05 + 0.20 = 0.25 -> never penetrated
        assert avoid_bar.h(x) >= -1e-6

    def test_keep_upright_compiles_tilt_barrier(self):
        comp = SemanticCBFCompiler()
        barriers = comp.compile_rules(["keep_upright(max_tilt_deg=15)"])
        assert len(barriers) == 1
        bar = barriers[0]
        assert isinstance(bar, TiltConeBarrier)
        assert bar.h(np.array([0.0, 0.0])) > 0.0
        assert bar.h(np.array([math.radians(25.0), 0.0])) < 0.0


class TestTimeVaryingBarrier:
    def test_funnel_shrinks_monotonically_and_integrates_with_filter(self):
        comp = SemanticCBFCompiler()
        barriers = comp.compile_rules(
            ["eventually(within_s=4.0, target=[1.0, 0.0, 0.0], radius=0.2, initial_radius=1.2, alpha=3.0)"],
            pos_slice=slice(0, 3),
            state_dim=3,
        )
        assert len(barriers) == 1
        tv = barriers[0]
        assert isinstance(tv, TimeVaryingBarrier)

        radii = [tv.radius_at(t) for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
        assert radii[0] == pytest.approx(1.2)
        assert radii[4] == pytest.approx(0.2)
        assert radii[5] == pytest.approx(0.2)
        for i in range(4):
            assert radii[i] > radii[i + 1]

        x_far = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # distance 1.0 from [1, 0, 0]
        assert tv.h_at(x_far, 0.0) > 0.0  # inside r(0)=1.2 funnel
        assert tv.h_at(x_far, 4.0) < 0.0  # outside r(4)=0.2 target ball
        assert tv.dh_dt(x_far, 1.0) < 0.0
        np.testing.assert_allclose(tv.grad_h_t(x_far, 1.0), np.array([2.0, 0.0, 0.0]))

        # Inflate shrinks safe set by -delta
        tv_inf = tv.inflate(0.5)
        assert tv_inf.h_at(x_far, 0.0) == pytest.approx(tv.h_at(x_far, 0.0) - 0.5)

        # Updating current_time drives StateCBFSafetyFilter toward the target
        filt = StateCBFSafetyFilter(
            LinearDynamics(np.zeros((3, 3)), np.eye(3)),
            [tv],
            u_min=[-2.0, -2.0, -2.0],
            u_max=[2.0, 2.0, 2.0],
        )
        x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        dt = 0.1
        for step in range(45):
            tv.set_time(step * dt)
            out = filt.filter(x, [0.0, 0.0, 0.0])  # zero nominal -> CBF pushes into shrinking funnel
            assert out["verified_safe"] is True
            x = x + dt * np.asarray(out["u_filtered"], dtype=np.float64)
        # By t=4.5, x has been guided close to the target ball (radius 0.2 around [1, 0, 0])
        assert np.linalg.norm(x - np.array([1.0, 0.0, 0.0])) < 0.35


# ---------------------------------------------------------------------------
# EmbodiedASTVerifier tests
# ---------------------------------------------------------------------------


class TestEmbodiedASTVerifier:
    def test_pure_stdlib_zero_numpy(self):
        path = _PKG_DIR / "core" / "embodied_ast.py"
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0].lower())
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0].lower())
        assert "numpy" not in imported
        assert "np" not in imported

    def test_catches_unguarded_actuator_loop(self):
        verifier = EmbodiedASTVerifier()
        bad_code = """
def spin_forever(pub, cmd):
    while True:
        pub.publish(cmd)
"""
        res = verifier.verify(bad_code)
        assert res["verified_safe"] is False
        assert res["decision"] == "deny"
        assert res["rules_checked"] == list(RULES_CHECKED)
        assert any(f["rule_id"] == "unguarded_actuator_loop" for f in res["findings"])

    def test_allows_guarded_actuator_loops(self):
        verifier = EmbodiedASTVerifier()
        sleep_guarded = """
def control_loop(pub, cmd, rate):
    while True:
        pub.publish(cmd)
        rate.sleep()
"""
        cbf_guarded = """
def cbf_loop(pub, cbf_filter, x, cmd):
    while True:
        out = cbf_filter.filter(x, cmd)
        if not out["verified_safe"]:
            break
        pub.publish(out["u_filtered"])
"""
        bounded_loop = """
def bounded_step(pub, cmd):
    i = 0
    while i < 10:
        pub.publish(cmd)
        i += 1
"""
        for code in (sleep_guarded, cbf_guarded, bounded_loop):
            res = verifier.verify(code)
            assert res["verified_safe"] is True, res
            assert res["decision"] == "allow"
            assert res["findings"] == []

    def test_catches_safety_override_forbidden(self):
        verifier = EmbodiedASTVerifier()
        for snippet in (
            "robot.unlock_protective_stop()\n",
            "arm.disable_collision_detection()\n",
            "bus.can_send_raw(0x123, b'\\x00')\n",
            "plc.modbus_write_coil(1, False)\n",
            "robot.safety_enabled = False\n",
        ):
            res = verifier.verify(snippet)
            assert res["verified_safe"] is False, snippet
            assert res["decision"] == "deny"
            assert any(f["rule_id"] == "safety_override_forbidden" for f in res["findings"])

    def test_catches_unit_hazard_deg_to_rad_and_allows_radians(self):
        verifier = EmbodiedASTVerifier()
        for bad in (
            "movej(90)\n",
            "movej([0.0, -90, 45, 0.0, 0.0, 0.0])\n",
            "set_joint_angles(180, 0.0)\n",
            "PublishJointTrajectory(positions=[0.0, 360.0])\n",
        ):
            res = verifier.verify(bad)
            assert res["verified_safe"] is False, bad
            assert res["decision"] == "deny"
            assert any(f["rule_id"] == "unit_hazard_deg_to_rad" for f in res["findings"])

        for good in (
            "import math\nmovej(math.radians(90))\n",
            "import math\nset_joint_positions([math.radians(45), math.radians(-90)])\n",
            "movej([0.0, -1.57, 0.78, 0.0, 0.0, 0.0])\n",
            "set_joint_angles(deg2rad(180), 0.0)\n",
        ):
            res = verifier.verify(good)
            assert res["verified_safe"] is True, good
            assert res["decision"] == "allow"

    def test_catches_coordinate_frame_mismatch_and_allows_transformed(self):
        verifier = EmbodiedASTVerifier()
        bad = """
def move_pick(arm):
    pose = PoseStamped(frame_id="ee_link")
    arm.movel(pose, frame_id="base_link")
"""
        res = verifier.verify(bad)
        assert res["verified_safe"] is False
        assert res["decision"] in ("deny", "escalate")
        assert any(f["rule_id"] == "coordinate_frame_mismatch" for f in res["findings"])

        good_same = """
def move_pick_same(arm):
    pose = PoseStamped(frame_id="base_link")
    arm.movel(pose, frame_id="base_link")
"""
        good_transformed = """
def move_pick_tf(arm, tf_buffer):
    pose = PoseStamped(frame_id="ee_link")
    tf_pose = tf_buffer.transform(pose, "base_link")
    arm.movel(tf_pose, frame_id="base_link")
"""
        for code in (good_same, good_transformed):
            ok = verifier.verify(code)
            assert ok["verified_safe"] is True, ok
            assert ok["decision"] == "allow"

    def test_urscript_fallback_detection(self):
        verifier = EmbodiedASTVerifier()
        urscript_bad = """
def pick_part():
  unlock_protective_stop()
  movej([0, -90, 45, 0, 0, 0], a=1.4, v=1.05)
end
"""
        res = verifier.verify(urscript_bad)
        assert res["verified_safe"] is False
        assert res["decision"] == "deny"
        rule_ids = {f["rule_id"] for f in res["findings"]}
        assert "safety_override_forbidden" in rule_ids
        assert "unit_hazard_deg_to_rad" in rule_ids

    def test_no_forbidden_template_labels_in_created_files(self):
        pat = re.compile(r"proposal\s*\d+", re.IGNORECASE)
        for rel in (
            "core/semantic_cbf_compiler.py",
            "core/embodied_ast.py",
            "tests/test_semantic_cbf_and_embodied_ast.py",
        ):
            text = (_PKG_DIR / rel).read_text(encoding="utf-8")
            assert pat.search(text) is None, f"Forbidden label found in {rel}"
