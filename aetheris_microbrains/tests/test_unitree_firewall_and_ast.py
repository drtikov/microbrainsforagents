"""Tests for Unitree firewall protocol, MCP proxy gating, and Unitree AST rules."""

from __future__ import annotations

import ast
from pathlib import Path

from aetheris_microbrains.core.capability_tokens import CapabilityCaveat, mint_root_capability
from aetheris_microbrains.core.embodied_ast import EmbodiedASTVerifier
from aetheris_microbrains.hosts.ros2_adapter import EmbodiedActionFirewall
from aetheris_microbrains.mcp_proxy import is_embodied_mcp_tool


def test_unitree_firewall_and_mcp():
    fw = EmbodiedActionFirewall()
    secret = b"unitree-secret-key"
    tok = mint_root_capability(
        secret,
        subject="go2_agent",
        caveat=CapabilityCaveat(max_linear_vel_mps=0.8, max_contact_force_n=25.0),
    )

    ok = fw.evaluate_embodied_action({"rpc": "Move", "vx": 0.4, "vy": 0.0}, protocol="unitree", capability_token=tok, root_secret=secret)
    assert ok["authorized"] is True

    fast = fw.evaluate_embodied_action({"rpc": "Move", "vx": 1.5, "vy": 0.0}, protocol="unitree", capability_token=tok, root_secret=secret)
    assert fast["authorized"] is False

    stunt = fw.evaluate_embodied_action({"rpc": "FrontFlip"}, protocol="unitree")
    assert stunt["authorized"] is False
    assert stunt["reason"] == "unitree_acrobatic_stunt_blocked"

    mid_drop = fw.evaluate_embodied_action({"rpc": "StandDown", "base_speed_mps": 0.8}, protocol="unitree")
    assert mid_drop["authorized"] is False
    assert mid_drop["reason"] == "unitree_mid_motion_stand_down_blocked"

    assert is_embodied_mcp_tool("unitree_sport_move") is True
    assert is_embodied_mcp_tool("dex3_set_grasp") is True

    # Stunt obfuscation / separator / Cyrillic homoglyph resistance
    for obfuscated in ("front-flip", "front flip", "front_flip", "back-flip", "zero torque", "do frontflip now", "Fr\u043entFlip"):
        res = fw.evaluate_embodied_action({"rpc": obfuscated}, protocol="unitree")
        assert res["authorized"] is False, obfuscated
        assert res["reason"] == "unitree_acrobatic_stunt_blocked", obfuscated

    # Mid-motion StandDown bypass variants (vx/vy key, negative speed, spaces)
    for bad_standdown in (
        {"rpc": "StandDown", "vx": 0.8, "vy": 0.0},
        {"rpc": "StandDown", "base_speed_mps": -0.5},
        {"rpc": "stand down", "base_speed_mps": 0.8},
    ):
        res = fw.evaluate_embodied_action(bad_standdown, protocol="unitree")
        assert res["authorized"] is False
        assert res["reason"] == "unitree_mid_motion_stand_down_blocked"

    ok_standdown = fw.evaluate_embodied_action(
        {"rpc": "StandDown", "base_speed_mps": 0.05},
        protocol="unitree",
        capability_token=tok,
        root_secret=secret,
    )
    assert ok_standdown["authorized"] is True

    # Geofence must block out-of-bounds waypoints even when target_position is inside
    fence = (
        ((1.0, 0.0), 2.0),
        ((-1.0, 0.0), 2.0),
        ((0.0, 1.0), 2.0),
        ((0.0, -1.0), 2.0),
    )
    wp_hard = fw.evaluate_embodied_action(
        {"rpc": "Move", "target_position": [1.0, 1.0], "waypoints": [[5.0, 0.0], [1.0, 1.0]]},
        protocol="unitree",
        hard_limits={"geofence_polytope": fence},
    )
    assert wp_hard["authorized"] is False
    assert wp_hard["reason"] == "hard_limit_geofence_violation"

    tok_fenced = mint_root_capability(
        secret,
        subject="go2_fenced",
        caveat=CapabilityCaveat(max_linear_vel_mps=0.8, geofence_polytope=fence),
    )
    wp_tok = fw.evaluate_embodied_action(
        {"rpc": "Move", "target_position": [1.0, 1.0], "waypoints": [[5.0, 0.0], [1.0, 1.0]]},
        protocol="unitree",
        capability_token=tok_fenced,
        root_secret=secret,
    )
    assert wp_tok["authorized"] is False
    assert wp_tok["reason"] == "geofence_violation"

    # Fail-closed when neither capability_token nor hard_limits nor mission_envelope is provided on unitree/mavlink
    no_limits = fw.evaluate_embodied_action({"rpc": "Move", "vx": 0.4}, protocol="unitree")
    assert no_limits["authorized"] is False
    assert no_limits["reason"] == "missing_capability_token_or_hard_limits"
    no_limits_mav = fw.evaluate_embodied_action({"x": 1.0, "y": 1.0}, protocol="mavlink")
    assert no_limits_mav["authorized"] is False
    assert no_limits_mav["reason"] == "missing_capability_token_or_hard_limits"


def test_unitree_ast_rules_and_pure_stdlib():
    verifier = EmbodiedASTVerifier()
    bad_code = """
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
while True:
    cmd.motor_cmd[0].kp = 500.0
    pub.Write(cmd)
"""
    res = verifier.verify(bad_code)
    rule_ids = {f["rule_id"] for f in res["findings"]}
    assert res["decision"] == "deny"
    assert {"unitree_unhomed_lowcmd", "unitree_missing_crc", "unitree_missing_damping_finally", "unitree_excessive_kp_gain"} <= rule_ids

    good_code = """
sub = ChannelSubscriber("rt/lowstate", LowState_)
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
try:
    for _ in range(10):
        cmd.motor_cmd[0].kp = 40.0
        cmd.crc = CRC().Crc(cmd)
        pub.Write(cmd)
        time.sleep(0.002)
finally:
    damp(cmd)
"""
    res_ok = verifier.verify(good_code)
    assert res_ok["decision"] == "allow"

    root = Path(__file__).resolve().parents[1]
    for rel in ("hosts/ros2_adapter.py", "mcp_proxy.py", "core/embodied_ast.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "numpy" for a in node.names), rel
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "numpy", rel
