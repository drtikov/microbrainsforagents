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
