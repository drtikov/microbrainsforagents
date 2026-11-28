"""Tests for loco-manipulation balance coupling, 2.5D foothold projection, and dexterous hand CBF."""

from __future__ import annotations

import numpy as np

from aetheris_microbrains.core.dexterous_hand_cbf import DexterousContact, DexterousGraspModel, DexterousTactileCBF
from aetheris_microbrains.core.legged_centroidal_cbf import CapturePointDCMBarrier
from aetheris_microbrains.core.loco_manipulation_foothold_cbf import (
    LocoManipulationBalanceCBF,
    SafeFootholdProjector,
    TerrainHeightmap25D,
)
from aetheris_microbrains.core.urdf_spatial_kinematics import URDFRobotModel

_HUMANOID_MINI_URDF = """<robot name="mini_humanoid">
  <link name="pelvis"><inertial><mass value="10.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="arm_link"><inertial><origin xyz="0.2 0 0"/><mass value="3.0"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial></link>
  <joint name="shoulder" type="revolute">
    <parent link="pelvis"/><child link="arm_link"/>
    <origin xyz="0 0 0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="40" velocity="5"/>
  </joint>
</robot>"""


def test_loco_manip_and_terrain_foothold():
    robot = URDFRobotModel.from_urdf_string(_HUMANOID_MINI_URDF, floating_base=True)
    dcm = CapturePointDCMBarrier.from_feet([(0.15, 0.1), (0.15, -0.1), (-0.15, -0.1), (-0.15, 0.1)], z_com=0.6, alpha=1.5)
    loco_cbf = LocoManipulationBalanceCBF(robot, dcm, loco_indices=list(range(6)), arm_indices=[6])
    q = np.zeros(robot.n_q)
    q[2] = 0.6
    v = np.zeros(robot.n_v)
    A_arm, b_arm = loco_cbf.arm_balance_halfspaces(q, v)
    assert A_arm.shape == (4, 1)
    assert b_arm.shape == (4,)

    elev = np.zeros((10, 10), dtype=np.float64)
    elev[5:, :] = 0.5  # steep step wall
    hm = TerrainHeightmap25D(elev, origin_xy=(0.0, 0.0), resolution=0.1, max_slope_rad=0.35, max_roughness=0.05)
    proj = SafeFootholdProjector(hm)
    res = proj.project([0.50, 0.5])
    assert res["projected"] is True
    ix, iy = hm.world_to_grid(res["xy_safe"])
    assert bool(hm.traversable[ix, iy]) is True


def test_dexterous_grasp_and_tactile_thermal():
    # 3-finger symmetric grasp -> force closure > 0
    c1 = DexterousContact("thumb", (0.05, 0.0, 0.0), (-1.0, 0.0, 0.0), friction_mu=0.7)
    c2 = DexterousContact("index", (-0.025, 0.043, 0.0), (0.5, -0.866, 0.0), friction_mu=0.7)
    c3 = DexterousContact("middle", (-0.025, -0.043, 0.0), (0.5, 0.866, 0.0), friction_mu=0.7)
    gm = DexterousGraspModel([c1, c2, c3])
    assert gm.force_closure_margin() > 0.0

    # All 3 fingers pushing from same side -> no force closure
    bad = DexterousGraspModel([
        DexterousContact("f1", (0.05, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        DexterousContact("f2", (0.05, 0.02, 0.0), (-1.0, 0.0, 0.0)),
        DexterousContact("f3", (0.05, -0.02, 0.0), (-1.0, 0.0, 0.0)),
    ])
    assert bad.force_closure_margin() <= 0.0

    tact = DexterousTactileCBF(n_joints=7, tau_cont=0.5, i2t_max=0.05, f_fragile_max=10.0)
    out = tact.step_tactile_and_thermal(tau_nom=[2.0] * 7, f_normal_meas=[1.0, 1.0, 1.0], f_tan_meas=[2.0, 2.0, 2.0], dt=0.01)
    assert out["slip_force_boost_n"] > 0.0
    # Run until I^2*t thermal derating engages
    for _ in range(10):
        out = tact.step_tactile_and_thermal(tau_nom=[2.0] * 7, f_normal_meas=[5.0] * 3, f_tan_meas=[0.5] * 3, dt=0.01)
    assert out["thermal_derated"] is True
