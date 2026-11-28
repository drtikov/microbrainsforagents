"""Dependency-free SE(3) Lie algebra, Featherstone spatial dynamics, and URDF robot model.

Provides pure-NumPy analytical kinematics, geometric Jacobians, O(n_dof) Recursive
Newton-Euler (RNEA) inverse dynamics, Composite Rigid-Body Algorithm (CRBA) mass
matrix assembly, Euler-Lagrange safety-filter wiring, and analytical-gradient
self-collision capsule barriers parsed directly from URDF XML.

Spatial 6-vectors follow Featherstone's ``[angular; linear]`` ordering:
- Spatial motion twist: ``v = [omega_x, omega_y, omega_z, v_x, v_y, v_z]^T``
- Spatial force wrench: ``f = [tau_x, tau_y, tau_z, f_x, f_y, f_z]^T``
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

from .state_cbf_filter import EulerLagrangeDynamics, HalfPlaneBarrier
from .whole_body_barriers import CapsulePairBarrier


# ---------------------------------------------------------------------------
# SE(3) Lie Group & Featherstone Spatial Algebra
# ---------------------------------------------------------------------------


def hat_so3(omega: Sequence[float]) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix ``[omega]_x`` in so(3) such that ``[omega]_x p = omega x p``."""
    w = np.asarray(omega, dtype=np.float64).flatten()
    if w.shape[0] != 3:
        raise ValueError(f"hat_so3 expects a 3-vector, got shape {w.shape}")
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=np.float64,
    )


def vee_so3(mat: np.ndarray) -> np.ndarray:
    """Inverse of ``hat_so3``: extract 3-vector from a 3x3 skew-symmetric matrix."""
    M = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    return np.array(
        [
            0.5 * (M[2, 1] - M[1, 2]),
            0.5 * (M[0, 2] - M[2, 0]),
            0.5 * (M[1, 0] - M[0, 1]),
        ],
        dtype=np.float64,
    )


def exp_so3(omega: Sequence[float], theta: Optional[float] = None) -> np.ndarray:
    """Exponential map ``exp([phi]_x) -> SO(3)`` via Rodrigues' formula with Taylor series near 0.

    If ``theta`` is ``None``, ``omega`` is treated as the 3D rotation vector ``phi = axis * angle``.
    If ``theta`` is provided, ``omega`` is normalized as the rotation axis and scaled by ``theta``.
    """
    w = np.asarray(omega, dtype=np.float64).flatten()
    if w.shape[0] != 3:
        raise ValueError(f"exp_so3 expects a 3-vector, got shape {w.shape}")
    if theta is not None:
        norm_w = float(np.linalg.norm(w))
        if norm_w <= 1e-15:
            return np.eye(3, dtype=np.float64)
        phi = (w / norm_w) * float(theta)
    else:
        phi = w

    angle2 = float(np.dot(phi, phi))
    angle = math.sqrt(angle2)
    K = hat_so3(phi)
    if angle < 1e-8:
        c1 = 1.0 - angle2 / 6.0 + (angle2 * angle2) / 120.0
        c2 = 0.5 - angle2 / 24.0 + (angle2 * angle2) / 720.0
    else:
        c1 = math.sin(angle) / angle
        c2 = (1.0 - math.cos(angle)) / angle2
    return np.eye(3, dtype=np.float64) + c1 * K + c2 * (K @ K)


def exp_se3(xi: Sequence[float], theta: Optional[float] = None) -> np.ndarray:
    """Exponential map ``exp(xi^) -> SE(3)`` (4x4) for twist ``xi = [omega; v]``.

    If ``theta`` is ``None``, ``xi`` is the integrated 6-vector twist.
    If ``theta`` is provided, ``xi`` is scaled by ``float(theta)``.
    """
    twist = np.asarray(xi, dtype=np.float64).flatten()
    if twist.shape[0] != 6:
        raise ValueError(f"exp_se3 expects a 6-vector [omega; v], got shape {twist.shape}")
    if theta is not None:
        twist = twist * float(theta)

    phi = twist[:3]
    v = twist[3:6]
    angle2 = float(np.dot(phi, phi))
    angle = math.sqrt(angle2)
    K = hat_so3(phi)
    K2 = K @ K

    if angle < 1e-8:
        c1 = 1.0 - angle2 / 6.0 + (angle2 * angle2) / 120.0
        c2 = 0.5 - angle2 / 24.0 + (angle2 * angle2) / 720.0
        c3 = (1.0 / 6.0) - angle2 / 120.0 + (angle2 * angle2) / 5040.0
    else:
        c1 = math.sin(angle) / angle
        c2 = (1.0 - math.cos(angle)) / angle2
        c3 = (angle - math.sin(angle)) / (angle2 * angle)

    R = np.eye(3, dtype=np.float64) + c1 * K + c2 * K2
    G = np.eye(3, dtype=np.float64) + c2 * K + c3 * K2
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


def adjoint_se3(T: np.ndarray) -> np.ndarray:
    """Return 6x6 SE(3) Adjoint matrix ``Ad_T`` in ``[omega; v]`` ordering.

    For ``T = [[R, p], [0, 1]]``, ``Ad_T = [[R, 0], [[p]_x R, R]]``.
    """
    T_arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T_arr[:3, :3]
    p = T_arr[:3, 3]
    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[3:, :3] = hat_so3(p) @ R
    Ad[3:, 3:] = R
    return Ad


def spatial_cross_motion(
    v: Sequence[float], u: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Featherstone spatial motion cross product ``v x`` in ``[omega; v_lin]`` ordering.

    If ``u`` is ``None``, returns the 6x6 operator matrix ``crm(v) = [[w_x, 0], [v_x, w_x]]``.
    If ``u`` is provided, returns the 6-vector ``crm(v) @ u``.
    """
    v_arr = np.asarray(v, dtype=np.float64).flatten()
    if v_arr.shape[0] != 6:
        raise ValueError(f"spatial_cross_motion expects a 6-vector, got shape {v_arr.shape}")
    wx, wy, wz, vx, vy, vz = (
        float(v_arr[0]),
        float(v_arr[1]),
        float(v_arr[2]),
        float(v_arr[3]),
        float(v_arr[4]),
        float(v_arr[5]),
    )
    if u is not None:
        u_arr = np.asarray(u, dtype=np.float64).flatten()
        if u_arr.shape[0] != 6:
            raise ValueError(f"spatial_cross_motion second operand must be 6D, got {u_arr.shape}")
        uwx, uwy, uwz, uvx, uvy, uvz = (
            float(u_arr[0]),
            float(u_arr[1]),
            float(u_arr[2]),
            float(u_arr[3]),
            float(u_arr[4]),
            float(u_arr[5]),
        )
        return np.array(
            [
                wy * uwz - wz * uwy,
                wz * uwx - wx * uwz,
                wx * uwy - wy * uwx,
                (wy * uvz - wz * uvy) + (vy * uwz - vz * uwy),
                (wz * uvx - wx * uvz) + (vz * uwx - vx * uwz),
                (wx * uvy - wy * uvx) + (vx * uwy - vy * uwx),
            ],
            dtype=np.float64,
        )

    W = hat_so3(v_arr[:3])
    V = hat_so3(v_arr[3:6])
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = W
    out[3:, :3] = V
    out[3:, 3:] = W
    return out


def spatial_cross_force(
    v: Sequence[float], f: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Featherstone spatial force cross product ``v x* = -crm(v)^T`` in ``[tau; f_lin]`` ordering.

    If ``f`` is ``None``, returns the 6x6 operator matrix ``crf(v) = [[w_x, v_x], [0, w_x]]``.
    If ``f`` is provided, returns the 6-vector ``crf(v) @ f``.
    """
    v_arr = np.asarray(v, dtype=np.float64).flatten()
    if v_arr.shape[0] != 6:
        raise ValueError(f"spatial_cross_force expects a 6-vector, got shape {v_arr.shape}")
    wx, wy, wz, vx, vy, vz = (
        float(v_arr[0]),
        float(v_arr[1]),
        float(v_arr[2]),
        float(v_arr[3]),
        float(v_arr[4]),
        float(v_arr[5]),
    )
    if f is not None:
        f_arr = np.asarray(f, dtype=np.float64).flatten()
        if f_arr.shape[0] != 6:
            raise ValueError(f"spatial_cross_force second operand must be 6D, got {f_arr.shape}")
        tx, ty, tz, fx, fy, fz = (
            float(f_arr[0]),
            float(f_arr[1]),
            float(f_arr[2]),
            float(f_arr[3]),
            float(f_arr[4]),
            float(f_arr[5]),
        )
        return np.array(
            [
                (wy * tz - wz * ty) + (vy * fz - vz * fy),
                (wz * tx - wx * tz) + (vz * fx - vx * fz),
                (wx * ty - wy * tx) + (vx * fy - vy * fx),
                wy * fz - wz * fy,
                wz * fx - wx * fz,
                wx * fy - wy * fx,
            ],
            dtype=np.float64,
        )

    W = hat_so3(v_arr[:3])
    V = hat_so3(v_arr[3:6])
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = W
    out[:3, 3:] = V
    out[3:, 3:] = W
    return out


def rpy_to_so3(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert URDF Roll-Pitch-Yaw (extrinsic XYZ / intrinsic ZYX: ``Rz(yaw) Ry(pitch) Rx(roll)``) to SO(3)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _fast_unit_axis_rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Fast exact 3x3 rotation matrix around pre-normalized 3D unit axis."""
    ux, uy, uz = float(axis[0]), float(axis[1]), float(axis[2])
    c = math.cos(theta)
    s = math.sin(theta)
    v = 1.0 - c
    return np.array(
        [
            [c + ux * ux * v, ux * uy * v - uz * s, ux * uz * v + uy * s],
            [uy * ux * v + uz * s, c + uy * uy * v, uy * uz * v - ux * s],
            [uz * ux * v - uy * s, uz * uy * v + ux * s, c + uz * uz * v],
        ],
        dtype=np.float64,
    )


def _plucker_motion_from_parent(R_pk: np.ndarray, r_pk: np.ndarray) -> np.ndarray:
    """Featherstone Plucker motion transform ``^k X_p = Ad_{T_{p,k}^{-1}}``."""
    R_T = R_pk.T
    rx, ry, rz = float(r_pk[0]), float(r_pk[1]), float(r_pk[2])
    skew_r = np.array(
        [[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]],
        dtype=np.float64,
    )
    X = np.zeros((6, 6), dtype=np.float64)
    X[:3, :3] = R_T
    X[3:, :3] = -R_T @ skew_r
    X[3:, 3:] = R_T
    return X


def _spatial_inertia_from_urdf(
    mass: float,
    com_xyz: np.ndarray,
    com_R: np.ndarray,
    inertia_com: np.ndarray,
) -> np.ndarray:
    """Assemble 6x6 symmetric positive-semidefinite spatial inertia in link frame."""
    if mass <= 0.0 and not np.any(inertia_com):
        return np.zeros((6, 6), dtype=np.float64)
    I_bar = com_R @ inertia_com @ com_R.T
    C_skew = hat_so3(com_xyz)
    I_spatial = np.zeros((6, 6), dtype=np.float64)
    I_spatial[:3, :3] = I_bar + mass * (C_skew @ C_skew.T)
    I_spatial[:3, 3:] = mass * C_skew
    I_spatial[3:, :3] = mass * C_skew.T
    I_spatial[3:, 3:] = mass * np.eye(3, dtype=np.float64)
    return 0.5 * (I_spatial + I_spatial.T)


# ---------------------------------------------------------------------------
# Safe stdlib XML Parsing & URDF Data Structures
# ---------------------------------------------------------------------------


def _safe_parse_urdf_xml(xml_str: str) -> ET.Element:
    """Parse URDF XML string via stdlib ElementTree while rejecting DTDs and entities."""
    if not isinstance(xml_str, str) or not xml_str.strip():
        raise ValueError("URDF XML string must be non-empty")

    lowered = xml_str.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValueError("XML DTD and entity declarations are forbidden in URDF input")

    parser = ET.XMLParser()
    expat_parser = getattr(parser, "parser", None)
    if expat_parser is not None:

        def _forbid(*_args, **_kwargs):
            raise ValueError("XML DTD and entity expansion are forbidden in URDF input")

        expat_parser.StartDoctypeDeclHandler = _forbid
        expat_parser.EntityDeclHandler = _forbid
        expat_parser.UnparsedEntityDeclHandler = _forbid
        expat_parser.ExternalEntityRefHandler = _forbid

    try:
        root = ET.fromstring(xml_str, parser=parser)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML: {exc}") from exc

    if root.tag != "robot":
        raise ValueError(f"Expected root <robot> tag, got <{root.tag}>")
    return root


def _parse_vec3(attr: Optional[str], default: Tuple[float, float, float]) -> np.ndarray:
    if attr is None or not attr.strip():
        return np.array(default, dtype=np.float64)
    parts = attr.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 space-separated floats, got {attr!r}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def _parse_origin(elem: Optional[ET.Element]) -> Tuple[np.ndarray, np.ndarray]:
    if elem is None:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = _parse_vec3(elem.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_vec3(elem.get("rpy"), (0.0, 0.0, 0.0))
    return rpy_to_so3(float(rpy[0]), float(rpy[1]), float(rpy[2])), xyz


class CollisionCapsule(NamedTuple):
    """Link-local bounding capsule ``(local_p0, local_p1, radius)`` fitted to a URDF collision primitive."""

    local_p0: np.ndarray
    local_p1: np.ndarray
    radius: float


def _fit_collision_capsule(collision_elem: ET.Element) -> Optional[CollisionCapsule]:
    """Fit a sound outer bounding capsule ``(local_p0, local_p1, radius)`` to cylinder/box/sphere."""
    geom = collision_elem.find("geometry")
    if geom is None:
        return None
    R_col, t_col = _parse_origin(collision_elem.find("origin"))

    cyl = geom.find("cylinder")
    if cyl is not None:
        radius = float(cyl.get("radius", "0.0"))
        length = float(cyl.get("length", "0.0"))
        if radius < 0.0 or length < 0.0:
            raise ValueError("URDF cylinder radius and length must be non-negative")
        half_vec = R_col @ np.array([0.0, 0.0, 0.5 * length], dtype=np.float64)
        return CollisionCapsule(
            local_p0=t_col - half_vec,
            local_p1=t_col + half_vec,
            radius=radius,
        )

    sph = geom.find("sphere")
    if sph is not None:
        radius = float(sph.get("radius", "0.0"))
        if radius < 0.0:
            raise ValueError("URDF sphere radius must be non-negative")
        return CollisionCapsule(
            local_p0=t_col.copy(),
            local_p1=t_col.copy(),
            radius=radius,
        )

    box = geom.find("box")
    if box is not None:
        size = _parse_vec3(box.get("size"), (0.0, 0.0, 0.0))
        if np.any(size < 0.0):
            raise ValueError("URDF box dimensions must be non-negative")
        major_axis = int(np.argmax(size))
        other_axes = [i for i in range(3) if i != major_axis]
        radius = 0.5 * math.hypot(float(size[other_axes[0]]), float(size[other_axes[1]]))
        local_axis_vec = np.zeros(3, dtype=np.float64)
        local_axis_vec[major_axis] = 0.5 * float(size[major_axis])
        half_vec = R_col @ local_axis_vec
        return CollisionCapsule(
            local_p0=t_col - half_vec,
            local_p1=t_col + half_vec,
            radius=radius,
        )

    return None


@dataclass(frozen=True)
class _BodyNode:
    """Internal topological tree node (link 0 is fixed base; k >= 1 has a parent link < k)."""

    link_idx: int
    link_name: str
    parent_idx: int
    joint_name: Optional[str]
    joint_type: str  # "root", "revolute", "continuous", "prismatic", "fixed"
    dof_index: Optional[int]
    origin_R: np.ndarray
    origin_t: np.ndarray
    axis: np.ndarray
    motion_subspace: np.ndarray  # (6,)
    mass: float
    com_xyz: np.ndarray
    spatial_inertia: np.ndarray  # (6, 6)
    capsules: Tuple[CollisionCapsule, ...]
    ancestor_chain: Tuple[int, ...]  # link indices from root (0) to link_idx inclusive
    movable_depth: int  # number of movable joints from root to this link


# ---------------------------------------------------------------------------
# URDFRobotModel
# ---------------------------------------------------------------------------


class URDFRobotModel:
    """Pure-NumPy URDF spatial kinematics, RNEA/CRBA dynamics, and self-collision CBF builder."""

    _SUPPORTED_JOINTS = ("revolute", "continuous", "prismatic", "fixed")

    def __init__(
        self,
        name: str,
        nodes: List[_BodyNode],
        joint_names: List[str],
        q_min: np.ndarray,
        q_max: np.ndarray,
        dq_max: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ):
        self.name = name
        self._nodes = nodes
        self.n_links = len(nodes)
        self.n_dof = len(joint_names)
        self.joint_names = list(joint_names)
        self.link_names = [n.link_name for n in nodes]
        self._link_name_to_idx: Dict[str, int] = {n.link_name: n.link_idx for n in nodes}
        self.root_link_name = nodes[0].link_name
        self.end_effector_link_name = self._select_default_ee_link()

        self.q_min = np.asarray(q_min, dtype=np.float64).flatten()
        self.q_max = np.asarray(q_max, dtype=np.float64).flatten()
        self.dq_max = np.asarray(dq_max, dtype=np.float64).flatten()
        self.u_min = np.asarray(u_min, dtype=np.float64).flatten()
        self.u_max = np.asarray(u_max, dtype=np.float64).flatten()
        self.collision_capsules: Dict[str, List[CollisionCapsule]] = {
            n.link_name: list(n.capsules) for n in nodes if n.capsules
        }

    def _select_default_ee_link(self) -> str:
        parents = {n.parent_idx for n in self._nodes if n.parent_idx >= 0}
        leaves = [n.link_name for n in self._nodes if n.link_idx not in parents]
        return leaves[-1] if leaves else self._nodes[-1].link_name

    @classmethod
    def from_urdf_string(cls, xml_str: str) -> "URDFRobotModel":
        """Parse a ``URDFRobotModel`` safely from a URDF XML string."""
        root_elem = _safe_parse_urdf_xml(xml_str)
        robot_name = root_elem.get("name", "urdf_robot")

        # Parse links
        raw_links: Dict[str, Tuple[float, np.ndarray, np.ndarray, Tuple[CollisionCapsule, ...]]] = {}
        for link_elem in root_elem.findall("link"):
            lname = link_elem.get("name")
            if not lname:
                raise ValueError("URDF <link> missing required 'name' attribute")
            if lname in raw_links:
                raise ValueError(f"Duplicate URDF link name: {lname!r}")

            mass = 0.0
            com_xyz = np.zeros(3, dtype=np.float64)
            I_spatial = np.zeros((6, 6), dtype=np.float64)
            inertial_elem = link_elem.find("inertial")
            if inertial_elem is not None:
                mass_elem = inertial_elem.find("mass")
                if mass_elem is not None:
                    mass = float(mass_elem.get("value", "0.0"))
                if mass < 0.0:
                    raise ValueError(f"Negative mass {mass} on link {lname!r}")
                com_R, com_xyz = _parse_origin(inertial_elem.find("origin"))
                inertia_elem = inertial_elem.find("inertia")
                I_com = np.zeros((3, 3), dtype=np.float64)
                if inertia_elem is not None:
                    ixx = float(inertia_elem.get("ixx", "0.0"))
                    ixy = float(inertia_elem.get("ixy", "0.0"))
                    ixz = float(inertia_elem.get("ixz", "0.0"))
                    iyy = float(inertia_elem.get("iyy", "0.0"))
                    iyz = float(inertia_elem.get("iyz", "0.0"))
                    izz = float(inertia_elem.get("izz", "0.0"))
                    I_com = np.array(
                        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
                        dtype=np.float64,
                    )
                I_spatial = _spatial_inertia_from_urdf(mass, com_xyz, com_R, I_com)

            capsules: List[CollisionCapsule] = []
            for col_elem in link_elem.findall("collision"):
                cap = _fit_collision_capsule(col_elem)
                if cap is not None:
                    capsules.append(cap)

            raw_links[lname] = (mass, com_xyz, I_spatial, tuple(capsules))

        if not raw_links:
            raise ValueError("URDF contains no <link> elements")

        # Parse joints
        children_by_parent: Dict[str, List[dict]] = {lname: [] for lname in raw_links}
        parent_of_child: Dict[str, str] = {}
        seen_joints = set()

        for joint_elem in root_elem.findall("joint"):
            jname = joint_elem.get("name")
            jtype = joint_elem.get("type", "").strip()
            if not jname:
                raise ValueError("URDF <joint> missing required 'name' attribute")
            if jname in seen_joints:
                raise ValueError(f"Duplicate URDF joint name: {jname!r}")
            seen_joints.add(jname)
            if jtype not in cls._SUPPORTED_JOINTS:
                raise ValueError(f"Unsupported URDF joint type {jtype!r} on joint {jname!r}")

            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                raise ValueError(f"Joint {jname!r} missing <parent> or <child>")
            plink = parent_elem.get("link", "")
            clink = child_elem.get("link", "")
            if plink not in raw_links:
                raise ValueError(f"Joint {jname!r} references unknown parent link {plink!r}")
            if clink not in raw_links:
                raise ValueError(f"Joint {jname!r} references unknown child link {clink!r}")
            if clink in parent_of_child:
                raise ValueError(f"Kinematic loop detected: link {clink!r} has multiple parent joints")
            parent_of_child[clink] = plink

            origin_R, origin_t = _parse_origin(joint_elem.find("origin"))
            axis_elem = joint_elem.find("axis")
            axis_raw = _parse_vec3(
                axis_elem.get("xyz") if axis_elem is not None else None,
                (1.0, 0.0, 0.0),
            )
            if jtype == "fixed":
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                S_vec = np.zeros(6, dtype=np.float64)
            else:
                norm_a = float(np.linalg.norm(axis_raw))
                if norm_a <= 1e-12:
                    raise ValueError(f"Joint {jname!r} has zero axis vector")
                axis = axis_raw / norm_a
                S_vec = np.zeros(6, dtype=np.float64)
                if jtype in ("revolute", "continuous"):
                    S_vec[:3] = axis
                else:
                    S_vec[3:6] = axis

            limit_elem = joint_elem.find("limit")
            lower = -math.inf
            upper = math.inf
            effort = math.inf
            velocity = math.inf
            if limit_elem is not None:
                if jtype in ("revolute", "prismatic"):
                    lower = float(limit_elem.get("lower", "0.0"))
                    upper = float(limit_elem.get("upper", "0.0"))
                if limit_elem.get("effort") is not None:
                    effort = abs(float(limit_elem.get("effort", "0.0")))
                if limit_elem.get("velocity") is not None:
                    velocity = abs(float(limit_elem.get("velocity", "0.0")))

            children_by_parent[plink].append(
                {
                    "joint_name": jname,
                    "joint_type": jtype,
                    "child_link": clink,
                    "origin_R": origin_R,
                    "origin_t": origin_t,
                    "axis": axis,
                    "S": S_vec,
                    "lower": lower,
                    "upper": upper,
                    "effort": effort,
                    "velocity": velocity,
                }
            )

        roots = [lname for lname in raw_links if lname not in parent_of_child]
        if len(roots) != 1:
            raise ValueError(
                f"Expected exactly 1 root link in kinematic tree, found {len(roots)}: {roots}"
            )
        root_name = roots[0]

        # Deterministic depth-first topological sort preserving XML joint declaration order
        nodes: List[_BodyNode] = []
        joint_names: List[str] = []
        q_min_list: List[float] = []
        q_max_list: List[float] = []
        dq_max_list: List[float] = []
        u_min_list: List[float] = []
        u_max_list: List[float] = []

        root_mass, root_com, root_I, root_caps = raw_links[root_name]
        nodes.append(
            _BodyNode(
                link_idx=0,
                link_name=root_name,
                parent_idx=-1,
                joint_name=None,
                joint_type="root",
                dof_index=None,
                origin_R=np.eye(3, dtype=np.float64),
                origin_t=np.zeros(3, dtype=np.float64),
                axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                motion_subspace=np.zeros(6, dtype=np.float64),
                mass=root_mass,
                com_xyz=root_com,
                spatial_inertia=root_I,
                capsules=root_caps,
                ancestor_chain=(0,),
                movable_depth=0,
            )
        )

        def _visit(parent_idx: int, parent_name: str) -> None:
            parent_node = nodes[parent_idx]
            for jinfo in children_by_parent[parent_name]:
                cname = jinfo["child_link"]
                jtype = jinfo["joint_type"]
                if jtype == "fixed":
                    dof_idx: Optional[int] = None
                    mov_depth = parent_node.movable_depth
                else:
                    dof_idx = len(joint_names)
                    joint_names.append(jinfo["joint_name"])
                    q_min_list.append(jinfo["lower"])
                    q_max_list.append(jinfo["upper"])
                    dq_max_list.append(jinfo["velocity"])
                    u_min_list.append(-jinfo["effort"])
                    u_max_list.append(jinfo["effort"])
                    mov_depth = parent_node.movable_depth + 1

                c_idx = len(nodes)
                cmass, ccom, cI, ccaps = raw_links[cname]
                nodes.append(
                    _BodyNode(
                        link_idx=c_idx,
                        link_name=cname,
                        parent_idx=parent_idx,
                        joint_name=jinfo["joint_name"],
                        joint_type=jtype,
                        dof_index=dof_idx,
                        origin_R=jinfo["origin_R"],
                        origin_t=jinfo["origin_t"],
                        axis=jinfo["axis"],
                        motion_subspace=jinfo["S"],
                        mass=cmass,
                        com_xyz=ccom,
                        spatial_inertia=cI,
                        capsules=ccaps,
                        ancestor_chain=parent_node.ancestor_chain + (c_idx,),
                        movable_depth=mov_depth,
                    )
                )
                _visit(c_idx, cname)

        _visit(0, root_name)
        if len(nodes) != len(raw_links):
            raise ValueError("URDF contains disconnected or cyclic links")

        return cls(
            name=robot_name,
            nodes=nodes,
            joint_names=joint_names,
            q_min=np.asarray(q_min_list, dtype=np.float64),
            q_max=np.asarray(q_max_list, dtype=np.float64),
            dq_max=np.asarray(dq_max_list, dtype=np.float64),
            u_min=np.asarray(u_min_list, dtype=np.float64),
            u_max=np.asarray(u_max_list, dtype=np.float64),
        )

    @classmethod
    def from_urdf_file(cls, path: Union[str, Path]) -> "URDFRobotModel":
        """Load and parse a ``URDFRobotModel`` from a filesystem URDF path."""
        urdf_path = Path(path)
        xml_text = urdf_path.read_text(encoding="utf-8")
        return cls.from_urdf_string(xml_text)

    def _validate_q(self, q: Sequence[float], label: str = "q") -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64).flatten()
        if arr.shape[0] != self.n_dof:
            raise ValueError(f"Expected {label} of length {self.n_dof}, got {arr.shape[0]}")
        return arr

    def _parent_to_child_pose(self, node: _BodyNode, q_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(R_parent_child, r_parent_child)`` for ``node`` at joint configuration ``q_arr``."""
        jtype = node.joint_type
        if jtype == "fixed":
            return node.origin_R, node.origin_t
        qd = float(q_arr[node.dof_index])  # type: ignore[index]
        if jtype in ("revolute", "continuous"):
            R_j = _fast_unit_axis_rodrigues(node.axis, qd)
            return node.origin_R @ R_j, node.origin_t
        # prismatic
        return node.origin_R, node.origin_t + qd * (node.origin_R @ node.axis)

    # ------------------------------------------------------------------
    # Forward Kinematics & Analytical Jacobians
    # ------------------------------------------------------------------

    def forward_kinematics(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        """Compute 4x4 world homogeneous transforms ``T_world_link`` for all links in O(n_dof)."""
        q_arr = self._validate_q(q, "q")
        transforms: List[np.ndarray] = [np.eye(4, dtype=np.float64)]
        out: Dict[str, np.ndarray] = {self._nodes[0].link_name: transforms[0]}

        for k in range(1, self.n_links):
            node = self._nodes[k]
            T_parent = transforms[node.parent_idx]
            R_w_p = T_parent[:3, :3]
            p_w_p = T_parent[:3, 3]
            R_p_k, r_p_k = self._parent_to_child_pose(node, q_arr)
            T_w_k = np.eye(4, dtype=np.float64)
            T_w_k[:3, :3] = R_w_p @ R_p_k
            T_w_k[:3, 3] = p_w_p + R_w_p @ r_p_k
            transforms.append(T_w_k)
            out[node.link_name] = T_w_k
        return out

    def _chain_kinematics(
        self, q_arr: np.ndarray, link_idx: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        List[int],
        List[np.ndarray],
        List[np.ndarray],
        List[int],
        List[np.ndarray],
    ]:
        """Evaluate world pose and active joint axes along the root-to-link chain."""
        chain = self._nodes[link_idx].ancestor_chain
        R_w = np.eye(3, dtype=np.float64)
        p_w = np.zeros(3, dtype=np.float64)

        rev_dofs: List[int] = []
        rev_axes: List[np.ndarray] = []
        rev_origins: List[np.ndarray] = []
        pris_dofs: List[int] = []
        pris_axes: List[np.ndarray] = []

        for idx in chain[1:]:
            node = self._nodes[idx]
            R_p_k, r_p_k = self._parent_to_child_pose(node, q_arr)
            p_w = p_w + R_w @ r_p_k
            R_w = R_w @ R_p_k
            if node.dof_index is not None:
                z_world = R_w @ node.axis
                if node.joint_type in ("revolute", "continuous"):
                    rev_dofs.append(node.dof_index)
                    rev_axes.append(z_world)
                    rev_origins.append(p_w)
                else:
                    pris_dofs.append(node.dof_index)
                    pris_axes.append(z_world)

        return R_w, p_w, rev_dofs, rev_axes, rev_origins, pris_dofs, pris_axes

    def link_point_jacobian(
        self,
        q: Sequence[float],
        link_name: str,
        local_point: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ``(world_point_3d, J_linear_3xn)`` analytically in O(n_dof) with zero finite differences."""
        if link_name not in self._link_name_to_idx:
            raise KeyError(f"Unknown link name: {link_name!r}")
        q_arr = self._validate_q(q, "q")
        pt_local = _parse_vec3(None, (0.0, 0.0, 0.0)) if local_point is None else np.asarray(local_point, dtype=np.float64).flatten()
        if pt_local.shape[0] != 3:
            raise ValueError(f"local_point must be a 3-vector, got shape {pt_local.shape}")

        link_idx = self._link_name_to_idx[link_name]
        R_w, p_w, rev_dofs, rev_axes, rev_origins, pris_dofs, pris_axes = self._chain_kinematics(
            q_arr, link_idx
        )
        p_world = p_w + R_w @ pt_local
        J_lin = np.zeros((3, self.n_dof), dtype=np.float64)

        if rev_dofs:
            Z = np.asarray(rev_axes, dtype=np.float64)
            O = np.asarray(rev_origins, dtype=np.float64)
            J_lin[:, rev_dofs] = np.cross(Z, p_world[None, :] - O).T
        if pris_dofs:
            J_lin[:, pris_dofs] = np.asarray(pris_axes, dtype=np.float64).T

        return p_world, J_lin

    def end_effector_jacobian(
        self,
        q: Sequence[float],
        link_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute 6 x n_dof world-aligned geometric Jacobian ``[J_angular; J_linear]``."""
        target = self.end_effector_link_name if link_name is None else link_name
        if target not in self._link_name_to_idx:
            raise KeyError(f"Unknown link name: {target!r}")
        q_arr = self._validate_q(q, "q")
        link_idx = self._link_name_to_idx[target]
        _, p_ee, rev_dofs, rev_axes, rev_origins, pris_dofs, pris_axes = self._chain_kinematics(
            q_arr, link_idx
        )

        J = np.zeros((6, self.n_dof), dtype=np.float64)
        if rev_dofs:
            Z = np.asarray(rev_axes, dtype=np.float64)
            O = np.asarray(rev_origins, dtype=np.float64)
            J[:3, rev_dofs] = Z.T
            J[3:6, rev_dofs] = np.cross(Z, p_ee[None, :] - O).T
        if pris_dofs:
            J[3:6, pris_dofs] = np.asarray(pris_axes, dtype=np.float64).T
        return J

    def _link_segment_fk(
        self, q_arr: np.ndarray, link_idx: int, local_p0: np.ndarray, local_p1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        chain = self._nodes[link_idx].ancestor_chain
        R_w = np.eye(3, dtype=np.float64)
        p_w = np.zeros(3, dtype=np.float64)
        for idx in chain[1:]:
            node = self._nodes[idx]
            R_p_k, r_p_k = self._parent_to_child_pose(node, q_arr)
            p_w = p_w + R_w @ r_p_k
            R_w = R_w @ R_p_k
        return p_w + R_w @ local_p0, p_w + R_w @ local_p1

    def _link_segment_jacobians(
        self, q_arr: np.ndarray, link_idx: int, local_p0: np.ndarray, local_p1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        R_w, p_w, rev_dofs, rev_axes, rev_origins, pris_dofs, pris_axes = self._chain_kinematics(
            q_arr, link_idx
        )
        w0 = p_w + R_w @ local_p0
        w1 = p_w + R_w @ local_p1
        J0 = np.zeros((3, self.n_dof), dtype=np.float64)
        J1 = np.zeros((3, self.n_dof), dtype=np.float64)
        if rev_dofs:
            Z = np.asarray(rev_axes, dtype=np.float64)
            O = np.asarray(rev_origins, dtype=np.float64)
            J0[:, rev_dofs] = np.cross(Z, w0[None, :] - O).T
            J1[:, rev_dofs] = np.cross(Z, w1[None, :] - O).T
        if pris_dofs:
            P_axes = np.asarray(pris_axes, dtype=np.float64).T
            J0[:, pris_dofs] = P_axes
            J1[:, pris_dofs] = P_axes
        return J0, J1

    # ------------------------------------------------------------------
    # Featherstone RNEA, CRBA, Energy & Euler-Lagrange Wiring
    # ------------------------------------------------------------------

    def inverse_dynamics_rnea(
        self,
        q: Sequence[float],
        qdot: Sequence[float],
        qddot: Optional[Sequence[float]] = None,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
    ) -> np.ndarray:
        """Exact O(n_dof) Recursive Newton-Euler inverse dynamics ``tau = M(q)qddot + C(q,qdot)qdot + g(q)``."""
        q_arr = self._validate_q(q, "q")
        dq_arr = self._validate_q(qdot, "qdot")
        ddq_arr = (
            np.zeros(self.n_dof, dtype=np.float64)
            if qddot is None
            else self._validate_q(qddot, "qddot")
        )
        g_vec = np.asarray(gravity, dtype=np.float64).flatten()
        if g_vec.shape[0] != 3:
            raise ValueError(f"gravity must be a 3-vector, got shape {g_vec.shape}")

        v_links = [np.zeros(6, dtype=np.float64) for _ in range(self.n_links)]
        a_links = [np.zeros(6, dtype=np.float64) for _ in range(self.n_links)]
        f_links = [np.zeros(6, dtype=np.float64) for _ in range(self.n_links)]
        X_up = [np.eye(6, dtype=np.float64) for _ in range(self.n_links)]

        a_links[0][3:6] = -g_vec

        # Forward pass: root -> leaves
        for k in range(1, self.n_links):
            node = self._nodes[k]
            p = node.parent_idx
            R_p_k, r_p_k = self._parent_to_child_pose(node, q_arr)
            X_kp = _plucker_motion_from_parent(R_p_k, r_p_k)
            X_up[k] = X_kp

            if node.dof_index is None:
                v_k = X_kp @ v_links[p]
                a_k = X_kp @ a_links[p]
            else:
                d = node.dof_index
                v_j = node.motion_subspace * float(dq_arr[d])
                v_k = (X_kp @ v_links[p]) + v_j
                a_k = (
                    (X_kp @ a_links[p])
                    + node.motion_subspace * float(ddq_arr[d])
                    + spatial_cross_motion(v_k, v_j)
                )

            v_links[k] = v_k
            a_links[k] = a_k
            I_k = node.spatial_inertia
            f_links[k] = (I_k @ a_k) + spatial_cross_force(v_k, I_k @ v_k)

        # Backward pass: leaves -> root
        tau = np.zeros(self.n_dof, dtype=np.float64)
        for k in range(self.n_links - 1, 0, -1):
            node = self._nodes[k]
            if node.dof_index is not None:
                tau[node.dof_index] = float(np.dot(node.motion_subspace, f_links[k]))
            p = node.parent_idx
            if p > 0:
                f_links[p] = f_links[p] + (X_up[k].T @ f_links[k])
        return tau

    def mass_matrix_crba(self, q: Sequence[float]) -> np.ndarray:
        """Compute symmetric positive-definite joint-space inertia matrix ``M(q)`` via CRBA."""
        q_arr = self._validate_q(q, "q")
        X_up = [np.eye(6, dtype=np.float64) for _ in range(self.n_links)]
        I_comp = [node.spatial_inertia.copy() for node in self._nodes]

        for k in range(1, self.n_links):
            node = self._nodes[k]
            R_p_k, r_p_k = self._parent_to_child_pose(node, q_arr)
            X_up[k] = _plucker_motion_from_parent(R_p_k, r_p_k)

        M = np.zeros((self.n_dof, self.n_dof), dtype=np.float64)
        for k in range(self.n_links - 1, 0, -1):
            node = self._nodes[k]
            p = node.parent_idx
            if node.dof_index is not None:
                di = node.dof_index
                F = I_comp[k] @ node.motion_subspace
                M[di, di] = float(np.dot(node.motion_subspace, F))

                curr = k
                while self._nodes[curr].parent_idx > 0:
                    F = X_up[curr].T @ F
                    curr = self._nodes[curr].parent_idx
                    anc_node = self._nodes[curr]
                    if anc_node.dof_index is not None:
                        dj = anc_node.dof_index
                        val = float(np.dot(F, anc_node.motion_subspace))
                        M[di, dj] = val
                        M[dj, di] = val

            if p >= 0:
                X_kp = X_up[k]
                I_comp[p] = I_comp[p] + (X_kp.T @ I_comp[k] @ X_kp)

        return 0.5 * (M + M.T)

    def gravity_torques(
        self, q: Sequence[float], gravity: Sequence[float] = (0.0, 0.0, -9.81)
    ) -> np.ndarray:
        """Return generalized gravity vector ``g(q)`` via RNEA with zero velocity/acceleration."""
        zeros = np.zeros(self.n_dof, dtype=np.float64)
        return self.inverse_dynamics_rnea(q, zeros, zeros, gravity=gravity)

    def coriolis_torques(self, q: Sequence[float], qdot: Sequence[float]) -> np.ndarray:
        """Return Coriolis/centrifugal bias ``C(q, qdot) qdot`` via gravity-free RNEA."""
        zeros = np.zeros(self.n_dof, dtype=np.float64)
        return self.inverse_dynamics_rnea(q, qdot, zeros, gravity=(0.0, 0.0, 0.0))

    def potential_energy(
        self, q: Sequence[float], gravity: Sequence[float] = (0.0, 0.0, -9.81)
    ) -> float:
        """Compute gravitational potential energy ``V(q) = -sum_i m_i g^T p_{com,i}(q)``."""
        fk = self.forward_kinematics(q)
        g_vec = np.asarray(gravity, dtype=np.float64).flatten()
        V = 0.0
        for node in self._nodes:
            if node.mass > 0.0:
                T_w = fk[node.link_name]
                p_com_w = T_w[:3, 3] + T_w[:3, :3] @ node.com_xyz
                V -= node.mass * float(np.dot(g_vec, p_com_w))
        return float(V)

    def kinetic_energy(self, q: Sequence[float], qdot: Sequence[float]) -> float:
        """Compute joint-space kinetic energy ``0.5 * qdot^T M(q) qdot``."""
        dq_arr = self._validate_q(qdot, "qdot")
        M = self.mass_matrix_crba(q)
        return 0.5 * float(dq_arr @ M @ dq_arr)

    def to_euler_lagrange_dynamics(
        self, gravity: Sequence[float] = (0.0, 0.0, -9.81)
    ) -> EulerLagrangeDynamics:
        """Wire CRBA ``M_fn`` and RNEA ``C_fn`` / ``g_fn`` into ``EulerLagrangeDynamics`` with ``u_min``/``u_max``."""
        g_tuple = tuple(float(v) for v in np.asarray(gravity, dtype=np.float64).flatten())
        zeros = np.zeros(self.n_dof, dtype=np.float64)

        def M_fn(q: np.ndarray) -> np.ndarray:
            return self.mass_matrix_crba(q)

        def C_fn(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
            return self.inverse_dynamics_rnea(q, dq, zeros, gravity=(0.0, 0.0, 0.0))

        def g_fn(q: np.ndarray) -> np.ndarray:
            return self.inverse_dynamics_rnea(q, zeros, zeros, gravity=g_tuple)

        dyn = EulerLagrangeDynamics(
            n_dof=self.n_dof,
            control_dim=self.n_dof,
            M_fn=M_fn,
            C_fn=C_fn,
            g_fn=g_fn,
        )
        dyn.u_min = self.u_min.copy()
        dyn.u_max = self.u_max.copy()
        return dyn

    # ------------------------------------------------------------------
    # Whole-Body Self-Collision & Joint-Limit Barrier Builders
    # ------------------------------------------------------------------

    def _movable_tree_separation(self, idx_a: int, idx_b: int) -> int:
        """Number of movable joints along the unique kinematic tree path between two links."""
        chain_a = self._nodes[idx_a].ancestor_chain
        chain_b = self._nodes[idx_b].ancestor_chain
        lca = 0
        for a, b in zip(chain_a, chain_b):
            if a == b:
                lca = a
            else:
                break
        return (
            self._nodes[idx_a].movable_depth
            + self._nodes[idx_b].movable_depth
            - 2 * self._nodes[lca].movable_depth
        )

    def build_self_collision_barriers(
        self,
        min_link_separation: int = 2,
        margin: float = 0.02,
        relative_degree: int = 2,
        alpha: float = 1.0,
        alpha2: float = 2.0,
        broadphase_cutoff: Optional[float] = None,
        hard: bool = True,
    ) -> List[CapsulePairBarrier]:
        """Build analytical-gradient ``CapsulePairBarrier`` constraints for non-adjacent collision links."""
        sep_threshold = max(1, int(min_link_separation))
        barriers: List[CapsulePairBarrier] = []
        collision_nodes = [n for n in self._nodes if n.capsules]

        for i_pos, node_a in enumerate(collision_nodes):
            for node_b in collision_nodes[i_pos + 1 :]:
                if self._movable_tree_separation(node_a.link_idx, node_b.link_idx) < sep_threshold:
                    continue
                for cap_a in node_a.capsules:
                    for cap_b in node_b.capsules:
                        r_safe = float(cap_a.radius + cap_b.radius + float(margin))
                        idx_a = node_a.link_idx
                        idx_b = node_b.link_idx
                        p0_a, p1_a = cap_a.local_p0.copy(), cap_a.local_p1.copy()
                        p0_b, p1_b = cap_b.local_p0.copy(), cap_b.local_p1.copy()

                        def _make_fk(l_idx: int, lp0: np.ndarray, lp1: np.ndarray):
                            def _fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                                return self._link_segment_fk(
                                    self._validate_q(q, "q"), l_idx, lp0, lp1
                                )

                            return _fk

                        def _make_jac(l_idx: int, lp0: np.ndarray, lp1: np.ndarray):
                            def _jac(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                                return self._link_segment_jacobians(
                                    self._validate_q(q, "q"), l_idx, lp0, lp1
                                )

                            return _jac

                        bar = CapsulePairBarrier(
                            link_fk_fn=_make_fk(idx_a, p0_a, p1_a),
                            opponent_segment=_make_fk(idx_b, p0_b, p1_b),
                            r_safe=r_safe,
                            alpha=alpha,
                            alpha2=alpha2,
                            relative_degree=relative_degree,
                            hard=hard,
                            n_dof=self.n_dof,
                            link_jacobians_fn=_make_jac(idx_a, p0_a, p1_a),
                            opponent_jacobians_fn=_make_jac(idx_b, p0_b, p1_b),
                            broadphase_cutoff=broadphase_cutoff,
                        )
                        barriers.append(bar)
        return barriers

    def build_joint_limit_barriers(
        self,
        alpha: float = 1.0,
        alpha2: float = 2.0,
        relative_degree: int = 2,
        hard: bool = True,
    ) -> List[HalfPlaneBarrier]:
        """Build ``HalfPlaneBarrier`` constraints over state ``x = [q, qdot]`` for finite URDF joint limits."""
        barriers: List[HalfPlaneBarrier] = []
        state_dim = 2 * self.n_dof
        for i in range(self.n_dof):
            if math.isfinite(float(self.q_min[i])):
                a_low = np.zeros(state_dim, dtype=np.float64)
                a_low[i] = 1.0
                barriers.append(
                    HalfPlaneBarrier(
                        a=a_low,
                        b=float(self.q_min[i]),
                        alpha=alpha,
                        alpha2=alpha2,
                        relative_degree=relative_degree,
                        hard=hard,
                    )
                )
            if math.isfinite(float(self.q_max[i])):
                a_high = np.zeros(state_dim, dtype=np.float64)
                a_high[i] = -1.0
                barriers.append(
                    HalfPlaneBarrier(
                        a=a_high,
                        b=-float(self.q_max[i]),
                        alpha=alpha,
                        alpha2=alpha2,
                        relative_degree=relative_degree,
                        hard=hard,
                    )
                )
        return barriers
