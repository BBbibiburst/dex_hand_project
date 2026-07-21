"""Reusable MuJoCo probe primitives for tactile diagnostics."""

from __future__ import annotations

import mujoco
import numpy as np

PROBE_BODY_NAME = "tactile_probe"
PROBE_JOINT_NAME = "tactile_probe_freejoint"
PROBE_GEOM_NAME = "tactile_probe_geom"


def add_probe_to_spec(
    spec: mujoco.MjSpec,
    *,
    radius: float,
    initial_pos: np.ndarray,
    gravity_comp: bool = True,
) -> None:
    """Inject a movable spherical probe into an uncompiled ``MjSpec``."""
    if radius <= 0.0:
        raise ValueError("probe radius must be positive")

    probe = spec.worldbody.add_body()
    probe.name = PROBE_BODY_NAME
    probe.pos = np.asarray(initial_pos, dtype=np.float64).tolist()
    if gravity_comp and hasattr(probe, "gravcomp"):
        probe.gravcomp = 1.0

    joint = probe.add_joint()
    joint.name = PROBE_JOINT_NAME
    joint.type = mujoco.mjtJoint.mjJNT_FREE

    geom = probe.add_geom()
    geom.name = PROBE_GEOM_NAME
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [float(radius), 0.0, 0.0]
    geom.rgba = [1.0, 0.12, 0.08, 0.85]
    geom.mass = 0.01
    geom.condim = 3
    geom.contype = 1
    geom.conaffinity = 3
    geom.friction = [0.8, 0.01, 0.001]


def probe_joint_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    """Return qpos and qvel addresses for the injected free joint."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROBE_JOINT_NAME)
    if joint_id < 0:
        raise RuntimeError(f"Probe joint {PROBE_JOINT_NAME!r} was not compiled.")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def set_probe_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    pos: np.ndarray,
    quat: np.ndarray | None = None,
    forward: bool = False,
) -> None:
    """Set probe pose, clear its velocity, and optionally run ``mj_forward``."""
    qpos_adr, qvel_adr = probe_joint_addresses(model)
    data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(
        quat if quat is not None else [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    if forward:
        mujoco.mj_forward(model, data)
