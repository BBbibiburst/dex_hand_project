"""Closed-chain end-effector geometry extraction and shape sampling."""

from __future__ import annotations

import mujoco
import numpy as np

from source.grasping.mujoco_safety import capture_mujoco_warnings, checked_mj_step
from source.grasping.search.common import progress
from source.grasping.search.types import Device, Surface

def mesh_vertices(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)


def mesh_faces(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    start, count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    return np.asarray(model.mesh_face[start : start + count], dtype=np.int64)


def geom_label(device: Device, name: str) -> int:
    if device.name == "pika_gripper":
        if "left_link" in name:
            return 0
        if "right_link" in name:
            return 1
        return 2
    if "skin_palm" in name:
        return 5
    for finger in range(5):
        if f"skin_{finger}_" in name:
            return finger
    return 6


def surface_for(device: Device, fractions: np.ndarray, *, seed: int) -> Surface:
    model = mujoco.MjModel.from_xml_path(str(device.xml))
    data = mujoco.MjData(model)
    values = np.empty(len(device.actuators))
    for index, (name, fraction) in enumerate(zip(device.actuators, fractions, strict=True)):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        low, high = model.actuator_ctrlrange[actuator]
        if device.name == "pika_gripper":
            value = np.clip(low + 0.05 * float(fraction), low, high)
        else:
            value = low + float(fraction) * (high - low)
        data.ctrl[actuator] = values[index] = value
    with capture_mujoco_warnings() as warnings:
        for step in range(600):
            checked_mj_step(
                model,
                data,
                warnings,
                phase=f"{device.name} surface solve",
                step=step + 1,
            )

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, device.root_body)
    root_pos = data.xpos[root].copy()
    root_rot = data.xmat[root].reshape(3, 3).copy()
    rng = np.random.default_rng(seed)
    point_groups, label_groups, meshes = [], [], []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or not name:
            continue
        if device.name == "pika_gripper" and not name.endswith("_collision"):
            continue
        mesh_id = int(model.geom_dataid[geom])
        vertices = mesh_vertices(model, mesh_id)
        rotation = data.geom_xmat[geom].reshape(3, 3)
        local = (vertices @ rotation.T + data.geom_xpos[geom] - root_pos) @ root_rot
        faces = mesh_faces(model, mesh_id)
        meshes.append((local, faces))
        selected = local
        if len(selected) > 350:
            selected = selected[rng.choice(len(selected), 350, replace=False)]
        point_groups.append(selected)
        label_groups.append(np.full(len(selected), geom_label(device, name), dtype=int))
    points = np.concatenate(point_groups)
    labels = np.concatenate(label_groups)
    if device.name == "pika_gripper":
        midpoint = 0.5 * (points[labels == 0].mean(0) + points[labels == 1].mean(0))
    else:
        finger = np.concatenate([points[labels == i] for i in range(4)]).mean(0)
        midpoint = 0.5 * (finger + points[labels == 4].mean(0))
    return Surface(points, labels, meshes, values, fractions.copy(), midpoint)


def fraction_candidates(device: Device, count: int) -> list[np.ndarray]:
    progress = np.linspace(0.12, 0.92, count)
    if device.name == "pika_gripper":
        return [np.asarray([value]) for value in progress[::-1]]
    candidates = []
    for value in progress:
        candidates.append(np.asarray([value, value, value, value, 1.0, value]))
        candidates.append(np.asarray([value, value, 0.5 * value, 0.5 * value, 1.0, value]))
    return candidates


def _open_fractions(device: Device) -> np.ndarray:
    if device.name == "pika_gripper":
        return np.ones(1, dtype=np.float64)
    return np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
