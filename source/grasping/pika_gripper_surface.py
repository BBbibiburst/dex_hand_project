"""Extract a posed Pika parallel-gripper surface for geometric grasp search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from source.assets import PIKA_GRIPPER_XML_PATH


@dataclass(frozen=True)
class PosedPikaGripperSurface:
    points: np.ndarray
    labels: np.ndarray
    contact_centers: np.ndarray
    actuator_values: np.ndarray


def _mesh_vertices(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)


def load_posed_pika_gripper_surface(
    *,
    opening_fraction: float = 1.0,
    max_points_per_geom: int = 700,
    seed: int = 0,
    xml_path: str | Path = PIKA_GRIPPER_XML_PATH,
) -> PosedPikaGripperSurface:
    """Return collision meshes in gripper-root coordinates.

    ``opening_fraction`` follows the controller convention: zero is closed and
    one is the maximum 0.1 m jaw opening.
    """
    if not 0.0 <= opening_fraction <= 1.0:
        raise ValueError("opening_fraction must be in [0, 1].")
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    data = mujoco.MjData(model)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_position")
    low, high = model.actuator_ctrlrange[actuator_id]
    joint_target = low + 0.5 * (0.1 * float(opening_fraction))
    data.ctrl[actuator_id] = np.clip(joint_target, low, high)
    for _ in range(500):
        mujoco.mj_step(model, data)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base_link")
    root_position = data.xpos[root_id].copy()
    root_rotation = data.xmat[root_id].reshape(3, 3).copy()
    rng = np.random.default_rng(seed)
    point_groups = []
    label_groups = []
    centers = np.empty((2, 3), dtype=np.float64)
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if (
            not name.endswith("_collision")
            or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        label = 2
        if "left_link" in name:
            label = 0
        elif "right_link" in name:
            label = 1
        vertices = _mesh_vertices(model, int(model.geom_dataid[geom_id]))
        if vertices.shape[0] > max_points_per_geom:
            vertices = vertices[rng.choice(vertices.shape[0], max_points_per_geom, replace=False)]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world = vertices @ rotation.T + data.geom_xpos[geom_id]
        local = (world - root_position) @ root_rotation
        point_groups.append(local)
        label_groups.append(np.full(len(local), label, dtype=np.int64))
        if label < 2:
            # The distal half is the useful opposing contact region.
            distal = local[local[:, 0] >= np.quantile(local[:, 0], 0.55)]
            centers[label] = distal.mean(axis=0)
    if len(point_groups) != 3:
        raise RuntimeError("Failed to extract all Pika collision meshes.")
    return PosedPikaGripperSurface(
        points=np.concatenate(point_groups),
        labels=np.concatenate(label_groups),
        contact_centers=centers,
        actuator_values=np.asarray([joint_target], dtype=np.float64),
    )
