"""Extract a posed Dex Hand STL surface for geometry-only grasp search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from source.assets import DEX_HAND_XML_PATH


@dataclass(frozen=True)
class PosedDexHandSurface:
    points: np.ndarray
    labels: np.ndarray
    fingertip_centers: np.ndarray
    actuator_values: np.ndarray


_ACTUATORS = (
    "act_push_0_j",
    "act_push_1_j",
    "act_push_2_j",
    "act_push_3_j",
    "thumb_rotate_act_push_j",
    "thumb_grasp_act_push_j",
)


def _mesh_vertices(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)


def _geom_label(name: str) -> int:
    if "skin_palm" in name:
        return 5
    for finger in range(5):
        if f"skin_{finger}_" in name:
            return finger
    # Non-skin meshes still belong to the physical hand and must participate
    # in object penetration checks.  They are not eligible contact pads.
    return 6


def load_posed_dex_hand_surface(
    *,
    actuator_fractions: np.ndarray | None = None,
    max_points_per_geom: int = 350,
    seed: int = 0,
    xml_path: str | Path = DEX_HAND_XML_PATH,
) -> PosedDexHandSurface:
    """Resolve the closed linkage once and return STL vertices in hand-root space.

    MuJoCo is used only as the MJCF closed-chain kinematics resolver here. The
    returned arrays are plain NumPy data; the subsequent search has no simulator.
    """
    if max_points_per_geom <= 0:
        raise ValueError("max_points_per_geom must be positive.")
    fractions = (
        np.asarray([0.75, 0.75, 0.75, 0.75, 1.0, 0.75], dtype=np.float64)
        if actuator_fractions is None
        else np.asarray(actuator_fractions, dtype=np.float64)
    )
    if fractions.shape != (6,) or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("actuator_fractions must contain six values in [0, 1].")

    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    data = mujoco.MjData(model)
    values = np.empty(6, dtype=np.float64)
    for index, name in enumerate(_ACTUATORS):
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        low, high = model.actuator_ctrlrange[actuator_id]
        values[index] = low + fractions[index] * (high - low)
        data.ctrl[actuator_id] = values[index]

    # Let position actuators and equality constraints resolve the passive links.
    for _ in range(600):
        mujoco.mj_step(model, data)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
    root_position = data.xpos[root_id].copy()
    root_rotation = data.xmat[root_id].reshape(3, 3).copy()
    rng = np.random.default_rng(seed)
    point_groups = []
    label_groups = []
    fingertip_centers = np.full((5, 3), np.nan, dtype=np.float64)
    fingertip_points: dict[int, list[np.ndarray]] = {index: [] for index in range(5)}

    for geom_id in range(model.ngeom):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        )
        # The MJCF contains a named physical mesh followed by an unnamed visual
        # duplicate for many parts.  Keep every named mesh once: this includes
        # the palm, base, phalanges and linkage parts without double-sampling.
        if not name or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        label = _geom_label(name)
        vertices = _mesh_vertices(model, int(model.geom_dataid[geom_id]))
        if vertices.shape[0] > max_points_per_geom:
            selected = rng.choice(
                vertices.shape[0], max_points_per_geom, replace=False
            )
            vertices = vertices[selected]
        geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world = vertices @ geom_rotation.T + data.geom_xpos[geom_id]
        local = (world - root_position) @ root_rotation
        point_groups.append(local)
        label_groups.append(np.full(local.shape[0], label, dtype=np.int64))
        for finger in range(5):
            if f"skin_{finger}_2" in name:
                fingertip_points[finger].append(local)

    for finger, groups in fingertip_points.items():
        if groups:
            fingertip_centers[finger] = np.concatenate(groups).mean(axis=0)
    if not point_groups or np.isnan(fingertip_centers).any():
        raise RuntimeError("Failed to extract Dex Hand skin STL surfaces.")
    return PosedDexHandSurface(
        points=np.concatenate(point_groups),
        labels=np.concatenate(label_groups),
        fingertip_centers=fingertip_centers,
        actuator_values=values,
    )
