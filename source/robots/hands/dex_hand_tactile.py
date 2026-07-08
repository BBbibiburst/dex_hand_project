# -*- coding: utf-8 -*-
"""Dex hand's tactile sensor implementation.

This is a concrete ``TactileSensorBase`` implementation, not a framework
concept. It happens to work by fitting STL skin meshes to generate a taxel
grid and injecting MuJoCo ``touch`` sensors; that choice is entirely local
to this file. A different hand implementation is free to do something
completely different (or nothing at all) and would never need to look at
this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from gymnasium import spaces
import mujoco
import numpy as np

from source.environments.assets import DEX_HAND_DIR
from source.environments.tactile_sensors import TactileSensorBase
from source.robots.hands._surface_fitting import (
    finger_segment_grid_points,
    fingertip_grid_points,
    palm_grid_points,
)


SITE_PREFIX = "taxel_"
SENSOR_PREFIX = "touch_"
TACTILE_GROUP = 4
DEFAULT_TAXEL_RADIUS = 0.0018

# (mesh_name, rows, cols, grid_fn) — the only place in the whole project that
# needs to know how many taxels each dex-hand skin patch has and how its
# surface is shaped.
_GRID_FN = {
    "segment": finger_segment_grid_points,
    "fingertip": fingertip_grid_points,
    "palm": palm_grid_points,
}


def _dex_hand_patch_layout() -> tuple[tuple[str, int, int, str], ...]:
    layout: list[tuple[str, int, int, str]] = []
    for finger_id in range(5):
        layout.append((f"skin_{finger_id}_0_p", 7, 8, "segment"))
        layout.append((f"skin_{finger_id}_1_p", 4, 8, "segment"))
        layout.append((f"skin_{finger_id}_2_p", 4, 8, "fingertip"))
    layout.append(("skin_palm_p", 7, 16, "palm"))
    return tuple(layout)


DEX_HAND_PATCH_LAYOUT = _dex_hand_patch_layout()


def site_name(mesh_name: str, row: int, col: int) -> str:
    return f"{SITE_PREFIX}{mesh_name}_r{row:02d}_c{col:02d}"


def sensor_name(mesh_name: str, row: int, col: int) -> str:
    return f"{SENSOR_PREFIX}{mesh_name}_r{row:02d}_c{col:02d}"


def _items(value):
    """Return MuJoCo MjSpec child collections across API variants."""
    return value() if callable(value) else value


def _body_by_geom_name(spec: mujoco.MjSpec) -> Dict[str, mujoco.MjsBody]:
    """Recursively map geom name -> owning body, directly from the MjSpec
    object (no XML text parsing required)."""
    result: Dict[str, mujoco.MjsBody] = {}

    def visit(body: mujoco.MjsBody) -> None:
        for geom in _items(body.geoms):
            if geom.name:
                result[geom.name] = body
        for child in _items(body.bodies):
            visit(child)

    for body in _items(spec.worldbody.bodies):
        visit(body)
    return result


def _mesh_file_map(spec: mujoco.MjSpec) -> Dict[str, str]:
    return {mesh.name: mesh.file for mesh in _items(spec.meshes) if mesh.name and mesh.file}


def _resolve_mesh_path(mesh_root, mesh_file: str):
    mesh_path = Path(mesh_file)
    return mesh_path if mesh_path.is_absolute() else Path(mesh_root) / mesh_path


class DexHandTouchSensor(TactileSensorBase):
    """Touch-sensor array generated from the dex hand's STL skin meshes."""

    def __init__(
        self,
        *,
        taxel_radius: float = DEFAULT_TAXEL_RADIUS,
        patch_layout: Sequence[tuple[str, int, int, str]] = DEX_HAND_PATCH_LAYOUT,
        mesh_dir=DEX_HAND_DIR,
    ) -> None:
        self.taxel_radius = taxel_radius
        self.patch_layout = tuple(patch_layout)
        self.mesh_dir = mesh_dir
        self.sensor_names: tuple[str, ...] = tuple(
            sensor_name(mesh_name, row, col)
            for mesh_name, rows, cols, _ in self.patch_layout
            for row in range(rows)
            for col in range(cols)
        )
        # Set by the environment before bind(); mirrors the prefix applied to
        # everything under the attached hand subtree (``attach_body`` renames
        # all named children with this prefix).
        self.name_prefix: str = ""
        self._sensor_adrs: Optional[np.ndarray] = None

    @property
    def patch_shapes(self) -> Dict[str, tuple[int, int]]:
        return {mesh_name: (rows, cols) for mesh_name, rows, cols, _ in self.patch_layout}

    def set_name_prefix(self, prefix: str) -> None:
        self.name_prefix = prefix

    # ------------------------------------------------------------------
    # Spec augmentation — runs before the hand is attached to the arm.
    # ------------------------------------------------------------------

    def augment_spec(self, hand_spec: mujoco.MjSpec) -> None:
        body_by_geom = _body_by_geom_name(hand_spec)
        mesh_files = _mesh_file_map(hand_spec)

        for mesh_name, rows, cols, kind in self.patch_layout:
            body = body_by_geom.get(mesh_name)
            if body is None:
                raise ValueError(f"Skin geom {mesh_name!r} was not found in the hand model.")

            mesh_file = mesh_files.get(mesh_name)
            if mesh_file is None:
                raise ValueError(f"Skin mesh asset {mesh_name!r} was not found in the hand model.")

            grid_fn = _GRID_FN[kind]
            mesh_points = grid_fn(_resolve_mesh_path(self.mesh_dir, mesh_file), rows, cols)
            geom = hand_spec.geom(mesh_name)
            body_points = _transform_points(mesh_points, np.asarray(geom.pos), np.asarray(geom.quat))

            for row in range(rows):
                for col in range(cols):
                    idx = row * cols + col
                    taxel_site_name = site_name(mesh_name, row, col)
                    taxel_sensor_name = sensor_name(mesh_name, row, col)

                    site = body.add_site()
                    site.name = taxel_site_name
                    site.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    site.size = [self.taxel_radius, 0.0, 0.0]
                    site.pos = body_points[idx].tolist()
                    site.rgba = [0.0, 0.8, 1.0, 0.35]
                    site.group = TACTILE_GROUP

                    sensor = hand_spec.add_sensor()
                    sensor.name = taxel_sensor_name
                    sensor.type = mujoco.mjtSensor.mjSENS_TOUCH
                    sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
                    sensor.objname = taxel_site_name

    # ------------------------------------------------------------------
    # Runtime lifecycle
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(low=0.0, high=np.inf, shape=(len(self.sensor_names),), dtype=np.float32)

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        _ = data
        sensor_ids = []
        missing = []
        for name in self.sensor_names:
            full_name = self.name_prefix + name
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, full_name)
            if sensor_id < 0:
                missing.append(full_name)
            else:
                sensor_ids.append(sensor_id)
        if missing:
            sample = missing[:8]
            raise ValueError(
                f"Missing tactile touch sensor(s): {sample}"
                f"{' ...' if len(missing) > len(sample) else ''}. "
                "Make sure add_tactile_sensors=True was passed to build_robot_spec()."
            )
        sensor_ids = np.asarray(sensor_ids, dtype=np.int32)
        self._sensor_adrs = model.sensor_adr[sensor_ids].astype(np.int32)
        if not np.all(model.sensor_dim[sensor_ids] == 1):
            raise ValueError("Dex hand tactile touch sensors must be scalar sensors.")

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = model, data, rng, options
        return {
            "tactile_sensor": "dex_hand_touch",
            "tactile_size": len(self.sensor_names),
            "tactile_patches": self.patch_shapes,
        }

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        _ = model
        if self._sensor_adrs is None:
            raise RuntimeError("DexHandTouchSensor.bind() must be called first.")
        return data.sensordata[self._sensor_adrs].astype(np.float32).copy()


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return points @ _quat_to_mat(quat).T + pos
