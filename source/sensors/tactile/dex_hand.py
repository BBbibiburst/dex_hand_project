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
from source.sensors.tactile._surface_fitting import (
    DEX_HAND_PATCH_LAYOUT,
    grid_points_for_kind,
)

SITE_PREFIX = "taxel_"
SENSOR_PREFIX = "touch_"
TACTILE_GROUP = 4

# 1. 修改参数
DEFAULT_TAXEL_HALF_DEPTH = 0.0015
DEFAULT_TAXEL_OVERLAP = 1.08
DEFAULT_TAXEL_MIN_HALF_SIZE = 0.0005


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

    # 2. 修改 DexHandTouchSensor.__init__
    def __init__(
        self,
        *,
        taxel_half_depth: float = DEFAULT_TAXEL_HALF_DEPTH,
        taxel_overlap: float = DEFAULT_TAXEL_OVERLAP,
        taxel_min_half_size: float = DEFAULT_TAXEL_MIN_HALF_SIZE,
        patch_layout: Sequence[tuple[str, int, int, str]] = DEX_HAND_PATCH_LAYOUT,
        mesh_dir=DEX_HAND_DIR,
    ) -> None:
        if taxel_half_depth <= 0.0:
            raise ValueError("taxel_half_depth must be positive.")
        if taxel_overlap <= 0.0:
            raise ValueError("taxel_overlap must be positive.")
        if taxel_min_half_size <= 0.0:
            raise ValueError("taxel_min_half_size must be positive.")
            
        self.taxel_half_depth = float(taxel_half_depth)
        self.taxel_overlap = float(taxel_overlap)
        self.taxel_min_half_size = float(taxel_min_half_size)
        self.patch_layout = tuple(patch_layout)
        self.mesh_dir = mesh_dir
        self.sensor_names: tuple[str, ...] = tuple(
            sensor_name(mesh_name, row, col)
            for mesh_name, rows, cols, _ in self.patch_layout
            for row in range(rows)
            for col in range(cols)
        )
        self.name_prefix: str = ""
        self._sensor_adrs: Optional[np.ndarray] = None

    @property
    def patch_shapes(self) -> Dict[str, tuple[int, int]]:
        return {mesh_name: (rows, cols) for mesh_name, rows, cols, _ in self.patch_layout}

    def set_name_prefix(self, prefix: str) -> None:
        self.name_prefix = prefix

    # ------------------------------------------------------------------
    # Spec augmentation runs before the hand is attached to the arm.
    # ------------------------------------------------------------------

    # 3. 添加网格局部坐标系计算辅助函数
    def _normalize(self, vector: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm < eps:
            raise ValueError("Cannot normalize a near-zero vector.")
        return vector / norm

    def _grid_difference(
        self, grid: np.ndarray, row: int, col: int, *, axis: int,
    ) -> np.ndarray:
        """Estimate a tangent using centered or one-sided finite differences."""
        rows, cols, _ = grid.shape
        if axis == 0: # Row direction.
            if row == 0:
                return grid[row + 1, col] - grid[row, col]
            if row == rows - 1:
                return grid[row, col] - grid[row - 1, col]
            return grid[row + 1, col] - grid[row - 1, col]
        if axis == 1: # Column direction.
            if col == 0:
                return grid[row, col + 1] - grid[row, col]
            if col == cols - 1:
                return grid[row, col] - grid[row, col - 1]
            return grid[row, col + 1] - grid[row, col - 1]
        raise ValueError(f"Unsupported grid axis: {axis}")

    def _neighbor_spacing(
        self, grid: np.ndarray, row: int, col: int, *, axis: int,
    ) -> float:
        """Estimate center-to-center spacing around one taxel."""
        rows, cols, _ = grid.shape
        distances: list[float] = []
        if axis == 0:
            if row > 0:
                distances.append(
                    float(np.linalg.norm(grid[row, col] - grid[row - 1, col]))
                )
            if row + 1 < rows:
                distances.append(
                    float(np.linalg.norm(grid[row + 1, col] - grid[row, col]))
                )
        elif axis == 1:
            if col > 0:
                distances.append(
                    float(np.linalg.norm(grid[row, col] - grid[row, col - 1]))
                )
            if col + 1 < cols:
                distances.append(
                    float(np.linalg.norm(grid[row, col + 1] - grid[row, col]))
                )
        else:
            raise ValueError(f"Unsupported grid axis: {axis}")
        if not distances:
            raise ValueError(
                f"Cannot estimate taxel spacing at row={row}, col={col}, axis={axis}."
            )
        return float(np.mean(distances))

    def _taxel_frame(
        self, grid: np.ndarray, row: int, col: int,
    ) -> np.ndarray:
        """Return a local-to-body rotation matrix for one box taxel.
        Local X: grid column direction
        Local Y: grid row direction
        Local Z: surface normal
        """
        tangent_x = self._normalize(self._grid_difference(grid, row, col, axis=1))
        tangent_y_raw = self._grid_difference(grid, row, col, axis=0)
        # Gram-Schmidt: remove the tangent_x component so the box axes
        # remain orthogonal even if the fitted grid is skewed.
        tangent_y_raw = tangent_y_raw - np.dot(
            tangent_y_raw, tangent_x
        ) * tangent_x
        tangent_y = self._normalize(tangent_y_raw)
        normal = self._normalize(np.cross(tangent_x, tangent_y))
        # Columns are the box's local axes expressed in body coordinates.
        return np.column_stack((tangent_x, tangent_y, normal))

    def _rotation_matrix_to_quat(self, matrix: np.ndarray) -> np.ndarray:
        """Convert a 3x3 local-to-body matrix to MuJoCo wxyz quaternion."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Rotation matrix must have shape (3, 3), got {matrix.shape}."
            )
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, matrix.reshape(9))
        return quat

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

            mesh_points = grid_points_for_kind(
                kind,
                _resolve_mesh_path(self.mesh_dir, mesh_file),
                rows,
                cols,
            )
            geom = hand_spec.geom(mesh_name)
            body_points = _transform_points(mesh_points, np.asarray(geom.pos), np.asarray(geom.quat))
            
            # Restore the regular row-column structure.
            point_grid = body_points.reshape(rows, cols, 3)

            for row in range(rows):
                for col in range(cols):
                    point = point_grid[row, col]
                    taxel_site_name = site_name(mesh_name, row, col)
                    taxel_sensor_name = sensor_name(mesh_name, row, col)

                    # Local dimensions are determined from neighboring
                    # center-to-center spacing.
                    col_spacing = self._neighbor_spacing(
                        point_grid, row, col, axis=1,
                    )
                    row_spacing = self._neighbor_spacing(
                        point_grid, row, col, axis=0,
                    )
                    half_x = max(
                        self.taxel_min_half_size,
                        0.5 * col_spacing * self.taxel_overlap,
                    )
                    half_y = max(
                        self.taxel_min_half_size,
                        0.5 * row_spacing * self.taxel_overlap,
                    )
                    
                    rotation = self._taxel_frame(point_grid, row, col)
                    quat = self._rotation_matrix_to_quat(rotation)

                    site = body.add_site()
                    site.name = taxel_site_name
                    site.type = mujoco.mjtGeom.mjGEOM_BOX
                    # MuJoCo box size uses half-lengths.
                    site.size = [
                        half_x,
                        half_y,
                        self.taxel_half_depth,
                    ]
                    site.pos = point.tolist()
                    site.quat = quat.tolist()
                    site.rgba = [0.0, 0.8, 1.0, 0.25]
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