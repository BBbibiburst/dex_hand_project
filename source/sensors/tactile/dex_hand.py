# -*- coding: utf-8 -*-
"""Dex-hand tactile sensor backend.

The dex hand currently uses one oriented box-shaped MuJoCo touch site per
taxel. This keeps the tactile model stable and avoids adding extra collision
bodies to the hand.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import mujoco
import numpy as np
from gymnasium import spaces

from source.assets import DEX_HAND_DIR
from source.sensors.base import TactileSensorBase, TactileSiteRef, TactileSurfacePlotData
from source.sensors.tactile.signal_processing import (
    TactileSignalProcessor,
    TactileSignalProcessorConfig,
    TaxelPatch,
)
from source.sensors.tactile.surface_fitting import (
    DEX_HAND_PATCH_LAYOUT,
    grid_points_for_kind,
)

SITE_PREFIX = "taxel_"
SENSOR_PREFIX = "touch_"
TACTILE_GROUP = 4

DEFAULT_TACTILE_BACKEND = "simple_box"
SUPPORTED_TACTILE_BACKENDS = ("simple_box",)


def site_name(mesh_name: str, row: int, col: int) -> str:
    return f"{SITE_PREFIX}{mesh_name}_r{row:02d}_c{col:02d}"


def sensor_name(mesh_name: str, row: int, col: int) -> str:
    return f"{SENSOR_PREFIX}{mesh_name}_r{row:02d}_c{col:02d}"


def _items(value):
    return value() if callable(value) else value


def _body_by_geom_name(spec: mujoco.MjSpec) -> Dict[str, mujoco.MjsBody]:
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


def _resolve_mesh_path(mesh_root, mesh_file: str) -> Path:
    mesh_path = Path(mesh_file)
    return mesh_path if mesh_path.is_absolute() else Path(mesh_root) / mesh_path


def _normalize(vector: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _grid_difference(grid: np.ndarray, row: int, col: int, *, axis: int) -> np.ndarray:
    rows, cols, _ = grid.shape
    if axis == 0:
        if rows < 2:
            raise ValueError("Taxel grid needs at least two rows.")
        if row == 0:
            return grid[1, col] - grid[0, col]
        if row == rows - 1:
            return grid[row, col] - grid[row - 1, col]
        return grid[row + 1, col] - grid[row - 1, col]
    if axis == 1:
        if cols < 2:
            raise ValueError("Taxel grid needs at least two columns.")
        if col == 0:
            return grid[row, 1] - grid[row, 0]
        if col == cols - 1:
            return grid[row, col] - grid[row, col - 1]
        return grid[row, col + 1] - grid[row, col - 1]
    raise ValueError(f"Unsupported grid axis: {axis}")


def _neighbor_spacing(grid: np.ndarray, row: int, col: int, *, axis: int) -> float:
    rows, cols, _ = grid.shape
    distances: list[float] = []
    if axis == 0:
        if row > 0:
            distances.append(float(np.linalg.norm(grid[row, col] - grid[row - 1, col])))
        if row + 1 < rows:
            distances.append(float(np.linalg.norm(grid[row + 1, col] - grid[row, col])))
    elif axis == 1:
        if col > 0:
            distances.append(float(np.linalg.norm(grid[row, col] - grid[row, col - 1])))
        if col + 1 < cols:
            distances.append(float(np.linalg.norm(grid[row, col + 1] - grid[row, col])))
    else:
        raise ValueError(f"Unsupported grid axis: {axis}")
    if not distances:
        raise ValueError(f"Cannot estimate spacing at row={row}, col={col}, axis={axis}.")
    # A symmetric box cannot represent different distances on its two sides.
    # Using the mean makes it cross the nearer Voronoi boundary on non-uniform
    # fingertip grids, so adjacent touch sites can report exactly the same
    # contact. The nearest spacing keeps each site inside that boundary.
    return float(np.min(distances))


def _taxel_frame(grid: np.ndarray, row: int, col: int) -> np.ndarray:
    """Local-to-body frame: X=column tangent, Y=row tangent, Z=normal."""
    tangent_x = _normalize(_grid_difference(grid, row, col, axis=1))
    tangent_y = _grid_difference(grid, row, col, axis=0)
    tangent_y = tangent_y - np.dot(tangent_y, tangent_x) * tangent_x
    tangent_y = _normalize(tangent_y)
    normal = _normalize(np.cross(tangent_x, tangent_y))
    return np.column_stack((tangent_x, tangent_y, normal))


def _orient_frame_outward(
    frame: np.ndarray, point: np.ndarray, patch_center: np.ndarray
) -> np.ndarray:
    """Choose a consistent outward normal using the body origin, then patch centre.

    Skin mesh points are expressed in the owning body's coordinates.  The body
    origin is normally inside the link.  If it is inconclusive, the direction
    from the patch centre to the current point is used instead.
    """
    result = np.asarray(frame, dtype=np.float64).copy()
    reference = np.asarray(point, dtype=np.float64)
    if np.linalg.norm(reference) < 1e-8:
        reference = np.asarray(point, dtype=np.float64) - np.asarray(patch_center, dtype=np.float64)
    if np.linalg.norm(reference) >= 1e-8 and np.dot(result[:, 2], reference) < 0.0:
        # Flip Y and Z together to preserve a right-handed frame.
        result[:, 1] *= -1.0
        result[:, 2] *= -1.0
    return result


def _rotation_matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix must have shape (3, 3), got {matrix.shape}.")
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, matrix.reshape(9))
    return quat


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ _quat_to_mat(quat).T + np.asarray(pos)


class DexHandTactileSensorBase(TactileSensorBase):
    """Shared geometry, naming, binding, and readout for dex-hand backends."""

    backend_name = "base"

    def __init__(
        self,
        *,
        patch_layout: Sequence[tuple[str, int, int, str]] = DEX_HAND_PATCH_LAYOUT,
        mesh_dir=DEX_HAND_DIR,
        image_force_max: float = 5.0,
        signal_processor: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if image_force_max <= 0.0:
            raise ValueError("image_force_max must be positive.")
        self.patch_layout = tuple(patch_layout)
        self.mesh_dir = Path(mesh_dir)
        self.image_force_max = float(image_force_max)
        self.signal_processor = TactileSignalProcessor(signal_processor)
        self.name_prefix = ""
        self._sensor_adrs: Optional[np.ndarray] = None

        patches: list[TaxelPatch] = []
        names: list[str] = []
        offset = 0
        for mesh_name, rows, cols, kind in self.patch_layout:
            count = rows * cols
            patches.append(TaxelPatch(mesh_name, rows, cols, kind, offset, offset + count))
            names.extend(
                sensor_name(mesh_name, row, col) for row in range(rows) for col in range(cols)
            )
            offset += count
        self.patches = tuple(patches)
        self.sensor_names = tuple(names)

    @property
    def patch_shapes(self) -> Dict[str, tuple[int, int]]:
        return {patch.name: patch.shape for patch in self.patches}

    @property
    def total_taxels(self) -> int:
        return len(self.sensor_names)

    @property
    def observation_space(self) -> spaces.Space:
        high = 1.0 if self.signal_processor.normalized else np.inf
        return spaces.Box(low=0.0, high=high, shape=(self.total_taxels,), dtype=np.float32)

    def set_name_prefix(self, prefix: str) -> None:
        self.name_prefix = prefix

    def visualization_sites(self) -> tuple[TactileSiteRef, ...]:
        return tuple(
            TactileSiteRef(
                site_name(patch.name, row, col),
                patch.name,
                patch.start + row * patch.cols + col,
            )
            for patch in self.patches
            for row in range(patch.rows)
            for col in range(patch.cols)
        )

    def surface_patch_names(self) -> tuple[str, ...]:
        return tuple(patch.name for patch in self.patches)

    def default_surface_patch_names(self) -> tuple[str, ...]:
        return ("skin_0_0_p", "skin_0_2_p", "skin_palm_p")

    def surface_plot_data(self, patch_name: str) -> TactileSurfacePlotData:
        from source.sensors.tactile.surface_fitting import (
            GRID_POINT_FUNCTIONS,
            finger_segment_fit_surface,
            patch_fingertip_ellipsoid_plot_data,
            patch_mesh_uv_plot_data,
            patch_plot_data,
        )

        try:
            patch = next(item for item in self.patches if item.name == patch_name)
        except StopIteration as exc:
            raise ValueError(
                f"Unknown tactile patch {patch_name!r}; known: {list(self.surface_patch_names())}."
            ) from exc
        filename = f"{patch.name}.STL"
        direct_path = self.mesh_dir / filename
        stl_path = direct_path if direct_path.is_file() else self.mesh_dir / "meshes" / filename
        if patch.kind == "mesh-uv":
            data = patch_mesh_uv_plot_data(stl_path, patch.name, patch.rows, patch.cols)
        elif patch.kind == "fingertip-ellipsoid":
            data = patch_fingertip_ellipsoid_plot_data(stl_path, patch.name, patch.rows, patch.cols)
        else:
            data = patch_plot_data(
                stl_path,
                patch.name,
                patch.rows,
                patch.cols,
                GRID_POINT_FUNCTIONS[patch.kind],
                finger_segment_fit_surface,
            )
        return TactileSurfacePlotData(
            patch.name,
            patch.rows,
            patch.cols,
            patch.kind,
            data.samples,
            data.triangles,
            tuple(data.fit_surfaces),
            f"{patch.name}: {patch.rows} x {patch.cols} ({patch.kind})",
        )

    def augment_spec(self, hand_spec: mujoco.MjSpec) -> None:
        body_by_geom = _body_by_geom_name(hand_spec)
        mesh_files = _mesh_file_map(hand_spec)

        for patch in self.patches:
            body = body_by_geom.get(patch.name)
            if body is None:
                raise ValueError(f"Skin geom {patch.name!r} was not found in the hand model.")
            mesh_file = mesh_files.get(patch.name)
            if mesh_file is None:
                raise ValueError(f"Skin mesh asset {patch.name!r} was not found in the hand model.")

            mesh_points = grid_points_for_kind(
                patch.kind,
                _resolve_mesh_path(self.mesh_dir, mesh_file),
                patch.rows,
                patch.cols,
            )
            geom = hand_spec.geom(patch.name)
            body_points = _transform_points(
                mesh_points,
                np.asarray(geom.pos, dtype=np.float64),
                np.asarray(geom.quat, dtype=np.float64),
            )
            expected = (patch.rows * patch.cols, 3)
            if body_points.shape != expected:
                raise ValueError(
                    f"Generated grid for {patch.name!r} has shape {body_points.shape}, "
                    f"expected {expected}."
                )

            grid = body_points.reshape(patch.rows, patch.cols, 3)
            patch_center = grid.mean(axis=(0, 1))
            for row in range(patch.rows):
                for col in range(patch.cols):
                    frame = _orient_frame_outward(
                        _taxel_frame(grid, row, col), grid[row, col], patch_center
                    )
                    self._add_taxel(
                        hand_spec=hand_spec,
                        parent_body=body,
                        mesh_name=patch.name,
                        row=row,
                        col=col,
                        point=grid[row, col],
                        frame=frame,
                        row_spacing=_neighbor_spacing(grid, row, col, axis=0),
                        col_spacing=_neighbor_spacing(grid, row, col, axis=1),
                    )

    @abstractmethod
    def _add_taxel(
        self,
        *,
        hand_spec: mujoco.MjSpec,
        parent_body: mujoco.MjsBody,
        mesh_name: str,
        row: int,
        col: int,
        point: np.ndarray,
        frame: np.ndarray,
        row_spacing: float,
        col_spacing: float,
    ) -> None:
        raise NotImplementedError

    def _add_touch_sensor(
        self,
        hand_spec: mujoco.MjSpec,
        *,
        mesh_name: str,
        row: int,
        col: int,
        site: str,
        cutoff: float = 0.0,
        noise: float = 0.0,
    ) -> None:
        sensor = hand_spec.add_sensor()
        sensor.name = sensor_name(mesh_name, row, col)
        sensor.type = mujoco.mjtSensor.mjSENS_TOUCH
        sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
        sensor.objname = site
        sensor.cutoff = float(cutoff)
        sensor.noise = float(noise)

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        _ = data
        sensor_ids: list[int] = []
        missing: list[str] = []
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
                f"Missing tactile sensor(s): {sample}"
                f"{' ...' if len(missing) > len(sample) else ''}."
            )
        ids = np.asarray(sensor_ids, dtype=np.int32)
        if not np.all(model.sensor_dim[ids] == 1):
            raise ValueError("Dex-hand tactile sensors must all be scalar.")
        # Store sensordata addresses, never raw sensor IDs.
        self._sensor_adrs = model.sensor_adr[ids].astype(np.int32)

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = model, data, rng, options
        self.signal_processor.reset(self.total_taxels)
        return {
            "tactile_sensor": "dex_hand",
            "tactile_backend": self.backend_name,
            "tactile_size": self.total_taxels,
            "tactile_patches": self.patch_shapes,
            "tactile_signal_processor": self.signal_processor.metadata(),
        }

    def read_raw(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        _ = model
        if self._sensor_adrs is None:
            raise RuntimeError(f"{type(self).__name__}.bind() must be called first.")
        return data.sensordata[self._sensor_adrs].astype(np.float32).copy()

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return self.signal_processor.process(self.read_raw(model, data), self.patches)

    def diagnostic_values(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return self.read_raw(model, data)

    def read_concat(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return self.read(model, data)

    def read_patches(self, model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        return self.patches_from_values(self.read(model, data))

    def patches_from_values(self, values: Any) -> Dict[str, np.ndarray]:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        return {
            patch.name: flat[patch.start : patch.stop].reshape(patch.shape)
            for patch in self.patches
        }

    def read_images(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        force_max: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        maximum = self.image_force_max if force_max is None else float(force_max)
        if maximum <= 0.0:
            raise ValueError("force_max must be positive.")
        return {
            name: np.rint(np.clip(values, 0.0, maximum) * (255.0 / maximum)).astype(np.uint8)
            for name, values in self.read_patches(model, data).items()
        }

    def read_image(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        patch_name: Optional[str] = None,
        force_max: Optional[float] = None,
    ) -> Dict[str, np.ndarray] | np.ndarray:
        images = self.read_images(model, data, force_max=force_max)
        if patch_name is None:
            return images
        try:
            return images[patch_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown tactile patch {patch_name!r}; known: {sorted(images)}"
            ) from exc

    def metadata(self) -> Dict[str, Any]:
        patches = {
            patch.name: {
                "rows": patch.rows,
                "cols": patch.cols,
                "kind": patch.kind,
                "flat_slice": (patch.start, patch.stop),
            }
            for patch in self.patches
        }
        return {
            "backend": self.backend_name,
            "patches": patches,
            "signal_processor": self.signal_processor.metadata(),
        }


class SimpleBoxTactileSensor(DexHandTactileSensorBase):
    """Fast backend: one oriented box touch site per taxel."""

    backend_name = "simple_box"

    def __init__(
        self,
        *,
        taxel_half_depth: float = 0.0015,
        taxel_overlap: float = 1.0,
        taxel_min_half_size: float = 0.0005,
        site_rgba: Sequence[float] = (0.0, 0.8, 1.0, 0.25),
        site_group: int = TACTILE_GROUP,
        sensor_noise: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if taxel_half_depth <= 0.0:
            raise ValueError("taxel_half_depth must be positive.")
        if taxel_overlap <= 0.0:
            raise ValueError("taxel_overlap must be positive.")
        if taxel_min_half_size <= 0.0:
            raise ValueError("taxel_min_half_size must be positive.")
        self.taxel_half_depth = float(taxel_half_depth)
        self.taxel_overlap = float(taxel_overlap)
        self.taxel_min_half_size = float(taxel_min_half_size)
        self.site_rgba = tuple(float(v) for v in site_rgba)
        self.site_group = int(site_group)
        self.sensor_noise = float(sensor_noise)

    def _add_taxel(
        self, *, hand_spec, parent_body, mesh_name, row, col, point, frame, row_spacing, col_spacing
    ) -> None:
        half_x = max(self.taxel_min_half_size, 0.5 * col_spacing * self.taxel_overlap)
        half_y = max(self.taxel_min_half_size, 0.5 * row_spacing * self.taxel_overlap)
        name = site_name(mesh_name, row, col)

        site = parent_body.add_site()
        site.name = name
        site.type = mujoco.mjtGeom.mjGEOM_BOX
        site.size = [half_x, half_y, self.taxel_half_depth]
        site.pos = np.asarray(point, dtype=np.float64).tolist()
        site.quat = _rotation_matrix_to_quat(frame).tolist()
        site.group = self.site_group
        site.rgba = list(self.site_rgba)
        self._add_touch_sensor(
            hand_spec,
            mesh_name=mesh_name,
            row=row,
            col=col,
            site=name,
            noise=self.sensor_noise,
        )


def create_dex_hand_tactile_sensor(
    backend: str = DEFAULT_TACTILE_BACKEND,
    **kwargs: Any,
) -> DexHandTactileSensorBase:
    """Create one dex-hand tactile backend through the unified interface."""
    normalized = str(backend).strip().lower()
    mapping = {
        "simple_box": SimpleBoxTactileSensor,
    }
    try:
        sensor_type = mapping[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dex-hand tactile backend {backend!r}; "
            f"supported backends: {', '.join(mapping)}."
        ) from exc
    return sensor_type(**kwargs)


# Backward compatibility: the previous class name now means the fast default.
DexHandTouchSensor = SimpleBoxTactileSensor


__all__ = [
    "DEFAULT_TACTILE_BACKEND",
    "SUPPORTED_TACTILE_BACKENDS",
    "DEX_HAND_PATCH_LAYOUT",
    "DexHandTactileSensorBase",
    "DexHandTouchSensor",
    "SimpleBoxTactileSensor",
    "TactileSignalProcessor",
    "TactileSignalProcessorConfig",
    "TaxelPatch",
    "create_dex_hand_tactile_sensor",
    "site_name",
    "sensor_name",
]
