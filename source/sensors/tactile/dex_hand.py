# -*- coding: utf-8 -*-
"""Dex-hand tactile sensor backend.

The dex hand currently uses one oriented box-shaped MuJoCo touch site per
taxel. This keeps the tactile model stable and avoids adding extra collision
bodies to the hand.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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
    return float(np.mean(distances))


def _taxel_frame(grid: np.ndarray, row: int, col: int) -> np.ndarray:
    """Local-to-body frame: X=column tangent, Y=row tangent, Z=normal."""
    tangent_x = _normalize(_grid_difference(grid, row, col, axis=1))
    tangent_y = _grid_difference(grid, row, col, axis=0)
    tangent_y = tangent_y - np.dot(tangent_y, tangent_x) * tangent_x
    tangent_y = _normalize(tangent_y)
    normal = _normalize(np.cross(tangent_x, tangent_y))
    return np.column_stack((tangent_x, tangent_y, normal))


def _orient_frame_outward(frame: np.ndarray, point: np.ndarray, patch_center: np.ndarray) -> np.ndarray:
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
            [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)],
            [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
            [2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ _quat_to_mat(quat).T + np.asarray(pos)


def _validate_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve_axis_reflect(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if len(kernel) == 1:
        return values.copy()
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    result = np.zeros_like(values, dtype=np.float64)
    for index, weight in enumerate(kernel):
        start = index
        stop = start + values.shape[axis]
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(start, stop)
        result += weight * padded[tuple(slices)]
    return result


def _gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma)
    blurred = _convolve_axis_reflect(values, kernel, axis=0)
    return _convolve_axis_reflect(blurred, kernel, axis=1)


def _neighbor_crosstalk(values: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return values.copy()
    amount = float(np.clip(amount, 0.0, 1.0))
    rows, cols = values.shape
    result = (1.0 - amount) * values
    neighbor_sum = np.zeros_like(values, dtype=np.float64)
    neighbor_count = np.zeros_like(values, dtype=np.float64)

    if rows > 1:
        neighbor_sum[1:, :] += values[:-1, :]
        neighbor_count[1:, :] += 1.0
        neighbor_sum[:-1, :] += values[1:, :]
        neighbor_count[:-1, :] += 1.0
    if cols > 1:
        neighbor_sum[:, 1:] += values[:, :-1]
        neighbor_count[:, 1:] += 1.0
        neighbor_sum[:, :-1] += values[:, 1:]
        neighbor_count[:, :-1] += 1.0

    valid = neighbor_count > 0.0
    result[valid] += amount * neighbor_sum[valid] / neighbor_count[valid]
    result[~valid] = values[~valid]
    return result


@dataclass(frozen=True)
class TaxelPatch:
    name: str
    rows: int
    cols: int
    kind: str
    start: int
    stop: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)


@dataclass(frozen=True)
class TactileSignalProcessorConfig:
    deadzone: float = 0.0
    saturation: float = 1.0
    nonlinear_exponent: float = 1.0
    lowpass_alpha: float = 1.0
    crosstalk: float = 0.0
    gaussian_sigma: float = 0.0
    noise_std: float = 0.0
    normalize: bool = True
    seed: Optional[int] = None

    @classmethod
    def from_mapping(
        cls,
        values: Optional[Mapping[str, Any]],
    ) -> "TactileSignalProcessorConfig":
        if values is None:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown tactile signal processor option(s): {unknown}")
        return cls(**dict(values))

    def __post_init__(self) -> None:
        _validate_nonnegative("deadzone", self.deadzone)
        if self.saturation <= 0.0:
            raise ValueError("saturation must be positive.")
        if self.nonlinear_exponent <= 0.0:
            raise ValueError("nonlinear_exponent must be positive.")
        if not 0.0 <= self.lowpass_alpha <= 1.0:
            raise ValueError("lowpass_alpha must be in [0, 1].")
        if not 0.0 <= self.crosstalk <= 1.0:
            raise ValueError("crosstalk must be in [0, 1].")
        _validate_nonnegative("gaussian_sigma", self.gaussian_sigma)
        _validate_nonnegative("noise_std", self.noise_std)


class TactileSignalProcessor:
    """Post-process raw MuJoCo touch readings into tactile sensor signals."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = TactileSignalProcessorConfig.from_mapping(config)
        self._filtered: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def normalized(self) -> bool:
        return self.config.normalize

    def reset(self, size: Optional[int] = None) -> None:
        self._filtered = None if size is None else np.zeros(size, dtype=np.float64)

    def process(self, raw: np.ndarray, patches: Sequence[TaxelPatch]) -> np.ndarray:
        cfg = self.config
        values = np.asarray(raw, dtype=np.float64).copy()
        values = np.maximum(values - cfg.deadzone, 0.0)
        values = np.clip(values, 0.0, cfg.saturation)
        if cfg.nonlinear_exponent != 1.0:
            values = cfg.saturation * (values / cfg.saturation) ** cfg.nonlinear_exponent

        if cfg.lowpass_alpha < 1.0:
            if self._filtered is None or self._filtered.shape != values.shape:
                self._filtered = values.copy()
            else:
                alpha = cfg.lowpass_alpha
                self._filtered = alpha * values + (1.0 - alpha) * self._filtered
            values = self._filtered.copy()

        if cfg.crosstalk > 0.0 or cfg.gaussian_sigma > 0.0:
            values = self._process_patch_images(values, patches)

        if cfg.noise_std > 0.0:
            values += self._rng.normal(0.0, cfg.noise_std, size=values.shape)
            values = np.clip(values, 0.0, cfg.saturation)

        if cfg.normalize:
            values = values / cfg.saturation
        return values.astype(np.float32)

    def _process_patch_images(
        self,
        values: np.ndarray,
        patches: Sequence[TaxelPatch],
    ) -> np.ndarray:
        cfg = self.config
        result = values.copy()
        for patch in patches:
            image = values[patch.start:patch.stop].reshape(patch.shape)
            if cfg.crosstalk > 0.0:
                image = _neighbor_crosstalk(image, cfg.crosstalk)
            if cfg.gaussian_sigma > 0.0:
                image = _gaussian_blur(image, cfg.gaussian_sigma)
            result[patch.start:patch.stop] = image.reshape(-1)
        return np.clip(result, 0.0, cfg.saturation)

    def metadata(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "deadzone": cfg.deadzone,
            "saturation": cfg.saturation,
            "nonlinear_exponent": cfg.nonlinear_exponent,
            "lowpass_alpha": cfg.lowpass_alpha,
            "crosstalk": cfg.crosstalk,
            "gaussian_sigma": cfg.gaussian_sigma,
            "noise_std": cfg.noise_std,
            "normalize": cfg.normalize,
            "seed": cfg.seed,
        }


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
                sensor_name(mesh_name, row, col)
                for row in range(rows)
                for col in range(cols)
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

    def read_concat(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return self.read(model, data)

    def read_patches(self, model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        flat = self.read(model, data)
        return {
            patch.name: flat[patch.start:patch.stop].reshape(patch.shape)
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
        taxel_overlap: float = 1.08,
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

    def _add_taxel(self, *, hand_spec, parent_body, mesh_name, row, col, point,
                   frame, row_spacing, col_spacing) -> None:
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
