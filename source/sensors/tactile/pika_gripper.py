# -*- coding: utf-8 -*-
"""32 x 58 film tactile arrays for the Pika parallel gripper.

The physical films cover the inward faces of the two jaws.  Their electrical
layout remains rectangular even though the jaw mesh clips the two corners at
the tip.  Consequently every matrix cell gets a MuJoCo touch site; cells over
the clipped corners naturally remain zero because there is no jaw collision
geometry beneath them.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import mujoco
import numpy as np
from gymnasium import spaces

from source.assets import PIKA_GRIPPER_DIR
from source.sensors.base import TactileSensorBase, TactileSiteRef, TactileSurfacePlotData
from source.sensors.tactile.fitting.stl import read_stl_triangles
from source.sensors.tactile.signal_processing import TactileSignalProcessor, TaxelPatch

ROWS = 32
COLS = 58
SIDES = ("left", "right")
TACTILE_GROUP = 4

# Measured from gripper_left_link.STL / gripper_right_link.STL.  X runs from
# jaw root to tip and Z spans the film width.  The inward gripping face is a
# very slightly tilted plane; the larger absolute Y coordinate is the surface
# facing the grasp gap (the smaller Y planes belong to internal jaw features).
FACE_X_REAR = 0.040911
FACE_X_TIP = 0.108539
X_MIN = FACE_X_REAR
X_MAX = FACE_X_TIP
Z_MIN = -0.024543
Z_MAX = 0.012971
INNER_Y_REAR = 0.040896
INNER_Y_TIP = 0.041497


def site_name(side: str, row: int, col: int) -> str:
    return f"taxel_pika_{side}_r{row:02d}_c{col:02d}"


def sensor_name(side: str, row: int, col: int) -> str:
    return f"touch_pika_{side}_r{row:02d}_c{col:02d}"


def _inward_y_magnitude(x: float) -> float:
    fraction = np.clip((x - FACE_X_REAR) / (FACE_X_TIP - FACE_X_REAR), 0.0, 1.0)
    return float(INNER_Y_REAR + fraction * (INNER_Y_TIP - INNER_Y_REAR))


def _inward_y_slope(x: float) -> float:
    del x
    return (INNER_Y_TIP - INNER_Y_REAR) / (FACE_X_TIP - FACE_X_REAR)


def _frame(side: str, x: float) -> np.ndarray:
    """Return a local-to-body frame with +Z pointing into the grasp gap."""
    side_sign = -1.0 if side == "left" else 1.0
    tangent_col = np.asarray([1.0, side_sign * _inward_y_slope(x), 0.0])
    tangent_col /= np.linalg.norm(tangent_col)
    normal = np.asarray([-_inward_y_slope(x), side_sign, 0.0])
    normal /= np.linalg.norm(normal)
    tangent_row = np.cross(normal, tangent_col)
    return np.column_stack((tangent_col, tangent_row, normal))


def _quat(matrix: np.ndarray) -> list[float]:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat.tolist()


class PikaGripperTactileSensor(TactileSensorBase):
    """Two dense film arrays returned as ``(left/right, 32, 58)``."""

    backend_name = "simple_box"

    def __init__(
        self,
        *,
        taxel_half_depth: float = 0.0015,
        surface_embed: float = 0.0002,
        taxel_overlap: float = 1.05,
        site_rgba: Sequence[float] = (1.0, 0.45, 0.0, 0.18),
        site_group: int = TACTILE_GROUP,
        sensor_noise: float = 0.0,
        signal_processor: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if taxel_half_depth <= 0.0:
            raise ValueError("taxel_half_depth must be positive.")
        if not 0.0 <= surface_embed < taxel_half_depth:
            raise ValueError("surface_embed must be non-negative and smaller than taxel_half_depth.")
        if taxel_overlap <= 0.0:
            raise ValueError("taxel_overlap must be positive.")
        self.taxel_half_depth = float(taxel_half_depth)
        self.surface_embed = float(surface_embed)
        self.taxel_overlap = float(taxel_overlap)
        self.site_rgba = [float(value) for value in site_rgba]
        self.site_group = int(site_group)
        self.sensor_noise = float(sensor_noise)
        self.signal_processor = TactileSignalProcessor(signal_processor)
        self.name_prefix = ""
        self.patches = tuple(
            TaxelPatch(side, ROWS, COLS, "planar_clipped_tip", index * ROWS * COLS,
                       (index + 1) * ROWS * COLS)
            for index, side in enumerate(SIDES)
        )
        self.sensor_names = tuple(
            sensor_name(side, row, col)
            for side in SIDES
            for row in range(ROWS)
            for col in range(COLS)
        )
        self._sensor_adrs: Optional[np.ndarray] = None

    @property
    def total_taxels(self) -> int:
        return len(self.sensor_names)

    @property
    def observation_space(self) -> spaces.Space:
        high = 1.0 if self.signal_processor.normalized else np.inf
        return spaces.Box(low=0.0, high=high, shape=(2, ROWS, COLS), dtype=np.float32)

    def set_name_prefix(self, prefix: str) -> None:
        self.name_prefix = prefix

    def visualization_sites(self) -> tuple[TactileSiteRef, ...]:
        return tuple(
            TactileSiteRef(
                site_name(side, row, col),
                side,
                side_index * ROWS * COLS + row * COLS + col,
            )
            for side_index, side in enumerate(SIDES)
            for row in range(ROWS)
            for col in range(COLS)
        )

    def surface_patch_names(self) -> tuple[str, ...]:
        return SIDES

    def surface_plot_data(self, patch_name: str) -> TactileSurfacePlotData:
        if patch_name not in SIDES:
            raise ValueError(f"Unknown tactile patch {patch_name!r}; known: {list(SIDES)}.")
        side_sign = -1.0 if patch_name == "left" else 1.0
        row_spacing = (Z_MAX - Z_MIN) / ROWS
        col_spacing = (X_MAX - X_MIN) / COLS
        samples = np.asarray(
            [
                [
                    X_MIN + (col + 0.5) * col_spacing,
                    side_sign
                    * _inward_y_magnitude(X_MIN + (col + 0.5) * col_spacing),
                    Z_MIN + (row + 0.5) * row_spacing,
                ]
                for row in range(ROWS)
                for col in range(COLS)
            ],
            dtype=np.float64,
        )
        mesh_name = f"gripper_{patch_name}_link.STL"
        return TactileSurfacePlotData(
            patch_name,
            ROWS,
            COLS,
            "planar-clipped-tip",
            samples,
            read_stl_triangles(PIKA_GRIPPER_DIR / "meshes" / mesh_name),
            (samples.reshape(ROWS, COLS, 3),),
            f"Pika {patch_name} film: {ROWS} x {COLS}",
        )

    def augment_spec(self, hand_spec: mujoco.MjSpec) -> None:
        col_spacing = (X_MAX - X_MIN) / COLS
        row_spacing = (Z_MAX - Z_MIN) / ROWS
        half_col = 0.5 * col_spacing * self.taxel_overlap
        half_row = 0.5 * row_spacing * self.taxel_overlap

        for side in SIDES:
            body = hand_spec.body(f"gripper_{side}_link")
            if body is None:
                raise ValueError(f"Pika jaw body gripper_{side}_link was not found.")
            side_sign = -1.0 if side == "left" else 1.0
            for row in range(ROWS):
                z = Z_MIN + (row + 0.5) * row_spacing
                for col in range(COLS):
                    x = X_MIN + (col + 0.5) * col_spacing
                    name = site_name(side, row, col)
                    frame = _frame(side, x)
                    surface_point = np.asarray(
                        [x, side_sign * _inward_y_magnitude(x), z], dtype=np.float64
                    )
                    # Embed the bottom face slightly into the collision mesh so
                    # contact points remain inside the touch volume despite
                    # solver tolerances. Most of the film still stays outside.
                    outward_offset = self.taxel_half_depth - self.surface_embed
                    point = surface_point + outward_offset * frame[:, 2]
                    site = body.add_site()
                    site.name = name
                    site.type = mujoco.mjtGeom.mjGEOM_BOX
                    site.size = [half_col, half_row, self.taxel_half_depth]
                    site.pos = point.tolist()
                    site.quat = _quat(frame)
                    site.group = self.site_group
                    site.rgba = self.site_rgba

                    sensor = hand_spec.add_sensor()
                    sensor.name = sensor_name(side, row, col)
                    sensor.type = mujoco.mjtSensor.mjSENS_TOUCH
                    sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
                    sensor.objname = name
                    sensor.noise = self.sensor_noise

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del data
        ids = []
        for name in self.sensor_names:
            full_name = self.name_prefix + name
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, full_name)
            if sensor_id < 0:
                raise ValueError(f"Missing Pika tactile sensor {full_name!r}.")
            ids.append(sensor_id)
        sensor_ids = np.asarray(ids, dtype=np.int32)
        if not np.all(model.sensor_dim[sensor_ids] == 1):
            raise ValueError("Pika tactile sensors must all be scalar.")
        self._sensor_adrs = model.sensor_adr[sensor_ids].astype(np.int32)

    def reset(self, model, data, *, rng, options) -> Dict[str, Any]:
        del model, data, rng, options
        self.signal_processor.reset(self.total_taxels)
        return {
            "tactile_sensor": "pika_gripper",
            "tactile_backend": self.backend_name,
            "tactile_size": self.total_taxels,
            "tactile_patches": {side: (ROWS, COLS) for side in SIDES},
            "tactile_signal_processor": self.signal_processor.metadata(),
        }

    def read_raw(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        del model
        if self._sensor_adrs is None:
            raise RuntimeError("PikaGripperTactileSensor.bind() must be called first.")
        return data.sensordata[self._sensor_adrs].astype(np.float32).copy()

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        values = self.signal_processor.process(self.read_raw(model, data), self.patches)
        return values.reshape(2, ROWS, COLS)

    def diagnostic_values(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return self.read_raw(model, data)

    def read_patches(self, model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        return self.patches_from_values(self.read(model, data))

    def patches_from_values(self, values: Any) -> Dict[str, np.ndarray]:
        matrices = np.asarray(values, dtype=np.float32).reshape(2, ROWS, COLS)
        return {side: matrices[index] for index, side in enumerate(SIDES)}


def create_pika_gripper_tactile_sensor(
    backend: str = "simple_box", **kwargs: Any
) -> PikaGripperTactileSensor:
    if str(backend).strip().lower() != "simple_box":
        raise ValueError("Pika gripper supports only the 'simple_box' tactile backend.")
    return PikaGripperTactileSensor(**kwargs)


__all__ = [
    "COLS",
    "ROWS",
    "PikaGripperTactileSensor",
    "create_pika_gripper_tactile_sensor",
    "sensor_name",
    "site_name",
]
