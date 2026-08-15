"""Differentiable surrogate for the six-drive closed-chain Dex Hand.

The original hand URDF exposes passive linkage joints as independent joints,
which makes it invalid for BODex's serial-tree optimizer.  This module instead
samples the authoritative MuJoCo closed-chain model and fits smooth polynomial
pad/surface trajectories as functions of the six physical actuator fractions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HAND_XML = PROJECT_ROOT / "assets" / "grippers" / "dex_hand" / "dex_hand.xml"
SURROGATE_SCHEMA_VERSION = 1
ACTUATOR_NAMES = (
    "act_push_0_j",
    "act_push_1_j",
    "act_push_2_j",
    "act_push_3_j",
    "thumb_rotate_act_push_j",
    "thumb_grasp_act_push_j",
)
OPEN_FRACTIONS = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _farthest_indices(points: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or len(points) < count:
        raise ValueError(f"Cannot select {count} points from a mesh with {len(points)} vertices.")
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    minimum = np.linalg.norm(points - points[selected[0]], axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum))
        minimum = np.minimum(minimum, np.linalg.norm(points - points[selected[index]], axis=1))
    return selected


def _mesh_points(model: mujoco.MjModel, geom_id: int, count: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise ValueError("Dex Hand surrogate expects mesh geoms.")
    start = int(model.mesh_vertadr[mesh_id])
    size = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start : start + size], dtype=np.float64)
    return vertices[_farthest_indices(vertices, count)]


def _transform_geom_points(
    data: mujoco.MjData,
    geom_id: int,
    local_points: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    return local_points @ rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=np.float64)


def _thumb_powers(values: np.ndarray, degree: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    terms = tuple((i, j) for total in range(degree + 1) for i in range(total + 1) for j in [total - i])
    matrix = np.stack(
        [(values[:, 0] ** i) * (values[:, 1] ** j) for i, j in terms],
        axis=1,
    )
    return matrix, terms


@dataclass(frozen=True)
class DexHandSurrogate:
    finger_coefficients: np.ndarray
    thumb_coefficients: np.ndarray
    palm_points: np.ndarray
    contact_offsets: np.ndarray
    points_per_geom: int
    finger_degree: int
    thumb_degree: int
    thumb_terms: tuple[tuple[int, int], ...]
    calibration_rms: float

    @property
    def points_per_digit(self) -> int:
        return self.points_per_geom * 3

    @property
    def surface_point_count(self) -> int:
        return len(self.palm_points) + 5 * self.points_per_digit

    @property
    def contact_indices(self) -> tuple[np.ndarray, ...]:
        palm = len(self.palm_points)
        groups = []
        distal_start = 2 * self.points_per_geom
        for digit in range(5):
            start = palm + digit * self.points_per_digit + distal_start
            groups.append(np.arange(start, start + self.points_per_geom, dtype=np.int64))
        return tuple(groups)

    def evaluate_numpy(self, fractions: np.ndarray) -> np.ndarray:
        fractions = np.asarray(fractions, dtype=np.float64)
        single = fractions.ndim == 1
        fractions = fractions.reshape(-1, 6)
        if np.any((fractions < 0.0) | (fractions > 1.0)):
            raise ValueError("fractions must lie in [0, 1].")
        powers = np.stack(
            [fractions[:, :4] ** degree for degree in range(self.finger_degree + 1)],
            axis=2,
        )
        fingers = np.einsum("bfd,fpdc->bfpc", powers, self.finger_coefficients)
        thumb_basis = np.stack(
            [
                (fractions[:, 4] ** i) * (fractions[:, 5] ** j)
                for i, j in self.thumb_terms
            ],
            axis=1,
        )
        thumb = np.einsum("bt,ptc->bpc", thumb_basis, self.thumb_coefficients)
        palm = np.broadcast_to(self.palm_points, (len(fractions), *self.palm_points.shape))
        result = np.concatenate([palm, fingers.reshape(len(fractions), -1, 3), thumb], axis=1)
        return result[0] if single else result

    def evaluate_torch(self, fractions):
        import torch

        if fractions.ndim != 2 or fractions.shape[1] != 6:
            raise ValueError("fractions must have shape (B, 6).")
        coefficients = torch.as_tensor(
            self.finger_coefficients,
            device=fractions.device,
            dtype=fractions.dtype,
        )
        powers = torch.stack(
            [fractions[:, :4] ** degree for degree in range(self.finger_degree + 1)],
            dim=2,
        )
        fingers = torch.einsum("bfd,fpdc->bfpc", powers, coefficients)
        thumb_basis = torch.stack(
            [
                (fractions[:, 4] ** i) * (fractions[:, 5] ** j)
                for i, j in self.thumb_terms
            ],
            dim=1,
        )
        thumb_coefficients = torch.as_tensor(
            self.thumb_coefficients,
            device=fractions.device,
            dtype=fractions.dtype,
        )
        thumb = torch.einsum("bt,ptc->bpc", thumb_basis, thumb_coefficients)
        palm = torch.as_tensor(
            self.palm_points,
            device=fractions.device,
            dtype=fractions.dtype,
        ).unsqueeze(0).expand(len(fractions), -1, -1)
        return torch.cat([palm, fingers.reshape(len(fractions), -1, 3), thumb], dim=1)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SURROGATE_SCHEMA_VERSION,
            "actuator_names": ACTUATOR_NAMES,
            "points_per_geom": self.points_per_geom,
            "finger_degree": self.finger_degree,
            "thumb_degree": self.thumb_degree,
            "thumb_terms": self.thumb_terms,
            "calibration_rms": self.calibration_rms,
        }
        np.savez_compressed(
            path,
            finger_coefficients=self.finger_coefficients,
            thumb_coefficients=self.thumb_coefficients,
            palm_points=self.palm_points,
            contact_offsets=self.contact_offsets,
            metadata=np.asarray(json.dumps(metadata)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> DexHandSurrogate:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            if metadata.get("schema_version") != SURROGATE_SCHEMA_VERSION:
                raise ValueError("Unsupported Dex Hand surrogate schema.")
            if tuple(metadata.get("actuator_names", ())) != ACTUATOR_NAMES:
                raise ValueError("Dex Hand surrogate actuator order does not match this project.")
            return cls(
                finger_coefficients=np.asarray(payload["finger_coefficients"], dtype=np.float64),
                thumb_coefficients=np.asarray(payload["thumb_coefficients"], dtype=np.float64),
                palm_points=np.asarray(payload["palm_points"], dtype=np.float64),
                contact_offsets=np.asarray(payload["contact_offsets"], dtype=np.int64),
                points_per_geom=int(metadata["points_per_geom"]),
                finger_degree=int(metadata["finger_degree"]),
                thumb_degree=int(metadata["thumb_degree"]),
                thumb_terms=tuple(tuple(term) for term in metadata["thumb_terms"]),
                calibration_rms=float(metadata["calibration_rms"]),
            )


def calibrate_dex_hand_surrogate(
    *,
    hand_xml: str | Path = DEFAULT_HAND_XML,
    points_per_geom: int = 12,
    finger_samples: int = 15,
    thumb_samples: int = 9,
    finger_degree: int = 7,
    thumb_degree: int = 5,
    settle_steps: int = 700,
) -> DexHandSurrogate:
    if finger_samples <= finger_degree or thumb_samples < thumb_degree + 1:
        raise ValueError("Not enough calibration samples for the requested polynomial degree.")
    model = mujoco.MjModel.from_xml_path(str(Path(hand_xml).resolve()))
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(actuator_ids < 0):
        raise ValueError("The Dex Hand XML is missing one or more expected actuators.")
    low = model.actuator_ctrlrange[actuator_ids, 0]
    high = model.actuator_ctrlrange[actuator_ids, 1]

    digit_geoms: list[list[int]] = []
    digit_local_points: list[list[np.ndarray]] = []
    for digit in range(5):
        geom_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"skin_{digit}_{part}_p")
            for part in range(3)
        ]
        if any(geom_id < 0 for geom_id in geom_ids):
            raise ValueError(f"Dex Hand XML is missing skin geoms for digit {digit}.")
        digit_geoms.append(geom_ids)
        digit_local_points.append(
            [_mesh_points(model, geom_id, points_per_geom) for geom_id in geom_ids]
        )
    palm_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "skin_palm_p")
    palm_local = _mesh_points(model, palm_geom, points_per_geom * 3)

    def sample(fractions: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        data = mujoco.MjData(model)
        data.ctrl[actuator_ids] = low + fractions * (high - low)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        digits = [
            np.concatenate(
                [
                    _transform_geom_points(data, geom_id, local_points)
                    for geom_id, local_points in zip(
                        digit_geoms[digit], digit_local_points[digit], strict=True
                    )
                ],
                axis=0,
            )
            for digit in range(5)
        ]
        palm = _transform_geom_points(data, palm_geom, palm_local)
        return digits, palm

    finger_values = np.linspace(0.0, 1.0, finger_samples)
    finger_design = np.vander(finger_values, N=finger_degree + 1, increasing=True)
    finger_coefficients = np.empty(
        (4, points_per_geom * 3, finger_degree + 1, 3),
        dtype=np.float64,
    )
    squared_errors: list[float] = []
    palm_points = None
    for finger in range(4):
        samples = []
        for value in finger_values:
            fractions = OPEN_FRACTIONS.copy()
            fractions[finger] = value
            digits, palm = sample(fractions)
            samples.append(digits[finger])
            if palm_points is None:
                palm_points = palm
        targets = np.asarray(samples)
        coefficients = np.linalg.lstsq(
            finger_design,
            targets.reshape(finger_samples, -1),
            rcond=None,
        )[0].reshape(finger_degree + 1, points_per_geom * 3, 3)
        finger_coefficients[finger] = coefficients.transpose(1, 0, 2)
        predicted = np.einsum("sd,pdc->spc", finger_design, finger_coefficients[finger])
        squared_errors.extend(np.square(predicted - targets).reshape(-1).tolist())

    thumb_grid = np.asarray(
        [(rotate, grasp) for rotate in np.linspace(0.0, 1.0, thumb_samples) for grasp in np.linspace(0.0, 1.0, thumb_samples)],
        dtype=np.float64,
    )
    thumb_design, thumb_terms = _thumb_powers(thumb_grid, thumb_degree)
    thumb_targets = []
    for rotate, grasp in thumb_grid:
        fractions = OPEN_FRACTIONS.copy()
        fractions[4:] = [rotate, grasp]
        digits, _ = sample(fractions)
        thumb_targets.append(digits[4])
    thumb_targets_array = np.asarray(thumb_targets)
    thumb_coefficients = np.linalg.lstsq(
        thumb_design,
        thumb_targets_array.reshape(len(thumb_grid), -1),
        rcond=None,
    )[0].reshape(len(thumb_terms), points_per_geom * 3, 3).transpose(1, 0, 2)
    thumb_predicted = np.einsum("st,ptc->spc", thumb_design, thumb_coefficients)
    squared_errors.extend(np.square(thumb_predicted - thumb_targets_array).reshape(-1).tolist())

    if palm_points is None:
        raise RuntimeError("Dex Hand calibration produced no palm sample.")
    contact_offsets = np.asarray(
        [len(palm_points) + digit * points_per_geom * 3 + points_per_geom * 2 for digit in range(5)],
        dtype=np.int64,
    )
    return DexHandSurrogate(
        finger_coefficients=finger_coefficients,
        thumb_coefficients=thumb_coefficients,
        palm_points=np.asarray(palm_points, dtype=np.float64),
        contact_offsets=contact_offsets,
        points_per_geom=points_per_geom,
        finger_degree=finger_degree,
        thumb_degree=thumb_degree,
        thumb_terms=thumb_terms,
        calibration_rms=float(np.sqrt(np.mean(squared_errors))),
    )


def load_or_calibrate_surrogate(
    cache_path: str | Path,
    **calibration_options: Any,
) -> DexHandSurrogate:
    cache_path = Path(cache_path)
    if cache_path.is_file():
        surrogate = DexHandSurrogate.load(cache_path)
        expected = {
            "points_per_geom": calibration_options.get("points_per_geom"),
            "finger_degree": calibration_options.get("finger_degree"),
            "thumb_degree": calibration_options.get("thumb_degree"),
        }
        if all(
            value is None or getattr(surrogate, name) == value
            for name, value in expected.items()
        ):
            return surrogate
    surrogate = calibrate_dex_hand_surrogate(**calibration_options)
    surrogate.save(cache_path)
    return surrogate
