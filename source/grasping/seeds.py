"""Geometry-only enclosure seeds for GraspQP and DexEvolve."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from source.grasping.catalog import ObjectGeometry
from source.grasping.contracts import GraspCandidate
from source.grasping.hand_surrogate import DexHandSurrogate

@dataclass(frozen=True)
class SeedConfig:
    table_margin: float = 0.0015
    enclosure_prior_count: int = 128
    seed: int = 0

    def validate(self) -> None:
        if self.enclosure_prior_count < 0:
            raise ValueError("enclosure_prior_count cannot be negative.")
        if self.table_margin < 0.0:
            raise ValueError("table_margin cannot be negative.")


def _normalize(vector, *, eps: float = 1e-8):
    import torch

    return vector / torch.clamp(torch.linalg.norm(vector, dim=-1, keepdim=True), min=eps)


def rotation_6d_to_matrix(rotation_6d):
    """Convert Zhou et al. 6D rotations to right-handed matrices."""
    import torch

    first = _normalize(rotation_6d[..., :3])
    second_raw = rotation_6d[..., 3:]
    second = _normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def _inverse_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _rotation_from_approach(
    approach: np.ndarray,
    preferred_spread: np.ndarray,
    roll: float,
) -> np.ndarray:
    """Build a hand rotation whose local +Y follows ``approach``."""
    y_axis = approach / max(float(np.linalg.norm(approach)), 1e-9)
    z_axis = preferred_spread - np.dot(preferred_spread, y_axis) * y_axis
    if np.linalg.norm(z_axis) < 1e-6:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        z_axis = fallback - np.dot(fallback, y_axis) * y_axis
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    z_axis = np.cross(x_axis, y_axis)

    cosine = math.cos(roll)
    sine = math.sin(roll)
    rolled_x = cosine * x_axis - sine * z_axis
    rolled_z = sine * x_axis + cosine * z_axis
    return np.column_stack([rolled_x, y_axis, rolled_z])


def _seed_rotations(
    count: int,
    side_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[str, ...]]:
    side_count = round(count * side_fraction)
    side_count = min(max(side_count, 0), count)
    top_count = count - side_count
    rotations: list[np.ndarray] = []
    families: list[str] = []
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    roll_pattern = np.deg2rad(np.asarray([0.0, -18.0, 18.0], dtype=np.float64))

    for index in range(side_count):
        azimuth = phase + 2.0 * math.pi * index / max(side_count, 1)
        inward = np.asarray([-math.cos(azimuth), -math.sin(azimuth), 0.0])
        roll = float(roll_pattern[index % len(roll_pattern)])
        rotations.append(
            _rotation_from_approach(inward, np.asarray([0.0, 0.0, 1.0]), roll)
        )
        families.append("side")

    for index in range(top_count):
        azimuth = phase + 2.0 * math.pi * index / max(top_count, 1)
        spread = np.asarray([math.cos(azimuth), math.sin(azimuth), 0.0])
        rotations.append(
            _rotation_from_approach(
                np.asarray([0.0, 0.0, -1.0]),
                spread,
                float(roll_pattern[index % len(roll_pattern)]),
            )
        )
        families.append("top")
    return np.asarray(rotations, dtype=np.float64), tuple(families)


def _digit_centers(surrogate: DexHandSurrogate, fractions: np.ndarray) -> np.ndarray:
    points = surrogate.evaluate_numpy(fractions)
    return np.asarray([points[group].mean(axis=0) for group in surrogate.contact_indices])


def _object_width(vertices: np.ndarray, direction: np.ndarray) -> float:
    unit = direction / max(float(np.linalg.norm(direction)), 1e-9)
    projection = vertices @ unit
    return float(projection.max() - projection.min())


def _initialize_seeds(
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    *,
    count: int,
    side_fraction: float,
    table_margin: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotations, families = _seed_rotations(
        count,
        side_fraction,
        rng,
    )
    fractions = np.empty((count, 6), dtype=np.float64)
    translations = np.empty((count, 3), dtype=np.float64)
    target_centers = np.empty((count, 3), dtype=np.float64)
    object_center = 0.5 * (geometry.bounds[0] + geometry.bounds[1])
    object_height = float(geometry.bounds[1, 2] - geometry.bounds[0, 2])
    closing_grid = np.linspace(0.03, 0.62, 60)

    for index, (rotation, family) in enumerate(zip(rotations, families, strict=True)):
        thumb_rotate = float(
            rng.uniform(0.82, 1.0) if family == "side" else rng.uniform(0.2, 1.0)
        )
        candidates = np.repeat(closing_grid[:, None], 6, axis=1)
        candidates[:, 4] = thumb_rotate
        surfaces = surrogate.evaluate_numpy(candidates)
        centers = np.stack(
            [
                np.stack(
                    [surface[group].mean(axis=0) for group in surrogate.contact_indices]
                )
                for surface in surfaces
            ],
            axis=0,
        )
        finger_centers = centers[:, :4].mean(axis=1)
        gaps = centers[:, 4] - finger_centers
        gap_lengths = np.linalg.norm(gaps, axis=1)
        world_directions = np.einsum(
            "ij,nj->ni",
            rotation,
            gaps / np.maximum(gap_lengths[:, None], 1e-9),
        )
        object_widths = np.asarray(
            [_object_width(geometry.vertices, direction) for direction in world_directions]
        )
        desired_gaps = object_widths + 0.010
        feasible = np.flatnonzero(gap_lengths >= desired_gaps)
        if len(feasible):
            selected = int(
                feasible[np.argmin(gap_lengths[feasible] - desired_gaps[feasible])]
            )
        else:
            selected = int(np.argmax(gap_lengths))

        selected_fractions = candidates[selected].copy()
        selected_fractions[:4] += rng.normal(0.0, 0.018, size=4)
        selected_fractions[5] += float(rng.normal(0.0, 0.015))
        selected_fractions = np.clip(selected_fractions, 0.01, 0.99)
        selected_centers = _digit_centers(surrogate, selected_fractions)
        cavity_center = 0.5 * (
            selected_centers[:4].mean(axis=0) + selected_centers[4]
        )

        target = object_center.copy()
        if family == "top":
            target[2] += 0.20 * object_height
        translation = target - rotation @ cavity_center
        translation += rng.normal(0.0, 0.0015, size=3)

        hand_points = surrogate.evaluate_numpy(selected_fractions) @ rotation.T + translation
        minimum_z = float(hand_points[:, 2].min())
        required_z = geometry.table_z + table_margin
        if minimum_z < required_z:
            shift = required_z - minimum_z
            translation[2] += shift
            target[2] += shift

        fractions[index] = selected_fractions
        translations[index] = translation
        target_centers[index] = target
    return translations, rotations, fractions, target_centers


def generate_enclosure_seeds(
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    config: SeedConfig,
) -> tuple[GraspCandidate, ...]:
    """Generate centered side-enclosure proposals without contact-QP drift.

    These proposals preserve the underactuated mechanical prior: the object is
    placed between the opposed digit groups and the final stationary close is
    left to MuJoCo.  They intentionally bypass the free contact-force optimizer,
    which tends to turn large cylinders into marginal fingertip pinches.
    """
    if config.enclosure_prior_count == 0:
        return ()
    # Each azimuth is expanded over a compact depth/height/closure lattice.
    # Repeating nearly identical azimuths wastes the budget on rotationally
    # symmetric cans while never correcting an off-centre closing resultant.
    variants = tuple(
        (depth, height, lateral, middle)
        for depth in (-0.010, 0.010)
        for height in (-0.008, 0.008)
        for lateral in (-0.020, 0.020)
        for middle in (0.00, 0.08)
    )
    base_count = int(math.ceil(config.enclosure_prior_count / len(variants)))
    translations, rotations, fractions, _ = _initialize_seeds(
        geometry,
        surrogate,
        count=base_count,
        side_fraction=1.0,
        table_margin=config.table_margin,
        rng=np.random.default_rng(config.seed + 104_729),
    )
    candidates: list[GraspCandidate] = []
    object_points = np.asarray(geometry.surface_points, dtype=np.float64)
    object_normals = np.asarray(geometry.surface_normals, dtype=np.float64)
    expanded = []
    for base_translation, rotation, base_fractions in zip(
        translations, rotations, fractions, strict=True
    ):
        centers = _digit_centers(surrogate, base_fractions)
        gap = centers[4] - centers[:4].mean(axis=0)
        gap_world = rotation @ (gap / max(float(np.linalg.norm(gap)), 1e-9))
        lateral_world = rotation[:, 0]
        for depth, height, lateral, middle_delta in variants:
            actuator_fractions = base_fractions.copy()
            actuator_fractions[1] = np.clip(
                actuator_fractions[1] + middle_delta, 0.0, 1.0
            )
            translation = base_translation + depth * gap_world + lateral * lateral_world
            translation = translation.copy()
            translation[2] += height
            expanded.append(
                (translation, rotation, actuator_fractions, depth, height, lateral, middle_delta)
            )
    for index, (translation, rotation, actuator_fractions, depth, height, lateral, middle_delta) in enumerate(
        expanded[: config.enclosure_prior_count]
    ):
        hand_points = surrogate.evaluate_numpy(actuator_fractions) @ rotation.T + translation
        contacts, normals, distances = [], [], []
        for group in surrogate.contact_indices:
            delta = hand_points[group, None, :] - object_points[None, :, :]
            pair = np.unravel_index(np.argmin(np.sum(delta * delta, axis=2)), delta.shape[:2])
            object_index = int(pair[1])
            contacts.append(object_points[object_index])
            normals.append(object_normals[object_index])
            distances.append(float(np.linalg.norm(delta[pair])))
        candidates.append(
            GraspCandidate(
                object_id=geometry.object_id,
                seed_index=1_000_000 + index,
                hand_translation=translation,
                hand_rotation_matrix=rotation,
                actuator_fractions=actuator_fractions,
                contact_points=np.asarray(contacts),
                contact_normals=np.asarray(normals),
                contact_distances=np.asarray(distances),
                metrics={
                    # A geometry proposal is not a grasp until real MuJoCo
                    # close/hold validation observes persistent opposition.
                    "valid": 0.0,
                    "enclosure_prior": 1.0,
                    "mean_contact_distance": float(np.mean(distances)),
                    "enclosure_depth_offset": depth,
                    "enclosure_height_offset": height,
                    "enclosure_lateral_offset": lateral,
                    "enclosure_middle_delta": middle_delta,
                },
                backend="pca-centered-enclosure",
            )
        )
    return tuple(candidates)


def convex_outside_distance(points, plane_normals, plane_offsets, plane_part_offsets=None):
    """Positive outside distance for one convex hull or a union of hulls."""
    import torch

    values = torch.einsum("bni,pi->bnp", points, plane_normals) - plane_offsets
    if plane_part_offsets is None:
        return values.max(dim=-1).values
    offsets = [int(value) for value in plane_part_offsets]
    per_part = [
        values[..., offsets[index] : offsets[index + 1]].max(dim=-1).values
        for index in range(len(offsets) - 1)
    ]
    return torch.stack(per_part, dim=-1).min(dim=-1).values
