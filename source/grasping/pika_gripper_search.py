"""Point-cloud grasp and approach search for the Pika parallel gripper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from source.grasping.mesh_pointcloud import SurfacePointCloud
from source.grasping.pika_gripper_surface import (
    PosedPikaGripperSurface,
    load_posed_pika_gripper_surface,
)


@dataclass(frozen=True)
class PikaGraspResult:
    gripper: PosedPikaGripperSurface
    points: np.ndarray
    translation: np.ndarray
    rotation_matrix: np.ndarray
    actuator_fractions: np.ndarray
    maximum_penetration: float
    maximum_noncontact_penetration: float
    mean_contact_distance: float
    contacting_fingers: tuple[int, ...]
    contact_points: np.ndarray
    contact_normals: np.ndarray
    force_closure_residual: float
    table_clearance: float
    pca_axis_index: int
    robustness_margin: float
    preload_weights: np.ndarray
    preload_directions: np.ndarray
    approach_translations: np.ndarray
    approach_rotation_matrices: np.ndarray
    approach_actuator_fractions: np.ndarray
    success: bool


def _metrics(
    points: np.ndarray,
    labels: np.ndarray,
    cloud: SurfacePointCloud,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    tree = cKDTree(cloud.points)
    count = min(8, len(cloud.points))
    _, indices = tree.query(points, k=count)
    if count == 1:
        indices = indices[:, None]
    offsets = points[:, None, :] - cloud.points[indices]
    signed = np.max(
        np.sum(offsets * cloud.normals[indices], axis=2),
        axis=1,
    )
    jaw = labels < 2
    rigid = labels == 2
    penetration = float(np.maximum(-signed[jaw], 0.0).max())
    rigid_penetration = float(np.maximum(-signed[rigid], 0.0).max())
    distances = np.empty(2, dtype=np.float64)
    object_indices = np.empty(2, dtype=np.int64)
    for jaw_index in range(2):
        jaw_distances, jaw_indices = tree.query(points[labels == jaw_index])
        closest = int(np.argmin(jaw_distances))
        distances[jaw_index] = jaw_distances[closest]
        object_indices[jaw_index] = int(jaw_indices[closest])
    return penetration, rigid_penetration, distances, object_indices


def _approach(
    cloud: SurfacePointCloud,
    result: PikaGraspResult,
    *,
    waypoint_count: int = 14,
    clearance: float = 0.10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    progress = np.linspace(0.0, 1.0, waypoint_count)
    final_fraction = float(result.actuator_fractions[0])
    # Stay fully open through most of the approach; close only near contact.
    close_progress = np.clip((progress - 0.72) / 0.28, 0.0, 1.0)
    fractions = (1.0 + close_progress * (final_fraction - 1.0))[:, None]
    local_forward = result.rotation_matrix @ np.asarray([1.0, 0.0, 0.0])
    directions = [
        -local_forward,
        np.asarray([0.0, 0.0, 1.0]),
        -local_forward + np.asarray([0.0, 0.0, 1.0]),
    ]
    rng = np.random.default_rng(seed + 30_000)
    directions.extend(rng.normal(size=(32, 3)))
    best = None
    best_key = None
    surfaces = [
        load_posed_pika_gripper_surface(
            opening_fraction=float(fraction[0]),
            seed=seed + 40_000 + index,
        )
        for index, fraction in enumerate(fractions)
    ]
    for raw_direction in directions:
        direction = np.asarray(raw_direction, dtype=np.float64)
        direction /= max(np.linalg.norm(direction), 1e-9)
        translations = (
            result.translation[None, :] + (1.0 - progress[:, None]) * clearance * direction[None, :]
        )
        violation = 0.0
        for index, (surface, translation) in enumerate(zip(surfaces, translations, strict=True)):
            if index + 1 == waypoint_count:
                continue
            points = surface.points @ result.rotation_matrix.T + translation
            penetration, rigid, _, _ = _metrics(points, surface.labels, cloud)
            violation = max(violation, penetration - 0.003, rigid - 0.0015)
        key = (
            violation > 0.0,
            max(violation, 0.0),
            1.0 - float(direction @ (-local_forward)),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = translations
    if best is None or best_key is None or best_key[0]:
        raise RuntimeError("No collision-free Pika approach path was found.")
    rotations = np.repeat(result.rotation_matrix[None, :, :], waypoint_count, axis=0)
    return best, rotations, fractions


def search_pika_grasp(
    cloud: SurfacePointCloud,
    *,
    samples: int = 128,
    seed: int = 0,
) -> PikaGraspResult:
    """Search opposing-jaw grasps across PCA axes and wrist rolls."""
    if samples < 8:
        raise ValueError("samples must be at least 8.")
    centered = cloud.points - cloud.points.mean(axis=0)
    _, _, rows = np.linalg.svd(centered, full_matrices=False)
    axes = rows.T
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    half_extents = 0.5 * np.ptp(centered @ axes, axis=0)
    table_height = float(cloud.points[:, 2].min())
    opening_fractions = np.linspace(0.18, 0.95, 11)
    surfaces = [
        load_posed_pika_gripper_surface(
            opening_fraction=float(fraction),
            seed=seed + index,
        )
        for index, fraction in enumerate(opening_fractions)
    ]
    best = None
    best_key = None
    for closing_axis_index in range(3):
        for forward_axis_index in range(3):
            if forward_axis_index == closing_axis_index:
                continue
            for forward_sign in (-1.0, 1.0):
                closing_axis = axes[:, closing_axis_index]
                forward_axis = forward_sign * axes[:, forward_axis_index]
                up_axis = np.cross(forward_axis, closing_axis)
                up_axis /= max(np.linalg.norm(up_axis), 1e-9)
                closing_axis = np.cross(up_axis, forward_axis)
                rotation = np.column_stack([forward_axis, closing_axis, up_axis])
                for height_fraction in (-0.45, 0.0, 0.45):
                    target_midpoint = (
                        height_fraction
                        * half_extents[forward_axis_index]
                        * axes[:, forward_axis_index]
                    )
                    for opening_fraction, surface in zip(opening_fractions, surfaces, strict=True):
                        midpoint = surface.contact_centers.mean(axis=0)
                        translation = target_midpoint - rotation @ midpoint
                        points = surface.points @ rotation.T + translation
                        (
                            penetration,
                            rigid_penetration,
                            distances,
                            object_indices,
                        ) = _metrics(points, surface.labels, cloud)
                        contacts = tuple(int(index) for index in np.flatnonzero(distances <= 0.005))
                        contact_points = cloud.points[object_indices]
                        inward = -cloud.normals[object_indices]
                        opposition = float(np.dot(inward[0], inward[1]))
                        force_residual = 0.5 * float(np.linalg.norm(inward[0] + inward[1]))
                        table_clearance = float(points[:, 2].min() - table_height)
                        feasible = (
                            penetration <= 0.003
                            and rigid_penetration <= 0.0015
                            and len(contacts) == 2
                            and opposition <= -0.35
                            and table_clearance >= 0.005
                        )
                        robustness = min(
                            (0.003 - penetration) / 0.003,
                            (0.0015 - rigid_penetration) / 0.0015,
                            (table_clearance - 0.005) / 0.010,
                            (0.005 - float(distances.max())) / 0.005,
                            (-0.35 - opposition) / 0.65,
                        )
                        key = (
                            not feasible,
                            table_clearance < 0.005,
                            rigid_penetration > 0.0015,
                            penetration > 0.003,
                            len(contacts) != 2,
                            opposition > -0.35,
                            # A horizontal approach is substantially easier
                            # for the mounted RM75B and avoids wrist
                            # singularities. Fall back to top-down only when
                            # no collision-free side grasp exists.
                            abs(float(forward_axis[2])) > 0.70,
                            -robustness,
                            force_residual,
                            float(distances.mean()),
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best = PikaGraspResult(
                                gripper=surface,
                                points=points,
                                translation=translation,
                                rotation_matrix=rotation,
                                actuator_fractions=np.asarray([opening_fraction], dtype=np.float64),
                                maximum_penetration=penetration,
                                maximum_noncontact_penetration=rigid_penetration,
                                mean_contact_distance=float(distances.mean()),
                                contacting_fingers=contacts,
                                contact_points=contact_points,
                                contact_normals=inward,
                                force_closure_residual=force_residual,
                                table_clearance=table_clearance,
                                pca_axis_index=closing_axis_index,
                                robustness_margin=robustness,
                                preload_weights=np.ones(1, dtype=np.float64),
                                preload_directions=-np.ones(1, dtype=np.float64),
                                approach_translations=np.empty((0, 3)),
                                approach_rotation_matrices=np.empty((0, 3, 3)),
                                approach_actuator_fractions=np.empty((0, 1)),
                                success=feasible,
                            )
    if best is None:
        raise RuntimeError("Pika grasp search produced no candidates.")
    if not best.success:
        return best
    translations, rotations, fractions = _approach(cloud, best, seed=seed)
    return PikaGraspResult(
        **{
            **best.__dict__,
            "approach_translations": translations,
            "approach_rotation_matrices": rotations,
            "approach_actuator_fractions": fractions,
        }
    )
