"""Geometry-only staged closure with the object initialized inside the hand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from source.grasping.dex_hand_surface import (
    PosedDexHandSurface,
    load_posed_dex_hand_surface,
)
from source.grasping.mesh_pointcloud import SurfacePointCloud


@dataclass(frozen=True)
class HandClosureResult:
    hand: PosedDexHandSurface
    points: np.ndarray
    fingertip_centers: np.ndarray
    translation: np.ndarray
    rotation_matrix: np.ndarray
    actuator_fractions: np.ndarray
    closure: float
    maximum_penetration: float
    mean_contact_distance: float
    contacting_fingers: tuple[int, ...]
    contact_points: np.ndarray
    contact_normals: np.ndarray
    force_closure_residual: float
    success: bool


def _grasp_midpoint(hand: PosedDexHandSurface) -> np.ndarray:
    finger_side = hand.fingertip_centers[:4].mean(axis=0)
    thumb_side = hand.fingertip_centers[4]
    return 0.5 * (finger_side + thumb_side)


def _geometry_metrics(
    points: np.ndarray,
    labels: np.ndarray,
    cloud: SurfacePointCloud,
) -> tuple[float, float, tuple[int, ...]]:
    tree = cKDTree(cloud.points)
    _, point_indices = tree.query(points)
    signed = np.sum(
        (points - cloud.points[point_indices]) * cloud.normals[point_indices],
        axis=1,
    )
    maximum_penetration = float(np.maximum(-signed, 0.0).max())
    finger_distances = []
    contacting = []
    for finger in range(5):
        selected = labels == finger
        distances, _ = tree.query(points[selected])
        minimum = float(np.min(distances))
        finger_distances.append(minimum)
        if minimum <= 0.004:
            contacting.append(finger)
    return maximum_penetration, float(np.mean(finger_distances)), tuple(contacting)


def _actual_contact_quality(
    points: np.ndarray,
    labels: np.ndarray,
    cloud: SurfacePointCloud,
    contacting: tuple[int, ...],
) -> tuple[float, np.ndarray, np.ndarray]:
    if len(contacting) < 2:
        return float("inf"), np.empty((0, 3)), np.empty((0, 3))
    tree = cKDTree(cloud.points)
    wrenches = []
    contact_points = []
    contact_normals = []
    for finger in contacting:
        finger_points = points[labels == finger]
        distances, indices = tree.query(finger_points)
        closest = int(np.argmin(distances))
        object_index = int(indices[closest])
        contact_point = cloud.points[object_index]
        inward = -cloud.normals[object_index]
        contact_points.append(contact_point)
        contact_normals.append(inward)
        reference = (
            np.asarray([0.0, 1.0, 0.0])
            if abs(inward[2]) > 0.9
            else np.asarray([0.0, 0.0, 1.0])
        )
        tangent_1 = np.cross(inward, reference)
        tangent_1 /= np.linalg.norm(tangent_1)
        tangent_2 = np.cross(inward, tangent_1)
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            force = (
                inward
                + 0.8 * np.cos(angle) * tangent_1
                + 0.8 * np.sin(angle) * tangent_2
            )
            force /= np.linalg.norm(force)
            wrenches.append(
                np.concatenate([force, np.cross(contact_point, force)])
            )
    matrix = np.asarray(wrenches, dtype=np.float64).T
    count = matrix.shape[1]
    initial = np.full(count, 1.0 / count)
    solution = minimize(
        lambda weights: float(np.sum(np.square(matrix @ weights))),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 60, "ftol": 1e-10},
    )
    return (
        float(np.linalg.norm(matrix @ solution.x)),
        np.asarray(contact_points),
        np.asarray(contact_normals),
    )


def search_hand_grasp(
    cloud: SurfacePointCloud,
    *,
    samples: int = 64,
    seed: int = 0,
) -> HandClosureResult:
    """Jointly search wrist pose, object-in-hand depth, and six actuators."""
    if samples < 8:
        raise ValueError("samples must be at least 8.")
    open_fractions = np.asarray([0.05, 0.05, 0.05, 0.05, 1.0, 0.05])
    closed_fractions = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    best = None
    best_key = None
    rng = np.random.default_rng(seed)
    for index in range(samples):
        if index < 8:
            closure = index / 7.0
            fractions = open_fractions + closure * (closed_fractions - open_fractions)
            euler = np.zeros(3)
            depth = 0.004 * index
            lateral = np.zeros(2)
        else:
            closure = float(rng.uniform(0.15, 1.0))
            fractions = np.clip(
                open_fractions
                + closure * (closed_fractions - open_fractions)
                + rng.normal(0.0, 0.16, size=6),
                0.0,
                1.0,
            )
            # Preserve the user-specified thumb-opposition prior.
            fractions[4] = rng.uniform(0.70, 1.0)
            euler = rng.uniform(
                [-0.45, -0.45, -np.pi],
                [0.45, 0.45, np.pi],
            )
            depth = float(rng.uniform(0.0, 0.028))
            lateral = rng.uniform(-0.010, 0.010, size=2)
        hand = load_posed_dex_hand_surface(
            actuator_fractions=fractions,
            seed=seed + index,
        )
        rotation = Rotation.from_euler("xyz", euler).as_matrix()
        midpoint = _grasp_midpoint(hand)
        palm_center = hand.points[hand.labels == 5].mean(axis=0)
        outward = midpoint - palm_center
        outward /= max(np.linalg.norm(outward), 1e-9)
        side_1 = np.cross(outward, np.asarray([0.0, 0.0, 1.0]))
        if np.linalg.norm(side_1) < 1e-6:
            side_1 = np.asarray([1.0, 0.0, 0.0])
        side_1 /= np.linalg.norm(side_1)
        side_2 = np.cross(outward, side_1)
        object_in_hand = (
            midpoint + depth * outward + lateral[0] * side_1 + lateral[1] * side_2
        )
        translation = -(rotation @ object_in_hand)
        points = hand.points @ rotation.T + translation
        tips = hand.fingertip_centers @ rotation.T + translation
        penetration, tip_distance, contacting = _geometry_metrics(
            points, hand.labels, cloud
        )
        antagonistic = 4 in contacting and any(finger < 4 for finger in contacting)
        force_closure_residual, contact_points, contact_normals = (
            _actual_contact_quality(points, hand.labels, cloud, contacting)
        )
        feasible = penetration <= 0.003
        # In object coordinates the origin must remain within the opposing
        # fingertip region, not merely somewhere outside the collision mesh.
        tip_low = tips.min(axis=0) - 0.012
        tip_high = tips.max(axis=0) + 0.012
        object_inside_hand = bool(np.all(tip_low <= 0.0) and np.all(0.0 <= tip_high))
        # Lexicographic ordering prevents a contact-rich penetrating candidate
        # from beating a collision-free candidate.
        key = (
            not (feasible and object_inside_hand),
            penetration if not feasible else 0.0,
            not object_inside_hand,
            not antagonistic,
            -len(contacting),
            force_closure_residual,
            tip_distance,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = HandClosureResult(
                hand=hand,
                points=points,
                fingertip_centers=tips,
                translation=translation,
                rotation_matrix=rotation,
                actuator_fractions=fractions,
                closure=float(closure),
                maximum_penetration=penetration,
                mean_contact_distance=tip_distance,
                contacting_fingers=contacting,
                contact_points=contact_points,
                contact_normals=contact_normals,
                force_closure_residual=force_closure_residual,
                success=(
                    feasible
                    and object_inside_hand
                    and antagonistic
                    and force_closure_residual <= 0.35
                ),
            )
    if best is None:
        raise RuntimeError("Closure search produced no candidates.")
    return best
