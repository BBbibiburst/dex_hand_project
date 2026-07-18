"""Geometry-only staged closure with the object initialized inside the hand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from source.grasping.approach_path_search import plan_approach_path
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
    maximum_noncontact_penetration: float
    mean_contact_distance: float
    contacting_fingers: tuple[int, ...]
    contact_points: np.ndarray
    contact_normals: np.ndarray
    force_closure_residual: float
    palmward_force_component: float
    palmward_direction: np.ndarray
    palmward_depth: float
    approach_translations: np.ndarray
    approach_rotation_matrices: np.ndarray
    approach_actuator_fractions: np.ndarray
    success: bool


def _grasp_midpoint(hand: PosedDexHandSurface) -> np.ndarray:
    finger_side = hand.fingertip_centers[:4].mean(axis=0)
    thumb_side = hand.fingertip_centers[4]
    return 0.5 * (finger_side + thumb_side)


def geometry_metrics(
    points: np.ndarray,
    labels: np.ndarray,
    cloud: SurfacePointCloud,
) -> tuple[float, float, float, tuple[int, ...]]:
    tree = cKDTree(cloud.points)
    # A single nearest sampled point is unstable around sparse edges (for
    # example, a cylinder-side query may select a lid sample).  Evaluate a
    # small normal neighbourhood and retain the most exterior tangent-plane
    # projection.  For locally convex surfaces this is a much less noisy
    # approximation of signed distance while staying point-cloud-only.
    neighbour_count = min(8, cloud.points.shape[0])
    _, point_indices = tree.query(points, k=neighbour_count)
    if neighbour_count == 1:
        point_indices = point_indices[:, None]
    offsets = points[:, None, :] - cloud.points[point_indices]
    projections = np.sum(offsets * cloud.normals[point_indices], axis=2)
    signed = np.max(projections, axis=1)
    maximum_penetration = float(np.maximum(-signed, 0.0).max())
    noncontact = labels == 6
    maximum_noncontact_penetration = (
        float(np.maximum(-signed[noncontact], 0.0).max())
        if np.any(noncontact)
        else 0.0
    )
    finger_distances = []
    contacting = []
    for finger in range(5):
        selected = labels == finger
        distances, _ = tree.query(points[selected])
        minimum = float(np.min(distances))
        finger_distances.append(minimum)
        if minimum <= 0.004:
            contacting.append(finger)
    return (
        maximum_penetration,
        maximum_noncontact_penetration,
        float(np.mean(finger_distances)),
        tuple(contacting),
    )


def _actual_contact_quality(
    points: np.ndarray,
    labels: np.ndarray,
    cloud: SurfacePointCloud,
    contacting: tuple[int, ...],
    palmward: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    if len(contacting) < 2:
        return (
            float("inf"),
            np.empty((0, 3)),
            np.empty((0, 3)),
            -1.0,
        )
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
    # Find a realizable preload distribution whose resultant points toward the
    # palm while suppressing sideways force and torque.  Unlike averaging the
    # contact normals, this respects non-negative friction-cone forces and lets
    # different fingers carry different loads.
    def palmward_objective(weights: np.ndarray) -> float:
        wrench = matrix @ weights
        force = wrench[:3]
        along = float(force @ palmward)
        perpendicular = force - along * palmward
        scaled_torque = wrench[3:] / 0.05
        return float(
            np.sum(np.square(perpendicular))
            + np.sum(np.square(scaled_torque))
            - 0.5 * along
        )

    preload = minimize(
        palmward_objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 80, "ftol": 1e-10},
    )
    palmward_force = float((matrix[:3] @ preload.x) @ palmward)
    return (
        float(np.linalg.norm(matrix @ solution.x)),
        np.asarray(contact_points),
        np.asarray(contact_normals),
        palmward_force,
    )


def _legacy_plan_approach(
    cloud: SurfacePointCloud,
    grasp: HandClosureResult,
    *,
    seed: int,
    waypoint_count: int = 14,
    clearance: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Search a short collision-free hand path backwards from the grasp."""
    open_fractions = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    progress = np.linspace(0.0, 1.0, waypoint_count)
    fractions = (
        open_fractions[None, :]
        + progress[:, None]
        * (grasp.actuator_fractions - open_fractions)[None, :]
    )
    posed_hands = [
        load_posed_dex_hand_surface(
            actuator_fractions=waypoint_fractions,
            seed=seed + 10_000 + index,
        )
        for index, waypoint_fractions in enumerate(fractions)
    ]

    palmward = grasp.palmward_direction / np.linalg.norm(grasp.palmward_direction)
    axes = np.eye(3)
    tabletop_preference = palmward + 1.5 * np.asarray([0.0, 0.0, 1.0])
    tabletop_preference /= np.linalg.norm(tabletop_preference)
    directions = [palmward, tabletop_preference]
    directions.extend(axes)
    directions.extend(-axes)
    rng = np.random.default_rng(seed + 20_000)
    directions.extend(rng.normal(size=(48, 3)))

    best = None
    best_key = None
    for raw_direction in directions:
        direction = np.asarray(raw_direction, dtype=np.float64)
        direction /= max(np.linalg.norm(direction), 1e-9)
        translations = []
        maximum_pad_violation = 0.0
        maximum_rigid_violation = 0.0
        # Waypoints are ordered from a clear pregrasp to the final grasp.
        for index, hand in enumerate(posed_hands):
            remaining = 1.0 - progress[index]
            translation = grasp.translation + clearance * remaining * direction
            points = hand.points @ grasp.rotation_matrix.T + translation
            penetration, rigid_penetration, _, _ = geometry_metrics(
                points,
                hand.labels,
                cloud,
            )
            # The final optimized grasp is intentionally in contact.
            if index + 1 < waypoint_count:
                maximum_pad_violation = max(
                    maximum_pad_violation,
                    penetration - 0.004,
                )
                maximum_rigid_violation = max(
                    maximum_rigid_violation,
                    rigid_penetration - 0.0015,
                )
            translations.append(translation)
        key = (
            max(maximum_pad_violation, 0.0) > 0.0
            or max(maximum_rigid_violation, 0.0) > 0.0,
            max(maximum_rigid_violation, 0.0),
            max(maximum_pad_violation, 0.0),
            1.0 - float(direction @ tabletop_preference),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = np.asarray(translations)

    if best is None or best_key is None or best_key[0]:
        raise RuntimeError("No collision-free point-cloud approach path was found.")
    rotations = np.repeat(
        grasp.rotation_matrix[None, :, :],
        waypoint_count,
        axis=0,
    )
    return best, rotations, fractions


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
    # Deterministic seeds decouple finger closure from in-hand depth.  The old
    # diagonal schedule changed both together and skipped the narrow interval
    # where pads touch without penetrating.
    seed_closures = np.asarray([0.0, 0.05, 0.10, 0.14, 0.18, 0.25, 0.40, 0.60])
    seed_depths = np.asarray([0.0, 0.006, 0.012, 0.018])
    deterministic_count = min(samples, seed_closures.size * seed_depths.size)
    for index in range(samples):
        if index < deterministic_count:
            closure_index, depth_index = divmod(index, seed_depths.size)
            closure = float(seed_closures[closure_index])
            fractions = open_fractions + closure * (closed_fractions - open_fractions)
            euler = np.zeros(3)
            depth = float(seed_depths[depth_index])
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
        # Positive depth must move the object from the fingertip midpoint
        # toward the palm.  The previous plus sign moved it toward the open
        # side of the hand and made outward slip more likely.
        object_in_hand = (
            midpoint - depth * outward + lateral[0] * side_1 + lateral[1] * side_2
        )
        translation = -(rotation @ object_in_hand)
        points = hand.points @ rotation.T + translation
        tips = hand.fingertip_centers @ rotation.T + translation
        palm_in_object = rotation @ (palm_center - object_in_hand)
        palmward = palm_in_object / max(np.linalg.norm(palm_in_object), 1e-9)
        penetration, noncontact_penetration, tip_distance, contacting = geometry_metrics(
            points, hand.labels, cloud
        )
        antagonistic = 4 in contacting and any(finger < 4 for finger in contacting)
        (
            force_closure_residual,
            contact_points,
            contact_normals,
            palmward_force,
        ) = (
            _actual_contact_quality(
                points,
                hand.labels,
                cloud,
                contacting,
                palmward,
            )
        )
        # Soft finger pads may overlap the point-cloud surface slightly while
        # making contact.  Rigid base/linkage meshes use a tighter limit.
        feasible = penetration <= 0.004 and noncontact_penetration <= 0.0015
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
            palmward_force < 0.0,
            -palmward_force,
            -len(contacting),
            -depth,
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
                maximum_noncontact_penetration=noncontact_penetration,
                mean_contact_distance=tip_distance,
                contacting_fingers=contacting,
                contact_points=contact_points,
                contact_normals=contact_normals,
                force_closure_residual=force_closure_residual,
                palmward_force_component=palmward_force,
                palmward_direction=palmward,
                palmward_depth=depth,
                approach_translations=np.empty((0, 3)),
                approach_rotation_matrices=np.empty((0, 3, 3)),
                approach_actuator_fractions=np.empty((0, 6)),
                success=(
                    feasible
                    and object_inside_hand
                    and antagonistic
                    and palmward_force >= 0.0
                    and force_closure_residual <= 0.35
                ),
            )
    if best is None:
        raise RuntimeError("Closure search produced no candidates.")
    if not best.success:
        return best
    approach_translations, approach_rotations, approach_fractions = plan_approach_path(
        cloud,
        best,
        seed=seed,
    )
    return HandClosureResult(
        **{
            **best.__dict__,
            "approach_translations": approach_translations,
            "approach_rotation_matrices": approach_rotations,
            "approach_actuator_fractions": approach_fractions,
        }
    )
