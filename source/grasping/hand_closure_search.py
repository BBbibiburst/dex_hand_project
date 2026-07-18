"""Geometry-only staged closure with the object initialized inside the hand."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

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
    table_clearance: float
    pca_axis_index: int
    robustness_margin: float
    object_inside_hand: bool
    preload_weights: np.ndarray
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
        float(np.maximum(-signed[noncontact], 0.0).max()) if np.any(noncontact) else 0.0
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
) -> tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    if len(contacting) < 2:
        return (
            float("inf"),
            np.empty((0, 3)),
            np.empty((0, 3)),
            -1.0,
            np.zeros(5, dtype=np.float64),
        )
    tree = cKDTree(cloud.points)
    wrenches = []
    wrench_fingers = []
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
            np.asarray([0.0, 1.0, 0.0]) if abs(inward[2]) > 0.9 else np.asarray([0.0, 0.0, 1.0])
        )
        tangent_1 = np.cross(inward, reference)
        tangent_1 /= np.linalg.norm(tangent_1)
        tangent_2 = np.cross(inward, tangent_1)
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            force = inward + 0.8 * np.cos(angle) * tangent_1 + 0.8 * np.sin(angle) * tangent_2
            force /= np.linalg.norm(force)
            wrenches.append(np.concatenate([force, np.cross(contact_point, force)]))
            wrench_fingers.append(finger)
    matrix = np.asarray(wrenches, dtype=np.float64).T
    wrench_fingers = np.asarray(wrench_fingers, dtype=np.int64)
    count = matrix.shape[1]

    def closure_residual(selected_matrix: np.ndarray) -> float:
        selected_count = selected_matrix.shape[1]
        initial_weights = np.full(selected_count, 1.0 / selected_count)
        solution = minimize(
            lambda weights: float(np.sum(np.square(selected_matrix @ weights))),
            initial_weights,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * selected_count,
            constraints={
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
            },
            options={"maxiter": 60, "ftol": 1e-10},
        )
        return float(np.linalg.norm(selected_matrix @ solution.x))

    force_closure_residual = closure_residual(matrix)
    active_fingers = tuple(contacting)
    subset_key = None
    # Point-contact friction needs at least three fingers for a useful
    # spatial grasp.  Choose the smallest subset that keeps the analytic
    # residual low, always retaining thumb opposition.
    for subset_size in range(3, len(contacting) + 1):
        for subset in combinations(contacting, subset_size):
            if 4 not in subset or not any(finger < 4 for finger in subset):
                continue
            selected = np.isin(wrench_fingers, subset)
            residual = closure_residual(matrix[:, selected])
            key = (residual > 0.08, subset_size, residual)
            if subset_key is None or key < subset_key:
                subset_key = key
                active_fingers = subset
        if subset_key is not None and not subset_key[0]:
            break
    preload_weights = np.zeros(5, dtype=np.float64)
    preload_weights[np.asarray(active_fingers, dtype=np.int64)] = 1.0

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
            np.sum(np.square(perpendicular)) + np.sum(np.square(scaled_torque)) - 0.5 * along
        )

    initial = np.full(count, 1.0 / count)
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
        force_closure_residual,
        np.asarray(contact_points),
        np.asarray(contact_normals),
        palmward_force,
        preload_weights,
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
        + progress[:, None] * (grasp.actuator_fractions - open_fractions)[None, :]
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
            max(maximum_pad_violation, 0.0) > 0.0 or max(maximum_rigid_violation, 0.0) > 0.0,
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
    centered = cloud.points - cloud.points.mean(axis=0)
    _, _, principal_rows = np.linalg.svd(centered, full_matrices=False)
    principal_axes = principal_rows.T
    for axis_index in range(3):
        axis = principal_axes[:, axis_index]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            principal_axes[:, axis_index] *= -1.0
    if np.linalg.det(principal_axes) < 0.0:
        principal_axes[:, 2] *= -1.0
    projections = centered @ principal_axes
    principal_half_extents = 0.5 * np.ptp(projections, axis=0)
    table_height = float(cloud.points[:, 2].min())
    # Deterministic seeds decouple finger closure from in-hand depth.  The old
    # diagonal schedule changed both together and skipped the narrow interval
    # where pads touch without penetrating.
    seed_closures = np.asarray([0.0, 0.05, 0.10, 0.14, 0.18, 0.25, 0.40, 0.60])
    seed_depths = np.asarray([0.0, 0.006, 0.012, 0.018])
    deterministic_count = min(samples, seed_closures.size * seed_depths.size)
    # 24 orientations cover +/- each PCA axis at four wrist rolls.  Repeating
    # that bank with several closure/depth/height variants leaves part of the
    # budget for genuinely random refinement at the default 128 samples.
    pca_count = min(max(samples - deterministic_count, 0), 64)
    for index in range(samples):
        if index < deterministic_count:
            closure_index, depth_index = divmod(index, seed_depths.size)
            closure = float(seed_closures[closure_index])
            fractions = open_fractions + closure * (closed_fractions - open_fractions)
            euler = np.zeros(3)
            depth = float(seed_depths[depth_index])
            lateral = np.zeros(2)
            pca_axis_index = 2
            axial_fraction = 0.0
            use_pca_orientation = False
        elif index < deterministic_count + pca_count:
            pca_index = index - deterministic_count
            legacy_pca_count = pca_count // 2
            if pca_index < legacy_pca_count:
                pca_axis_index = pca_index % 3
                axial_levels = np.asarray([0.0, 0.25, 0.45, -0.20])
                axial_fraction = float(axial_levels[(pca_index // 3) % len(axial_levels)])
                closure_levels = np.asarray([0.10, 0.14, 0.18, 0.25, 0.35])
                closure = float(closure_levels[(pca_index // 12) % len(closure_levels)])
                depth = float(seed_depths[(pca_index // 48) % len(seed_depths)])
                use_pca_orientation = False
            else:
                orientation_index = (pca_index - legacy_pca_count) % 24
                variant = (pca_index - legacy_pca_count) // 24
                pca_axis_index = orientation_index // 8
                axis_sign = 1.0 if (orientation_index // 4) % 2 == 0 else -1.0
                wrist_roll = 0.5 * np.pi * (orientation_index % 4)
                axial_levels = np.asarray([0.0, 0.35, -0.25, 0.60])
                axial_fraction = float(axial_levels[variant % len(axial_levels)])
                closure_levels = np.asarray([0.10, 0.20, 0.32, 0.45])
                closure = float(closure_levels[variant % len(closure_levels)])
                depth = float(seed_depths[variant % len(seed_depths)])
                use_pca_orientation = True
            fractions = open_fractions + closure * (closed_fractions - open_fractions)
            fractions[4] = 1.0
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
            lateral = rng.uniform(-0.025, 0.025, size=2)
            pca_axis_index = int(rng.integers(0, 3))
            axial_fraction = float(rng.uniform(-0.25, 0.50))
            use_pca_orientation = bool(rng.integers(0, 2))
            axis_sign = float(rng.choice([-1.0, 1.0]))
            wrist_roll = float(rng.uniform(-np.pi, np.pi))
        hand = load_posed_dex_hand_surface(
            actuator_fractions=fractions,
            seed=seed + index,
        )
        midpoint = _grasp_midpoint(hand)
        palm_center = hand.points[hand.labels == 5].mean(axis=0)
        outward = midpoint - palm_center
        outward /= max(np.linalg.norm(outward), 1e-9)
        side_1 = np.cross(outward, np.asarray([0.0, 0.0, 1.0]))
        if np.linalg.norm(side_1) < 1e-6:
            side_1 = np.asarray([1.0, 0.0, 0.0])
        side_1 /= np.linalg.norm(side_1)
        side_2 = np.cross(outward, side_1)
        if use_pca_orientation:
            remaining = [axis for axis in range(3) if axis != pca_axis_index]
            target_outward = axis_sign * principal_axes[:, pca_axis_index]
            base_side_1 = principal_axes[:, remaining[0]]
            base_side_2 = np.cross(target_outward, base_side_1)
            base_side_2 /= max(np.linalg.norm(base_side_2), 1e-9)
            base_side_1 = np.cross(base_side_2, target_outward)
            target_side_1 = np.cos(wrist_roll) * base_side_1 + np.sin(wrist_roll) * base_side_2
            target_side_2 = np.cross(target_outward, target_side_1)
            target_basis = np.column_stack([target_outward, target_side_1, target_side_2])
            source_basis = np.column_stack([outward, side_1, side_2])
            rotation = target_basis @ source_basis.T
        else:
            rotation = Rotation.from_euler("xyz", euler).as_matrix()
        axis_offset = (
            axial_fraction
            * principal_half_extents[pca_axis_index]
            * principal_axes[:, pca_axis_index]
        )
        desired_midpoint = (
            axis_offset
            + depth * (rotation @ outward)
            - lateral[0] * (rotation @ side_1)
            - lateral[1] * (rotation @ side_2)
        )
        translation = desired_midpoint - rotation @ midpoint
        points = hand.points @ rotation.T + translation
        tips = hand.fingertip_centers @ rotation.T + translation
        palm_in_object = rotation @ palm_center + translation
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
            finger_preload_weights,
        ) = _actual_contact_quality(
            points,
            hand.labels,
            cloud,
            contacting,
            palmward,
        )
        # Soft finger pads may overlap the point-cloud surface slightly while
        # making contact.  Rigid base/linkage meshes use a tighter limit.
        feasible = penetration <= 0.004 and noncontact_penetration <= 0.0015
        table_clearance = float(points[:, 2].min() - table_height)
        # In object coordinates the origin must remain within the opposing
        # fingertip region, not merely somewhere outside the collision mesh.
        tip_low = tips.min(axis=0) - 0.012
        tip_high = tips.max(axis=0) + 0.012
        object_inside_hand = bool(
            np.any(
                np.all(
                    (tip_low[None, :] <= cloud.points) & (cloud.points <= tip_high[None, :]),
                    axis=1,
                )
            )
        )
        robustness_margin = min(
            (0.004 - penetration) / 0.004,
            (0.0015 - noncontact_penetration) / 0.0015,
            (table_clearance - 0.005) / 0.010,
            palmward_force / 0.20,
            (0.35 - force_closure_residual) / 0.35,
        )
        actuator_preload_weights = np.asarray(
            [
                finger_preload_weights[0],
                finger_preload_weights[1],
                finger_preload_weights[2],
                finger_preload_weights[3],
                0.0,
                finger_preload_weights[4],
            ],
            dtype=np.float64,
        )
        # Lexicographic ordering prevents a contact-rich penetrating candidate
        # from beating a collision-free candidate.
        key = (
            not (feasible and object_inside_hand),
            penetration if not feasible else 0.0,
            not object_inside_hand,
            table_clearance < 0.005,
            not antagonistic,
            palmward_force < 0.0,
            force_closure_residual > 0.35,
            -robustness_margin,
            -table_clearance,
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
                table_clearance=table_clearance,
                pca_axis_index=pca_axis_index,
                robustness_margin=robustness_margin,
                object_inside_hand=object_inside_hand,
                preload_weights=actuator_preload_weights,
                approach_translations=np.empty((0, 3)),
                approach_rotation_matrices=np.empty((0, 3, 3)),
                approach_actuator_fractions=np.empty((0, 6)),
                success=(
                    feasible
                    and object_inside_hand
                    and table_clearance >= 0.005
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
