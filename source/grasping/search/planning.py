"""Object-relative pose generation and collision-checked approach planning."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from source.grasping.search.common import progress
from source.grasping.search.scoring import _full_mesh_table_clearance, _signed_surface_distances
from source.grasping.search.types import ApproachPlan, Candidate, Cloud, Device, Surface

def _orthonormal_frame_from_normal(normal: np.ndarray, roll: float) -> np.ndarray:
    """Build a hand frame whose +X axis points inward from the object surface."""
    x_axis = -np.asarray(normal, dtype=np.float64)
    x_axis /= max(np.linalg.norm(x_axis), 1e-9)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(x_axis @ reference)) > 0.92:
        reference = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    base = np.column_stack([x_axis, y_axis, z_axis])
    return base @ Rotation.from_rotvec(np.array([roll, 0.0, 0.0])).as_matrix()


def _spread_anchor_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Cheap farthest-point sampling for spatially spread anchors."""
    rng = np.random.default_rng(seed)
    count = min(max(1, count), len(points))
    selected = [int(rng.integers(len(points)))]
    minimum_distance = np.full(len(points), np.inf)
    for _ in range(1, count):
        delta = points - points[selected[-1]]
        minimum_distance = np.minimum(minimum_distance, np.einsum("ij,ij->i", delta, delta))
        selected.append(int(np.argmax(minimum_distance)))
    return np.asarray(selected, dtype=int)


def _grasp_center_from_anchor(
    cloud: Cloud,
    anchor_index: int,
    *,
    lateral_radius: float,
) -> tuple[np.ndarray, float]:
    """Estimate the interior grasp center from a surface anchor.

    The old implementation aligned the hand's grasp midpoint directly with the
    surface anchor.  That leaves most of a convex object outside the fingers.
    Here we cast a small point-cloud ray along the inward surface normal, find
    the opposite side, and place the grasp midpoint halfway through the local
    object chord.
    """
    anchor = cloud.points[anchor_index]
    inward = -cloud.normals[anchor_index]
    inward /= max(np.linalg.norm(inward), 1e-9)

    delta = cloud.points - anchor
    axial = delta @ inward
    lateral_vector = delta - axial[:, None] * inward[None, :]
    lateral = np.linalg.norm(lateral_vector, axis=1)

    mask = (axial > 0.004) & (lateral <= lateral_radius)
    if np.any(mask):
        # Use a high percentile instead of the single furthest point, which is
        # much less sensitive to sparse/noisy point-cloud outliers.
        local_depth = float(np.percentile(axial[mask], 90.0))
    else:
        # Conservative fallback: move inward a little rather than leaving the
        # grasp center exactly on the surface.
        positive = axial[axial > 0.004]
        local_depth = float(np.percentile(positive, 35.0)) if len(positive) else 0.012

    local_depth = float(np.clip(local_depth, 0.010, 0.090))
    return anchor + 0.5 * local_depth * inward, local_depth


def local_pose_candidates(
    cloud: Cloud,
    *,
    anchor_count: int,
    rolls_per_anchor: int,
    support_margin: float,
    seed: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Generate support-aware poses with grasp centers inside the object."""
    table_z = float(cloud.points[:, 2].min())
    usable = np.flatnonzero(cloud.points[:, 2] >= table_z + support_margin)
    if not len(usable):
        usable = np.arange(len(cloud.points))
        progress("[anchors] warning: support filter removed every point; using all points")

    local = _spread_anchor_indices(cloud.points[usable], anchor_count, seed)
    anchor_indices = usable[local]
    rolls = np.linspace(0.0, 2.0 * np.pi, max(1, rolls_per_anchor), endpoint=False)

    object_extent = float(np.ptp(cloud.points, axis=0).max())
    lateral_radius = float(np.clip(0.16 * object_extent, 0.008, 0.018))
    poses = []
    chord_depths = []
    for anchor_index in anchor_indices:
        normal = cloud.normals[anchor_index]
        grasp_center, chord_depth = _grasp_center_from_anchor(
            cloud, int(anchor_index), lateral_radius=lateral_radius
        )
        chord_depths.append(chord_depth)
        for roll_index, roll in enumerate(rolls):
            poses.append(
                (
                    int(anchor_index),
                    roll_index,
                    _orthonormal_frame_from_normal(normal, roll),
                    grasp_center,
                )
            )

    median_depth = float(np.median(chord_depths)) if chord_depths else 0.0
    progress(
        f"[anchors] usable={len(usable)}/{len(cloud.points)} "
        f"selected={len(anchor_indices)} poses={len(poses)} "
        f"median_chord={median_depth * 1000.0:.1f}mm"
    )
    return poses


def _approach_directions(candidate: Candidate, seed: int) -> list[np.ndarray]:
    """Generate outward-biased straight-line approach directions."""
    # The hand's local +X axis describes contact orientation, not the vector
    # from the grasp center back to the wrist/root. Dex Hand's latter direction
    # is mostly local -Y, so using -X can move the palm into the object.
    outward = -(candidate.surface.midpoint @ candidate.rotation.T)
    outward /= max(np.linalg.norm(outward), 1e-9)
    world_up = np.asarray([0.0, 0.0, 1.0])
    lateral = candidate.rotation[:, 1]
    vertical = candidate.rotation[:, 2]
    raw = [
        outward,
        outward + 0.5 * world_up,
        outward + world_up,
        outward + 0.35 * lateral + 0.5 * world_up,
        outward - 0.35 * lateral + 0.5 * world_up,
        outward + 0.35 * vertical + 0.5 * world_up,
        outward - 0.35 * vertical + 0.5 * world_up,
    ]
    rng = np.random.default_rng(seed)
    for _ in range(6):
        raw.append(outward + rng.normal(scale=0.45, size=3) + rng.uniform(0.0, 0.8) * world_up)

    directions = []
    for value in raw:
        direction = np.asarray(value, dtype=np.float64)
        direction /= max(np.linalg.norm(direction), 1e-9)
        if float(direction @ outward) < 0.2 or direction[2] < -0.1:
            continue
        if not any(np.allclose(direction, existing, atol=1e-6) for existing in directions):
            directions.append(direction)
    return directions


def approach_direction_metadata(direction: np.ndarray) -> dict[str, float | str]:
    """Describe an object-frame approach direction for coverage accounting."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    azimuth = float(np.arctan2(direction[1], direction[0]))
    elevation = float(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    sector = int(np.floor(((azimuth + np.pi) % (2.0 * np.pi)) / (np.pi / 3.0)))
    label = ("back", "back_left", "front_left", "front", "front_right", "back_right")[
        sector
    ]
    elevation_label = "upper" if elevation >= np.deg2rad(15.0) else "level"
    return {
        "approach_azimuth": azimuth,
        "approach_elevation": elevation,
        "approach_bin": f"{label}_{elevation_label}",
    }


def plan_approach(
    cloud: Cloud,
    device: Device,
    candidate: Candidate,
    open_surface: Surface,
    surface_cache: dict[tuple[float, ...], Surface],
    *,
    seed: int,
    approach_waypoint_count: int = 10,
    grasp_waypoint_count: int = 7,
    clearance: float = 0.10,
    pregrasp_clearance: float = 0.05,
) -> tuple[ApproachPlan, ...]:
    """Find a free-space approach followed by a checked closing trajectory."""
    from source.grasping.search.hand_geometry import _open_fractions, surface_for
    open_fractions = _open_fractions(device)
    approach_progress = np.linspace(0.0, 1.0, approach_waypoint_count)
    approach_fractions = np.repeat(
        open_fractions[None, :],
        approach_waypoint_count,
        axis=0,
    )
    closing_progress = np.linspace(0.0, 1.0, grasp_waypoint_count)
    closing_fractions = (
        open_fractions[None, :]
        + closing_progress[:, None] * (candidate.surface.fractions - open_fractions)[None, :]
    )
    closing_surfaces = []
    for index, fractions in enumerate(closing_fractions):
        if index == 0:
            closing_surfaces.append(open_surface)
            continue
        if index + 1 == grasp_waypoint_count:
            closing_surfaces.append(candidate.surface)
            continue
        key = tuple(np.round(fractions, 8))
        if key not in surface_cache:
            surface_cache[key] = surface_for(
                device,
                fractions,
                seed=seed + 1_000 + len(surface_cache),
            )
        closing_surfaces.append(surface_cache[key])

    table_z = float(cloud.points[:, 2].min())
    directions = _approach_directions(candidate, seed)
    if not directions:
        return ()
    preferred = directions[0]
    ranked_plans: list[tuple[tuple, ApproachPlan]] = []

    for direction in directions:
        approach_distances = clearance + approach_progress * (pregrasp_clearance - clearance)
        approach_translations = (
            candidate.translation[None, :] + approach_distances[:, None] * direction
        )
        move_progress = np.linspace(0.0, 1.0, grasp_waypoint_count)
        move_translations = (
            candidate.translation[None, :]
            + (1.0 - move_progress[:, None]) * pregrasp_clearance * direction
        )
        # Construct the path in the collision-sensitive direction: start at
        # the stable final grasp, withdraw upward and outward, then reverse it
        # for execution. The sinusoidal lift leaves both endpoints unchanged
        # while preventing low straight-line sweeps across the table.
        reverse_arc_height = max(0.0, 0.035 * (1.0 - max(float(direction[2]), 0.0)))
        # pregrasp -> grasp execution must descend monotonically from the safe
        # lifted corridor; constructing the reverse withdrawal means height is
        # maximal at pregrasp and exactly zero at the validated final grasp.
        move_translations[:, 2] += reverse_arc_height * (1.0 - move_progress)
        maximum_penetration = 0.0
        minimum_object_clearance = np.inf
        minimum_table_clearance = np.inf
        for translation in approach_translations:
            posed = open_surface.points @ candidate.rotation.T + translation
            distances, _, signed = _signed_surface_distances(cloud, posed)
            maximum_penetration = max(
                maximum_penetration,
                float(np.maximum(-signed, 0.0).max()),
            )
            minimum_object_clearance = min(
                minimum_object_clearance,
                float(distances.min()),
            )
            minimum_table_clearance = min(
                minimum_table_clearance,
                _full_mesh_table_clearance(
                    open_surface,
                    candidate.rotation,
                    translation,
                    table_z,
                ),
            )
            if minimum_table_clearance < 0.005:
                break
        approach_blocked = minimum_table_clearance < 0.005
        pregrasp_translation = move_translations[0]
        final_translation = move_translations[-1]
        variants = (
            (
                np.concatenate(
                    [
                        np.repeat(
                            pregrasp_translation[None, :],
                            grasp_waypoint_count,
                            axis=0,
                        ),
                        move_translations[1:],
                    ]
                ),
                np.concatenate(
                    [
                        closing_fractions,
                        np.repeat(
                            candidate.surface.fractions[None, :],
                            grasp_waypoint_count - 1,
                            axis=0,
                        ),
                    ]
                ),
                [*closing_surfaces, *([candidate.surface] * (grasp_waypoint_count - 1))],
            ),
            (
                np.concatenate(
                    [
                        move_translations,
                        np.repeat(
                            final_translation[None, :],
                            grasp_waypoint_count - 1,
                            axis=0,
                        ),
                    ]
                ),
                np.concatenate(
                    [
                        np.repeat(
                            open_fractions[None, :],
                            grasp_waypoint_count,
                            axis=0,
                        ),
                        closing_fractions[1:],
                    ]
                ),
                [*([open_surface] * grasp_waypoint_count), *closing_surfaces[1:]],
            ),
            (
                move_translations,
                closing_fractions,
                closing_surfaces,
            ),
        )

        for variant_index, (
            grasp_translations,
            grasp_fractions,
            grasp_surfaces,
        ) in enumerate(variants):
            maximum_grasp_penetration = 0.0
            maximum_grasp_rigid_penetration = 0.0
            variant_table_clearance = minimum_table_clearance
            if not approach_blocked:
                for translation, surface in zip(
                    grasp_translations,
                    grasp_surfaces,
                    strict=True,
                ):
                    posed = surface.points @ candidate.rotation.T + translation
                    _, _, signed = _signed_surface_distances(cloud, posed)
                    contact_mask = np.isin(surface.labels, device.contact_labels)
                    rigid_mask = ~contact_mask
                    maximum_grasp_penetration = max(
                        maximum_grasp_penetration,
                        float(np.maximum(-signed[contact_mask], 0.0).max()),
                    )
                    if np.any(rigid_mask):
                        maximum_grasp_rigid_penetration = max(
                            maximum_grasp_rigid_penetration,
                            float(np.maximum(-signed[rigid_mask], 0.0).max()),
                        )
                    variant_table_clearance = min(
                        variant_table_clearance,
                        _full_mesh_table_clearance(
                            surface,
                            candidate.rotation,
                            translation,
                            table_z,
                        ),
                    )
                    if (
                        maximum_grasp_penetration > 0.004
                        or maximum_grasp_rigid_penetration > 0.0015
                        or variant_table_clearance < 0.005
                    ):
                        break
            blocked = (
                approach_blocked
                or maximum_grasp_penetration > 0.004
                or maximum_grasp_rigid_penetration > 0.0015
                or variant_table_clearance < 0.005
            )
            key = (
                blocked,
                maximum_grasp_rigid_penetration,
                maximum_grasp_penetration,
                maximum_penetration,
                max(0.005 - variant_table_clearance, 0.0),
                variant_index,
                1.0 - float(direction @ preferred),
            )
            ranked_plans.append(
                (
                    key,
                    ApproachPlan(
                        approach_translations=approach_translations,
                        approach_fractions=approach_fractions,
                        grasp_translations=grasp_translations,
                        grasp_fractions=grasp_fractions,
                        direction=direction,
                        maximum_penetration=maximum_penetration,
                        minimum_object_clearance=float(minimum_object_clearance),
                        maximum_grasp_penetration=maximum_grasp_penetration,
                        maximum_grasp_rigid_penetration=maximum_grasp_rigid_penetration,
                        minimum_table_clearance=float(variant_table_clearance),
                        collision_free=not blocked,
                    ),
                )
            )

    ranked_plans.sort(key=lambda item: item[0])
    return tuple(plan for _, plan in ranked_plans[:6])


def approach(candidate: Candidate, waypoint_count: int = 14) -> tuple[np.ndarray, np.ndarray]:
    if candidate.approach_plan is not None:
        return (
            candidate.approach_plan.approach_translations,
            candidate.approach_plan.approach_fractions,
        )
    progress = np.linspace(0.0, 1.0, waypoint_count)
    direction = candidate.rotation @ np.asarray([-1.0, 0.0, 0.0])
    direction[2] = max(direction[2], 0.35)
    direction /= np.linalg.norm(direction)
    translations = candidate.translation[None, :] + (1.0 - progress[:, None]) * 0.10 * direction
    if len(candidate.surface.fractions) == 1:
        open_fractions = np.ones(1, dtype=np.float64)
    else:
        open_fractions = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    fractions = np.repeat(open_fractions[None, :], waypoint_count, axis=0)
    return translations, fractions
