"""Point-cloud-only collision-free approach path search for Dex Hand."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from source.grasping.dex_hand_surface import load_posed_dex_hand_surface

if TYPE_CHECKING:
    from source.grasping.hand_closure_search import HandClosureResult
    from source.grasping.mesh_pointcloud import SurfacePointCloud


def plan_approach_path(
    cloud: SurfacePointCloud,
    grasp: HandClosureResult,
    *,
    seed: int,
    waypoint_count: int = 14,
    clearance: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return collision-free pose and hand-command waypoints."""
    # Lazy import avoids a runtime cycle with hand_closure_search.
    from source.grasping.hand_closure_search import geometry_metrics

    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least 2.")
    if clearance <= 0.0:
        raise ValueError("clearance must be positive.")
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
    tabletop_preference = palmward + 1.5 * np.asarray([0.0, 0.0, 1.0])
    tabletop_preference /= np.linalg.norm(tabletop_preference)
    axes = np.eye(3)
    directions = [palmward, tabletop_preference, *axes, *(-axes)]
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
        for index, hand in enumerate(posed_hands):
            remaining = 1.0 - progress[index]
            translation = grasp.translation + clearance * remaining * direction
            points = hand.points @ grasp.rotation_matrix.T + translation
            penetration, rigid_penetration, _, _ = geometry_metrics(
                points,
                hand.labels,
                cloud,
            )
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
