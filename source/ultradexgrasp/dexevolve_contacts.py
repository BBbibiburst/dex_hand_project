"""Adaptive contact resampling for underactuated DexEvolve individuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from source.ultradexgrasp.catalog import ObjectGeometry
from source.ultradexgrasp.hand_surrogate import DexHandSurrogate


@dataclass(frozen=True)
class AdaptiveContactCommand:
    contact_fractions: np.ndarray
    command_delta: np.ndarray
    contact_points: np.ndarray
    contact_normals: np.ndarray
    contact_distances: np.ndarray
    hand_point_indices: np.ndarray
    distance_energy: float
    penetration_energy: float

    @property
    def grip_fractions(self) -> np.ndarray:
        return np.clip(self.contact_fractions + self.command_delta, 0.0, 1.0)


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic FPS, matching DexEvolve's spatial contact diversification."""
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    selected = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    minimum = np.linalg.norm(points - points[selected[0]], axis=1)
    for _ in range(1, count):
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(points - points[index], axis=1))
    return np.asarray(selected, dtype=np.int64)


def _surface(
    surrogate: DexHandSurrogate,
    fractions: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    return surrogate.evaluate_numpy(fractions) @ rotation.T + translation


def _actuator_jacobian(
    surrogate: DexHandSurrogate,
    fractions: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    point_indices: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    base = _surface(surrogate, fractions, translation, rotation)[point_indices]
    jacobian = np.empty((len(point_indices), 3, 6), dtype=np.float64)
    for drive in range(6):
        shifted = fractions.copy()
        direction = 1.0 if fractions[drive] <= 1.0 - epsilon else -1.0
        shifted[drive] = np.clip(shifted[drive] + direction * epsilon, 0.0, 1.0)
        actual_step = shifted[drive] - fractions[drive]
        moved = _surface(surrogate, shifted, translation, rotation)[point_indices]
        jacobian[:, :, drive] = (moved - base) / actual_step
    return jacobian


def resample_contact_command(
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    *,
    translation: np.ndarray,
    rotation: np.ndarray,
    contact_fractions: np.ndarray,
    contact_threshold: float = 0.012,
    active_contacts: int = 12,
    desired_travel: float = 0.010,
    regularization: float = 2e-3,
    maximum_delta: float = 0.45,
    jacobian_epsilon: float = 1e-3,
) -> AdaptiveContactCommand:
    """Ball-query contacts, FPS, then solve an actuator-space contact Jacobian.

    The official method solves in independent joint space.  Here the calibrated
    tendon surrogate differentiates skin motion with respect to the six real
    actuator coordinates, retaining the hand's passive coupling.
    """
    fractions = np.asarray(contact_fractions, dtype=np.float64)
    hand = _surface(surrogate, fractions, translation, rotation)
    delta = hand[:, None, :] - geometry.surface_points[None, :, :]
    squared = np.einsum("hoi,hoi->ho", delta, delta)
    nearest_object = np.argmin(squared, axis=1)
    nearest_distance = np.sqrt(squared[np.arange(len(hand)), nearest_object])
    potential = np.flatnonzero(nearest_distance <= contact_threshold)
    if not len(potential):
        potential = np.argsort(nearest_distance)[: min(active_contacts, len(hand))]
    if active_contacts < len(surrogate.contact_indices):
        raise ValueError("active_contacts must cover all five digits.")
    # DexEvolve's global FPS can otherwise spend the full budget on the four
    # opposed fingers. Reserve indices 0..4 for one nearest distal-skin point
    # per digit; index 4 is therefore always the thumb target.
    mandatory = np.asarray(
        [group[np.argmin(nearest_distance[group])] for group in surrogate.contact_indices],
        dtype=np.int64,
    )
    remaining = np.setdiff1d(potential, mandatory, assume_unique=False)
    fill_count = min(active_contacts - len(mandatory), len(remaining))
    if fill_count:
        fps = _farthest_point_indices(
            geometry.surface_points[nearest_object[remaining]], fill_count
        )
        hand_indices = np.concatenate([mandatory, remaining[fps]])
    else:
        hand_indices = mandatory
    object_indices = nearest_object[hand_indices]
    points = geometry.surface_points[object_indices]
    normals = geometry.surface_normals[object_indices]
    distances = nearest_distance[hand_indices]

    jacobian = _actuator_jacobian(
        surrogate,
        fractions,
        translation,
        rotation,
        hand_indices,
        jacobian_epsilon,
    ).reshape(-1, 6)
    # Move each selected skin point into the object along the inward surface
    # normal. Position control then turns the blocked displacement into force.
    target = (-desired_travel * normals).reshape(-1)
    system = jacobian.T @ jacobian + regularization * np.eye(6)
    command = np.linalg.solve(system, jacobian.T @ target)
    command = np.clip(command, -maximum_delta, maximum_delta)
    # The four finger flexors and thumb flexor are one-sided closing commands.
    command[[0, 1, 2, 3, 5]] = np.maximum(command[[0, 1, 2, 3, 5]], 0.0)
    command = np.minimum(command, 1.0 - fractions)

    plane_values = hand @ geometry.plane_normals.T - geometry.plane_offsets
    outside = plane_values.max(axis=1)
    penetration = np.maximum(-outside, 0.0)
    return AdaptiveContactCommand(
        contact_fractions=fractions.copy(),
        command_delta=command,
        contact_points=points.copy(),
        contact_normals=normals.copy(),
        contact_distances=distances.copy(),
        hand_point_indices=hand_indices.copy(),
        distance_energy=float(np.mean(distances)),
        penetration_energy=float(np.mean(penetration) + np.max(penetration)),
    )


def depenetrate_pose(
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    *,
    translation: np.ndarray,
    rotation: np.ndarray,
    fractions: np.ndarray,
    steps: int = 2,
    clearance: float = 5e-4,
) -> np.ndarray:
    """Apply the original method's short post-mutation penetration repair."""
    repaired = np.asarray(translation, dtype=np.float64).copy()
    for _ in range(steps):
        hand = _surface(surrogate, fractions, repaired, rotation)
        values = hand @ geometry.plane_normals.T - geometry.plane_offsets
        plane = np.argmax(values, axis=1)
        outside = values[np.arange(len(hand)), plane]
        penetrated = outside < clearance
        if not np.any(penetrated):
            break
        depths = clearance - outside[penetrated]
        directions = geometry.plane_normals[plane[penetrated]]
        shift = np.average(directions, axis=0, weights=depths)
        norm = float(np.linalg.norm(shift))
        if norm < 1e-9:
            break
        repaired += min(float(depths.max()), 0.010) * shift / norm
    return repaired
