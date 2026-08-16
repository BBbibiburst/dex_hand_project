"""Grasp contact, force-closure, penetration, and clearance scoring."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from source.grasping.search.common import progress
from source.grasping.search.types import Candidate, Cloud, Device, Surface

def friction_wrenches(
    points: np.ndarray,
    inward_normals: np.ndarray,
    *,
    center_of_mass: np.ndarray,
    characteristic_length: float,
    friction: float = 0.8,
    edges: int = 12,
) -> np.ndarray:
    """Build a normalized 6-D grasp-wrench matrix.

    Forces are expressed on the object.  Torque rows are divided by a
    characteristic object length so force and torque residuals have comparable
    numerical scale.  Using torque about the object COM fixes the previous
    origin-dependent closure score.
    """
    columns = []
    length = max(float(characteristic_length), 1e-6)
    for point, normal in zip(points, inward_normals, strict=True):
        normal = np.asarray(normal, dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-9)
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(normal @ reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(normal, reference)
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        bitangent = np.cross(normal, tangent)
        arm = point - center_of_mass
        for angle in np.linspace(0.0, 2.0 * np.pi, edges, endpoint=False):
            force = normal + friction * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
            force /= max(np.linalg.norm(force), 1e-9)
            torque = np.cross(arm, force) / length
            columns.append(np.r_[force, torque])
    return np.asarray(columns, dtype=np.float64).T


def _normalized_nnls_residual(matrix: np.ndarray, target: np.ndarray) -> float:
    if matrix.size == 0:
        return 1.0
    _, residual = nnls(matrix, target)
    return float(residual / max(np.linalg.norm(target), 1e-9))


def grasp_equilibrium_metrics(
    cloud: Cloud,
    contact_points: np.ndarray,
    contact_normals: np.ndarray,
) -> tuple[float, float, float, float]:
    """Evaluate gravity support and true six-axis disturbance resistance.

    Returns ``(closure, gravity, worst_disturbance, normal_coverage)``.  The
    previous implementation only searched for a zero wrench in a convex cone;
    that could reward a hand merely supporting a sphere from below.  Here a
    valid grasp must generate the opposite wrench for gravity and for both
    signs of all three forces and all three torques.
    """
    if len(contact_points) < 2:
        return 1.0, 1.0, 1.0, 0.0

    try:
        center_of_mass = np.asarray(cloud.mesh.center_mass, dtype=np.float64)
        if center_of_mass.shape != (3,) or not np.all(np.isfinite(center_of_mass)):
            raise ValueError
    except Exception:
        center_of_mass = np.asarray(cloud.mesh.centroid, dtype=np.float64)

    radius = float(np.max(np.linalg.norm(cloud.points - center_of_mass, axis=1)))
    matrix = friction_wrenches(
        contact_points,
        contact_normals,
        center_of_mass=center_of_mass,
        characteristic_length=radius,
    )

    # Gravity acts along -Z, so the contacts must be able to create +Z.
    gravity_target = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    gravity_residual = _normalized_nnls_residual(matrix, gravity_target)

    residuals = []
    for axis in range(6):
        for sign in (-1.0, 1.0):
            target = np.zeros(6)
            target[axis] = sign
            residuals.append(_normalized_nnls_residual(matrix, target))
    disturbance_residual = float(max(residuals))
    closure_residual = float(np.sqrt(np.mean(np.square(residuals))))

    # A cheap geometric diagnostic: inward normals should cover both signs of
    # every spatial axis.  A one-sided bowl/support grasp scores near zero.
    normals = contact_normals / np.maximum(
        np.linalg.norm(contact_normals, axis=1, keepdims=True), 1e-9
    )
    directional = []
    for axis in np.eye(3):
        directional.append(float(np.max(normals @ axis)))
        directional.append(float(np.max(normals @ -axis)))
    normal_coverage = float(min(directional))
    return closure_residual, gravity_residual, disturbance_residual, normal_coverage


def _full_mesh_table_clearance(
    surface: Surface,
    rotation: np.ndarray,
    translation: np.ndarray,
    table_z: float,
) -> float:
    minimum = np.inf
    for vertices, _ in surface.meshes:
        posed = vertices @ rotation.T + translation
        minimum = min(minimum, float(posed[:, 2].min() - table_z))
    return float(minimum)


def _approach_table_clearance(
    surface: Surface,
    rotation: np.ndarray,
    translation: np.ndarray,
    table_z: float,
    waypoint_count: int = 10,
) -> float:
    # Use the same raised approach direction as the exported trajectory, but
    # inspect the complete collision meshes at every waypoint.
    direction = rotation @ np.asarray([-1.0, 0.0, 0.0])
    direction[2] = max(direction[2], 0.35)
    direction /= max(np.linalg.norm(direction), 1e-9)
    minimum = np.inf
    for progress in np.linspace(0.0, 1.0, waypoint_count):
        waypoint = translation + (1.0 - progress) * 0.10 * direction
        minimum = min(
            minimum,
            _full_mesh_table_clearance(surface, rotation, waypoint, table_z),
        )
    return float(minimum)


def _signed_surface_distances(
    cloud: Cloud,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest distances/indices and robust local signed distances."""
    neighbour_count = min(8, len(cloud.points))
    neighbour_distances, neighbour_indices = cloud.tree.query(
        points,
        k=neighbour_count,
    )
    if neighbour_count == 1:
        neighbour_distances = neighbour_distances[:, None]
        neighbour_indices = neighbour_indices[:, None]
    offsets = points[:, None, :] - cloud.points[neighbour_indices]
    projections = np.sum(offsets * cloud.normals[neighbour_indices], axis=2)
    signed = np.max(projections, axis=1)
    return neighbour_distances[:, 0], neighbour_indices[:, 0], signed


def _robot_execution_penalty(
    device: Device,
    contacts: tuple[int, ...],
    table_clearance: float,
) -> float:
    """Prefer contact-rich Dex poses with enough clearance for the robot wrist."""
    missing_contact_count = len(device.contact_labels) - len(contacts)
    contact_penalty = 0.04 * missing_contact_count
    clearance_penalty = 0.15 if device.name == "dex_hand" and table_clearance < 0.025 else 0.0
    return contact_penalty + clearance_penalty


def evaluate(
    cloud: Cloud,
    device: Device,
    surface: Surface,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    roll_index: int,
    anchor_index: int = -1,
    full_checks: bool = False,
) -> Candidate:
    posed = surface.points @ rotation.T + translation
    distances, indices, signed = _signed_surface_distances(cloud, posed)
    contact_mask = np.isin(surface.labels, device.contact_labels)
    rigid_mask = ~contact_mask
    penetration = float(np.maximum(-signed[contact_mask], 0.0).max())
    rigid = float(np.maximum(-signed[rigid_mask], 0.0).max()) if np.any(rigid_mask) else 0.0
    contacts, contact_points, contact_normals, per_label = [], [], [], []
    for label in device.contact_labels:
        selected = np.flatnonzero(surface.labels == label)
        closest = selected[int(np.argmin(distances[selected]))]
        per_label.append(float(distances[closest]))
        if distances[closest] <= 0.005:
            contacts.append(label)
            object_index = int(indices[closest])
            contact_points.append(cloud.points[object_index])
            contact_normals.append(-cloud.normals[object_index])
    contact_points_array = np.asarray(contact_points, dtype=np.float64).reshape(-1, 3)
    contact_normals_array = np.asarray(contact_normals, dtype=np.float64).reshape(-1, 3)
    if full_checks:
        (
            force_closure,
            gravity_balance_residual,
            disturbance_residual,
            normal_coverage,
        ) = grasp_equilibrium_metrics(cloud, contact_points_array, contact_normals_array)
    else:
        # The coarse stage is geometric. Fourteen NNLS solves per coarse pose
        # dominated runtime without contributing to the final validity check.
        force_closure = 0.0
        gravity_balance_residual = 0.0
        disturbance_residual = 0.0
        normal_coverage = 1.0

    table_z = float(cloud.points[:, 2].min())
    if full_checks:
        clearance = _full_mesh_table_clearance(surface, rotation, translation, table_z)
        approach_clearance = _approach_table_clearance(surface, rotation, translation, table_z)
    else:
        clearance = float(posed[:, 2].min() - table_z)
        approach_clearance = clearance

    opposing = (
        4 in contacts and any(label < 4 for label in contacts)
        if device.name == "dex_hand"
        else 0 in contacts and 1 in contacts
    )
    pika_normal_opposition = (
        len(contact_normals_array) == 2
        and float(contact_normals_array[0] @ contact_normals_array[1]) <= -0.5
    )
    mean_distance = float(np.mean(per_label))
    rejection_reasons = []
    if rigid > 0.0015:
        rejection_reasons.append("rigid_penetration")
    if penetration > 0.004:
        rejection_reasons.append("contact_penetration")
    if clearance < 0.005:
        rejection_reasons.append("table_clearance")
    if full_checks and approach_clearance < 0.005:
        rejection_reasons.append("approach_table_collision")
    if full_checks:
        if gravity_balance_residual > 0.18:
            rejection_reasons.append("gravity_unbalanced")
        if device.name == "dex_hand":
            if disturbance_residual > 0.32:
                rejection_reasons.append("insufficient_wrench_resistance")
            if normal_coverage < 0.08:
                rejection_reasons.append("one_sided_contacts")
            if force_closure > 0.24:
                rejection_reasons.append("force_closure")
        else:
            # A two-finger point-contact model cannot resist every six-axis
            # wrench. Judge Pika by achievable closure, opposition, and gravity.
            if force_closure > 0.45:
                rejection_reasons.append("poor_two_finger_closure")
            if opposing and not pika_normal_opposition:
                rejection_reasons.append("nonopposing_contact_normals")
    if not opposing:
        rejection_reasons.append("missing_opposition")
    valid = not rejection_reasons
    # Force-closure can be mathematically satisfied by only a thumb and two
    # fingertips, but those sparse contacts have little tolerance for the
    # millimetre-scale pose error of the full robot. Prefer poses that place
    # more of the available digits close to the object before trading that
    # coverage for a small amount of allowed fingertip penetration.
    score = (
        30.0 * penetration
        + 120.0 * rigid
        + 3.0 * mean_distance
        + _robot_execution_penalty(
            device,
            tuple(contacts),
            clearance,
        )
        + 0.8 * force_closure
        + 1.2 * gravity_balance_residual
        + 1.8 * disturbance_residual
        + 0.8 * max(0.08 - normal_coverage, 0.0)
        + max(0.005 - clearance, 0.0) * 80.0
        + max(0.005 - approach_clearance, 0.0) * 100.0
        + (0.0 if opposing else 1.5)
        + 0.5 * len(rejection_reasons)
    )
    return Candidate(
        surface,
        rotation,
        translation,
        posed,
        tuple(contacts),
        contact_points_array,
        contact_normals_array,
        penetration,
        rigid,
        mean_distance,
        force_closure,
        gravity_balance_residual,
        disturbance_residual,
        normal_coverage,
        clearance,
        approach_clearance,
        roll_index,
        score,
        valid,
        tuple(rejection_reasons),
        anchor_index,
    )
