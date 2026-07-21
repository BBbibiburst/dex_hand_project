#!/usr/bin/env python3
"""Production grasp-search implementation for Dex Hand and Pika.

This module keeps object loading, end-effector FK/mesh extraction, candidate
generation, geometric scoring, force-closure estimation, approach generation,
JSON output, and optional visualization in one implementation.

Run from the repository root, for example:

    python -m source.grasping.grasp_config_search \
      --object-id ycb:002_master_chef_can --viewer

The output uses the production grasp-config schema and can be inspected with:

    python -m tools.grasping.validate_grasp \
      configs/grasps/dex_hand/ycb_002_master_chef_can.json --viewer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from source.grasping.constants import (
    DEFAULT_GRIP_PRELOAD,
    GRASP_CONFIG_SCHEMA_VERSION,
    GRASP_SEARCH_STRATEGY,
)
from source.grasping.standalone_validator import (
    StandaloneValidationResult,
    validate_grasp_config,
    validate_grasp_trajectory_payload,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT
MANIFEST = ROOT / "assets" / "maniskill" / "manifest.json"
DEX_XML = ROOT / "assets" / "grippers" / "dex_hand" / "dex_hand.xml"
PIKA_XML = ROOT / "assets" / "grippers" / "pika_gripper" / "pika_gripper.xml"


LOGGER = logging.getLogger(__name__)


def progress(message: str) -> None:
    """Emit optional internal search diagnostics without polluting CLI output."""
    LOGGER.debug(message)


@dataclass(frozen=True)
class Device:
    name: str
    xml: Path
    root_body: str
    actuators: tuple[str, ...]
    contact_labels: tuple[int, ...]


DEVICES = {
    "dex_hand": Device(
        "dex_hand",
        DEX_XML,
        "hand_root",
        (
            "act_push_0_j",
            "act_push_1_j",
            "act_push_2_j",
            "act_push_3_j",
            "thumb_rotate_act_push_j",
            "thumb_grasp_act_push_j",
        ),
        (0, 1, 2, 3, 4),
    ),
    "pika_gripper": Device(
        "pika_gripper",
        PIKA_XML,
        "gripper_base_link",
        ("gripper_position",),
        (0, 1),
    ),
}


@dataclass
class Cloud:
    points: np.ndarray
    normals: np.ndarray
    center: np.ndarray
    scale: float
    mesh: trimesh.Trimesh
    tree: cKDTree


@dataclass
class Surface:
    points: np.ndarray
    labels: np.ndarray
    meshes: list[tuple[np.ndarray, np.ndarray]]
    actuator_values: np.ndarray
    fractions: np.ndarray
    midpoint: np.ndarray


@dataclass(frozen=True)
class ApproachPlan:
    approach_translations: np.ndarray
    approach_fractions: np.ndarray
    grasp_translations: np.ndarray
    grasp_fractions: np.ndarray
    direction: np.ndarray
    maximum_penetration: float
    minimum_object_clearance: float
    maximum_grasp_penetration: float
    maximum_grasp_rigid_penetration: float
    minimum_table_clearance: float
    collision_free: bool


@dataclass
class Candidate:
    surface: Surface
    rotation: np.ndarray
    translation: np.ndarray
    points: np.ndarray
    contacts: tuple[int, ...]
    contact_points: np.ndarray
    contact_normals: np.ndarray
    penetration: float
    rigid_penetration: float
    mean_distance: float
    force_closure: float
    gravity_balance_residual: float
    disturbance_residual: float
    normal_coverage: float
    table_clearance: float
    approach_table_clearance: float
    roll_index: int
    score: float
    valid: bool
    rejection_reasons: tuple[str, ...]
    anchor_index: int
    approach_plan: ApproachPlan | None = None
    approach_alternatives: tuple[ApproachPlan, ...] = ()


@dataclass(frozen=True)
class GraspConfigSearchResult:
    """Artifacts returned by the reusable grasp-search API."""

    output_path: Path
    mesh_path: Path
    cloud: Cloud
    candidates: tuple[Candidate, ...]
    config: dict
    published: bool

    @property
    def grasp(self) -> Candidate:
        """Return the selected best candidate."""
        return self.candidates[0]


@dataclass(frozen=True)
class ValidatedGraspConfigResult:
    """A grasp candidate that passed standalone dynamics validation."""

    output_path: Path
    selected_seed: int
    attempts_used: int
    validation: StandaloneValidationResult


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def grasp_config_name(object_id: str) -> str:
    """Return the canonical filesystem-safe name for an object grasp."""
    return safe_name(object_id)


def grasp_config_directory(
    end_effector_name: str,
    *,
    benchmark: bool = False,
) -> Path:
    """Return the canonical config directory for one end effector."""
    directory = ROOT / "configs" / "grasps" / end_effector_name
    return directory / "benchmark" if benchmark else directory


def grasp_benchmark_report_path(end_effector_name: str) -> Path:
    """Return the canonical grasp-catalog benchmark report path."""
    return grasp_config_directory(end_effector_name) / "grasp_catalog_benchmark.json"


def resolve_object(object_id: str) -> Path:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in payload["objects"]:
        key = f"{record['dataset']}:{record['object_id']}"
        if key != object_id:
            continue
        source = Path(record["source_path"])
        root = source if source.is_absolute() else ROOT / source
        files = record.get("model_files", ())
        preferred = next((name for name in files if Path(name).name == "textured.obj"), None)
        selected = preferred or next(
            (name for name in files if Path(name).suffix.lower() in {".obj", ".stl", ".ply"}),
            None,
        )
        if selected is None:
            break
        return root / selected
    raise ValueError(f"Unknown object or missing mesh: {object_id}")


def object_mesh_path(object_id: str) -> Path:
    """Resolve one manifest object to its preferred local triangle mesh."""
    return resolve_object(object_id)


def manifest_objects() -> list[tuple[str, Path]]:
    """Return all manifest objects that have a supported local mesh file."""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    objects: list[tuple[str, Path]] = []
    for record in payload.get("objects", []):
        object_id = f"{record['dataset']}:{record['object_id']}"
        source = Path(record["source_path"])
        root = source if source.is_absolute() else ROOT / source
        files = record.get("model_files", ())
        preferred = next(
            (name for name in files if Path(name).name == "textured.obj"),
            None,
        )
        selected = preferred or next(
            (name for name in files if Path(name).suffix.lower() in {".obj", ".stl", ".ply"}),
            None,
        )
        if selected is None:
            continue
        mesh_path = root / selected
        if mesh_path.exists():
            objects.append((object_id, mesh_path))
    objects.sort(key=lambda item: item[0])
    return objects


def load_cloud(path: Path, *, count: int, target_size: float, seed: int) -> Cloud:
    loaded = trimesh.load_mesh(path, process=True)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh in {path}")
    mesh = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(0) + vertices.max(0))
    scale = target_size / max(float(np.ptp(vertices, axis=0).max()), 1e-9)
    mesh.vertices = (vertices - center) * scale
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    points_array = np.asarray(points, dtype=np.float64)
    return Cloud(points_array, normals, center, scale, mesh, cKDTree(points_array))


def mesh_vertices(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)


def mesh_faces(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    start, count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    return np.asarray(model.mesh_face[start : start + count], dtype=np.int64)


def geom_label(device: Device, name: str) -> int:
    if device.name == "pika_gripper":
        if "left_link" in name:
            return 0
        if "right_link" in name:
            return 1
        return 2
    if "skin_palm" in name:
        return 5
    for finger in range(5):
        if f"skin_{finger}_" in name:
            return finger
    return 6


def surface_for(device: Device, fractions: np.ndarray, *, seed: int) -> Surface:
    model = mujoco.MjModel.from_xml_path(str(device.xml))
    data = mujoco.MjData(model)
    values = np.empty(len(device.actuators))
    for index, (name, fraction) in enumerate(zip(device.actuators, fractions, strict=True)):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        low, high = model.actuator_ctrlrange[actuator]
        if device.name == "pika_gripper":
            value = np.clip(low + 0.05 * float(fraction), low, high)
        else:
            value = low + float(fraction) * (high - low)
        data.ctrl[actuator] = values[index] = value
    for _ in range(600):
        mujoco.mj_step(model, data)

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, device.root_body)
    root_pos = data.xpos[root].copy()
    root_rot = data.xmat[root].reshape(3, 3).copy()
    rng = np.random.default_rng(seed)
    point_groups, label_groups, meshes = [], [], []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or not name:
            continue
        if device.name == "pika_gripper" and not name.endswith("_collision"):
            continue
        mesh_id = int(model.geom_dataid[geom])
        vertices = mesh_vertices(model, mesh_id)
        rotation = data.geom_xmat[geom].reshape(3, 3)
        local = (vertices @ rotation.T + data.geom_xpos[geom] - root_pos) @ root_rot
        faces = mesh_faces(model, mesh_id)
        meshes.append((local, faces))
        selected = local
        if len(selected) > 350:
            selected = selected[rng.choice(len(selected), 350, replace=False)]
        point_groups.append(selected)
        label_groups.append(np.full(len(selected), geom_label(device, name), dtype=int))
    points = np.concatenate(point_groups)
    labels = np.concatenate(label_groups)
    if device.name == "pika_gripper":
        midpoint = 0.5 * (points[labels == 0].mean(0) + points[labels == 1].mean(0))
    else:
        finger = np.concatenate([points[labels == i] for i in range(4)]).mean(0)
        midpoint = 0.5 * (finger + points[labels == 4].mean(0))
    return Surface(points, labels, meshes, values, fractions.copy(), midpoint)


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
    clearance_penalty = (
        0.15
        if device.name == "dex_hand" and table_clearance < 0.025
        else 0.0
    )
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


def fraction_candidates(device: Device, count: int) -> list[np.ndarray]:
    progress = np.linspace(0.12, 0.92, count)
    if device.name == "pika_gripper":
        return [np.asarray([value]) for value in progress[::-1]]
    candidates = []
    for value in progress:
        candidates.append(np.asarray([value, value, value, value, 1.0, value]))
        candidates.append(np.asarray([value, value, 0.5 * value, 0.5 * value, 1.0, value]))
    return candidates


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


def _retain(bucket: list[Candidate], candidate: Candidate, keep: int) -> None:
    bucket.append(candidate)
    bucket.sort(key=lambda item: (not item.valid, item.score))
    del bucket[keep:]


def _open_fractions(device: Device) -> np.ndarray:
    if device.name == "pika_gripper":
        return np.ones(1, dtype=np.float64)
    return np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


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


def search(
    cloud: Cloud,
    device: Device,
    *,
    joint_candidates: int,
    anchor_count: int,
    rolls_per_anchor: int,
    coarse_keep: int,
    top_k: int,
    support_margin: float,
    seed: int,
) -> list[Candidate]:
    all_fractions = fraction_candidates(device, max(3, joint_candidates // 16))
    coarse_stride = max(1, len(all_fractions) // 8)
    coarse_fractions = all_fractions[::coarse_stride]
    if all_fractions[-1] is not coarse_fractions[-1]:
        coarse_fractions.append(all_fractions[-1])
    coarse_rolls = max(2, rolls_per_anchor // 2)
    poses = local_pose_candidates(
        cloud,
        anchor_count=anchor_count,
        rolls_per_anchor=coarse_rolls,
        support_margin=support_margin,
        seed=seed,
    )
    coarse_depths = (-0.018, -0.006, 0.006)
    estimated = len(coarse_fractions) * len(poses) * len(coarse_depths)
    progress(
        f"[coarse] hand_shapes={len(coarse_fractions)} poses={len(poses)} "
        f"depths={len(coarse_depths)} evaluations={estimated}"
    )
    coarse: list[Candidate] = []
    progress_step = max(1, estimated // 10)
    evaluated = 0
    for fraction_index, fraction in enumerate(coarse_fractions):
        progress(f"[coarse] building hand shape {fraction_index + 1}/{len(coarse_fractions)}")
        shape_started = time.perf_counter()
        surface = surface_for(device, fraction, seed=seed + fraction_index)
        progress(
            f"[coarse] hand shape {fraction_index + 1}/{len(coarse_fractions)} ready "
            f"({time.perf_counter() - shape_started:.1f}s, points={len(surface.points)})"
        )
        for anchor_index, roll_index, rotation, grasp_center in poses:
            base_translation = grasp_center - surface.midpoint @ rotation.T
            for depth in coarse_depths:
                candidate = evaluate(
                    cloud,
                    device,
                    surface,
                    rotation,
                    base_translation + rotation[:, 0] * depth,
                    roll_index=roll_index,
                    anchor_index=anchor_index,
                    full_checks=False,
                )
                _retain(coarse, candidate, max(1, coarse_keep))
                evaluated += 1
                if evaluated % progress_step == 0 or evaluated == estimated:
                    best = coarse[0]
                    progress(
                        f"[coarse] {evaluated}/{estimated} best={best.score:.4f} valid={best.valid}"
                    )

    progress(f"[fine] refining {len(coarse)} coarse seeds")
    fine: list[Candidate] = []
    angle_offsets = np.deg2rad((-6.0, 0.0, 6.0))
    depth_offsets = (-0.004, 0.0, 0.004)
    lateral_offsets = (-0.003, 0.0, 0.003)
    fine_total = len(coarse) * len(angle_offsets) * len(depth_offsets) * len(lateral_offsets)
    fine_step = max(1, fine_total // 10)
    evaluated = 0
    for seed_index, coarse_candidate in enumerate(coarse):
        for angle in angle_offsets:
            local_delta = Rotation.from_rotvec(np.array([angle, 0.0, 0.0])).as_matrix()
            rotation = coarse_candidate.rotation @ local_delta
            for depth in depth_offsets:
                for lateral in lateral_offsets:
                    translation = (
                        coarse_candidate.translation
                        + rotation[:, 0] * depth
                        + rotation[:, 1] * lateral
                    )
                    candidate = evaluate(
                        cloud,
                        device,
                        coarse_candidate.surface,
                        rotation,
                        translation,
                        roll_index=coarse_candidate.roll_index,
                        anchor_index=coarse_candidate.anchor_index,
                        full_checks=True,
                    )
                    _retain(fine, candidate, max(1, top_k, coarse_keep))
                    evaluated += 1
                    if evaluated % fine_step == 0 or evaluated == fine_total:
                        best = fine[0]
                        progress(
                            f"[fine] {evaluated}/{fine_total} "
                            f"best={best.score:.4f} valid={best.valid}"
                        )

    # Always preserve at least one result for visualization and debugging.
    selected = fine or coarse
    if not selected:
        raise RuntimeError("No candidate was evaluated.")
    selected.sort(key=lambda item: (not item.valid, item.score))
    valid_count = sum(item.valid for item in selected)
    progress(
        f"[search] saved={len(selected[: max(1, top_k)])} "
        f"valid={valid_count} fallback={valid_count == 0}"
    )
    return selected[: max(1, top_k)]


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


def candidate_summary(candidate: Candidate) -> dict:
    return {
        "score": candidate.score,
        "valid": candidate.valid,
        "rejection_reasons": list(candidate.rejection_reasons),
        "anchor_index": candidate.anchor_index,
        "roll_index": candidate.roll_index,
        "translation": candidate.translation.tolist(),
        "rotation_matrix": candidate.rotation.tolist(),
        "actuator_fractions": candidate.surface.fractions.tolist(),
        "contacts": list(candidate.contacts),
        "contact_points": candidate.contact_points.tolist(),
        "contact_normals": candidate.contact_normals.tolist(),
        "penetration": candidate.penetration,
        "rigid_penetration": candidate.rigid_penetration,
        "mean_contact_distance": candidate.mean_distance,
        "force_closure_residual": candidate.force_closure,
        "gravity_balance_residual": candidate.gravity_balance_residual,
        "worst_disturbance_residual": candidate.disturbance_residual,
        "contact_normal_coverage": candidate.normal_coverage,
        "table_clearance": candidate.table_clearance,
        "approach_table_clearance": candidate.approach_table_clearance,
        "approach_planned": candidate.approach_plan is not None,
        "approach_maximum_penetration": (
            candidate.approach_plan.maximum_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "approach_minimum_object_clearance": (
            candidate.approach_plan.minimum_object_clearance
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_maximum_penetration": (
            candidate.approach_plan.maximum_grasp_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_maximum_rigid_penetration": (
            candidate.approach_plan.maximum_grasp_rigid_penetration
            if candidate.approach_plan is not None
            else None
        ),
    }


def payload(
    object_id: str | None,
    mesh_path: Path,
    cloud: Cloud,
    device: Device,
    candidates: list[Candidate],
) -> dict:
    candidate = candidates[0]
    translations, fractions = approach(candidate)
    if candidate.approach_plan is not None:
        grasp_translations = candidate.approach_plan.grasp_translations
        grasp_fractions = candidate.approach_plan.grasp_fractions
    else:
        grasp_translations = candidate.translation[None, :]
        grasp_fractions = candidate.surface.fractions[None, :]
    opposing = (
        4 in candidate.contacts and any(label < 4 for label in candidate.contacts)
        if device.name == "dex_hand"
        else 0 in candidate.contacts and 1 in candidate.contacts
    )
    success = candidate.valid and opposing
    preload_directions = (
        np.ones(len(device.actuators))
        if device.name == "dex_hand"
        else -np.ones(len(device.actuators))
    )
    preload_weights = np.ones(len(device.actuators))
    if device.name == "dex_hand":
        preload_weights[4] = 0.0
    return {
        "schema_version": GRASP_CONFIG_SCHEMA_VERSION,
        "search_strategy": GRASP_SEARCH_STRATEGY,
        "object_id": object_id,
        "end_effector_name": device.name,
        "mesh": str(mesh_path),
        "mesh_center": cloud.center.tolist(),
        "mesh_scale": cloud.scale,
        "object_table_height": float(cloud.points[:, 2].min()),
        "contact_points": candidate.contact_points.tolist(),
        "contact_normals": candidate.contact_normals.tolist(),
        "hand_actuator_fractions": candidate.surface.fractions.tolist(),
        "hand_actuator_values": candidate.surface.actuator_values.tolist(),
        "hand_preload_directions": preload_directions.tolist(),
        "hand_preload_weights": preload_weights.tolist(),
        "hand_translation": candidate.translation.tolist(),
        "hand_rotation_matrix": candidate.rotation.tolist(),
        "hand_mean_actuator_fraction": float(np.mean(candidate.surface.fractions)),
        "hand_maximum_penetration": candidate.penetration,
        "hand_maximum_noncontact_penetration": candidate.rigid_penetration,
        "hand_mean_contact_distance": candidate.mean_distance,
        "hand_contacting_fingers": list(candidate.contacts),
        "hand_force_closure_residual": candidate.force_closure,
        "hand_gravity_balance_residual": candidate.gravity_balance_residual,
        "hand_worst_disturbance_residual": candidate.disturbance_residual,
        "hand_contact_normal_coverage": candidate.normal_coverage,
        "hand_table_clearance": candidate.table_clearance,
        "approach_minimum_table_clearance": candidate.approach_table_clearance,
        "approach_maximum_object_penetration": (
            candidate.approach_plan.maximum_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "approach_minimum_object_clearance": (
            candidate.approach_plan.minimum_object_clearance
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_trajectory_maximum_penetration": (
            candidate.approach_plan.maximum_grasp_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "grasp_trajectory_maximum_rigid_penetration": (
            candidate.approach_plan.maximum_grasp_rigid_penetration
            if candidate.approach_plan is not None
            else None
        ),
        "approach_direction": (
            candidate.approach_plan.direction.tolist()
            if candidate.approach_plan is not None
            else None
        ),
        "hand_orientation_roll_index": candidate.roll_index,
        "hand_contact_distance_margin": max(0.0, 0.005 - candidate.mean_distance),
        "approach_hand_translations": translations.tolist(),
        "approach_hand_rotation_matrices": np.repeat(
            candidate.rotation[None, :, :], len(translations), axis=0
        ).tolist(),
        "approach_hand_actuator_fractions": fractions.tolist(),
        "grasp_hand_translations": grasp_translations.tolist(),
        "grasp_hand_rotation_matrices": np.repeat(
            candidate.rotation[None, :, :],
            len(grasp_translations),
            axis=0,
        ).tolist(),
        "grasp_hand_actuator_fractions": grasp_fractions.tolist(),
        "hand_fit_success": success,
        "search_debug_fallback_used": not candidate.valid,
        "search_candidate_count_saved": len(candidates),
        "search_candidates": [candidate_summary(item) for item in candidates],
    }


def _search_failure_detail(candidate: Candidate) -> str:
    reasons = ", ".join(candidate.rejection_reasons)
    return reasons or f"score={candidate.score:.4f}"


def select_executable_config(
    object_id: str | None,
    mesh_path: Path,
    cloud: Cloud,
    device: Device,
    candidates: list[Candidate],
    *,
    seed: int,
) -> dict:
    """Select the first analytically valid candidate with an exact free approach."""
    open_surface = surface_for(device, _open_fractions(device), seed=seed + 50_000)
    surface_cache = {
        tuple(np.round(_open_fractions(device), 8)): open_surface,
    }
    for candidate_index, candidate in enumerate(candidates):
        if not candidate.valid:
            continue
        alternatives = plan_approach(
            cloud,
            device,
            candidate,
            open_surface,
            surface_cache,
            seed=seed + 60_000 + candidate_index,
        )
        if not alternatives:
            candidate.valid = False
            candidate.rejection_reasons = (
                *candidate.rejection_reasons,
                "approach_object_collision",
            )
            candidate.score += 2.0
            continue
        candidate.approach_alternatives = alternatives
        candidate.approach_plan = alternatives[0]
        alternatives = candidate.approach_alternatives or (
            (candidate.approach_plan,) if candidate.approach_plan is not None else ()
        )
        ordered = [candidate, *(item for item in candidates if item is not candidate)]
        for plan in alternatives:
            candidate.approach_plan = plan
            candidate_payload = payload(
                object_id,
                mesh_path,
                cloud,
                device,
                ordered,
            )
            try:
                validate_grasp_trajectory_payload(candidate_payload)
            except ValueError:
                continue
            candidate.approach_table_clearance = plan.minimum_table_clearance
            candidates[:] = ordered
            return candidate_payload
        candidate.valid = False
        candidate.rejection_reasons = (
            *candidate.rejection_reasons,
            "mujoco_approach_collision",
        )
        candidate.score += 2.0
    candidates.sort(key=lambda item: (not item.valid, item.score))
    return payload(object_id, mesh_path, cloud, device, candidates)


def search_grasp_config(
    *,
    object_id: str | None = None,
    mesh: str | Path | None = None,
    output: str | Path | None = None,
    points: int = 2048,
    joint_candidates: int = 128,
    surface_anchors: int = 24,
    rolls_per_anchor: int = 8,
    coarse_keep: int = 24,
    top_k: int = 8,
    support_margin: float = 0.008,
    seed: int = 0,
    target_size: float = 0.09,
    end_effector_name: str = "dex_hand",
    require_valid: bool = True,
    publish_invalid: bool = False,
) -> GraspConfigSearchResult:
    """Run the new two-stage search and write a production-schema grasp config."""
    if (object_id is None) == (mesh is None):
        raise ValueError("Provide exactly one of object_id or mesh.")
    if points <= 0:
        raise ValueError("points must be positive.")
    for name, value in (
        ("joint_candidates", joint_candidates),
        ("surface_anchors", surface_anchors),
        ("rolls_per_anchor", rolls_per_anchor),
        ("coarse_keep", coarse_keep),
        ("top_k", top_k),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if support_margin < 0.0:
        raise ValueError("support_margin must be non-negative.")
    if target_size <= 0.0:
        raise ValueError("target_size must be positive.")
    try:
        device = DEVICES[end_effector_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported end effector {end_effector_name!r}; available={tuple(DEVICES)}."
        ) from exc

    mesh_path = resolve_object(object_id) if mesh is None else Path(mesh)
    if not mesh_path.is_absolute():
        mesh_path = ROOT / mesh_path
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Object mesh does not exist: {mesh_path}")
    name = "custom_mesh" if object_id is None else grasp_config_name(object_id)
    output_path = (
        grasp_config_directory(end_effector_name) / f"{name}.json"
        if output is None
        else Path(output)
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    progress(f"[setup] loading object: {mesh_path}")
    cloud = load_cloud(
        mesh_path,
        count=points,
        target_size=target_size,
        seed=seed,
    )
    candidates = search(
        cloud,
        device,
        joint_candidates=joint_candidates,
        anchor_count=surface_anchors,
        rolls_per_anchor=rolls_per_anchor,
        coarse_keep=coarse_keep,
        top_k=top_k,
        support_margin=support_margin,
        seed=seed,
    )
    config = select_executable_config(
        object_id,
        mesh_path,
        cloud,
        device,
        candidates,
        seed=seed,
    )
    result = GraspConfigSearchResult(
        output_path=output_path,
        mesh_path=mesh_path,
        cloud=cloud,
        candidates=tuple(candidates),
        config=config,
        published=False,
    )
    if require_valid and not config["hand_fit_success"]:
        raise RuntimeError(
            f"No valid grasp was found for {object_id or mesh_path!r}: "
            f"{_search_failure_detail(result.grasp)}."
        )
    published = bool(config["hand_fit_success"] or publish_invalid)
    if published:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
        temporary_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary_path, output_path)
        progress(f"[output] wrote {output_path}")
    return GraspConfigSearchResult(
        output_path=output_path,
        mesh_path=mesh_path,
        cloud=cloud,
        candidates=tuple(candidates),
        config=config,
        published=published,
    )


def generate_grasp_config(
    object_id: str,
    *,
    output: str | Path | None = None,
    **search_kwargs,
) -> Path:
    """Generate one object config and return its cached path."""
    return search_grasp_config(
        object_id=object_id,
        output=output,
        **search_kwargs,
    ).output_path


def generate_validated_grasp_config(
    object_id: str,
    *,
    output: str | Path | None = None,
    attempts: int = 3,
    validation_seconds: float = 3.0,
    settle_seconds: float = 0.8,
    grip_preload: float = DEFAULT_GRIP_PRELOAD,
    **search_kwargs,
) -> ValidatedGraspConfigResult:
    """Publish the first new-search candidate that is dynamically stable."""
    if attempts <= 0:
        raise ValueError("attempts must be positive.")
    end_effector_name = str(search_kwargs.get("end_effector_name", "dex_hand"))
    name = grasp_config_name(object_id)
    output_path = (
        grasp_config_directory(end_effector_name) / f"{name}.json"
        if output is None
        else Path(output)
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".candidate")
    failures: list[str] = []

    try:
        base_seed = int(search_kwargs.pop("seed", 0))
        for attempt in range(attempts):
            candidate_seed = base_seed + attempt
            try:
                search_grasp_config(
                    object_id=object_id,
                    output=temporary_path,
                    seed=candidate_seed,
                    **search_kwargs,
                )
                validation = validate_grasp_config(
                    temporary_path,
                    seconds=validation_seconds,
                    settle_seconds=settle_seconds,
                    grip_preload=grip_preload,
                )
            except Exception as exc:
                failures.append(f"seed={candidate_seed}: {exc}")
                continue
            if not validation.stable:
                failures.append(
                    f"seed={candidate_seed}: unstable "
                    f"drift={validation.position_drift:.4f}m "
                    f"rotation={validation.rotation_drift:.3f}rad "
                    f"drop={validation.vertical_drop:.4f}m "
                    f"contacts={validation.final_contacts}"
                )
                continue
            os.replace(temporary_path, output_path)
            return ValidatedGraspConfigResult(
                output_path=output_path,
                selected_seed=candidate_seed,
                attempts_used=attempt + 1,
                validation=validation,
            )
    finally:
        temporary_path.unlink(missing_ok=True)
    detail = " | ".join(failures) or "no candidates evaluated"
    raise RuntimeError(f"No dynamically stable grasp was found for {object_id!r}: {detail}")


def draw(cloud: Cloud, candidate: Candidate, *, output: Path | None, show: bool) -> None:
    """Visualize the object and end effector as lightweight point clouds.

    The search still uses the original sampled geometry.  This function only
    changes rendering, so visualization density has no effect on grasp scores.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")

    # Object: point-cloud rendering avoids the jagged appearance and interaction
    # cost of a heavily decimated triangle mesh.
    object_limit = 1_200
    object_stride = max(1, (len(cloud.points) + object_limit - 1) // object_limit)
    object_points = cloud.points[::object_stride]
    axis.scatter(
        *object_points.T,
        s=6,
        alpha=0.48,
        color="#5b87ad",
        linewidths=0.0,
        depthshade=False,
        label="object point cloud",
    )

    # Hand: render each semantic region separately so the palm and fingers remain
    # recognizable even when points overlap in the projected view.
    labels = candidate.surface.labels
    posed_points = candidate.points
    unique_labels = sorted(int(value) for value in np.unique(labels))
    is_pika = len(candidate.surface.fractions) == 1
    body_label = 2 if is_pika else 6
    palm_label = None if is_pika else 5
    hand_colors = {
        0: "#ef4444",
        1: "#8b5cf6",
        2: "#06b6d4",
        3: "#22c55e",
        4: "#eab308",
        5: "#d97706",
        6: "#6b7280",
    }
    hand_names = {
        0: "finger 0",
        1: "finger 1",
        2: "finger 2",
        3: "finger 3",
        4: "thumb",
        5: "palm",
        6: "hand body",
    }
    if is_pika:
        hand_colors[2] = "#6b7280"
        hand_names.update({0: "left finger", 1: "right finger", 2: "gripper body"})
    for label in unique_labels:
        region = posed_points[labels == label]
        if not len(region):
            continue
        # Contact regions retain the most detail. The non-contact hand body is
        # deliberately sparse and translucent so it cannot hide the fingers.
        region_limit = 350 if label == body_label else 500 if label == palm_label else 650
        stride = max(1, (len(region) + region_limit - 1) // region_limit)
        region = region[::stride]
        axis.scatter(
            *region.T,
            s=5 if label == body_label else 9 if label == palm_label else 11,
            alpha=0.22 if label == body_label else 0.62 if label == palm_label else 0.88,
            color=hand_colors.get(label, "#6b7280"),
            linewidths=0.0,
            depthshade=False,
            label=hand_names.get(label, f"hand region {label}"),
        )

    if len(candidate.contact_points):
        axis.scatter(
            *candidate.contact_points.T,
            s=95,
            color="#111827",
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
            label="contacts",
        )
        axis.quiver(
            *candidate.contact_points.T,
            *candidate.contact_normals.T,
            length=0.02,
            color="#111827",
            linewidth=1.4,
        )

    translations, _ = approach(candidate)
    axis.plot(*translations.T, color="#2ca02c", linewidth=2.5, label="approach")
    axis.scatter(
        *translations[0],
        s=45,
        marker="o",
        color="#2ca02c",
        depthshade=False,
        label="pregrasp",
    )
    if candidate.approach_plan is not None:
        axis.plot(
            *candidate.approach_plan.grasp_translations.T,
            color="#dc2626",
            linewidth=2.5,
            label="checked closing trajectory",
        )

    # Draw the inferred support plane as a wire grid.  It is cheap and makes
    # table-clearance failures much easier to understand.
    visible = np.concatenate([cloud.points, candidate.points, translations])
    low, high = visible.min(0), visible.max(0)
    center = 0.5 * (low + high)
    radius = max(0.01, 0.55 * float(np.ptp(visible, axis=0).max()))
    table_z = float(cloud.points[:, 2].min())
    grid_values = np.linspace(-radius, radius, 7)
    for offset in grid_values:
        axis.plot(
            [center[0] - radius, center[0] + radius],
            [center[1] + offset, center[1] + offset],
            [table_z, table_z],
            color="#6b7280",
            alpha=0.18,
            linewidth=0.7,
        )
        axis.plot(
            [center[0] + offset, center[0] + offset],
            [center[1] - radius, center[1] + radius],
            [table_z, table_z],
            color="#6b7280",
            alpha=0.18,
            linewidth=0.7,
        )

    axis.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(min(table_z - 0.01, center[2] - radius), center[2] + radius),
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title=(
            f"point-cloud grasp view {candidate.surface.fractions.tolist()}\n"
            f"score={candidate.score:.3f}, Efc={candidate.force_closure:.3f}, "
            f"Eg={candidate.gravity_balance_residual:.3f}, "
            f"Eworst={candidate.disturbance_residual:.3f}, valid={candidate.valid}"
        ),
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--object-id",
        help="Manifest object id. Omit both --object-id and --mesh to process all objects.",
    )
    source.add_argument("--mesh", type=Path, help="Process one custom mesh.")
    parser.add_argument(
        "--end-effector",
        choices=tuple(DEVICES),
        default="dex_hand",
    )
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--joint-candidates", type=int, default=128)
    parser.add_argument("--surface-anchors", type=int, default=24)
    parser.add_argument("--rolls-per-anchor", type=int, default=8)
    parser.add_argument("--coarse-keep", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--support-margin", type=float, default=0.008)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Single mode: output JSON file. Batch mode: output directory. "
            "Defaults to configs/grasps/END_EFFECTOR."
        ),
    )
    parser.add_argument(
        "--preview-image",
        type=Path,
        help="Single-object preview image path (also enables image saving).",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save a PNG for every processed object; in batch mode all images are saved.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory for saved PNG previews. Defaults to OUTPUT_DIR/previews.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the interactive viewer (batch mode opens one window per object).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N manifest objects; useful for testing batch mode.",
    )
    return parser.parse_args()


def default_output_dir(device: Device) -> Path:
    return grasp_config_directory(device.name)


def process_object(
    *,
    object_id: str | None,
    mesh_path: Path,
    device: Device,
    args: argparse.Namespace,
    output_path: Path,
    image_path: Path | None,
    seed: int,
    item_index: int,
    item_count: int,
) -> dict:
    label = object_id or mesh_path.stem
    progress(f"[batch {item_index}/{item_count}] begin {label}")
    progress(f"[setup] loading object: {mesh_path}")
    cloud = load_cloud(
        mesh_path,
        count=args.points,
        target_size=args.target_size,
        seed=seed,
    )
    candidates = search(
        cloud,
        device,
        joint_candidates=args.joint_candidates,
        anchor_count=args.surface_anchors,
        rolls_per_anchor=args.rolls_per_anchor,
        coarse_keep=args.coarse_keep,
        top_k=args.top_k,
        support_margin=args.support_margin,
        seed=seed,
    )
    candidate = candidates[0]
    result = select_executable_config(
        object_id,
        mesh_path,
        cloud,
        device,
        candidates,
        seed=seed,
    )
    candidate = candidates[0]
    published = bool(result["hand_fit_success"])
    if published:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
        temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        os.replace(temporary_path, output_path)
        progress(f"[output] wrote {output_path}")

    if image_path is not None or args.viewer:
        if image_path is not None:
            progress(f"[image] rendering {image_path}")
        draw(cloud, candidate, output=image_path, show=args.viewer)
        if image_path is not None:
            progress(f"[image] wrote {image_path}")

    progress(
        f"[batch {item_index}/{item_count}] done {label}: "
        f"score={candidate.score:.4f} contacts={candidate.contacts} "
        f"Efc={candidate.force_closure:.4f} "
        f"Eg={candidate.gravity_balance_residual:.4f} "
        f"Eworst={candidate.disturbance_residual:.4f} "
        f"fit={result['hand_fit_success']}"
    )
    return {
        "object_id": object_id,
        "mesh": str(mesh_path),
        "output": str(output_path) if published else None,
        "image": str(image_path) if image_path is not None else None,
        "success": bool(result["hand_fit_success"]),
        "fallback_used": bool(result["search_debug_fallback_used"]),
        "score": float(candidate.score),
        "contacts": list(candidate.contacts),
        "error": None,
    }


def main() -> None:
    args = parse_args()
    device = DEVICES[args.end_effector]
    batch_mode = args.object_id is None and args.mesh is None
    progress(
        "[setup] starting two-stage grasp search "
        f"mode={'batch' if batch_mode else 'single'} end_effector={device.name}"
    )

    if batch_mode:
        items = manifest_objects()
        if args.limit is not None:
            items = items[: max(0, args.limit)]
        if not items:
            raise RuntimeError(f"No usable mesh entries found in {MANIFEST}")
        output_dir = args.output or default_output_dir(device)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_images = bool(args.save_images or args.preview_image or args.image_dir)
        image_dir = args.image_dir or (output_dir / "previews")
        if save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        progress(
            f"[batch] discovered={len(items)} output_dir={output_dir} save_images={save_images}"
        )

        records: list[dict] = []
        for index, (object_id, mesh_path) in enumerate(items, start=1):
            name = safe_name(object_id)
            output_path = output_dir / f"{name}.json"
            image_path = image_dir / f"{name}.png" if save_images else None
            try:
                record = process_object(
                    object_id=object_id,
                    mesh_path=mesh_path,
                    device=device,
                    args=args,
                    output_path=output_path,
                    image_path=image_path,
                    seed=args.seed + index - 1,
                    item_index=index,
                    item_count=len(items),
                )
            except Exception as exc:  # keep the remaining batch alive
                progress(
                    f"[batch {index}/{len(items)}] ERROR {object_id}: {type(exc).__name__}: {exc}"
                )
                record = {
                    "object_id": object_id,
                    "mesh": str(mesh_path),
                    "output": str(output_path),
                    "image": str(image_path) if image_path is not None else None,
                    "success": False,
                    "fallback_used": False,
                    "score": None,
                    "contacts": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)

        succeeded = sum(item["success"] for item in records)
        failed_fit = sum(item["error"] is None and not item["success"] for item in records)
        errors = sum(item["error"] is not None for item in records)
        summary = {
            "mode": "batch",
            "end_effector": device.name,
            "total": len(records),
            "successful_grasps": succeeded,
            "failed_fits": failed_fit,
            "errors": errors,
            "images_saved": save_images,
            "records": records,
        }
        summary_path = output_dir / "batch_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        progress(
            f"[batch] complete total={len(records)} success={succeeded} "
            f"failed_fit={failed_fit} errors={errors}"
        )
        progress(f"[batch] summary={summary_path}")
        return

    object_id = args.object_id
    mesh_path = resolve_object(object_id) if object_id else args.mesh
    assert mesh_path is not None
    name = safe_name(object_id or mesh_path.stem)
    output_path = args.output
    if output_path is None:
        output_path = default_output_dir(device) / f"{name}.json"

    image_path = args.preview_image
    if image_path is None and args.save_images:
        image_dir = args.image_dir or (output_path.parent / "previews")
        image_path = image_dir / f"{name}.png"

    process_object(
        object_id=object_id,
        mesh_path=mesh_path,
        device=device,
        args=args,
        output_path=output_path,
        image_path=image_path,
        seed=args.seed,
        item_index=1,
        item_count=1,
    )


if __name__ == "__main__":
    main()
