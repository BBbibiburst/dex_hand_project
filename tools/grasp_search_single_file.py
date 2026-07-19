#!/usr/bin/env python3
"""Single-file grasp-search workbench for Dex Hand and Pika.

This development copy intentionally does not import ``source.grasping``.  It
keeps object loading, end-effector FK/mesh extraction, candidate generation,
geometric scoring, force-closure estimation, approach generation, JSON output,
and visualization in one file so an experimental model can edit one artifact.

Run from the repository root, for example:

    python tools/grasp_search_single_file.py \
      --object-id ycb:002_master_chef_can --viewer

The output uses the production grasp-config schema and can be inspected with:

    python -m source.demos.validate_standalone_grasp \
      configs/grasps/single_file/ycb_002_master_chef_can.json --viewer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mujoco
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "maniskill" / "manifest.json"
DEX_XML = ROOT / "assets" / "grippers" / "dex_hand" / "dex_hand.xml"
PIKA_XML = ROOT / "assets" / "grippers" / "pika_gripper" / "pika_gripper.xml"


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


@dataclass
class Surface:
    points: np.ndarray
    labels: np.ndarray
    meshes: list[tuple[np.ndarray, np.ndarray]]
    actuator_values: np.ndarray
    fractions: np.ndarray
    midpoint: np.ndarray


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
    table_clearance: float
    pca_axis: int
    score: float


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


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
    return Cloud(np.asarray(points), normals, center, scale, mesh)


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
    friction: float = 0.8,
    edges: int = 8,
) -> np.ndarray:
    columns = []
    for point, normal in zip(points, inward_normals, strict=True):
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(normal @ reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(normal, reference)
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        bitangent = np.cross(normal, tangent)
        for angle in np.linspace(0.0, 2.0 * np.pi, edges, endpoint=False):
            force = normal + friction * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
            force /= np.linalg.norm(force)
            columns.append(np.r_[force, np.cross(point, force)])
    return np.asarray(columns, dtype=np.float64).T


def evaluate(
    cloud: Cloud,
    device: Device,
    surface: Surface,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    pca_axis: int,
) -> Candidate:
    posed = surface.points @ rotation.T + translation
    tree = cKDTree(cloud.points)
    distances, indices = tree.query(posed)
    offsets = posed - cloud.points[indices]
    signed = np.sum(offsets * cloud.normals[indices], axis=1)
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
    if len(contact_points_array) >= 2:
        matrix = friction_wrenches(contact_points_array, contact_normals_array)
        # nnls(A, 0) has the trivial zero solution. Add a unit-sum row so the
        # residual measures whether a nonzero convex contact-force mixture can
        # balance the wrench.
        augmented = np.vstack([matrix, np.ones(matrix.shape[1])])
        target = np.r_[np.zeros(6), 1.0]
        weights, residual = nnls(augmented, target)
        _ = weights
        force_closure = float(residual)
    else:
        force_closure = 1.0
    table_z = float(cloud.points[:, 2].min())
    clearance = float(posed[:, 2].min() - table_z)
    opposing = (
        4 in contacts and any(label < 4 for label in contacts)
        if device.name == "dex_hand"
        else 0 in contacts and 1 in contacts
    )
    score = (
        35.0 * penetration
        + 80.0 * rigid
        + 2.0 * float(np.mean(per_label))
        + force_closure
        + max(0.005 - clearance, 0.0) * 50.0
        + (0.0 if opposing else 2.0)
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
        float(np.mean(per_label)),
        force_closure,
        clearance,
        pca_axis,
        score,
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


def orientation_candidates(cloud: Cloud, seed: int) -> list[tuple[int, np.ndarray]]:
    covariance = np.cov(cloud.points.T)
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1]
    rotations = []
    cube = Rotation.create_group("O").as_matrix()
    for axis_index in range(3):
        basis = np.roll(axes, -axis_index, axis=1)
        if np.linalg.det(basis) < 0:
            basis[:, 2] *= -1
        rotations.extend((axis_index, basis @ symmetry) for symmetry in cube)
    rng = np.random.default_rng(seed)
    rotations.extend((-1, matrix) for matrix in Rotation.random(24, random_state=rng).as_matrix())
    return rotations


def search(
    cloud: Cloud,
    device: Device,
    *,
    joint_candidates: int,
    seed: int,
) -> Candidate:
    fractions = fraction_candidates(device, max(3, joint_candidates // 16))
    orientations = orientation_candidates(cloud, seed)
    best = None
    for fraction_index, fraction in enumerate(fractions):
        surface = surface_for(device, fraction, seed=seed + fraction_index)
        for pca_axis, rotation in orientations:
            translation = -(surface.midpoint @ rotation.T)
            for depth in (-0.012, -0.006, 0.0, 0.006, 0.012):
                candidate_translation = translation + rotation[:, 0] * depth
                candidate = evaluate(
                    cloud,
                    device,
                    surface,
                    rotation,
                    candidate_translation,
                    pca_axis=pca_axis,
                )
                if best is None or candidate.score < best.score:
                    best = candidate
    if best is None:
        raise RuntimeError("No candidate was evaluated.")
    return best


def approach(candidate: Candidate, waypoint_count: int = 14) -> tuple[np.ndarray, np.ndarray]:
    progress = np.linspace(0.0, 1.0, waypoint_count)
    direction = candidate.rotation @ np.asarray([-1.0, 0.0, 0.0])
    direction[2] = max(direction[2], 0.35)
    direction /= np.linalg.norm(direction)
    translations = candidate.translation[None, :] + (1.0 - progress[:, None]) * 0.10 * direction
    fractions = np.repeat(candidate.surface.fractions[None, :], waypoint_count, axis=0)
    if len(fractions[0]) == 1:
        fractions[:, 0] = 1.0 + np.clip((progress - 0.72) / 0.28, 0.0, 1.0) * (
            candidate.surface.fractions[0] - 1.0
        )
    else:
        fractions[: int(0.7 * waypoint_count), :4] *= 0.0
        fractions[: int(0.7 * waypoint_count), 5] *= 0.0
    return translations, fractions


def payload(
    object_id: str | None,
    mesh_path: Path,
    cloud: Cloud,
    device: Device,
    candidate: Candidate,
) -> dict:
    translations, fractions = approach(candidate)
    opposing = (
        4 in candidate.contacts and any(label < 4 for label in candidate.contacts)
        if device.name == "dex_hand"
        else 0 in candidate.contacts and 1 in candidate.contacts
    )
    success = (
        candidate.penetration <= 0.004
        and candidate.rigid_penetration <= 0.0015
        and candidate.table_clearance >= 0.005
        and candidate.force_closure <= 0.35
        and opposing
    )
    preload_directions = (
        np.ones(len(device.actuators))
        if device.name == "dex_hand"
        else -np.ones(len(device.actuators))
    )
    preload_weights = np.ones(len(device.actuators))
    if device.name == "dex_hand":
        preload_weights[4] = 0.0
    return {
        "schema_version": 1,
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
        "hand_closure": float(np.mean(candidate.surface.fractions)),
        "hand_maximum_penetration": candidate.penetration,
        "hand_maximum_noncontact_penetration": candidate.rigid_penetration,
        "hand_mean_contact_distance": candidate.mean_distance,
        "hand_contacting_fingers": list(candidate.contacts),
        "hand_force_closure_residual": candidate.force_closure,
        "hand_palmward_force_component": 0.0,
        "hand_palmward_direction": [1.0, 0.0, 0.0],
        "hand_palmward_depth": 0.0,
        "hand_table_clearance": candidate.table_clearance,
        "hand_pca_axis_index": candidate.pca_axis,
        "hand_robustness_margin": max(0.0, 0.005 - candidate.mean_distance),
        "hand_object_inside": True,
        "approach_hand_translations": translations.tolist(),
        "approach_hand_rotation_matrices": np.repeat(
            candidate.rotation[None, :, :], len(translations), axis=0
        ).tolist(),
        "approach_hand_actuator_fractions": fractions.tolist(),
        "hand_fit_success": success,
    }


def draw(cloud: Cloud, candidate: Candidate, *, output: Path | None, show: bool) -> None:
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    object_faces = cloud.mesh.faces
    if len(object_faces) > 20_000:
        object_faces = object_faces[np.linspace(0, len(object_faces) - 1, 20_000, dtype=int)]
    axis.add_collection3d(
        Poly3DCollection(
            np.asarray(cloud.mesh.vertices)[object_faces],
            facecolor="#6f91ad",
            edgecolor="none",
            alpha=0.30,
            label="object mesh",
        )
    )
    hand_triangles = []
    for vertices, faces in candidate.surface.meshes:
        posed = vertices @ candidate.rotation.T + candidate.translation
        count = min(len(faces), 500)
        indices = np.linspace(0, len(faces) - 1, count, dtype=int)
        hand_triangles.append(posed[faces[indices]])
    axis.add_collection3d(
        Poly3DCollection(
            np.concatenate(hand_triangles),
            facecolor="#f4a261",
            edgecolor="none",
            alpha=0.5,
            label="end-effector mesh",
        )
    )
    axis.scatter(*cloud.points.T, s=2, alpha=0.18, color="#5b87ad")
    axis.scatter(*candidate.points.T, s=2, alpha=0.12, color="#e76f51")
    if len(candidate.contact_points):
        axis.scatter(
            *candidate.contact_points.T,
            s=90,
            color="#2a9d8f",
            edgecolors="black",
            label="contacts",
        )
        axis.quiver(
            *candidate.contact_points.T,
            *candidate.contact_normals.T,
            length=0.02,
            color="#2a9d8f",
        )
    translations, _ = approach(candidate)
    axis.plot(*translations.T, color="#2ca02c", linewidth=2.5, label="approach")
    visible = np.concatenate([cloud.points, candidate.points, translations])
    low, high = visible.min(0), visible.max(0)
    center, radius = 0.5 * (low + high), 0.55 * float(np.ptp(visible, axis=0).max())
    axis.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title=(
            f"single-file {candidate.surface.fractions.tolist()}\n"
            f"score={candidate.score:.3f}, Efc={candidate.force_closure:.3f}"
        ),
    )
    axis.set_box_aspect((1, 1, 1))
    axis.legend()
    figure.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--object-id")
    source.add_argument("--mesh", type=Path)
    parser.add_argument(
        "--end-effector",
        choices=tuple(DEVICES),
        default="dex_hand",
    )
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--joint-candidates", type=int, default=128)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-image", type=Path)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = resolve_object(args.object_id) if args.object_id else args.mesh
    device = DEVICES[args.end_effector]
    cloud = load_cloud(
        mesh_path,
        count=args.points,
        target_size=args.target_size,
        seed=args.seed,
    )
    candidate = search(
        cloud,
        device,
        joint_candidates=args.joint_candidates,
        seed=args.seed,
    )
    output = args.output
    if output is None:
        name = safe_name(args.object_id or mesh_path.stem)
        output = ROOT / "configs" / "grasps" / "single_file"
        if device.name != "dex_hand":
            output /= device.name
        output /= f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = payload(args.object_id, mesh_path, cloud, device, candidate)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.preview_image or args.viewer:
        draw(cloud, candidate, output=args.preview_image, show=args.viewer)
    print(
        f"output={output} end_effector={device.name} "
        f"score={candidate.score:.4f} contacts={candidate.contacts} "
        f"Efc={candidate.force_closure:.4f} "
        f"penetration={candidate.penetration:.4f}m "
        f"rigid_penetration={candidate.rigid_penetration:.4f}m "
        f"table_clearance={candidate.table_clearance:.4f}m "
        f"fit={result['hand_fit_success']}"
    )


if __name__ == "__main__":
    main()
