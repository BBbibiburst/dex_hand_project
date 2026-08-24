"""Object loading for the project-native Ultra pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from source.envs.manipulation.object_catalog import (
    DEFAULT_MAX_HORIZONTAL_DIAMETER,
    MANIFEST_PATH,
    limited_mesh_scale,
    resolve_record,
    resolve_record_path,
)


@dataclass(frozen=True)
class ObjectGeometry:
    object_id: str
    source_path: Path
    center: np.ndarray
    scale: float
    vertices: np.ndarray
    faces: np.ndarray
    surface_points: np.ndarray
    surface_normals: np.ndarray
    bounds: np.ndarray
    plane_normals: np.ndarray
    plane_offsets: np.ndarray

    @property
    def table_z(self) -> float:
        return float(self.bounds[0, 2])


def resolve_object_mesh(object_id: str) -> Path:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Object manifest is missing: {MANIFEST_PATH}. "
            "Run tools/download_maniskill_objects.py first."
        )
    record = resolve_record(object_id)
    root = resolve_record_path(record, "source_path")
    files = tuple(record.get("model_files", ()))
    selected = next((item for item in files if Path(item).name == "textured.obj"), None)
    selected = selected or next(
        (item for item in files if Path(item).suffix.lower() in {".obj", ".ply", ".stl"}),
        None,
    )
    if selected is not None and (root / selected).is_file():
        return (root / selected).resolve()
    raise ValueError(f"Unknown object or missing mesh: {object_id}")


def load_object_geometry(
    object_id: str,
    *,
    target_size: float | None = None,
    maximum_horizontal_diameter: float | None = DEFAULT_MAX_HORIZONTAL_DIAMETER,
    surface_points: int = 2048,
    seed: int = 0,
) -> ObjectGeometry:
    if (target_size is not None and target_size <= 0.0) or surface_points < 128:
        raise ValueError("target_size must be positive when set and surface_points at least 128.")
    import trimesh

    path = resolve_object_mesh(object_id)
    record = resolve_record(object_id)
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"Unable to load a triangle mesh from {path}.")
    mesh = loaded.copy()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    raw_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = 0.5 * (raw_bounds[0] + raw_bounds[1])
    extent = raw_bounds[1] - raw_bounds[0]
    scale = limited_mesh_scale(
        record,
        extent,
        target_size=target_size,
        maximum_horizontal_diameter=maximum_horizontal_diameter,
    )
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)

    # MuJoCo uses a convex collision hull for the single mesh geom created by
    # MeshObjectSpec. Synthesis must target that same geometry; optimizing on
    # a visual concavity (the inside of a bowl, for example) would otherwise
    # create candidates that cannot exist in the task simulator.
    collision_mesh = mesh.convex_hull
    collision_mesh.remove_unreferenced_vertices()
    collision_mesh.fix_normals()

    points, face_ids = trimesh.sample.sample_surface_even(
        collision_mesh,
        surface_points,
        seed=seed,
    )
    if len(points) < surface_points:
        extra, extra_faces = trimesh.sample.sample_surface(
            collision_mesh,
            surface_points - len(points),
            seed=seed + 1,
        )
        points = np.concatenate([points, extra], axis=0)
        face_ids = np.concatenate([face_ids, extra_faces], axis=0)
    normals = np.asarray(
        collision_mesh.face_normals[np.asarray(face_ids, dtype=np.int64)],
        dtype=np.float64,
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    plane_normals = np.array(collision_mesh.face_normals, dtype=np.float64, copy=True)
    plane_normals /= np.maximum(np.linalg.norm(plane_normals, axis=1, keepdims=True), 1e-12)
    plane_offsets = np.einsum(
        "fi,fi->f",
        plane_normals,
        np.asarray(collision_mesh.triangles[:, 0], dtype=np.float64),
    )
    return ObjectGeometry(
        object_id=object_id,
        source_path=path,
        center=center,
        scale=float(scale),
        vertices=np.asarray(collision_mesh.vertices, dtype=np.float64),
        faces=np.asarray(collision_mesh.faces, dtype=np.int64),
        surface_points=np.asarray(points, dtype=np.float64),
        surface_normals=normals,
        bounds=np.asarray(collision_mesh.bounds, dtype=np.float64),
        plane_normals=plane_normals,
        plane_offsets=plane_offsets,
    )
