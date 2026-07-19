"""Load an STL/OBJ mesh and sample an object-centred surface point cloud."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class SurfacePointCloud:
    points: np.ndarray
    normals: np.ndarray
    center: np.ndarray
    scale: float


@dataclass(frozen=True)
class TriangleMesh:
    """One triangle mesh expressed in an end-effector root frame."""

    vertices: np.ndarray
    faces: np.ndarray


def sample_surface_pointcloud(
    mesh_path: str | Path,
    *,
    count: int = 2048,
    target_size: float | None = None,
    seed: int = 0,
) -> SurfacePointCloud:
    """Sample points/normals without constructing a simulator."""
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if count < 32:
        raise ValueError("count must be at least 32.")

    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError(f"Mesh contains no triangle surface: {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    extent = float(np.ptp(vertices, axis=0).max())
    scale = 1.0 if target_size is None else float(target_size) / max(extent, 1e-9)

    # trimesh uses NumPy's global RNG in sample_surface.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float64)
    points = (np.asarray(points, dtype=np.float64) - center) * scale
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return SurfacePointCloud(
        points=points,
        normals=normals,
        center=center,
        scale=scale,
    )
