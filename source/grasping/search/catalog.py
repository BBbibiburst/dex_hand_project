"""Object-catalog and grasp-config path utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from source.grasping.search.common import MANIFEST, ROOT
from source.grasping.search.types import Cloud


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def grasp_config_name(object_id: str) -> str:
    """Return the canonical filesystem-safe name for an object grasp."""
    return _safe_name(object_id)


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
