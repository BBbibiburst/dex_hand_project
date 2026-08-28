"""Cached deterministic convex decomposition shared by synthesis and MuJoCo."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "collision_decomposition"
SETTINGS = {
    "schema_version": 3,
    "threshold": 0.05,
    "max_convex_hull": 24,
    "resolution": 1000,
    "mcts_iterations": 40,
    "max_ch_vertex": 96,
    "decimate": True,
    "seed": 0,
}
_MINIMUM_RELATIVE_PART_VOLUME = 1e-9


@contextmanager
def _exclusive_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _cache_key(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(json.dumps(SETTINGS, sort_keys=True).encode())
    return digest.hexdigest()[:24]


def _mujoco_compatible_part(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Give PhysX-compatible planar pieces a tiny deterministic thickness."""

    part = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mean = np.asarray(part.vertices).mean(axis=0)
    centered = np.asarray(part.vertices) - mean
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    local = centered @ axes.T
    extents = np.ptp(local, axis=0)
    maximum = max(float(extents.max()), 1e-9)
    if float(extents.min()) < maximum * 1e-5:
        thin_axis = int(np.argmin(extents))
        thickness = maximum * 2e-4
        lower, upper = local.copy(), local.copy()
        lower[:, thin_axis] -= 0.5 * thickness
        upper[:, thin_axis] += 0.5 * thickness
        points = np.concatenate([lower, upper], axis=0) @ axes + mean
        part = trimesh.convex.convex_hull(points)
    return part


def _drop_negligible_parts(parts: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """Discard numerical debris that MuJoCo cannot assign solid inertia.

    Some official collision files contain duplicate micro-triangles many
    orders of magnitude smaller than the actual object.  They are not useful
    contact geometry and make MuJoCo reject the complete model with
    ``mesh volume is too small``.  The relative threshold deliberately keeps
    real small features and thin components.
    """

    volumes = np.asarray([abs(float(part.volume)) for part in parts], dtype=np.float64)
    total = float(volumes.sum())
    if not parts or total <= 0.0:
        return parts
    keep = volumes >= total * _MINIMUM_RELATIVE_PART_VOLUME
    filtered = [part for part, usable in zip(parts, keep, strict=True) if bool(usable)]
    return filtered or [parts[int(np.argmax(volumes))]]


def convex_decomposition_paths(path: Path) -> tuple[Path, ...]:
    """Return cached OBJ files, one for every CoACD convex component."""

    source = Path(path).resolve()
    key = _cache_key(source)
    directory = CACHE_ROOT / key
    manifest_path = directory / "manifest.json"
    with _exclusive_lock(CACHE_ROOT / f"{key}.lock"):
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            cached = tuple(directory / name for name in payload.get("parts", ()))
            if cached and all(item.is_file() for item in cached):
                loaded_parts = [
                    trimesh.load(item, force="mesh", process=True) for item in cached
                ]
                usable = _drop_negligible_parts(loaded_parts)
                usable_ids = {id(part) for part in usable}
                return tuple(
                    path
                    for path, part in zip(cached, loaded_parts, strict=True)
                    if id(part) in usable_ids
                )

        loaded = trimesh.load(source, force="mesh", process=True)
        if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
            raise ValueError(f"Unable to decompose collision mesh: {source}")
        loaded.remove_unreferenced_vertices()
        loaded.fix_normals()
        # ManiSkill YCB collision.ply files already contain disconnected convex
        # components and are loaded with add_multiple_convex_collisions_from_file.
        # Preserve those official components exactly. Other datasets receive a
        # deterministic CoACD decomposition once and then use the same cache.
        if source.name == "collision.ply":
            components = loaded.split(only_watertight=False)
            result = [
                (
                    np.asarray(part.vertices, dtype=np.float64),
                    np.asarray(part.faces, dtype=np.int32),
                )
                for part in components
            ]
        else:
            import coacd

            coacd.set_log_level("error")
            result = coacd.run_coacd(
                coacd.Mesh(
                    np.asarray(loaded.vertices, dtype=np.float64),
                    np.asarray(loaded.faces, dtype=np.int32),
                ),
                threshold=float(SETTINGS["threshold"]),
                max_convex_hull=int(SETTINGS["max_convex_hull"]),
                resolution=int(SETTINGS["resolution"]),
                mcts_iterations=int(SETTINGS["mcts_iterations"]),
                max_ch_vertex=int(SETTINGS["max_ch_vertex"]),
                decimate=bool(SETTINGS["decimate"]),
                seed=int(SETTINGS["seed"]),
            )
        if not result:
            raise RuntimeError(f"No convex collision components found for {source}")
        directory.mkdir(parents=True, exist_ok=True)
        compatible_parts = _drop_negligible_parts(
            [_mujoco_compatible_part(vertices, faces) for vertices, faces in result]
        )
        names: list[str] = []
        for index, part in enumerate(compatible_parts):
            name = f"part_{index:03d}.obj"
            destination = directory / name
            temporary = directory / f".{name}.{os.getpid()}.partial"
            part.export(temporary, file_type="obj")
            os.replace(temporary, destination)
            names.append(name)
        temporary_manifest = directory / f".manifest.{os.getpid()}.partial"
        temporary_manifest.write_text(
            json.dumps(
                {"source": str(source), "settings": SETTINGS, "parts": names},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        return tuple(directory / name for name in names)


def load_convex_parts(path: Path) -> tuple[trimesh.Trimesh, ...]:
    """Load the same cached convex components used by MuJoCo."""

    parts = []
    for part_path in convex_decomposition_paths(path):
        mesh = trimesh.load(part_path, force="mesh", process=True)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"Invalid cached collision component: {part_path}")
        mesh.fix_normals()
        parts.append(mesh)
    return tuple(parts)
