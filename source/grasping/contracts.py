"""Serializable contracts for the GraspQP + DexEvolve pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GRASP_SCHEMA_VERSION = 1
EPISODE_SCHEMA_VERSION = 1
PIPELINE_NAME = "graspqp-dexevolve-rm75b-dex-hand-v1"


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}.")
    return array


@dataclass(frozen=True)
class GraspCandidate:
    """One object-relative hand pose synthesized by the new optimizer."""

    object_id: str
    seed_index: int
    hand_translation: np.ndarray
    hand_rotation_matrix: np.ndarray
    actuator_fractions: np.ndarray
    contact_points: np.ndarray
    contact_normals: np.ndarray
    contact_distances: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)
    backend: str = "native-differentiable"

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty.")
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative.")
        translation = _array(self.hand_translation, (3,), "hand_translation")
        rotation = _array(self.hand_rotation_matrix, (3, 3), "hand_rotation_matrix")
        fractions = _array(self.actuator_fractions, (6,), "actuator_fractions")
        if np.any((fractions < 0.0) | (fractions > 1.0)):
            raise ValueError("actuator_fractions must lie in [0, 1].")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
            raise ValueError("hand_rotation_matrix must be orthonormal.")
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("hand_rotation_matrix must be right-handed.")
        points = np.asarray(self.contact_points, dtype=np.float64)
        normals = np.asarray(self.contact_normals, dtype=np.float64)
        distances = np.asarray(self.contact_distances, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("contact_points must be finite with shape (N, 3).")
        if normals.shape != points.shape or not np.all(np.isfinite(normals)):
            raise ValueError("contact_normals must match contact_points.")
        if distances.shape != (len(points),) or not np.all(np.isfinite(distances)):
            raise ValueError("contact_distances must have shape (N,).")
        object.__setattr__(self, "hand_translation", translation)
        object.__setattr__(self, "hand_rotation_matrix", rotation)
        object.__setattr__(self, "actuator_fractions", fractions)
        object.__setattr__(self, "contact_points", points)
        object.__setattr__(self, "contact_normals", normals)
        object.__setattr__(self, "contact_distances", distances)
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRASP_SCHEMA_VERSION,
            "pipeline": PIPELINE_NAME,
            "backend": self.backend,
            "object_id": self.object_id,
            "seed_index": self.seed_index,
            "hand_translation": self.hand_translation.tolist(),
            "hand_rotation_matrix": self.hand_rotation_matrix.tolist(),
            "actuator_fractions": self.actuator_fractions.tolist(),
            "contact_points": self.contact_points.tolist(),
            "contact_normals": self.contact_normals.tolist(),
            "contact_distances": self.contact_distances.tolist(),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraspCandidate:
        if payload.get("schema_version") != GRASP_SCHEMA_VERSION:
            raise ValueError("Unsupported grasp candidate schema.")
        return cls(
            object_id=str(payload["object_id"]),
            seed_index=int(payload["seed_index"]),
            hand_translation=payload["hand_translation"],
            hand_rotation_matrix=payload["hand_rotation_matrix"],
            actuator_fractions=payload["actuator_fractions"],
            contact_points=payload["contact_points"],
            contact_normals=payload["contact_normals"],
            contact_distances=payload["contact_distances"],
            metrics=dict(payload.get("metrics", {})),
            backend=str(payload.get("backend", "native-differentiable")),
        )


@dataclass
class DemonstrationEpisode:
    """Dense control episode plus a compact, human-readable manifest."""

    object_id: str
    seed: int
    candidate: GraspCandidate
    arrays: dict[str, np.ndarray]
    success: bool
    terminal_stage: str
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    REQUIRED_ARRAYS = (
        "qpos",
        "qvel",
        "ctrl",
        "action",
        "object_position",
        "object_quaternion_wxyz",
        "stage",
        "reward",
        "task_success",
    )

    def validate(self) -> None:
        missing = [name for name in self.REQUIRED_ARRAYS if name not in self.arrays]
        if missing:
            raise ValueError(f"Episode arrays are missing {missing}.")
        lengths = {name: len(np.asarray(self.arrays[name])) for name in self.REQUIRED_ARRAYS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Episode arrays have inconsistent lengths: {lengths}.")
        if next(iter(lengths.values())) == 0:
            raise ValueError("Episode must contain at least one frame.")

    def save(self, directory: str | Path) -> Path:
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        array_path = directory / "episode.npz"
        np.savez_compressed(array_path, **self.arrays)
        manifest = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "pipeline": PIPELINE_NAME,
            "object_id": self.object_id,
            "seed": self.seed,
            "success": bool(self.success),
            "terminal_stage": self.terminal_stage,
            "failure_reason": self.failure_reason,
            "frames": len(next(iter(self.arrays.values()))),
            "arrays": array_path.name,
            "candidate": self.candidate.to_dict(),
            "metadata": self.metadata,
        }
        manifest_path = directory / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
        return manifest_path

    @classmethod
    def load(cls, path: str | Path) -> DemonstrationEpisode:
        path = Path(path)
        manifest_path = path / "manifest.json" if path.is_dir() else path
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != EPISODE_SCHEMA_VERSION:
            raise ValueError("Unsupported grasp episode schema.")
        array_path = manifest_path.parent / str(payload["arrays"])
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        episode = cls(
            object_id=str(payload["object_id"]),
            seed=int(payload["seed"]),
            candidate=GraspCandidate.from_dict(payload["candidate"]),
            arrays=arrays,
            success=bool(payload["success"]),
            terminal_stage=str(payload["terminal_stage"]),
            failure_reason=payload.get("failure_reason"),
            metadata=dict(payload.get("metadata", {})),
        )
        episode.validate()
        return episode
