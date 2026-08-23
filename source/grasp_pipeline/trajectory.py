"""Serializable residual trajectories produced by the grasp RL search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TRAJECTORY_SCHEMA_VERSION = 1


@dataclass
class GraspTrajectory:
    object_id: str
    source_manifest: str
    start_stage: str
    action_mode: str
    residual_actions: np.ndarray
    controls: np.ndarray
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    success: bool
    episode_return: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        residual = np.asarray(self.residual_actions, dtype=np.float32)
        controls = np.asarray(self.controls, dtype=np.float32)
        if residual.ndim != 2 or controls.ndim != 2 or len(residual) != len(controls):
            raise ValueError("Residual and control trajectories must be 2-D with equal length.")
        if not len(controls):
            raise ValueError("Residual trajectory is empty.")
        for name, values in (
            ("residual_actions", residual),
            ("controls", controls),
            ("initial_qpos", self.initial_qpos),
            ("initial_qvel", self.initial_qvel),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or infinity.")

    def save(self, directory: str | Path) -> Path:
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays_path = directory / "trajectory.npz"
        np.savez_compressed(
            arrays_path,
            residual_actions=np.asarray(self.residual_actions, dtype=np.float32),
            controls=np.asarray(self.controls, dtype=np.float32),
            initial_qpos=np.asarray(self.initial_qpos, dtype=np.float32),
            initial_qvel=np.asarray(self.initial_qvel, dtype=np.float32),
        )
        manifest = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "object_id": self.object_id,
            "source_manifest": self.source_manifest,
            "start_stage": self.start_stage,
            "action_mode": self.action_mode,
            "success": bool(self.success),
            "episode_return": float(self.episode_return),
            "frames": int(len(self.controls)),
            "arrays": arrays_path.name,
            "metadata": self.metadata,
        }
        path = directory / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GraspTrajectory":
        path = Path(path)
        manifest = path / "manifest.json" if path.is_dir() else path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("Unsupported residual trajectory schema.")
        with np.load(manifest.parent / payload["arrays"], allow_pickle=False) as archive:
            result = cls(
                object_id=str(payload["object_id"]),
                source_manifest=str(payload["source_manifest"]),
                start_stage=str(payload["start_stage"]),
                action_mode=str(payload["action_mode"]),
                residual_actions=np.asarray(archive["residual_actions"], dtype=np.float32),
                controls=np.asarray(archive["controls"], dtype=np.float32),
                initial_qpos=np.asarray(archive["initial_qpos"], dtype=np.float32),
                initial_qvel=np.asarray(archive["initial_qvel"], dtype=np.float32),
                success=bool(payload["success"]),
                episode_return=float(payload["episode_return"]),
                metadata=dict(payload.get("metadata", {})),
            )
        result.validate()
        return result
