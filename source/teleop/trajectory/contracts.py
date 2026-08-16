"""Portable raw/optimized Vive + glove trajectory records.

The record deliberately stores both the *command* trajectory and the MuJoCo state
observed while the command was collected.  Optimized trajectories keep the
observed arrays as a reference only; replay always starts from ``initial_*`` and
executes ``actions`` through the normal environment/controller stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

TRAJECTORY_SCHEMA_VERSION = 1


def _array(value, *, dtype=None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError("Trajectory arrays must contain only finite values.")
    return result


@dataclass(frozen=True)
class TeleopTrajectory:
    """A self-contained teleoperation command trajectory and replay snapshot."""

    metadata: dict[str, Any]
    timestamps: np.ndarray
    actions: np.ndarray
    vive_pose: np.ndarray
    glove: np.ndarray
    observed_qpos: np.ndarray
    observed_qvel: np.ndarray
    observed_ctrl: np.ndarray
    observed_object_position: np.ndarray
    observed_object_quaternion: np.ndarray
    observed_tactile: np.ndarray
    task_success: np.ndarray
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    initial_ctrl: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    def __post_init__(self) -> None:
        timestamps = _array(self.timestamps, dtype=np.float64).reshape(-1)
        actions = _array(self.actions, dtype=np.float32)
        vive_pose = _array(self.vive_pose, dtype=np.float32)
        glove = _array(self.glove, dtype=np.float32)
        task_success = np.asarray(self.task_success, dtype=bool).reshape(-1)
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError("actions must have shape (T, action_dim) with T > 0.")
        horizon, action_dim = actions.shape
        if timestamps.shape != (horizon,):
            raise ValueError("timestamps must contain one value per action frame.")
        if vive_pose.shape != (horizon, 7):
            raise ValueError("vive_pose must have shape (T, 7), xyz + quaternion_wxyz.")
        if glove.shape != (horizon, 6):
            raise ValueError("glove must have shape (T, 6).")
        if task_success.shape != (horizon,):
            raise ValueError("task_success must contain one flag per action frame.")
        if np.any(np.diff(timestamps) < 0.0):
            raise ValueError("timestamps must be non-decreasing.")
        if np.asarray(self.action_low).shape != (action_dim,) or np.asarray(self.action_high).shape != (
            action_dim,
        ):
            raise ValueError("action bounds must match action dimension.")

        observed = {
            "observed_qpos": self.observed_qpos,
            "observed_qvel": self.observed_qvel,
            "observed_ctrl": self.observed_ctrl,
            "observed_object_position": self.observed_object_position,
            "observed_object_quaternion": self.observed_object_quaternion,
            "observed_tactile": self.observed_tactile,
        }
        for name, value in observed.items():
            arr = _array(value)
            if len(arr) != horizon:
                raise ValueError(f"{name} must contain one sample per action frame.")

        if np.asarray(self.observed_object_position).shape != (horizon, 3):
            raise ValueError("observed_object_position must have shape (T, 3).")
        if np.asarray(self.observed_object_quaternion).shape != (horizon, 4):
            raise ValueError("observed_object_quaternion must have shape (T, 4).")
        for name, value in (
            ("initial_qpos", self.initial_qpos),
            ("initial_qvel", self.initial_qvel),
            ("initial_ctrl", self.initial_ctrl),
        ):
            _array(value)

        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "vive_pose", vive_pose)
        object.__setattr__(self, "glove", glove)
        object.__setattr__(self, "task_success", task_success)
        object.__setattr__(self, "action_low", np.asarray(self.action_low, dtype=np.float32))
        object.__setattr__(self, "action_high", np.asarray(self.action_high, dtype=np.float32))

    @property
    def horizon(self) -> int:
        return int(len(self.actions))

    @property
    def control_dt(self) -> float:
        value = float(self.metadata.get("control_dt", 0.05))
        if value <= 0.0:
            raise ValueError("metadata.control_dt must be positive.")
        return value

    @property
    def hand_action_size(self) -> int:
        return int(self.metadata.get("hand_action_size", 6))

    @property
    def arm_action_size(self) -> int:
        return self.actions.shape[1] - self.hand_action_size

    def with_actions(self, actions: np.ndarray, *, metadata_updates: dict[str, Any]) -> "TeleopTrajectory":
        metadata = dict(self.metadata)
        metadata.update(metadata_updates)
        return replace(self, actions=np.asarray(actions, dtype=np.float32), metadata=metadata)

    def save(self, path: str | Path) -> Path:
        """Atomically write a compressed ``.npz`` record without pickle payloads."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata.setdefault("schema_version", TRAJECTORY_SCHEMA_VERSION)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            timestamps=self.timestamps,
            actions=self.actions,
            vive_pose=self.vive_pose,
            glove=self.glove,
            observed_qpos=self.observed_qpos,
            observed_qvel=self.observed_qvel,
            observed_ctrl=self.observed_ctrl,
            observed_object_position=self.observed_object_position,
            observed_object_quaternion=self.observed_object_quaternion,
            observed_tactile=self.observed_tactile,
            task_success=self.task_success.astype(np.uint8),
            initial_qpos=self.initial_qpos,
            initial_qvel=self.initial_qvel,
            initial_ctrl=self.initial_ctrl,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TeleopTrajectory":
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            schema = int(metadata.get("schema_version", 0))
            if schema != TRAJECTORY_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported teleop trajectory schema {schema}; expected {TRAJECTORY_SCHEMA_VERSION}."
                )
            return cls(
                metadata=metadata,
                timestamps=payload["timestamps"],
                actions=payload["actions"],
                vive_pose=payload["vive_pose"],
                glove=payload["glove"],
                observed_qpos=payload["observed_qpos"],
                observed_qvel=payload["observed_qvel"],
                observed_ctrl=payload["observed_ctrl"],
                observed_object_position=payload["observed_object_position"],
                observed_object_quaternion=payload["observed_object_quaternion"],
                observed_tactile=payload["observed_tactile"],
                task_success=payload["task_success"].astype(bool),
                initial_qpos=payload["initial_qpos"],
                initial_qvel=payload["initial_qvel"],
                initial_ctrl=payload["initial_ctrl"],
                action_low=payload["action_low"],
                action_high=payload["action_high"],
            )


class TeleopTrajectoryBuffer:
    """Mutable frame buffer used by the interactive collector."""

    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray,
        initial_ctrl: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        self.metadata = dict(metadata)
        self.initial_qpos = np.asarray(initial_qpos, dtype=np.float64).copy()
        self.initial_qvel = np.asarray(initial_qvel, dtype=np.float64).copy()
        self.initial_ctrl = np.asarray(initial_ctrl, dtype=np.float64).copy()
        self.action_low = np.asarray(action_low, dtype=np.float32).copy()
        self.action_high = np.asarray(action_high, dtype=np.float32).copy()
        self._timestamps: list[float] = []
        self._actions: list[np.ndarray] = []
        self._vive_pose: list[np.ndarray] = []
        self._glove: list[np.ndarray] = []
        self._qpos: list[np.ndarray] = []
        self._qvel: list[np.ndarray] = []
        self._ctrl: list[np.ndarray] = []
        self._object_position: list[np.ndarray] = []
        self._object_quaternion: list[np.ndarray] = []
        self._tactile: list[np.ndarray] = []
        self._success: list[bool] = []
        self._time_origin: float | None = None

    @property
    def frame_count(self) -> int:
        return len(self._actions)

    def add_frame(self, *, observation, action, glove, vive, success: bool, timestamp: float) -> None:
        if self._time_origin is None:
            self._time_origin = float(timestamp)
        self._timestamps.append(float(timestamp) - self._time_origin)
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        self._vive_pose.append(
            np.concatenate([vive.position, vive.quaternion_wxyz]).astype(np.float32)
        )
        self._glove.append(np.asarray(glove.stretch, dtype=np.float32).copy())
        self._qpos.append(np.asarray(observation["qpos"], dtype=np.float32).copy())
        self._qvel.append(np.asarray(observation["qvel"], dtype=np.float32).copy())
        self._ctrl.append(np.asarray(observation["ctrl"], dtype=np.float32).copy())
        self._object_position.append(np.asarray(observation["object_pos"], dtype=np.float32).copy())
        self._object_quaternion.append(np.asarray(observation["object_quat"], dtype=np.float32).copy())
        self._tactile.append(np.asarray(observation["tactile"], dtype=np.float32).copy())
        self._success.append(bool(success))

    def build(self) -> TeleopTrajectory:
        if not self.frame_count:
            raise ValueError("Cannot build an empty teleop trajectory.")
        return TeleopTrajectory(
            metadata=self.metadata,
            timestamps=np.asarray(self._timestamps, dtype=np.float64),
            actions=np.stack(self._actions),
            vive_pose=np.stack(self._vive_pose),
            glove=np.stack(self._glove),
            observed_qpos=np.stack(self._qpos),
            observed_qvel=np.stack(self._qvel),
            observed_ctrl=np.stack(self._ctrl),
            observed_object_position=np.stack(self._object_position),
            observed_object_quaternion=np.stack(self._object_quaternion),
            observed_tactile=np.stack(self._tactile),
            task_success=np.asarray(self._success, dtype=bool),
            initial_qpos=self.initial_qpos,
            initial_qvel=self.initial_qvel,
            initial_ctrl=self.initial_ctrl,
            action_low=self.action_low,
            action_high=self.action_high,
        )
