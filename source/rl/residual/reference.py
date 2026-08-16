"""Load UltraDexGrasp episodes and expose low-level RL reference trajectories.

This module intentionally does not import ``source.ultradexgrasp``.  That keeps
RL data inspection independent from the Ultra package import side effects and
also makes the migration compatible with older Ultra package initializers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STAGE_CODES = {
    "settle": 0,
    "transit": 1,
    "pregrasp": 2,
    "approach": 3,
    "close": 4,
    "hold": 5,
    "lift": 6,
    "verify": 7,
}


@dataclass(frozen=True)
class EpisodeRecord:
    """Minimal Ultra episode representation needed by residual RL."""

    object_id: str
    seed: int
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    manifest: Path

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeRecord":
        path = Path(path)
        manifest = path / "manifest.json" if path.is_dir() else path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        array_path = manifest.parent / str(payload["arrays"])
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        required = {
            "qpos",
            "qvel",
            "ctrl",
            "object_position",
            "stage",
        }
        missing = sorted(required - arrays.keys())
        if missing:
            raise ValueError(f"Ultra episode is missing arrays: {missing}")
        lengths = {name: len(arrays[name]) for name in required}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values())) <= 0:
            raise ValueError(f"Ultra episode arrays have inconsistent lengths: {lengths}")
        return cls(
            object_id=str(payload["object_id"]),
            seed=int(payload.get("seed", 0)),
            arrays=arrays,
            metadata=dict(payload.get("metadata", {})),
            manifest=manifest,
        )


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Low-level actuator trajectory plus the simulator state that precedes it."""

    object_id: str
    source_manifest: Path
    source_seed: int
    start_stage: str
    control_dt: float
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    initial_ctrl: np.ndarray
    controls: np.ndarray
    stages: np.ndarray
    actuator_ids: np.ndarray
    actuator_names: tuple[str, ...]
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray
    arm_action_size: int
    initial_object_position: np.ndarray

    def __post_init__(self) -> None:
        controls = np.asarray(self.controls, dtype=np.float32)
        actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)
        ctrl_low = np.asarray(self.ctrl_low, dtype=np.float32)
        ctrl_high = np.asarray(self.ctrl_high, dtype=np.float32)
        if controls.ndim != 2 or len(controls) == 0:
            raise ValueError("Reference controls must have shape (T, action_dim) with T > 0.")
        action_dim = controls.shape[1]
        if actuator_ids.shape != (action_dim,):
            raise ValueError("actuator_ids must match the reference action dimension.")
        if ctrl_low.shape != (action_dim,) or ctrl_high.shape != (action_dim,):
            raise ValueError("Control limits must match the reference action dimension.")
        if not 0 <= self.arm_action_size <= action_dim:
            raise ValueError("arm_action_size is outside the reference action dimension.")
        if np.any(ctrl_high <= ctrl_low):
            raise ValueError("Every residual-controlled actuator needs a non-empty ctrl range.")
        if self.stages.shape != (len(controls),):
            raise ValueError("stages must contain one code per reference control frame.")
        for name, array in (
            ("initial_qpos", self.initial_qpos),
            ("initial_qvel", self.initial_qvel),
            ("initial_ctrl", self.initial_ctrl),
            ("controls", controls),
            ("initial_object_position", self.initial_object_position),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinity.")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "actuator_ids", actuator_ids)
        object.__setattr__(self, "ctrl_low", ctrl_low)
        object.__setattr__(self, "ctrl_high", ctrl_high)

    @property
    def horizon(self) -> int:
        return int(len(self.controls))

    @property
    def action_dim(self) -> int:
        return int(self.controls.shape[1])

    @property
    def hand_action_size(self) -> int:
        return self.action_dim - self.arm_action_size

    @property
    def hand_slice(self) -> slice:
        return slice(self.arm_action_size, self.action_dim)

    @classmethod
    def from_episode(
        cls,
        episode: EpisodeRecord,
        env,
        *,
        source_manifest: str | Path | None = None,
        start_stage: str = "approach",
    ) -> "ReferenceTrajectory":
        if start_stage not in STAGE_CODES:
            raise ValueError(
                f"Unknown start stage {start_stage!r}; choose from {tuple(STAGE_CODES)}."
            )
        stages = np.asarray(episode.arrays["stage"], dtype=np.int16).reshape(-1)
        matches = np.flatnonzero(stages == STAGE_CODES[start_stage])
        if not len(matches):
            available = sorted({int(value) for value in stages})
            raise ValueError(
                f"Reference episode has no {start_stage!r} stage; stage codes present={available}."
            )
        start = int(matches[0])
        state_index = max(0, start - 1)

        arm = env.controller.arm_controller
        hand = env.controller.hand_controller
        actuator_ids = np.concatenate([arm.actuator_ids, hand.actuator_ids]).astype(np.int32)
        actuator_names = tuple(env.controller.actuator_names)
        if len(actuator_names) != len(actuator_ids):
            raise RuntimeError("Controller actuator names and ids have inconsistent lengths.")
        recorded_names = episode.metadata.get("position_actuator_names")
        if recorded_names is not None and tuple(recorded_names) != actuator_names:
            raise ValueError(
                "Reference actuator ordering does not match the current robot model: "
                f"recorded={tuple(recorded_names)}, current={actuator_names}."
            )

        qpos = np.asarray(episode.arrays["qpos"], dtype=np.float32)
        qvel = np.asarray(episode.arrays["qvel"], dtype=np.float32)
        ctrl = np.asarray(episode.arrays["ctrl"], dtype=np.float32)
        if qpos.shape[1:] != (env.model.nq,) or qvel.shape[1:] != (env.model.nv,):
            raise ValueError(
                "Reference episode robot state does not match the current model: "
                f"qpos={qpos.shape[1:]}/{env.model.nq}, qvel={qvel.shape[1:]}/{env.model.nv}."
            )
        if ctrl.shape[1:] != (env.model.nu,):
            raise ValueError(
                f"Reference ctrl dimension {ctrl.shape[1:]} does not match model.nu={env.model.nu}."
            )

        object_position = np.asarray(episode.arrays["object_position"], dtype=np.float32)
        control_dt = float(episode.metadata.get("control_dt", getattr(env.config, "control_dt", 0.05)))
        if control_dt <= 0.0:
            raise ValueError("Reference control_dt must be positive.")
        return cls(
            object_id=episode.object_id,
            source_manifest=Path(source_manifest or episode.manifest),
            source_seed=episode.seed,
            start_stage=start_stage,
            control_dt=control_dt,
            initial_qpos=qpos[state_index].copy(),
            initial_qvel=qvel[state_index].copy(),
            initial_ctrl=ctrl[state_index].copy(),
            controls=ctrl[start:, actuator_ids].copy(),
            stages=stages[start:].copy(),
            actuator_ids=actuator_ids,
            actuator_names=actuator_names,
            ctrl_low=np.concatenate([arm.ctrl_low, hand.ctrl_low]).astype(np.float32),
            ctrl_high=np.concatenate([arm.ctrl_high, hand.ctrl_high]).astype(np.float32),
            arm_action_size=int(arm.action_size),
            initial_object_position=object_position[state_index].copy(),
        )


def resolve_reference_manifest(path: str | Path) -> Path:
    """Resolve an Ultra output, attempt directory, or manifest to one full episode."""
    path = Path(path)
    if path.is_file():
        return path
    direct = path / "manifest.json"
    if direct.is_file():
        return direct
    run_path = path / "run.json"
    if run_path.is_file():
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        manifest_name = payload.get("manifest")
        if manifest_name and (path / str(manifest_name)).is_file():
            return path / str(manifest_name)
    required = {STAGE_CODES["lift"], STAGE_CODES["verify"]}
    for candidate in sorted(path.glob("attempts/*/manifest.json")):
        episode = EpisodeRecord.load(candidate)
        stage_codes = {int(value) for value in np.asarray(episode.arrays["stage"]).reshape(-1)}
        if required.issubset(stage_codes):
            return candidate
    raise FileNotFoundError(
        f"No full UltraDexGrasp reference episode was found under {path}. "
        "The reference must reach lift and verify."
    )


def load_reference(
    manifest_or_directory: str | Path,
    env,
    *,
    start_stage: str = "approach",
) -> ReferenceTrajectory:
    manifest = resolve_reference_manifest(manifest_or_directory)
    episode = EpisodeRecord.load(manifest)
    return ReferenceTrajectory.from_episode(
        episode,
        env,
        source_manifest=manifest,
        start_stage=start_stage,
    )
