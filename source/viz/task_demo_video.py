"""Task-agnostic MuJoCo episode replay and compact video recording."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import mujoco
import numpy as np

SUPPORTED_TASKS = ("lift", "pick_place", "stack", "nut_assembly", "push")


@dataclass(frozen=True)
class RecordedTaskEpisode:
    """Minimal task episode contract required by the video recorder."""

    manifest_path: Path
    task: str
    task_config: dict[str, Any]
    env_config: dict[str, Any]
    arrays: dict[str, np.ndarray]
    stage_names: dict[int, str]
    object_id: str | None
    camera: str | int | None


def _task_from_manifest(payload: Mapping[str, Any], override: str | None) -> str:
    metadata = payload.get("metadata", {})
    stored = metadata.get("task") if isinstance(metadata, Mapping) else None
    task = str(override or stored or "lift")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task {task!r}; choose one of {SUPPORTED_TASKS}.")
    return task


def load_recorded_task_episode(
    manifest: str | Path,
    *,
    task: str | None = None,
    task_config_override: Mapping[str, Any] | None = None,
) -> RecordedTaskEpisode:
    """Load a generated or teleoperated episode without requiring grasp fields."""

    manifest_path = Path(manifest)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if manifest_path.suffix == ".npz":
        with np.load(manifest_path, allow_pickle=False) as archive:
            if "metadata_json" not in archive or "observed_qpos" not in archive:
                raise ValueError(f"Unsupported standalone episode archive: {manifest_path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {
                "qpos": np.asarray(archive["observed_qpos"]),
                "qvel": np.asarray(archive["observed_qvel"]),
                "ctrl": np.asarray(archive["observed_ctrl"]),
            }
        payload: dict[str, Any] = {
            "metadata": metadata,
            "object_id": metadata.get("object_id"),
        }
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        array_path = manifest_path.parent / str(payload.get("arrays", "episode.npz"))
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    resolved_task = _task_from_manifest(payload, task)
    if "qpos" not in arrays:
        raise ValueError(f"Episode archive has no qpos array: {array_path}")
    frames = len(arrays["qpos"])
    if frames == 0:
        raise ValueError("Episode contains no frames.")
    for optional in ("qvel", "ctrl", "stage"):
        if optional in arrays and len(arrays[optional]) != frames:
            raise ValueError(f"Episode array {optional!r} has a different frame count.")

    task_config = dict(metadata.get("task_config", {}))
    object_id = payload.get("object_id") or metadata.get("object_id")
    if object_id and resolved_task in {"lift", "pick_place", "push"}:
        task_config.setdefault("object_id", str(object_id))
    if task_config_override:
        task_config.update(task_config_override)
    task_config.setdefault("terminate_on_success", False)

    stored_codes = metadata.get("stage_codes", {})
    stage_names = (
        {int(code): str(name) for name, code in stored_codes.items()}
        if isinstance(stored_codes, Mapping)
        else {}
    )
    return RecordedTaskEpisode(
        manifest_path=manifest_path,
        task=resolved_task,
        task_config=task_config,
        env_config=dict(metadata.get("env_config", {})),
        arrays=arrays,
        stage_names=stage_names,
        object_id=None if object_id is None else str(object_id),
        camera=metadata.get("camera"),
    )


def _open_video_writer(path: Path, *, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer (codec: mp4v).")
    return writer


def record_task_episode_video(
    episode: RecordedTaskEpisode,
    output: str | Path,
    *,
    width: int = 480,
    height: int = 270,
    fps: float = 20.0,
    frame_stride: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    hold_last_seconds: float = 0.75,
    camera: str | int | None = None,
    overlay: bool = True,
    environment_factory: Callable[[RecordedTaskEpisode, int], Any],
) -> Path:
    """Replay exact MuJoCo states and encode a compact MP4 video."""

    if width <= 0 or height <= 0 or fps <= 0.0 or frame_stride <= 0:
        raise ValueError("Video dimensions, fps, and frame_stride must be positive.")
    if width % 2 or height % 2:
        raise ValueError("MP4 width and height must be even numbers.")
    if hold_last_seconds < 0.0:
        raise ValueError("hold_last_seconds must be non-negative.")

    arrays = episode.arrays
    total_frames = len(arrays["qpos"])
    stop = total_frames if end_frame is None else min(end_frame, total_frames)
    if not 0 <= start_frame < stop:
        raise ValueError(
            f"Invalid frame interval [{start_frame}, {stop}) for {total_frames} frames."
        )

    env = environment_factory(episode, total_frames + 10)
    renderer: mujoco.Renderer | None = None
    writer: cv2.VideoWriter | None = None
    output_path = Path(output)
    written = 0
    last_frame: np.ndarray | None = None
    try:
        env.reset(seed=int(json.loads(episode.manifest_path.read_text()).get("seed", 0)))
        if arrays["qpos"].shape[1:] != env.data.qpos.shape:
            raise ValueError(
                f"Recorded qpos shape {arrays['qpos'].shape[1:]} does not match "
                f"the {episode.task} model shape {env.data.qpos.shape}. "
                "Supply the task's original task_config in the manifest or CLI."
            )
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        writer = _open_video_writer(output_path, width=width, height=height, fps=fps)
        selected_camera: str | int = camera if camera is not None else episode.camera or "agentview"
        if isinstance(selected_camera, str):
            camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, selected_camera)
            if camera_id < 0:
                available = [
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, index)
                    for index in range(env.model.ncam)
                ]
                raise ValueError(
                    f"Unknown camera {selected_camera!r}; available cameras: {available}"
                )
        previous_stage: int | None = None
        for index in range(start_frame, stop, frame_stride):
            env.data.qpos[:] = arrays["qpos"][index]
            if "qvel" in arrays and arrays["qvel"].shape[1:] == env.data.qvel.shape:
                env.data.qvel[:] = arrays["qvel"][index]
            if "ctrl" in arrays and arrays["ctrl"].shape[1:] == env.data.ctrl.shape:
                env.data.ctrl[:] = arrays["ctrl"][index]
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=selected_camera)
            rgb = renderer.render()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            stage = int(arrays["stage"][index]) if "stage" in arrays else -1
            if overlay:
                stage_label = episode.stage_names.get(stage, str(stage) if stage >= 0 else "")
                title = episode.task.replace("_", " ").title()
                if episode.object_id:
                    title += f"  |  {episode.object_id}"
                cv2.putText(
                    frame,
                    title,
                    (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"frame {index:04d}" + (f"  |  {stage_label}" if stage_label else ""),
                    (12, height - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            written += 1
            last_frame = frame
            if stage != previous_stage:
                print(
                    f"[stage] source_frame={index} {episode.stage_names.get(stage, stage)}",
                    flush=True,
                )
                previous_stage = stage
        if last_frame is not None:
            for _ in range(round(hold_last_seconds * fps)):
                writer.write(last_frame)
                written += 1
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        env.close()

    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "task": episode.task,
                "object_id": episode.object_id,
                "source_manifest": str(episode.manifest_path),
                "video": str(output_path),
                "resolution": [width, height],
                "fps": fps,
                "frame_stride": frame_stride,
                "camera": selected_camera,
                "encoded_frames": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
