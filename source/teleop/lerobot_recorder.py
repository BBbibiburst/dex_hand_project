"""Thin adapter around the official LeRobotDataset incremental writer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from source.imitation.schema import (
    ACTION_KEY,
    AGENTVIEW_IMAGE_KEY,
    OPERATOR_GLOVE_KEY,
    OPERATOR_VIVE_POSE_KEY,
    STATE_KEY,
    TACTILE_KEY,
    TASK_KEY,
)


class LeRobotEpisodeRecorder:
    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path,
        fps: int,
        state_dim: int,
        action_dim: int,
        tactile_shape: tuple[int, ...],
        image_shape: tuple[int, int, int],
        robot_type: str = "dex_hand_project",
        use_videos: bool = True,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobotDataset could not be imported. Install a complete "
                "lerobot>=0.4 package in the active Python environment."
            ) from exc

        self.has_tactile = bool(np.prod(tactile_shape, dtype=np.int64))
        features = {
            STATE_KEY: {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": None,
            },
            OPERATOR_GLOVE_KEY: {
                "dtype": "float32",
                "shape": (6,),
                "names": None,
            },
            OPERATOR_VIVE_POSE_KEY: {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
            AGENTVIEW_IMAGE_KEY: {
                "dtype": "video" if use_videos else "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            ACTION_KEY: {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": None,
            },
        }
        if self.has_tactile:
            features[TACTILE_KEY] = {
                "dtype": "float32",
                "shape": tactile_shape,
                "names": None,
            }

        root = Path(root)
        if root.exists():
            info_path = root / "meta" / "info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                total_episodes = int(info.get("total_episodes", 0))
                total_frames = int(info.get("total_frames", 0))
                if total_episodes == 0 and total_frames == 0:
                    archived = _archive_incomplete_dataset(root)
                    print(f"Archived an incomplete zero-frame LeRobot dataset: {archived}")
                    self.dataset = _create_dataset(
                        LeRobotDataset,
                        repo_id=repo_id,
                        root=root,
                        fps=fps,
                        robot_type=robot_type,
                        features=features,
                        use_videos=use_videos,
                    )
                else:
                    self.dataset = LeRobotDataset(repo_id=repo_id, root=root)
                    _validate_dataset_compatibility(
                        self.dataset,
                        fps=fps,
                        features=features,
                    )
                    print(
                        f"Resuming LeRobot dataset at {root}: "
                        f"{self.dataset.meta.total_episodes} existing episodes"
                    )
            elif any(root.iterdir()):
                raise RuntimeError(
                    f"Dataset output directory exists but has no LeRobot metadata: {root}. "
                    "Choose another --output path."
                )
            else:
                archived = _archive_incomplete_dataset(root)
                print(f"Archived an empty dataset directory: {archived}")
                self.dataset = _create_dataset(
                    LeRobotDataset,
                    repo_id=repo_id,
                    root=root,
                    fps=fps,
                    robot_type=robot_type,
                    features=features,
                    use_videos=use_videos,
                )
        else:
            self.dataset = _create_dataset(
                LeRobotDataset,
                repo_id=repo_id,
                root=root,
                fps=fps,
                robot_type=robot_type,
                features=features,
                use_videos=use_videos,
            )
        self.frame_count = 0

    def add_frame(self, *, observation, image, action, glove, vive, task: str) -> None:
        state = np.concatenate(
            [observation["qpos"], observation["qvel"], observation["ctrl"]]
        ).astype(np.float32)
        frame = {
            STATE_KEY: state,
            OPERATOR_GLOVE_KEY: np.asarray(glove.stretch, dtype=np.float32),
            OPERATOR_VIVE_POSE_KEY: np.concatenate([vive.position, vive.quaternion_wxyz]).astype(
                np.float32
            ),
            AGENTVIEW_IMAGE_KEY: np.asarray(image, dtype=np.uint8),
            ACTION_KEY: np.asarray(action, dtype=np.float32),
            TASK_KEY: task,
        }
        if self.has_tactile:
            frame[TACTILE_KEY] = np.asarray(observation["tactile"], dtype=np.float32)
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def save_episode(self) -> None:
        if self.frame_count:
            self.frame_count = 0
            self.dataset.save_episode()

    def clear_episode(self) -> None:
        if hasattr(self.dataset, "clear_episode_buffer"):
            self.dataset.clear_episode_buffer()
        self.frame_count = 0

    def finalize(self) -> None:
        if self.frame_count:
            self.save_episode()
        finalize = getattr(self.dataset, "finalize", None)
        if finalize is not None:
            finalize()


def _create_dataset(dataset_type, **kwargs):
    """Create a new dataset while keeping constructor details in one place."""
    return dataset_type.create(**kwargs)


def _archive_incomplete_dataset(root: Path) -> Path:
    """Preserve a zero-frame partial dataset before recreating its output path."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    candidate = root.with_name(f"{root.name}.incomplete-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = root.with_name(f"{root.name}.incomplete-{timestamp}-{suffix}")
        suffix += 1
    root.replace(candidate)
    return candidate


def _validate_dataset_compatibility(dataset, *, fps: int, features: dict) -> None:
    """Reject appends that would mix incompatible recording schemas."""
    if int(dataset.fps) != int(fps):
        raise ValueError(f"Existing dataset FPS is {dataset.fps}, but this run requested {fps}.")
    existing = dataset.features
    for name, expected in features.items():
        if name not in existing:
            raise ValueError(f"Existing dataset is missing feature {name!r}.")
        actual = existing[name]
        if str(actual.get("dtype")) != str(expected["dtype"]):
            raise ValueError(
                f"Existing feature {name!r} has dtype {actual.get('dtype')!r}; "
                f"expected {expected['dtype']!r}."
            )
        if tuple(actual.get("shape", ())) != tuple(expected["shape"]):
            raise ValueError(
                f"Existing feature {name!r} has shape {actual.get('shape')!r}; "
                f"expected {expected['shape']!r}."
            )
