"""Thin adapter around the official LeRobotDataset incremental writer."""
from __future__ import annotations

from pathlib import Path
import numpy as np


class LeRobotEpisodeRecorder:
    def __init__(self, *, repo_id: str, root: str | Path, fps: int,
                 state_dim: int, action_dim: int, tactile_shape: tuple[int, ...],
                 image_shape: tuple[int, int, int], robot_type="dex_hand_project",
                 use_videos=True):
        try:
            # lerobot 0.4.x keeps ``datasets`` as a namespace package and does
            # not re-export LeRobotDataset from lerobot.datasets.__init__.
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobotDataset could not be imported. Install a complete "
                "lerobot>=0.4 package in the active Python environment."
            ) from exc
        self.has_tactile = bool(np.prod(tactile_shape, dtype=np.int64))
        features = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "observation.operator.glove": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.operator.vive_pose": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.images.agentview": {"dtype": "video" if use_videos else "image", "shape": image_shape, "names": ["height", "width", "channels"]},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        }
        if self.has_tactile:
            features["observation.tactile"] = {
                "dtype": "float32",
                "shape": tactile_shape,
                "names": None,
            }
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id, root=Path(root), fps=fps, robot_type=robot_type,
            features=features, use_videos=use_videos,
        )
        self.frame_count = 0

    def add_frame(self, *, observation, image, action, glove, vive, task: str) -> None:
        state = np.concatenate([observation["qpos"], observation["qvel"], observation["ctrl"]]).astype(np.float32)
        tactile = np.asarray(observation["tactile"], dtype=np.float32)
        frame = {
            "observation.state": state,
            "observation.operator.glove": np.asarray(glove.stretch, dtype=np.float32),
            "observation.operator.vive_pose": np.concatenate(
                [vive.position, vive.quaternion_wxyz]
            ).astype(np.float32),
            "observation.images.agentview": np.asarray(image, dtype=np.uint8),
            "action": np.asarray(action, dtype=np.float32),
            "task": task,
        }
        if self.has_tactile:
            frame["observation.tactile"] = tactile
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def save_episode(self) -> None:
        if self.frame_count:
            # Clear our pending flag before committing so a failing save is
            # not attempted again by ``finalize`` during exception cleanup.
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
