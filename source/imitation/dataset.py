"""LeRobot sequence adapter for vision-tactile diffusion policy training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from source.imitation.schema import (
    ACTION_KEY,
    AGENTVIEW_IMAGE_KEY,
    STATE_KEY,
    TACTILE_KEY,
)


class LeRobotDiffusionDataset(Dataset):
    def __init__(self, *, repo_id: str, root: str | Path, horizon: int):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"LeRobot dataset not found: {root}")
        # LeRobot pads timestamps beyond an episode boundary, which is the
        # desired diffusion-policy action-horizon behavior.
        metadata = LeRobotDataset(repo_id, root=root)
        fps = metadata.fps
        self.dataset = LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps={ACTION_KEY: [step / fps for step in range(horizon)]},
        )
        self.image_key = AGENTVIEW_IMAGE_KEY
        self.has_tactile = TACTILE_KEY in self.dataset.features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[self.image_key].float()
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.max() > 1:
            image = image / 255.0
        tactile = (
            item[TACTILE_KEY].float().flatten()
            if self.has_tactile
            else torch.zeros(0, dtype=torch.float32)
        )
        action = item[ACTION_KEY].float()
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return {
            "image": image,
            "tactile": tactile,
            "state": item[STATE_KEY].float(),
            "action": action,
        }
