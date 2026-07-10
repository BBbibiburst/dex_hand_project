"""LeRobot sequence adapter for vision-tactile diffusion policy training."""
from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import Dataset


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
            delta_timestamps={"action": [step / fps for step in range(horizon)]},
        )
        self.image_key = "observation.images.agentview"
        self.has_tactile = "observation.tactile" in self.dataset.features

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
            item["observation.tactile"].float().flatten()
            if self.has_tactile
            else torch.zeros(0, dtype=torch.float32)
        )
        action = item["action"].float()
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return {
            "image": image,
            "tactile": tactile,
            "state": item["observation.state"].float(),
            "action": action,
        }
