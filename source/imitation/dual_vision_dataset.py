"""Root-separated P11 global/ROI dataset reader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DualVisionRootDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        selected = [row for row in self.manifest["rows"] if row["split"] == split]
        if not selected:
            raise ValueError(f"No roots in split {split!r}")
        self.roots = []
        self.index = []
        for root_index, row in enumerate(selected):
            with np.load(row["path"], allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            length = len(arrays["state"])
            if any(len(arrays[key]) != length for key in ("global_rgb", "local_rgb", "action")):
                raise ValueError(f"Misaligned root arrays: {row['path']}")
            self.roots.append(arrays)
            self.index.extend((root_index, item) for item in range(length))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        root_index, item = self.index[index]
        arrays = self.roots[root_index]
        global_rgb = torch.from_numpy(arrays["global_rgb"][item]).permute(2, 0, 1).float() / 255
        local_rgb = torch.from_numpy(arrays["local_rgb"][item]).permute(2, 0, 1).float() / 255
        return {
            "global_rgb": global_rgb,
            "local_rgb": local_rgb,
            "state": torch.from_numpy(arrays["state"][item]).float(),
            "action": torch.from_numpy(arrays["action"][item]).float(),
        }

    def moments(self):
        states = np.concatenate([root["state"] for root in self.roots], axis=0)
        actions = np.concatenate([root["action"] for root in self.roots], axis=0)
        return tuple(
            torch.from_numpy(value).float()
            for value in (
                states.mean(0),
                states.std(0),
                actions.reshape(-1, actions.shape[-1]).mean(0),
                actions.reshape(-1, actions.shape[-1]).std(0),
            )
        )
