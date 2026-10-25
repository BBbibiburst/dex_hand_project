"""Small bounded residual policies for frozen Cartesian behavior cloning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


def tactile_region_features(tactile: np.ndarray, groups: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(tactile, dtype=np.float32).reshape(-1)
    digit_values = [float(values[groups[f"digit_{index}"]].sum()) for index in range(5)]
    return np.asarray([values.sum(), digit_values[4], sum(digit_values[:4]), *digit_values])


@dataclass(frozen=True)
class CartesianResidualConfig:
    input_dim: int
    horizon: int = 4
    hidden_dim: int = 128
    maximum_residual_m: float = 0.005
    use_tactile: bool = False


class CartesianResidualMLP(nn.Module):
    def __init__(self, config: CartesianResidualConfig):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.horizon * 3),
        )
        self.register_buffer("input_mean", torch.zeros(config.input_dim))
        self.register_buffer("input_std", torch.ones(config.input_dim))
        self.register_buffer("target_scale", torch.ones(3))

    def set_normalization(self, inputs, targets):
        self.input_mean.copy_(inputs.mean(0))
        self.input_std.copy_(inputs.std(0).clamp_min(1e-6))
        self.target_scale.copy_(targets.reshape(-1, 3).std(0).clamp_min(1e-4))

    def forward(self, inputs):
        normalized = (inputs - self.input_mean) / self.input_std
        values = self.network(normalized).reshape(-1, self.config.horizon, 3)
        residual = values * self.target_scale
        return torch.clamp(
            residual, -self.config.maximum_residual_m, self.config.maximum_residual_m
        )

    def checkpoint(self):
        return {"config": asdict(self.config), "model": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, payload, device="cpu"):
        model = cls(CartesianResidualConfig(**payload["config"]))
        model.load_state_dict(payload["model"])
        return model.to(device)
