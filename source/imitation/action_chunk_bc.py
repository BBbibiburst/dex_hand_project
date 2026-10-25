"""Deterministic action-chunk behavior-cloning baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from source.imitation.diffusion_policy import VisionEncoder


@dataclass(frozen=True)
class ActionChunkBCConfig:
    state_dim: int
    action_dim: int
    horizon: int = 16
    feature_dim: int = 256


class ActionChunkBC(nn.Module):
    """Map one deployable RGB+state observation to an absolute-action chunk."""

    def __init__(self, config: ActionChunkBCConfig):
        super().__init__()
        self.config = config
        d = config.feature_dim
        self.vision = VisionEncoder(d)
        self.state = nn.Sequential(nn.Linear(config.state_dim, d), nn.LayerNorm(d), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(2 * d, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, config.horizon * config.action_dim),
        )
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))

    def set_normalization(self, state_mean, state_std, action_mean, action_std) -> None:
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std.clamp_min(1e-6))
        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std.clamp_min(1e-6))

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        condition = torch.cat(
            [
                self.vision(image),
                self.state((state - self.state_mean) / self.state_std),
            ],
            dim=-1,
        )
        normalized = self.head(condition).reshape(
            image.shape[0], self.config.horizon, self.config.action_dim
        )
        return normalized * self.action_std + self.action_mean

    def loss(self, image, state, action) -> torch.Tensor:
        target = (action - self.action_mean) / self.action_std
        condition = torch.cat(
            [
                self.vision(image),
                self.state((state - self.state_mean) / self.state_std),
            ],
            dim=-1,
        )
        predicted = self.head(condition).reshape_as(target)
        return F.mse_loss(predicted, target)

    @torch.inference_mode()
    def sample(self, image, state) -> torch.Tensor:
        return self.forward(image, state)

    def checkpoint(self) -> dict:
        return {
            "policy_type": "action_chunk_bc",
            "config": asdict(self.config),
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: str = "cpu") -> "ActionChunkBC":
        model = cls(ActionChunkBCConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        return model.to(device)
