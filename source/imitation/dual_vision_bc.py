"""Global RGB plus fixed high-resolution ROI action-chunk behavior cloning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from source.imitation.diffusion_policy import VisionEncoder


@dataclass(frozen=True)
class DualVisionBCConfig:
    state_dim: int
    action_dim: int
    horizon: int = 16
    global_feature_dim: int = 256
    local_feature_dim: int = 128
    state_feature_dim: int = 256


class DualVisionActionChunkBC(nn.Module):
    """Predict the existing XYZ9 chunk without changing its control contract."""

    def __init__(self, config: DualVisionBCConfig):
        super().__init__()
        self.config = config
        self.global_vision = VisionEncoder(config.global_feature_dim)
        self.local_vision = VisionEncoder(config.local_feature_dim)
        self.state = nn.Sequential(
            nn.Linear(config.state_dim, config.state_feature_dim),
            nn.LayerNorm(config.state_feature_dim),
            nn.SiLU(),
        )
        fused = config.global_feature_dim + config.local_feature_dim + config.state_feature_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, config.horizon * config.action_dim),
        )
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))

    def set_normalization(self, state_mean, state_std, action_mean, action_std):
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std.clamp_min(1e-6))
        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std.clamp_min(1e-6))

    def _features(self, global_rgb, local_rgb, state):
        return torch.cat(
            (
                self.global_vision(global_rgb),
                self.local_vision(local_rgb),
                self.state((state - self.state_mean) / self.state_std),
            ),
            dim=-1,
        )

    def forward(self, global_rgb, local_rgb, state):
        normalized = self.head(self._features(global_rgb, local_rgb, state))
        normalized = normalized.reshape(
            global_rgb.shape[0], self.config.horizon, self.config.action_dim
        )
        return normalized * self.action_std + self.action_mean

    def loss(self, global_rgb, local_rgb, state, action):
        target = (action - self.action_mean) / self.action_std
        predicted = self.head(self._features(global_rgb, local_rgb, state)).reshape_as(target)
        return F.mse_loss(predicted, target)

    @torch.inference_mode()
    def sample(self, global_rgb, local_rgb, state):
        return self.forward(global_rgb, local_rgb, state)

    def checkpoint(self):
        return {
            "policy_type": "dual_vision_action_chunk_bc",
            "config": asdict(self.config),
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        model = cls(DualVisionBCConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        return model.to(device)
