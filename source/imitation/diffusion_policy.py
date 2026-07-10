"""Compact conditional DDPM for fused RGB, tactile and robot state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    state_dim: int
    tactile_dim: int
    action_dim: int
    horizon: int = 16
    diffusion_steps: int = 50
    feature_dim: int = 256


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(-scale * torch.arange(half, device=time.device))
        values = time.float()[:, None] * frequencies[None]
        return torch.cat([values.sin(), values.cos()], dim=-1)


class VisionEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.SiLU(),
        )

    def forward(self, image):
        return self.network(image)


class DiffusionPolicy(nn.Module):
    def __init__(self, config: DiffusionPolicyConfig):
        super().__init__()
        self.config = config
        d = config.feature_dim
        self.vision = VisionEncoder(d)
        self.state = nn.Sequential(nn.Linear(config.state_dim, d), nn.LayerNorm(d), nn.SiLU())
        self.tactile = (
            nn.Sequential(nn.Linear(config.tactile_dim, d), nn.LayerNorm(d), nn.SiLU())
            if config.tactile_dim
            else None
        )
        modality_count = 3 if self.tactile is not None else 2
        self.condition = nn.Sequential(nn.Linear(modality_count * d, d), nn.SiLU())
        self.time = nn.Sequential(SinusoidalTimeEmbedding(d), nn.Linear(d, d), nn.SiLU())
        input_dim = config.horizon * config.action_dim
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim + 2 * d, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, input_dim),
        )
        betas = torch.linspace(1e-4, 0.02, config.diffusion_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))
        self.register_buffer("tactile_mean", torch.zeros(config.tactile_dim))
        self.register_buffer("tactile_std", torch.ones(config.tactile_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))

    def set_normalization(
        self, state_mean, state_std, tactile_mean, tactile_std, action_mean, action_std
    ):
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std.clamp_min(1e-6))
        if self.config.tactile_dim:
            self.tactile_mean.copy_(tactile_mean)
            self.tactile_std.copy_(tactile_std.clamp_min(1e-6))
        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std.clamp_min(1e-6))

    def encode(self, image, tactile, state):
        features = [self.vision(image), self.state((state - self.state_mean) / self.state_std)]
        if self.tactile is not None:
            features.append(self.tactile((tactile - self.tactile_mean) / self.tactile_std))
        return self.condition(torch.cat(features, dim=-1))

    def _predict_noise(self, noisy, time, condition):
        batch = noisy.shape[0]
        flat = noisy.reshape(batch, -1)
        return self.denoiser(torch.cat([flat, self.time(time), condition], -1)).reshape_as(noisy)

    def loss(self, image, tactile, state, action):
        action = (action - self.action_mean) / self.action_std
        batch = action.shape[0]
        time = torch.randint(0, self.config.diffusion_steps, (batch,), device=action.device)
        noise = torch.randn_like(action)
        alpha_bar = self.alpha_bars[time].view(batch, 1, 1)
        noisy = alpha_bar.sqrt() * action + (1 - alpha_bar).sqrt() * noise
        predicted = self._predict_noise(noisy, time, self.encode(image, tactile, state))
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(self, image, tactile, state):
        condition = self.encode(image, tactile, state)
        action = torch.randn(
            image.shape[0], self.config.horizon, self.config.action_dim, device=image.device
        )
        for step in reversed(range(self.config.diffusion_steps)):
            time = torch.full((image.shape[0],), step, device=image.device, dtype=torch.long)
            noise = self._predict_noise(action, time, condition)
            alpha, alpha_bar, beta = self.alphas[step], self.alpha_bars[step], self.betas[step]
            action = (action - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise) / torch.sqrt(alpha)
            if step:
                action += beta.sqrt() * torch.randn_like(action)
        return action * self.action_std + self.action_mean

    def checkpoint(self):
        return {"config": asdict(self.config), "model": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        model = cls(DiffusionPolicyConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        return model.to(device)
