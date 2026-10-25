"""Small deployable state-history controllers for the Lift phase."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LiftTemporalConfig:
    state_dim: int = 39
    history: int = 8
    horizon: int = 4
    hidden_dim: int = 256
    kind: str = "history_mlp"
    include_action_history: bool = True


class LiftTemporalController(nn.Module):
    def __init__(self, config: LiftTemporalConfig):
        super().__init__()
        self.config = config
        input_dim = config.state_dim + (3 if config.include_action_history else 0)
        if config.kind == "single_mlp":
            encoder_dim = input_dim
            self.encoder = None
        elif config.kind == "history_mlp":
            encoder_dim = config.history * input_dim
            self.encoder = None
        elif config.kind == "gru":
            encoder_dim = config.hidden_dim
            self.encoder = nn.GRU(input_dim, config.hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown Lift controller kind: {config.kind}")
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.horizon * 3),
        )
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))

    def set_normalization(self, inputs, targets) -> None:
        self.input_mean.copy_(inputs.reshape(-1, inputs.shape[-1]).mean(0))
        self.input_std.copy_(inputs.reshape(-1, inputs.shape[-1]).std(0).clamp_min(1e-6))
        self.target_mean.copy_(targets.reshape(-1, 3).mean(0))
        self.target_std.copy_(targets.reshape(-1, 3).std(0).clamp_min(1e-6))

    def _features(self, states, actions):
        values = torch.cat([states, actions], -1) if self.config.include_action_history else states
        values = (values - self.input_mean) / self.input_std
        if self.config.kind == "single_mlp":
            return values[:, -1]
        if self.config.kind == "history_mlp":
            return values.flatten(1)
        _, hidden = self.encoder(values)
        return hidden[-1]

    def forward(self, states, actions):
        normalized = self.head(self._features(states, actions)).reshape(-1, self.config.horizon, 3)
        return normalized * self.target_std + self.target_mean

    def loss(self, states, actions, targets):
        normalized = (targets - self.target_mean) / self.target_std
        predicted = self.head(self._features(states, actions)).reshape_as(normalized)
        return nn.functional.mse_loss(predicted, normalized)

    def checkpoint(self):
        return {"policy_type": "lift_temporal", "config": asdict(self.config), "model": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        model = cls(LiftTemporalConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        return model.to(device)
