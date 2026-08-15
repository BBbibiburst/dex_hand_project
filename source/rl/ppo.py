"""Small self-contained PPO implementation for CUDA vector environments.

Keeping PPO in-tree avoids coupling the project to Isaac Lab or an additional RL
framework.  Physics stays in MJWarp and the policy/value networks stay in
PyTorch; the environment interface intentionally mirrors the tiny subset needed
by this trainer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 64
    update_epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    target_kl: float = 0.03
    initial_std: float = 0.55
    hidden_sizes: tuple[int, ...] = (256, 256, 128)

    def validate(self) -> None:
        if min(self.rollout_steps, self.update_epochs, self.minibatches) <= 0:
            raise ValueError("PPO rollout/update/minibatch counts must be positive.")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("PPO gamma and gae_lambda must lie in (0, 1].")
        if self.learning_rate <= 0.0 or self.initial_std <= 0.0:
            raise ValueError("PPO learning_rate and initial_std must be positive.")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("All PPO hidden layer sizes must be positive.")


class RunningNormalizer(nn.Module):
    def __init__(self, size: int, *, clip: float = 10.0) -> None:
        super().__init__()
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        batch = values.reshape(-1, values.shape[-1]).float()
        batch_mean = batch.mean(0)
        batch_var = batch.var(0, unbiased=False)
        batch_count = float(len(batch))
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(m2 / total, min=1e-6))
        self.count.copy_(total)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.mean) / torch.sqrt(self.var + 1e-6)
        return torch.clamp(normalized, -self.clip, self.clip)


def _mlp(input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        layer = nn.Linear(previous, size)
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.ELU()])
        previous = size
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig) -> None:
        super().__init__()
        self.normalizer = RunningNormalizer(obs_dim)
        self.actor = _mlp(obs_dim, action_dim, config.hidden_sizes)
        self.critic = _mlp(obs_dim, 1, config.hidden_sizes)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(config.initial_std))))

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.normalizer(obs)

    def _distribution(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        features = self._features(obs)
        mean = self.actor(features)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.5))
        return torch.distributions.Normal(mean, std), features

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, features = self._distribution(obs)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_prob = (
            distribution.log_prob(raw) - torch.log(torch.clamp(1.0 - action.square(), min=1e-6))
        ).sum(-1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, value

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        distribution, _ = self._distribution(obs)
        return torch.tanh(distribution.mean)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, features = self._distribution(obs)
        clipped = torch.clamp(actions, -0.999999, 0.999999)
        raw = torch.atanh(clipped)
        log_prob = (
            distribution.log_prob(raw) - torch.log(torch.clamp(1.0 - clipped.square(), min=1e-6))
        ).sum(-1)
        entropy = distribution.entropy().sum(-1)
        value = self.critic(features).squeeze(-1)
        return log_prob, entropy, value


class PPOTrainer:
    def __init__(
        self,
        env,
        config: PPOConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.config = config or PPOConfig()
        self.config.validate()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = env.torch_device
        self.model = ActorCritic(env.obs_dim, env.action_dim, self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.obs = env.reset()
        self.total_steps = 0
        self.update_index = 0

    def _collect_rollout(self) -> dict[str, torch.Tensor]:
        cfg = self.config
        shape = (cfg.rollout_steps, self.env.num_envs)
        obs = torch.empty((*shape, self.env.obs_dim), device=self.device)
        actions = torch.empty((*shape, self.env.action_dim), device=self.device)
        log_probs = torch.empty(shape, device=self.device)
        rewards = torch.empty(shape, device=self.device)
        dones = torch.empty(shape, device=self.device)
        values = torch.empty(shape, device=self.device)

        self.model.normalizer.update(self.obs)
        for step in range(cfg.rollout_steps):
            obs[step] = self.obs
            with torch.no_grad():
                action, log_prob, value = self.model.act(self.obs)
            next_obs, reward, done, _ = self.env.step(action)
            actions[step] = action
            log_probs[step] = log_prob
            rewards[step] = reward
            dones[step] = done.float()
            values[step] = value
            self.obs = next_obs
            self.total_steps += self.env.num_envs

        # Freeze observation-normalization statistics for the entire rollout and
        # PPO update.  Otherwise old and recomputed log probabilities would use
        # different normalized observations and invalidate the importance ratio.
        with torch.no_grad():
            last_value = self.model.value(self.obs)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(self.env.num_envs, device=self.device)
        for step in reversed(range(cfg.rollout_steps)):
            next_value = last_value if step == cfg.rollout_steps - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + cfg.gamma * next_value * nonterminal - values[step]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values
        return {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "values": values,
            "rewards": rewards,
            "dones": dones,
        }

    def _update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.config
        batch_size = cfg.rollout_steps * self.env.num_envs
        obs = rollout["obs"].reshape(batch_size, self.env.obs_dim)
        actions = rollout["actions"].reshape(batch_size, self.env.action_dim)
        old_log_probs = rollout["log_probs"].reshape(batch_size)
        advantages = rollout["advantages"].reshape(batch_size)
        returns = rollout["returns"].reshape(batch_size)
        old_values = rollout["values"].reshape(batch_size)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        minibatch_size = max(1, batch_size // cfg.minibatches)

        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
        }
        stop = False
        for _ in range(cfg.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                log_prob, entropy, value = self.model.evaluate_actions(obs[idx], actions[idx])
                ratio = torch.exp(log_prob - old_log_probs[idx])
                unclipped = ratio * advantages[idx]
                clipped = (
                    torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                    * advantages[idx]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                value_delta = value - old_values[idx]
                value_clipped = old_values[idx] + torch.clamp(
                    value_delta, -cfg.clip_ratio, cfg.clip_ratio
                )
                value_loss = 0.5 * torch.maximum(
                    (value - returns[idx]).square(),
                    (value_clipped - returns[idx]).square(),
                ).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                approx_kl = (old_log_probs[idx] - log_prob).mean().detach()
                metrics["policy_loss"].append(float(policy_loss.detach()))
                metrics["value_loss"].append(float(value_loss.detach()))
                metrics["entropy"].append(float(entropy_mean.detach()))
                metrics["kl"].append(float(approx_kl))
                if float(approx_kl) > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}

    def train(
        self,
        updates: int,
        *,
        callback=None,
    ) -> None:
        if updates <= 0:
            raise ValueError("updates must be positive.")
        for _ in range(updates):
            rollout = self._collect_rollout()
            update_metrics = self._update(rollout)
            self.update_index += 1
            metrics: dict[str, Any] = {
                **update_metrics,
                "update": self.update_index,
                "total_steps": self.total_steps,
                "mean_reward": float(rollout["rewards"].mean()),
                **self.env.training_metrics(),
            }
            if callback is not None:
                callback(self, metrics)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update": self.update_index,
                "total_steps": self.total_steps,
                "ppo_config": asdict(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        stored_config = PPOConfig(**payload["ppo_config"])
        if stored_config != self.config:
            raise ValueError(
                "Checkpoint PPO configuration differs from the requested training configuration."
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_index = int(payload.get("update", 0))
        self.total_steps = int(payload.get("total_steps", 0))
        self.obs = self.env.reset()
