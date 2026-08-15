"""Hybrid PPO for categorical wrist-template selection plus continuous hand editing."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from source.rl.ppo import PPOConfig


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


class HybridActorCritic(nn.Module):
    """Categorical wrist-template actor + squashed-Gaussian 6D hand actor."""

    def __init__(
        self,
        obs_dim: int,
        template_count: int,
        hand_action_dim: int,
        config: PPOConfig,
    ) -> None:
        super().__init__()
        if template_count <= 0 or hand_action_dim <= 0:
            raise ValueError("Hybrid actor dimensions must be positive.")
        self.template_count = int(template_count)
        self.hand_action_dim = int(hand_action_dim)
        self.normalizer = RunningNormalizer(obs_dim)
        self.template_actor = _mlp(obs_dim, self.template_count, config.hidden_sizes)
        self.hand_actor = _mlp(obs_dim, self.hand_action_dim, config.hidden_sizes)
        self.critic = _mlp(obs_dim, 1, config.hidden_sizes)
        self.hand_log_std = nn.Parameter(
            torch.full((self.hand_action_dim,), float(np.log(config.initial_std)))
        )

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.normalizer(obs)

    def _distributions(
        self, obs: torch.Tensor
    ) -> tuple[torch.distributions.Categorical, torch.distributions.Normal, torch.Tensor]:
        features = self._features(obs)
        template_distribution = torch.distributions.Categorical(
            logits=self.template_actor(features)
        )
        hand_mean = self.hand_actor(features)
        hand_std = torch.exp(torch.clamp(self.hand_log_std, -5.0, 1.5))
        hand_distribution = torch.distributions.Normal(hand_mean, hand_std)
        return template_distribution, hand_distribution, features

    @staticmethod
    def _hand_log_prob(
        distribution: torch.distributions.Normal,
        raw: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return (
            distribution.log_prob(raw)
            - torch.log(torch.clamp(1.0 - action.square(), min=1e-6))
        ).sum(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        template_dist, hand_dist, features = self._distributions(obs)
        template_id = template_dist.sample()
        hand_raw = hand_dist.rsample()
        hand_action = torch.tanh(hand_raw)
        log_prob = template_dist.log_prob(template_id) + self._hand_log_prob(
            hand_dist, hand_raw, hand_action
        )
        # Environment transport is a dense tensor: column 0 is an exact integer
        # template id encoded as float, columns 1: are the six continuous edits.
        action = torch.cat(
            [template_id.to(dtype=hand_action.dtype).unsqueeze(-1), hand_action], dim=-1
        )
        value = self.critic(features).squeeze(-1)
        return action, log_prob, value

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        template_dist, hand_dist, _ = self._distributions(obs)
        template_id = torch.argmax(template_dist.logits, dim=-1)
        hand_action = torch.tanh(hand_dist.mean)
        return torch.cat(
            [template_id.to(dtype=hand_action.dtype).unsqueeze(-1), hand_action], dim=-1
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        template_dist, hand_dist, features = self._distributions(obs)
        template_id = torch.round(actions[:, 0]).long().clamp(0, self.template_count - 1)
        hand_action = torch.clamp(actions[:, 1:], -0.999999, 0.999999)
        hand_raw = torch.atanh(hand_action)
        log_prob = template_dist.log_prob(template_id) + self._hand_log_prob(
            hand_dist, hand_raw, hand_action
        )
        # This is the entropy of the unsquashed Gaussian plus categorical
        # entropy, matching the project's existing PPO approximation.
        entropy = template_dist.entropy() + hand_dist.entropy().sum(-1)
        value = self.critic(features).squeeze(-1)
        return log_prob, entropy, value


class HybridPPOTrainer:
    """PPO trainer for a hybrid categorical + continuous single-step action."""

    def __init__(self, env, config: PPOConfig | None = None, *, seed: int = 0) -> None:
        self.env = env
        self.config = config or PPOConfig()
        self.config.validate()
        if not hasattr(env, "template_count") or not hasattr(env, "hand_action_dim"):
            raise TypeError("Hybrid PPO environment must expose template_count and hand_action_dim.")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = env.torch_device
        self.model = HybridActorCritic(
            env.obs_dim,
            env.template_count,
            env.hand_action_dim,
            self.config,
        ).to(self.device)
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
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
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
        return {
            name: float(np.mean(values)) if values else 0.0
            for name, values in metrics.items()
        }

    def train(self, updates: int, *, callback=None) -> None:
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
                "trainer_type": "grasp_edit_hybrid_v10",
                "template_count": int(self.env.template_count),
                "hand_action_dim": int(self.env.hand_action_dim),
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
        if payload.get("trainer_type") != "grasp_edit_hybrid_v10":
            raise ValueError("Checkpoint is not a v10 hybrid grasp-edit PPO checkpoint.")
        if int(payload.get("template_count", -1)) != int(self.env.template_count):
            raise ValueError("Checkpoint template count differs from the current lattice.")
        if int(payload.get("hand_action_dim", -1)) != int(self.env.hand_action_dim):
            raise ValueError("Checkpoint hand action dimension differs from the current environment.")
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
