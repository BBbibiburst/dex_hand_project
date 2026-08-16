"""Behavior-cloning utilities for a grasp-formation prior.

The policy deliberately predicts only the six physical Dex Hand actuator
commands.  Arm motion stays anchored to the per-object reference trajectory;
PPO is responsible for residual arm corrections later.  This avoids averaging
incompatible RM75B joint trajectories across different objects while still
teaching the missing behavior observed in free residual RL: close the hand,
form thumb/finger opposition, and keep the grasp closed through lift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from source.rl.residual.env import MjWarpResidualLiftEnv, ResidualLiftConfig
from source.rl.residual.reference import STAGE_CODES


BC_SCHEMA_VERSION = 1
DEFAULT_HIDDEN_SIZES = (256, 256, 128)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


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


class BCHandPolicy(nn.Module):
    """State-conditioned six-drive hand policy trained from successful grasps."""

    def __init__(
        self,
        obs_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        ignored_tail_dim: int = 13,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        if ignored_tail_dim < 0 or ignored_tail_dim >= obs_dim:
            raise ValueError("ignored_tail_dim must be non-negative and smaller than obs_dim.")
        self.obs_dim = int(obs_dim)
        self.hand_action_dim = 6
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.ignored_tail_dim = int(ignored_tail_dim)
        self.actor = _mlp(self.obs_dim, self.hand_action_dim, self.hidden_sizes)
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_var", torch.ones(self.obs_dim))

    def _prepare(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observation dim {self.obs_dim}, got {obs.shape[-1]}.")
        values = obs.float()
        if self.ignored_tail_dim:
            values = values.clone()
            values[..., -self.ignored_tail_dim :] = 0.0
        values = (values - self.obs_mean) / torch.sqrt(self.obs_var + 1e-6)
        return torch.clamp(values, -10.0, 10.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(self._prepare(obs)))

    @torch.no_grad()
    def set_observation_statistics(self, observations: torch.Tensor) -> None:
        values = observations.float().clone()
        if self.ignored_tail_dim:
            values[..., -self.ignored_tail_dim :] = 0.0
        self.obs_mean.copy_(values.mean(dim=0))
        self.obs_var.copy_(torch.clamp(values.var(dim=0, unbiased=False), min=1e-6))


@dataclass(frozen=True)
class BCDatasetInfo:
    observations: int
    experts: int
    objects: int
    obs_dim: int
    hand_action_dim: int
    actuator_names: tuple[str, ...]
    expert_manifests: tuple[str, ...]


@dataclass(frozen=True)
class BCTrainConfig:
    epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    seed: int = 0
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    ignored_tail_dim: int = 13

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("BC epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid BC optimizer parameters.")
        if any(value <= 0 for value in self.hidden_sizes):
            raise ValueError("BC hidden sizes must be positive.")


def _stage_weight(stage: int) -> float:
    # Grasp formation receives the strongest supervision.  Approach is retained
    # so the learned hand prior knows to stay relatively open before closure.
    if stage == STAGE_CODES["approach"]:
        return 0.50
    if stage == STAGE_CODES["close"]:
        return 4.00
    if stage == STAGE_CODES["hold"]:
        return 5.00
    if stage == STAGE_CODES["lift"]:
        return 2.50
    if stage == STAGE_CODES["verify"]:
        return 2.50
    return 0.25


def collect_bc_dataset(
    expert_manifests: Iterable[str | Path],
    *,
    output: str | Path,
    device: str = "cuda:0",
    nconmax: int = 192,
    njmax: int = 768,
) -> BCDatasetInfo:
    """Replay successful references with zero residual and record state->hand action pairs."""
    manifests = tuple(Path(path) for path in expert_manifests)
    if not manifests:
        raise ValueError("No expert manifests were supplied for behavior cloning.")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    weights: list[float] = []
    stages: list[int] = []
    object_indices: list[int] = []
    object_ids: list[str] = []
    actuator_names: tuple[str, ...] | None = None
    obs_dim: int | None = None

    for expert_index, manifest in enumerate(manifests):
        env = MjWarpResidualLiftEnv(
            manifest,
            ResidualLiftConfig(
                num_envs=1,
                device=device,
                action_mode="arm_hand",
                start_stage="approach",
                nconmax=nconmax,
                njmax=njmax,
            ),
        )
        try:
            if env.action_dim != 13 or env.reference.hand_action_size != 6:
                raise ValueError(
                    "BC collection expects RM75B(7)+DexHand(6), got "
                    f"action_dim={env.action_dim}, hand={env.reference.hand_action_size}."
                )
            names = tuple(env.reference.actuator_names)
            if actuator_names is None:
                actuator_names = names
            elif actuator_names != names:
                raise ValueError("Expert actuator ordering is inconsistent across trajectories.")
            if obs_dim is None:
                obs_dim = env.obs_dim
            elif obs_dim != env.obs_dim:
                raise ValueError("Expert observation dimensions are inconsistent.")

            object_index = len(object_ids)
            object_ids.append(env.reference.object_id)
            zero_action = torch.zeros((1, env.action_dim), device=env.torch_device)
            hand_start = env.reference.arm_action_size
            hand_low = env.ctrl_low[hand_start:].detach().cpu().numpy()
            hand_high = env.ctrl_high[hand_start:].detach().cpu().numpy()
            hand_span = np.maximum(hand_high - hand_low, 1e-8)
            obs = env.reset()

            for step in range(env.reference.horizon):
                stage = int(round(float(env.reference.stages[step])))
                control = env.reference.controls[step, hand_start:]
                normalized = np.clip(2.0 * (control - hand_low) / hand_span - 1.0, -1.0, 1.0)
                observations.append(obs[0].detach().cpu().numpy().astype(np.float32))
                actions.append(np.asarray(normalized, dtype=np.float32))
                weights.append(_stage_weight(stage))
                stages.append(stage)
                object_indices.append(object_index)
                obs, _, _, _ = env.step(zero_action)
        finally:
            env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"[bc:data] {expert_index + 1:03d}/{len(manifests):03d} "
            f"object={object_ids[-1]} manifest={manifest}",
            flush=True,
        )

    obs_array = np.stack(observations).astype(np.float32)
    action_array = np.stack(actions).astype(np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    stage_array = np.asarray(stages, dtype=np.int16)
    object_array = np.asarray(object_indices, dtype=np.int32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=obs_array,
        hand_actions=action_array,
        weights=weight_array,
        stages=stage_array,
        object_indices=object_array,
    )
    info = BCDatasetInfo(
        observations=len(obs_array),
        experts=len(manifests),
        objects=len(set(object_ids)),
        obs_dim=int(obs_array.shape[1]),
        hand_action_dim=int(action_array.shape[1]),
        actuator_names=tuple(actuator_names or ()),
        expert_manifests=tuple(str(path) for path in manifests),
    )
    _atomic_json(
        output.with_suffix(".json"),
        {
            "schema_version": BC_SCHEMA_VERSION,
            "dataset": asdict(info),
            "stage_weights": {name: _stage_weight(code) for name, code in STAGE_CODES.items()},
        },
    )
    return info


def load_bc_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != BC_SCHEMA_VERSION:
        raise ValueError(f"Unsupported BC dataset schema: {metadata.get('schema_version')}.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"observations", "hand_actions", "weights", "stages", "object_indices"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"BC dataset is missing arrays: {sorted(missing)}")
    return arrays, metadata


def train_bc_policy(
    dataset: str | Path,
    *,
    checkpoint: str | Path,
    device: str = "cuda:0",
    config: BCTrainConfig | None = None,
) -> dict[str, float]:
    cfg = config or BCTrainConfig()
    cfg.validate()
    arrays, metadata = load_bc_dataset(dataset)
    observations = torch.as_tensor(arrays["observations"], dtype=torch.float32)
    targets = torch.as_tensor(arrays["hand_actions"], dtype=torch.float32)
    weights = torch.as_tensor(arrays["weights"], dtype=torch.float32)
    if targets.shape[1] != 6:
        raise ValueError(f"Expected six hand actions, got {targets.shape}.")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch_device = torch.device(device)
    policy = BCHandPolicy(
        observations.shape[1],
        hidden_sizes=cfg.hidden_sizes,
        ignored_tail_dim=cfg.ignored_tail_dim,
    ).to(torch_device)
    policy.set_observation_statistics(observations.to(torch_device))
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    count = len(observations)
    generator = torch.Generator().manual_seed(cfg.seed)
    final_loss = float("nan")
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, cfg.epochs + 1):
        permutation = torch.randperm(count, generator=generator)
        total = 0.0
        total_weight = 0.0
        policy.train()
        for start in range(0, count, cfg.batch_size):
            index = permutation[start : start + cfg.batch_size]
            obs = observations[index].to(torch_device, non_blocking=True)
            target = targets[index].to(torch_device, non_blocking=True)
            sample_weight = weights[index].to(torch_device, non_blocking=True)
            prediction = policy(obs)
            per_sample = (prediction - target).square().mean(dim=1)
            loss = (per_sample * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            total += float((per_sample.detach() * sample_weight).sum().item())
            total_weight += float(sample_weight.sum().item())
        final_loss = total / max(total_weight, 1e-8)
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        if epoch == 1 or epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs:
            print(
                f"[bc:train] epoch={epoch:04d}/{cfg.epochs:04d} loss={final_loss:.6f} "
                f"best={best_loss:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("BC training produced no checkpoint state.")
    policy.load_state_dict(best_state)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BC_SCHEMA_VERSION,
        "policy_type": "grasp_hand_bc",
        "obs_dim": policy.obs_dim,
        "hand_action_dim": policy.hand_action_dim,
        "hidden_sizes": list(policy.hidden_sizes),
        "ignored_tail_dim": policy.ignored_tail_dim,
        "state_dict": policy.state_dict(),
        "train_config": asdict(cfg),
        "dataset": str(Path(dataset)),
        "dataset_metadata": metadata,
        "metrics": {"final_loss": final_loss, "best_loss": best_loss},
    }
    torch.save(payload, checkpoint)
    _atomic_json(
        checkpoint.with_suffix(".json"),
        {
            key: value
            for key, value in payload.items()
            if key not in {"state_dict", "dataset_metadata"}
        }
        | {"dataset_metadata": metadata},
    )
    print(f"[bc:done] checkpoint={checkpoint} best_loss={best_loss:.6f}", flush=True)
    return {"final_loss": final_loss, "best_loss": best_loss}


def load_bc_policy(checkpoint: str | Path, *, device: torch.device | str) -> BCHandPolicy:
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if payload.get("schema_version") != BC_SCHEMA_VERSION or payload.get("policy_type") != "grasp_hand_bc":
        raise ValueError("Unsupported grasp BC checkpoint.")
    policy = BCHandPolicy(
        int(payload["obs_dim"]),
        hidden_sizes=tuple(int(value) for value in payload["hidden_sizes"]),
        ignored_tail_dim=int(payload.get("ignored_tail_dim", 13)),
    ).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy
