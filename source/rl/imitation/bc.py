"""Geometry-conditioned behavior cloning for Dex Hand grasp formation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from source.rl.imitation.geometry_env import (
    GEOMETRY_OBSERVATION_SCHEMA_VERSION,
    GeometryAwareResidualLiftEnv,
)
from source.rl.residual.env import ResidualLiftConfig
from source.rl.residual.reference import STAGE_CODES

BC_SCHEMA_VERSION = 5
BC_POLICY_TYPE = "grasp_hand_residual_bc_geometry"
BC_TARGET_TYPE = "expert_minus_coarse_reference_hand_fraction"
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
    """Predict six expert corrections as fractions of actuator range."""

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
    target_type: str
    actuator_names: tuple[str, ...]
    expert_manifests: tuple[str, ...]
    expert_object_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    observation_schema: dict


@dataclass(frozen=True)
class BCTrainConfig:
    epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    seed: int = 0
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    ignored_tail_dim: int = 13
    validation_fraction: float = 0.25
    coarse_reference_noise_std: float = 0.08

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("BC epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid BC optimizer parameters.")
        if any(value <= 0 for value in self.hidden_sizes):
            raise ValueError("BC hidden sizes must be positive.")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in [0, 0.5).")
        if self.coarse_reference_noise_std < 0.0:
            raise ValueError("coarse_reference_noise_std must be non-negative.")


def _stage_weight(stage: int) -> float:
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


def _normalize_hand_controls(
    controls: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    span = np.maximum(np.asarray(high) - np.asarray(low), 1e-8)
    normalized = 2.0 * (np.asarray(controls) - np.asarray(low)) / span - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _hand_residual_fraction(
    expert_controls: np.ndarray,
    coarse_controls: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Return expert-minus-coarse hand correction in actuator-range units."""

    span = np.maximum(np.asarray(high) - np.asarray(low), 1e-8)
    residual = (np.asarray(expert_controls) - np.asarray(coarse_controls)) / span
    return np.clip(residual, -1.0, 1.0).astype(np.float32)


def _split_object_groups(
    object_group_indices: np.ndarray,
    object_ids: tuple[str, ...] | list[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Split frames by unique object id, never by trajectory/manifest index."""

    groups = np.asarray(object_group_indices, dtype=np.int64)
    names = tuple(str(value) for value in object_ids)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("object_group_indices must be a non-empty 1-D array.")
    if len(set(names)) != len(names):
        raise ValueError("BC dataset object_ids must be unique group names.")
    unique_groups = np.unique(groups)
    if np.any(unique_groups < 0) or np.any(unique_groups >= len(names)):
        raise ValueError("object_group_indices reference an unknown object_ids entry.")

    shuffled = unique_groups.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    if len(shuffled) >= 4 and validation_fraction > 0.0:
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation_groups = {int(value) for value in shuffled[:validation_count]}
    else:
        validation_groups = set()
    validation_mask = np.asarray(
        [int(value) in validation_groups for value in groups],
        dtype=bool,
    )
    training_mask = ~validation_mask
    training_ids = tuple(names[index] for index in sorted(set(map(int, groups[training_mask]))))
    validation_ids = tuple(names[index] for index in sorted(validation_groups))
    return training_mask, validation_mask, training_ids, validation_ids


def collect_bc_dataset(
    expert_manifests: Iterable[str | Path],
    *,
    output: str | Path,
    device: str = "cuda:0",
    nconmax: int = 192,
    njmax: int = 768,
) -> BCDatasetInfo:
    """Replay experts while retaining independent coarse and expert controls."""
    manifests = tuple(Path(path) for path in expert_manifests)
    if not manifests:
        raise ValueError("No expert manifests were supplied for behavior cloning.")

    observations: list[np.ndarray] = []
    coarse_reference_actions: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    residual_targets: list[np.ndarray] = []
    weights: list[float] = []
    stages: list[int] = []
    object_group_indices: list[int] = []
    expert_indices: list[int] = []
    object_ids: list[str] = []
    object_group_by_id: dict[str, int] = {}
    expert_object_ids: list[str] = []
    actuator_names: tuple[str, ...] | None = None
    obs_dim: int | None = None
    observation_schema: dict | None = None

    for expert_index, manifest in enumerate(manifests):
        env = GeometryAwareResidualLiftEnv(
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
                observation_schema = env.observation_schema()
            elif obs_dim != env.obs_dim:
                raise ValueError("Expert observation dimensions are inconsistent.")

            object_id = env.reference.object_id
            expert_object_ids.append(object_id)
            if object_id not in object_group_by_id:
                object_group_by_id[object_id] = len(object_ids)
                object_ids.append(object_id)
            object_group_index = object_group_by_id[object_id]
            zero_action = torch.zeros((1, env.action_dim), device=env.torch_device)
            hand_start = env.reference.arm_action_size
            hand_low = env.ctrl_low[hand_start:].detach().cpu().numpy()
            hand_high = env.ctrl_high[hand_start:].detach().cpu().numpy()
            if env.coarse_reference.controls.shape != env.reference.controls.shape:
                raise ValueError(
                    "BC expert and coarse reference must share the same control shape."
                )
            obs = env.reset()

            for step in range(env.reference.horizon):
                stage = round(float(env.reference.stages[step]))
                expert_control = env.reference.controls[step, hand_start:]
                coarse_control = env.coarse_reference.controls[step, hand_start:]
                observations.append(obs[0].detach().cpu().numpy().astype(np.float32))
                coarse_reference_actions.append(
                    _normalize_hand_controls(coarse_control, hand_low, hand_high)
                )
                expert_actions.append(_normalize_hand_controls(expert_control, hand_low, hand_high))
                residual_targets.append(
                    _hand_residual_fraction(
                        expert_control,
                        coarse_control,
                        hand_low,
                        hand_high,
                    )
                )
                weights.append(_stage_weight(stage))
                stages.append(stage)
                object_group_indices.append(object_group_index)
                expert_indices.append(expert_index)
                obs, _, _, _ = env.step(zero_action)
        finally:
            env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"[bc:data] {expert_index + 1:03d}/{len(manifests):03d} "
            f"object={expert_object_ids[-1]} manifest={manifest}",
            flush=True,
        )

    obs_array = np.stack(observations).astype(np.float32)
    coarse_array = np.stack(coarse_reference_actions).astype(np.float32)
    expert_array = np.stack(expert_actions).astype(np.float32)
    residual_array = np.stack(residual_targets).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=obs_array,
        coarse_reference_hand_actions=coarse_array,
        expert_hand_actions=expert_array,
        hand_residual_targets=residual_array,
        weights=np.asarray(weights, dtype=np.float32),
        stages=np.asarray(stages, dtype=np.int16),
        object_group_indices=np.asarray(object_group_indices, dtype=np.int32),
        expert_indices=np.asarray(expert_indices, dtype=np.int32),
    )
    info = BCDatasetInfo(
        observations=len(obs_array),
        experts=len(manifests),
        objects=len(set(object_ids)),
        obs_dim=int(obs_array.shape[1]),
        hand_action_dim=int(residual_array.shape[1]),
        target_type=BC_TARGET_TYPE,
        actuator_names=tuple(actuator_names or ()),
        expert_manifests=tuple(str(path) for path in manifests),
        expert_object_ids=tuple(expert_object_ids),
        object_ids=tuple(object_ids),
        observation_schema=dict(observation_schema or {}),
    )
    _atomic_json(
        output.with_suffix(".json"),
        {
            "schema_version": BC_SCHEMA_VERSION,
            "geometry_observation_schema_version": GEOMETRY_OBSERVATION_SCHEMA_VERSION,
            "target_type": BC_TARGET_TYPE,
            "dataset": asdict(info),
            "stage_weights": {name: _stage_weight(code) for name, code in STAGE_CODES.items()},
        },
    )
    return info


def load_bc_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != BC_SCHEMA_VERSION:
        raise ValueError(f"Unsupported BC dataset schema: {metadata.get('schema_version')}.")
    if metadata.get("target_type") != BC_TARGET_TYPE:
        raise ValueError(f"Unsupported BC target type: {metadata.get('target_type')!r}.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "observations",
        "coarse_reference_hand_actions",
        "expert_hand_actions",
        "hand_residual_targets",
        "weights",
        "stages",
        "object_group_indices",
        "expert_indices",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"BC dataset is missing arrays: {sorted(missing)}")
    expected = 0.5 * (arrays["expert_hand_actions"] - arrays["coarse_reference_hand_actions"])
    if not np.allclose(arrays["hand_residual_targets"], expected, atol=2e-5):
        raise ValueError("BC residual targets do not equal expert minus coarse reference controls.")
    return arrays, metadata


def _weighted_loss(policy, obs, target, weight) -> torch.Tensor:
    prediction = policy(obs)
    per_sample = (prediction - target).square().mean(dim=1)
    return (per_sample * weight).sum() / torch.clamp(weight.sum(), min=1e-6)


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
    targets = torch.as_tensor(arrays["hand_residual_targets"], dtype=torch.float32)
    weights = torch.as_tensor(arrays["weights"], dtype=torch.float32)
    object_group_indices = np.asarray(arrays["object_group_indices"], dtype=np.int64)
    if targets.shape[1] != 6:
        raise ValueError(f"Expected six hand actions, got {targets.shape}.")

    schema = metadata.get("dataset", {}).get("observation_schema", {})
    feature_slices = schema.get("feature_slices", {})
    reference_slice = feature_slices.get("coarse_reference_hand")
    if not (
        isinstance(reference_slice, list)
        and len(reference_slice) == 2
        and 0 <= int(reference_slice[0]) < int(reference_slice[1]) <= observations.shape[1]
    ):
        raise ValueError("BC dataset lacks a valid coarse_reference_hand observation slice.")
    ref_lo, ref_hi = map(int, reference_slice)

    dataset_info = metadata.get("dataset", {})
    object_ids = tuple(str(value) for value in dataset_info.get("object_ids", ()))
    train_mask_np, val_mask_np, training_object_ids, validation_object_ids = _split_object_groups(
        object_group_indices,
        object_ids,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
    )
    train_indices = torch.as_tensor(np.flatnonzero(train_mask_np), dtype=torch.long)
    val_indices = torch.as_tensor(np.flatnonzero(val_mask_np), dtype=torch.long)
    if not len(train_indices):
        raise RuntimeError("BC object-level split produced an empty training set.")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch_device = torch.device(device)
    policy = BCHandPolicy(
        observations.shape[1],
        hidden_sizes=cfg.hidden_sizes,
        ignored_tail_dim=cfg.ignored_tail_dim,
    ).to(torch_device)
    policy.set_observation_statistics(observations[train_indices].to(torch_device))
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    final_train = float("nan")
    final_val = float("nan")
    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, cfg.epochs + 1):
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        total = 0.0
        total_weight = 0.0
        policy.train()
        for start in range(0, len(permutation), cfg.batch_size):
            index = permutation[start : start + cfg.batch_size]
            obs = observations[index].to(torch_device, non_blocking=True).clone()
            if cfg.coarse_reference_noise_std > 0.0:
                noise = torch.randn_like(obs[:, ref_lo:ref_hi]) * cfg.coarse_reference_noise_std
                obs[:, ref_lo:ref_hi] = torch.clamp(obs[:, ref_lo:ref_hi] + noise, -1.0, 1.0)
            target = targets[index].to(torch_device, non_blocking=True)
            sample_weight = weights[index].to(torch_device, non_blocking=True)
            loss = _weighted_loss(policy, obs, target, sample_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().item()) * float(sample_weight.sum().item())
            total_weight += float(sample_weight.sum().item())
        final_train = total / max(total_weight, 1e-8)

        policy.eval()
        if len(val_indices):
            with torch.no_grad():
                final_val = float(
                    _weighted_loss(
                        policy,
                        observations[val_indices].to(torch_device),
                        targets[val_indices].to(torch_device),
                        weights[val_indices].to(torch_device),
                    ).item()
                )
            metric = final_val
        else:
            final_val = final_train
            metric = final_train
        if metric < best_metric:
            best_metric = metric
            best_state = {
                name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
            }
        if epoch == 1 or epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs:
            print(
                f"[bc:train] epoch={epoch:04d}/{cfg.epochs:04d} "
                f"train={final_train:.6f} val={final_val:.6f} best={best_metric:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("BC training produced no checkpoint state.")
    policy.load_state_dict(best_state)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BC_SCHEMA_VERSION,
        "policy_type": BC_POLICY_TYPE,
        "target_type": BC_TARGET_TYPE,
        "obs_dim": policy.obs_dim,
        "hand_action_dim": policy.hand_action_dim,
        "hidden_sizes": list(policy.hidden_sizes),
        "ignored_tail_dim": policy.ignored_tail_dim,
        "state_dict": policy.state_dict(),
        "train_config": asdict(cfg),
        "dataset": str(Path(dataset)),
        "dataset_metadata": metadata,
        "training_object_ids": list(training_object_ids),
        "validation_object_ids": list(validation_object_ids),
        "metrics": {
            "final_train_loss": final_train,
            "final_validation_loss": final_val,
            "best_validation_loss": best_metric,
        },
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
    print(
        f"[bc:done] checkpoint={checkpoint} best_val={best_metric:.6f} "
        f"validation_objects={list(validation_object_ids)}",
        flush=True,
    )
    return {
        "final_train_loss": final_train,
        "final_validation_loss": final_val,
        "best_validation_loss": best_metric,
    }


def load_bc_policy(checkpoint: str | Path, *, device: torch.device | str) -> BCHandPolicy:
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if (
        payload.get("schema_version") != BC_SCHEMA_VERSION
        or payload.get("policy_type") != BC_POLICY_TYPE
        or payload.get("target_type") != BC_TARGET_TYPE
    ):
        raise ValueError("Unsupported grasp BC checkpoint; rebuild it with the current pipeline.")
    policy = BCHandPolicy(
        int(payload["obs_dim"]),
        hidden_sizes=tuple(int(value) for value in payload["hidden_sizes"]),
        ignored_tail_dim=int(payload.get("ignored_tail_dim", 13)),
    ).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy
