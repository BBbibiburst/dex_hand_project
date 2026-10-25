"""Train deterministic RGB+state action-chunk behavior cloning."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from apps.train_diffusion import moments
from source.imitation.action_chunk_bc import ActionChunkBC, ActionChunkBCConfig
from source.imitation.dataset import LeRobotDiffusionDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--loss-plot", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2011)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--include-all-episodes", action="store_true")
    parser.add_argument(
        "--max-extra-frames-per-episode",
        type=int,
        default=0,
        help="Uniformly subsample non-primary episodes; 0 keeps every frame.",
    )
    parser.add_argument(
        "--primary-sampling-weight",
        type=float,
        default=0.5,
        help="Sampling probability assigned to the complete primary episode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.horizon <= 0 or args.frame_stride <= 0:
        raise ValueError("epochs, horizon, and frame-stride must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = LeRobotDiffusionDataset(
        repo_id=args.repo_id, root=args.dataset, horizon=args.horizon, use_tactile=False
    )
    by_episode = dataset.episode_indices()
    if args.episode not in by_episode:
        raise ValueError(f"Episode {args.episode} absent; available={sorted(by_episode)}")
    primary_indices = by_episode[args.episode][:: args.frame_stride]
    indices = list(primary_indices)
    extra_indices: list[int] = []
    if args.include_all_episodes:
        for episode, episode_indices in sorted(by_episode.items()):
            if episode == args.episode:
                continue
            selected = episode_indices[:: args.frame_stride]
            if (
                args.max_extra_frames_per_episode
                and len(selected) > args.max_extra_frames_per_episode
            ):
                positions = np.linspace(
                    0, len(selected) - 1, args.max_extra_frames_per_episode, dtype=np.int64
                )
                selected = [selected[int(position)] for position in positions]
            extra_indices.extend(selected)
        indices.extend(extra_indices)
    sample = dataset[indices[0]]
    config = ActionChunkBCConfig(
        state_dim=sample["state"].numel(),
        action_dim=sample["action"].shape[-1],
        horizon=args.horizon,
        feature_dim=args.feature_dim,
    )
    model = ActionChunkBC(config).to(args.device)
    state_mean, state_std, _, _, action_mean, action_std = moments(dataset, indices)
    model.set_normalization(
        state_mean.to(args.device),
        state_std.to(args.device),
        action_mean.to(args.device),
        action_std.to(args.device),
    )
    sampler = None
    shuffle = True
    if extra_indices:
        if not 0.0 < args.primary_sampling_weight < 1.0:
            raise ValueError("--primary-sampling-weight must lie in (0, 1).")
        weights = [args.primary_sampling_weight / len(primary_indices)] * len(primary_indices)
        weights += [(1.0 - args.primary_sampling_weight) / len(extra_indices)] * len(extra_indices)
        sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)
        shuffle = False
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    best = float("inf")
    history = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            image = batch["image"].to(args.device, non_blocking=True)
            state = batch["state"].to(args.device, non_blocking=True)
            action = batch["action"].to(args.device, non_blocking=True)
            loss = model.loss(image, state, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * state.shape[0]
        mean_loss = total / len(indices)
        history.append({"epoch": epoch, "train_loss": mean_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train={mean_loss:.8f}", flush=True)
        if mean_loss < best:
            best = mean_loss
            payload = model.checkpoint()
            payload.update(
                {
                    "epoch": epoch,
                    "train_loss": best,
                    "repo_id": args.repo_id,
                    "state_schema": "actuator_position_velocity_command_v1",
                    "action_schema": "arm7_hand6_position_command_v1",
                    "train_episodes": sorted(by_episode)
                    if args.include_all_episodes
                    else [args.episode],
                }
            )
            torch.save(payload, args.output)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history])
    axis.set(xlabel="Epoch", ylabel="Normalized action MSE", title="Deterministic action-chunk BC")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    args.loss_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.loss_plot, dpi=180)
    plt.close(figure)

    summary = {
        "command_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": asdict(config),
        "seed": args.seed,
        "dataset": str(args.dataset),
        "episode": args.episode,
        "frames": len(indices),
        "primary_frames": len(primary_indices),
        "recovery_frames": len(extra_indices),
        "best_train_loss": best,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "loss_plot": str(args.loss_plot),
        "history": history,
        "scope": "single-trajectory memorization diagnostic; not generalization evidence",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
