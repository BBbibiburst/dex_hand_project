"""Train a fused vision-tactile diffusion policy on a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from source.imitation.dataset import LeRobotDiffusionDataset
from source.imitation.diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("datasets/lerobot"))
    p.add_argument("--repo-id", default="local/dex-hand-demonstrations")
    p.add_argument("--output", type=Path, default=Path("checkpoints/diffusion_policy.pt"))
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--noise-schedule", choices=("linear", "cosine"), default="linear")
    p.add_argument(
        "--clip-sample",
        type=float,
        default=5.0,
        help="Clamp reconstructed normalized x0 during sampling; set <=0 to disable.",
    )
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth observation as a training sample; action horizon remains at dataset FPS.",
    )
    p.add_argument("--no-tactile", action="store_true")
    p.add_argument(
        "--overfit-episode",
        type=int,
        help=(
            "Diagnostic mode: train and report loss on one named episode. "
            "This intentionally has no independent validation set and must not be used "
            "as generalization evidence."
        ),
    )
    p.add_argument("--summary", type=Path)
    p.add_argument("--loss-plot", type=Path)
    return p.parse_args()


def moments(dataset, indices):
    states, tactile, actions = [], [], []
    for index in indices:
        state, tactile_value, action = dataset.statistics_item(index)
        states.append(state)
        actions.append(action.reshape(-1, action.shape[-1]))
        if tactile_value.numel():
            tactile.append(tactile_value)
    state = torch.stack(states)
    action = torch.cat(actions)
    tact = torch.stack(tactile) if tactile else torch.zeros(0)
    tactile_mean = tact.mean(0) if tactile else tact
    tactile_std = tact.std(0, unbiased=False) if tactile else tact
    return (
        state.mean(0),
        state.std(0, unbiased=False),
        tactile_mean,
        tactile_std,
        action.mean(0),
        action.std(0, unbiased=False),
    )


def split_episode_indices(dataset, *, val_fraction: float, seed: int):
    """Split complete episodes, never neighboring frames from one episode."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must lie in (0, 1).")
    by_episode = dataset.episode_indices()
    episode_ids = sorted(by_episode)
    if len(episode_ids) < 2:
        raise ValueError("Dataset needs at least two episodes for an episode-level split.")
    random.Random(seed).shuffle(episode_ids)
    val_count = min(len(episode_ids) - 1, max(1, round(len(episode_ids) * val_fraction)))
    val_episodes = tuple(sorted(episode_ids[:val_count]))
    train_episodes = tuple(sorted(episode_ids[val_count:]))
    train_ids = [index for episode in train_episodes for index in by_episode[episode]]
    val_ids = [index for episode in val_episodes for index in by_episode[episode]]
    return train_ids, val_ids, train_episodes, val_episodes


def main():
    args = args_parser()
    if (
        args.num_workers < 0
        or args.frame_stride <= 0
        or args.diffusion_steps <= 0
        or args.feature_dim <= 0
    ):
        raise ValueError("--num-workers must be non-negative and --frame-stride positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = LeRobotDiffusionDataset(
        repo_id=args.repo_id,
        root=args.dataset,
        horizon=args.horizon,
        use_tactile=not args.no_tactile,
    )
    if args.overfit_episode is None:
        train_ids, val_ids, train_episodes, val_episodes = split_episode_indices(
            dataset, val_fraction=args.val_fraction, seed=args.seed
        )
    else:
        by_episode = dataset.episode_indices()
        if args.overfit_episode not in by_episode:
            raise ValueError(
                f"Episode {args.overfit_episode} is absent; available={sorted(by_episode)}."
            )
        train_ids = list(by_episode[args.overfit_episode])
        # Deliberate train-set reconstruction diagnostic, not validation.
        val_ids = list(train_ids)
        train_episodes = val_episodes = (args.overfit_episode,)
    train_ids = train_ids[:: args.frame_stride]
    val_ids = val_ids[:: args.frame_stride]
    sample = dataset[train_ids[0]]
    config = DiffusionPolicyConfig(
        sample["state"].numel(),
        sample["tactile"].numel(),
        sample["action"].shape[-1],
        args.horizon,
        diffusion_steps=args.diffusion_steps,
        feature_dim=args.feature_dim,
        noise_schedule=args.noise_schedule,
        clip_sample=args.clip_sample if args.clip_sample > 0 else None,
    )
    model = DiffusionPolicy(config).to(args.device)
    stats = moments(dataset, train_ids)
    model.set_normalization(*(value.to(args.device) for value in stats))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(Subset(dataset, train_ids), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_ids), **loader_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    best = float("inf")
    history = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model.loss(**batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch["state"].shape[0]
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                val_loss += model.loss(**batch).item() * batch["state"].shape[0]
        train_loss /= len(train_ids)
        val_loss /= len(val_ids)
        print(f"epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            payload = model.checkpoint()
            payload.update(
                {
                    "epoch": epoch,
                    "val_loss": best,
                    "repo_id": args.repo_id,
                    "state_schema": "actuator_position_velocity_command_v1",
                    "action_schema": "arm7_hand6_position_command_v1",
                    "use_tactile": not args.no_tactile,
                    "train_episodes": list(train_episodes),
                    "val_episodes": list(val_episodes),
                }
            )
            torch.save(payload, args.output)
    print(f"best checkpoint: {args.output} val_loss={best:.6f}")
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    loss_plot = args.loss_plot or summary_path.with_name("loss_curve.png")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(
        [row["epoch"] for row in history], [row["train_loss"] for row in history], label="train"
    )
    axis.plot(
        [row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation"
    )
    axis.set(xlabel="Epoch", ylabel="DDPM noise-prediction MSE", title="Diffusion Policy training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    loss_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(loss_plot, dpi=180)
    plt.close(figure)

    summary = {
        "command_config": {
            **vars(args),
            "dataset": str(args.dataset),
            "output": str(args.output),
            "summary": None if args.summary is None else str(args.summary),
        },
        "model_config": asdict(config),
        "device": str(args.device),
        "cuda_device_name": torch.cuda.get_device_name(args.device)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else None,
        "seed": args.seed,
        "use_tactile": not args.no_tactile,
        "diagnostic_overfit_episode": args.overfit_episode,
        "validation_is_training_data": args.overfit_episode is not None,
        "train_episodes": list(train_episodes),
        "val_episodes": list(val_episodes),
        "train_frames": len(train_ids),
        "val_frames": len(val_ids),
        "best_val_loss": best,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "loss_plot": str(loss_plot),
        "history": history,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
