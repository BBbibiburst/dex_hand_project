"""Bounded single-trajectory training with the official LeRobot ACT policy."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from apps.train_diffusion import moments
from source.imitation.dataset import LeRobotDiffusionDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--loss-plot", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2011)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dim-model", type=int, default=128)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--vae-layers", type=int, default=2)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    return parser


def _config(args, sample):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (sample["state"].numel(),)),
            "observation.images.agentview": PolicyFeature(
                FeatureType.VISUAL, tuple(sample["image"].shape)
            ),
        },
        output_features={
            "action": PolicyFeature(FeatureType.ACTION, (sample["action"].shape[-1],))
        },
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=args.action_steps,
        pretrained_backbone_weights=None,
        dim_model=args.dim_model,
        n_heads=4,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.encoder_layers,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=args.vae_layers,
        kl_weight=args.kl_weight,
        dropout=0.0,
    )


def _normalized_batch(batch, stats, device):
    state_mean, state_std, action_mean, action_std = stats
    state = batch["state"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)
    image = batch["image"].to(device, non_blocking=True)
    return {
        "observation.state": (state - state_mean) / state_std,
        "observation.images.agentview": image,
        "action": (action - action_mean) / action_std,
        "action_is_pad": torch.zeros(action.shape[:2], dtype=torch.bool, device=device),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = LeRobotDiffusionDataset(
        repo_id=args.repo_id, root=args.dataset, horizon=args.chunk_size, use_tactile=False
    )
    indices = dataset.episode_indices().get(args.episode)
    if not indices:
        raise ValueError(f"Episode {args.episode} is absent.")
    sample = dataset[indices[0]]
    config = _config(args, sample)
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy(config).to(args.device)
    state_mean, state_std, _, _, action_mean, action_std = moments(dataset, indices)
    stats = tuple(
        value.to(args.device)
        for value in (
            state_mean,
            state_std.clamp_min(1e-6),
            action_mean,
            action_std.clamp_min(1e-6),
        )
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    best = float("inf")
    history = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total = l1_total = kl_total = 0.0
        for raw in loader:
            batch = _normalized_batch(raw, stats, args.device)
            loss, details = policy(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            count = raw["state"].shape[0]
            total += loss.item() * count
            l1_total += details["l1_loss"] * count
            kl_total += details.get("kld_loss", 0.0) * count
        row = {
            "epoch": epoch,
            "loss": total / len(indices),
            "l1_loss": l1_total / len(indices),
            "kl_loss": kl_total / len(indices),
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} loss={row['loss']:.6f} l1={row['l1_loss']:.6f} kl={row['kl_loss']:.6f}",
                flush=True,
            )
        if row["loss"] < best:
            best = row["loss"]
            torch.save(
                {
                    "policy_type": "lerobot_act_0.4.4",
                    "config": config,
                    "model": policy.state_dict(),
                    "state_mean": stats[0].cpu(),
                    "state_std": stats[1].cpu(),
                    "action_mean": stats[2].cpu(),
                    "action_std": stats[3].cpu(),
                    "epoch": epoch,
                    "train_loss": best,
                    "repo_id": args.repo_id,
                    "state_schema": "actuator_position_velocity_command_v1",
                    "action_schema": "arm7_hand6_position_command_v1",
                },
                args.output,
            )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot([x["epoch"] for x in history], [x["l1_loss"] for x in history], label="L1")
    axis.plot([x["epoch"] for x in history], [x["kl_loss"] for x in history], label="KL")
    axis.set(xlabel="Epoch", ylabel="Loss", title="LeRobot ACT single-trajectory training")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.loss_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.loss_plot, dpi=180)
    plt.close(figure)
    summary = {
        "command_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "lerobot_version": "0.4.4",
        "official_policy": "lerobot.policies.act.ACTPolicy",
        "frames": len(indices),
        "best_train_loss": best,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "loss_plot": str(args.loss_plot),
        "history": history,
        "scope": "single-trajectory memorization diagnostic; not generalization evidence",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
