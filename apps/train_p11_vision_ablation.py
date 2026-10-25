"""Fairly train global-only and global-plus-ROI XYZ9 behavior cloning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from source.imitation.action_chunk_bc import ActionChunkBC, ActionChunkBCConfig
from source.imitation.dual_vision_bc import DualVisionActionChunkBC, DualVisionBCConfig
from source.imitation.dual_vision_dataset import DualVisionRootDataset


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2011)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(model, batch, dual):
    if dual:
        return model.loss(batch["global_rgb"], batch["local_rgb"], batch["state"], batch["action"])
    return model.loss(batch["global_rgb"], batch["state"], batch["action"])


def _train_one(args, name, train, validation, state_stats, action_stats, dual):
    _seed(args.seed)
    sample = train[0]
    if dual:
        config = DualVisionBCConfig(
            state_dim=sample["state"].numel(),
            action_dim=sample["action"].shape[-1],
            horizon=sample["action"].shape[0],
        )
        model = DualVisionActionChunkBC(config).to(args.device)
    else:
        config = ActionChunkBCConfig(
            state_dim=sample["state"].numel(),
            action_dim=sample["action"].shape[-1],
            horizon=sample["action"].shape[0],
            feature_dim=256,
        )
        model = ActionChunkBC(config).to(args.device)
    model.set_normalization(
        state_stats[0].to(args.device),
        state_stats[1].clamp_min(1e-6).to(args.device),
        action_stats[0].to(args.device),
        action_stats[1].clamp_min(1e-6).to(args.device),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, num_workers=0, pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    directory = args.output / name
    directory.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
            loss = _loss(model, batch, dual)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch["state"])
        train_loss = total / len(train)
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for batch in validation_loader:
                batch = {
                    key: value.to(args.device, non_blocking=True) for key, value in batch.items()
                }
                total += float(_loss(model, batch, dual).item()) * len(batch["state"])
        validation_loss = total / len(validation)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        print(
            f"{name} epoch={epoch:03d} train={train_loss:.6f} val={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best:
            best = validation_loss
            best_epoch = epoch
            checkpoint = model.checkpoint()
            checkpoint.update(
                {
                    "action_schema": "xyz9_shared_anchor_v1",
                    "orientation_mode": "freeze measured orientation at rollout reset",
                    "dataset": str(args.dataset),
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_loss": best,
                }
            )
            torch.save(checkpoint, directory / "checkpoint.pt")
        if epoch - best_epoch >= args.patience:
            break
    runtime = time.perf_counter() - started
    checkpoint_size = (directory / "checkpoint.pt").stat().st_size
    summary = {
        "name": name,
        "config": asdict(config),
        "seed": args.seed,
        "train_roots": 7,
        "validation_roots": 1,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "checkpoint_bytes": checkpoint_size,
        "best_epoch": best_epoch,
        "best_validation_loss": best,
        "runtime_seconds": runtime,
        "history": history,
    }
    (directory / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    train = DualVisionRootDataset(args.dataset, "train")
    validation = DualVisionRootDataset(args.dataset, "validation")
    state_mean, state_std, action_mean, action_std = train.moments()
    summaries = []
    for name, dual in (("global", False), ("global_roi", True)):
        summaries.append(
            _train_one(
                args,
                name,
                train,
                validation,
                (state_mean, state_std),
                (action_mean, action_std),
                dual,
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, summary in zip(axes, summaries, strict=True):
        history = summary["history"]
        axis.plot(
            [row["epoch"] for row in history], [row["train_loss"] for row in history], label="train"
        )
        axis.plot(
            [row["epoch"] for row in history],
            [row["validation_loss"] for row in history],
            label="validation",
        )
        axis.set(title=summary["name"], xlabel="epoch", ylabel="normalized action MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "loss_curves.png", dpi=180)
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
