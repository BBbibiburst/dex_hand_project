"""Train single-episode BC with a shared-anchor XYZ9 or Pose12 action contract."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from apps.trace_diffusion_rollout import _make_env, _reset_exact
from source.imitation.action_chunk_bc import ActionChunkBC, ActionChunkBCConfig
from source.imitation.cartesian_action import POSE12, XYZ9, CartesianChunkDataset
from source.imitation.dataset import LeRobotDiffusionDataset
from source.verification.strict_replay import load_replay_controls


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=(XYZ9, POSE12), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2011)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--split-file",
        type=Path,
        help="Optional root-level split JSON; LeRobot episode indices must match root IDs.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    payload = json.loads(args.manifest.read_text())
    env = _make_env(payload, 400)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    _, qpos, qvel, ctrl = load_replay_controls(args.manifest, env)
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    _reset_exact(env, qpos, qvel, ctrl)
    arm = env.controller.arm_controller
    from source.geometry import mat_to_quat

    fixed_quat = mat_to_quat(env.data.site_xmat[arm.site_id]).astype(np.float32)
    base = LeRobotDiffusionDataset(
        repo_id=args.repo_id, root=args.dataset, horizon=args.horizon, use_tactile=False
    )
    started = time.perf_counter()
    dataset = CartesianChunkDataset(
        base, env, args.mode, fixed_orientation_wxyz=fixed_quat if args.mode == XYZ9 else None
    )
    env.close()
    sample = dataset[0]
    config = ActionChunkBCConfig(
        state_dim=sample["state"].numel(),
        action_dim=sample["action"].shape[-1],
        horizon=args.horizon,
        feature_dim=args.feature_dim,
    )
    model = ActionChunkBC(config).to(args.device)
    episode_rows = dataset.episode_indices()
    if args.split_file is None:
        train_indices = list(range(len(dataset)))
        validation_indices: list[int] = []
        split_roots = {"train": sorted(episode_rows), "validation": [], "test": []}
    else:
        split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))
        split_roots = {
            name: [int(value) for value in split_payload["roots"].get(name, [])]
            for name in ("train", "validation", "test")
        }
        flattened = [root for values in split_roots.values() for root in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Root-level split contains leakage between partitions.")
        missing = sorted(set(flattened) - set(episode_rows))
        if missing:
            raise ValueError(f"Split references missing LeRobot episodes: {missing}")
        train_indices = [index for root in split_roots["train"] for index in episode_rows[root]]
        validation_indices = [
            index for root in split_roots["validation"] for index in episode_rows[root]
        ]
        if not train_indices or not validation_indices:
            raise ValueError("Split training requires non-empty train and validation roots.")
    states = torch.stack([dataset.base.statistics_item(i)[0] for i in train_indices])
    actions = dataset.labels[train_indices].reshape(-1, config.action_dim)
    model.set_normalization(
        states.mean(0).to(args.device),
        states.std(0).clamp_min(1e-6).to(args.device),
        actions.mean(0).to(args.device),
        actions.std(0).clamp_min(1e-6).to(args.device),
    )
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=True,
    )
    validation_loader = (
        DataLoader(
            Subset(dataset, validation_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if validation_indices
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    history, best = [], float("inf")
    args.output.mkdir(parents=True, exist_ok=True)
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
            total += loss.item() * len(state)
        mean = total / len(train_indices)
        model.eval()
        validation = 0.0
        if validation_loader is not None:
            with torch.no_grad():
                for batch in validation_loader:
                    image = batch["image"].to(args.device, non_blocking=True)
                    state = batch["state"].to(args.device, non_blocking=True)
                    action = batch["action"].to(args.device, non_blocking=True)
                    validation += model.loss(image, state, action).item() * len(state)
            validation /= len(validation_indices)
        else:
            validation = mean
        history.append({"epoch": epoch, "train_loss": mean, "validation_loss": validation})
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} train={mean:.8f} validation={validation:.8f}",
                flush=True,
            )
        if validation < best:
            best = validation
            checkpoint = model.checkpoint()
            checkpoint.update(
                {
                    "action_schema": f"{args.mode}_shared_anchor_v1",
                    "fixed_orientation_wxyz": fixed_quat.tolist() if args.mode == XYZ9 else None,
                    "source_manifest": str(args.manifest),
                    "epoch": epoch,
                    "train_loss": mean,
                    "validation_loss": best,
                    "split_roots": split_roots,
                }
            )
            torch.save(checkpoint, args.output / "checkpoint.pt")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot([x["epoch"] for x in history], [x["train_loss"] for x in history], label="train")
    ax.plot(
        [x["epoch"] for x in history],
        [x["validation_loss"] for x in history],
        label="validation",
    )
    ax.set(xlabel="Epoch", ylabel="Normalized action MSE", title=f"{args.mode.upper()} BC")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "loss_curve.png", dpi=180)
    plt.close(fig)
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_config": asdict(config),
        "frames": len(dataset),
        "train_frames": len(train_indices),
        "validation_frames": len(validation_indices),
        "held_out_test_frames": sum(len(episode_rows[root]) for root in split_roots["test"]),
        "split_roots": split_roots,
        "best_validation_loss": best,
        "runtime_seconds": time.perf_counter() - started,
        "history": history,
        "scope": (
            "root-split state-conditioned Sugar Box dataset"
            if args.split_file is not None
            else "single/all-episode diagnostic"
        ),
    }
    (args.output / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
