"""Train a fused vision-tactile diffusion policy on a LeRobot dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
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
    p.add_argument("--horizon", type=int, default=16); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def moments(dataset, indices):
    states, tactile, actions = [], [], []
    for index in indices:
        sample = dataset[index]; states.append(sample["state"]); actions.append(sample["action"].reshape(-1, sample["action"].shape[-1]))
        if sample["tactile"].numel(): tactile.append(sample["tactile"])
    state = torch.stack(states); action = torch.cat(actions)
    tact = torch.stack(tactile) if tactile else torch.zeros(len(states), 0)
    return state.mean(0), state.std(0), tact.mean(0), tact.std(0), action.mean(0), action.std(0)


def main():
    args = args_parser(); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dataset = LeRobotDiffusionDataset(repo_id=args.repo_id, root=args.dataset, horizon=args.horizon)
    indices = list(range(len(dataset))); random.shuffle(indices)
    val_count = max(1, int(len(indices)*args.val_fraction)); val_ids, train_ids = indices[:val_count], indices[val_count:]
    if not train_ids: raise ValueError("Dataset needs at least two frames for train/validation split.")
    sample = dataset[train_ids[0]]
    config = DiffusionPolicyConfig(sample["state"].numel(), sample["tactile"].numel(), sample["action"].shape[-1], args.horizon)
    model = DiffusionPolicy(config).to(args.device)
    stats = moments(dataset, train_ids); model.set_normalization(*(value.to(args.device) for value in stats))
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_ids), batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    best = float("inf"); args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}; loss = model.loss(**batch)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item()*batch["state"].shape[0]
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}; val_loss += model.loss(**batch).item()*batch["state"].shape[0]
        train_loss /= len(train_ids); val_loss /= len(val_ids)
        print(f"epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best:
            best = val_loss; payload = model.checkpoint(); payload.update({"epoch": epoch, "val_loss": best, "repo_id": args.repo_id})
            torch.save(payload, args.output)
    print(f"best checkpoint: {args.output} val_loss={best:.6f}")


if __name__ == "__main__": main()
