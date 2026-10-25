"""Train a bounded Lift-only Cartesian controller with root-level splits."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from source.imitation.lift_temporal import LiftTemporalConfig, LiftTemporalController


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("single_mlp", "history_mlp", "gru"), required=True)
    parser.add_argument("--no-action-history", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--single-root", type=int)
    return parser


def _load(rows):
    merged = {}
    for row in rows:
        with np.load(row["path"]) as archive:
            for key in ("state_history", "action_history", "target_xyz"):
                merged.setdefault(key, []).append(np.asarray(archive[key]))
    return {key: torch.from_numpy(np.concatenate(value)) for key, value in merged.items()}


def main(argv=None):
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata = json.loads((args.dataset / "manifest.json").read_text())
    train_rows = [row for row in metadata["rows"] if row["split"] == "train"]
    if args.single_root is None:
        validation_rows = train_rows[-1:]
        fit_rows = train_rows[:-1]
    else:
        fit_rows = [row for row in train_rows if row["root_id"] == args.single_root]
        if not fit_rows:
            raise ValueError(f"Training root {args.single_root} is unavailable.")
        # Intentional memorization gate: validation is the same root and is
        # never interpreted as generalization performance.
        validation_rows = fit_rows
    fit, validation = _load(fit_rows), _load(validation_rows)
    config = LiftTemporalConfig(
        state_dim=fit["state_history"].shape[-1],
        history=fit["state_history"].shape[1],
        horizon=fit["target_xyz"].shape[1],
        kind=args.kind,
        include_action_history=not args.no_action_history,
    )
    model = LiftTemporalController(config).to(args.device)
    inputs = torch.cat([fit["state_history"], fit["action_history"]], -1) if config.include_action_history else fit["state_history"]
    model.set_normalization(inputs.to(args.device), fit["target_xyz"].to(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(args.seed)
    best = float("inf")
    history = []
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(fit["target_xyz"]), generator=generator)
        total = 0.0
        for start in range(0, len(permutation), args.batch_size):
            index = permutation[start : start + args.batch_size]
            state = fit["state_history"][index].to(args.device)
            action = fit["action_history"][index].to(args.device)
            target = fit["target_xyz"][index].to(args.device)
            loss = model.loss(state, action, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        with torch.inference_mode():
            val = float(model.loss(validation["state_history"].to(args.device), validation["action_history"].to(args.device), validation["target_xyz"].to(args.device)))
        train = total / len(permutation)
        history.append({"epoch": epoch, "train_loss": train, "validation_loss": val})
        if val < best:
            best = val
            checkpoint = model.checkpoint()
            checkpoint.update({"epoch": epoch, "validation_loss": val, "fit_roots": [row["root_id"] for row in fit_rows], "validation_roots": [row["root_id"] for row in validation_rows]})
            torch.save(checkpoint, args.output / "checkpoint.pt")
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch:04d} train={train:.7f} val={val:.7f}", flush=True)
    summary = {"config": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)}, "model_config": config.__dict__, "fit_roots": [row["root_id"] for row in fit_rows], "validation_roots": [row["root_id"] for row in validation_rows], "held_out_test_roots": [row["root_id"] for row in metadata["rows"] if row["split"] == "test"], "memorization_diagnostic": args.single_root is not None, "best_validation_loss": best, "runtime_seconds": time.perf_counter() - started, "history": history}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
