"""Train state-only or state+tactile XYZ residuals on validated recoveries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from apps.analyze_tactile_failure_warning import _groups
from source.imitation.cartesian_residual import (
    CartesianResidualConfig,
    CartesianResidualMLP,
    tactile_region_features,
)
from source.sensors.tactile.dex_hand import SimpleBoxTactileSensor


def _chunks(path, use_tactile, groups):
    data = np.load(path)
    count = len(data["state"]) // 4
    indices = np.arange(count) * 4
    state = data["state"][indices]
    if use_tactile:
        tactile = np.asarray(
            [tactile_region_features(row, groups) for row in data["tactile"][indices]]
        )
        state = np.concatenate([state, tactile], axis=1)
    target = data["residual_xyz"][: count * 4].reshape(count, 4, 3)
    return state.astype(np.float32), target.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2011)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    groups = _groups(SimpleBoxTactileSensor())
    files = sorted(args.recovery_root.glob("*/expert_xyz_bc_hand/validated_recovery.npz"))
    train_files = [path for path in files if "lift_start" not in str(path)]
    test_files = [path for path in files if "lift_start" in str(path)]
    train_parts = [_chunks(path, args.use_tactile, groups) for path in train_files]
    test_parts = [_chunks(path, args.use_tactile, groups) for path in test_files]
    state = np.concatenate([part[0] for part in train_parts], axis=0)
    target = np.concatenate([part[1] for part in train_parts], axis=0)
    inputs = torch.from_numpy(state).to(args.device)
    targets = torch.from_numpy(target).to(args.device)
    config = CartesianResidualConfig(input_dim=state.shape[1], use_tactile=args.use_tactile)
    model = CartesianResidualMLP(config).to(args.device)
    model.set_normalization(inputs, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        prediction = model(inputs)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        history.append(value)
        if value < best:
            best = value
            payload = model.checkpoint()
            payload.update(
                {
                    "train_files": [str(x) for x in train_files],
                    "test_files": [str(x) for x in test_files],
                }
            )
            torch.save(payload, args.output / "checkpoint.pt")
    test_mse = None
    if test_parts:
        x, y = test_parts[0]
        with torch.inference_mode():
            test_mse = float(
                torch.nn.functional.mse_loss(
                    model(torch.from_numpy(x).to(args.device)), torch.from_numpy(y).to(args.device)
                ).item()
            )
    summary = {
        "use_tactile": args.use_tactile,
        "train_recovery_states": [str(x) for x in train_files],
        "heldout_recovery_states": [str(x) for x in test_files],
        "train_chunks": len(state),
        "heldout_chunks": sum(len(x[0]) for x in test_parts),
        "best_train_mse": best,
        "heldout_mse": test_mse,
        "runtime_seconds": time.perf_counter() - started,
        "epochs": args.epochs,
        "seed": args.seed,
        "activation_scope": "contact-state recovery only; full rollout requires an explicit non-privileged trigger",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
