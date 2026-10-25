"""Measure teacher-forced action errors for BC or ACT on one LeRobot episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from source.imitation.dataset import LeRobotDiffusionDataset


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("bc", "act"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    horizon = int(
        checkpoint["config"]["horizon"] if args.policy == "bc" else checkpoint["config"].chunk_size
    )
    dataset = LeRobotDiffusionDataset(
        repo_id=args.repo_id, root=args.dataset, horizon=horizon, use_tactile=False
    )
    indices = dataset.episode_indices().get(args.episode)
    if not indices:
        raise ValueError(f"Episode {args.episode} is absent.")
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size)
    if args.policy == "bc":
        from source.imitation.action_chunk_bc import ActionChunkBC

        policy = ActionChunkBC.from_checkpoint(checkpoint, args.device).eval()

        def predict(batch):
            return policy.sample(batch["image"].to(args.device), batch["state"].to(args.device))

    else:
        from lerobot.policies.act.modeling_act import ACTPolicy

        policy = ACTPolicy(checkpoint["config"]).to(args.device)
        policy.load_state_dict(checkpoint["model"])
        policy.eval()
        state_mean = checkpoint["state_mean"].to(args.device)
        state_std = checkpoint["state_std"].to(args.device)
        action_mean = checkpoint["action_mean"].to(args.device)
        action_std = checkpoint["action_std"].to(args.device)

        def predict(batch):
            inputs = {
                "observation.images.agentview": batch["image"].to(args.device),
                "observation.state": (batch["state"].to(args.device) - state_mean) / state_std,
            }
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(inputs)
            return normalized * action_std + action_mean

    predictions, targets = [], []
    for batch in loader:
        with torch.inference_mode():
            predictions.append(predict(batch).cpu())
        targets.append(batch["action"])
    predicted = torch.cat(predictions)
    target = torch.cat(targets)
    delta = predicted - target
    first_delta = delta[:, 0]
    report = {
        "policy": args.policy,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "episode": args.episode,
        "frames": len(indices),
        "horizon": horizon,
        "teacher_forced_action_rmse": float(delta.square().mean().sqrt()),
        "teacher_forced_first_action_rmse": float(first_delta.square().mean().sqrt()),
        "per_dimension_first_action_rmse": first_delta.square().mean(0).sqrt().tolist(),
        "internal_adjacent_action_delta_l2": float(
            torch.linalg.vector_norm(torch.diff(predicted, dim=1), dim=-1).mean()
        ),
        "fixed_initial_observation_first_action_rmse": float(first_delta[0].square().mean().sqrt()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
