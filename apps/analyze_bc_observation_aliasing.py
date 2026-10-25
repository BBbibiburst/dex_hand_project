"""Measure whether similar deployable BC observations require different expert actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from source.grasping.contracts import DemonstrationEpisode
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.dataset import LeRobotDiffusionDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDiffusionDataset(
        repo_id=args.repo_id, root=args.dataset, horizon=16, use_tactile=False
    )
    policy = ActionChunkBC.from_checkpoint(
        torch.load(args.checkpoint, map_location=args.device, weights_only=False), args.device
    ).eval()
    images, states, actions = [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        images.append(sample["image"])
        states.append(sample["state"])
        actions.append(sample["action"][0])
    state = torch.stack(states).to(args.device)
    action = torch.stack(actions).to(args.device)
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            features.append(
                policy.vision(torch.stack(images[start : start + args.batch_size]).to(args.device))
            )
    visual = torch.cat(features)
    state_n = (state - policy.state_mean) / policy.state_std
    action_n = (action - policy.action_mean) / policy.action_std
    visual = visual / visual.norm(dim=1, keepdim=True).clamp_min(1e-8)
    visual_distance = (1.0 - visual @ visual.T).clamp_min(0.0).cpu().numpy()
    state_distance = torch.cdist(state_n, state_n).cpu().numpy() / np.sqrt(state_n.shape[1])
    action_distance = torch.cdist(action_n, action_n).cpu().numpy() / np.sqrt(action_n.shape[1])
    observation_distance = visual_distance + state_distance
    n = len(dataset)
    valid = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= 10
    triangle = np.triu(np.ones((n, n), dtype=bool), 1)
    mask = valid & triangle
    flat_obs = observation_distance[mask]
    threshold = float(np.percentile(flat_obs, 2.0))
    candidates = np.argwhere(mask & (observation_distance <= threshold))
    ranked = sorted(
        candidates.tolist(), key=lambda pair: float(action_distance[pair[0], pair[1]]), reverse=True
    )[:20]
    episode = DemonstrationEpisode.load(args.manifest)
    offset = len(episode.arrays["stage"]) - len(dataset) - 1
    codes = {int(value): name for name, value in episode.metadata["stage_codes"].items()}
    rows = []
    for first, second in ranked:
        rows.append(
            {
                "first_dataset_index": first,
                "second_dataset_index": second,
                "first_source_index": offset + first,
                "second_source_index": offset + second,
                "first_stage": codes[int(episode.arrays["stage"][offset + first])],
                "second_stage": codes[int(episode.arrays["stage"][offset + second])],
                "visual_cosine_distance": float(visual_distance[first, second]),
                "state_normalized_distance": float(state_distance[first, second]),
                "combined_observation_distance": float(observation_distance[first, second]),
                "action_normalized_distance": float(action_distance[first, second]),
                "action_l2_rad": float(
                    torch.linalg.vector_norm(action[first] - action[second]).cpu()
                ),
            }
        )
    result = {
        "frames": n,
        "source_pre_action_offset": offset,
        "near_observation_percentile": 2.0,
        "near_observation_threshold": threshold,
        "candidate_pairs": int(len(candidates)),
        "top_ambiguous_pairs": rows,
        "interpretation": (
            "A pair is observationally aliased only if deployable normalized state and learned "
            "visual feature are both close while normalized expert-action distance is large."
        ),
    }
    (args.output / "aliasing.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(flat_obs, action_distance[mask], s=5, alpha=0.25)
    axis.axvline(threshold, color="tab:red", linestyle="--", label="closest 2% observations")
    axis.set(
        xlabel="Visual cosine distance + normalized state distance",
        ylabel="Normalized expert-action distance",
        title="Deployable observation similarity vs required expert action",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "observation_action_aliasing.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
