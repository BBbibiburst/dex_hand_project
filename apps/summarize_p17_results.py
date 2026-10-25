"""Summarize P17 offline XYZ9 errors and paired closed-loop evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from apps.trace_diffusion_rollout import _make_env, _reset_exact
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.cartesian_action import XYZ9, CartesianChunkDataset
from source.imitation.dataset import LeRobotDiffusionDataset
from source.verification.strict_replay import load_replay_controls


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _offline(policy, dataset, indices, device):
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False)
    squared = first_squared = 0.0
    count = first_count = 0
    for batch in loader:
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        target = batch["action"].to(device)
        with torch.inference_mode():
            predicted = policy.sample(image, state)
        delta = predicted - target
        squared += float(delta.square().sum())
        count += delta.numel()
        first_squared += float(delta[:, 0].square().sum())
        first_count += delta[:, 0].numel()
    return {
        "action_rmse_physical_units": float(np.sqrt(squared / count)),
        "first_action_rmse": float(np.sqrt(first_squared / first_count)),
        "frames": len(indices),
    }


def _first_contact_per_digit(counts):
    result = []
    for digit in range(counts.shape[1]):
        indices = np.flatnonzero(counts[:, digit] > 0)
        result.append(None if not len(indices) else int(indices[0]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.root / "dataset_audit" / "split.json").read_text())["roots"]
    payload = json.loads(args.reference_manifest.read_text())
    env = _make_env(payload, 400)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
    _, qpos, qvel, ctrl = load_replay_controls(args.reference_manifest, env)
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    _reset_exact(env, qpos, qvel, ctrl)
    from source.geometry import mat_to_quat

    fixed = mat_to_quat(env.data.site_xmat[env.controller.arm_controller.site_id])
    base = LeRobotDiffusionDataset(
        repo_id=args.repo_id, root=args.dataset, horizon=16, use_tactile=False
    )
    cartesian = CartesianChunkDataset(base, env, XYZ9, fixed_orientation_wxyz=fixed)
    env.close()
    episode_rows = cartesian.episode_indices()
    policies = {}
    checkpoints = {}
    for name, path in (("old_p6", args.old_checkpoint), ("new_p17", args.new_checkpoint)):
        checkpoint = torch.load(path, map_location=args.device, weights_only=False)
        policies[name] = ActionChunkBC.from_checkpoint(checkpoint, args.device).eval()
        checkpoints[name] = checkpoint
    offline = {}
    for name, policy in policies.items():
        offline[name] = {}
        for partition in ("train", "validation", "test"):
            indices = [index for root in split[partition] for index in episode_rows[int(root)]]
            offline[name][partition] = _offline(policy, cartesian, indices, args.device)

    old_summary = json.loads(
        (args.root / "evaluation/root_04/old_p6/summary.json").read_text()
    )
    new_summary = json.loads(
        (args.root / "evaluation/root_04/new_p17/summary.json").read_text()
    )
    p14 = np.load("outputs/dp_p14_2026-10-24/divergence/root_04/alignment.npz")
    new_trace = np.load(args.root / "evaluation/root_04/new_p17/rollout_trace.npz")
    old_trace = np.load(args.root / "evaluation/root_04/old_p6/rollout_trace.npz")
    expert_wrist = p14["expert_wrist"]
    early = slice(5, 21)
    closed_loop = {}
    for name, summary, trace in (
        ("old_p6", old_summary, old_trace),
        ("new_p17", new_summary, new_trace),
    ):
        wrist = trace["bc_wrist"]
        limit = min(len(wrist), len(expert_wrist))
        digit = trace["bc_digit_contact_count"]
        closed_loop[name] = {
            "approach_wrist_error_5_20_m": float(
                np.mean(np.linalg.norm(wrist[early] - expert_wrist[early], axis=1))
            ),
            "first_contact_step": summary["bc"]["first_contact_step"],
            "first_contact_per_digit": _first_contact_per_digit(digit),
            "contact_fraction": summary["bc"]["contact_fraction"],
            "maximum_lift_m": summary["bc"]["maximum_lift_m"],
            "final_lift_m": summary["bc"]["final_lift_m"],
            "strict_success": summary["strict_replay"]["success"],
            "strict_status": summary["strict_replay"]["verification_status"],
            "wrist_trace_length": limit,
            "inference_ik_p50_ms": summary["inference_ik_p50_ms"],
            "inference_ik_p95_ms": summary["inference_ik_p95_ms"],
        }
    normalization = {
        name: {
            "state_mean": checkpoints[name]["model"]["state_mean"].cpu().tolist(),
            "state_std": checkpoints[name]["model"]["state_std"].cpu().tolist(),
            "action_mean": checkpoints[name]["model"]["action_mean"].cpu().tolist(),
            "action_std": checkpoints[name]["model"]["action_std"].cpu().tolist(),
        }
        for name in checkpoints
    }
    # Audit state aliasing and near-duplicate episodes without using object truth.
    new_policy = policies["new_p17"]
    states = torch.stack([base.statistics_item(i)[0] for i in range(len(base))])
    state_n = (states - new_policy.state_mean.cpu()) / new_policy.state_std.cpu()
    first_actions = cartesian.labels[:, 0]
    action_n = (
        first_actions - new_policy.action_mean.cpu()
    ) / new_policy.action_std.cpu()
    state_distance = torch.cdist(state_n, state_n).numpy() / np.sqrt(state_n.shape[1])
    action_distance = torch.cdist(action_n, action_n).numpy() / np.sqrt(action_n.shape[1])
    episode_of = np.empty(len(base), dtype=np.int32)
    local_frame = np.empty(len(base), dtype=np.int32)
    for episode, indices in episode_rows.items():
        episode_of[indices] = episode
        local_frame[indices] = np.arange(len(indices))
    triangle = np.triu(np.ones_like(state_distance, dtype=bool), 1)
    separated = (episode_of[:, None] != episode_of[None, :]) | (
        np.abs(local_frame[:, None] - local_frame[None, :]) >= 10
    )
    mask = triangle & separated
    near_threshold = float(np.percentile(state_distance[mask], 1.0))
    pairs = np.argwhere(mask & (state_distance <= near_threshold))
    ranked = sorted(
        pairs.tolist(), key=lambda pair: float(action_distance[pair[0], pair[1]]), reverse=True
    )[:20]
    aliasing = {
        "state_distance_percentile": 1.0,
        "state_distance_threshold": near_threshold,
        "candidate_pairs": len(pairs),
        "top_pairs": [
            {
                "first_episode": int(episode_of[first]),
                "first_frame": int(local_frame[first]),
                "second_episode": int(episode_of[second]),
                "second_frame": int(local_frame[second]),
                "state_normalized_distance": float(state_distance[first, second]),
                "xyz9_action_normalized_distance": float(action_distance[first, second]),
            }
            for first, second in ranked
        ],
        "limitation": "39D deployable-state audit only; RGB may disambiguate some pairs.",
    }
    episode_near_duplicates = []
    for first in range(5):
        for second in range(first + 1, 5):
            first_labels = cartesian.labels[episode_rows[first]]
            second_labels = cartesian.labels[episode_rows[second]]
            episode_near_duplicates.append(
                {
                    "first_root": first,
                    "second_root": second,
                    "xyz9_chunk_rmse": float(
                        torch.mean((first_labels - second_labels).square()).sqrt()
                    ),
                }
            )
    result = {
        "schema_version": 1,
        "checkpoint_sha256": {
            "old_p6": _sha(args.old_checkpoint),
            "new_p17": _sha(args.new_checkpoint),
        },
        "parameter_count": {
            name: sum(value.numel() for value in checkpoint["model"].values())
            for name, checkpoint in checkpoints.items()
        },
        "offline": offline,
        "closed_loop_held_out_root_04": closed_loop,
        "normalization": normalization,
        "state_action_aliasing": aliasing,
        "episode_near_duplicate_matrix": episode_near_duplicates,
        "evaluation_attempts": 1,
        "sample_size_warning": "One held-out root only; no generalization claim.",
    }
    (args.root / "comparison_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    figure_dir = args.root / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for name, trace in (("Old P6", old_trace), ("New P17", new_trace)):
        axes[0].plot(
            np.linalg.norm(trace["bc_wrist"] - trace["bc_object"], axis=1), label=name
        )
        axes[1].plot(
            trace["bc_object"][:, 2] - trace["bc_object"][0, 2], label=name
        )
    axes[0].set_ylabel("Wrist-object distance [m]")
    axes[1].set_ylabel("Object lift [m]")
    axes[1].set_xlabel("Control step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "old_vs_new_closed_loop.png", dpi=180)
    plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key != "normalization"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
