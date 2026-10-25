"""Diagnose generated-demo action semantics and fixed-observation DP predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from source.envs.manipulation import make_lift_env
from source.imitation.diffusion_policy import DiffusionPolicy
from source.imitation.observation import action_layout, actuator_ids, real_observable_state
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--legacy-dataset", type=Path)
    parser.add_argument("--legacy-repo-id")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _episode_actions(root: Path, repo_id: str, episode: int = 0) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    indices = [
        index
        for index, value in enumerate(dataset.hf_dataset["episode_index"])
        if int(value) == episode
    ]
    return np.stack(
        [np.asarray(dataset.hf_dataset[index]["action"], dtype=np.float32) for index in indices]
    )


def _environment(payload: dict, frames: int):
    metadata = dict(payload.get("metadata", {}))
    return make_lift_env(
        task_config={
            "object_id": str(payload["object_id"]),
            "reward_shaping": False,
            "terminate_on_success": False,
        },
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=None,
        control_dt=float(metadata.get("control_dt", 0.05)),
        episode_length=frames + 20,
    )


def _error_summary(reference: np.ndarray, candidate: np.ndarray) -> dict:
    count = min(len(reference), len(candidate))
    delta = np.asarray(candidate[:count]) - np.asarray(reference[:count])
    return {
        "reference_frames": int(len(reference)),
        "candidate_frames": int(len(candidate)),
        "compared_frames": count,
        "per_dimension_max_abs": np.max(np.abs(delta), axis=0).tolist(),
        "per_dimension_rmse": np.sqrt(np.mean(delta * delta, axis=0)).tolist(),
        "global_max_abs": float(np.max(np.abs(delta))),
        "global_rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def _execute(env, actions, initial_qpos, initial_qvel, initial_ctrl) -> dict:
    env.reset(seed=0)
    env.data.qpos[:] = initial_qpos
    env.data.qvel[:] = initial_qvel
    env.data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(env.model, env.data)
    binding = env.task._require_bindings().objects["object"]
    initial_z = float(env.data.xpos[binding.body_id, 2])
    contact_steps = 0
    successes = 0
    maximum_lift = 0.0
    for action in np.asarray(actions, dtype=np.float32):
        _, _, terminated, truncated, info = env.step(
            np.clip(action, env.action_space.low, env.action_space.high)
        )
        lift = float(env.data.xpos[binding.body_id, 2]) - initial_z
        maximum_lift = max(maximum_lift, lift)
        successes += int(bool(info.get("task_success", False)))
        contact_steps += int(env.task._is_robot_touching_object(env.model, env.data, "object"))
        if terminated or truncated:
            break
    final_lift = float(env.data.xpos[binding.body_id, 2]) - initial_z
    return {
        "frames": int(len(actions)),
        "task_success_frames": successes,
        "maximum_lift_m": maximum_lift,
        "final_lift_m": final_lift,
        "contact_fraction": contact_steps / max(len(actions), 1),
    }


def _fixed_inputs(env, renderer, use_tactile: bool, device: str):
    observation = env._get_observation()
    renderer.update_scene(env.data, camera="agentview")
    image = renderer.render().copy()
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    tactile = np.asarray(observation["tactile"], dtype=np.float32).reshape(-1)
    if not use_tactile:
        tactile = np.zeros(0, dtype=np.float32)
    return (
        image_tensor,
        torch.from_numpy(tactile).unsqueeze(0).to(device),
        torch.from_numpy(real_observable_state(env)).unsqueeze(0).to(device),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    env = _environment(payload, int(payload.get("frames", 500)))
    try:
        env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
        controls, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(manifest, env)
        if initial_ctrl is None:
            initial_ctrl = env.data.ctrl.copy()
        fixed_actions = _episode_actions(args.dataset, args.repo_id)
        report = {
            "manifest": str(manifest),
            "object_id": payload["object_id"],
            "action_layout": list(action_layout(env)),
            "action_units": ["radian"] * 7 + ["meter"] * 6,
            "strict_replay": asdict(
                strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=False)
            ),
            "fixed_dataset_error": _error_summary(controls, fixed_actions),
            "raw_execution": _execute(env, controls, initial_qpos, initial_qvel, initial_ctrl),
            "fixed_dataset_execution": _execute(
                env, fixed_actions, initial_qpos, initial_qvel, initial_ctrl
            ),
            "initial_action_minus_ctrl": (
                controls[0] - np.asarray(initial_ctrl)[actuator_ids(env)]
            ).tolist(),
            "checkpoints": {},
        }
        if args.legacy_dataset is not None:
            if not args.legacy_repo_id:
                raise ValueError("--legacy-repo-id is required with --legacy-dataset.")
            legacy_actions = _episode_actions(args.legacy_dataset, args.legacy_repo_id)
            report["legacy_dataset_error"] = _error_summary(controls, legacy_actions)
            report["legacy_dataset_execution"] = _execute(
                env, legacy_actions, initial_qpos, initial_qvel, initial_ctrl
            )

        env.data.qpos[:] = initial_qpos
        env.data.qvel[:] = initial_qvel
        env.data.ctrl[:] = initial_ctrl
        mujoco.mj_forward(env.model, env.data)
        renderer = mujoco.Renderer(env.model, height=120, width=160)
        prediction_groups = {}
        try:
            for checkpoint_path in args.checkpoint:
                checkpoint = torch.load(
                    checkpoint_path, map_location=args.device, weights_only=False
                )
                policy = DiffusionPolicy.from_checkpoint(checkpoint, args.device).eval()
                inputs = _fixed_inputs(env, renderer, policy.config.tactile_dim > 0, args.device)
                samples = []
                for seed in range(args.samples):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    samples.append(policy.sample(*inputs)[0].cpu().numpy())
                predictions = np.stack(samples)
                prediction_groups[checkpoint_path.stem] = predictions
                first = predictions[:, 0]
                current = np.asarray(initial_ctrl)[actuator_ids(env)]
                expert = controls[0]
                normalized = (
                    controls - policy.action_mean.cpu().numpy()
                ) / policy.action_std.cpu().numpy()
                restored = (
                    normalized * policy.action_std.cpu().numpy() + policy.action_mean.cpu().numpy()
                )
                low = np.asarray(env.action_space.low)
                high = np.asarray(env.action_space.high)
                report["checkpoints"][str(checkpoint_path)] = {
                    "samples": args.samples,
                    "first_action_mean": first.mean(axis=0).tolist(),
                    "first_action_std": first.std(axis=0).tolist(),
                    "first_action_vs_expert_rmse": float(
                        np.sqrt(np.mean((first.mean(axis=0) - expert) ** 2))
                    ),
                    "first_action_vs_current_ctrl_rmse": float(
                        np.sqrt(np.mean((first.mean(axis=0) - current) ** 2))
                    ),
                    "mean_adjacent_action_delta_l2": float(
                        np.mean(np.linalg.norm(np.diff(predictions, axis=1), axis=-1))
                    ),
                    "out_of_bounds_fraction": float(
                        np.mean((predictions < low) | (predictions > high))
                    ),
                    "normalization_roundtrip_max_abs": float(np.max(np.abs(restored - controls))),
                }
        finally:
            renderer.close()

        np.savez_compressed(
            args.output / "fixed_observation_predictions.npz",
            expert=controls[:16],
            current_ctrl=np.asarray(initial_ctrl)[actuator_ids(env)],
            **prediction_groups,
        )
        if prediction_groups:
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
            for dimension, axis in enumerate(axes.flat):
                if dimension >= controls.shape[1]:
                    axis.axis("off")
                    continue
                axis.plot(controls[:16, dimension], color="black", label="expert", linewidth=2)
                for label, predictions in prediction_groups.items():
                    mean = predictions.mean(axis=0)[:, dimension]
                    std = predictions.std(axis=0)[:, dimension]
                    steps = np.arange(len(mean))
                    axis.plot(steps, mean, label=label)
                    axis.fill_between(steps, mean - std, mean + std, alpha=0.15)
                axis.set_title(action_layout(env)[dimension], fontsize=8)
                axis.grid(alpha=0.2)
            axes.flat[0].legend(fontsize=7)
            figure.suptitle("Fixed initial observation: expert and sampled 13-D action chunks")
            figure.tight_layout()
            figure.savefig(args.output / "fixed_observation_action_curves.png", dpi=180)
            plt.close(figure)
        (args.output / "action_chain_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
