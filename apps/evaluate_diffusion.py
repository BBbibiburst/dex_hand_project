"""Closed-loop task-success evaluation for a trained diffusion policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from source.envs.manipulation import make_manipulation_env, registered_tasks
from source.imitation.diffusion_policy import DiffusionPolicy
from source.imitation.observation import real_observable_state, validate_position_action_contract


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", choices=registered_tasks(), default="lift")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--action-steps", type=int, default=4)
    p.add_argument(
        "--success-hold-steps",
        type=int,
        default=1,
        help="Require this many consecutive task-success steps before counting success.",
    )
    p.add_argument(
        "--inference-steps",
        type=int,
        help="Optional reduced deterministic DDIM step count; default uses full DDPM sampling.",
    )
    p.add_argument("--camera", default="agentview")
    # Generated LeRobot datasets use 160x120 RGB. Keep closed-loop inputs on
    # the same pixel grid unless a checkpoint was explicitly trained otherwise.
    p.add_argument("--image-width", type=int, default=160)
    p.add_argument("--image-height", type=int, default=120)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hand-name")
    p.add_argument("--robot-config")
    p.add_argument("--object-id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    p.add_argument("--video", type=Path)
    return p.parse_args()


def tensor_observation(obs, image, state, device, *, use_tactile=True):
    image = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    tactile_values = (
        np.asarray(obs["tactile"], np.float32).reshape(-1)
        if use_tactile
        else np.zeros(0, np.float32)
    )
    tactile = torch.from_numpy(tactile_values).unsqueeze(0).to(device)
    state = torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0).to(device)
    return image, tactile, state


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy = DiffusionPolicy.from_checkpoint(checkpoint, args.device).eval()
    # Automatically generated demonstrations contain low-level position controls,
    # so closed-loop evaluation must use the same action semantics.
    kwargs = {"control_mode": "position", "render_mode": None, "episode_length": args.max_steps}
    if args.hand_name:
        kwargs["hand_name"] = args.hand_name
    if args.robot_config:
        kwargs["robot_config_path"] = args.robot_config
    task_config = {"reward_shaping": False}
    if args.object_id is not None:
        if args.task != "lift":
            raise ValueError("--object-id is currently supported only for Lift evaluation.")
        task_config["object_id"] = args.object_id
    env = make_manipulation_env(args.task, task_config=task_config, **kwargs)
    validate_position_action_contract(env)
    renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
    if args.success_hold_steps <= 0:
        raise ValueError("--success-hold-steps must be positive.")
    successes, returns, lengths, latencies, episode_rows = 0, [], [], [], []
    action_deltas: list[float] = []
    drops = 0
    video_writer = None
    if args.video is not None:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(args.video, fps=20)
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            obs, _ = env.reset(seed=episode_seed)
            total = 0.0
            success = False
            steps = 0
            success_run = 0
            ever_lifted = False
            dropped = False
            contact_steps = 0
            contact_run = 0
            longest_contact_run = 0
            previous_action = None
            while steps < args.max_steps and not success:
                renderer.update_scene(env.data, camera=args.camera)
                image = renderer.render().copy()
                if video_writer is not None and episode == 0:
                    video_writer.append_data(image)
                inputs = tensor_observation(
                    obs,
                    image,
                    real_observable_state(env),
                    args.device,
                    use_tactile=policy.config.tactile_dim > 0,
                )
                if (
                    inputs[1].shape[-1] != policy.config.tactile_dim
                    or inputs[2].shape[-1] != policy.config.state_dim
                ):
                    raise ValueError(
                        "Evaluation robot/task observation dimensions do not match the "
                        "training checkpoint."
                    )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                actions = (
                    policy.sample(*inputs, inference_steps=args.inference_steps)[0].cpu().numpy()
                )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - started)
                for action in actions[: args.action_steps]:
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    if previous_action is not None:
                        action_deltas.append(float(np.linalg.norm(action - previous_action)))
                    previous_action = action.copy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    steps += 1
                    task_success = bool(info.get("task_success", False))
                    if task_success:
                        ever_lifted = True
                        success_run += 1
                    else:
                        dropped = dropped or ever_lifted
                        success_run = 0
                    success = success_run >= args.success_hold_steps
                    touching = bool(
                        env.task._is_robot_touching_object(env.model, env.data, "object")
                    )
                    contact_steps += int(touching)
                    contact_run = contact_run + 1 if touching else 0
                    longest_contact_run = max(longest_contact_run, contact_run)
                    if success or terminated or truncated or steps >= args.max_steps:
                        break
            successes += int(success)
            drops += int(dropped)
            returns.append(total)
            lengths.append(steps)
            episode_rows.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "success": success,
                    "return": total,
                    "steps": steps,
                    "drop": dropped,
                    "contact_fraction": contact_steps / steps if steps else 0.0,
                    "longest_contact_run": longest_contact_run,
                }
            )
            print(f"episode={episode} success={success} return={total:.3f} steps={steps}")
    finally:
        if video_writer is not None:
            video_writer.close()
        renderer.close()
        env.close()
    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "object_id": args.object_id,
        "seed": args.seed,
        "inference_steps": args.inference_steps or policy.config.diffusion_steps,
        "successes": successes,
        "episodes": args.episodes,
        "success_rate": successes / args.episodes,
        "drops": drops,
        "drop_rate": drops / args.episodes,
        "success_hold_steps": args.success_hold_steps,
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(lengths)),
        "mean_inference_latency_ms": 1000.0 * float(np.mean(latencies)) if latencies else None,
        "p95_inference_latency_ms": 1000.0 * float(np.percentile(latencies, 95))
        if latencies
        else None,
        "mean_contact_fraction": float(np.mean([row["contact_fraction"] for row in episode_rows])),
        "mean_action_delta_l2": float(np.mean(action_deltas)) if action_deltas else None,
        "p95_action_delta_l2": (float(np.percentile(action_deltas, 95)) if action_deltas else None),
        "episode_results": episode_rows,
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
