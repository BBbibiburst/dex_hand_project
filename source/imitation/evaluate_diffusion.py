"""Closed-loop task-success evaluation for a trained diffusion policy."""
from __future__ import annotations

import argparse
from pathlib import Path
import mujoco
import numpy as np
import torch

from source.envs.manipulation import make_manipulation_env, registered_tasks
from source.imitation.diffusion_policy import DiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--task", choices=registered_tasks(), required=True)
    p.add_argument("--episodes", type=int, default=20); p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--action-steps", type=int, default=4); p.add_argument("--camera", default="agentview")
    p.add_argument("--image-width", type=int, default=640); p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hand-name"); p.add_argument("--robot-config")
    return p.parse_args()


def tensor_observation(obs, image, device):
    image = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    tactile = torch.from_numpy(np.asarray(obs["tactile"], np.float32)).flatten().unsqueeze(0).to(device)
    state = torch.from_numpy(np.concatenate([obs["qpos"], obs["qvel"], obs["ctrl"]])).float().unsqueeze(0).to(device)
    return image, tactile, state


def main():
    args = parse_args(); checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy = DiffusionPolicy.from_checkpoint(checkpoint, args.device).eval()
    kwargs = {"control_mode": "ik", "render_mode": None, "episode_length": args.max_steps}
    if args.hand_name: kwargs["hand_name"] = args.hand_name
    if args.robot_config: kwargs["robot_config_path"] = args.robot_config
    env = make_manipulation_env(args.task, task_config={"reward_shaping": False}, **kwargs)
    renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
    successes, returns, lengths = 0, [], []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode); total = 0.0; success = False; steps = 0
            while steps < args.max_steps and not success:
                renderer.update_scene(env.data, camera=args.camera); image = renderer.render().copy()
                inputs = tensor_observation(obs, image, args.device)
                if inputs[1].shape[-1] != policy.config.tactile_dim or inputs[2].shape[-1] != policy.config.state_dim:
                    raise ValueError("Evaluation robot/task observation dimensions do not match the training checkpoint.")
                actions = policy.sample(*inputs)[0].cpu().numpy()
                for action in actions[:args.action_steps]:
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, terminated, truncated, info = env.step(action); total += reward; steps += 1
                    success = bool(info.get("task_success", False))
                    if success or terminated or truncated or steps >= args.max_steps: break
            successes += int(success); returns.append(total); lengths.append(steps)
            print(f"episode={episode} success={success} return={total:.3f} steps={steps}")
    finally:
        renderer.close(); env.close()
    print(f"success_rate={successes/args.episodes:.3f} ({successes}/{args.episodes}) mean_return={np.mean(returns):.3f} mean_steps={np.mean(lengths):.1f}")


if __name__ == "__main__": main()
