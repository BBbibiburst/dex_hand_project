"""Automatically collect phase-scripted demonstrations into LeRobot format."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

from source.demos.common import add_robot_config_args
from source.demos.strategies import create_strategy, registered_strategies
from source.envs.manipulation import make_manipulation_env
from source.teleop.devices import GloveSample, ViveSample
from source.teleop.lerobot_recorder import LeRobotEpisodeRecorder
from source.viz.overlays import clear_markers, draw_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=registered_strategies(), default="lift")
    parser.add_argument("--repo-id", default="local/dex-hand-scripted-demonstrations")
    parser.add_argument("--output", type=Path, default=Path("datasets/scripted_lerobot"))
    parser.add_argument("--episodes", type=int, default=20, help="Number of successful episodes.")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    add_robot_config_args(parser)
    return parser.parse_args()


def _make_env(args):
    overrides = {
        "robot_config_path": getattr(args, "robot_config", None),
        "arm_name": getattr(args, "arm_name", None),
        "hand_name": getattr(args, "hand_name", None),
        "base_name": getattr(args, "base_name", None),
        "control_mode": "ik",
        "control_dt": 1.0 / args.fps,
        "episode_length": args.max_steps,
        "enable_tactile_sensors": not getattr(args, "no_tactile", False),
        "render_mode": None,
    }
    return make_manipulation_env(
        args.task,
        task_config={"reward_shaping": True, "terminate_on_success": True},
        **{key: value for key, value in overrides.items() if value is not None},
    )


def _operator_samples(action: np.ndarray, env, timestamp: float):
    hand_size = env.controller.hand_controller.action_size
    hand = np.asarray(action[-hand_size:], dtype=np.float32)
    low = np.asarray(env.action_space.low[-hand_size:], dtype=np.float32)
    high = np.asarray(env.action_space.high[-hand_size:], dtype=np.float32)
    denominator = np.maximum(high - low, 1e-8)
    normalized_opening = np.clip((hand - low) / denominator, 0.0, 1.0)
    if hand_size == 1:
        normalized_opening = np.repeat(normalized_opening, 6)
    glove = GloveSample(1.0 - normalized_opening, timestamp)
    vive = ViveSample(action[:3].copy(), action[3:7].copy(), timestamp)
    return glove, vive


def run(args) -> None:
    for name in ("episodes", "max_attempts", "max_steps", "fps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    env = _make_env(args)
    strategy = create_strategy(args.task)
    renderer = None
    recorder = None
    view_handle = None
    successful_episodes = 0
    attempts = 0
    try:
        observation, info = env.reset(seed=args.seed)
        if not args.dry_run:
            renderer = mujoco.Renderer(
                env.model, height=args.image_height, width=args.image_width
            )
            renderer.update_scene(env.data, camera=args.camera)
            first_image = renderer.render()
            recorder = LeRobotEpisodeRecorder(
                repo_id=args.repo_id,
                root=args.output,
                fps=args.fps,
                state_dim=env.model.nq + env.model.nv + env.model.nu,
                action_dim=env.action_space.shape[0],
                tactile_shape=np.asarray(observation["tactile"]).shape,
                image_shape=first_image.shape,
                use_videos=not args.no_video,
            )
        if args.viewer:
            view_handle = viewer.launch_passive(env.model, env.data)

        while successful_episodes < args.episodes and attempts < args.max_attempts:
            seed = args.seed + attempts
            observation, info = env.reset(seed=seed)
            strategy.reset()
            success = False
            steps = 0
            episode_return = 0.0
            for step in range(args.max_steps):
                action, _ = strategy.tick(observation, info, step, env)
                observation, reward, terminated, truncated, info = env.step(action)
                steps = step + 1
                episode_return += reward
                success = bool(info.get("task_success", False))
                if recorder is not None and renderer is not None:
                    renderer.update_scene(env.data, camera=args.camera)
                    image = renderer.render().copy()
                    glove, vive = _operator_samples(action, env, float(env.data.time))
                    recorder.add_frame(
                        observation=observation,
                        image=image,
                        action=action,
                        glove=glove,
                        vive=vive,
                        task=args.task,
                    )
                if view_handle is not None:
                    if not view_handle.is_running():
                        raise KeyboardInterrupt
                    clear_markers(view_handle)
                    draw_label(
                        view_handle,
                        np.asarray([0.0, -0.32, 1.15], dtype=np.float32),
                        f"scripted | {strategy.phase_name} | attempt {attempts + 1}",
                    )
                    view_handle.sync()
                if success or terminated or truncated or strategy.aborted:
                    break

            attempts += 1
            should_save = success or args.save_failures
            if recorder is not None:
                if should_save:
                    recorder.save_episode()
                else:
                    recorder.clear_episode()
            if success:
                successful_episodes += 1
            outcome = "SUCCESS" if success else "DISCARDED"
            print(
                f"attempt={attempts} seed={seed} outcome={outcome} steps={steps} "
                f"return={episode_return:.3f} phase={strategy.phase_name}"
            )
    except KeyboardInterrupt:
        print("Scripted collection interrupted.")
    finally:
        if recorder is not None:
            if recorder.frame_count:
                recorder.clear_episode()
            recorder.finalize()
        if view_handle is not None:
            view_handle.close()
        if renderer is not None:
            renderer.close()
        env.close()
    print(f"collected={successful_episodes}/{args.episodes} attempts={attempts}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
