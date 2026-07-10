# -*- coding: utf-8 -*-
"""Environment viewer for the migrated single-arm manipulation tasks.

This demo does not solve the task, teleport objects, or drive an oracle robot
motion. It only runs the environment so task geometry, observations, rewards,
contacts, and reset placement can be inspected in the MuJoCo viewer.

Usage::

    python -m source.demos.manipulation_task_playback --task lift
    python -m source.demos.manipulation_task_playback --task stack
    python -m source.demos.manipulation_task_playback --task pick_place --single-object can
    python -m source.demos.manipulation_task_playback --task nut_assembly --single-nut square_nut
    python -m source.demos.manipulation_task_playback --task door --no-latch
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
from mujoco import viewer

from source.demos.common import RealtimePacer, add_robot_config_args
from source.envs.manipulation import make_manipulation_env, registered_tasks
from source.viz.overlays import clear_markers, draw_label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Viewer for migrated single-arm manipulation environments."
    )
    parser.add_argument(
        "--task",
        choices=registered_tasks(),
        default="lift",
        help="Task environment to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reset seed controlling initial task placement.",
    )
    parser.add_argument(
        "--single-object",
        choices=("milk", "bread", "cereal", "can"),
        help="Run PickPlace with only the selected object.",
    )
    parser.add_argument(
        "--single-nut",
        choices=("square_nut", "round_nut"),
        help="Run NutAssembly with only the selected nut.",
    )
    parser.add_argument(
        "--no-latch",
        action="store_true",
        help="Run Door without the rotatable latch joint.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        help="Viewer sync frequency.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=20.0,
        help="Environment control frequency.",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Sample random actions instead of holding the reset action.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.20,
        help="Scale for sampled random actions around the reset action.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run as fast as possible instead of wall-clock realtime.",
    )
    parser.add_argument(
        "--print-status",
        action="store_true",
        help="Print reward and success once per second.",
    )
    add_robot_config_args(parser)
    return parser.parse_args()


def _make_env(args: argparse.Namespace):
    if args.control_hz <= 0.0:
        raise ValueError(f"--control-hz must be positive, got {args.control_hz}.")
    if args.render_fps <= 0.0:
        raise ValueError(f"--render-fps must be positive, got {args.render_fps}.")
    if args.action_scale < 0.0:
        raise ValueError(f"--action-scale must be non-negative, got {args.action_scale}.")
    if args.single_object is not None and args.task != "pick_place":
        raise ValueError("--single-object is only valid with --task pick_place.")
    if args.single_nut is not None and args.task != "nut_assembly":
        raise ValueError("--single-nut is only valid with --task nut_assembly.")
    if args.no_latch and args.task != "door":
        raise ValueError("--no-latch is only valid with --task door.")

    env_kwargs = dict(
        robot_config_path=getattr(args, "robot_config", None),
        arm_name=getattr(args, "arm_name", None),
        hand_name=getattr(args, "hand_name", None),
        base_name=getattr(args, "base_name", None),
        enable_tactile_sensors=False if getattr(args, "no_tactile", False) else None,
        control_mode="ik",
        render_mode=None,
        control_dt=1.0 / args.control_hz,
    )
    # Avoid passing None as an explicit robot-config override. This lets the
    # project's current robot profile provide values for omitted CLI options.
    env_kwargs = {key: value for key, value in env_kwargs.items() if value is not None}
    task_config: Dict[str, Any] = {"reward_shaping": True}
    if args.task == "pick_place" and args.single_object is not None:
        task_config["single_object"] = args.single_object
    if args.task == "nut_assembly" and args.single_nut is not None:
        task_config["single_nut"] = args.single_nut
    if args.task == "door":
        task_config["use_latch"] = not args.no_latch
    return make_manipulation_env(args.task, task_config=task_config, **env_kwargs)


def _sample_action(
    env, base_action: np.ndarray, rng: np.random.Generator, scale: float
) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    noise = rng.uniform(-scale, scale, size=base_action.shape).astype(np.float32)
    action = base_action + noise
    finite = np.isfinite(low) & np.isfinite(high)
    action[finite] = np.clip(action[finite], low[finite], high[finite])
    return action.astype(np.float32)


def run_viewer(args: argparse.Namespace) -> None:
    env = _make_env(args)
    rng = np.random.default_rng(args.seed)
    obs, _info = env.reset(seed=args.seed)
    base_action = env.controller.current_action(env.model, env.data)
    action = base_action.copy()

    sim_start = float(env.data.time)
    pacer = RealtimePacer()
    pacer.reset(sim_start)
    last_print_second = -1
    reward = 0.0
    reward_info: Dict[str, Any] = {}

    try:
        with viewer.launch_passive(env.model, env.data) as handle:
            while handle.is_running():
                sim_elapsed = float(env.data.time) - sim_start

                if args.random_actions:
                    action = _sample_action(env, base_action, rng, args.action_scale)

                obs, reward, terminated, truncated, info = env.step(action)
                reward_info = {
                    key: value
                    for key, value in info.items()
                    if key.startswith("reward_") or key == "task_success"
                }

                clear_markers(handle)
                draw_label(
                    handle,
                    np.asarray([0.0, -0.32, 1.15], dtype=np.float32),
                    (
                        f"{args.task} env | reward {reward:.3f} | "
                        f"success {bool(info.get('task_success', False))}"
                    ),
                )
                handle.sync()

                if args.print_status and int(sim_elapsed) != last_print_second:
                    last_print_second = int(sim_elapsed)
                    print(
                        f"t={sim_elapsed:5.2f} reward={reward:.3f} "
                        f"terminated={terminated} truncated={truncated} info={reward_info}"
                    )

                if terminated or truncated:
                    obs, _info = env.reset(seed=args.seed)
                    base_action = env.controller.current_action(env.model, env.data)
                    action = base_action.copy()
                    sim_start = float(env.data.time)
                    pacer.reset(sim_start)
                    last_print_second = -1

                if not args.no_realtime:
                    pacer.sleep_until(float(env.data.time))
    finally:
        env.close()


def main() -> None:
    run_viewer(_parse_args())


if __name__ == "__main__":
    main()
