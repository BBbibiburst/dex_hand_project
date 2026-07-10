# -*- coding: utf-8 -*-
"""Random action demo for RobotGymEnv."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import numpy as np

from source.demos.common import RealtimePacer, add_robot_config_args, make_demo_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random action demo for RobotGymEnv.")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=2.0,
        help="Random action update frequency. Default is 2 Hz.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        help="Viewer refresh frequency. Default is 60 FPS.",
    )
    parser.add_argument(
        "--action-filter",
        type=float,
        default=0.08,
        help="Low-pass factor for random target changes. 1.0 disables smoothing.",
    )
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_config_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.control_hz <= 0.0:
        raise ValueError(f"--control-hz must be positive, got {args.control_hz}.")
    if args.render_fps <= 0.0:
        raise ValueError(f"--render-fps must be positive, got {args.render_fps}.")
    if not 0.0 < args.action_filter <= 1.0:
        raise ValueError(f"--action-filter must be in (0, 1], got {args.action_filter}.")

    env = make_demo_env(
        args,
        render_mode="human",
        control_mode="position",
        control_dt=1.0 / args.control_hz,
    )
    obs, info = env.reset(seed=args.seed)
    reward = 0.0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    env.action_space.seed(args.seed)
    action = env.controller.current_action(env.model, env.data)
    target_action = action.copy()
    render_dt = 1.0 / args.render_fps
    next_control_time = float(env.data.time)
    sim_start = float(env.data.time)
    pacer = RealtimePacer()
    pacer.reset(sim_start)

    try:
        while True:
            control_updates = 0
            if float(env.data.time) >= next_control_time:
                target_action = env.action_space.sample().astype(np.float32)
                next_control_time += env.config.control_dt
                control_updates = 1

            action = (
                action + args.action_filter * (target_action - action)
            ).astype(np.float32)
            info = env.controller.apply_action(env.model, env.data, action)

            physics_steps = max(1, int(round(render_dt / env.model.opt.timestep)))
            env.step_physics(physics_steps, control_updates=control_updates)
            obs = env._get_observation()
            reward = 0.0
            terminated = False
            truncated = env.elapsed_steps >= env.config.episode_length

            env.render()
            if not env.is_rendering:
                break
            if terminated or truncated:
                obs, info = env.reset()
                action = env.controller.current_action(env.model, env.data)
                target_action = action.copy()
                next_control_time = float(env.data.time)
                sim_start = float(env.data.time)
                pacer.reset(sim_start)
                terminated = False
                truncated = False

            if not args.no_realtime:
                sim_elapsed = float(env.data.time) - sim_start
                pacer.sleep_until(float(env.data.time))
    finally:
        env.close()

    print("observation keys:", list(obs.keys()))
    print("action shape:", env.action_space.shape)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("info:", info)


if __name__ == "__main__":
    main()
