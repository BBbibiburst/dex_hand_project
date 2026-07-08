# -*- coding: utf-8 -*-
"""Random action demo for RobotGymEnv."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict

from source.environments.rl_env import RLEnvConfig, make_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random action demo for RobotGymEnv.")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=2.0,
        help="Random action update frequency. Default is 2 Hz.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.control_hz <= 0.0:
        raise ValueError(f"--control-hz must be positive, got {args.control_hz}.")

    config = RLEnvConfig(control_dt=1.0 / args.control_hz)
    env = make_env(render_mode="human", control_mode="position", config=config)
    obs, info = env.reset(seed=args.seed)
    reward = 0.0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if not env.is_rendering:
                break
            if terminated or truncated:
                obs, info = env.reset()
                terminated = False
                truncated = False
            time.sleep(env.config.control_dt)
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
