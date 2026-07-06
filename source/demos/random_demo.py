# -*- coding: utf-8 -*-
"""Random action demo for DexHandGymEnv."""

from __future__ import annotations

import time
from typing import Any, Dict

try:
    from source.environments.rl_env import make_env
except ImportError:
    from source.environments.rl_env import make_env


def main() -> None:
    env = make_env(render_mode="human", control_mode="position")
    obs, info = env.reset(seed=0)
    reward = 0.0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if env._viewer is None:
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
