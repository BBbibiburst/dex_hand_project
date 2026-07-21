"""CLI helpers for selecting and overriding robot assemblies."""

from __future__ import annotations

import argparse
from typing import Any

from source.envs.rl_env import make_env
from source.robots.config import load_robot_config


def add_robot_config_args(
    parser: argparse.ArgumentParser,
    *,
    include_device_overrides: bool = True,
    include_tactile_toggle: bool = True,
) -> None:
    """Add the standard robot assembly arguments to ``parser``."""
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help="Robot config JSON. Defaults to configs/current_robot.json.",
    )
    if not include_device_overrides:
        return
    parser.add_argument("--arm-name", type=str, default=None)
    parser.add_argument("--hand-name", type=str, default=None)
    parser.add_argument("--base-name", type=str, default=None)
    if include_tactile_toggle:
        parser.add_argument("--no-tactile", action="store_true")


def robot_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return non-``None`` assembly overrides from parsed arguments."""
    return {
        "arm_name": getattr(args, "arm_name", None),
        "hand_name": getattr(args, "hand_name", None),
        "base_name": getattr(args, "base_name", None),
        "enable_tactile_sensors": False if getattr(args, "no_tactile", False) else None,
    }


def make_configured_env(
    args: argparse.Namespace,
    *,
    render_mode: str | None = None,
    control_mode: str | None = None,
    **overrides: Any,
):
    """Create ``RobotGymEnv`` from standard robot CLI arguments."""
    config_overrides = robot_config_overrides(args)
    config_overrides.update(overrides)
    return make_env(
        render_mode=render_mode,
        control_mode=control_mode,
        robot_config_path=getattr(args, "robot_config", None),
        **config_overrides,
    )


def load_configured_robot(args: argparse.Namespace) -> dict[str, Any]:
    """Load a robot config and apply standard CLI overrides."""
    config = load_robot_config(getattr(args, "robot_config", None))
    for key, value in robot_config_overrides(args).items():
        if value is not None:
            config[key] = value
    return config
