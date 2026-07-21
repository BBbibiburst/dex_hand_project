# -*- coding: utf-8 -*-
"""Preview the robot assembly selected by the robot configuration."""

from __future__ import annotations

import argparse
import logging

import mujoco
from mujoco import viewer

from source.cli.robot_config import add_robot_config_args
from source.robots.builder import build_robot_model_from_config

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the configured robot assembly.")
    add_robot_config_args(parser)
    parser.add_argument("--no-scene", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model, data = build_robot_model_from_config(
        args.robot_config,
        arm_name=args.arm_name,
        hand_name=args.hand_name,
        base_name=args.base_name,
        add_preview_scene=False if args.no_scene else None,
        enable_tactile_sensors=False if args.no_tactile else None,
    )
    LOGGER.info("Robot model compiled: nq=%d nv=%d nu=%d", model.nq, model.nv, model.nu)
    with viewer.launch_passive(model, data) as handle:
        while handle.is_running():
            mujoco.mj_step(model, data)
            handle.sync()


if __name__ == "__main__":
    main()
