"""Plot tactile sampling grids exposed by the configured sensor backend."""

from __future__ import annotations

import argparse

from source.cli.robot_config import add_robot_config_args, load_configured_robot
from source.robots.registry import get_hand
from source.viz.tactile import plot_tactile_sampling_grids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", nargs="+", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--surface-alpha", type=float, default=0.32)
    parser.add_argument("--save", type=str, default="")
    add_robot_config_args(parser, include_tactile_toggle=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_configured_robot(args)
    descriptor = get_hand(str(config["hand_name"]))
    if descriptor.tactile_sensor_factory is None:
        raise ValueError(f"End effector {descriptor.name!r} does not provide tactile sensing.")
    sensor = descriptor.tactile_sensor_factory(
        args.backend or str(config.get("tactile_backend", "simple_box")),
        **dict(config.get("tactile_options") or {}),
    )
    plot_tactile_sampling_grids(
        sensor,
        patches=args.patches,
        point_size=args.point_size,
        surface_alpha=args.surface_alpha,
        save=args.save,
    )


if __name__ == "__main__":
    main()
