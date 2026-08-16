"""Replay a raw or optimized Vive + glove trajectory in authoritative MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.teleop.trajectory import TeleopTrajectory, replay_teleop_trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--no-compare-reference", action="store_true")
    args = parser.parse_args(argv)

    trajectory = TeleopTrajectory.load(args.trajectory)
    result = replay_teleop_trajectory(
        trajectory,
        render=not args.no_render,
        realtime=not args.no_realtime,
        hold_seconds=args.hold_seconds,
        compare_reference=not args.no_compare_reference,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
