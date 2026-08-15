"""Replay a residual grasp trajectory in authoritative C MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    from source.rl.replay import replay_residual_trajectory

    result = replay_residual_trajectory(
        args.trajectory,
        render_mode="human" if args.render else None,
    )
    print(
        f"success={result.success} success_fraction={result.success_fraction:.1%} "
        f"lift={result.object_lift:.3f}m frames={result.frames}",
        flush=True,
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
