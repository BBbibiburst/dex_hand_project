"""Search and physics-validate grasps for catalogue objects."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.grasping.constants import DEFAULT_GRIP_PRELOAD
from source.workflows.grasp_benchmark import GraspBenchmarkConfig, run_grasp_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--joint-candidates", type=int, default=128)
    parser.add_argument("--surface-anchors", type=int, default=24)
    parser.add_argument("--rolls-per-anchor", type=int, default=8)
    parser.add_argument("--coarse-keep", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--support-margin", type=float, default=0.008)
    parser.add_argument("--search-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument(
        "--end-effector", choices=("dex_hand", "pika_gripper"), default="dex_hand"
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--grip-preload", type=float, default=DEFAULT_GRIP_PRELOAD)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_grasp_benchmark(GraspBenchmarkConfig(**vars(args))))


if __name__ == "__main__":
    main()
