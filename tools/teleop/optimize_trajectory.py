"""Smooth a raw teleop trajectory while preserving tactile/contact-heavy frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.teleop.trajectory import (
    TeleopTrajectory,
    TrajectoryOptimizationConfig,
    optimize_teleop_trajectory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--position-smoothness", type=float, default=18.0)
    parser.add_argument("--orientation-smoothness", type=float, default=10.0)
    parser.add_argument("--hand-smoothness", type=float, default=4.0)
    parser.add_argument("--contact-fidelity", type=float, default=8.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--contact-padding-frames", type=int, default=2)
    parser.add_argument("--edge-fidelity", type=float, default=1000.0)
    parser.add_argument("--edge-frames", type=int, default=2)
    args = parser.parse_args(argv)

    trajectory = TeleopTrajectory.load(args.trajectory)
    config = TrajectoryOptimizationConfig(
        position_smoothness=args.position_smoothness,
        orientation_smoothness=args.orientation_smoothness,
        hand_smoothness=args.hand_smoothness,
        contact_fidelity=args.contact_fidelity,
        contact_threshold=args.contact_threshold,
        contact_padding_frames=args.contact_padding_frames,
        edge_fidelity=args.edge_fidelity,
        edge_frames=args.edge_frames,
    )
    optimized = optimize_teleop_trajectory(trajectory, config)
    output = args.output or args.trajectory.with_name(args.trajectory.stem + "_optimized.npz")
    optimized.metadata["source_path"] = str(args.trajectory)
    output = optimized.save(output)
    print(
        f"optimized={output} frames={optimized.horizon} "
        f"contact_frames={optimized.metadata['optimizer_contact_frames']} "
        f"mean_dpos={optimized.metadata['optimizer_mean_position_change_m']:.5f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
