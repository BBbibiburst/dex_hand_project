"""Probe the experimental GraspQP bridge for the closed-chain Dex Hand."""

from __future__ import annotations

import argparse

import numpy as np

from source.grasping.graspqp_adapter import (
    check_graspqp_compatibility,
    sample_closed_chain_kinematics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs=6, type=float, default=[0.5] * 6)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--points-per-geom", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compatibility = check_graspqp_compatibility()
    print(
        f"graspqp_installed={compatibility.installed} "
        f"torch_installed={compatibility.torch_installed} "
        f"cuda_available={compatibility.cuda_available}"
    )
    if compatibility.reason:
        print(f"runtime_note={compatibility.reason}")
    sample = sample_closed_chain_kinematics(
        np.asarray(args.fractions),
        epsilon=args.epsilon,
        max_points_per_geom=args.points_per_geom,
    )
    print(
        f"surface_points={len(sample.surface.points)} "
        f"point_jacobian_shape={sample.point_jacobian.shape} "
        f"fingertip_jacobian_shape={sample.fingertip_jacobian.shape} "
        f"finite={np.isfinite(sample.point_jacobian).all()}"
    )


if __name__ == "__main__":
    main()
