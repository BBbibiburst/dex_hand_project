"""Probe the GraspQP + DexEvolve generation dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.grasping.config import load_pipeline_config
from source.grasping.graspqp_adapter import graspqp_available
from source.grasping.hand_surrogate import load_or_calibrate_surrogate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recalibrate-surrogate", action="store_true")
    parser.add_argument("--mjwarp", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_pipeline_config()
    if args.recalibrate_surrogate:
        Path(config.surrogate_cache).unlink(missing_ok=True)
    surrogate = load_or_calibrate_surrogate(
        config.surrogate_cache,
        **config.surrogate_options,
    )
    available = graspqp_available()
    mjwarp_available = True
    if args.mjwarp:
        try:
            import mujoco_warp  # noqa: F401
            import warp as wp

            wp.init()
            mjwarp_available = bool(wp.get_device(args.device).is_cuda)
        except (ImportError, RuntimeError):
            mjwarp_available = False
    print(
        f"[grasp-generation] graspqp={available} "
        f"mjwarp={mjwarp_available if args.mjwarp else 'not-requested'} "
        f"surface_points={surrogate.surface_point_count} cache={config.surrogate_cache}",
        flush=True,
    )
    healthy = available and (mjwarp_available or not args.mjwarp)
    return 0 if healthy or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
