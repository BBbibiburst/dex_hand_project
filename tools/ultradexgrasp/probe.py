"""Probe the project-native Ultra Prior, Dex Hand surrogate, and optional MJWarp."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from source.ultradexgrasp.catalog import load_object_geometry
from source.ultradexgrasp.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from source.ultradexgrasp.hand_surrogate import load_or_calibrate_surrogate
from source.ultradexgrasp.synthesizer import synthesize_grasps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--object-id", default="ycb:002_master_chef_can")
    parser.add_argument("--device")
    parser.add_argument(
        "--mjwarp",
        action="store_true",
        help="Also require the MuJoCo Warp runtime on the selected device.",
    )
    parser.add_argument(
        "--recalibrate-surrogate",
        action="store_true",
        help="Discard the generated hand surrogate and rebuild it from the current MJCF.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    pipeline = load_pipeline_config(args.config)
    device = (
        args.device
        or pipeline.synthesis.device
        or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    cuda_requested = str(device).startswith("cuda")
    cuda_ok = bool(torch.cuda.is_available())
    print(
        f"[torch] version={torch.__version__} cuda={cuda_ok} requested={device} "
        f"gpu={torch.cuda.get_device_name(0) if cuda_ok else 'none'}",
        flush=True,
    )
    runtime_ok = not cuda_requested or cuda_ok

    if args.recalibrate_surrogate and pipeline.surrogate_cache.is_file():
        pipeline.surrogate_cache.unlink()
        print(f"[surrogate] removed={pipeline.surrogate_cache}", flush=True)
    surrogate = load_or_calibrate_surrogate(
        pipeline.surrogate_cache,
        **pipeline.surrogate_options,
    )
    print(
        f"[surrogate] points={surrogate.surface_point_count} "
        f"rms={1000.0 * surrogate.calibration_rms:.3f}mm",
        flush=True,
    )
    geometry = load_object_geometry(
        args.object_id,
        target_size=pipeline.target_size,
        surface_points=min(pipeline.surface_points, 1024),
        seed=0,
    )
    synthesis = replace(
        pipeline.synthesis,
        seed_count=min(pipeline.synthesis.seed_count, 16),
        optimization_steps=min(pipeline.synthesis.optimization_steps, 80),
        top_k=min(pipeline.synthesis.top_k, 4),
        device=device,
        seed=0,
    )
    candidates = synthesize_grasps(geometry, surrogate, synthesis)
    valid = sum(bool(candidate.metrics.get("valid", 0.0)) for candidate in candidates)
    best = candidates[0].metrics if candidates else {}
    print(f"[native] candidates={len(candidates)} valid={valid} best={best}", flush=True)

    mjwarp_ok = True
    if args.mjwarp:
        try:
            import mujoco_warp  # noqa: F401
            import warp as wp

            wp.init()
            warp_device = wp.get_device(device)
            mjwarp_ok = bool(warp_device.is_cuda)
            print(f"[mjwarp] device={warp_device} cuda={warp_device.is_cuda}", flush=True)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            mjwarp_ok = False
            print(f"[mjwarp] unavailable: {type(exc).__name__}: {exc}", flush=True)

    if args.strict and (not runtime_ok or not mjwarp_ok or valid == 0):
        return 3
    return 0 if candidates else 3


if __name__ == "__main__":
    raise SystemExit(main())
