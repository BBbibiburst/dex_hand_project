"""Probe the pinned UltraDexGrasp stack and native Dex Hand synthesizer."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from source.ultradexgrasp.catalog import load_object_geometry
from source.ultradexgrasp.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from source.ultradexgrasp.hand_surrogate import load_or_calibrate_surrogate
from source.ultradexgrasp.synthesizer import synthesize_grasps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_PATH = PROJECT_ROOT / "deps" / "ultradexgrasp" / "versions.json"
CHECKOUT_PATHS = {
    "ultradexgrasp": PROJECT_ROOT / "deps" / "ultradexgrasp" / "upstream",
    "bodex_api": PROJECT_ROOT / "deps" / "ultradexgrasp" / "third_party" / "BODex_api",
    "curobo": PROJECT_ROOT / "deps" / "ultradexgrasp" / "third_party" / "curobo",
    "pytorch3d": PROJECT_ROOT / "deps" / "ultradexgrasp" / "third_party" / "pytorch3d",
}


def _revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--object-id", default="ycb:002_master_chef_can")
    parser.add_argument("--device")
    parser.add_argument("--dependency-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    versions = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    dependency_ok = True
    for name, expected in versions.items():
        actual = _revision(CHECKOUT_PATHS[name])
        ok = actual == expected["revision"]
        dependency_ok = dependency_ok and ok
        print(
            f"[dependency] {name}: {'ok' if ok else 'mismatch'} "
            f"expected={expected['revision']} actual={actual}",
            flush=True,
        )
    if args.dependency_only:
        return 0 if dependency_ok else 2

    import torch

    print(
        f"[torch] version={torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    pipeline = load_pipeline_config(args.config)
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
        device=args.device or pipeline.synthesis.device,
        seed=0,
    )
    candidates = synthesize_grasps(geometry, surrogate, synthesis)
    valid = sum(bool(candidate.metrics.get("valid", 0.0)) for candidate in candidates)
    best = candidates[0].metrics if candidates else {}
    print(f"[native] candidates={len(candidates)} valid={valid} best={best}", flush=True)
    if args.strict and (not dependency_ok or valid == 0):
        return 3
    return 0 if candidates else 3


if __name__ == "__main__":
    raise SystemExit(main())
