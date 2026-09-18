"""Transfer verified Lift episodes into a stable Stack demonstration."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from source.grasping.executor import ExecutionConfig
from source.grasping.stack_transfer import (
    STACK_TRANSFER_VERSION,
    StackTransferConfig,
    execute_stack_transfer,
)
from source.workflows.batch import atomic_write_json
from tools.grasping.transfer_lift_to_pick_place import (
    discover_lift_manifests,
    load_lift_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lift-manifest", action="append", type=Path, dest="lift_manifests")
    parser.add_argument("--lift-root", action="append", type=Path, dest="lift_roots")
    parser.add_argument("--object-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-attempts", type=int, default=3)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    parser.add_argument("--render-mode", choices=("human", "rgb_array"))
    return parser


def _execution_config(source) -> ExecutionConfig:
    payload = dict(source.metadata.get("execution_config", {}))
    allowed = {item.name for item in fields(ExecutionConfig)}
    return ExecutionConfig(**{key: value for key, value in payload.items() if key in allowed})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = discover_lift_manifests(
        args.lift_manifests, args.lift_roots, object_id=args.object_id
    )
    if not manifests:
        raise ValueError("No successful Lift episode manifests were found.")
    attempts: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    candidates = []
    for manifest in manifests:
        source = load_lift_source(manifest)
        identity = (
            source.candidate.seed_index,
            *source.candidate.hand_translation.round(6),
            *source.candidate.hand_rotation_matrix.reshape(-1).round(6),
            *source.arrays["qpos"][-1].round(5),
        )
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append((manifest, source))
        if len(candidates) >= args.maximum_candidates:
            break
    grip_modes = (
        ("adaptive", True, 0.0, 0.0),
        ("strong_balanced", False, 0.35, 0.35),
        ("balanced_preload", False, 0.15, 0.15),
        ("source_grip", False, 0.0, 0.0),
        ("finger_preload", False, 0.20, 0.0),
        ("thumb_preload", False, 0.0, 0.20),
        ("strong_finger", False, 0.40, 0.0),
        ("strong_thumb", False, 0.0, 0.40),
    )
    direction_offsets = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
    search_modes = [(angle, grip_modes[0]) for angle in direction_offsets] + [
        (angle, grip) for grip in grip_modes[1:] for angle in direction_offsets
    ]
    for seed in range(args.seed_attempts):
        for angle, (mode, adaptive, finger_boost, thumb_boost) in search_modes:
            for candidate_index, (manifest, source) in enumerate(candidates, 1):
                motion = f"audited_a{angle:+04.0f}"
                result = execute_stack_transfer(
                    source,
                    seed=seed,
                    grasp_config=_execution_config(source),
                    transfer_config=StackTransferConfig(
                        adaptive_transport_grip=adaptive,
                        transport_finger_boost=finger_boost,
                        transport_thumb_boost=thumb_boost,
                        support_direction_offset_degrees=angle,
                    ),
                    render_mode=args.render_mode,
                )
                attempt_dir = (
                    args.output / "attempts" / f"c{candidate_index:02d}_s{seed:04d}_{motion}_{mode}"
                )
                attempt_manifest = result.save(attempt_dir)
                attempts.append(
                    {
                        "source_manifest": str(manifest),
                        "seed": seed,
                        "mode": mode,
                        "motion": motion,
                        "support_direction_offset_degrees": angle,
                        "success": result.success,
                        "failure_reason": result.failure_reason,
                        "manifest": str(attempt_manifest),
                    }
                )
                print(
                    f"[stack] candidate={candidate_index:02d} seed={seed:04d} "
                    f"motion={motion} mode={mode} success={result.success}",
                    flush=True,
                )
                if result.success:
                    best = result.save(args.output / "best_trajectory")
                    summary = {
                        "task": "stack",
                        "pipeline_version": STACK_TRANSFER_VERSION,
                        "object_id": source.object_id,
                        "success": True,
                        "best_manifest": str(best),
                        "attempts": attempts,
                    }
                    args.output.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(args.output / "summary.json", summary)
                    return 0
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output / "summary.json",
        {
            "task": "stack",
            "pipeline_version": STACK_TRANSFER_VERSION,
            "object_id": args.object_id,
            "success": False,
            "attempts": attempts,
        },
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
