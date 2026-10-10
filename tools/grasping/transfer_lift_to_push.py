"""Generate one feedback-controlled Push episode from Lift proposals."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from source.grasping.push_transfer import (
    PUSH_TRANSFER_VERSION,
    PushTransferConfig,
    execute_push_transfer,
)
from source.workflows.batch import atomic_write_json
from tools.grasping.transfer_lift_to_pick_place import discover_lift_manifests, load_lift_source
from tools.grasping.transfer_lift_to_stack import _execution_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lift-root", action="append", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    args = parser.parse_args(argv)
    manifests = discover_lift_manifests(None, args.lift_root, object_id=args.object_id)
    attempts = []
    seen = set()
    for index, manifest in enumerate(manifests, 1):
        source = load_lift_source(manifest)
        # Push reuses only the reset object pose, not the Lift grasp. Different
        # grasp candidates with an identical reset pose therefore produce the
        # same deterministic Push episode and must not be rerun.
        identity = (
            *source.metadata["object_initial_position"],
            *source.metadata["object_initial_quaternion_wxyz"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen) > args.maximum_candidates:
            break
        for roll_degrees, height_fraction, minimum_height in (
            (225.0, 0.50, 0.055),
            (210.0, 0.70, 0.075),
            (205.0, 0.70, 0.080),
            (215.0, 0.70, 0.080),
            (225.0, 0.80, 0.090),
            (225.0, 0.85, 0.110),
        ):
            for margin in (0.0, 0.008, 0.012, 0.016):
                try:
                    result = execute_push_transfer(
                        source,
                        grasp_config=_execution_config(source),
                        transfer_config=replace(
                            PushTransferConfig(),
                            post_success_push_distance=margin,
                            finger_plane_roll_degrees=roll_degrees,
                            contact_height_fraction=height_fraction,
                            minimum_contact_height=minimum_height,
                        ),
                    )
                    attempt = (
                        args.output
                        / "attempts"
                        / (
                            f"c{index:02d}_h{round(100 * height_fraction):03d}"
                            f"_r{round(roll_degrees):03d}"
                            f"_m{round(1000 * margin):03d}"
                        )
                    )
                    saved = result.save(attempt)
                    attempts.append(
                        {
                            "source_manifest": str(manifest),
                            "contact_height_fraction": height_fraction,
                            "minimum_contact_height": minimum_height,
                            "finger_plane_roll_degrees": roll_degrees,
                            "post_success_push_distance": margin,
                            "manifest": str(saved),
                            "success": result.success,
                        }
                    )
                    print(
                        f"[push] candidate={index:02d} roll={roll_degrees:.0f} "
                        f"height={height_fraction:.2f} "
                        f"margin={margin:.3f} success={result.success}",
                        flush=True,
                    )
                    if result.success:
                        best = result.save(args.output / "best_trajectory")
                        payload = {
                            "task": "push",
                            "pipeline_version": PUSH_TRANSFER_VERSION,
                            "object_id": source.object_id,
                            "success": True,
                            "best_manifest": str(best),
                            "attempts": attempts,
                        }
                        args.output.mkdir(parents=True, exist_ok=True)
                        atomic_write_json(args.output / "summary.json", payload)
                        return 0
                    if (
                        result.metadata["finger_object_contact_frames"] == 0
                        or result.metadata["hand_table_contact_frames"] > 0
                    ):
                        break
                except ValueError as exc:
                    attempts.append(
                        {
                            "source_manifest": str(manifest),
                            "contact_height_fraction": height_fraction,
                            "minimum_contact_height": minimum_height,
                            "finger_plane_roll_degrees": roll_degrees,
                            "post_success_push_distance": margin,
                            "success": False,
                            "failure_reason": str(exc),
                        }
                    )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output / "summary.json",
        {
            "task": "push",
            "pipeline_version": PUSH_TRANSFER_VERSION,
            "object_id": args.object_id,
            "success": False,
            "attempts": attempts,
        },
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
