"""Generate one fixed-nut / grasped-bolt assembly demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.grasping.nut_transfer import (
    NUT_TRANSFER_VERSION,
    NutTransferConfig,
    execute_nut_transfer,
)
from source.workflows.batch import atomic_write_json
from tools.grasping.transfer_lift_to_pick_place import load_lift_source
from tools.grasping.transfer_lift_to_stack import _execution_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle-manifest", type=Path, required=True)
    parser.add_argument("--variant-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    source = load_lift_source(args.handle_manifest)
    attempts: list[dict[str, object]] = []
    for seed in range(args.seed_attempts):
        for mode, corrections, yaw_corrections, verify_steps in (
            ("passive", 0, 0, 100),
            # A seated polygon can briefly rebound after the closed-loop
            # insertion stage. Keep the same physical success thresholds and
            # simply allow the held bolt to settle for another two seconds.
            ("long_settle", 0, 0, 200),
        ):
            result = execute_nut_transfer(
                source,
                variant_index=args.variant_index,
                seed=seed,
                grasp_config=_execution_config(source),
                transfer_config=NutTransferConfig(
                    orientation_corrections=corrections,
                    yaw_corrections=yaw_corrections,
                    verify_steps=verify_steps,
                ),
            )
            manifest = result.save(args.output / "attempts" / f"seed_{seed:04d}_{mode}")
            attempts.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "orientation_corrections": corrections,
                    "yaw_corrections": yaw_corrections,
                    "verify_steps": verify_steps,
                    "success": result.success,
                    "failure_reason": result.failure_reason,
                    "manifest": str(manifest),
                }
            )
            print(
                f"[nut] variant={args.variant_index:03d} seed={seed:04d} "
                f"mode={mode} success={result.success}",
                flush=True,
            )
            if result.success:
                best = result.save(args.output / "best_trajectory")
                payload = {
                    "task": "nut_assembly",
                    "pipeline_version": NUT_TRANSFER_VERSION,
                    "variant_index": args.variant_index,
                    "success": True,
                    "best_manifest": str(best),
                    "attempts": attempts,
                }
                args.output.mkdir(parents=True, exist_ok=True)
                atomic_write_json(args.output / "summary.json", payload)
                return 0
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output / "summary.json",
        {
            "task": "nut_assembly",
            "pipeline_version": NUT_TRANSFER_VERSION,
            "variant_index": args.variant_index,
            "success": False,
            "attempts": attempts,
        },
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
