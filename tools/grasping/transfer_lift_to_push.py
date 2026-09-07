"""Generate one feedback-controlled Push episode from Lift proposals."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.grasping.push_transfer import PUSH_TRANSFER_VERSION, execute_push_transfer
from tools.grasping.transfer_lift_to_pick_place import discover_lift_manifests, load_lift_source
from tools.grasping.transfer_lift_to_stack import _execution_config
from tools.grasping.batch_pick_place_transfer import _atomic_json


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
        identity = tuple(source.candidate.hand_rotation_matrix.round(6).flat)
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen) > args.maximum_candidates:
            break
        try:
            result = execute_push_transfer(source, grasp_config=_execution_config(source))
            attempt = args.output / "attempts" / f"c{index:02d}"
            saved = result.save(attempt)
            attempts.append(
                {
                    "source_manifest": str(manifest),
                    "manifest": str(saved),
                    "success": result.success,
                }
            )
            print(f"[push] candidate={index:02d} success={result.success}", flush=True)
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
                _atomic_json(args.output / "summary.json", payload)
                return 0
        except ValueError as exc:
            attempts.append(
                {"source_manifest": str(manifest), "success": False, "failure_reason": str(exc)}
            )
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
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
