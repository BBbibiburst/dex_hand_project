"""Evaluate whether selected objects remain still before grasp planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.grasping.affordance import simulate_initial_pose_stability
from source.workflows.batch import atomic_write_json, load_selection_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate objects already present in the output report.",
    )
    args = parser.parse_args(argv)
    object_ids = load_selection_ids(args.selection)
    cached: dict[str, dict] = {}
    if args.output.is_file() and not args.force:
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            float(previous.get("settle_seconds", -1.0)) == args.settle_seconds
            and int(previous.get("seed", -1)) == args.seed
        ):
            cached = {str(row["object_id"]): row for row in previous.get("results", [])}
    results = []
    for index, object_id in enumerate(object_ids, 1):
        row = cached.get(object_id)
        if row is None:
            try:
                stability = simulate_initial_pose_stability(
                    object_id, seed=args.seed, settle_seconds=args.settle_seconds
                )
                row = {"object_id": object_id, **stability.to_dict(), "error": None}
            except Exception as exc:  # Keep catalogue diagnostics complete.
                row = {
                    "object_id": object_id,
                    "stable": False,
                    "settled": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        results.append(row)
        state = "STABLE" if row["stable"] else "UNSTABLE"
        source = "CACHED" if object_id in cached else "MEASURED"
        print(f"[{index:03d}/{len(object_ids):03d}] {object_id:<60} {state} {source}")

    payload = {
        "schema_version": 1,
        "selection": str(args.selection),
        "settle_seconds": args.settle_seconds,
        "seed": args.seed,
        "count": len(results),
        "stable_count": sum(bool(row["stable"]) for row in results),
        "results": results,
    }
    atomic_write_json(args.output, payload)
    print(f"[done] stable={payload['stable_count']}/{payload['count']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
