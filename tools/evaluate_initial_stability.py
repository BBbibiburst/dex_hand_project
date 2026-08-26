"""Evaluate whether selected objects remain still before grasp planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.grasping.affordance import simulate_initial_pose_stability


def _selection_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("objects", payload) if isinstance(payload, dict) else payload
    return tuple(
        str(value["object_id"] if isinstance(value, dict) else value) for value in values
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    object_ids = _selection_ids(args.selection)
    results = []
    for index, object_id in enumerate(object_ids, 1):
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
        print(f"[{index:03d}/{len(object_ids):03d}] {object_id:<60} {state}")

    payload = {
        "schema_version": 1,
        "selection": str(args.selection),
        "settle_seconds": args.settle_seconds,
        "seed": args.seed,
        "count": len(results),
        "stable_count": sum(bool(row["stable"]) for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] stable={payload['stable_count']}/{payload['count']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
