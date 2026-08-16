"""Physics-validate an optimized teleop trajectory with a final hold test.

Table contact is reported, not rejected: table-assisted finger/object contact is
valid for objects such as bananas and other low-profile shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.teleop.trajectory import TeleopTrajectory, replay_teleop_trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--max-penetration",
        type=float,
        help="Optional hard rejection threshold in metres. Omit to report contact only.",
    )
    args = parser.parse_args(argv)

    trajectory = TeleopTrajectory.load(args.trajectory)
    result = replay_teleop_trajectory(
        trajectory,
        render=args.render,
        realtime=args.realtime,
        hold_seconds=args.hold_seconds,
        compare_reference=False,
    )
    payload = {
        "trajectory": str(args.trajectory),
        "trajectory_kind": trajectory.metadata.get("trajectory_kind"),
        "result": result.to_dict(),
        "validation": {
            "table_contact_policy": "report_only",
            "hold_seconds": args.hold_seconds,
            "max_penetration_limit_m": args.max_penetration,
        },
    }
    penetration_ok = args.max_penetration is None or result.max_penetration <= args.max_penetration
    payload["validation"]["passed"] = bool(result.stable and penetration_ok)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.report)
    return 0 if payload["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
