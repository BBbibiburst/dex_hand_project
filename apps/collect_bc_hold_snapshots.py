"""Extract complete frozen-BC Hold snapshots without counting them as successes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def snapshot_payload(trace, index: int) -> dict[str, np.ndarray]:
    """Return the complete simulator/controller state needed for exact restore."""
    required = ("bc_qpos", "bc_qvel", "bc_ctrl", "bc_qacc_warmstart")
    missing = [name for name in required if name not in trace]
    if missing:
        raise ValueError(f"BC trace is missing complete snapshot fields: {missing}")
    return {
        "qpos": trace["bc_qpos"][index],
        "qvel": trace["bc_qvel"][index],
        "ctrl": trace["bc_ctrl"][index],
        "qacc_warmstart": trace["bc_qacc_warmstart"][index],
        "fixed_orientation_wxyz": trace["bc_wrist_quat_wxyz"][index],
        "rollout_step": np.asarray(index),
        "tactile_history": trace["bc_tactile"][max(0, index - 7) : index + 1],
        "state_arm_qpos": trace["bc_arm_qpos"][max(0, index - 7) : index + 1],
        "state_arm_ctrl": trace["bc_arm_ctrl"][max(0, index - 7) : index + 1],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=91)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in sorted(args.bc_root.glob("root_*")):
        root_id = int(directory.name.split("_")[-1])
        trace_path = directory / "rollout_trace.npz"
        if not trace_path.is_file():
            continue
        with np.load(trace_path) as trace:
            index = min(args.step, len(trace["bc_qpos"]) - 1)
            snapshot = snapshot_payload(trace, index)
        destination = args.output / f"root_{root_id:02d}.npz"
        np.savez_compressed(destination, **snapshot)
        source = (
            args.source_root / f"p{root_id:02d}" / "candidate_00" / "manifest.json"
        ).resolve()
        payload = json.loads(source.read_text())
        rows.append(
            {
                "root_id": root_id,
                "seed": payload["seed"],
                "source_manifest": str(source),
                "snapshot": str(destination.resolve()),
                "snapshot_step": index,
                "initial_object_position": payload["metadata"]["object_initial_position"],
                "initial_object_quaternion_wxyz": payload["metadata"][
                    "object_initial_quaternion_wxyz"
                ],
                "bc_trace": str(trace_path.resolve()),
                "bc_video": str((directory / "bc_rollout.mp4").resolve()),
            }
        )
    summary = {"schema_version": 1, "count": len(rows), "rows": rows}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
