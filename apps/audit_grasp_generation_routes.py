"""Audit multi-initial-state support and observed cost of two demonstration routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.envs.manipulation import make_lift_env


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--local-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", default="ycb:004_sugar_box")
    parser.add_argument("--seed", type=int, default=8011)
    parser.add_argument("--cases", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    full = json.loads(args.full_run.read_text(encoding="utf-8"))
    local = json.loads(args.local_summary.read_text(encoding="utf-8"))
    local_rows = [row for row in local["results"] if row["object_id"] == args.object_id][
        : args.cases
    ]
    env = make_lift_env(
        task_config={"object_id": args.object_id, "reward_shaping": False},
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
    )
    reset_rows = []
    try:
        for seed in range(args.seed, args.seed + args.cases):
            observation, _ = env.reset(seed=seed)
            reset_rows.append(
                {
                    "seed": seed,
                    "object_position": np.asarray(observation["object_pos"]).tolist(),
                    "object_quaternion_wxyz": np.asarray(observation["object_quat"]).tolist(),
                }
            )
    finally:
        env.close()
    positions = np.asarray([row["object_position"] for row in reset_rows])
    distinct_positions = len({tuple(np.round(value, 8)) for value in positions})
    successes = sum(bool(row["accepted"]) for row in local_rows)
    local_seconds = sum(float(row["runtime_seconds"]) for row in local_rows)
    full_seconds = float(full["timings"]["total"])
    report = {
        "object_id": args.object_id,
        "empirical_seed_resets": reset_rows,
        "distinct_reset_positions": distinct_positions,
        "full_pipeline": {
            "entrypoint": "tools.grasp_generation.graspqp_evolve",
            "supports_object_id": True,
            "supports_seed": True,
            "supports_explicit_initial_pose": False,
            "supports_explicit_xy_yaw": False,
            "seed_changes_environment_reset": distinct_positions == args.cases,
            "candidate_pose_frame": "object-relative; ranked against the reset scene pose",
            "dexevolve_physics_reexecution": True,
            "strict_replay_initial_state_saved": True,
            "safe_batch_output": "only when caller supplies a unique --output per run",
            "observed_reference_runs": 1,
            "observed_successes": int(bool(full["success"])),
            "observed_total_seconds": full_seconds,
            "observed_seconds_per_success": full_seconds if full["success"] else None,
        },
        "local_perturbation": {
            "entrypoint": "apps.generate_verified_lift_demonstrations",
            "supports_explicit_xy_yaw": True,
            "recomputes_world_grasp_pose_from_object_relative_candidate": True,
            "reruns_ik_and_mujoco_execution": True,
            "strict_replay": True,
            "cases": len(local_rows),
            "successes": successes,
            "success_rate": successes / len(local_rows),
            "total_seconds": local_seconds,
            "mean_seconds_per_attempt": local_seconds / len(local_rows),
            "seconds_per_success": local_seconds / successes if successes else None,
            "rows": local_rows,
        },
        "matched_five_pose_comparison_completed": False,
        "matched_comparison_blocker": (
            "The full pipeline CLI has no explicit initial-pose/XY/yaw input, so it cannot be "
            "run on the exact five FixedTablePlacementSampler poses without a new scoped interface."
        ),
        "recommendation": (
            "Hybrid: local verified perturbation is the cost-effective primary augmenter around "
            "known skills; run the full pipeline for new objects or when local retries fail."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
