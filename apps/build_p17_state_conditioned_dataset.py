"""Audit and select one boundary-compatible P16 demonstration per source root."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from apps.collect_generated_lerobot import recorded_pre_action_indices
from source.envs.manipulation import make_lift_env
from source.grasping.contracts import DemonstrationEpisode
from source.verification.strict_replay import load_replay_controls


SPLITS = {"train": (0, 1, 2), "validation": (3,), "test": (4,)}


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def _boundary(arrays: dict[str, np.ndarray], query_step: int) -> dict[str, float]:
    end = query_step + 1
    ctrl = np.asarray(arrays["ctrl"])
    return {
        "arm_l2_rad": float(np.linalg.norm(ctrl[end + 1, :7] - ctrl[end, :7])),
        "hand_l2_m": float(np.linalg.norm(ctrl[end + 1, 7:13] - ctrl[end, 7:13])),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p16-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    coverage = json.loads((args.p16_root / "coverage_matrix.json").read_text())
    compatible = {
        (int(root["root_id"]), plan["plan"])
        for root in coverage["roots"]
        for plan in root["plans"]
        if plan["label"] == "STATE_COVERED_ACTION_COMPATIBLE"
    }
    rows, excluded = [], []
    for result_path in sorted(
        (args.p16_root / "state_conditioned_query").glob("root_*/*/query_result.json")
    ):
        result = json.loads(result_path.read_text())
        root_id, plan = int(result["root_id"]), str(result["plan"])
        manifest = Path(result["manifest"])
        episode = DemonstrationEpisode.load(manifest)
        boundary = _boundary(episode.arrays, int(result["query_step"]))
        eligible = bool(
            result["strict_success"]
            and (root_id, plan) in compatible
            and result["ik_failures"] == 0
            and result["joint_limit_violations"] == 0
            and max(result["restore_error"].values()) <= 1e-6
        )
        record = {
            "root_id": root_id,
            "plan": plan,
            "manifest": str(manifest.resolve()),
            "source_manifest": result["source_manifest"],
            "seed": episode.seed,
            "query_step": result["query_step"],
            "frames": len(episode.arrays["ctrl"]),
            "strict_success": result["strict_success"],
            "boundary": boundary,
            "restore_max_abs": max(result["restore_error"].values()),
            "action_hash": _hash(episode.arrays["ctrl"]),
            "grasp_mode": "mode_xminus" if plan.startswith("xminus") else "mode_pitch",
            "planning_method": "state_conditioned_object_relative_cartesian_replan",
            "eligible": eligible,
        }
        if eligible and plan == "xminus":
            rows.append(record)
        else:
            record["exclusion_reason"] = (
                "ONE_EPISODE_PER_ROOT_SLOW_VARIANT"
                if eligible and plan == "xminus_slow"
                else "BOUNDARY_ACTION_INCOMPATIBLE"
                if result["strict_success"]
                else "NOT_FINAL_VERIFIED"
            )
            excluded.append(record)
    if sorted(row["root_id"] for row in rows) != list(range(5)):
        raise RuntimeError("Expected exactly one compatible xminus episode for each P14 root.")

    root_to_split = {root: split for split, roots in SPLITS.items() for root in roots}
    for row in rows:
        row["split"] = root_to_split[row["root_id"]]
        episode = DemonstrationEpisode.load(row["manifest"])
        row["stage_frames"] = {
            str(code): int(count)
            for code, count in zip(*np.unique(episode.arrays["stage"], return_counts=True))
        }
        payload = json.loads(Path(row["manifest"]).read_text())
        env = make_lift_env(
            task_config={"object_id": payload["object_id"], "terminate_on_success": False},
            control_mode="position",
            enable_tactile_sensors=True,
            render_mode=None,
            episode_length=row["frames"] + 20,
        )
        try:
            controls, _, _, _ = load_replay_controls(row["manifest"], env)
            indices = recorded_pre_action_indices(episode.arrays, controls)
        finally:
            env.close()
        row["pre_action_index_first"] = int(indices[0])
        row["pre_action_index_last"] = int(indices[-1])
        row["export_frames"] = len(indices)

    split_roots = {split: list(roots) for split, roots in SPLITS.items()}
    flattened = [root for roots in split_roots.values() for root in roots]
    leakage_free = len(flattened) == len(set(flattened)) == 5
    hashes = [row["action_hash"] for row in rows]
    stage_totals = Counter()
    for row in rows:
        stage_totals.update(row["stage_frames"])
    summary = {
        "schema_version": 1,
        "selected_episodes": len(rows),
        "selected_frames_recorded": sum(row["frames"] for row in rows),
        "selected_frames_exported": sum(row["export_frames"] for row in rows),
        "independent_roots": len({row["root_id"] for row in rows}),
        "episodes_per_root": dict(Counter(str(row["root_id"]) for row in rows)),
        "episodes_per_mode": dict(Counter(row["grasp_mode"] for row in rows)),
        "stage_frames": dict(stage_totals),
        "action_dim": 13,
        "state_dim": 39,
        "pre_action_alignment": all(row["pre_action_index_first"] == 0 for row in rows),
        "duplicate_action_hashes": len(hashes) - len(set(hashes)),
        "boundary_arm_jump_rad": [row["boundary"]["arm_l2_rad"] for row in rows],
        "boundary_hand_jump_m": [row["boundary"]["hand_l2_m"] for row in rows],
        "split_roots": split_roots,
        "root_leakage_free": leakage_free,
        "test_root_count": len(SPLITS["test"]),
        "sample_size_warning": "Only one held-out test root; results are exploratory, not a generalization estimate.",
        "rows": rows,
        "excluded": excluded,
    }
    if not leakage_free or not summary["pre_action_alignment"]:
        raise RuntimeError("P17 data quality gate failed.")
    (args.output / "state_conditioned_manifest.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "split.json").write_text(
        json.dumps({"schema_version": 1, "roots": split_roots}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "data_quality_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"rows", "excluded"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
