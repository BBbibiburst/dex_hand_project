"""Aggregate P16 randomness controls and state-conditioned query evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARM_BOUNDARY_LIMIT_RAD = 0.05
HAND_BOUNDARY_LIMIT_M = 0.001
RESTORE_LIMIT = 1e-6


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _randomness_rows(root: Path) -> list[dict]:
    rows = []
    groups = {
        "object_pose": root / "object_pose" / "summary.json",
        "grasp_target": None,
        "close_timing": None,
    }
    payload = _read(groups["object_pose"])
    for row in payload["results"]:
        rows.append({"dimension": "object_pose", **row})
    for path in sorted((root / "grasp_target").glob("*/summary.json")):
        for row in _read(path)["results"]:
            rows.append({"dimension": "grasp_target", "level": path.parent.name, **row})
    for path in sorted((root / "close_timing").glob("*/summary.json")):
        for row in _read(path)["results"]:
            rows.append({"dimension": "close_timing", "level": path.parent.name, **row})
    return rows


def _query_rows(root: Path) -> list[dict]:
    return [_read(path) for path in sorted(root.glob("root_*/*/query_result.json"))]


def _boundary_metrics(manifest: Path, query_step: int) -> tuple[float, float]:
    payload = _read(manifest)
    with np.load(manifest.parent / payload["arrays"]) as archive:
        controls = np.asarray(archive["ctrl"])
    prefix_end = query_step + 1  # frame 0 is the exact pre-action initial state
    return (
        float(np.linalg.norm(controls[prefix_end + 1, :7] - controls[prefix_end, :7])),
        float(np.linalg.norm(controls[prefix_end + 1, 7:13] - controls[prefix_end, 7:13])),
    )


def generation_audit() -> dict:
    return {
        "schema_version": 1,
        "dimensions": [
            {
                "name": "object_initial_pose",
                "entry": "generate_verified_lift_demonstrations.py::perturbations/run",
                "effect": "FixedTablePlacementSampler changes object XY/yaw; candidate remains object-relative, so world grasp and approach transform with the object.",
                "independent_control": True,
                "limitation": "seed also labels/reset scene; yaw can change IK branch and reachability.",
            },
            {
                "name": "grasp_target_and_wrist_pose",
                "entry": "templates.py::edit_candidate_pose/local_wrist_lattice",
                "effect": "Changes object-relative hand translation/rotation, pregrasp direction and contact timing.",
                "independent_control": True,
                "limitation": "compile_key ranks low IK/edit norm first, so automatic selection can collapse to nearby variants.",
            },
            {
                "name": "hand6_target",
                "entry": "executor.py::grasp_hand_targets",
                "effect": "Candidate fractions plus fixed finger/thumb preload define final closure.",
                "independent_control": "candidate-dependent",
                "limitation": "No per-digit timing schedule; all six targets use one linear Close segment.",
            },
            {
                "name": "hand6_close_timing",
                "entry": "executor.py::_run_pose_segment / ExecutionConfig.close_steps",
                "effect": "Changes the rate and duration of the same linear hand target curve.",
                "independent_control": True,
                "limitation": "Does not change close order or target topology.",
            },
            {
                "name": "graspqp_seed",
                "entry": "graspqp_evolve.py -> load_object_geometry/generate_enclosure_seeds/evolve_candidates",
                "effect": "Controls geometry sampling, seed-bank azimuth/hand fractions and DexEvolve mutations; also controls env reset.",
                "independent_control": False,
                "limitation": "One seed couples geometry sampling, optimizer/evolution randomness and scene reset.",
            },
            {
                "name": "candidate_ranking",
                "entry": "rank_candidates_for_scene + templates.py::compile_key",
                "effect": "Filters by IK residual then favors small local edits and source tracking quality.",
                "independent_control": False,
                "limitation": "Can map different random banks to the same high-ranked local topology.",
            },
            {
                "name": "ik_branch",
                "entry": "arm controller _solve_ik / ContinuousCartesianAdapter",
                "effect": "Changes redundant arm joints and can cause reachability failures.",
                "independent_control": False,
                "limitation": "A different arm branch alone is not counted as a different grasp mode.",
            },
        ],
        "deduplication_classes": {
            "SEED_ONLY": "same task-space path, hand curve and contacts; metadata seed differs",
            "OBJECT_POSE_ONLY": "object/root differs while object-relative grasp behavior remains equivalent",
            "IK_BRANCH_ONLY": "joint path differs but wrist/contact behavior is equivalent",
            "LOCAL_VARIANT": "small pose/timing perturbation with unchanged contact topology",
            "DISTINCT_SUCCESSFUL_GRASP": "task-space pose plus contact timing/topology differ and both are FINAL",
            "STATE_CONDITIONED_SUPERVISION": "trajectory contains the real policy prefix and a newly planned FINAL suffix",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    random_rows = _randomness_rows(args.root / "randomness")
    query_rows = _query_rows(args.root / "state_conditioned_query")
    if len(random_rows) != 12 or len(query_rows) != 15:
        raise RuntimeError(
            f"Expected the fixed P16 budgets 12/15, got {len(random_rows)}/{len(query_rows)}."
        )
    by_dimension = []
    for dimension in ("object_pose", "grasp_target", "close_timing"):
        selected = [row for row in random_rows if row["dimension"] == dimension]
        by_dimension.append(
            {
                "dimension": dimension,
                "attempts": len(selected),
                "final_verified": sum(bool(row["strict_success"]) for row in selected),
                "failure_categories": dict(Counter(row["failure_category"] for row in selected)),
                "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in selected])),
            }
        )

    coverage = []
    for row in query_rows:
        arm_jump, hand_jump = _boundary_metrics(Path(row["manifest"]), row["query_step"])
        restored = max(row["restore_error"].values()) <= RESTORE_LIMIT
        compatible = bool(
            restored
            and row["strict_success"]
            and not row["ik_failures"]
            and not row["joint_limit_violations"]
            and arm_jump <= ARM_BOUNDARY_LIMIT_RAD
            and hand_jump <= HAND_BOUNDARY_LIMIT_M
        )
        coverage.append(
            {
                "root_id": row["root_id"],
                "plan": row["plan"],
                "state_distance_max_abs": max(row["restore_error"].values()),
                "boundary_arm_target_jump_l2_rad": arm_jump,
                "boundary_hand_target_jump_l2_m": hand_jump,
                "strict_success": row["strict_success"],
                "label": (
                    "STATE_COVERED_ACTION_COMPATIBLE"
                    if compatible
                    else "STATE_COVERED_ACTION_INCOMPATIBLE"
                    if restored
                    else "NOT_COVERED"
                ),
            }
        )
    root_matrix = []
    for root_id in range(5):
        rows = [row for row in coverage if row["root_id"] == root_id]
        label = (
            "STATE_COVERED_ACTION_COMPATIBLE"
            if any(row["label"] == "STATE_COVERED_ACTION_COMPATIBLE" for row in rows)
            else "STATE_COVERED_ACTION_INCOMPATIBLE"
            if any(row["label"] == "STATE_COVERED_ACTION_INCOMPATIBLE" for row in rows)
            else "NOT_COVERED"
        )
        root_matrix.append({"root_id": root_id, "label": label, "plans": rows})

    audit = generation_audit()
    (args.root / "generation_randomness_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (args.root / "randomness_behavior_matrix.json").write_text(
        json.dumps({"dimensions": by_dimension, "results": random_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.root / "coverage_matrix.json").write_text(
        json.dumps(
            {
                "thresholds": {
                    "restore_max_abs": RESTORE_LIMIT,
                    "boundary_arm_target_jump_l2_rad": ARM_BOUNDARY_LIMIT_RAD,
                    "boundary_hand_target_jump_l2_m": HAND_BOUNDARY_LIMIT_M,
                    "source": "P14 persistent-divergence thresholds; fixed before P16 aggregation",
                },
                "roots": root_matrix,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    compatible_roots = [
        row["root_id"]
        for row in root_matrix
        if row["label"] == "STATE_COVERED_ACTION_COMPATIBLE"
    ]
    summary = {
        "schema_version": 1,
        "randomness_attempts": len(random_rows),
        "randomness_final_verified": sum(row["strict_success"] for row in random_rows),
        "randomness_by_dimension": by_dimension,
        "state_conditioned_attempts": len(query_rows),
        "state_conditioned_final_verified": sum(row["strict_success"] for row in query_rows),
        "state_conditioned_successful_roots": sorted(
            {row["root_id"] for row in query_rows if row["strict_success"]}
        ),
        "compatible_coverage_roots": compatible_roots,
        "coverage_gate_3_of_5": len(compatible_roots) >= 3,
        "dataset_construction_gate": len(compatible_roots) >= 3
        and any(row["strict_success"] for row in query_rows),
        "training_started": False,
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure_dir = args.root / "figures"
    figure_dir.mkdir(exist_ok=True)
    labels = [row["dimension"] for row in by_dimension]
    values = [row["final_verified"] for row in by_dimension]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, values, color="#3977b7")
    ax.set_ylim(0, 4.5)
    ax.set_ylabel("FINAL_VERIFIED / 4")
    ax.set_title("P16 bounded randomness controls")
    for index, value in enumerate(values):
        ax.text(index, value + 0.08, f"{value}/4", ha="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "randomness_success.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    plans = ("xminus", "pitch", "xminus_slow")
    for plan in plans:
        rows = sorted((row for row in coverage if row["plan"] == plan), key=lambda x: x["root_id"])
        ax.plot(
            [row["root_id"] for row in rows],
            [row["boundary_arm_target_jump_l2_rad"] for row in rows],
            marker="o",
            label=plan,
        )
    ax.axhline(ARM_BOUNDARY_LIMIT_RAD, color="#c44e52", linestyle="--", label="0.05 rad limit")
    ax.set_xlabel("P14 root")
    ax.set_ylabel("Boundary arm target jump L2 [rad]")
    ax.set_xticks(range(5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "query_boundary_continuity.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
