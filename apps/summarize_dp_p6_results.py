"""Aggregate P6 Joint13/XYZ9/Pose12 evidence and create comparison plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/dp_p6_2026-10-11"))
    parser.add_argument(
        "--joint-summary",
        type=Path,
        default=Path("outputs/dp_p4_2026-10-11/bc_resolution_fixed_action4/summary.json"),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    joint = json.loads(args.joint_summary.read_text())
    xyz = json.loads((args.root / "xyz9_bc/rollout_action4/summary.json").read_text())
    pose = json.loads((args.root / "pose12_bc/rollout_action4/summary.json").read_text())
    rows = [
        {
            "method": "Joint13",
            "dimension": 13,
            "contact_fraction": joint["bc"]["contact_fraction"],
            "max_lift_m": joint["bc"]["maximum_lift_m"],
            "final_lift_m": joint["bc"]["final_lift_m"],
            "strict_success": False,
            "target_delta_l2": joint["bc"]["mean_target_delta_l2"],
            "p95_latency_ms": joint["inference_p95_ms"],
            "ik_failures": None,
        },
        *[
            {
                "method": label,
                "dimension": dimension,
                "contact_fraction": value["bc"]["contact_fraction"],
                "max_lift_m": value["bc"]["maximum_lift_m"],
                "final_lift_m": value["bc"]["final_lift_m"],
                "strict_success": value["strict_replay"]["success"],
                "target_delta_l2": value["bc"]["mean_target_delta_l2"],
                "p95_latency_ms": value["inference_ik_p95_ms"],
                "ik_failures": value["ik_failure_count"],
            }
            for label, dimension, value in (("XYZ9", 9, xyz), ("Pose12", 12, pose))
        ],
    ]
    traces = {
        "XYZ9": np.load(args.root / "xyz9_bc/rollout_action4/rollout_trace.npz"),
        "Pose12": np.load(args.root / "pose12_bc/rollout_action4/rollout_trace.npz"),
    }
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = [row["method"] for row in rows]
    axes[0].bar(labels, [100 * row["contact_fraction"] for row in rows])
    axes[0].set_ylabel("Contact fraction (%)")
    axes[1].bar(labels, [1000 * row["max_lift_m"] for row in rows])
    axes[1].axhline(55, color="red", linestyle="--", label="FINAL 55 mm")
    axes[1].set_ylabel("Maximum lift (mm)")
    axes[1].legend(fontsize=8)
    axes[2].bar(labels, [row["target_delta_l2"] for row in rows])
    axes[2].set_ylabel("Mean Joint13 target change (L2)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_path = args.root / "joint_xyz_pose_bc_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for label, trace in traces.items():
        wrist = trace["bc_wrist"]
        obj = trace["bc_object"]
        axes[0].plot(np.linalg.norm(wrist - obj, axis=1), label=label)
        axes[1].plot(trace["bc_contact"].astype(float), label=label)
    for axis in axes:
        axis.axvspan(88, 92, color="red", alpha=0.12, label="P5 instability onset")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Wrist-object distance (m)")
    axes[1].set_ylabel("Any contact")
    axes[1].set_xlabel("Control step")
    fig.tight_layout()
    close_path = args.root / "close_hold_cartesian_comparison.png"
    fig.savefig(close_path, dpi=180)
    plt.close(fig)
    result = {
        "fixed_reset_single_trajectory_bc": rows,
        "comparison_figure": str(figure_path),
        "close_hold_figure": str(close_path),
        "scope": "one fixed Sugar Box reset; no generalization claim",
    }
    (args.root / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
