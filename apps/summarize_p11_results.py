"""Aggregate P11 training, rollout, grasp-quality, and latency evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path):
    return json.loads(Path(path).read_text())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    figures = args.root / "figures"
    figures.mkdir(exist_ok=True)
    methods = {}
    for method in ("global", "global_roi"):
        rows = []
        for root_id in (8, 9):
            directory = args.root / "evaluation" / method / f"root_{root_id:02d}"
            rollout = _read(directory / "summary.json")
            quality = _read(directory / "grasp_quality/summary.json")
            rows.append(
                {
                    "root_id": root_id,
                    "contact_fraction": rollout["bc"]["contact_fraction"],
                    "multi_finger_fraction": quality["multi_finger_fraction"],
                    "thumb_opposition_fraction": quality["thumb_opposition_fraction"],
                    "object_following_ratio": quality["object_following_ratio"],
                    "maximum_lift_mm": 1000 * rollout["bc"]["maximum_lift_m"],
                    "final_lift_mm": 1000 * rollout["bc"]["final_lift_m"],
                    "strict_success": rollout["strict_replay"]["success"],
                    "model_p50_ms": rollout["model_p50_ms"],
                    "model_p95_ms": rollout["model_p95_ms"],
                    "end_to_end_p50_ms": rollout["observation_to_action_p50_ms"],
                    "end_to_end_p95_ms": rollout["observation_to_action_p95_ms"],
                }
            )
        numeric = [key for key in rows[0] if key not in ("root_id", "strict_success")]
        aggregate = {key: float(np.mean([row[key] for row in rows])) for key in numeric}
        aggregate.update(
            {
                "attempts": len(rows),
                "strict_successes": sum(row["strict_success"] for row in rows),
            }
        )
        methods[method] = {"test_roots": rows, "aggregate": aggregate}
    snapshots = {}
    for method in ("global", "global_roi"):
        directory = args.root / "evaluation_snapshot" / method / "root_00"
        rollout = _read(directory / "summary.json")
        quality = _read(directory / "grasp_quality/summary.json")
        snapshots[method] = {
            "contact_fraction": rollout["bc"]["contact_fraction"],
            "multi_finger_fraction": quality["multi_finger_fraction"],
            "thumb_opposition_fraction": quality["thumb_opposition_fraction"],
            "lift_object_following_ratio": quality["object_following_ratio"],
            "maximum_lift_mm": 1000 * rollout["bc"]["maximum_lift_m"],
            "final_lift_mm": 1000 * rollout["bc"]["final_lift_m"],
            "strict_success": rollout["strict_replay"]["success"],
            "snapshot_restore": rollout["snapshot_restore"],
        }
    training = {
        method: _read(args.root / "training" / method / "train_summary.json")
        for method in ("global", "global_roi")
    }
    summary = {
        "scope": "one fixed-seed model per method; two independent held-out full rollouts and one paired P9 Close snapshot",
        "training": {
            method: {
                key: training[method][key]
                for key in (
                    "parameters",
                    "trainable_parameters",
                    "checkpoint_bytes",
                    "best_epoch",
                    "best_validation_loss",
                    "runtime_seconds",
                )
            }
            for method in training
        },
        "full_rollouts": methods,
        "paired_close_snapshot": snapshots,
        "conclusion": (
            "ROI increased contact prevalence but did not improve load-bearing object following "
            "or FINAL verification; tactile expansion gate is not met."
        ),
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    labels = ("Contact", "Multi-finger", "Thumb opposition")
    keys = ("contact_fraction", "multi_finger_fraction", "thumb_opposition_fraction")
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for offset, method in zip((-0.2, 0.2), ("global", "global_roi"), strict=True):
        aggregate = methods[method]["aggregate"]
        axes[0].bar(x + offset, [100 * aggregate[key] for key in keys], 0.4, label=method)
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylabel("Held-out fraction (%)")
    axes[0].legend(frameon=False)
    width = 0.35
    for offset, method in zip((-width / 2, width / 2), ("global", "global_roi"), strict=True):
        values = [row["maximum_lift_mm"] for row in methods[method]["test_roots"]]
        axes[1].bar(np.arange(2) + offset, values, width, label=method)
    axes[1].set_xticks((0, 1), ("root 8", "root 9"))
    axes[1].set_ylabel("Maximum lift (mm)")
    axes[1].axhline(55, color="black", linestyle="--", linewidth=1, label="55 mm threshold")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "heldout_contact_and_lift.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = ("Global", "Global+ROI")
    ax.bar(
        np.arange(2) - 0.18,
        [100 * snapshots[key]["contact_fraction"] for key in ("global", "global_roi")],
        0.36,
        label="contact (%)",
    )
    ax.bar(
        np.arange(2) + 0.18,
        [100 * snapshots[key]["lift_object_following_ratio"] for key in ("global", "global_roi")],
        0.36,
        label="Lift object-following (%)",
    )
    ax.set_xticks((0, 1), names)
    ax.set_title("Same restored P9 Close snapshot")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "snapshot_contact_vs_following.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
