"""Aggregate P7 recoverability and residual ablation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/dp_p7_2026-10-11"))
    args = parser.parse_args(argv)
    recovery = json.loads((args.root / "recoverability/summary.json").read_text())
    rows = recovery["rows"]
    by_variant = {}
    for row in rows:
        item = by_variant.setdefault(row["variant"], {"attempts": 0, "successes": 0})
        item["attempts"] += 1
        item["successes"] += int(row["strict_replay"]["success"])
    snapshot_classification = {}
    for snapshot in recovery["snapshots"]:
        selected = [row for row in rows if row["snapshot"] == snapshot]
        expert = [row for row in selected if row["variant"] == "expert_xyz_bc_hand"][0]
        residual_success = any(
            row["strict_replay"]["success"] for row in selected if row["variant"].startswith("res_")
        )
        if expert["strict_replay"]["success"]:
            category = "nominal-replay recoverable"
        elif residual_success:
            category = "local-replan recoverable"
        else:
            category = "not recovered within tested budget"
        snapshot_classification[snapshot] = category
    ablation = []
    for name in ("baseline", "fixed_rule", "state_residual", "tactile_residual"):
        row = json.loads((args.root / f"ablation/{name}/summary.json").read_text())
        strict = row["strict_replay"]
        ablation.append(
            {
                "variant": name,
                "trigger_step": row["trigger_step"],
                "uses_tactile": row["uses_tactile"],
                "strict_success": strict["success"],
                "max_lift_m": strict["max_lift"],
                "final_lift_m": strict["final_lift"],
                "tail_grasp_fraction": strict["tail_grasp_fraction"],
                "p95_ms": row["control_p95_ms"],
            }
        )
    result = {
        "recoverability_attempts": len(rows),
        "recoverability_strict_successes": sum(row["strict_replay"]["success"] for row in rows),
        "by_variant": by_variant,
        "snapshot_classification": snapshot_classification,
        "full_rollout_ablation": ablation,
    }
    (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [row["variant"] for row in ablation]
    axes[0].bar(labels, [1000 * row["max_lift_m"] for row in ablation])
    axes[0].axhline(55, color="red", linestyle="--", label="FINAL 55 mm")
    axes[0].set_ylabel("Maximum lift (mm)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()
    axes[1].bar(labels, [100 * row["tail_grasp_fraction"] for row in ablation])
    axes[1].set_ylabel("Tail grasp fraction (%)")
    axes[1].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.root / "residual_ablation.png", dpi=180)
    plt.close(fig)
    variants = list(by_variant)
    snapshots = list(recovery["snapshots"])
    matrix = np.zeros((len(snapshots), len(variants)))
    for row in rows:
        matrix[snapshots.index(row["snapshot"]), variants.index(row["variant"])] = row[
            "strict_replay"
        ]["success"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(variants)), variants, rotation=35, ha="right")
    ax.set_yticks(range(len(snapshots)), snapshots)
    ax.set_title("Strict recovery success from real XYZ9 BC states")
    fig.colorbar(image, ax=ax, label="FINAL success")
    fig.tight_layout()
    fig.savefig(args.root / "recoverability_matrix.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
