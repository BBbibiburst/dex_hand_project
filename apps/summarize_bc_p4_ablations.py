"""Aggregate P4 BC action-step and image-resolution ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = []
    for steps in (1, 2, 4):
        path = args.root / f"bc_resolution_fixed_action{steps}" / "summary.json"
        payload = json.loads(path.read_text())
        rows.append(
            {
                "configuration": f"160x120, action_steps={steps}",
                "history_frames": 1,
                "action_steps": steps,
                "contact_fraction": payload["bc"]["contact_fraction"],
                "maximum_lift_m": payload["bc"]["maximum_lift_m"],
                "final_lift_m": payload["bc"]["final_lift_m"],
                "chunk_jump_l2": payload["chunk_boundary_jump_l2_mean"],
                "p50_ms": payload["inference_p50_ms"],
                "p95_ms": payload["inference_p95_ms"],
                "success": payload["strict_success"],
            }
        )
    high = json.loads(
        (args.root / "resolution_320" / "rollout_action4" / "summary.json").read_text()
    )
    rows.append(
        {
            "configuration": "320x240, action_steps=4",
            "history_frames": 1,
            "action_steps": 4,
            "contact_fraction": high["bc"]["contact_fraction"],
            "maximum_lift_m": high["bc"]["maximum_lift_m"],
            "final_lift_m": high["bc"]["final_lift_m"],
            "chunk_jump_l2": high["chunk_boundary_jump_l2_mean"],
            "p50_ms": high["inference_p50_ms"],
            "p95_ms": high["inference_p95_ms"],
            "success": high["strict_success"],
        }
    )
    result = {
        "rows": rows,
        "history_ablation": {
            "executed": False,
            "reason": (
                "C1 found no material observation-action aliasing; action-step and resolution "
                "ablations both failed to produce Lift, so the two-negative-ablation stop rule applied."
            ),
        },
    }
    (args.root / "ablation_summary.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["configuration"] for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, [100 * row["contact_fraction"] for row in rows], color="tab:blue")
    axes[0].set_ylabel("Contact fraction (%)")
    axes[1].bar(x, [1000 * row["maximum_lift_m"] for row in rows], color="tab:orange")
    axes[1].axhline(40, color="tab:red", linestyle="--", label="Lift success threshold")
    axes[1].set_ylabel("Maximum object lift (mm)")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("P4 deterministic BC closed-loop ablations (fixed Sugar Box reset)")
    figure.tight_layout()
    figure.savefig(args.root / "action_steps_resolution_ablation.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
