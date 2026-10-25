"""Create compact P5 causal, recovery, and failure-event figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import matplotlib.pyplot as plt

    cross = json.loads((args.root / "cross_control/cross_control_summary.json").read_text())
    snapshots = ("close_end", "hold_start", "hold_mid", "lift_start")
    combinations = (
        "expert_arm_expert_hand",
        "expert_arm_bc_hand",
        "bc_arm_expert_hand",
        "bc_arm_bc_hand",
    )
    values = {(row["snapshot"], row["combination"]): row for row in cross["rows"]}
    x = np.arange(len(snapshots))
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.2
    for offset, combination in enumerate(combinations):
        axis.bar(
            x + (offset - 1.5) * width,
            [1000 * values[(snapshot, combination)]["maximum_lift_m"] for snapshot in snapshots],
            width,
            label=combination.replace("_", " "),
        )
    axis.axhline(55, color="red", linestyle="--", label="strict lift threshold")
    axis.set_xticks(x, snapshots)
    axis.set_ylabel("Maximum lift (mm)")
    axis.set_title("Arm/hand cross-control causal intervention")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.root / "cross_control/cross_control_lift.png", dpi=180)
    plt.close(figure)

    recovery = json.loads((args.root / "recovery/summary.json").read_text())
    source_indices = recovery["source_indices"]
    divergence = recovery["divergence_steps"]
    success = np.zeros((len(source_indices), len(divergence)))
    for row in recovery["rows"]:
        i = source_indices.index(row["source_index"])
        j = divergence.index(row["divergence_steps"])
        success[i, j] = row["verification_status"].startswith("LOCAL_RECOVERY_VERIFIED")
    figure, axis = plt.subplots(figsize=(9, 3.8))
    image = axis.imshow(success, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(divergence)), divergence)
    axis.set_yticks(range(len(source_indices)), [f"source {value}" for value in source_indices])
    axis.set_xlabel("BC divergence steps before expert recovery")
    axis.set_title("Locally verified recovery region (1=pass, 0=reject)")
    figure.colorbar(image, ax=axis, ticks=(0, 1))
    figure.tight_layout()
    figure.savefig(args.root / "recovery/recoverability_heatmap.png", dpi=180)
    plt.close(figure)

    with np.load(args.root / "baseline_detailed/rollout_trace.npz", allow_pickle=False) as data:
        digit = data["bc_digit_contact_count"] > 0
        tactile = data["bc_tactile"]
        arm_error = np.linalg.norm(data["bc_arm_target"] - data["expert_arm_target"], axis=1)
        relative = data["bc_object"] - data["bc_wrist"]
        slip = np.linalg.norm(np.diff(relative, axis=0, prepend=relative[:1]), axis=1)
    window = slice(75, 111)
    steps = np.arange(340)[window]
    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(steps, digit[window].astype(int))
    axes[0].set_ylabel("Digit contact")
    axes[0].legend([f"digit {i}" for i in range(5)], ncol=5, fontsize=8)
    axes[1].plot(steps, tactile[window].sum(axis=1), label="total tactile")
    axes[1].set_ylabel("Taxel sum")
    axes[2].plot(steps, arm_error[window], label="arm target error", color="tab:red")
    axes[2].axhline(0.05, color="grey", linestyle="--")
    axes[2].set_ylabel("Arm error L2")
    axes[3].plot(steps, 1000 * slip[window], label="relative slip", color="tab:orange")
    axes[3].set_ylabel("Slip (mm/step)")
    axes[3].set_xlabel("Control step")
    for axis in axes:
        axis.axvline(88, color="purple", linestyle="--", label="contact loss 88")
        axis.axvline(92, color="black", linestyle=":", label="arm error 92")
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    figure.suptitle(
        "Close/Hold failure event: physical contact loss precedes arm divergence threshold"
    )
    figure.tight_layout()
    figure.savefig(args.root / "baseline_detailed/step_92_failure_zoom.png", dpi=180)
    plt.close(figure)

    import imageio.v2 as imageio

    reader = imageio.get_reader(args.root / "baseline_detailed/bc_rollout.mp4")
    selected = (82, 88, 92, 100)
    frames = [reader.get_data(index) for index in selected]
    reader.close()
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for axis, frame, index in zip(axes, frames, selected, strict=True):
        axis.imshow(frame)
        axis.set_title(f"step {index}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.root / "baseline_detailed/failure_event_frames.png", dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
