"""Create stage-level expert/BC diagnostics from an exact fixed-reset rollout trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.grasping.contracts import DemonstrationEpisode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    episode = DemonstrationEpisode.load(args.manifest)
    with np.load(args.trace, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    length = len(arrays["bc_object"])
    source_start = len(episode.arrays["stage"]) - length
    stage = np.asarray(episode.arrays["stage"])[source_start : source_start + length]
    names = {int(value): name for name, value in episode.metadata["stage_codes"].items()}
    rows = []
    for code in np.unique(stage):
        mask = stage == code
        first = int(np.flatnonzero(mask)[0])
        last = int(np.flatnonzero(mask)[-1])
        row = {"stage": names[int(code)], "first_step": first, "last_step": last}
        for prefix in ("expert", "bc"):
            obj = arrays[f"{prefix}_object"]
            wrist = arrays[f"{prefix}_wrist"]
            contact = arrays[f"{prefix}_contact"].astype(bool)
            relative = obj - wrist
            relative_motion = np.linalg.norm(
                np.diff(relative, axis=0, prepend=relative[:1]), axis=1
            )
            row[prefix] = {
                "contact_fraction": float(contact[mask].mean()),
                "mean_contact_force": float(arrays[f"{prefix}_contact_force"][mask].mean()),
                "maximum_object_lift_m": float(obj[mask, 2].max() - obj[0, 2]),
                "final_object_lift_m": float(obj[last, 2] - obj[0, 2]),
                "minimum_wrist_distance_m": float(arrays[f"{prefix}_wrist_distance"][mask].min()),
                "mean_relative_motion_m": float(relative_motion[mask].mean()),
                "mean_tactile_sum": float(arrays[f"{prefix}_tactile_sum"][mask].mean()),
                "digit_contact_fraction": (
                    (arrays[f"{prefix}_digit_contact_count"][mask] > 0).mean(axis=0).tolist()
                ),
            }
        rows.append(row)

    # Earliest sustained deviation in commanded arm targets relative to expert.
    target_error = np.linalg.norm(arrays["bc_arm_target"] - arrays["expert_arm_target"], axis=1)
    qpos_error = np.linalg.norm(arrays["bc_arm_qpos"] - arrays["expert_arm_qpos"], axis=1)
    result = {
        "manifest": str(args.manifest),
        "trace": str(args.trace),
        "source_stage_start_index": source_start,
        "first_target_error_gt_0_05_step": (
            int(np.flatnonzero(target_error > 0.05)[0]) if np.any(target_error > 0.05) else None
        ),
        "first_qpos_error_gt_0_05_step": (
            int(np.flatnonzero(qpos_error > 0.05)[0]) if np.any(qpos_error > 0.05) else None
        ),
        "target_error_rmse": float(
            np.sqrt(np.mean((arrays["bc_arm_target"] - arrays["expert_arm_target"]) ** 2))
        ),
        "stages": rows,
    }
    (args.output / "stage_failure_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    boundaries = [int(np.flatnonzero(stage == code)[0]) for code in np.unique(stage)]
    labels = [names[int(code)] for code in np.unique(stage)]
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for prefix, label in (("expert", "Expert"), ("bc", "BC")):
        axes[0].plot(arrays[f"{prefix}_wrist"][:, 2], label=f"{label} wrist")
        axes[0].plot(arrays[f"{prefix}_object"][:, 2], label=f"{label} object", linestyle="--")
        axes[1].plot(arrays[f"{prefix}_contact"].astype(float), label=f"{label} contact")
        axes[1].plot(
            arrays[f"{prefix}_digit_contact_count"].sum(axis=1) / 5.0,
            label=f"{label} digit count / 5",
            alpha=0.8,
        )
        relative = arrays[f"{prefix}_object"] - arrays[f"{prefix}_wrist"]
        axes[2].plot(np.linalg.norm(relative - relative[0], axis=1), label=label)
    axes[0].set_ylabel("Height (m)")
    axes[1].set_ylabel("Contact")
    axes[2].set_ylabel("Relative object-wrist change (m)")
    axes[2].set_xlabel("Control step")
    for axis in axes:
        for boundary, label in zip(boundaries, labels, strict=True):
            axis.axvline(boundary, color="grey", alpha=0.25, linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend(ncol=2)
    axes[0].set_title("Expert vs BC stage diagnostics")
    top = axes[0].twiny()
    top.set_xlim(axes[0].get_xlim())
    top.set_xticks(boundaries, labels, rotation=30, ha="left")
    figure.tight_layout()
    figure.savefig(args.output / "height_contact_slip_by_stage.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    for joint in range(7):
        axis = axes.flat[joint]
        axis.plot(arrays["expert_arm_target"][:, joint], label="Expert")
        axis.plot(arrays["bc_arm_target"][:, joint], label="BC", alpha=0.85)
        for boundary in boundaries:
            axis.axvline(boundary, color="grey", alpha=0.15, linewidth=0.7)
        axis.set_title(f"Arm target joint {joint + 1}")
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(args.output / "expert_bc_joint_targets_by_stage.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
