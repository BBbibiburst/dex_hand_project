"""Analyze multi-finger grasp quality of the frozen P6 XYZ9 rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from apps.trace_diffusion_rollout import _make_env
from source.geometry import normalize_quat, quat_conjugate, quat_multiply, quat_to_rotvec
from source.grasping.executor import _contact_digit


def _longest(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _relative_rotation(wrist, obj):
    return quat_multiply(quat_conjugate(normalize_quat(wrist)), normalize_quat(obj))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-step-offset",
        type=int,
        default=0,
        help="Original rollout step represented by trace step zero (for restored snapshots).",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    trace = np.load(args.trace)
    payload = json.loads(args.manifest.read_text())
    env = _make_env(payload, 400)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    binding = env.task._require_bindings().objects["object"]
    object_ids = set(binding.geom_ids)
    contact_points = []
    per_step_points = []
    try:
        for qpos, qvel, ctrl in zip(trace["bc_qpos"], trace["bc_qvel"], trace["bc_ctrl"]):
            env.data.qpos[:] = qpos
            env.data.qvel[:] = qvel
            env.data.ctrl[:] = ctrl
            mujoco.mj_forward(env.model, env.data)
            points = []
            for index in range(env.data.ncon):
                contact = env.data.contact[index]
                if int(contact.geom1) in object_ids or int(contact.geom2) in object_ids:
                    other = (
                        int(contact.geom2)
                        if int(contact.geom1) in object_ids
                        else int(contact.geom1)
                    )
                    digit = _contact_digit(env, other)
                    if digit >= 0:
                        point = np.r_[digit, np.asarray(contact.pos)]
                        points.append(point)
                        contact_points.append(point)
            per_step_points.append(len(points))
    finally:
        env.close()
    digit = trace["bc_digit_contact_count"] > 0
    multi = digit.sum(axis=1) >= 3
    opposing = digit[:, 4] & (digit[:, :4].sum(axis=1) >= 2)
    wrist = trace["bc_wrist"]
    obj = trace["bc_object"]
    relative = obj - wrist
    relative_translation_slip = np.linalg.norm(relative - relative[0], axis=1)
    relative_quat = np.asarray(
        [
            _relative_rotation(wrist_q, object_q)
            for wrist_q, object_q in zip(trace["bc_wrist_quat_wxyz"], trace["bc_object_quat_wxyz"])
        ]
    )
    relative_rotation_slip = np.asarray(
        [
            np.linalg.norm(quat_to_rotvec(quat_multiply(q, quat_conjugate(relative_quat[0]))))
            for q in relative_quat
        ]
    )
    original_stage_ranges = {
        "close_early": (46, 70),
        "close_late": (70, 91),
        "hold_start": (91, 101),
        "hold_mid": (101, 131),
        "lift": (131, 221),
        "verify": (221, 340),
    }
    stage_ranges = {
        name: (
            max(0, start - args.source_step_offset),
            min(len(digit), end - args.source_step_offset),
        )
        for name, (start, end) in original_stage_ranges.items()
        if end > args.source_step_offset and start - args.source_step_offset < len(digit)
    }
    stages = {}
    for name, (start, end) in stage_ranges.items():
        if end <= start:
            continue
        stages[name] = {
            "steps": end - start,
            "any_contact_fraction": float(trace["bc_contact"][start:end].mean()),
            "multi_finger_fraction": float(multi[start:end].mean()),
            "thumb_opposition_fraction": float(opposing[start:end].mean()),
            "digit_contact_fraction": digit[start:end].mean(axis=0).tolist(),
            "mean_contact_force": float(trace["bc_contact_force"][start:end].mean()),
            "translation_slip_change_m": float(
                relative_translation_slip[end - 1] - relative_translation_slip[start]
            ),
        }
    lift_start, lift_end = stage_ranges.get("lift", (0, len(digit)))
    lift = slice(lift_start, lift_end)
    wrist_rise = float(wrist[lift][-1, 2] - wrist[lift][0, 2])
    object_rise = float(obj[lift][-1, 2] - obj[lift][0, 2])
    result = {
        "steps": len(digit),
        "any_contact_fraction": float(trace["bc_contact"].mean()),
        "multi_finger_fraction": float(multi.mean()),
        "thumb_opposition_fraction": float(opposing.mean()),
        "longest_multi_finger_steps": _longest(multi),
        "longest_thumb_opposition_steps": _longest(opposing),
        "digit_contact_fraction": digit.mean(axis=0).tolist(),
        "maximum_relative_translation_slip_m": float(relative_translation_slip.max()),
        "maximum_relative_rotation_slip_rad": float(relative_rotation_slip.max()),
        "lift_wrist_rise_m": wrist_rise,
        "lift_object_rise_m": object_rise,
        "object_following_ratio": object_rise / wrist_rise if abs(wrist_rise) > 1e-8 else None,
        "contact_points": len(contact_points),
        "stage_ranges": stage_ranges,
        "stages": stages,
        "note": "Diagnostic metrics only; original FINAL profile remains the success criterion.",
    }
    np.savez_compressed(
        args.output / "grasp_quality_trace.npz",
        digit_contact=digit,
        multi_finger=multi,
        thumb_opposition=opposing,
        relative_translation=relative,
        relative_translation_slip=relative_translation_slip,
        relative_rotation_slip=relative_rotation_slip,
        contact_points=np.asarray(contact_points),
        contact_point_count=np.asarray(per_step_points),
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for index in range(5):
        axes[0].plot(digit[:, index].astype(float) + 1.1 * index, label=f"digit {index}")
    axes[0].plot(opposing.astype(float) - 1.2, label="thumb opposition", linewidth=2)
    axes[1].plot(1000 * relative_translation_slip, label="translation slip")
    axes[1].plot(1000 * relative_rotation_slip, label="rotation slip (mrad)")
    axes[2].plot(1000 * (wrist[:, 2] - wrist[0, 2]), label="wrist rise")
    axes[2].plot(1000 * (obj[:, 2] - obj[0, 2]), label="object rise")
    for axis in axes:
        for boundary in (46, 91, 131, 221):
            shifted = boundary - args.source_step_offset
            if 0 <= shifted < len(digit):
                axis.axvline(shifted, color="grey", alpha=0.25)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    axes[1].set_ylabel("Slip (mm / mrad)")
    axes[2].set_ylabel("Rise (mm)")
    axes[2].set_xlabel("Control step")
    fig.tight_layout()
    fig.savefig(args.output / "contact_slip_following.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
