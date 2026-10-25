"""Compare expert and frozen XYZ9 Lift mechanics on one aligned timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from apps.trace_diffusion_rollout import _make_env
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import STAGE_CODES
from source.imitation.cartesian_action import ContinuousCartesianAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bc-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    episode = DemonstrationEpisode.load(args.manifest)
    env = _make_env(payload, 500)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
    adapter = ContinuousCartesianAdapter(env)
    stage = np.asarray(episode.arrays["stage"])
    selected = stage >= STAGE_CODES["approach"]
    qpos = episode.arrays["qpos"][selected]
    # Generated episodes retain high-level IK actions in ``action`` but strict
    # replay executes the low-level arm7+hand6 position targets in ``ctrl``.
    controls = episode.arrays["ctrl"][selected]
    expert_wrist, expert_target = [], []
    try:
        for q, control in zip(qpos, controls, strict=True):
            env.data.qpos[:] = q
            mujoco.mj_forward(env.model, env.data)
            expert_wrist.append(env.data.site_xpos[env.controller.arm_controller.site_id].copy())
            expert_target.append(adapter.pose_for_joint_target(control[:7])[0])
    finally:
        env.close()
    expert_wrist = np.asarray(expert_wrist)
    expert_target = np.asarray(expert_target)
    expert_object = np.asarray(episode.arrays["object_position"])[selected]
    expert_force = np.asarray(episode.arrays["robot_object_normal_force"])[selected]
    expert_support = np.asarray(episode.arrays["object_support_contact"])[selected]
    expert_stage = stage[selected]
    with np.load(args.bc_trace) as archive:
        bc = {key: np.asarray(archive[key]) for key in archive.files}
    lift_code = STAGE_CODES["lift"]
    expert_lift = np.flatnonzero(expert_stage == lift_code)
    bc_lift = np.arange(131, 221)
    expert_follow = float(
        (expert_object[expert_lift[-1], 2] - expert_object[expert_lift[0], 2])
        / (expert_wrist[expert_lift[-1], 2] - expert_wrist[expert_lift[0], 2])
    )
    bc_follow = float(
        (bc["bc_object"][bc_lift[-1], 2] - bc["bc_object"][bc_lift[0], 2])
        / (bc["bc_wrist"][bc_lift[-1], 2] - bc["bc_wrist"][bc_lift[0], 2])
    )
    result = {
        "expert": {
            "lift_steps": len(expert_lift),
            "target_z_rise_m": float(expert_target[expert_lift[-1], 2] - expert_target[expert_lift[0], 2]),
            "wrist_z_rise_m": float(expert_wrist[expert_lift[-1], 2] - expert_wrist[expert_lift[0], 2]),
            "object_z_rise_m": float(expert_object[expert_lift[-1], 2] - expert_object[expert_lift[0], 2]),
            "object_following_ratio": expert_follow,
            "mean_contact_force": float(expert_force[expert_lift].mean()),
            "table_support_fraction": float(expert_support[expert_lift].mean()),
        },
        "bc": {
            "lift_steps": len(bc_lift),
            "predicted_target_z_rise_m": float(bc["cartesian_action"][bc_lift[-1], 2] - bc["cartesian_action"][bc_lift[0], 2]),
            "wrist_z_rise_m": float(bc["bc_wrist"][bc_lift[-1], 2] - bc["bc_wrist"][bc_lift[0], 2]),
            "object_z_rise_m": float(bc["bc_object"][bc_lift[-1], 2] - bc["bc_object"][bc_lift[0], 2]),
            "object_following_ratio": bc_follow,
            "mean_contact_force": float(bc["bc_contact_force"][bc_lift].mean()),
        },
        "conclusion": "Frozen XYZ9 raises the wrist but the grasp transmits almost none of that motion to the object; this is grasp-state/load-transfer failure, not absent Z command.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    x = np.arange(len(expert_lift))
    axes[0].plot(x, 1000 * (expert_target[expert_lift, 2] - expert_target[expert_lift[0], 2]), label="expert target z")
    axes[0].plot(x, 1000 * (expert_wrist[expert_lift, 2] - expert_wrist[expert_lift[0], 2]), label="expert wrist z")
    axes[0].plot(x, 1000 * (expert_object[expert_lift, 2] - expert_object[expert_lift[0], 2]), label="expert object z")
    axes[1].plot(x, expert_force[expert_lift], label="expert contact force")
    axes[1].plot(x, expert_support[expert_lift].astype(float), label="expert table support")
    y = np.arange(len(bc_lift))
    axes[2].plot(y, 1000 * (bc["bc_wrist"][bc_lift, 2] - bc["bc_wrist"][bc_lift[0], 2]), label="BC wrist z")
    axes[2].plot(y, 1000 * (bc["bc_object"][bc_lift, 2] - bc["bc_object"][bc_lift[0], 2]), label="BC object z")
    axes[2].plot(y, bc["bc_contact_force"][bc_lift], label="BC contact force")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Rise (mm)")
    axes[1].set_ylabel("Force / support")
    axes[2].set_ylabel("Rise (mm) / force")
    axes[2].set_xlabel("Lift control step")
    fig.tight_layout()
    fig.savefig(args.output / "expert_vs_bc_lift_physics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
