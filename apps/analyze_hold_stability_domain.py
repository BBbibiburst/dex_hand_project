"""Bounded local control-sensitivity study around a verified Hold snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from apps.trace_diffusion_rollout import _make_env
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import _robot_object_contact_summary
from source.verification.strict_replay import load_replay_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", choices=("hold_start", "hold_mid"), default="hold_start")
    parser.add_argument("--max-steps", type=int, default=250)
    return parser


def _restore(env, episode, index):
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    env.data.qpos[:] = episode.arrays["qpos"][index]
    env.data.qvel[:] = episode.arrays["qvel"][index]
    env.data.ctrl[:] = episode.arrays["ctrl"][index]
    env.data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _wrist_z_joint_direction(env) -> np.ndarray:
    arm = env.controller.arm_controller
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, arm.site_id)
    jac = jacp[:, arm.arm_dof_addrs]
    return np.linalg.pinv(jac) @ np.asarray([0.0, 0.0, 1.0])


def _run(env, episode, controls, source_index, control_start, arm_bias, hand_bias, max_steps):
    _restore(env, episode, source_index)
    initial_z = float(episode.arrays["object_position"][0, 2])
    opposing_values, forces, relative, successes = [], [], [], []
    binding = env.task._require_bindings().objects["object"]
    arm = env.controller.arm_controller
    positions = []
    for expert in controls[control_start : control_start + max_steps]:
        action = np.asarray(expert, dtype=np.float32).copy()
        action[:7] += arm_bias
        action[7:] += hand_bias
        action = np.clip(action, env.action_space.low, env.action_space.high)
        _, _, terminated, truncated, info = env.step(action)
        _, force, digit_count, _ = _robot_object_contact_summary(env, "object")
        active = digit_count > 0
        opposing_values.append(bool(active[4] and active[:4].sum() >= 2))
        forces.append(force)
        wrist = env.data.site_xpos[arm.site_id].copy()
        obj = env.data.xpos[binding.body_id].copy()
        positions.append(obj)
        relative.append(obj - wrist)
        successes.append(bool(info.get("task_success", False)))
        if terminated or truncated:
            break
    opposing = np.asarray(opposing_values)
    positions = np.asarray(positions)
    relative = np.asarray(relative)
    slip = np.linalg.norm(np.diff(relative, axis=0), axis=1)
    return {
        "steps": len(positions),
        "opposing_contact_fraction": float(opposing.mean()),
        "opposing_contact_tail_fraction": float(opposing[-40:].mean()),
        "mean_contact_force": float(np.mean(forces)),
        "mean_slip_m": float(slip.mean()) if len(slip) else 0.0,
        "maximum_lift_m": float(positions[:, 2].max() - initial_z),
        "final_lift_m": float(positions[-1, 2] - initial_z),
        "task_success": bool(np.any(successes)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    episode = DemonstrationEpisode.load(args.manifest)
    env = _make_env(payload, int(payload["frames"]) + 20)
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    controls, _, _, _ = load_replay_controls(args.manifest, env)
    stage = np.asarray(episode.arrays["stage"])
    hold = np.flatnonzero(stage == episode.metadata["stage_codes"]["hold"])
    source_index = int(hold[0] if args.snapshot == "hold_start" else hold[len(hold) // 2])
    pre_action_start = len(stage) - len(controls) - 1
    control_start = source_index - pre_action_start
    _restore(env, episode, source_index)
    wrist_z_direction = _wrist_z_joint_direction(env)
    rows = []

    def evaluate(kind, label, arm_bias=None, hand_bias=None, **parameters):
        arm_value = np.zeros(7) if arm_bias is None else np.asarray(arm_bias, dtype=np.float64)
        hand_value = np.zeros(6) if hand_bias is None else np.asarray(hand_bias, dtype=np.float64)
        metrics = _run(
            env,
            episode,
            controls,
            source_index,
            control_start,
            arm_value,
            hand_value,
            args.max_steps,
        )
        rows.append(
            {
                "kind": kind,
                "label": label,
                "arm_bias": arm_value.tolist(),
                "hand_bias": hand_value.tolist(),
                **parameters,
                **metrics,
            }
        )

    evaluate("baseline", "expert")
    for joint in range(7):
        for delta in (-0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01):
            bias = np.zeros(7)
            bias[joint] = delta
            evaluate(
                "arm_joint", f"j{joint + 1}:{delta:+.4f}", arm_bias=bias, joint=joint, delta=delta
            )
    for drive in range(6):
        for delta in (-0.001, -0.0005, -0.00025, 0.00025, 0.0005, 0.001):
            bias = np.zeros(6)
            bias[drive] = delta
            evaluate(
                "hand_drive", f"h{drive + 1}:{delta:+.5f}", hand_bias=bias, drive=drive, delta=delta
            )
    closure_indices = (0, 1, 2, 3, 5)
    for delta in (-0.001, -0.0005, -0.00025, 0.00025, 0.0005, 0.001):
        bias = np.zeros(6)
        bias[list(closure_indices)] = delta
        evaluate("hand_closure", f"closure:{delta:+.5f}", hand_bias=bias, delta=delta)
    for dz in (-0.005, -0.002, -0.001, 0.001, 0.002, 0.005):
        evaluate(
            "wrist_z",
            f"wrist_z:{dz:+.4f}",
            arm_bias=wrist_z_direction * dz,
            requested_wrist_z_m=dz,
            joint_bias_norm=float(np.linalg.norm(wrist_z_direction * dz)),
        )
    grid_values = (-0.001, -0.0005, 0.0, 0.0005, 0.001)
    for dz in grid_values:
        for closure in grid_values:
            hand_bias = np.zeros(6)
            hand_bias[list(closure_indices)] = closure
            evaluate(
                "grid",
                f"z={dz:+.4f},close={closure:+.4f}",
                arm_bias=wrist_z_direction * dz,
                hand_bias=hand_bias,
                requested_wrist_z_m=dz,
                closure_bias=closure,
            )
    env.close()
    result = {
        "manifest": str(args.manifest),
        "snapshot": args.snapshot,
        "source_index": source_index,
        "control_start": control_start,
        "max_steps": args.max_steps,
        "wrist_z_joint_direction": wrist_z_direction.tolist(),
        "success_definition": "existing Lift task_success; no threshold or physics changes",
        "rows": rows,
    }
    (args.output / "stability_domain.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt

    grid = [row for row in rows if row["kind"] == "grid"]
    matrix = np.asarray([row["task_success"] for row in grid], dtype=float).reshape(5, 5)
    lift = 1000.0 * np.asarray([row["maximum_lift_m"] for row in grid]).reshape(5, 5)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, values, title in (
        (axes[0], matrix, "Lift success"),
        (axes[1], lift, "Maximum lift (mm)"),
    ):
        plot = axis.imshow(values, origin="lower", cmap="viridis", aspect="auto")
        axis.set_xticks(range(5), [f"{1000 * x:+.1f}" for x in grid_values])
        axis.set_yticks(range(5), [f"{1000 * x:+.1f}" for x in grid_values])
        axis.set_xlabel("Closure target bias (mm actuator coordinate)")
        axis.set_ylabel("Requested wrist-z target bias (mm)")
        axis.set_title(title)
        figure.colorbar(plot, ax=axis)
    figure.tight_layout()
    figure.savefig(args.output / "hold_stability_grid.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps({"runs": len(rows), "successes": sum(r["task_success"] for r in rows)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
