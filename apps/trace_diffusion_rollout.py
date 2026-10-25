"""Trace expert and diffusion-policy control on one exact verified reset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from apps.evaluate_diffusion import tensor_observation
from source.envs.manipulation import make_lift_env
from source.imitation.diffusion_policy import DiffusionPolicy
from source.imitation.observation import real_observable_state
from source.verification.strict_replay import load_replay_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--inference-steps", type=int)
    return parser


def _make_env(payload, max_steps):
    return make_lift_env(
        task_config={
            "object_id": payload["object_id"],
            "reward_shaping": False,
            "terminate_on_success": False,
        },
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=None,
        control_dt=float(payload.get("metadata", {}).get("control_dt", 0.05)),
        episode_length=max_steps + 20,
    )


def _reset_exact(env, qpos, qvel, ctrl):
    observation, _ = env.reset(seed=0)
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = ctrl
    mujoco.mj_forward(env.model, env.data)
    return observation


def _snapshot(env, target, chunk_start):
    from source.grasping.executor import _palm_object_contact, _robot_object_contact_summary
    from source.geometry import mat_to_quat

    arm = env.controller.arm_controller
    binding = env.task._require_bindings().objects["object"]
    wrist = env.data.site_xpos[arm.site_id].copy()
    obj = env.data.xpos[binding.body_id].copy()
    count, force, digit_counts, digit_forces = _robot_object_contact_summary(env, "object")
    tactile = np.asarray(env._get_observation().get("tactile", np.zeros(0)), dtype=np.float32)
    return {
        "arm_qpos": env.data.qpos[arm.qpos_addrs].copy(),
        "arm_ctrl": env.data.ctrl[arm.actuator_ids].copy(),
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "ctrl": env.data.ctrl.copy(),
        "qacc_warmstart": env.data.qacc_warmstart.copy(),
        "arm_target": np.asarray(target[:7]).copy(),
        "hand_target": np.asarray(target[7:]).copy(),
        "wrist": wrist,
        "wrist_quat_wxyz": mat_to_quat(env.data.site_xmat[arm.site_id]),
        "object": obj,
        "object_quat_wxyz": env.data.xquat[binding.body_id].copy(),
        "wrist_distance": float(np.linalg.norm(wrist - obj)),
        "contact": bool(env.task._is_robot_touching_object(env.model, env.data, "object")),
        "contact_count": int(count),
        "contact_force": float(force),
        "digit_contact_count": digit_counts.copy(),
        "digit_contact_force": digit_forces.copy(),
        "palm_contact": bool(_palm_object_contact(env, "object")),
        "tactile_sum": float(tactile.sum()),
        "tactile_max": float(tactile.max(initial=0.0)),
        "tactile": tactile.copy(),
        "chunk_start": bool(chunk_start),
    }


def _stack(rows, key):
    return np.asarray([row[key] for row in rows])


def _summary(rows, initial_object_z):
    distances = _stack(rows, "wrist_distance")
    targets = _stack(rows, "arm_target")
    qpos = _stack(rows, "arm_qpos")
    contacts = _stack(rows, "contact")
    lifts = _stack(rows, "object")[:, 2] - initial_object_z
    first_contact = np.flatnonzero(contacts)
    return {
        "steps": len(rows),
        "initial_wrist_distance_m": float(distances[0]),
        "minimum_wrist_distance_m": float(distances.min()),
        "final_wrist_distance_m": float(distances[-1]),
        "first_contact_step": int(first_contact[0]) if len(first_contact) else None,
        "contact_fraction": float(contacts.mean()),
        "maximum_lift_m": float(lifts.max()),
        "final_lift_m": float(lifts[-1]),
        "mean_target_delta_l2": float(np.mean(np.linalg.norm(np.diff(targets, axis=0), axis=1))),
        "p95_target_delta_l2": float(
            np.percentile(np.linalg.norm(np.diff(targets, axis=0), axis=1), 95)
        ),
        "mean_real_motion_l2": float(np.mean(np.linalg.norm(np.diff(qpos, axis=0), axis=1))),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action_steps <= 0:
        raise ValueError("--action-steps must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    env = _make_env(payload, args.max_steps)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    controls, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(args.manifest, env)
    if initial_ctrl is None:
        initial_ctrl = env.data.ctrl.copy()
    binding = env.task._require_bindings().objects["object"]
    _reset_exact(env, initial_qpos, initial_qvel, initial_ctrl)
    initial_object_z = float(env.data.xpos[binding.body_id, 2])

    expert_rows = []
    for action in controls[: args.max_steps]:
        env.step(action)
        expert_rows.append(_snapshot(env, action, False))

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy = DiffusionPolicy.from_checkpoint(checkpoint, args.device).eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    observation = _reset_exact(env, initial_qpos, initial_qvel, initial_ctrl)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    import imageio.v2 as imageio

    video_path = args.output / "dp_rollout.mp4"
    writer = imageio.get_writer(video_path, fps=20)
    dp_rows = []
    try:
        while len(dp_rows) < args.max_steps:
            renderer.update_scene(env.data, camera="agentview")
            image = renderer.render().copy()
            inputs = tensor_observation(
                observation,
                image,
                real_observable_state(env),
                args.device,
                use_tactile=policy.config.tactile_dim > 0,
            )
            chunk = policy.sample(*inputs, inference_steps=args.inference_steps)[0].cpu().numpy()
            for chunk_index, predicted in enumerate(chunk[: args.action_steps]):
                if len(dp_rows) >= args.max_steps:
                    break
                target = np.clip(predicted, env.action_space.low, env.action_space.high)
                observation, _, terminated, truncated, _ = env.step(target)
                dp_rows.append(_snapshot(env, target, chunk_index == 0))
                renderer.update_scene(env.data, camera="agentview")
                writer.append_data(renderer.render().copy())
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
    finally:
        writer.close()
        renderer.close()
        env.close()

    arrays = {}
    for prefix, rows in (("expert", expert_rows), ("dp", dp_rows)):
        for key in rows[0]:
            arrays[f"{prefix}_{key}"] = _stack(rows, key)
    np.savez_compressed(args.output / "rollout_trace.npz", **arrays)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    chunk_steps = np.flatnonzero(_stack(dp_rows, "chunk_start"))
    for joint in range(7):
        axis = axes.flat[joint]
        axis.plot(_stack(expert_rows, "arm_target")[:, joint], label="expert", linewidth=1.5)
        axis.plot(_stack(dp_rows, "arm_target")[:, joint], label="DP", alpha=0.85)
        for boundary in chunk_steps:
            axis.axvline(boundary, color="grey", alpha=0.08, linewidth=0.5)
        axis.set_title(f"Arm joint {joint + 1} target")
        axis.grid(alpha=0.2)
    axes.flat[7].axis("off")
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(args.output / "arm_target_curves.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(_stack(expert_rows, "wrist_distance"), label="expert")
    axes[0].plot(_stack(dp_rows, "wrist_distance"), label="DP")
    for boundary in chunk_steps:
        axes[0].axvline(boundary, color="grey", alpha=0.08, linewidth=0.5)
    axes[0].set(xlabel="Control step", ylabel="Wrist-to-object distance (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    expert_wrist = _stack(expert_rows, "wrist")
    dp_wrist = _stack(dp_rows, "wrist")
    axes[1].plot(expert_wrist[:, 0], expert_wrist[:, 2], label="expert")
    axes[1].plot(dp_wrist[:, 0], dp_wrist[:, 2], label="DP")
    axes[1].set(xlabel="Wrist x (m)", ylabel="Wrist z (m)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output / "approach_comparison.png", dpi=180)
    plt.close(figure)

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "seed": args.seed,
        "prediction_horizon": policy.config.horizon,
        "observation_horizon": 1,
        "action_horizon": args.action_steps,
        "inference_steps": args.inference_steps or policy.config.diffusion_steps,
        "chunk_execution": (
            "sample one prediction_horizon chunk, execute indices "
            f"[0:{args.action_steps}], then observe and replan"
        ),
        "expert": _summary(expert_rows, initial_object_z),
        "dp": _summary(dp_rows, initial_object_z),
        "video": str(video_path),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
