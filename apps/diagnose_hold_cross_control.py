"""Run expert/BC arm-hand cross-control interventions from verified Hold snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from apps.trace_action_chunk_bc import _inputs, execution_prefix
from apps.trace_diffusion_rollout import _make_env
from source.geometry import mat_to_quat
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import _palm_object_contact, _robot_object_contact_summary
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.observation import real_observable_state
from source.verification.strict_replay import load_replay_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--max-local-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2011)
    return parser


def _restore(env, episode: DemonstrationEpisode, source_index: int) -> None:
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    env.data.qpos[:] = episode.arrays["qpos"][source_index]
    env.data.qvel[:] = episode.arrays["qvel"][source_index]
    env.data.ctrl[:] = episode.arrays["ctrl"][source_index]
    env.data.time = source_index * float(episode.metadata["control_dt"])
    env.data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _row(env, executed, expert, predicted, chunk_index, latency, success):
    arm = env.controller.arm_controller
    binding = env.task._require_bindings().objects["object"]
    count, force, digit_count, digit_force = _robot_object_contact_summary(env, "object")
    wrist = env.data.site_xpos[arm.site_id].copy()
    wrist_quat = mat_to_quat(env.data.site_xmat[arm.site_id]).copy()
    obj = env.data.xpos[binding.body_id].copy()
    obj_velocity = env.data.qvel[binding.qvel_adr : binding.qvel_adr + 6].copy()
    tactile = np.asarray(env._get_observation()["tactile"], dtype=np.float32)
    return {
        "arm_qpos": env.data.qpos[arm.qpos_addrs].copy(),
        "arm_target": np.asarray(executed[:7]).copy(),
        "hand_target": np.asarray(executed[7:]).copy(),
        "hand_joint_qpos": env.data.qpos[7:19].copy(),
        "wrist_position": wrist,
        "wrist_quat_wxyz": wrist_quat,
        "object_position": obj,
        "object_quat_wxyz": env.data.xquat[binding.body_id].copy(),
        "object_velocity": obj_velocity,
        "wrist_object_relative": obj - wrist,
        "contact_count": count,
        "contact_force": force,
        "digit_contact_count": digit_count,
        "digit_contact_force": digit_force,
        "palm_contact": _palm_object_contact(env, "object"),
        "tactile_sum": float(tactile.sum()),
        "executed_action": np.asarray(executed).copy(),
        "expert_action": np.asarray(expert).copy(),
        "predicted_action": np.asarray(predicted).copy(),
        "action_error": np.asarray(executed - expert).copy(),
        "chunk_index": chunk_index,
        "inference_latency_s": latency,
        "task_success": success,
    }


def _longest_true(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _summary(rows, original_object_z):
    digit = np.asarray([row["digit_contact_count"] for row in rows]) > 0
    opposing = digit[:, 4] & (digit[:, :4].sum(axis=1) >= 2)
    positions = np.asarray([row["object_position"] for row in rows])
    relative = np.asarray([row["wrist_object_relative"] for row in rows])
    slip = np.linalg.norm(np.diff(relative, axis=0), axis=1)
    success = np.asarray([row["task_success"] for row in rows], dtype=bool)
    first_loss = np.flatnonzero(~opposing)
    return {
        "steps": len(rows),
        "opposing_contact_fraction": float(opposing.mean()),
        "longest_opposing_contact_steps": _longest_true(opposing),
        "first_opposing_contact_loss_step": int(first_loss[0]) if len(first_loss) else None,
        "mean_contact_force": float(np.mean([row["contact_force"] for row in rows])),
        "mean_relative_slip_per_step_m": float(slip.mean()) if len(slip) else 0.0,
        "maximum_lift_m": float(positions[:, 2].max() - original_object_z),
        "final_lift_m": float(positions[-1, 2] - original_object_z),
        "task_success": bool(success.any()),
        "mean_arm_action_error_l2": float(
            np.mean([np.linalg.norm(row["action_error"][:7]) for row in rows])
        ),
        "mean_hand_action_error_l2": float(
            np.mean([np.linalg.norm(row["action_error"][7:]) for row in rows])
        ),
        "inference_p95_ms": 1000.0
        * float(np.percentile([row["inference_latency_s"] for row in rows], 95)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    payload = json.loads(args.manifest.read_text())
    episode = DemonstrationEpisode.load(args.manifest)
    env = _make_env(payload, int(payload["frames"]) + 20)
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    controls, _, _, _ = load_replay_controls(args.manifest, env)
    source_pre_action_start = len(episode.arrays["stage"]) - len(controls) - 1
    codes = episode.metadata["stage_codes"]
    stage = np.asarray(episode.arrays["stage"])
    close = np.flatnonzero(stage == codes["close"])
    hold = np.flatnonzero(stage == codes["hold"])
    lift = np.flatnonzero(stage == codes["lift"])
    snapshots = {
        "close_end": int(close[-1]),
        "hold_start": int(hold[0]),
        "hold_mid": int(hold[len(hold) // 2]),
        "lift_start": int(lift[0]),
    }
    checkpoint_bytes = args.checkpoint.read_bytes()
    policy = ActionChunkBC.from_checkpoint(
        torch.load(args.checkpoint, map_location=args.device, weights_only=False), args.device
    ).eval()
    renderer = mujoco.Renderer(env.model, height=120, width=160)
    original_object_z = float(episode.arrays["object_position"][0, 2])
    combinations = {
        "expert_arm_expert_hand": (False, False),
        "expert_arm_bc_hand": (False, True),
        "bc_arm_expert_hand": (True, False),
        "bc_arm_bc_hand": (True, True),
    }
    summaries = []
    try:
        for snapshot_name, source_index in snapshots.items():
            control_start = source_index - source_pre_action_start
            for combination, (bc_arm, bc_hand) in combinations.items():
                _restore(env, episode, source_index)
                rows = []
                writer = None
                if snapshot_name == "hold_start":
                    import imageio.v2 as imageio

                    video = args.output / f"{snapshot_name}_{combination}.mp4"
                    writer = imageio.get_writer(video, fps=20)
                local = 0
                while local < min(args.max_local_steps, len(controls) - control_start):
                    observation = env._get_observation()
                    renderer.update_scene(env.data, camera="agentview")
                    image = renderer.render().copy()
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        chunk = (
                            policy.sample(
                                *_inputs(
                                    observation,
                                    image,
                                    real_observable_state(env),
                                    args.device,
                                )
                            )[0]
                            .cpu()
                            .numpy()
                        )
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - started
                    for chunk_index, predicted in enumerate(
                        execution_prefix(chunk, args.action_steps)
                    ):
                        expert_index = control_start + local
                        if local >= args.max_local_steps or expert_index >= len(controls):
                            break
                        expert = np.asarray(controls[expert_index], dtype=np.float32)
                        executed = expert.copy()
                        if bc_arm:
                            executed[:7] = predicted[:7]
                        if bc_hand:
                            executed[7:] = predicted[7:]
                        executed = np.clip(executed, env.action_space.low, env.action_space.high)
                        _, _, terminated, truncated, info = env.step(executed)
                        rows.append(
                            _row(
                                env,
                                executed,
                                expert,
                                predicted,
                                chunk_index,
                                latency if chunk_index == 0 else 0.0,
                                bool(info.get("task_success", False)),
                            )
                        )
                        local += 1
                        if writer is not None:
                            renderer.update_scene(env.data, camera="agentview")
                            writer.append_data(renderer.render().copy())
                        if terminated or truncated:
                            break
                    if terminated or truncated:
                        break
                if writer is not None:
                    writer.close()
                arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                np.savez_compressed(args.output / f"{snapshot_name}_{combination}.npz", **arrays)
                summary = {
                    "snapshot": snapshot_name,
                    "source_index": source_index,
                    "control_start": control_start,
                    "combination": combination,
                    **_summary(rows, original_object_z),
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    finally:
        renderer.close()
        env.close()
    result = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "action_steps": args.action_steps,
        "control_hz": 1.0 / float(episode.metadata["control_dt"]),
        "source_pre_action_start": source_pre_action_start,
        "snapshots": snapshots,
        "snapshot_contract": (
            "qpos/qvel/ctrl restored; model has na=0; qacc_warmstart cleared; position controller "
            "has no hidden filter state during direct position control"
        ),
        "rows": summaries,
    }
    (args.output / "cross_control_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
