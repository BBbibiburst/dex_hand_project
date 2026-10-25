"""Generate bounded BC-deviation -> expert-recovery trajectories near verified Hold."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from apps.trace_action_chunk_bc import _inputs, execution_prefix
from apps.trace_diffusion_rollout import _make_env
from source.grasping.contracts import DemonstrationEpisode
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.observation import real_observable_state
from source.verification.strict_replay import _contact_sets, _contact_snapshot, load_replay_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--verify-tail", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2011)
    return parser


def _restore(env, episode, index):
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    env.data.qpos[:] = episode.arrays["qpos"][index]
    env.data.qvel[:] = episode.arrays["qvel"][index]
    env.data.ctrl[:] = episode.arrays["ctrl"][index]
    env.data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _recover(env, initial, controls, original_z, tail):
    env.reset(seed=0)
    env.data.qpos[:] = initial["qpos"]
    env.data.qvel[:] = initial["qvel"]
    env.data.ctrl[:] = initial["ctrl"]
    env.data.qacc_warmstart[:] = initial["qacc_warmstart"]
    mujoco.mj_forward(env.model, env.data)
    binding = env.task._require_bindings().objects["object"]
    sets = _contact_sets(env)
    rows = {
        key: []
        for key in (
            "qpos",
            "qvel",
            "ctrl",
            "action",
            "state",
            "tactile",
            "object_position",
            "object_velocity",
            "contact",
            "grasp",
            "opposition",
            "robot_table",
            "penetration",
            "task_success",
        )
    }
    for action in controls:
        observation = env._get_observation()
        rows["qpos"].append(env.data.qpos.copy())
        rows["qvel"].append(env.data.qvel.copy())
        rows["ctrl"].append(env.data.ctrl.copy())
        rows["state"].append(real_observable_state(env))
        rows["tactile"].append(np.asarray(observation["tactile"], np.float32))
        rows["action"].append(np.asarray(action, np.float32))
        _, _, terminated, truncated, info = env.step(action)
        contact, digits, _palm, grasp, opposition, robot_table, penetration = _contact_snapshot(
            env, sets
        )
        rows["object_position"].append(env.data.xpos[binding.body_id].copy())
        rows["object_velocity"].append(
            env.data.qvel[binding.qvel_adr : binding.qvel_adr + 6].copy()
        )
        rows["contact"].append(contact > 0)
        rows["grasp"].append(bool(grasp and digits >= 2))
        rows["opposition"].append(opposition)
        rows["robot_table"].append(robot_table)
        rows["penetration"].append(penetration)
        rows["task_success"].append(bool(info.get("task_success", False)))
        if terminated or truncated:
            break
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    count = min(tail, len(arrays["action"]))
    speed = np.linalg.norm(arrays["object_velocity"][-count:, :3], axis=1)
    angular = np.linalg.norm(arrays["object_velocity"][-count:, 3:], axis=1)
    lift = arrays["object_position"][:, 2] - original_z
    metrics = {
        "frames": len(arrays["action"]),
        "max_lift_m": float(lift.max()),
        "final_lift_m": float(lift[-1]),
        "tail_min_lift_m": float(lift[-count:].min()),
        "tail_contact_fraction": float(arrays["contact"][-count:].mean()),
        "tail_grasp_fraction": float(arrays["grasp"][-count:].mean()),
        "tail_opposition_mean": float(arrays["opposition"][-count:].mean()),
        "tail_max_speed": float(speed.max()),
        "tail_max_angular_speed": float(angular.max()),
        "tail_robot_table_fraction": float((arrays["robot_table"][-count:] > 0).mean()),
        "max_penetration": float(arrays["penetration"].max()),
    }
    metrics["success"] = bool(
        metrics["tail_min_lift_m"] >= 0.055
        and metrics["tail_max_speed"] <= 0.10
        and metrics["tail_max_angular_speed"] <= 0.10
        and metrics["tail_contact_fraction"] >= 0.80
        and metrics["tail_grasp_fraction"] >= 0.80
        and metrics["tail_opposition_mean"] >= 0.25
        and metrics["tail_robot_table_fraction"] <= 0.0
        and metrics["max_penetration"] <= 0.003
    )
    return arrays, metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    payload = json.loads(args.manifest.read_text())
    episode = DemonstrationEpisode.load(args.manifest)
    env = _make_env(payload, int(payload["frames"]) + 50)
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    controls, _, _, _ = load_replay_controls(args.manifest, env)
    policy = ActionChunkBC.from_checkpoint(
        torch.load(args.checkpoint, map_location=args.device, weights_only=False), args.device
    ).eval()
    renderer = mujoco.Renderer(env.model, height=120, width=160)
    stage = np.asarray(episode.arrays["stage"])
    hold = np.flatnonzero(stage == episode.metadata["stage_codes"]["hold"])
    lift = np.flatnonzero(stage == episode.metadata["stage_codes"]["lift"])
    snapshots = (int(hold[0]), int(hold[len(hold) // 2]), int(lift[0]))
    divergence_steps = tuple(range(4, 41, 4))
    pre_action_start = len(stage) - len(controls) - 1
    original_z = float(episode.arrays["object_position"][0, 2])
    attempts = []
    try:
        for source_index in snapshots:
            control_start = source_index - pre_action_start
            for divergence in divergence_steps:
                started = time.perf_counter()
                _restore(env, episode, source_index)
                local = 0
                while local < divergence:
                    observation = env._get_observation()
                    renderer.update_scene(env.data, camera="agentview")
                    image = renderer.render().copy()
                    with torch.inference_mode():
                        chunk = (
                            policy.sample(
                                *_inputs(
                                    observation, image, real_observable_state(env), args.device
                                )
                            )[0]
                            .cpu()
                            .numpy()
                        )
                    for predicted in execution_prefix(chunk, args.action_steps):
                        if local >= divergence:
                            break
                        env.step(np.clip(predicted, env.action_space.low, env.action_space.high))
                        local += 1
                initial = {
                    "qpos": env.data.qpos.copy(),
                    "qvel": env.data.qvel.copy(),
                    "ctrl": env.data.ctrl.copy(),
                    "qacc_warmstart": env.data.qacc_warmstart.copy(),
                }
                recovery_controls = np.asarray(controls[control_start + divergence :], np.float32)
                arrays, metrics = _recover(
                    env, initial, recovery_controls, original_z, args.verify_tail
                )
                # A second independent restoration/replay is mandatory before acceptance.
                _, replay_metrics = _recover(
                    env, initial, recovery_controls, original_z, args.verify_tail
                )
                verified = bool(metrics["success"] and replay_metrics["success"])
                attempt_id = f"s{source_index:03d}_d{divergence:02d}"
                directory = args.output / attempt_id
                directory.mkdir(exist_ok=True)
                if verified:
                    np.savez_compressed(
                        directory / "recovery.npz",
                        initial_qpos=initial["qpos"],
                        initial_qvel=initial["qvel"],
                        initial_ctrl=initial["ctrl"],
                        initial_qacc_warmstart=initial["qacc_warmstart"],
                        **arrays,
                    )
                record = {
                    "id": attempt_id,
                    "source_manifest": str(args.manifest),
                    "source_index": source_index,
                    "control_start": control_start,
                    "perturbation_type": "closed_loop_bc_divergence",
                    "divergence_steps": divergence,
                    "recovery_source": "aligned_expert_future_controls",
                    "verification_protocol": "local recovery using unchanged FINAL profile thresholds",
                    "verification_status": (
                        "LOCAL_RECOVERY_VERIFIED_FINAL_CRITERIA"
                        if verified
                        else "LOCAL_RECOVERY_REJECTED"
                    ),
                    "metrics": metrics,
                    "replay_metrics": replay_metrics,
                    "runtime_seconds": time.perf_counter() - started,
                    "artifact": str(directory / "recovery.npz") if verified else None,
                }
                (directory / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
                attempts.append(record)
                print(json.dumps({"id": attempt_id, "verified": verified, **metrics}), flush=True)
    finally:
        renderer.close()
        env.close()
    verified = [
        row for row in attempts if row["verification_status"].startswith("LOCAL_RECOVERY_VERIFIED")
    ]
    summary = {
        "attempts": len(attempts),
        "verified": len(verified),
        "success_rate": len(verified) / len(attempts),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in attempts])),
        "source_indices": list(snapshots),
        "divergence_steps": list(divergence_steps),
        "note": "Local recovery artifacts are not mislabeled as full-episode FINAL_VERIFIED trajectories.",
        "rows": attempts,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
