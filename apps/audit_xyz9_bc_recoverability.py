"""Test oracle takeovers and bounded XYZ residuals from real frozen-BC states."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from apps.trace_action_chunk_bc import _inputs, execution_prefix
from apps.trace_diffusion_rollout import _make_env
from source.grasp_pipeline.trajectory import GraspTrajectory
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.imitation.observation import real_observable_state
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


SNAPSHOTS = {
    "close_early": 55,
    "close_end": 90,
    "hold_start": 91,
    "hold_mid": 111,
    "lift_start": 131,
}


def _restore(env, trace, step):
    env.reset(seed=0)
    env.data.qpos[:] = trace["bc_qpos"][step]
    env.data.qvel[:] = trace["bc_qvel"][step]
    env.data.ctrl[:] = trace["bc_ctrl"][step]
    env.data.qacc_warmstart[:] = trace["bc_qacc_warmstart"][step]
    mujoco.mj_forward(env.model, env.data)


def _save_full_trajectory(directory, payload, source, controls, qpos, qvel, schema):
    trajectory = GraspTrajectory(
        object_id=str(payload["object_id"]),
        source_manifest=str(source.resolve()),
        start_stage="approach",
        action_mode="absolute_position",
        residual_actions=np.zeros((len(controls), 1), dtype=np.float32),
        controls=np.asarray(controls, dtype=np.float32),
        initial_qpos=qpos,
        initial_qvel=qvel,
        success=False,
        episode_return=0.0,
        metadata={
            "control_dt": float(payload.get("metadata", {}).get("control_dt", 0.05)),
            "source_seed": int(payload.get("metadata", {}).get("source_seed", 0)),
            "p7_recovery_schema": schema,
        },
    )
    return trajectory.save(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--snapshots", nargs="*", choices=tuple(SNAPSHOTS))
    parser.add_argument("--variants", nargs="*")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    trace = np.load(args.baseline_trace)
    env = _make_env(payload, 400)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    expert, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(args.manifest, env)
    if initial_ctrl is None:
        initial_ctrl = env.data.ctrl.copy()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy = ActionChunkBC.from_checkpoint(checkpoint, args.device).eval()
    fixed_orientation = np.asarray(checkpoint["fixed_orientation_wxyz"], dtype=np.float64)
    renderer = mujoco.Renderer(env.model, height=120, width=160)
    residuals = {
        "bc_continue": np.zeros(3),
        "res_z1mm": np.asarray([0.0, 0.0, 0.001]),
        "res_z2mm": np.asarray([0.0, 0.0, 0.002]),
        "res_z3mm": np.asarray([0.0, 0.0, 0.003]),
        "res_z5mm": np.asarray([0.0, 0.0, 0.005]),
    }
    variants = [
        "expert_xyz_expert_hand",
        "expert_xyz_bc_hand",
        "bc_xyz_expert_hand",
        *residuals,
    ]
    rows = []
    try:
        selected_snapshots = args.snapshots or list(SNAPSHOTS)
        selected_variants = args.variants or variants
        for snapshot in selected_snapshots:
            step = SNAPSHOTS[snapshot]
            for variant in selected_variants:
                started = time.perf_counter()
                _restore(env, trace, step)
                adapter = ContinuousCartesianAdapter(env)
                adapter.reset(env.data.ctrl[env.controller.arm_controller.actuator_ids])
                emitted = []
                training = {
                    "state": [],
                    "tactile": [],
                    "bc_action": [],
                    "recovery_action": [],
                    "residual_xyz": [],
                }
                local = step + 1
                while local < min(len(expert), 340):
                    observation = env._get_observation()
                    renderer.update_scene(env.data, camera="agentview")
                    image = renderer.render().copy()
                    anchor = adapter.measured_pose()
                    with torch.inference_mode():
                        bc_chunk = (
                            policy.sample(
                                *_inputs(
                                    observation, image, real_observable_state(env), args.device
                                )
                            )[0]
                            .cpu()
                            .numpy()
                        )
                    for bc_label in execution_prefix(bc_chunk, args.action_steps):
                        if local >= min(len(expert), 340):
                            break
                        expert_action = np.asarray(expert[local], dtype=np.float32)
                        expert_label = adapter.encode(expert_action, anchor, XYZ9)
                        if variant.startswith("expert_xyz"):
                            label = expert_label.copy()
                        else:
                            label = bc_label.copy()
                        if variant in residuals:
                            label[:3] += residuals[variant]
                        if variant.endswith("expert_hand"):
                            label[3:] = expert_action[7:]
                        training["state"].append(real_observable_state(env))
                        training["tactile"].append(
                            np.asarray(observation["tactile"], dtype=np.float32)
                        )
                        training["bc_action"].append(np.asarray(bc_label, dtype=np.float32))
                        training["recovery_action"].append(np.asarray(label, dtype=np.float32))
                        training["residual_xyz"].append(
                            np.asarray(label[:3] - bc_label[:3], dtype=np.float32)
                        )
                        control, ik = adapter.decode(
                            label,
                            anchor,
                            XYZ9,
                            fixed_orientation_wxyz=fixed_orientation,
                        )
                        if not ik.success:
                            break
                        env.step(np.clip(control, env.action_space.low, env.action_space.high))
                        emitted.append(control)
                        local += 1
                full = np.concatenate([trace["bc_arm_target"][: step + 1], np.empty((0, 7))])
                _ = full  # arm-only trace is intentionally not used for strict replay.
                prefix = np.asarray(trace["bc_arm_target"][: step + 1])
                prefix_hand = np.asarray(trace["bc_hand_target"][: step + 1])
                prefix_joint13 = np.concatenate([prefix, prefix_hand], axis=1)
                full_controls = np.concatenate([prefix_joint13, np.asarray(emitted)], axis=0)
                directory = args.output / snapshot / variant
                manifest = _save_full_trajectory(
                    directory / "trajectory",
                    payload,
                    args.manifest,
                    full_controls,
                    initial_qpos,
                    initial_qvel,
                    variant,
                )
                strict = strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=False)
                record = {
                    "snapshot": snapshot,
                    "snapshot_step": step,
                    "variant": variant,
                    "oracle_xyz": variant.startswith("expert_xyz"),
                    "oracle_hand": variant.endswith("expert_hand"),
                    "uses_tactile": False,
                    "residual_xyz_m": residuals.get(variant, np.zeros(3)).tolist(),
                    "emitted_recovery_steps": len(emitted),
                    "runtime_seconds": time.perf_counter() - started,
                    "strict_replay": asdict(strict),
                    "trajectory_manifest": str(manifest),
                }
                (directory / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
                if strict.success and variant == "expert_xyz_bc_hand":
                    np.savez_compressed(
                        directory / "validated_recovery.npz",
                        **{key: np.asarray(value) for key, value in training.items()},
                        snapshot_step=np.asarray(step),
                    )
                rows.append(record)
                print(
                    json.dumps(
                        {
                            "snapshot": snapshot,
                            "variant": variant,
                            "success": strict.success,
                            "max_lift": strict.max_lift,
                            "final_lift": strict.final_lift,
                        }
                    ),
                    flush=True,
                )
    finally:
        renderer.close()
        env.close()
    result = {
        "snapshots": SNAPSHOTS,
        "attempts": len(rows),
        "strict_successes": sum(row["strict_replay"]["success"] for row in rows),
        "rows": rows,
        "classification_note": "not recovered means not recovered within this bounded search only",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
