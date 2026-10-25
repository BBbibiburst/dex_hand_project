"""Evaluate fixed-height stop and controlled Hold around a frozen Lift GRU."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from apps.trace_diffusion_rollout import _make_env, _reset_exact, _snapshot, _stack, _summary
from source.grasp_pipeline.trajectory import GraspTrajectory
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.imitation.lift_supervisor import LiftPhase, LiftSupervisor, LiftSupervisorConfig
from source.imitation.lift_temporal import LiftTemporalController
from source.imitation.observation import real_observable_state
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import strict_replay_manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed_stop", "stop_hold"), required=True)
    parser.add_argument("--target-rise-mm", type=float, default=85.0)
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument("--max-active-steps", type=int, default=180)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--preparation-xyz-mm", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--preparation-hand-delta", nargs=6, type=float, default=(0.0,) * 6)
    parser.add_argument("--preparation-steps", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.dataset / "manifest.json").read_text())
    row = next(item for item in metadata["rows"] if item["root_id"] == args.root_id)
    source = Path(row["manifest"])
    payload = json.loads(source.read_text())
    with np.load(row["path"]) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = LiftTemporalController.from_checkpoint(checkpoint, args.device).eval()
    env = _make_env(
        payload, args.preparation_steps + args.max_active_steps + args.hold_steps + 30
    )
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
    if args.snapshot is None:
        snapshot = data
    else:
        with np.load(args.snapshot) as archive:
            snapshot = {key: np.asarray(archive[key]) for key in archive.files}
    _reset_exact(
        env,
        snapshot["qpos"] if "qpos" in snapshot else snapshot["hold_qpos"],
        snapshot["qvel"] if "qvel" in snapshot else snapshot["hold_qvel"],
        snapshot["ctrl"] if "ctrl" in snapshot else snapshot["hold_ctrl"],
    )
    initial_qpos = env.data.qpos.copy()
    initial_qvel = env.data.qvel.copy()
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset()
    start_pose = adapter.measured_pose()
    fixed_orientation = start_pose[1]
    binding = env.task._require_bindings().objects["object"]
    initial_object_z = float(env.data.xpos[binding.body_id, 2])
    rows, controls, labels, phases = [], [], [], []
    preparation_target = start_pose[0] + np.asarray(args.preparation_xyz_mm) / 1000.0
    initial_hand = env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
    target_hand = initial_hand + np.asarray(args.preparation_hand_delta, dtype=np.float64)
    for _ in range(args.preparation_steps):
        anchor = adapter.measured_pose()
        xyz = preparation_target - anchor[0]
        control, result = adapter.decode(
            np.r_[xyz, target_hand], anchor, XYZ9, fixed_orientation_wxyz=fixed_orientation
        )
        control = np.clip(control, env.action_space.low, env.action_space.high)
        env.step(control)
        rows.append(_snapshot(env, control, False))
        controls.append(control)
        labels.append(xyz)
        phases.append("PREPARATION")
        if not result.success:
            break
    start_pose = adapter.measured_pose()
    supervisor = LiftSupervisor(
        LiftSupervisorConfig(
            target_rise_m=args.target_rise_mm / 1000.0,
            hold_steps=args.hold_steps,
            maximum_active_steps=args.max_active_steps,
        )
    )
    supervisor.reset(start_pose[0][2])
    state_history = deque(
        [real_observable_state(env)] * model.config.history, maxlen=model.config.history
    )
    action_history = deque(
        [np.zeros(3, np.float32)] * model.config.history, maxlen=model.config.history
    )
    renderer = mujoco.Renderer(env.model, height=240, width=320) if args.video else None
    writer = imageio.get_writer(args.output / "rollout.mp4", fps=20) if args.video else None
    hold_position = None
    discarded_chunk_actions = 0
    safety_clamps = 0
    try:
        while not supervisor.terminal:
            if supervisor.phase is LiftPhase.TARGET_REACHED:
                hold_position = adapter.measured_pose()[0]
                if args.mode == "fixed_stop":
                    break
                supervisor.observe(hold_position[2])
                continue
            if supervisor.phase is LiftPhase.HOLD_STABILIZE:
                current = adapter.measured_pose()[0]
                xyz = hold_position - current
                anchor = adapter.measured_pose()
                hand = env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
                control, result = adapter.decode(
                    np.r_[xyz, hand], anchor, XYZ9, fixed_orientation_wxyz=fixed_orientation
                )
                control = np.clip(control, env.action_space.low, env.action_space.high)
                env.step(control)
                supervisor.observe(adapter.measured_pose()[0][2], control_ok=result.success)
                candidates = [
                    (xyz, control, True, _snapshot(env, control, True), real_observable_state(env))
                ]
            else:
                states = torch.from_numpy(np.stack(state_history))[None].to(args.device)
                actions = torch.from_numpy(np.stack(action_history))[None].to(args.device)
                with torch.inference_mode():
                    chunk = model(states, actions)[0].cpu().numpy()
                anchor = adapter.measured_pose()
                candidates = []
                for chunk_index, requested in enumerate(chunk[: args.action_steps]):
                    xyz = requested.copy()
                    bounded_z = supervisor.clamp_delta_z(anchor[0][2], xyz[2])
                    safety_clamps += int(bounded_z < xyz[2] - 1e-9)
                    xyz[2] = bounded_z
                    hand = env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
                    control, result = adapter.decode(
                        np.r_[xyz, hand], anchor, XYZ9, fixed_orientation_wxyz=fixed_orientation
                    )
                    control = np.clip(control, env.action_space.low, env.action_space.high)
                    env.step(control)
                    candidates.append(
                        (
                            xyz,
                            control,
                            chunk_index == 0,
                            _snapshot(env, control, chunk_index == 0),
                            real_observable_state(env),
                        )
                    )
                    phase = supervisor.observe(
                        adapter.measured_pose()[0][2], control_ok=result.success
                    )
                    if phase is LiftPhase.TARGET_REACHED:
                        discarded_chunk_actions += args.action_steps - chunk_index - 1
                        break
            for xyz, control, chunk_start, snapshot, observable_state in candidates:
                rows.append(snapshot)
                controls.append(control)
                labels.append(xyz)
                phases.append(supervisor.phase.value)
                state_history.append(observable_state)
                action_history.append(np.asarray(xyz, np.float32))
                if writer is not None:
                    renderer.update_scene(env.data, camera="agentview")
                    writer.append_data(renderer.render().copy())
    finally:
        if writer is not None:
            writer.close()
            renderer.close()
        env.close()
    controls_a = np.asarray(controls, np.float32)
    trajectory = GraspTrajectory(
        object_id=payload["object_id"],
        source_manifest=str(source.resolve()),
        start_stage="hold",
        action_mode="absolute_position",
        residual_actions=np.zeros((len(controls_a), 1), np.float32),
        controls=controls_a,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        success=False,
        episode_return=0.0,
        metadata={"control_dt": 0.05, "source_seed": int(payload.get("metadata", {}).get("source_seed", payload["seed"])), "lift_supervisor": asdict(supervisor.config)},
    )
    manifest = trajectory.save(args.output / "emitted_trajectory")
    strict = strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=False)
    wrist = _stack(rows, "wrist")
    obj = _stack(rows, "object")
    target_rise = supervisor.config.target_rise_m
    summary = {
        "mode": args.mode,
        "root_id": args.root_id,
        "checkpoint": str(args.checkpoint),
        "snapshot": None if args.snapshot is None else str(args.snapshot),
        "preparation_xyz_mm": list(args.preparation_xyz_mm),
        "preparation_hand_delta": list(args.preparation_hand_delta),
        "preparation_steps": args.preparation_steps,
        "config": asdict(supervisor.config),
        "final_phase": supervisor.phase.value,
        "steps": len(rows),
        "discarded_chunk_actions": discarded_chunk_actions,
        "safety_clamps": safety_clamps,
        "maximum_wrist_rise_m": float(np.max(wrist[:, 2] - start_pose[0][2])),
        "wrist_overshoot_m": float(np.max(wrist[:, 2] - start_pose[0][2]) - target_rise),
        "object_following_ratio": float((obj[-1, 2] - initial_object_z) / (wrist[-1, 2] - start_pose[0][2])),
        "rollout": _summary(rows, initial_object_z),
        "strict_replay": asdict(strict),
        "video": str(args.output / "rollout.mp4") if args.video else None,
    }
    np.savez_compressed(
        args.output / "trace.npz",
        predicted_xyz=np.asarray(labels),
        phase=np.asarray(phases),
        **{f"rollout_{key}": _stack(rows, key) for key in rows[0]},
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
