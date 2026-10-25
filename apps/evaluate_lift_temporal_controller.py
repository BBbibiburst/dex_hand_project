"""Evaluate a Lift-only Cartesian controller from saved Hold snapshots."""

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
from source.imitation.lift_temporal import LiftTemporalController
from source.imitation.observation import real_observable_state
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import strict_replay_manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--hand-mode", choices=("hold", "expert"), default="hold")
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--root-id", type=int)
    parser.add_argument("--snapshot", type=Path)
    return parser


def _append(history: deque, value):
    history.append(np.asarray(value, dtype=np.float32))


def evaluate_root(args, model, row):
    source = Path(row["manifest"])
    payload = json.loads(source.read_text())
    with np.load(row["path"]) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    env = _make_env(payload, args.max_steps + 30)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
    if args.snapshot is None:
        snapshot = data
    else:
        with np.load(args.snapshot) as archive:
            snapshot = {key: np.asarray(archive[key]) for key in archive.files}
    observation = _reset_exact(env, snapshot["qpos"] if "qpos" in snapshot else snapshot["hold_qpos"], snapshot["qvel"] if "qvel" in snapshot else snapshot["hold_qvel"], snapshot["ctrl"] if "ctrl" in snapshot else snapshot["hold_ctrl"])
    _ = observation
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset()
    _, fixed_orientation = adapter.measured_pose()
    binding = env.task._require_bindings().objects["object"]
    initial_z = float(env.data.xpos[binding.body_id, 2])
    state_history = deque(maxlen=model.config.history)
    action_history = deque(maxlen=model.config.history)
    state = real_observable_state(env)
    for _ in range(model.config.history):
        _append(state_history, state)
        _append(action_history, np.zeros(3, dtype=np.float32))
    destination = args.output / f"root_{row['root_id']:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    writer = None
    renderer = None
    if args.video:
        renderer = mujoco.Renderer(env.model, height=240, width=320)
        writer = imageio.get_writer(destination / "rollout.mp4", fps=20)
    rows, controls, labels, ik = [], [], [], []
    try:
        while len(rows) < args.max_steps:
            states = torch.from_numpy(np.stack(state_history))[None].to(args.device)
            actions = torch.from_numpy(np.stack(action_history))[None].to(args.device)
            with torch.inference_mode():
                chunk = model(states, actions)[0].cpu().numpy()
            anchor = adapter.measured_pose()
            for chunk_index, xyz in enumerate(chunk[: args.action_steps]):
                if len(rows) >= args.max_steps:
                    break
                step = len(rows)
                hand = (
                    data["expert_hand"][min(step, len(data["expert_hand"]) - 1), chunk_index]
                    if args.hand_mode == "expert"
                    else env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
                )
                control, result = adapter.decode(
                    np.r_[xyz, hand], anchor, XYZ9, fixed_orientation_wxyz=fixed_orientation
                )
                control = np.clip(control, env.action_space.low, env.action_space.high)
                _, _, terminated, truncated, _ = env.step(control)
                rows.append(_snapshot(env, control, chunk_index == 0))
                controls.append(control)
                labels.append(xyz)
                ik.append(asdict(result))
                _append(state_history, real_observable_state(env))
                _append(action_history, xyz)
                if writer is not None:
                    renderer.update_scene(env.data, camera="agentview")
                    writer.append_data(renderer.render().copy())
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.close()
            renderer.close()
        env.close()
    controls_a = np.asarray(controls, dtype=np.float32)
    trajectory = GraspTrajectory(
        object_id=payload["object_id"],
        source_manifest=str(source.resolve()),
        start_stage="hold",
        action_mode="absolute_position",
        residual_actions=np.zeros((len(controls_a), 1), np.float32),
        controls=controls_a,
        initial_qpos=env.data.qpos.copy() if not rows else (snapshot["qpos"] if "qpos" in snapshot else snapshot["hold_qpos"]),
        initial_qvel=snapshot["qvel"] if "qvel" in snapshot else snapshot["hold_qvel"],
        success=False,
        episode_return=0.0,
        metadata={"control_dt": 0.05, "source_seed": int(payload.get("metadata", {}).get("source_seed", payload["seed"])), "p8_lift_only": True},
    )
    manifest = trajectory.save(destination / "emitted_trajectory")
    strict = strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=False)
    np.savez_compressed(destination / "trace.npz", predicted_xyz=np.asarray(labels), controls=controls_a, **{f"rollout_{key}": _stack(rows, key) for key in rows[0]})
    summary = {"root_id": row["root_id"], "split": row["split"], "hand_mode": args.hand_mode, "rollout": _summary(rows, initial_z), "strict_replay": asdict(strict), "ik_failures": sum(not item["success"] for item in ik), "video": str(destination / "rollout.mp4") if args.video else None}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.dataset / "manifest.json").read_text())
    selected = [row for row in metadata["rows"] if args.split == "all" or row["split"] == args.split]
    if args.root_id is not None:
        selected = [row for row in selected if row["root_id"] == args.root_id]
    if not selected:
        raise ValueError("No dataset roots match the requested selection.")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = LiftTemporalController.from_checkpoint(checkpoint, args.device).eval()
    results = []
    for row in selected:
        result = evaluate_root(args, model, row)
        results.append(result)
        print(f"root={row['root_id']:02d} strict={result['strict_replay']['verification_status']} lift={1000*result['strict_replay']['max_lift']:.2f}mm", flush=True)
    summary = {"checkpoint": str(args.checkpoint), "split": args.split, "hand_mode": args.hand_mode, "successes": sum(result["strict_replay"]["success"] for result in results), "trials": len(results), "results": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
