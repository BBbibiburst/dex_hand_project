"""Evaluate frozen XYZ9 BC with bounded state/tactile Cartesian residuals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from apps.analyze_tactile_failure_warning import _groups
from apps.trace_action_chunk_bc import _inputs, execution_prefix
from apps.trace_diffusion_rollout import _make_env, _reset_exact
from source.grasp_pipeline.trajectory import GraspTrajectory
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.imitation.cartesian_residual import CartesianResidualMLP, tactile_region_features
from source.imitation.observation import real_observable_state
from source.sensors.tactile.dex_hand import SimpleBoxTactileSensor
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


def _trajectory(directory, payload, source, controls, qpos, qvel, variant):
    return GraspTrajectory(
        object_id=str(payload["object_id"]),
        source_manifest=str(source.resolve()),
        start_stage="approach",
        action_mode="absolute_position",
        residual_actions=np.zeros((len(controls), 1), np.float32),
        controls=np.asarray(controls, np.float32),
        initial_qpos=qpos,
        initial_qvel=qvel,
        success=False,
        episode_return=0.0,
        metadata={
            "control_dt": float(payload.get("metadata", {}).get("control_dt", 0.05)),
            "source_seed": int(payload.get("metadata", {}).get("source_seed", 0)),
            "p7_residual_variant": variant,
        },
    ).save(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        choices=("baseline", "fixed_rule", "state_residual", "tactile_residual"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2011)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    payload = json.loads(args.manifest.read_text())
    env = _make_env(payload, 380)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    _, qpos, qvel, ctrl = load_replay_controls(args.manifest, env)
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    bc_payload = torch.load(args.bc_checkpoint, map_location=args.device, weights_only=False)
    bc = ActionChunkBC.from_checkpoint(bc_payload, args.device).eval()
    residual = None
    if args.variant.endswith("residual"):
        residual = CartesianResidualMLP.from_checkpoint(
            torch.load(args.residual_checkpoint, map_location=args.device, weights_only=False),
            args.device,
        ).eval()
    fixed = np.asarray(bc_payload["fixed_orientation_wxyz"])
    observation = _reset_exact(env, qpos, qvel, ctrl)
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset()
    groups = _groups(SimpleBoxTactileSensor())
    renderer = mujoco.Renderer(env.model, height=120, width=160)
    writer = imageio.get_writer(args.output / "rollout.mp4", fps=20)
    controls = []
    residual_rows = []
    tactile_rows = []
    state_rows = []
    trigger_rows = []
    latencies = []
    consecutive = 0
    active = False
    trigger_step = None
    step = 0
    ik_failures = 0
    try:
        while step < 340:
            renderer.update_scene(env.data, camera="agentview")
            image = renderer.render().copy()
            anchor = adapter.measured_pose()
            state = real_observable_state(env)
            tactile = np.asarray(observation["tactile"], np.float32)
            features = tactile_region_features(tactile, groups)
            hand_norm = float(np.linalg.norm(state[-6:]))
            condition = (
                features[0] > 2.0 if args.variant == "tactile_residual" else hand_norm > 0.008
            )
            consecutive = consecutive + 1 if condition else 0
            if not active and consecutive >= 3:
                active = True
                trigger_step = step
            started = time.perf_counter()
            chunk = bc.sample(*_inputs(observation, image, state, args.device))[0].cpu().numpy()
            if args.variant == "fixed_rule" and active:
                res_chunk = np.tile(np.asarray([0, 0, 0.002], np.float32), (args.action_steps, 1))
            elif residual is not None and active:
                model_input = state
                if residual.config.use_tactile:
                    model_input = np.r_[state, features]
                res_chunk = (
                    residual(
                        torch.from_numpy(model_input.astype(np.float32))
                        .unsqueeze(0)
                        .to(args.device)
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                res_chunk = np.zeros((args.action_steps, 3), np.float32)
            latencies.append(time.perf_counter() - started)
            for index, label in enumerate(execution_prefix(chunk, args.action_steps)):
                if step >= 340:
                    break
                final = label.copy()
                final[:3] += res_chunk[index]
                control, ik = adapter.decode(final, anchor, XYZ9, fixed_orientation_wxyz=fixed)
                if not ik.success:
                    ik_failures += 1
                    final = label.copy()
                    control, ik = adapter.decode(final, anchor, XYZ9, fixed_orientation_wxyz=fixed)
                observation, _, terminated, truncated, _ = env.step(
                    np.clip(control, env.action_space.low, env.action_space.high)
                )
                controls.append(control)
                residual_rows.append(res_chunk[index])
                tactile_rows.append(features)
                state_rows.append(state)
                trigger_rows.append(active)
                step += 1
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
    manifest = _trajectory(
        args.output / "trajectory", payload, args.manifest, controls, qpos, qvel, args.variant
    )
    strict = strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=False)
    np.savez_compressed(
        args.output / "trace.npz",
        controls=np.asarray(controls),
        residual_xyz=np.asarray(residual_rows),
        tactile_features=np.asarray(tactile_rows),
        state=np.asarray(state_rows),
        trigger=np.asarray(trigger_rows),
    )
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "uses_tactile": args.variant == "tactile_residual",
        "uses_recovery": args.variant != "baseline",
        "trigger_step": trigger_step,
        "trigger_count": int(np.sum(np.diff(np.r_[False, trigger_rows]) > 0)),
        "maximum_residual_m": float(np.max(np.abs(residual_rows))),
        "ik_failures": ik_failures,
        "control_p50_ms": 1000 * float(np.percentile(latencies, 50)),
        "control_p95_ms": 1000 * float(np.percentile(latencies, 95)),
        "strict_replay": asdict(strict),
        "video": str(args.output / "rollout.mp4"),
        "trajectory_manifest": str(manifest),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
