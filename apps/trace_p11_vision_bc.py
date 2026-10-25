"""Evaluate retrained global or global-plus-ROI XYZ9 BC with exact replay."""

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

from apps.trace_action_chunk_bc import execution_prefix
from apps.trace_diffusion_rollout import _make_env, _reset_exact, _snapshot, _stack, _summary
from source.grasp_pipeline.trajectory import GraspTrajectory
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.imitation.dual_vision_bc import DualVisionActionChunkBC
from source.imitation.observation import real_observable_state
from source.imitation.vision_input import crop_normalized_rgb
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roi-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2011)
    return parser


def _tensor(image, device):
    return (
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
        .to(device)
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    env = _make_env(payload, args.max_steps + 30)
    env.model.vis.global_.offwidth = 1920
    env.model.vis.global_.offheight = 1080
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
    _, qpos, qvel, ctrl = load_replay_controls(args.manifest, env)
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    snapshot_payload = None
    if args.snapshot is not None:
        with np.load(args.snapshot, allow_pickle=False) as archive:
            snapshot_payload = {key: np.asarray(archive[key]) for key in archive.files}
        qpos = snapshot_payload["qpos"]
        qvel = snapshot_payload["qvel"]
        ctrl = snapshot_payload["ctrl"]
    observation = _reset_exact(env, qpos, qvel, ctrl)
    if snapshot_payload is not None and "qacc_warmstart" in snapshot_payload:
        env.data.qacc_warmstart[:] = snapshot_payload["qacc_warmstart"]
        mujoco.mj_forward(env.model, env.data)
        observation = env._get_observation()
    restored = {
        "qpos_max_abs_error": float(np.max(np.abs(env.data.qpos - qpos))),
        "qvel_max_abs_error": float(np.max(np.abs(env.data.qvel - qvel))),
        "ctrl_max_abs_error": float(np.max(np.abs(env.data.ctrl - ctrl))),
        "qacc_warmstart_max_abs_error": (
            None
            if snapshot_payload is None or "qacc_warmstart" not in snapshot_payload
            else float(np.max(np.abs(env.data.qacc_warmstart - snapshot_payload["qacc_warmstart"])))
        ),
    }
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    dual = checkpoint["policy_type"] == "dual_vision_action_chunk_bc"
    policy = (
        DualVisionActionChunkBC.from_checkpoint(checkpoint, args.device)
        if dual
        else ActionChunkBC.from_checkpoint(checkpoint, args.device)
    ).eval()
    roi = json.loads(args.roi_summary.read_text())["fixed_roi"]["normalized_xyxy"]
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset()
    fixed_orientation = adapter.measured_pose()[1]
    binding = env.task._require_bindings().objects["object"]
    initial_z = float(env.data.xpos[binding.body_id, 2])
    initial_qpos = env.data.qpos.copy()
    initial_qvel = env.data.qvel.copy()
    global_renderer = mujoco.Renderer(env.model, height=120, width=160)
    local_renderer = mujoco.Renderer(env.model, height=1080, width=1920) if dual else None
    writer = imageio.get_writer(args.output / "rollout.mp4", fps=20)
    rows, diagnostics, emitted, labels = [], [], [], []
    model_times, global_render_times, local_render_times, preprocess_times, ik_times = (
        [],
        [],
        [],
        [],
        [],
    )
    render_time_error = 0.0
    render_qpos_error = 0.0
    try:
        while len(rows) < args.max_steps:
            before_time = float(env.data.time)
            before_qpos = env.data.qpos.copy()
            started = time.perf_counter()
            global_renderer.update_scene(env.data, camera="agentview")
            global_rgb = global_renderer.render().copy()
            global_render_times.append(time.perf_counter() - started)
            local_rgb = None
            if dual:
                started = time.perf_counter()
                local_renderer.update_scene(env.data, camera="agentview")
                high = local_renderer.render().copy()
                local_render_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                local_rgb = crop_normalized_rgb(high, roi, output_width=96, output_height=256)
                preprocess_times.append(time.perf_counter() - started)
            render_time_error = max(render_time_error, abs(float(env.data.time) - before_time))
            render_qpos_error = max(
                render_qpos_error, float(np.max(np.abs(env.data.qpos - before_qpos)))
            )
            state = torch.from_numpy(real_observable_state(env))[None].to(args.device)
            global_tensor = _tensor(global_rgb, args.device)
            anchor = adapter.measured_pose()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            chunk = (
                (
                    policy.sample(global_tensor, _tensor(local_rgb, args.device), state)
                    if dual
                    else policy.sample(global_tensor, state)
                )[0]
                .cpu()
                .numpy()
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            model_times.append(time.perf_counter() - started)
            for chunk_index, label in enumerate(execution_prefix(chunk, args.action_steps)):
                if len(rows) >= args.max_steps:
                    break
                started = time.perf_counter()
                control, result = adapter.decode(
                    label, anchor, XYZ9, fixed_orientation_wxyz=fixed_orientation
                )
                ik_times.append(time.perf_counter() - started)
                control = np.clip(control, env.action_space.low, env.action_space.high)
                observation, _, terminated, truncated, _ = env.step(control)
                _ = observation
                rows.append(_snapshot(env, control, chunk_index == 0))
                diagnostics.append(asdict(result))
                emitted.append(control)
                labels.append(label)
                global_renderer.update_scene(env.data, camera="agentview")
                writer.append_data(global_renderer.render().copy())
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
    finally:
        writer.close()
        global_renderer.close()
        if local_renderer is not None:
            local_renderer.close()
        env.close()
    emitted = np.asarray(emitted, dtype=np.float32)
    trajectory = GraspTrajectory(
        object_id=str(payload["object_id"]),
        source_manifest=str(args.manifest.resolve()),
        start_stage="approach",
        action_mode="absolute_position",
        residual_actions=np.zeros((len(emitted), 1), np.float32),
        controls=emitted,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        success=False,
        episode_return=0.0,
        metadata={
            "control_dt": float(payload.get("metadata", {}).get("control_dt", 0.05)),
            "source_seed": int(payload.get("metadata", {}).get("source_seed", payload["seed"])),
            "p11_policy_type": checkpoint["policy_type"],
            "snapshot": None if args.snapshot is None else str(args.snapshot),
        },
    )
    generated_manifest = trajectory.save(args.output / "emitted_trajectory")
    strict = strict_replay_manifest(generated_manifest, profile=FINAL_PROFILE, use_cache=False)
    numeric = {
        name: np.asarray([row[name] for row in diagnostics])
        for name in ("position_error", "orientation_error", "success", "limit_violation")
    }
    for key in rows[0]:
        numeric[f"bc_{key}"] = _stack(rows, key)
    np.savez_compressed(
        args.output / "rollout_trace.npz",
        cartesian_action=np.asarray(labels),
        joint13=emitted,
        **numeric,
    )
    bc = _summary(rows, initial_z)
    render_total = np.asarray(global_render_times) + (
        np.asarray(local_render_times) if dual else 0.0
    )
    end_to_end = render_total + np.asarray(model_times)
    if dual:
        end_to_end = end_to_end + np.asarray(preprocess_times)
    summary = {
        "policy_type": checkpoint["policy_type"],
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "snapshot": None if args.snapshot is None else str(args.snapshot),
        "snapshot_restore": restored,
        "action_steps": args.action_steps,
        "attempts": 1,
        "bc": bc,
        "strict_replay": asdict(strict),
        "ik_success_count": int(np.sum(numeric["success"])),
        "ik_failure_count": int(np.sum(~numeric["success"])),
        "limit_violation_count": int(np.sum(numeric["limit_violation"])),
        "global_render_p50_ms": 1000 * float(np.percentile(global_render_times, 50)),
        "global_render_p95_ms": 1000 * float(np.percentile(global_render_times, 95)),
        "local_render_p50_ms": (
            None if not dual else 1000 * float(np.percentile(local_render_times, 50))
        ),
        "local_render_p95_ms": (
            None if not dual else 1000 * float(np.percentile(local_render_times, 95))
        ),
        "preprocess_p50_ms": (
            0.0 if not dual else 1000 * float(np.percentile(preprocess_times, 50))
        ),
        "preprocess_p95_ms": (
            0.0 if not dual else 1000 * float(np.percentile(preprocess_times, 95))
        ),
        "model_p50_ms": 1000 * float(np.percentile(model_times, 50)),
        "model_p95_ms": 1000 * float(np.percentile(model_times, 95)),
        "ik_p50_ms": 1000 * float(np.percentile(ik_times, 50)),
        "ik_p95_ms": 1000 * float(np.percentile(ik_times, 95)),
        "observation_to_action_p50_ms": 1000 * float(np.percentile(end_to_end, 50)),
        "observation_to_action_p95_ms": 1000 * float(np.percentile(end_to_end, 95)),
        "render_does_not_step_max_time_error": render_time_error,
        "render_does_not_change_qpos_max_error": render_qpos_error,
        "video": str(args.output / "rollout.mp4"),
        "emitted_manifest": str(generated_manifest),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
