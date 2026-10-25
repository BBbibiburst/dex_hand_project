"""Run shared-anchor Cartesian BC and verify the emitted Joint13 controls."""

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

from apps.trace_action_chunk_bc import _inputs, execution_prefix
from apps.trace_diffusion_rollout import _make_env, _reset_exact, _snapshot, _stack, _summary
from source.grasp_pipeline.trajectory import GraspTrajectory
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.cartesian_action import POSE12, XYZ9, ContinuousCartesianAdapter
from source.imitation.observation import real_observable_state
from source.imitation.vision_input import resize_rgb_uint8
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument(
        "--render-width",
        type=int,
        help="Source render width; defaults to --image-width for backwards compatibility.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        help="Source render height; defaults to --image-height for backwards compatibility.",
    )
    parser.add_argument("--seed", type=int, default=2011)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    render_width = args.render_width or args.image_width
    render_height = args.render_height or args.image_height
    if min(args.image_width, args.image_height, render_width, render_height) <= 0:
        raise ValueError("image and render dimensions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    env = _make_env(payload, args.max_steps + 30)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    expert, qpos, qvel, ctrl = load_replay_controls(args.manifest, env)
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    mode = str(checkpoint["action_schema"]).split("_")[0]
    if mode not in (XYZ9, POSE12):
        raise ValueError(f"Unsupported schema {checkpoint['action_schema']}")
    policy = ActionChunkBC.from_checkpoint(checkpoint, args.device).eval()
    torch.manual_seed(args.seed)
    observation = _reset_exact(env, qpos, qvel, ctrl)
    binding = env.task._require_bindings().objects["object"]
    initial_z = float(env.data.xpos[binding.body_id, 2])
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset()
    fixed = checkpoint.get("fixed_orientation_wxyz")
    fixed = None if fixed is None else np.asarray(fixed, dtype=np.float64)
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), render_width)
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), render_height)
    renderer = mujoco.Renderer(env.model, height=render_height, width=render_width)
    writer = imageio.get_writer(args.output / "bc_rollout.mp4", fps=20)
    rows = []
    diagnostics = []
    emitted = []
    labels = []
    latencies = []
    render_latencies = []
    preprocess_latencies = []
    ik_latencies = []
    boundary = []
    previous = None
    try:
        while len(rows) < args.max_steps:
            render_started = time.perf_counter()
            renderer.update_scene(env.data, camera="agentview")
            rendered = renderer.render().copy()
            render_latencies.append(time.perf_counter() - render_started)
            preprocess_started = time.perf_counter()
            image = resize_rgb_uint8(
                rendered, width=args.image_width, height=args.image_height
            )
            preprocess_latencies.append(time.perf_counter() - preprocess_started)
            anchor = adapter.measured_pose()
            inputs = _inputs(observation, image, real_observable_state(env), args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            chunk = policy.sample(*inputs)[0].cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            for chunk_index, label in enumerate(execution_prefix(chunk, args.action_steps)):
                if len(rows) >= args.max_steps:
                    break
                ik_started = time.perf_counter()
                control, result = adapter.decode(label, anchor, mode, fixed_orientation_wxyz=fixed)
                ik_latencies.append(time.perf_counter() - ik_started)
                control = np.clip(control, env.action_space.low, env.action_space.high)
                if previous is not None and chunk_index == 0:
                    boundary.append(float(np.linalg.norm(control - previous)))
                observation, _, terminated, truncated, _ = env.step(control)
                rows.append(_snapshot(env, control, chunk_index == 0))
                diagnostics.append(asdict(result))
                emitted.append(control)
                labels.append(label)
                previous = control.copy()
                renderer.update_scene(env.data, camera="agentview")
                video_frame = resize_rgb_uint8(
                    renderer.render().copy(), width=min(render_width, 640), height=min(render_height, 480)
                )
                writer.append_data(video_frame)
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
    finally:
        writer.close()
        renderer.close()
        env.close()
    emitted = np.asarray(emitted, np.float32)
    trajectory = GraspTrajectory(
        object_id=str(payload["object_id"]),
        source_manifest=str(args.manifest.resolve()),
        start_stage="approach",
        action_mode="absolute_position",
        residual_actions=np.zeros((len(emitted), 1), np.float32),
        controls=emitted,
        initial_qpos=qpos,
        initial_qvel=qvel,
        success=False,
        episode_return=0.0,
        metadata={
            "control_dt": float(payload.get("metadata", {}).get("control_dt", 0.05)),
            "source_seed": int(
                payload.get("metadata", {}).get("source_seed", payload.get("seed", 0))
            ),
            "p6_bc_action_schema": checkpoint["action_schema"],
        },
    )
    generated_manifest = trajectory.save(args.output / "emitted_trajectory")
    strict = strict_replay_manifest(generated_manifest, profile=FINAL_PROFILE, use_cache=False)
    numeric = {
        name: np.asarray([row[name] for row in diagnostics])
        for name in (
            "position_error",
            "orientation_error",
            "joint_delta",
            "minimum_singular_value",
            "success",
            "limit_violation",
        )
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
    summary = {
        "mode": mode,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "action_steps": args.action_steps,
        "render_resolution_wh": [render_width, render_height],
        "model_input_resolution_wh": [args.image_width, args.image_height],
        "preprocessing": (
            "identity"
            if (render_width, render_height) == (args.image_width, args.image_height)
            else "Pillow bilinear RGB resize"
        ),
        "bc": bc,
        "strict_replay": asdict(strict),
        "ik_success_count": int(np.sum(numeric["success"])),
        "ik_failure_count": int(np.sum(~numeric["success"])),
        "limit_violation_count": int(np.sum(numeric["limit_violation"])),
        "ik_position_error_max_m": float(numeric["position_error"].max()),
        "ik_orientation_error_max_rad": float(numeric["orientation_error"].max()),
        "chunk_boundary_joint13_jump_mean": float(np.mean(boundary)) if boundary else None,
        "inference_ik_p50_ms": 1000 * float(np.percentile(latencies, 50)),
        "inference_ik_p95_ms": 1000 * float(np.percentile(latencies, 95)),
        "render_p50_ms": 1000 * float(np.percentile(render_latencies, 50)),
        "render_p95_ms": 1000 * float(np.percentile(render_latencies, 95)),
        "preprocess_p50_ms": 1000 * float(np.percentile(preprocess_latencies, 50)),
        "preprocess_p95_ms": 1000 * float(np.percentile(preprocess_latencies, 95)),
        "ik_p50_ms": 1000 * float(np.percentile(ik_latencies, 50)),
        "ik_p95_ms": 1000 * float(np.percentile(ik_latencies, 95)),
        "video": str(args.output / "bc_rollout.mp4"),
        "emitted_manifest": str(generated_manifest),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    wrist = _stack(rows, "wrist")
    obj = _stack(rows, "object")
    contacts = _stack(rows, "contact")
    axes[0].plot(np.linalg.norm(wrist - obj, axis=1), label="wrist-object distance")
    axes[1].plot(contacts.astype(float), label="contact")
    axes[1].plot(obj[:, 2] - initial_z, label="object lift (m)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("Control step")
    fig.tight_layout()
    fig.savefig(args.output / "wrist_contact_lift.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
