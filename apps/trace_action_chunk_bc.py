"""Trace expert and deterministic action-chunk BC on one exact verified reset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from apps.trace_diffusion_rollout import _make_env, _reset_exact, _snapshot, _stack, _summary
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.observation import real_observable_state
from source.verification.strict_replay import load_replay_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2011)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=120)
    return parser


def _inputs(observation, image, state, device):
    image_tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    state_tensor = torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0).to(device)
    return image_tensor, state_tensor


def execution_prefix(chunk: np.ndarray, action_steps: int) -> np.ndarray:
    """Return each predicted timestep once, in order, for one control chunk."""

    values = np.asarray(chunk)
    if values.ndim != 2:
        raise ValueError("Predicted action chunk must be 2-D.")
    if action_steps <= 0 or action_steps > len(values):
        raise ValueError("action_steps must lie in [1, prediction_horizon].")
    return values[:action_steps]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action_steps <= 0:
        raise ValueError("--action-steps must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
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
    policy = ActionChunkBC.from_checkpoint(checkpoint, args.device).eval()
    observation = _reset_exact(env, initial_qpos, initial_qvel, initial_ctrl)
    renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
    import imageio.v2 as imageio

    video_path = args.output / "bc_rollout.mp4"
    writer = imageio.get_writer(video_path, fps=20)
    rows, latencies, boundary_jumps = [], [], []
    previous_target = None
    try:
        while len(rows) < args.max_steps:
            renderer.update_scene(env.data, camera="agentview")
            image = renderer.render().copy()
            inputs = _inputs(observation, image, real_observable_state(env), args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            chunk = policy.sample(*inputs)[0].cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            if previous_target is not None:
                boundary_jumps.append(float(np.linalg.norm(chunk[0] - previous_target)))
            terminated = truncated = False
            for chunk_index, predicted in enumerate(execution_prefix(chunk, args.action_steps)):
                if len(rows) >= args.max_steps:
                    break
                target = np.clip(predicted, env.action_space.low, env.action_space.high)
                observation, _, terminated, truncated, _ = env.step(target)
                rows.append(_snapshot(env, target, chunk_index == 0))
                previous_target = target.copy()
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
    for prefix, values in (("expert", expert_rows), ("bc", rows)):
        for key in values[0]:
            arrays[f"{prefix}_{key}"] = _stack(values, key)
    np.savez_compressed(args.output / "rollout_trace.npz", **arrays)

    import matplotlib.pyplot as plt

    boundaries = np.flatnonzero(_stack(rows, "chunk_start"))
    figure, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    for joint in range(7):
        axis = axes.flat[joint]
        axis.plot(_stack(expert_rows, "arm_target")[:, joint], label="expert")
        axis.plot(_stack(rows, "arm_target")[:, joint], label="BC", alpha=0.85)
        for boundary in boundaries:
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
    axes[0].plot(_stack(rows, "wrist_distance"), label="BC")
    axes[0].set(xlabel="Control step", ylabel="Wrist-to-object distance (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    expert_wrist, bc_wrist = _stack(expert_rows, "wrist"), _stack(rows, "wrist")
    axes[1].plot(expert_wrist[:, 0], expert_wrist[:, 2], label="expert")
    axes[1].plot(bc_wrist[:, 0], bc_wrist[:, 2], label="BC")
    axes[1].set(xlabel="Wrist x (m)", ylabel="Wrist z (m)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output / "approach_comparison.png", dpi=180)
    plt.close(figure)

    bc_summary = _summary(rows, initial_object_z)
    summary = {
        "policy": "deterministic_action_chunk_bc",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "seed": args.seed,
        "prediction_horizon": policy.config.horizon,
        "observation_horizon": 1,
        "network_image_shape": [3, args.image_height, args.image_width],
        "action_horizon": args.action_steps,
        "expert": _summary(expert_rows, initial_object_z),
        "bc": bc_summary,
        "chunk_boundary_jump_l2_mean": float(np.mean(boundary_jumps)) if boundary_jumps else None,
        "inference_p50_ms": 1000.0 * float(np.percentile(latencies, 50)),
        "inference_p95_ms": 1000.0 * float(np.percentile(latencies, 95)),
        "video": str(video_path),
        "strict_success": bool(bc_summary["maximum_lift_m"] > 0.04),
        "note": "strict_success field is task-threshold screening; manifest-level strict replay applies only to saved expert trajectories",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
