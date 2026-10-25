"""Trace a trained official LeRobot ACT policy on one exact verified reset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from apps.trace_diffusion_rollout import _make_env, _reset_exact, _snapshot, _stack, _summary
from source.imitation.observation import real_observable_state
from source.verification.strict_replay import load_replay_controls


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2011)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    env = _make_env(payload, args.max_steps)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    controls, qpos, qvel, ctrl = load_replay_controls(args.manifest, env)
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    binding = env.task._require_bindings().objects["object"]
    _reset_exact(env, qpos, qvel, ctrl)
    initial_z = float(env.data.xpos[binding.body_id, 2])
    expert_rows = []
    for action in controls[: args.max_steps]:
        env.step(action)
        expert_rows.append(_snapshot(env, action, False))
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy(checkpoint["config"]).to(args.device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    state_mean = checkpoint["state_mean"].to(args.device)
    state_std = checkpoint["state_std"].to(args.device)
    action_mean = checkpoint["action_mean"].to(args.device)
    action_std = checkpoint["action_std"].to(args.device)
    observation = _reset_exact(env, qpos, qvel, ctrl)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    import imageio.v2 as imageio

    video = args.output / "act_rollout.mp4"
    writer = imageio.get_writer(video, fps=20)
    rows = []
    latencies = []
    jumps = []
    previous = None
    try:
        while len(rows) < args.max_steps:
            renderer.update_scene(env.data, camera="agentview")
            image = renderer.render().copy()
            image_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                .div(255)
                .unsqueeze(0)
                .to(args.device)
            )
            state = torch.from_numpy(real_observable_state(env)).unsqueeze(0).to(args.device)
            batch = {
                "observation.images.agentview": image_tensor,
                "observation.state": (state - state_mean) / state_std,
            }
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(batch)[0]
                chunk = (normalized * action_std + action_mean).cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            if previous is not None:
                jumps.append(float(np.linalg.norm(chunk[0] - previous)))
            terminated = truncated = False
            for index, predicted in enumerate(chunk[: args.action_steps]):
                target = np.clip(predicted, env.action_space.low, env.action_space.high)
                observation, _, terminated, truncated, _ = env.step(target)
                rows.append(_snapshot(env, target, index == 0))
                previous = target.copy()
                renderer.update_scene(env.data, camera="agentview")
                writer.append_data(renderer.render().copy())
                if len(rows) >= args.max_steps or terminated or truncated:
                    break
            if terminated or truncated:
                break
    finally:
        writer.close()
        renderer.close()
        env.close()
    arrays = {}
    for prefix, values in (("expert", expert_rows), ("act", rows)):
        for key in values[0]:
            arrays[f"{prefix}_{key}"] = _stack(values, key)
    np.savez_compressed(args.output / "rollout_trace.npz", **arrays)
    act_summary = _summary(rows, initial_z)
    summary = {
        "policy": "official_lerobot_act_0.4.4",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "seed": args.seed,
        "prediction_horizon": checkpoint["config"].chunk_size,
        "action_horizon": args.action_steps,
        "expert": _summary(expert_rows, initial_z),
        "act": act_summary,
        "chunk_boundary_jump_l2_mean": float(np.mean(jumps)) if jumps else None,
        "inference_p50_ms": 1000 * float(np.percentile(latencies, 50)),
        "inference_p95_ms": 1000 * float(np.percentile(latencies, 95)),
        "video": str(video),
        "strict_success": bool(act_summary["maximum_lift_m"] > 0.04),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
