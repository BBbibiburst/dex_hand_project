"""Render aligned global/ROI observations and XYZ9 chunks from verified roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from apps.trace_diffusion_rollout import _make_env, _reset_exact
from source.grasping.contracts import DemonstrationEpisode
from source.imitation.cartesian_action import ContinuousCartesianAdapter
from source.imitation.observation import real_observable_state
from source.imitation.vision_input import crop_normalized_rgb
from source.verification.strict_replay import load_replay_controls


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--roi-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--local-width", type=int, default=96)
    parser.add_argument("--local-height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2011)
    return parser


def _chunks(env, controls, states, horizon):
    adapter = ContinuousCartesianAdapter(env)
    result = []
    for index, state in enumerate(states):
        env.data.qpos[adapter.arm.qpos_addrs] = state["qpos"][adapter.arm.qpos_addrs]
        env.data.qvel[:] = state["qvel"]
        env.data.ctrl[:] = state["ctrl"]
        mujoco.mj_forward(env.model, env.data)
        anchor, _ = adapter.measured_pose()
        chunk = []
        for offset in range(horizon):
            action = controls[min(index + offset, len(controls) - 1)]
            target, _ = adapter.pose_for_joint_target(action[:7])
            chunk.append(np.r_[target - anchor, action[7:]])
        result.append(chunk)
    return np.asarray(result, dtype=np.float32)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    previews = args.output / "previews"
    previews.mkdir(exist_ok=True)
    source = json.loads(args.source_a.read_text())
    roi = json.loads(args.roi_summary.read_text())["fixed_roi"]["normalized_xyxy"]
    rows = []
    for root_id, manifest_value in enumerate(source["accepted_manifests"]):
        started = time.perf_counter()
        manifest = Path(manifest_value)
        payload = json.loads(manifest.read_text())
        episode = DemonstrationEpisode.load(manifest)
        env = _make_env(payload, int(payload["frames"]) + 30)
        env.model.vis.global_.offwidth = 1920
        env.model.vis.global_.offheight = 1080
        env.reset(seed=int(episode.metadata.get("source_seed", payload["seed"])))
        controls, qpos, qvel, ctrl = load_replay_controls(manifest, env)
        if ctrl is None:
            ctrl = env.data.ctrl.copy()
        _reset_exact(env, qpos, qvel, ctrl)
        global_renderer = mujoco.Renderer(env.model, height=120, width=160)
        local_renderer = mujoco.Renderer(env.model, height=1080, width=1920)
        global_frames, local_frames, states, state_vectors = [], [], [], []
        max_time_error = 0.0
        max_qpos_error = 0.0
        try:
            for control in np.asarray(controls, dtype=np.float32):
                before_time = float(env.data.time)
                before_qpos = env.data.qpos.copy()
                global_renderer.update_scene(env.data, camera="agentview")
                global_frames.append(global_renderer.render().copy())
                local_renderer.update_scene(env.data, camera="agentview")
                high = local_renderer.render().copy()
                local_frames.append(
                    crop_normalized_rgb(
                        high,
                        roi,
                        output_width=args.local_width,
                        output_height=args.local_height,
                    )
                )
                max_time_error = max(max_time_error, abs(float(env.data.time) - before_time))
                max_qpos_error = max(
                    max_qpos_error, float(np.max(np.abs(env.data.qpos - before_qpos)))
                )
                states.append(
                    {
                        "qpos": env.data.qpos.copy(),
                        "qvel": env.data.qvel.copy(),
                        "ctrl": env.data.ctrl.copy(),
                    }
                )
                state_vectors.append(real_observable_state(env))
                env.step(control)
        finally:
            global_renderer.close()
            local_renderer.close()
        actions = _chunks(env, np.asarray(controls, np.float32), states, args.horizon)
        env.close()
        stages = np.asarray(episode.arrays["stage"])[-len(controls) :]
        output_path = (args.output / f"root_{root_id:02d}.npz").resolve()
        np.savez_compressed(
            output_path,
            global_rgb=np.asarray(global_frames, dtype=np.uint8),
            local_rgb=np.asarray(local_frames, dtype=np.uint8),
            state=np.asarray(state_vectors, dtype=np.float32),
            action=actions,
            stage=stages.astype(np.int16),
        )
        codes = episode.metadata["stage_codes"]
        canvas = Image.new("RGB", (640, 4 * 256), "white")
        draw = ImageDraw.Draw(canvas)
        for panel, stage_name in enumerate(("approach", "close", "hold", "lift")):
            indices = np.flatnonzero(stages == int(codes[stage_name]))
            index = int(indices[len(indices) // 2])
            global_image = Image.fromarray(global_frames[index]).resize((320, 240))
            local_image = Image.fromarray(local_frames[index]).resize((90, 240))
            canvas.paste(global_image, (0, panel * 256))
            canvas.paste(local_image, (360, panel * 256))
            draw.text((470, panel * 256 + 100), stage_name, fill="black")
        canvas.save(previews / f"root_{root_id:02d}.jpg", quality=92)
        split = "train" if root_id < 7 else "validation" if root_id == 7 else "test"
        row = {
            "root_id": root_id,
            "split": split,
            "manifest": str(manifest),
            "strict_verified_source": bool(payload["success"]),
            "path": str(output_path),
            "samples": len(controls),
            "render_does_not_step_max_time_error": max_time_error,
            "render_does_not_change_qpos_max_error": max_qpos_error,
            "runtime_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(f"root={root_id:02d} split={split} frames={len(controls)}", flush=True)
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "action_schema": "xyz9_shared_anchor_v1",
        "state_schema": "actuator_position_velocity_command_v1",
        "global_rgb": {"shape_hwc": [120, 160, 3], "source": "direct agentview render"},
        "local_rgb": {
            "shape_hwc": [args.local_height, args.local_width, 3],
            "source": "fixed normalized ROI of direct 1920x1080 agentview render",
            "normalized_xyxy": roi,
            "interpolation": "Pillow bilinear",
        },
        "online_privileged_inputs": [],
        "pre_action_alignment": True,
        "horizon": args.horizon,
        "roots": len(rows),
        "rows": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
