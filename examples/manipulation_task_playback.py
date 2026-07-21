# -*- coding: utf-8 -*-
"""Environment viewer for the migrated single-arm manipulation tasks.

This demo does not solve the task, teleport objects, or drive an oracle robot
motion. It only runs the environment so task geometry, observations, rewards,
contacts, and reset placement can be inspected in the MuJoCo viewer.

Usage::

    python -m examples.manipulation_task_playback --task lift
    python -m examples.manipulation_task_playback --task stack
    python -m examples.manipulation_task_playback --task pick_place --object-id ycb:025_mug
    python -m examples.manipulation_task_playback --task nut_assembly --single-nut square_nut
    python -m examples.manipulation_task_playback --task push --object-id ycb:006_mustard_bottle
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import mujoco
import numpy as np
from mujoco import viewer
from PIL import Image

from source.cli.robot_config import add_robot_config_args
from source.runtime.pacing import RealtimePacer
from source.envs.manipulation import make_manipulation_env, registered_tasks
from source.viz.overlays import clear_markers, draw_label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Viewer for migrated single-arm manipulation environments."
    )
    parser.add_argument(
        "--task",
        choices=registered_tasks(),
        default="lift",
        help="Task environment to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reset seed controlling initial task placement.",
    )
    parser.add_argument(
        "--object-id",
        help="Catalogue object ID for Lift, PickPlace, or Push, for example ycb:025_mug.",
    )
    parser.add_argument(
        "--stack-object-ids",
        nargs=2,
        metavar=("TOP", "BOTTOM"),
        help="Two stack-compatible catalogue IDs.",
    )
    parser.add_argument(
        "--single-nut",
        choices=("square_nut", "round_nut"),
        help="Run NutAssembly with only the selected nut.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        help="Viewer sync frequency.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=20.0,
        help="Environment control frequency.",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Sample random actions instead of holding the reset action.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.20,
        help="Scale for sampled random actions around the reset action.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run as fast as possible instead of wall-clock realtime.",
    )
    parser.add_argument(
        "--print-status",
        action="store_true",
        help="Print reward and success once per second.",
    )
    parser.add_argument(
        "--snapshot",
        nargs="?",
        const="docs/manipulation_snapshot.png",
        metavar="PATH",
        help=(
            "Render one fixed-camera PNG and exit without opening the viewer. "
            "Default path when omitted: docs/manipulation_snapshot.png."
        ),
    )
    parser.add_argument(
        "--camera",
        default="agentview",
        help="Named MuJoCo camera used by --snapshot (default: agentview).",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Snapshot width in pixels (default: 640).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Snapshot height in pixels (default: 480).",
    )
    add_robot_config_args(parser)
    return parser.parse_args()


def _make_env(args: argparse.Namespace):
    if args.control_hz <= 0.0:
        raise ValueError(f"--control-hz must be positive, got {args.control_hz}.")
    if args.render_fps <= 0.0:
        raise ValueError(f"--render-fps must be positive, got {args.render_fps}.")
    if args.action_scale < 0.0:
        raise ValueError(f"--action-scale must be non-negative, got {args.action_scale}.")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("--image-width and --image-height must be positive.")
    if args.object_id is not None and args.task not in {"lift", "pick_place", "push"}:
        raise ValueError("--object-id is only valid with --task lift, pick_place, or push.")
    if args.stack_object_ids is not None and args.task != "stack":
        raise ValueError("--stack-object-ids is only valid with --task stack.")
    if args.single_nut is not None and args.task != "nut_assembly":
        raise ValueError("--single-nut is only valid with --task nut_assembly.")

    env_kwargs = dict(
        robot_config_path=getattr(args, "robot_config", None),
        arm_name=getattr(args, "arm_name", None),
        hand_name=getattr(args, "hand_name", None),
        base_name=getattr(args, "base_name", None),
        enable_tactile_sensors=False if getattr(args, "no_tactile", False) else None,
        control_mode="ik",
        render_mode=None,
        control_dt=1.0 / args.control_hz,
    )
    # Avoid passing None as an explicit robot-config override. This lets the
    # project's current robot profile provide values for omitted CLI options.
    env_kwargs = {key: value for key, value in env_kwargs.items() if value is not None}
    task_config: Dict[str, Any] = {"reward_shaping": True}
    if args.object_id is not None:
        task_config["object_id"] = args.object_id
    if args.stack_object_ids is not None:
        task_config["object_ids"] = tuple(args.stack_object_ids)
    if args.task == "nut_assembly" and args.single_nut is not None:
        task_config["single_nut"] = args.single_nut
    return make_manipulation_env(args.task, task_config=task_config, **env_kwargs)


def render_snapshot(args: argparse.Namespace) -> Path:
    """Reset once, render one named-camera frame, and close immediately."""
    env = _make_env(args)
    output = Path(args.snapshot).expanduser().resolve()
    try:
        env.reset(seed=args.seed)
        camera_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            args.camera,
        )
        if camera_id < 0:
            names = [
                mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, index)
                for index in range(env.model.ncam)
            ]
            raise ValueError(f"Unknown camera {args.camera!r}; available cameras: {names}")
        with mujoco.Renderer(
            env.model,
            height=args.image_height,
            width=args.image_width,
        ) as renderer:
            renderer.update_scene(env.data, camera=args.camera)
            frame = renderer.render()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(output)
        print(f"Snapshot saved: {output}")
        return output
    finally:
        env.close()


def _sample_action(
    env, base_action: np.ndarray, rng: np.random.Generator, scale: float
) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    noise = rng.uniform(-scale, scale, size=base_action.shape).astype(np.float32)
    action = base_action + noise
    finite = np.isfinite(low) & np.isfinite(high)
    action[finite] = np.clip(action[finite], low[finite], high[finite])
    return action.astype(np.float32)


def run_viewer(args: argparse.Namespace) -> None:
    env = _make_env(args)
    rng = np.random.default_rng(args.seed)
    obs, _info = env.reset(seed=args.seed)
    base_action = env.controller.current_action(env.model, env.data)
    action = base_action.copy()

    sim_start = float(env.data.time)
    pacer = RealtimePacer()
    pacer.reset(sim_start)
    last_print_second = -1
    reward = 0.0
    reward_info: Dict[str, Any] = {}

    try:
        with viewer.launch_passive(env.model, env.data) as handle:
            while handle.is_running():
                sim_elapsed = float(env.data.time) - sim_start

                if args.random_actions:
                    action = _sample_action(env, base_action, rng, args.action_scale)

                obs, reward, terminated, truncated, info = env.step(action)
                reward_info = {
                    key: value
                    for key, value in info.items()
                    if key.startswith("reward_") or key == "task_success"
                }

                clear_markers(handle)
                draw_label(
                    handle,
                    np.asarray([0.0, -0.32, 1.15], dtype=np.float32),
                    (
                        f"{args.task} env | reward {reward:.3f} | "
                        f"success {bool(info.get('task_success', False))}"
                    ),
                )
                handle.sync()

                if args.print_status and int(sim_elapsed) != last_print_second:
                    last_print_second = int(sim_elapsed)
                    print(
                        f"t={sim_elapsed:5.2f} reward={reward:.3f} "
                        f"terminated={terminated} truncated={truncated} info={reward_info}"
                    )

                if terminated or truncated:
                    obs, _info = env.reset(seed=args.seed)
                    base_action = env.controller.current_action(env.model, env.data)
                    action = base_action.copy()
                    sim_start = float(env.data.time)
                    pacer.reset(sim_start)
                    last_print_second = -1

                if not args.no_realtime:
                    pacer.sleep_until(float(env.data.time))
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    if args.snapshot is not None:
        render_snapshot(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
