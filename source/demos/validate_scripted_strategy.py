"""Interactively validate every phase of a scripted manipulation strategy.

The simulation pauses after each phase transition. Press SPACE in the MuJoCo
viewer to allow the next phase, or Q to stop. No dataset is created.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
from mujoco import viewer

from source.demos.common import add_robot_config_args
from source.demos.strategies import create_strategy, registered_strategies
from source.envs.manipulation import make_manipulation_env
from source.viz.overlays import (
    clear_markers,
    draw_label,
    draw_line_marker,
    draw_pose_frame,
    draw_sphere_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=registered_strategies(), default="lift")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--viewer-speed",
        type=float,
        default=1.0,
        help="Playback speed relative to wall-clock time (default: 1.0).",
    )
    add_robot_config_args(parser)
    return parser.parse_args()


def _make_env(args):
    overrides = {
        "robot_config_path": getattr(args, "robot_config", None),
        "arm_name": getattr(args, "arm_name", None),
        "hand_name": getattr(args, "hand_name", None),
        "base_name": getattr(args, "base_name", None),
        "control_mode": "ik",
        "control_dt": 1.0 / args.fps,
        "episode_length": args.max_steps,
        "enable_tactile_sensors": not getattr(args, "no_tactile", False),
        "render_mode": None,
    }
    return make_manipulation_env(
        args.task,
        task_config={"reward_shaping": True, "terminate_on_success": False},
        **{key: value for key, value in overrides.items() if value is not None},
    )


def _format_vector(values: np.ndarray, *, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _draw_state(
    handle,
    env,
    observation,
    action,
    info,
    strategy,
    *,
    status: str,
    step: int,
) -> None:
    target_position = np.asarray(info.get("ik_target_position", action[:3]), dtype=np.float64)
    target_quaternion = np.asarray(info.get("ik_target_quat", action[3:7]), dtype=np.float64)
    hand_size = env.controller.hand_controller.action_size
    fallback_hand = action[-hand_size:] if hand_size else np.zeros(0, dtype=np.float32)
    hand_target = np.asarray(
        info.get("hand_position_target", fallback_hand),
        dtype=np.float64,
    )

    clear_markers(handle)
    draw_pose_frame(
        handle,
        target_position,
        target_quaternion,
        axis_length=0.09,
        line_width=0.005,
        label="COMMAND",
    )
    midpoint_getter = getattr(strategy, "grasp_midpoint", None)
    if midpoint_getter is not None:
        midpoint = np.asarray(midpoint_getter(env), dtype=np.float64)
        draw_sphere_marker(
            handle,
            midpoint,
            radius=0.018,
            rgba=(1.0, 0.1, 0.85, 0.95),
        )
        draw_label(
            handle,
            midpoint + np.asarray([0.0, 0.0, 0.025]),
            "MIDPOINT",
            rgba=(1.0, 0.1, 0.85, 1.0),
        )
        object_position = observation.get("object_pos")
        if object_position is not None:
            draw_line_marker(
                handle,
                midpoint,
                np.asarray(object_position, dtype=np.float64),
                width=0.004,
                rgba=(1.0, 0.4, 0.8, 0.9),
            )
    draw_label(
        handle,
        np.asarray([0.0, -0.32, 1.15], dtype=np.float32),
        strategy.phase_prompt,
    )
    draw_label(
        handle,
        np.asarray([0.0, -0.32, 1.10], dtype=np.float32),
        status,
        rgba=(1.0, 0.85, 0.1, 1.0),
    )
    draw_label(
        handle,
        np.asarray([0.0, -0.32, 1.05], dtype=np.float32),
        f"xyz {_format_vector(target_position)} | quat {_format_vector(target_quaternion)}",
    )
    draw_label(
        handle,
        np.asarray([0.0, -0.32, 1.00], dtype=np.float32),
        f"hand {_format_vector(hand_target, precision=4)} | step {step}",
        rgba=(0.3, 1.0, 0.8, 1.0),
    )
    handle.sync()


def run(args) -> None:
    if args.max_steps <= 0 or args.fps <= 0 or args.viewer_speed <= 0:
        raise ValueError("--max-steps, --fps, and --viewer-speed must be positive.")

    env = _make_env(args)
    strategy = create_strategy(args.task)
    observation, info = env.reset(seed=args.seed)
    confirm = threading.Event()
    stop = threading.Event()

    def on_key(keycode: int) -> None:
        if keycode == 32:  # SPACE
            confirm.set()
        elif keycode in (ord("Q"), ord("q")):
            stop.set()
            confirm.set()

    handle = viewer.launch_passive(env.model, env.data, key_callback=on_key)
    action = env.controller.current_ik_action(env.model, env.data)
    period = 1.0 / (args.fps * args.viewer_speed)
    deadline = time.monotonic()
    try:
        print("Phase validation: SPACE confirms the next phase; Q quits.")
        previous_phase = strategy.phase_name
        previous_index = strategy.phase_index
        for step in range(args.max_steps):
            if stop.is_set() or not handle.is_running():
                break

            action, _ = strategy.tick(observation, info, step, env)
            observation, _, terminated, truncated, info = env.step(action)
            _draw_state(
                handle,
                env,
                observation,
                action,
                info,
                strategy,
                status=f"RUNNING | speed {args.viewer_speed:.2f}x",
                step=step + 1,
            )

            if strategy.phase_name != previous_phase:
                advanced = strategy.phase_index > previous_index
                transition = (
                    f"COMPLETED {previous_phase} -> {strategy.phase_name}"
                    if advanced
                    else f"FAILED {previous_phase} -> RESTART"
                )
                print(f"{transition}; press SPACE to continue or Q to quit.")
                confirm.clear()
                while handle.is_running() and not confirm.is_set():
                    _draw_state(
                        handle,
                        env,
                        observation,
                        action,
                        info,
                        strategy,
                        status=f"{transition} | PAUSED: press SPACE",
                        step=step + 1,
                    )
                    time.sleep(0.05)
                deadline = time.monotonic()
                previous_phase = strategy.phase_name
                previous_index = strategy.phase_index

            if strategy.finished:
                print("Strategy validation completed.")
                break
            if terminated or truncated or strategy.aborted:
                print(
                    f"Validation stopped: terminated={terminated}, "
                    f"truncated={truncated}, aborted={strategy.aborted}."
                )
                break
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        handle.close()
        env.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
