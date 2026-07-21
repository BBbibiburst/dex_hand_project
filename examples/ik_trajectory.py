# -*- coding: utf-8 -*-
"""Standalone IK circular trajectory demo for RobotGymEnv.

Usage::

    python -m examples.ik_trajectory --radius-x 0.03 --radius-y 0.03
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import numpy as np

from source.cli.robot_config import add_robot_config_args, make_configured_env
from source.runtime.pacing import RealtimePacer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IK circular trajectory demo.")

    parser.add_argument("--frequency", type=float, default=0.08)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--no-realtime", action="store_true")

    # Trajectory center; defaults to the current end-effector position.
    parser.add_argument("--center-x", type=float, default=0.3)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--center-z", type=float, default=1.0)

    # Trajectory radius / amplitude.
    parser.add_argument("--radius-x", type=float, default=0.1)
    parser.add_argument("--radius-y", type=float, default=0.1)
    parser.add_argument("--radius-z", type=float, default=0.00)

    # Whether the hand joints move periodically as well.
    parser.add_argument("--hand-scale", type=float, default=0.85)

    # Visualization.
    parser.add_argument("--no-target-marker", action="store_true")
    parser.add_argument("--target-marker-size", type=float, default=0.018)

    parser.add_argument("--no-center-marker", action="store_true")
    parser.add_argument("--center-marker-size", type=float, default=0.012)

    parser.add_argument("--no-trajectory-marker", action="store_true")
    parser.add_argument("--trajectory-segments", type=int, default=96)

    parser.add_argument("--no-stats-label", action="store_true")
    parser.add_argument("--print-error", action="store_true")
    add_robot_config_args(parser)

    return parser.parse_args()


def run_ik_sine_demo(
    env,
    *,
    frequency: float = 0.08,
    render_fps: float = 60.0,
    realtime: bool = True,
    center: Optional[np.ndarray] = None,
    radius_x: float = 0.03,
    radius_y: float = 0.03,
    radius_z: float = 0.00,
    hand_scale: float = 0.85,
    draw_target_marker: bool = True,
    target_marker_size: float = 0.018,
    draw_center_marker: bool = True,
    center_marker_size: float = 0.012,
    draw_trajectory_marker: bool = True,
    trajectory_segments: int = 96,
    draw_stats_label_flag: bool = True,
    print_error: bool = False,
) -> None:
    """Run an interactive IK circular trajectory demo in the passive viewer."""
    import mujoco
    from mujoco import viewer

    from source.viz.overlays import (
        clear_markers,
        draw_ellipse_marker,
        draw_sphere_marker,
        draw_stats_label,
    )

    controller = env.controller

    previous_mode = controller.control_mode
    if previous_mode != "ik":
        env.set_control_mode("ik")

    base_action = controller.current_ik_action(env.model, env.data)

    if center is None:
        center = base_action[:3].copy()
    else:
        center = np.asarray(center, dtype=np.float32)

    base_quat = base_action[3:7].copy()
    stats_label_pos = center + np.asarray([0.0, -0.16, 0.16], dtype=np.float32)

    hand_center = None
    hand_half_range = None
    if controller.include_hand_action:
        hand_space = controller.hand_controller.action_space
        hand_low = np.asarray(hand_space.low, dtype=np.float32).reshape(-1)
        hand_high = np.asarray(hand_space.high, dtype=np.float32).reshape(-1)
        hand_center = 0.5 * (hand_low + hand_high)
        hand_half_range = 0.5 * (hand_high - hand_low)

    render_dt = 1.0 / render_fps
    sim_start = float(env.data.time)
    pacer = RealtimePacer()
    pacer.reset(sim_start)
    last_control_sim_time = -np.inf
    last_print_second = -1

    action = base_action.copy()
    action[:3] = center
    action[3:7] = base_quat

    info: Dict[str, Any] = {"ik_error": np.zeros(6, dtype=np.float32), "ik_iterations": 0}

    with viewer.launch_passive(env.model, env.data) as handle:
        while handle.is_running():
            frame_sim_start = float(env.data.time)
            frame_steps = 0
            frame_control_updates = 0

            while float(env.data.time) - frame_sim_start < render_dt:
                sim_elapsed = float(env.data.time) - sim_start

                if float(env.data.time) - last_control_sim_time >= env.config.control_dt:
                    last_control_sim_time = float(env.data.time)

                    phase = 2.0 * np.pi * frequency * sim_elapsed
                    action = base_action.copy()

                    # End-effector target position trajectory.
                    action[0] = center[0] + radius_x * np.cos(phase)
                    action[1] = center[1] + radius_y * np.sin(phase)
                    action[2] = center[2] + radius_z * np.sin(2.0 * phase)

                    # Fixed end-effector orientation.
                    action[3:7] = base_quat

                    # Optional periodic hand joint motion.
                    if hand_center is not None and hand_half_range is not None:
                        hand_phase = phase + 0.6 * np.arange(hand_center.size, dtype=np.float32)
                        action[7:] = hand_center + hand_scale * hand_half_range * np.sin(hand_phase)

                    info = controller.apply_action(env.model, env.data, action)
                    frame_control_updates += 1

                mujoco.mj_step(env.model, env.data)
                frame_steps += 1

            env.record_simulation_steps(
                physics_steps=frame_steps, control_updates=frame_control_updates
            )

            clear_markers(handle)

            if draw_trajectory_marker:
                draw_ellipse_marker(
                    handle,
                    center,
                    radius_x=radius_x,
                    radius_y=radius_y,
                    radius_z=radius_z,
                    segments=trajectory_segments,
                    rgba=(0.0, 0.3, 1.0, 1.0),
                )

            if draw_center_marker:
                draw_sphere_marker(
                    handle, center, radius=center_marker_size, rgba=(1.0, 0.0, 0.0, 1.0)
                )

            if draw_target_marker:
                draw_sphere_marker(
                    handle, action[:3], radius=target_marker_size, rgba=(0.0, 1.0, 0.0, 1.0)
                )

            if draw_stats_label_flag:
                draw_stats_label(handle, env.simulation_stats, stats_label_pos, control_label="IK")

            handle.sync()

            sim_elapsed = float(env.data.time) - sim_start
            if print_error and int(sim_elapsed) != last_print_second:
                last_print_second = int(sim_elapsed)
                print(
                    "target_pos=",
                    np.array2string(action[:3], precision=4, suppress_small=True),
                    "center=",
                    np.array2string(center, precision=4, suppress_small=True),
                    "ik_error=",
                    np.array2string(info["ik_error"], precision=4, suppress_small=True),
                    "iterations=",
                    info["ik_iterations"],
                    "stats=",
                    env.simulation_stats,
                )

            if realtime:
                pacer.sleep_until(float(env.data.time))

    if previous_mode != "ik":
        env.set_control_mode(previous_mode)


def main() -> None:
    args = _parse_args()

    center = None
    if None not in (args.center_x, args.center_y, args.center_z):
        center = np.asarray([args.center_x, args.center_y, args.center_z], dtype=np.float32)

    env = make_configured_env(
        args,
        control_mode="ik",
    )
    env.reset(seed=0)

    try:
        run_ik_sine_demo(
            env,
            frequency=args.frequency,
            render_fps=args.render_fps,
            realtime=not args.no_realtime,
            center=center,
            radius_x=args.radius_x,
            radius_y=args.radius_y,
            radius_z=args.radius_z,
            hand_scale=args.hand_scale,
            draw_target_marker=not args.no_target_marker,
            target_marker_size=args.target_marker_size,
            draw_center_marker=not args.no_center_marker,
            center_marker_size=args.center_marker_size,
            draw_trajectory_marker=not args.no_trajectory_marker,
            trajectory_segments=args.trajectory_segments,
            draw_stats_label_flag=not args.no_stats_label,
            print_error=args.print_error,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
