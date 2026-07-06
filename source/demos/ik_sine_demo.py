# -*- coding: utf-8 -*-
"""Standalone IK sine-wave trajectory demo for DexHandGymEnv.

Usage::

    python -m source.demos.ik_sine_demo --frequency 0.08 --radius 0.03
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import numpy as np

from ..environments.transforms import axis_angle_quat, quat_multiply


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IK sine-wave trajectory demo.")
    parser.add_argument("--frequency", type=float, default=0.08)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--radius", type=float, default=0.03)
    parser.add_argument("--z-scale", type=float, default=0.02)
    parser.add_argument("--rot-scale", type=float, default=0.25)
    parser.add_argument("--hand-scale", type=float, default=0.85)
    parser.add_argument("--no-target-marker", action="store_true")
    parser.add_argument("--target-marker-size", type=float, default=0.018)
    parser.add_argument("--no-stats-label", action="store_true")
    parser.add_argument("--print-error", action="store_true")
    return parser.parse_args()


def run_ik_sine_demo(
    env,
    *,
    frequency: float = 0.08,
    render_fps: float = 60.0,
    realtime: bool = True,
    radius: float = 0.03,
    z_scale: float = 0.02,
    rot_scale: float = 0.25,
    hand_scale: float = 0.85,
    draw_target_marker: bool = True,
    target_marker_size: float = 0.018,
    draw_stats_label: bool = True,
    print_error: bool = False,
) -> None:
    """Run an interactive IK sine-wave demo in the passive viewer."""
    import mujoco
    from mujoco import viewer
    from ..environments.overlays import clear_markers, draw_sphere_marker, draw_stats_label

    controller = env.controller
    previous_mode = controller.control_mode
    if previous_mode != "ik":
        env.set_control_mode("ik")

    base_action = controller.current_ik_action(env.model, env.data)
    base_pos = base_action[:3].copy()
    base_quat = base_action[3:7].copy()
    stats_label_pos = base_pos + np.asarray([0.0, -0.16, 0.16], dtype=np.float32)

    hand_center = None
    hand_half_range = None
    if controller.include_hand_action:
        hand_low = controller.ctrl_low[7:].astype(np.float32)
        hand_high = controller.ctrl_high[7:].astype(np.float32)
        hand_center = 0.5 * (hand_low + hand_high)
        hand_half_range = 0.5 * (hand_high - hand_low)

    render_dt = 1.0 / render_fps
    wall_start = time.perf_counter()
    sim_start = float(env.data.time)
    last_control_sim_time = -np.inf
    last_print_second = -1
    action = base_action.copy()
    info: Dict[str, Any] = {
        "ik_error": np.zeros(6, dtype=np.float32),
        "ik_iterations": 0,
    }

    with viewer.launch_passive(env.model, env.data) as handle:
        while handle.is_running():
            frame_sim_start = float(env.data.time)
            frame_steps = 0
            frame_control_updates = 0

            while float(env.data.time) - frame_sim_start < render_dt:
                sim_elapsed = float(env.data.time) - sim_start
                if (
                    float(env.data.time) - last_control_sim_time
                    >= env.config.control_dt
                ):
                    last_control_sim_time = float(env.data.time)
                    phase = 2.0 * np.pi * frequency * sim_elapsed
                    action = base_action.copy()
                    action[0] = base_pos[0] + radius * np.cos(phase)
                    action[1] = base_pos[1] + radius * np.sin(phase)
                    action[2] = base_pos[2] + z_scale * np.sin(2.0 * phase)

                    yaw = axis_angle_quat([0.0, 0.0, 1.0], rot_scale * np.sin(phase))
                    pitch = axis_angle_quat(
                        [0.0, 1.0, 0.0], 0.5 * rot_scale * np.cos(phase)
                    )
                    action[3:7] = quat_multiply(quat_multiply(yaw, pitch), base_quat)

                    if hand_center is not None and hand_half_range is not None:
                        hand_phase = phase + 0.6 * np.arange(
                            hand_center.size, dtype=np.float32
                        )
                        action[7:] = hand_center + hand_scale * hand_half_range * np.sin(
                            hand_phase
                        )

                    info = controller.apply_action(env.model, env.data, action)
                    frame_control_updates += 1

                mujoco.mj_step(env.model, env.data)
                frame_steps += 1

            env.record_simulation_steps(
                physics_steps=frame_steps,
                control_updates=frame_control_updates,
            )
            clear_markers(handle)
            if draw_target_marker:
                draw_sphere_marker(handle, action[:3], radius=target_marker_size)
            if draw_stats_label:
                draw_stats_label(handle, env.simulation_stats, stats_label_pos, control_label="IK")
            handle.sync()

            sim_elapsed = float(env.data.time) - sim_start
            if print_error and int(sim_elapsed) != last_print_second:
                last_print_second = int(sim_elapsed)
                print(
                    "ik_error=",
                    np.array2string(info["ik_error"], precision=4, suppress_small=True),
                    "iterations=",
                    info["ik_iterations"],
                    "stats=",
                    env.simulation_stats,
                )

            if realtime:
                sleep_time = wall_start + sim_elapsed - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    if previous_mode != "ik":
        env.set_control_mode(previous_mode)


def main() -> None:
    args = _parse_args()
    from ..environments.rl_env import make_env
    env = make_env(control_mode="ik")
    env.reset(seed=0)
    try:
        run_ik_sine_demo(
            env,
            frequency=args.frequency,
            render_fps=args.render_fps,
            realtime=not args.no_realtime,
            radius=args.radius,
            z_scale=args.z_scale,
            rot_scale=args.rot_scale,
            hand_scale=args.hand_scale,
            draw_target_marker=not args.no_target_marker,
            target_marker_size=args.target_marker_size,
            draw_stats_label=not args.no_stats_label,
            print_error=args.print_error,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
