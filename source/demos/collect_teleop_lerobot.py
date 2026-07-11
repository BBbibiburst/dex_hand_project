"""Collect Vive + stretch-glove demonstrations into a LeRobot dataset.

The default mock devices make the full control loop testable before hardware
drivers are implemented. Use ``--dry-run`` to skip the LeRobot dependency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

from source.demos.common import add_robot_config_args
from source.envs.manipulation import make_manipulation_env, registered_tasks
from source.teleop.devices import (
    MockStretchGlove,
    MockViveTracker,
    SineStretchGlove,
    SineViveTracker,
    StretchGloveApiDevice,
    ViveApiTracker,
)
from source.teleop.lerobot_recorder import LeRobotEpisodeRecorder
from source.teleop.mapping import TeleopMapper
from source.teleop.ui import TeleopUIState
from source.viz.overlays import clear_markers, draw_label


def parse_args():
    parser = argparse.ArgumentParser(description="Collect teleoperated LeRobot demonstrations.")
    parser.add_argument("--task", choices=registered_tasks(), default="lift")
    parser.add_argument("--repo-id", default="local/dex-hand-demonstrations")
    parser.add_argument("--output", type=Path, default=Path("datasets/lerobot"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("hardware", "sine", "mock"),
        default="sine",
        help="Input source: hardware uses device APIs; sine/mock are test inputs.",
    )
    parser.add_argument("--glove-inverted", action="store_true")
    parser.add_argument("--glove-mac", help="Classic-Bluetooth glove MAC address.")
    parser.add_argument("--glove-channel", type=int, default=1, help="Glove RFCOMM channel.")
    parser.add_argument("--glove-calibration-seconds", type=float, default=3.0)
    parser.add_argument("--vive-device-index", type=int)
    parser.add_argument("--vive-serial", help="Select a Vive tracker by serial number.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Control and render without writing data."
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Store images instead of encoded MP4."
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
        # Interactive episodes are ended explicitly with N / R / Q. Keep the
        # Gymnasium time limit from expiring while the operator is calibrating,
        # positioning, or recording a long demonstration.
        "episode_length": np.iinfo(np.int32).max,
        "enable_tactile_sensors": not getattr(args, "no_tactile", False),
        "render_mode": None,
    }
    return make_manipulation_env(
        args.task,
        task_config={"reward_shaping": True},
        **{key: value for key, value in overrides.items() if value is not None},
    )


def run(args) -> None:
    if args.fps <= 0 or args.episodes <= 0 or args.episode_frames <= 0:
        raise ValueError("fps, episodes and episode-frames must be positive.")
    env = _make_env(args)
    if args.device == "hardware":
        if not args.glove_mac:
            raise ValueError("--glove-mac is required when --device hardware is used.")
        glove = StretchGloveApiDevice(
            args.glove_mac,
            channel=args.glove_channel,
            calibration_seconds=args.glove_calibration_seconds,
        )
        vive = ViveApiTracker(device_index=args.vive_device_index, serial=args.vive_serial)
    elif args.device == "sine":
        glove, vive = SineStretchGlove(), SineViveTracker()
    else:
        glove, vive = MockStretchGlove(), MockViveTracker()
    renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
    recorder = None
    view_handle = None
    ui = TeleopUIState()

    try:
        glove.connect()
        vive.connect()

        def read_valid_vive(timeout: float = 10.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                sample = vive.read()
                if sample.valid:
                    return sample
                time.sleep(0.05)
            raise RuntimeError("Vive did not provide a valid tracked pose within 10 seconds.")

        observation, _ = env.reset(seed=0)
        initial_action = env.controller.current_ik_action(env.model, env.data)
        vive.set_pose(initial_action[:3], initial_action[3:7])
        mapper = TeleopMapper(
            env, position_scale=args.position_scale, glove_inverted=args.glove_inverted
        )
        mapper.calibrate(read_valid_vive())
        view_handle = viewer.launch_passive(env.model, env.data, key_callback=ui.handle_key)

        renderer.update_scene(env.data, camera=args.camera)
        first_image = renderer.render()
        if not args.dry_run:
            recorder = LeRobotEpisodeRecorder(
                repo_id=args.repo_id,
                root=args.output,
                fps=args.fps,
                state_dim=env.model.nq + env.model.nv + env.model.nu,
                action_dim=env.action_space.shape[0],
                tactile_shape=np.asarray(observation["tactile"]).shape,
                image_shape=first_image.shape,
                use_videos=not args.no_video,
            )

        def reset_episode(seed: int) -> None:
            nonlocal observation
            observation, _ = env.reset(seed=seed)
            initial_action = env.controller.current_ik_action(env.model, env.data)
            vive.set_pose(initial_action[:3], initial_action[3:7])
            mapper.calibrate(read_valid_vive())

        print(
            "Controls: SPACE record/pause | N save episode | R discard/reset | "
            "C recalibrate | Q quit"
        )
        period = 1.0 / args.fps
        deadline = time.monotonic()
        episode = 0
        episode_frames = 0
        success = False
        reset_episode(episode)
        while episode < args.episodes and not ui.quit_requested:
            if view_handle is not None and not view_handle.is_running():
                print("viewer closed; stopping collection")
                break
            if ui.consume_calibration_request():
                mapper.calibrate(read_valid_vive())
                print("Vive pose recalibrated")
            if ui.consume_discard_request():
                if recorder is not None:
                    recorder.clear_episode()
                episode_frames = 0
                success = False
                ui.recording = False
                reset_episode(episode)
                print(f"episode={episode} discarded and reset")
            if ui.consume_save_request():
                if episode_frames == 0:
                    print("episode is empty; nothing saved")
                else:
                    if recorder is not None:
                        recorder.save_episode()
                    print(f"episode={episode} saved success={success} frames={episode_frames}")
                    episode += 1
                    episode_frames = 0
                    success = False
                    ui.recording = False
                    if episode < args.episodes:
                        reset_episode(episode)
                    continue

            vive_sample = vive.read()
            if not vive_sample.valid:
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
                continue
            glove_sample = glove.read()
            action = mapper.action(vive_sample, glove_sample)
            observation, reward, terminated, truncated, info = env.step(action)
            success = bool(info.get("task_success", False))
            if ui.recording and episode_frames < args.episode_frames:
                renderer.update_scene(env.data, camera=args.camera)
                image = renderer.render().copy()
                if recorder is not None:
                    recorder.add_frame(
                        observation=observation,
                        image=image,
                        action=action,
                        glove=glove_sample,
                        vive=vive_sample,
                        task=args.task,
                    )
                episode_frames += 1
                if episode_frames == args.episode_frames:
                    ui.recording = False
                    print("frame limit reached; press N to save or R to discard")
            if view_handle is not None:
                clear_markers(view_handle)
                state = "REC" if ui.recording else "PAUSED"
                draw_label(
                    view_handle,
                    np.asarray([0.0, -0.32, 1.15], dtype=np.float32),
                    f"{state} | ep {episode + 1}/{args.episodes} | "
                    f"frames {episode_frames}/{args.episode_frames} | "
                    f"success {success}",
                )
                view_handle.sync()
            if terminated or truncated:
                ui.recording = False
                print("environment ended; press N to save or R to discard")
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        if recorder is not None:
            if recorder.frame_count:
                print(f"discarding {recorder.frame_count} unsaved frames")
                recorder.clear_episode()
            recorder.finalize()
        if view_handle is not None:
            view_handle.close()
        renderer.close()
        glove.close()
        vive.close()
        env.close()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
