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
from source.teleop.config import load_teleop_config
from source.teleop.devices import (
    MockStretchGlove,
    MockViveTracker,
    SineStretchGlove,
    SineViveTracker,
    StretchGloveApiDevice,
    ViveApiTracker,
)
from source.teleop.lerobot_recorder import LeRobotEpisodeRecorder
from source.teleop.glove_processing import read_latest_glove
from source.teleop.mapping import TeleopMapper
from source.teleop.ui import TeleopUIState
from source.viz.overlays import clear_markers, draw_label, draw_pose_frame
from source.viz.teleop_dashboard import TeleopDashboard


def parse_args():
    teleop_config = load_teleop_config()
    glove_calibration = dict(teleop_config.get("glove_calibration") or {})
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
        "--workspace-yaw-degrees",
        type=float,
        default=float(teleop_config.get("vive_robot_yaw_degrees", -90.0)),
        help=(
            "Yaw alignment from the Vive workspace to the robot world. "
            "The default -90 maps Vive forward (+Y) to robot forward (+X)."
        ),
    )
    parser.add_argument(
        "--neutral-hand-pitch-degrees",
        type=float,
        default=float(
            teleop_config.get("vive_robot_neutral_hand_pitch_degrees", 90.0)
        ),
        help=(
            "World-Y pitch applied to the robot's initial end-effector orientation "
            "to define the flat, forward neutral hand pose."
        ),
    )
    parser.add_argument(
        "--arm-home-qpos",
        type=float,
        nargs=7,
        default=teleop_config.get(
            "teleop_arm_home_qpos", [0.0, 0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
        ),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Seven arm joint positions used as the teleoperation reset pose.",
    )
    parser.add_argument(
        "--device",
        choices=("hardware", "sine", "mock"),
        default="sine",
        help="Input source: hardware uses device APIs; sine/mock are test inputs.",
    )
    parser.add_argument("--glove-inverted", action="store_true")
    parser.add_argument(
        "--thumb-rotation",
        type=float,
        default=float(teleop_config.get("teleop_thumb_rotation", 0.25)),
        help=(
            "Fixed normalized Dex Hand thumb-opposition command in [0, 1]. "
            "Thumb grasp remains controlled by the glove."
        ),
    )
    parser.add_argument(
        "--ik-posture-weight",
        type=float,
        default=float(teleop_config.get("teleop_ik_posture_weight", 0.002)),
        help="Soft joint-posture weight used to prefer the bent teleop home pose.",
    )
    parser.add_argument(
        "--ik-posture-qpos",
        type=float,
        nargs=7,
        default=teleop_config.get(
            "teleop_ik_posture_qpos", [0.0, 1.1, 0.0, 1.3, 0.0, -0.5, 0.0]
        ),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Preferred bent-arm posture used by the soft teleoperation IK objective.",
    )
    parser.add_argument(
        "--glove-smoothing",
        type=float,
        default=float(teleop_config.get("teleop_glove_smoothing", 0.90)),
        help="EMA response factor in (0, 1]; larger values respond faster.",
    )
    parser.add_argument(
        "--glove-deadzone",
        type=float,
        default=float(teleop_config.get("glove_open_deadzone", 0.10)),
        help="Normalized open-end interval mapped exactly to 0.",
    )
    parser.add_argument(
        "--glove-closed-deadzone",
        type=float,
        default=float(teleop_config.get("glove_closed_deadzone", 0.10)),
        help="Normalized closed-end interval mapped exactly to 1.",
    )
    parser.add_argument(
        "--finger-curve-gamma",
        type=float,
        default=float(teleop_config.get("teleop_finger_curve_gamma", 1.4)),
        help=(
            "Flexion response exponent; values above 1 provide finer control "
            "near the open pose, while 1 keeps a linear mapping."
        ),
    )
    parser.add_argument(
        "--glove-mac",
        default=teleop_config.get("glove_mac"),
        help="Classic-Bluetooth glove MAC address (default: configs/teleop.json).",
    )
    parser.add_argument(
        "--glove-channel",
        type=int,
        default=int(teleop_config.get("glove_channel", 1)),
        help="Glove RFCOMM channel.",
    )
    parser.add_argument(
        "--glove-serial-port",
        default=teleop_config.get("glove_serial_port"),
        help="Windows HC-06 outgoing COM port; preferred over direct RFCOMM.",
    )
    parser.add_argument(
        "--glove-baudrate",
        type=int,
        default=int(teleop_config.get("glove_baudrate", 9600)),
    )
    parser.add_argument(
        "--glove-calibration-seconds",
        type=float,
        default=float(teleop_config.get("glove_calibration_seconds", 3.0)),
    )
    parser.add_argument("--vive-device-index", type=int)
    parser.add_argument("--vive-serial", help="Select a Vive tracker by serial number.")
    parser.add_argument(
        "--no-calibration-prompt",
        action="store_true",
        help="Do not wait for the hardware neutral-pose confirmation before Vive calibration.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Control and render without writing data."
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Store images instead of encoded MP4."
    )
    parser.set_defaults(mujoco_viewer=True)
    parser.add_argument(
        "--mujoco-viewer",
        dest="mujoco_viewer",
        action="store_true",
        help="Open the standard MuJoCo Viewer alongside the teleop dashboard (default).",
    )
    parser.add_argument(
        "--no-mujoco-viewer",
        dest="mujoco_viewer",
        action="store_false",
        help="Disable the standard MuJoCo Viewer and show only the teleop dashboard.",
    )
    parser.add_argument(
        "--target-frame-size",
        type=float,
        default=0.08,
        help="Length in metres of the displayed ideal IK target pose axes.",
    )
    add_robot_config_args(parser)
    parser.set_defaults(
        glove_calibration_minimum=glove_calibration.get("open_minimum"),
        glove_calibration_maximum=glove_calibration.get("fist_maximum"),
    )
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
    if args.target_frame_size <= 0:
        raise ValueError("--target-frame-size must be positive.")
    if args.ik_posture_weight < 0:
        raise ValueError("--ik-posture-weight must be non-negative.")
    if not np.isfinite(args.finger_curve_gamma) or args.finger_curve_gamma <= 0:
        raise ValueError("--finger-curve-gamma must be positive and finite.")
    env = _make_env(args)
    arm_controller = env.controller.arm_controller
    arm_controller.posture_weight = args.ik_posture_weight
    ik_posture_qpos = np.asarray(args.ik_posture_qpos, dtype=np.float64)
    if ik_posture_qpos.shape != (arm_controller.position_action_size,):
        raise ValueError(
            "--ik-posture-qpos must contain exactly "
            f"{arm_controller.position_action_size} values."
        )
    if np.any(ik_posture_qpos < arm_controller.ctrl_low) or np.any(
        ik_posture_qpos > arm_controller.ctrl_high
    ):
        raise ValueError("--ik-posture-qpos exceeds the configured arm joint limits.")
    arm_controller.nullspace_posture = ik_posture_qpos
    if args.device == "hardware":
        if not args.glove_mac:
            raise ValueError("--glove-mac is required when --device hardware is used.")
        glove = StretchGloveApiDevice(
            args.glove_mac,
            channel=args.glove_channel,
            serial_port=args.glove_serial_port,
            baudrate=args.glove_baudrate,
            calibration_seconds=args.glove_calibration_seconds,
            calibration_minimum=args.glove_calibration_minimum,
            calibration_maximum=args.glove_calibration_maximum,
        )
        vive = ViveApiTracker(device_index=args.vive_device_index, serial=args.vive_serial)
    elif args.device == "sine":
        glove, vive = SineStretchGlove(), SineViveTracker()
    else:
        glove, vive = MockStretchGlove(), MockViveTracker()
    renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
    recorder = None
    view_handle = None
    dashboard = None
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
        mapper = TeleopMapper(
            env,
            position_scale=args.position_scale,
            workspace_yaw_degrees=args.workspace_yaw_degrees,
            neutral_hand_pitch_degrees=args.neutral_hand_pitch_degrees,
            dex_thumb_rotation=args.thumb_rotation,
            glove_inverted=args.glove_inverted,
            glove_smoothing=args.glove_smoothing,
            glove_deadzone=args.glove_deadzone,
            glove_closed_deadzone=args.glove_closed_deadzone,
            finger_curve_gamma=args.finger_curve_gamma,
        )
        if args.mujoco_viewer:
            # Collection shortcuts belong exclusively to TeleopDashboard.
            # Leaving out key_callback prevents MuJoCo's built-in shortcuts
            # from also toggling recorder state.
            view_handle = viewer.launch_passive(env.model, env.data)

        renderer.update_scene(env.data, camera=args.camera)
        first_image = renderer.render()
        dashboard = TeleopDashboard(env.tactile_sensor)
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

        def update_dashboard(
            *,
            state: str,
            episode_index: int,
            frames: int,
            success: bool,
            message: str = "",
            target_position=None,
            hand_values=None,
        ) -> np.ndarray:
            renderer.update_scene(env.data, camera=args.camera)
            image = renderer.render().copy()
            key = dashboard.update(
                image,
                observation["tactile"],
                state=state,
                episode=episode_index + 1,
                episodes=args.episodes,
                frames=frames,
                frame_limit=args.episode_frames,
                success=success,
                message=message,
                target_position=target_position,
                hand_values=hand_values,
            )
            if key not in (-1, 255):
                ui.handle_key(key)
            return image

        def calibrate_vive(*, wait_for_dashboard_confirmation: bool) -> None:
            if args.device == "hardware" and not args.no_calibration_prompt:
                print(
                    "\nVive 中立位姿校准：请将手掌水平放平，"
                    "手指朝向机器人正前方。"
                )
                if wait_for_dashboard_confirmation:
                    print("保持稳定，然后在 Teleop Data Collection 窗口中按 C 采集基准。")
                    # Discard a C event left over from an earlier interaction.
                    ui.consume_calibration_request()
                    while True:
                        if dashboard is None or not dashboard.is_open:
                            raise KeyboardInterrupt
                        update_dashboard(
                            state="CALIBRATION",
                            episode_index=0,
                            frames=0,
                            success=False,
                            message="HAND FLAT + FORWARD, THEN PRESS C",
                        )
                        if ui.consume_calibration_request():
                            break
                        time.sleep(0.03)
            mapper.calibrate(read_valid_vive())
            print(
                "Vive 中立位姿已校准：当前位置对应机器人当前末端位置，"
                "水平朝前对应机器人当前末端朝向。"
            )

        def reset_episode(seed: int) -> None:
            nonlocal observation
            observation, _ = env.reset(seed=seed)
            arm_controller = env.controller.arm_controller
            home_qpos = np.asarray(args.arm_home_qpos, dtype=np.float64)
            if home_qpos.shape != (arm_controller.position_action_size,):
                raise ValueError(
                    "--arm-home-qpos must contain exactly "
                    f"{arm_controller.position_action_size} values."
                )
            if np.any(home_qpos < arm_controller.ctrl_low) or np.any(
                home_qpos > arm_controller.ctrl_high
            ):
                raise ValueError(
                    "--arm-home-qpos exceeds the configured arm joint limits."
                )
            env.data.qpos[arm_controller.qpos_addrs] = home_qpos
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            env.controller.reset(
                env.model,
                env.data,
                rng=env.np_random,
                options=None,
            )
            # reset() does not replace an explicitly configured preferred posture.
            arm_controller.nullspace_posture = ik_posture_qpos.copy()
            mujoco.mj_forward(env.model, env.data)
            initial_action = env.controller.current_ik_action(env.model, env.data)
            vive.set_pose(initial_action[:3], initial_action[3:7])
            print(
                "Teleop home pose: "
                f"EE position={np.round(initial_action[:3], 3).tolist()}"
            )
            calibrate_vive(wait_for_dashboard_confirmation=True)

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
            if dashboard is None or not dashboard.is_open:
                print("teleoperation dashboard closed; stopping collection")
                break
            if view_handle is not None and not view_handle.is_running():
                print("MuJoCo Viewer closed; continuing with the teleop dashboard")
                view_handle.close()
                view_handle = None
            if ui.consume_calibration_request():
                ui.recording = False
                calibrate_vive(wait_for_dashboard_confirmation=False)
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
                update_dashboard(
                    state="TRACKING LOST",
                    episode_index=episode,
                    frames=episode_frames,
                    success=success,
                    message="WAITING FOR A VALID VIVE POSE",
                )
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
                continue
            glove_sample = read_latest_glove(glove) if args.device == "hardware" else glove.read()
            action = mapper.action(vive_sample, glove_sample)
            observation, reward, terminated, truncated, info = env.step(action)
            success = bool(info.get("task_success", False))
            target_position = info.get("ik_target_position")
            image = update_dashboard(
                state="REC" if ui.recording else "PAUSED",
                episode_index=episode,
                frames=episode_frames,
                success=success,
                target_position=target_position,
                hand_values=mapper.last_hand_values,
            )
            if ui.recording and episode_frames < args.episode_frames:
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
                target_quaternion = info.get("ik_target_quat")
                if target_position is not None and target_quaternion is not None:
                    draw_pose_frame(
                        view_handle,
                        target_position,
                        target_quaternion,
                        axis_length=args.target_frame_size,
                        label="TARGET",
                    )
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
        if dashboard is not None:
            dashboard.close()
        renderer.close()
        glove.close()
        vive.close()
        env.close()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
