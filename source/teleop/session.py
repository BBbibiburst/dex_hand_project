"""Shared Vive + glove teleoperation runtime used by data-collection apps.

The hardware adapters, calibrated mapping, dashboard, viewer overlays, and home
pose handling live here so LeRobot and raw-trajectory collection cannot drift
into separate teleoperation implementations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Any

import mujoco
import numpy as np
from mujoco import viewer

from source.teleop.config import load_teleop_config
from source.teleop.devices import (
    GloveSample,
    MockStretchGlove,
    MockViveTracker,
    SineStretchGlove,
    SineViveTracker,
    StretchGloveApiDevice,
    ViveApiTracker,
    ViveSample,
)
from source.teleop.glove_processing import read_latest_glove
from source.teleop.mapping import TeleopMapper
from source.teleop.ui import TeleopUIState
from source.teleop.vive.coordinates import remap_pose, rotation_matrix_to_rpy_degrees
from source.viz.overlays import clear_markers, draw_label, draw_line_marker, draw_pose_frame
from source.viz.teleop_dashboard import TeleopDashboard


@dataclass(frozen=True)
class TeleopControlSample:
    vive: ViveSample
    glove: GloveSample
    action: np.ndarray


@dataclass(frozen=True)
class TeleopSceneSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    elapsed_steps: int


def add_teleop_session_args(
    parser: argparse.ArgumentParser,
    *,
    config: dict[str, Any] | None = None,
    default_device: str = "sine",
) -> None:
    """Add the shared operator/device/control/visualization CLI arguments."""
    config = load_teleop_config() if config is None else config
    calibration = dict(config.get("glove_calibration") or {})
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument(
        "--workspace-yaw-degrees",
        type=float,
        default=float(config.get("vive_robot_yaw_degrees", -90.0)),
        help="Yaw alignment from the Vive workspace to the robot world.",
    )
    parser.add_argument(
        "--neutral-hand-pitch-degrees",
        type=float,
        default=float(config.get("vive_robot_neutral_hand_pitch_degrees", 90.0)),
        help="Pitch defining the flat, forward robot neutral hand pose.",
    )
    parser.add_argument(
        "--arm-home-qpos",
        type=float,
        nargs=7,
        default=config.get("teleop_arm_home_qpos", [0.0, 0.7, 0.0, 0.7, 0.0, 0.0, 0.0]),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Seven arm joint positions used as the teleoperation reset pose.",
    )
    parser.add_argument(
        "--device",
        choices=("hardware", "sine", "mock"),
        default=default_device,
        help="Input source: hardware uses device APIs; sine/mock are test inputs.",
    )
    parser.add_argument("--glove-inverted", action="store_true")
    parser.add_argument(
        "--thumb-rotation",
        type=float,
        default=float(config.get("teleop_thumb_rotation", 0.25)),
    )
    parser.add_argument(
        "--ik-posture-weight",
        type=float,
        default=float(config.get("teleop_ik_posture_weight", 0.002)),
    )
    parser.add_argument(
        "--ik-posture-qpos",
        type=float,
        nargs=7,
        default=config.get("teleop_ik_posture_qpos", [0.0, 1.1, 0.0, 1.3, 0.0, -0.5, 0.0]),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument(
        "--glove-smoothing",
        type=float,
        default=float(config.get("teleop_glove_smoothing", 0.90)),
    )
    parser.add_argument(
        "--glove-deadzone",
        type=float,
        default=float(config.get("glove_open_deadzone", 0.10)),
    )
    parser.add_argument(
        "--glove-closed-deadzone",
        type=float,
        default=float(config.get("glove_closed_deadzone", 0.10)),
    )
    parser.add_argument(
        "--finger-curve-gamma",
        type=float,
        default=float(config.get("teleop_finger_curve_gamma", 1.4)),
    )
    parser.add_argument("--glove-mac", default=config.get("glove_mac"))
    parser.add_argument("--glove-channel", type=int, default=int(config.get("glove_channel", 1)))
    parser.add_argument("--glove-serial-port", default=config.get("glove_serial_port"))
    parser.add_argument("--glove-baudrate", type=int, default=int(config.get("glove_baudrate", 9600)))
    parser.add_argument(
        "--glove-calibration-seconds",
        type=float,
        default=float(config.get("glove_calibration_seconds", 3.0)),
    )
    parser.add_argument("--vive-device-index", type=int)
    parser.add_argument("--vive-serial", help="Select a Vive tracker by serial number.")
    parser.add_argument("--no-calibration-prompt", action="store_true")
    parser.set_defaults(mujoco_viewer=True)
    parser.add_argument("--mujoco-viewer", dest="mujoco_viewer", action="store_true")
    parser.add_argument("--no-mujoco-viewer", dest="mujoco_viewer", action="store_false")
    parser.add_argument("--target-frame-size", type=float, default=0.08)
    parser.set_defaults(
        glove_calibration_minimum=calibration.get("open_minimum"),
        glove_calibration_maximum=calibration.get("fist_maximum"),
    )


def validate_teleop_session_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("image dimensions must be positive.")
    if args.target_frame_size <= 0:
        raise ValueError("--target-frame-size must be positive.")
    if args.ik_posture_weight < 0:
        raise ValueError("--ik-posture-weight must be non-negative.")
    if not np.isfinite(args.finger_curve_gamma) or args.finger_curve_gamma <= 0:
        raise ValueError("--finger-curve-gamma must be positive and finite.")
    if not 0.0 <= args.thumb_rotation <= 1.0:
        raise ValueError("--thumb-rotation must be in [0, 1].")
    if args.device == "hardware" and not args.glove_mac:
        raise ValueError("--glove-mac is required when --device hardware is used.")


def make_teleop_devices(args: argparse.Namespace):
    """Create the same tested adapters used by the standalone glove/Vive tools."""
    if args.device == "hardware":
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
        return glove, vive
    if args.device == "sine":
        return SineStretchGlove(), SineViveTracker()
    return MockStretchGlove(), MockViveTracker()


class TeleopSession:
    """One shared operator loop around an already-created IK manipulation env."""

    def __init__(
        self,
        env,
        args: argparse.Namespace,
        *,
        episodes: int,
        frame_limit: int,
    ) -> None:
        validate_teleop_session_args(args)
        self.env = env
        self.args = args
        self.episodes = int(episodes)
        self.frame_limit = int(frame_limit)
        if self.episodes <= 0 or self.frame_limit <= 0:
            raise ValueError("episodes and frame_limit must be positive.")

        self.arm = env.controller.arm_controller
        self.arm.posture_weight = float(args.ik_posture_weight)
        self.posture = np.asarray(args.ik_posture_qpos, dtype=np.float64)
        self.home = np.asarray(args.arm_home_qpos, dtype=np.float64)
        expected = (self.arm.position_action_size,)
        if self.posture.shape != expected or self.home.shape != expected:
            raise ValueError(
                f"arm-home-qpos and ik-posture-qpos must contain {expected[0]} values."
            )
        if np.any(self.posture < self.arm.ctrl_low) or np.any(self.posture > self.arm.ctrl_high):
            raise ValueError("--ik-posture-qpos exceeds the configured arm joint limits.")
        if np.any(self.home < self.arm.ctrl_low) or np.any(self.home > self.arm.ctrl_high):
            raise ValueError("--arm-home-qpos exceeds the configured arm joint limits.")
        self.arm.nullspace_posture = self.posture.copy()

        self.glove, self.vive = make_teleop_devices(args)
        self.ui = TeleopUIState()
        self.mapper: TeleopMapper | None = None
        self.observation: dict[str, Any] | None = None
        self.latest_control: TeleopControlSample | None = None

        self.renderer = mujoco.Renderer(
            env.model,
            height=args.image_height,
            width=args.image_width,
        )
        self.dashboard = TeleopDashboard(env.tactile_sensor)
        self.view_handle = (
            viewer.launch_passive(env.model, env.data) if args.mujoco_viewer else None
        )

    @property
    def period(self) -> float:
        return 1.0 / float(self.args.fps)

    @property
    def is_open(self) -> bool:
        return self.dashboard.is_open

    def connect(self) -> None:
        self.glove.connect()
        self.vive.connect()

    def _make_mapper(self) -> TeleopMapper:
        return TeleopMapper(
            self.env,
            position_scale=self.args.position_scale,
            workspace_yaw_degrees=self.args.workspace_yaw_degrees,
            neutral_hand_pitch_degrees=self.args.neutral_hand_pitch_degrees,
            dex_thumb_rotation=self.args.thumb_rotation,
            glove_inverted=self.args.glove_inverted,
            glove_smoothing=self.args.glove_smoothing,
            glove_deadzone=self.args.glove_deadzone,
            glove_closed_deadzone=self.args.glove_closed_deadzone,
            finger_curve_gamma=self.args.finger_curve_gamma,
        )

    def reset_home(self, seed: int) -> dict[str, Any]:
        """Reset task placement, then put the arm in the shared teleop home pose."""
        self.observation, _ = self.env.reset(seed=seed)
        self.env.data.qpos[self.arm.qpos_addrs] = self.home
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.controller.reset(
            self.env.model,
            self.env.data,
            rng=self.env.np_random,
            options=None,
        )
        self.arm.nullspace_posture = self.posture.copy()
        mujoco.mj_forward(self.env.model, self.env.data)
        self.observation = self.env._get_observation()
        # Construct after the home pose is established so the neutral orientation
        # is anchored to the pose the operator actually sees.
        self.mapper = self._make_mapper()
        initial_action = self.env.controller.current_ik_action(self.env.model, self.env.data)
        self.vive.set_pose(initial_action[:3], initial_action[3:7])
        self.latest_control = None
        print(f"Teleop home pose: EE position={np.round(initial_action[:3], 3).tolist()}")
        return self.observation

    def snapshot(self) -> TeleopSceneSnapshot:
        return TeleopSceneSnapshot(
            qpos=self.env.data.qpos.copy(),
            qvel=self.env.data.qvel.copy(),
            ctrl=self.env.data.ctrl.copy(),
            elapsed_steps=int(self.env.elapsed_steps),
        )

    def restore(self, snapshot: TeleopSceneSnapshot) -> dict[str, Any]:
        """Restore an exact same-scene snapshot without re-randomizing the object."""
        self.env.data.qpos[:] = snapshot.qpos
        self.env.data.qvel[:] = snapshot.qvel
        self.env.data.ctrl[:] = snapshot.ctrl
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.controller.reset(
            self.env.model,
            self.env.data,
            rng=self.env.np_random,
            options=None,
        )
        self.arm.nullspace_posture = self.posture.copy()
        self.env.data.ctrl[:] = snapshot.ctrl
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.elapsed_steps = snapshot.elapsed_steps
        self.observation = self.env._get_observation()
        self.mapper = self._make_mapper()
        initial_action = self.env.controller.current_ik_action(self.env.model, self.env.data)
        self.vive.set_pose(initial_action[:3], initial_action[3:7])
        self.latest_control = None
        return self.observation

    def read_valid_vive(self, timeout: float = 10.0) -> ViveSample:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self.vive.read()
            if sample.valid:
                return sample
            time.sleep(0.05)
        raise RuntimeError(f"Vive did not provide a valid tracked pose within {timeout:g} seconds.")

    def calibrate(
        self,
        *,
        wait_for_dashboard_confirmation: bool,
        episode_index: int = 0,
        frames: int = 0,
    ) -> None:
        if self.mapper is None:
            self.mapper = self._make_mapper()
        if (
            self.args.device == "hardware"
            and not self.args.no_calibration_prompt
            and wait_for_dashboard_confirmation
        ):
            print("\nVive 中立位姿校准：请将手掌水平放平，手指朝向机器人正前方。")
            print("保持稳定，然后在 Teleop Data Collection 窗口中按 C 采集基准。")
            self.ui.consume_calibration_request()
            while True:
                if not self.dashboard.is_open:
                    raise KeyboardInterrupt
                self.render(
                    info={},
                    state="CALIBRATION",
                    episode_index=episode_index,
                    frames=frames,
                    success=False,
                    message="HAND FLAT + FORWARD, THEN PRESS C",
                )
                if self.ui.consume_calibration_request():
                    break
                time.sleep(0.03)
        self.mapper.calibrate(self.read_valid_vive())
        print(
            "Vive 中立位姿已校准：当前位置对应机器人当前末端位置，"
            "水平朝前对应机器人当前末端朝向。"
        )

    def read_control(self) -> TeleopControlSample | None:
        if self.mapper is None:
            raise RuntimeError("reset_home()/restore() and calibrate() must run before teleoperation.")
        vive_sample = self.vive.read()
        if not vive_sample.valid:
            self.latest_control = None
            return None
        glove_sample = (
            read_latest_glove(self.glove)
            if self.args.device == "hardware"
            else self.glove.read()
        )
        action = self.mapper.action(vive_sample, glove_sample)
        sample = TeleopControlSample(vive_sample, glove_sample, action)
        self.latest_control = sample
        return sample

    def camera_image(self) -> np.ndarray:
        """Render the configured agent camera without consuming dashboard input."""
        self.renderer.update_scene(self.env.data, camera=self.args.camera)
        return self.renderer.render().copy()

    def _grasp_contact_count(self) -> int:
        bindings = getattr(self.env.task, "bindings", None)
        if bindings is None:
            return 0
        object_ids: set[int] = set()
        for binding in bindings.objects.values():
            object_ids.update(int(value) for value in binding.geom_ids)
        robot_ids = {int(value) for value in bindings.robot_geom_ids}
        count = 0
        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if (first in object_ids and second in robot_ids) or (
                second in object_ids and first in robot_ids
            ):
                count += 1
        return count

    def render(
        self,
        *,
        info: dict[str, Any],
        state: str,
        episode_index: int,
        frames: int,
        success: bool,
        message: str = "",
    ) -> np.ndarray:
        if self.observation is None:
            self.observation = self.env._get_observation()
        if self.view_handle is not None and not self.view_handle.is_running():
            print("MuJoCo Viewer closed; continuing with the teleop dashboard")
            self.view_handle.close()
            self.view_handle = None

        image = self.camera_image()
        read_raw = getattr(self.env.tactile_sensor, "read_raw", None)
        raw_tactile = (
            read_raw(self.env.model, self.env.data)
            if callable(read_raw)
            else self.observation["tactile"]
        )
        target_position = info.get("ik_target_position")
        target_quat = info.get("ik_target_quat")
        actual_action = self.env.controller.current_ik_action(self.env.model, self.env.data)
        actual_position = actual_action[:3]
        actual_quat = actual_action[3:7]
        ik_error = info.get("ik_error")

        vive_position = vive_rpy = None
        if self.latest_control is not None:
            vive_position, vive_rotation = remap_pose(
                self.latest_control.vive.position,
                self.latest_control.vive.quaternion_wxyz,
            )
            vive_rpy = rotation_matrix_to_rpy_degrees(vive_rotation)

        hand_values = None if self.mapper is None else self.mapper.last_hand_values
        key = self.dashboard.update(
            image,
            self.observation["tactile"],
            state=state,
            episode=episode_index + 1,
            episodes=self.episodes,
            frames=frames,
            frame_limit=self.frame_limit,
            success=success,
            message=message,
            target_position=target_position,
            actual_position=actual_position,
            ik_error=ik_error,
            vive_position=vive_position,
            vive_rpy=vive_rpy,
            hand_values=hand_values,
            raw_tactile_values=raw_tactile,
            grasp_contacts=self._grasp_contact_count(),
        )
        if key not in (-1, 255):
            self.ui.handle_key(key)

        if self.view_handle is not None:
            clear_markers(self.view_handle)
            if target_position is not None and target_quat is not None:
                draw_pose_frame(
                    self.view_handle,
                    target_position,
                    target_quat,
                    axis_length=self.args.target_frame_size,
                    label="TARGET",
                )
                draw_line_marker(
                    self.view_handle,
                    actual_position,
                    target_position,
                    width=0.002,
                    rgba=(1.0, 0.8, 0.1, 0.9),
                )
            draw_pose_frame(
                self.view_handle,
                actual_position,
                actual_quat,
                axis_length=self.args.target_frame_size * 0.65,
                label="ACTUAL",
            )
            object_position = self.observation.get("object_pos")
            if object_position is not None:
                draw_line_marker(
                    self.view_handle,
                    actual_position,
                    np.asarray(object_position, dtype=np.float64),
                    width=0.0025,
                    rgba=(0.95, 0.3, 0.95, 0.9),
                )
                draw_label(
                    self.view_handle,
                    np.asarray(object_position, dtype=np.float64),
                    "OBJECT",
                    rgba=(1.0, 0.4, 1.0, 1.0),
                )
            draw_label(
                self.view_handle,
                np.asarray([0.0, -0.32, 1.15], dtype=np.float32),
                f"{state} | ep {episode_index + 1}/{self.episodes} | "
                f"frames {frames}/{self.frame_limit} | success {success}",
            )
            self.view_handle.sync()
        return image

    def close(self) -> None:
        if self.view_handle is not None:
            self.view_handle.close()
            self.view_handle = None
        self.dashboard.close()
        self.renderer.close()
        self.glove.close()
        self.vive.close()
