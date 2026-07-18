"""Calibrated Vive/glove samples to the project's absolute IK action."""

from __future__ import annotations

import math

import numpy as np

from source.teleop.devices import GloveSample, ViveSample
from source.teleop.glove_processing import GloveValueFilter
from source.teleop.vive.coordinates import (
    quaternion_to_rotation_matrix,
    remap_pose,
    rotation_matrix_to_quaternion_wxyz,
)


class TeleopMapper:
    """Relative Vive motion anchored to the robot EE pose at calibration."""

    def __init__(
        self,
        env,
        *,
        position_scale=1.0,
        workspace_yaw_degrees=0.0,
        neutral_hand_pitch_degrees=0.0,
        dex_thumb_rotation=0.25,
        glove_inverted=False,
        glove_smoothing=0.90,
        glove_deadzone=0.10,
        glove_closed_deadzone=None,
        finger_curve_gamma=1.4,
    ):
        if env.controller.control_mode != "ik":
            raise ValueError("TeleopMapper requires env control_mode='ik'.")
        self.env = env
        self.position_scale = float(position_scale)
        if not math.isfinite(workspace_yaw_degrees):
            raise ValueError("workspace_yaw_degrees must be finite.")
        yaw = math.radians(float(workspace_yaw_degrees))
        self.workspace_rotation = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        if not math.isfinite(neutral_hand_pitch_degrees):
            raise ValueError("neutral_hand_pitch_degrees must be finite.")
        pitch = math.radians(float(neutral_hand_pitch_degrees))
        neutral_pitch_rotation = np.asarray(
            [
                [math.cos(pitch), 0.0, math.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch), 0.0, math.cos(pitch)],
            ],
            dtype=float,
        )
        initial_action = self.env.controller.current_ik_action(self.env.model, self.env.data)
        initial_robot_rotation = quaternion_to_rotation_matrix(initial_action[3:7])
        self._neutral_robot_quaternion = rotation_matrix_to_quaternion_wxyz(
            neutral_pitch_rotation @ initial_robot_rotation
        )
        if not 0.0 <= dex_thumb_rotation <= 1.0:
            raise ValueError("dex_thumb_rotation must be in [0, 1].")
        self.dex_thumb_rotation = float(dex_thumb_rotation)
        self.glove_inverted = glove_inverted
        self.glove_filter = GloveValueFilter(
            glove_smoothing,
            glove_deadzone,
            glove_closed_deadzone,
        )
        if not math.isfinite(finger_curve_gamma) or finger_curve_gamma <= 0:
            raise ValueError("finger_curve_gamma must be positive and finite.")
        self.finger_curve_gamma = float(finger_curve_gamma)
        self._vive_origin = None
        self._robot_origin = None
        self.last_hand_values = np.asarray(
            [0.0, 0.0, 0.0, 0.0, self.dex_thumb_rotation, 0.0],
            dtype=np.float32,
        )

    def calibrate(self, vive: ViveSample) -> None:
        if not vive.valid:
            raise ValueError("Cannot calibrate from an invalid Vive pose.")
        action = self.env.controller.current_ik_action(self.env.model, self.env.data)
        vive_position, vive_rotation = remap_pose(vive.position, vive.quaternion_wxyz)
        self._vive_origin = (vive_position, vive_rotation)
        self._robot_origin = (
            action[:3].astype(float),
            self._neutral_robot_quaternion.copy(),
        )
        self.glove_filter.reset()

    def action(self, vive: ViveSample, glove: GloveSample) -> np.ndarray:
        if self._vive_origin is None or self._robot_origin is None:
            self.calibrate(vive)
        vp, vive_origin_rotation = self._vive_origin
        rp, rq = self._robot_origin
        vive_position, vive_rotation = remap_pose(vive.position, vive.quaternion_wxyz)
        target_pos = rp + self.position_scale * (self.workspace_rotation @ (vive_position - vp))
        robot_origin_rotation = quaternion_to_rotation_matrix(rq)
        relative_rotation = vive_rotation @ vive_origin_rotation.T
        relative_rotation = self.workspace_rotation @ relative_rotation @ self.workspace_rotation.T
        target_q = rotation_matrix_to_quaternion_wxyz(relative_rotation @ robot_origin_rotation)

        hand = np.asarray(glove.stretch, dtype=np.float32).reshape(-1)
        if hand.shape != (6,):
            raise ValueError(f"Glove sample must have six channels, got {hand.shape}.")
        hand = self.glove_filter.update(hand)
        # Device and dex-hand conventions both use 0=open and 1=flexed.
        if self.glove_inverted:
            hand = 1.0 - hand
        # The linkage moves much more visibly in the first part of its pushrod
        # travel. Shape flexion commands so the operator gets finer control
        # near the open pose and more useful travel in the second half.
        linear_hand = hand.copy()
        hand = hand.copy()
        hand[[0, 1, 2, 3, 5]] = np.power(hand[[0, 1, 2, 3, 5]], self.finger_curve_gamma)
        self.last_hand_values = hand.copy()
        self.last_hand_values[4] = self.dex_thumb_rotation
        controller = self.env.controller.hand_controller
        count = controller.action_size
        if count == 1:
            # The Pika position actuator uses the opposite convention from
            # glove flexion: actuator low is closed and high is open. Control
            # it from the glove's thumb-flex channel (5); channel 4 is the
            # duplicated/fixed thumb-opposition signal used only by Dex Hand.
            thumb_flexion = float(linear_hand[5])
            normalized = np.asarray([1.0 - thumb_flexion], dtype=np.float32)
            self.last_hand_values = linear_hand.copy()
            self.last_hand_values[4] = self.dex_thumb_rotation
        elif count == 6:
            # GloveSample: index, middle, ring, pinky, thumb rotate, thumb grasp.
            # The glove and this Dex Hand use opposite four-finger actuator
            # order. Thumb actuator order already matches.
            normalized = hand[[3, 2, 1, 0, 4, 5]].copy()
            # Keep thumb opposition fixed; the glove's thumb sensor controls
            # only thumb grasp/flexion (channel 5).
            normalized[4] = self.dex_thumb_rotation
        else:
            raise ValueError(f"Unsupported hand action size {count}; expected 1 or 6.")
        low = np.asarray(self.env.action_space.low[-count:], dtype=np.float32)
        high = np.asarray(self.env.action_space.high[-count:], dtype=np.float32)
        hand_action = low + np.clip(normalized, 0, 1) * (high - low)
        action = np.concatenate([target_pos, target_q, hand_action]).astype(np.float32)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)
