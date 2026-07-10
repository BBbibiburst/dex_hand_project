"""Calibrated Vive/glove samples to the project's absolute IK action."""
from __future__ import annotations

import numpy as np

from source.geometry import normalize_quat, quat_conjugate, quat_multiply
from source.teleop.devices import GloveSample, ViveSample



class TeleopMapper:
    """Relative Vive motion anchored to the robot EE pose at calibration."""

    def __init__(self, env, *, position_scale=1.0, glove_inverted=False):
        if env.controller.control_mode != "ik":
            raise ValueError("TeleopMapper requires env control_mode='ik'.")
        self.env = env
        self.position_scale = float(position_scale)
        self.glove_inverted = glove_inverted
        self._vive_origin = None
        self._robot_origin = None

    def calibrate(self, vive: ViveSample) -> None:
        if not vive.valid:
            raise ValueError("Cannot calibrate from an invalid Vive pose.")
        action = self.env.controller.current_ik_action(self.env.model, self.env.data)
        self._vive_origin = (vive.position.astype(float), vive.quaternion_wxyz.astype(float))
        self._robot_origin = (action[:3].astype(float), action[3:7].astype(float))

    def action(self, vive: ViveSample, glove: GloveSample) -> np.ndarray:
        if self._vive_origin is None or self._robot_origin is None:
            self.calibrate(vive)
        vp, vq = self._vive_origin
        rp, rq = self._robot_origin
        target_pos = rp + self.position_scale * (vive.position - vp)
        relative_q = quat_multiply(vive.quaternion_wxyz, quat_conjugate(vq))
        target_q = quat_multiply(relative_q, rq)
        target_q = normalize_quat(target_q)

        hand = np.asarray(glove.stretch, dtype=np.float32).reshape(-1)
        if hand.shape != (6,):
            raise ValueError(f"Glove sample must have six channels, got {hand.shape}.")
        # Device convention is 0=stretched/open and 1=flexed/closed.
        # Controller position ranges conventionally increase towards open.
        if not self.glove_inverted:
            hand = 1.0 - hand
        controller = self.env.controller.hand_controller
        count = controller.action_size
        if count == 1:  # parallel gripper: use only one glove channel
            normalized = hand[:1]
        elif count == 6:  # dex hand: one channel per actuator
            normalized = hand
        else:
            raise ValueError(f"Unsupported hand action size {count}; expected 1 or 6.")
        low = np.asarray(self.env.action_space.low[-count:], dtype=np.float32)
        high = np.asarray(self.env.action_space.high[-count:], dtype=np.float32)
        hand_action = low + np.clip(normalized, 0, 1) * (high - low)
        action = np.concatenate([target_pos, target_q, hand_action]).astype(np.float32)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)
