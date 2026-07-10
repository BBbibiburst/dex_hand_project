# -*- coding: utf-8 -*-
"""End-effector controllers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np

from source.control.common import _actuator_ids_or_raise, _empty_box, prefixed_names
from source.robots.descriptors import EndEffectorDescriptor

class EndEffectorPositionController:
    """Direct position controller for a hand/gripper descriptor."""

    def __init__(
        self,
        *,
        hand_descriptor: EndEffectorDescriptor,
        hand_prefix: str | None = None,
        include_action: bool = True,
        normalized_position: bool = False,
        reset_to_current_position: bool = True,
    ) -> None:
        self.hand_descriptor = hand_descriptor
        self.hand_prefix = (
            hand_descriptor.default_prefix if hand_prefix is None else hand_prefix
        )
        self.include_action = include_action
        self.normalized_position = normalized_position
        self.reset_to_current_position = reset_to_current_position
        self.local_action_names = tuple(hand_descriptor.position_actuator_names)
        self.actuator_names = (
            prefixed_names(self.local_action_names, self.hand_prefix)
            if include_action
            else ()
        )

        self._actuator_ids: Optional[np.ndarray] = None
        self._qpos_addrs: Optional[np.ndarray] = None
        self._ctrl_low: Optional[np.ndarray] = None
        self._ctrl_high: Optional[np.ndarray] = None
        self._action_space = self._unbound_action_space()

    @property
    def action_size(self) -> int:
        return len(self.actuator_names)

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def actuator_ids(self) -> np.ndarray:
        if self._actuator_ids is None:
            raise RuntimeError("EndEffectorPositionController.bind() must be called first.")
        return self._actuator_ids

    @property
    def qpos_addrs(self) -> np.ndarray:
        if self._qpos_addrs is None:
            raise RuntimeError("EndEffectorPositionController.bind() must be called first.")
        return self._qpos_addrs

    @property
    def ctrl_low(self) -> np.ndarray:
        if self._ctrl_low is None:
            raise RuntimeError("EndEffectorPositionController.bind() must be called first.")
        return self._ctrl_low

    @property
    def ctrl_high(self) -> np.ndarray:
        if self._ctrl_high is None:
            raise RuntimeError("EndEffectorPositionController.bind() must be called first.")
        return self._ctrl_high

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        _ = data
        if self.action_size == 0:
            self._actuator_ids = np.zeros(0, dtype=np.int32)
            self._qpos_addrs = np.zeros(0, dtype=np.int32)
            self._ctrl_low = np.zeros(0, dtype=np.float32)
            self._ctrl_high = np.zeros(0, dtype=np.float32)
            self._action_space = _empty_box()
            return

        self._actuator_ids = _actuator_ids_or_raise(
            model,
            self.actuator_names,
            owner=f"{self.hand_descriptor.name} controller",
        )
        joint_ids = model.actuator_trnid[self._actuator_ids, 0].astype(np.int32)
        self._qpos_addrs = model.jnt_qposadr[joint_ids].astype(np.int32)
        ctrlrange = model.actuator_ctrlrange[self._actuator_ids].astype(np.float32)
        self._ctrl_low = ctrlrange[:, 0].copy()
        self._ctrl_high = ctrlrange[:, 1].copy()
        self._action_space = self._bound_action_space()

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = rng
        _ = options
        if self.action_size == 0:
            return {
                "hand_controller": self.hand_descriptor.name,
                "hand_position_actuators": self.actuator_names,
            }

        if self.reset_to_current_position:
            target = np.clip(data.qpos[self.qpos_addrs], self.ctrl_low, self.ctrl_high)
        else:
            target = np.clip(
                np.zeros(self.action_size, dtype=np.float32),
                self.ctrl_low,
                self.ctrl_high,
            )
        data.ctrl[self.actuator_ids] = target
        mujoco.mj_forward(model, data)
        return {
            "hand_controller": self.hand_descriptor.name,
            "hand_position_target": target.astype(np.float32).copy(),
            "hand_position_actuators": self.actuator_names,
        }

    def apply_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        _ = model
        target = self._coerce_action(action)
        if self.action_size:
            data.ctrl[self.actuator_ids] = target
        return {
            "hand_controller": self.hand_descriptor.name,
            "hand_position_target": target.astype(np.float32).copy(),
        }

    def current_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        _ = model
        if self.action_size == 0:
            return np.zeros(0, dtype=np.float32)
        return data.ctrl[self.actuator_ids].astype(np.float32).copy()

    def action_layout(self) -> tuple[str, ...]:
        return self.local_action_names if self.include_action else ()

    def _coerce_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32)
        expected_shape = (self.action_size,)
        if action_arr.shape != expected_shape:
            raise ValueError(
                f"{self.hand_descriptor.name} hand action must have shape "
                f"{expected_shape}, got {action_arr.shape}."
            )
        if self.action_size == 0:
            return action_arr
        if self.normalized_position:
            clipped = np.clip(action_arr, -1.0, 1.0)
            midpoint = 0.5 * (self.ctrl_low + self.ctrl_high)
            half_range = 0.5 * (self.ctrl_high - self.ctrl_low)
            return (midpoint + clipped * half_range).astype(np.float32)
        return np.clip(action_arr, self.ctrl_low, self.ctrl_high).astype(np.float32)

    def _bound_action_space(self) -> spaces.Box:
        if self.action_size == 0:
            return _empty_box()
        if self.normalized_position:
            return spaces.Box(
                low=-np.ones(self.action_size, dtype=np.float32),
                high=np.ones(self.action_size, dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Box(
            low=self.ctrl_low.astype(np.float32),
            high=self.ctrl_high.astype(np.float32),
            dtype=np.float32,
        )

    def _unbound_action_space(self) -> spaces.Box:
        if self.action_size == 0:
            return _empty_box()
        if self.normalized_position:
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.action_size,),
                dtype=np.float32,
            )
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.action_size,),
            dtype=np.float32,
        )


class PikaGripperController(EndEffectorPositionController):
    """Single-DOF opening-width controller for the Pika parallel gripper.

    The MJCF exposes one actuated slide joint whose ctrl range is ``[-0.05, 0]``:
    ``0`` is fully open, ``-0.05`` is fully closed. This controller presents a
    gripper-level action instead: jaw opening width in meters, where ``0`` is
    closed and ``0.1`` is fully open.
    """

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().bind(model, data)
        if self.action_size not in (0, 1):
            raise ValueError(
                f"{self.hand_descriptor.name} expects exactly one gripper actuator, "
                f"got {self.action_size}."
            )
        if self.action_size:
            self._action_space = self._bound_action_space()

    @property
    def max_opening(self) -> float:
        if self.action_size == 0:
            return 0.0
        return float(2.0 * (self.ctrl_high[0] - self.ctrl_low[0]))

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        info = super().reset(model, data, rng=rng, options=options)
        if self.action_size:
            info["gripper_opening"] = self.current_action(model, data)
        return info

    def apply_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        target = self._coerce_action(action)
        if self.action_size:
            joint_target = self._opening_to_joint_target(target)
            data.ctrl[self.actuator_ids] = joint_target
        else:
            joint_target = target
        return {
            "hand_controller": self.hand_descriptor.name,
            "gripper_opening": target.astype(np.float32).copy(),
            "hand_position_target": joint_target.astype(np.float32).copy(),
        }

    def current_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        _ = model
        if self.action_size == 0:
            return np.zeros(0, dtype=np.float32)
        joint_target = data.ctrl[self.actuator_ids].astype(np.float32)
        return self._joint_target_to_opening(joint_target).astype(np.float32)

    def action_layout(self) -> tuple[str, ...]:
        return ("gripper_opening",) if self.include_action else ()

    def _coerce_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32)
        expected_shape = (self.action_size,)
        if action_arr.shape != expected_shape:
            raise ValueError(
                f"{self.hand_descriptor.name} gripper action must have shape "
                f"{expected_shape}, got {action_arr.shape}."
            )
        if self.action_size == 0:
            return action_arr
        if self.normalized_position:
            clipped = np.clip(action_arr, -1.0, 1.0)
            return (0.5 * (clipped + 1.0) * self.max_opening).astype(np.float32)
        return np.clip(action_arr, 0.0, self.max_opening).astype(np.float32)

    def _bound_action_space(self) -> spaces.Box:
        if self.action_size == 0:
            return _empty_box()
        if self.normalized_position:
            return spaces.Box(
                low=-np.ones(self.action_size, dtype=np.float32),
                high=np.ones(self.action_size, dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Box(
            low=np.zeros(self.action_size, dtype=np.float32),
            high=np.full(self.action_size, self.max_opening, dtype=np.float32),
            dtype=np.float32,
        )

    def _opening_to_joint_target(self, opening: np.ndarray) -> np.ndarray:
        return (self.ctrl_low + 0.5 * opening).astype(np.float32)

    def _joint_target_to_opening(self, joint_target: np.ndarray) -> np.ndarray:
        return (2.0 * (joint_target - self.ctrl_low)).astype(np.float32)
