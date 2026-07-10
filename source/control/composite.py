# -*- coding: utf-8 -*-
"""Composite robot controllers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np

from source.control.arm import ArmPositionIkController
from source.control.end_effectors import EndEffectorPositionController
from source.robots.descriptors import ArmDescriptor, EndEffectorDescriptor

class CompositeRobotController:
    """Compose an arm controller and an end-effector controller."""

    def __init__(
        self,
        *,
        arm_controller: ArmPositionIkController,
        hand_controller: EndEffectorPositionController,
    ) -> None:
        self.arm_controller = arm_controller
        self.hand_controller = hand_controller
        self._action_space = self._combine_action_spaces()

    @property
    def control_mode(self) -> str:
        return self.arm_controller.control_mode

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return self.arm_controller.actuator_names + self.hand_controller.actuator_names

    @property
    def arm_actuator_count(self) -> int:
        """Number of direct position actuators owned by the arm controller.

        Kept as a compatibility property for demos that need to split arm and
        hand position targets.
        """
        return self.arm_controller.position_action_size

    @property
    def include_hand_action(self) -> bool:
        """Whether the composed action currently includes hand commands."""
        return self.hand_controller.action_size > 0

    @property
    def ctrl_low(self) -> np.ndarray:
        """Combined actuator lower bounds in arm-then-hand order."""
        return np.concatenate(
            [self.arm_controller.ctrl_low, self.hand_controller.ctrl_low]
        ).astype(np.float32)

    @property
    def ctrl_high(self) -> np.ndarray:
        """Combined actuator upper bounds in arm-then-hand order."""
        return np.concatenate(
            [self.arm_controller.ctrl_high, self.hand_controller.ctrl_high]
        ).astype(np.float32)

    def set_timestep(self, dt: float) -> None:
        self.arm_controller.set_timestep(dt)

    def set_control_mode(self, control_mode: str) -> None:
        self.arm_controller.set_control_mode(control_mode)
        self._action_space = self._combine_action_spaces()

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.arm_controller.bind(model, data)
        self.hand_controller.bind(model, data)
        self._action_space = self._combine_action_spaces()

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        info.update(self.arm_controller.reset(model, data, rng=rng, options=options))
        info.update(self.hand_controller.reset(model, data, rng=rng, options=options))
        info.update(
            {
                "controller": "composite",
                "position_actuators": self.actuator_names,
                "position_target": data.ctrl[
                    np.concatenate(
                        [
                            self.arm_controller.actuator_ids,
                            self.hand_controller.actuator_ids,
                        ]
                    )
                ].astype(np.float32).copy(),
                "ik_action_layout": self.ik_action_layout(),
            }
        )
        return info

    def apply_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: Any,
    ) -> Dict[str, Any]:
        action_arr = np.asarray(action, dtype=np.float32)
        expected_shape = self.action_space.shape
        if action_arr.shape != expected_shape:
            raise ValueError(
                f"Composite action must have shape {expected_shape}, got {action_arr.shape}."
            )

        arm_size = self.arm_controller.action_size
        arm_action = action_arr[:arm_size]
        hand_action = action_arr[arm_size:]
        info: Dict[str, Any] = {}
        info.update(self.arm_controller.apply_action(model, data, arm_action))
        info.update(self.hand_controller.apply_action(model, data, hand_action))
        if self.control_mode == "ik":
            info["position_target"] = self._current_position_target(data)
        else:
            info["position_target"] = action_arr.copy()
        return info

    def current_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        if self.control_mode == "ik":
            return self.current_ik_action(model, data)
        return np.concatenate(
            [
                self.arm_controller.current_action(model, data),
                self.hand_controller.current_action(model, data),
            ]
        ).astype(np.float32)

    def current_ik_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return np.concatenate(
            [
                self.arm_controller.current_ik_action(model, data),
                self.hand_controller.current_action(model, data),
            ]
        ).astype(np.float32)

    def ik_action_layout(self) -> tuple[str, ...]:
        return self.arm_controller.ik_action_layout() + self.hand_controller.action_layout()

    def _combine_action_spaces(self) -> spaces.Box:
        arm_space = self.arm_controller.action_space
        hand_space = self.hand_controller.action_space
        return spaces.Box(
            low=np.concatenate(
                [
                    np.asarray(arm_space.low, dtype=np.float32).reshape(-1),
                    np.asarray(hand_space.low, dtype=np.float32).reshape(-1),
                ]
            ),
            high=np.concatenate(
                [
                    np.asarray(arm_space.high, dtype=np.float32).reshape(-1),
                    np.asarray(hand_space.high, dtype=np.float32).reshape(-1),
                ]
            ),
            dtype=np.float32,
        )

    def _current_position_target(self, data: mujoco.MjData) -> np.ndarray:
        actuator_ids = np.concatenate(
            [self.arm_controller.actuator_ids, self.hand_controller.actuator_ids]
        )
        return data.ctrl[actuator_ids].astype(np.float32).copy()


def build_robot_controller(
    *,
    arm_descriptor: ArmDescriptor,
    hand_descriptor: EndEffectorDescriptor,
    hand_prefix: str | None = None,
    control_mode: str = "position",
    ee_site_name: str | None = None,
    include_hand_action: bool = True,
    normalized_position: bool = False,
    **arm_controller_kwargs: Any,
) -> CompositeRobotController:
    """Build a composite controller from descriptor-declared factories."""

    arm_factory = arm_descriptor.controller_factory or ArmPositionIkController
    hand_factory = hand_descriptor.controller_factory or EndEffectorPositionController

    arm_controller = arm_factory(
        arm_descriptor=arm_descriptor,
        control_mode=control_mode,
        ee_site_name=ee_site_name or arm_descriptor.ee_site_name,
        normalized_position=normalized_position,
        **arm_controller_kwargs,
    )
    hand_controller = hand_factory(
        hand_descriptor=hand_descriptor,
        hand_prefix=hand_prefix,
        include_action=include_hand_action,
        normalized_position=normalized_position,
    )
    return CompositeRobotController(
        arm_controller=arm_controller,
        hand_controller=hand_controller,
    )


class RobotPositionIkController(CompositeRobotController):
    """Compatibility wrapper for older code that constructed one controller."""

    def __init__(
        self,
        *,
        arm_descriptor: ArmDescriptor | None = None,
        hand_descriptor: EndEffectorDescriptor | None = None,
        hand_prefix: str | None = None,
        control_mode: str = "position",
        ee_site_name: str | None = None,
        include_hand_action: bool = True,
        normalized_position: bool = False,
        **arm_controller_kwargs: Any,
    ) -> None:
        if arm_descriptor is None or hand_descriptor is None:
            from source.robots.defaults import DEFAULT_ARM, DEFAULT_HAND

            arm_descriptor = DEFAULT_ARM if arm_descriptor is None else arm_descriptor
            hand_descriptor = DEFAULT_HAND if hand_descriptor is None else hand_descriptor

        composite = build_robot_controller(
            arm_descriptor=arm_descriptor,
            hand_descriptor=hand_descriptor,
            hand_prefix=hand_prefix,
            control_mode=control_mode,
            ee_site_name=ee_site_name,
            include_hand_action=include_hand_action,
            normalized_position=normalized_position,
            **arm_controller_kwargs,
        )
        super().__init__(
            arm_controller=composite.arm_controller,
            hand_controller=composite.hand_controller,
        )
