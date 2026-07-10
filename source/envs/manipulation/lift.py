# -*- coding: utf-8 -*-
"""Lift manipulation task."""

from __future__ import annotations

import mujoco
import numpy as np

from source.envs.core.registry import register_task
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.objects import FreeBoxSpec


@register_task("lift")
class LiftTask(SingleArmManipulationTask):
    """Lift one cube above the table."""

    success_reward = 2.25

    @property
    def name(self) -> str:
        return "lift"

    def create_objects(self) -> tuple[FreeBoxSpec, ...]:
        return (
            FreeBoxSpec(
                name="cube",
                half_size=(0.021, 0.021, 0.021),
                rgba=(0.86, 0.12, 0.10, 1.0),
            ),
        )

    def compute_task_reward(self, obs, action, model, data, success: bool):
        _ = action
        reward = self.success_reward if success else 0.0
        info: dict[str, float] = {}

        if not success and self.reward_shaping:
            distance = float(np.linalg.norm(obs["gripper_to_cube_pos"]))
            reaching = 1.0 - np.tanh(10.0 * distance)
            contact = 0.25 if self._is_robot_touching_object(model, data, "cube") else 0.0
            reward += reaching + contact
            info.update(
                reward_reaching=float(reaching),
                reward_contact=float(contact),
            )

        return self.scale_reward(float(reward)), info

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        cube_z = self._body_pos(model, data, "cube")[2]
        return bool(cube_z > self.table_top_z + 0.04)
