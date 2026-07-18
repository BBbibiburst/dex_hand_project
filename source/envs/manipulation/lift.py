# -*- coding: utf-8 -*-
"""Lift manipulation task."""

from __future__ import annotations

import mujoco
import numpy as np

from source.envs.core.registry import register_task
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.object_catalog import DEFAULT_LIFT_OBJECT, lift_object_ids
from source.envs.manipulation.objects import MeshObjectSpec


@register_task("lift")
class LiftTask(SingleArmManipulationTask):
    """Lift one catalogue object above the table."""

    success_reward = 2.25

    def __init__(self, *, object_id: str = DEFAULT_LIFT_OBJECT, **kwargs) -> None:
        if object_id not in lift_object_ids():
            raise ValueError(f"Object {object_id!r} is not available for lift.")
        self.object_id = object_id
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "lift"

    def create_objects(self) -> tuple[MeshObjectSpec, ...]:
        return (MeshObjectSpec(name="object", object_id=self.object_id),)

    def compute_task_reward(self, obs, action, model, data, success: bool):
        _ = action
        reward = self.success_reward if success else 0.0
        info: dict[str, float] = {}

        if not success and self.reward_shaping:
            distance = float(np.linalg.norm(obs["gripper_to_object_pos"]))
            reaching = 1.0 - np.tanh(10.0 * distance)
            contact = 0.25 if self._is_robot_touching_object(model, data, "object") else 0.0
            reward += reaching + contact
            info.update(
                reward_reaching=float(reaching),
                reward_contact=float(contact),
            )

        return self.scale_reward(float(reward)), info

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        object_z = self._body_pos(model, data, "object")[2]
        return bool(object_z > self.table_top_z + self.objects[0].bottom_offset + 0.04)
