# -*- coding: utf-8 -*-
"""Lift manipulation task."""
from __future__ import annotations

import mujoco
import numpy as np

from source.environments.core.registry import register_task
from source.environments.manipulation.base import SingleArmManipulationTask
from source.environments.manipulation.objects import FreeBoxSpec


@register_task("lift")
class LiftTask(SingleArmManipulationTask):
    success_reward = 2.25

    @property
    def name(self) -> str: return "lift"

    def create_objects(self):
        return (FreeBoxSpec(name="cube", half_size=(.021,.021,.021), rgba=(.86,.12,.10,1.0)),)

    def compute_task_reward(self, obs, action, model, data, success: bool):
        _=action
        reward=self.success_reward if success else 0.0
        info={}
        if not success and self.reward_shaping:
            reaching=1.0-np.tanh(10.0*float(np.linalg.norm(obs["gripper_to_cube_pos"])))
            contact=.25 if self._is_robot_touching_object(model,data,"cube") else 0.0
            reward += reaching + contact
            info.update(reward_reaching=float(reaching), reward_contact=float(contact))
        return self.scale_reward(float(reward)), info

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        return bool(self._body_pos(model,data,"cube")[2] > self.table_top_z + .04)
