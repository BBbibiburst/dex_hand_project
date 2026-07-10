# -*- coding: utf-8 -*-
"""Lift manipulation task."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import mujoco
import numpy as np

from source.environments.manipulation.base import SingleArmManipulationTask
from source.environments.manipulation.objects import FreeBoxSpec
from source.environments.tasks import Observation


Array = np.ndarray


class LiftTask(SingleArmManipulationTask):
    """Single-block lifting task ported onto RobotGymEnv."""

    @property
    def name(self) -> str:
        return "lift"

    @property
    def boxes(self) -> Tuple[FreeBoxSpec, ...]:
        return (
            FreeBoxSpec(
                name="cube",
                half_size=(0.021, 0.021, 0.021),
                rgba=(0.86, 0.12, 0.10, 1.0),
            ),
        )

    def compute_reward(
        self,
        obs: Observation,
        action: Array,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Tuple[float, Dict[str, Any]]:
        _ = action
        success = self.check_success(model, data)
        reward = 2.25 if success else 0.0

        if not success and self.reward_shaping:
            dist = float(np.linalg.norm(obs["gripper_to_cube_pos"]))
            reward += 1.0 - np.tanh(10.0 * dist)
            if self._is_robot_touching_object(model, data, "cube"):
                reward += 0.25

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return float(reward), {"task_success": bool(success)}

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        cube_height = self._body_pos(model, data, "cube")[2]
        return bool(cube_height > self.table_top_z + 0.04)
