# -*- coding: utf-8 -*-
"""Stack manipulation task."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from source.envs.core.registry import register_task
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.objects import FreeBoxSpec
from source.envs.manipulation.placement import UniformTablePlacementSampler


@register_task("stack")
class StackTask(SingleArmManipulationTask):
    """Stack cube A on cube B."""

    success_reward = 2.0

    def __init__(self, **kwargs: Any) -> None:
        sampler = kwargs.pop(
            "placement_sampler",
            UniformTablePlacementSampler(
                x_range=(-0.08, 0.08),
                y_range=(-0.08, 0.08),
                min_separation=0.10,
            ),
        )
        super().__init__(placement_sampler=sampler, **kwargs)

    @property
    def name(self) -> str:
        return "stack"

    def create_objects(self) -> tuple[FreeBoxSpec, ...]:
        return (
            FreeBoxSpec(
                name="cubeA",
                half_size=(0.02, 0.02, 0.02),
                rgba=(0.86, 0.12, 0.10, 1.0),
            ),
            FreeBoxSpec(
                name="cubeB",
                half_size=(0.025, 0.025, 0.025),
                rgba=(0.18, 0.62, 0.20, 1.0),
            ),
        )

    def extra_observation_space(self) -> dict[str, spaces.Space]:
        return {
            "cubeA_to_cubeB_pos": spaces.Box(
                -np.inf,
                np.inf,
                shape=(3,),
                dtype=np.float32,
            )
        }

    def get_extra_observation(self, model, data, obs) -> dict[str, np.ndarray]:
        _ = model
        _ = data
        return {"cubeA_to_cubeB_pos": (obs["cubeB_pos"] - obs["cubeA_pos"]).astype(np.float32)}

    def compute_task_reward(self, obs, action, model, data, success: bool):
        _ = obs
        _ = action
        _ = success

        reach, lift, stack = self.staged_rewards(model, data)
        reward = (
            max(reach, lift, stack)
            if self.reward_shaping
            else (self.success_reward if stack > 0.0 else 0.0)
        )
        return self.scale_reward(float(reward)), {
            "reward_reach": reach,
            "reward_lift": lift,
            "reward_stack": stack,
        }

    def staged_rewards(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, float, float]:
        bindings = self._require_bindings()
        cube_a_pos = self._body_pos(model, data, "cubeA")
        cube_b_pos = self._body_pos(model, data, "cubeB")
        ee_pos = (
            np.zeros(3, dtype=np.float64)
            if bindings.ee_site_id is None
            else data.site_xpos[bindings.ee_site_id]
        )

        reach = 0.25 * (1.0 - np.tanh(10.0 * float(np.linalg.norm(ee_pos - cube_a_pos))))
        grasping = self._is_robot_touching_object(model, data, "cubeA")
        if grasping:
            reach += 0.25

        lifted = cube_a_pos[2] > self.table_top_z + 0.04
        lift = 1.0 if lifted else 0.0
        if lifted:
            horizontal_distance = float(np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2]))
            lift += 0.5 * (1.0 - np.tanh(horizontal_distance))

        stacked = not grasping and lifted and self._objects_touching(data, "cubeA", "cubeB")
        stack = self.success_reward if stacked else 0.0
        return float(reach), float(lift), float(stack)

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        return self.staged_rewards(model, data)[2] > 0.0
