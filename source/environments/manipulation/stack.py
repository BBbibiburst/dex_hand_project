# -*- coding: utf-8 -*-
"""Stack manipulation task."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from gymnasium import spaces
import mujoco
import numpy as np

from source.environments.manipulation.base import SingleArmManipulationTask
from source.environments.manipulation.objects import FreeBoxSpec
from source.environments.manipulation.placement import UniformTablePlacementSampler
from source.environments.tasks import Observation


Array = np.ndarray


class StackTask(SingleArmManipulationTask):
    """Two-block stacking task ported onto RobotGymEnv."""

    def __init__(self, **kwargs: Any) -> None:
        placement_sampler = kwargs.pop(
            "placement_sampler",
            UniformTablePlacementSampler(
                x_range=(-0.08, 0.08),
                y_range=(-0.08, 0.08),
                min_separation=0.10,
            ),
        )
        super().__init__(placement_sampler=placement_sampler, **kwargs)

    @property
    def name(self) -> str:
        return "stack"

    @property
    def boxes(self) -> Tuple[FreeBoxSpec, ...]:
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

    @property
    def observation_space(self) -> Dict[str, spaces.Space]:
        obs_spaces = super().observation_space
        obs_spaces["cubeA_to_cubeB_pos"] = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32
        )
        return obs_spaces

    def get_observation(self, model: mujoco.MjModel, data: mujoco.MjData) -> Observation:
        obs = super().get_observation(model, data)
        obs["cubeA_to_cubeB_pos"] = (
            obs["cubeB_pos"] - obs["cubeA_pos"]
        ).astype(np.float32)
        return obs

    def compute_reward(
        self,
        obs: Observation,
        action: Array,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Tuple[float, Dict[str, Any]]:
        _ = obs, action
        r_reach, r_lift, r_stack = self.staged_rewards(model, data)
        reward = max(r_reach, r_lift, r_stack) if self.reward_shaping else 0.0
        if not self.reward_shaping and r_stack > 0.0:
            reward = 2.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return float(reward), {
            "task_success": bool(r_stack > 0.0),
            "stack_reach_reward": float(r_reach),
            "stack_lift_reward": float(r_lift),
            "stack_reward": float(r_stack),
        }

    def staged_rewards(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, float, float]:
        cube_a_pos = self._body_pos(model, data, "cubeA")
        cube_b_pos = self._body_pos(model, data, "cubeB")
        ee_pos = (
            data.site_xpos[self._ee_site_id]
            if self._ee_site_id is not None
            else np.zeros(3, dtype=np.float64)
        )
        dist = float(np.linalg.norm(ee_pos - cube_a_pos))
        r_reach = 0.25 * (1.0 - np.tanh(10.0 * dist))

        grasping_cube_a = self._is_robot_touching_object(model, data, "cubeA")
        if grasping_cube_a:
            r_reach += 0.25

        cube_a_lifted = cube_a_pos[2] > self.table_top_z + 0.04
        r_lift = 1.0 if cube_a_lifted else 0.0
        if cube_a_lifted:
            horiz_dist = float(np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2]))
            r_lift += 0.5 * (1.0 - np.tanh(horiz_dist))

        r_stack = 0.0
        if (
            not grasping_cube_a
            and cube_a_lifted
            and self._objects_touching(data, "cubeA", "cubeB")
        ):
            r_stack = 2.0

        return float(r_reach), float(r_lift), float(r_stack)

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        _, _, r_stack = self.staged_rewards(model, data)
        return bool(r_stack > 0.0)
