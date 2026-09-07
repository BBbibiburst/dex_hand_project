# -*- coding: utf-8 -*-
"""Stack manipulation task."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from source.envs.core.registry import register_task
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.object_catalog import (
    DEFAULT_STACK_OBJECTS,
    lift_object_ids,
    stack_object_ids,
)
from source.envs.manipulation.objects import FreeBoxSpec, MeshObjectSpec
from source.envs.manipulation.placement import (
    STACK_REACHABLE_REGION,
    UniformTablePlacementSampler,
)


@register_task("stack")
class StackTask(SingleArmManipulationTask):
    """Stack cube A on cube B."""

    success_reward = 2.0

    def __init__(
        self,
        *,
        object_ids: tuple[str, str] = DEFAULT_STACK_OBJECTS,
        stacked_object_id: str | None = None,
        stable_steps: int = 25,
        support_height: float = 0.07,
        support_center: tuple[float, float] = (0.46, 0.18),
        support_half_size: float = 0.05,
        **kwargs: Any,
    ) -> None:
        if stable_steps < 1:
            raise ValueError("stable_steps must be at least one.")
        self.stable_steps = int(stable_steps)
        self._stack_stable_count = 0
        if not 0.055 <= support_height <= 0.085:
            raise ValueError("support_height must lie in [0.055, 0.085] metres.")
        self.support_height = float(support_height)
        if not 0.045 <= support_half_size <= 0.09:
            raise ValueError("support_half_size must lie in [0.045, 0.09] metres.")
        self.support_half_size = float(support_half_size)
        self.support_center = (float(support_center[0]), float(support_center[1]))
        if stacked_object_id is None:
            if len(object_ids) != 2 or any(item not in stack_object_ids() for item in object_ids):
                raise ValueError("object_ids must contain two stack-compatible catalogue IDs.")
            self.object_ids = tuple(object_ids)
        else:
            if stacked_object_id not in lift_object_ids():
                raise ValueError(f"Unknown stacked_object_id {stacked_object_id!r}.")
            self.object_ids = (stacked_object_id, "standard_support")
        sampler = kwargs.pop(
            "placement_sampler",
            UniformTablePlacementSampler.for_object_centers(
                STACK_REACHABLE_REGION,
                min_separation=0.09,
            ),
        )
        super().__init__(placement_sampler=sampler, **kwargs)

    @property
    def name(self) -> str:
        return "stack"

    def create_objects(self):
        if self.object_ids[1] == "standard_support":
            return (
                MeshObjectSpec(name="cubeA", object_id=self.object_ids[0]),
                FreeBoxSpec(
                    name="cubeB",
                    half_size=(
                        self.support_half_size,
                        self.support_half_size,
                        0.5 * self.support_height,
                    ),
                    rgba=(0.18, 0.42, 0.78, 1.0),
                    fixed_to_world=True,
                ),
            )
        return (
            MeshObjectSpec(
                name="cubeA",
                object_id=self.object_ids[0],
                target_size=0.065,
            ),
            MeshObjectSpec(
                name="cubeB",
                object_id=self.object_ids[1],
                target_size=0.075,
            ),
        )

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        """Compile the standard support at its weld reference pose."""

        if self.object_ids[1] != "standard_support":
            super().augment_spec(spec)
            return
        self.arena.augment_spec(spec)
        stacked, support = self.objects
        stacked.add_to_spec(
            spec,
            np.asarray(
                [
                    self.table_offset[0],
                    self.table_offset[1],
                    self.table_top_z + stacked.bottom_offset + 0.002,
                ],
                dtype=np.float64,
            ),
        )
        support.add_to_spec(
            spec,
            np.asarray(
                [
                    self.support_center[0],
                    self.support_center[1],
                    self.table_top_z + support.bottom_offset,
                ],
                dtype=np.float64,
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

        stacked_now = not grasping and self._stable_platform_contact(model, data)
        stacked = stacked_now and self._stack_stable_count >= self.stable_steps
        stack = self.success_reward if stacked else 0.0
        return float(reach), float(lift), float(stack)

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        candidate = not self._is_robot_touching_object(
            model, data, "cubeA"
        ) and self._stable_platform_contact(model, data)
        self._stack_stable_count = self._stack_stable_count + 1 if candidate else 0
        return self._stack_stable_count >= self.stable_steps

    def _stable_platform_contact(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Recognise a physically stable placement in any object orientation."""

        cube_a_pos = self._body_pos(model, data, "cubeA")
        cube_b_pos = self._body_pos(model, data, "cubeB")
        support_top = cube_b_pos[2] + self.objects[1].bottom_offset
        if self.object_ids[1] == "standard_support":
            half_x, half_y, _ = self.objects[1].half_size
            inside_surface = (
                abs(float(cube_a_pos[0] - cube_b_pos[0])) <= half_x - 0.004
                and abs(float(cube_a_pos[1] - cube_b_pos[1])) <= half_y - 0.004
            )
        else:
            inside_surface = bool(
                np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2]) <= self.objects[1].horizontal_radius
            )
        binding = self._require_bindings().objects["cubeA"]
        velocity = data.qvel[binding.qvel_adr : binding.qvel_adr + 6]
        slow = bool(np.linalg.norm(velocity[:3]) <= 0.04 and np.linalg.norm(velocity[3:]) <= 0.5)
        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.arena.table_geom_name)
        touches_table = False
        if table_id >= 0:
            object_geoms = binding.geom_ids
            for index in range(data.ncon):
                pair = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
                if table_id in pair and pair.intersection(object_geoms):
                    touches_table = True
                    break
        return bool(
            inside_surface
            and cube_a_pos[2] >= support_top + 0.004
            and self._objects_touching(data, "cubeA", "cubeB")
            and not touches_table
            and slow
        )

    def reset(self, model, data, *, rng, options):
        info = super().reset(model, data, rng=rng, options=options)
        if self.object_ids[1] == "standard_support":
            support = self._require_bindings().objects["cubeB"]
            data.qpos[support.qpos_adr : support.qpos_adr + 3] = (
                self.support_center[0],
                self.support_center[1],
                self.table_top_z + self.objects[1].bottom_offset,
            )
            data.qpos[support.qpos_adr + 3 : support.qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[support.qvel_adr : support.qvel_adr + 6] = 0.0
            mujoco.mj_forward(model, data)
        self._stack_stable_count = 0
        return info
