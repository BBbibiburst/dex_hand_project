# -*- coding: utf-8 -*-
"""Planar object-pushing task with a randomized tabletop goal."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from source.envs.core.registry import register_task
from source.envs.manipulation.arenas import PushArena
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.object_catalog import DEFAULT_PUSH_OBJECT, push_object_ids
from source.envs.manipulation.objects import MeshObjectSpec


@register_task("push")
class PushTask(SingleArmManipulationTask):
    """Push one catalogue object into a marked target without lifting it."""

    success_reward = 2.0

    def __init__(
        self,
        *,
        object_id: str = DEFAULT_PUSH_OBJECT,
        success_radius: float = 0.055,
        stable_steps: int = 5,
        **kwargs: Any,
    ) -> None:
        if object_id not in push_object_ids():
            raise ValueError(f"Object {object_id!r} is not available for push.")
        if success_radius <= 0.0:
            raise ValueError("success_radius must be positive.")
        if stable_steps < 1:
            raise ValueError("stable_steps must be at least 1.")
        self.object_id = object_id
        self.success_radius = float(success_radius)
        self.stable_steps = int(stable_steps)
        self.target_site_id: int | None = None
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.initial_goal_distance = 1.0
        self._stable_count = 0
        kwargs["arena"] = kwargs.pop("arena", PushArena())
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "push"

    def create_objects(self):
        return (MeshObjectSpec("object", self.object_id),)

    def extra_observation_space(self):
        return {
            "target_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "object_to_target_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        }

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        arena: PushArena = self.arena  # type: ignore[assignment]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, arena.target_site_name)
        if site_id < 0:
            raise ValueError(f"Push target site {arena.target_site_name!r} was not compiled.")
        self.target_site_id = int(site_id)

    def reset(self, model, data, *, rng, options):
        if self.bindings is None:
            self.bind(model)
        bindings = self._require_bindings()
        obj = self.objects[0]
        binding = bindings.objects["object"]

        object_xy = np.asarray(
            [rng.uniform(0.43, 0.53), rng.uniform(-0.22, -0.10)],
            dtype=np.float64,
        )
        target_xy = np.asarray(
            [rng.uniform(0.52, 0.66), rng.uniform(0.10, 0.23)],
            dtype=np.float64,
        )
        object_pos = np.asarray(
            [object_xy[0], object_xy[1], self.table_top_z + obj.bottom_offset + 0.002]
        )
        data.qpos[binding.qpos_adr : binding.qpos_adr + 3] = object_pos
        data.qpos[binding.qpos_adr + 3 : binding.qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[binding.qvel_adr : binding.qvel_adr + 6] = 0.0

        self.target_pos = np.asarray(
            [target_xy[0], target_xy[1], self.table_top_z + 0.001],
            dtype=np.float64,
        )
        assert self.target_site_id is not None
        model.site_pos[self.target_site_id] = self.target_pos
        self.initial_goal_distance = max(float(np.linalg.norm(object_xy - target_xy)), 1e-6)
        self._stable_count = 0
        mujoco.mj_forward(model, data)
        return {
            "task": self.name,
            "task_objects": ("object",),
            "object_id": self.object_id,
            "target_pos": self.target_pos.astype(np.float32).copy(),
        }

    def get_extra_observation(self, model, data, obs):
        _ = model, data
        target = self.target_pos.astype(np.float32).copy()
        return {
            "target_pos": target,
            "object_to_target_pos": (target - obs["object_pos"]).astype(np.float32),
        }

    def check_success(self, model, data) -> bool:
        binding = self._require_bindings().objects["object"]
        position = self._body_pos(model, data, "object")
        distance = float(np.linalg.norm(position[:2] - self.target_pos[:2]))
        planar_speed = float(np.linalg.norm(data.qvel[binding.qvel_adr : binding.qvel_adr + 2]))
        maximum_z = self.table_top_z + self.objects[0].bottom_offset + 0.025
        candidate = (
            distance <= self.success_radius and position[2] <= maximum_z and planar_speed <= 0.08
        )
        self._stable_count = self._stable_count + 1 if candidate else 0
        return self._stable_count >= self.stable_steps

    def compute_task_reward(self, obs, action, model, data, success):
        _ = action
        distance = float(np.linalg.norm(obs["object_to_target_pos"][:2]))
        if success:
            reward = self.success_reward
            return self.scale_reward(reward), {
                "reward_goal_distance": distance,
                "reward_stable_steps": self._stable_count,
            }
        if not self.reward_shaping:
            return 0.0, {"reward_goal_distance": distance}

        ee_distance = float(np.linalg.norm(obs["gripper_to_object_pos"]))
        reaching = 0.35 * (1.0 - np.tanh(10.0 * ee_distance))
        contact = 0.15 if self._is_robot_touching_object(model, data, "object") else 0.0
        progress = np.clip(
            (self.initial_goal_distance - distance) / self.initial_goal_distance,
            -1.0,
            1.0,
        )
        proximity = 1.1 * (1.0 - np.tanh(5.0 * distance))
        object_z = float(obs["object_pos"][2])
        lifted = max(
            object_z - (self.table_top_z + self.objects[0].bottom_offset + 0.02),
            0.0,
        )
        lift_penalty = min(5.0 * lifted, 0.25)
        reward = reaching + contact + 0.4 * float(progress) + proximity - lift_penalty
        return self.scale_reward(float(reward)), {
            "reward_reaching": float(reaching),
            "reward_contact": float(contact),
            "reward_progress": float(progress),
            "reward_proximity": float(proximity),
            "reward_lift_penalty": float(lift_penalty),
            "reward_goal_distance": distance,
            "reward_stable_steps": self._stable_count,
        }
