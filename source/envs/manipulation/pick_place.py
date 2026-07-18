"""Single-object transport from one bin to another."""

from __future__ import annotations

from typing import Any

import numpy as np

from source.envs.core.registry import register_task
from source.envs.manipulation.arenas import BinsArena
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.object_catalog import (
    DEFAULT_PICK_PLACE_OBJECT,
    pick_place_object_ids,
)
from source.envs.manipulation.objects import MeshObjectSpec
from source.envs.manipulation.placement import UniformTablePlacementSampler
from source.envs.manipulation.rewards import staged_multi_object_reward


@register_task("pick_place")
class PickPlaceTask(SingleArmManipulationTask):
    success_reward = 1.0

    def __init__(self, *, object_id: str = DEFAULT_PICK_PLACE_OBJECT, **kwargs: Any) -> None:
        if object_id not in pick_place_object_ids():
            raise ValueError(f"Object {object_id!r} is not available for pick_place.")
        self.object_id = object_id
        arena = kwargs.pop("arena", BinsArena())
        sampler = kwargs.pop(
            "placement_sampler",
            UniformTablePlacementSampler(
                x_range=(-0.13, 0.13), y_range=(-0.13, 0.13), min_separation=0.075
            ),
        )
        super().__init__(arena=arena, placement_sampler=sampler, **kwargs)
        self.object_in_target = False

    @property
    def name(self) -> str:
        return "pick_place"

    def create_objects(self):
        return (MeshObjectSpec("object", self.object_id),)

    def reset(self, model, data, *, rng, options):
        info = super().reset(model, data, rng=rng, options=options)
        # Base sampler is relative to the source-bin centre.
        self.object_in_target = False
        return info

    @property
    def table_offset(self):
        arena = self.arena
        if isinstance(arena, BinsArena):
            return np.asarray((arena.source_center[0], arena.source_center[1], arena.table_top_z))
        return super().table_offset

    def _target_center(self, _name: str = "object") -> np.ndarray:
        arena: BinsArena = self.arena  # type: ignore[assignment]
        return np.asarray((*arena.target_center, arena.table_top_z))

    def _in_target(self, pos: np.ndarray) -> bool:
        arena: BinsArena = self.arena  # type: ignore[assignment]
        center = self._target_center()
        return bool(
            abs(pos[0] - center[0]) < arena.bin_half_size[0] * 0.82
            and abs(pos[1] - center[1]) < arena.bin_half_size[1] * 0.82
            and arena.table_top_z < pos[2] < arena.table_top_z + 0.18
        )

    def check_success(self, model, data) -> bool:
        bindings = self._require_bindings()
        ee = None if bindings.ee_site_id is None else data.site_xpos[bindings.ee_site_id]
        pos = self._body_pos(model, data, "object")
        released = ee is None or np.linalg.norm(ee - pos) > 0.05
        self.object_in_target = self._in_target(pos) and released
        return self.object_in_target

    def compute_task_reward(self, obs, action, model, data, success):
        _ = action
        if success:
            reward = self.success_reward
        elif not self.reward_shaping:
            reward = float(self.object_in_target)
        else:
            reward = staged_multi_object_reward(
                object_names=("object",),
                placed=(self.object_in_target,),
                gripper_distance=lambda name: float(np.linalg.norm(obs[f"gripper_to_{name}_pos"])),
                is_grasped=lambda name: self._is_robot_touching_object(model, data, name),
                object_position=lambda name: self._body_pos(model, data, name),
                target_position=self._target_center,
            )
        return self.scale_reward(float(reward)), {"object_in_target": self.object_in_target}
