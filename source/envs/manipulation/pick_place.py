"""Four-object pick-and-place task, implemented without robosuite imports."""
from __future__ import annotations

from typing import Any
import mujoco
import numpy as np

from source.envs.core.registry import register_task
from source.envs.manipulation.arenas import BinsArena
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.objects import FreeBoxSpec, FreeCylinderSpec
from source.envs.manipulation.placement import UniformTablePlacementSampler


@register_task("pick_place")
class PickPlaceTask(SingleArmManipulationTask):
    success_reward = 4.0
    object_names = ("milk", "bread", "cereal", "can")

    def __init__(self, *, single_object: str | None = None, **kwargs: Any) -> None:
        if single_object is not None and single_object not in self.object_names:
            raise ValueError(f"single_object must be one of {self.object_names!r}")
        self.single_object = single_object
        arena = kwargs.pop("arena", BinsArena())
        sampler = kwargs.pop("placement_sampler", UniformTablePlacementSampler(
            x_range=(-0.13, 0.13), y_range=(-0.13, 0.13), min_separation=0.075
        ))
        super().__init__(arena=arena, placement_sampler=sampler, **kwargs)
        self.objects_in_bins = np.zeros(len(self.objects), dtype=bool)

    @property
    def name(self) -> str:
        return "pick_place"

    def create_objects(self):
        specs = {
            "milk": FreeBoxSpec("milk", (0.022, 0.035, 0.055), (0.95, 0.95, 0.95, 1)),
            "bread": FreeBoxSpec("bread", (0.025, 0.035, 0.045), (0.85, 0.55, 0.20, 1)),
            "cereal": FreeBoxSpec("cereal", (0.022, 0.038, 0.060), (0.85, 0.25, 0.15, 1)),
            "can": FreeCylinderSpec("can", 0.027, 0.045, (0.25, 0.55, 0.85, 1)),
        }
        names = self.object_names if self.single_object is None else (self.single_object,)
        return tuple(specs[name] for name in names)

    def reset(self, model, data, *, rng, options):
        info = super().reset(model, data, rng=rng, options=options)
        # Base sampler is relative to the source-bin centre.
        self.objects_in_bins[:] = False
        return info

    @property
    def table_offset(self):
        arena = self.arena
        if isinstance(arena, BinsArena):
            return np.asarray((arena.source_center[0], arena.source_center[1], arena.table_top_z))
        return super().table_offset

    def _target_center(self, index: int) -> np.ndarray:
        arena: BinsArena = self.arena  # type: ignore[assignment]
        dx = arena.bin_half_size[0] * 0.5
        dy = arena.bin_half_size[1] * 0.5
        offsets = ((-dx, -dy), (-dx, dy), (dx, -dy), (dx, dy))
        global_index = self.object_names.index(self.objects[index].name)
        ox, oy = offsets[global_index]
        return np.asarray((arena.target_center[0] + ox, arena.target_center[1] + oy, arena.table_top_z))

    def _in_target(self, pos: np.ndarray, index: int) -> bool:
        arena: BinsArena = self.arena  # type: ignore[assignment]
        center = self._target_center(index)
        return bool(abs(pos[0] - center[0]) < arena.bin_half_size[0] * 0.48 and
                    abs(pos[1] - center[1]) < arena.bin_half_size[1] * 0.48 and
                    arena.table_top_z < pos[2] < arena.table_top_z + 0.18)

    def check_success(self, model, data) -> bool:
        bindings = self._require_bindings()
        ee = None if bindings.ee_site_id is None else data.site_xpos[bindings.ee_site_id]
        for i, obj in enumerate(self.objects):
            pos = self._body_pos(model, data, obj.name)
            released = ee is None or np.linalg.norm(ee - pos) > 0.05
            self.objects_in_bins[i] = self._in_target(pos, i) and released
        return bool(np.all(self.objects_in_bins))

    def compute_task_reward(self, obs, action, model, data, success):
        _ = action
        if success:
            reward = self.success_reward
        elif not self.reward_shaping:
            reward = float(np.sum(self.objects_in_bins))
        else:
            active = [o for i, o in enumerate(self.objects) if not self.objects_in_bins[i]]
            if not active:
                reward = self.success_reward
            else:
                d = min(float(np.linalg.norm(obs[f"gripper_to_{o.name}_pos"])) for o in active)
                reach = 0.1 * (1 - np.tanh(10 * d))
                grasp = 0.35 if any(self._is_robot_touching_object(model, data, o.name) for o in active) else 0.0
                hover = 0.0
                for o in active:
                    i = self.objects.index(o)
                    pos = self._body_pos(model, data, o.name)
                    hover = max(hover, 0.7 * (1 - np.tanh(10 * np.linalg.norm(pos[:2] - self._target_center(i)[:2]))))
                reward = float(np.sum(self.objects_in_bins)) + max(reach, grasp, hover)
        return self.scale_reward(float(reward)), {"objects_in_bins": self.objects_in_bins.copy()}

