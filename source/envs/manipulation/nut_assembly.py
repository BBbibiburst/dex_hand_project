"""Square and round nut assembly task without robosuite runtime dependencies."""
from __future__ import annotations

from typing import Any
import numpy as np

from source.envs.core.registry import register_task
from source.envs.manipulation.arenas import PegsArena
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.objects import XmlNutSpec
from source.envs.manipulation.placement import UniformTablePlacementSampler


@register_task("nut_assembly")
class NutAssemblyTask(SingleArmManipulationTask):
    success_reward = 2.0
    nut_names = ("square_nut", "round_nut")

    def __init__(self, *, single_nut: str | None = None, **kwargs: Any) -> None:
        if single_nut is not None and single_nut not in self.nut_names:
            raise ValueError(f"single_nut must be one of {self.nut_names!r}")
        self.single_nut = single_nut
        arena = kwargs.pop("arena", PegsArena())
        sampler = kwargs.pop("placement_sampler", UniformTablePlacementSampler(
            x_range=(-0.13, 0.13),
            y_range=(-0.16, 0.16),
            ensure_object_boundary_in_range=False,
            min_separation=0.10,
        ))
        super().__init__(arena=arena, placement_sampler=sampler, **kwargs)
        self.objects_on_pegs = np.zeros(len(self.objects), dtype=bool)

    @property
    def name(self):
        return "nut_assembly"

    def create_objects(self):
        specs = {
            "square_nut": XmlNutSpec("square_nut", "square-nut.xml"),
            "round_nut": XmlNutSpec("round_nut", "round-nut.xml"),
        }
        names = self.nut_names if self.single_nut is None else (self.single_nut,)
        return tuple(specs[n] for n in names)

    def reset(self, model, data, *, rng, options):
        info = super().reset(model, data, rng=rng, options=options)
        self.objects_on_pegs[:] = False
        return info

    def _peg_center(self, obj_name: str):
        arena: PegsArena = self.arena  # type: ignore[assignment]
        idx = self.nut_names.index(obj_name)
        return np.asarray((*arena.peg_centers[idx], arena.table_top_z))

    def check_success(self, model, data):
        bindings = self._require_bindings()
        ee = None if bindings.ee_site_id is None else data.site_xpos[bindings.ee_site_id]
        for i, obj in enumerate(self.objects):
            pos = self._body_pos(model, data, obj.name)
            peg = self._peg_center(obj.name)
            released = ee is None or np.linalg.norm(ee - pos) > 0.05
            self.objects_on_pegs[i] = np.linalg.norm(pos[:2] - peg[:2]) < 0.018 and pos[2] < self.table_top_z + 0.10 and released
        return bool(np.all(self.objects_on_pegs))

    def compute_task_reward(self, obs, action, model, data, success):
        _ = action
        if success:
            reward = self.success_reward
        elif not self.reward_shaping:
            reward = float(np.sum(self.objects_on_pegs))
        else:
            active = [o for i, o in enumerate(self.objects) if not self.objects_on_pegs[i]]
            d = min((float(np.linalg.norm(obs[f"gripper_to_{o.name}_pos"])) for o in active), default=0.0)
            reach = 0.1 * (1 - np.tanh(10 * d))
            grasp = 0.35 if any(self._is_robot_touching_object(model, data, o.name) for o in active) else 0.0
            hover = max((0.7 * (1 - np.tanh(10 * np.linalg.norm(self._body_pos(model, data, o.name)[:2] - self._peg_center(o.name)[:2]))) for o in active), default=0.0)
            reward = float(np.sum(self.objects_on_pegs)) + max(reach, grasp, hover)
        return self.scale_reward(float(reward)), {"objects_on_pegs": self.objects_on_pegs.copy()}
