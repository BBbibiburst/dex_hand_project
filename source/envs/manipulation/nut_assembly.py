"""Square and round nut assembly task without robosuite runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from source.envs.core.registry import register_task
from source.envs.manipulation.arenas import FixedNutArena, PegsArena
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.objects import FreeBoltSpec, XmlNutSpec
from source.envs.manipulation.placement import (
    NUT_ASSEMBLY_REACHABLE_REGION,
    UniformTablePlacementSampler,
)
from source.envs.manipulation.rewards import staged_multi_object_reward


@register_task("nut_assembly")
class NutAssemblyTask(SingleArmManipulationTask):
    success_reward = 2.0
    nut_names = ("square_nut", "round_nut")

    def __init__(
        self,
        *,
        single_nut: str | None = None,
        variant_index: int | None = None,
        fixed_bolt_position: tuple[float, float, float] | None = None,
        fixed_bolt_quaternion_wxyz: tuple[float, float, float, float] | None = None,
        stable_steps: int = 20,
        **kwargs: Any,
    ) -> None:
        if variant_index is not None and not 0 <= variant_index < 100:
            raise ValueError("variant_index must lie in [0, 99].")
        if variant_index is not None and single_nut is not None:
            raise ValueError("variant_index and single_nut are mutually exclusive.")
        if single_nut is not None and single_nut not in self.nut_names:
            raise ValueError(f"single_nut must be one of {self.nut_names!r}")
        self.single_nut = single_nut
        self.variant = None if variant_index is None else procedural_nut_variant(variant_index)
        if stable_steps < 1:
            raise ValueError("stable_steps must be positive.")
        self.stable_steps = int(stable_steps)
        self._stable_count = 0
        self.fixed_bolt_position = fixed_bolt_position
        self.fixed_bolt_quaternion_wxyz = fixed_bolt_quaternion_wxyz
        if self.variant is None:
            arena = kwargs.pop("arena", PegsArena())
        else:
            arena = kwargs.pop(
                "arena",
                FixedNutArena(
                    polygon_sides=self.variant.polygon_sides,
                    inner_apothem=self.variant.hole_apothem,
                ),
            )
        sampler = kwargs.pop(
            "placement_sampler",
            UniformTablePlacementSampler.for_object_centers(
                NUT_ASSEMBLY_REACHABLE_REGION,
                rotation=(-np.deg2rad(20.0), np.deg2rad(20.0)),
                min_separation=0.11,
            ),
        )
        super().__init__(arena=arena, placement_sampler=sampler, **kwargs)
        self.objects_on_pegs = np.zeros(len(self.objects), dtype=bool)

    @property
    def name(self):
        return "nut_assembly"

    def create_objects(self):
        if self.variant is not None:
            return (
                FreeBoltSpec(
                    "assembly_bolt",
                    polygon_sides=self.variant.polygon_sides,
                    shaft_radius=self.variant.peg_radius,
                ),
            )
        specs = {
            "square_nut": XmlNutSpec("square_nut", "square-nut.xml"),
            "round_nut": XmlNutSpec("round_nut", "round-nut.xml"),
        }
        names = self.nut_names if self.single_nut is None else (self.single_nut,)
        return tuple(specs[n] for n in names)

    def reset(self, model, data, *, rng, options):
        info = super().reset(model, data, rng=rng, options=options)
        if self.variant is not None and self.fixed_bolt_position is not None:
            binding = self._require_bindings().objects["assembly_bolt"]
            data.qpos[binding.qpos_adr : binding.qpos_adr + 3] = self.fixed_bolt_position
            if self.fixed_bolt_quaternion_wxyz is not None:
                data.qpos[binding.qpos_adr + 3 : binding.qpos_adr + 7] = (
                    self.fixed_bolt_quaternion_wxyz
                )
            data.qvel[binding.qvel_adr : binding.qvel_adr + 6] = 0.0
            mujoco.mj_forward(model, data)
        self.objects_on_pegs[:] = False
        self._stable_count = 0
        return info

    def _peg_center(self, obj_name: str):
        if self.variant is not None:
            arena: FixedNutArena = self.arena  # type: ignore[assignment]
            return np.asarray((*arena.nut_center, arena.table_top_z))
        arena: PegsArena = self.arena  # type: ignore[assignment]
        idx = self.nut_names.index(obj_name)
        return np.asarray((*arena.peg_centers[idx], arena.table_top_z))

    def check_success(self, model, data):
        if self.variant is not None:
            binding = self._require_bindings().objects["assembly_bolt"]
            position = self._body_pos(model, data, "assembly_bolt")
            target = self._peg_center("assembly_bolt")
            rotation = data.xmat[binding.body_id].reshape(3, 3)
            vertical = float(rotation[2, 2]) >= float(np.cos(np.deg2rad(8.0)))
            horizontal = float(np.linalg.norm(position[:2] - target[:2]))
            shaft_tip = float(position[2] - self.objects[0].bottom_offset)
            arena: FixedNutArena = self.arena  # type: ignore[assignment]
            socket_top = self.table_top_z + 2.0 * arena.wall_half_height
            shoulder_bottom = float(
                position[2]
                + self.objects[0].shoulder_center_z
                - self.objects[0].shoulder_half_height
            )
            insertion_depth = socket_top - shaft_tip
            inserted = bool(
                insertion_depth >= 0.060
                # Keep a real gap above the table while allowing the shoulder
                # to settle naturally a little farther down the countersink.
                and shaft_tip >= self.table_top_z + 0.002
                # The shoulder remains within the physical countersink. Its
                # resting height depends on polygon phase/contact, so use the
                # actual guide top instead of an artificial 4 mm band around
                # the lower socket lip. The independent 60 mm shaft-depth
                # requirement still proves genuine insertion.
                and socket_top - 0.010 <= shoulder_bottom <= socket_top + arena.guide_height
            )
            velocity = data.qvel[binding.qvel_adr : binding.qvel_adr + 6]
            slow = np.linalg.norm(velocity[:3]) <= 0.03 and np.linalg.norm(velocity[3:]) <= 0.4
            candidate = bool(
                # Allow 0.5 mm for contact-solver and polygon-mesh tolerance
                # around the declared radial clearance. The shaft must still
                # physically pass through the straight lower socket.
                horizontal <= max(0.0025, self.variant.clearance + 0.0005)
                and vertical
                and inserted
                and slow
            )
            self._stable_count = self._stable_count + 1 if candidate else 0
            self.objects_on_pegs[0] = self._stable_count >= self.stable_steps
            return bool(self.objects_on_pegs[0])
        bindings = self._require_bindings()
        ee = None if bindings.ee_site_id is None else data.site_xpos[bindings.ee_site_id]
        for i, obj in enumerate(self.objects):
            pos = self._body_pos(model, data, obj.name)
            peg = self._peg_center(obj.name)
            released = ee is None or np.linalg.norm(ee - pos) > 0.05
            tolerance = 0.012 if self.variant is not None else 0.018
            maximum_z = self.table_top_z + (0.035 if self.variant is not None else 0.10)
            self.objects_on_pegs[i] = (
                np.linalg.norm(pos[:2] - peg[:2]) < tolerance and pos[2] < maximum_z and released
            )
        return bool(np.all(self.objects_on_pegs))

    def compute_task_reward(self, obs, action, model, data, success):
        _ = action
        if success:
            reward = self.success_reward
        elif not self.reward_shaping:
            reward = float(np.sum(self.objects_on_pegs))
        else:
            reward = staged_multi_object_reward(
                object_names=tuple(obj.name for obj in self.objects),
                placed=self.objects_on_pegs,
                gripper_distance=lambda name: float(np.linalg.norm(obs[f"gripper_to_{name}_pos"])),
                is_grasped=lambda name: self._is_robot_touching_object(model, data, name),
                object_position=lambda name: self._body_pos(model, data, name),
                target_position=self._peg_center,
            )
        return self.scale_reward(float(reward)), {"objects_on_pegs": self.objects_on_pegs.copy()}


@dataclass(frozen=True)
class ProceduralNutVariant:
    index: int
    polygon_sides: int
    ring_radius: float
    tube_radius: float
    peg_radius: float
    hole_apothem: float
    clearance: float


def procedural_nut_variant(index: int) -> ProceduralNutVariant:
    """Map [0, 99] to reproducible round/square/regular-polygon pairs."""

    if not 0 <= index < 100:
        raise ValueError("index must lie in [0, 99].")
    shapes = (0, 4, 3, 5, 6, 7, 8, 10, 12, 16)
    sides = shapes[index % len(shapes)]
    family = index // len(shapes)
    tube = 0.008 + 0.0005 * (family % 4)
    hole_circumradius = 0.034 + 0.001 * (family % 5)
    # The shaft and socket are a real mating pair. Keep only 4--4.5 mm radial
    # clearance; the previous 14--15 mm gap let the handle enter the opening
    # while the nominal shaft contributed almost nothing to insertion.
    clearance = 0.004 + 0.0005 * (family // 5)
    # Five- and six-sided shafts have broad corners that bind much more
    # readily than round/high-sided profiles during passive insertion.  Give
    # those two families a still-visible 8--8.5 mm radial running clearance.
    if sides in {3, 5, 6, 7}:
        clearance += 0.004
    cosine = 1.0 if sides == 0 else float(np.cos(np.pi / sides))
    hole_apothem = hole_circumradius * cosine
    ring_radius = (hole_apothem + tube) / cosine
    peg_radius = hole_circumradius - clearance
    return ProceduralNutVariant(
        index=index,
        polygon_sides=sides,
        ring_radius=ring_radius,
        tube_radius=tube,
        peg_radius=peg_radius,
        hole_apothem=hole_apothem,
        clearance=clearance,
    )
