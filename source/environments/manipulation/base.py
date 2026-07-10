# -*- coding: utf-8 -*-
"""Base classes for robosuite-style single-arm manipulation tasks."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from gymnasium import spaces
import mujoco
import numpy as np

from source.environments.manipulation.arenas import TableArena
from source.environments.manipulation.objects import FreeBoxSpec
from source.environments.manipulation.placement import UniformTablePlacementSampler
from source.environments.tasks import Observation, RobotTask


class SingleArmManipulationTask(RobotTask):
    """Base class for robosuite-style single-arm table manipulation tasks."""

    def __init__(
        self,
        *,
        arena: Optional[TableArena] = None,
        table_full_size: Tuple[float, float, float] = (0.8, 0.8, 0.05),
        table_offset: Tuple[float, float, float] = (0.55, 0.0, 0.8),
        table_friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001),
        table_has_legs: bool = True,
        ee_site_name: str = "right_hand_site",
        use_object_obs: bool = True,
        reward_scale: Optional[float] = 1.0,
        reward_shaping: bool = False,
        terminate_on_success: bool = False,
        placement_sampler: Optional[UniformTablePlacementSampler] = None,
    ) -> None:
        self.arena = arena or TableArena(
            table_full_size=table_full_size,
            table_offset=table_offset,
            table_friction=table_friction,
            table_has_legs=table_has_legs,
        )
        self.ee_site_name = ee_site_name
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.terminate_on_success = terminate_on_success
        self.placement_sampler = placement_sampler or UniformTablePlacementSampler(
            x_range=(-0.08, 0.08),
            y_range=(-0.08, 0.08),
        )
        self._body_ids: dict[str, int] = {}
        self._joint_qpos_addrs: dict[str, int] = {}
        self._joint_qvel_addrs: dict[str, int] = {}
        self._geom_ids: dict[str, set[int]] = {}
        self._ee_site_id: Optional[int] = None

    @property
    def table_top_z(self) -> float:
        return self.arena.table_top_z

    @property
    def table_offset(self) -> np.ndarray:
        """World position of the table top center."""
        return self.arena.table_top_pos

    @property
    @abstractmethod
    def boxes(self) -> Tuple[FreeBoxSpec, ...]:
        """Free block objects owned by this task."""

    @property
    def observation_space(self) -> Dict[str, spaces.Space]:
        if not self.use_object_obs:
            return {}
        obs_spaces: Dict[str, spaces.Space] = {}
        for box in self.boxes:
            obs_spaces[f"{box.name}_pos"] = spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )
            obs_spaces[f"{box.name}_quat"] = spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            )
            obs_spaces[f"gripper_to_{box.name}_pos"] = spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )
        return obs_spaces

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        self.arena.augment_spec(spec)
        for box in self.boxes:
            self._add_free_box(spec, box)

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = options
        self._bind(model)
        placements = self.placement_sampler.sample(
            self.boxes,
            rng=rng,
            reference_pos=self.table_offset,
        )

        for box in self.boxes:
            pos, quat = placements[box.name]
            qpos_addr = self._joint_qpos_addrs[box.name]
            qvel_addr = self._joint_qvel_addrs[box.name]
            data.qpos[qpos_addr:qpos_addr + 3] = pos
            data.qpos[qpos_addr + 3:qpos_addr + 7] = quat
            data.qvel[qvel_addr:qvel_addr + 6] = 0.0

        return {
            "task": self.name,
            "task_objects": tuple(box.name for box in self.boxes),
        }

    def get_observation(self, model: mujoco.MjModel, data: mujoco.MjData) -> Observation:
        self._bind(model)
        ee_pos = (
            data.site_xpos[self._ee_site_id]
            if self._ee_site_id is not None
            else np.zeros(3, dtype=np.float64)
        )
        obs: Observation = {}
        if not self.use_object_obs:
            return obs
        for box in self.boxes:
            body_id = self._body_ids[box.name]
            box_pos = data.xpos[body_id].astype(np.float32).copy()
            obs[f"{box.name}_pos"] = box_pos
            obs[f"{box.name}_quat"] = data.xquat[body_id].astype(np.float32).copy()
            obs[f"gripper_to_{box.name}_pos"] = (
                box_pos - ee_pos.astype(np.float32)
            ).astype(np.float32)
        return obs

    def is_terminated(
        self,
        obs: Observation,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Tuple[bool, Dict[str, Any]]:
        success = self.check_success(model, data)
        return bool(self.terminate_on_success and success), {"success": bool(success)}

    @abstractmethod
    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Return whether the task is solved."""

    def _add_free_box(self, spec: mujoco.MjSpec, box: FreeBoxSpec) -> None:
        body = spec.worldbody.add_body()
        body.name = box.body_name
        body.pos = [
            float(self.table_offset[0]),
            float(self.table_offset[1]),
            self.table_top_z + box.half_size[2] + 0.002,
        ]

        joint = body.add_joint()
        joint.name = box.joint_name
        joint.type = mujoco.mjtJoint.mjJNT_FREE

        geom = body.add_geom()
        geom.name = box.geom_name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = list(box.half_size)
        geom.density = box.density
        geom.friction = list(box.friction)
        geom.rgba = list(box.rgba)
        geom.condim = 3
        geom.contype = 1
        geom.conaffinity = 1

        if box.duplicate_collision_geoms:
            visual = body.add_geom()
            visual.name = f"{box.name}_visual"
            visual.type = mujoco.mjtGeom.mjGEOM_BOX
            visual.size = list(box.half_size)
            visual.contype = 0
            visual.conaffinity = 0
            visual.rgba = list(box.rgba)

    def _bind(self, model: mujoco.MjModel) -> None:
        if len(self._body_ids) == len(self.boxes):
            return

        self._body_ids.clear()
        self._joint_qpos_addrs.clear()
        self._joint_qvel_addrs.clear()
        self._geom_ids.clear()

        for box in self.boxes:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, box.body_name
            )
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, box.joint_name
            )
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, box.geom_name
            )
            if min(body_id, joint_id, geom_id) < 0:
                raise ValueError(f"Task object {box.name!r} was not compiled.")
            self._body_ids[box.name] = int(body_id)
            self._joint_qpos_addrs[box.name] = int(model.jnt_qposadr[joint_id])
            self._joint_qvel_addrs[box.name] = int(model.jnt_dofadr[joint_id])
            self._geom_ids[box.name] = {int(geom_id)}

        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name
        )
        self._ee_site_id = int(site_id) if site_id >= 0 else None

    def _body_pos(self, model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
        self._bind(model)
        return data.xpos[self._body_ids[name]]

    def _is_robot_touching_object(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_name: str,
    ) -> bool:
        object_geoms = self._geom_ids[object_name]
        ignored = set().union(*(ids for ids in self._geom_ids.values()))
        table_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self.arena.table_geom_name
        )
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ignored.update(idx for idx in (table_id, floor_id) if idx >= 0)

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair.intersection(object_geoms):
                continue
            other = next(iter(pair - object_geoms), -1)
            if other not in ignored:
                return True
        return False

    def _objects_touching(
        self,
        data: mujoco.MjData,
        first_name: str,
        second_name: str,
    ) -> bool:
        first_geoms = self._geom_ids[first_name]
        second_geoms = self._geom_ids[second_name]
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair.intersection(first_geoms) and pair.intersection(second_geoms):
                return True
        return False
