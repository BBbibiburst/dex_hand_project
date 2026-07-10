# -*- coding: utf-8 -*-
"""Reusable base for single-arm table manipulation tasks."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Optional, Tuple

from gymnasium import spaces
import mujoco
import numpy as np

from source.envs.manipulation.arenas import TableArena
from source.envs.manipulation.bindings import ObjectBinding, TaskBindings
from source.envs.manipulation.objects import ManipulationObjectSpec
from source.envs.manipulation.placement import UniformTablePlacementSampler
from source.envs.core.tasks import RobotTask, TaskStepResult


class SingleArmManipulationTask(RobotTask):
    """Template implementation shared by table-top single-arm tasks."""

    success_reward: float = 1.0

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
        self._objects = tuple(self.create_objects())
        self.bindings: TaskBindings | None = None

    @abstractmethod
    def create_objects(self) -> tuple[ManipulationObjectSpec, ...]:
        """Declare all free objects added by the task."""

    @property
    def objects(self) -> tuple[ManipulationObjectSpec, ...]:
        return self._objects

    @property
    def boxes(self) -> tuple[ManipulationObjectSpec, ...]:
        """Deprecated compatibility alias for older box-only tasks."""
        return self._objects

    @property
    def table_top_z(self) -> float:
        return self.arena.table_top_z

    @property
    def table_offset(self) -> np.ndarray:
        return self.arena.table_top_pos

    @property
    def observation_space(self) -> Dict[str, spaces.Space]:
        if not self.use_object_obs:
            return {}

        result: Dict[str, spaces.Space] = {}
        for obj in self.objects:
            result[f"{obj.name}_pos"] = spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )
            result[f"{obj.name}_quat"] = spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            )
            result[f"gripper_to_{obj.name}_pos"] = spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )
        result.update(self.extra_observation_space())
        return result

    def extra_observation_space(self) -> Dict[str, spaces.Space]:
        return {}

    def get_extra_observation(self, model, data, obs) -> dict[str, np.ndarray]:
        return {}

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        self.arena.augment_spec(spec)
        for obj in self.objects:
            initial = np.asarray(
                [
                    self.table_offset[0],
                    self.table_offset[1],
                    self.table_top_z + obj.bottom_offset + 0.002,
                ],
                dtype=np.float64,
            )
            obj.add_to_spec(spec, initial)

    def bind(self, model: mujoco.MjModel) -> None:
        object_bindings: dict[str, ObjectBinding] = {}
        object_geom_ids: set[int] = set()

        for obj in self.objects:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name
            )
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, obj.joint_name
            )
            geom_ids = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in obj.geom_names
            }
            if body_id < 0 or joint_id < 0 or min(geom_ids, default=-1) < 0:
                raise ValueError(f"Task object {obj.name!r} was not compiled correctly.")

            valid_geom_ids = {int(value) for value in geom_ids}
            object_geom_ids.update(valid_geom_ids)
            object_bindings[obj.name] = ObjectBinding(
                body_id=int(body_id),
                joint_id=int(joint_id),
                qpos_adr=int(model.jnt_qposadr[joint_id]),
                qvel_adr=int(model.jnt_dofadr[joint_id]),
                geom_ids=frozenset(valid_geom_ids),
            )

        ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name
        )
        environment_ids = set(object_geom_ids)
        for name in (self.arena.table_geom_name, "floor"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                environment_ids.add(int(geom_id))

        robot_ids = frozenset(
            geom_id for geom_id in range(model.ngeom) if geom_id not in environment_ids
        )
        self.bindings = TaskBindings(
            objects=object_bindings,
            ee_site_id=int(ee_site_id) if ee_site_id >= 0 else None,
            robot_geom_ids=robot_ids,
            environment_geom_ids=frozenset(environment_ids),
        )

    def _require_bindings(self) -> TaskBindings:
        if self.bindings is None:
            raise RuntimeError("Task.bind(model) must be called before use.")
        return self.bindings

    def reset(self, model, data, *, rng, options):
        _ = options
        if self.bindings is None:
            self.bind(model)

        bindings = self._require_bindings()
        placements = self.placement_sampler.sample(
            self.objects,
            rng=rng,
            reference_pos=self.table_offset,
        )
        for obj in self.objects:
            position, quaternion = placements[obj.name]
            binding = bindings.objects[obj.name]
            data.qpos[binding.qpos_adr : binding.qpos_adr + 3] = position
            data.qpos[binding.qpos_adr + 3 : binding.qpos_adr + 7] = quaternion
            data.qvel[binding.qvel_adr : binding.qvel_adr + 6] = 0.0

        return {
            "task": self.name,
            "task_objects": tuple(obj.name for obj in self.objects),
        }

    def get_observation(self, model, data):
        _ = model
        if not self.use_object_obs:
            return {}

        bindings = self._require_bindings()
        ee_position = (
            np.zeros(3, dtype=np.float32)
            if bindings.ee_site_id is None
            else data.site_xpos[bindings.ee_site_id].astype(np.float32)
        )
        observation: dict[str, np.ndarray] = {}
        for obj in self.objects:
            binding = bindings.objects[obj.name]
            position = data.xpos[binding.body_id].astype(np.float32).copy()
            observation[f"{obj.name}_pos"] = position
            observation[f"{obj.name}_quat"] = (
                data.xquat[binding.body_id].astype(np.float32).copy()
            )
            observation[f"gripper_to_{obj.name}_pos"] = (
                position - ee_position
            ).astype(np.float32)

        observation.update(self.get_extra_observation(model, data, observation))
        return observation

    def evaluate(self, obs, action, model, data) -> TaskStepResult:
        success = bool(self.check_success(model, data))
        reward, task_info = self.compute_task_reward(
            obs, action, model, data, success
        )
        return TaskStepResult(
            reward=float(reward),
            success=success,
            terminated=bool(self.terminate_on_success and success),
            info={"task_success": success, **task_info},
        )

    @abstractmethod
    def compute_task_reward(self, obs, action, model, data, success: bool):
        """Return ``(scaled_reward, reward_info)`` for the current state."""

    @abstractmethod
    def check_success(self, model, data) -> bool:
        """Return whether the task goal is currently satisfied."""

    def scale_reward(self, reward: float) -> float:
        if self.reward_scale is None:
            return reward
        return reward * self.reward_scale / self.success_reward

    def _body_pos(self, model, data, name: str) -> np.ndarray:
        _ = model
        binding = self._require_bindings().objects[name]
        return data.xpos[binding.body_id]

    def _is_robot_touching_object(self, model, data, object_name: str) -> bool:
        _ = model
        bindings = self._require_bindings()
        object_geoms = bindings.objects[object_name].geom_ids
        for index in range(data.ncon):
            pair = {
                int(data.contact[index].geom1),
                int(data.contact[index].geom2),
            }
            if pair.intersection(object_geoms) and pair.intersection(
                bindings.robot_geom_ids
            ):
                return True
        return False

    def _objects_touching(self, data, first_name: str, second_name: str) -> bool:
        bindings = self._require_bindings()
        first = bindings.objects[first_name].geom_ids
        second = bindings.objects[second_name].geom_ids
        for index in range(data.ncon):
            pair = {
                int(data.contact[index].geom1),
                int(data.contact[index].geom2),
            }
            if pair.intersection(first) and pair.intersection(second):
                return True
        return False
