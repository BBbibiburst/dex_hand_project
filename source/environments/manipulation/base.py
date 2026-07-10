# -*- coding: utf-8 -*-
"""Reusable base for single-arm table manipulation tasks."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from gymnasium import spaces
import mujoco
import numpy as np

from source.environments.manipulation.arenas import TableArena
from source.environments.manipulation.bindings import ObjectBinding, TaskBindings
from source.environments.manipulation.objects import ManipulationObjectSpec
from source.environments.manipulation.placement import UniformTablePlacementSampler
from source.environments.tasks import Observation, RobotTask, TaskStepResult


class SingleArmManipulationTask(RobotTask):
    success_reward: float = 1.0

    def __init__(self, *, arena: Optional[TableArena] = None,
                 table_full_size: Tuple[float,float,float]=(0.8,0.8,0.05),
                 table_offset: Tuple[float,float,float]=(0.55,0.0,0.8),
                 table_friction: Tuple[float,float,float]=(1.0,0.005,0.0001),
                 table_has_legs: bool=True, ee_site_name: str="right_hand_site",
                 use_object_obs: bool=True, reward_scale: Optional[float]=1.0,
                 reward_shaping: bool=False, terminate_on_success: bool=False,
                 placement_sampler: Optional[UniformTablePlacementSampler]=None) -> None:
        self.arena = arena or TableArena(table_full_size=table_full_size, table_offset=table_offset,
                                         table_friction=table_friction, table_has_legs=table_has_legs)
        self.ee_site_name=ee_site_name; self.use_object_obs=use_object_obs
        self.reward_scale=reward_scale; self.reward_shaping=reward_shaping
        self.terminate_on_success=terminate_on_success
        self.placement_sampler = placement_sampler or UniformTablePlacementSampler(x_range=(-.08,.08), y_range=(-.08,.08))
        self._objects = tuple(self.create_objects())
        self.bindings: TaskBindings | None = None

    @abstractmethod
    def create_objects(self) -> tuple[ManipulationObjectSpec, ...]: ...

    @property
    def objects(self): return self._objects
    @property
    def boxes(self): return self._objects  # compatibility
    @property
    def table_top_z(self): return self.arena.table_top_z
    @property
    def table_offset(self): return self.arena.table_top_pos

    @property
    def observation_space(self) -> Dict[str, spaces.Space]:
        if not self.use_object_obs: return {}
        result = {}
        for obj in self.objects:
            result[f"{obj.name}_pos"] = spaces.Box(-np.inf,np.inf,shape=(3,),dtype=np.float32)
            result[f"{obj.name}_quat"] = spaces.Box(-np.inf,np.inf,shape=(4,),dtype=np.float32)
            result[f"gripper_to_{obj.name}_pos"] = spaces.Box(-np.inf,np.inf,shape=(3,),dtype=np.float32)
        result.update(self.extra_observation_space())
        return result

    def extra_observation_space(self): return {}
    def get_extra_observation(self, model, data, obs): return {}

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        self.arena.augment_spec(spec)
        for obj in self.objects:
            initial = np.asarray([self.table_offset[0], self.table_offset[1], self.table_top_z + obj.bottom_offset + .002])
            obj.add_to_spec(spec, initial)

    def bind(self, model: mujoco.MjModel) -> None:
        object_bindings = {}
        object_geom_ids: set[int] = set()
        for obj in self.objects:
            body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.body_name)
            joint_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,obj.joint_name)
            geom_ids={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in obj.geom_names}
            if body_id < 0 or joint_id < 0 or min(geom_ids, default=-1) < 0:
                raise ValueError(f"Task object {obj.name!r} was not compiled correctly.")
            geom_ids={int(x) for x in geom_ids}; object_geom_ids.update(geom_ids)
            object_bindings[obj.name]=ObjectBinding(int(body_id),int(joint_id),int(model.jnt_qposadr[joint_id]),
                                                    int(model.jnt_dofadr[joint_id]),frozenset(geom_ids))
        ee=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,self.ee_site_name)
        environment_ids=set(object_geom_ids)
        for name in (self.arena.table_geom_name,"floor"):
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
            if gid>=0: environment_ids.add(int(gid))
        robot_ids=frozenset(i for i in range(model.ngeom) if i not in environment_ids)
        self.bindings=TaskBindings(object_bindings, int(ee) if ee>=0 else None, robot_ids, frozenset(environment_ids))

    def _require_bindings(self):
        if self.bindings is None: raise RuntimeError("Task.bind(model) must be called before use.")
        return self.bindings

    def reset(self, model, data, *, rng, options):
        _=options
        if self.bindings is None: self.bind(model)
        b=self._require_bindings(); placements=self.placement_sampler.sample(self.objects,rng=rng,reference_pos=self.table_offset)
        for obj in self.objects:
            pos,quat=placements[obj.name]; ob=b.objects[obj.name]
            data.qpos[ob.qpos_adr:ob.qpos_adr+3]=pos; data.qpos[ob.qpos_adr+3:ob.qpos_adr+7]=quat
            data.qvel[ob.qvel_adr:ob.qvel_adr+6]=0.0
        return {"task":self.name,"task_objects":tuple(o.name for o in self.objects)}

    def get_observation(self, model, data):
        _=model
        if not self.use_object_obs: return {}
        b=self._require_bindings(); ee=np.zeros(3) if b.ee_site_id is None else data.site_xpos[b.ee_site_id]
        obs={}
        for obj in self.objects:
            ob=b.objects[obj.name]; pos=data.xpos[ob.body_id].astype(np.float32).copy()
            obs[f"{obj.name}_pos"]=pos; obs[f"{obj.name}_quat"]=data.xquat[ob.body_id].astype(np.float32).copy()
            obs[f"gripper_to_{obj.name}_pos"]=(pos-ee.astype(np.float32)).astype(np.float32)
        obs.update(self.get_extra_observation(model,data,obs)); return obs

    def evaluate(self, obs, action, model, data):
        success=bool(self.check_success(model,data)); reward, info=self.compute_task_reward(obs,action,model,data,success)
        return TaskStepResult(float(reward), success, bool(self.terminate_on_success and success),
                              {"task_success":success, **info})

    @abstractmethod
    def compute_task_reward(self, obs, action, model, data, success: bool): ...
    @abstractmethod
    def check_success(self, model, data) -> bool: ...

    def scale_reward(self, reward: float) -> float:
        return reward if self.reward_scale is None else reward * self.reward_scale / self.success_reward

    def _body_pos(self, model, data, name):
        _=model; return data.xpos[self._require_bindings().objects[name].body_id]
    def _is_robot_touching_object(self, model, data, object_name):
        _=model; b=self._require_bindings(); object_geoms=b.objects[object_name].geom_ids
        for i in range(data.ncon):
            pair={int(data.contact[i].geom1),int(data.contact[i].geom2)}
            if pair.intersection(object_geoms) and pair.intersection(b.robot_geom_ids): return True
        return False
    def _objects_touching(self, data, first_name, second_name):
        b=self._require_bindings(); first=b.objects[first_name].geom_ids; second=b.objects[second_name].geom_ids
        return any(({int(data.contact[i].geom1),int(data.contact[i].geom2)}.intersection(first) and
                    {int(data.contact[i].geom1),int(data.contact[i].geom2)}.intersection(second)) for i in range(data.ncon))
