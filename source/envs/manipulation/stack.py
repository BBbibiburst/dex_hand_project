# -*- coding: utf-8 -*-
"""Stack manipulation task."""
from __future__ import annotations

from typing import Any
from gymnasium import spaces
import mujoco
import numpy as np

from source.envs.core.registry import register_task
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.objects import FreeBoxSpec
from source.envs.manipulation.placement import UniformTablePlacementSampler


@register_task("stack")
class StackTask(SingleArmManipulationTask):
    success_reward=2.0

    def __init__(self, **kwargs: Any):
        sampler=kwargs.pop("placement_sampler",UniformTablePlacementSampler(x_range=(-.08,.08),y_range=(-.08,.08),min_separation=.10))
        super().__init__(placement_sampler=sampler,**kwargs)

    @property
    def name(self): return "stack"

    def create_objects(self):
        return (FreeBoxSpec("cubeA",(.02,.02,.02),(.86,.12,.10,1.0)),
                FreeBoxSpec("cubeB",(.025,.025,.025),(.18,.62,.20,1.0)))

    def extra_observation_space(self):
        return {"cubeA_to_cubeB_pos":spaces.Box(-np.inf,np.inf,shape=(3,),dtype=np.float32)}

    def get_extra_observation(self, model, data, obs):
        _=model,data
        return {"cubeA_to_cubeB_pos":(obs["cubeB_pos"]-obs["cubeA_pos"]).astype(np.float32)}

    def compute_task_reward(self, obs, action, model, data, success):
        _=obs,action,success
        reach,lift,stack=self.staged_rewards(model,data)
        reward=max(reach,lift,stack) if self.reward_shaping else (self.success_reward if stack>0 else 0.0)
        return self.scale_reward(float(reward)), {"reward_reach":reach,"reward_lift":lift,"reward_stack":stack}

    def staged_rewards(self, model, data):
        b=self._require_bindings()
        a=self._body_pos(model,data,"cubeA")
        c=self._body_pos(model,data,"cubeB")
        ee=np.zeros(3) if b.ee_site_id is None else data.site_xpos[b.ee_site_id]
        reach=.25*(1.0-np.tanh(10.0*float(np.linalg.norm(ee-a))))
        grasp=self._is_robot_touching_object(model,data,"cubeA")
        if grasp: reach += .25
        lifted=a[2] > self.table_top_z+.04
        lift=1.0 if lifted else 0.0
        if lifted: lift += .5*(1.0-np.tanh(float(np.linalg.norm(a[:2]-c[:2]))))
        stack=2.0 if (not grasp and lifted and self._objects_touching(data,"cubeA","cubeB")) else 0.0
        return float(reach),float(lift),float(stack)

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        return self.staged_rewards(model,data)[2] > 0.0
