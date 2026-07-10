# -*- coding: utf-8 -*-
"""Factory helpers for manipulation environments."""

from __future__ import annotations

from typing import Any

from source.environments.manipulation.lift import LiftTask
from source.environments.manipulation.stack import StackTask


def make_lift_env(**kwargs: Any):
    """Create a RobotGymEnv configured with the migrated Lift task."""
    from source.environments.rl_env import make_env

    task_kwargs = _pop_task_kwargs(kwargs)
    task = LiftTask(**task_kwargs)
    kwargs.setdefault("add_default_scene", False)
    return make_env(task=task, **kwargs)


def make_stack_env(**kwargs: Any):
    """Create a RobotGymEnv configured with the migrated Stack task."""
    from source.environments.rl_env import make_env

    task_kwargs = _pop_task_kwargs(kwargs)
    task = StackTask(**task_kwargs)
    kwargs.setdefault("add_default_scene", False)
    return make_env(task=task, **kwargs)


def _pop_task_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    task_keys = {
        "arena",
        "table_full_size",
        "table_offset",
        "table_friction",
        "table_has_legs",
        "ee_site_name",
        "use_object_obs",
        "reward_scale",
        "reward_shaping",
        "terminate_on_success",
        "placement_sampler",
    }
    return {key: kwargs.pop(key) for key in list(kwargs) if key in task_keys}
