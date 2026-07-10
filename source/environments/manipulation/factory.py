# -*- coding: utf-8 -*-
"""Single generic factory plus compatibility wrappers."""
from __future__ import annotations
from typing import Any, Mapping

from source.environments.core.registry import make_task
# Import built-ins for registration side effects.
from source.environments.manipulation import lift as _lift  # noqa: F401
from source.environments.manipulation import stack as _stack  # noqa: F401


def make_manipulation_env(task_name: str, *, task_config: Mapping[str,Any] | None=None, **env_kwargs: Any):
    from source.environments.rl_env import make_env
    task=make_task(task_name, **dict(task_config or {}))
    env_kwargs.setdefault("add_default_scene",False)
    return make_env(task=task,**env_kwargs)


def make_lift_env(**kwargs: Any):
    return make_manipulation_env("lift", task_config=_extract_legacy_task_config(kwargs), **kwargs)


def make_stack_env(**kwargs: Any):
    return make_manipulation_env("stack", task_config=_extract_legacy_task_config(kwargs), **kwargs)


def _extract_legacy_task_config(kwargs: dict[str,Any]):
    keys={"arena","table_full_size","table_offset","table_friction","table_has_legs","ee_site_name",
          "use_object_obs","reward_scale","reward_shaping","terminate_on_success","placement_sampler"}
    return {k:kwargs.pop(k) for k in list(kwargs) if k in keys}
