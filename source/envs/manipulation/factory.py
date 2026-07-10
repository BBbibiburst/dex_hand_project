# -*- coding: utf-8 -*-
"""Factories for manipulation-task environments."""

from __future__ import annotations

from typing import Any, Mapping

from source.envs.core.registry import make_task
from source.envs.rl_env import make_env


def make_manipulation_env(
    task_name: str,
    *,
    task_config: Mapping[str, Any] | None = None,
    **env_kwargs: Any,
):
    """Create an environment for a registered manipulation task."""
    task = make_task(task_name, **dict(task_config or {}))
    env_kwargs.setdefault("add_default_scene", False)
    return make_env(task=task, **env_kwargs)


def make_lift_env(
    *,
    task_config: Mapping[str, Any] | None = None,
    **env_kwargs: Any,
):
    """Create the built-in lift environment."""
    return make_manipulation_env("lift", task_config=task_config, **env_kwargs)


def make_stack_env(
    *,
    task_config: Mapping[str, Any] | None = None,
    **env_kwargs: Any,
):
    """Create the built-in stack environment."""
    return make_manipulation_env("stack", task_config=task_config, **env_kwargs)


def make_pick_place_env(*, task_config=None, **env_kwargs):
    return make_manipulation_env("pick_place", task_config=task_config, **env_kwargs)


def make_nut_assembly_env(*, task_config=None, **env_kwargs):
    return make_manipulation_env("nut_assembly", task_config=task_config, **env_kwargs)


def make_door_env(*, task_config=None, **env_kwargs):
    return make_manipulation_env("door", task_config=task_config, **env_kwargs)
