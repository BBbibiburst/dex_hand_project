# -*- coding: utf-8 -*-
"""Extensible manipulation tasks.

Task modules are discovered automatically by :mod:`source.envs.core.registry`.
Adding a task only requires placing a public module in this package and using
``@register_task`` on its task class.
"""

from source.envs.core.registry import make_task, register_task, registered_tasks
from source.envs.manipulation.arenas import TableArena
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.factory import (
    make_lift_env,
    make_manipulation_env,
    make_stack_env,
)
from source.envs.manipulation.objects import FreeBoxSpec, ManipulationObjectSpec
from source.envs.manipulation.placement import UniformTablePlacementSampler

__all__ = [
    "FreeBoxSpec",
    "ManipulationObjectSpec",
    "SingleArmManipulationTask",
    "TableArena",
    "UniformTablePlacementSampler",
    "make_task",
    "register_task",
    "registered_tasks",
    "make_manipulation_env",
    "make_lift_env",
    "make_stack_env",
]
