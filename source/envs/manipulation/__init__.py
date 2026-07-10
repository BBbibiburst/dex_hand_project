# -*- coding: utf-8 -*-
"""Manipulation tasks and environment factories.

Built-in tasks are imported from an explicit module list. Third-party tasks can
still register themselves with :func:`register_task`.
"""

from source.envs.core.registry import make_task, register_task, registered_tasks
from source.envs.manipulation.arenas import BinsArena, PegsArena, TableArena
from source.envs.manipulation.base import SingleArmManipulationTask
from source.envs.manipulation.factory import (
    make_door_env,
    make_lift_env,
    make_manipulation_env,
    make_nut_assembly_env,
    make_pick_place_env,
    make_stack_env,
)
from source.envs.manipulation.objects import (
    FreeBoxSpec,
    FreeCylinderSpec,
    FreeNutSpec,
    ManipulationObjectSpec,
    XmlNutSpec,
)
from source.envs.manipulation.placement import UniformTablePlacementSampler

__all__ = [
    "FreeBoxSpec",
    "ManipulationObjectSpec",
    "SingleArmManipulationTask",
    "TableArena",
    "BinsArena",
    "PegsArena",
    "FreeCylinderSpec",
    "FreeNutSpec",
    "XmlNutSpec",
    "UniformTablePlacementSampler",
    "make_task",
    "register_task",
    "registered_tasks",
    "make_manipulation_env",
    "make_lift_env",
    "make_stack_env",
    "make_pick_place_env",
    "make_nut_assembly_env",
    "make_door_env",
]
