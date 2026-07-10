# -*- coding: utf-8 -*-
"""Compatibility imports for manipulation tasks.

New code should import from ``source.environments.manipulation``. This module
is kept so older demos and scripts using ``source.environments.manipulation_tasks``
continue to work.
"""

from source.environments.manipulation import (
    FreeBoxSpec,
    LiftTask,
    SingleArmManipulationTask,
    StackTask,
    TableArena,
    UniformTablePlacementSampler,
    make_lift_env,
    make_stack_env,
)

__all__ = [
    "FreeBoxSpec",
    "LiftTask",
    "SingleArmManipulationTask",
    "StackTask",
    "TableArena",
    "UniformTablePlacementSampler",
    "make_lift_env",
    "make_stack_env",
]
