# -*- coding: utf-8 -*-
"""Compatibility exports for the manipulation package."""
from source.environments.manipulation import (
    FreeBoxSpec,
    LiftTask,
    ManipulationObjectSpec,
    SingleArmManipulationTask,
    StackTask,
    TableArena,
    UniformTablePlacementSampler,
    make_lift_env,
    make_manipulation_env,
    make_stack_env,
    make_task,
    register_task,
    registered_tasks,
)

__all__ = [name for name in globals() if not name.startswith("_")]
