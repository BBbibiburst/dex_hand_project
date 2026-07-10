"""Extensible manipulation task package."""
from source.environments.core.registry import make_task, register_task, registered_tasks
from source.environments.manipulation.arenas import TableArena
from source.environments.manipulation.base import SingleArmManipulationTask
from source.environments.manipulation.factory import make_lift_env, make_manipulation_env, make_stack_env
from source.environments.manipulation.lift import LiftTask
from source.environments.manipulation.objects import FreeBoxSpec, ManipulationObjectSpec
from source.environments.manipulation.placement import UniformTablePlacementSampler
from source.environments.manipulation.stack import StackTask

__all__=["FreeBoxSpec","ManipulationObjectSpec","LiftTask","StackTask","SingleArmManipulationTask",
         "TableArena","UniformTablePlacementSampler","make_task","register_task","registered_tasks",
         "make_manipulation_env","make_lift_env","make_stack_env"]
