# -*- coding: utf-8 -*-
"""Environment task contracts and registry."""

from source.envs.core.registry import make_task, register_task, registered_tasks
from source.envs.core.tasks import NoopTask, RobotTask, TaskStepResult

__all__ = [
    "NoopTask",
    "RobotTask",
    "TaskStepResult",
    "make_task",
    "register_task",
    "registered_tasks",
]
