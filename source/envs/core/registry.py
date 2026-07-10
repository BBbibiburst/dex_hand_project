# -*- coding: utf-8 -*-
"""Task registration and construction."""

from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

from source.envs.core.tasks import RobotTask
from source.registry import Registry

TaskClass = type[RobotTask]
TaskType = TypeVar("TaskType", bound=TaskClass)
_TASKS = Registry[TaskClass]("task", normalize=str.lower)
_BUILTIN_TASK_MODULES = (
    "source.envs.manipulation.lift",
    "source.envs.manipulation.stack",
)
_BUILTINS_LOADED = False


def load_builtin_tasks() -> None:
    """Import the explicitly supported built-in task modules once."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    for module_name in _BUILTIN_TASK_MODULES:
        importlib.import_module(module_name)
    _BUILTINS_LOADED = True


def register_task(name: str) -> Callable[[TaskType], TaskType]:
    """Register a task class under ``name``."""

    def decorator(cls: TaskType) -> TaskType:
        _TASKS.register(name, cls)
        return cls

    return decorator


def make_task(name: str, **kwargs: Any) -> RobotTask:
    load_builtin_tasks()
    return _TASKS.get(name)(**kwargs)


def registered_tasks() -> tuple[str, ...]:
    load_builtin_tasks()
    return _TASKS.names()
