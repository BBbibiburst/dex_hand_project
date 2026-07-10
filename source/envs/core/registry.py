# -*- coding: utf-8 -*-
"""Task registration, discovery, and construction."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, TypeVar

from source.envs.core.tasks import RobotTask
from source.registry import Registry

TaskClass = type[RobotTask]
TaskType = TypeVar("TaskType", bound=TaskClass)
_TASKS = Registry[TaskClass]("task", normalize=str.lower)
_BUILTINS_LOADED = False


def load_builtin_tasks() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    package_name = "source.envs.manipulation"
    package = importlib.import_module(package_name)
    ignored = {"base", "bindings", "factory", "objects", "placement", "arenas"}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_") or module_info.name in ignored:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")


def register_task(name: str) -> Callable[[TaskType], TaskType]:
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
