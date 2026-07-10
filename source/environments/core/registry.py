# -*- coding: utf-8 -*-
"""Generic task registration and construction."""
from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

from source.environments.tasks import RobotTask

TaskType = TypeVar("TaskType", bound=type[RobotTask])
_TASKS: Dict[str, type[RobotTask]] = {}


def register_task(name: str) -> Callable[[TaskType], TaskType]:
    key = name.strip().lower()
    if not key:
        raise ValueError("Task name cannot be empty.")

    def decorator(cls: TaskType) -> TaskType:
        previous = _TASKS.get(key)
        if previous is not None and previous is not cls:
            raise KeyError(f"Task {key!r} is already registered by {previous.__name__}.")
        _TASKS[key] = cls
        return cls

    return decorator


def make_task(name: str, **kwargs: Any) -> RobotTask:
    key = name.strip().lower()
    try:
        task_cls = _TASKS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_TASKS)) or "<none>"
        raise ValueError(f"Unknown task {name!r}. Available tasks: {available}.") from exc
    return task_cls(**kwargs)


def registered_tasks() -> tuple[str, ...]:
    return tuple(sorted(_TASKS))
