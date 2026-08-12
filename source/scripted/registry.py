"""Registry for built-in scripted demonstration policies."""

from __future__ import annotations

from typing import Any

from source.scripted.base import TaskStrategy
from source.scripted.lift import LiftStrategy

_STRATEGIES: dict[str, tuple[str, type[TaskStrategy]]] = {
    "lift": ("lift", LiftStrategy),
}


def registered_strategies() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def strategy_task_name(strategy: str) -> str:
    """Return the explicitly associated manipulation task name."""
    try:
        return _STRATEGIES[strategy][0]
    except KeyError as exc:
        raise ValueError(f"Unknown scripted strategy {strategy!r}.") from exc


def create_strategy(task: str, **kwargs: Any) -> TaskStrategy:
    try:
        task_name, strategy_class = _STRATEGIES[task]
    except KeyError as exc:
        raise ValueError(
            f"No scripted strategy for task {task!r}; available={registered_strategies()}."
        ) from exc
    from source.envs.manipulation import registered_tasks

    if task_name not in registered_tasks():
        raise RuntimeError(f"Scripted strategy {task!r} references missing task {task_name!r}.")
    return strategy_class(**kwargs)
