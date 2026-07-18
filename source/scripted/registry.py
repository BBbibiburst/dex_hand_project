"""Registry for built-in scripted demonstration policies."""

from __future__ import annotations

from typing import Any

from source.scripted.base import TaskStrategy
from source.scripted.lift import LiftStrategy

_STRATEGIES: dict[str, type[TaskStrategy]] = {"lift": LiftStrategy}


def registered_strategies() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def create_strategy(task: str, **kwargs: Any) -> TaskStrategy:
    try:
        strategy_class = _STRATEGIES[task]
    except KeyError as exc:
        raise ValueError(
            f"No scripted strategy for task {task!r}; available={registered_strategies()}."
        ) from exc
    return strategy_class(**kwargs)
