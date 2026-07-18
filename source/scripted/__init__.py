"""Phase-based scripted policies for automated demonstration collection."""

from source.scripted.base import (
    ActionContext,
    PhaseContext,
    PhaseResult,
    TaskStrategy,
)
from source.scripted.registry import create_strategy, registered_strategies

__all__ = [
    "ActionContext",
    "PhaseContext",
    "PhaseResult",
    "TaskStrategy",
    "create_strategy",
    "registered_strategies",
]
