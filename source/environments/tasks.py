# -*- coding: utf-8 -*-
"""Stable task contract used by :mod:`source.environments.rl_env`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gymnasium import spaces
import numpy as np

Array = np.ndarray
Observation = Dict[str, Any]


@dataclass(frozen=True)
class TaskStepResult:
    """Complete result of evaluating one task step."""

    reward: float
    success: bool = False
    terminated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class RobotTask(ABC):
    """Task plug-in interface.

    Environments own simulation and controllers; tasks own scene additions,
    reset randomisation, observations, reward and task termination.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, spaces.Space]:
        ...

    def augment_spec(self, spec: Any) -> None:
        _ = spec

    def bind(self, model: Any) -> None:
        """Cache compiled model identifiers. Called once after compilation."""
        _ = model

    @abstractmethod
    def reset(
        self,
        model: Any,
        data: Any,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_observation(self, model: Any, data: Any) -> Observation:
        ...

    @abstractmethod
    def evaluate(
        self,
        obs: Observation,
        action: Array,
        model: Any,
        data: Any,
    ) -> TaskStepResult:
        ...


class NoopTask(RobotTask):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def observation_space(self) -> Dict[str, spaces.Space]:
        return {}

    def reset(self, model, data, *, rng, options):
        _ = model, data, rng, options
        return {"task": self.name}

    def get_observation(self, model, data) -> Observation:
        _ = model, data
        return {}

    def evaluate(self, obs, action, model, data) -> TaskStepResult:
        _ = obs, action, model, data
        return TaskStepResult(reward=0.0)
