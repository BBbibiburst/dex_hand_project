# -*- coding: utf-8 -*-
"""Task interface and built-in task implementations for DexHand env."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from gymnasium import spaces
import numpy as np


Array = np.ndarray
Observation = Dict[str, Any]


class DexHandTask(ABC):
    """Abstract task interface for DexHandGymEnv.

    Implementations define what the robot should do (reach, grasp, etc.)
    by providing observation augmentation, reward, and termination logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name for logging/debugging."""

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, spaces.Space]:
        """Additional observation spaces beyond the base robot state."""

    @abstractmethod
    def reset(
        self,
        model: Any,
        data: Any,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        """Called at the start of each episode. Returns info dict."""

    @abstractmethod
    def get_observation(
        self,
        model: Any,
        data: Any,
    ) -> Observation:
        """Current task-specific observation. Must match ``observation_space``."""

    @abstractmethod
    def compute_reward(
        self,
        obs: Observation,
        action: Array,
        model: Any,
        data: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        """Returns (reward, reward_info)."""

    @abstractmethod
    def is_terminated(
        self,
        obs: Observation,
        model: Any,
        data: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Returns (terminated, terminated_info)."""


class NoopTask(DexHandTask):
    """Default no-op task: zero reward, never terminates, no extra observations."""

    @property
    def name(self) -> str:
        return "noop"

    @property
    def observation_space(self) -> Dict[str, spaces.Space]:
        return {}

    def reset(
        self,
        model: Any,
        data: Any,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = model, data, rng, options
        return {"task": self.name}

    def get_observation(self, model: Any, data: Any) -> Observation:
        _ = model, data
        return {}

    def compute_reward(
        self,
        obs: Observation,
        action: Array,
        model: Any,
        data: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        _ = obs, action, model, data
        return 0.0, {}

    def is_terminated(
        self,
        obs: Observation,
        model: Any,
        data: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        _ = obs, model, data
        return False, {}
