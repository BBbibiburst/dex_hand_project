# -*- coding: utf-8 -*-
"""Tactile sensor interface.

This module defines the *only* contract the framework imposes on tactile
sensing: a lifecycle of four steps.

    1. ``augment_spec`` — called on the end effector's ``MjSpec`` before it is
       attached to the arm. Implementations that need extra sites/sensors
       (e.g. a taxel grid) add them here directly via the MjSpec API. The
       default does nothing, so end effectors with no tactile sensing don't
       need to override it at all.
    2. ``bind`` — called once after the combined model is compiled; cache
       whatever ids/addresses you need.
    3. ``reset`` — called at the start of every episode.
    4. ``read`` — called every step; must return a value inside
       ``observation_space``.

How a concrete implementation gets its data (STL surface fitting, raw
contact-force summation, a learned model, or nothing) is entirely up to that
implementation. The framework does not know and does not need to know.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np


class TactileSensorBase(ABC):
    """Abstract base class for tactile sensors."""

    def augment_spec(self, hand_spec: mujoco.MjSpec) -> None:
        """Optionally modify the end effector's ``MjSpec`` before it is
        attached to the arm (e.g. add sites/sensors). Default: no-op."""
        _ = hand_spec

    def set_name_prefix(self, prefix: str) -> None:
        """Receive the prefix applied when the end effector is attached.

        MuJoCo's ``attach_body`` prefixes names under the attached subtree.
        Implementations that look up generated sites/sensors by name can
        store the prefix here. Default: no-op.
        """
        _ = prefix

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """The Gymnasium space corresponding to tactile observations."""

    @abstractmethod
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called after the environment compiles the model; used to cache
        ids/addresses needed by ``read``."""

    @abstractmethod
    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        """Called on every episode reset; returns diagnostic info to be
        merged into the environment's info dict."""

    @abstractmethod
    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        """Read the current tactile observation. The returned value must lie
        within ``observation_space``."""


class NullTactileSensor(TactileSensorBase):
    """No-op tactile sensor for end effectors with no tactile sensing."""

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        _ = model
        _ = data

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        _ = model
        _ = data
        _ = rng
        _ = options
        return {}

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        _ = model
        _ = data
        return np.zeros(0, dtype=np.float32)
