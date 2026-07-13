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
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import mujoco
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class TactileSiteRef:
    """One generated site exposed to device-independent visualization tools."""

    name: str
    patch: str
    flat_index: int


@dataclass(frozen=True)
class TactileSurfacePlotData:
    """Backend-provided geometry for the generic surface-sampling demo."""

    name: str
    rows: int
    cols: int
    kind: str
    samples: np.ndarray
    triangles: np.ndarray
    fit_surfaces: tuple[np.ndarray, ...] = ()
    title: str = ""


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

    def visualization_sites(self) -> Sequence[TactileSiteRef]:
        """Return generated sites that generic visualization tools may draw.

        Backends that do not represent sensing with MuJoCo sites can keep the
        default empty result and remain fully valid tactile implementations.
        """
        return ()

    def read_patches(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> Mapping[str, np.ndarray]:
        """Return named 2-D tactile arrays for diagnostics and visualization."""
        raise NotImplementedError(f"{type(self).__name__} does not expose tactile patches.")

    def surface_patch_names(self) -> Sequence[str]:
        """Return patches supported by the optional offline surface demo."""
        return ()

    def surface_plot_data(self, patch_name: str) -> TactileSurfacePlotData:
        """Return backend-specific surface geometry through a common data model."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide surface plot data for {patch_name!r}."
        )

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
