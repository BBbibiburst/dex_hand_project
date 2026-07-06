# -*- coding: utf-8 -*-
"""
Tactile sensor abstract interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np

from source.environments.tactile_layout import (
    DEX_HAND_TACTILE_COUNT,
    DEX_HAND_TACTILE_PATCHES,
    tactile_sensor_names,
)


class TactileSensorBase(ABC):
    """Abstract base class for tactile sensors.

    Concrete implementations may compute tactile signals based on STL mesh geoms
    in the dexterous hand MJCF, contacts, ray casting, custom sites/taxels, or
    external models. The environment only cares that the observation space and
    the return value of ``read`` are consistent.
    """

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """The Gymnasium space corresponding to tactile observations."""

    @abstractmethod
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called after the environment compiles the model; used to cache
        geom/site/body IDs or STL-derived data."""

    @abstractmethod
    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        """Called on every episode reset; returns diagnostic info to be merged
        into the environment's info dict."""

    @abstractmethod
    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        """Read the current tactile observation. The returned value must lie
        within ``observation_space``."""


class NullTactileSensor(TactileSensorBase):
    """No-op tactile sensor for getting the environment lifecycle working first."""

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


class DexHandTouchSensor(TactileSensorBase):
    """Read generated MuJoCo touch sensors on the dex-hand skin meshes."""

    def __init__(self, *, hand_prefix: str = "") -> None:
        self.hand_prefix = hand_prefix
        self.sensor_names = tactile_sensor_names(hand_prefix)
        self._sensor_ids: Optional[np.ndarray] = None
        self._sensor_adrs: Optional[np.ndarray] = None
        self._sensor_dims: Optional[np.ndarray] = None

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(DEX_HAND_TACTILE_COUNT,),
            dtype=np.float32,
        )

    @property
    def patch_shapes(self) -> Dict[str, tuple[int, int]]:
        return {
            patch.mesh_name: (patch.rows, patch.cols)
            for patch in DEX_HAND_TACTILE_PATCHES
        }

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        _ = data
        sensor_ids = []
        missing = []
        for name in self.sensor_names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0:
                missing.append(name)
            else:
                sensor_ids.append(sensor_id)
        if missing:
            sample = missing[:8]
            raise ValueError(
                f"Missing tactile touch sensor(s): {sample}"
                f"{' ...' if len(missing) > len(sample) else ''}. "
                "Make sure build_combined_spec(add_tactile_sensors=True) is used."
            )

        self._sensor_ids = np.asarray(sensor_ids, dtype=np.int32)
        self._sensor_adrs = model.sensor_adr[self._sensor_ids].astype(np.int32)
        self._sensor_dims = model.sensor_dim[self._sensor_ids].astype(np.int32)
        if not np.all(self._sensor_dims == 1):
            raise ValueError("Dex hand tactile touch sensors must be scalar sensors.")

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
        return {
            "tactile_sensor": "dex_hand_touch",
            "tactile_size": DEX_HAND_TACTILE_COUNT,
            "tactile_patches": self.patch_shapes,
        }

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        _ = model
        if self._sensor_adrs is None:
            raise RuntimeError("DexHandTouchSensor.bind() must be called first.")
        return data.sensordata[self._sensor_adrs].astype(np.float32).copy()
