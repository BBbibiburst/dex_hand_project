# -*- coding: utf-8 -*-
"""触觉传感器抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np


class TactileSensorBase(ABC):
    """触觉传感器抽象基类。

    具体实现可以基于灵巧手 MJCF 中的 STL mesh geom、contact、ray cast、
    自定义 site/taxel 或外部模型计算触觉信号。环境只关心 observation space
    和 ``read`` 的返回值是否匹配。
    """

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """触觉观测对应的 Gymnasium space。"""

    @abstractmethod
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """环境编译模型后调用，用于缓存 geom/site/body id 或 STL 派生数据。"""

    @abstractmethod
    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        options: Optional[dict],
    ) -> Dict[str, Any]:
        """每个 episode reset 时调用，返回要并入 info 的诊断信息。"""

    @abstractmethod
    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        """读取当前触觉观测。返回值必须落在 ``observation_space`` 内。"""


class NullTactileSensor(TactileSensorBase):
    """空触觉传感器，用于先跑通环境生命周期。"""

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
