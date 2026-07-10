# -*- coding: utf-8 -*-
"""Composable task-object specifications."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import mujoco
import numpy as np


class ManipulationObjectSpec(ABC):
    name: str

    @property
    @abstractmethod
    def body_name(self) -> str: ...

    @property
    @abstractmethod
    def joint_name(self) -> str: ...

    @property
    @abstractmethod
    def geom_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def horizontal_radius(self) -> float: ...

    @property
    @abstractmethod
    def bottom_offset(self) -> float: ...

    @abstractmethod
    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None: ...


@dataclass(frozen=True)
class FreeBoxSpec(ManipulationObjectSpec):
    """Free box; retained as the compatible name used by existing tasks."""

    name: str
    half_size: Tuple[float, float, float]
    rgba: Tuple[float, float, float, float]
    density: float = 500.0
    friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)
    duplicate_collision_geoms: bool = True

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_name(self) -> str:
        return f"{self.name}_geom"

    @property
    def geom_names(self) -> tuple[str, ...]:
        return (self.geom_name,)

    @property
    def horizontal_radius(self) -> float:
        return float(np.linalg.norm(self.half_size[:2]))

    @property
    def bottom_offset(self) -> float:
        return float(self.half_size[2])

    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None:
        body = spec.worldbody.add_body()
        body.name = self.body_name
        body.pos = np.asarray(initial_pos, dtype=float).tolist()
        joint = body.add_joint()
        joint.name = self.joint_name
        joint.type = mujoco.mjtJoint.mjJNT_FREE
        geom = body.add_geom()
        geom.name = self.geom_name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = list(self.half_size)
        geom.density = self.density
        geom.friction = list(self.friction)
        geom.rgba = list(self.rgba)
        geom.condim = 3
        geom.contype = 1
        geom.conaffinity = 1
        if self.duplicate_collision_geoms:
            visual = body.add_geom()
            visual.name = f"{self.name}_visual"
            visual.type = mujoco.mjtGeom.mjGEOM_BOX
            visual.size = list(self.half_size)
            visual.contype = 0
            visual.conaffinity = 0
            visual.rgba = list(self.rgba)
