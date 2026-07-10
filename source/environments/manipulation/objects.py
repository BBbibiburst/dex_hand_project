# -*- coding: utf-8 -*-
"""Movable object specifications for manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FreeBoxSpec:
    """Specification for one free block object."""

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
