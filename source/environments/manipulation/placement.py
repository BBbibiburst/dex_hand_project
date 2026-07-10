# -*- coding: utf-8 -*-
"""Placement samplers for manipulation task resets."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np

from source.environments.manipulation.objects import ManipulationObjectSpec


class UniformTablePlacementSampler:
    def __init__(self, *, x_range: Tuple[float, float], y_range: Tuple[float, float],
                 rotation: Optional[Tuple[float, float]] = None,
                 ensure_object_boundary_in_range: bool = False,
                 ensure_valid_placement: bool = True, min_separation: float = 0.08,
                 max_attempts: int = 100) -> None:
        self.x_range = x_range; self.y_range = y_range; self.rotation = rotation
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.ensure_valid_placement = ensure_valid_placement
        self.min_separation = min_separation; self.max_attempts = max_attempts

    def sample(self, objects: Sequence[ManipulationObjectSpec], *, rng: np.random.Generator,
               reference_pos: np.ndarray, z_offset: float = 0.002):
        placements = {}; xy_points: list[np.ndarray] = []
        reference_pos = np.asarray(reference_pos, dtype=np.float64)
        for obj in objects:
            x_low, x_high = self._effective_range(self.x_range, obj.horizontal_radius)
            y_low, y_high = self._effective_range(self.y_range, obj.horizontal_radius)
            for _ in range(self.max_attempts):
                xy = np.asarray([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)])
                if not self.ensure_valid_placement or all(np.linalg.norm(xy-p) >= self.min_separation for p in xy_points):
                    break
            yaw_range = (-np.pi, np.pi) if self.rotation is None else self.rotation
            yaw = rng.uniform(*yaw_range); half = 0.5 * yaw
            quat = np.asarray([np.cos(half), 0., 0., np.sin(half)], dtype=np.float64)
            pos = np.asarray([reference_pos[0]+xy[0], reference_pos[1]+xy[1],
                              reference_pos[2]+obj.bottom_offset+z_offset], dtype=np.float64)
            xy_points.append(xy); placements[obj.name] = (pos, quat)
        return placements

    def _effective_range(self, value_range, radius):
        if not self.ensure_object_boundary_in_range: return value_range
        low, high = value_range[0]+radius, value_range[1]-radius
        if low > high:
            mid = 0.5*(low+high); return mid, mid
        return low, high
