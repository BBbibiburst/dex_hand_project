# -*- coding: utf-8 -*-
"""Placement samplers for manipulation task resets."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from source.environments.manipulation.objects import FreeBoxSpec


class UniformTablePlacementSampler:
    """robosuite-style uniform table sampler."""

    def __init__(
        self,
        *,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        rotation: Optional[Tuple[float, float]] = None,
        ensure_object_boundary_in_range: bool = False,
        ensure_valid_placement: bool = True,
        min_separation: float = 0.08,
        max_attempts: int = 100,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.ensure_valid_placement = ensure_valid_placement
        self.min_separation = min_separation
        self.max_attempts = max_attempts

    def sample(
        self,
        boxes: Sequence[FreeBoxSpec],
        *,
        rng: np.random.Generator,
        reference_pos: np.ndarray,
        z_offset: float = 0.002,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        placements: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        xy_points: list[np.ndarray] = []
        reference_pos = np.asarray(reference_pos, dtype=np.float64)

        for box in boxes:
            radius = float(np.linalg.norm(box.half_size[:2]))
            x_low, x_high = self._effective_range(self.x_range, radius)
            y_low, y_high = self._effective_range(self.y_range, radius)

            for _ in range(self.max_attempts):
                xy = np.asarray([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)])
                if (
                    not self.ensure_valid_placement
                    or all(
                        np.linalg.norm(xy - other_xy) >= self.min_separation
                        for other_xy in xy_points
                    )
                ):
                    break
            else:
                xy = np.asarray([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)])

            yaw_range = (-np.pi, np.pi) if self.rotation is None else self.rotation
            yaw = rng.uniform(*yaw_range)
            half_yaw = 0.5 * yaw
            quat = np.asarray(
                [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
                dtype=np.float64,
            )
            pos = np.asarray(
                [
                    reference_pos[0] + xy[0],
                    reference_pos[1] + xy[1],
                    reference_pos[2] + box.half_size[2] + z_offset,
                ],
                dtype=np.float64,
            )
            xy_points.append(xy)
            placements[box.name] = (pos, quat)

        return placements

    def _effective_range(
        self,
        value_range: Tuple[float, float],
        radius: float,
    ) -> Tuple[float, float]:
        if not self.ensure_object_boundary_in_range:
            return value_range
        low, high = value_range
        low += radius
        high -= radius
        if low > high:
            midpoint = 0.5 * (low + high)
            return midpoint, midpoint
        return low, high
