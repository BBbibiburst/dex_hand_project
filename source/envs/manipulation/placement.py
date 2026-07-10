# -*- coding: utf-8 -*-
"""Placement samplers for manipulation-task objects."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from source.envs.manipulation.objects import ManipulationObjectSpec

Placement = tuple[np.ndarray, np.ndarray]


class UniformTablePlacementSampler:
    """Sample non-overlapping object poses from rectangular table ranges.

    A complete layout is retried when a later object cannot be placed. This is
    important for multi-object tasks: keeping an unlucky first sample fixed can
    make every subsequent placement impossible even though valid joint layouts
    exist.
    """

    def __init__(
        self,
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        rotation: tuple[float, float] | None = None,
        ensure_object_boundary_in_range: bool = True,
        ensure_valid_placement: bool = True,
        min_separation: float = 0.06,
        max_attempts: int = 500,
        candidates_per_object: int = 32,
    ) -> None:
        if x_range[0] > x_range[1]:
            raise ValueError(f"Invalid x_range: {x_range!r}.")
        if y_range[0] > y_range[1]:
            raise ValueError(f"Invalid y_range: {y_range!r}.")
        if min_separation < 0.0:
            raise ValueError("min_separation must be non-negative.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        if candidates_per_object <= 0:
            raise ValueError("candidates_per_object must be positive.")

        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.ensure_valid_placement = ensure_valid_placement
        self.min_separation = min_separation
        self.max_attempts = max_attempts
        self.candidates_per_object = candidates_per_object

    def sample(
        self,
        objects: Sequence[ManipulationObjectSpec],
        *,
        rng: np.random.Generator,
        reference_pos: np.ndarray,
        z_offset: float = 0.002,
    ) -> dict[str, Placement]:
        """Return a valid placement for every object.

        ``max_attempts`` counts complete-layout retries. Within each layout,
        every object gets ``candidates_per_object`` candidate samples before
        the whole layout is discarded and restarted.
        """

        reference_pos = np.asarray(reference_pos, dtype=np.float64)
        if reference_pos.shape != (3,):
            raise ValueError(f"reference_pos must have shape (3,), got {reference_pos.shape}.")

        if not objects:
            return {}

        effective_ranges = {
            obj.name: (
                self._effective_range(self.x_range, obj.horizontal_radius),
                self._effective_range(self.y_range, obj.horizontal_radius),
            )
            for obj in objects
        }

        for _layout_attempt in range(self.max_attempts):
            xy_points: list[np.ndarray] = []
            placements: dict[str, Placement] = {}

            for obj in objects:
                x_limits, y_limits = effective_ranges[obj.name]
                xy = self._sample_valid_xy(
                    rng=rng,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    previous_points=xy_points,
                )
                if xy is None:
                    # An earlier object was placed unfavourably. Discard the
                    # entire layout instead of retrying this object forever.
                    break

                quaternion = self._sample_yaw_quaternion(rng)
                position = np.asarray(
                    [
                        reference_pos[0] + xy[0],
                        reference_pos[1] + xy[1],
                        reference_pos[2] + obj.bottom_offset + z_offset,
                    ],
                    dtype=np.float64,
                )
                xy_points.append(xy)
                placements[obj.name] = (position, quaternion)
            else:
                return placements

        names = ", ".join(repr(obj.name) for obj in objects)
        raise RuntimeError(
            "Unable to sample a valid joint placement for objects "
            f"[{names}] after {self.max_attempts} complete-layout attempts. "
            "Consider widening x_range/y_range or reducing min_separation."
        )

    def _sample_valid_xy(
        self,
        *,
        rng: np.random.Generator,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        previous_points: Sequence[np.ndarray],
    ) -> np.ndarray | None:
        for _ in range(self.candidates_per_object):
            xy = np.asarray(
                [rng.uniform(*x_limits), rng.uniform(*y_limits)],
                dtype=np.float64,
            )
            if not self.ensure_valid_placement or all(
                np.linalg.norm(xy - previous) >= self.min_separation for previous in previous_points
            ):
                return xy
        return None

    def _sample_yaw_quaternion(self, rng: np.random.Generator) -> np.ndarray:
        yaw_range = (-np.pi, np.pi) if self.rotation is None else self.rotation
        yaw = rng.uniform(*yaw_range)
        half_yaw = 0.5 * yaw
        return np.asarray(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )

    def _effective_range(
        self,
        value_range: tuple[float, float],
        radius: float,
    ) -> tuple[float, float]:
        if not self.ensure_object_boundary_in_range:
            return value_range

        low = value_range[0] + radius
        high = value_range[1] - radius
        if low > high:
            raise ValueError(
                f"Object radius {radius:.6f} does not fit inside range {value_range!r}."
            )
        return low, high
