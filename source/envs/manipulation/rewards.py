# -*- coding: utf-8 -*-
"""Reusable reward-shaping helpers for manipulation tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def staged_multi_object_reward(
    *,
    object_names: Sequence[str],
    placed: Sequence[bool],
    gripper_distance: Callable[[str], float],
    is_grasped: Callable[[str], bool],
    object_position: Callable[[str], np.ndarray],
    target_position: Callable[[str], np.ndarray],
    reach_weight: float = 0.1,
    grasp_reward: float = 0.35,
    hover_weight: float = 0.7,
    distance_gain: float = 10.0,
) -> float:
    """Return staged reach / grasp / hover reward for unplaced objects.

    The already placed object count forms the integer part of the reward. Only
    the best active stage is added, matching the original PickPlace and
    NutAssembly behavior.
    """
    active_names = [name for name, is_placed in zip(object_names, placed) if not is_placed]
    placed_count = float(np.count_nonzero(placed))
    if not active_names:
        return placed_count

    reach_distance = min(gripper_distance(name) for name in active_names)
    reach = reach_weight * (1.0 - np.tanh(distance_gain * reach_distance))

    grasp = grasp_reward if any(is_grasped(name) for name in active_names) else 0.0

    hover = max(
        hover_weight
        * (
            1.0
            - np.tanh(
                distance_gain
                * np.linalg.norm(object_position(name)[:2] - target_position(name)[:2])
            )
        )
        for name in active_names
    )
    return placed_count + max(float(reach), float(grasp), float(hover))
