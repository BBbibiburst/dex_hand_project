# -*- coding: utf-8 -*-
"""Shared helpers for robot controllers."""

from __future__ import annotations

from typing import Sequence

from gymnasium import spaces
import mujoco
import numpy as np

CONTROL_MODES = ("position", "ik")
IK_ACTION_LAYOUT = ("x", "y", "z", "qw", "qx", "qy", "qz")


def prefixed_names(names: Sequence[str], prefix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in names)


def _validate_mode(control_mode: str) -> str:
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"control_mode must be one of {CONTROL_MODES}, got {control_mode!r}.")
    return control_mode


def _actuator_ids_or_raise(
    model: mujoco.MjModel,
    actuator_names: Sequence[str],
    *,
    owner: str,
) -> np.ndarray:
    actuator_ids = []
    missing = []
    for name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            missing.append(name)
        else:
            actuator_ids.append(actuator_id)

    if missing:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
            for idx in range(model.nu)
        ]
        raise ValueError(f"{owner} missing actuator(s): {missing}. Available actuators: {available}")

    return np.asarray(actuator_ids, dtype=np.int32)


def _empty_box() -> spaces.Box:
    return spaces.Box(
        low=np.zeros(0, dtype=np.float32),
        high=np.zeros(0, dtype=np.float32),
        dtype=np.float32,
    )

