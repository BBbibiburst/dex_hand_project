"""Numerically checked MuJoCo stepping for grasp generation and validation."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import mujoco
import numpy as np


@contextmanager
def capture_mujoco_warnings() -> Iterator[list[str]]:
    warnings: list[str] = []
    previous_handler = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    try:
        yield warnings
    finally:
        mujoco.set_mju_user_warning(previous_handler)


def checked_mj_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    warnings: list[str],
    *,
    phase: str,
    step: int,
) -> None:
    mujoco.mj_step(model, data)
    finite = all(np.all(np.isfinite(values)) for values in (data.qpos, data.qvel, data.qacc))
    bounded = (
        np.max(np.abs(data.qvel), initial=0.0) < 1e6
        and np.max(np.abs(data.qacc), initial=0.0) < 1e12
    )
    if warnings or not finite or not bounded:
        detail = warnings[-1] if warnings else "non-finite or unbounded simulation state"
        raise FloatingPointError(
            f"MuJoCo numerical instability during {phase} at step {step}: {detail}"
        )
