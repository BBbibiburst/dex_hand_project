# -*- coding: utf-8 -*-
"""Viewer overlay utilities (markers, labels, stats) — decoupled from Env."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import mujoco
from mujoco import viewer
import numpy as np


Array = np.ndarray


def clear_markers(handle: viewer.Handle) -> None:
    """Remove all user-drawn geoms from the viewer scene."""
    handle.user_scn.ngeom = 0


def draw_sphere_marker(
    handle: viewer.Handle,
    pos: Array,
    *,
    radius: float = 0.018,
    rgba: Optional[Sequence[float]] = None,
) -> None:
    """Draw a sphere marker at ``pos``."""
    scene = handle.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    color = [0.0, 0.9, 1.0, 0.55] if rgba is None else rgba
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_label(
    handle: viewer.Handle,
    pos: Array,
    text: str,
    *,
    rgba: Optional[Sequence[float]] = None,
) -> None:
    """Draw a text label at ``pos``."""
    scene = handle.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    color = [1.0, 1.0, 1.0, 1.0] if rgba is None else rgba
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LABEL,
        np.zeros(3, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.asarray(color, dtype=np.float32),
    )
    geom.label = text
    scene.ngeom += 1


def format_stats(
    stats: Dict[str, float],
    *,
    control_label: str = "ctrl",
) -> str:
    """Format simulation statistics into a display string."""
    return (
        f"sim {stats.get('sim_step_hz', 0.0):5.0f} Hz | "
        f"RTF {stats.get('real_time_factor', 0.0):4.2f} | "
        f"{control_label} {stats.get('control_hz', 0.0):4.1f} Hz"
    )


def draw_stats_label(
    handle: viewer.Handle,
    stats: Dict[str, float],
    pos: Optional[Array] = None,
    *,
    control_label: str = "ctrl",
) -> None:
    """Draw simulation statistics as a viewer label."""
    if pos is None:
        pos = np.asarray([0.0, -0.25, 1.2], dtype=np.float32)
    draw_label(handle, pos, format_stats(stats, control_label=control_label))
