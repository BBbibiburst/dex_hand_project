# -*- coding: utf-8 -*-
"""Viewer overlay utilities (markers, labels, stats) — decoupled from Env."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import mujoco
import numpy as np
from mujoco import viewer

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


def draw_line_marker(
    handle: viewer.Handle,
    p1: Array,
    p2: Array,
    *,
    width: float = 0.003,
    rgba: Optional[Sequence[float]] = None,
) -> None:
    """Draw a line segment marker between ``p1`` and ``p2``."""
    scene = handle.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    color = [0.0, 0.3, 1.0, 1.0] if rgba is None else rgba
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.asarray([width, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )
    scene.ngeom += 1


def draw_ellipse_marker(
    handle: viewer.Handle,
    center: Array,
    *,
    radius_x: float,
    radius_y: float,
    radius_z: float = 0.0,
    segments: int = 96,
    rgba: Optional[Sequence[float]] = None,
) -> None:
    """Draw a closed ellipse/circle as a chain of line segments."""
    center = np.asarray(center, dtype=np.float64)
    segments = max(8, int(segments))
    points = []
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        points.append(
            center
            + np.asarray(
                [
                    radius_x * np.cos(theta),
                    radius_y * np.sin(theta),
                    radius_z * np.sin(2.0 * theta),
                ],
                dtype=np.float64,
            )
        )
    for i in range(segments):
        draw_line_marker(handle, points[i], points[(i + 1) % segments], rgba=rgba)


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
