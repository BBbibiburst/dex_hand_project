"""Finger-segment outer-shell fitting and regular taxel sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from source.sensors.tactile.fitting.bezier import (
    evaluate_bezier_surface as _evaluate_bezier_surface,
    fit_bezier_surface as _fit_bezier_surface,
)
from source.sensors.tactile.fitting.ellipse import _fit_rotated_section_ellipse
from source.sensors.tactile.fitting.primitives import (
    _ellipse_arc_mid_angles, _occupied_angle_arc, _supersample_triangles,
    _triangle_normals, _vertices_as_triangles,
)
from source.sensors.tactile.fitting.stl import read_stl_triangles

FINGER_SEGMENT_FIT_SUBDIVISIONS = 4

@dataclass(frozen=True)
class FingerSegmentSurfaceFit:
    center: np.ndarray
    axis: np.ndarray
    section_x: np.ndarray
    section_y: np.ndarray
    axial_low: float
    axial_high: float
    surface_center: np.ndarray
    surface_radius_x: float
    surface_radius_y: float
    surface_angle: float
    arc_start: float
    arc_end: float


def finger_segment_grid_points(mesh_path: Path, rows: int, cols: int) -> np.ndarray:
    """Regular grid fitted to the real outer shell of a finger segment."""
    triangles = read_stl_triangles(mesh_path)
    return _finger_segment_regular_surface_grid_points(triangles, rows, cols)


def finger_segment_fit_surface(vertices: np.ndarray) -> np.ndarray:
    triangles = _vertices_as_triangles(vertices)
    return _finger_segment_regular_surface_grid_points(triangles, 32, 64).reshape(32, 64, 3)


def _fit_finger_segment_surfaces(vertices: np.ndarray) -> FingerSegmentSurfaceFit:
    """Fit the exposed contact surface of a segment skin as a partial
    elliptic cylinder."""
    triangles = _vertices_as_triangles(vertices)
    fit_points = _supersample_triangles(triangles, subdivisions=FINGER_SEGMENT_FIT_SUBDIVISIONS)
    fit_center = fit_points.mean(axis=0)
    centered = fit_points - fit_center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis, section_x, section_y = vh[0], vh[1], vh[2]

    face_centers = triangles.mean(axis=1)
    face_normals = _triangle_normals(triangles)
    face_centered = face_centers - fit_center
    face_section = np.column_stack([face_centered @ section_x, face_centered @ section_y])

    coarse_center = np.median(face_section, axis=0)
    face_rel = face_section - coarse_center
    coarse_radius_x = max(np.percentile(np.abs(face_rel[:, 0]), 95.0), 1e-9)
    coarse_radius_y = max(np.percentile(np.abs(face_rel[:, 1]), 95.0), 1e-9)
    r_norm = np.sqrt(
        (face_rel[:, 0] / coarse_radius_x) ** 2 + (face_rel[:, 1] / coarse_radius_y) ** 2
    )

    radial_len = np.linalg.norm(face_rel, axis=1) + 1e-12
    radial = face_rel / radial_len[:, None]
    normal_section = np.column_stack([face_normals @ section_x, face_normals @ section_y])
    normal_radial = np.sum(normal_section * radial, axis=1)

    if np.std(r_norm) < 0.02:
        outer_faces = normal_radial > 0.0
    else:
        pos_vote = (r_norm > np.median(r_norm)).astype(np.float64)
        normal_vote = (normal_radial > 0.0).astype(np.float64)
        outer_faces = (0.6 * pos_vote + 0.4 * normal_vote) > 0.5

    if outer_faces.sum() < 0.05 * len(outer_faces) or (~outer_faces).sum() < 0.05 * len(
        outer_faces
    ):
        outer_faces = r_norm >= np.percentile(r_norm, 60.0)

    surface_vertices = _supersample_triangles(
        triangles[outer_faces], subdivisions=FINGER_SEGMENT_FIT_SUBDIVISIONS
    )
    if len(surface_vertices) == 0:
        surface_vertices = fit_points

    (
        surface_center,
        surface_radius_x,
        surface_radius_y,
        surface_angle,
    ) = _fit_rotated_section_ellipse(surface_vertices, fit_center, section_x, section_y)

    surface_centered = surface_vertices - fit_center
    surface_section = np.column_stack([surface_centered @ section_x, surface_centered @ section_y])
    surface_rel = surface_section - surface_center
    surface_cos = np.cos(surface_angle)
    surface_sin = np.sin(surface_angle)
    ellipse_u = surface_rel[:, 0] * surface_cos + surface_rel[:, 1] * surface_sin
    ellipse_v = -surface_rel[:, 0] * surface_sin + surface_rel[:, 1] * surface_cos
    angles = np.mod(
        np.arctan2(
            ellipse_v / max(surface_radius_y, 1e-9),
            ellipse_u / max(surface_radius_x, 1e-9),
        ),
        2.0 * np.pi,
    )
    arc_start, arc_end = _occupied_angle_arc(angles)

    axial = centered @ axis
    axial_low, axial_high = np.percentile(axial, [7.5, 92.5])
    return FingerSegmentSurfaceFit(
        center=fit_center,
        axis=axis,
        section_x=section_x,
        section_y=section_y,
        axial_low=float(axial_low),
        axial_high=float(axial_high),
        surface_center=surface_center,
        surface_radius_x=float(surface_radius_x),
        surface_radius_y=float(surface_radius_y),
        surface_angle=float(surface_angle),
        arc_start=float(arc_start),
        arc_end=float(arc_end),
    )


def _finger_segment_raw_shell_grid_points(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Select a regular grid from the true outer shell of a segment STL."""
    vertices = triangles.reshape(-1, 3)
    fit = _fit_finger_segment_surfaces(vertices)
    points = _supersample_triangles(triangles, subdivisions=5)

    rel = points - fit.center
    axial = rel @ fit.axis
    sx = rel @ fit.section_x
    sy = rel @ fit.section_y
    dx = sx - fit.surface_center[0]
    dy = sy - fit.surface_center[1]
    ca = np.cos(fit.surface_angle)
    sa = np.sin(fit.surface_angle)
    eu = dx * ca + dy * sa
    ev = -dx * sa + dy * ca
    rx = max(fit.surface_radius_x, 1e-9)
    ry = max(fit.surface_radius_y, 1e-9)
    radius_norm = np.sqrt((eu / rx) ** 2 + (ev / ry) ** 2)
    theta = np.arctan2(ev / ry, eu / rx)

    arc_start = float(fit.arc_start)
    arc_end = float(fit.arc_end)
    while arc_end <= arc_start:
        arc_end += 2.0 * np.pi
    arc_mid = 0.5 * (arc_start + arc_end)
    theta = arc_mid + (theta - arc_mid + np.pi) % (2.0 * np.pi) - np.pi

    axial_low = float(fit.axial_low)
    axial_high = float(fit.axial_high)
    axial_span = max(axial_high - axial_low, 1e-9)
    angle_span = max(arc_end - arc_start, 1e-9)
    margin = np.deg2rad(8.0)
    usable = (
        (axial >= axial_low - 0.04 * axial_span)
        & (axial <= axial_high + 0.04 * axial_span)
        & (theta >= arc_start - margin)
        & (theta <= arc_end + margin)
    )
    if int(usable.sum()) < max(32, rows * cols):
        usable = np.ones(len(points), dtype=bool)

    points = points[usable]
    axial = axial[usable]
    theta = theta[usable]
    radius_norm = radius_norm[usable]

    axial_bins = max(28, rows * 4)
    angle_bins = max(96, cols * 8)
    ai = np.clip(
        ((axial - axial_low) / axial_span * axial_bins).astype(np.int32), 0, axial_bins - 1
    )
    ti = np.clip(
        ((theta - arc_start) / angle_span * angle_bins).astype(np.int32), 0, angle_bins - 1
    )
    key = ai * angle_bins + ti
    order = np.lexsort((-radius_norm, key))
    _, first = np.unique(key[order], return_index=True)
    envelope = order[first]
    env_points = points[envelope]
    env_axial = axial[envelope]
    env_theta = theta[envelope]
    env_radius = radius_norm[envelope]

    axial_edges = np.linspace(axial_low, axial_high, rows + 1)
    axial_values = 0.5 * (axial_edges[:-1] + axial_edges[1:])
    theta_values = _ellipse_arc_mid_angles(rx, ry, cols, arc_start, arc_end)

    result = []
    for target_axial in axial_values:
        for target_theta in theta_values:
            da = np.abs(env_axial - target_axial) / axial_span
            dt = np.abs(env_theta - target_theta) / angle_span
            local = np.flatnonzero((da <= 0.10) & (dt <= 0.10))
            if len(local) < 5:
                local = np.argsort((da / 0.055) ** 2 + (dt / 0.055) ** 2)[:32]
            local_r = env_radius[local]
            outer = local[local_r >= np.percentile(local_r, 60.0)]
            if len(outer) == 0:
                outer = local
            score = (da[outer] / 0.050) ** 2 + (dt[outer] / 0.050) ** 2
            result.append(env_points[outer[int(np.argmin(score))]])
    return np.asarray(result, dtype=np.float64)


def _finger_segment_regular_surface_grid_points(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Fit a smooth Bezier patch to a segment's detected outer shell."""
    dense_rows = max(18, rows * 4)
    dense_cols = max(32, cols * 4)
    raw = _finger_segment_raw_shell_grid_points(triangles, dense_rows, dense_cols)
    controls = _fit_bezier_surface(
        raw,
        dense_rows,
        dense_cols,
        degree_u=min(5, dense_rows - 1),
        degree_v=min(7, dense_cols - 1),
        regularization=5.0e-5,
    )
    u_values = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    v_values = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    return _evaluate_bezier_surface(controls, u_values, v_values)
