"""Fingertip swept-shell detection and smooth taxel-surface fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from source.sensors.tactile.fitting.bezier import (
    evaluate_bezier_surface as _evaluate_bezier_surface,
    fit_bezier_surface as _fit_bezier_surface,
)
from source.sensors.tactile.fitting.primitives import _supersample_triangles
from source.sensors.tactile.fitting.stl import read_stl_triangles

def fingertip_ellipsoid_grid_points(mesh_path: Path, rows: int, cols: int) -> np.ndarray:
    """Surface-following grid for rounded fingertip pads.

    A local PCA frame establishes consistent axial and circumferential
    coordinates for a swept-shell fit. Taxels are then interpolated directly
    on the real STL triangles, rather than placed on an ideal ellipsoid.
    ``phi`` is clamped to ``[0, pi/2]`` and the ``theta`` arc is centered from
    the outer-surface samples with width ``pi/2``, so the model covers a
    quarter ellipsoid instead of a half ellipsoid. The fingertip patch is concave, so the outer-surface sample
    cloud can be mirrored across the local plane parallel to global xy by
    ``FINGERTIP_MIRROR_ACROSS_XY`` before the ellipsoid axes and angular range
    are estimated.
    The mirrored outer-surface cloud is used to estimate one robust average
    ``a, b, c`` and a valid ``phi/theta`` range; every taxel is then sampled on
    that same regular ellipsoid.
    """
    triangles = read_stl_triangles(mesh_path)
    return _fingertip_swept_shell_grid_points_from_triangles(triangles, rows, cols)


def _fingertip_swept_shell_grid_points_from_triangles(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Sample the fingertip as a swept U-shaped shell, not an ellipsoid."""
    return _fingertip_regular_surface_grid_points(triangles, rows, cols)


def _fingertip_swept_shell_raw_grid_points(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    points = _supersample_triangles(triangles, subdivisions=5)
    if len(points) == 0:
        raise ValueError("Fingertip STL contains no usable surface points.")

    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].copy()
    section_x = vh[1].copy()

    axial_probe = centered @ axis
    low_mask = axial_probe <= np.percentile(axial_probe, 15.0)
    high_mask = axial_probe >= np.percentile(axial_probe, 85.0)

    def section_spread(mask: np.ndarray) -> float:
        local = centered[mask]
        section_y_probe = vh[2]
        return float(np.std(local @ section_x) + np.std(local @ section_y_probe))

    if section_spread(low_mask) < section_spread(high_mask):
        axis = -axis

    section_x = section_x - axis * float(section_x @ axis)
    section_x /= max(np.linalg.norm(section_x), 1e-12)
    section_y = np.cross(axis, section_x)
    section_y /= max(np.linalg.norm(section_y), 1e-12)

    local = points - center
    axial = local @ axis
    section_u = local @ section_x
    section_v = local @ section_y

    axial_low, axial_high = np.percentile(axial, [3.0, 97.0])
    axial_span = max(float(axial_high - axial_low), 1e-9)

    slice_count = 32
    slice_edges = np.linspace(axial_low, axial_high, slice_count + 1)
    slice_centres = 0.5 * (slice_edges[:-1] + slice_edges[1:])
    centre_u = np.full(slice_count, np.nan, dtype=np.float64)
    centre_v = np.full(slice_count, np.nan, dtype=np.float64)

    for idx, (start, end) in enumerate(zip(slice_edges[:-1], slice_edges[1:])):
        mask = (axial >= start) & (axial < end)
        if int(mask.sum()) < 20:
            continue
        centre_u[idx] = 0.5 * (
            np.percentile(section_u[mask], 2.0) + np.percentile(section_u[mask], 98.0)
        )
        centre_v[idx] = 0.5 * (
            np.percentile(section_v[mask], 2.0) + np.percentile(section_v[mask], 98.0)
        )

    valid = np.isfinite(centre_u) & np.isfinite(centre_v)
    if int(valid.sum()) < 2:
        centre_u[:] = np.median(section_u)
        centre_v[:] = np.median(section_v)
    else:
        centre_u = np.interp(
            slice_centres,
            slice_centres[valid],
            centre_u[valid],
        )
        centre_v = np.interp(
            slice_centres,
            slice_centres[valid],
            centre_v[valid],
        )

    point_centre_u = np.interp(axial, slice_centres, centre_u)
    point_centre_v = np.interp(axial, slice_centres, centre_v)
    radial_u = section_u - point_centre_u
    radial_v = section_v - point_centre_v
    radius = np.hypot(radial_u, radial_v)
    angle = np.mod(np.arctan2(radial_v, radial_u), 2.0 * np.pi)

    axial_bin_count = 28
    angle_bin_count = 144
    axial_bin = np.clip(
        ((axial - axial_low) / axial_span * axial_bin_count).astype(np.int32),
        0,
        axial_bin_count - 1,
    )
    angle_bin = np.clip(
        (angle / (2.0 * np.pi) * angle_bin_count).astype(np.int32),
        0,
        angle_bin_count - 1,
    )
    bin_key = axial_bin * angle_bin_count + angle_bin

    order = np.lexsort((-radius, bin_key))
    _, first = np.unique(bin_key[order], return_index=True)
    envelope_indices = order[first]

    envelope_axial = axial[envelope_indices]
    envelope_angle = angle[envelope_indices]
    envelope_radius = radius[envelope_indices]

    # Determine the complete exterior U-shell arc from cross-section support.
    # A fixed pi-wide window only captures the lowest/bottom part of this STL.
    # Instead, count whether each angular ray exists in many longitudinal
    # slices.  Genuine outer-shell directions persist along the finger, while
    # the opening, inner wall and the small tip-closing region occur in only a
    # few slices.
    support_slices = 20
    support_axial_edges = np.linspace(
        np.percentile(envelope_axial, 4.0),
        np.percentile(envelope_axial, 96.0),
        support_slices + 1,
    )
    angular_presence = np.zeros(angle_bin_count, dtype=np.float64)
    usable_slice_count = 0
    for start_u, end_u in zip(support_axial_edges[:-1], support_axial_edges[1:]):
        slice_mask = (envelope_axial >= start_u) & (envelope_axial < end_u)
        if int(slice_mask.sum()) < 8:
            continue
        hist, _ = np.histogram(
            envelope_angle[slice_mask],
            bins=angle_bin_count,
            range=(0.0, 2.0 * np.pi),
        )
        angular_presence += (hist > 0).astype(np.float64)
        usable_slice_count += 1

    if usable_slice_count == 0:
        angular_presence[:] = 1.0
        usable_slice_count = 1

    angular_presence = (
        0.20 * np.roll(angular_presence, 2)
        + 0.20 * np.roll(angular_presence, 1)
        + 0.20 * angular_presence
        + 0.20 * np.roll(angular_presence, -1)
        + 0.20 * np.roll(angular_presence, -2)
    )

    # Keep directions supported by at least roughly one quarter of the finger
    # length, then find their longest circular run.  This normally gives the
    # whole outside of the U: left wall -> rounded underside -> right wall.
    threshold = max(2.0, 0.24 * usable_slice_count)
    occupied = angular_presence >= threshold
    if int(occupied.sum()) < angle_bin_count // 4:
        threshold = max(1.0, 0.12 * usable_slice_count)
        occupied = angular_presence >= threshold

    doubled_occupied = np.concatenate([occupied, occupied])
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    for idx, is_occupied in enumerate(doubled_occupied):
        if is_occupied:
            if current_length == 0:
                current_start = idx
            current_length += 1
            if current_length > best_length and current_length <= angle_bin_count:
                best_start = current_start
                best_length = current_length
        else:
            current_length = 0

    bin_width = 2.0 * np.pi / angle_bin_count
    if best_length < angle_bin_count // 4:
        # Conservative fallback for an unusually sparse mesh.
        best_length = int(round(1.35 * np.pi / bin_width))
        best_start = int(np.argmax(angular_presence)) - best_length // 2

    # Include the edge/side-wall directions that may be one or two bins less
    # persistent because of tessellation and tapering near the fingertip.
    expansion_bins = max(3, int(round(np.deg2rad(8.0) / bin_width)))
    best_start -= expansion_bins
    best_length = min(
        angle_bin_count - 2,
        best_length + 2 * expansion_bins,
    )
    arc_start = best_start * bin_width
    arc_end = arc_start + best_length * bin_width

    arc_mid = 0.5 * (arc_start + arc_end)
    unwrapped_angle = arc_mid + (envelope_angle - arc_mid + np.pi) % (2.0 * np.pi) - np.pi
    arc_margin = np.deg2rad(8.0)
    in_arc = (unwrapped_angle >= arc_start - arc_margin) & (unwrapped_angle <= arc_end + arc_margin)
    envelope_axial = envelope_axial[in_arc]
    envelope_angle = unwrapped_angle[in_arc]
    envelope_radius = envelope_radius[in_arc]
    envelope_points = points[envelope_indices[in_arc]]

    # Include the distal tip explicitly.  The previous midpoint-only sampling
    # never reached the longitudinal boundary, so the fitted Bezier patch was
    # inevitably truncated before the fingertip nose.
    axial_start = float(np.percentile(axial, 2.0))
    axial_tip = float(np.percentile(axial, 99.7))
    row_parameter = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    # Slightly concentrate dense source rows near the distal end, where the
    # shell bends and closes most rapidly.
    row_values = axial_start + (axial_tip - axial_start) * (1.0 - (1.0 - row_parameter) ** 1.45)
    col_values = (
        arc_start + (np.arange(cols, dtype=np.float64) + 0.5) * (arc_end - arc_start) / cols
    )

    result: list[np.ndarray] = []
    axial_scale = max(float(axial_tip - axial_start), 1e-9)
    angle_scale = max(float(arc_end - arc_start), 1e-9)

    # Dedicated distal-cap candidates.  The swept-shell angular envelope gets
    # sparse where the U-shaped wall closes, so relying on it alone truncates
    # the surface before the physical nose.
    all_unwrapped_angle = arc_mid + (angle - arc_mid + np.pi) % (2.0 * np.pi) - np.pi
    tip_threshold = float(np.percentile(axial, 97.5))
    tip_candidate_mask = (
        (axial >= tip_threshold)
        & (all_unwrapped_angle >= arc_start - arc_margin)
        & (all_unwrapped_angle <= arc_end + arc_margin)
    )
    tip_points = points[tip_candidate_mask]
    tip_axial = axial[tip_candidate_mask]
    tip_angle = all_unwrapped_angle[tip_candidate_mask]

    for row_index, target_axial in enumerate(row_values):
        for target_angle in col_values:
            if row_index == rows - 1 and len(tip_points) >= cols:
                angular_error = (
                    np.abs((tip_angle - target_angle + np.pi) % (2.0 * np.pi) - np.pi) / angle_scale
                )
                axial_error = (axial_tip - tip_axial) / axial_scale
                score = (angular_error / 0.055) ** 2 + (axial_error / 0.035) ** 2
                nearby = np.argsort(score)[: max(12, min(48, len(score)))]
                # Prefer the front-most samples among angularly compatible
                # candidates, while retaining lateral variation across cols.
                best_pool = nearby[tip_axial[nearby] >= np.percentile(tip_axial[nearby], 70.0)]
                if len(best_pool) == 0:
                    best_pool = nearby
                best = best_pool[int(np.argmin(score[best_pool]))]
                result.append(tip_points[best])
                continue
            axial_distance = np.abs(envelope_axial - target_axial) / axial_scale
            angular_distance = (
                np.abs((envelope_angle - target_angle + np.pi) % (2.0 * np.pi) - np.pi)
                / angle_scale
            )

            local_mask = (axial_distance <= 0.11) & (angular_distance <= 0.11)
            local_indices = np.flatnonzero(local_mask)
            if len(local_indices) < 4:
                score = (axial_distance / 0.065) ** 2 + (angular_distance / 0.065) ** 2
                local_indices = np.argsort(score)[:24]

            local_radii = envelope_radius[local_indices]
            radial_threshold = np.percentile(local_radii, 65.0)
            outer_local = local_indices[local_radii >= radial_threshold]
            if len(outer_local) == 0:
                outer_local = local_indices

            score = (axial_distance[outer_local] / 0.060) ** 2 + (
                angular_distance[outer_local] / 0.060
            ) ** 2
            best = outer_local[int(np.argmin(score))]
            result.append(envelope_points[best])

    return np.asarray(result, dtype=np.float64)


def _fingertip_regular_surface_grid_points(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Fit a regular smooth surface to the detected outer fingertip shell.

    A denser swept-shell grid first identifies the correct outer contact
    surface.  A tensor-product Bezier patch is then fitted to those samples,
    and the requested taxels are sampled at equally spaced parameter-cell
    centres.  This keeps the grid visually regular without reverting to an
    inaccurate ellipsoid model.
    """
    dense_rows = max(12, rows * 3)
    dense_cols = max(24, cols * 3)
    raw_samples = _fingertip_swept_shell_raw_grid_points(
        triangles,
        dense_rows,
        dense_cols,
    )
    controls = _fit_bezier_surface(
        raw_samples,
        dense_rows,
        dense_cols,
        degree_u=min(6, dense_rows - 1),
        degree_v=min(7, dense_cols - 1),
        regularization=8.0e-6,
        include_u_endpoints=True,
        boundary_weight=10.0,
    )

    # Sample parameter-cell centres. Placing the final row exactly at u=1
    # collapses its columns at the fingertip pole, producing multiple touch
    # sites for effectively the same contact point. The last cell centre still
    # reaches the distal cap without creating that singular row.
    if rows == 1:
        u_values = np.asarray([0.5], dtype=np.float64)
    else:
        u_edges = np.linspace(0.06, 1.0, rows + 1, dtype=np.float64)
        u_values = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_values = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    return _evaluate_bezier_surface(controls, u_values, v_values)


def _fingertip_swept_shell_surface_from_triangles(
    triangles: np.ndarray,
    *,
    surface_rows: int = 28,
    surface_cols: int = 48,
) -> np.ndarray:
    dense_rows = max(16, surface_rows // 2)
    dense_cols = max(32, surface_cols // 2)
    raw_samples = _fingertip_swept_shell_raw_grid_points(triangles, dense_rows, dense_cols)
    controls = _fit_bezier_surface(
        raw_samples,
        dense_rows,
        dense_cols,
        degree_u=min(6, dense_rows - 1),
        degree_v=min(7, dense_cols - 1),
        regularization=8.0e-6,
        include_u_endpoints=True,
        boundary_weight=10.0,
    )
    u_values = np.linspace(0.0, 1.0, surface_rows, dtype=np.float64)
    v_values = np.linspace(0.0, 1.0, surface_cols, dtype=np.float64)
    return _evaluate_bezier_surface(controls, u_values, v_values).reshape(
        surface_rows, surface_cols, 3
    )
