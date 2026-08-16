"""Mesh-UV projection and barycentric surface interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from source.sensors.tactile.fitting.primitives import (
    _barycentric_2d, _barycentric_coordinates_2d, _barycentric_interpolate,
    _linspace_midpoints,
)
from source.sensors.tactile.fitting.stl import read_stl_triangles

def mesh_uv_grid_points(mesh_path: Path, rows: int, cols: int) -> np.ndarray:
    """Taxel grid by projecting the STL patch to a local 2D parameter domain.

    This is a mesh-first fallback for irregular skin surfaces. It does not try
    to fit the surface to a cylinder, ellipsoid, or plane. Instead it:

    1. Builds a local PCA frame for the mesh patch.
    2. Projects every triangle into the first two PCA coordinates.
    3. Places a regular ``rows x cols`` grid in that 2D domain.
    4. Maps each 2D grid point back to a 3D triangle with barycentric
       interpolation, falling back to nearest projected triangle if the 2D
       point lands in a small hole/outside the projected hull.

    It works best when a skin patch is topologically disk-like and does not
    fold over itself heavily in the chosen local PCA projection.
    """
    triangles = read_stl_triangles(mesh_path)
    return _mesh_uv_grid_points_from_triangles(triangles, rows, cols)


def _projected_surface_candidates(
    triangles: np.ndarray,
    tri_uv: np.ndarray,
    uv: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Find every 3D triangle whose 2D projection contains uv."""

    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []

    for tri_index, projected_triangle in enumerate(tri_uv):
        # Cheap bounding-box rejection.
        uv_min = projected_triangle.min(axis=0)
        uv_max = projected_triangle.max(axis=0)

        if np.any(uv < uv_min - 1e-9) or np.any(uv > uv_max + 1e-9):
            continue

        bary = _barycentric_coordinates_2d(
            projected_triangle,
            uv,
        )

        if bary is None:
            continue

        point_3d = _barycentric_interpolate(
            triangles[tri_index],
            bary,
        )

        candidates.append((tri_index, bary, point_3d))

    return candidates


def _mesh_uv_grid_points_from_triangles(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Sample a regular grid on the palm's outer surface."""

    triangles = np.asarray(triangles, dtype=np.float64)
    vertices = triangles.reshape(-1, 3)

    center = vertices.mean(axis=0)
    centered = vertices - center

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    u_axis = vh[0].copy()
    v_axis = vh[1].copy()
    normal_axis = vh[2].copy()

    # Keep a right-handed local coordinate frame.
    if np.dot(np.cross(u_axis, v_axis), normal_axis) < 0.0:
        normal_axis *= -1.0

    local_triangles = triangles - center

    tri_uv = np.stack(
        [
            local_triangles @ u_axis,
            local_triangles @ v_axis,
        ],
        axis=-1,
    )

    vertex_uv = tri_uv.reshape(-1, 2)

    # Retain the existing regular interior sampling.
    u_values = _linspace_midpoints(
        vertex_uv[:, 0],
        cols,
    )
    v_values = _linspace_midpoints(
        vertex_uv[:, 1],
        rows,
    )

    # Determine which normal side is the tactile outer surface.
    #
    # For a closed shell the two major surfaces lie on opposite sides of
    # the PCA centre. The side with more outward-facing triangle normals
    # is selected below. If your STL winding is unreliable, the fallback
    # is controlled by PALM_OUTER_SIGN.
    outer_sign = -1.0

    points: list[np.ndarray] = []

    for v_value in v_values:
        for u_value in u_values:
            uv = np.asarray(
                [u_value, v_value],
                dtype=np.float64,
            )

            candidates = _projected_surface_candidates(
                triangles,
                tri_uv,
                uv,
            )

            if not candidates:
                # Preserve the old nearest-triangle fallback around mesh edges.
                tri_index, bary = _locate_projected_triangle(
                    tri_uv,
                    uv,
                )

                point = _barycentric_interpolate(
                    triangles[tri_index],
                    bary,
                )

                points.append(point)
                continue

            candidate_points = np.asarray(
                [candidate[2] for candidate in candidates],
                dtype=np.float64,
            )

            heights = (candidate_points - center) @ normal_axis

            if outer_sign > 0.0:
                selected_index = int(np.argmax(heights))
            else:
                selected_index = int(np.argmin(heights))

            points.append(candidate_points[selected_index])

    return np.asarray(points, dtype=np.float64)


def _locate_projected_triangle(
    tri_uv: np.ndarray,
    uv: np.ndarray,
) -> tuple[int, np.ndarray]:
    best_index = 0
    best_bary = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    best_score = np.inf

    mins = tri_uv.min(axis=1)
    maxs = tri_uv.max(axis=1)
    candidate_mask = np.all(uv >= mins - 1e-12, axis=1) & np.all(uv <= maxs + 1e-12, axis=1)
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(tri_uv))

    for tri_index in candidate_indices:
        bary = _barycentric_2d(uv, tri_uv[tri_index])
        min_bary = float(bary.min())
        if min_bary >= -1e-8:
            return int(tri_index), bary

        projected = _barycentric_interpolate(tri_uv[tri_index], np.clip(bary, 0.0, 1.0))
        score = float(np.sum((projected - uv) ** 2) - min_bary * 1e-12)
        if score < best_score:
            best_score = score
            best_index = int(tri_index)
            best_bary = np.clip(bary, 0.0, 1.0)
            total = best_bary.sum()
            if total > 1e-12:
                best_bary /= total
            else:
                best_bary = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    return best_index, best_bary
