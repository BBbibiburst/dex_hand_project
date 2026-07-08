# -*- coding: utf-8 -*-
"""Private STL parsing / surface-fitting helpers used only by the dex hand's
tactile sensor implementation.

Nothing in this module is part of any framework-level contract. It exists
purely because *this particular hand's* skin meshes are STL surfaces that
need to be turned into a taxel grid, and *this particular hand's* author
chose to implement that via geometric surface fitting. A different hand is
free to implement ``TactileSensorBase`` with a completely different strategy
(hand-authored coordinates, contact-force summation, a learned model, ...)
without ever importing this file.

The public surface consists of the grid strategies currently used by the dex
hand: fixed-shape segment fitting, ellipsoid-cap fingertip fitting,
outer-face free-form RBF fitting, and mesh-UV sampling for the palm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy.interpolate import RBFInterpolator


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
    arc_start: float
    arc_end: float


@dataclass(frozen=True)
class PatchPlotData:
    mesh_name: str
    rows: int
    cols: int
    triangles: np.ndarray
    samples: np.ndarray
    fit_surfaces: tuple[np.ndarray, ...]


# ---------------------------------------------------------------------------
# STL I/O
# ---------------------------------------------------------------------------


def read_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        tri_count = struct.unpack_from("<I", data, 80)[0]
        triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
        offset = 84
        for tri_idx in range(tri_count):
            offset += 12
            for vertex_idx in range(3):
                triangles[tri_idx, vertex_idx] = struct.unpack_from("<3f", data, offset)
                offset += 12
            offset += 2
        return triangles
    return _read_ascii_stl_triangles(data.decode("utf-8", errors="ignore"))


def read_stl_vertices(path: Path) -> np.ndarray:
    return read_stl_triangles(path).reshape(-1, 3)


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    tri_count = struct.unpack_from("<I", data, 80)[0]
    return 84 + tri_count * 50 == len(data)


def _read_ascii_stl_triangles(text: str) -> np.ndarray:
    vertices: list[list[float]] = []
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError("STL file contains no vertices.")
    if len(vertices) % 3 != 0:
        raise ValueError("ASCII STL vertex count is not divisible by 3.")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# Public entry points: mesh -> taxel grid points, by patch shape
# ---------------------------------------------------------------------------


def finger_segment_grid_points(mesh_path: Path, rows: int, cols: int) -> np.ndarray:
    """Taxel grid for a finger-segment skin pad (thick partial elliptic
    cylinder: one axis along the segment, the other around the exposed arc).
    """
    vertices = read_stl_vertices(mesh_path)
    fit = _fit_finger_segment_surfaces(vertices)
    theta_values = _ellipse_arc_mid_angles(
        fit.surface_radius_x, fit.surface_radius_y, cols, fit.arc_start, fit.arc_end
    )
    axial_edges = np.linspace(fit.axial_low, fit.axial_high, rows + 1, dtype=np.float64)
    axial_values = 0.5 * (axial_edges[:-1] + axial_edges[1:])

    points: list[np.ndarray] = []
    for z_value in axial_values:
        for theta in theta_values:
            x_value = fit.surface_center[0] + fit.surface_radius_x * np.cos(theta)
            y_value = fit.surface_center[1] + fit.surface_radius_y * np.sin(theta)
            points.append(
                fit.center
                + z_value * fit.axis
                + x_value * fit.section_x
                + y_value * fit.section_y
            )
    return np.asarray(points, dtype=np.float64)


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


def fingertip_ellipsoid_grid_points(mesh_path: Path, rows: int, cols: int) -> np.ndarray:
    """Stable ellipsoid-cap style grid for rounded fingertip pads.

    This follows the same spirit as the segment fitter: build a local PCA
    frame, keep likely outer/contact surface samples, detect the occupied
    angular arc, then sample a smooth grid on elliptical cross-sections. The
    radial scale is estimated per axial row from the actual mesh, which makes
    it more tolerant than a single ideal half-ellipsoid while staying much
    more stable than unconstrained RBF fitting.
    """
    triangles = read_stl_triangles(mesh_path)
    return _fingertip_ellipsoid_grid_points_from_triangles(triangles, rows, cols)


def freeform_rbf_outer_grid_points(
    mesh_path: Path,
    rows: int,
    cols: int,
    *,
    smoothing: float = 1e-7,
    max_training_points: int = 1200,
    subdivisions: int = 4,
) -> np.ndarray:
    """Free-form RBF grid fitted only from likely outer/contact faces.

    This keeps the flexible ``S(u, v) -> R3`` surface idea, but first filters
    thick STL patches down to the exposed side and supersamples those selected
    triangles. It is an experimental alternative to the fixed-shape segment
    and fingertip fitters.
    """
    triangles = read_stl_triangles(mesh_path)
    return _freeform_rbf_outer_grid_points_from_triangles(
        triangles,
        rows,
        cols,
        smoothing=smoothing,
        max_training_points=max_training_points,
        subdivisions=subdivisions,
    )


# ---------------------------------------------------------------------------
# Fit-surface reconstructions, used only by the offline plotting demo
# ---------------------------------------------------------------------------


def finger_segment_fit_surface(vertices: np.ndarray) -> np.ndarray:
    fit = _fit_finger_segment_surfaces(vertices)
    z_values = np.linspace(fit.axial_low, fit.axial_high, 32)
    theta_values = np.linspace(fit.arc_start, fit.arc_end, 48)
    z_grid, theta_grid = np.meshgrid(z_values, theta_values, indexing="ij")
    x_grid = fit.surface_center[0] + fit.surface_radius_x * np.cos(theta_grid)
    y_grid = fit.surface_center[1] + fit.surface_radius_y * np.sin(theta_grid)
    return (
        fit.center
        + z_grid[..., None] * fit.axis
        + x_grid[..., None] * fit.section_x
        + y_grid[..., None] * fit.section_y
    )


def patch_plot_data(
    mesh_path: Path,
    mesh_name: str,
    rows: int,
    cols: int,
    grid_fn,
    fit_fn,
) -> PatchPlotData:
    """Return all arrays needed to visualize one tactile patch (used by the
    offline sampling-plot demo only)."""
    triangles = read_stl_triangles(mesh_path)
    vertices = triangles.reshape(-1, 3)
    samples = grid_fn(mesh_path, rows, cols)
    fit_surfaces = (fit_fn(vertices),)
    return PatchPlotData(
        mesh_name=mesh_name,
        rows=rows,
        cols=cols,
        triangles=triangles,
        samples=samples,
        fit_surfaces=fit_surfaces,
    )


def patch_mesh_uv_plot_data(mesh_path: Path, mesh_name: str, rows: int, cols: int) -> PatchPlotData:
    """Plot-data helper for the mesh-UV sampling prototype."""
    triangles = read_stl_triangles(mesh_path)
    samples = _mesh_uv_grid_points_from_triangles(triangles, rows, cols)
    return PatchPlotData(
        mesh_name=mesh_name,
        rows=rows,
        cols=cols,
        triangles=triangles,
        samples=samples,
        fit_surfaces=(),
    )


def patch_fingertip_ellipsoid_plot_data(
    mesh_path: Path,
    mesh_name: str,
    rows: int,
    cols: int,
) -> PatchPlotData:
    """Plot-data helper for fingertip ellipsoid-cap fitting."""
    triangles = read_stl_triangles(mesh_path)
    samples = _fingertip_ellipsoid_grid_points_from_triangles(triangles, rows, cols)
    surface = _fingertip_ellipsoid_surface_from_triangles(triangles)
    return PatchPlotData(
        mesh_name=mesh_name,
        rows=rows,
        cols=cols,
        triangles=triangles,
        samples=samples,
        fit_surfaces=(surface,),
    )


def patch_freeform_rbf_outer_plot_data(
    mesh_path: Path,
    mesh_name: str,
    rows: int,
    cols: int,
) -> PatchPlotData:
    """Plot-data helper for outer-face supersampled RBF fitting."""
    triangles = read_stl_triangles(mesh_path)
    mask = _section_outer_face_mask(triangles)
    outer_triangles = triangles[mask]
    samples = _freeform_rbf_outer_grid_points_from_triangles(triangles, rows, cols)
    surface = _freeform_rbf_surface_from_points(
        _supersample_triangles(outer_triangles, subdivisions=4)
    )
    return PatchPlotData(
        mesh_name=mesh_name,
        rows=rows,
        cols=cols,
        triangles=triangles,
        samples=samples,
        fit_surfaces=(surface,),
    )


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


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

    if outer_faces.sum() < 0.05 * len(outer_faces) or (~outer_faces).sum() < 0.05 * len(outer_faces):
        outer_faces = r_norm >= np.percentile(r_norm, 60.0)

    surface_vertices = _supersample_triangles(
        triangles[outer_faces], subdivisions=FINGER_SEGMENT_FIT_SUBDIVISIONS
    )
    if len(surface_vertices) == 0:
        surface_vertices = fit_points

    surface_center, surface_radius_x, surface_radius_y = _fit_axis_aligned_section_ellipse(
        surface_vertices, fit_center, section_x, section_y
    )

    surface_centered = surface_vertices - fit_center
    surface_section = np.column_stack(
        [surface_centered @ section_x, surface_centered @ section_y]
    )
    surface_rel = surface_section - surface_center
    angles = np.mod(
        np.arctan2(
            surface_rel[:, 1] / max(surface_radius_y, 1e-9),
            surface_rel[:, 0] / max(surface_radius_x, 1e-9),
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
        arc_start=float(arc_start),
        arc_end=float(arc_end),
    )


def _fit_axis_aligned_section_ellipse(
    points: np.ndarray, center: np.ndarray, section_x: np.ndarray, section_y: np.ndarray
) -> tuple[np.ndarray, float, float]:
    centered = points - center
    section = np.column_stack([centered @ section_x, centered @ section_y])
    ellipse_center = np.median(section, axis=0)
    rel = section - ellipse_center
    radius_x = max(np.percentile(np.abs(rel[:, 0]), 95.0), 1e-9)
    radius_y = max(np.percentile(np.abs(rel[:, 1]), 95.0), 1e-9)
    return ellipse_center, radius_x, radius_y


def _vertices_as_triangles(vertices: np.ndarray) -> np.ndarray:
    usable = (len(vertices) // 3) * 3
    if usable == 0:
        raise ValueError("STL vertices cannot form triangles.")
    return vertices[:usable].reshape(-1, 3, 3)


def _supersample_triangles(triangles: np.ndarray, *, subdivisions: int) -> np.ndarray:
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=np.float64)
    subdivisions = max(1, int(subdivisions))
    barycentric = []
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            k = subdivisions - i - j
            barycentric.append((i, j, k))
    weights = np.asarray(barycentric, dtype=np.float64) / float(subdivisions)
    return (
        triangles[:, None, 0, :] * weights[None, :, 0, None]
        + triangles[:, None, 1, :] * weights[None, :, 1, None]
        + triangles[:, None, 2, :] * weights[None, :, 2, None]
    ).reshape(-1, 3)


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0
    return normals


def _mesh_uv_grid_points_from_triangles(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    vertices = triangles.reshape(-1, 3)
    center = vertices.mean(axis=0)
    centered = vertices - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u_axis, v_axis = vh[0], vh[1]

    tri_uv = np.stack(
        [
            (triangles - center) @ u_axis,
            (triangles - center) @ v_axis,
        ],
        axis=-1,
    )
    vertex_uv = tri_uv.reshape(-1, 2)
    u_values = _linspace_midpoints(vertex_uv[:, 0], cols)
    v_values = _linspace_midpoints(vertex_uv[:, 1], rows)

    points: list[np.ndarray] = []
    for v_value in v_values:
        for u_value in u_values:
            uv = np.asarray([u_value, v_value], dtype=np.float64)
            tri_index, bary = _locate_projected_triangle(tri_uv, uv)
            points.append(_barycentric_interpolate(triangles[tri_index], bary))
    return np.asarray(points, dtype=np.float64)


def _fingertip_ellipsoid_grid_points_from_triangles(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    fit = _fit_fingertip_ellipsoid_cap(triangles)
    theta_values = _ellipse_arc_mid_angles(
        fit["radius_x"],
        fit["radius_y"],
        cols,
        fit["arc_start"],
        fit["arc_end"],
    )
    axial_values = _linspace_midpoints(fit["outer_axial"], rows)
    scales = _fingertip_row_scales(fit, axial_values)
    return _fingertip_points_from_fit(fit, axial_values, theta_values, scales)


def _fingertip_ellipsoid_surface_from_triangles(
    triangles: np.ndarray,
    *,
    surface_rows: int = 28,
    surface_cols: int = 48,
) -> np.ndarray:
    fit = _fit_fingertip_ellipsoid_cap(triangles)
    axial_values = _linspace_percentile(fit["outer_axial"], surface_rows)
    theta_values = np.linspace(fit["arc_start"], fit["arc_end"], surface_cols)
    scales = _fingertip_row_scales(fit, axial_values)
    points = _fingertip_points_from_fit(fit, axial_values, theta_values, scales)
    return points.reshape(surface_rows, surface_cols, 3)


def _fit_fingertip_ellipsoid_cap(triangles: np.ndarray) -> dict[str, np.ndarray | float]:
    fit_points = _supersample_triangles(triangles, subdivisions=3)
    center = fit_points.mean(axis=0)
    centered = fit_points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis, section_x, section_y = vh[0], vh[1], vh[2]

    outer_triangles = triangles[_section_outer_face_mask(triangles)]
    if len(outer_triangles) == 0:
        outer_triangles = triangles
    outer_points = _supersample_triangles(outer_triangles, subdivisions=4)
    outer_centered = outer_points - center
    outer_axial = outer_centered @ axis
    outer_section = np.column_stack([outer_centered @ section_x, outer_centered @ section_y])
    section_center = np.median(outer_section, axis=0)
    outer_rel = outer_section - section_center

    radius_x = max(np.percentile(np.abs(outer_rel[:, 0]), 92.5), 1e-9)
    radius_y = max(np.percentile(np.abs(outer_rel[:, 1]), 92.5), 1e-9)
    norm_radius = np.sqrt((outer_rel[:, 0] / radius_x) ** 2 + (outer_rel[:, 1] / radius_y) ** 2)
    angles = np.mod(
        np.arctan2(outer_rel[:, 1] / radius_y, outer_rel[:, 0] / radius_x),
        2.0 * np.pi,
    )
    arc_start, arc_end = _occupied_angle_arc(angles)
    return {
        "center": center,
        "axis": axis,
        "section_x": section_x,
        "section_y": section_y,
        "section_center": section_center,
        "radius_x": float(radius_x),
        "radius_y": float(radius_y),
        "outer_axial": outer_axial,
        "norm_radius": norm_radius,
        "arc_start": float(arc_start),
        "arc_end": float(arc_end),
    }


def _fingertip_row_scales(
    fit: dict[str, np.ndarray | float],
    axial_values: np.ndarray,
) -> np.ndarray:
    outer_axial = np.asarray(fit["outer_axial"], dtype=np.float64)
    norm_radius = np.asarray(fit["norm_radius"], dtype=np.float64)
    axial_span = max(np.percentile(outer_axial, 92.5) - np.percentile(outer_axial, 7.5), 1e-9)
    window = 0.20 * axial_span
    scales = []
    for axial_value in axial_values:
        mask = np.abs(outer_axial - axial_value) <= window
        if int(mask.sum()) < 8:
            mask = np.ones_like(outer_axial, dtype=bool)
        scales.append(np.percentile(norm_radius[mask], 82.0))
    scales_arr = np.asarray(scales, dtype=np.float64)
    return np.clip(scales_arr, 0.25, 1.15)


def _fingertip_points_from_fit(
    fit: dict[str, np.ndarray | float],
    axial_values: np.ndarray,
    theta_values: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    center = np.asarray(fit["center"], dtype=np.float64)
    axis = np.asarray(fit["axis"], dtype=np.float64)
    section_x = np.asarray(fit["section_x"], dtype=np.float64)
    section_y = np.asarray(fit["section_y"], dtype=np.float64)
    section_center = np.asarray(fit["section_center"], dtype=np.float64)
    radius_x = float(fit["radius_x"])
    radius_y = float(fit["radius_y"])

    points: list[np.ndarray] = []
    for axial_value, scale in zip(axial_values, scales):
        for theta in theta_values:
            x_value = section_center[0] + scale * radius_x * np.cos(theta)
            y_value = section_center[1] + scale * radius_y * np.sin(theta)
            points.append(
                center
                + axial_value * axis
                + x_value * section_x
                + y_value * section_y
            )
    return np.asarray(points, dtype=np.float64)


def _freeform_rbf_outer_grid_points_from_triangles(
    triangles: np.ndarray,
    rows: int,
    cols: int,
    *,
    smoothing: float = 1e-7,
    max_training_points: int = 1200,
    subdivisions: int = 4,
) -> np.ndarray:
    outer_triangles = triangles[_section_outer_face_mask(triangles)]
    if len(outer_triangles) == 0:
        outer_triangles = triangles
    points = _supersample_triangles(outer_triangles, subdivisions=subdivisions)
    uv, xyz = _pca_parameter_points_from_vertices(points)
    rbf = _fit_rbf_surface(uv, xyz, smoothing=smoothing, max_training_points=max_training_points)
    grid_uv = _regular_uv_midpoint_grid(uv, rows, cols)
    return rbf(grid_uv).astype(np.float64)


def _freeform_rbf_surface_from_points(
    points: np.ndarray,
    *,
    smoothing: float = 1e-7,
    max_training_points: int = 1200,
    surface_rows: int = 32,
    surface_cols: int = 48,
) -> np.ndarray:
    uv, xyz = _pca_parameter_points_from_vertices(points)
    rbf = _fit_rbf_surface(uv, xyz, smoothing=smoothing, max_training_points=max_training_points)
    grid_uv = _regular_uv_line_grid(uv, surface_rows, surface_cols)
    return rbf(grid_uv).reshape(surface_rows, surface_cols, 3).astype(np.float64)


def _pca_parameter_points_from_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    centered = vertices - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    uv = np.column_stack([centered @ vh[0], centered @ vh[1]])
    return _normalize_uv(uv), vertices.astype(np.float64)


def _section_outer_face_mask(triangles: np.ndarray) -> np.ndarray:
    """Heuristically select the exposed side of a thick skin patch.

    The selection mirrors the robust idea used by the fixed segment fitter:
    in a PCA frame, faces farther from the section center and whose normals
    point radially outward are preferred. If normals are unreliable or the
    vote degenerates, it falls back to high normalized section radius.
    """
    vertices = triangles.reshape(-1, 3)
    center = vertices.mean(axis=0)
    centered = vertices - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    section_x, section_y = vh[1], vh[2]

    face_centers = triangles.mean(axis=1)
    face_normals = _triangle_normals(triangles)
    face_centered = face_centers - center
    face_section = np.column_stack([face_centered @ section_x, face_centered @ section_y])

    coarse_center = np.median(face_section, axis=0)
    face_rel = face_section - coarse_center
    radius_x = max(np.percentile(np.abs(face_rel[:, 0]), 95.0), 1e-9)
    radius_y = max(np.percentile(np.abs(face_rel[:, 1]), 95.0), 1e-9)
    r_norm = np.sqrt((face_rel[:, 0] / radius_x) ** 2 + (face_rel[:, 1] / radius_y) ** 2)

    radial_len = np.linalg.norm(face_rel, axis=1) + 1e-12
    radial = face_rel / radial_len[:, None]
    normal_section = np.column_stack([face_normals @ section_x, face_normals @ section_y])
    normal_radial = np.sum(normal_section * radial, axis=1)

    if np.std(r_norm) < 0.02:
        mask = normal_radial > 0.0
    else:
        pos_vote = (r_norm >= np.median(r_norm)).astype(np.float64)
        normal_vote = (normal_radial > 0.0).astype(np.float64)
        mask = (0.7 * pos_vote + 0.3 * normal_vote) > 0.5

    min_faces = max(8, int(0.1 * len(triangles)))
    if int(mask.sum()) < min_faces or int((~mask).sum()) < min_faces:
        mask = r_norm >= np.percentile(r_norm, 60.0)
    return mask


def _normalize_uv(uv: np.ndarray) -> np.ndarray:
    center = np.median(uv, axis=0)
    span = np.percentile(uv, 95.0, axis=0) - np.percentile(uv, 5.0, axis=0)
    span = np.where(span > 1e-12, span, 1.0)
    return (uv - center) / span


def _fit_rbf_surface(
    uv: np.ndarray,
    xyz: np.ndarray,
    *,
    smoothing: float,
    max_training_points: int,
) -> RBFInterpolator:
    if len(uv) > max_training_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(uv), size=max_training_points, replace=False)
        uv = uv[indices]
        xyz = xyz[indices]
    return RBFInterpolator(
        uv,
        xyz,
        kernel="thin_plate_spline",
        smoothing=smoothing,
        degree=1,
    )


def _regular_uv_midpoint_grid(uv: np.ndarray, rows: int, cols: int) -> np.ndarray:
    u_values = _linspace_midpoints(uv[:, 0], cols)
    v_values = _linspace_midpoints(uv[:, 1], rows)
    u_grid, v_grid = np.meshgrid(u_values, v_values, indexing="xy")
    return np.column_stack([u_grid.ravel(), v_grid.ravel()])


def _regular_uv_line_grid(uv: np.ndarray, rows: int, cols: int) -> np.ndarray:
    u_values = _linspace_percentile(uv[:, 0], cols)
    v_values = _linspace_percentile(uv[:, 1], rows)
    u_grid, v_grid = np.meshgrid(u_values, v_values, indexing="xy")
    return np.column_stack([u_grid.ravel(), v_grid.ravel()])


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


def _barycentric_2d(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) <= 1e-14:
        distances = np.sum((triangle - point) ** 2, axis=1)
        bary = np.zeros(3, dtype=np.float64)
        bary[int(np.argmin(distances))] = 1.0
        return bary
    inv = 1.0 / denom
    beta = (v2[0] * v1[1] - v1[0] * v2[1]) * inv
    gamma = (v0[0] * v2[1] - v2[0] * v0[1]) * inv
    alpha = 1.0 - beta - gamma
    return np.asarray([alpha, beta, gamma], dtype=np.float64)


def _barycentric_interpolate(values: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return values[0] * bary[0] + values[1] * bary[1] + values[2] * bary[2]


def _occupied_angle_arc(angles: np.ndarray, *, bins: int = 96) -> tuple[float, float]:
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi))
    occupied = hist > 0
    doubled_empty = np.concatenate([~occupied, ~occupied])

    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for idx, is_empty in enumerate(doubled_empty):
        if is_empty:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_len = 0

    if best_len == 0 or best_len >= bins:
        return 0.0, 2.0 * np.pi

    bin_width = 2.0 * np.pi / bins
    start = ((best_start + best_len) % bins) * bin_width
    arc_bins = bins - min(best_len, bins)
    end = start + arc_bins * bin_width
    return start, end


def _ellipse_arc_mid_angles(
    radius_x: float, radius_y: float, count: int, start: float, end: float
) -> np.ndarray:
    if count == 1:
        return np.asarray([(start + end) * 0.5], dtype=np.float64)

    samples = np.linspace(start, end, 512, dtype=np.float64)
    speed = np.sqrt((radius_x * np.sin(samples)) ** 2 + (radius_y * np.cos(samples)) ** 2)
    cumulative = np.zeros_like(samples)
    cumulative[1:] = np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(samples))
    targets = (np.arange(count, dtype=np.float64) + 0.5) * cumulative[-1] / count
    return np.interp(targets, cumulative, samples)


def _angle_distance(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((values - target + np.pi) % (2.0 * np.pi) - np.pi)


def _linspace_midpoints(values: np.ndarray, count: int) -> np.ndarray:
    low, high = np.percentile(values, [7.5, 92.5])
    if count == 1:
        return np.asarray([(low + high) * 0.5], dtype=np.float64)
    edges = np.linspace(low, high, count + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def _linspace_percentile(values: np.ndarray, count: int) -> np.ndarray:
    low, high = np.percentile(values, [7.5, 92.5])
    if count == 1:
        return np.asarray([(low + high) * 0.5], dtype=np.float64)
    return np.linspace(low, high, count, dtype=np.float64)
