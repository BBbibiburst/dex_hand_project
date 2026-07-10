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
hand: fixed-shape segment fitting, quarter-ellipsoid fingertip fitting,
and mesh-UV sampling for the palm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


FINGER_SEGMENT_FIT_SUBDIVISIONS = 4
FINGERTIP_MAX_PHI = 0.5 * np.pi
FINGERTIP_MAX_THETA_ARC = np.pi
FINGERTIP_MIRROR_ACROSS_XY = True
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEX_HAND_MESH_DIR = PROJECT_ROOT / "assets" / "grippers" / "dex_hand" / "meshes"
DEFAULT_PLOT_PATCHES = ("skin_0_0_p", "skin_0_2_p", "skin_palm_p")


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


def dex_hand_patch_layout() -> tuple[tuple[str, int, int, str], ...]:
    """Return the dex-hand tactile patch layout and fitting strategy names."""
    layout: list[tuple[str, int, int, str]] = []
    for finger_id in range(5):
        layout.append((f"skin_{finger_id}_0_p", 7, 8, "segment"))
        layout.append((f"skin_{finger_id}_1_p", 4, 8, "segment"))
        layout.append((f"skin_{finger_id}_2_p", 4, 8, "fingertip-ellipsoid"))
    layout.append(("skin_palm_p", 7, 16, "mesh-uv"))
    return tuple(layout)


DEX_HAND_PATCH_LAYOUT = dex_hand_patch_layout()


def dex_hand_patch_info() -> dict[str, tuple[int, int, str]]:
    return {
        mesh_name: (rows, cols, kind)
        for mesh_name, rows, cols, kind in DEX_HAND_PATCH_LAYOUT
    }


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
    """Stable quarter-ellipsoid style grid for rounded fingertip pads.

    This follows the same spirit as the segment fitter: build a local PCA
    frame, keep likely outer/contact surface samples, detect the occupied
    angular arc, then sample a smooth grid on a concave quarter ellipsoid.
    ``phi`` is clamped to ``[0, pi/2]`` and the ``theta`` arc is clamped to at
    most ``pi``, so the model covers a quarter ellipsoid instead of a half
    ellipsoid. The fingertip patch is concave, so the generated cap is mirrored
    across the local plane parallel to global xy by ``FINGERTIP_MIRROR_ACROSS_XY``.
    Local scale is estimated per ``phi`` row from the actual mesh, which makes
    it more tolerant than a single ideal ellipsoid while staying much more
    stable than unconstrained free-form fitting.
    """
    triangles = read_stl_triangles(mesh_path)
    return _fingertip_ellipsoid_grid_points_from_triangles(triangles, rows, cols)


GRID_POINT_FUNCTIONS = {
    "segment": finger_segment_grid_points,
    "mesh-uv": mesh_uv_grid_points,
    "fingertip-ellipsoid": fingertip_ellipsoid_grid_points,
}


def grid_points_for_kind(
    kind: str,
    mesh_path: Path,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Compute grid points for one dex-hand fitting strategy."""
    try:
        grid_fn = GRID_POINT_FUNCTIONS[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dex-hand tactile fitting strategy {kind!r}. "
            f"Known: {sorted(GRID_POINT_FUNCTIONS)}"
        ) from exc
    return grid_fn(mesh_path, rows, cols)


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
    """Plot-data helper for fingertip quarter-ellipsoid fitting."""
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
    phi_values = _linspace_interval_midpoints(fit["phi_lo"], fit["phi_hi"], rows)
    theta_values = _ellipse_arc_mid_angles(
        fit["b"], fit["c"], cols, fit["arc_start"], fit["arc_end"]
    )
    scales = _fingertip_phi_scales(fit, phi_values)
    return _fingertip_points_from_fit(fit, phi_values, theta_values, scales)


def _fingertip_ellipsoid_surface_from_triangles(
    triangles: np.ndarray,
    *,
    surface_rows: int = 28,
    surface_cols: int = 48,
) -> np.ndarray:
    fit = _fit_fingertip_ellipsoid_cap(triangles)
    phi_values = np.linspace(fit["phi_lo"], fit["phi_hi"], surface_rows)
    theta_values = np.linspace(fit["arc_start"], fit["arc_end"], surface_cols)
    scales = _fingertip_phi_scales(fit, phi_values)
    points = _fingertip_points_from_fit(fit, phi_values, theta_values, scales)
    return points.reshape(surface_rows, surface_cols, 3)


def _fit_fingertip_ellipsoid_cap(triangles: np.ndarray) -> dict[str, np.ndarray | float]:
    fit_points = _supersample_triangles(triangles, subdivisions=3)
    pca_center = fit_points.mean(axis=0)
    centered = fit_points - pca_center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis, section_x, section_y = vh[0], vh[1], vh[2]

    axial_all = centered @ axis
    lo_mask = axial_all <= np.percentile(axial_all, 15.0)
    hi_mask = axial_all >= np.percentile(axial_all, 85.0)
    lo_spread = np.std(centered[lo_mask] @ section_x) + np.std(centered[lo_mask] @ section_y)
    hi_spread = np.std(centered[hi_mask] @ section_x) + np.std(centered[hi_mask] @ section_y)
    if lo_spread < hi_spread:
        axis = -axis

    outer_triangles = triangles[_fingertip_outer_face_mask(triangles, pca_center, axis)]
    if len(outer_triangles) == 0:
        outer_triangles = triangles
    outer_points = _supersample_triangles(outer_triangles, subdivisions=4)
    outer_centered_pca = outer_points - pca_center
    outer_u0 = outer_centered_pca @ axis
    base_u = np.percentile(outer_u0, 5.0)
    tip_u = np.percentile(outer_u0, 97.0)
    center = pca_center + base_u * axis
    outer_centered = outer_points - center

    u = outer_centered @ axis
    v = outer_centered @ section_x
    w = outer_centered @ section_y

    a = max(tip_u - base_u, np.percentile(np.abs(u), 90.0), 1e-9)
    b = max(np.percentile(np.abs(v), 92.5), 1e-9)
    c = max(np.percentile(np.abs(w), 92.5), 1e-9)

    un = np.clip(u / a, -1.0, 1.0)
    phi = np.arccos(un)
    theta = np.mod(np.arctan2(w / c, v / b), 2.0 * np.pi)

    phi_hi = min(float(np.percentile(phi, 96.0)), float(FINGERTIP_MAX_PHI))
    arc_start, arc_end = _occupied_angle_arc_limited(
        theta,
        max_width=FINGERTIP_MAX_THETA_ARC,
    )
    return {
        "center": center,
        "axis": axis,
        "section_x": section_x,
        "section_y": section_y,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "phi_lo": 0.0,
        "phi_hi": phi_hi,
        "arc_start": float(arc_start),
        "arc_end": float(arc_end),
        "outer_u": u,
        "outer_v": v,
        "outer_w": w,
        "outer_phi": phi,
    }


def _fingertip_outer_face_mask(
    triangles: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    face_centers = triangles.mean(axis=1)
    face_normals = _triangle_normals(triangles)
    face_rel = face_centers - center
    radial_len = np.linalg.norm(face_rel, axis=1) + 1e-12
    radial_dir = face_rel / radial_len[:, None]
    normal_radial_3d = np.sum(face_normals * radial_dir, axis=1)

    normal_axial = face_normals @ axis
    face_axial = face_rel @ axis
    axial_low = np.percentile(face_axial, 10.0)
    is_base_cut = (normal_axial < -0.85) & (face_axial < axial_low)

    r_norm3d = radial_len / max(np.percentile(radial_len, 95.0), 1e-9)
    if np.std(r_norm3d) < 0.02:
        mask = normal_radial_3d > 0.0
    else:
        pos_vote = (r_norm3d >= np.median(r_norm3d)).astype(np.float64)
        normal_vote = (normal_radial_3d > 0.0).astype(np.float64)
        mask = (0.6 * pos_vote + 0.4 * normal_vote) > 0.5

    mask = mask & (~is_base_cut)
    min_faces = max(8, int(0.1 * len(triangles)))
    if int(mask.sum()) < min_faces:
        mask = (r_norm3d >= np.percentile(r_norm3d, 60.0)) & (~is_base_cut)
    return mask


def _fingertip_phi_scales(
    fit: dict[str, np.ndarray | float],
    phi_values: np.ndarray,
) -> np.ndarray:
    outer_phi = np.asarray(fit["outer_phi"], dtype=np.float64)
    u = np.asarray(fit["outer_u"], dtype=np.float64)
    v = np.asarray(fit["outer_v"], dtype=np.float64)
    w = np.asarray(fit["outer_w"], dtype=np.float64)
    a = float(fit["a"])
    b = float(fit["b"])
    c = float(fit["c"])
    norm_radius = np.sqrt((u / a) ** 2 + (v / b) ** 2 + (w / c) ** 2)

    phi_span = max(float(fit["phi_hi"]) - float(fit["phi_lo"]), 1e-9)
    window = 0.15 * phi_span
    scales = []
    for phi_value in phi_values:
        mask = np.abs(outer_phi - phi_value) <= window
        if int(mask.sum()) < 8:
            mask = np.ones_like(outer_phi, dtype=bool)
        scales.append(np.percentile(norm_radius[mask], 80.0))
    scales_arr = np.asarray(scales, dtype=np.float64)
    return np.clip(scales_arr, 0.5, 1.3)


def _fingertip_points_from_fit(
    fit: dict[str, np.ndarray | float],
    phi_values: np.ndarray,
    theta_values: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    center = np.asarray(fit["center"], dtype=np.float64)
    axis = np.asarray(fit["axis"], dtype=np.float64)
    section_x = np.asarray(fit["section_x"], dtype=np.float64)
    section_y = np.asarray(fit["section_y"], dtype=np.float64)
    a = float(fit["a"])
    b = float(fit["b"])
    c = float(fit["c"])

    points: list[np.ndarray] = []
    for phi_value, scale in zip(phi_values, scales):
        sin_phi = np.sin(phi_value)
        for theta in theta_values:
            x_value = scale * a * np.cos(phi_value)
            y_value = scale * b * sin_phi * np.cos(theta)
            z_value = scale * c * sin_phi * np.sin(theta)
            point = (
                center
                + x_value * axis
                + y_value * section_x
                + z_value * section_y
            )
            if FINGERTIP_MIRROR_ACROSS_XY:
                point = point.copy()
                point[2] = 2.0 * center[2] - point[2]
            points.append(point)
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


def _occupied_angle_arc_limited(
    angles: np.ndarray,
    *,
    max_width: float,
    bins: int = 96,
) -> tuple[float, float]:
    start, end = _occupied_angle_arc(angles, bins=bins)
    if end - start <= max_width:
        return start, end

    hist, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi))
    bin_width = 2.0 * np.pi / bins
    window_bins = max(1, min(bins, int(round(max_width / bin_width))))
    doubled = np.concatenate([hist, hist])
    prefix = np.concatenate([[0], np.cumsum(doubled)])
    scores = prefix[window_bins : window_bins + bins] - prefix[:bins]
    best_start_bin = int(np.argmax(scores))
    limited_start = best_start_bin * bin_width
    limited_end = limited_start + window_bins * bin_width
    return limited_start, limited_end


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


def _linspace_interval_midpoints(low: float, high: float, count: int) -> np.ndarray:
    if count == 1:
        return np.asarray([(low + high) * 0.5], dtype=np.float64)
    edges = np.linspace(low, high, count + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def _linspace_percentile(values: np.ndarray, count: int) -> np.ndarray:
    low, high = np.percentile(values, [7.5, 92.5])
    if count == 1:
        return np.asarray([(low + high) * 0.5], dtype=np.float64)
    return np.linspace(low, high, count, dtype=np.float64)


# ---------------------------------------------------------------------------
# Offline visualization CLI
# ---------------------------------------------------------------------------


FIT_SURFACE_FUNCTIONS = {
    "segment": finger_segment_fit_surface,
    "mesh-uv": None,
    "fingertip-ellipsoid": patch_fingertip_ellipsoid_plot_data,
}


def _parse_plot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dex-hand tactile STL sampling grids."
    )
    parser.add_argument(
        "--patches",
        nargs="+",
        default=list(DEFAULT_PLOT_PATCHES),
        help="Skin mesh names without .STL, e.g. skin_0_0_p skin_0_2_p skin_palm_p.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_DEX_HAND_MESH_DIR,
        help="Directory containing dex-hand skin STL files.",
    )
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--surface-alpha", type=float, default=0.32)
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args()


def _patch_title(mesh_name: str, kind: str) -> str:
    rows, cols, _ = dex_hand_patch_info()[mesh_name]
    if kind == "mesh-uv":
        return f"Palm pad: {rows} x {cols}"
    if kind == "fingertip-ellipsoid":
        return f"{mesh_name}: fingertip {rows} x {cols}"
    return f"{mesh_name}: segment {rows} x {cols}"


def _fit_surface_style(kind: str) -> tuple[str, float]:
    if kind == "mesh-uv":
        return "tomato", 0.32
    if kind == "fingertip-ellipsoid":
        return "gold", 0.38
    return "deepskyblue", 0.42


def _set_equal_axes(ax, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.55 * np.max(upper - lower)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _draw_grid(ax, samples: np.ndarray, rows: int, cols: int) -> None:
    grid = samples.reshape(rows, cols, 3)
    for row in range(rows):
        ax.plot(
            grid[row, :, 0],
            grid[row, :, 1],
            grid[row, :, 2],
            color="black",
            linewidth=1.0,
            alpha=0.7,
        )
    for col in range(cols):
        ax.plot(
            grid[:, col, 0],
            grid[:, col, 1],
            grid[:, col, 2],
            color="dimgray",
            linewidth=0.8,
            alpha=0.55,
        )


def _plot_stl_surface(ax, triangles: np.ndarray, *, alpha: float) -> None:
    flat_vertices = triangles.reshape(-1, 3)
    tri_indices = np.arange(flat_vertices.shape[0], dtype=np.int32).reshape(-1, 3)
    ax.plot_trisurf(
        flat_vertices[:, 0],
        flat_vertices[:, 1],
        flat_vertices[:, 2],
        triangles=tri_indices,
        color="silver",
        alpha=alpha,
        linewidth=0.08,
        edgecolor="gray",
        shade=True,
        antialiased=True,
    )


def _plot_surface_array(ax, surface: np.ndarray, *, color: str, alpha: float) -> None:
    ax.plot_surface(
        surface[..., 0],
        surface[..., 1],
        surface[..., 2],
        color=color,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
        shade=False,
    )


def _plot_data_for_kind(
    stl_path: Path,
    mesh_name: str,
    rows: int,
    cols: int,
    kind: str,
) -> PatchPlotData:
    fit_fn = FIT_SURFACE_FUNCTIONS[kind]
    if fit_fn is None:
        return patch_mesh_uv_plot_data(stl_path, mesh_name, rows, cols)
    if kind == "fingertip-ellipsoid":
        return patch_fingertip_ellipsoid_plot_data(stl_path, mesh_name, rows, cols)
    return patch_plot_data(
        stl_path,
        mesh_name,
        rows,
        cols,
        GRID_POINT_FUNCTIONS[kind],
        fit_fn,
    )


def _plot_patch(
    ax,
    mesh_name: str,
    *,
    mesh_dir: Path,
    surface_alpha: float,
    point_size: float,
) -> None:
    patch_info = dex_hand_patch_info()
    if mesh_name not in patch_info:
        raise ValueError(
            f"Unknown dex-hand patch {mesh_name!r}. Known: {sorted(patch_info)}"
        )
    rows, cols, kind = patch_info[mesh_name]
    stl_path = mesh_dir / f"{mesh_name}.STL"
    plot_data = _plot_data_for_kind(stl_path, mesh_name, rows, cols, kind)

    triangles = plot_data.triangles
    vertices = triangles.reshape(-1, 3)
    samples = plot_data.samples

    _plot_stl_surface(ax, triangles, alpha=surface_alpha)
    color, alpha = _fit_surface_style(kind)
    for surface in plot_data.fit_surfaces:
        _plot_surface_array(ax, surface, color=color, alpha=alpha)

    colors = np.linspace(0.0, 1.0, rows * cols)
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        samples[:, 2],
        s=point_size,
        c=colors,
        cmap="turbo",
        edgecolors="black",
        linewidths=0.45,
        depthshade=False,
        label="taxels",
    )
    _draw_grid(ax, samples, rows, cols)

    ax.set_title(f"{_patch_title(mesh_name, kind)}\n{kind}")
    _set_equal_axes(ax, np.vstack([vertices, samples]))
    ax.view_init(elev=22, azim=-58)


def plot_tactile_sampling_grids(
    *,
    patches: tuple[str, ...] | list[str] = DEFAULT_PLOT_PATCHES,
    mesh_dir: Path = DEFAULT_DEX_HAND_MESH_DIR,
    point_size: float = 42.0,
    surface_alpha: float = 0.32,
    save: str = "",
) -> None:
    """Plot dex-hand tactile sampling grids for algorithm debugging."""
    import matplotlib.pyplot as plt

    column_count = len(patches)
    fig = plt.figure(figsize=(6.2 * column_count, 6.0))
    fig.suptitle("Dex-hand tactile sampling grids", fontsize=13)

    for plot_index, mesh_name in enumerate(patches, start=1):
        ax = fig.add_subplot(1, column_count, plot_index, projection="3d")
        _plot_patch(
            ax,
            mesh_name,
            mesh_dir=mesh_dir,
            surface_alpha=surface_alpha,
            point_size=point_size,
        )

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=220)
        print(f"Saved tactile sampling plot to {save}")
    else:
        plt.show()


def main() -> None:
    args = _parse_plot_args()
    plot_tactile_sampling_grids(
        patches=args.patches,
        mesh_dir=args.mesh_dir,
        point_size=args.point_size,
        surface_alpha=args.surface_alpha,
        save=args.save,
    )


if __name__ == "__main__":
    main()
