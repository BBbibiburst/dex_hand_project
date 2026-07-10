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

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


FINGER_SEGMENT_FIT_SUBDIVISIONS = 4
FINGERTIP_MAX_PHI = 0.5 * np.pi
FINGERTIP_THETA_ARC = 0.5 * np.pi
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
    surface_angle: float
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
        layout.append((f"skin_{finger_id}_0_p", 4, 8, "segment"))
        layout.append((f"skin_{finger_id}_1_p", 4, 8, "segment"))
        layout.append((f"skin_{finger_id}_2_p", 7, 8, "fingertip-ellipsoid"))
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
    """Regular grid fitted to the real outer shell of a finger segment."""
    triangles = read_stl_triangles(mesh_path)
    return _finger_segment_regular_surface_grid_points(triangles, rows, cols)


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
    """Surface-following grid for rounded fingertip pads.

    A local PCA/ellipsoid frame is used only to establish consistent axial
    and circumferential coordinates. Taxels are then interpolated directly
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
    triangles = _vertices_as_triangles(vertices)
    return _finger_segment_regular_surface_grid_points(
        triangles, 32, 64
    ).reshape(32, 64, 3)


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
    """Plot-data helper for the regular fingertip Bezier fit.

    The dense fitted surface is included in ``fit_surfaces`` so the offline
    visualizer renders it as a translucent overlay on top of the STL mesh.
    """
    triangles = read_stl_triangles(mesh_path)
    samples = _fingertip_ellipsoid_grid_points_from_triangles(triangles, rows, cols)
    fitted_surface = _fingertip_ellipsoid_surface_from_triangles(
        triangles,
        surface_rows=36,
        surface_cols=72,
    )
    return PatchPlotData(
        mesh_name=mesh_name,
        rows=rows,
        cols=cols,
        triangles=triangles,
        samples=samples,
        fit_surfaces=(fitted_surface,),
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

    (
        surface_center,
        surface_radius_x,
        surface_radius_y,
        surface_angle,
    ) = _fit_rotated_section_ellipse(
        surface_vertices, fit_center, section_x, section_y
    )

    surface_centered = surface_vertices - fit_center
    surface_section = np.column_stack(
        [surface_centered @ section_x, surface_centered @ section_y]
    )
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


def _fit_rotated_section_ellipse(
    points: np.ndarray,
    center: np.ndarray,
    section_x: np.ndarray,
    section_y: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    centered = points - center
    section = np.column_stack([centered @ section_x, centered @ section_y])
    result, _ = _fit_ellipse_ransac(section)
    if result is not None:
        cx, cy, radius_x, radius_y, angle = result
        return (
            np.asarray([cx, cy], dtype=np.float64),
            float(radius_x),
            float(radius_y),
            float(angle),
        )

    ellipse_center = np.median(section, axis=0)
    rel = section - ellipse_center
    radius_x = max(np.percentile(np.abs(rel[:, 0]), 95.0), 1e-9)
    radius_y = max(np.percentile(np.abs(rel[:, 1]), 95.0), 1e-9)
    return ellipse_center, radius_x, radius_y, 0.0


def _rotated_ellipse_point(
    center: np.ndarray,
    radius_x: float,
    radius_y: float,
    angle: float,
    theta: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    ellipse_u = radius_x * np.cos(theta)
    ellipse_v = radius_y * np.sin(theta)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_value = center[0] + ellipse_u * cos_angle - ellipse_v * sin_angle
    y_value = center[1] + ellipse_u * sin_angle + ellipse_v * cos_angle
    return x_value, y_value


def _ellipse_sampson_objective(params: np.ndarray, points_2d: np.ndarray) -> float:
    center_x, center_y, radius_x, radius_y, angle = params
    if radius_x <= 0.0 or radius_y <= 0.0:
        return 1e12

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    dx = points_2d[:, 0] - center_x
    dy = points_2d[:, 1] - center_y
    ellipse_u = dx * cos_angle + dy * sin_angle
    ellipse_v = -dx * sin_angle + dy * cos_angle
    residual = (ellipse_u / radius_x) ** 2 + (ellipse_v / radius_y) ** 2 - 1.0
    grad_u = 2.0 * ellipse_u / (radius_x**2)
    grad_v = 2.0 * ellipse_v / (radius_y**2)
    denom = grad_u**2 + grad_v**2 + 1e-12
    return float(np.sum((residual**2) / denom))


def _initial_ellipse_params(points_2d: np.ndarray) -> list[float]:
    center = np.median(points_2d, axis=0)
    centered = points_2d - center
    if len(points_2d) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        angle = float(np.arctan2(vh[0, 1], vh[0, 0]))
        axes = centered @ vh[:2].T
    else:
        angle = 0.0
        axes = centered
    radius_x = max(0.5 * np.ptp(axes[:, 0]), 1e-9)
    radius_y = max(0.5 * np.ptp(axes[:, 1]), 1e-9)
    if radius_x < radius_y:
        radius_x, radius_y = radius_y, radius_x
        angle += 0.5 * np.pi
    angle = (angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return [float(center[0]), float(center[1]), float(radius_x), float(radius_y), angle]


def _fit_ellipse_geometric(
    points_2d: np.ndarray,
    init_params: list[float] | None = None,
) -> tuple[float, float, float, float, float] | None:
    if len(points_2d) < 10:
        return None
    if init_params is None:
        init_params = _initial_ellipse_params(points_2d)
    if minimize is None:
        return tuple(init_params)  # type: ignore[return-value]

    ref = max(np.ptp(points_2d[:, 0]), np.ptp(points_2d[:, 1]), 1e-9)
    bounds = [
        (init_params[0] - ref, init_params[0] + ref),
        (init_params[1] - ref, init_params[1] + ref),
        (ref * 0.02, ref * 5.0),
        (ref * 0.02, ref * 5.0),
        (-0.5 * np.pi, 0.5 * np.pi),
    ]
    result = minimize(
        lambda params: _ellipse_sampson_objective(params, points_2d),
        init_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 180, "ftol": 1e-11},
    )
    if not result.success and result.fun > 1e6:
        return None

    center_x, center_y, radius_x, radius_y, angle = map(float, result.x)
    if radius_x < radius_y:
        radius_x, radius_y = radius_y, radius_x
        angle += 0.5 * np.pi
    angle = (angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return center_x, center_y, radius_x, radius_y, angle


def _ellipse_point_distances(
    points_2d: np.ndarray,
    params: tuple[float, float, float, float, float],
) -> np.ndarray:
    center_x, center_y, radius_x, radius_y, angle = params
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    dx = points_2d[:, 0] - center_x
    dy = points_2d[:, 1] - center_y
    ellipse_u = dx * cos_angle + dy * sin_angle
    ellipse_v = -dx * sin_angle + dy * cos_angle
    normalized_radius = np.sqrt((ellipse_u / radius_x) ** 2 + (ellipse_v / radius_y) ** 2)
    normalized_radius = np.maximum(normalized_radius, 1e-9)
    euclidean_radius = np.sqrt(ellipse_u**2 + ellipse_v**2)
    return np.abs(normalized_radius - 1.0) * euclidean_radius / normalized_radius


def _fit_ellipse_ransac(
    points_2d: np.ndarray,
    *,
    iterations: int = 32,
    inlier_tol: float = 0.10,
    min_inliers: int = 15,
) -> tuple[tuple[float, float, float, float, float] | None, np.ndarray]:
    if len(points_2d) < min_inliers:
        result = _fit_ellipse_geometric(points_2d)
        return result, np.ones(len(points_2d), dtype=bool)

    ref = max(np.ptp(points_2d[:, 0]), np.ptp(points_2d[:, 1]), 1e-9)
    tolerance = inlier_tol * ref
    rng = np.random.default_rng(42)
    best_result = None
    best_inliers = np.zeros(len(points_2d), dtype=bool)
    best_count = 0

    for _ in range(iterations):
        sample_size = int(
            rng.integers(
                max(min_inliers, len(points_2d) // 4),
                max(min_inliers + 1, len(points_2d) // 2),
            )
        )
        sample_indices = rng.choice(
            len(points_2d),
            size=min(sample_size, len(points_2d)),
            replace=False,
        )
        result = _fit_ellipse_geometric(points_2d[sample_indices])
        if result is None:
            continue
        inliers = _ellipse_point_distances(points_2d, result) < tolerance
        count = int(inliers.sum())
        if count > best_count:
            best_result = result
            best_inliers = inliers
            best_count = count

    if best_result is None or best_count < min_inliers:
        best_result = _fit_ellipse_geometric(points_2d)
        best_inliers = np.ones(len(points_2d), dtype=bool)
    elif best_inliers.any():
        refined = _fit_ellipse_geometric(points_2d[best_inliers], list(best_result))
        if refined is not None:
            best_result = refined

    return best_result, best_inliers


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
        return float(
            np.std(local @ section_x)
            + np.std(local @ section_y_probe)
        )

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
            np.percentile(section_u[mask], 2.0)
            + np.percentile(section_u[mask], 98.0)
        )
        centre_v[idx] = 0.5 * (
            np.percentile(section_v[mask], 2.0)
            + np.percentile(section_v[mask], 98.0)
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
    for start_u, end_u in zip(
        support_axial_edges[:-1], support_axial_edges[1:]
    ):
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
    unwrapped_angle = (
        arc_mid
        + (envelope_angle - arc_mid + np.pi) % (2.0 * np.pi)
        - np.pi
    )
    arc_margin = np.deg2rad(8.0)
    in_arc = (
        (unwrapped_angle >= arc_start - arc_margin)
        & (unwrapped_angle <= arc_end + arc_margin)
    )
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
    row_values = axial_start + (axial_tip - axial_start) * (
        1.0 - (1.0 - row_parameter) ** 1.45
    )
    row_edges = np.asarray([axial_start, axial_tip], dtype=np.float64)
    col_values = arc_start + (
        np.arange(cols, dtype=np.float64) + 0.5
    ) * (arc_end - arc_start) / cols

    result: list[np.ndarray] = []
    axial_scale = max(float(axial_tip - axial_start), 1e-9)
    angle_scale = max(float(arc_end - arc_start), 1e-9)

    # Dedicated distal-cap candidates.  The swept-shell angular envelope gets
    # sparse where the U-shaped wall closes, so relying on it alone truncates
    # the surface before the physical nose.
    all_unwrapped_angle = (
        arc_mid + (angle - arc_mid + np.pi) % (2.0 * np.pi) - np.pi
    )
    tip_threshold = float(np.percentile(axial, 97.5))
    tip_candidate_mask = (
        (axial >= tip_threshold)
        & (all_unwrapped_angle >= arc_start - arc_margin)
        & (all_unwrapped_angle <= arc_end + arc_margin)
    )
    tip_points = points[tip_candidate_mask]
    tip_axial = axial[tip_candidate_mask]
    tip_angle = all_unwrapped_angle[tip_candidate_mask]
    tip_radius = radius[tip_candidate_mask]

    for row_index, target_axial in enumerate(row_values):
        for target_angle in col_values:
            if row_index == rows - 1 and len(tip_points) >= cols:
                angular_error = np.abs(
                    (tip_angle - target_angle + np.pi)
                    % (2.0 * np.pi)
                    - np.pi
                ) / angle_scale
                axial_error = (axial_tip - tip_axial) / axial_scale
                score = (angular_error / 0.055) ** 2 + (axial_error / 0.035) ** 2
                nearby = np.argsort(score)[: max(12, min(48, len(score)))]
                # Prefer the front-most samples among angularly compatible
                # candidates, while retaining lateral variation across cols.
                best_pool = nearby[
                    tip_axial[nearby] >= np.percentile(tip_axial[nearby], 70.0)
                ]
                if len(best_pool) == 0:
                    best_pool = nearby
                best = best_pool[int(np.argmin(score[best_pool]))]
                result.append(tip_points[best])
                continue
            axial_distance = (
                np.abs(envelope_axial - target_axial) / axial_scale
            )
            angular_distance = (
                np.abs(
                    (envelope_angle - target_angle + np.pi)
                    % (2.0 * np.pi)
                    - np.pi
                )
                / angle_scale
            )

            local_mask = (
                (axial_distance <= 0.11)
                & (angular_distance <= 0.11)
            )
            local_indices = np.flatnonzero(local_mask)
            if len(local_indices) < 4:
                score = (
                    (axial_distance / 0.065) ** 2
                    + (angular_distance / 0.065) ** 2
                )
                local_indices = np.argsort(score)[:24]

            local_radii = envelope_radius[local_indices]
            radial_threshold = np.percentile(local_radii, 65.0)
            outer_local = local_indices[local_radii >= radial_threshold]
            if len(outer_local) == 0:
                outer_local = local_indices

            score = (
                (axial_distance[outer_local] / 0.060) ** 2
                + (angular_distance[outer_local] / 0.060) ** 2
            )
            best = outer_local[int(np.argmin(score))]
            result.append(envelope_points[best])

    return np.asarray(result, dtype=np.float64)





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
    ai = np.clip(((axial - axial_low) / axial_span * axial_bins).astype(np.int32), 0, axial_bins - 1)
    ti = np.clip(((theta - arc_start) / angle_span * angle_bins).astype(np.int32), 0, angle_bins - 1)
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


def _bernstein_basis(values: np.ndarray, degree: int) -> np.ndarray:
    """Return Bernstein basis values with shape (N, degree + 1)."""
    from math import comb

    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    basis = np.empty((len(values), degree + 1), dtype=np.float64)
    one_minus = 1.0 - values
    for index in range(degree + 1):
        basis[:, index] = (
            comb(degree, index)
            * values**index
            * one_minus ** (degree - index)
        )
    return basis


def _fit_bezier_surface(
    samples: np.ndarray,
    sample_rows: int,
    sample_cols: int,
    *,
    degree_u: int = 5,
    degree_v: int = 7,
    regularization: float = 2.0e-5,
    include_u_endpoints: bool = False,
    boundary_weight: float = 1.0,
) -> np.ndarray:
    """Fit a smooth tensor-product Bezier surface to a regular sample grid.

    ``samples`` may be slightly noisy or uneven because each source sample was
    selected from the STL shell.  The lower-dimensional Bezier control net
    removes that nearest-neighbour jitter while preserving the global U-shaped
    fingertip geometry.
    """
    grid = np.asarray(samples, dtype=np.float64).reshape(sample_rows, sample_cols, 3)
    if include_u_endpoints:
        u_values = np.linspace(0.0, 1.0, sample_rows, dtype=np.float64)
    else:
        u_values = (np.arange(sample_rows, dtype=np.float64) + 0.5) / sample_rows
    v_values = (np.arange(sample_cols, dtype=np.float64) + 0.5) / sample_cols
    basis_u = _bernstein_basis(u_values, degree_u)
    basis_v = _bernstein_basis(v_values, degree_v)

    design = np.einsum("ri,cj->rcij", basis_u, basis_v).reshape(
        sample_rows * sample_cols,
        (degree_u + 1) * (degree_v + 1),
    )
    targets = grid.reshape(-1, 3)

    # Give the longitudinal boundary rows extra weight for fingertip fits.
    # Without this, regularization tends to pull the distal boundary inward
    # and visibly flatten/truncate the nose.
    weights = np.ones(sample_rows * sample_cols, dtype=np.float64)
    if include_u_endpoints and boundary_weight > 1.0:
        weights[:sample_cols] *= boundary_weight
        weights[-sample_cols:] *= boundary_weight
    sqrt_weights = np.sqrt(weights)[:, None]
    weighted_design = design * sqrt_weights
    weighted_targets = targets * sqrt_weights

    # Mild Tikhonov regularization is enough because Bernstein bases are
    # already smooth and non-oscillatory.  Scale it to the data matrix so the
    # behavior is insensitive to mesh units and sampling resolution.
    gram = weighted_design.T @ weighted_design
    ridge = regularization * max(float(np.trace(gram) / len(gram)), 1e-12)
    controls = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64),
        weighted_design.T @ weighted_targets,
    )
    return controls.reshape(degree_u + 1, degree_v + 1, 3)


def _evaluate_bezier_surface(
    controls: np.ndarray,
    u_values: np.ndarray,
    v_values: np.ndarray,
) -> np.ndarray:
    degree_u = controls.shape[0] - 1
    degree_v = controls.shape[1] - 1
    basis_u = _bernstein_basis(u_values, degree_u)
    basis_v = _bernstein_basis(v_values, degree_v)
    surface = np.einsum("ri,cj,ijk->rck", basis_u, basis_v, controls)
    return surface.reshape(-1, 3)


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

    # Keep the first row slightly inside the base, but place the final row on
    # the distal boundary so the taxel grid and preview both reach the nose.
    if rows == 1:
        u_values = np.asarray([0.5], dtype=np.float64)
    else:
        u_values = np.linspace(0.06, 1.0, rows, dtype=np.float64)
    v_values = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    return _evaluate_bezier_surface(controls, u_values, v_values)


def _fingertip_ellipsoid_surface_from_triangles(
    triangles: np.ndarray,
    *,
    surface_rows: int = 28,
    surface_cols: int = 48,
) -> np.ndarray:
    dense_rows = max(16, surface_rows // 2)
    dense_cols = max(32, surface_cols // 2)
    raw_samples = _fingertip_swept_shell_raw_grid_points(
        triangles, dense_rows, dense_cols
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
    u_values = np.linspace(0.0, 1.0, surface_rows, dtype=np.float64)
    v_values = np.linspace(0.0, 1.0, surface_cols, dtype=np.float64)
    return _evaluate_bezier_surface(controls, u_values, v_values).reshape(
        surface_rows, surface_cols, 3
    )


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
    if FINGERTIP_MIRROR_ACROSS_XY:
        outer_points = outer_points.copy()
        outer_points[:, 2] = 2.0 * center[2] - outer_points[:, 2]
    outer_centered = outer_points - center

    u = outer_centered @ axis
    v = outer_centered @ section_x
    w = outer_centered @ section_y

    a = max(tip_u - base_u, _robust_high_abs_mean(u), 1e-9)
    b = max(_robust_high_abs_mean(v), 1e-9)
    c = max(_robust_high_abs_mean(w), 1e-9)

    un = np.clip(u / a, -1.0, 1.0)
    phi = np.arccos(un)
    theta = np.mod(np.arctan2(w / c, v / b), 2.0 * np.pi)

    phi_hi = min(float(np.percentile(phi, 96.0)), float(FINGERTIP_MAX_PHI))
    arc_start, arc_end = _axis_ray_angle_arc_fixed_width(
        u,
        v,
        w,
        b,
        c,
        width=FINGERTIP_THETA_ARC,
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
        "outer_theta": theta,
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


def _fingertip_points_from_fit(
    fit: dict[str, np.ndarray | float],
    phi_values: np.ndarray,
    theta_values: np.ndarray,
) -> np.ndarray:
    center = np.asarray(fit["center"], dtype=np.float64)
    axis = np.asarray(fit["axis"], dtype=np.float64)
    section_x = np.asarray(fit["section_x"], dtype=np.float64)
    section_y = np.asarray(fit["section_y"], dtype=np.float64)
    a = float(fit["a"])
    b = float(fit["b"])
    c = float(fit["c"])

    points: list[np.ndarray] = []
    for phi_value in phi_values:
        sin_phi = np.sin(phi_value)
        for theta in theta_values:
            point = (
                center
                + a * np.cos(phi_value) * axis
                + b * sin_phi * np.cos(theta) * section_x
                + c * sin_phi * np.sin(theta) * section_y
            )
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


def _axis_ray_angle_arc_fixed_width(
    axial: np.ndarray,
    section_x_values: np.ndarray,
    section_y_values: np.ndarray,
    radius_x: float,
    radius_y: float,
    *,
    width: float,
    bins: int = 96,
    axial_slices: int = 7,
) -> tuple[float, float]:
    section_x_norm = section_x_values / max(radius_x, 1e-9)
    section_y_norm = section_y_values / max(radius_y, 1e-9)
    radial_norm = np.sqrt(section_x_norm**2 + section_y_norm**2)
    angles = np.mod(np.arctan2(section_y_norm, section_x_norm), 2.0 * np.pi)

    target_width = min(float(width), 2.0 * np.pi)
    if len(angles) == 0:
        return 0.0, target_width

    axial_low, axial_high = np.percentile(axial, [8.0, 92.0])
    radial_floor = np.percentile(radial_norm, 35.0)
    usable = (
        (axial >= axial_low)
        & (axial <= axial_high)
        & (radial_norm >= radial_floor)
    )
    if int(usable.sum()) < 12:
        usable = radial_norm >= radial_floor
    if int(usable.sum()) < 12:
        usable = np.ones_like(radial_norm, dtype=bool)

    bin_width = 2.0 * np.pi / bins
    support = np.zeros(bins, dtype=np.float64)
    axial_edges = np.linspace(axial_low, axial_high, axial_slices + 1)
    for start_u, end_u in zip(axial_edges[:-1], axial_edges[1:]):
        slice_mask = usable & (axial >= start_u) & (axial <= end_u)
        if int(slice_mask.sum()) < 4:
            continue
        hist, _ = np.histogram(angles[slice_mask], bins=bins, range=(0.0, 2.0 * np.pi))
        # Count ray directions per axial slice, not raw point density, so one
        # dense local patch does not dominate the whole angle range.
        support += (hist > 0).astype(np.float64)

    if not np.any(support):
        hist, _ = np.histogram(angles[usable], bins=bins, range=(0.0, 2.0 * np.pi))
        support = hist.astype(np.float64)
    if not np.any(support):
        return 0.0, target_width

    support = (
        0.25 * np.roll(support, 1)
        + 0.5 * support
        + 0.25 * np.roll(support, -1)
    )
    window_bins = max(1, min(bins, int(round(target_width / bin_width))))
    doubled = np.concatenate([support, support])
    prefix = np.concatenate([[0.0], np.cumsum(doubled)])
    scores = prefix[window_bins : window_bins + bins] - prefix[:bins]
    start_bin = int(np.argmax(scores))
    arc_start = start_bin * bin_width
    return arc_start, arc_start + window_bins * bin_width


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


def _robust_high_abs_mean(values: np.ndarray) -> float:
    abs_values = np.abs(np.asarray(values, dtype=np.float64))
    if len(abs_values) == 0:
        return 0.0
    low, high = np.percentile(abs_values, [55.0, 95.0])
    band = abs_values[(abs_values >= low) & (abs_values <= high)]
    if len(band) == 0:
        band = abs_values
    return float(np.mean(band))


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
        return f"{mesh_name}: fingertip {rows} x {cols} (Bezier fit)"
    return f"{mesh_name}: segment {rows} x {cols} (Bezier shell fit)"


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
