# -*- coding: utf-8 -*-
"""Plot tactile sampling grids for dex-hand skin STL meshes."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np

from source.environments.tactile_layout import _surface_grid_points
from source.environments.tactile_layout import (
    _ellipse_arc_mid_angles,
    _fit_finger_segment_surfaces,
    _linspace_midpoints,
    _occupied_angle_arc,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MESH_DIR = PROJECT_ROOT / "assets" / "grippers" / "dex_hand" / "meshes"
DEFAULT_PATCHES = ("skin_0_0_p", "skin_0_2_p", "skin_palm_p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tactile STL sampling grids.")
    parser.add_argument(
        "--patches",
        nargs="+",
        default=list(DEFAULT_PATCHES),
        help="Skin mesh names without .STL, e.g. skin_0_0_p skin_0_2_p skin_palm_p.",
    )
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--surface-alpha", type=float, default=0.32)
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args()


def _grid_shape(mesh_name: str) -> tuple[int, int]:
    if mesh_name == "skin_palm_p":
        return 7, 16
    if mesh_name.endswith("_0_p"):
        return 7, 8
    if mesh_name.endswith("_1_p") or mesh_name.endswith("_2_p"):
        return 4, 8
    raise ValueError(f"Unknown tactile skin patch type: {mesh_name!r}.")


def _patch_title(mesh_name: str) -> str:
    if mesh_name == "skin_palm_p":
        return "Palm pad: 7 x 16"
    if mesh_name.endswith("_2_p"):
        return f"{mesh_name}: fingertip 4 x 8"
    return f"{mesh_name}: segment 7/4 x 8"


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


def _read_stl_triangles(path: Path) -> np.ndarray:
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
    if len(vertices) % 3 != 0:
        raise ValueError("ASCII STL vertex count is not divisible by 3.")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


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


def _plot_fitted_surface(ax, mesh_name: str, vertices: np.ndarray) -> None:
    if mesh_name == "skin_palm_p":
        _plot_palm_fit_surface(ax, vertices)
    elif mesh_name.endswith("_2_p"):
        _plot_fingertip_fit_surface(ax, vertices)
    else:
        _plot_segment_fit_surface(ax, vertices)


def _plot_segment_fit_surface(ax, vertices: np.ndarray) -> None:
    fit = _fit_finger_segment_surfaces(vertices)
    outer_surface = _segment_surface_from_fit(
        fit,
        fit.outer_center,
        fit.outer_radius_x,
        fit.outer_radius_y,
    )
    inner_surface = _segment_surface_from_fit(
        fit,
        fit.inner_center,
        fit.inner_radius_x,
        fit.inner_radius_y,
    )
    _plot_surface_array(ax, outer_surface, color="deepskyblue", alpha=0.42)
    _plot_surface_array(ax, inner_surface, color="salmon", alpha=0.36)


def _segment_surface_from_fit(fit, center_2d, radius_x, radius_y) -> np.ndarray:
    z_values = np.linspace(fit.axial_low, fit.axial_high, 32)
    theta_values = np.linspace(fit.arc_start, fit.arc_end, 48)
    z_grid, theta_grid = np.meshgrid(z_values, theta_values, indexing="ij")
    x_grid = center_2d[0] + radius_x * np.cos(theta_grid)
    y_grid = center_2d[1] + radius_y * np.sin(theta_grid)
    return (
        fit.center
        + z_grid[..., None] * fit.axis
        + x_grid[..., None] * fit.section_x
        + y_grid[..., None] * fit.section_y
    )


def _plot_fingertip_fit_surface(ax, vertices: np.ndarray) -> None:
    center = vertices.mean(axis=0)
    centered = vertices - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis, section_x, section_y = vh[0], vh[1], vh[2]

    axial = centered @ axis
    section = np.column_stack([centered @ section_x, centered @ section_y])
    center_2d = np.median(section, axis=0)
    rel = section - center_2d
    radius_x = max(np.percentile(np.abs(rel[:, 0]), 95.0), 1e-9)
    radius_y = max(np.percentile(np.abs(rel[:, 1]), 95.0), 1e-9)
    norm_radius = np.sqrt((rel[:, 0] / radius_x) ** 2 + (rel[:, 1] / radius_y) ** 2)
    outer_mask = norm_radius >= np.percentile(norm_radius, 58.0)
    outer_axial = axial[outer_mask]
    outer_rel = rel[outer_mask]
    outer_angles = np.mod(
        np.arctan2(outer_rel[:, 1] / radius_y, outer_rel[:, 0] / radius_x),
        2.0 * np.pi,
    )
    arc_start, arc_end = _occupied_angle_arc(outer_angles)

    z_values = np.linspace(*np.percentile(axial, [7.5, 92.5]), 28)
    theta_values = _ellipse_arc_mid_angles(radius_x, radius_y, 52, arc_start, arc_end)
    local_scale = _fingertip_radial_scale(axial, norm_radius, z_values)

    z_grid, theta_grid = np.meshgrid(z_values, theta_values, indexing="ij")
    scale_grid = local_scale[:, None]
    x_grid = center_2d[0] + scale_grid * radius_x * np.cos(theta_grid)
    y_grid = center_2d[1] + scale_grid * radius_y * np.sin(theta_grid)
    surface = (
        center
        + z_grid[..., None] * axis
        + x_grid[..., None] * section_x
        + y_grid[..., None] * section_y
    )
    _plot_surface_array(ax, surface, color="gold", alpha=0.38)


def _fingertip_radial_scale(
    axial: np.ndarray,
    norm_radius: np.ndarray,
    z_values: np.ndarray,
) -> np.ndarray:
    z_span = max(np.percentile(axial, 92.5) - np.percentile(axial, 7.5), 1e-9)
    window = 0.18 * z_span
    scales = []
    for z_value in z_values:
        mask = np.abs(axial - z_value) <= window
        if int(mask.sum()) < 8:
            mask = np.ones_like(axial, dtype=bool)
        scales.append(np.percentile(norm_radius[mask], 88.0))
    return np.asarray(scales, dtype=np.float64)


def _plot_palm_fit_surface(ax, vertices: np.ndarray) -> None:
    x_value = np.percentile(vertices[:, 0], 88.0)
    z_values = np.linspace(*np.percentile(vertices[:, 2], [7.5, 92.5]), 36)
    y_values = np.linspace(*np.percentile(vertices[:, 1], [7.5, 92.5]), 24)
    z_grid, y_grid = np.meshgrid(z_values, y_values, indexing="ij")
    surface = np.stack(
        [
            np.full_like(z_grid, x_value),
            y_grid,
            z_grid,
        ],
        axis=-1,
    )
    _plot_surface_array(ax, surface, color="tomato", alpha=0.32)


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


def _plot_patch(
    ax,
    mesh_name: str,
    *,
    surface_alpha: float,
    point_size: float,
) -> None:
    rows, cols = _grid_shape(mesh_name)
    stl_path = MESH_DIR / f"{mesh_name}.STL"
    triangles = _read_stl_triangles(stl_path)
    vertices = triangles.reshape(-1, 3)
    samples = _surface_grid_points(stl_path, rows, cols, mesh_name=mesh_name)

    _plot_stl_surface(ax, triangles, alpha=surface_alpha)
    _plot_fitted_surface(ax, mesh_name, vertices)

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

    ax.set_title(_patch_title(mesh_name))
    _set_equal_axes(ax, np.vstack([vertices, samples]))
    ax.view_init(elev=22, azim=-58)


def main() -> None:
    args = _parse_args()

    import matplotlib.pyplot as plt

    patch_count = len(args.patches)
    fig = plt.figure(figsize=(6.2 * patch_count, 6.0))
    fig.suptitle("Dex-hand tactile sampling grids", fontsize=13)

    for index, mesh_name in enumerate(args.patches, start=1):
        ax = fig.add_subplot(1, patch_count, index, projection="3d")
        _plot_patch(
            ax,
            mesh_name,
            surface_alpha=args.surface_alpha,
            point_size=args.point_size,
        )

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=220)
        print(f"Saved tactile sampling plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
