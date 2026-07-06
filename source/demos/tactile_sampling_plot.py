# -*- coding: utf-8 -*-
"""Plot tactile sampling grids for dex-hand skin STL meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from source.environments.tactile_layout import tactile_patch_plot_data


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


def _plot_patch(
    ax,
    mesh_name: str,
    *,
    surface_alpha: float,
    point_size: float,
) -> None:
    stl_path = MESH_DIR / f"{mesh_name}.STL"
    plot_data = tactile_patch_plot_data(stl_path, mesh_name)
    rows, cols = plot_data.rows, plot_data.cols
    triangles = plot_data.triangles
    vertices = triangles.reshape(-1, 3)
    samples = plot_data.samples

    _plot_stl_surface(ax, triangles, alpha=surface_alpha)
    color, alpha = _fit_surface_style(mesh_name)
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

    ax.set_title(_patch_title(mesh_name))
    _set_equal_axes(ax, np.vstack([vertices, samples]))
    ax.view_init(elev=22, azim=-58)


def _fit_surface_style(mesh_name: str) -> tuple[str, float]:
    if mesh_name == "skin_palm_p":
        return "tomato", 0.32
    if mesh_name.endswith("_2_p"):
        return "gold", 0.38
    return "deepskyblue", 0.42


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
