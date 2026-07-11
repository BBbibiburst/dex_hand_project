"""Offline visualization CLI for dex-hand tactile surface fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from source.sensors.tactile.fitting.layout import (
    DEFAULT_DEX_HAND_MESH_DIR,
    DEFAULT_PLOT_PATCHES,
    dex_hand_patch_info,
)
from source.sensors.tactile.surface_fitting import (
    GRID_POINT_FUNCTIONS,
    PatchPlotData,
    finger_segment_fit_surface,
    patch_fingertip_ellipsoid_plot_data,
    patch_mesh_uv_plot_data,
    patch_plot_data,
)

FIT_SURFACE_FUNCTIONS = {
    "segment": finger_segment_fit_surface,
    "mesh-uv": None,
    "fingertip-ellipsoid": patch_fingertip_ellipsoid_plot_data,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dex-hand tactile STL sampling grids.")
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


def _set_equal_axes(axis, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.55 * np.max(upper - lower)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")


def _draw_grid(axis, samples: np.ndarray, rows: int, cols: int) -> None:
    grid = samples.reshape(rows, cols, 3)
    for row in range(rows):
        axis.plot(*grid[row].T, color="black", linewidth=1.0, alpha=0.7)
    for column in range(cols):
        axis.plot(*grid[:, column].T, color="dimgray", linewidth=0.8, alpha=0.55)


def _plot_stl_surface(axis, triangles: np.ndarray, *, alpha: float) -> None:
    flat_vertices = triangles.reshape(-1, 3)
    triangle_indices = np.arange(flat_vertices.shape[0], dtype=np.int32).reshape(-1, 3)
    axis.plot_trisurf(
        *flat_vertices.T,
        triangles=triangle_indices,
        color="silver",
        alpha=alpha,
        linewidth=0.08,
        edgecolor="gray",
        shade=True,
        antialiased=True,
    )


def _plot_surface_array(axis, surface: np.ndarray, *, color: str, alpha: float) -> None:
    axis.plot_surface(
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
    fit_function = FIT_SURFACE_FUNCTIONS[kind]
    if fit_function is None:
        return patch_mesh_uv_plot_data(stl_path, mesh_name, rows, cols)
    if kind == "fingertip-ellipsoid":
        return patch_fingertip_ellipsoid_plot_data(stl_path, mesh_name, rows, cols)
    return patch_plot_data(
        stl_path,
        mesh_name,
        rows,
        cols,
        GRID_POINT_FUNCTIONS[kind],
        fit_function,
    )


def _plot_patch(
    axis,
    mesh_name: str,
    *,
    mesh_dir: Path,
    surface_alpha: float,
    point_size: float,
) -> None:
    patch_info = dex_hand_patch_info()
    if mesh_name not in patch_info:
        raise ValueError(f"Unknown dex-hand patch {mesh_name!r}. Known: {sorted(patch_info)}")
    rows, cols, kind = patch_info[mesh_name]
    plot_data = _plot_data_for_kind(
        mesh_dir / f"{mesh_name}.STL", mesh_name, rows, cols, kind
    )
    vertices = plot_data.triangles.reshape(-1, 3)
    _plot_stl_surface(axis, plot_data.triangles, alpha=surface_alpha)
    color, alpha = _fit_surface_style(kind)
    for surface in plot_data.fit_surfaces:
        _plot_surface_array(axis, surface, color=color, alpha=alpha)
    axis.scatter(
        *plot_data.samples.T,
        s=point_size,
        c=np.linspace(0.0, 1.0, rows * cols),
        cmap="turbo",
        edgecolors="black",
        linewidths=0.45,
        depthshade=False,
        label="taxels",
    )
    _draw_grid(axis, plot_data.samples, rows, cols)
    axis.set_title(f"{_patch_title(mesh_name, kind)}\n{kind}")
    _set_equal_axes(axis, np.vstack([vertices, plot_data.samples]))
    axis.view_init(elev=22, azim=-58)


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

    figure = plt.figure(figsize=(6.2 * len(patches), 6.0))
    figure.suptitle("Dex-hand tactile sampling grids", fontsize=13)
    for plot_index, mesh_name in enumerate(patches, start=1):
        axis = figure.add_subplot(1, len(patches), plot_index, projection="3d")
        _plot_patch(
            axis,
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
    args = parse_args()
    plot_tactile_sampling_grids(
        patches=args.patches,
        mesh_dir=args.mesh_dir,
        point_size=args.point_size,
        surface_alpha=args.surface_alpha,
        save=args.save,
    )


if __name__ == "__main__":
    main()
