# -*- coding: utf-8 -*-
"""Plot tactile sampling grids for dex-hand skin STL meshes.

This tool is intentionally dex-hand specific. It uses the DexHand tactile
implementation directly because plotting fitted skin surfaces is a debugging
aid for that sensor model, not a robot-agnostic framework operation.
"""

from __future__ import annotations

import argparse

import numpy as np

from source.demos.common import add_robot_config_args, load_demo_robot_config, require_hand
from source.environments.assets import DEX_HAND_MESH_DIR
from source.sensors.tactile import _surface_fitting as fit
from source.sensors.tactile.dex_hand import DEX_HAND_PATCH_LAYOUT


DEFAULT_PATCHES = ("skin_0_0_p", "skin_0_2_p", "skin_palm_p")

_GRID_FN = {
    "segment": fit.finger_segment_grid_points,
    "mesh-uv": fit.mesh_uv_grid_points,
    "rbf-outer": fit.freeform_rbf_outer_grid_points,
    "fingertip-ellipsoid": fit.fingertip_ellipsoid_grid_points,
}
_FIT_FN = {
    "segment": fit.finger_segment_fit_surface,
    "mesh-uv": None,
    "rbf-outer": fit.patch_freeform_rbf_outer_plot_data,
    "fingertip-ellipsoid": fit.patch_fingertip_ellipsoid_plot_data,
}
_PATCH_INFO = {
    mesh_name: (rows, cols, kind) for mesh_name, rows, cols, kind in DEX_HAND_PATCH_LAYOUT
}


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
    parser.add_argument(
        "--strategy",
        choices=(
            "fit",
            "mesh-uv",
            "rbf-outer",
            "fingertip-ellipsoid",
            "compare-all",
            "compare-fingertip",
        ),
        default="fit",
        help=(
            "Sampling strategy to plot. 'fit' means the current dex-hand "
            "per-patch strategy; 'compare-all' draws fit, mesh-uv, and rbf-outer; "
            "'compare-fingertip' draws current fingertip fit vs ellipsoid-cap."
        ),
    )
    parser.add_argument("--save", type=str, default="")
    add_robot_config_args(
        parser,
        include_device_overrides=False,
        include_tactile_toggle=False,
    )
    return parser.parse_args()


def _patch_title(mesh_name: str, kind: str) -> str:
    rows, cols, _ = _PATCH_INFO[mesh_name]
    if kind == "mesh-uv":
        return f"Palm pad: {rows} x {cols}"
    if kind in ("rbf-outer", "fingertip-ellipsoid"):
        return f"{mesh_name}: fingertip {rows} x {cols}"
    return f"{mesh_name}: segment {rows} x {cols}"


def _fit_surface_style(kind: str) -> tuple[str, float]:
    if kind == "mesh-uv":
        return "tomato", 0.32
    if kind in ("rbf-outer", "fingertip-ellipsoid"):
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
            grid[row, :, 0], grid[row, :, 1], grid[row, :, 2],
            color="black", linewidth=1.0, alpha=0.7,
        )
    for col in range(cols):
        ax.plot(
            grid[:, col, 0], grid[:, col, 1], grid[:, col, 2],
            color="dimgray", linewidth=0.8, alpha=0.55,
        )


def _plot_stl_surface(ax, triangles: np.ndarray, *, alpha: float) -> None:
    flat_vertices = triangles.reshape(-1, 3)
    tri_indices = np.arange(flat_vertices.shape[0], dtype=np.int32).reshape(-1, 3)
    ax.plot_trisurf(
        flat_vertices[:, 0], flat_vertices[:, 1], flat_vertices[:, 2],
        triangles=tri_indices, color="silver", alpha=alpha,
        linewidth=0.08, edgecolor="gray", shade=True, antialiased=True,
    )


def _plot_surface_array(ax, surface: np.ndarray, *, color: str, alpha: float) -> None:
    ax.plot_surface(
        surface[..., 0], surface[..., 1], surface[..., 2],
        color=color, alpha=alpha, linewidth=0, antialiased=True, shade=False,
    )


def _plot_patch(
    ax,
    mesh_name: str,
    *,
    surface_alpha: float,
    point_size: float,
    strategy: str,
) -> None:
    if mesh_name not in _PATCH_INFO:
        raise ValueError(
            f"Unknown dex-hand patch {mesh_name!r}. Known: {sorted(_PATCH_INFO)}"
        )
    rows, cols, kind = _PATCH_INFO[mesh_name]
    stl_path = DEX_HAND_MESH_DIR / f"{mesh_name}.STL"
    if strategy == "fit":
        fit_fn = _FIT_FN[kind]
        if fit_fn is None:
            plot_data = fit.patch_mesh_uv_plot_data(stl_path, mesh_name, rows, cols)
        elif kind in ("rbf-outer", "fingertip-ellipsoid"):
            plot_data = fit_fn(stl_path, mesh_name, rows, cols)
        else:
            plot_data = fit.patch_plot_data(
                stl_path, mesh_name, rows, cols, _GRID_FN[kind], fit_fn
            )
    elif strategy == "mesh-uv":
        plot_data = fit.patch_mesh_uv_plot_data(stl_path, mesh_name, rows, cols)
    elif strategy == "rbf-outer":
        plot_data = fit.patch_freeform_rbf_outer_plot_data(stl_path, mesh_name, rows, cols)
    elif strategy == "fingertip-ellipsoid":
        plot_data = fit.patch_fingertip_ellipsoid_plot_data(
            stl_path, mesh_name, rows, cols
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}.")

    triangles = plot_data.triangles
    vertices = triangles.reshape(-1, 3)
    samples = plot_data.samples

    _plot_stl_surface(ax, triangles, alpha=surface_alpha)
    if strategy == "fit":
        color, alpha = _fit_surface_style(kind)
    elif strategy == "rbf-outer":
        color, alpha = "mediumseagreen", 0.32
    elif strategy == "fingertip-ellipsoid":
        color, alpha = "gold", 0.38
    else:
        color, alpha = "mediumseagreen", 0.0
    for surface in plot_data.fit_surfaces:
        _plot_surface_array(ax, surface, color=color, alpha=alpha)

    colors = np.linspace(0.0, 1.0, rows * cols)
    ax.scatter(
        samples[:, 0], samples[:, 1], samples[:, 2],
        s=point_size, c=colors, cmap="turbo",
        edgecolors="black", linewidths=0.45, depthshade=False, label="taxels",
    )
    _draw_grid(ax, samples, rows, cols)

    title_suffix = kind if strategy == "fit" else strategy
    ax.set_title(f"{_patch_title(mesh_name, kind)}\n{title_suffix}")
    _set_equal_axes(ax, np.vstack([vertices, samples]))
    ax.view_init(elev=22, azim=-58)


def main() -> None:
    args = _parse_args()
    config = load_demo_robot_config(args)
    require_hand(config, "dex_hand", demo_name="tactile_sampling_plot")

    import matplotlib.pyplot as plt

    if args.strategy == "compare-all":
        strategies = ("fit", "mesh-uv", "rbf-outer")
    elif args.strategy == "compare-fingertip":
        strategies = ("fit", "fingertip-ellipsoid")
    else:
        strategies = (args.strategy,)
    patch_count = len(args.patches)
    column_count = patch_count * len(strategies)
    fig = plt.figure(figsize=(6.2 * column_count, 6.0))
    fig.suptitle("Dex-hand tactile sampling grids", fontsize=13)

    plot_index = 1
    for mesh_name in args.patches:
        for strategy in strategies:
            ax = fig.add_subplot(1, column_count, plot_index, projection="3d")
            _plot_patch(
                ax,
                mesh_name,
                surface_alpha=args.surface_alpha,
                point_size=args.point_size,
                strategy=strategy,
            )
            plot_index += 1

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=220)
        print(f"Saved tactile sampling plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
