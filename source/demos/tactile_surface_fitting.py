"""Offline visualization of backend-provided tactile surface sampling."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from source.demos.common import add_robot_config_args, load_demo_robot_config
from source.robots.registry import get_hand
from source.sensors.base import TactileSensorBase, TactileSurfacePlotData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tactile sampling grids supplied by the configured sensor backend."
    )
    parser.add_argument(
        "--patches",
        nargs="+",
        default=None,
        help="Patch names exposed by the backend. Defaults to its first three patches.",
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--surface-alpha", type=float, default=0.32)
    parser.add_argument("--save", type=str, default="")
    add_robot_config_args(parser, include_tactile_toggle=False)
    return parser.parse_args()


def _set_equal_axes(axis, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(1e-6, 0.55 * float(np.max(upper - lower)))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")


def _draw_grid(axis, samples: np.ndarray, rows: int, cols: int) -> None:
    grid = samples.reshape(rows, cols, 3)
    for row in range(rows):
        axis.plot(*grid[row].T, color="black", linewidth=0.8, alpha=0.65)
    for column in range(cols):
        axis.plot(*grid[:, column].T, color="dimgray", linewidth=0.65, alpha=0.5)


def _plot_stl_surface(axis, triangles: np.ndarray, *, alpha: float) -> None:
    flat = triangles.reshape(-1, 3)
    indices = np.arange(flat.shape[0], dtype=np.int32).reshape(-1, 3)
    axis.plot_trisurf(
        *flat.T,
        triangles=indices,
        color="silver",
        alpha=alpha,
        linewidth=0.05,
        edgecolor="gray",
        shade=True,
        antialiased=True,
    )


def _plot_patch(axis, data: TactileSurfacePlotData, *, point_size: float, alpha: float) -> None:
    _plot_stl_surface(axis, data.triangles, alpha=alpha)
    colors = ("deepskyblue", "gold", "tomato", "mediumseagreen")
    for index, surface in enumerate(data.fit_surfaces):
        axis.plot_surface(
            surface[..., 0],
            surface[..., 1],
            surface[..., 2],
            color=colors[index % len(colors)],
            alpha=0.38,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
    axis.scatter(
        *data.samples.T,
        s=point_size,
        c=np.linspace(0.0, 1.0, data.rows * data.cols),
        cmap="turbo",
        edgecolors="black",
        linewidths=0.35,
        depthshade=False,
    )
    _draw_grid(axis, data.samples, data.rows, data.cols)
    axis.set_title(data.title or f"{data.name}: {data.rows} x {data.cols} ({data.kind})")
    _set_equal_axes(axis, np.vstack([data.triangles.reshape(-1, 3), data.samples]))
    axis.view_init(elev=22, azim=-58)


def plot_tactile_sampling_grids(
    sensor: TactileSensorBase,
    *,
    patches: Sequence[str] | None = None,
    point_size: float = 42.0,
    surface_alpha: float = 0.32,
    save: str = "",
) -> None:
    """Plot sampling geometry exposed by any tactile sensor backend."""
    import matplotlib.pyplot as plt

    available = tuple(sensor.surface_patch_names())
    if not available:
        raise ValueError(
            f"Tactile backend {type(sensor).__name__!r} does not expose surface sampling data."
        )
    selected = tuple(patches) if patches else tuple(sensor.default_surface_patch_names())
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown patch(es) {unknown}; available: {list(available)}.")

    figure = plt.figure(figsize=(6.2 * len(selected), 6.0))
    figure.suptitle(f"{type(sensor).__name__} tactile sampling grids", fontsize=13)
    for index, patch_name in enumerate(selected, start=1):
        axis = figure.add_subplot(1, len(selected), index, projection="3d")
        _plot_patch(
            axis,
            sensor.surface_plot_data(patch_name),
            point_size=point_size,
            alpha=surface_alpha,
        )
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=220)
        print(f"Saved tactile sampling plot to {save}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    config = load_demo_robot_config(args)
    descriptor = get_hand(str(config["hand_name"]))
    if descriptor.tactile_sensor_factory is None:
        raise ValueError(f"End effector {descriptor.name!r} does not provide tactile sensing.")
    sensor = descriptor.tactile_sensor_factory(
        args.backend or str(config.get("tactile_backend", "simple_box")),
        **dict(config.get("tactile_options") or {}),
    )
    plot_tactile_sampling_grids(
        sensor,
        patches=args.patches,
        point_size=args.point_size,
        surface_alpha=args.surface_alpha,
        save=args.save,
    )


if __name__ == "__main__":
    main()
