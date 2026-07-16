"""Shared dark dashboard styling for live Vive visualizations."""

from __future__ import annotations

import numpy as np


BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
CYAN = "#58d6f5"
ORANGE = "#f0883e"
GREEN = "#39d353"
RED = "#f85149"
WHITE = "#e6edf3"
GRAY = "#8b949e"
PURPLE = "#bc8cff"
YELLOW = "#e3b341"


def apply_theme(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "text.color": WHITE,
            "axes.labelcolor": GRAY,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.edgecolor": GRID,
            "grid.color": GRID,
            "font.family": "monospace",
        }
    )


def style_3d_axis(axis, origin, radius: float) -> None:
    origin = np.asarray(origin, dtype=float)
    axis.set_facecolor(BG)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(GRID)
    axis.grid(True, alpha=0.35)
    axis.tick_params(colors=GRAY, labelsize=7)
    axis.set_xlabel("RIGHT  +X (m)", fontsize=8, color=RED)
    axis.set_ylabel("FORWARD +Y (m)", fontsize=8, color=GREEN)
    axis.set_zlabel("UP     +Z (m)", fontsize=8, color=CYAN)
    axis.set_xlim(origin[0] - radius, origin[0] + radius)
    axis.set_ylim(origin[1] - radius, origin[1] + radius)
    axis.set_zlim(origin[2] - radius, origin[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-58)


def draw_floor(axis, origin, radius: float):
    origin = np.asarray(origin, dtype=float)
    values = np.linspace(-radius, radius, 9)
    gx, gy = np.meshgrid(origin[0] + values, origin[1] + values)
    gz = np.full_like(gx, origin[2] - radius)
    return axis.plot_surface(gx, gy, gz, color=GRAY, alpha=0.055, shade=False)


def style_panel(axis, title: str) -> None:
    axis.set_facecolor(PANEL)
    axis.set_title(title, color=WHITE, fontsize=9, fontweight="bold", pad=8)
    axis.grid(True, axis="x", alpha=0.25)
    axis.tick_params(colors=GRAY, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def update_frame_axes(axis, artists, position, rotation, length: float) -> None:
    colors = (RED, GREEN, CYAN)
    for index, (artist, color) in enumerate(zip(artists, colors)):
        endpoint = position + rotation[:, index] * length
        artist.set_data_3d(
            [position[0], endpoint[0]],
            [position[1], endpoint[1]],
            [position[2], endpoint[2]],
        )
        artist.set_color(color)
