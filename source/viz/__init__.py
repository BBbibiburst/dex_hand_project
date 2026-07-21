"""Visualization helpers with lazy optional-dependency imports."""

from __future__ import annotations

_OVERLAY_EXPORTS = {
    "clear_markers",
    "draw_ellipse_marker",
    "draw_label",
    "draw_line_marker",
    "draw_pose_frame",
    "draw_sphere_marker",
    "draw_stats_label",
    "format_stats",
}

__all__ = [
    *_OVERLAY_EXPORTS,
    "render_grasp_benchmark_report",
    "plot_tactile_sampling_grids",
]


def __getattr__(name: str):
    if name in _OVERLAY_EXPORTS:
        from source.viz import overlays

        return getattr(overlays, name)
    if name == "render_grasp_benchmark_report":
        from source.viz.grasp_benchmark import render_grasp_benchmark_report

        return render_grasp_benchmark_report
    if name == "plot_tactile_sampling_grids":
        from source.viz.tactile import plot_tactile_sampling_grids

        return plot_tactile_sampling_grids
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
