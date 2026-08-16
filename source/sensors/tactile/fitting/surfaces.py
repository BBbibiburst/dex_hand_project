"""Dex-hand tactile surface-fitting facade.

Implementation is split by geometry responsibility: primitives, ellipse,
segment, projection, fingertip, and plot-data preparation.
"""

from source.sensors.tactile.fitting.fingertip import fingertip_ellipsoid_grid_points
from source.sensors.tactile.fitting.layout import (
    DEFAULT_DEX_HAND_MESH_DIR, DEFAULT_PLOT_PATCHES, DEX_HAND_PATCH_LAYOUT,
    dex_hand_patch_info, dex_hand_patch_layout,
)
from source.sensors.tactile.fitting.plot_data import (
    PatchPlotData, patch_fingertip_ellipsoid_plot_data, patch_mesh_uv_plot_data, patch_plot_data,
)
from source.sensors.tactile.fitting.projection import mesh_uv_grid_points
from source.sensors.tactile.fitting.segment import finger_segment_fit_surface, finger_segment_grid_points
from source.sensors.tactile.fitting.stl import read_stl_triangles

fingertip_swept_shell_grid_points = fingertip_ellipsoid_grid_points

GRID_POINT_FUNCTIONS = {
    "segment": finger_segment_grid_points,
    "mesh-uv": mesh_uv_grid_points,
    "fingertip-ellipsoid": fingertip_ellipsoid_grid_points,
}

def grid_points_for_kind(kind, mesh_path, rows, cols):
    """Compute grid points for one dex-hand fitting strategy."""
    try:
        grid_fn = GRID_POINT_FUNCTIONS[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dex-hand tactile fitting strategy {kind!r}. "
            f"Known: {sorted(GRID_POINT_FUNCTIONS)}"
        ) from exc
    return grid_fn(mesh_path, rows, cols)

__all__ = [
    "DEFAULT_DEX_HAND_MESH_DIR", "DEFAULT_PLOT_PATCHES", "DEX_HAND_PATCH_LAYOUT",
    "PatchPlotData", "dex_hand_patch_info", "dex_hand_patch_layout",
    "finger_segment_fit_surface", "finger_segment_grid_points",
    "fingertip_ellipsoid_grid_points", "fingertip_swept_shell_grid_points",
    "grid_points_for_kind", "mesh_uv_grid_points", "patch_fingertip_ellipsoid_plot_data",
    "patch_mesh_uv_plot_data", "patch_plot_data", "read_stl_triangles",
]
