"""Public tactile surface-fitting API."""

from source.sensors.tactile._surface_fitting import (
    PATCH_FITTERS,
    dex_hand_patch_info,
    finger_segment_grid_points,
    fingertip_ellipsoid_grid_points,
    mesh_uv_grid_points,
    palm_grid_points,
    read_stl_triangles,
    read_stl_vertices,
)

__all__ = [
    "PATCH_FITTERS",
    "dex_hand_patch_info",
    "finger_segment_grid_points",
    "fingertip_ellipsoid_grid_points",
    "mesh_uv_grid_points",
    "palm_grid_points",
    "read_stl_triangles",
    "read_stl_vertices",
]
