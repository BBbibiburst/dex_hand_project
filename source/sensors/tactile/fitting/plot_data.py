"""Geometry payloads consumed by offline tactile visualizers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from source.sensors.tactile.fitting.fingertip import (
    _fingertip_swept_shell_grid_points_from_triangles,
    _fingertip_swept_shell_surface_from_triangles,
)
from source.sensors.tactile.fitting.projection import _mesh_uv_grid_points_from_triangles
from source.sensors.tactile.fitting.stl import read_stl_triangles

@dataclass(frozen=True)
class PatchPlotData:
    mesh_name: str
    rows: int
    cols: int
    triangles: np.ndarray
    samples: np.ndarray
    fit_surfaces: tuple[np.ndarray, ...]


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
    samples = _fingertip_swept_shell_grid_points_from_triangles(triangles, rows, cols)
    fitted_surface = _fingertip_swept_shell_surface_from_triangles(
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
