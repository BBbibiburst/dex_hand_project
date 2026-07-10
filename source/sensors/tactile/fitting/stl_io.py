"""STL loading entry points used by tactile fitting."""

from source.sensors.tactile._surface_fitting import read_stl_triangles, read_stl_vertices

__all__ = ["read_stl_triangles", "read_stl_vertices"]
