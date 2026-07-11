"""Minimal binary and ASCII STL readers used by tactile fitting."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        triangles = np.empty((triangle_count, 3, 3), dtype=np.float64)
        offset = 84
        for triangle_index in range(triangle_count):
            offset += 12
            for vertex_index in range(3):
                triangles[triangle_index, vertex_index] = struct.unpack_from("<3f", data, offset)
                offset += 12
            offset += 2
        return triangles
    return _read_ascii_stl_triangles(data.decode("utf-8", errors="ignore"))


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    return 84 + triangle_count * 50 == len(data)


def _read_ascii_stl_triangles(text: str) -> np.ndarray:
    vertices: list[list[float]] = []
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError("STL file contains no vertices.")
    if len(vertices) % 3 != 0:
        raise ValueError("ASCII STL vertex count is not divisible by 3.")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
