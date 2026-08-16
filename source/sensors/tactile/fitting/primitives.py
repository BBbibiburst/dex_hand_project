"""Low-level triangle, barycentric, and sampling primitives."""

from __future__ import annotations

import numpy as np

def _vertices_as_triangles(vertices: np.ndarray) -> np.ndarray:
    usable = (len(vertices) // 3) * 3
    if usable == 0:
        raise ValueError("STL vertices cannot form triangles.")
    return vertices[:usable].reshape(-1, 3, 3)


def _supersample_triangles(triangles: np.ndarray, *, subdivisions: int) -> np.ndarray:
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=np.float64)
    subdivisions = max(1, int(subdivisions))
    barycentric = []
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            k = subdivisions - i - j
            barycentric.append((i, j, k))
    weights = np.asarray(barycentric, dtype=np.float64) / float(subdivisions)
    return (
        triangles[:, None, 0, :] * weights[None, :, 0, None]
        + triangles[:, None, 1, :] * weights[None, :, 1, None]
        + triangles[:, None, 2, :] * weights[None, :, 2, None]
    ).reshape(-1, 3)


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0
    return normals


def _barycentric_coordinates_2d(
    triangle: np.ndarray,
    point: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray | None:
    """Return barycentric coordinates when point lies inside a 2D triangle."""

    triangle = np.asarray(triangle, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)

    a = triangle[0]
    b = triangle[1]
    c = triangle[2]

    v0 = b - a
    v1 = c - a
    v2 = point - a

    denominator = v0[0] * v1[1] - v1[0] * v0[1]

    if abs(denominator) < 1e-12:
        return None

    beta = (v2[0] * v1[1] - v1[0] * v2[1]) / denominator

    gamma = (v0[0] * v2[1] - v2[0] * v0[1]) / denominator

    alpha = 1.0 - beta - gamma

    bary = np.asarray([alpha, beta, gamma], dtype=np.float64)

    if np.all(bary >= -tolerance) and np.all(bary <= 1.0 + tolerance):
        return bary

    return None


def _barycentric_2d(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) <= 1e-14:
        distances = np.sum((triangle - point) ** 2, axis=1)
        bary = np.zeros(3, dtype=np.float64)
        bary[int(np.argmin(distances))] = 1.0
        return bary
    inv = 1.0 / denom
    beta = (v2[0] * v1[1] - v1[0] * v2[1]) * inv
    gamma = (v0[0] * v2[1] - v2[0] * v0[1]) * inv
    alpha = 1.0 - beta - gamma
    return np.asarray([alpha, beta, gamma], dtype=np.float64)


def _barycentric_interpolate(values: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return values[0] * bary[0] + values[1] * bary[1] + values[2] * bary[2]


def _occupied_angle_arc(angles: np.ndarray, *, bins: int = 96) -> tuple[float, float]:
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi))
    occupied = hist > 0
    doubled_empty = np.concatenate([~occupied, ~occupied])

    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for idx, is_empty in enumerate(doubled_empty):
        if is_empty:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_len = 0

    if best_len == 0 or best_len >= bins:
        return 0.0, 2.0 * np.pi

    bin_width = 2.0 * np.pi / bins
    start = ((best_start + best_len) % bins) * bin_width
    arc_bins = bins - min(best_len, bins)
    end = start + arc_bins * bin_width
    return start, end


def _ellipse_arc_mid_angles(
    radius_x: float, radius_y: float, count: int, start: float, end: float
) -> np.ndarray:
    if count == 1:
        return np.asarray([(start + end) * 0.5], dtype=np.float64)

    samples = np.linspace(start, end, 512, dtype=np.float64)
    speed = np.sqrt((radius_x * np.sin(samples)) ** 2 + (radius_y * np.cos(samples)) ** 2)
    cumulative = np.zeros_like(samples)
    cumulative[1:] = np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(samples))
    targets = (np.arange(count, dtype=np.float64) + 0.5) * cumulative[-1] / count
    return np.interp(targets, cumulative, samples)


def _linspace_midpoints(values: np.ndarray, count: int) -> np.ndarray:
    low, high = np.percentile(values, [7.5, 92.5])
    if count == 1:
        return np.asarray([(low + high) * 0.5], dtype=np.float64)
    edges = np.linspace(low, high, count + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])
