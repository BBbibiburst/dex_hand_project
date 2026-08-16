"""Robust 2D ellipse fitting used by segment-surface reconstruction."""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

def _rotated_ellipse_point(
    center: np.ndarray,
    radius_x: float,
    radius_y: float,
    angle: float,
    theta: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    ellipse_u = radius_x * np.cos(theta)
    ellipse_v = radius_y * np.sin(theta)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_value = center[0] + ellipse_u * cos_angle - ellipse_v * sin_angle
    y_value = center[1] + ellipse_u * sin_angle + ellipse_v * cos_angle
    return x_value, y_value


def _ellipse_sampson_objective(params: np.ndarray, points_2d: np.ndarray) -> float:
    center_x, center_y, radius_x, radius_y, angle = params
    if radius_x <= 0.0 or radius_y <= 0.0:
        return 1e12

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    dx = points_2d[:, 0] - center_x
    dy = points_2d[:, 1] - center_y
    ellipse_u = dx * cos_angle + dy * sin_angle
    ellipse_v = -dx * sin_angle + dy * cos_angle
    residual = (ellipse_u / radius_x) ** 2 + (ellipse_v / radius_y) ** 2 - 1.0
    grad_u = 2.0 * ellipse_u / (radius_x**2)
    grad_v = 2.0 * ellipse_v / (radius_y**2)
    denom = grad_u**2 + grad_v**2 + 1e-12
    return float(np.sum((residual**2) / denom))


def _initial_ellipse_params(points_2d: np.ndarray) -> list[float]:
    center = np.median(points_2d, axis=0)
    centered = points_2d - center
    if len(points_2d) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        angle = float(np.arctan2(vh[0, 1], vh[0, 0]))
        axes = centered @ vh[:2].T
    else:
        angle = 0.0
        axes = centered
    radius_x = max(0.5 * np.ptp(axes[:, 0]), 1e-9)
    radius_y = max(0.5 * np.ptp(axes[:, 1]), 1e-9)
    if radius_x < radius_y:
        radius_x, radius_y = radius_y, radius_x
        angle += 0.5 * np.pi
    angle = (angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return [float(center[0]), float(center[1]), float(radius_x), float(radius_y), angle]


def _fit_ellipse_geometric(
    points_2d: np.ndarray,
    init_params: list[float] | None = None,
) -> tuple[float, float, float, float, float] | None:
    if len(points_2d) < 10:
        return None
    if init_params is None:
        init_params = _initial_ellipse_params(points_2d)
    if minimize is None:
        return tuple(init_params)  # type: ignore[return-value]

    ref = max(np.ptp(points_2d[:, 0]), np.ptp(points_2d[:, 1]), 1e-9)
    bounds = [
        (init_params[0] - ref, init_params[0] + ref),
        (init_params[1] - ref, init_params[1] + ref),
        (ref * 0.02, ref * 5.0),
        (ref * 0.02, ref * 5.0),
        (-0.5 * np.pi, 0.5 * np.pi),
    ]
    result = minimize(
        lambda params: _ellipse_sampson_objective(params, points_2d),
        init_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 180, "ftol": 1e-11},
    )
    if not result.success and result.fun > 1e6:
        return None

    center_x, center_y, radius_x, radius_y, angle = map(float, result.x)
    if radius_x < radius_y:
        radius_x, radius_y = radius_y, radius_x
        angle += 0.5 * np.pi
    angle = (angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return center_x, center_y, radius_x, radius_y, angle


def _ellipse_point_distances(
    points_2d: np.ndarray,
    params: tuple[float, float, float, float, float],
) -> np.ndarray:
    center_x, center_y, radius_x, radius_y, angle = params
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    dx = points_2d[:, 0] - center_x
    dy = points_2d[:, 1] - center_y
    ellipse_u = dx * cos_angle + dy * sin_angle
    ellipse_v = -dx * sin_angle + dy * cos_angle
    normalized_radius = np.sqrt((ellipse_u / radius_x) ** 2 + (ellipse_v / radius_y) ** 2)
    normalized_radius = np.maximum(normalized_radius, 1e-9)
    euclidean_radius = np.sqrt(ellipse_u**2 + ellipse_v**2)
    return np.abs(normalized_radius - 1.0) * euclidean_radius / normalized_radius


def _fit_ellipse_ransac(
    points_2d: np.ndarray,
    *,
    iterations: int = 32,
    inlier_tol: float = 0.10,
    min_inliers: int = 15,
) -> tuple[tuple[float, float, float, float, float] | None, np.ndarray]:
    if len(points_2d) < min_inliers:
        result = _fit_ellipse_geometric(points_2d)
        return result, np.ones(len(points_2d), dtype=bool)

    ref = max(np.ptp(points_2d[:, 0]), np.ptp(points_2d[:, 1]), 1e-9)
    tolerance = inlier_tol * ref
    rng = np.random.default_rng(42)
    best_result = None
    best_inliers = np.zeros(len(points_2d), dtype=bool)
    best_count = 0

    for _ in range(iterations):
        sample_size = int(
            rng.integers(
                max(min_inliers, len(points_2d) // 4),
                max(min_inliers + 1, len(points_2d) // 2),
            )
        )
        sample_indices = rng.choice(
            len(points_2d),
            size=min(sample_size, len(points_2d)),
            replace=False,
        )
        result = _fit_ellipse_geometric(points_2d[sample_indices])
        if result is None:
            continue
        inliers = _ellipse_point_distances(points_2d, result) < tolerance
        count = int(inliers.sum())
        if count > best_count:
            best_result = result
            best_inliers = inliers
            best_count = count

    if best_result is None or best_count < min_inliers:
        best_result = _fit_ellipse_geometric(points_2d)
        best_inliers = np.ones(len(points_2d), dtype=bool)
    elif best_inliers.any():
        refined = _fit_ellipse_geometric(points_2d[best_inliers], list(best_result))
        if refined is not None:
            best_result = refined

    return best_result, best_inliers


def _fit_rotated_section_ellipse(
    points: np.ndarray,
    center: np.ndarray,
    section_x: np.ndarray,
    section_y: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    centered = points - center
    section = np.column_stack([centered @ section_x, centered @ section_y])
    result, _ = _fit_ellipse_ransac(section)
    if result is not None:
        cx, cy, radius_x, radius_y, angle = result
        return (
            np.asarray([cx, cy], dtype=np.float64),
            float(radius_x),
            float(radius_y),
            float(angle),
        )

    ellipse_center = np.median(section, axis=0)
    rel = section - ellipse_center
    radius_x = max(np.percentile(np.abs(rel[:, 0]), 95.0), 1e-9)
    radius_y = max(np.percentile(np.abs(rel[:, 1]), 95.0), 1e-9)
    return ellipse_center, radius_x, radius_y, 0.0
