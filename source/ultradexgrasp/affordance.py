"""Computable underactuated-enclosure affordance metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeometryAffordance:
    extents_m: tuple[float, float, float]
    grasp_span_m: float
    sphericity: float
    axis_ratio: float
    convexity: float
    size_compatibility: float
    shape_regularity: float
    power_grasp_suitability: float
    rotational_symmetry: float
    geometry_prior: float
    eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _window_score(
    value: float,
    *,
    minimum: float,
    ideal_minimum: float,
    ideal_maximum: float,
    maximum: float,
) -> float:
    if value <= minimum or value >= maximum:
        return 0.0
    if ideal_minimum <= value <= ideal_maximum:
        return 1.0
    if value < ideal_minimum:
        return (value - minimum) / (ideal_minimum - minimum)
    return (maximum - value) / (maximum - ideal_maximum)


def _principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    _, axes = np.linalg.eigh(covariance)
    local = centered @ axes
    extents = np.ptp(local, axis=0)
    order = np.argsort(extents)
    return local[:, order], extents[order]


def _symmetry_score(points: np.ndarray, extents: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(extents)), 1e-9)
    tree = cKDTree(points)
    scores: list[float] = []
    for axis in range(3):
        rotated = points.copy()
        other = [item for item in range(3) if item != axis]
        rotated[:, other] *= -1.0
        distances, _ = tree.query(rotated, k=1)
        normalized_error = float(np.sqrt(np.mean(distances**2)) / scale)
        scores.append(float(np.exp(-35.0 * normalized_error)))
    return max(scores)


def geometry_affordance(mesh, *, scale_to_meters: float) -> GeometryAffordance:
    """Score geometry without using category names or grasp outcomes.

    The score is deliberately a prior. Adaptive contact and perturbation
    robustness must be supplied by MuJoCo before an object can enter a final
    benchmark.
    """
    scaled = mesh.copy()
    scaled.apply_scale(float(scale_to_meters))
    hull = scaled.convex_hull
    samples, _ = __import__("trimesh").sample.sample_surface(hull, 1200, seed=17)
    local, extents = _principal_frame(np.asarray(samples, dtype=np.float64))
    minimum, middle, maximum = map(float, extents)
    grasp_span = float(np.sqrt(max(minimum * middle, 0.0)))
    axis_ratio = maximum / max(minimum, 1e-9)

    area = max(float(hull.area), 1e-12)
    volume = max(abs(float(hull.volume)), 1e-12)
    sphericity = float(np.clip(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area, 0.0, 1.0))
    raw_volume = max(abs(float(scaled.volume)), 0.0)
    convexity = float(np.clip(raw_volume / volume, 0.0, 1.0))

    size = _window_score(
        grasp_span,
        minimum=0.025,
        ideal_minimum=0.035,
        ideal_maximum=0.080,
        maximum=0.100,
    )
    aspect = _window_score(
        axis_ratio,
        minimum=0.95,
        ideal_minimum=1.0,
        ideal_maximum=2.5,
        maximum=5.0,
    )
    shape = float(np.clip(0.65 * sphericity + 0.35 * aspect, 0.0, 1.0))
    # A power grasp needs two substantial transverse dimensions and a mostly
    # continuous surface. Convexity is used as a conservative collision-model
    # proxy, not as a preference for semantic object classes.
    transverse_balance = minimum / max(middle, 1e-9)
    power = float(
        np.clip(
            0.45 * size + 0.30 * transverse_balance + 0.25 * np.sqrt(convexity),
            0.0,
            1.0,
        )
    )
    symmetry = _symmetry_score(local, extents)
    prior = 0.25 * shape + 0.25 * power + 0.30 * size + 0.20 * symmetry

    reasons: list[str] = []
    if grasp_span < 0.025:
        reasons.append("grasp_span_too_small")
    if grasp_span > 0.100:
        reasons.append("grasp_span_too_large")
    if minimum < 0.018:
        reasons.append("too_thin_for_enclosure")
    if maximum > 0.240:
        reasons.append("overall_size_too_large")
    if axis_ratio > 5.0:
        reasons.append("too_slender")
    return GeometryAffordance(
        extents_m=(minimum, middle, maximum),
        grasp_span_m=grasp_span,
        sphericity=sphericity,
        axis_ratio=axis_ratio,
        convexity=convexity,
        size_compatibility=size,
        shape_regularity=shape,
        power_grasp_suitability=power,
        rotational_symmetry=symmetry,
        geometry_prior=float(prior),
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def adaptive_contact_score(arrays: dict[str, np.ndarray], close_stage: int = 4) -> float | None:
    """Measure whether distinct digit contacts accumulate during closure."""
    if "robot_object_digit_contact_count" not in arrays:
        return None
    mask = np.asarray(arrays["stage"]) == close_stage
    counts = np.asarray(arrays["robot_object_digit_contact_count"])[mask]
    if len(counts) < 2:
        return None
    active = (counts > 0).sum(axis=1).astype(np.float64)
    coverage = float(active.max() / counts.shape[1])
    nondecreasing = float(np.mean(np.diff(active) >= 0.0))
    persistence = float(np.mean(active[-max(1, len(active) // 4) :] / counts.shape[1]))
    return float(np.clip(0.45 * coverage + 0.25 * nondecreasing + 0.30 * persistence, 0.0, 1.0))


def complete_uas(
    geometry: GeometryAffordance,
    *,
    adaptive_contact: float,
    robustness: float,
) -> float:
    """Combine geometry and measured dynamics into the final UAS."""
    values = (adaptive_contact, robustness)
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Dynamic UAS components must lie in [0, 1].")
    return float(
        0.20 * geometry.shape_regularity
        + 0.20 * geometry.power_grasp_suitability
        + 0.20 * geometry.size_compatibility
        + 0.15 * geometry.rotational_symmetry
        + 0.15 * adaptive_contact
        + 0.10 * robustness
    )


def benchmark_eligible(
    geometry: GeometryAffordance,
    *,
    nominal_success: bool,
    adaptive_contact: float,
    robustness: float,
    minimum_uas: float = 0.70,
) -> bool:
    """Apply hard physical gates before admitting an object to DexHand-100."""
    if not geometry.eligible or not nominal_success:
        return False
    if adaptive_contact < 0.60 or robustness < 0.50:
        return False
    return complete_uas(
        geometry,
        adaptive_contact=adaptive_contact,
        robustness=robustness,
    ) >= minimum_uas
