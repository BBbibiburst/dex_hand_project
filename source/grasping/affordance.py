"""Computable underactuated-enclosure affordance metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


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
    support_margin_m: float
    center_of_mass_height_m: float
    tipping_angle_deg: float
    initial_stability: float
    geometry_prior: float
    eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InitialPoseStability:
    """Measured free-settling stability in the real lift environment."""

    stable: bool
    settled: bool
    horizontal_displacement_m: float
    vertical_displacement_m: float
    orientation_change_deg: float
    tail_max_linear_speed_m_s: float
    tail_max_angular_speed_rad_s: float
    simulated_seconds: float

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


def _initial_pose_stability(hull) -> tuple[float, float, float, float]:
    """Estimate support geometry in the mesh frame (not a physical gate)."""
    vertices = np.asarray(hull.vertices, dtype=np.float64)
    minimum_z = float(vertices[:, 2].min())
    height = max(float(np.ptp(vertices[:, 2])), 1e-9)
    # A narrow band tolerates scan noise while keeping point-contact spheres
    # distinguishable from objects with an actual planar support surface.
    support_band = max(1e-5, min(2.5e-4, 0.005 * height))
    support_xy = vertices[vertices[:, 2] <= minimum_z + support_band, :2]
    center = np.asarray(hull.center_mass, dtype=np.float64)
    if center.shape != (3,) or np.any(~np.isfinite(center)):
        center = vertices.mean(axis=0)
    com_height = max(float(center[2] - minimum_z), 1e-9)
    margin = 0.0
    if len(support_xy) >= 3:
        try:
            support_hull = ConvexHull(support_xy)
        except QhullError:
            pass
        else:
            equations = np.asarray(support_hull.equations, dtype=np.float64)
            signed = -(
                equations[:, :2] @ center[:2] + equations[:, 2]
            ) / np.maximum(np.linalg.norm(equations[:, :2], axis=1), 1e-12)
            margin = max(0.0, float(signed.min()))
    angle = float(np.degrees(np.arctan2(margin, com_height)))
    stability = float(np.clip((angle - 8.0) / 17.0, 0.0, 1.0))
    return margin, com_height, angle, stability


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
    support_margin, com_height, tipping_angle, stability = _initial_pose_stability(hull)
    # Mesh-frame support is retained for diagnostics only. Catalogue meshes do
    # not share a reliable upright convention, so physical free settling owns
    # the stability gate and the geometry prior remains orientation agnostic.
    prior = 0.26 * shape + 0.26 * power + 0.30 * size + 0.18 * symmetry

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
        support_margin_m=support_margin,
        center_of_mass_height_m=com_height,
        tipping_angle_deg=tipping_angle,
        initial_stability=stability,
        geometry_prior=float(prior),
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def initial_pose_stability_from_trajectory(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    velocities: np.ndarray,
    *,
    timestep: float,
    tail_seconds: float = 0.25,
    maximum_horizontal_displacement_m: float = 0.005,
    maximum_orientation_change_deg: float = 10.0,
    maximum_tail_linear_speed_m_s: float = 0.01,
    maximum_tail_angular_speed_rad_s: float = 0.10,
) -> InitialPoseStability:
    """Classify whether the initially placed object remains usable for planning."""

    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must have shape (T, 3) with T >= 2")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("quaternions_wxyz must have shape (T, 4)")
    if velocities.shape != (len(positions), 6):
        raise ValueError("velocities must have shape (T, 6)")
    if timestep <= 0.0 or tail_seconds <= 0.0:
        raise ValueError("timestep and tail_seconds must be positive")

    q0 = quaternions[0] / max(float(np.linalg.norm(quaternions[0])), 1e-12)
    q1 = quaternions[-1] / max(float(np.linalg.norm(quaternions[-1])), 1e-12)
    orientation_change = float(
        np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(q0, q1))), 0.0, 1.0)))
    )
    displacement = positions[-1] - positions[0]
    horizontal_displacement = float(np.linalg.norm(displacement[:2]))
    tail_count = min(len(velocities), max(1, int(np.ceil(tail_seconds / timestep))))
    tail = velocities[-tail_count:]
    tail_linear_speed = float(np.linalg.norm(tail[:, :3], axis=1).max())
    tail_angular_speed = float(np.linalg.norm(tail[:, 3:], axis=1).max())
    # Planning only becomes invalid when the object actually migrates or tips.
    # Instantaneous free-joint velocity can remain noisy under stiff mesh-table
    # contact even while the measured pose is effectively constant, so expose
    # quiet settling separately instead of turning contact chatter into a tip.
    stable = bool(
        horizontal_displacement <= maximum_horizontal_displacement_m
        and orientation_change <= maximum_orientation_change_deg
    )
    settled = bool(
        tail_linear_speed <= maximum_tail_linear_speed_m_s
        and tail_angular_speed <= maximum_tail_angular_speed_rad_s
    )
    return InitialPoseStability(
        stable=stable,
        settled=settled,
        horizontal_displacement_m=horizontal_displacement,
        vertical_displacement_m=float(displacement[2]),
        orientation_change_deg=orientation_change,
        tail_max_linear_speed_m_s=tail_linear_speed,
        tail_max_angular_speed_rad_s=tail_angular_speed,
        simulated_seconds=float((len(positions) - 1) * timestep),
    )


def simulate_initial_pose_stability(
    object_id: str,
    *,
    seed: int = 0,
    settle_seconds: float = 1.5,
) -> InitialPoseStability:
    """Run free settling with the exact object scale and lift-task placement."""

    import mujoco

    from source.envs.manipulation import make_lift_env

    if settle_seconds <= 0.0:
        raise ValueError("settle_seconds must be positive")
    env = make_lift_env(
        task_config={"object_id": object_id},
        control_mode="ik",
        enable_tactile_sensors=False,
        episode_length=2,
    )
    try:
        env.reset(seed=seed)
        binding = env.task._require_bindings().objects["object"]
        timestep = float(env.model.opt.timestep)
        steps = max(1, int(np.ceil(settle_seconds / timestep)))
        positions = [env.data.xpos[binding.body_id].copy()]
        quaternions = [env.data.xquat[binding.body_id].copy()]
        velocities = [env.data.qvel[binding.qvel_adr : binding.qvel_adr + 6].copy()]
        for _ in range(steps):
            mujoco.mj_step(env.model, env.data)
            positions.append(env.data.xpos[binding.body_id].copy())
            quaternions.append(env.data.xquat[binding.body_id].copy())
            velocities.append(env.data.qvel[binding.qvel_adr : binding.qvel_adr + 6].copy())
        return initial_pose_stability_from_trajectory(
            np.asarray(positions),
            np.asarray(quaternions),
            np.asarray(velocities),
            timestep=timestep,
        )
    finally:
        env.close()


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
