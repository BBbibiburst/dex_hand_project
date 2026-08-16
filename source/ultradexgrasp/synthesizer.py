"""Native UltraDexGrasp-style optimizer for the underactuated Dex Hand.

The optimizer deliberately uses the six physical hand drives rather than a
serial-joint approximation. It combines a MuJoCo-calibrated differentiable
hand surrogate with the same convex object geometry used by the task model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from source.ultradexgrasp.catalog import ObjectGeometry
from source.ultradexgrasp.contracts import GraspCandidate
from source.ultradexgrasp.hand_surrogate import DexHandSurrogate

ProgressCallback = Callable[[int, int, dict[str, float]], None]


@dataclass(frozen=True)
class SynthesisConfig:
    seed_count: int = 64
    optimization_steps: int = 220
    translation_learning_rate: float = 0.0025
    rotation_learning_rate: float = 0.012
    actuator_learning_rate: float = 0.025
    force_learning_rate: float = 0.05
    contact_distance: float = 0.004
    maximum_penetration: float = 0.003
    penetration_allowance: float = 0.0005
    minimum_contact_fingers: int = 4
    top_k: int = 32
    friction_coefficient: float = 0.8
    contact_temperature: float = 0.002
    table_margin: float = 0.0015
    force_residual_threshold: float = 0.28
    side_seed_fraction: float = 0.8
    device: str | None = None
    dtype: str = "float32"
    seed: int = 0

    def validate(self) -> None:
        for name in ("seed_count", "optimization_steps", "minimum_contact_fingers", "top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.minimum_contact_fingers > 5:
            raise ValueError("minimum_contact_fingers cannot exceed five.")
        rates = (
            self.translation_learning_rate,
            self.rotation_learning_rate,
            self.actuator_learning_rate,
            self.force_learning_rate,
        )
        if any(rate <= 0.0 for rate in rates) or self.contact_temperature <= 0.0:
            raise ValueError("Learning rates and contact_temperature must be positive.")
        if not 0.0 <= self.side_seed_fraction <= 1.0:
            raise ValueError("side_seed_fraction must lie in [0, 1].")


def _normalize(vector, *, eps: float = 1e-8):
    import torch

    return vector / torch.clamp(torch.linalg.norm(vector, dim=-1, keepdim=True), min=eps)


def rotation_6d_to_matrix(rotation_6d):
    """Convert Zhou et al. 6D rotations to right-handed matrices."""
    import torch

    first = _normalize(rotation_6d[..., :3])
    second_raw = rotation_6d[..., 3:]
    second = _normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def _inverse_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _rotation_from_approach(
    approach: np.ndarray,
    preferred_spread: np.ndarray,
    roll: float,
) -> np.ndarray:
    """Build a hand rotation whose local +Y follows ``approach``."""
    y_axis = approach / max(float(np.linalg.norm(approach)), 1e-9)
    z_axis = preferred_spread - np.dot(preferred_spread, y_axis) * y_axis
    if np.linalg.norm(z_axis) < 1e-6:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        z_axis = fallback - np.dot(fallback, y_axis) * y_axis
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    z_axis = np.cross(x_axis, y_axis)

    cosine = math.cos(roll)
    sine = math.sin(roll)
    rolled_x = cosine * x_axis - sine * z_axis
    rolled_z = sine * x_axis + cosine * z_axis
    return np.column_stack([rolled_x, y_axis, rolled_z])


def _seed_rotations(
    count: int,
    side_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[str, ...]]:
    side_count = round(count * side_fraction)
    side_count = min(max(side_count, 0), count)
    top_count = count - side_count
    rotations: list[np.ndarray] = []
    families: list[str] = []
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    roll_pattern = np.deg2rad(np.asarray([0.0, -18.0, 18.0], dtype=np.float64))

    for index in range(side_count):
        azimuth = phase + 2.0 * math.pi * index / max(side_count, 1)
        inward = np.asarray([-math.cos(azimuth), -math.sin(azimuth), 0.0])
        roll = float(roll_pattern[index % len(roll_pattern)])
        rotations.append(
            _rotation_from_approach(inward, np.asarray([0.0, 0.0, 1.0]), roll)
        )
        families.append("side")

    for index in range(top_count):
        azimuth = phase + 2.0 * math.pi * index / max(top_count, 1)
        spread = np.asarray([math.cos(azimuth), math.sin(azimuth), 0.0])
        rotations.append(
            _rotation_from_approach(
                np.asarray([0.0, 0.0, -1.0]),
                spread,
                float(roll_pattern[index % len(roll_pattern)]),
            )
        )
        families.append("top")
    return np.asarray(rotations, dtype=np.float64), tuple(families)


def _digit_centers(surrogate: DexHandSurrogate, fractions: np.ndarray) -> np.ndarray:
    points = surrogate.evaluate_numpy(fractions)
    return np.asarray([points[group].mean(axis=0) for group in surrogate.contact_indices])


def _object_width(vertices: np.ndarray, direction: np.ndarray) -> float:
    unit = direction / max(float(np.linalg.norm(direction)), 1e-9)
    projection = vertices @ unit
    return float(projection.max() - projection.min())


def _initialize_seeds(
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    config: SynthesisConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotations, families = _seed_rotations(
        config.seed_count,
        config.side_seed_fraction,
        rng,
    )
    fractions = np.empty((config.seed_count, 6), dtype=np.float64)
    translations = np.empty((config.seed_count, 3), dtype=np.float64)
    target_centers = np.empty((config.seed_count, 3), dtype=np.float64)
    object_center = 0.5 * (geometry.bounds[0] + geometry.bounds[1])
    object_height = float(geometry.bounds[1, 2] - geometry.bounds[0, 2])
    closing_grid = np.linspace(0.03, 0.62, 60)

    for index, (rotation, family) in enumerate(zip(rotations, families, strict=True)):
        thumb_rotate = float(
            rng.uniform(0.82, 1.0) if family == "side" else rng.uniform(0.2, 1.0)
        )
        candidates = np.repeat(closing_grid[:, None], 6, axis=1)
        candidates[:, 4] = thumb_rotate
        surfaces = surrogate.evaluate_numpy(candidates)
        centers = np.stack(
            [
                np.stack(
                    [surface[group].mean(axis=0) for group in surrogate.contact_indices]
                )
                for surface in surfaces
            ],
            axis=0,
        )
        finger_centers = centers[:, :4].mean(axis=1)
        gaps = centers[:, 4] - finger_centers
        gap_lengths = np.linalg.norm(gaps, axis=1)
        world_directions = np.einsum(
            "ij,nj->ni",
            rotation,
            gaps / np.maximum(gap_lengths[:, None], 1e-9),
        )
        object_widths = np.asarray(
            [_object_width(geometry.vertices, direction) for direction in world_directions]
        )
        desired_gaps = object_widths + 0.010
        feasible = np.flatnonzero(gap_lengths >= desired_gaps)
        if len(feasible):
            selected = int(
                feasible[np.argmin(gap_lengths[feasible] - desired_gaps[feasible])]
            )
        else:
            selected = int(np.argmax(gap_lengths))

        selected_fractions = candidates[selected].copy()
        selected_fractions[:4] += rng.normal(0.0, 0.018, size=4)
        selected_fractions[5] += float(rng.normal(0.0, 0.015))
        selected_fractions = np.clip(selected_fractions, 0.01, 0.99)
        selected_centers = _digit_centers(surrogate, selected_fractions)
        cavity_center = 0.5 * (
            selected_centers[:4].mean(axis=0) + selected_centers[4]
        )

        target = object_center.copy()
        if family == "top":
            target[2] += 0.20 * object_height
        translation = target - rotation @ cavity_center
        translation += rng.normal(0.0, 0.0015, size=3)

        hand_points = surrogate.evaluate_numpy(selected_fractions) @ rotation.T + translation
        minimum_z = float(hand_points[:, 2].min())
        required_z = geometry.table_z + config.table_margin
        if minimum_z < required_z:
            shift = required_z - minimum_z
            translation[2] += shift
            target[2] += shift

        fractions[index] = selected_fractions
        translations[index] = translation
        target_centers[index] = target
    return translations, rotations, fractions, target_centers


def _friction_wrenches(contact_points, normals, coefficient: float, length_scale: float):
    import torch

    reference_z = torch.tensor(
        [0.0, 0.0, 1.0],
        device=normals.device,
        dtype=normals.dtype,
    )
    reference_x = torch.tensor(
        [1.0, 0.0, 0.0],
        device=normals.device,
        dtype=normals.dtype,
    )
    use_x = torch.abs((normals * reference_z).sum(dim=-1, keepdim=True)) > 0.9
    reference = torch.where(use_x, reference_x, reference_z)
    tangent_1 = _normalize(torch.cross(normals, reference, dim=-1))
    tangent_2 = _normalize(torch.cross(normals, tangent_1, dim=-1))
    directions = []
    for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        tangent = math.cos(angle) * tangent_1 + math.sin(angle) * tangent_2
        directions.append(_normalize(-normals + coefficient * tangent))
    forces = torch.stack(directions, dim=2)
    arms = contact_points.unsqueeze(2).expand_as(forces)
    torques = torch.cross(arms, forces, dim=-1) / max(length_scale, 1e-4)
    return torch.cat([forces, torques], dim=-1)


def _nearest_surface(points, object_points, object_normals):
    import torch

    distances = torch.cdist(
        points,
        object_points.unsqueeze(0).expand(len(points), -1, -1),
    )
    minimum, indices = distances.min(dim=-1)
    flat = indices.reshape(-1)
    nearest_points = object_points.index_select(0, flat).reshape(*indices.shape, 3)
    nearest_normals = object_normals.index_select(0, flat).reshape(*indices.shape, 3)
    return minimum, nearest_points, nearest_normals


def _convex_outside_distance(points, plane_normals, plane_offsets):
    """Return positive outside / negative inside distance to a convex hull."""
    import torch

    plane_values = torch.einsum("bni,fi->bnf", points, plane_normals) - plane_offsets
    return plane_values.max(dim=-1).values


def _contact_state(
    points,
    distances,
    nearest_points,
    nearest_normals,
    contact_indices,
    temperature: float,
):
    import torch

    hand_contacts = []
    object_contacts = []
    normals = []
    digit_distances = []
    for indices in contact_indices:
        group_distances = distances.index_select(1, indices)
        weights = group_distances.div(-temperature).softmax(dim=1)
        hand_contacts.append(
            (points.index_select(1, indices) * weights[..., None]).sum(dim=1)
        )
        object_contacts.append(
            (nearest_points.index_select(1, indices) * weights[..., None]).sum(dim=1)
        )
        normals.append(
            _normalize(
                (nearest_normals.index_select(1, indices) * weights[..., None]).sum(dim=1)
            )
        )
        digit_distances.append((group_distances * weights).sum(dim=1))
    return (
        torch.stack(hand_contacts, dim=1),
        torch.stack(object_contacts, dim=1),
        torch.stack(normals, dim=1),
        torch.stack(digit_distances, dim=1),
    )


def synthesize_grasps(
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    config: SynthesisConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[GraspCandidate, ...]:
    """Optimize hand root pose and six actuator fractions in one batched solve."""
    import torch

    config = config or SynthesisConfig()
    config.validate()
    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for UltraDexGrasp synthesis but is unavailable.")
    dtype = getattr(torch, config.dtype)
    device = torch.device(device_name)
    rng = np.random.default_rng(config.seed)

    object_points_np = np.asarray(geometry.surface_points, dtype=np.float64)
    object_normals_np = np.asarray(geometry.surface_normals, dtype=np.float64)
    if len(object_points_np) < 128:
        raise ValueError("Object point cloud must contain at least 128 points.")

    initial_translations, initial_rotations, initial_fractions, target_centers_np = (
        _initialize_seeds(geometry, surrogate, config, rng)
    )
    translation = torch.nn.Parameter(
        torch.as_tensor(initial_translations, device=device, dtype=dtype)
    )
    rotation_6d = torch.nn.Parameter(
        torch.as_tensor(matrix_to_rotation_6d(initial_rotations), device=device, dtype=dtype)
    )
    fraction_logits = torch.nn.Parameter(
        torch.as_tensor(_inverse_sigmoid(initial_fractions), device=device, dtype=dtype)
    )
    force_logits = torch.nn.Parameter(
        torch.zeros((config.seed_count, 20), device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [translation], "lr": config.translation_learning_rate},
            {"params": [rotation_6d], "lr": config.rotation_learning_rate},
            {"params": [fraction_logits], "lr": config.actuator_learning_rate},
            {"params": [force_logits], "lr": config.force_learning_rate},
        ]
    )

    object_points = torch.as_tensor(object_points_np, device=device, dtype=dtype)
    object_normals = torch.as_tensor(object_normals_np, device=device, dtype=dtype)
    plane_normals = torch.as_tensor(geometry.plane_normals, device=device, dtype=dtype)
    plane_offsets = torch.as_tensor(geometry.plane_offsets, device=device, dtype=dtype)
    table_z = torch.as_tensor(geometry.table_z, device=device, dtype=dtype)
    initial_translation_t = torch.as_tensor(initial_translations, device=device, dtype=dtype)
    target_centers = torch.as_tensor(target_centers_np, device=device, dtype=dtype)
    contact_index_tensors = [
        torch.as_tensor(group, device=device, dtype=torch.long)
        for group in surrogate.contact_indices
    ]
    length_scale = float(np.max(geometry.bounds[1] - geometry.bounds[0]))
    identity = torch.eye(5, device=device, dtype=torch.bool).unsqueeze(0)
    penetration_count = min(20, surrogate.surface_point_count)

    for step in range(config.optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        fractions = torch.sigmoid(fraction_logits)
        rotations_t = rotation_6d_to_matrix(rotation_6d)
        local_points = surrogate.evaluate_torch(fractions)
        points = torch.einsum("bij,bnj->bni", rotations_t, local_points) + translation[:, None, :]
        distances, nearest_points, nearest_normals = _nearest_surface(
            points,
            object_points,
            object_normals,
        )
        hand_contacts, contacts, normals, digit_distances = _contact_state(
            points,
            distances,
            nearest_points,
            nearest_normals,
            contact_index_tensors,
            config.contact_temperature,
        )

        outside_distance = _convex_outside_distance(points, plane_normals, plane_offsets)
        penetration = torch.relu(-outside_distance - config.penetration_allowance)
        worst_penetration = torch.topk(penetration, penetration_count, dim=1).values
        table_penetration = torch.relu(table_z + config.table_margin - points[..., 2])
        worst_table = torch.topk(table_penetration, penetration_count, dim=1).values

        contact_loss = torch.square(digit_distances / config.contact_distance).mean(dim=1)
        penetration_loss = torch.square(
            worst_penetration / config.maximum_penetration
        ).mean(dim=1)
        table_loss = torch.square(worst_table / config.maximum_penetration).mean(dim=1)

        wrenches = _friction_wrenches(
            contacts,
            normals,
            config.friction_coefficient,
            length_scale,
        ).reshape(config.seed_count, 20, 6)
        force_weights = torch.softmax(force_logits, dim=1)
        force_residual = torch.linalg.norm(
            (wrenches * force_weights[..., None]).sum(dim=1),
            dim=1,
        )
        singular_values = torch.linalg.svdvals(wrenches.transpose(1, 2))
        isotropy_loss = -torch.log(torch.clamp(singular_values[:, -1], min=1e-5))

        pairwise = torch.cdist(contacts, contacts)
        separation = torch.relu(0.014 - pairwise.masked_fill(identity, 1.0)) / 0.014
        separation_loss = torch.square(separation).sum(dim=(1, 2)) / 20.0
        finger_normal = _normalize(normals[:, :4].mean(dim=1))
        opposition_loss = 1.0 + (finger_normal * normals[:, 4]).sum(dim=1)
        enclosure_center = 0.5 * (
            hand_contacts[:, :4].mean(dim=1) + hand_contacts[:, 4]
        )
        enclosure_loss = torch.square((enclosure_center - target_centers) / 0.025).mean(
            dim=1
        )
        pose_drift = torch.square((translation - initial_translation_t) / 0.05).mean(dim=1)
        finger_coupling = torch.square(
            fractions[:, :4] - fractions[:, :4].mean(dim=1, keepdim=True)
        ).mean(dim=1)

        phase = step / max(config.optimization_steps - 1, 1)
        force_phase = float(np.clip((phase - 0.30) / 0.45, 0.0, 1.0))
        enclosure_weight = 1.4 - 1.0 * force_phase
        per_seed = (
            3.5 * contact_loss
            + 8.0 * penetration_loss
            + 10.0 * table_loss
            + (0.15 + 1.35 * force_phase) * force_residual
            + 0.02 * force_phase * isotropy_loss
            + 0.45 * separation_loss
            + 0.65 * force_phase * opposition_loss
            + enclosure_weight * enclosure_loss
            + 0.08 * pose_drift
            + 0.03 * finger_coupling
        )
        loss = per_seed.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [translation, rotation_6d, fraction_logits, force_logits],
            max_norm=25.0,
        )
        optimizer.step()

        if progress is not None and (
            step == 0 or (step + 1) % 20 == 0 or step + 1 == config.optimization_steps
        ):
            progress(
                step + 1,
                config.optimization_steps,
                {
                    "loss": float(loss.detach().cpu()),
                    "contact_distance": float(digit_distances.mean().detach().cpu()),
                    "maximum_penetration": float(
                        torch.relu(-outside_distance).max().detach().cpu()
                    ),
                    "force_residual": float(force_residual.mean().detach().cpu()),
                },
            )

    with torch.no_grad():
        fractions = torch.sigmoid(fraction_logits)
        rotations_t = rotation_6d_to_matrix(rotation_6d)
        local_points = surrogate.evaluate_torch(fractions)
        points = torch.einsum("bij,bnj->bni", rotations_t, local_points) + translation[:, None, :]
        distances, nearest_points, nearest_normals = _nearest_surface(
            points,
            object_points,
            object_normals,
        )
        contacts = []
        normals = []
        digit_distances = []
        for indices in contact_index_tensors:
            group_distances = distances.index_select(1, indices)
            selected = group_distances.argmin(dim=1)
            batch = torch.arange(config.seed_count, device=device)
            absolute = indices[selected]
            contacts.append(nearest_points[batch, absolute])
            normals.append(nearest_normals[batch, absolute])
            digit_distances.append(group_distances[batch, selected])
        contacts_t = torch.stack(contacts, dim=1)
        normals_t = torch.stack(normals, dim=1)
        digit_distances_t = torch.stack(digit_distances, dim=1)
        wrenches = _friction_wrenches(
            contacts_t,
            normals_t,
            config.friction_coefficient,
            length_scale,
        ).reshape(config.seed_count, 20, 6)
        force_weights = torch.softmax(force_logits, dim=1)
        force_residual = torch.linalg.norm(
            (wrenches * force_weights[..., None]).sum(dim=1),
            dim=1,
        )
        isotropy = torch.linalg.svdvals(wrenches.transpose(1, 2))[:, -1]
        outside_distance = _convex_outside_distance(points, plane_normals, plane_offsets)
        maximum_penetration = torch.relu(-outside_distance).max(dim=1).values
        table_clearance = points[..., 2].min(dim=1).values - table_z
        contact_fingers = (digit_distances_t <= config.contact_distance).sum(dim=1)
        score = (
            digit_distances_t.mean(dim=1) / config.contact_distance
            + 2.0 * maximum_penetration / config.maximum_penetration
            + force_residual
            - 0.04 * isotropy
            + 3.0 * torch.relu(-table_clearance / config.maximum_penetration)
        )
        valid = (
            (contact_fingers >= config.minimum_contact_fingers)
            & (maximum_penetration <= config.maximum_penetration)
            & (table_clearance >= -0.0005)
            & (force_residual <= config.force_residual_threshold)
        )

        order = torch.argsort(score).detach().cpu().numpy().tolist()
        valid_order = [index for index in order if bool(valid[index].item())]
        selected_order = (valid_order or order)[: config.top_k]
        candidates = []
        for index in selected_order:
            candidates.append(
                GraspCandidate(
                    object_id=geometry.object_id,
                    seed_index=int(index),
                    hand_translation=translation[index].detach().cpu().numpy(),
                    hand_rotation_matrix=rotations_t[index].detach().cpu().numpy(),
                    actuator_fractions=fractions[index].detach().cpu().numpy(),
                    contact_points=contacts_t[index].detach().cpu().numpy(),
                    contact_normals=normals_t[index].detach().cpu().numpy(),
                    contact_distances=digit_distances_t[index].detach().cpu().numpy(),
                    metrics={
                        "score": float(score[index].item()),
                        "valid": float(valid[index].item()),
                        "contact_fingers": float(contact_fingers[index].item()),
                        "mean_contact_distance": float(
                            digit_distances_t[index].mean().item()
                        ),
                        "maximum_penetration": float(maximum_penetration[index].item()),
                        "table_clearance": float(table_clearance[index].item()),
                        "force_residual": float(force_residual[index].item()),
                        "wrench_isotropy": float(isotropy[index].item()),
                        "surrogate_rms": float(surrogate.calibration_rms),
                    },
                )
            )
    return tuple(candidates)
