"""Official GraspQP metric adapted to the six-drive closed-chain hand surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from source.grasping.catalog import ObjectGeometry
from source.grasping.contracts import GraspCandidate
from source.grasping.hand_surrogate import DexHandSurrogate
from source.grasping.seeds import (
    convex_outside_distance,
    _inverse_sigmoid,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


@dataclass(frozen=True)
class GraspQPConfig:
    steps: int = 100
    translation_learning_rate: float = 0.0015
    rotation_learning_rate: float = 0.006
    actuator_learning_rate: float = 0.012
    contact_temperature: float = 0.002
    contact_distance: float = 0.004
    maximum_penetration: float = 0.003
    friction: float = 0.8
    maximum_force: float = 20.0
    closure_reserve: float = 0.0
    device: str | None = None
    dtype: str = "float32"

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("GraspQP steps must be positive.")
        if (
            min(
                self.translation_learning_rate,
                self.rotation_learning_rate,
                self.actuator_learning_rate,
                self.contact_temperature,
                self.contact_distance,
                self.maximum_penetration,
                self.friction,
                self.maximum_force,
            )
            <= 0.0
        ):
            raise ValueError("GraspQP numeric settings must be positive.")
        if not 0.0 <= self.closure_reserve < 1.0:
            raise ValueError("closure_reserve must lie in [0, 1).")


def graspqp_available() -> bool:
    try:
        from graspqp.metrics import GraspSpanMetricFactory  # noqa: F401
    except ImportError:
        return False
    return True


def refine_candidates_with_graspqp(
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    seeds: tuple[GraspCandidate, ...],
    config: GraspQPConfig | None = None,
) -> tuple[GraspCandidate, ...]:
    """Jointly refine wrist pose and six physical drives with official GraspQP."""
    import torch
    from graspqp.metrics import GraspSpanMetricFactory

    if not seeds:
        return ()
    config = config or GraspQPConfig()
    config.validate()
    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for GraspQP but is unavailable.")
    device = torch.device(device_name)
    dtype = getattr(torch, config.dtype)
    count = len(seeds)
    initial_t = np.stack([item.hand_translation for item in seeds])
    initial_r = np.stack([item.hand_rotation_matrix for item in seeds])
    initial_f = np.stack([item.actuator_fractions for item in seeds])
    translation = torch.nn.Parameter(torch.as_tensor(initial_t, device=device, dtype=dtype))
    rotation_6d = torch.nn.Parameter(
        torch.as_tensor(matrix_to_rotation_6d(initial_r), device=device, dtype=dtype)
    )
    fraction_logits = torch.nn.Parameter(
        torch.as_tensor(_inverse_sigmoid(initial_f), device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [translation], "lr": config.translation_learning_rate},
            {"params": [rotation_6d], "lr": config.rotation_learning_rate},
            {"params": [fraction_logits], "lr": config.actuator_learning_rate},
        ]
    )
    metric = GraspSpanMetricFactory.create(
        GraspSpanMetricFactory.MetricType.GRASPQP,
        solver_kwargs={
            "friction": config.friction,
            "max_limit": config.maximum_force,
            "n_cone_vecs": 4,
        },
    ).to(device)
    object_points = torch.as_tensor(geometry.surface_points, device=device, dtype=dtype)
    object_normals = torch.as_tensor(geometry.surface_normals, device=device, dtype=dtype)
    plane_normals = torch.as_tensor(geometry.plane_normals, device=device, dtype=dtype)
    plane_offsets = torch.as_tensor(geometry.plane_offsets, device=device, dtype=dtype)
    cog = object_points.mean(dim=0, keepdim=True).expand(count, -1)
    initial_t_tensor = torch.as_tensor(initial_t, device=device, dtype=dtype)
    contact_indices = [
        torch.as_tensor(group, device=device, dtype=torch.long)
        for group in surrogate.contact_indices
    ]

    final_qp = None
    final_distance = None
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        fractions = torch.sigmoid(fraction_logits)
        rotations = rotation_6d_to_matrix(rotation_6d)
        local = surrogate.evaluate_torch(fractions)
        hand_points = torch.einsum("bij,bnj->bni", rotations, local) + translation[:, None]
        distances = torch.cdist(hand_points, object_points[None].expand(count, -1, -1))
        contacts, inward_normals, digit_distances = [], [], []
        for indices in contact_indices:
            group_distances = distances.index_select(1, indices)
            per_hand_min = group_distances.min(dim=2).values
            hand_weights = torch.softmax(-per_hand_min / config.contact_temperature, dim=1)
            object_weights = torch.softmax(-group_distances / config.contact_temperature, dim=2)
            surface = torch.einsum("bho,oi->bhi", object_weights, object_points)
            normals = torch.einsum("bho,oi->bhi", object_weights, object_normals)
            contacts.append(torch.einsum("bh,bhi->bi", hand_weights, surface))
            normal = torch.einsum("bh,bhi->bi", hand_weights, normals)
            inward_normals.append(-normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-6))
            digit_distances.append((hand_weights * per_hand_min).sum(dim=1))
        contacts_t = torch.stack(contacts, dim=1)
        normals_t = torch.stack(inward_normals, dim=1)
        digit_distance_t = torch.stack(digit_distances, dim=1)
        qp_energy = metric(contacts_t, normals_t, cog=cog)
        outside = convex_outside_distance(hand_points, plane_normals, plane_offsets)
        penetration = torch.relu(-outside - 0.0005)
        table = torch.relu(geometry.table_z + 0.0015 - hand_points[..., 2])
        pose_drift = torch.square((translation - initial_t_tensor) / 0.04).mean(dim=1)
        coupling = torch.square(fractions[:, :4] - fractions[:, :4].mean(dim=1, keepdim=True)).mean(
            dim=1
        )
        per_seed = (
            qp_energy
            + 3.0 * torch.square(digit_distance_t / config.contact_distance).mean(dim=1)
            + 8.0 * torch.square(penetration / config.maximum_penetration).mean(dim=1)
            + 10.0 * torch.square(table / config.maximum_penetration).mean(dim=1)
            + 0.08 * pose_drift
            + 0.03 * coupling
        )
        per_seed.mean().backward()
        torch.nn.utils.clip_grad_norm_([translation, rotation_6d, fraction_logits], 20.0)
        optimizer.step()
        final_qp = qp_energy.detach()
        final_distance = digit_distance_t.detach()

    with torch.no_grad():
        translations = translation.cpu().numpy()
        rotations = rotation_6d_to_matrix(rotation_6d).cpu().numpy()
        contact_fractions = torch.sigmoid(fraction_logits).cpu().numpy()
        fractions = contact_fractions.copy()
        # GraspQP returns a zero-load first-contact geometry. Position control
        # needs additional travel after contact to generate normal force.
        # Keep the thumb opposition drive (4) unchanged and add reserve only
        # to the five actual closing drives.
        fractions[:, :4] = np.clip(fractions[:, :4] + config.closure_reserve, 0.0, 1.0)
        fractions[:, 5] = np.clip(fractions[:, 5] + config.closure_reserve, 0.0, 1.0)
        qp_values = final_qp.cpu().numpy()
        distance_values = final_distance.cpu().numpy()
    candidates = []
    for index, seed in enumerate(seeds):
        contacts, normals = [], []
        points = (
            surrogate.evaluate_numpy(contact_fractions[index]) @ rotations[index].T
            + translations[index]
        )
        for group in surrogate.contact_indices:
            delta = points[group, None] - geometry.surface_points[None]
            pair = np.unravel_index(np.argmin(np.sum(delta * delta, axis=2)), delta.shape[:2])
            contacts.append(geometry.surface_points[pair[1]])
            normals.append(geometry.surface_normals[pair[1]])
        candidates.append(
            GraspCandidate(
                object_id=geometry.object_id,
                seed_index=4_000_000 + index,
                hand_translation=translations[index],
                hand_rotation_matrix=rotations[index],
                actuator_fractions=fractions[index],
                contact_points=np.asarray(contacts),
                contact_normals=np.asarray(normals),
                contact_distances=distance_values[index],
                metrics={
                    **seed.metrics,
                    "valid": 0.0,
                    "graspqp_prior": 1.0,
                    "graspqp_energy": float(qp_values[index]),
                    "mean_contact_distance": float(distance_values[index].mean()),
                    "graspqp_closure_reserve": config.closure_reserve,
                },
                backend="official-graspqp-closed-chain",
            )
        )
    return tuple(candidates)
