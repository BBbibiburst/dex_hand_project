"""Closed-chain Dex Hand integration for GraspQP seed optimization.

GraspQP expects differentiable hand kinematics.  The project's hand is an MJCF
closed linkage, so it cannot be loaded by GraspQP's URDF kinematics directly.
MuJoCo therefore resolves the linkage and central differences expose derivatives
with respect to the six independent actuator fractions.  GraspQP's force-closure
metric then jointly refines the hand drives and free-wrist pose before the
simulator-based evolutionary stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stdout
from importlib.util import find_spec
import io

import numpy as np

from source.grasping.dex_hand_surface import PosedDexHandSurface, load_posed_dex_hand_surface


@dataclass(frozen=True)
class GraspQPCompatibility:
    installed: bool
    torch_installed: bool
    cuda_available: bool
    reason: str | None


@dataclass(frozen=True)
class DexHandKinematicsSample:
    fractions: np.ndarray
    surface: PosedDexHandSurface
    point_jacobian: np.ndarray
    fingertip_jacobian: np.ndarray


@dataclass(frozen=True)
class GraspQPPoseRefinement:
    rotation: np.ndarray
    translation: np.ndarray
    actuator_fractions: np.ndarray
    initial_energy: float
    final_energy: float
    minimum_table_clearance: float


def check_graspqp_compatibility() -> GraspQPCompatibility:
    """Report whether the optional official GraspQP runtime can be imported."""
    installed = find_spec("graspqp") is not None
    torch_installed = find_spec("torch") is not None
    cuda_available = False
    if torch_installed:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    if not installed:
        reason = "graspqp is not installed; install the official package with its lite extra"
    elif not torch_installed:
        reason = "GraspQP requires PyTorch"
    elif not cuda_available:
        reason = "GraspQP is importable, but CUDA is unavailable; full optimization will be slow"
    else:
        reason = None
    return GraspQPCompatibility(installed, torch_installed, cuda_available, reason)


def sample_closed_chain_kinematics(
    fractions: np.ndarray,
    *,
    epsilon: float = 1e-3,
    max_points_per_geom: int = 80,
    seed: int = 0,
) -> DexHandKinematicsSample:
    """Resolve Dex Hand geometry and its Jacobian over six actuator fractions.

    The returned Jacobians use shapes ``(points, 3, 6)`` and ``(5, 3, 6)``.
    One-sided differences are used at actuator limits.
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.shape != (6,) or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("fractions must contain six values in [0, 1].")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5).")

    kwargs = {"max_points_per_geom": max_points_per_geom, "seed": seed}
    center = load_posed_dex_hand_surface(actuator_fractions=fractions, **kwargs)
    point_jacobian = np.empty((*center.points.shape, 6), dtype=np.float64)
    fingertip_jacobian = np.empty((5, 3, 6), dtype=np.float64)

    for axis in range(6):
        lower = fractions.copy()
        upper = fractions.copy()
        lower[axis] = max(0.0, lower[axis] - epsilon)
        upper[axis] = min(1.0, upper[axis] + epsilon)
        span = upper[axis] - lower[axis]
        low_surface = load_posed_dex_hand_surface(actuator_fractions=lower, **kwargs)
        high_surface = load_posed_dex_hand_surface(actuator_fractions=upper, **kwargs)
        if (
            low_surface.points.shape != center.points.shape
            or high_surface.points.shape != center.points.shape
        ):
            raise RuntimeError("Dex Hand surface sampling changed during finite differences.")
        point_jacobian[..., axis] = (high_surface.points - low_surface.points) / span
        fingertip_jacobian[..., axis] = (
            high_surface.fingertip_centers - low_surface.fingertip_centers
        ) / span

    return DexHandKinematicsSample(
        fractions=fractions.copy(),
        surface=center,
        point_jacobian=point_jacobian,
        fingertip_jacobian=fingertip_jacobian,
    )


def refine_closed_chain_grasp(
    *,
    hand_points: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    object_points: np.ndarray,
    object_normals: np.ndarray,
    initial_fractions: np.ndarray | None = None,
    table_z: float | None = None,
    table_clearance: float = 0.005,
    iterations: int = 120,
    learning_rate: float = 2e-3,
    device: str = "cuda",
) -> GraspQPPoseRefinement:
    """Jointly refine a closed-chain Dex Hand grasp with the GraspQP metric.

    MuJoCo resolves the closed-chain kinematics and finite differences expose
    derivatives for the six independent drives. Wrist pose and drive fractions
    are optimized together. GraspQP supplies force closure while surface
    distance, drive bounds, and the table half-space provide task constraints.
    """
    import torch

    # GraspQP prints optional-backend warnings at import time and solver details
    # every time its lazy metric is initialized. The catalogue benchmark has
    # its own concise per-object progress output, so suppress only those known
    # informational prints; Python exceptions still propagate normally.
    with redirect_stdout(io.StringIO()):
        from graspqp.metrics import GraspSpanMetricFactory

    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    dtype = torch.float32
    target_device = torch.device(device)
    cloud = torch.as_tensor(object_points, dtype=dtype, device=target_device)
    normals = torch.as_tensor(object_normals, dtype=dtype, device=target_device)
    base_rotation = torch.as_tensor(initial_rotation, dtype=dtype, device=target_device)
    translation = torch.tensor(
        initial_translation, dtype=dtype, device=target_device, requires_grad=True
    )
    rotation_delta = torch.zeros(3, dtype=dtype, device=target_device, requires_grad=True)
    if initial_fractions is None:
        raise ValueError("initial_fractions are required for closed-chain grasp refinement")
    base_fractions_np = np.asarray(initial_fractions, dtype=np.float64)
    kinematics = sample_closed_chain_kinematics(
        base_fractions_np,
        epsilon=2e-3,
        max_points_per_geom=20,
    )
    requested_contacts = np.asarray(hand_points, dtype=np.float64)
    nearest_indices = np.linalg.norm(
        kinematics.surface.points[:, None, :] - requested_contacts[None, :, :],
        axis=-1,
    ).argmin(axis=0)
    local = torch.as_tensor(
        kinematics.surface.points[nearest_indices], dtype=dtype, device=target_device
    )
    local_jacobian_np = kinematics.point_jacobian[nearest_indices]
    table_points_np = kinematics.surface.points
    table_jacobian_np = kinematics.point_jacobian
    base_fractions = torch.as_tensor(base_fractions_np, dtype=dtype, device=target_device)
    local_jacobian = torch.as_tensor(local_jacobian_np, dtype=dtype, device=target_device)
    table_points = torch.as_tensor(table_points_np, dtype=dtype, device=target_device)
    table_jacobian = torch.as_tensor(table_jacobian_np, dtype=dtype, device=target_device)
    fraction_delta = torch.zeros(6, dtype=dtype, device=target_device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [translation, rotation_delta, fraction_delta], lr=learning_rate
    )
    with redirect_stdout(io.StringIO()):
        metric = GraspSpanMetricFactory.create(
            GraspSpanMetricFactory.MetricType.GRASPQP,
            solver_kwargs={"friction": 0.6, "max_limit": 20.0},
        )

    def rotation_matrix(vector):
        angle = torch.linalg.vector_norm(vector).clamp_min(1e-8)
        axis = vector / angle
        x, y, z = axis.unbind()
        zero = torch.zeros((), device=target_device)
        skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
        eye = torch.eye(3, device=target_device)
        return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)

    initial_energy = None
    for _ in range(iterations):
        optimizer.zero_grad()
        rotation = base_rotation @ rotation_matrix(rotation_delta)
        fractions = base_fractions + fraction_delta
        local_contacts = local + torch.einsum("pij,j->pi", local_jacobian, fraction_delta)
        contacts = local_contacts @ rotation.T + translation
        squared = torch.cdist(contacts, cloud).square()
        weights = torch.softmax(-squared / (0.004**2), dim=-1)
        surface = weights @ cloud
        contact_normals = -(weights @ normals)
        contact_normals = contact_normals / contact_normals.norm(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        distance_energy = (contacts - surface).square().sum(-1).sqrt().sum()
        cog = cloud.mean(0, keepdim=True)
        with redirect_stdout(io.StringIO()):
            qp_energy = metric(contacts.unsqueeze(0), contact_normals.unsqueeze(0), cog=cog).sum()
        pose_regularizer = (
            20.0
            * (
                translation
                - torch.as_tensor(initial_translation, dtype=dtype, device=target_device)
            )
            .square()
            .sum()
            + 0.02 * rotation_delta.square().sum()
        )
        joint_limit_energy = (
            torch.relu(-fractions).square() + torch.relu(fractions - 1.0).square()
        ).sum()
        table_energy = torch.zeros((), dtype=dtype, device=target_device)
        if table_z is not None:
            local_table_points = table_points + torch.einsum(
                "pij,j->pi", table_jacobian, fraction_delta
            )
            world_z = (local_table_points @ rotation.T + translation)[:, 2]
            floor = float(table_z) + float(table_clearance)
            table_energy = torch.relu(floor - world_z).square().sum()
        energy = (
            100.0 * distance_energy
            + qp_energy
            + pose_regularizer
            + 500.0 * joint_limit_energy
            + 2_000.0 * table_energy
        )
        if initial_energy is None:
            initial_energy = float(energy.detach().cpu())
        energy.backward()
        optimizer.step()

    final_fractions = np.clip(
        (base_fractions + fraction_delta).detach().cpu().numpy(), 0.0, 1.0
    )
    final_rotation = (base_rotation @ rotation_matrix(rotation_delta)).detach().cpu().numpy()
    final_translation = translation.detach().cpu().numpy()
    final_surface = load_posed_dex_hand_surface(
        actuator_fractions=final_fractions,
        max_points_per_geom=20,
        seed=0,
    )
    world_points = final_surface.points @ final_rotation.T + final_translation
    minimum_table_clearance = (
        float("inf") if table_z is None else float(world_points[:, 2].min() - table_z)
    )
    return GraspQPPoseRefinement(
        rotation=final_rotation,
        translation=final_translation,
        actuator_fractions=final_fractions,
        initial_energy=float(initial_energy),
        final_energy=float(energy.detach().cpu()),
        minimum_table_clearance=minimum_table_clearance,
    )
