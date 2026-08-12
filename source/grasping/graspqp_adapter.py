"""Experimental kinematics bridge for using GraspQP with the closed-chain Dex Hand.

GraspQP expects differentiable hand kinematics.  The project's hand is an MJCF
closed linkage, so it cannot be loaded by GraspQP's URDF kinematics directly.
This module provides the first, dependency-light bridge: MuJoCo resolves the
linkage and central differences expose derivatives with respect to the six
independent actuator fractions.  It is suitable for validating an optimizer
integration and for generating data for a later differentiable surrogate.
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
    initial_energy: float
    final_energy: float


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


def refine_wrist_pose(
    *,
    hand_points: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    object_points: np.ndarray,
    object_normals: np.ndarray,
    iterations: int = 120,
    learning_rate: float = 2e-3,
    device: str = "cuda",
) -> GraspQPPoseRefinement:
    """Refine a wrist pose with the official differentiable GraspQP metric.

    ``hand_points`` contains one fixed contact candidate per digit in hand-root
    coordinates. A differentiable soft nearest-neighbour surface supplies the
    contact-distance term while GraspQP supplies the force-closure QP term.
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
    local = torch.as_tensor(hand_points, dtype=dtype, device=target_device)
    cloud = torch.as_tensor(object_points, dtype=dtype, device=target_device)
    normals = torch.as_tensor(object_normals, dtype=dtype, device=target_device)
    base_rotation = torch.as_tensor(initial_rotation, dtype=dtype, device=target_device)
    translation = torch.tensor(
        initial_translation, dtype=dtype, device=target_device, requires_grad=True
    )
    rotation_delta = torch.zeros(3, dtype=dtype, device=target_device, requires_grad=True)
    optimizer = torch.optim.Adam([translation, rotation_delta], lr=learning_rate)
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
        contacts = local @ rotation.T + translation
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
        energy = 100.0 * distance_energy + qp_energy + pose_regularizer
        if initial_energy is None:
            initial_energy = float(energy.detach().cpu())
        energy.backward()
        optimizer.step()

    final_rotation = (base_rotation @ rotation_matrix(rotation_delta)).detach().cpu().numpy()
    return GraspQPPoseRefinement(
        rotation=final_rotation,
        translation=translation.detach().cpu().numpy(),
        initial_energy=float(initial_energy),
        final_energy=float(energy.detach().cpu()),
    )
