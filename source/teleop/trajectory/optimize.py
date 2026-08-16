"""Deterministic offline smoothing for raw Vive + glove trajectories.

This is intentionally *contact preserving*, not collision-free.  Frames with
non-trivial tactile signal receive higher data fidelity so a table-assisted or
object-contact motion is not averaged away merely because it contains contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from source.teleop.trajectory.contracts import TeleopTrajectory


@dataclass(frozen=True)
class TrajectoryOptimizationConfig:
    position_smoothness: float = 18.0
    orientation_smoothness: float = 10.0
    hand_smoothness: float = 4.0
    contact_fidelity: float = 8.0
    contact_threshold: float = 0.02
    contact_padding_frames: int = 2
    edge_fidelity: float = 1000.0
    edge_frames: int = 2

    def validate(self) -> None:
        for name in ("position_smoothness", "orientation_smoothness", "hand_smoothness"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.contact_fidelity < 1.0:
            raise ValueError("contact_fidelity must be >= 1.")
        if self.contact_threshold < 0.0:
            raise ValueError("contact_threshold must be non-negative.")
        if self.contact_padding_frames < 0 or self.edge_frames < 0:
            raise ValueError("frame padding values must be non-negative.")
        if self.edge_fidelity < 1.0:
            raise ValueError("edge_fidelity must be >= 1.")


def _contact_mask(tactile: np.ndarray, threshold: float, padding: int) -> np.ndarray:
    tactile = np.asarray(tactile)
    horizon = len(tactile)
    if tactile.size == 0:
        return np.zeros(horizon, dtype=bool)
    flat = np.abs(tactile).reshape(horizon, -1)
    mask = np.max(flat, axis=1) > threshold
    if padding and np.any(mask):
        kernel = np.ones(2 * padding + 1, dtype=np.int32)
        mask = np.convolve(mask.astype(np.int32), kernel, mode="same") > 0
    return mask


def _weights(horizon: int, contact: np.ndarray, cfg: TrajectoryOptimizationConfig) -> np.ndarray:
    weights = np.ones(horizon, dtype=np.float64)
    weights[contact] *= cfg.contact_fidelity
    edge = min(cfg.edge_frames, horizon)
    if edge:
        weights[:edge] = np.maximum(weights[:edge], cfg.edge_fidelity)
        weights[-edge:] = np.maximum(weights[-edge:], cfg.edge_fidelity)
    return weights


def _smooth(values: np.ndarray, smoothness: float, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if smoothness <= 0.0 or len(values) < 3:
        return values.copy()
    horizon = len(values)
    d2 = np.zeros((horizon - 2, horizon), dtype=np.float64)
    rows = np.arange(horizon - 2)
    d2[rows, rows] = 1.0
    d2[rows, rows + 1] = -2.0
    d2[rows, rows + 2] = 1.0
    system = np.diag(weights) + smoothness * (d2.T @ d2)
    rhs = weights[:, None] * values.reshape(horizon, -1)
    solved = np.linalg.solve(system, rhs)
    return solved.reshape(values.shape)


def _continuous_quaternions(quaternions_wxyz: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions_wxyz, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result /= np.maximum(norms, 1e-12)
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _smooth_orientation(
    quaternions_wxyz: np.ndarray,
    smoothness: float,
    weights: np.ndarray,
) -> np.ndarray:
    quats = _continuous_quaternions(quaternions_wxyz)
    if smoothness <= 0.0 or len(quats) < 3:
        return quats.astype(np.float32)
    xyzw = quats[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(xyzw)
    origin = rotations[0]
    relative = origin.inv() * rotations
    rotvec = relative.as_rotvec()
    smoothed = _smooth(rotvec, smoothness, weights)
    result = origin * Rotation.from_rotvec(smoothed)
    out_xyzw = result.as_quat()
    out = out_xyzw[:, [3, 0, 1, 2]]
    out = _continuous_quaternions(out)
    return out.astype(np.float32)


def optimize_teleop_trajectory(
    trajectory: TeleopTrajectory,
    config: TrajectoryOptimizationConfig | None = None,
) -> TeleopTrajectory:
    """Smooth Cartesian pose and hand commands while retaining contact intent."""
    cfg = config or TrajectoryOptimizationConfig()
    cfg.validate()
    if trajectory.arm_action_size != 7:
        raise ValueError(
            "Teleop trajectory optimizer expects a 7D IK arm action (xyz + quaternion_wxyz)."
        )

    contact = _contact_mask(
        trajectory.observed_tactile,
        cfg.contact_threshold,
        cfg.contact_padding_frames,
    )
    weights = _weights(trajectory.horizon, contact, cfg)
    actions = trajectory.actions.astype(np.float64).copy()
    actions[:, :3] = _smooth(actions[:, :3], cfg.position_smoothness, weights)
    actions[:, 3:7] = _smooth_orientation(actions[:, 3:7], cfg.orientation_smoothness, weights)
    if trajectory.hand_action_size:
        actions[:, 7:] = _smooth(actions[:, 7:], cfg.hand_smoothness, weights)

    finite_low = np.isfinite(trajectory.action_low)
    finite_high = np.isfinite(trajectory.action_high)
    actions[:, finite_low] = np.maximum(actions[:, finite_low], trajectory.action_low[finite_low])
    actions[:, finite_high] = np.minimum(actions[:, finite_high], trajectory.action_high[finite_high])
    actions[:, 3:7] /= np.maximum(np.linalg.norm(actions[:, 3:7], axis=1, keepdims=True), 1e-12)

    position_delta = np.linalg.norm(actions[:, :3] - trajectory.actions[:, :3], axis=1)
    hand_delta = (
        np.linalg.norm(actions[:, 7:] - trajectory.actions[:, 7:], axis=1)
        if trajectory.hand_action_size
        else np.zeros(trajectory.horizon)
    )
    metadata_updates = {
        "trajectory_kind": "optimized",
        "source_trajectory_kind": trajectory.metadata.get("trajectory_kind", "raw"),
        "optimizer": "weighted_second_difference_se3_v1",
        "optimizer_parameters": {
            "position_smoothness": cfg.position_smoothness,
            "orientation_smoothness": cfg.orientation_smoothness,
            "hand_smoothness": cfg.hand_smoothness,
            "contact_fidelity": cfg.contact_fidelity,
            "contact_threshold": cfg.contact_threshold,
            "contact_padding_frames": cfg.contact_padding_frames,
            "edge_fidelity": cfg.edge_fidelity,
            "edge_frames": cfg.edge_frames,
        },
        "optimizer_contact_frames": int(contact.sum()),
        "optimizer_mean_position_change_m": float(position_delta.mean()),
        "optimizer_max_position_change_m": float(position_delta.max()),
        "optimizer_mean_hand_change": float(hand_delta.mean()),
    }
    return trajectory.with_actions(actions.astype(np.float32), metadata_updates=metadata_updates)
