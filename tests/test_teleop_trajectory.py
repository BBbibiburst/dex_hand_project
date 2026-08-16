"""Pure regression tests for teleop trajectory I/O and offline optimization."""

from __future__ import annotations

import numpy as np

from source.teleop.trajectory.contracts import TeleopTrajectory
from source.teleop.trajectory.optimize import (
    TrajectoryOptimizationConfig,
    optimize_teleop_trajectory,
)


def _trajectory(horizon: int = 12) -> TeleopTrajectory:
    t = np.arange(horizon, dtype=np.float64) * 0.05
    actions = np.zeros((horizon, 13), dtype=np.float32)
    actions[:, 0] = np.linspace(0.2, 0.3, horizon)
    actions[:, 1] = 0.01 * ((-1.0) ** np.arange(horizon))
    actions[:, 2] = 0.8
    actions[:, 3] = 1.0
    actions[:, 7:] = np.linspace(0.0, 0.8, horizon)[:, None]
    actions[:, 7:] += (0.03 * ((-1.0) ** np.arange(horizon)))[:, None]
    tactile = np.zeros((horizon, 4), dtype=np.float32)
    tactile[5:8] = 0.5
    return TeleopTrajectory(
        metadata={
            "schema_version": 1,
            "trajectory_kind": "raw",
            "control_dt": 0.05,
            "hand_action_size": 6,
            "task": "lift",
        },
        timestamps=t,
        actions=actions,
        vive_pose=np.tile(np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (horizon, 1)),
        glove=np.zeros((horizon, 6), dtype=np.float32),
        observed_qpos=np.zeros((horizon, 3), dtype=np.float32),
        observed_qvel=np.zeros((horizon, 3), dtype=np.float32),
        observed_ctrl=np.zeros((horizon, 2), dtype=np.float32),
        observed_object_position=np.tile(np.asarray([0.5, 0.0, 0.6], dtype=np.float32), (horizon, 1)),
        observed_object_quaternion=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (horizon, 1)),
        observed_tactile=tactile,
        task_success=np.zeros(horizon, dtype=bool),
        initial_qpos=np.zeros(3),
        initial_qvel=np.zeros(3),
        initial_ctrl=np.zeros(2),
        action_low=np.asarray([-np.inf] * 3 + [-1] * 4 + [0] * 6, dtype=np.float32),
        action_high=np.asarray([np.inf] * 3 + [1] * 4 + [1] * 6, dtype=np.float32),
    )


def test_teleop_trajectory_round_trip(tmp_path) -> None:
    source = _trajectory()
    path = source.save(tmp_path / "raw.npz")
    loaded = TeleopTrajectory.load(path)
    assert loaded.metadata["trajectory_kind"] == "raw"
    assert np.allclose(loaded.actions, source.actions)
    assert np.allclose(loaded.observed_tactile, source.observed_tactile)
    assert loaded.horizon == source.horizon


def test_optimizer_reduces_cartesian_jitter_and_normalizes_quaternion() -> None:
    source = _trajectory()
    optimized = optimize_teleop_trajectory(source)
    raw_jitter = np.mean(np.abs(np.diff(source.actions[:, 1], n=2)))
    optimized_jitter = np.mean(np.abs(np.diff(optimized.actions[:, 1], n=2)))
    assert optimized_jitter < raw_jitter
    assert np.allclose(np.linalg.norm(optimized.actions[:, 3:7], axis=1), 1.0, atol=1e-5)
    assert optimized.metadata["trajectory_kind"] == "optimized"
    assert optimized.metadata["optimizer_contact_frames"] > 0


def test_contact_fidelity_keeps_contact_frames_closer_to_raw() -> None:
    source = _trajectory()
    low = optimize_teleop_trajectory(
        source,
        TrajectoryOptimizationConfig(contact_fidelity=1.0, edge_frames=0),
    )
    high = optimize_teleop_trajectory(
        source,
        TrajectoryOptimizationConfig(contact_fidelity=50.0, edge_frames=0),
    )
    contact = slice(5, 8)
    low_error = np.mean(np.abs(low.actions[contact, 7:] - source.actions[contact, 7:]))
    high_error = np.mean(np.abs(high.actions[contact, 7:] - source.actions[contact, 7:]))
    assert high_error < low_error
