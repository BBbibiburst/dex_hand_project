import numpy as np
import pytest

from source.grasping.graspqp_adapter import sample_closed_chain_kinematics


def test_rejects_invalid_actuator_fractions() -> None:
    with pytest.raises(ValueError, match="six values"):
        sample_closed_chain_kinematics(np.zeros(5))


def test_closed_chain_jacobian_is_finite() -> None:
    sample = sample_closed_chain_kinematics(np.full(6, 0.5), epsilon=2e-3, max_points_per_geom=8)
    assert sample.point_jacobian.shape == (*sample.surface.points.shape, 6)
    assert sample.fingertip_jacobian.shape == (5, 3, 6)
    assert np.isfinite(sample.point_jacobian).all()
    assert np.linalg.norm(sample.fingertip_jacobian) > 0.0


def test_closed_chain_jacobian_tracks_six_independent_drives() -> None:
    sample = sample_closed_chain_kinematics(
        np.asarray([0.2, 0.3, 0.4, 0.5, 0.7, 0.6]),
        epsilon=2e-3,
        max_points_per_geom=6,
    )

    per_drive_motion = np.linalg.norm(sample.point_jacobian, axis=(0, 1))
    assert np.count_nonzero(per_drive_motion > 1e-6) == 6
