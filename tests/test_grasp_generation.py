"""Contracts and architecture checks for GraspQP + DexEvolve generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from source.grasping.catalog import MANIFEST_PATH, load_object_geometry
from source.grasping.affordance import (
    adaptive_contact_score,
    benchmark_eligible,
    complete_uas,
    geometry_affordance,
    initial_pose_stability_from_trajectory,
)
from source.grasping.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from source.grasping.contracts import DemonstrationEpisode, GraspCandidate
from source.grasping.executor import (
    ExecutionConfig,
    candidate_world_pose,
    grasp_hand_targets,
)
from source.grasping.hand_surrogate import OPEN_FRACTIONS
from source.grasping.seeds import SeedConfig, convex_outside_distance
from tools.grasp_generation.visualize_episode import contact_points_world


def _candidate() -> GraspCandidate:
    return GraspCandidate(
        object_id="ycb:test",
        seed_index=3,
        hand_translation=np.asarray([0.1, -0.2, 0.03]),
        hand_rotation_matrix=np.eye(3),
        actuator_fractions=np.asarray([0.1, 0.2, 0.3, 0.4, 0.8, 0.25]),
        contact_points=np.zeros((5, 3)),
        contact_normals=np.tile(np.asarray([1.0, 0.0, 0.0]), (5, 1)),
        contact_distances=np.full(5, 0.001),
        metrics={"valid": 1.0},
    )


def test_outside_distance_uses_union_of_convex_collision_parts() -> None:
    torch = pytest.importorskip("torch")
    points = torch.tensor([[[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
    normals = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] * 2)
    offsets = torch.tensor([-1.0, 2.0, 2.0, -1.0])

    distance = convex_outside_distance(points, normals, offsets, [0, 2, 4])

    np.testing.assert_allclose(distance.numpy(), [[-0.5, 1.0, -0.5]])


def test_ycb_geometry_preserves_official_multiple_collision_parts() -> None:
    geometry = load_object_geometry("ycb:025_mug", surface_points=128)

    assert len(geometry.plane_part_offsets) - 1 > 1


def test_episode_roundtrip_uses_independent_npz_contract(tmp_path: Path) -> None:
    frames = 4
    arrays = {
        "qpos": np.zeros((frames, 7), dtype=np.float32),
        "qvel": np.zeros((frames, 7), dtype=np.float32),
        "ctrl": np.zeros((frames, 13), dtype=np.float32),
        "action": np.zeros((frames, 13), dtype=np.float32),
        "object_position": np.zeros((frames, 3), dtype=np.float32),
        "object_quaternion_wxyz": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (frames, 1),
        ),
        "stage": np.arange(frames, dtype=np.int16),
        "reward": np.zeros(frames, dtype=np.float32),
        "task_success": np.asarray([False, False, True, True]),
    }
    episode = DemonstrationEpisode(
        object_id="ycb:test",
        seed=7,
        candidate=_candidate(),
        arrays=arrays,
        success=True,
        terminal_stage="verify",
    )

    manifest = episode.save(tmp_path)
    restored = DemonstrationEpisode.load(manifest)

    assert restored.success
    assert restored.candidate.seed_index == 3
    assert np.array_equal(restored.arrays["task_success"], arrays["task_success"])


def test_candidate_world_pose_respects_object_and_attach_frames() -> None:
    candidate = _candidate()
    position, rotation, quaternion = candidate_world_pose(
        candidate,
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.eye(3),
    )

    assert np.allclose(position, [1.1, 1.8, 3.03])
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(np.abs(quaternion), [1.0, 0.0, 0.0, 0.0])


def test_default_pipeline_config_is_valid() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG_PATH)

    assert config.seeds.enclosure_prior_count > 0
    assert config.execution.lift_height > 0.04
    assert config.surrogate_options["finger_degree"] == 7
    assert config.target_size is None
    assert config.maximum_horizontal_diameter == pytest.approx(0.075)
    assert config.seeds.enclosure_prior_count >= 100


def test_open_hand_uses_collision_free_neutral_thumb_opposition() -> None:
    np.testing.assert_allclose(OPEN_FRACTIONS, [0.0, 0.0, 0.0, 0.0, 0.25, 0.0])


def test_visualized_contacts_follow_current_object_pose() -> None:
    candidate = _candidate()
    candidate = GraspCandidate(
        **{
            **candidate.__dict__,
            "contact_points": np.asarray([[1.0, 0.0, 0.0]]),
            "contact_normals": np.asarray([[1.0, 0.0, 0.0]]),
            "contact_distances": np.asarray([0.0]),
        }
    )
    episode = DemonstrationEpisode(
        object_id="ycb:test",
        seed=0,
        candidate=candidate,
        arrays={
            "object_position": np.asarray([[2.0, 3.0, 4.0]]),
            "object_quaternion_wxyz": np.asarray(
                [[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]]
            ),
        },
        success=False,
        terminal_stage="verify",
    )
    points, normals = contact_points_world(episode, 0)
    np.testing.assert_allclose(points, [[2.0, 4.0, 4.0]], atol=1e-7)
    np.testing.assert_allclose(normals, [[0.0, 1.0, 0.0]], atol=1e-7)


def test_underactuated_affordance_prefers_enclosable_body_over_thin_rod() -> None:
    import trimesh

    cylinder = geometry_affordance(
        trimesh.creation.cylinder(radius=0.03, height=0.10),
        scale_to_meters=1.0,
    )
    rod = geometry_affordance(
        trimesh.creation.box(extents=[0.008, 0.008, 0.20]),
        scale_to_meters=1.0,
    )
    assert cylinder.eligible
    assert not rod.eligible
    assert cylinder.geometry_prior > rod.geometry_prior


def test_mesh_support_estimate_is_not_a_hard_physical_gate() -> None:
    import trimesh

    box = geometry_affordance(
        trimesh.creation.box(extents=[0.05, 0.06, 0.08]),
        scale_to_meters=1.0,
    )
    sphere = geometry_affordance(
        trimesh.creation.icosphere(radius=0.035),
        scale_to_meters=1.0,
    )

    assert box.tipping_angle_deg > 20.0
    assert box.initial_stability > sphere.initial_stability
    assert sphere.eligible


def test_measured_initial_pose_stability_rejects_motion_after_placement() -> None:
    positions = np.zeros((5, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.012, len(positions))
    quaternions = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (len(positions), 1))
    velocities = np.zeros((len(positions), 6), dtype=np.float64)

    result = initial_pose_stability_from_trajectory(
        positions, quaternions, velocities, timestep=0.1
    )

    assert not result.stable
    assert result.horizontal_displacement_m == pytest.approx(0.012)


def test_measured_initial_pose_stability_accepts_quiet_settling() -> None:
    positions = np.zeros((5, 3), dtype=np.float64)
    positions[:, 2] = np.linspace(0.002, 0.0, len(positions))
    quaternions = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (len(positions), 1))
    velocities = np.zeros((len(positions), 6), dtype=np.float64)

    result = initial_pose_stability_from_trajectory(
        positions, quaternions, velocities, timestep=0.1
    )

    assert result.stable
    assert result.settled


def test_contact_velocity_chatter_is_reported_without_calling_it_a_tip() -> None:
    positions = np.zeros((5, 3), dtype=np.float64)
    quaternions = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (len(positions), 1))
    velocities = np.zeros((len(positions), 6), dtype=np.float64)
    velocities[-3:, 5] = 0.2

    result = initial_pose_stability_from_trajectory(
        positions, quaternions, velocities, timestep=0.1
    )

    assert result.stable
    assert not result.settled


def test_dynamic_uas_requires_measured_contact_and_robustness() -> None:
    import trimesh

    geometry = geometry_affordance(
        trimesh.creation.cylinder(radius=0.035, height=0.08),
        scale_to_meters=1.0,
    )
    arrays = {
        "stage": np.asarray([4, 4, 4, 4]),
        "robot_object_digit_contact_count": np.asarray(
            [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 1, 0, 1]]
        ),
    }
    adaptive = adaptive_contact_score(arrays)
    assert adaptive is not None and adaptive > 0.7
    assert 0.0 <= complete_uas(geometry, adaptive_contact=adaptive, robustness=0.5) <= 1.0
    assert benchmark_eligible(
        geometry,
        nominal_success=True,
        adaptive_contact=adaptive,
        robustness=0.8,
    )
    assert not benchmark_eligible(
        geometry,
        nominal_success=False,
        adaptive_contact=1.0,
        robustness=1.0,
    )


def test_execution_config_rejects_invalid_preload() -> None:
    with pytest.raises(ValueError, match="finger_preload"):
        ExecutionConfig(finger_preload=0.41).validate()


def test_seed_config_rejects_negative_enclosure_prior_count() -> None:
    with pytest.raises(ValueError, match="enclosure_prior_count"):
        SeedConfig(enclosure_prior_count=-1).validate()


def test_grasp_approach_stays_open_and_preload_is_applied_only_at_close() -> None:
    approach, closed = grasp_hand_targets(
        np.asarray([0.20, 0.30, 0.40, 0.50, 0.70, 0.60]),
        ExecutionConfig(finger_preload=0.10, thumb_grasp_preload=0.15),
    )

    np.testing.assert_allclose(approach, OPEN_FRACTIONS)
    np.testing.assert_allclose(closed, [0.30, 0.40, 0.50, 0.60, 0.70, 0.75])


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="optional ManiSkill assets are absent")
def test_object_surface_sampling_is_seed_deterministic() -> None:
    first = load_object_geometry("ycb:002_master_chef_can", surface_points=256, seed=19)
    second = load_object_geometry("ycb:002_master_chef_can", surface_points=256, seed=19)

    assert np.array_equal(first.surface_points, second.surface_points)
    assert np.array_equal(first.surface_normals, second.surface_normals)
    halfspace_values = first.vertices @ first.plane_normals.T - first.plane_offsets
    per_part = [
        halfspace_values[:, start:stop].max(axis=1)
        for start, stop in zip(
            first.plane_part_offsets[:-1], first.plane_part_offsets[1:], strict=True
        )
    ]
    assert float(np.stack(per_part, axis=1).min(axis=1).max()) < 2e-6
    np.testing.assert_allclose(
        first.bounds[1] - first.bounds[0],
        [0.075435, 0.075435, 0.103723],
        rtol=2e-3,
    )


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="optional ManiSkill assets are absent")
def test_egad_legacy_manifest_units_are_millimetres() -> None:
    geometry = load_object_geometry("egad:A0", surface_points=128, seed=3)
    horizontal = float(np.max(np.ptp(geometry.vertices, axis=0)[:2]))
    assert 0.074 < horizontal <= 0.0751
