"""Contracts and architecture checks for the independent UltraDexGrasp path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from source.ultradexgrasp.catalog import MANIFEST_PATH, load_object_geometry
from source.ultradexgrasp.affordance import (
    adaptive_contact_score,
    benchmark_eligible,
    complete_uas,
    geometry_affordance,
)
from source.ultradexgrasp.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from source.ultradexgrasp.contracts import DemonstrationEpisode, GraspCandidate
from source.ultradexgrasp.executor import (
    ExecutionConfig,
    ReachabilityResult,
    candidate_world_pose,
    grasp_hand_targets,
)
from tools.ultradexgrasp.generate import select_execution_candidates
from source.ultradexgrasp.hand_surrogate import OPEN_FRACTIONS
from source.ultradexgrasp.synthesizer import SynthesisConfig
from tools.ultradexgrasp.visualize_episode import contact_points_world


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

    assert config.synthesis.minimum_contact_fingers == 4
    assert config.execution.lift_height > 0.04
    assert config.surrogate_options["finger_degree"] == 7
    assert config.target_size is None
    assert config.maximum_horizontal_diameter == pytest.approx(0.075)
    assert config.synthesis.enclosure_prior_count >= 100
    assert config.synthesis.contact_partition_prior_count > 0


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


def test_dynamic_uas_requires_measured_contact_and_robustness() -> None:
    import trimesh

    geometry = geometry_affordance(
        trimesh.creation.icosphere(radius=0.035),
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


def test_synthesis_config_rejects_negative_enclosure_prior_count() -> None:
    with pytest.raises(ValueError, match="enclosure_prior_count"):
        SynthesisConfig(enclosure_prior_count=-1).validate()


def test_grasp_approach_stays_open_and_preload_is_applied_only_at_close() -> None:
    approach, closed = grasp_hand_targets(
        np.asarray([0.20, 0.30, 0.40, 0.50, 0.70, 0.60]),
        ExecutionConfig(finger_preload=0.10, thumb_grasp_preload=0.15),
    )

    np.testing.assert_allclose(approach, OPEN_FRACTIONS)
    np.testing.assert_allclose(closed, [0.30, 0.40, 0.50, 0.60, 0.70, 0.75])


def test_execution_selection_reserves_budget_for_diverse_enclosure_cells() -> None:
    ranked = []
    for index in range(8):
        base = _candidate()
        metrics = {"valid": 1.0}
        backend = "native-differentiable"
        if index >= 2:
            metrics.update(
                {
                    "valid": 0.0,
                    "enclosure_prior": 1.0,
                    "enclosure_depth_offset": (-0.01, 0.0, 0.01)[index % 3],
                    "enclosure_height_offset": (-0.008, 0.008)[index % 2],
                    "enclosure_lateral_offset": (-0.02, 0.02)[index % 2],
                    "enclosure_middle_delta": 0.08 * (index % 2),
                }
            )
            backend = "pca-centered-enclosure"
        candidate = GraspCandidate(
            **{
                **base.__dict__,
                "seed_index": index,
                "metrics": metrics,
                "backend": backend,
            }
        )
        ranked.append(ReachabilityResult(candidate, index * 0.01, 0.0, 0.0))

    selected = select_execution_candidates(
        tuple(ranked), limit=4, execution=ExecutionConfig()
    )

    assert len(selected) == 4
    assert sum(item.candidate.metrics.get("enclosure_prior", 0.0) for item in selected) == 3


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="optional ManiSkill assets are absent")
def test_object_surface_sampling_is_seed_deterministic() -> None:
    first = load_object_geometry("ycb:002_master_chef_can", surface_points=256, seed=19)
    second = load_object_geometry("ycb:002_master_chef_can", surface_points=256, seed=19)

    assert np.array_equal(first.surface_points, second.surface_points)
    assert np.array_equal(first.surface_normals, second.surface_normals)
    halfspace_values = first.vertices @ first.plane_normals.T - first.plane_offsets
    assert float(halfspace_values.max()) < 1e-8
    np.testing.assert_allclose(
        first.bounds[1] - first.bounds[0],
        [0.075000, 0.074889, 0.102540],
        rtol=2e-3,
    )


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="optional ManiSkill assets are absent")
def test_egad_legacy_manifest_units_are_millimetres() -> None:
    geometry = load_object_geometry("egad:A0", surface_points=128, seed=3)
    horizontal = float(np.max(np.ptp(geometry.vertices, axis=0)[:2]))
    assert 0.074 < horizontal <= 0.0751
