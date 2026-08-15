"""Contracts and architecture checks for the independent UltraDexGrasp path."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from source.ultradexgrasp.catalog import MANIFEST_PATH, load_object_geometry
from source.ultradexgrasp.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from source.ultradexgrasp.contracts import DemonstrationEpisode, GraspCandidate
from source.ultradexgrasp.executor import (
    ExecutionConfig,
    candidate_world_pose,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_execution_config_rejects_invalid_preload() -> None:
    with pytest.raises(ValueError, match="finger_preload"):
        ExecutionConfig(finger_preload=0.3).validate()


def test_ultradexgrasp_source_does_not_import_legacy_grasping() -> None:
    violations = []
    for path in (PROJECT_ROOT / "source" / "ultradexgrasp").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            violations.extend(
                f"{path.name}: {module}"
                for module in modules
                if module.startswith("source.grasping")
            )
    assert not violations


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="optional ManiSkill assets are absent")
def test_object_surface_sampling_is_seed_deterministic() -> None:
    first = load_object_geometry("ycb:002_master_chef_can", surface_points=256, seed=19)
    second = load_object_geometry("ycb:002_master_chef_can", surface_points=256, seed=19)

    assert np.array_equal(first.surface_points, second.surface_points)
    assert np.array_equal(first.surface_normals, second.surface_normals)
    halfspace_values = first.vertices @ first.plane_normals.T - first.plane_offsets
    assert float(halfspace_values.max()) < 1e-8
