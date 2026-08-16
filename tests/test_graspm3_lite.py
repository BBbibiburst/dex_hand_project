"""Contracts for the GraspM3-lite multimode temporal search."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apps.run_graspm3_lite_single import _prepare_output
from apps.train_grasp_primitive_rl import _primitive_names
from source.rl.grasp_edit.graspm3_lite import (
    TEMPORAL_PARAMETER_DIM,
    GraspM3LiteConfig,
    TemporalBatch,
    TemporalCandidate,
    TemporalCEMSearch,
    mode_close_alpha,
    normalized_sigmoid,
)
from source.rl.grasp_edit.primitives import (
    available_grasp_primitives,
    resolve_grasp_primitives,
)
from source.rl.residual.trajectory import ResidualTrajectory


def test_all_macro_grasp_modes_and_compatibility_aliases() -> None:
    modes = available_grasp_primitives()
    assert modes == (
        "wrap",
        "pinch",
        "tripod",
        "spherical",
        "hook",
        "cradle",
        "lateral",
        "table_assisted",
    )
    assert tuple(item.name for item in resolve_grasp_primitives("all")) == modes
    assert tuple(item.name for item in resolve_grasp_primitives("power_wrap,support")) == (
        "wrap",
        "cradle",
    )
    assert _primitive_names("wrap,pinch,support,hook") == (
        "wrap",
        "pinch",
        "cradle",
        "hook",
    )


def test_temporal_batch_validates_mode_ids() -> None:
    batch = TemporalBatch(
        np.zeros((4, TEMPORAL_PARAMETER_DIM), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=bool),
        np.arange(4, dtype=np.int64),
    )
    batch.validate(population_size=4, template_count=1, mode_count=4)

    invalid = TemporalBatch(
        batch.parameters,
        batch.template_ids,
        batch.reference_mask,
        np.array([0, 1, 2, 4], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="mode_ids"):
        invalid.validate(population_size=4, template_count=1, mode_count=4)


def test_temporal_profiles_have_normalized_endpoints_and_worldwise_close_power() -> None:
    progress = np.linspace(0.0, 1.0, 11)
    profile = normalized_sigmoid(progress, center=0.55, width=0.12)
    assert profile[0] == pytest.approx(0.0)
    assert profile[-1] == pytest.approx(1.0)
    assert np.all(np.diff(profile) >= 0.0)

    alpha = torch.full((3, 6), 0.5)
    powered = mode_close_alpha(alpha, torch.tensor([0.5, 1.0, 2.0]))
    assert powered.shape == (3, 6)
    assert torch.allclose(powered[0], torch.full((6,), 0.5**0.5))
    assert torch.allclose(powered[1], torch.full((6,), 0.5))
    assert torch.allclose(powered[2], torch.full((6,), 0.25))


def test_cem_stratifies_all_modes_and_keeps_reference_schedule() -> None:
    config = GraspM3LiteConfig(num_envs=17, population_size=17, iterations=1)
    fake_env = SimpleNamespace(
        templates=(object(), object(), object()),
        mode_count=8,
        mode_names=available_grasp_primitives(),
        reference_mode_id=0,
        parameter_bounds_low=np.zeros(TEMPORAL_PARAMETER_DIM, dtype=np.float32),
        parameter_bounds_high=np.ones(TEMPORAL_PARAMETER_DIM, dtype=np.float32),
    )
    search = TemporalCEMSearch(fake_env, config, seed=7)
    batch = search._sample()

    assert set(batch.mode_ids[: fake_env.mode_count]) == set(range(fake_env.mode_count))
    assert int(batch.reference_mask.sum()) == 2
    reference_indices = np.flatnonzero(batch.reference_mask)
    assert np.array_equal(batch.template_ids[reference_indices], np.arange(2))
    assert np.all(batch.mode_ids[reference_indices] == fake_env.reference_mode_id)


def test_population_must_fit_modes_and_reference() -> None:
    with pytest.raises(ValueError, match="reference schedule"):
        GraspM3LiteConfig(num_envs=8, population_size=8).validate()


def test_verification_pool_keeps_one_non_reference_candidate_per_mode() -> None:
    mode_names = ("wrap", "pinch", "cradle")
    config = GraspM3LiteConfig(
        num_envs=8,
        population_size=8,
        iterations=1,
        verification_candidates=3,
        grasp_modes=mode_names,
    )
    fake_env = SimpleNamespace(
        templates=(object(),),
        mode_count=3,
        parameter_bounds_low=np.zeros(TEMPORAL_PARAMETER_DIM, dtype=np.float32),
        parameter_bounds_high=np.ones(TEMPORAL_PARAMETER_DIM, dtype=np.float32),
    )
    search = TemporalCEMSearch(fake_env, config, seed=1)

    def candidate(mode_id: int, score: float, *, reference: bool = False):
        trajectory = ResidualTrajectory(
            object_id="test",
            source_manifest="test",
            start_stage="approach",
            action_mode="test",
            residual_actions=np.zeros((1, 1), dtype=np.float32),
            controls=np.zeros((1, 1), dtype=np.float32),
            initial_qpos=np.zeros(1, dtype=np.float32),
            initial_qvel=np.zeros(1, dtype=np.float32),
            success=False,
            episode_return=score,
            metadata={
                "template_id": 0,
                "temporal_parameters": [float(mode_id), score],
            },
        )
        return TemporalCandidate(
            trajectory=trajectory,
            score=score,
            mjwarp_success=False,
            reference_schedule=reference,
            mode_id=mode_id,
            mode_name=mode_names[mode_id],
        )

    selected = search._select_verification_pool(
        [
            candidate(0, 100.0, reference=True),
            candidate(0, 10.0),
            candidate(1, 2.0),
            candidate(2, 1.0),
        ]
    )

    assert {item.mode_id for item in selected} == {0, 1, 2}
    assert all(not item.reference_schedule for item in selected)


def test_existing_single_object_result_requires_explicit_overwrite(tmp_path) -> None:
    output = tmp_path / "object"
    (output / "candidates" / "candidate_000").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="--overwrite-output"):
        _prepare_output(output, overwrite=False)

    _prepare_output(output, overwrite=True)
    assert output.is_dir()
    assert not (output / "candidates").exists()
