"""Regression tests for geometry-aware residual BC and verification states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apps.run_grasp_il_rl_catalog import _write_summary
from source.rl.imitation.bc import (
    BC_SCHEMA_VERSION,
    BC_TARGET_TYPE,
    _hand_residual_fraction,
    _split_object_groups,
    load_bc_dataset,
)
from source.rl.imitation.verification import (
    EXPERT_POOL_REJECTED,
    EXPERT_POOL_VALID,
    EXPERT_PROFILE,
    FINAL_PROFILE,
    FINAL_REJECTED,
    FINAL_VERIFIED,
    verification_status,
)


def test_bc_target_is_expert_minus_coarse_in_actuator_range_units() -> None:
    low = np.asarray([-1.0, 0.0, -2.0], dtype=np.float32)
    high = np.asarray([1.0, 4.0, 2.0], dtype=np.float32)
    coarse = np.asarray([-0.5, 1.0, -1.0], dtype=np.float32)
    expert = np.asarray([0.5, 3.0, 1.0], dtype=np.float32)

    target = _hand_residual_fraction(expert, coarse, low, high)

    np.testing.assert_allclose(target, [0.5, 0.5, 0.5])


def test_bc_target_is_zero_when_no_coarser_sequence_exists() -> None:
    controls = np.asarray([0.1, 0.4, 0.8, 0.3, 0.6, 0.2], dtype=np.float32)

    target = _hand_residual_fraction(
        controls,
        controls,
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
    )

    np.testing.assert_array_equal(target, np.zeros(6, dtype=np.float32))


def test_object_group_split_never_separates_manifests_of_the_same_object() -> None:
    groups = np.asarray([0, 0, 1, 2, 0, 3, 1, 2, 3], dtype=np.int32)
    object_ids = ("object-a", "object-b", "object-c", "object-d")

    first = _split_object_groups(
        groups,
        object_ids,
        validation_fraction=0.25,
        seed=17,
    )
    second = _split_object_groups(
        groups,
        object_ids,
        validation_fraction=0.25,
        seed=17,
    )
    train_mask, validation_mask, training_ids, validation_ids = first

    np.testing.assert_array_equal(train_mask, second[0])
    np.testing.assert_array_equal(validation_mask, second[1])
    assert training_ids == second[2]
    assert validation_ids == second[3]
    assert set(training_ids).isdisjoint(validation_ids)
    assert set(training_ids) | set(validation_ids) == set(object_ids)
    assert len(validation_ids) == 1
    for group in np.unique(groups):
        indices = groups == group
        assert np.all(train_mask[indices]) or np.all(validation_mask[indices])


def _write_bc_dataset(path: Path, *, residual_offset: float = 0.0) -> None:
    coarse = np.asarray(
        [[-0.5, 0.0], [0.25, -0.25], [0.0, 0.5]],
        dtype=np.float32,
    )
    expert = np.asarray(
        [[0.5, 0.0], [0.75, 0.25], [-0.5, 1.0]],
        dtype=np.float32,
    )
    residual = 0.5 * (expert - coarse) + residual_offset
    np.savez_compressed(
        path,
        observations=np.zeros((3, 8), dtype=np.float32),
        coarse_reference_hand_actions=coarse,
        expert_hand_actions=expert,
        hand_residual_targets=residual,
        weights=np.ones(3, dtype=np.float32),
        stages=np.zeros(3, dtype=np.int16),
        object_group_indices=np.asarray([0, 0, 1], dtype=np.int32),
        expert_indices=np.asarray([0, 0, 1], dtype=np.int32),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": BC_SCHEMA_VERSION,
                "target_type": BC_TARGET_TYPE,
                "dataset": {
                    "object_ids": ["object-a", "object-b"],
                    "observation_schema": {
                        "feature_slices": {"coarse_reference_hand": [0, 2]}
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_bc_dataset_persists_coarse_expert_and_residual_sequences(tmp_path: Path) -> None:
    dataset = tmp_path / "bc.npz"
    _write_bc_dataset(dataset)

    arrays, metadata = load_bc_dataset(dataset)

    assert metadata["target_type"] == BC_TARGET_TYPE
    assert {
        "coarse_reference_hand_actions",
        "expert_hand_actions",
        "hand_residual_targets",
    }.issubset(arrays)


def test_bc_dataset_rejects_inconsistent_residual_targets(tmp_path: Path) -> None:
    dataset = tmp_path / "bc.npz"
    _write_bc_dataset(dataset, residual_offset=0.1)

    with pytest.raises(ValueError, match="expert minus coarse"):
        load_bc_dataset(dataset)


@pytest.mark.parametrize(
    ("profile", "success", "expected"),
    (
        (EXPERT_PROFILE, True, EXPERT_POOL_VALID),
        (EXPERT_PROFILE, False, EXPERT_POOL_REJECTED),
        (FINAL_PROFILE, True, FINAL_VERIFIED),
        (FINAL_PROFILE, False, FINAL_REJECTED),
    ),
)
def test_verification_profiles_emit_disjoint_public_states(
    profile: str,
    success: bool,
    expected: str,
) -> None:
    assert verification_status(profile, success) == expected
    assert EXPERT_POOL_VALID != FINAL_VERIFIED


def test_catalog_summary_does_not_count_expert_admission_as_final_verification(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path,
        ["object-a", "object-b"],
        {
            "object-a": {"object_id": "object-a", "status": EXPERT_POOL_VALID},
            "object-b": {"object_id": "object-b", "status": FINAL_VERIFIED},
        },
    )

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["expert_pool_valid"] == 1
    assert payload["final_verified"] == 1
    assert payload["verified_total"] == 1
