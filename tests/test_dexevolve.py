from __future__ import annotations

import numpy as np

from source.grasping.contracts import GraspCandidate
from source.grasping.budget import FORMAL_GENERATION_BUDGET
from source.grasping.dexevolve import (
    DexEvolveConfig,
    candidate_embedding,
    crossover_candidates,
    mutate_candidate,
)
from source.grasping.graspqp_adapter import GraspQPConfig, graspqp_available


def _candidate(seed: int = 1) -> GraspCandidate:
    return GraspCandidate(
        object_id="ycb:test",
        seed_index=seed,
        hand_translation=np.asarray([0.1, 0.0, 0.03]),
        hand_rotation_matrix=np.eye(3),
        actuator_fractions=np.asarray([0.2, 0.3, 0.4, 0.5, 0.8, 0.3]),
        contact_points=np.zeros((5, 3)),
        contact_normals=np.tile([1.0, 0.0, 0.0], (5, 1)),
        contact_distances=np.zeros(5),
    )


def test_dexevolve_mutation_preserves_candidate_contract() -> None:
    child = mutate_candidate(
        _candidate(),
        np.random.default_rng(4),
        DexEvolveConfig(mutation_probability=1.0),
        seed_index=3_000_001,
    )
    assert child.seed_index == 3_000_001
    assert child.backend.endswith("+dexevolve")
    assert np.all((child.actuator_fractions >= 0.0) & (child.actuator_fractions <= 1.0))
    assert candidate_embedding(child).shape == (12,)


def test_dexevolve_crossover_swaps_complete_hand_and_pose_blocks() -> None:
    first = _candidate(1)
    second = GraspCandidate(
        **{
            **first.__dict__,
            "seed_index": 2,
            "hand_translation": np.asarray([0.2, 0.1, 0.04]),
            "actuator_fractions": 1.0 - first.actuator_fractions,
        }
    )
    child = crossover_candidates(first, second, np.random.default_rng(9), seed_index=3_000_002)
    pose_is_first = np.array_equal(child.hand_translation, first.hand_translation)
    assert pose_is_first or np.array_equal(child.hand_translation, second.hand_translation)
    expected_hand = second.actuator_fractions if pose_is_first else first.actuator_fractions
    assert np.array_equal(child.actuator_fractions, expected_hand)


def test_official_graspqp_runtime_and_adapter_config_are_available() -> None:
    assert graspqp_available()
    config = GraspQPConfig()
    config.validate()
    assert config.closure_reserve == 0.0


def test_dexevolve_requires_sustained_opposed_contact() -> None:
    config = DexEvolveConfig()
    assert 0.0 < config.minimum_opposed_contact_fraction <= 1.0


def test_formal_generation_budget_is_shared_by_dexevolve_defaults() -> None:
    budget = FORMAL_GENERATION_BUDGET
    config = DexEvolveConfig()
    assert budget.dexevolve_evaluations == 216
    assert config.population_size == budget.population == 24
    assert config.offspring == budget.offspring == 12
    assert config.generations == budget.generations == 16
    assert budget.archive_candidates == 6
