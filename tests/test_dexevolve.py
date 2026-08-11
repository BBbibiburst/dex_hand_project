from copy import deepcopy

import numpy as np

from source.grasping.dexevolve import (
    EvolutionConfig,
    crossover,
    embedding,
    evaluate_population,
    mutate,
)


def payload() -> dict:
    return {
        "end_effector_name": "dex_hand",
        "hand_translation": [0.0, 0.0, 0.0],
        "hand_rotation_matrix": np.eye(3).tolist(),
        "hand_actuator_fractions": [0.5] * 6,
        "approach_hand_translations": [[0.0, 0.0, 0.0]],
        "grasp_hand_translations": [[0.0, 0.0, 0.0]],
        "approach_hand_rotation_matrices": [np.eye(3).tolist()],
        "grasp_hand_rotation_matrices": [np.eye(3).tolist()],
        "grasp_hand_actuator_fractions": [[0.5] * 6],
    }


def test_embedding_matches_dexevolve_scaling() -> None:
    value = embedding(payload())
    assert value.shape == (12,)
    np.testing.assert_allclose(value[:6], 0.0, atol=1e-12)


def test_mutation_preserves_input_and_limits() -> None:
    source = payload()
    original = deepcopy(source)
    child = mutate(source, np.random.default_rng(0), EvolutionConfig())
    assert source == original
    assert np.all((np.asarray(child["hand_actuator_fractions"]) >= 0.0))
    assert np.all((np.asarray(child["hand_actuator_fractions"]) <= 1.0))
    np.testing.assert_allclose(
        child["grasp_hand_rotation_matrices"][-1], child["hand_rotation_matrix"]
    )
    np.testing.assert_allclose(
        child["grasp_hand_actuator_fractions"][-1], child["hand_actuator_fractions"]
    )


def test_crossover_moves_trajectory_with_final_translation() -> None:
    first = payload()
    second = payload()
    second["hand_translation"] = [0.1, 0.0, 0.0]
    child = crossover(first, second, np.random.default_rng(1))
    np.testing.assert_allclose(
        child["approach_hand_translations"][0], child["hand_translation"]
    )
    np.testing.assert_allclose(
        child["grasp_hand_rotation_matrices"][-1], child["hand_rotation_matrix"]
    )
    np.testing.assert_allclose(
        child["grasp_hand_actuator_fractions"][-1], child["hand_actuator_fractions"]
    )


def test_parallel_evaluation_preserves_submission_order() -> None:
    payloads = [{"candidate_id": index} for index in range(4)]
    results = evaluate_population(payloads, EvolutionConfig(jobs=2, seconds=0.01))
    assert [result.payload["candidate_id"] for result in results] == list(range(4))
