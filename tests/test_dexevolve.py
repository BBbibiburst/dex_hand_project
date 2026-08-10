from copy import deepcopy

import numpy as np

from source.grasping.dexevolve import EvolutionConfig, crossover, embedding, mutate


def payload() -> dict:
    return {
        "end_effector_name": "dex_hand",
        "hand_translation": [0.0, 0.0, 0.0],
        "hand_rotation_matrix": np.eye(3).tolist(),
        "hand_actuator_fractions": [0.5] * 6,
        "approach_hand_translations": [[0.0, 0.0, 0.0]],
        "grasp_hand_translations": [[0.0, 0.0, 0.0]],
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


def test_crossover_moves_trajectory_with_final_translation() -> None:
    first = payload()
    second = payload()
    second["hand_translation"] = [0.1, 0.0, 0.0]
    child = crossover(first, second, np.random.default_rng(1))
    np.testing.assert_allclose(
        child["approach_hand_translations"][0], child["hand_translation"]
    )
