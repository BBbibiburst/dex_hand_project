from copy import deepcopy

import numpy as np

from source.grasping.dexevolve import (
    EvolutionConfig,
    Individual,
    crossover,
    embedding,
    evolve,
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
    }


def test_embedding_matches_dexevolve_scaling() -> None:
    value = embedding(payload())
    assert value.shape == (13,)
    np.testing.assert_allclose(value[:6], 0.0, atol=1e-12)


def test_mutation_preserves_input_and_limits() -> None:
    source = payload()
    original = deepcopy(source)
    child = mutate(source, np.random.default_rng(0), EvolutionConfig())
    assert source == original
    assert np.all((np.asarray(child["hand_actuator_fractions"]) >= 0.0))
    assert np.all((np.asarray(child["hand_actuator_fractions"]) <= 1.0))
    assert 0.0 <= child["evolution_grip_preload"] <= 0.35


def test_crossover_moves_trajectory_with_final_translation() -> None:
    first = payload()
    second = payload()
    second["hand_translation"] = [0.1, 0.0, 0.0]
    child = crossover(first, second, np.random.default_rng(1))
    np.testing.assert_allclose(
        child["approach_hand_translations"][0], child["hand_translation"]
    )


def test_evolve_returns_a_long_horizon_elite_first(monkeypatch) -> None:
    calls = 0

    def fake_evaluate(payloads, config):
        nonlocal calls
        calls += 1
        if calls < 3:
            return [Individual(item, fitness=1000.0 - index) for index, item in enumerate(payloads)]
        return [
            Individual(item, fitness=10.0 - index, metrics={"simulated_seconds": 3.0})
            for index, item in enumerate(payloads)
        ]

    monkeypatch.setattr("source.grasping.dexevolve.evaluate_population", fake_evaluate)
    archive, _ = evolve(
        payload(),
        EvolutionConfig(
            population_size=2,
            offspring=1,
            generations=1,
            elite_count=2,
            jobs=1,
        ),
    )
    assert archive[0].fitness == 10.0
    assert archive[0].metrics == {"simulated_seconds": 3.0}
