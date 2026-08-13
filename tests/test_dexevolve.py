from copy import deepcopy

import numpy as np

from source.grasping.dexevolve import (
    EvolutionConfig,
    crossover,
    embedding,
    evaluate_population,
    mutate,
    table_clearance_metrics,
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


def test_robustness_defaults_are_bounded() -> None:
    config = EvolutionConfig()
    assert config.robustness_samples == 2
    assert config.robustness_translation_sigma == 0.0025
    assert config.robustness_orientation_sigma == 0.025


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
    np.testing.assert_allclose(child["approach_hand_translations"][0], child["hand_translation"])
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


def test_population_uses_batch_evaluator(monkeypatch) -> None:
    payloads = [{"candidate_id": index} for index in range(3)]
    monkeypatch.setattr("source.grasping.dexevolve.table_clearance_metrics", lambda _: None)

    class FakeEvaluator:
        def evaluate(self, values, *, seconds, settle_seconds):
            assert values == payloads
            assert seconds == 0.01
            assert settle_seconds == 0.4
            from source.grasping.standalone_validator import DirectHoldValidationResult

            return [
                DirectHoldValidationResult(
                    direct_hold_stable=True,
                    initial_displacement=0.0,
                    position_drift=0.0,
                    rotation_drift=0.0,
                    vertical_drop=0.0,
                    initial_contacts=2,
                    final_contacts=2,
                    simulated_seconds=seconds,
                )
                for _ in values
            ]

    results = evaluate_population(
        payloads,
        EvolutionConfig(jobs=1, seconds=0.01),
        batch_evaluator=FakeEvaluator(),
    )

    assert [result.payload["candidate_id"] for result in results] == [0, 1, 2]
    assert all(result.direct_hold_stable for result in results)


def test_table_clearance_uses_mutated_pose_and_full_trajectory(monkeypatch) -> None:
    value = payload()
    value["object_table_height"] = 0.0
    value["hand_translation"] = [0.0, 0.0, 0.02]
    value["approach_hand_translations"] = [[0.0, 0.0, 0.03]]
    value["grasp_hand_translations"] = [[0.0, 0.0, 0.004]]
    monkeypatch.setattr(
        "source.grasping.dexevolve._dex_hand_vertices",
        lambda fractions: np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]),
    )

    metrics = table_clearance_metrics(value)

    assert metrics is not None
    assert metrics["hand_table_clearance"] == 0.02
    assert metrics["approach_minimum_table_clearance"] == 0.03
    assert metrics["grasp_minimum_table_clearance"] == 0.004
    assert metrics["trajectory_minimum_table_clearance"] == 0.004
