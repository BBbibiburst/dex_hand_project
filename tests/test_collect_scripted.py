from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

from apps import collect_scripted_lerobot
from apps.collect_scripted_lerobot import _yaw_from_quaternion
from apps.collect_scripted_lerobot import (
    _evaluation_target_met,
    _rotated_candidate_indices,
    _successful_diversity,
)
from source.scripted.lift import LiftStrategy, _mesh_symmetry_yaws_from_vertices


def test_yaw_from_quaternion() -> None:
    angle = 1.25
    quaternion = np.asarray([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    assert np.isclose(_yaw_from_quaternion(quaternion), angle)


def test_lift_strategy_accepts_explicit_grasp_config() -> None:
    path = Path("configs/grasps/dex_hand/example.json")
    strategy = LiftStrategy(grasp_config_path=path)
    assert strategy.grasp_config_override == path


def test_mesh_symmetry_rejects_invalid_quarter_turn_for_box() -> None:
    vertices = np.asarray(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-2.0, 2.0) for z in (-3.0, 3.0)]
    )

    yaws = _mesh_symmetry_yaws_from_vertices(vertices)

    assert any(np.isclose(yaw, np.pi) for yaw in yaws)
    assert not any(np.isclose(yaw, np.pi / 2.0) for yaw in yaws)


def test_candidate_priority_rotates_across_randomized_seeds() -> None:
    assert _rotated_candidate_indices(0, 4) == (0, 1, 2, 3)
    assert _rotated_candidate_indices(1, 4) == (1, 2, 3, 0)
    assert _rotated_candidate_indices(5, 4) == (1, 2, 3, 0)


def test_coverage_resume_only_skips_objects_that_reached_target() -> None:
    assert _evaluation_target_met({"successes": 20}, coverage_search=True, target=20)
    assert not _evaluation_target_met({"successes": 3}, coverage_search=True, target=20)
    assert _evaluation_target_met({"successes": 0}, coverage_search=False, target=20)


def test_successful_diversity_counts_only_saved_successes() -> None:
    trials = [
        {
            "success": True,
            "seed": 10,
            "candidate_index": 0,
            "initial_position": [0.0, 0.0, 0.1],
            "initial_yaw": -3.0,
        },
        {
            "success": False,
            "seed": 11,
            "candidate_index": 1,
            "initial_position": [1.0, 1.0, 0.1],
            "initial_yaw": 0.0,
        },
        {
            "success": True,
            "seed": 12,
            "candidate_index": 2,
            "initial_position": [0.2, 0.1, 0.1],
            "initial_yaw": 0.2,
        },
    ]

    diversity = _successful_diversity(trials, candidate_count=4)

    assert diversity["unique_successful_seeds"] == 2
    assert diversity["unique_successful_candidates"] == 2
    assert diversity["candidate_coverage_rate"] == 0.5
    assert np.allclose(diversity["initial_position_span_xyz"], [0.2, 0.1, 0.0])
    assert diversity["initial_yaw_bins"] == 2


def test_catalog_evaluation_writes_per_object_success_rate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "grasps.json"
    output = tmp_path / "lift.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "parameters": {"validation_semantics": "trajectory-hold-v2"},
                "objects": [
                    {
                        "object_id": "ycb:test",
                            "status": "trajectory_stable",
                            "config": "grasp.json",
                            "robot_lift": {"robot_lift_verified": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(collect_scripted_lerobot, "lift_object_ids", lambda: ("ycb:test",))
    monkeypatch.setattr(collect_scripted_lerobot, "_make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        collect_scripted_lerobot, "create_strategy", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(collect_scripted_lerobot, "scripted_grasp_search_options", lambda args: {})
    monkeypatch.setattr(
        collect_scripted_lerobot,
        "_evaluate_episode",
        lambda env, strategy, *, seed, max_steps: {
            "seed": seed,
            "success": seed % 2 == 0,
            "final_phase": "verify",
            "initial_position": [0.0, 0.0, 0.1],
            "initial_yaw": 0.0,
        },
    )
    args = SimpleNamespace(
        task="lift",
        trials_per_object=2,
        coverage_search=False,
        target_successes_per_object=1,
        max_coverage_seeds=30,
        max_coverage_candidates=16,
        no_tactile=False,
        limit=None,
        grasp_benchmark_report=source,
        object_ids=None,
        dataset="all",
        evaluation_output=output,
        resume_evaluation=False,
        seed=10,
        max_steps=20,
        fps=20,
    )

    collect_scripted_lerobot._run_catalog_evaluation(args)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["completed_objects"] == 1
    assert report["summary"]["total_episodes"] == 2
    assert report["objects"][0]["success_rate"] == 0.5


def test_coverage_resume_continues_partial_object_with_unused_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    grasp = tmp_path / "grasp.json"
    grasp.write_text('{"trajectory_stable_candidates": []}', encoding="utf-8")
    source = tmp_path / "grasps.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "parameters": {"validation_semantics": "trajectory-hold-v2"},
                "objects": [
                    {
                        "object_id": "ycb:test",
                            "status": "trajectory_stable",
                            "config": str(grasp),
                            "robot_lift": {"robot_lift_verified": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "lift.json"
    parameters = {
        "task": "lift",
        "trials_per_object": 10,
        "coverage_search": True,
        "target_successes_per_object": 2,
        "max_coverage_seeds": 3,
        "max_coverage_candidates": 1,
        "one_success_per_seed": True,
        "candidate_order": "rotated_by_seed",
        "seed": 10,
        "max_steps": 20,
        "fps": 20,
        "dataset": "all",
        "object_ids": None,
        "limit": None,
    }
    first_trial = {
        "seed": 10,
        "success": True,
        "final_phase": "verify",
        "initial_position": [0.0, 0.0, 0.1],
        "initial_yaw": 0.0,
        "candidate_index": 0,
    }
    output.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "source_grasp_report": str(source),
                "parameters": parameters,
                "objects": [
                    {
                        "object_id": "ycb:test",
                        "successes": 1,
                        "trials": 1,
                        "episodes": [first_trial],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(collect_scripted_lerobot, "lift_object_ids", lambda: ("ycb:test",))
    monkeypatch.setattr(
        collect_scripted_lerobot,
        "_make_env",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        collect_scripted_lerobot, "create_strategy", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(collect_scripted_lerobot, "scripted_grasp_search_options", lambda args: {})

    def evaluate(env, strategy, *, seed, max_steps, frame_callback=None):
        _ = env, strategy, max_steps, frame_callback
        calls.append(seed)
        return {
            "seed": seed,
            "success": True,
            "final_phase": "verify",
            "initial_position": [0.1, 0.0, 0.1],
            "initial_yaw": 0.5,
        }

    monkeypatch.setattr(collect_scripted_lerobot, "_evaluate_episode", evaluate)
    args = SimpleNamespace(
        task="lift",
        trials_per_object=10,
        coverage_search=True,
        target_successes_per_object=2,
        max_coverage_seeds=3,
        max_coverage_candidates=1,
        no_tactile=False,
        limit=None,
        grasp_benchmark_report=source,
        object_ids=None,
        dataset="all",
        evaluation_output=output,
        resume_evaluation=True,
        seed=10,
        max_steps=20,
        fps=20,
        dry_run=True,
    )

    collect_scripted_lerobot._run_catalog_evaluation(args)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [11]
    assert report["objects"][0]["successes"] == 2
    assert report["objects"][0]["trials"] == 2
    assert report["objects"][0]["coverage_success"] is True
