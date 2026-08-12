"""Cross-module contracts for public evaluation and configuration semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

from source.cli import robot_config
from source.envs.manipulation import registered_tasks
from source.evaluation.grasp_schema import (
    CURRENT_BENCHMARK_STATUSES,
    LEGACY_STABLE,
    TRAJECTORY_STABLE,
)
from source.grasping.standalone_validator import (
    DirectHoldValidationResult,
    TrajectoryValidationResult,
)
from source.scripted.lift import LiftStrategyState
from source.scripted import registered_strategies, strategy_task_name
from source.workflows.grasp_benchmark import GraspBenchmarkConfig, _write_report


def test_direct_and_trajectory_validation_results_are_distinct() -> None:
    direct_fields = DirectHoldValidationResult.__dataclass_fields__
    trajectory_fields = TrajectoryValidationResult.__dataclass_fields__

    assert "direct_hold_stable" in direct_fields
    assert "trajectory_hold_stable" not in direct_fields
    assert "trajectory_hold_stable" in trajectory_fields
    assert "trajectory_collision_free" in trajectory_fields
    assert "direct_hold_stable" not in trajectory_fields


def test_current_statuses_do_not_publish_legacy_stable() -> None:
    assert TRAJECTORY_STABLE in CURRENT_BENCHMARK_STATUSES
    assert "stable" not in CURRENT_BENCHMARK_STATUSES
    assert LEGACY_STABLE not in CURRENT_BENCHMARK_STATUSES


def test_benchmark_writer_uses_only_current_status_semantics(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    _write_report(
        output,
        args=GraspBenchmarkConfig(),
        selected=["ycb:test"],
        rows=[{"object_id": "ycb:test", "status": TRAJECTORY_STABLE}],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["objects"][0]["status"] == TRAJECTORY_STABLE
    assert "stable" not in payload["summary"]
    assert "stable_rate" not in payload["summary"]


def test_task_and_strategy_success_are_separate_public_states() -> None:
    state = LiftStrategyState()

    assert hasattr(state, "strategy_verified_success")
    assert not hasattr(state, "task_success")


def test_every_scripted_strategy_names_its_registered_task() -> None:
    tasks = set(registered_tasks())
    assert all(strategy_task_name(name) in tasks for name in registered_strategies())


def test_robot_config_preserves_tactile_setting_without_cli_override(monkeypatch) -> None:
    monkeypatch.setattr(
        robot_config,
        "load_robot_config",
        lambda path: {"enable_tactile_sensors": False},
    )
    args = SimpleNamespace(robot_config=None, no_tactile=False)

    assert robot_config.load_configured_robot(args)["enable_tactile_sensors"] is False


def test_no_tactile_cli_option_is_an_explicit_override(monkeypatch) -> None:
    monkeypatch.setattr(
        robot_config,
        "load_robot_config",
        lambda path: {"enable_tactile_sensors": True},
    )
    args = SimpleNamespace(robot_config=None, no_tactile=True)

    assert robot_config.load_configured_robot(args)["enable_tactile_sensors"] is False
