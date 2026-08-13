"""CLI preset and benchmark progress formatting contracts."""

from tools.grasping.benchmark_catalog import (
    GIB,
    _apply_full_pipeline_preset,
    _recommended_parallelism,
)
from source.workflows.grasp_benchmark import (
    _attempt_satisfies_goal,
    _failure_reason,
    _format_duration,
    _incomplete_attempt_key,
    _payload_after_robot_lift_attempts,
    _progress_timing,
)


def test_full_pipeline_keeps_explicit_parallelism() -> None:
    values = {"jobs": 4, "evolution_jobs": 1}

    _apply_full_pipeline_preset(values, {"jobs", "evolution_jobs"})

    assert values["jobs"] == 4
    assert values["evolution_jobs"] == 1
    assert values["generator"] == "graspqp"


def test_full_pipeline_keeps_safe_cpu_evolution_backend() -> None:
    values = {"jobs": 4, "evolution_jobs": 1, "evolution_backend": "cpu"}

    _apply_full_pipeline_preset(values, set())

    assert values["evolution_backend"] == "cpu"
    assert values["retry_incomplete"] is True


def test_duration_format_is_compact() -> None:
    assert _format_duration(19.7) == "20s"
    assert _format_duration(125) == "2m05s"
    assert _format_duration(7500) == "2h05m"


def test_eta_waits_for_one_full_worker_wave() -> None:
    average, eta = _progress_timing(elapsed=3600.0, completed=1, total=127, worker_count=8)
    assert average == 3600.0
    assert eta is None

    average, eta = _progress_timing(elapsed=4200.0, completed=8, total=127, worker_count=8)
    assert average == 525.0
    assert eta == 525.0 * 119


def test_robot_lift_failure_only_triggers_explicit_refinement_retry() -> None:
    failed_lift = {"robot_lift_verified": False}
    assert _attempt_satisfies_goal(
        trajectory_hold_stable=True,
        require_robot_lift_success=False,
        robot_lift=failed_lift,
    )
    assert not _attempt_satisfies_goal(
        trajectory_hold_stable=True,
        require_robot_lift_success=True,
        robot_lift=failed_lift,
    )


def test_parallelism_reserves_cpu_and_memory() -> None:
    assert _recommended_parallelism(cpu_count=32, available_memory_bytes=64 * GIB) == 8
    assert _recommended_parallelism(cpu_count=8, available_memory_bytes=7 * GIB) == 3
    assert _recommended_parallelism(cpu_count=2, available_memory_bytes=64 * GIB) == 1


def test_benchmark_failure_reason_classification() -> None:
    assert _failure_reason({"status": "search_error"}) == "search_error"
    assert (
        _failure_reason(
            {
                "status": "direct_hold_only",
                "evolution": {
                    "trajectory_validation_errors": [
                        "Grasp trajectory collides with the object via v3_base_link"
                    ]
                },
            }
        )
        == "trajectory_object_collision"
    )
    assert (
        _failure_reason(
            {
                "status": "trajectory_stable",
                "robot_lift": {
                    "robot_lift_verified": False,
                    "precheck_reason": "robot_ik_unreachable_waypoint_2",
                    "table_collision": False,
                },
            }
        )
        == "robot_ik_unreachable"
    )


def test_failed_robot_lift_restores_preferred_trajectory_payload() -> None:
    preferred = {"candidate": "trajectory_best"}
    attempted = {"candidate": "last_robot_failure"}

    assert (
        _payload_after_robot_lift_attempts(preferred, attempted, robot_lift_verified=False)
        == preferred
    )
    assert (
        _payload_after_robot_lift_attempts(preferred, attempted, robot_lift_verified=True)
        == attempted
    )


def test_incomplete_attempt_prefers_later_robot_phase() -> None:
    precheck = {"robot_lift": {"final_phase": "precheck", "table_collision": False}}
    lift = {"robot_lift": {"final_phase": "lift", "table_collision": False}}
    assert _incomplete_attempt_key(lift) < _incomplete_attempt_key(precheck)
