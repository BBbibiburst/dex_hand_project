"""CLI preset and benchmark progress formatting contracts."""

from tools.grasping.benchmark_catalog import (
    GIB,
    _apply_full_pipeline_preset,
    _recommended_parallelism,
)
from source.workflows.grasp_benchmark import (
    _attempt_satisfies_goal,
    _append_diverse_candidates,
    _approach_bins,
    _candidate_is_diverse,
    _failure_reason,
    _format_duration,
    _incomplete_attempt_key,
    _pilot_stop_reason,
    _payload_after_robot_lift_attempts,
    _progress_timing,
    _robot_candidate_precheck_key,
    _task_outcome_label,
    _task_scene_label,
    _write_payload_atomic,
)


def test_full_pipeline_keeps_explicit_parallelism() -> None:
    values = {"jobs": 4, "evolution_jobs": 1}

    _apply_full_pipeline_preset(values, {"jobs", "evolution_jobs"})

    assert values["jobs"] == 4
    assert values["evolution_jobs"] == 1
    assert values["generator"] == "heuristic"


def test_full_pipeline_keeps_safe_cpu_evolution_backend() -> None:
    values = {"jobs": 4, "evolution_jobs": 1, "evolution_backend": "cpu"}

    _apply_full_pipeline_preset(values, set())

    assert values["evolution_backend"] == "cpu"
    assert values["retry_incomplete"] is True
    assert values["target_lift_candidates"] == 3
    assert values["maximum_object_seconds"] == 2700.0


def test_duration_format_is_compact() -> None:
    assert _format_duration(19.7) == "20s"
    assert _format_duration(125) == "2m05s"
    assert _format_duration(7500) == "2h05m"


def test_task_conditioned_progress_uses_lift_outcome_and_scene() -> None:
    solved = {
        "status": "trajectory_stable",
        "lift_verified_candidate_count": 3,
        "robot_lift": {
            "robot_lift_verified": True,
            "task_scene": {
                "scene_index": 4,
                "object_xy": [-0.05, 0.02],
                "object_yaw": 1.5708,
                "pull_toward_robot": 0.05,
            },
        },
    }
    assert _task_outcome_label(solved, target_lift_candidates=3) == "TASK_SOLVED"
    assert _task_outcome_label(solved, target_lift_candidates=4) == "TASK_PARTIAL"
    assert _task_scene_label(solved) == (
        "scene=4 xy=(-0.05,+0.02)m yaw=+90deg pull=5cm "
    )
    assert (
        _task_outcome_label(
            {
                "status": "trajectory_stable",
                "lift_verified_candidate_count": 0,
                "robot_lift": {"final_phase": "precheck"},
            },
            target_lift_candidates=1,
        )
        == "TASK_INFEASIBLE"
    )


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


def test_robot_precheck_ranking_prefers_executable_candidate() -> None:
    unreachable = {
        "precheck_passed": False,
        "table_collision": False,
        "maximum_ik_position_error": 0.4,
        "maximum_ik_orientation_error": 1.0,
    }
    reachable = {
        "precheck_passed": True,
        "table_collision": False,
        "maximum_ik_position_error": 0.001,
        "maximum_ik_orientation_error": 0.01,
    }
    assert _robot_candidate_precheck_key({}, 100.0, reachable) < (
        _robot_candidate_precheck_key({}, 1000.0, unreachable)
    )


def test_atomic_payload_writer_replaces_previous_result(tmp_path) -> None:
    path = tmp_path / "grasp.json"
    _write_payload_atomic(path, {"attempt": 1})
    _write_payload_atomic(path, {"attempt": 2})
    assert path.read_text(encoding="utf-8").strip().endswith("2\n}")
    assert not path.with_suffix(".json.tmp").exists()


def _candidate(x: float, angle: float = 0.0, joint: float = 0.2) -> dict:
    cosine, sine = __import__("math").cos(angle), __import__("math").sin(angle)
    return {
        "hand_translation": [x, 0.0, 0.0],
        "hand_rotation_matrix": [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "hand_actuator_fractions": [joint] * 6,
    }


def test_candidate_archive_deduplicates_pose_and_hand_shape() -> None:
    archive = [_candidate(0.0)]
    assert not _candidate_is_diverse(_candidate(0.005, angle=0.05, joint=0.21), archive)
    assert _candidate_is_diverse(_candidate(0.05), archive)
    assert _candidate_is_diverse(_candidate(0.0, angle=0.5), archive)
    assert _candidate_is_diverse(_candidate(0.0, joint=0.4), archive)

    _append_diverse_candidates(
        archive,
        [_candidate(0.005), _candidate(0.05), _candidate(0.10)],
        maximum=2,
    )
    assert len(archive) == 2


def test_approach_bin_coverage_is_explicit() -> None:
    candidates = [{"approach_bin": "front_level"}, {"approach_bin": "left_upper"}]
    covered = set().union(*(_approach_bins(candidate) for candidate in candidates))
    assert covered == {"front_level", "left_upper"}


def test_incomplete_attempt_prefers_later_robot_phase() -> None:
    precheck = {"robot_lift": {"final_phase": "precheck", "table_collision": False}}
    lift = {"robot_lift": {"final_phase": "lift", "table_collision": False}}
    assert _incomplete_attempt_key(lift) < _incomplete_attempt_key(precheck)


def test_pilot_stops_on_low_lift_rate_after_warmup() -> None:
    rows = [
        {"status": "trajectory_stable", "robot_lift": {"robot_lift_verified": False}}
        for _ in range(4)
    ]
    assert (
        _pilot_stop_reason(
            rows[:3],
            minimum_results=4,
            minimum_lift_rate=0.25,
            maximum_repeated_failure=3,
        )
        is None
    )
    assert "lift_rate=" in _pilot_stop_reason(
        rows,
        minimum_results=4,
        minimum_lift_rate=0.25,
        maximum_repeated_failure=3,
    )


def test_pilot_stops_on_repeated_failure_even_with_acceptable_lift_rate() -> None:
    rows = [
        {"status": "trajectory_stable", "robot_lift": {"robot_lift_verified": True}},
        {"status": "search_error"},
        {"status": "search_error"},
        {"status": "search_error"},
    ]
    reason = _pilot_stop_reason(
        rows,
        minimum_results=4,
        minimum_lift_rate=0.25,
        maximum_repeated_failure=3,
    )
    assert reason == "repeated_failure=search_error count=3"
