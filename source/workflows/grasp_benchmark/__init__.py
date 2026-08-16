"""High-level, resumable catalogue grasp benchmark workflow."""

from importlib import import_module

from source.workflows.grasp_benchmark.candidates import (
    _append_diverse_candidates,
    _approach_bins,
    _candidate_is_diverse,
    _incomplete_attempt_key,
    _payload_after_robot_lift_attempts,
    _robot_candidate_precheck_key,
    _write_payload_atomic,
)
from source.workflows.grasp_benchmark.config import GraspBenchmarkConfig
from source.workflows.grasp_benchmark.reporting import (
    _attempt_satisfies_goal,
    _failure_reason,
    _format_duration,
    _pilot_stop_reason,
    _progress_timing,
    _write_report,
    _task_outcome_label,
    _task_scene_label,
)

__all__ = ["GraspBenchmarkConfig", "run_grasp_benchmark"]


def __getattr__(name: str):
    if name != "run_grasp_benchmark":
        raise AttributeError(name)
    value = getattr(import_module("source.workflows.grasp_benchmark.runner"), name)
    globals()[name] = value
    return value
