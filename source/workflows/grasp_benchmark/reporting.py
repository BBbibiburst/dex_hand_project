"""Benchmark status classification, persistence, and human-readable reporting."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from source.evaluation.grasp_schema import (
    BENCHMARK_SCHEMA_VERSION, DIRECT_HOLD_ONLY, SEARCH_ERROR, TRAJECTORY_STABLE,
    UNSTABLE, VALIDATION_ERROR, VALIDATION_SEMANTICS,
)
from source.grasping.constants import GRASP_CONFIG_SCHEMA_VERSION, GRASP_SEARCH_STRATEGY
from source.workflows.grasp_benchmark.config import GraspBenchmarkConfig

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _task_outcome_label(row: dict, *, target_lift_candidates: int) -> str:
    """Return the user-facing result of a concrete robot Lift task search."""
    verified = int(row.get("lift_verified_candidate_count", 0))
    if verified >= target_lift_candidates:
        return "TASK_SOLVED"
    if verified > 0:
        return "TASK_PARTIAL"
    status = row.get("status")
    if status in {SEARCH_ERROR, VALIDATION_ERROR}:
        return str(status).upper()
    if status in {DIRECT_HOLD_ONLY, UNSTABLE}:
        return "GRASP_FAILED"
    lift = row.get("robot_lift") or {}
    phase = str(lift.get("final_phase") or "").lower()
    if phase == "precheck":
        return "TASK_INFEASIBLE"
    if phase in {"approach", "grasp", "lift", "verify"}:
        return f"LIFT_{phase.upper()}_FAILED"
    return "LIFT_NOT_EXECUTED"


def _task_scene_label(row: dict) -> str:
    """Format the concrete object placement selected by task-conditioned search."""
    scene = (row.get("robot_lift") or {}).get("task_scene")
    if not isinstance(scene, dict):
        return ""
    xy = scene.get("object_xy")
    if not isinstance(xy, list) or len(xy) != 2:
        return ""
    yaw_degrees = np.rad2deg(float(scene.get("object_yaw", 0.0)))
    pull_centimetres = 100.0 * float(scene.get("pull_toward_robot", 0.0))
    return (
        f"scene={int(scene.get('scene_index', 0))} "
        f"xy=({float(xy[0]):+.2f},{float(xy[1]):+.2f})m "
        f"yaw={yaw_degrees:+.0f}deg pull={pull_centimetres:.0f}cm "
    )


def _progress_timing(
    *, elapsed: float, completed: int, total: int, worker_count: int
) -> tuple[float, float | None]:
    """Return observed throughput time and a post-warmup ETA."""
    average = elapsed / max(1, completed)
    if completed < min(worker_count, total):
        return average, None
    return average, average * max(0, total - completed)


def _attempt_satisfies_goal(
    *, trajectory_hold_stable: bool, require_robot_lift_success: bool, robot_lift: dict | None
) -> bool:
    """Use Robot Lift as a retry gate only during explicit refinement."""
    return trajectory_hold_stable and (
        not require_robot_lift_success or bool((robot_lift or {}).get("robot_lift_verified"))
    )


def _failure_reason(row: dict) -> str | None:
    """Classify an incomplete result without discarding its detailed diagnostics."""
    status = row.get("status")
    if status == SEARCH_ERROR:
        return SEARCH_ERROR
    if status == VALIDATION_ERROR:
        return VALIDATION_ERROR
    if status == UNSTABLE:
        return "hold_unstable"
    if status == DIRECT_HOLD_ONLY:
        errors = " ".join(
            (row.get("evolution") or {}).get("trajectory_validation_errors") or []
        ).lower()
        if "table clearance" in errors:
            return "trajectory_table_clearance"
        if "collides with the object" in errors:
            return "trajectory_object_collision"
        return "trajectory_validation_failed"
    lift = row.get("robot_lift")
    if lift is None or lift.get("robot_lift_verified"):
        return None
    reason = str(lift.get("precheck_reason") or "")
    if reason.startswith("robot_ik_unreachable"):
        return "robot_ik_unreachable"
    if lift.get("table_collision") or reason.startswith("robot_table_collision"):
        return "robot_table_collision"
    phase = str(lift.get("final_phase") or "unknown")
    return f"robot_lift_{phase}_failed"


def _pilot_stop_reason(
    rows: list[dict],
    *,
    minimum_results: int,
    minimum_lift_rate: float,
    maximum_repeated_failure: int,
) -> str | None:
    """Return a diagnostic reason when a pilot already shows systematic failure."""
    if len(rows) < minimum_results:
        return None
    lift_successes = sum(
        bool((row.get("robot_lift") or {}).get("robot_lift_verified")) for row in rows
    )
    lift_rate = lift_successes / len(rows)
    if lift_rate < minimum_lift_rate:
        return (
            f"lift_rate={lift_successes}/{len(rows)} ({lift_rate:.1%}) "
            f"below {minimum_lift_rate:.1%}"
        )
    recent_reasons = [_failure_reason(row) for row in rows[-maximum_repeated_failure:]]
    if recent_reasons and recent_reasons[0] is not None and len(set(recent_reasons)) == 1:
        return f"repeated_failure={recent_reasons[0]} count={len(recent_reasons)}"
    return None


def _report_parameters(args: "GraspBenchmarkConfig") -> dict:
    return {
        "dataset": args.dataset,
        "object_ids": args.object_ids,
        "limit": args.limit,
        "search_strategy": GRASP_SEARCH_STRATEGY,
        "grasp_schema_version": GRASP_CONFIG_SCHEMA_VERSION,
        "validation_semantics": VALIDATION_SEMANTICS,
        "validate_robot_lift": args.validate_robot_lift,
        "pilot": args.pilot,
        "pilot_min_results": args.pilot_min_results,
        "pilot_min_lift_rate": args.pilot_min_lift_rate,
        "pilot_max_repeated_failure": args.pilot_max_repeated_failure,
        "points": args.points,
        "joint_candidates": args.joint_candidates,
        "surface_anchors": args.surface_anchors,
        "rolls_per_anchor": args.rolls_per_anchor,
        "coarse_keep": args.coarse_keep,
        "top_k": args.top_k,
        "support_margin": args.support_margin,
        "generator": args.generator,
        "graspqp_iterations": args.graspqp_iterations,
        "evolve": args.evolve,
        "evolution_population": args.evolution_population,
        "evolution_offspring": args.evolution_offspring,
        "evolution_generations": args.evolution_generations,
        "evolution_jobs": args.evolution_jobs,
        "evolution_seconds": args.evolution_seconds,
        "evolution_backend": args.evolution_backend,
        "mjwarp_device": args.mjwarp_device,
        "mjwarp_batch_size": args.mjwarp_batch_size,
        "mjwarp_nconmax": args.mjwarp_nconmax,
        "mjwarp_njmax": args.mjwarp_njmax,
        "search_attempts": args.search_attempts,
        "target_lift_candidates": args.target_lift_candidates,
        "maximum_saved_candidates": args.maximum_saved_candidates,
        "maximum_robot_candidates_per_attempt": args.maximum_robot_candidates_per_attempt,
        "task_conditioned_search": args.task_conditioned_search,
        "task_scene_attempts": args.task_scene_attempts,
        "task_rotations_per_distance": args.task_rotations_per_distance,
        "task_pull_step": args.task_pull_step,
        "task_maximum_pull": args.task_maximum_pull,
        "maximum_object_seconds": args.maximum_object_seconds,
        "seed": args.seed,
        "target_size": args.target_size,
        "end_effector": args.end_effector,
        "seconds": args.seconds,
        "settle_seconds": args.settle_seconds,
        "grip_preload": args.grip_preload,
    }


def _write_report(
    path: Path,
    *,
    args: "GraspBenchmarkConfig",
    selected: list[str],
    rows: list[dict],
) -> None:
    generated = sum(row["status"] != SEARCH_ERROR for row in rows)
    stable = sum(row["status"] == TRAJECTORY_STABLE for row in rows)
    robot_lift_tested = sum(row.get("robot_lift") is not None for row in rows)
    robot_lift_verified = sum(
        bool((row.get("robot_lift") or {}).get("robot_lift_verified")) for row in rows
    )
    lift_verified_candidates = sum(int(row.get("lift_verified_candidate_count", 0)) for row in rows)
    lift_candidate_targets_met = sum(
        int(row.get("lift_verified_candidate_count", 0)) >= args.target_lift_candidates
        for row in rows
    )
    task_solved = sum(int(row.get("lift_verified_candidate_count", 0)) > 0 for row in rows)
    task_targets_met = lift_candidate_targets_met
    object_time_budgets_reached = sum(bool(row.get("object_time_budget_reached")) for row in rows)
    failure_reasons: dict[str, int] = {}
    for row in rows:
        reason = _failure_reason(row)
        if reason is not None:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    failed = [row["object_id"] for row in rows if row["status"] != TRAJECTORY_STABLE]
    incomplete_lift_archives = [
        row["object_id"]
        for row in rows
        if args.validate_robot_lift
        and int(row.get("lift_verified_candidate_count", 0)) < args.target_lift_candidates
    ]
    unsolved = [
        row["object_id"] for row in rows if int(row.get("lift_verified_candidate_count", 0)) == 0
    ]
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": _report_parameters(args),
        "summary": {
            "selected": len(selected),
            "completed": len(rows),
            "grasp_generated": generated,
            "trajectory_stable": stable,
            "generation_rate": generated / len(rows) if rows else 0.0,
            "trajectory_stable_rate": stable / len(rows) if rows else 0.0,
            "robot_lift_tested": robot_lift_tested,
            "robot_lift_verified": robot_lift_verified,
            "robot_lift_verified_rate": (
                robot_lift_verified / robot_lift_tested if robot_lift_tested else 0.0
            ),
            "lift_verified_candidates": lift_verified_candidates,
            "lift_candidate_targets_met": lift_candidate_targets_met,
            "task_solved": task_solved,
            "task_success_rate": task_solved / len(rows) if rows else 0.0,
            "task_targets_met": task_targets_met,
            "unsolved_object_ids": unsolved,
            "object_time_budgets_reached": object_time_budgets_reached,
            "failure_reasons": failure_reasons,
            "failed_object_ids": failed,
            "incomplete_lift_archive_object_ids": incomplete_lift_archives,
        },
        "objects": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_completed(path: Path, args: "GraspBenchmarkConfig") -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"Cannot resume unsupported report {path}.")
    parameters = payload.get("parameters")
    expected = _report_parameters(args)
    if parameters != expected:
        raise ValueError(
            f"Cannot resume {path} with different parameters. "
            f"stored={parameters}, requested={expected}"
        )
    return list(payload.get("objects", []))
