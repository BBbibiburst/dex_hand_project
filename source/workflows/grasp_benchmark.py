"""Search and physics-validate grasps for every catalogue object."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import queue
import time

import numpy as np

from source.envs.manipulation.object_catalog import object_ids
from source.evaluation.grasp_schema import (
    BENCHMARK_SCHEMA_VERSION,
    DIRECT_HOLD_ONLY,
    SEARCH_ERROR,
    TRAJECTORY_STABLE,
    UNSTABLE,
    VALIDATION_ERROR,
    VALIDATION_SEMANTICS,
)
from source.grasping.constants import (
    DEFAULT_GRIP_PRELOAD,
    GRASP_CONFIG_SCHEMA_VERSION,
    GRASP_SEARCH_STRATEGY,
)
from source.grasping.grasp_config_search import (
    grasp_benchmark_report_path,
    grasp_config_directory,
    grasp_config_name,
    replan_evolved_payload,
    search_grasp_config,
)
from source.grasping.standalone_validator import (
    validate_grasp_config,
    validate_grasp_payload_trajectory,
)
from source.grasping.dexevolve import (
    EvolutionConfig,
    evolve,
    mjwarp_available,
    table_clearance_metrics,
)
from source.runtime.progress import LiveWorkerProgress


_PROGRESS_QUEUE = None


def _init_progress_worker(progress_queue) -> None:
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue


def _emit_progress(
    object_id: str,
    phase: str,
    *,
    current: int | None = None,
    total: int | None = None,
    detail: str = "",
) -> None:
    if _PROGRESS_QUEUE is None:
        return
    _PROGRESS_QUEUE.put(
        {
            "worker": multiprocessing.current_process().name,
            "object_id": object_id,
            "phase": phase,
            "current": current,
            "total": total,
            "detail": detail,
        }
    )


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


def _payload_after_robot_lift_attempts(
    preferred_payload: dict,
    attempted_payload: dict,
    *,
    robot_lift_verified: bool,
) -> dict:
    """Publish a successful Lift candidate, otherwise restore the trajectory-first choice."""
    return dict(attempted_payload if robot_lift_verified else preferred_payload)


def _write_payload_atomic(path: Path, payload: dict) -> None:
    """Publish one grasp payload without exposing a partially-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _robot_candidate_precheck_key(
    payload: dict, individual_fitness: float, precheck: dict
) -> tuple:
    """Prioritize executable, collision-free and well-cleared robot candidates."""
    return (
        0 if precheck["precheck_passed"] else 1,
        1 if precheck["table_collision"] else 0,
        float(precheck["maximum_ik_position_error"]),
        float(precheck["maximum_ik_orientation_error"]),
        -float(payload.get("trajectory_minimum_table_clearance", 0.0)),
        -float(individual_fitness),
    )


def _candidate_is_diverse(
    candidate: dict,
    archive: list[dict],
    *,
    translation_threshold: float = 0.025,
    rotation_threshold: float = np.deg2rad(15.0),
    joint_threshold: float = 0.08,
) -> bool:
    """Reject near-identical wrist poses and hand shapes across search attempts."""
    translation = np.asarray(candidate["hand_translation"], dtype=np.float64)
    rotation = np.asarray(candidate["hand_rotation_matrix"], dtype=np.float64)
    joints = np.asarray(candidate["hand_actuator_fractions"], dtype=np.float64)
    for existing in archive:
        existing_translation = np.asarray(existing["hand_translation"], dtype=np.float64)
        existing_rotation = np.asarray(existing["hand_rotation_matrix"], dtype=np.float64)
        existing_joints = np.asarray(existing["hand_actuator_fractions"], dtype=np.float64)
        translation_distance = float(np.linalg.norm(translation - existing_translation))
        relative_trace = float(np.trace(rotation @ existing_rotation.T))
        rotation_distance = float(np.arccos(np.clip((relative_trace - 1.0) / 2.0, -1.0, 1.0)))
        joint_distance = float(np.sqrt(np.mean(np.square(joints - existing_joints))))
        if (
            translation_distance < translation_threshold
            and rotation_distance < rotation_threshold
            and joint_distance < joint_threshold
        ):
            return False
    return True


def _approach_bins(candidate: dict) -> set[str]:
    value = candidate.get("approach_bin")
    return {str(value)} if value else set()


def _append_diverse_candidates(
    archive: list[dict], candidates: list[dict], *, maximum: int
) -> None:
    for candidate in candidates:
        if len(archive) >= maximum:
            break
        if _candidate_is_diverse(candidate, archive):
            archive.append(dict(candidate))


def _incomplete_attempt_key(row: dict) -> tuple:
    """Rank failed attempts by progress toward an executable robot Lift."""
    lift = row.get("robot_lift") or {}
    phase_rank = {
        "precheck": 0,
        "approach": 1,
        "grasp": 2,
        "lift": 3,
        "verify": 4,
        "done": 5,
    }.get(str(lift.get("final_phase") or ""), -1)
    return (
        0 if lift.get("robot_lift_verified") else 1,
        -phase_rank,
        1 if lift.get("table_collision") else 0,
        float(row.get("vertical_drop", float("inf"))),
        float(row.get("position_drift", float("inf"))),
        float(row.get("rotation_drift", float("inf"))),
        -int(row.get("final_contacts", 0)),
    )


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


@dataclass
class GraspBenchmarkConfig:
    """Configuration for a catalogue-wide grasp search and validation run."""

    dataset: str = "all"
    object_ids: list[str] | None = None
    limit: int | None = None
    points: int = 2048
    joint_candidates: int = 128
    surface_anchors: int = 24
    rolls_per_anchor: int = 8
    coarse_keep: int = 24
    top_k: int = 8
    support_margin: float = 0.008
    generator: str = "heuristic"
    graspqp_iterations: int = 120
    evolve: bool = False
    evolution_population: int = 32
    evolution_offspring: int = 16
    evolution_generations: int = 20
    evolution_jobs: int = 4
    evolution_seconds: float = 1.5
    evolution_backend: str = "cpu"
    mjwarp_device: str = "cuda:0"
    mjwarp_batch_size: int = 32
    mjwarp_nconmax: int = 128
    mjwarp_njmax: int = 512
    evolution_dir: Path | None = None
    search_attempts: int = 3
    target_lift_candidates: int = 1
    maximum_saved_candidates: int = 24
    maximum_robot_candidates_per_attempt: int = 6
    task_conditioned_search: bool = False
    task_scene_attempts: int = 12
    task_rotations_per_distance: int = 4
    task_pull_step: float = 0.05
    task_maximum_pull: float = 0.15
    maximum_object_seconds: float = 2700.0
    seed: int = 0
    target_size: float = 0.09
    end_effector: str = "dex_hand"
    seconds: float = 3.0
    settle_seconds: float = 0.8
    grip_preload: float = DEFAULT_GRIP_PRELOAD
    jobs: int = 1
    reuse: bool = False
    validate_robot_lift: bool = False
    pilot: bool = False
    pilot_min_results: int = 4
    pilot_min_lift_rate: float = 0.25
    pilot_max_repeated_failure: int = 3
    resume: bool = False
    retry_incomplete: bool = False
    config_dir: Path | None = None
    output: Path | None = None


def _selected_ids(args: "GraspBenchmarkConfig") -> list[str]:
    available = object_ids(None if args.dataset == "all" else args.dataset)
    if args.object_ids:
        unknown = sorted(set(args.object_ids) - set(available))
        if unknown:
            raise ValueError(f"Objects outside selected catalogue: {unknown}")
        selected = list(dict.fromkeys(args.object_ids))
    else:
        selected = list(available)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        selected = selected[: args.limit]
    return selected


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


def _validate_config(
    path: Path,
    *,
    seconds: float,
    settle_seconds: float,
    grip_preload: float,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = validate_grasp_config(
        path,
        seconds=seconds,
        settle_seconds=settle_seconds,
        grip_preload=grip_preload,
    )
    metrics = asdict(result)
    metrics.update(
        {
            "table_clearance": payload.get("hand_table_clearance"),
            "orientation_roll_index": payload.get(
                "hand_orientation_roll_index",
                payload.get("hand_pca_axis_index"),
            ),
            "contact_distance_margin": payload.get(
                "hand_contact_distance_margin",
                payload.get("hand_robustness_margin"),
            ),
            "force_closure_residual": payload.get("hand_force_closure_residual"),
            "contacting_fingers": payload.get("hand_contacting_fingers"),
            "preload_weights": payload.get("hand_preload_weights"),
        }
    )
    return metrics


def _run_one(task: dict) -> dict:
    object_id = task["object_id"]
    _emit_progress(object_id, "TASK_SETUP", detail="creating concrete Lift scenes")
    started = time.monotonic()
    config_path = Path(task["config_path"])
    search_errors = []
    validation_errors = []
    best_unstable = None
    accumulated_trajectory_candidates: list[dict] = []
    lift_verified_candidates: list[dict] = []
    accumulated_robot_lift_attempts: list[dict] = []
    best_lift_result = None
    best_lift_payload = None
    phase_seconds = {
        "search": 0.0,
        "evolution": 0.0,
        "trajectory_replan_and_validation": 0.0,
        "robot_candidate_precheck": 0.0,
        "robot_lift_validation": 0.0,
    }
    for attempt in range(task["search_attempts"]):
        if time.monotonic() - started >= task["maximum_object_seconds"]:
            break
        attempt_scenes = [None]
        if task["task_conditioned_search"]:
            from source.grasping.robot_lift_validator import task_scene_schedule

            attempt_scenes = task_scene_schedule(
                seed=task["seed"] + attempt,
                scene_attempts=task["task_scene_attempts"],
                rotations_per_distance=task["task_rotations_per_distance"],
                pull_step=task["task_pull_step"],
                maximum_pull=task["task_maximum_pull"],
            )
        attempt_started = time.monotonic()
        try:
            reuse_this_attempt = task["reuse"] and attempt == 0 and config_path.is_file()
            if not reuse_this_attempt:
                _emit_progress(
                    object_id,
                    "GRASP_SEARCH",
                    detail=f"attempt={attempt + 1}/{task['search_attempts']} "
                    "GraspQP candidate generation",
                )
                search_kwargs = {
                    "object_id": object_id,
                    "output": config_path,
                    "points": task["points"],
                    "joint_candidates": task["joint_candidates"],
                    "surface_anchors": task["surface_anchors"],
                    "rolls_per_anchor": task["rolls_per_anchor"],
                    "coarse_keep": task["coarse_keep"],
                    "top_k": task["top_k"],
                    "support_margin": task["support_margin"],
                    "seed": task["seed"] + attempt,
                    "target_size": task["target_size"],
                    "end_effector_name": task["end_effector"],
                    "generator": task["generator"],
                    "graspqp_iterations": task["graspqp_iterations"],
                    "require_valid": not task["evolve"],
                    "publish_invalid": task["evolve"],
                }
                if task["task_conditioned_search"]:
                    from source.grasping.robot_lift_validator import (
                        RobotTaskCandidateFilter,
                    )

                    with RobotTaskCandidateFilter(
                        object_id,
                        attempt_scenes,
                        seed=task["seed"] + attempt,
                    ) as task_filter:
                        search_grasp_config(
                            **search_kwargs,
                            candidate_filter=task_filter,
                        )
                else:
                    search_grasp_config(**search_kwargs)
            if task["task_conditioned_search"]:
                generated_payload = json.loads(config_path.read_text(encoding="utf-8"))
                if not generated_payload.get("task_scene"):
                    raise RuntimeError(
                        "No generated grasp passed full-robot task precheck; "
                        "skipping DexEvolve for this search attempt"
                    )
            search_seconds = time.monotonic() - attempt_started
            phase_seconds["search"] += search_seconds
            _emit_progress(
                object_id,
                "GRASP_SEARCH_DONE",
                current=attempt + 1,
                total=task["search_attempts"],
            )
        except Exception as exc:
            search_errors.append(f"seed={task['seed'] + attempt}: {exc}")
            continue
        try:
            seed_metrics = None
            seed_payload = json.loads(config_path.read_text(encoding="utf-8"))
            if task["task_conditioned_search"] and seed_payload.get("task_scene"):
                selected_scene_index = int(seed_payload["task_scene"]["scene_index"])
                attempt_scenes = sorted(
                    attempt_scenes,
                    key=lambda scene: int(scene["scene_index"]) != selected_scene_index,
                )
            if seed_payload.get("hand_fit_success"):
                seed_metrics = _validate_config(
                    config_path,
                    seconds=task["seconds"],
                    settle_seconds=task["settle_seconds"],
                    grip_preload=task["grip_preload"],
                )

            evolution_summary = None
            validation_path = config_path
            if task["evolve"]:
                evolution_config = EvolutionConfig(
                    population_size=task["evolution_population"],
                    offspring=task["evolution_offspring"],
                    generations=task["evolution_generations"],
                    jobs=task["evolution_jobs"],
                    seconds=task["evolution_seconds"],
                    seed=task["seed"] + attempt,
                    backend=task["evolution_backend"],
                    mjwarp_device=task["mjwarp_device"],
                    mjwarp_batch_size=task["mjwarp_batch_size"],
                    mjwarp_nconmax=task["mjwarp_nconmax"],
                    mjwarp_njmax=task["mjwarp_njmax"],
                )
                evolution_started = time.monotonic()
                archive, history = evolve(
                    seed_payload,
                    evolution_config,
                    progress_callback=lambda current, total, summary: _emit_progress(
                        object_id,
                        "EVOLUTION",
                        current=current,
                        total=total,
                        detail=f"stable={summary['direct_hold_stable']} "
                        f"archive={summary['archive']}",
                    ),
                )
                phase_seconds["evolution"] += time.monotonic() - evolution_started
                best = archive[0]
                trajectory_candidates = []
                trajectory_errors = []
                trajectory_started = time.monotonic()
                for archive_index, individual in enumerate(archive):
                    _emit_progress(
                        object_id,
                        "TRAJECTORY_VALIDATION",
                        current=archive_index + 1,
                        total=len(archive),
                        detail=f"accepted={len(trajectory_candidates)}",
                    )
                    if time.monotonic() - started >= task["maximum_object_seconds"]:
                        break
                    if not individual.direct_hold_stable:
                        continue
                    candidate = dict(individual.payload)
                    candidate.pop("trajectory_stable_candidates", None)
                    candidate["direct_hold_stable"] = True
                    try:
                        candidate = replan_evolved_payload(
                            candidate,
                            seed=(task["seed"] + attempt * max(1, len(archive)) + archive_index),
                        )
                        replanned_clearance = table_clearance_metrics(candidate)
                        if (
                            replanned_clearance is None
                            or replanned_clearance["trajectory_minimum_table_clearance"]
                            < evolution_config.minimum_table_clearance
                        ):
                            raise ValueError("replanned trajectory violates table clearance")
                        candidate.update(replanned_clearance)
                        result = validate_grasp_payload_trajectory(
                            candidate,
                            seconds=task["seconds"],
                            settle_seconds=task["settle_seconds"],
                            grip_preload=task["grip_preload"],
                        )
                    except Exception as exc:
                        trajectory_errors.append(str(exc))
                        continue
                    if not result.trajectory_hold_stable:
                        continue
                    candidate.update(
                        trajectory_collision_free=True,
                        trajectory_hold_stable=True,
                        validation_stage="trajectory_hold_stable",
                        search_seed=task["seed"] + attempt,
                        search_attempt=attempt,
                    )
                    trajectory_candidates.append((candidate, result, individual))
                    if len(trajectory_candidates) >= 16:
                        break
                phase_seconds["trajectory_replan_and_validation"] += (
                    time.monotonic() - trajectory_started
                )
                robot_prechecks = []
                if (
                    trajectory_candidates
                    and task["validate_robot_lift"]
                    and not task["task_conditioned_search"]
                    and time.monotonic() - started < task["maximum_object_seconds"]
                ):
                    from source.grasping.robot_lift_validator import (
                        precheck_robot_lift_candidates,
                    )

                    robot_precheck_started = time.monotonic()
                    robot_prechecks = precheck_robot_lift_candidates(
                        object_id,
                        [candidate for candidate, _, _ in trajectory_candidates],
                        seed=task["seed"] + attempt,
                    )
                    phase_seconds["robot_candidate_precheck"] += (
                        time.monotonic() - robot_precheck_started
                    )
                    indexed_prechecks = {item["candidate_index"]: item for item in robot_prechecks}

                    def robot_candidate_key(index_and_candidate):
                        index, candidate = index_and_candidate
                        payload, result, individual = candidate
                        precheck = indexed_prechecks[index]
                        return _robot_candidate_precheck_key(
                            payload,
                            individual.fitness,
                            precheck,
                        )

                    ordered = sorted(enumerate(trajectory_candidates), key=robot_candidate_key)
                    trajectory_candidates = [candidate for _, candidate in ordered]
                    robot_prechecks = [indexed_prechecks[index] for index, _ in ordered]
                _append_diverse_candidates(
                    accumulated_trajectory_candidates,
                    [candidate for candidate, _, _ in trajectory_candidates],
                    maximum=task["maximum_saved_candidates"],
                )
                selected = trajectory_candidates[0] if trajectory_candidates else None
                selected_individual = selected[2] if selected else best
                published_payload = dict(selected[0] if selected else best.payload)
                published_payload.update(
                    evolution_refined=True,
                    direct_hold_stable=bool(selected_individual.direct_hold_stable),
                    trajectory_collision_free=bool(selected),
                    trajectory_hold_stable=bool(selected),
                    validation_stage=("trajectory_hold_stable" if selected else DIRECT_HOLD_ONLY),
                    trajectory_stable_candidates=[
                        candidate for candidate, _, _ in trajectory_candidates
                    ],
                )
                validation_path = Path(task["evolution_path"])
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                _write_payload_atomic(validation_path, published_payload)
                evolution_summary = {
                    "archive": len(archive),
                    "direct_hold_candidates": sum(item.direct_hold_stable for item in archive),
                    "trajectory_stable_candidates": len(trajectory_candidates),
                    "trajectory_validation_errors": trajectory_errors[:8],
                    "best_fitness": best.fitness,
                    "history": history,
                    "backend": history[-1].get("backend", task["evolution_backend"]),
                    "backend_fallback_error": history[-1].get("backend_fallback_error"),
                    "robot_prechecks": robot_prechecks,
                }
                metrics = (
                    asdict(selected[1])
                    if selected
                    else dict(best.metrics or {}, trajectory_hold_stable=False)
                )
                if selected:
                    for key in (
                        "hand_table_clearance",
                        "approach_minimum_table_clearance",
                        "grasp_minimum_table_clearance",
                        "trajectory_minimum_table_clearance",
                    ):
                        if key in selected[0]:
                            metrics[key] = selected[0][key]
            else:
                metrics = _validate_config(
                    validation_path,
                    seconds=task["seconds"],
                    settle_seconds=task["settle_seconds"],
                    grip_preload=task["grip_preload"],
                )
                if metrics["trajectory_hold_stable"]:
                    published_payload = json.loads(validation_path.read_text(encoding="utf-8"))
                    published_payload.update(
                        trajectory_collision_free=True,
                        trajectory_hold_stable=True,
                        validation_stage="trajectory_hold_stable",
                    )
                    temporary = validation_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(published_payload, indent=2), encoding="utf-8")
                    temporary.replace(validation_path)
            trajectory_hold_stable = bool(metrics["trajectory_hold_stable"])
            robot_lift = None
            robot_lift_attempts = []
            if trajectory_hold_stable and task["validate_robot_lift"]:
                from source.grasping.robot_lift_validator import (
                    precheck_robot_lift_candidates,
                    precheck_robot_lift_task_scenes,
                    validate_robot_lift,
                )

                robot_candidates = (
                    trajectory_candidates if task["evolve"] else [(published_payload, None, None)]
                )
                if not task["task_conditioned_search"] and task["evolve"] and robot_prechecks:
                    feasible_count = sum(item["precheck_passed"] for item in robot_prechecks)
                    # Do not spend up to 900 dynamic steps on a candidate that
                    # has already failed deterministic IK/table precheck.
                    robot_candidates = robot_candidates[: max(1, feasible_count)]
                robot_candidates = robot_candidates[: task["maximum_robot_candidates_per_attempt"]]
                preferred_payload = dict(published_payload)
                robot_lift_started = time.monotonic()
                scenes = attempt_scenes
                candidate_scene_pairs = []
                failed_scene_prechecks = []
                if task["task_conditioned_search"]:
                    _emit_progress(
                        object_id,
                        "TASK_PRECHECK",
                        current=0,
                        total=len(robot_candidates) * len(scenes),
                        detail=f"scenes={len(scenes)} candidates={len(robot_candidates)}",
                    )
                    payloads = [item[0] for item in robot_candidates]
                    scene_prechecks = precheck_robot_lift_task_scenes(
                        object_id,
                        payloads,
                        scenes,
                        seed=task["seed"] + attempt,
                        progress_callback=lambda current, total, scene: _emit_progress(
                            object_id,
                            "TASK_PRECHECK",
                            current=current,
                            total=total,
                            detail=f"scene={scene['scene_index']} "
                            f"pull={100.0 * scene['pull_toward_robot']:.0f}cm",
                        ),
                    )
                    for precheck in scene_prechecks:
                        candidate_index = int(precheck["candidate_index"])
                        scene = precheck["task_scene"]
                        if precheck["precheck_passed"]:
                            candidate_scene_pairs.append(
                                (candidate_index, robot_candidates[candidate_index], scene)
                            )
                        else:
                            failed_scene_prechecks.append(precheck)
                    # Dynamic rollout is expensive. Rank feasible combinations
                    # by least object relocation and then preserve grasp order.
                    candidate_scene_pairs.sort(
                        key=lambda item: (
                            float(item[2]["pull_toward_robot"]),
                            int(item[2]["scene_index"]),
                            item[0],
                        )
                    )
                    # Spend the dynamic-rollout budget across different task
                    # poses before trying a second grasp in the same pose.
                    first_per_scene = []
                    remaining_pairs = []
                    seen_scenes = set()
                    for pair in candidate_scene_pairs:
                        scene_index = int(pair[2]["scene_index"])
                        if scene_index in seen_scenes:
                            remaining_pairs.append(pair)
                        else:
                            seen_scenes.add(scene_index)
                            first_per_scene.append(pair)
                    candidate_scene_pairs = first_per_scene + remaining_pairs
                else:
                    candidate_scene_pairs = [
                        (index, candidate, None) for index, candidate in enumerate(robot_candidates)
                    ]
                candidate_scene_pairs = candidate_scene_pairs[
                    : task["maximum_robot_candidates_per_attempt"]
                ]
                if not candidate_scene_pairs and failed_scene_prechecks:
                    best_precheck = min(
                        failed_scene_prechecks,
                        key=lambda item: (
                            bool(item["table_collision"]),
                            float(item["maximum_ik_position_error"]),
                            float(item["maximum_ik_orientation_error"]),
                        ),
                    )
                    robot_lift = {
                        **best_precheck,
                        "robot_lift_verified": False,
                        "steps": 0,
                        "final_phase": "precheck",
                        "aborted": False,
                        "error": None,
                        "search_attempt": attempt,
                        "search_seed": task["seed"] + attempt,
                    }
                    robot_lift_attempts.append(robot_lift)
                    accumulated_robot_lift_attempts.append(robot_lift)
                for candidate_index, robot_candidate, scene in candidate_scene_pairs:
                    if time.monotonic() - started >= task["maximum_object_seconds"]:
                        break
                    candidate_payload = dict(robot_candidate[0])
                    _emit_progress(
                        object_id,
                        "DYNAMIC_LIFT",
                        current=len(robot_lift_attempts) + 1,
                        total=len(candidate_scene_pairs),
                        detail=(
                            "default scene"
                            if scene is None
                            else f"scene={scene['scene_index']} "
                            f"pull={100.0 * scene['pull_toward_robot']:.0f}cm"
                        ),
                    )
                    if task["evolve"]:
                        candidate_payload.pop("trajectory_stable_candidates", None)
                        candidate_payload.update(
                            direct_hold_stable=True,
                            trajectory_collision_free=True,
                            trajectory_hold_stable=True,
                            validation_stage="trajectory_hold_stable",
                        )
                    _write_payload_atomic(validation_path, candidate_payload)
                    candidate_lift = validate_robot_lift(
                        object_id,
                        validation_path,
                        seed=task["seed"] + attempt,
                        scene=scene,
                    ).as_dict()
                    candidate_lift["candidate_index"] = candidate_index
                    candidate_lift["search_attempt"] = attempt
                    candidate_lift["search_seed"] = task["seed"] + attempt
                    if scene is not None:
                        candidate_lift["task_scene"] = dict(scene)
                    robot_lift_attempts.append(candidate_lift)
                    accumulated_robot_lift_attempts.append(candidate_lift)
                    robot_lift = candidate_lift
                    if not candidate_lift["robot_lift_verified"]:
                        continue
                    verified_payload = dict(candidate_payload)
                    verified_payload.update(
                        robot_lift_verified=True,
                        robot_table_collision=False,
                        validation_stage="robot_lift_verified",
                    )
                    if scene is not None:
                        verified_payload["task_scene"] = dict(scene)
                    if _candidate_is_diverse(verified_payload, lift_verified_candidates):
                        lift_verified_candidates.append(verified_payload)
                    if best_lift_payload is None:
                        best_lift_payload = dict(verified_payload)
                        best_lift_result = dict(candidate_lift)
                    if task["evolve"]:
                        selected = robot_candidate
                        metrics = asdict(robot_candidate[1])
                        for key in (
                            "hand_table_clearance",
                            "approach_minimum_table_clearance",
                            "grasp_minimum_table_clearance",
                            "trajectory_minimum_table_clearance",
                        ):
                            if key in candidate_payload:
                                metrics[key] = candidate_payload[key]
                    covered_bins = set().union(
                        *(_approach_bins(item) for item in lift_verified_candidates)
                    )
                    if len(lift_verified_candidates) >= task["target_lift_candidates"] and len(
                        covered_bins
                    ) >= min(2, task["target_lift_candidates"]):
                        break
                phase_seconds["robot_lift_validation"] += time.monotonic() - robot_lift_started
                attempted_payload = json.loads(validation_path.read_text(encoding="utf-8"))
                if best_lift_result is not None:
                    robot_lift = dict(best_lift_result)
                robot_lift_verified = bool((robot_lift or {}).get("robot_lift_verified"))
                published_payload = (
                    dict(best_lift_payload)
                    if best_lift_payload is not None
                    else _payload_after_robot_lift_attempts(
                        preferred_payload,
                        attempted_payload,
                        robot_lift_verified=robot_lift_verified,
                    )
                )
                if task["evolve"]:
                    published_payload["trajectory_stable_candidates"] = list(
                        accumulated_trajectory_candidates
                    )
                    published_payload["lift_verified_candidates"] = list(lift_verified_candidates)
                published_payload.update(
                    robot_lift_verified=robot_lift_verified,
                    robot_table_collision=bool((robot_lift or {}).get("table_collision")),
                    validation_stage=(
                        "robot_lift_verified" if robot_lift_verified else "trajectory_hold_stable"
                    ),
                    lift_verified_candidate_count=len(lift_verified_candidates),
                    lift_candidate_target=task["target_lift_candidates"],
                )
                _write_payload_atomic(validation_path, published_payload)
            # The first catalogue pass optimizes for broad trajectory coverage.
            # Robot Lift remains an independent result and only becomes a retry
            # gate during the explicit incomplete-object refinement pass.
            fully_validated = _attempt_satisfies_goal(
                trajectory_hold_stable=trajectory_hold_stable,
                require_robot_lift_success=task["require_robot_lift_success"],
                robot_lift=robot_lift,
            ) and (
                not task["validate_robot_lift"]
                or (
                    len(lift_verified_candidates) >= task["target_lift_candidates"]
                    and len(
                        set().union(*(_approach_bins(item) for item in lift_verified_candidates))
                    )
                    >= min(2, task["target_lift_candidates"])
                )
            )
            status = (
                TRAJECTORY_STABLE
                if trajectory_hold_stable
                else DIRECT_HOLD_ONLY
                if task["evolve"] and best.direct_hold_stable
                else UNSTABLE
            )
            row = {
                "object_id": object_id,
                "status": status,
                "config": str(validation_path),
                "seed_config": str(config_path),
                "seed_trajectory_stable": (
                    None if seed_metrics is None else seed_metrics["trajectory_hold_stable"]
                ),
                "trajectory_hold_stable": trajectory_hold_stable,
                "robot_lift": robot_lift,
                "robot_lift_attempts": list(accumulated_robot_lift_attempts),
                "lift_verified_candidate_count": len(lift_verified_candidates),
                "lift_candidate_target": task["target_lift_candidates"],
                "object_time_budget_reached": (
                    time.monotonic() - started >= task["maximum_object_seconds"]
                ),
                "evolution": evolution_summary,
                "selected_seed": task["seed"] + attempt,
                "attempts_used": attempt + 1,
                "search_seconds": phase_seconds["search"],
                "phase_seconds": dict(phase_seconds),
                "elapsed_seconds": time.monotonic() - started,
                **metrics,
            }
            if fully_validated:
                return row
            unstable_key = _incomplete_attempt_key(row)
            if (
                best_unstable is None
                or unstable_key < best_unstable[0]
                or (
                    unstable_key == best_unstable[0]
                    and row["lift_verified_candidate_count"]
                    > best_unstable[1].get("lift_verified_candidate_count", 0)
                )
            ):
                best_unstable = (
                    unstable_key,
                    row,
                    dict(published_payload),
                    validation_path,
                )
        except Exception as exc:
            validation_errors.append(f"seed={task['seed'] + attempt} validation: {exc}")
    if best_unstable is not None:
        _write_payload_atomic(best_unstable[3], best_unstable[2])
        return best_unstable[1]
    if validation_errors:
        return {
            "object_id": object_id,
            "status": VALIDATION_ERROR,
            "error": " | ".join(validation_errors),
            "attempts_used": task["search_attempts"],
            "elapsed_seconds": time.monotonic() - started,
        }
    return {
        "object_id": object_id,
        "status": SEARCH_ERROR,
        "error": " | ".join(search_errors),
        "attempts_used": task["search_attempts"],
        "elapsed_seconds": time.monotonic() - started,
    }


def run_grasp_benchmark(args: GraspBenchmarkConfig) -> int:
    run_started = time.monotonic()
    if args.seconds <= 0 or args.settle_seconds < 0:
        raise ValueError("Simulation durations are invalid.")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.search_attempts <= 0:
        raise ValueError("--search-attempts must be positive.")
    if (
        min(
            args.target_lift_candidates,
            args.maximum_saved_candidates,
            args.maximum_robot_candidates_per_attempt,
        )
        <= 0
    ):
        raise ValueError("Candidate targets and limits must be positive.")
    if args.maximum_object_seconds <= 0.0:
        raise ValueError("--maximum-object-seconds must be positive.")
    if min(args.task_scene_attempts, args.task_rotations_per_distance) <= 0:
        raise ValueError("Task scene attempt counts must be positive.")
    if args.task_pull_step < 0.0 or args.task_maximum_pull < 0.0:
        raise ValueError("Task pull distances must be non-negative.")
    if args.pilot_min_results <= 0 or args.pilot_max_repeated_failure <= 0:
        raise ValueError("Pilot result and repeated-failure thresholds must be positive.")
    if not 0.0 <= args.pilot_min_lift_rate <= 1.0:
        raise ValueError("--pilot-min-lift-rate must be between zero and one.")
    if args.evolution_backend not in {"auto", "cpu", "mjwarp"}:
        raise ValueError("--evolution-backend must be auto, cpu, or mjwarp.")
    if min(args.mjwarp_batch_size, args.mjwarp_nconmax, args.mjwarp_njmax) <= 0:
        raise ValueError("MJWarp batch size and capacities must be positive.")
    gpu_evolution = args.evolution_backend == "mjwarp" or (
        args.evolution_backend == "auto" and mjwarp_available()
    )
    if args.evolve and gpu_evolution and args.jobs != 1:
        raise ValueError(
            "MJWarp evolution requires --jobs 1 so object workers do not duplicate GPU buffers."
        )
    total_workers = args.jobs * (args.evolution_jobs if args.evolve else 1)
    if total_workers > 8:
        raise ValueError(
            "Unsafe nested parallelism: --jobs * --evolution-jobs must be <= 8 "
            f"(got {args.jobs} * {args.evolution_jobs} = {total_workers})."
        )
    if args.evolve and args.end_effector != "dex_hand":
        raise ValueError("DexEvolve refinement currently supports dex_hand only.")
    if (
        args.evolve
        and min(
            args.evolution_population,
            args.evolution_offspring,
            args.evolution_generations,
            args.evolution_jobs,
        )
        <= 0
    ):
        raise ValueError("Evolution sizes, generations, and jobs must be positive.")
    for name in (
        "points",
        "joint_candidates",
        "surface_anchors",
        "rolls_per_anchor",
        "coarse_keep",
        "top_k",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.support_margin < 0.0:
        raise ValueError("--support-margin must be non-negative.")
    selected = _selected_ids(args)
    if args.config_dir is None:
        args.config_dir = grasp_config_directory(args.end_effector, benchmark=True)
    if args.output is None:
        args.output = grasp_benchmark_report_path(args.end_effector)
    if args.evolution_dir is None:
        args.evolution_dir = grasp_config_directory(args.end_effector) / "dexevolve"
    print(
        f"workers: objects={args.jobs} evolution_per_object="
        f"{args.evolution_jobs if args.evolve else 0} "
        f"maximum_process_parallelism={total_workers}",
        f"evolution_backend={args.evolution_backend}",
        flush=True,
    )
    rows = _load_completed(args.output, args) if args.resume else []
    rows = [row for row in rows if row["object_id"] in selected]
    if args.retry_incomplete:
        retained = []
        for row in rows:
            trajectory_ready = row.get("status") == TRAJECTORY_STABLE
            robot_ready = not args.validate_robot_lift or bool(
                (row.get("robot_lift") or {}).get("robot_lift_verified")
            )
            if args.validate_robot_lift and robot_ready:
                candidate_count = int(
                    row.get(
                        "lift_verified_candidate_count",
                        1,
                    )
                )
                robot_ready = candidate_count >= args.target_lift_candidates
            if trajectory_ready and robot_ready:
                retained.append(row)
            else:
                print(
                    f"RETRY {row['object_id']} status={row.get('status')} "
                    f"robot_lift_verified={robot_ready if args.validate_robot_lift else 'n/a'}",
                    flush=True,
                )
        rows = retained
    completed = {row["object_id"] for row in rows}
    resumed_count = len(rows)

    pending = [object_id for object_id in selected if object_id not in completed]
    for object_id in selected:
        if object_id in completed:
            print(f"SKIP {object_id}", flush=True)
    tasks = [
        {
            "object_id": object_id,
            "config_path": str(args.config_dir / f"{grasp_config_name(object_id)}.json"),
            "reuse": args.reuse,
            "validate_robot_lift": args.validate_robot_lift,
            "require_robot_lift_success": args.retry_incomplete,
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
            "evolution_path": str(args.evolution_dir / f"{grasp_config_name(object_id)}.json"),
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
        for object_id in pending
    ]
    progress_manager = multiprocessing.Manager()
    progress_queue = progress_manager.Queue()
    executor = ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_progress_worker,
        initargs=(progress_queue,),
    )
    futures = {executor.submit(_run_one, task): task for task in tasks}
    pending = set(futures)
    live_progress = LiveWorkerProgress(
        total=len(tasks),
        workers=args.jobs,
    )
    interrupt_count = 0
    force_stop = False
    pilot_stop = None
    try:
        while pending:
            try:
                completed_futures, _ = wait(
                    pending,
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                while True:
                    try:
                        live_progress.update(progress_queue.get_nowait())
                    except queue.Empty:
                        break
                live_progress.render()
                for future in completed_futures:
                    pending.remove(future)
                    row = future.result()
                    object_id = row["object_id"]
                    rows.append(row)
                    rows.sort(key=lambda row: selected.index(row["object_id"]))
                    _write_report(args.output, args=args, selected=selected, rows=rows)
                    detail = row.get("error", "")
                    task_label = (
                        _task_outcome_label(
                            row,
                            target_lift_candidates=args.target_lift_candidates,
                        )
                        if args.validate_robot_lift
                        else row["status"].upper()
                    )
                    lift_archive_label = (
                        f"lift_grasps={row.get('lift_verified_candidate_count', 0)}/"
                        f"{args.target_lift_candidates} "
                        if args.validate_robot_lift
                        else ""
                    )
                    scene_label = _task_scene_label(row) if args.task_conditioned_search else ""
                    completed_this_run = len(rows) - resumed_count
                    run_elapsed = time.monotonic() - run_started
                    average, eta = _progress_timing(
                        elapsed=run_elapsed,
                        completed=completed_this_run,
                        total=len(selected) - resumed_count,
                        worker_count=args.jobs,
                    )
                    eta_label = "warming_up" if eta is None else _format_duration(eta)
                    live_progress.mark_completed(
                        object_id=object_id,
                        solved=int(row.get("lift_verified_candidate_count", 0)) > 0,
                    )
                    live_progress.clear()
                    print(
                        f"[{len(rows)}/{len(selected)}] "
                        f"{task_label:20} {object_id} "
                        f"{lift_archive_label}{scene_label}"
                        f"object={_format_duration(row.get('elapsed_seconds', 0.0))} "
                        f"throughput={_format_duration(average)}/object eta={eta_label} "
                        f"{detail}",
                        flush=True,
                    )
                    live_progress.render()
                    if args.pilot:
                        lift_successes = sum(
                            bool((item.get("robot_lift") or {}).get("robot_lift_verified"))
                            for item in rows
                        )
                        reasons = {}
                        for item in rows:
                            reason = _failure_reason(item)
                            if reason is not None:
                                reasons[reason] = reasons.get(reason, 0) + 1
                        print(
                            f"pilot_progress={len(rows)}/{len(selected)} "
                            f"lift_rate={lift_successes / len(rows):.1%} "
                            f"failure_reasons={reasons or '{}'}",
                            flush=True,
                        )
                        pilot_stop = _pilot_stop_reason(
                            rows,
                            minimum_results=args.pilot_min_results,
                            minimum_lift_rate=args.pilot_min_lift_rate,
                            maximum_repeated_failure=args.pilot_max_repeated_failure,
                        )
                        if pilot_stop is not None:
                            force_stop = True
                            print(f"PILOT_STOP {pilot_stop}", flush=True)
                            break
                if pilot_stop is not None:
                    break
            except KeyboardInterrupt:
                interrupt_count += 1
                _write_report(args.output, args=args, selected=selected, rows=rows)
                if interrupt_count == 1:
                    print(
                        "\nSIGINT received; progress saved and benchmark continues. "
                        "Press Ctrl-C again to stop.",
                        flush=True,
                    )
                    continue
                force_stop = True
                print(
                    "\nSecond SIGINT received; cancelling pending objects. "
                    f"Progress saved at {len(rows)}/{len(selected)}.",
                    flush=True,
                )
                break
    finally:
        live_progress.close()
        executor.shutdown(wait=not force_stop, cancel_futures=force_stop)
        progress_manager.shutdown()

    if pilot_stop is not None:
        _write_report(args.output, args=args, selected=selected, rows=rows)
        print(f"report={args.output}")
        return 2
    if force_stop:
        return 130

    _write_report(args.output, args=args, selected=selected, rows=rows)
    stable = sum(row["status"] == TRAJECTORY_STABLE for row in rows)
    generated = sum(row["status"] != SEARCH_ERROR for row in rows)
    robot_lift_tested = sum(row.get("robot_lift") is not None for row in rows)
    robot_lift_verified = sum(
        bool((row.get("robot_lift") or {}).get("robot_lift_verified")) for row in rows
    )
    lift_verified_candidate_count = sum(
        int(row.get("lift_verified_candidate_count", 0)) for row in rows
    )
    lift_candidate_targets_met = sum(
        int(row.get("lift_verified_candidate_count", 0)) >= args.target_lift_candidates
        for row in rows
    )
    task_solved = sum(int(row.get("lift_verified_candidate_count", 0)) > 0 for row in rows)
    unsolved = [
        row["object_id"] for row in rows if int(row.get("lift_verified_candidate_count", 0)) == 0
    ]
    incomplete_lift_archives = [
        row["object_id"]
        for row in rows
        if args.validate_robot_lift
        and int(row.get("lift_verified_candidate_count", 0)) < args.target_lift_candidates
    ]
    print(f"\ncompleted={len(rows)}/{len(selected)}")
    print(
        f"task_solved={task_solved}/{len(rows)} "
        f"object_success_rate={task_solved / len(rows):.1%} "
        f"task_targets_met={lift_candidate_targets_met}/{len(rows)}"
    )
    print(
        f"lift_verified_grasps={lift_verified_candidate_count} "
        f"target_per_object={args.target_lift_candidates}"
    )
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            TRAJECTORY_STABLE,
            DIRECT_HOLD_ONLY,
            UNSTABLE,
            VALIDATION_ERROR,
            SEARCH_ERROR,
        )
    }
    print(
        "diagnostic_status_counts="
        + " ".join(f"{status}:{count}" for status, count in status_counts.items())
    )
    print(
        f"diagnostics: generated={generated}/{len(rows)} "
        f"trajectory_stable={stable}/{len(rows)} "
        f"robot_lift_tested={robot_lift_tested} "
        f"robot_lift_verified={robot_lift_verified}"
    )
    print(f"total_elapsed={_format_duration(time.monotonic() - run_started)}")
    print("unsolved_objects:")
    print(*(unsolved or ["(none)"]), sep="\n")
    print("objects_below_lift_grasp_target:")
    print(*(incomplete_lift_archives or ["(none)"]), sep="\n")
    print(f"report={args.output}")
    return int(bool(unsolved or incomplete_lift_archives))
