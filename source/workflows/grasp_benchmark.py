"""Search and physics-validate grasps for every catalogue object."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

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
from source.grasping.dexevolve import EvolutionConfig, evolve, table_clearance_metrics


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


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
    evolution_dir: Path | None = None
    search_attempts: int = 3
    seed: int = 0
    target_size: float = 0.09
    end_effector: str = "dex_hand"
    seconds: float = 3.0
    settle_seconds: float = 0.8
    grip_preload: float = DEFAULT_GRIP_PRELOAD
    jobs: int = 1
    reuse: bool = False
    validate_robot_lift: bool = False
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
    failure_reasons: dict[str, int] = {}
    for row in rows:
        reason = _failure_reason(row)
        if reason is not None:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    failed = [row["object_id"] for row in rows if row["status"] != TRAJECTORY_STABLE]
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
            "failure_reasons": failure_reasons,
            "failed_object_ids": failed,
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
        "search_attempts": args.search_attempts,
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
    started = time.monotonic()
    config_path = Path(task["config_path"])
    search_errors = []
    validation_errors = []
    best_unstable = None
    phase_seconds = {
        "search": 0.0,
        "evolution": 0.0,
        "trajectory_replan_and_validation": 0.0,
        "robot_lift_validation": 0.0,
    }
    for attempt in range(task["search_attempts"]):
        attempt_started = time.monotonic()
        try:
            reuse_this_attempt = task["reuse"] and attempt == 0 and config_path.is_file()
            if not reuse_this_attempt:
                search_grasp_config(
                    object_id=object_id,
                    output=config_path,
                    points=task["points"],
                    joint_candidates=task["joint_candidates"],
                    surface_anchors=task["surface_anchors"],
                    rolls_per_anchor=task["rolls_per_anchor"],
                    coarse_keep=task["coarse_keep"],
                    top_k=task["top_k"],
                    support_margin=task["support_margin"],
                    seed=task["seed"] + attempt,
                    target_size=task["target_size"],
                    end_effector_name=task["end_effector"],
                    generator=task["generator"],
                    graspqp_iterations=task["graspqp_iterations"],
                    require_valid=not task["evolve"],
                    publish_invalid=task["evolve"],
                )
            search_seconds = time.monotonic() - attempt_started
            phase_seconds["search"] += search_seconds
        except Exception as exc:
            search_errors.append(f"seed={task['seed'] + attempt}: {exc}")
            continue
        try:
            seed_metrics = None
            seed_payload = json.loads(config_path.read_text(encoding="utf-8"))
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
                )
                evolution_started = time.monotonic()
                archive, history = evolve(seed_payload, evolution_config)
                phase_seconds["evolution"] += time.monotonic() - evolution_started
                best = archive[0]
                trajectory_candidates = []
                trajectory_errors = []
                trajectory_started = time.monotonic()
                for archive_index, individual in enumerate(archive):
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
                    )
                    trajectory_candidates.append((candidate, result, individual))
                    if len(trajectory_candidates) >= 16:
                        break
                phase_seconds["trajectory_replan_and_validation"] += (
                    time.monotonic() - trajectory_started
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
                temporary = validation_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(published_payload, indent=2), encoding="utf-8")
                temporary.replace(validation_path)
                evolution_summary = {
                    "archive": len(archive),
                    "direct_hold_candidates": sum(item.direct_hold_stable for item in archive),
                    "trajectory_stable_candidates": len(trajectory_candidates),
                    "trajectory_validation_errors": trajectory_errors[:8],
                    "best_fitness": best.fitness,
                    "history": history,
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
                from source.grasping.robot_lift_validator import validate_robot_lift

                robot_candidates = (
                    trajectory_candidates if task["evolve"] else [(published_payload, None, None)]
                )
                preferred_payload = dict(published_payload)
                robot_lift_started = time.monotonic()
                for candidate_index, robot_candidate in enumerate(robot_candidates):
                    candidate_payload = dict(robot_candidate[0])
                    if task["evolve"]:
                        candidate_payload.pop("trajectory_stable_candidates", None)
                        candidate_payload.update(
                            direct_hold_stable=True,
                            trajectory_collision_free=True,
                            trajectory_hold_stable=True,
                            validation_stage="trajectory_hold_stable",
                        )
                    temporary = validation_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
                    temporary.replace(validation_path)
                    candidate_lift = validate_robot_lift(
                        object_id,
                        validation_path,
                        seed=task["seed"] + attempt,
                    ).as_dict()
                    candidate_lift["candidate_index"] = candidate_index
                    robot_lift_attempts.append(candidate_lift)
                    robot_lift = candidate_lift
                    if not candidate_lift["robot_lift_verified"]:
                        continue
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
                    break
                phase_seconds["robot_lift_validation"] += time.monotonic() - robot_lift_started
                attempted_payload = json.loads(validation_path.read_text(encoding="utf-8"))
                published_payload = _payload_after_robot_lift_attempts(
                    preferred_payload,
                    attempted_payload,
                    robot_lift_verified=robot_lift["robot_lift_verified"],
                )
                if task["evolve"]:
                    published_payload["trajectory_stable_candidates"] = [
                        candidate for candidate, _, _ in trajectory_candidates
                    ]
                published_payload.update(
                    robot_lift_verified=robot_lift["robot_lift_verified"],
                    robot_table_collision=robot_lift["table_collision"],
                    validation_stage=(
                        "robot_lift_verified"
                        if robot_lift["robot_lift_verified"]
                        else "trajectory_hold_stable"
                    ),
                )
                temporary = validation_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(published_payload, indent=2), encoding="utf-8")
                temporary.replace(validation_path)
            # The first catalogue pass optimizes for broad trajectory coverage.
            # Robot Lift remains an independent result and only becomes a retry
            # gate during the explicit incomplete-object refinement pass.
            fully_validated = _attempt_satisfies_goal(
                trajectory_hold_stable=trajectory_hold_stable,
                require_robot_lift_success=task["require_robot_lift_success"],
                robot_lift=robot_lift,
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
                "robot_lift_attempts": robot_lift_attempts,
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
            unstable_key = (
                float(row.get("vertical_drop", float("inf"))),
                float(row.get("position_drift", float("inf"))),
                float(row.get("rotation_drift", float("inf"))),
                -int(row.get("final_contacts", 0)),
            )
            if best_unstable is None or unstable_key < best_unstable[0]:
                best_unstable = (unstable_key, row)
        except Exception as exc:
            validation_errors.append(f"seed={task['seed'] + attempt} validation: {exc}")
    if best_unstable is not None:
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
            "evolution_path": str(args.evolution_dir / f"{grasp_config_name(object_id)}.json"),
            "search_attempts": args.search_attempts,
            "seed": args.seed,
            "target_size": args.target_size,
            "end_effector": args.end_effector,
            "seconds": args.seconds,
            "settle_seconds": args.settle_seconds,
            "grip_preload": args.grip_preload,
        }
        for object_id in pending
    ]
    executor = ProcessPoolExecutor(max_workers=args.jobs)
    futures = {executor.submit(_run_one, task): task for task in tasks}
    pending = set(futures)
    interrupt_count = 0
    force_stop = False
    try:
        while pending:
            try:
                for future in as_completed(pending):
                    pending.remove(future)
                    row = future.result()
                    object_id = row["object_id"]
                    rows.append(row)
                    rows.sort(key=lambda row: selected.index(row["object_id"]))
                    _write_report(args.output, args=args, selected=selected, rows=rows)
                    detail = row.get("error", "")
                    lift = row.get("robot_lift")
                    lift_label = (
                        ""
                        if lift is None
                        else f"lift={'PASS' if lift.get('robot_lift_verified') else 'FAIL'} "
                    )
                    completed_this_run = len(rows) - resumed_count
                    run_elapsed = time.monotonic() - run_started
                    average, eta = _progress_timing(
                        elapsed=run_elapsed,
                        completed=completed_this_run,
                        total=len(selected) - resumed_count,
                        worker_count=args.jobs,
                    )
                    eta_label = "warming_up" if eta is None else _format_duration(eta)
                    print(
                        f"[{len(rows)}/{len(selected)}] "
                        f"{row['status'].upper():16} {object_id} "
                        f"{lift_label}"
                        f"object={_format_duration(row.get('elapsed_seconds', 0.0))} "
                        f"throughput={_format_duration(average)}/object eta={eta_label} "
                        f"{detail}",
                        flush=True,
                    )
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
        executor.shutdown(wait=not force_stop, cancel_futures=force_stop)

    if force_stop:
        return 130

    _write_report(args.output, args=args, selected=selected, rows=rows)
    stable = sum(row["status"] == TRAJECTORY_STABLE for row in rows)
    generated = sum(row["status"] != SEARCH_ERROR for row in rows)
    robot_lift_tested = sum(row.get("robot_lift") is not None for row in rows)
    robot_lift_verified = sum(
        bool((row.get("robot_lift") or {}).get("robot_lift_verified")) for row in rows
    )
    failed = [row["object_id"] for row in rows if row["status"] != TRAJECTORY_STABLE]
    print(
        f"\ncompleted={len(rows)}/{len(selected)} "
        f"generated={generated}/{len(rows)} "
        f"trajectory_stable={stable}/{len(rows)} "
        f"trajectory_stable_rate={stable / len(rows):.1%}"
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
        "status_counts=" + " ".join(f"{status}:{count}" for status, count in status_counts.items())
    )
    if robot_lift_tested:
        print(
            f"robot_lift_verified={robot_lift_verified}/{robot_lift_tested} "
            f"robot_lift_verified_rate={robot_lift_verified / robot_lift_tested:.1%}"
        )
    else:
        print("robot_lift_verified=0/0 robot_lift_verified_rate=n/a")
    print(f"total_elapsed={_format_duration(time.monotonic() - run_started)}")
    print("cannot_grasp_or_hold:")
    print(*(failed or ["(none)"]), sep="\n")
    print(f"report={args.output}")
    return int(bool(failed))
