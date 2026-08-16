"""One-object benchmark worker and progress transport."""

from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import asdict
from pathlib import Path

from source.evaluation.grasp_schema import (
    DIRECT_HOLD_ONLY,
    SEARCH_ERROR,
    TRAJECTORY_STABLE,
    UNSTABLE,
    VALIDATION_ERROR,
)
from source.grasping.dexevolve import EvolutionConfig, evolve, table_clearance_metrics
from source.grasping.search import replan_evolved_payload, search_grasp_config
from source.grasping.standalone_validator import (
    validate_grasp_config,
    validate_grasp_payload_trajectory,
)
from source.workflows.grasp_benchmark.candidates import (
    _append_diverse_candidates,
    _approach_bins,
    _candidate_is_diverse,
    _incomplete_attempt_key,
    _payload_after_robot_lift_attempts,
    _robot_candidate_precheck_key,
    _write_payload_atomic,
)
from source.workflows.grasp_benchmark.reporting import _attempt_satisfies_goal

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
            from source.execution.robot_lift import task_scene_schedule

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
                    "geometric candidate generation",
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
                    "require_valid": not task["evolve"],
                    "publish_invalid": task["evolve"],
                }
                if task["task_conditioned_search"]:
                    from source.execution.robot_lift import (
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
        except Exception as exc:  # noqa: BLE001 - continue with the next search seed
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
                    except Exception as exc:  # noqa: BLE001 - reject only this candidate
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
                    from source.execution.robot_lift import (
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

                    def robot_candidate_key(
                        index_and_candidate,
                        prechecks=indexed_prechecks,
                    ):
                        index, candidate = index_and_candidate
                        payload, _result, individual = candidate
                        precheck = prechecks[index]
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
                from source.execution.robot_lift import (
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
        except Exception as exc:  # noqa: BLE001 - preserve the best prior attempt
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
