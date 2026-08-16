"""Catalogue-level benchmark orchestration."""

from __future__ import annotations

import multiprocessing
import queue
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from source.evaluation.grasp_schema import (
    DIRECT_HOLD_ONLY,
    SEARCH_ERROR,
    TRAJECTORY_STABLE,
    UNSTABLE,
    VALIDATION_ERROR,
)
from source.grasping.dexevolve import mjwarp_available
from source.grasping.search import (
    grasp_benchmark_report_path,
    grasp_config_directory,
    grasp_config_name,
)
from source.runtime.progress import LiveWorkerProgress
from source.workflows.grasp_benchmark.config import GraspBenchmarkConfig, _selected_ids
from source.workflows.grasp_benchmark.reporting import (
    _failure_reason,
    _format_duration,
    _load_completed,
    _pilot_stop_reason,
    _progress_timing,
    _task_outcome_label,
    _task_scene_label,
    _write_report,
)
from source.workflows.grasp_benchmark.worker import _init_progress_worker, _run_one


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
