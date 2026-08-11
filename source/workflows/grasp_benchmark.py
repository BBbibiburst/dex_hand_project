"""Search and physics-validate grasps for every catalogue object."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from source.envs.manipulation.object_catalog import object_ids
from source.grasping.constants import (
    DEFAULT_GRIP_PRELOAD,
    GRASP_CONFIG_SCHEMA_VERSION,
    GRASP_SEARCH_STRATEGY,
)
from source.grasping.grasp_config_search import (
    grasp_benchmark_report_path,
    grasp_config_directory,
    grasp_config_name,
    search_grasp_config,
)
from source.grasping.standalone_validator import (
    validate_grasp_config,
    validate_grasp_payload_direct,
)
from source.grasping.dexevolve import EvolutionConfig, evolve


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
    resume: bool = False
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
    generated = sum(row["status"] != "search_error" for row in rows)
    stable = sum(row["status"] == "stable" for row in rows)
    failed = [row["object_id"] for row in rows if row["status"] != "stable"]
    payload = {
        "schema_version": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": _report_parameters(args),
        "summary": {
            "selected": len(selected),
            "completed": len(rows),
            "grasp_generated": generated,
            "stable": stable,
            "generation_rate": generated / len(rows) if rows else 0.0,
            "stable_rate": stable / len(rows) if rows else 0.0,
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
    if payload.get("schema_version") != 2:
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
                # DexEvolve intentionally accepts analytically invalid seeds;
                # simulator fitness, rather than the geometric filter, decides
                # whether offspring survive.
                seed_payload["hand_fit_success"] = True
                evolution_config = EvolutionConfig(
                    population_size=task["evolution_population"],
                    offspring=task["evolution_offspring"],
                    generations=task["evolution_generations"],
                    jobs=task["evolution_jobs"],
                    seconds=task["evolution_seconds"],
                    seed=task["seed"] + attempt,
                )
                archive, history = evolve(seed_payload, evolution_config)
                best = archive[0]
                # Keep a compact, diverse set of simulator-stable alternatives.
                # A single object-relative grasp cannot remain arm-reachable for
                # every randomized object yaw; Lift selects among these without
                # inventing contact-invalid rotations at execution time.
                stable_payloads = []
                for individual in archive:
                    if not individual.stable:
                        continue
                    candidate = dict(individual.payload)
                    candidate.pop("stable_grasp_candidates", None)
                    stable_payloads.append(candidate)
                    if len(stable_payloads) >= 16:
                        break
                published_payload = dict(best.payload)
                published_payload["stable_grasp_candidates"] = stable_payloads
                validation_path = Path(task["evolution_path"])
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = validation_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(published_payload, indent=2), encoding="utf-8")
                temporary.replace(validation_path)
                evolution_summary = {
                    "archive": len(archive),
                    "stable_candidates": sum(item.stable for item in archive),
                    "best_fitness": best.fitness,
                    "history": history,
                }

            if task["evolve"]:
                final_payload = json.loads(validation_path.read_text(encoding="utf-8"))
                metrics = asdict(
                    validate_grasp_payload_direct(
                        final_payload,
                        seconds=task["seconds"],
                        settle_seconds=task["settle_seconds"],
                        grip_preload=task["grip_preload"],
                    )
                )
                # Evolution-level hard constraints (notably swept hand/table
                # clearance) must not be overwritten by the final free-hand
                # dynamics check, whose table is visual-only.
                metrics["stable"] = bool(metrics["stable"] and best.stable)
                if not best.stable and best.metrics is not None:
                    metrics["evolution_rejection"] = best.metrics.get(
                        "rejection_reason",
                        best.metrics.get("error"),
                    )
                metrics.update(
                    {
                        "table_clearance": final_payload.get("hand_table_clearance"),
                        "approach_table_clearance": final_payload.get(
                            "approach_minimum_table_clearance"
                        ),
                        "grasp_table_clearance": final_payload.get("grasp_minimum_table_clearance"),
                        "trajectory_table_clearance": final_payload.get(
                            "trajectory_minimum_table_clearance"
                        ),
                    }
                )
            else:
                metrics = _validate_config(
                    validation_path,
                    seconds=task["seconds"],
                    settle_seconds=task["settle_seconds"],
                    grip_preload=task["grip_preload"],
                )
            row = {
                "object_id": object_id,
                "status": "stable" if metrics["stable"] else "unstable",
                "config": str(validation_path),
                "seed_config": str(config_path),
                "seed_stable": None if seed_metrics is None else seed_metrics["stable"],
                "evolution": evolution_summary,
                "selected_seed": task["seed"] + attempt,
                "attempts_used": attempt + 1,
                "search_seconds": search_seconds,
                "elapsed_seconds": time.monotonic() - started,
                **metrics,
            }
            if metrics["stable"]:
                return row
            unstable_key = (
                float(row["vertical_drop"]),
                float(row["position_drift"]),
                float(row["rotation_drift"]),
                -int(row["final_contacts"]),
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
            "status": "validation_error",
            "error": " | ".join(validation_errors),
            "attempts_used": task["search_attempts"],
            "elapsed_seconds": time.monotonic() - started,
        }
    return {
        "object_id": object_id,
        "status": "search_error",
        "error": " | ".join(search_errors),
        "attempts_used": task["search_attempts"],
        "elapsed_seconds": time.monotonic() - started,
    }


def run_grasp_benchmark(args: GraspBenchmarkConfig) -> int:
    if args.seconds <= 0 or args.settle_seconds < 0:
        raise ValueError("Simulation durations are invalid.")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.search_attempts <= 0:
        raise ValueError("--search-attempts must be positive.")
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
    rows = _load_completed(args.output, args) if args.resume else []
    rows = [row for row in rows if row["object_id"] in selected]
    completed = {row["object_id"] for row in rows}

    pending = [object_id for object_id in selected if object_id not in completed]
    for object_id in selected:
        if object_id in completed:
            print(f"SKIP {object_id}", flush=True)
    tasks = [
        {
            "object_id": object_id,
            "config_path": str(args.config_dir / f"{grasp_config_name(object_id)}.json"),
            "reuse": args.reuse,
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
                    print(
                        f"[{len(rows)}/{len(selected)}] "
                        f"{row['status'].upper():16} {object_id} {detail}",
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
    stable = sum(row["status"] == "stable" for row in rows)
    generated = sum(row["status"] != "search_error" for row in rows)
    failed = [row["object_id"] for row in rows if row["status"] != "stable"]
    print(
        f"\ncompleted={len(rows)}/{len(selected)} "
        f"generated={generated}/{len(rows)} "
        f"stable={stable}/{len(rows)} "
        f"stable_rate={stable / len(rows):.1%}"
    )
    print("cannot_grasp_or_hold:")
    print(*(failed or ["(none)"]), sep="\n")
    print(f"report={args.output}")
    return int(bool(failed))
