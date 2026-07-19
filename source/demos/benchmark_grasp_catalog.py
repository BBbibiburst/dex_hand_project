"""Search and physics-validate grasps for every catalogue object."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from source.envs.manipulation.object_catalog import object_ids
from source.grasping.grasp_config_search import (
    grasp_benchmark_report_path,
    grasp_config_directory,
    grasp_config_name,
    search_grasp_config,
)
from source.grasping.standalone_validator import (
    validate_grasp_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "ycb", "egad"),
        default="all",
        help="Catalogue subset to test.",
    )
    parser.add_argument(
        "--object-id",
        action="append",
        dest="object_ids",
        help="Test only this object; repeat for multiple objects.",
    )
    parser.add_argument("--limit", type=int, help="Test only the first N objects.")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--joint-candidates", type=int, default=128)
    parser.add_argument(
        "--search-attempts",
        type=int,
        default=3,
        help="Independent seeds tried per object; stops at the first stable grasp.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument(
        "--end-effector",
        choices=("dex_hand", "pika_gripper"),
        default="dex_hand",
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--grip-preload", type=float, default=0.25)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes; use 1 for deterministic debugging.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing per-object grasp JSON files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows in an existing report and test the rest.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help=(
            "Per-object output directory "
            "(default: configs/grasps/<end_effector>/benchmark)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Benchmark report JSON "
            "(default: configs/grasps/<end_effector>/grasp_catalog_benchmark.json)."
        ),
    )
    return parser.parse_args()


def _selected_ids(args: argparse.Namespace) -> list[str]:
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
    args: argparse.Namespace,
    selected: list[str],
    rows: list[dict],
) -> None:
    generated = sum(row["status"] != "search_error" for row in rows)
    stable = sum(row["status"] == "stable" for row in rows)
    failed = [row["object_id"] for row in rows if row["status"] != "stable"]
    payload = {
        "schema_version": 1,
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


def _report_parameters(args: argparse.Namespace) -> dict:
    return {
        "dataset": args.dataset,
        "points": args.points,
        "joint_candidates": args.joint_candidates,
        "search_attempts": args.search_attempts,
        "seed": args.seed,
        "target_size": args.target_size,
        "end_effector": args.end_effector,
        "seconds": args.seconds,
        "settle_seconds": args.settle_seconds,
        "grip_preload": args.grip_preload,
    }


def _load_completed(path: Path, args: argparse.Namespace) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
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
            "pca_axis_index": payload.get("hand_pca_axis_index"),
            "robustness_margin": payload.get("hand_robustness_margin"),
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
                    seed=task["seed"] + attempt,
                    target_size=task["target_size"],
                    end_effector_name=task["end_effector"],
                )
            search_seconds = time.monotonic() - attempt_started
        except Exception as exc:
            search_errors.append(f"seed={task['seed'] + attempt}: {exc}")
            continue
        try:
            metrics = _validate_config(
                config_path,
                seconds=task["seconds"],
                settle_seconds=task["settle_seconds"],
                grip_preload=task["grip_preload"],
            )
            row = {
                "object_id": object_id,
                "status": "stable" if metrics["stable"] else "unstable",
                "config": str(config_path),
                "selected_seed": task["seed"] + attempt,
                "attempts_used": attempt + 1,
                "search_seconds": search_seconds,
                "elapsed_seconds": time.monotonic() - started,
                **metrics,
            }
            if metrics["stable"]:
                return row
            best_unstable = row
        except Exception as exc:
            validation_errors.append(f"seed={task['seed'] + attempt} validation: {exc}")
    if best_unstable is not None:
        return best_unstable
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


def run(args: argparse.Namespace) -> int:
    if args.seconds <= 0 or args.settle_seconds < 0:
        raise ValueError("Simulation durations are invalid.")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.search_attempts <= 0:
        raise ValueError("--search-attempts must be positive.")
    selected = _selected_ids(args)
    if args.config_dir is None:
        args.config_dir = grasp_config_directory(args.end_effector, benchmark=True)
    if args.output is None:
        args.output = grasp_benchmark_report_path(args.end_effector)
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
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(_run_one, task): task for task in tasks}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            object_id = row["object_id"]
            rows.append(row)
            rows.sort(key=lambda row: selected.index(row["object_id"]))
            _write_report(args.output, args=args, selected=selected, rows=rows)
            detail = row.get("error", "")
            print(
                f"[{len(completed) + completed_count}/{len(selected)}] "
                f"{row['status'].upper():16} {object_id} {detail}",
                flush=True,
            )

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


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()

