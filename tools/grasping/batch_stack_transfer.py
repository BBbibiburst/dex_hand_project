"""Migrate the Top100 verified Lift benchmark into the Stack task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from source.grasping.stack_transfer import STACK_TRANSFER_VERSION
from source.envs.manipulation.object_catalog import partition_task_objects
from source.workflows.batch import (
    atomic_write_json,
    lift_source_roots,
    load_selection_ids,
    memory_safe_worker_count,
    object_slug,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/stack_top100"))
    parser.add_argument(
        "--lattice-root", type=Path, default=Path("outputs/dex_hand_top100_v2/lattice")
    )
    parser.add_argument("--rl-root", type=Path, default=Path("outputs/dex_hand_top100_v2/rl"))
    parser.add_argument("--grasp-root", type=Path, default=Path("outputs/dex_hand_top100_v2/grasp"))
    parser.add_argument("--extra-lift-root", action="append", type=Path, dest="extra_lift_roots")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--memory-per-worker-gib", type=float, default=1.5)
    parser.add_argument("--seed-attempts", type=int, default=3)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Run every selected object as a coverage stress test instead of the Stack domain.",
    )
    return parser


def _run_object(object_id: str, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output / object_slug(object_id)
    summary_path = output / "summary.json"
    if not args.force and summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("success") and cached.get("pipeline_version") == STACK_TRANSFER_VERSION:
            return {
                "object_id": object_id,
                "status": "SUCCESS",
                "attempts": len(cached.get("attempts", [])),
                "runtime_sec": 0.0,
                "cached": True,
                "failure_reason": "",
            }
    roots = lift_source_roots(
        args.lattice_root,
        args.rl_root,
        args.grasp_root,
        object_id,
        tuple(args.extra_lift_roots or ()),
    )
    command = [
        sys.executable,
        "-m",
        "tools.grasping.transfer_lift_to_stack",
        "--object-id",
        object_id,
        "--output",
        str(output),
        "--seed-attempts",
        str(args.seed_attempts),
        "--maximum-candidates",
        str(args.maximum_candidates),
    ]
    for root in roots:
        command.extend(("--lift-root", str(root)))
    log = args.output / "logs" / f"{object_slug(object_id)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        child = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    attempts = payload.get("attempts", [])
    success = bool(payload.get("success"))
    failure = ""
    if attempts and not success:
        failure = str(attempts[-1].get("failure_reason", ""))
    if child.returncode not in (0, 2) and not success:
        failure = f"child exit {child.returncode}; see {log}"
    return {
        "object_id": object_id,
        "status": "SUCCESS" if success else "FAILED",
        "attempts": len(attempts),
        "runtime_sec": round(time.perf_counter() - started, 3),
        "cached": False,
        "failure_reason": failure,
    }


def _write_summary(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    selected_count: int,
    excluded: tuple[dict[str, str], ...],
) -> None:
    counts = Counter(row["status"] for row in rows)
    atomic_write_json(
        output / "summary.json",
        {
            "task": "stack",
            "pipeline_version": STACK_TRANSFER_VERSION,
            "benchmark_domain": "all-selected" if not excluded else "task-valid",
            "selected_count": selected_count,
            "excluded_count": len(excluded),
            "excluded": list(excluded),
            "count": len(rows),
            "status_counts": dict(counts),
            "success_rate": counts["SUCCESS"] / max(len(rows), 1),
            "results": rows,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.workers, args.seed_attempts, args.maximum_candidates) <= 0:
        raise ValueError("Worker and search counts must be positive.")
    selected = load_selection_ids(args.selection)
    if args.expect_count and len(selected) != args.expect_count:
        raise ValueError(f"Expected {args.expect_count} objects, found {len(selected)}.")
    eligible, excluded = partition_task_objects("stack", list(selected))
    objects = selected if args.include_ineligible else eligible
    if args.include_ineligible:
        excluded = ()
    if not objects:
        raise ValueError("No Stack-eligible objects remain in the selection.")
    args.output.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[str, dict[str, Any]] = {}
    workers = memory_safe_worker_count(
        min(args.workers, len(objects)), gib_per_worker=args.memory_per_worker_gib
    )
    print(
        f"[resource] requested_workers={args.workers} memory_safe_workers={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_object, item, args=args): item for item in objects}
        for completed, future in enumerate(as_completed(futures), 1):
            object_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "object_id": object_id,
                    "status": "PIPELINE_ERROR",
                    "attempts": 0,
                    "runtime_sec": 0.0,
                    "cached": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            rows_by_id[object_id] = row
            rows = [rows_by_id[item] for item in objects if item in rows_by_id]
            _write_summary(
                args.output, rows, selected_count=len(selected), excluded=excluded
            )
            print(
                f"[{completed:03d}/{len(objects):03d}] {object_id:<36} "
                f"{row['status']:<14} attempts={row['attempts']:02d} "
                f"runtime={row['runtime_sec']:.1f}s",
                flush=True,
            )
    rows = [rows_by_id[item] for item in objects]
    _write_summary(args.output, rows, selected_count=len(selected), excluded=excluded)
    counts = Counter(row["status"] for row in rows)
    print(f"[summary] {dict(counts)}", flush=True)
    return 0 if counts["SUCCESS"] == len(objects) else 2


if __name__ == "__main__":
    raise SystemExit(main())
