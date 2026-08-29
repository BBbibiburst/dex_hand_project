"""Migrate a ranked Lift benchmark to PickPlace with resumable object workers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from source.grasping.task_transfer import PICK_PLACE_PIPELINE_VERSION


def _slug(object_id: str) -> str:
    return object_id.replace(":", "_").replace("/", "_")


def _selection_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("objects", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Selection has no object list: {path}")
    result = tuple(str(row["object_id"] if isinstance(row, dict) else row) for row in rows)
    if len(set(result)) != len(result):
        raise ValueError("Selection contains duplicate object IDs.")
    return result


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _source_roots(
    lattice_root: Path,
    rl_root: Path,
    grasp_root: Path,
    object_id: str,
) -> tuple[Path, ...]:
    slug = _slug(object_id)
    candidates = (
        grasp_root / slug,
        lattice_root / slug,
        lattice_root / "recovery_lift_085mm" / slug,
        rl_root / slug,
    )
    return tuple(path for path in candidates if path.is_dir())


def _run_object(
    object_id: str,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    object_output = args.output / _slug(object_id)
    summary_path = object_output / "summary.json"
    if not args.force and summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("success") and cached.get("pipeline_version") == PICK_PLACE_PIPELINE_VERSION:
            return {
                "object_id": object_id,
                "status": "SUCCESS",
                "attempts": len(cached.get("attempts", [])),
                "runtime_sec": 0.0,
                "cached": True,
                "failure_reason": "",
            }

    roots = _source_roots(args.lattice_root, args.rl_root, args.grasp_root, object_id)
    if not roots:
        return {
            "object_id": object_id,
            "status": "NO_LIFT_SOURCE",
            "attempts": 0,
            "runtime_sec": round(time.perf_counter() - started, 3),
            "cached": False,
            "failure_reason": "no Lift Lattice/PPO directory",
        }
    command = [
        sys.executable,
        "-m",
        "tools.grasping.transfer_lift_to_pick_place",
        "--object-id",
        object_id,
        "--output",
        str(object_output),
        "--seed-attempts",
        str(args.seed_attempts),
        "--maximum-candidates",
        str(args.maximum_candidates),
        "--clearance-height",
        str(args.clearance_height),
    ]
    for root in roots:
        command.extend(("--lift-root", str(root)))
    log_path = args.output / "logs" / f"{_slug(object_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    attempts = payload.get("attempts", [])
    success = bool(payload.get("success"))
    last_failure = ""
    if attempts and not success:
        last_failure = str(attempts[-1].get("failure_reason", ""))
    if child.returncode not in (0, 2) and not success:
        last_failure = f"child exit {child.returncode}; see {log_path}"
    return {
        "object_id": object_id,
        "status": "SUCCESS" if success else "FAILED",
        "attempts": len(attempts),
        "runtime_sec": round(time.perf_counter() - started, 3),
        "cached": False,
        "failure_reason": last_failure,
    }


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["status"] for row in rows)
    _atomic_json(
        output / "summary.json",
        {
            "task": "pick_place",
            "pipeline_version": PICK_PLACE_PIPELINE_VERSION,
            "count": len(rows),
            "status_counts": dict(counts),
            "success_rate": counts["SUCCESS"] / max(len(rows), 1),
            "results": rows,
        },
    )
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/pick_place_top100"))
    parser.add_argument(
        "--lattice-root", type=Path, default=Path("outputs/dex_hand_top100_v2/lattice")
    )
    parser.add_argument("--rl-root", type=Path, default=Path("outputs/dex_hand_top100_v2/rl"))
    parser.add_argument("--grasp-root", type=Path, default=Path("outputs/dex_hand_top100_v2/grasp"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-attempts", type=int, default=6)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    parser.add_argument("--clearance-height", type=float, default=0.065)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.workers, args.seed_attempts, args.maximum_candidates) <= 0:
        raise ValueError("Worker and search counts must be positive.")
    objects = _selection_ids(args.selection)
    if args.expect_count and len(objects) != args.expect_count:
        raise ValueError(f"Expected {args.expect_count} objects, found {len(objects)}.")
    args.output.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(objects))) as pool:
        futures = {
            pool.submit(_run_object, object_id, args=args): object_id for object_id in objects
        }
        completed = 0
        for future in as_completed(futures):
            object_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve full benchmark progress
                row = {
                    "object_id": object_id,
                    "status": "PIPELINE_ERROR",
                    "attempts": 0,
                    "runtime_sec": 0.0,
                    "cached": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            rows_by_id[object_id] = row
            completed += 1
            rows = [rows_by_id[item] for item in objects if item in rows_by_id]
            _write_summary(args.output, rows)
            print(
                f"[{completed:03d}/{len(objects):03d}] {object_id:<36} "
                f"{row['status']:<14} attempts={row['attempts']:02d} "
                f"runtime={row['runtime_sec']:.1f}s",
                flush=True,
            )
    rows = [rows_by_id[item] for item in objects]
    _write_summary(args.output, rows)
    counts = Counter(row["status"] for row in rows)
    print(f"[summary] {dict(counts)}", flush=True)
    return 0 if counts["SUCCESS"] == len(objects) else 2


if __name__ == "__main__":
    raise SystemExit(main())
