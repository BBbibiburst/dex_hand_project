"""Generate and verify the 100 fixed-nut / grasped-bolt variants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from source.grasping.nut_transfer import NUT_TRANSFER_VERSION
from source.workflows.batch import atomic_write_json, memory_safe_worker_count

# Earlier versions used a full-length tapered pin, 14--15 mm clearance and a
# handle overlapping the socket. Their 100/100 artifacts are not compatible
# with the corrected mating-shaft geometry.
_COMPATIBLE_SUCCESS_VERSIONS = frozenset({NUT_TRANSFER_VERSION})


def _run(variant: int, args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    output = args.output / f"variant_{variant:03d}"
    summary = output / "summary.json"
    if summary.is_file() and not args.force:
        old = json.loads(summary.read_text(encoding="utf-8"))
        if old.get("success") and old.get("pipeline_version") in _COMPATIBLE_SUCCESS_VERSIONS:
            return {"variant_index": variant, "status": "SUCCESS", "runtime_sec": 0.0}
    command = [
        sys.executable,
        "-m",
        "tools.grasping.transfer_lift_to_nut",
        "--handle-manifest",
        str(args.handle_manifest),
        "--variant-index",
        str(variant),
        "--output",
        str(output),
        "--seed-attempts",
        str(args.seed_attempts),
    ]
    log = args.output / "logs" / f"variant_{variant:03d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    payload = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
    success = bool(payload.get("success"))
    return {
        "variant_index": variant,
        "status": "SUCCESS" if success else "FAILED",
        "runtime_sec": round(time.perf_counter() - started, 3),
        "failure_reason": ""
        if child.returncode in (0, 2)
        else f"child exit {child.returncode}; see {log}",
    }


def _write_summary(output: Path, rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row["status"]) for row in rows)
    atomic_write_json(
        output / "summary.json",
        {
            "task": "nut_assembly",
            "pipeline_version": NUT_TRANSFER_VERSION,
            "count": len(rows),
            "status_counts": dict(counts),
            "success_rate": counts["SUCCESS"] / max(1, len(rows)),
            "results": rows,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/nut_assembly_top100"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-per-worker-gib", type=float, default=1.5)
    parser.add_argument("--seed-attempts", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.handle_manifest.is_file():
        raise FileNotFoundError(args.handle_manifest)
    if min(args.workers, args.seed_attempts) < 1:
        raise ValueError("workers and seed-attempts must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    variants = tuple(range(100))
    workers = memory_safe_worker_count(
        min(args.workers, len(variants)), gib_per_worker=args.memory_per_worker_gib
    )
    print(f"[resource] requested_workers={args.workers} memory_safe_workers={workers}", flush=True)
    rows_by_variant: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, variant, args): variant for variant in variants}
        for completed, future in enumerate(as_completed(futures), 1):
            variant = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "variant_index": variant,
                    "status": "PIPELINE_ERROR",
                    "runtime_sec": 0.0,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            rows_by_variant[variant] = row
            rows = [rows_by_variant[index] for index in variants if index in rows_by_variant]
            _write_summary(args.output, rows)
            print(
                f"[{completed:03d}/100] variant={variant:03d} "
                f"{row['status']:<14} runtime={float(row['runtime_sec']):.1f}s",
                flush=True,
            )
    rows = [rows_by_variant[index] for index in variants]
    _write_summary(args.output, rows)
    counts = Counter(str(row["status"]) for row in rows)
    print(f"[summary] {dict(counts)}", flush=True)
    return 0 if counts["SUCCESS"] == 100 else 2


if __name__ == "__main__":
    raise SystemExit(main())
