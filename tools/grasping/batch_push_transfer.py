"""Generate and verify Push demonstrations for a selected object benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from source.envs.manipulation.object_catalog import partition_task_objects
from source.grasping.push_transfer import PUSH_TRANSFER_VERSION
from source.workflows.batch import (
    atomic_write_json,
    lift_source_roots,
    load_selection_ids,
    memory_safe_worker_count,
    object_slug,
)
from source.workflows.task_domains import is_prevalidated_task_manifest

# Only reuse artifacts produced by the current physical protocol. Older
# successes may use different contact geometry or verification semantics.
_COMPATIBLE_SUCCESS_VERSIONS = frozenset({PUSH_TRANSFER_VERSION})


def _run(oid, args):
    start = time.perf_counter()
    out = args.output / object_slug(oid)
    summary = out / "summary.json"
    if summary.is_file() and not args.force:
        old = json.loads(summary.read_text())
        if old.get("pipeline_version") in _COMPATIBLE_SUCCESS_VERSIONS:
            return {
                "object_id": oid,
                "status": "SUCCESS" if old.get("success") else "FAILED",
                "runtime_sec": 0.0,
                "cached": True,
                "failure_reason": "",
            }
    roots = lift_source_roots(
        args.lattice_root, args.rl_root, args.grasp_root, oid, tuple(args.extra_lift_roots or ())
    )
    cmd = [
        sys.executable,
        "-m",
        "tools.grasping.transfer_lift_to_push",
        "--object-id",
        oid,
        "--output",
        str(out),
        "--maximum-candidates",
        str(args.maximum_candidates),
    ]
    for root in roots:
        cmd.extend(("--lift-root", str(root)))
    log = args.output / "logs" / f"{object_slug(oid)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        child = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    payload = json.loads(summary.read_text()) if summary.is_file() else {}
    return {
        "object_id": oid,
        "status": "SUCCESS" if payload.get("success") else "FAILED",
        "runtime_sec": round(time.perf_counter() - start, 3),
        "cached": False,
        "failure_reason": ""
        if child.returncode in (0, 2)
        else f"child exit {child.returncode}; see {log}",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--expect-count", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("outputs/push_top100"))
    p.add_argument("--lattice-root", type=Path, default=Path("outputs/dex_hand_top100_v2/lattice"))
    p.add_argument("--rl-root", type=Path, default=Path("outputs/dex_hand_top100_v2/rl"))
    p.add_argument("--grasp-root", type=Path, default=Path("outputs/dex_hand_top100_v2/grasp"))
    p.add_argument("--extra-lift-root", action="append", type=Path, dest="extra_lift_roots")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--memory-per-worker-gib", type=float, default=1.5)
    p.add_argument("--maximum-candidates", type=int, default=12)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Run every selected object as a coverage stress test instead of the Push domain.",
    )
    args = p.parse_args(argv)
    selected = load_selection_ids(args.selection)
    if args.expect_count and len(selected) != args.expect_count:
        raise ValueError(f"Expected {args.expect_count} objects, found {len(selected)}.")
    selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
    prevalidated = is_prevalidated_task_manifest(selection_payload, "push")
    if prevalidated and args.include_ineligible:
        raise ValueError(
            "--include-ineligible requires the source Top100 selection, not a task-valid subset."
        )
    if prevalidated:
        eligible = selected
        excluded = tuple(selection_payload.get("excluded", ()))
    else:
        eligible, excluded = partition_task_objects("push", list(selected))
    ids = selected if args.include_ineligible else eligible
    if args.include_ineligible:
        excluded = ()
    if not ids:
        raise ValueError("No Push-eligible objects remain in the selection.")
    rows = {}
    args.output.mkdir(parents=True, exist_ok=True)
    workers = memory_safe_worker_count(
        min(args.workers, len(ids)), gib_per_worker=args.memory_per_worker_gib
    )
    print(
        f"[resource] requested_workers={args.workers} memory_safe_workers={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs = {pool.submit(_run, x, args): x for x in ids}
        for n, f in enumerate(as_completed(fs), 1):
            oid = fs[f]
            try:
                row = f.result()
            except Exception as exc:
                row = {
                    "object_id": oid,
                    "status": "PIPELINE_ERROR",
                    "runtime_sec": 0.0,
                    "cached": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            rows[oid] = row
            ordered = [rows[x] for x in ids if x in rows]
            counts = Counter(x["status"] for x in ordered)
            atomic_write_json(
                args.output / "summary.json",
                {
                    "task": "push",
                    "pipeline_version": PUSH_TRANSFER_VERSION,
                    "benchmark_domain": "task-valid" if prevalidated else (
                        "all-selected" if not excluded else "legacy-task-filter"
                    ),
                    "domain_version": selection_payload.get("domain_version"),
                    "source_count": selection_payload.get("source_count", len(selected)),
                    "selected_count": len(selected),
                    "excluded_count": len(excluded),
                    "excluded": list(excluded),
                    "count": len(ordered),
                    "status_counts": dict(counts),
                    "success_rate": counts["SUCCESS"] / max(len(ordered), 1),
                    "results": ordered,
                },
            )
            print(
                f"[{n:03d}/{len(ids):03d}] {oid:<36} {row['status']:<14} runtime={row['runtime_sec']:.1f}s",
                flush=True,
            )
    counts = Counter(x["status"] for x in rows.values())
    print(f"[summary] {dict(counts)}")
    return 0 if counts["SUCCESS"] == len(ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
