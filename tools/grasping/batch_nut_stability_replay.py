"""Independently replay terminal stability for 100 NutAssembly trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from source.verification.task_stability import replay_nut_terminal_stability
from tools.grasping.batch_pick_place_transfer import _atomic_json
from tools.grasping.batch_push_transfer import _memory_safe_workers


def _verify(index: int, root: Path) -> dict[str, object]:
    manifest = root / f"variant_{index:03d}" / "best_trajectory" / "manifest.json"
    if not manifest.is_file():
        return {"variant_index": index, "status": "MISSING", "error": str(manifest)}
    try:
        result = replay_nut_terminal_stability(manifest)
    except Exception as exc:  # noqa: BLE001
        return {
            "variant_index": index,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "variant_index": index,
        "status": "SUCCESS" if result.success else "FAILED",
        **result.__dict__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-per-worker-gib", type=float, default=0.3)
    args = parser.parse_args(argv)
    workers = _memory_safe_workers(args.workers, gib_per_worker=args.memory_per_worker_gib)
    rows: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_verify, index, args.input): index for index in range(100)}
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            rows[index] = future.result()
            row = rows[index]
            print(f"[{completed:03d}/100] variant={index:03d} {row['status']}", flush=True)
    ordered = [rows[index] for index in range(100)]
    counts = Counter(str(row["status"]) for row in ordered)
    payload = {
        "task": "nut_assembly_stability_replay",
        "count": 100,
        "status_counts": dict(counts),
        "success_rate": counts["SUCCESS"] / 100.0,
        "results": ordered,
    }
    _atomic_json(args.input / "stability_summary.json", payload)
    print(f"[summary] {dict(counts)}", flush=True)
    return 0 if counts["SUCCESS"] == 100 else 2


if __name__ == "__main__":
    raise SystemExit(main())
