"""End-to-end successful-demo BC -> parallel BC-guided residual RL catalogue sweep.

Pipeline:
  reuse any surviving validated trajectories -> if needed rebuild a bootstrap
  pool of successful experts in parallel across GPUs -> hand behavior cloning ->
  BC-guided stage-curriculum arm+hand residual PPO -> authoritative C MuJoCo replay.

The runner works even when the old benchmark trajectories were deleted.  A stale
benchmark summary is treated only as a priority hint for bootstrap scheduling.
Both bootstrap search and the final catalogue sweep are resumable and assign each
subprocess to a fixed CUDA_VISIBLE_DEVICES slot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from source.envs.manipulation.object_catalog import object_ids
from source.rl.grasp_edit.templates import discover_ultra_attempts
from source.rl.imitation.bc import BCTrainConfig, collect_bc_dataset, train_bc_policy
from source.rl.imitation.verification import (
    EXPERT_POOL_VALID,
    EXPERT_PROFILE,
    FINAL_REJECTED,
    FINAL_VERIFIED,
)

SUCCESS_SOURCE_STATUSES = {"ULTRA_SUCCESS", "LATTICE_SUCCESS", "RL_SUCCESS"}
FINAL_STATUSES = (
    EXPERT_POOL_VALID,
    FINAL_VERIFIED,
    FINAL_REJECTED,
    "RL_NO_SUCCESS",
    "MJWARP_SUCCESS_UNVERIFIED",
    "NO_REFERENCE",
    "ERROR",
)
CATALOG_RESULT_SCHEMA_VERSION = 4
BC_ARTIFACT_SCHEMA_VERSION = 5


def _slug(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _gpu_ids(spec: str) -> list[str]:
    if spec.strip().lower() != "auto":
        values = [item.strip() for item in spec.split(",") if item.strip()]
        if not values:
            raise ValueError("--gpus must contain at least one GPU id or 'auto'.")
        return values

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        values = [item.strip() for item in visible.split(",") if item.strip()]
        if values:
            return values

    try:
        child = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
        )
        values = [line.strip() for line in child.stdout.splitlines() if line.strip()]
        if child.returncode == 0 and values:
            return values
    except OSError:
        pass
    return ["0"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("outputs/grasp_edit_benchmark/summary.json"),
    )
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expect-count", type=int, default=127)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_il_rl_catalog"))
    parser.add_argument("--lattice-root", type=Path, default=Path("outputs/grasp_edit_lattice"))
    parser.add_argument("--primitive-root", type=Path, default=Path("outputs/grasp_primitive_rl"))
    parser.add_argument("--ultra-root", type=Path, action="append", dest="ultra_roots")

    # Imitation learning.
    parser.add_argument("--bc-device", default="cuda:0")
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--bc-max-experts-per-object", type=int, default=4)
    parser.add_argument("--bc-validation-objects", type=int, default=4)
    parser.add_argument("--bc-min-rollout-success", type=float, default=0.25)
    parser.add_argument("--rebuild-bc", action="store_true")
    parser.add_argument(
        "--bootstrap-experts",
        type=int,
        default=16,
        help=(
            "Minimum number of distinct expert objects required before BC. "
            "Missing experts are rebuilt automatically in parallel; 0 disables rebuilding."
        ),
    )
    parser.add_argument(
        "--bootstrap-workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent bootstrap-search subprocesses per physical GPU.",
    )
    parser.add_argument("--bootstrap-num-envs", type=int, default=64)
    parser.add_argument("--bootstrap-initial-updates", type=int, default=5)
    parser.add_argument("--bootstrap-mid-updates", type=int, default=10)
    parser.add_argument("--bootstrap-max-updates", type=int, default=15)
    parser.add_argument("--bootstrap-base-candidates", type=int, default=3)
    parser.add_argument("--bootstrap-lattice-max-templates", type=int, default=12)
    parser.add_argument("--bootstrap-lattice-max-executions", type=int, default=32)
    # Parallel RL workers.
    parser.add_argument(
        "--gpus",
        default="auto",
        help="'auto' uses every visible GPU; otherwise comma-separated physical GPU ids.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-std", type=float, default=0.20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--hand-residual-fraction", type=float, default=0.12)
    parser.add_argument("--arm-residual-radians", type=float, default=0.04)
    parser.add_argument("--success-hold-steps", type=int, default=12)
    parser.add_argument("--maximum-object-speed", type=float, default=0.10)
    parser.add_argument("--maximum-object-angular-speed", type=float, default=0.10)
    parser.add_argument("--nconmax", type=int, default=192)
    parser.add_argument("--njmax", type=int, default=768)

    # Missing-prior bootstrap.
    parser.add_argument("--no-auto-ultra", action="store_true")
    parser.add_argument("--ultra-generate-seeds", type=int, default=3)
    parser.add_argument("--ultra-seed-count", type=int, default=100)
    parser.add_argument("--ultra-max-execution-candidates", type=int, default=8)

    parser.add_argument("--train-successful", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _catalog(args: argparse.Namespace) -> list[str]:
    selected = list(object_ids(None if args.dataset == "all" else args.dataset))
    full_run = args.dataset == "all" and not args.object_ids and args.limit is None
    if full_run and args.expect_count and len(selected) != args.expect_count:
        raise RuntimeError(
            f"Catalogue count mismatch: expected {args.expect_count}, found {len(selected)}."
        )
    if args.object_ids:
        requested = set(args.object_ids)
        missing = sorted(requested.difference(selected))
        if missing:
            raise ValueError(f"Unknown object ids: {missing}")
        selected = [item for item in selected if item in requested]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("No catalogue objects selected.")
    return selected


def _benchmark_rows(path: Path) -> dict[str, dict]:
    # The old benchmark may survive after its trajectory directories were deleted.
    # Treat it as a scheduling hint, never as proof that an expert still exists.
    if not path.is_file():
        print(f"[benchmark:missing] {path}; bootstrap will use catalogue order", flush=True)
        return {}
    payload = _read_json(path)
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        print(f"[benchmark:invalid] {path}; bootstrap will use catalogue order", flush=True)
        return {}
    return {
        str(row["object_id"]): dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("object_id")
    }


def _lattice_rows(lattice_root: Path, object_id: str) -> list[dict]:
    payload = _read_json(lattice_root / _slug(object_id) / "index.json")
    rows = payload.get("templates", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _is_residual_manifest(manifest: Path) -> bool:
    return "action_mode" in _read_json(manifest)


def _validate_expert(manifest: Path) -> bool:
    """Admit BC experts only after the expert-profile C-MuJoCo replay."""
    payload = _read_json(manifest)
    if not payload or not bool(payload.get("success", False)):
        return False
    from source.rl.imitation.strict_replay import strict_replay_manifest

    try:
        result = strict_replay_manifest(
            manifest,
            render_mode=None,
            profile=EXPERT_PROFILE,
            use_cache=True,
        )
    except Exception as exc:  # noqa: BLE001 - reject only this expert candidate
        print(
            f"[expert:reject] manifest={manifest} expert_replay_error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    if not result.success:
        print(
            f"[expert:reject] manifest={manifest} "
            f"status={result.verification_status} "
            f"tail_min={result.tail_min_lift:.3f} "
            f"contact={result.tail_contact_fraction:.1%} "
            f"grasp={result.tail_grasp_fraction:.1%} "
            f"opp={result.tail_opposition_mean:.2f} "
            f"speed={result.tail_max_speed:.3f} "
            f"omega={result.tail_max_angular_speed:.3f} "
            f"tail_table={result.tail_robot_table_contact_fraction:.1%} "
            f"tail_pen={1000.0 * result.tail_max_penetration:.1f}mm "
            f"quality={result.quality_score:.2f}",
            flush=True,
        )
    return bool(result.success and result.verification_status == EXPERT_POOL_VALID)


def _expert_candidates_for_object(
    object_id: str,
    *,
    benchmark_root: Path,
    output_root: Path,
    bootstrap_root: Path,
    lattice_root: Path,
    ultra_roots: tuple[Path, ...],
) -> list[Path]:
    """Return every currently materialized success candidate for one object."""
    slug = _slug(object_id)
    candidates: list[Path] = []

    # Newly generated BC-guided success from a previous interrupted run.
    candidates.append(output_root / "rl" / slug / "best_trajectory" / "manifest.json")
    # Bootstrap grasp-edit success generated by this runner.
    candidates.append(
        bootstrap_root
        / "jobs"
        / slug
        / "benchmark"
        / "rl"
        / slug
        / "best_trajectory"
        / "manifest.json"
    )
    # Historical grasp-edit success, if it still exists.
    candidates.append(benchmark_root / "rl" / slug / "best_trajectory" / "manifest.json")

    for lattice in _lattice_rows(lattice_root, object_id):
        if not bool(lattice.get("success")) or not lattice.get("manifest"):
            continue
        candidates.append(_resolve_path(lattice["manifest"], base=Path.cwd()))

    try:
        for manifest, episode in discover_ultra_attempts(
            object_id,
            roots=ultra_roots,
            maximum=32,
        ):
            if episode.success:
                candidates.append(Path(manifest))
    except (FileNotFoundError, RuntimeError, ValueError):
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate.resolve())
    return unique


def _discover_experts(
    *,
    benchmark: Path,
    rows: dict[str, dict],
    catalog: list[str],
    output_root: Path,
    bootstrap_root: Path,
    lattice_root: Path,
    ultra_roots: tuple[Path, ...],
    max_per_object: int,
) -> tuple[list[Path], set[str]]:
    if max_per_object <= 0:
        raise ValueError("--bc-max-experts-per-object must be positive.")
    benchmark_root = benchmark.parent
    experts: list[Path] = []
    expert_objects: set[str] = set()

    for object_id in catalog:
        candidates = _expert_candidates_for_object(
            object_id,
            benchmark_root=benchmark_root,
            output_root=output_root,
            bootstrap_root=bootstrap_root,
            lattice_root=lattice_root,
            ultra_roots=ultra_roots,
        )
        accepted: list[Path] = []
        for candidate in candidates:
            if _validate_expert(candidate):
                accepted.append(candidate)
            if len(accepted) >= max_per_object:
                break
        if accepted:
            experts.extend(accepted)
            expert_objects.add(object_id)
            print(f"[expert] object={object_id} trajectories={len(accepted)}", flush=True)
        elif rows.get(object_id, {}).get("status") in SUCCESS_SOURCE_STATUSES:
            # A stale summary claiming success is useful for scheduling, but not
            # enough for BC because the actual episode/trajectory may have been deleted.
            print(
                f"[expert:stale] object={object_id} "
                f"old_status={rows.get(object_id, {}).get('status')} artifacts=missing",
                flush=True,
            )
    return experts, expert_objects


def _bootstrap_priority(
    catalog: list[str],
    rows: dict[str, dict],
    expert_objects: set[str],
) -> list[str]:
    """Success-first ordering with soft dataset/shape diversity.

    The previous scheduler strictly round-robined every
    dataset/shape bucket.  With no historical benchmark that placed hard EGAD
    objects at the very front and spent 20+ minutes on each before trying easy
    YCB candidates. The current scheduler keeps shape diversity but uses a
    2:1 YCB:EGAD pass when both datasets are present.
    """
    status_rank = {
        "ULTRA_SUCCESS": 0,
        "LATTICE_SUCCESS": 1,
        "RL_SUCCESS": 2,
        "RL_PROMISING": 3,
        "DIRECT_FAILED": 4,
        "NO_REACHABLE_TEMPLATE": 5,
        "NO_ULTRA_PRIOR": 6,
        "PIPELINE_ERROR": 7,
    }
    order = {object_id: index for index, object_id in enumerate(catalog)}
    shape_cache: dict[str, str] = {}

    def lift_hint(row: dict) -> float:
        values = (
            row.get("rl_best_final_lift_mm"),
            row.get("rl_best_lift_mm"),
            row.get("lattice_best_lift_mm"),
            row.get("ultra_best_final_lift_mm"),
            row.get("ultra_best_lift_mm"),
        )
        parsed = []
        for value in values:
            try:
                parsed.append(float(value or 0.0))
            except (TypeError, ValueError):
                parsed.append(0.0)
        return max(parsed, default=0.0)

    def shape_bucket(object_id: str) -> str:
        cached = shape_cache.get(object_id)
        if cached is not None:
            return cached
        dataset = object_id.split(":", 1)[0]
        try:
            import numpy as np
            import trimesh

            from source.grasping.search.catalog import resolve_object

            loaded = trimesh.load_mesh(resolve_object(object_id), process=False)
            mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
            vertices = np.asarray(mesh.vertices, dtype=float)
            extent = np.sort(np.maximum(np.ptp(vertices, axis=0), 1e-6))[::-1]
            a, b, c = map(float, extent)
            if a / b >= 1.8:
                shape = "elongated"
            elif b / c >= 1.8:
                shape = "flat"
            elif a / c <= 1.35:
                shape = "compact"
            else:
                shape = "irregular"
        except Exception:  # noqa: BLE001 - shape is only a scheduling hint
            shape = "unknown"
        value = f"{dataset}:{shape}"
        shape_cache[object_id] = value
        return value

    selected = [item for item in catalog if item not in expert_objects]
    buckets: dict[str, list[str]] = {}
    for object_id in selected:
        buckets.setdefault(shape_bucket(object_id), []).append(object_id)
    for values in buckets.values():
        values.sort(
            key=lambda object_id: (
                status_rank.get(str(rows.get(object_id, {}).get("status", "")), 8),
                -lift_hint(rows.get(object_id, {})),
                order[object_id],
            )
        )

    def drain_dataset(dataset: str, rounds: int) -> list[str]:
        result: list[str] = []
        names = sorted(name for name in buckets if name.startswith(f"{dataset}:"))
        for _ in range(rounds):
            for name in names:
                if buckets[name]:
                    result.append(buckets[name].pop(0))
        return result

    result: list[str] = []
    datasets = {item.split(":", 1)[0] for item in selected}
    while any(buckets[name] for name in buckets):
        if "ycb" in datasets:
            result.extend(drain_dataset("ycb", 2 if "egad" in datasets else 1))
        if "egad" in datasets:
            result.extend(drain_dataset("egad", 1))
        for dataset in sorted(datasets - {"ycb", "egad"}):
            result.extend(drain_dataset(dataset, 1))
    return result


BOOTSTRAP_RESULT_SCHEMA_VERSION = 5


def _bootstrap_job(
    object_id: str,
    *,
    gpu: str,
    budget: int,
    args: argparse.Namespace,
    bootstrap_root: Path,
    ultra_roots: tuple[Path, ...],
) -> dict[str, Any]:
    """Run one breadth-first bootstrap stage for one object.

    ``budget`` is the *maximum PPO updates for this pass*.  Ultra and wrist
    lattice are still tried first and can finish the object with zero PPO.
    Later passes revisit only unresolved objects with larger budgets.
    """
    if budget <= 0:
        raise ValueError("bootstrap budget must be positive.")

    slug = _slug(object_id)
    job_root = bootstrap_root / "jobs" / slug
    benchmark_output = job_root / "benchmark"
    result_path = job_root / "expert_result.json"
    log_path = bootstrap_root / "logs" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job_root.mkdir(parents=True, exist_ok=True)

    # First re-score every already materialized trajectory under the current
    # expert criteria. This lets a restarted run immediately recover trajectories
    # such as the 54 mm cracker-box grasp without recomputing search.
    preexisting = _expert_candidates_for_object(
        object_id,
        benchmark_root=benchmark_output,
        output_root=args.output,
        bootstrap_root=bootstrap_root,
        lattice_root=args.lattice_root,
        ultra_roots=ultra_roots,
    )
    accepted = [path for path in preexisting if _validate_expert(path)][
        : args.bc_max_experts_per_object
    ]
    if accepted:
        result = {
            "schema_version": BOOTSTRAP_RESULT_SCHEMA_VERSION,
            "object_id": object_id,
            "status": EXPERT_POOL_VALID,
            "pipeline_status": "REVALIDATED_EXISTING",
            "manifest": str(accepted[0]),
            "manifests": [str(path) for path in accepted],
            "gpu": gpu,
            "budget": 0,
            "log": str(log_path),
            "return_code": 0,
            "runtime_sec": 0.0,
        }
        _atomic_json(result_path, result)
        return result

    cached = _read_json(result_path)
    cached_budget = int(cached.get("budget", 0) or 0)
    cached_schema = int(cached.get("schema_version", 0) or 0)
    if (
        cached
        and not args.force
        and cached_schema == BOOTSTRAP_RESULT_SCHEMA_VERSION
        and cached.get("status") in {"NO_EXPERT", "ERROR"}
        and cached_budget >= budget
        and not args.retry_failed
    ):
        return dict(cached) | {
            "cached": True,
            "gpu": gpu,
            "log": str(log_path),
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "tools.grasping.batch_grasp_edit",
        "--object-id",
        object_id,
        "--expect-count",
        "0",
        "--output",
        str(benchmark_output),
        "--num-envs",
        str(args.bootstrap_num_envs),
        # Breadth-first pass: do not let one object automatically consume the
        # complete 5->10->15 budget.  Each catalogue pass grants one ceiling.
        "--initial-updates",
        str(budget),
        "--mid-updates",
        str(budget),
        "--max-updates",
        str(budget),
        "--device",
        "cuda:0",
        "--ultra-seed-count",
        str(args.ultra_seed_count),
        "--ultra-generate-seeds",
        str(args.ultra_generate_seeds),
        "--ultra-max-execution-candidates",
        str(args.ultra_max_execution_candidates),
        "--base-candidates",
        str(args.bootstrap_base_candidates),
        "--lattice-max-templates",
        str(args.bootstrap_lattice_max_templates),
        "--lattice-max-executions",
        str(args.bootstrap_lattice_max_executions),
    ]
    for ultra_root in ultra_roots:
        command.extend(["--ultra-root", str(ultra_root)])

    # Different breadth-first budgets produce different benchmark signatures.
    # Existing Ultra attempts and lattice templates are still reused internally;
    # --force only prevents the per-object summary from hiding a new stage.
    if args.force or cached_budget < budget or cached_schema != BOOTSTRAP_RESULT_SCHEMA_VERSION:
        command.append("--force")

    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n=== bootstrap object={object_id} gpu={gpu} "
            f"budget={budget} started={_utc_now()} ===\n"
        )
        log.write(f"[command] {' '.join(command)}\n")
        log.flush()
        child = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    candidates = _expert_candidates_for_object(
        object_id,
        benchmark_root=benchmark_output,
        output_root=args.output,
        bootstrap_root=bootstrap_root,
        lattice_root=args.lattice_root,
        ultra_roots=ultra_roots,
    )
    accepted = [path for path in candidates if _validate_expert(path)][
        : args.bc_max_experts_per_object
    ]

    summary = _read_json(benchmark_output / "summary.json")
    rows = summary.get("results", []) if isinstance(summary.get("results", []), list) else []
    old_status = rows[0].get("status") if rows else ""
    result = {
        "schema_version": BOOTSTRAP_RESULT_SCHEMA_VERSION,
        "object_id": object_id,
        "status": EXPERT_POOL_VALID if accepted else "NO_EXPERT",
        "pipeline_status": old_status,
        "manifest": str(accepted[0]) if accepted else "",
        "manifests": [str(path) for path in accepted],
        "gpu": gpu,
        "budget": int(budget),
        "log": str(log_path),
        "return_code": int(child.returncode),
        "runtime_sec": round(time.perf_counter() - started, 3),
    }
    _atomic_json(result_path, result)
    return result


def _parallel_bootstrap_experts(
    *,
    experts: list[Path],
    expert_objects: set[str],
    catalog: list[str],
    rows: dict[str, dict],
    gpu_ids: list[str],
    args: argparse.Namespace,
    bootstrap_root: Path,
    ultra_roots: tuple[Path, ...],
) -> tuple[list[Path], set[str]]:
    """Breadth-first parallel expert bootstrap.

    Every object first receives a cheap 1-update ceiling (Ultra/lattice can
    still succeed with zero PPO).  Only if the expert target is still unmet do
    unresolved objects receive 5, then 10, then 15 update ceilings.  This
    prevents a hard EGAD object from monopolizing a worker for ~20 minutes while
    easy expert candidates remain untried.
    """
    target = min(max(0, int(args.bootstrap_experts)), len(catalog))
    if target <= len(expert_objects):
        print(
            f"[bootstrap:skip] expert_objects={len(expert_objects)} target={target}",
            flush=True,
        )
        return experts, expert_objects
    if target == 0:
        return experts, expert_objects
    if args.bootstrap_workers_per_gpu <= 0 or args.bootstrap_num_envs <= 0:
        raise ValueError("bootstrap-workers-per-gpu and bootstrap-num-envs must be positive.")
    if not (
        0
        < args.bootstrap_initial_updates
        <= args.bootstrap_mid_updates
        <= args.bootstrap_max_updates
    ):
        raise ValueError(
            "Require 0 < bootstrap-initial-updates <= bootstrap-mid-updates "
            "<= bootstrap-max-updates."
        )

    candidates = _bootstrap_priority(catalog, rows, expert_objects)
    slots = [gpu for gpu in gpu_ids for _ in range(args.bootstrap_workers_per_gpu)]
    stage_budgets: list[int] = []
    for value in (
        1,
        int(args.bootstrap_initial_updates),
        int(args.bootstrap_mid_updates),
        int(args.bootstrap_max_updates),
    ):
        if value > 0 and value not in stage_budgets:
            stage_budgets.append(value)

    print(
        f"[bootstrap] need={target - len(expert_objects)} target={target} "
        f"candidates={len(candidates)} gpu_slots={slots} envs/job={args.bootstrap_num_envs} "
        f"breadth_first_budgets={stage_budgets}",
        flush=True,
    )

    if args.dry_run:
        for index, object_id in enumerate(candidates[: max(target * 2, len(slots))], 1):
            print(
                f"[bootstrap:plan {index:03d}] object={object_id} "
                f"old_status={rows.get(object_id, {}).get('status', '')}",
                flush=True,
            )
        return experts, expert_objects

    lock = threading.Lock()
    path_keys = {str(path.resolve()) for path in experts if path.is_file()}
    attempted: dict[str, dict] = {}

    # Load results that match the current schema. Older cache rows are ignored
    # because expert-admission criteria may have changed.
    previous_summary = _read_json(bootstrap_root / "summary.json")
    for item in previous_summary.get("results", []):
        if not isinstance(item, dict) or not item.get("object_id"):
            continue
        if int(item.get("schema_version", 0) or 0) == BOOTSTRAP_RESULT_SCHEMA_VERSION:
            attempted[str(item["object_id"])] = dict(item)

    def persist_bootstrap(stage_budget: int | None = None) -> None:
        _atomic_json(
            bootstrap_root / "summary.json",
            {
                "schema_version": BOOTSTRAP_RESULT_SCHEMA_VERSION,
                "updated_at": _utc_now(),
                "target_expert_objects": target,
                "expert_objects": len(expert_objects),
                "expert_trajectories": len(experts),
                "breadth_first_budgets": stage_budgets,
                "active_budget": stage_budget,
                "results": [attempted[key] for key in catalog if key in attempted],
            },
        )

    def run_stage(stage_budget: int) -> None:
        unresolved = [
            object_id
            for object_id in candidates
            if object_id not in expert_objects
            and (
                args.retry_failed
                or int(attempted.get(object_id, {}).get("budget", 0) or 0) < stage_budget
                or int(attempted.get(object_id, {}).get("schema_version", 0) or 0)
                != BOOTSTRAP_RESULT_SCHEMA_VERSION
            )
        ]
        if not unresolved:
            return

        print(
            f"[bootstrap:stage] budget={stage_budget} unresolved={len(unresolved)} "
            f"experts={len(expert_objects)}/{target}",
            flush=True,
        )
        work: queue.Queue[str] = queue.Queue()
        for object_id in unresolved:
            work.put(object_id)

        def worker(gpu: str) -> None:
            while True:
                with lock:
                    if len(expert_objects) >= target:
                        return
                try:
                    object_id = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    result = _bootstrap_job(
                        object_id,
                        gpu=gpu,
                        budget=stage_budget,
                        args=args,
                        bootstrap_root=bootstrap_root,
                        ultra_roots=ultra_roots,
                    )
                except Exception as exc:  # noqa: BLE001 - keep other bootstrap jobs running
                    result = {
                        "schema_version": BOOTSTRAP_RESULT_SCHEMA_VERSION,
                        "object_id": object_id,
                        "status": "ERROR",
                        "gpu": gpu,
                        "budget": stage_budget,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                with lock:
                    attempted[object_id] = result
                    for value in result.get("manifests", []):
                        path = Path(value)
                        if not path.is_file():
                            continue
                        key = str(path.resolve())
                        if key not in path_keys:
                            path_keys.add(key)
                            experts.append(path.resolve())
                    if result.get("status") == EXPERT_POOL_VALID:
                        expert_objects.add(object_id)
                    persist_bootstrap(stage_budget)
                    print(
                        f"[bootstrap:done] object={object_id:<34} "
                        f"status={result.get('status'):<12} gpu={gpu} "
                        f"budget={int(result.get('budget', stage_budget) or 0):>2} "
                        f"experts={len(expert_objects)}/{target} "
                        f"runtime={float(result.get('runtime_sec', 0.0) or 0.0):.1f}s",
                        flush=True,
                    )
                work.task_done()

        threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    for stage_budget in stage_budgets:
        if len(expert_objects) >= target:
            break
        run_stage(stage_budget)

    persist_bootstrap(None)
    if len(expert_objects) < target:
        raise RuntimeError(
            f"Breadth-first bootstrap exhausted budgets {stage_budgets} with only "
            f"{len(expert_objects)}/{target} validated expert objects. Inspect "
            f"{bootstrap_root / 'summary.json'} and logs."
        )
    print(
        f"[bootstrap:ready] expert_objects={len(expert_objects)} "
        f"expert_trajectories={len(experts)}",
        flush=True,
    )
    return experts, expert_objects


def _bc_signature(experts: list[Path], args: argparse.Namespace) -> str:
    source_files = (
        Path("source/rl/imitation/bc.py"),
        Path("source/rl/imitation/geometry_env.py"),
        Path("source/rl/imitation/guided_env.py"),
        Path("source/rl/imitation/strict_replay.py"),
        Path("source/rl/imitation/verification.py"),
        Path("source/rl/residual/env.py"),
    )
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
        if path.is_file()
    }
    expert_hashes = []
    for path in experts:
        try:
            expert_hashes.append(
                [str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()]
            )
        except OSError:
            expert_hashes.append([str(path), "missing"])
    payload = {
        "pipeline": "grasp_hand_bc",
        "experts": expert_hashes,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_learning_rate": args.bc_learning_rate,
        "bc_validation_objects": args.bc_validation_objects,
        "source_hashes": source_hashes,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _prepare_bc(experts: list[Path], args: argparse.Namespace) -> Path:
    bc_root = args.output / "bc"
    dataset = bc_root / "grasp_hand_bc_dataset.npz"
    checkpoint = bc_root / "grasp_hand_bc.pt"
    signature_path = bc_root / "signature.json"
    signature = _bc_signature(experts, args)
    stored = _read_json(signature_path)
    reusable = (
        not args.rebuild_bc
        and stored.get("schema_version") == BC_ARTIFACT_SCHEMA_VERSION
        and stored.get("signature") == signature
        and dataset.is_file()
        and dataset.with_suffix(".json").is_file()
        and checkpoint.is_file()
        and (bc_root / "validation.json").is_file()
    )
    if reusable:
        print(f"[bc:reuse] dataset={dataset} checkpoint={checkpoint}", flush=True)
        return checkpoint

    bc_root.mkdir(parents=True, exist_ok=True)
    info = collect_bc_dataset(
        experts,
        output=dataset,
        device=args.bc_device,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    _atomic_json(
        bc_root / "experts.json",
        {
            "schema_version": BC_ARTIFACT_SCHEMA_VERSION,
            "count": len(experts),
            "status": EXPERT_POOL_VALID,
            "experts": [
                {
                    "object_id": object_id,
                    "manifest": str(path),
                    "verification_status": EXPERT_POOL_VALID,
                }
                for path, object_id in zip(
                    experts,
                    info.expert_object_ids,
                    strict=True,
                )
            ],
        },
    )
    print(
        f"[bc:data-ready] frames={info.observations} experts={info.experts} "
        f"objects={info.objects} obs={info.obs_dim} schema={info.observation_schema.get('schema_version')}",
        flush=True,
    )
    train_bc_policy(
        dataset,
        checkpoint=checkpoint,
        device=args.bc_device,
        config=BCTrainConfig(
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            learning_rate=args.bc_learning_rate,
        ),
    )

    from source.rl.imitation.evaluate import evaluate_bc_checkpoint

    metadata = _read_json(checkpoint.with_suffix(".json"))
    validation_ids = set(metadata.get("validation_object_ids", []))
    validation_by_object: dict[str, Path] = {}
    for path, object_id in zip(experts, info.expert_object_ids, strict=True):
        if object_id in validation_ids and object_id not in validation_by_object:
            validation_by_object[object_id] = path
    if not validation_by_object:
        for path, object_id in zip(experts, info.expert_object_ids, strict=True):
            validation_by_object.setdefault(object_id, path)
            if len(validation_by_object) >= max(1, args.bc_validation_objects):
                break
    validation_manifests = list(validation_by_object.values())
    validation = evaluate_bc_checkpoint(
        validation_manifests,
        checkpoint=checkpoint,
        device=args.bc_device,
        maximum_objects=args.bc_validation_objects,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    _atomic_json(
        bc_root / "validation.json",
        {"schema_version": BC_ARTIFACT_SCHEMA_VERSION, **validation},
    )
    print(
        f"[bc:rollout-validation] objects={validation['objects']} "
        f"success={validation['success_rate']:.1%}",
        flush=True,
    )
    if validation["success_rate"] < args.bc_min_rollout_success:
        raise RuntimeError(
            f"BC-only held-out rollout success {validation['success_rate']:.1%} is below "
            f"--bc-min-rollout-success={args.bc_min_rollout_success:.1%}; refusing full RL sweep."
        )

    _atomic_json(
        signature_path,
        {"schema_version": BC_ARTIFACT_SCHEMA_VERSION, "signature": signature},
    )
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return checkpoint


def _best_reference(
    object_id: str,
    *,
    benchmark_root: Path,
    bootstrap_root: Path,
    primitive_root: Path,
    lattice_root: Path,
    ultra_roots: tuple[Path, ...],
) -> Path | None:
    """Choose the physically best available full reference, not max-lift-first."""
    slug = _slug(object_id)
    candidates: list[Path] = []
    direct = (
        primitive_root / slug / "best_attempt" / "manifest.json",
        bootstrap_root
        / "jobs"
        / slug
        / "benchmark"
        / "rl"
        / slug
        / "best_attempt"
        / "manifest.json",
        benchmark_root / "rl" / slug / "best_attempt" / "manifest.json",
    )
    candidates.extend(path.resolve() for path in direct if path.is_file())

    rows = _lattice_rows(lattice_root, object_id)
    rows.sort(
        key=lambda row: (
            bool(row.get("success", False)),
            float(row.get("source_lift", 0.0) or 0.0),
            -float(row.get("precheck_score", 1e9) or 1e9),
        ),
        reverse=True,
    )
    for row in rows[:4]:
        value = row.get("manifest")
        if value:
            path = _resolve_path(value, base=Path.cwd())
            if path.is_file():
                candidates.append(path.resolve())

    try:
        attempts = discover_ultra_attempts(object_id, roots=ultra_roots, maximum=3)
    except (FileNotFoundError, RuntimeError, ValueError):
        attempts = []
    candidates.extend(Path(row[0]).resolve() for row in attempts if Path(row[0]).is_file())

    unique = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        return None

    from source.rl.imitation.strict_replay import strict_replay_manifest

    scored = []
    for path in unique:
        try:
            result = strict_replay_manifest(
                path,
                render_mode=None,
                profile=EXPERT_PROFILE,
                use_cache=True,
            )
            scored.append((float(result.quality_score), bool(result.success), path))
        except Exception as exc:  # noqa: BLE001 - score the remaining references
            print(
                f"[reference:skip] manifest={path} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
    if not scored:
        return unique[0]
    scored.sort(key=lambda row: (row[1], row[0]), reverse=True)
    quality, strict_success, path = scored[0]
    print(
        f"[reference] object={object_id} expert_pool_valid={strict_success} "
        f"quality={quality:.2f} path={path}",
        flush=True,
    )
    return path


def _bootstrap_ultra(
    object_id: str,
    *,
    gpu: str,
    ultra_root: Path,
    args: argparse.Namespace,
    log,
) -> None:
    if args.no_auto_ultra:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    for seed in range(args.ultra_generate_seeds):
        output = ultra_root / _slug(object_id) / f"seed_{seed:04d}"
        command = [
            sys.executable,
            "-m",
            "tools.ultradexgrasp.generate",
            "--object-id",
            object_id,
            "--seed",
            str(seed),
            "--seed-count",
            str(args.ultra_seed_count),
            "--max-execution-candidates",
            str(args.ultra_max_execution_candidates),
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ]
        if output.exists():
            command.append("--overwrite")
        log.write(f"\n[bootstrap-ultra] {' '.join(command)}\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        try:
            if discover_ultra_attempts(object_id, roots=(ultra_root,), maximum=1):
                return
        except (FileNotFoundError, RuntimeError, ValueError):
            pass


def _run_object(
    object_id: str,
    *,
    gpu: str,
    bc_checkpoint: Path,
    args: argparse.Namespace,
    ultra_roots: tuple[Path, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output / "rl" / _slug(object_id)
    log_path = args.output / "logs" / f"{_slug(object_id)}.log"
    output.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_json(output / "result.json")
    if (
        existing
        and existing.get("schema_version") == CATALOG_RESULT_SCHEMA_VERSION
        and not args.force
    ):
        status = existing.get("status")
        if status == FINAL_VERIFIED or not args.retry_failed:
            return dict(existing) | {"cached": True, "gpu": gpu, "log": str(log_path)}
    elif existing and existing.get("schema_version") != CATALOG_RESULT_SCHEMA_VERSION:
        print(
            f"[cache:invalidate] object={object_id} old_schema={existing.get('schema_version')}",
            flush=True,
        )
    if args.force and output.exists():
        shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== object={object_id} gpu={gpu} started={_utc_now()} ===\n")
        reference = _best_reference(
            object_id,
            benchmark_root=args.benchmark.parent,
            bootstrap_root=args.output / "bootstrap",
            primitive_root=args.primitive_root,
            lattice_root=args.lattice_root,
            ultra_roots=ultra_roots,
        )
        if reference is None:
            _bootstrap_ultra(
                object_id,
                gpu=gpu,
                ultra_root=ultra_roots[0],
                args=args,
                log=log,
            )
            reference = _best_reference(
                object_id,
                benchmark_root=args.benchmark.parent,
                bootstrap_root=args.output / "bootstrap",
                primitive_root=args.primitive_root,
                lattice_root=args.lattice_root,
                ultra_roots=ultra_roots,
            )
        if reference is None:
            result = {
                "schema_version": CATALOG_RESULT_SCHEMA_VERSION,
                "object_id": object_id,
                "status": "NO_REFERENCE",
                "gpu": gpu,
                "log": str(log_path),
                "runtime_sec": round(time.perf_counter() - started, 3),
            }
            _atomic_json(output / "result.json", result)
            return result

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-m",
            "apps.train_bc_guided_grasp_rl",
            "--reference",
            str(reference),
            "--bc-checkpoint",
            str(bc_checkpoint),
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--num-envs",
            str(args.num_envs),
            "--updates",
            str(args.updates),
            "--rollout-steps",
            str(args.rollout_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--initial-std",
            str(args.initial_std),
            "--save-every",
            str(args.save_every),
            "--hand-residual-fraction",
            str(args.hand_residual_fraction),
            "--arm-residual-radians",
            str(args.arm_residual_radians),
            "--success-hold-steps",
            str(args.success_hold_steps),
            "--maximum-object-speed",
            str(args.maximum_object_speed),
            "--maximum-object-angular-speed",
            str(args.maximum_object_angular_speed),
            "--nconmax",
            str(args.nconmax),
            "--njmax",
            str(args.njmax),
            "--seed",
            str(int(hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:8], 16) % 100000),
        ]
        if args.force:
            command.append("--no-auto-resume")
        log.write(f"[command] {' '.join(command)}\n")
        log.flush()
        child = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    result = _read_json(output / "result.json")
    if not result:
        result = {
            "schema_version": CATALOG_RESULT_SCHEMA_VERSION,
            "object_id": object_id,
            "status": "ERROR",
            "error": f"child_returncode={child.returncode}; no result.json",
        }
        _atomic_json(output / "result.json", result)
    result.update(
        {
            "gpu": gpu,
            "reference": str(reference),
            "log": str(log_path),
            "return_code": child.returncode,
            "runtime_sec": round(time.perf_counter() - started, 3),
        }
    )
    _atomic_json(output / "result.json", result)
    return result


def _write_summary(output: Path, catalog: list[str], rows: dict[str, dict]) -> None:
    ordered = [rows[item] for item in catalog if item in rows]
    counts = Counter(str(row.get("status", "")) for row in ordered)
    payload = {
        "schema_version": 2,
        "updated_at": _utc_now(),
        "selected_objects": len(catalog),
        "completed_objects": len(ordered),
        "expert_pool_valid": counts.get(EXPERT_POOL_VALID, 0),
        "final_verified": counts.get(FINAL_VERIFIED, 0),
        "verified_total": counts.get(FINAL_VERIFIED, 0),
        "new_verified": counts.get(FINAL_VERIFIED, 0),
        "status_counts": {status: counts.get(status, 0) for status in FINAL_STATUSES},
        "results": ordered,
    }
    _atomic_json(output / "summary.json", payload)
    fields = [
        "object_id",
        "status",
        "gpu",
        "reference",
        "runtime_sec",
        "return_code",
        "log",
        "error",
    ]
    csv_path = output / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
    temp.replace(csv_path)


def run(args: argparse.Namespace) -> int:
    if args.workers_per_gpu <= 0 or args.num_envs <= 0 or args.updates <= 0:
        raise ValueError("workers-per-gpu, num-envs and updates must be positive.")
    if args.bootstrap_experts < 0:
        raise ValueError("--bootstrap-experts must be >= 0.")
    if args.bc_validation_objects <= 0:
        raise ValueError("--bc-validation-objects must be positive.")
    if not 0.0 <= args.bc_min_rollout_success <= 1.0:
        raise ValueError("--bc-min-rollout-success must lie in [0, 1].")
    if (
        min(
            args.bootstrap_base_candidates,
            args.bootstrap_lattice_max_templates,
            args.bootstrap_lattice_max_executions,
        )
        <= 0
    ):
        raise ValueError("bootstrap lattice/search counts must be positive.")
    gpu_ids = _gpu_ids(args.gpus)
    print(f"[gpu] selected={gpu_ids} source={args.gpus}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = _catalog(args)
    benchmark_rows = _benchmark_rows(args.benchmark)
    bootstrap_root = args.output / "bootstrap"
    bootstrap_ultra_root = bootstrap_root / "ultra"
    configured_ultra_roots = (
        tuple(args.ultra_roots)
        if args.ultra_roots
        else (Path("outputs/ultradexgrasp"), Path("outputs/ultradexgrasp_catalog"))
    )
    # Put the runner-owned root first so every newly generated prior is kept under
    # the resumable output tree rather than scattered into a deleted old output.
    ultra_roots_list = [bootstrap_ultra_root, *configured_ultra_roots]
    ultra_roots: tuple[Path, ...] = tuple(dict.fromkeys(Path(path) for path in ultra_roots_list))

    experts, expert_objects = _discover_experts(
        benchmark=args.benchmark,
        rows=benchmark_rows,
        catalog=catalog,
        output_root=args.output,
        bootstrap_root=bootstrap_root,
        lattice_root=args.lattice_root,
        ultra_roots=ultra_roots,
        max_per_object=args.bc_max_experts_per_object,
    )
    print(
        f"[prepare] catalogue={len(catalog)} existing_expert_objects={len(expert_objects)} "
        f"expert_trajectories={len(experts)}",
        flush=True,
    )

    experts, expert_objects = _parallel_bootstrap_experts(
        experts=experts,
        expert_objects=expert_objects,
        catalog=catalog,
        rows=benchmark_rows,
        gpu_ids=gpu_ids,
        args=args,
        bootstrap_root=bootstrap_root,
        ultra_roots=ultra_roots,
    )
    if args.dry_run:
        print("[done] dry-run stops before BC training and catalogue RL.", flush=True)
        return 0
    if not experts:
        raise RuntimeError(
            "No validated experts are available for BC. Set --bootstrap-experts to a positive "
            "value or restore at least one successful trajectory."
        )
    bc_checkpoint = _prepare_bc(experts, args)
    if args.prepare_only:
        print(f"[done] prepare-only bc={bc_checkpoint}", flush=True)
        return 0

    rows: dict[str, dict] = {}
    targets: list[str] = []
    for object_id in catalog:
        if object_id in expert_objects and not args.train_successful:
            rows[object_id] = {
                "object_id": object_id,
                "status": EXPERT_POOL_VALID,
                "gpu": "",
                "reference": "",
                "runtime_sec": 0.0,
                "return_code": 0,
                "log": "",
            }
        else:
            targets.append(object_id)
    _write_summary(args.output, catalog, rows)

    slots = [gpu for gpu in gpu_ids for _ in range(args.workers_per_gpu)]
    print(
        f"[sweep] targets={len(targets)} gpu_slots={slots} envs/job={args.num_envs} "
        f"updates={args.updates} output={args.output}",
        flush=True,
    )
    if args.dry_run:
        for index, object_id in enumerate(targets, 1):
            print(f"[{index:03d}/{len(targets):03d}] {object_id}")
        return 0

    work: queue.Queue[tuple[int, str]] = queue.Queue()
    for index, object_id in enumerate(targets, 1):
        work.put((index, object_id))
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                _index, object_id = work.get_nowait()
            except queue.Empty:
                return
            try:
                result = _run_object(
                    object_id,
                    gpu=gpu,
                    bc_checkpoint=bc_checkpoint,
                    args=args,
                    ultra_roots=ultra_roots,
                )
            except Exception as exc:  # noqa: BLE001 - keep the catalogue sweep running
                result = {
                    "schema_version": CATALOG_RESULT_SCHEMA_VERSION,
                    "object_id": object_id,
                    "status": "ERROR",
                    "gpu": gpu,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with lock:
                rows[object_id] = result
                _write_summary(args.output, catalog, rows)
                done = len([item for item in targets if item in rows])
                print(
                    f"[{done:03d}/{len(targets):03d}] object={object_id:<34} "
                    f"status={result.get('status'):<24} gpu={gpu} "
                    f"runtime={float(result.get('runtime_sec', 0.0) or 0.0):.1f}s",
                    flush=True,
                )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    _write_summary(args.output, catalog, rows)
    summary = _read_json(args.output / "summary.json")
    counts = summary.get("status_counts", {})
    print("\n[summary]", flush=True)
    for status in FINAL_STATUSES:
        if counts.get(status):
            print(f"  {status:<26} {counts[status]:3d}", flush=True)
    print(
        f"  {'EXPERT_POOL_VALID_TOTAL':<26} "
        f"{int(summary.get('expert_pool_valid', 0)):3d}\n"
        f"  {'FINAL_VERIFIED_TOTAL':<26} "
        f"{int(summary.get('final_verified', 0)):3d}\n"
        f"[done] json={args.output / 'summary.json'} csv={args.output / 'summary.csv'}",
        flush=True,
    )
    return 0 if counts.get("ERROR", 0) == 0 else 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
