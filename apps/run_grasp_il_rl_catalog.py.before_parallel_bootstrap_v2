"""End-to-end successful-demo BC -> parallel BC-guided residual RL catalogue sweep.

Pipeline:
  validated successful trajectories -> hand behavior cloning -> BC-guided
  stage-curriculum arm+hand residual PPO -> authoritative C MuJoCo replay.

The runner is resumable at object granularity and assigns each subprocess to a
fixed CUDA_VISIBLE_DEVICES slot, making it suitable for multi-GPU servers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from source.envs.manipulation.object_catalog import object_ids
from source.rl.grasp_edit.templates import discover_ultra_attempts
from source.rl.imitation.bc import BCTrainConfig, collect_bc_dataset, train_bc_policy


SUCCESS_SOURCE_STATUSES = {"ULTRA_SUCCESS", "LATTICE_SUCCESS", "RL_SUCCESS"}
FINAL_STATUSES = (
    "EXPERT_REUSED",
    "VERIFIED_SUCCESS",
    "REPLAY_FAILED",
    "RL_NO_SUCCESS",
    "MJWARP_SUCCESS_UNVERIFIED",
    "NO_REFERENCE",
    "ERROR",
)


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
    parser.add_argument(
        "--lattice-root", type=Path, default=Path("outputs/grasp_edit_lattice")
    )
    parser.add_argument(
        "--primitive-root", type=Path, default=Path("outputs/grasp_primitive_rl")
    )
    parser.add_argument(
        "--ultra-root", type=Path, action="append", dest="ultra_roots"
    )

    # Imitation learning.
    parser.add_argument("--bc-device", default="cuda:0")
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--bc-max-experts-per-object", type=int, default=4)
    parser.add_argument("--rebuild-bc", action="store_true")
    parser.add_argument(
        "--trust-rl-experts",
        action="store_true",
        help="Use RL_SUCCESS trajectories without re-validating them in classic MuJoCo.",
    )

    # Parallel RL workers.
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU ids, e.g. 0,1,2,3,4,5.",
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
    payload = _read_json(path)
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Benchmark {path} has no results list.")
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


def _validate_expert(manifest: Path, *, trust_rl: bool) -> bool:
    payload = _read_json(manifest)
    if not payload or not bool(payload.get("success", False)):
        return False
    if not _is_residual_manifest(manifest) or trust_rl:
        return True
    from source.rl.residual.replay import replay_residual_trajectory

    try:
        result = replay_residual_trajectory(manifest, render_mode=None)
    except Exception as exc:
        print(f"[expert:reject] manifest={manifest} replay_error={type(exc).__name__}: {exc}")
        return False
    if not result.success:
        print(
            f"[expert:reject] manifest={manifest} classic_replay=False "
            f"lift={result.object_lift:.3f} fraction={result.success_fraction:.1%}",
            flush=True,
        )
    return bool(result.success)


def _discover_experts(
    *,
    benchmark: Path,
    rows: dict[str, dict],
    catalog: list[str],
    lattice_root: Path,
    ultra_roots: tuple[Path, ...],
    max_per_object: int,
    trust_rl: bool,
) -> tuple[list[Path], set[str]]:
    if max_per_object <= 0:
        raise ValueError("--bc-max-experts-per-object must be positive.")
    benchmark_root = benchmark.parent
    experts: list[Path] = []
    expert_objects: set[str] = set()

    for object_id in catalog:
        row = rows.get(object_id, {})
        if row.get("status") not in SUCCESS_SOURCE_STATUSES:
            continue
        candidates: list[Path] = []

        rl_best = benchmark_root / "rl" / _slug(object_id) / "best_trajectory" / "manifest.json"
        if rl_best.is_file():
            candidates.append(rl_best.resolve())

        for lattice in _lattice_rows(lattice_root, object_id):
            if not bool(lattice.get("success")) or not lattice.get("manifest"):
                continue
            manifest = _resolve_path(lattice["manifest"], base=Path.cwd())
            if manifest.is_file():
                candidates.append(manifest)

        try:
            for manifest, episode in discover_ultra_attempts(
                object_id,
                roots=ultra_roots,
                maximum=max(8, max_per_object),
            ):
                if episode.success:
                    candidates.append(Path(manifest))
        except (FileNotFoundError, RuntimeError, ValueError):
            pass

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            if _validate_expert(candidate, trust_rl=trust_rl):
                unique.append(candidate.resolve())
            if len(unique) >= max_per_object:
                break
        if unique:
            experts.extend(unique)
            expert_objects.add(object_id)
            print(f"[expert] object={object_id} trajectories={len(unique)}", flush=True)
        else:
            print(f"[expert:missing] object={object_id} status={row.get('status')}", flush=True)

    if not experts:
        raise RuntimeError("No validated expert trajectories were discovered from the benchmark.")
    return experts, expert_objects


def _bc_signature(experts: list[Path], args: argparse.Namespace) -> str:
    payload = {
        "version": 1,
        "experts": [str(path.resolve()) for path in experts],
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_learning_rate": args.bc_learning_rate,
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
        and stored.get("signature") == signature
        and dataset.is_file()
        and dataset.with_suffix(".json").is_file()
        and checkpoint.is_file()
    )
    if reusable:
        print(f"[bc:reuse] dataset={dataset} checkpoint={checkpoint}", flush=True)
        return checkpoint

    bc_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        bc_root / "experts.json",
        {
            "schema_version": 1,
            "count": len(experts),
            "manifests": [str(path) for path in experts],
        },
    )
    info = collect_bc_dataset(
        experts,
        output=dataset,
        device=args.bc_device,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    print(
        f"[bc:data-ready] frames={info.observations} experts={info.experts} "
        f"objects={info.objects} obs={info.obs_dim}",
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
    _atomic_json(signature_path, {"schema_version": 1, "signature": signature})
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
    primitive_root: Path,
    lattice_root: Path,
    ultra_roots: tuple[Path, ...],
) -> Path | None:
    direct = (
        primitive_root / _slug(object_id) / "best_attempt" / "manifest.json",
        benchmark_root / "rl" / _slug(object_id) / "best_attempt" / "manifest.json",
    )
    for path in direct:
        if path.is_file():
            return path.resolve()

    rows = _lattice_rows(lattice_root, object_id)
    rows.sort(
        key=lambda row: (
            bool(row.get("success", False)),
            float(row.get("source_lift", 0.0) or 0.0),
            -float(row.get("precheck_score", 1e9) or 1e9),
        ),
        reverse=True,
    )
    for row in rows:
        value = row.get("manifest")
        if value:
            path = _resolve_path(value, base=Path.cwd())
            if path.is_file():
                return path

    try:
        attempts = discover_ultra_attempts(object_id, roots=ultra_roots, maximum=8)
    except (FileNotFoundError, RuntimeError, ValueError):
        attempts = []
    return Path(attempts[0][0]).resolve() if attempts else None


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
    if existing and not args.force:
        status = existing.get("status")
        if status == "VERIFIED_SUCCESS" or not args.retry_failed:
            return dict(existing) | {"cached": True, "gpu": gpu, "log": str(log_path)}
    if args.force and output.exists():
        shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== object={object_id} gpu={gpu} started={_utc_now()} ===\n")
        reference = _best_reference(
            object_id,
            benchmark_root=args.benchmark.parent,
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
                primitive_root=args.primitive_root,
                lattice_root=args.lattice_root,
                ultra_roots=ultra_roots,
            )
        if reference is None:
            result = {
                "schema_version": 1,
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
            "schema_version": 1,
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
        "schema_version": 1,
        "updated_at": _utc_now(),
        "selected_objects": len(catalog),
        "completed_objects": len(ordered),
        "verified_total": counts.get("EXPERT_REUSED", 0) + counts.get("VERIFIED_SUCCESS", 0),
        "new_verified": counts.get("VERIFIED_SUCCESS", 0),
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
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id.")
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = _catalog(args)
    benchmark_rows = _benchmark_rows(args.benchmark)
    ultra_roots = (
        tuple(args.ultra_roots)
        if args.ultra_roots
        else (Path("outputs/ultradexgrasp"), Path("outputs/ultradexgrasp_catalog"))
    )

    experts, expert_objects = _discover_experts(
        benchmark=args.benchmark,
        rows=benchmark_rows,
        catalog=catalog,
        lattice_root=args.lattice_root,
        ultra_roots=ultra_roots,
        max_per_object=args.bc_max_experts_per_object,
        trust_rl=args.trust_rl_experts,
    )
    print(
        f"[prepare] catalogue={len(catalog)} expert_objects={len(expert_objects)} "
        f"expert_trajectories={len(experts)}",
        flush=True,
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
                "status": "EXPERT_REUSED",
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
                index, object_id = work.get_nowait()
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
            except Exception as exc:
                result = {
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
        f"  {'VERIFIED_TOTAL':<26} {int(summary.get('verified_total', 0)):3d}\n"
        f"[done] json={args.output / 'summary.json'} csv={args.output / 'summary.csv'}",
        flush=True,
    )
    return 0 if counts.get("ERROR", 0) == 0 else 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
