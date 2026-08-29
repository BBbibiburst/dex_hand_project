"""Run a resumable parallel preflight + adaptive-budget grasp-edit diagnostic.

This benchmark is intentionally a screening pass, not final policy training.
For each object it first ensures an Grasp Prior exists, builds a CPU DIRECT
wrist-lattice preflight, and conditionally starts MJWarp PPO. By default an
Grasp- or lattice-successful object exits early; explicit stress-test options
can still run hybrid grasp-edit PPO with an adaptive 5 -> 10 -> 15 budget.
Object workers are assigned to fixed GPU slots; same-GPU workers pipeline CPU
lattice work around a bounded Grasp/PPO phase. Progress reports include a
resource-derived plan and a wall-clock ETA after the worker pipeline warms up.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from source.grasping.budget import FORMAL_GENERATION_BUDGET

STATUSES = (
    "LATTICE_SUCCESS",
    "RL_SUCCESS",
    "RL_PROMISING",
    "DIRECT_FAILED",
    "NO_GRASP_GENERATED",
    "NO_REACHABLE_TEMPLATE",
    "PIPELINE_ERROR",
)

_NONCACHEABLE_INTERRUPTION_ERRORS = (
    BrokenPipeError,
    BrokenProcessPool,
    CancelledError,
    ConnectionResetError,
    EOFError,
)


def _is_noncacheable_interruption(exc: BaseException) -> bool:
    """Return whether *exc* indicates runner/process-pool interruption.

    These failures say nothing about grasp quality and must never become a
    resumable ``PIPELINE_ERROR`` row.  Walk chained exceptions because process
    pools and IPC helpers may wrap the original transport error.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, _NONCACHEABLE_INTERRUPTION_ERRORS):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False

CSV_FIELDS = (
    "object_id",
    "dataset",
    "gpu",
    "status",
    "needs_motion_primitive",
    "grasp_attempts",
    "grasp_success",
    "grasp_seed_index",
    "grasp_best_lift_mm",
    "grasp_best_final_lift_mm",
    "lattice_candidates",
    "lattice_reachable",
    "lattice_templates",
    "lattice_successful_templates",
    "lattice_failed_templates",
    "lattice_best_lift_mm",
    "rl_updates",
    "rl_completed_episodes",
    "rl_best_success_rate",
    "rl_final_success_rate",
    "rl_best_lift_mm",
    "rl_best_final_lift_mm",
    "rl_top_template",
    "rl_top_template_fraction",
    "failure_category",
    "runtime_sec",
    "log_path",
)

_LATTICE_RE = re.compile(
    r"\[lattice\]\s+candidates=(?P<candidates>\d+)\s+"
    r"reachable=(?P<reachable>\d+)\s+templates=(?P<templates>\d+)"
)
_LATTICE_PRECHECK_RE = re.compile(
    r"\[lattice:precheck\]\s+object=\S+\s+"
    r"candidates=(?P<candidates>\d+)\s+reachable=(?P<reachable>\d+)"
)
_UPDATE_RE = re.compile(
    r"u\s+(?P<update>\d+)/(?P<total>\d+)\s+\|\s+"
    r"succ\s+(?P<success>[0-9.]+)%\s+\|\s+"
    r"lift\s+(?P<mean_lift>-?[0-9.]+)/\s*(?P<best_lift>-?[0-9.]+)mm\s+\|\s+"
    r"final\s+(?P<mean_final>-?[0-9.]+)/\s*(?P<best_final>-?[0-9.]+)mm\s+\|\s+"
    r"top\s+t(?P<template>\d+):(?P<rate>[0-9.]+)%"
)


@dataclass
class ChildResult:
    returncode: int
    duration_sec: float
    text: str


@dataclass
class LatticePreflight:
    duration_sec: float
    text: str
    candidates: int
    reachable: int
    templates: int
    successful_templates: int
    failed_templates: int
    best_lift_mm: float
    error: str = ""


@dataclass(frozen=True)
class ObjectWorkItem:
    object_id: str
    args: argparse.Namespace
    root: Path
    grasp_roots: tuple[Path, ...]
    object_dir: Path
    log_dir: Path
    rl_root: Path
    signature: str


_WORKER_GPU = ""
_GPU_PHASE_SEMAPHORE: Any | None = None
_PPO_PHASE_SEMAPHORE: Any | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _gpu_ids(spec: str, *, device: str) -> tuple[str, ...]:
    """Resolve physical GPU slots while preserving the old single-device default."""

    if not str(device).startswith("cuda"):
        return ("cpu",)
    if spec.strip().lower() != "auto":
        values = tuple(dict.fromkeys(item.strip() for item in spec.split(",") if item.strip()))
        if not values:
            raise ValueError("--gpus must contain at least one GPU id or 'auto'.")
        return values

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_env is not None:
        visible = visible_env.strip()
        if not visible or visible == "-1":
            raise ValueError(
                "--device requests CUDA, but CUDA_VISIBLE_DEVICES hides every GPU."
            )
        values = tuple(
            dict.fromkeys(item.strip() for item in visible.split(",") if item.strip())
        )
        if values:
            return values

    match = re.fullmatch(r"cuda(?::(?P<index>\d+))?", str(device))
    if match:
        return (match.group("index") or "0",)
    return ("0",)


def _gpu_resource_rows() -> dict[str, dict[str, float]]:
    try:
        child = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return {}
    if child.returncode:
        return {}

    result: dict[str, dict[str, float]] = {}
    for line in child.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            result[fields[0]] = {
                "total_memory_mb": float(fields[1]),
                "free_memory_mb": float(fields[2]),
                "utilization_percent": float(fields[3]),
            }
        except ValueError:
            continue
    return result


def _estimated_worker_memory_mb(num_envs: int) -> float:
    """Conservative MJWarp process estimate used only for scheduling."""

    return 1280.0 + 12.0 * float(num_envs)


def _resource_plan(
    gpus: tuple[str, ...],
    *,
    workers_per_gpu: str,
    gpu_jobs_per_gpu: str,
    ppo_jobs_per_gpu: str,
    num_envs: int,
    resource_rows: dict[str, dict[str, float]] | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    resources = _gpu_resource_rows() if resource_rows is None else resource_rows

    def positive_or_auto(value: str, option: str) -> int | None:
        spec = str(value).strip().lower()
        if spec == "auto":
            return None
        try:
            parsed = int(spec)
        except ValueError as exc:
            raise ValueError(f"{option} must be 'auto' or a positive integer.") from exc
        if parsed <= 0:
            raise ValueError(f"{option} must be positive.")
        return parsed

    explicit_workers = positive_or_auto(workers_per_gpu, "--workers-per-gpu")
    explicit_gpu_jobs = positive_or_auto(gpu_jobs_per_gpu, "--gpu-jobs-per-gpu")
    explicit_ppo_jobs = positive_or_auto(ppo_jobs_per_gpu, "--ppo-jobs-per-gpu")

    estimated_memory = _estimated_worker_memory_mb(num_envs)
    cpu_count = os.cpu_count() or 1
    cpu_cap_per_gpu = max(1, cpu_count // max(2 * len(gpus), 1))
    counts: dict[str, int] = {}
    gpu_job_limits: dict[str, int] = {}
    ppo_job_limits: dict[str, int] = {}
    details: dict[str, Any] = {}
    for gpu in gpus:
        resource = resources.get(gpu, {})
        total = float(resource.get("total_memory_mb", 0.0))
        free = float(resource.get("free_memory_mb", 0.0))
        utilization = float(resource.get("utilization_percent", 0.0))
        if total > 0.0 and free > 0.0:
            reserve = max(2048.0, 0.10 * total)
            memory_slots = max(1, int(max(0.0, free - reserve) // estimated_memory))
        else:
            reserve = 0.0
            # Missing telemetry must remain usable, but should not guess that
            # concurrent CUDA workloads are safe.
            memory_slots = 1

        if explicit_workers is not None:
            workers = explicit_workers
        elif gpu == "cpu":
            workers = 1
        else:
            # Two pipeline workers are enough to overlap CPU lattice work with
            # CUDA phases. More workers mainly increase contention and startup.
            workers = max(1, min(2, memory_slots, cpu_cap_per_gpu))

        if gpu == "cpu":
            gpu_jobs = workers
        elif explicit_gpu_jobs is not None:
            gpu_jobs = explicit_gpu_jobs
        else:
            launch_slots = 2 if utilization < 70.0 else 1
            gpu_jobs = max(1, min(2, workers, memory_slots, launch_slots))
        gpu_jobs = max(1, min(gpu_jobs, workers))

        if gpu == "cpu":
            ppo_jobs = workers
        elif explicit_ppo_jobs is not None:
            ppo_jobs = explicit_ppo_jobs
        else:
            # The 64-env MJWarp PPO phase saturates the observed 3090 compute
            # while using little memory. Keep PPO exclusive by default, but let
            # lightweight Grasp work occupy the second general GPU slot.
            ppo_jobs = 1
        ppo_jobs = max(1, min(ppo_jobs, gpu_jobs, workers))

        counts[gpu] = workers
        gpu_job_limits[gpu] = gpu_jobs
        ppo_job_limits[gpu] = ppo_jobs
        details[gpu] = {
            **resource,
            "workers": workers,
            "gpu_jobs": gpu_jobs,
            "ppo_jobs": ppo_jobs,
            "estimated_worker_memory_mb": estimated_memory,
            "reserved_memory_mb": reserve,
        }

    maximum_workers = max(counts.values(), default=0)
    slots = tuple(
        gpu
        for worker_index in range(maximum_workers)
        for gpu in gpus
        if worker_index < counts[gpu]
    )
    return slots, {
        "worker_mode": "auto" if explicit_workers is None else "explicit",
        "gpu_job_mode": "auto" if explicit_gpu_jobs is None else "explicit",
        "ppo_job_mode": "auto" if explicit_ppo_jobs is None else "explicit",
        "cpu_count": cpu_count,
        "gpu_details": details,
        "gpu_job_limits": gpu_job_limits,
        "ppo_job_limits": ppo_job_limits,
        "worker_slots": list(slots),
    }


def _runtime_estimate(
    rows: Iterable[dict[str, Any]],
    *,
    remaining: int,
    worker_count: int,
) -> tuple[float | None, float | None, int]:
    durations = [
        float(row.get("runtime_sec") or 0.0)
        for row in rows
        if float(row.get("runtime_sec") or 0.0) > 0.0
    ]
    if not durations:
        return None, None, 0
    average = sum(durations) / len(durations)
    eta = average * max(0, remaining) / max(1, worker_count)
    return average, eta, len(durations)


def _wall_clock_estimate(
    *,
    elapsed: float,
    completed: int,
    remaining: int,
    warmup_completions: int,
) -> tuple[float | None, float | None, int]:
    """Measure real object throughput after every active worker has returned once."""

    if completed < max(1, warmup_completions):
        return None, None, completed
    average = max(0.0, float(elapsed)) / max(1, completed)
    return average, average * max(0, remaining), completed


def _init_object_worker(
    slot_queue: Any,
    gpu_semaphores: dict[str, Any],
    ppo_semaphores: dict[str, Any],
) -> None:
    """Assign one immutable CUDA visibility slot to each spawned worker."""

    global _GPU_PHASE_SEMAPHORE, _PPO_PHASE_SEMAPHORE, _WORKER_GPU
    _WORKER_GPU = str(slot_queue.get())
    _GPU_PHASE_SEMAPHORE = gpu_semaphores.get(_WORKER_GPU)
    _PPO_PHASE_SEMAPHORE = ppo_semaphores.get(_WORKER_GPU)
    if _WORKER_GPU != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = _WORKER_GPU
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _worker_gpu_identity() -> tuple[str, str]:
    """Return the immutable worker assignment for scheduler diagnostics."""

    return _WORKER_GPU, os.environ.get("CUDA_VISIBLE_DEVICES", "")


@contextlib.contextmanager
def _gpu_phase(*, ppo: bool = False):
    general_semaphore = _GPU_PHASE_SEMAPHORE
    ppo_semaphore = _PPO_PHASE_SEMAPHORE if ppo else None
    if general_semaphore is None:
        yield
        return
    if ppo_semaphore is not None:
        ppo_semaphore.acquire()
    general_semaphore.acquire()
    try:
        yield
    finally:
        general_semaphore.release()
        if ppo_semaphore is not None:
            ppo_semaphore.release()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})
    temporary.replace(path)


def _repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("Run this benchmark inside dex_hand_project.")
    return Path(proc.stdout.strip())


def _prepare_shared_surrogate() -> None:
    """Populate the shared hand-surrogate cache before parallel Grasp workers."""

    from source.grasping.config import DEFAULT_CONFIG_PATH, load_pipeline_config
    from source.grasping.hand_surrogate import load_or_calibrate_surrogate

    pipeline = load_pipeline_config(DEFAULT_CONFIG_PATH)
    load_or_calibrate_surrogate(
        pipeline.surrogate_cache,
        **pipeline.surrogate_options,
    )


def _validate_runtime_dependencies(catalog: Iterable[str]) -> None:
    """Fail before spawning workers when on-demand collision tooling is absent."""

    if not any(not object_id.startswith("ycb:") for object_id in catalog):
        return
    try:
        importlib.import_module("coacd")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GSO/EGAD collision meshes require the 'coacd' package when their "
            "decomposition cache is missing. Install current project dependencies "
            "with `python -m pip install -e '.[grasping,mjwarp]'` or run "
            "`python -m pip install coacd`."
        ) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    files = (
        "assets/grippers/dex_hand/dex_hand.xml",
        "configs/grasping/default.json",
        "apps/train_grasp_edit_rl.py",
        "source/rl/common/ppo.py",
        "source/rl/grasp_edit/env.py",
        "source/rl/grasp_edit/ppo.py",
        "source/rl/grasp_edit/templates.py",
        "source/grasping/executor.py",
        "source/grasping/catalog.py",
        "source/grasping/collision_decomposition.py",
        "source/grasping/dexevolve_contacts.py",
        "source/grasping/hand_surrogate.py",
        "source/grasping/graspqp_adapter.py",
        "source/grasping/seeds.py",
        "source/grasping/dexevolve.py",
        "source/envs/manipulation/objects.py",
        "tools/grasp_generation/graspqp_evolve.py",
    )
    return {name: _sha256(root / name) for name in files}


def _dataset_ids(dataset: str) -> tuple[str, ...]:
    from source.envs.manipulation.object_catalog import object_ids

    if dataset == "all":
        return object_ids()
    if dataset == "original127":
        return tuple(
            object_id
            for object_id in object_ids()
            if object_id.startswith(("ycb:", "egad:"))
        )
    return object_ids(dataset)


def _selection_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise ValueError(f"Selection must contain an objects list: {path}")
    selected = tuple(str(row["object_id"]) for row in rows)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selection must contain unique object_id values.")
    known = set(_dataset_ids("all"))
    unknown = sorted(set(selected).difference(known))
    if unknown:
        raise ValueError(f"Selection contains unknown object id(s): {unknown}")
    return selected


def _episode_lifts(episode: Any) -> tuple[float, float]:
    try:
        import numpy as np

        position = np.asarray(episode.arrays["object_position"], dtype=float)
        if position.ndim == 2 and position.shape[0] and position.shape[1] >= 3:
            relative = position[:, 2] - position[0, 2]
            return float(np.max(relative)), float(relative[-1])
    except (KeyError, TypeError, ValueError):
        pass
    lift = float(episode.metadata.get("object_lift", 0.0))
    return lift, lift


def _discover_grasp(object_id: str, roots: tuple[Path, ...]):
    from source.rl.grasp_edit.templates import discover_grasp_attempts

    return discover_grasp_attempts(object_id, roots=roots, maximum=256)


def _grasp_summary(object_id: str, roots: tuple[Path, ...]) -> dict[str, Any]:
    attempts = _discover_grasp(object_id, roots)
    best = None
    best_key = (-1, float("-inf"), float("-inf"))
    for manifest, episode in attempts:
        max_lift, final_lift = _episode_lifts(episode)
        key = (1 if bool(episode.success) else 0, max_lift, final_lift)
        if key > best_key:
            best_key = key
            best = (manifest, episode, max_lift, final_lift)
    if best is None:
        return {
            "attempts": 0,
            "success": False,
            "seed_index": "",
            "best_lift_mm": 0.0,
            "best_final_lift_mm": 0.0,
        }
    _, episode, max_lift, final_lift = best
    return {
        "attempts": len(attempts),
        "success": any(bool(ep.success) for _, ep in attempts),
        "seed_index": int(episode.candidate.seed_index),
        "best_lift_mm": 1000.0 * max_lift,
        "best_final_lift_mm": 1000.0 * final_lift,
    }


def _run_child(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    verbose: bool,
) -> ChildResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lines: list[str] = []
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {_utc_now()} =====\n$ {' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            log.write(line)
            if verbose:
                print(line, end="", flush=True)
        returncode = proc.wait()
        log.write(f"[child-exit] returncode={returncode}\n")
    return ChildResult(returncode, time.perf_counter() - started, "".join(lines))


def _ensure_grasp(
    args: argparse.Namespace,
    object_id: str,
    roots: tuple[Path, ...],
    *,
    root: Path,
    log_path: Path,
) -> tuple[dict[str, Any], float, str]:
    summary = _grasp_summary(object_id, roots)
    if summary["attempts"]:
        return summary, 0.0, ""

    elapsed = 0.0
    combined = ""
    primary = roots[0]
    for offset in range(args.generation_attempts):
        rng_seed = args.seed + offset
        output = primary / _slug(object_id) / f"seed_{rng_seed:04d}"
        command = [
            sys.executable,
            "-m",
            "tools.grasp_generation.graspqp_evolve",
            "--object-id",
            object_id,
            "--seed",
            str(rng_seed),
            "--graspqp-seeds",
            str(args.graspqp_seeds),
            "--device",
            args.device,
            "--graspqp-executions",
            str(args.graspqp_executions),
            "--output",
            str(output),
        ]
        with _gpu_phase():
            child = _run_child(command, cwd=root, log_path=log_path, verbose=args.verbose_child)
        elapsed += child.duration_sec
        combined += child.text
        summary = _grasp_summary(object_id, roots)
        if summary["attempts"]:
            return summary, elapsed, combined
        if child.returncode not in (0, 2):
            expected_failures = (
                "GraspQP produced no RM75B-reachable candidates.",
                "DexEvolve archive has no strictly RM75B-reachable survivors.",
                "DexEvolve archive has no strict execution with sustained thumb-opposed contact.",
            )
            if not any(message in child.text for message in expected_failures):
                tail = "\n".join(child.text.rstrip().splitlines()[-12:])
                raise RuntimeError(
                    f"Grasp generator exited with code {child.returncode} for {object_id}.\n"
                    f"{tail}"
                )
    return summary, elapsed, combined


def _parse_lattice(text: str) -> dict[str, int]:
    matches = list(_LATTICE_RE.finditer(text))
    if not matches:
        return {"candidates": 0, "reachable": 0, "templates": 0}
    match = matches[-1]
    return {key: int(match.group(key)) for key in ("candidates", "reachable", "templates")}


def _parse_lattice_precheck(text: str) -> dict[str, int]:
    summary = _parse_lattice(text)
    if summary["candidates"] or summary["reachable"]:
        return summary
    matches = list(_LATTICE_PRECHECK_RE.finditer(text))
    if not matches:
        return summary
    match = matches[-1]
    return {
        "candidates": int(match.group("candidates")),
        "reachable": int(match.group("reachable")),
        "templates": 0,
    }


def _preflight_lattice(
    *,
    args: argparse.Namespace,
    object_id: str,
    grasp_roots: tuple[Path, ...],
    log_path: Path,
) -> LatticePreflight:
    """Compile/reuse DIRECT lattice on CPU before starting any MJWarp PPO."""
    from source.rl.grasp_edit.templates import build_grasp_edit_templates

    started = time.perf_counter()
    buffer = io.StringIO()
    templates = ()
    error = ""
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            templates = build_grasp_edit_templates(
                object_id,
                output_root=args.lattice_root,
                grasp_roots=grasp_roots,
                base_candidates=args.base_candidates,
                maximum_templates=args.lattice_max_templates,
                maximum_executions=args.lattice_max_executions,
                execution_lift_height=args.execution_lift_height,
                seed=args.seed,
                overwrite=False,
                failed_only=False,
                # Detailed compile diagnostics go to the per-object logfile,
                # not the benchmark terminal.  They also expose precheck counts
                # when no usable template survives execution.
                verbose=True,
            )
    except RuntimeError as exc:
        lowered = str(exc).lower()
        if "no usable reachable wrist-lattice templates" in lowered:
            error = str(exc)
        else:
            raise
    finally:
        text = buffer.getvalue()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {_utc_now()} lattice-preflight =====\n")
            log.write(text)
            if error:
                log.write(f"[lattice-preflight-error] {error}\n")
        if args.verbose_child and text:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)

    text = buffer.getvalue()
    lattice = _parse_lattice_precheck(text)
    successful = sum(bool(template.success) for template in templates)
    total = len(templates)
    best_lift_mm = max(
        (1000.0 * float(template.source_lift) for template in templates), default=0.0
    )
    return LatticePreflight(
        duration_sec=time.perf_counter() - started,
        text=text,
        candidates=lattice["candidates"],
        reachable=lattice["reachable"],
        templates=lattice["templates"] or total,
        successful_templates=successful,
        failed_templates=total - successful,
        best_lift_mm=best_lift_mm,
        error=error,
    )


def _parse_update_history(text: str) -> dict[str, Any]:
    matches = list(_UPDATE_RE.finditer(text))
    if not matches:
        return {
            "updates": 0,
            "best_success_rate": 0.0,
            "final_success_rate": 0.0,
            "mean_lift_mm": 0.0,
            "best_lift_mm": 0.0,
            "mean_final_lift_mm": 0.0,
            "best_final_lift_mm": 0.0,
            "recent_mean_lift_gain_mm": 0.0,
            "top_template": "",
            "top_template_fraction": 0.0,
        }

    best_success = max(float(match.group("success")) / 100.0 for match in matches)
    last = matches[-1]
    latest_update = int(last.group("update"))
    recent = [match for match in matches if int(match.group("update")) >= max(1, latest_update - 4)]
    first_recent = recent[0] if recent else last
    recent_gain = float(last.group("mean_lift")) - float(first_recent.group("mean_lift"))

    return {
        "updates": latest_update,
        "best_success_rate": best_success,
        "final_success_rate": float(last.group("success")) / 100.0,
        "mean_lift_mm": float(last.group("mean_lift")),
        "best_lift_mm": max(float(match.group("best_lift")) for match in matches),
        "mean_final_lift_mm": float(last.group("mean_final")),
        "best_final_lift_mm": max(float(match.group("best_final")) for match in matches),
        "recent_mean_lift_gain_mm": recent_gain,
        "top_template": f"t{int(last.group('template'))}",
        "top_template_fraction": float(last.group("rate")) / 100.0,
    }


def _adaptive_decision(
    *,
    history: dict[str, Any],
    train_output: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    target_update: int,
) -> str:
    """Return stop_success / stop_lattice / stop_hopeless / continue / stop_budget."""
    templates = config.get("templates", []) if isinstance(config.get("templates", []), list) else []
    if any(bool(row.get("success_before_edit")) for row in templates):
        return "stop_lattice"
    if (train_output / "best_trajectory" / "manifest.json").is_file():
        return "stop_success"
    if target_update >= args.max_updates:
        return "stop_budget"

    best_lift = float(history.get("best_lift_mm", 0.0))
    recent_gain = float(history.get("recent_mean_lift_gain_mm", 0.0))
    best_success = float(history.get("best_success_rate", 0.0))

    # Only call something hopeless early when *all* evidence is weak.  The
    # ambiguous middle band is deliberately allowed to continue so a slow
    # learner is not mislabeled DIRECT_FAILED after only five updates.
    if (
        target_update >= args.initial_updates
        and best_success <= 0.0
        and best_lift < args.early_fail_lift_mm
        and recent_gain < args.progress_gain_mm
    ):
        return "stop_hopeless"

    if (
        best_success > 0.0
        or best_lift >= args.continue_lift_mm
        or recent_gain >= args.progress_gain_mm
    ):
        return "continue"

    # Between the initial and middle budget, prefer one more stage for
    # ambiguous cases.  At the middle budget, only clearly promising cases
    # receive the final extension.
    if target_update < args.mid_updates:
        return "continue"
    return "stop_budget"


def _adaptive_train(
    *,
    args: argparse.Namespace,
    object_id: str,
    grasp_roots: tuple[Path, ...],
    root: Path,
    log_path: Path,
    rl_root: Path,
) -> ChildResult:
    """Train 5 -> 10 -> 15 updates using grasp-edit checkpoints rather than restarting."""
    train_output = rl_root / _slug(object_id)
    if (train_output / "best_trajectory" / "manifest.json").is_file():
        # A source-signature change may require rebuilding the summary and
        # lattice, but must never destroy an already successful RL artifact.
        return ChildResult(0, 0.0, "")
    checkpoint = train_output / "checkpoint_final.pt"
    completed_target = 0
    changed_physics = (
        abs(float(args.execution_lift_height) - 0.065) > 1e-12
        or abs(float(args.hand_edit_fraction) - 0.35) > 1e-12
    )
    if args.resume_existing_rl and checkpoint.is_file() and not changed_physics:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        completed_target = int(payload.get("update", 0))
        if completed_target < 0:
            raise ValueError(f"Invalid update index in {checkpoint}: {completed_target}")
    elif train_output.exists():
        shutil.rmtree(train_output)

    stage_targets = []
    for value in (args.initial_updates, args.mid_updates, args.max_updates):
        if not stage_targets or value != stage_targets[-1]:
            stage_targets.append(value)

    combined_text = ""
    total_duration = 0.0
    last_returncode = 2

    for target_update in stage_targets:
        additional_updates = target_update - completed_target
        if additional_updates <= 0:
            continue

        command = [
            sys.executable,
            "-m",
            "apps.train_grasp_edit_rl",
            "--object-id",
            object_id,
            "--output-root",
            str(rl_root),
            "--template-root",
            str(args.lattice_root),
            "--no-auto-grasp",
            "--num-envs",
            str(args.num_envs),
            "--updates",
            str(additional_updates),
            "--save-every",
            "0",
            "--log-every",
            "1",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--base-candidates",
            str(args.base_candidates),
            "--lattice-max-templates",
            str(args.lattice_max_templates),
            "--lattice-max-executions",
            str(args.lattice_max_executions),
            "--execution-lift-height",
            str(args.execution_lift_height),
            "--hand-edit-fraction",
            str(args.hand_edit_fraction),
        ]
        if completed_target and checkpoint.is_file():
            command.extend(["--resume", str(checkpoint)])
        for grasp_root in grasp_roots:
            command.extend(["--grasp-root", str(grasp_root)])

        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[adaptive-stage] target={target_update} additional={additional_updates}\n"
            )

        with _gpu_phase(ppo=True):
            child = _run_child(
                command,
                cwd=root,
                log_path=log_path,
                verbose=args.verbose_child,
            )
        total_duration += child.duration_sec
        combined_text += child.text
        last_returncode = child.returncode

        # A real exception should not be retried as a longer RL budget.
        if child.returncode not in (0, 2):
            break

        completed_target = target_update
        history = _parse_update_history(combined_text)
        config = _read_json(train_output / "config.json")
        decision = _adaptive_decision(
            history=history,
            train_output=train_output,
            config=config,
            args=args,
            target_update=target_update,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"[adaptive-decision] target={target_update} decision={decision} "
                f"best_success={history['best_success_rate']:.3f} "
                f"best_lift={history['best_lift_mm']:.1f}mm "
                f"gain={history['recent_mean_lift_gain_mm']:.1f}mm\n"
            )
        if decision != "continue":
            break

    return ChildResult(last_returncode, total_duration, combined_text)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _classify_training(
    *,
    preflight: LatticePreflight,
    child: ChildResult,
    train_output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = _read_json(train_output / "config.json")
    metrics = _read_json(train_output / "metrics.json")
    templates = config.get("templates", []) if isinstance(config.get("templates", []), list) else []
    successful_templates = max(
        preflight.successful_templates,
        sum(bool(row.get("success_before_edit")) for row in templates),
    )
    failed_templates = max(preflight.failed_templates, len(templates) - successful_templates)
    lattice = _parse_lattice(child.text)
    history = _parse_update_history(child.text)

    final_success_rate = float(metrics.get("episode_success_rate", 0.0))
    best_lift = max(
        float(metrics.get("best_attempt_lift", 0.0)),
        float(history.get("best_lift_mm", 0.0)) / 1000.0,
    )
    best_final_lift = max(
        float(metrics.get("best_attempt_final_lift", 0.0)),
        float(history.get("best_final_lift_mm", 0.0)) / 1000.0,
    )
    completed = int(args.num_envs * history["updates"])
    best_exists = (train_output / "best_trajectory" / "manifest.json").is_file()

    if successful_templates:
        status = "LATTICE_SUCCESS"
        failure = ""
    elif best_exists:
        status = "RL_SUCCESS"
        failure = ""
    elif child.returncode not in (0, 2):
        lowered = child.text.lower()
        if "no usable reachable wrist-lattice templates" in lowered or (
            "no usable" in lowered and "wrist-lattice templates" in lowered
        ):
            status = "NO_REACHABLE_TEMPLATE"
            failure = "no_reachable_template"
        else:
            status = "PIPELINE_ERROR"
            failure = "training_exception"
    elif (
        history["best_success_rate"] >= args.promising_success_rate
        or best_lift * 1000.0 >= args.promising_lift_mm
    ):
        status = "RL_PROMISING"
        failure = "adaptive_rl_not_yet_successful"
    else:
        status = "DIRECT_FAILED"
        failure = "adaptive_rl_no_progress"

    return {
        "status": status,
        "needs_motion_primitive": status
        in {"DIRECT_FAILED", "NO_GRASP_GENERATED", "NO_REACHABLE_TEMPLATE"},
        "lattice_candidates": lattice["candidates"] or preflight.candidates,
        "lattice_reachable": lattice["reachable"] or preflight.reachable,
        "lattice_templates": lattice["templates"] or preflight.templates or len(templates),
        "lattice_successful_templates": successful_templates,
        "lattice_failed_templates": failed_templates,
        "lattice_best_lift_mm": preflight.best_lift_mm,
        "rl_updates": history["updates"],
        "rl_completed_episodes": completed,
        "rl_best_success_rate": history["best_success_rate"],
        "rl_final_success_rate": final_success_rate,
        "rl_best_lift_mm": 1000.0 * best_lift,
        "rl_best_final_lift_mm": 1000.0 * best_final_lift,
        "rl_top_template": history["top_template"],
        "rl_top_template_fraction": history["top_template_fraction"],
        "failure_category": failure,
    }


def _empty_row(object_id: str) -> dict[str, Any]:
    return {field: "" for field in CSV_FIELDS} | {
        "object_id": object_id,
        "dataset": object_id.split(":", 1)[0],
        "gpu": "",
        "status": "PIPELINE_ERROR",
        "needs_motion_primitive": False,
        "grasp_attempts": 0,
        "grasp_success": False,
        "grasp_seed_index": "",
        "grasp_best_lift_mm": 0.0,
        "grasp_best_final_lift_mm": 0.0,
        "lattice_candidates": 0,
        "lattice_reachable": 0,
        "lattice_templates": 0,
        "lattice_successful_templates": 0,
        "lattice_failed_templates": 0,
        "lattice_best_lift_mm": 0.0,
        "rl_updates": 0,
        "rl_completed_episodes": 0,
        "rl_best_success_rate": 0.0,
        "rl_final_success_rate": 0.0,
        "rl_best_lift_mm": 0.0,
        "rl_best_final_lift_mm": 0.0,
        "rl_top_template": "",
        "rl_top_template_fraction": 0.0,
        "failure_category": "",
        "runtime_sec": 0.0,
        "log_path": "",
    }


def _save_object_result(
    path: Path,
    *,
    row: dict[str, Any],
    signature: str,
) -> None:
    _atomic_json(path, {"schema_version": 1, "signature": signature, "result": row})


def _load_object_result(path: Path, signature: str) -> dict[str, Any] | None:
    payload = _read_json(path)
    if payload.get("signature") != signature:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def _write_summary(
    output: Path,
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    signature: str,
    source_hashes: dict[str, str],
    total_count: int | None = None,
    worker_count: int = 1,
    eta_parallelism: int | None = None,
    gpus: tuple[str, ...] = (),
    resource_plan: dict[str, Any] | None = None,
    timing_override: tuple[float | None, float | None, int] | None = None,
) -> None:
    counts = Counter(str(row.get("status", "")) for row in rows)
    selected = len(rows) if total_count is None else int(total_count)
    remaining = max(0, selected - len(rows))
    parallelism = max(1, int(eta_parallelism or worker_count))
    if timing_override is None:
        average_runtime, eta, timing_samples = _runtime_estimate(
            rows,
            remaining=remaining,
            worker_count=parallelism,
        )
        timing_source = "historical_runtime"
    else:
        average_runtime, eta, timing_samples = timing_override
        timing_source = "current_wall_clock"
    _atomic_csv(output / "summary.csv", rows)
    _atomic_json(
        output / "summary.json",
        {
            "schema_version": 2,
            "updated_at": _utc_now(),
            "signature": signature,
            "dataset": args.dataset,
            "count": len(rows),
            "selected_objects": selected,
            "completed_objects": len(rows),
            "status_counts": {status: counts.get(status, 0) for status in STATUSES},
            "progress": {
                "remaining_objects": remaining,
                "worker_slots": worker_count,
                "eta_parallelism": parallelism,
                "gpus": list(gpus),
                "timing_samples": timing_samples,
                "timing_source": timing_source,
                "average_runtime_sec": average_runtime,
                "estimated_remaining_sec": eta,
            },
            "resource_plan": resource_plan or {},
            "source_hashes": source_hashes,
            "settings": {
                "gpus": args.gpus,
                "workers_per_gpu": args.workers_per_gpu,
                "gpu_jobs_per_gpu": args.gpu_jobs_per_gpu,
                "ppo_jobs_per_gpu": args.ppo_jobs_per_gpu,
                "num_envs": args.num_envs,
                "initial_updates": args.initial_updates,
                "mid_updates": args.mid_updates,
                "max_updates": args.max_updates,
                "graspqp_seeds": args.graspqp_seeds,
                "generation_attempts": args.generation_attempts,
                "base_candidates": args.base_candidates,
                "lattice_max_templates": args.lattice_max_templates,
                "lattice_max_executions": args.lattice_max_executions,
                "execution_lift_height": args.execution_lift_height,
                "hand_edit_fraction": args.hand_edit_fraction,
                "lattice_root": str(args.lattice_root),
                "promising_lift_mm": args.promising_lift_mm,
                "promising_success_rate": args.promising_success_rate,
                "early_fail_lift_mm": args.early_fail_lift_mm,
                "continue_lift_mm": args.continue_lift_mm,
                "progress_gain_mm": args.progress_gain_mm,
                "train_lattice_success": args.train_lattice_success,
                "resume_existing_rl": args.resume_existing_rl,
            },
            "results": rows,
        },
    )


def _format_progress(
    index: int,
    total: int,
    row: dict[str, Any],
    *,
    average_runtime: float | None = None,
    eta: float | None = None,
    cached: bool = False,
) -> str:
    status = str(row["status"])
    grasp = "Y" if row.get("grasp_success") else "N"
    templates = int(row.get("lattice_templates") or 0)
    success = 100.0 * float(row.get("rl_best_success_rate") or 0.0)
    updates = int(row.get("rl_updates") or 0)
    lift = float(
        row.get("rl_best_lift_mm")
        or row.get("lattice_best_lift_mm")
        or row.get("grasp_best_lift_mm")
        or 0.0
    )
    runtime = float(row.get("runtime_sec") or 0.0)
    gpu = str(row.get("gpu") or "-")
    timing = " eta=warming_up"
    if average_runtime is not None and eta is not None:
        timing = f" avg={_format_duration(average_runtime)}/obj eta={_format_duration(eta)}"
    cache_label = " [cached]" if cached else ""
    return (
        f"[{index:03d}/{total:03d}] {row['object_id']:<34} "
        f"{status:<21} grasp={grasp} tpl={templates:02d} "
        f"u={updates:02d} rl={success:5.1f}% lift={lift:5.1f}mm "
        f"gpu={gpu} object={_format_duration(runtime)}{timing}{cache_label}"
    )


def _run_object_item(item: ObjectWorkItem) -> dict[str, Any]:
    """Execute one isolated object pipeline inside a fixed resource slot."""

    os.chdir(item.root)
    args = argparse.Namespace(**vars(item.args))
    if _WORKER_GPU != "cpu" and str(args.device).startswith("cuda"):
        # CUDA_VISIBLE_DEVICES exposes exactly the physical GPU assigned by the
        # parent, so every child process addresses it as logical cuda:0.
        args.device = "cuda:0"

    object_id = item.object_id
    started = time.perf_counter()
    row = _empty_row(object_id)
    row["gpu"] = _WORKER_GPU or ("cpu" if args.device == "cpu" else str(args.device))
    log_path = item.log_dir / f"{_slug(object_id)}.log"
    row["log_path"] = str(log_path)
    deferred_error: Exception | None = None
    try:
        grasp, _, _ = _ensure_grasp(
            args,
            object_id,
            item.grasp_roots,
            root=item.root,
            log_path=log_path,
        )
        row.update(
            {
                "grasp_attempts": grasp["attempts"],
                "grasp_success": grasp["success"],
                "grasp_seed_index": grasp["seed_index"],
                "grasp_best_lift_mm": round(float(grasp["best_lift_mm"]), 3),
                "grasp_best_final_lift_mm": round(float(grasp["best_final_lift_mm"]), 3),
            }
        )
        if not grasp["attempts"]:
            row["status"] = "NO_GRASP_GENERATED"
            row["needs_motion_primitive"] = True
            row["failure_category"] = "grasp_no_full_attempt"
        else:
            preflight = _preflight_lattice(
                args=args,
                object_id=object_id,
                grasp_roots=item.grasp_roots,
                log_path=log_path,
            )
            row.update(
                {
                    "lattice_candidates": preflight.candidates,
                    "lattice_reachable": preflight.reachable,
                    "lattice_templates": preflight.templates,
                    "lattice_successful_templates": preflight.successful_templates,
                    "lattice_failed_templates": preflight.failed_templates,
                    "lattice_best_lift_mm": preflight.best_lift_mm,
                }
            )
            if preflight.error:
                row["status"] = "NO_REACHABLE_TEMPLATE"
                row["needs_motion_primitive"] = True
                row["failure_category"] = "no_reachable_template"
            elif preflight.successful_templates and not args.train_lattice_success:
                row["status"] = "LATTICE_SUCCESS"
                row["needs_motion_primitive"] = False
                row["failure_category"] = ""
            else:
                train_output = item.rl_root / _slug(object_id)
                child = _adaptive_train(
                    args=args,
                    object_id=object_id,
                    grasp_roots=item.grasp_roots,
                    root=item.root,
                    log_path=log_path,
                    rl_root=item.rl_root,
                )
                row.update(
                    _classify_training(
                        preflight=preflight,
                        child=child,
                        train_output=train_output,
                        args=args,
                    )
                )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve the rest of the catalogue sweep
        if _is_noncacheable_interruption(exc):
            raise
        row["status"] = "PIPELINE_ERROR"
        row["failure_category"] = f"{type(exc).__name__}: {exc}"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[benchmark-exception] {type(exc).__name__}: {exc}\n")
        if args.fail_fast:
            deferred_error = exc

    row["runtime_sec"] = round(time.perf_counter() - started, 3)
    _save_object_result(
        item.object_dir / f"{_slug(object_id)}.json",
        row=row,
        signature=item.signature,
    )
    if deferred_error is not None:
        raise deferred_error
    return row


def build_parser() -> argparse.ArgumentParser:
    budget = FORMAL_GENERATION_BUDGET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("original127", "all", "ycb", "egad", "gso"),
        default="original127",
        help=(
            "Object catalogue to run. original127 is the historical 78 YCB + "
            "49 EGAD benchmark; all also includes the later GSO catalogue."
        ),
    )
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument(
        "--selection",
        type=Path,
        help="JSON object selection in render_object_catalog/ranking format.",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=0,
        help="Optional selected-object count assertion; disabled by default.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_edit_benchmark"))
    parser.add_argument(
        "--lattice-root",
        type=Path,
        default=Path("outputs/grasp_edit_lattice"),
        help="Directory for compiled Wrist Lattice trajectories.",
    )
    parser.add_argument("--grasp-root", type=Path, action="append", dest="grasp_roots")
    parser.add_argument("--graspqp-seeds", type=int, default=budget.graspqp_seeds)
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--graspqp-executions", type=int, default=budget.graspqp_executions)
    parser.add_argument(
        "--train-lattice-success",
        action="store_true",
        help="Run PPO even when CPU lattice preflight already has a successful template.",
    )
    parser.add_argument(
        "--resume-existing-rl",
        action="store_true",
        help=(
            "Continue an existing per-object checkpoint up to --max-updates instead of "
            "deleting its RL directory. Intended for targeted RL_PROMISING retries."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--initial-updates", type=int, default=5)
    parser.add_argument("--mid-updates", type=int, default=10)
    parser.add_argument("--max-updates", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gpus",
        default="auto",
        help=(
            "Physical GPU ids for object-level scheduling. 'auto' uses CUDA_VISIBLE_DEVICES, "
            "or falls back to the index in --device."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        default="auto",
        help=(
            "Pipeline workers per GPU. 'auto' infers memory/CPU headroom and caps at two "
            "workers so CPU lattice and Grasp work can overlap PPO."
        ),
    )
    parser.add_argument(
        "--gpu-jobs-per-gpu",
        default="auto",
        help=(
            "Maximum simultaneous GPU subprocesses per card. 'auto' uses free memory, "
            "launch utilization, and worker count, capped at two."
        ),
    )
    parser.add_argument(
        "--ppo-jobs-per-gpu",
        default="auto",
        help=(
            "Maximum simultaneous PPO subprocesses per card. 'auto' keeps the observed "
            "compute-saturating PPO phase exclusive while general GPU slots overlap Grasp."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-candidates", type=int, default=3)
    parser.add_argument("--lattice-max-templates", type=int, default=12)
    parser.add_argument("--lattice-max-executions", type=int, default=32)
    parser.add_argument(
        "--execution-lift-height",
        type=float,
        default=0.065,
        help="C MuJoCo wrist lift trajectory height in metres; success remains 55 mm.",
    )
    parser.add_argument(
        "--hand-edit-fraction",
        type=float,
        default=0.35,
        help="Maximum normalized six-actuator edit applied around each lattice grip.",
    )
    parser.add_argument("--promising-lift-mm", type=float, default=20.0)
    parser.add_argument("--promising-success-rate", type=float, default=0.01)
    parser.add_argument(
        "--early-fail-lift-mm",
        type=float,
        default=10.0,
        help="At the first stage, stop only if best lift is below this and progress is flat.",
    )
    parser.add_argument(
        "--continue-lift-mm",
        type=float,
        default=20.0,
        help="A best lift at or above this is enough evidence to grant another stage.",
    )
    parser.add_argument(
        "--progress-gain-mm",
        type=float,
        default=5.0,
        help="Recent mean-lift gain that counts as meaningful training progress.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore matching cached benchmark rows."
    )
    parser.add_argument("--verbose-child", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expect_count < 0:
        raise ValueError("--expect-count must be >= 0; use 0 to disable the check.")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if not (0 < args.initial_updates <= args.mid_updates <= args.max_updates):
        raise ValueError("Require 0 < --initial-updates <= --mid-updates <= --max-updates.")
    if args.graspqp_seeds <= 0 or args.generation_attempts <= 0:
        raise ValueError("Grasp seed counts must be positive.")
    if args.promising_lift_mm < 0.0 or not 0.0 <= args.promising_success_rate <= 1.0:
        raise ValueError("Invalid promising thresholds.")
    if min(args.early_fail_lift_mm, args.continue_lift_mm, args.progress_gain_mm) < 0.0:
        raise ValueError("Adaptive lift/progress thresholds must be non-negative.")
    if args.early_fail_lift_mm > args.continue_lift_mm:
        raise ValueError("--early-fail-lift-mm cannot exceed --continue-lift-mm.")
    if args.execution_lift_height <= 0.0:
        raise ValueError("--execution-lift-height must be positive.")
    if not 0.0 < args.hand_edit_fraction <= 1.0:
        raise ValueError("--hand-edit-fraction must lie in (0, 1].")

    root = _repo_root()
    os.chdir(root)
    catalog = list(_selection_ids(args.selection) if args.selection else _dataset_ids(args.dataset))
    if args.object_ids:
        requested = set(args.object_ids)
        unknown = requested.difference(catalog)
        if unknown:
            raise ValueError(f"Unknown object id(s): {sorted(unknown)}")
        catalog = [item for item in catalog if item in requested]
    full_catalog_run = not args.object_ids and args.limit is None
    if full_catalog_run and args.expect_count and len(catalog) != args.expect_count:
        raise RuntimeError(
            f"Catalogue count mismatch: expected {args.expect_count}, found {len(catalog)}."
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        catalog = catalog[: args.limit]
    if not catalog:
        raise RuntimeError("No objects selected.")
    _validate_runtime_dependencies(catalog)

    gpus = _gpu_ids(args.gpus, device=args.device)
    slots, resource_plan = _resource_plan(
        gpus,
        workers_per_gpu=args.workers_per_gpu,
        gpu_jobs_per_gpu=args.gpu_jobs_per_gpu,
        ppo_jobs_per_gpu=args.ppo_jobs_per_gpu,
        num_envs=args.num_envs,
    )
    if not slots:
        raise RuntimeError("Resource planning produced no worker slots.")

    source_hashes = _source_hashes(root)
    signature_payload = {
        "pipeline": "graspqp_dexevolve_lattice_mjwarp_ppo",
        "dataset": args.dataset,
        "lattice_root": str(args.lattice_root.resolve()),
        "num_envs": args.num_envs,
        "initial_updates": args.initial_updates,
        "mid_updates": args.mid_updates,
        "max_updates": args.max_updates,
        "device": args.device,
        "seed": args.seed,
        "graspqp_seeds": args.graspqp_seeds,
        "generation_attempts": args.generation_attempts,
        "graspqp_executions": args.graspqp_executions,
        "train_lattice_success": args.train_lattice_success,
        "resume_existing_rl": args.resume_existing_rl,
        "base_candidates": args.base_candidates,
        "lattice_max_templates": args.lattice_max_templates,
        "lattice_max_executions": args.lattice_max_executions,
        "execution_lift_height": args.execution_lift_height,
        "hand_edit_fraction": args.hand_edit_fraction,
        "promising_lift_mm": args.promising_lift_mm,
        "promising_success_rate": args.promising_success_rate,
        "early_fail_lift_mm": args.early_fail_lift_mm,
        "continue_lift_mm": args.continue_lift_mm,
        "progress_gain_mm": args.progress_gain_mm,
        "source_hashes": source_hashes,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    grasp_roots = (
        tuple(args.grasp_roots)
        if args.grasp_roots
        else (Path("outputs/grasp_generation"),)
    )
    output = args.output
    object_dir = output / "objects"
    log_dir = output / "logs"
    rl_root = output / "rl"
    object_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[benchmark] objects={len(catalog)} envs={args.num_envs} "
        f"budget={args.initial_updates}->{args.mid_updates}->{args.max_updates} "
        f"preflight=lattice-first graspqp_seeds={args.graspqp_seeds} "
        f"resume={not args.force} slots={list(slots)} "
        f"gpu_job_limits={resource_plan['gpu_job_limits']} "
        f"ppo_job_limits={resource_plan['ppo_job_limits']} output={output}",
        flush=True,
    )
    for gpu, details in resource_plan.get("gpu_details", {}).items():
        memory = "unknown"
        if details.get("total_memory_mb"):
            memory = (
                f"free={details['free_memory_mb'] / 1024.0:.1f}/"
                f"{details['total_memory_mb'] / 1024.0:.1f}GiB"
            )
        utilization = details.get("utilization_percent")
        utilization_label = "unknown" if utilization is None else f"{utilization:.0f}%"
        print(
            f"[resource] gpu={gpu} workers={details['workers']} "
            f"gpu_jobs={details['gpu_jobs']} ppo_jobs={details['ppo_jobs']} "
            f"estimated_worker={details['estimated_worker_memory_mb'] / 1024.0:.1f}GiB "
            f"memory={memory} utilization={utilization_label}",
            flush=True,
        )
    if not args.train_lattice_success:
        print("[benchmark] CPU lattice success skips PPO entirely (0 RL updates).", flush=True)
    if args.dry_run:
        for index, object_id in enumerate(catalog, 1):
            slot = slots[(index - 1) % len(slots)]
            print(f"[{index:03d}/{len(catalog):03d}] gpu={slot} {object_id}")
        return 0

    rows_by_id: dict[str, dict[str, Any]] = {}
    for object_id in catalog:
        cached = (
            None
            if args.force
            else _load_object_result(object_dir / f"{_slug(object_id)}.json", signature)
        )
        if cached is not None:
            cached.setdefault("gpu", "")
            rows_by_id[object_id] = cached

    pending_objects = [object_id for object_id in catalog if object_id not in rows_by_id]
    active_worker_count = min(len(slots), len(pending_objects)) if pending_objects else 1
    selected_slots = slots[:active_worker_count]
    gpu_phase_capacity = (
        active_worker_count
        if gpus == ("cpu",)
        else sum(
            min(
                selected_slots.count(gpu),
                int(resource_plan["ppo_job_limits"][gpu]),
            )
            for gpu in set(selected_slots)
        )
    )
    eta_parallelism = max(1, min(active_worker_count, gpu_phase_capacity))
    average_runtime, eta, timing_samples = _runtime_estimate(
        rows_by_id.values(),
        remaining=len(pending_objects),
        worker_count=eta_parallelism,
    )
    eta_label = "warming_up" if eta is None else _format_duration(eta)
    average_label = "unknown" if average_runtime is None else _format_duration(average_runtime)
    print(
        f"[estimate] cached={len(rows_by_id)} pending={len(pending_objects)} "
        f"samples={timing_samples} avg={average_label}/object "
        f"workers={active_worker_count} gpu_parallelism={eta_parallelism} eta={eta_label}",
        flush=True,
    )
    for completed_index, object_id in enumerate(
        (item for item in catalog if item in rows_by_id),
        1,
    ):
        print(
            _format_progress(
                completed_index,
                len(catalog),
                rows_by_id[object_id],
                average_runtime=average_runtime,
                eta=eta,
                cached=True,
            ),
            flush=True,
        )

    ordered_rows = [rows_by_id[item] for item in catalog if item in rows_by_id]
    _write_summary(
        output,
        rows=ordered_rows,
        args=args,
        signature=signature,
        source_hashes=source_hashes,
        total_count=len(catalog),
        worker_count=active_worker_count,
        eta_parallelism=eta_parallelism,
        gpus=gpus,
        resource_plan=resource_plan,
    )

    if pending_objects:
        if active_worker_count > 1:
            print("[prepare] validating shared Dex Hand surrogate cache", flush=True)
            _prepare_shared_surrogate()
        parallel_run_started = time.perf_counter()
        completed_this_run = 0
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager:
            slot_queue = manager.Queue()
            for slot in selected_slots:
                slot_queue.put(slot)
            gpu_semaphores = {
                gpu: manager.BoundedSemaphore(int(resource_plan["gpu_job_limits"][gpu]))
                for gpu in set(selected_slots)
                if gpu != "cpu"
            }
            ppo_semaphores = {
                gpu: manager.BoundedSemaphore(int(resource_plan["ppo_job_limits"][gpu]))
                for gpu in set(selected_slots)
                if gpu != "cpu"
            }
            with ProcessPoolExecutor(
                max_workers=active_worker_count,
                mp_context=context,
                initializer=_init_object_worker,
                initargs=(slot_queue, gpu_semaphores, ppo_semaphores),
            ) as executor:
                futures = {
                    executor.submit(
                        _run_object_item,
                        ObjectWorkItem(
                            object_id=object_id,
                            args=args,
                            root=root,
                            grasp_roots=grasp_roots,
                            object_dir=object_dir,
                            log_dir=log_dir,
                            rl_root=rl_root,
                            signature=signature,
                        ),
                    ): object_id
                    for object_id in pending_objects
                }
                for future in as_completed(futures):
                    object_id = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        if _is_noncacheable_interruption(exc):
                            raise
                        if args.fail_fast:
                            raise
                        row = _empty_row(object_id)
                        row["failure_category"] = f"worker_{type(exc).__name__}: {exc}"
                        row["log_path"] = str(log_dir / f"{_slug(object_id)}.log")
                        _save_object_result(
                            object_dir / f"{_slug(object_id)}.json",
                            row=row,
                            signature=signature,
                        )
                    rows_by_id[object_id] = row
                    completed_this_run += 1
                    ordered_rows = [
                        rows_by_id[item] for item in catalog if item in rows_by_id
                    ]
                    remaining = len(catalog) - len(ordered_rows)
                    wall_timing = _wall_clock_estimate(
                        elapsed=time.perf_counter() - parallel_run_started,
                        completed=completed_this_run,
                        remaining=remaining,
                        warmup_completions=active_worker_count,
                    )
                    if wall_timing[0] is None:
                        average_runtime, eta, _ = _runtime_estimate(
                            ordered_rows,
                            remaining=remaining,
                            worker_count=eta_parallelism,
                        )
                        timing_override = None
                    else:
                        average_runtime, eta, _ = wall_timing
                        timing_override = wall_timing
                    _write_summary(
                        output,
                        rows=ordered_rows,
                        args=args,
                        signature=signature,
                        source_hashes=source_hashes,
                        total_count=len(catalog),
                        worker_count=active_worker_count,
                        eta_parallelism=eta_parallelism,
                        gpus=gpus,
                        resource_plan=resource_plan,
                        timing_override=timing_override,
                    )
                    print(
                        _format_progress(
                            len(ordered_rows),
                            len(catalog),
                            row,
                            average_runtime=average_runtime,
                            eta=eta,
                        ),
                        flush=True,
                    )

    rows = [rows_by_id[item] for item in catalog]
    _write_summary(
        output,
        rows=rows,
        args=args,
        signature=signature,
        source_hashes=source_hashes,
        total_count=len(catalog),
        worker_count=active_worker_count,
        eta_parallelism=eta_parallelism,
        gpus=gpus,
        resource_plan=resource_plan,
    )
    counts = Counter(row["status"] for row in rows)
    print("\n[summary]", flush=True)
    for status in STATUSES:
        if counts.get(status):
            print(f"  {status:<21} {counts[status]:3d}", flush=True)
    needs = sum(bool(row.get("needs_motion_primitive")) for row in rows)
    print(f"  {'NEEDS_MOTION_PRIMITIVE':<21} {needs:3d}", flush=True)
    print(f"[done] csv={output / 'summary.csv'} json={output / 'summary.json'}", flush=True)
    return 0 if counts.get("PIPELINE_ERROR", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
