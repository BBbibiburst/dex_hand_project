"""Run real online task attempts and transactionally export the five-task dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from apps.check_p30_f_drive import check as check_drive
from apps.export_five_task_dataset import build_parser as export_parser, export
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import ExecutionConfig, execute_grasp
from source.grasping.nut_transfer import NutTransferConfig, execute_nut_transfer
from source.grasping.push_transfer import PushTransferConfig, execute_push_transfer
from source.grasping.stack_transfer import StackTransferConfig, execute_stack_transfer
from source.grasping.task_transfer import PickPlaceTransferConfig, execute_pick_place_transfer
from source.workflows.batch import atomic_write_json
from tools.grasping.transfer_lift_to_pick_place import load_lift_source
from tools.grasping.transfer_lift_to_stack import _execution_config


class _TimedEnvironment:
    def __init__(self, environment, metrics: dict[str, float]) -> None:
        self._environment = environment
        self._metrics = metrics

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def reset(self, *args, **kwargs):
        started = time.perf_counter()
        result = self._environment.reset(*args, **kwargs)
        self._metrics["reset_seconds"] += time.perf_counter() - started
        return result

    def step(self, *args, **kwargs):
        started = time.perf_counter()
        result = self._environment.step(*args, **kwargs)
        self._metrics["simulation_step_seconds"] += time.perf_counter() - started
        return result


@contextmanager
def _instrument_task_environment(task: str, metrics: dict[str, float]):
    mapping = {
        "lift": (execute_grasp.__module__, "make_lift_env"),
        "pick_place": (execute_pick_place_transfer.__module__, "make_pick_place_env"),
        "stack": (execute_stack_transfer.__module__, "make_stack_env"),
        "push": (execute_push_transfer.__module__, "make_push_env"),
        "nut_assembly": (execute_nut_transfer.__module__, "make_nut_assembly_env"),
    }
    module_name, attribute = mapping[task]
    module = sys.modules[module_name]
    original = getattr(module, attribute)

    def timed_factory(*args, **kwargs):
        started = time.perf_counter()
        environment = original(*args, **kwargs)
        metrics["environment_init_seconds"] += time.perf_counter() - started
        return _TimedEnvironment(environment, metrics)

    setattr(module, attribute, timed_factory)
    try:
        yield
    finally:
        setattr(module, attribute, original)


def _dataclass_from_dict(kind, payload: dict):
    allowed = {item.name for item in fields(kind)}
    return kind(**{key: value for key, value in payload.items() if key in allowed})


def _successful_source(reference: Path) -> DemonstrationEpisode:
    summary_path = reference.parent.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for attempt in reversed(summary.get("attempts", [])):
        if attempt.get("success") and attempt.get("source_manifest"):
            source_path = Path(str(attempt["source_manifest"]))
            if not source_path.is_file() and "outputs" in source_path.parts:
                source_path = Path.cwd().joinpath(*source_path.parts[source_path.parts.index("outputs") :])
            return load_lift_source(source_path)
    raise ValueError(f"No successful source lineage in {summary_path}")


def _execute_task(job: dict) -> DemonstrationEpisode:
    task = job["task"]
    seed = int(job["seed"])
    reference_path = Path(job["reference_manifest"])
    # Some legacy transfer manifests contain a candidate matrix that was
    # serialized after task execution and no longer passes the stricter modern
    # orthonormality guard.  Pick/Stack/Push need only the recorded transfer
    # config plus their validated Lift lineage, so avoid unnecessarily loading
    # that legacy candidate.  Lift and Nut genuinely consume the episode.
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_metadata = dict(reference_payload.get("metadata", {}))
    if task == "lift":
        reference = DemonstrationEpisode.load(reference_path)
        config = _dataclass_from_dict(
            ExecutionConfig, reference.metadata.get("execution_config", {})
        )
        episode = execute_grasp(reference.candidate, seed=seed, config=config)
    elif task == "pick_place":
        source = _successful_source(reference_path)
        transfer = _dataclass_from_dict(
            PickPlaceTransferConfig, reference_metadata.get("transfer_config", {})
        )
        episode = execute_pick_place_transfer(
            source.candidate,
            seed=seed,
            grasp_config=_execution_config(source),
            transfer_config=transfer,
            source_object_position=np.asarray(
                source.metadata["object_initial_position"], dtype=np.float64
            ),
            source_object_quaternion_wxyz=np.asarray(
                source.metadata["object_initial_quaternion_wxyz"], dtype=np.float64
            ),
            source_episode=source,
        )
    elif task == "stack":
        source = _successful_source(reference_path)
        transfer = _dataclass_from_dict(
            StackTransferConfig, reference_metadata.get("transfer_config", {})
        )
        episode = execute_stack_transfer(
            source,
            seed=seed,
            grasp_config=_execution_config(source),
            transfer_config=transfer,
        )
    elif task == "push":
        source = _successful_source(reference_path)
        transfer = _dataclass_from_dict(
            PushTransferConfig, reference_metadata.get("transfer_config", {})
        )
        episode = execute_push_transfer(
            source,
            seed=seed,
            grasp_config=_execution_config(source),
            transfer_config=transfer,
        )
    elif task == "nut_assembly":
        reference = DemonstrationEpisode.load(reference_path)
        source = reference
        source.metadata.setdefault(
            "object_initial_position", np.asarray(source.arrays["object_position"][0]).tolist()
        )
        source.metadata.setdefault(
            "object_initial_quaternion_wxyz",
            np.asarray(source.arrays["object_quaternion_wxyz"][0]).tolist(),
        )
        transfer = _dataclass_from_dict(
            NutTransferConfig, reference.metadata.get("transfer_config", {})
        )
        variant = int(reference.object_id.rsplit(":", 1)[1])
        episode = execute_nut_transfer(
            source,
            variant_index=variant,
            seed=seed,
            grasp_config=ExecutionConfig(enable_tactile_sensors=True),
            transfer_config=transfer,
        )
    else:
        raise ValueError(f"Unsupported task {task!r}")
    return episode


def execute_online(job: dict) -> tuple[DemonstrationEpisode, dict]:
    metrics = {
        "environment_init_seconds": 0.0,
        "reset_seconds": 0.0,
        "simulation_step_seconds": 0.0,
    }
    started = time.perf_counter()
    with _instrument_task_environment(job["task"], metrics):
        episode = _execute_task(job)
    wall = time.perf_counter() - started
    policy = max(
        0.0,
        wall
        - metrics["environment_init_seconds"]
        - metrics["reset_seconds"]
        - metrics["simulation_step_seconds"],
    )
    timing = {
        "online_total_seconds": wall,
        "environment_init_seconds": metrics["environment_init_seconds"],
        "reset_seconds": metrics["reset_seconds"],
        "policy_planner_seconds": policy,
        "simulation_step_wall_seconds": metrics["simulation_step_seconds"],
        "note": "Simulation timing is accumulated env.step wall time; physics duration is reported separately from control timestamps.",
    }
    return episode, timing


def _single_job(job: dict, args: argparse.Namespace) -> dict:
    job_root = args.output_dir / job["job_id"]
    result_path = job_root / "job_result.json"
    if args.resume and result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    source_manifest = job_root / "source_episode/manifest.json"
    timing_path = job_root / "online_timing.json"
    resumed_source = args.resume and source_manifest.is_file() and timing_path.is_file()
    if job_root.exists() and not resumed_source:
        raise FileExistsError(f"Refusing non-resume existing job {job_root}")
    if resumed_source:
        episode = DemonstrationEpisode.load(source_manifest)
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    else:
        partial = args.output_dir / f".partial-{job['job_id']}"
        if partial.exists():
            raise FileExistsError(f"Stale online partial requires review: {partial}")
        partial.mkdir(parents=True)
        episode, timing = execute_online(job)
        episode.save(partial / "source_episode")
        timing_path = partial / "online_timing.json"
        atomic_write_json(timing_path, timing)
        partial.replace(job_root)
        source_manifest = job_root / "source_episode/manifest.json"
        timing_path = job_root / "online_timing.json"
    export_args = export_parser().parse_args(
        [
            "--source-manifest", str(source_manifest),
            "--output-root", str(args.dataset_root),
            "--object-registry", str(args.object_registry),
            "--attempt-id", job["job_id"],
            "--online-timing", str(timing_path),
            "--encoder", "libx264",
            "--queue-size", "8",
        ]
    )
    exported = export(export_args)
    result = {
        "job_id": job["job_id"],
        "task": job["task"],
        "object_id": job["object_id"],
        "seed": job["seed"],
        "online_success": bool(episode.success),
        "failure_reason": episode.failure_reason,
        "online_frames": len(episode.arrays["action"]),
        "online_physical_seconds": len(episode.arrays["action"]) * 0.05,
        "timing": timing,
        "export": exported,
        "status": "PASS"
        if exported["status"] in {"EXPORTED", "DUPLICATE_LINEAGE", "DUPLICATE_CONTENT"}
        else "FAIL",
    }
    atomic_write_json(result_path, result)
    return result


def _subprocess_job(job: dict, args: argparse.Namespace) -> dict:
    drive = check_drive(args.dataset_root)
    if not drive["ready"] or drive["safe_new_bytes_after_20_percent_reserve"] <= 0:
        return {
            "job_id": job["job_id"],
            "task": job["task"],
            "object_id": job["object_id"],
            "seed": job["seed"],
            "status": "SAFETY_STOP_DISK",
            "drive": drive,
        }
    command = [
        sys.executable,
        __file__,
        "--single-job-json", json.dumps(job),
        "--output-dir", str(args.output_dir),
        "--dataset-root", str(args.dataset_root),
        "--object-registry", str(args.object_registry),
    ]
    if args.resume:
        command.append("--resume")
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result_path = args.output_dir / job["job_id"] / "job_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            "job_id": job["job_id"],
            "task": job["task"],
            "object_id": job["object_id"],
            "seed": job["seed"],
            "status": "PROCESS_ERROR",
            "return_code": completed.returncode,
            "stderr": completed.stderr[-4000:],
        }
    result["worker_wall_seconds"] = time.perf_counter() - started
    result["command"] = command
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--single-job-json")
    parser.add_argument("--task", choices=("lift", "pick_place", "stack", "push", "nut_assembly"))
    parser.add_argument("--object-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--object-registry", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or args.workers > 2:
        raise ValueError("P31 worker count must be 1 or 2; P30 proved 4 unsafe.")
    if args.resolution != "1920x1080" or args.fps != 20:
        raise ValueError("P31 pilot contract is frozen at native 1920x1080 and 20 FPS.")
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    drive = check_drive(args.dataset_root)
    if not drive["ready"]:
        raise RuntimeError(f"F-volume preflight failed: {drive}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.single_job_json:
        result = _single_job(json.loads(args.single_job_json), args)
        print(json.dumps(result, ensure_ascii=False))
        return
    if not args.plan:
        raise ValueError("--plan is required for scheduler mode.")
    jobs = json.loads(args.plan.read_text(encoding="utf-8"))["jobs"]
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "jobs": jobs}, ensure_ascii=False))
        return
    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_subprocess_job, job, args): job for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            atomic_write_json(
                args.output_dir / "pilot_progress.json",
                {"completed": len(rows), "total": len(jobs), "rows": rows},
            )
    payload = {
        "wall_seconds": time.perf_counter() - started,
        "workers": args.workers,
        "jobs": rows,
        "passed": sum(row.get("status") == "PASS" for row in rows),
        "failed": sum(row.get("status") != "PASS" for row in rows),
    }
    atomic_write_json(args.output_dir / "pilot_run.json", payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
