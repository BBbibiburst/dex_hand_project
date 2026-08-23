"""Run the UltraDexGrasp episode generator over the complete object catalogue.

Each object/seed pair is executed in an isolated subprocess and output directory.
The batch state is atomically persisted after every attempt, so an interrupted run
can be continued with ``--resume`` without overwriting a completed episode.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from source.envs.manipulation.object_catalog import lift_object_ids
from source.ultradexgrasp.config import DEFAULT_CONFIG_PATH
from source.ultradexgrasp.contracts import PIPELINE_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_SCHEMA_VERSION = 1
TERMINAL_OBJECT_STATES = frozenset({"success", "failed"})
EXPECTED_FAILURE_TYPES = frozenset(
    {
        "geometry_load_failed",
        "no_valid_candidates",
        "ik_unreachable",
        "execution_failed",
        "timeout",
        "process_error",
    }
)


def _slug(object_id: str) -> str:
    return object_id.replace(":", "_").replace("/", "_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _selected_object_ids(
    *,
    dataset: str,
    requested: list[str] | None,
    limit: int | None,
) -> list[str]:
    available = list(lift_object_ids())
    available_set = set(available)
    if requested:
        missing = [object_id for object_id in requested if object_id not in available_set]
        if missing:
            raise ValueError(f"Unknown catalogue object ids: {missing}")
        selected = list(dict.fromkeys(requested))
    else:
        selected = available
        if dataset != "all":
            selected = [item for item in selected if item.startswith(f"{dataset}:")]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        selected = selected[:limit]
    if not selected:
        raise ValueError("Object selection is empty.")
    return selected


def _attempt_directory(root: Path, object_id: str, seed: int) -> Path:
    return root / "objects" / _slug(object_id) / f"seed_{seed:04d}"


def _manifest_is_success(path: Path, *, object_id: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        stored_seed = int(payload.get("seed", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("success")
        and payload.get("object_id") == object_id
        and stored_seed == seed
    )


def _completed_attempt_from_disk(
    attempt_dir: Path,
    *,
    object_id: str,
    seed: int,
) -> dict[str, Any] | None:
    """Recover a child result written just before the parent was interrupted."""
    manifest = attempt_dir / "manifest.json"
    if _manifest_is_success(manifest, object_id=object_id, seed=seed):
        return {
            "seed": seed,
            "status": "success",
            "exit_code": 0,
            "elapsed_seconds": None,
            "output": str(attempt_dir),
            "manifest": str(manifest),
            "log": str(attempt_dir / "attempt.log"),
            "recovered": True,
            "error": None,
        }

    run_path = attempt_dir / "run.json"
    if not run_path.is_file():
        return None
    try:
        payload = _read_json(run_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if payload.get("object_id") not in (None, object_id):
        return None
    if payload.get("seed") not in (None, seed):
        return None
    failure_type = str(payload.get("failure_type") or "process_error")
    if failure_type not in EXPECTED_FAILURE_TYPES:
        failure_type = "process_error"
    return {
        "seed": seed,
        "status": failure_type,
        "exit_code": None,
        "elapsed_seconds": None,
        "output": str(attempt_dir),
        "manifest": None,
        "log": str(attempt_dir / "attempt.log"),
        "recovered": True,
        "error": payload.get("error"),
    }


def _classify_child_result(
    attempt_dir: Path,
    *,
    object_id: str,
    seed: int,
    return_code: int | None,
    timed_out: bool,
) -> tuple[str, str | None, str | None]:
    if timed_out:
        return "timeout", None, "subprocess exceeded the configured timeout"

    manifest = attempt_dir / "manifest.json"
    if _manifest_is_success(manifest, object_id=object_id, seed=seed):
        return "success", str(manifest), None

    run_path = attempt_dir / "run.json"
    if run_path.is_file():
        try:
            payload = _read_json(run_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return "process_error", None, f"invalid run.json: {type(exc).__name__}: {exc}"
        failure_type = str(payload.get("failure_type") or "")
        if failure_type in EXPECTED_FAILURE_TYPES:
            return failure_type, None, payload.get("error")
        if payload.get("success"):
            return "process_error", None, "run.json reports success but no valid manifest exists"

    candidates_path = attempt_dir / "candidates.json"
    if candidates_path.is_file():
        try:
            candidates = _read_json(candidates_path)
            if int(candidates.get("valid_candidate_count", -1)) == 0:
                return "no_valid_candidates", None, None
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return (
        "process_error",
        None,
        f"child exited with code {return_code} without a structured failure result",
    )


def _child_command(args: argparse.Namespace, object_id: str, seed: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.ultradexgrasp.generate",
        "--object-id",
        object_id,
        "--config",
        str(args.config),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--max-execution-candidates",
        str(args.max_execution_candidates),
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.seed_count is not None:
        command.extend(["--seed-count", str(args.seed_count)])
    if args.optimization_steps is not None:
        command.extend(["--optimization-steps", str(args.optimization_steps)])
    return command


def _run_child(
    args: argparse.Namespace,
    *,
    object_id: str,
    seed: int,
    attempt_dir: Path,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / "attempt.log"
    command = _child_command(args, object_id, seed, attempt_dir)
    started = time.monotonic()
    timed_out = False
    return_code: int | None = None
    error: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            error = f"subprocess exceeded {args.timeout_seconds:.1f}s"
            log.write(f"\n[BATCH TIMEOUT] {error}\n")

    elapsed = time.monotonic() - started
    status, manifest, structured_error = _classify_child_result(
        attempt_dir,
        object_id=object_id,
        seed=seed,
        return_code=return_code,
        timed_out=timed_out,
    )
    return {
        "seed": seed,
        "status": status,
        "exit_code": return_code,
        "elapsed_seconds": elapsed,
        "output": str(attempt_dir),
        "manifest": manifest,
        "log": str(log_path),
        "recovered": False,
        "error": structured_error or error,
    }


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    rows = list(state["objects"])
    states = Counter(str(row["status"]) for row in rows)
    attempt_statuses = Counter(
        str(attempt["status"])
        for row in rows
        for attempt in row.get("attempts", [])
    )
    failed_objects = [row for row in rows if row["status"] == "failed"]
    final_failures = Counter(
        str(row["attempts"][-1]["status"])
        for row in failed_objects
        if row.get("attempts")
    )
    completed = states["success"] + states["failed"]
    return {
        "selected_objects": len(rows),
        "completed_objects": completed,
        "successful_objects": states["success"],
        "failed_objects": states["failed"],
        "pending_objects": len(rows) - completed,
        "object_coverage_rate": states["success"] / len(rows) if rows else 0.0,
        "total_attempts": sum(len(row.get("attempts", [])) for row in rows),
        "attempt_outcomes": dict(sorted(attempt_statuses.items())),
        "final_failure_types": dict(sorted(final_failures.items())),
    }


def _persist(root: Path, state: dict[str, Any]) -> None:
    state["summary"] = _state_summary(state)
    state["updated_at"] = _utc_now()
    _write_json_atomic(root / "batch_state.json", state)
    _write_json_atomic(
        root / "batch_summary.json",
        {
            "schema_version": state["schema_version"],
            "pipeline": state["pipeline"],
            "updated_at": state["updated_at"],
            "parameters": state["parameters"],
            "summary": state["summary"],
            "objects": [
                {
                    "object_id": row["object_id"],
                    "status": row["status"],
                    "success_seed": row.get("success_seed"),
                    "manifest": row.get("manifest"),
                    "attempts": len(row.get("attempts", [])),
                    "last_failure": (
                        row["attempts"][-1]["status"]
                        if row.get("attempts") and row["status"] != "success"
                        else None
                    ),
                }
                for row in state["objects"]
            ],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/ultradexgrasp_catalog"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=0, help="First retry seed for every object.")
    parser.add_argument("--max-seeds", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-execution-candidates", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--optimization-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parameters(args: argparse.Namespace, object_ids: list[str], seeds: list[int]) -> dict[str, Any]:
    return {
        "object_ids": object_ids,
        "seeds": seeds,
        "config": str(args.config.resolve()),
        "device": args.device,
        "seed_count": args.seed_count,
        "optimization_steps": args.optimization_steps,
        "max_execution_candidates": args.max_execution_candidates,
        "timeout_seconds": args.timeout_seconds,
    }


def _new_state(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "parameters": parameters,
        "summary": {},
        "objects": [
            {
                "object_id": object_id,
                "status": "pending",
                "attempts": [],
                "success_seed": None,
                "manifest": None,
            }
            for object_id in parameters["object_ids"]
        ],
    }


def _load_or_create_state(
    root: Path,
    *,
    parameters: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    path = root / "batch_state.json"
    if not path.is_file():
        return _new_state(parameters)
    if not resume:
        raise FileExistsError(
            f"Batch state already exists at {path}; pass --resume to continue it "
            "or choose a different --output directory."
        )
    state = _read_json(path)
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported batch state schema in {path}.")
    if state.get("pipeline") != PIPELINE_NAME:
        raise ValueError(f"Batch state belongs to another pipeline: {state.get('pipeline')!r}.")
    if state.get("parameters") != parameters:
        raise ValueError(
            "Cannot resume with different object selection, seed sequence, config, "
            "timeout, or generator overrides."
        )
    return state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_seeds <= 0:
        raise ValueError("--max-seeds must be positive.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.max_execution_candidates <= 0:
        raise ValueError("--max-execution-candidates must be positive.")

    object_ids = _selected_object_ids(
        dataset=args.dataset,
        requested=args.object_ids,
        limit=args.limit,
    )
    if args.expect_count is not None and len(object_ids) != args.expect_count:
        raise ValueError(
            f"Expected {args.expect_count} selected objects, but catalogue selection has "
            f"{len(object_ids)}."
        )
    seeds = [args.seed + offset for offset in range(args.max_seeds)]
    ycb_count = sum(item.startswith("ycb:") for item in object_ids)
    egad_count = sum(item.startswith("egad:") for item in object_ids)
    print(
        f"[plan] objects={len(object_ids)} ycb={ycb_count} egad={egad_count} "
        f"seeds={seeds} output={args.output}",
        flush=True,
    )
    if args.dry_run:
        for index, object_id in enumerate(object_ids, start=1):
            print(f"{index:03d}. {object_id}")
        return 0

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    parameters = _parameters(args, object_ids, seeds)
    state = _load_or_create_state(root, parameters=parameters, resume=args.resume)
    rows = {row["object_id"]: row for row in state["objects"]}
    _persist(root, state)

    try:
        for object_index, object_id in enumerate(object_ids, start=1):
            row = rows[object_id]
            attempted_seeds = {int(item["seed"]) for item in row.get("attempts", [])}
            if row["status"] == "success":
                print(f"[{object_index}/{len(object_ids)}] SKIP {object_id}: success", flush=True)
                continue
            if attempted_seeds.issuperset(seeds):
                row["status"] = "failed"
                _persist(root, state)
                print(f"[{object_index}/{len(object_ids)}] SKIP {object_id}: exhausted", flush=True)
                continue

            row["status"] = "running"
            _persist(root, state)
            for seed in seeds:
                if seed in attempted_seeds:
                    continue
                attempt_dir = _attempt_directory(root, object_id, seed)
                recovered = _completed_attempt_from_disk(
                    attempt_dir,
                    object_id=object_id,
                    seed=seed,
                )
                if recovered is not None:
                    attempt = recovered
                    print(
                        f"[{object_index}/{len(object_ids)}] {object_id} seed={seed} "
                        f"RECOVER {attempt['status']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{object_index}/{len(object_ids)}] {object_id} seed={seed} START",
                        flush=True,
                    )
                    attempt = _run_child(
                        args,
                        object_id=object_id,
                        seed=seed,
                        attempt_dir=attempt_dir,
                    )
                    print(
                        f"[{object_index}/{len(object_ids)}] {object_id} seed={seed} "
                        f"{attempt['status']} elapsed={attempt['elapsed_seconds']:.1f}s",
                        flush=True,
                    )

                row["attempts"].append(attempt)
                attempted_seeds.add(seed)
                if attempt["status"] == "success":
                    row["status"] = "success"
                    row["success_seed"] = seed
                    row["manifest"] = attempt["manifest"]
                    _persist(root, state)
                    break
                row["status"] = "running"
                _persist(root, state)

            if row["status"] != "success":
                row["status"] = "failed"
                _persist(root, state)
    except KeyboardInterrupt:
        for row in state["objects"]:
            if row["status"] == "running":
                row["status"] = "pending"
        _persist(root, state)
        print(f"\n[interrupted] state saved to {root / 'batch_state.json'}", flush=True)
        return 130

    _persist(root, state)
    summary = state["summary"]
    print(
        f"[done] coverage={summary['successful_objects']}/{summary['selected_objects']} "
        f"({summary['object_coverage_rate']:.1%}) attempts={summary['total_attempts']} "
        f"summary={root / 'batch_summary.json'}",
        flush=True,
    )
    return 0 if summary["failed_objects"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
