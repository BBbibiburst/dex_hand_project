"""Run a resumable low-cost v10 grasp-edit diagnostic over the object catalogue.

This benchmark is intentionally a screening pass, not final policy training.
For each object it ensures an UltraDexGrasp prior exists, optionally skips RL
when Ultra already succeeds, otherwise runs a short DIRECT wrist-lattice +
hybrid v10 PPO job and records a compact classification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STATUSES = (
    "ULTRA_SUCCESS",
    "LATTICE_SUCCESS",
    "RL_SUCCESS",
    "RL_PROMISING",
    "DIRECT_FAILED",
    "NO_ULTRA_PRIOR",
    "NO_REACHABLE_TEMPLATE",
    "PIPELINE_ERROR",
)

CSV_FIELDS = (
    "object_id",
    "dataset",
    "status",
    "needs_motion_primitive",
    "ultra_attempts",
    "ultra_success",
    "ultra_seed_index",
    "ultra_best_lift_mm",
    "ultra_best_final_lift_mm",
    "lattice_candidates",
    "lattice_reachable",
    "lattice_templates",
    "lattice_successful_templates",
    "lattice_failed_templates",
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
_UPDATE_RE = re.compile(
    r"u\s+(?P<update>\d+)/(?P<total>\d+)\s+\|\s+"
    r"succ\s+(?P<success>[0-9.]+)%.*?"
    r"top\s+t(?P<template>\d+):(?P<rate>[0-9.]+)%"
)


@dataclass
class ChildResult:
    returncode: int
    duration_sec: float
    text: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


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
    )
    if proc.returncode:
        raise RuntimeError("Run this benchmark inside dex_hand_project.")
    return Path(proc.stdout.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    files = (
        "apps/train_grasp_edit_rl.py",
        "source/rl/grasp_edit_hybrid_ppo.py",
        "source/rl/grasp_edit_templates.py",
        "source/rl/mjwarp_grasp_edit_env.py",
    )
    return {name: _sha256(root / name) for name in files if (root / name).is_file()}


def _dataset_ids(dataset: str) -> tuple[str, ...]:
    from source.envs.manipulation.object_catalog import object_ids

    if dataset == "all":
        return object_ids()
    return object_ids(dataset)


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


def _discover_ultra(object_id: str, roots: tuple[Path, ...]):
    from source.rl.grasp_edit_templates import discover_ultra_attempts

    return discover_ultra_attempts(object_id, roots=roots, maximum=256)


def _ultra_summary(object_id: str, roots: tuple[Path, ...]) -> dict[str, Any]:
    attempts = _discover_ultra(object_id, roots)
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


def _ensure_ultra(
    args: argparse.Namespace,
    object_id: str,
    roots: tuple[Path, ...],
    *,
    root: Path,
    log_path: Path,
) -> tuple[dict[str, Any], float, str]:
    summary = _ultra_summary(object_id, roots)
    if summary["attempts"]:
        return summary, 0.0, ""

    elapsed = 0.0
    combined = ""
    primary = roots[0]
    for offset in range(args.ultra_generate_seeds):
        rng_seed = args.seed + offset
        output = primary / _slug(object_id) / f"seed_{rng_seed:04d}"
        command = [
            sys.executable,
            "-m",
            "tools.ultradexgrasp.generate",
            "--object-id",
            object_id,
            "--seed",
            str(rng_seed),
            "--seed-count",
            str(args.ultra_seed_count),
            "--max-execution-candidates",
            str(args.ultra_max_execution_candidates),
            "--output",
            str(output),
        ]
        if output.exists():
            command.append("--overwrite")
        child = _run_child(command, cwd=root, log_path=log_path, verbose=args.verbose_child)
        elapsed += child.duration_sec
        combined += child.text
        summary = _ultra_summary(object_id, roots)
        if summary["attempts"]:
            return summary, elapsed, combined
    return summary, elapsed, combined


def _parse_lattice(text: str) -> dict[str, int]:
    matches = list(_LATTICE_RE.finditer(text))
    if not matches:
        return {"candidates": 0, "reachable": 0, "templates": 0}
    match = matches[-1]
    return {key: int(match.group(key)) for key in ("candidates", "reachable", "templates")}


def _parse_update_history(text: str) -> dict[str, Any]:
    matches = list(_UPDATE_RE.finditer(text))
    if not matches:
        return {
            "updates": 0,
            "best_success_rate": 0.0,
            "top_template": "",
            "top_template_fraction": 0.0,
        }
    best_success = max(float(match.group("success")) / 100.0 for match in matches)
    last = matches[-1]
    return {
        "updates": int(last.group("update")),
        "best_success_rate": best_success,
        "top_template": f"t{int(last.group('template'))}",
        "top_template_fraction": float(last.group("rate")) / 100.0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _classify_training(
    *,
    ultra: dict[str, Any],
    child: ChildResult,
    train_output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = _read_json(train_output / "config.json")
    metrics = _read_json(train_output / "metrics.json")
    templates = config.get("templates", []) if isinstance(config.get("templates", []), list) else []
    successful_templates = sum(bool(row.get("success_before_edit")) for row in templates)
    failed_templates = len(templates) - successful_templates
    lattice = _parse_lattice(child.text)
    history = _parse_update_history(child.text)

    final_success_rate = float(metrics.get("episode_success_rate", 0.0))
    best_lift = float(metrics.get("best_attempt_lift", 0.0))
    best_final_lift = float(metrics.get("best_attempt_final_lift", 0.0))
    completed = int(round(float(metrics.get("completed_episodes", 0.0))))
    best_exists = (train_output / "best_trajectory" / "manifest.json").is_file()

    if ultra["success"]:
        status = "ULTRA_SUCCESS"
        failure = ""
    elif successful_templates:
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
        failure = "short_rl_not_yet_successful"
    else:
        status = "DIRECT_FAILED"
        failure = "short_rl_no_progress"

    return {
        "status": status,
        "needs_motion_primitive": status
        in {"DIRECT_FAILED", "NO_ULTRA_PRIOR", "NO_REACHABLE_TEMPLATE"},
        "lattice_candidates": lattice["candidates"],
        "lattice_reachable": lattice["reachable"],
        "lattice_templates": lattice["templates"] or len(templates),
        "lattice_successful_templates": successful_templates,
        "lattice_failed_templates": failed_templates,
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
        "status": "PIPELINE_ERROR",
        "needs_motion_primitive": False,
        "ultra_attempts": 0,
        "ultra_success": False,
        "ultra_seed_index": "",
        "ultra_best_lift_mm": 0.0,
        "ultra_best_final_lift_mm": 0.0,
        "lattice_candidates": 0,
        "lattice_reachable": 0,
        "lattice_templates": 0,
        "lattice_successful_templates": 0,
        "lattice_failed_templates": 0,
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
) -> None:
    counts = Counter(str(row.get("status", "")) for row in rows)
    _atomic_csv(output / "summary.csv", rows)
    _atomic_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "updated_at": _utc_now(),
            "signature": signature,
            "dataset": args.dataset,
            "count": len(rows),
            "status_counts": {status: counts.get(status, 0) for status in STATUSES},
            "source_hashes": source_hashes,
            "settings": {
                "num_envs": args.num_envs,
                "updates": args.updates,
                "ultra_seed_count": args.ultra_seed_count,
                "ultra_generate_seeds": args.ultra_generate_seeds,
                "base_candidates": args.base_candidates,
                "lattice_max_templates": args.lattice_max_templates,
                "lattice_max_executions": args.lattice_max_executions,
                "promising_lift_mm": args.promising_lift_mm,
                "promising_success_rate": args.promising_success_rate,
                "train_ultra_success": args.train_ultra_success,
            },
            "results": rows,
        },
    )


def _format_progress(index: int, total: int, row: dict[str, Any]) -> str:
    status = str(row["status"])
    ultra = "Y" if row.get("ultra_success") else "N"
    templates = int(row.get("lattice_templates") or 0)
    success = 100.0 * float(row.get("rl_best_success_rate") or 0.0)
    lift = float(row.get("rl_best_lift_mm") or row.get("ultra_best_lift_mm") or 0.0)
    runtime = float(row.get("runtime_sec") or 0.0)
    return (
        f"[{index:03d}/{total:03d}] {row['object_id']:<34} "
        f"{status:<21} ultra={ultra} tpl={templates:02d} "
        f"rl={success:5.1f}% lift={lift:5.1f}mm {runtime:6.1f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--expect-count", type=int, default=127)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_edit_benchmark_v10"))
    parser.add_argument("--ultra-root", type=Path, action="append", dest="ultra_roots")
    parser.add_argument("--ultra-seed-count", type=int, default=100)
    parser.add_argument("--ultra-generate-seeds", type=int, default=3)
    parser.add_argument("--ultra-max-execution-candidates", type=int, default=8)
    parser.add_argument("--train-ultra-success", action="store_true")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-candidates", type=int, default=3)
    parser.add_argument("--lattice-max-templates", type=int, default=12)
    parser.add_argument("--lattice-max-executions", type=int, default=32)
    parser.add_argument("--promising-lift-mm", type=float, default=20.0)
    parser.add_argument("--promising-success-rate", type=float, default=0.01)
    parser.add_argument("--force", action="store_true", help="Ignore matching cached benchmark rows.")
    parser.add_argument("--verbose-child", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expect_count < 0:
        raise ValueError("--expect-count must be >= 0; use 0 to disable the check.")
    if args.num_envs <= 0 or args.updates <= 0:
        raise ValueError("--num-envs and --updates must be positive.")
    if args.ultra_seed_count <= 0 or args.ultra_generate_seeds <= 0:
        raise ValueError("Ultra seed counts must be positive.")
    if args.promising_lift_mm < 0.0 or not 0.0 <= args.promising_success_rate <= 1.0:
        raise ValueError("Invalid promising thresholds.")

    root = _repo_root()
    os.chdir(root)
    catalog = list(_dataset_ids(args.dataset))
    if args.object_ids:
        requested = set(args.object_ids)
        unknown = requested.difference(catalog)
        if unknown:
            raise ValueError(f"Unknown object id(s): {sorted(unknown)}")
        catalog = [item for item in catalog if item in requested]
    full_catalog_run = not args.object_ids and args.limit is None and args.dataset == "all"
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

    source_hashes = _source_hashes(root)
    signature_payload = {
        "version": 10,
        "dataset": args.dataset,
        "num_envs": args.num_envs,
        "updates": args.updates,
        "device": args.device,
        "seed": args.seed,
        "ultra_seed_count": args.ultra_seed_count,
        "ultra_generate_seeds": args.ultra_generate_seeds,
        "ultra_max_execution_candidates": args.ultra_max_execution_candidates,
        "train_ultra_success": args.train_ultra_success,
        "base_candidates": args.base_candidates,
        "lattice_max_templates": args.lattice_max_templates,
        "lattice_max_executions": args.lattice_max_executions,
        "promising_lift_mm": args.promising_lift_mm,
        "promising_success_rate": args.promising_success_rate,
        "source_hashes": source_hashes,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    ultra_roots = (
        tuple(args.ultra_roots)
        if args.ultra_roots
        else (Path("outputs/ultradexgrasp"), Path("outputs/ultradexgrasp_catalog"))
    )
    output = args.output
    object_dir = output / "objects"
    log_dir = output / "logs"
    rl_root = output / "rl"
    object_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[benchmark] objects={len(catalog)} envs={args.num_envs} updates={args.updates} "
        f"ultra_seeds={args.ultra_seed_count} resume={not args.force} output={output}",
        flush=True,
    )
    if not args.train_ultra_success:
        print("[benchmark] Ultra-success objects skip short RL by default.", flush=True)
    if args.dry_run:
        for index, object_id in enumerate(catalog, 1):
            print(f"[{index:03d}/{len(catalog):03d}] {object_id}")
        return 0

    rows_by_id: dict[str, dict[str, Any]] = {}
    for object_id in catalog:
        cached = None if args.force else _load_object_result(
            object_dir / f"{_slug(object_id)}.json", signature
        )
        if cached is not None:
            rows_by_id[object_id] = cached

    for index, object_id in enumerate(catalog, 1):
        if object_id in rows_by_id:
            print(_format_progress(index, len(catalog), rows_by_id[object_id]) + " [cached]", flush=True)
            continue

        started = time.perf_counter()
        row = _empty_row(object_id)
        log_path = log_dir / f"{_slug(object_id)}.log"
        row["log_path"] = str(log_path)
        try:
            ultra, ultra_time, _ = _ensure_ultra(
                args,
                object_id,
                ultra_roots,
                root=root,
                log_path=log_path,
            )
            row.update(
                {
                    "ultra_attempts": ultra["attempts"],
                    "ultra_success": ultra["success"],
                    "ultra_seed_index": ultra["seed_index"],
                    "ultra_best_lift_mm": round(float(ultra["best_lift_mm"]), 3),
                    "ultra_best_final_lift_mm": round(float(ultra["best_final_lift_mm"]), 3),
                }
            )
            if not ultra["attempts"]:
                row["status"] = "NO_ULTRA_PRIOR"
                row["needs_motion_primitive"] = True
                row["failure_category"] = "ultra_no_full_attempt"
            elif ultra["success"] and not args.train_ultra_success:
                row["status"] = "ULTRA_SUCCESS"
                row["needs_motion_primitive"] = False
            else:
                train_output = rl_root / _slug(object_id)
                command = [
                    sys.executable,
                    "-m",
                    "apps.train_grasp_edit_rl",
                    "--object-id",
                    object_id,
                    "--output-root",
                    str(rl_root),
                    "--template-root",
                    "outputs/grasp_edit_lattice_v9",
                    "--no-auto-ultra",
                    "--num-envs",
                    str(args.num_envs),
                    "--updates",
                    str(args.updates),
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
                ]
                for ultra_root in ultra_roots:
                    command.extend(["--ultra-root", str(ultra_root)])
                child = _run_child(
                    command,
                    cwd=root,
                    log_path=log_path,
                    verbose=args.verbose_child,
                )
                row.update(
                    _classify_training(
                        ultra=ultra,
                        child=child,
                        train_output=train_output,
                        args=args,
                    )
                )
                _ = ultra_time
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep the 127-object sweep alive
            row["status"] = "PIPELINE_ERROR"
            row["failure_category"] = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[benchmark-exception] {type(exc).__name__}: {exc}\n")
            if args.fail_fast:
                raise

        row["runtime_sec"] = round(time.perf_counter() - started, 3)
        rows_by_id[object_id] = row
        _save_object_result(
            object_dir / f"{_slug(object_id)}.json",
            row=row,
            signature=signature,
        )
        ordered_rows = [rows_by_id[item] for item in catalog if item in rows_by_id]
        _write_summary(
            output,
            rows=ordered_rows,
            args=args,
            signature=signature,
            source_hashes=source_hashes,
        )
        print(_format_progress(index, len(catalog), row), flush=True)

    rows = [rows_by_id[item] for item in catalog]
    _write_summary(
        output,
        rows=rows,
        args=args,
        signature=signature,
        source_hashes=source_hashes,
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
