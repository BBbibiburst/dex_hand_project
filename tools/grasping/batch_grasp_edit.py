"""Run a resumable preflight + adaptive-budget grasp-edit diagnostic.

This benchmark is intentionally a screening pass, not final policy training.
For each object it first ensures an Ultra Prior exists, builds a CPU DIRECT
wrist-lattice preflight, and conditionally starts MJWarp PPO. By default an
Ultra- or lattice-successful object exits early; explicit stress-test options
can still run hybrid grasp-edit PPO with an adaptive 5 -> 10 -> 15 budget.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("Run this benchmark inside dex_hand_project.")
    return Path(proc.stdout.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    files = (
        "assets/grippers/dex_hand/dex_hand.xml",
        "configs/ultradexgrasp/default.json",
        "apps/train_grasp_edit_rl.py",
        "source/rl/common/ppo.py",
        "source/rl/grasp_edit/env.py",
        "source/rl/grasp_edit/ppo.py",
        "source/rl/grasp_edit/templates.py",
        "source/ultradexgrasp/executor.py",
        "source/ultradexgrasp/hand_surrogate.py",
        "source/ultradexgrasp/synthesizer.py",
        "tools/ultradexgrasp/generate.py",
    )
    return {name: _sha256(root / name) for name in files}


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
    from source.rl.grasp_edit.templates import discover_ultra_attempts

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
    ultra_roots: tuple[Path, ...],
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
                ultra_roots=ultra_roots,
                base_candidates=args.base_candidates,
                maximum_templates=args.lattice_max_templates,
                maximum_executions=args.lattice_max_executions,
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
    ultra_roots: tuple[Path, ...],
    root: Path,
    log_path: Path,
    rl_root: Path,
) -> ChildResult:
    """Train 5 -> 10 -> 15 updates using grasp-edit checkpoints rather than restarting."""
    train_output = rl_root / _slug(object_id)
    if train_output.exists():
        shutil.rmtree(train_output)

    stage_targets = []
    for value in (args.initial_updates, args.mid_updates, args.max_updates):
        if not stage_targets or value != stage_targets[-1]:
            stage_targets.append(value)

    combined_text = ""
    total_duration = 0.0
    last_returncode = 2
    completed_target = 0

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
            "--no-auto-ultra",
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
        ]
        checkpoint = train_output / "checkpoint_final.pt"
        if completed_target and checkpoint.is_file():
            command.extend(["--resume", str(checkpoint)])
        for ultra_root in ultra_roots:
            command.extend(["--ultra-root", str(ultra_root)])

        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[adaptive-stage] target={target_update} additional={additional_updates}\n"
            )

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
    ultra: dict[str, Any],
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
        failure = "adaptive_rl_not_yet_successful"
    else:
        status = "DIRECT_FAILED"
        failure = "adaptive_rl_no_progress"

    return {
        "status": status,
        "needs_motion_primitive": status
        in {"DIRECT_FAILED", "NO_ULTRA_PRIOR", "NO_REACHABLE_TEMPLATE"},
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
                "initial_updates": args.initial_updates,
                "mid_updates": args.mid_updates,
                "max_updates": args.max_updates,
                "ultra_seed_count": args.ultra_seed_count,
                "ultra_generate_seeds": args.ultra_generate_seeds,
                "base_candidates": args.base_candidates,
                "lattice_max_templates": args.lattice_max_templates,
                "lattice_max_executions": args.lattice_max_executions,
                "lattice_root": str(args.lattice_root),
                "promising_lift_mm": args.promising_lift_mm,
                "promising_success_rate": args.promising_success_rate,
                "early_fail_lift_mm": args.early_fail_lift_mm,
                "continue_lift_mm": args.continue_lift_mm,
                "progress_gain_mm": args.progress_gain_mm,
                "train_ultra_success": args.train_ultra_success,
                "train_lattice_success": args.train_lattice_success,
            },
            "results": rows,
        },
    )


def _format_progress(index: int, total: int, row: dict[str, Any]) -> str:
    status = str(row["status"])
    ultra = "Y" if row.get("ultra_success") else "N"
    templates = int(row.get("lattice_templates") or 0)
    success = 100.0 * float(row.get("rl_best_success_rate") or 0.0)
    updates = int(row.get("rl_updates") or 0)
    lift = float(
        row.get("rl_best_lift_mm")
        or row.get("lattice_best_lift_mm")
        or row.get("ultra_best_lift_mm")
        or 0.0
    )
    runtime = float(row.get("runtime_sec") or 0.0)
    return (
        f"[{index:03d}/{total:03d}] {row['object_id']:<34} "
        f"{status:<21} ultra={ultra} tpl={templates:02d} "
        f"u={updates:02d} rl={success:5.1f}% lift={lift:5.1f}mm {runtime:6.1f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--expect-count", type=int, default=127)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_edit_benchmark"))
    parser.add_argument(
        "--lattice-root",
        type=Path,
        default=Path("outputs/grasp_edit_lattice"),
        help="Directory for compiled Wrist Lattice trajectories.",
    )
    parser.add_argument("--ultra-root", type=Path, action="append", dest="ultra_roots")
    parser.add_argument("--ultra-seed-count", type=int, default=100)
    parser.add_argument("--ultra-generate-seeds", type=int, default=3)
    parser.add_argument("--ultra-max-execution-candidates", type=int, default=8)
    parser.add_argument("--train-ultra-success", action="store_true")
    parser.add_argument(
        "--train-lattice-success",
        action="store_true",
        help="Run PPO even when CPU lattice preflight already has a successful template.",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--initial-updates", type=int, default=5)
    parser.add_argument("--mid-updates", type=int, default=10)
    parser.add_argument("--max-updates", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-candidates", type=int, default=3)
    parser.add_argument("--lattice-max-templates", type=int, default=12)
    parser.add_argument("--lattice-max-executions", type=int, default=32)
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
    if args.ultra_seed_count <= 0 or args.ultra_generate_seeds <= 0:
        raise ValueError("Ultra seed counts must be positive.")
    if args.promising_lift_mm < 0.0 or not 0.0 <= args.promising_success_rate <= 1.0:
        raise ValueError("Invalid promising thresholds.")
    if min(args.early_fail_lift_mm, args.continue_lift_mm, args.progress_gain_mm) < 0.0:
        raise ValueError("Adaptive lift/progress thresholds must be non-negative.")
    if args.early_fail_lift_mm > args.continue_lift_mm:
        raise ValueError("--early-fail-lift-mm cannot exceed --continue-lift-mm.")

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
        "pipeline": "ultra_lattice_mjwarp_ppo",
        "dataset": args.dataset,
        "lattice_root": str(args.lattice_root.resolve()),
        "num_envs": args.num_envs,
        "initial_updates": args.initial_updates,
        "mid_updates": args.mid_updates,
        "max_updates": args.max_updates,
        "device": args.device,
        "seed": args.seed,
        "ultra_seed_count": args.ultra_seed_count,
        "ultra_generate_seeds": args.ultra_generate_seeds,
        "ultra_max_execution_candidates": args.ultra_max_execution_candidates,
        "train_ultra_success": args.train_ultra_success,
        "train_lattice_success": args.train_lattice_success,
        "base_candidates": args.base_candidates,
        "lattice_max_templates": args.lattice_max_templates,
        "lattice_max_executions": args.lattice_max_executions,
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
        f"[benchmark] objects={len(catalog)} envs={args.num_envs} "
        f"budget={args.initial_updates}->{args.mid_updates}->{args.max_updates} "
        f"preflight=lattice-first ultra_seeds={args.ultra_seed_count} "
        f"resume={not args.force} output={output}",
        flush=True,
    )
    if not args.train_ultra_success:
        print("[benchmark] Ultra-success objects skip RL by default.", flush=True)
    if not args.train_lattice_success:
        print("[benchmark] CPU lattice success skips PPO entirely (0 RL updates).", flush=True)
    if args.dry_run:
        for index, object_id in enumerate(catalog, 1):
            print(f"[{index:03d}/{len(catalog):03d}] {object_id}")
        return 0

    rows_by_id: dict[str, dict[str, Any]] = {}
    for object_id in catalog:
        cached = (
            None
            if args.force
            else _load_object_result(object_dir / f"{_slug(object_id)}.json", signature)
        )
        if cached is not None:
            rows_by_id[object_id] = cached

    for index, object_id in enumerate(catalog, 1):
        if object_id in rows_by_id:
            print(
                _format_progress(index, len(catalog), rows_by_id[object_id]) + " [cached]",
                flush=True,
            )
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
                preflight = _preflight_lattice(
                    args=args,
                    object_id=object_id,
                    ultra_roots=ultra_roots,
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
                    train_output = rl_root / _slug(object_id)
                    child = _adaptive_train(
                        args=args,
                        object_id=object_id,
                        ultra_roots=ultra_roots,
                        root=root,
                        log_path=log_path,
                        rl_root=rl_root,
                    )
                    row.update(
                        _classify_training(
                            ultra=ultra,
                            preflight=preflight,
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
