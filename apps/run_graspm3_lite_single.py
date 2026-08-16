"""Run the GraspM3-lite temporal search and strict C MuJoCo replay for one object."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from apps.train_grasp_edit_rl import _ensure_ultra_priors
from source.rl.grasp_edit.graspm3_lite import (
    GraspM3LiteConfig,
    MjWarpGraspM3LiteEnv,
    TemporalCEMSearch,
)
from source.rl.grasp_edit.primitives import (
    available_grasp_primitives,
    resolve_grasp_primitives,
)
from source.rl.grasp_edit.templates import build_grasp_edit_templates
from source.rl.residual.replay import replay_residual_trajectory


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _slug(object_id: str) -> str:
    return object_id.replace(":", "_").replace("/", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/graspm3_lite"))
    parser.add_argument(
        "--template-root", type=Path, default=Path("outputs/graspm3_lite/lattice")
    )
    parser.add_argument("--ultra-root", type=Path, action="append", dest="ultra_roots")
    parser.add_argument("--ultra-seed-count", type=int, default=100)
    parser.add_argument("--ultra-generate-seeds", type=int, default=3)
    parser.add_argument("--ultra-max-execution-candidates", type=int, default=8)
    parser.add_argument("--no-auto-ultra", action="store_true")
    parser.add_argument("--base-candidates", type=int, default=3)
    parser.add_argument("--wrist-translation-step", type=float, default=0.01)
    parser.add_argument("--wrist-rotation-step-deg", type=float, default=15.0)
    parser.add_argument("--lattice-max-templates", type=int, default=12)
    parser.add_argument("--lattice-max-executions", type=int, default=32)
    parser.add_argument("--overwrite-templates", action="store_true")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace an existing completed/partial result for this object.",
    )
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--smoothing", type=float, default=0.70)
    parser.add_argument("--verification-candidates", type=int, default=8)
    parser.add_argument(
        "--grasp-modes",
        default="all",
        help=(
            "Comma-separated grasp families or 'all'. Available: "
            + ",".join(available_grasp_primitives())
        ),
    )
    parser.add_argument("--mode-bias-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-edit-fraction", type=float, default=0.35)
    parser.add_argument("--success-lift-height", type=float, default=0.055)
    parser.add_argument("--success-tail-steps", type=int, default=8)
    parser.add_argument("--maximum-object-speed", type=float, default=0.65)
    parser.add_argument("--nconmax", type=int, default=192)
    parser.add_argument("--njmax", type=int, default=768)
    parser.add_argument("--verify-tail", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Return zero after producing diagnostics even if C MuJoCo rejects every candidate.",
    )
    return parser


def _roots(args: argparse.Namespace) -> tuple[Path, ...]:
    return (
        tuple(args.ultra_roots)
        if args.ultra_roots
        else (Path("outputs/ultradexgrasp"), Path("outputs/ultradexgrasp_catalog"))
    )


def _candidate_summary(candidate, replay) -> dict[str, Any]:
    metadata = candidate.trajectory.metadata
    return {
        "score": candidate.score,
        "mjwarp_success": candidate.mjwarp_success,
        "reference_schedule": candidate.reference_schedule,
        "grasp_mode": candidate.mode_name,
        "grasp_mode_id": candidate.mode_id,
        "mode_family": metadata.get("mode_family"),
        "mode_objective": metadata.get("mode_objective"),
        "table_assisted": bool(metadata.get("table_assisted", False)),
        "mode_description": metadata.get("mode_description"),
        "template_id": metadata.get("template_id"),
        "template_label": metadata.get("template_label"),
        "mjwarp_max_lift": metadata.get("mjwarp_max_lift", 0.0),
        "mjwarp_final_lift": metadata.get("mjwarp_final_lift", 0.0),
        "c_mujoco_success": None if replay is None else replay.success,
        "c_mujoco_success_fraction": None if replay is None else replay.success_fraction,
        "c_mujoco_object_lift": None if replay is None else replay.object_lift,
        "c_mujoco_error": metadata.get("c_mujoco_error"),
    }


def _mode_names(value: str) -> tuple[str, ...]:
    modes = resolve_grasp_primitives(value)
    return tuple(mode.name for mode in modes)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    result_directories = ("candidates", "best_trajectory", "best_attempt")
    stale = (path / "summary.json").exists() or any(
        (path / name).exists() for name in result_directories
    )
    if stale and not overwrite:
        raise FileExistsError(
            f"Existing GraspM3-lite result found at {path}; "
            "pass --overwrite-output to replace it."
        )
    if overwrite:
        for name in result_directories:
            target = path / name
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        for name in ("config.json", "summary.json"):
            target = path / name
            if target.is_file():
                target.unlink()
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    if args.verify_tail <= 0:
        raise ValueError("--verify-tail must be positive.")
    grasp_modes = _mode_names(args.grasp_modes)
    ultra_roots = _roots(args)
    _ensure_ultra_priors(args, ultra_roots)
    templates = build_grasp_edit_templates(
        args.object_id,
        output_root=args.template_root,
        ultra_roots=ultra_roots,
        base_candidates=args.base_candidates,
        translation_step=args.wrist_translation_step,
        rotation_step_degrees=args.wrist_rotation_step_deg,
        maximum_templates=args.lattice_max_templates,
        maximum_executions=args.lattice_max_executions,
        seed=args.seed,
        overwrite=args.overwrite_templates,
        failed_only=False,
        verbose=args.verbose,
    )
    output = args.output_root / _slug(args.object_id)
    _prepare_output(output, overwrite=args.overwrite_output)
    config = GraspM3LiteConfig(
        num_envs=args.population_size,
        population_size=args.population_size,
        iterations=args.iterations,
        elite_fraction=args.elite_fraction,
        smoothing=args.smoothing,
        verification_candidates=args.verification_candidates,
        grasp_modes=grasp_modes,
        mode_bias_scale=args.mode_bias_scale,
        device=args.device,
        hand_edit_fraction=args.hand_edit_fraction,
        success_lift_height=args.success_lift_height,
        success_tail_steps=args.success_tail_steps,
        maximum_object_speed=args.maximum_object_speed,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    config.validate()
    _write_json(
        output / "config.json",
        {
            "pipeline": "graspm3-lite-temporal",
            "object_id": args.object_id,
            "search": asdict(config),
            "grasp_modes": list(config.grasp_modes),
            "mode_definitions": [
                {
                    "name": mode.name,
                    "family": mode.mode_family,
                    "description": mode.description,
                    "objective": mode.objective_name,
                    "score_weights": list(mode.score_weights),
                    "table_assisted": mode.table_assisted,
                }
                for mode in resolve_grasp_primitives(config.grasp_modes)
            ],
            "lattice": {
                "translation_step": args.wrist_translation_step,
                "rotation_step_degrees": args.wrist_rotation_step_deg,
                "maximum_templates": args.lattice_max_templates,
                "maximum_executions": args.lattice_max_executions,
            },
            "templates": [
                {
                    "label": item.label,
                    "manifest": str(item.manifest),
                    "success": item.success,
                }
                for item in templates
            ],
        },
    )
    print(
        f"[graspm3-lite] object={args.object_id} templates={len(templates)} "
        f"population={config.population_size} iterations={config.iterations} "
        f"modes={','.join(config.grasp_modes)} device={config.device}",
        flush=True,
    )

    env = MjWarpGraspM3LiteEnv(args.object_id, templates, config)
    try:
        search = TemporalCEMSearch(env, config, seed=args.seed)

        def report_iteration(row: dict[str, Any]) -> None:
            mode_successes = ",".join(
                f"{name}:{count}"
                for name, count in row["mode_mjwarp_successes"].items()
                if count
            ) or "none"
            print(
                f"[cem {row['iteration']:02d}/{config.iterations:02d}] "
                f"success={row['mjwarp_successes']}/{row['population']} "
                f"lift={1000.0 * row['best_max_lift']:.1f}mm "
                f"tail={1000.0 * row['best_tail_min_lift']:.1f}mm "
                f"modes={mode_successes}",
                flush=True,
            )

        result = search.run(callback=report_iteration)
    finally:
        env.close()

    candidate_root = output / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    verification_rows: list[dict[str, Any]] = []
    verified = None
    for rank, candidate in enumerate(result.verification_pool):
        directory = candidate_root / f"candidate_{rank:03d}"
        candidate.trajectory.save(directory)
        try:
            replay = replay_residual_trajectory(directory, verify_tail=args.verify_tail)
            candidate.trajectory.metadata.update(
                {
                    "c_mujoco_verified": bool(replay.success),
                    "verification_status": (
                        "C_MUJOCO_SUCCESS" if replay.success else "C_MUJOCO_REJECTED"
                    ),
                    "c_mujoco_success_fraction": float(replay.success_fraction),
                    "c_mujoco_object_lift": float(replay.object_lift),
                }
            )
            candidate.trajectory.success = bool(replay.success)
            candidate.trajectory.save(directory)
        except Exception as exc:  # noqa: BLE001 - preserve all candidate diagnostics
            replay = None
            candidate.trajectory.metadata.update(
                {
                    "c_mujoco_verified": False,
                    "verification_status": "C_MUJOCO_ERROR",
                    "c_mujoco_error": f"{type(exc).__name__}: {exc}",
                }
            )
            candidate.trajectory.success = False
            candidate.trajectory.save(directory)
        verification_rows.append(_candidate_summary(candidate, replay))
        if replay is not None and replay.success:
            candidate_rank = (not candidate.reference_schedule, candidate.score)
            if verified is None or candidate_rank > (
                not verified.reference_schedule,
                verified.score,
            ):
                verified = candidate

    if verified is not None:
        verified.trajectory.save(output / "best_trajectory")
        status = "C_MUJOCO_SUCCESS"
        print(
            f"[verified] template={verified.trajectory.metadata.get('template_label')} "
            f"lift={verified.trajectory.metadata.get('c_mujoco_object_lift', 0.0):.3f}m",
            flush=True,
        )
    else:
        if result.best_attempt is not None:
            result.best_attempt.save(output / "best_attempt")
        table_candidates = [
            row
            for row in verification_rows
            if row["table_assisted"] and row["mjwarp_success"]
        ]
        status = (
            "TABLE_ASSISTED_CANDIDATE_ONLY"
            if table_candidates
            else "NO_C_MUJOCO_SUCCESS"
        )
        print("[verified] no candidate passed authoritative C MuJoCo replay", flush=True)

    mode_stats: dict[str, dict[str, int]] = {
        name: {"candidates": 0, "mjwarp_successes": 0, "c_mujoco_successes": 0}
        for name in config.grasp_modes
    }
    for row in verification_rows:
        stats = mode_stats.setdefault(
            str(row["grasp_mode"]),
            {"candidates": 0, "mjwarp_successes": 0, "c_mujoco_successes": 0},
        )
        stats["candidates"] += 1
        stats["mjwarp_successes"] += int(bool(row["mjwarp_success"]))
        stats["c_mujoco_successes"] += int(bool(row["c_mujoco_success"]))

    summary = {
        "schema_version": 1,
        "pipeline": "graspm3-lite-temporal",
        "object_id": args.object_id,
        "status": status,
        "config": asdict(config),
        "search_history": list(result.history),
        "template_probabilities": list(result.template_probabilities),
        "mode_probabilities": list(result.mode_probabilities),
        "mode_stats": mode_stats,
        "verification_candidates": verification_rows,
        "best_trajectory": str(output / "best_trajectory") if verified else "",
        "best_attempt": str(output / "best_attempt") if verified is None else "",
    }
    _write_json(output / "summary.json", summary)
    print(f"[done] summary={output / 'summary.json'}", flush=True)
    return 0 if verified is not None or args.allow_unverified else 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
