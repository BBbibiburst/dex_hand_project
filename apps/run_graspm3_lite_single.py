"""Run the GraspM3-lite temporal search and strict C MuJoCo replay for one object."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from apps.train_grasp_edit_rl import _ensure_ultra_priors
from source.rl.grasp_edit.graspm3_lite import (
    TEMPORAL_PARAMETER_DIM,
    GraspM3LiteConfig,
    MjWarpGraspM3LiteEnv,
    TemporalCEMSearch,
    TemporalWarmStart,
)
from source.rl.grasp_edit.primitives import (
    available_grasp_primitives,
    resolve_grasp_primitives,
)
from source.rl.grasp_edit.templates import build_grasp_edit_templates
from source.rl.imitation.strict_replay import strict_replay_manifest
from source.rl.imitation.verification import FINAL_PROFILE
from source.rl.residual.trajectory import ResidualTrajectory


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
    parser.add_argument("--ingress-gain-max", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-edit-fraction", type=float, default=0.35)
    parser.add_argument("--success-lift-height", type=float, default=0.055)
    parser.add_argument("--success-tail-steps", type=int, default=20)
    parser.add_argument("--maximum-object-speed", type=float, default=0.10)
    parser.add_argument("--maximum-object-angular-speed", type=float, default=0.10)
    parser.add_argument("--minimum-tail-contact-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-tail-grasp-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-flat-thumb-fraction", type=float, default=0.55)
    parser.add_argument(
        "--warm-start",
        type=Path,
        action="append",
        default=[],
        help="Prior GraspM3-lite trajectory directory/manifest to refine.",
    )
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
        "mjwarp_tail_min_lift": metadata.get("mjwarp_tail_min_lift", 0.0),
        "mjwarp_tail_max_speed": metadata.get("mjwarp_tail_max_speed", 0.0),
        "mjwarp_tail_max_angular_speed": metadata.get(
            "mjwarp_tail_max_angular_speed", 0.0
        ),
        "mjwarp_tail_contact_fraction": metadata.get(
            "mjwarp_tail_contact_fraction", 0.0
        ),
        "mjwarp_tail_grasp_fraction": metadata.get(
            "mjwarp_tail_grasp_fraction", 0.0
        ),
        "mjwarp_tail_thumb_fraction": metadata.get(
            "mjwarp_tail_thumb_fraction", 0.0
        ),
        "c_mujoco_success": None if replay is None else replay.success,
        "verification_status": None if replay is None else replay.verification_status,
        "c_mujoco_final_lift": None if replay is None else replay.final_lift,
        "c_mujoco_max_lift": None if replay is None else replay.max_lift,
        "c_mujoco_tail_min_lift": None if replay is None else replay.tail_min_lift,
        "c_mujoco_tail_max_speed": None if replay is None else replay.tail_max_speed,
        "c_mujoco_tail_max_angular_speed": (
            None if replay is None else replay.tail_max_angular_speed
        ),
        "c_mujoco_tail_contact_fraction": (
            None if replay is None else replay.tail_contact_fraction
        ),
        "c_mujoco_tail_grasp_fraction": (
            None if replay is None else replay.tail_grasp_fraction
        ),
        "c_mujoco_tail_opposition_mean": (
            None if replay is None else replay.tail_opposition_mean
        ),
        "c_mujoco_quality_score": None if replay is None else replay.quality_score,
        "c_mujoco_error": metadata.get("c_mujoco_error"),
    }


def _mode_names(value: str) -> tuple[str, ...]:
    modes = resolve_grasp_primitives(value)
    return tuple(mode.name for mode in modes)


def _load_warm_starts(
    paths: list[Path],
    *,
    object_id: str,
    templates,
    mode_names: tuple[str, ...],
) -> tuple[TemporalWarmStart, ...]:
    template_by_label = {item.label: index for index, item in enumerate(templates)}
    mode_by_name = {name: index for index, name in enumerate(mode_names)}
    rows: list[TemporalWarmStart] = []
    for path in paths:
        trajectory = ResidualTrajectory.load(path)
        if trajectory.object_id != object_id:
            raise ValueError(
                f"Warm start {path} targets {trajectory.object_id}, not {object_id}."
            )
        parameters = np.asarray(
            trajectory.metadata.get("temporal_parameters", []), dtype=np.float32
        )
        if parameters.ndim != 1 or not 1 <= len(parameters) <= TEMPORAL_PARAMETER_DIM:
            raise ValueError(f"Warm start {path} has invalid temporal parameters.")
        if len(parameters) < TEMPORAL_PARAMETER_DIM:
            parameters = np.pad(
                parameters,
                (0, TEMPORAL_PARAMETER_DIM - len(parameters)),
            )
        template_label = str(trajectory.metadata.get("template_label", ""))
        mode_name = str(trajectory.metadata.get("grasp_mode", ""))
        if template_label not in template_by_label:
            raise ValueError(
                f"Warm-start template {template_label!r} is not in the current lattice."
            )
        if mode_name not in mode_by_name:
            raise ValueError(
                f"Warm-start mode {mode_name!r} is not among {mode_names}."
            )
        rows.append(
            TemporalWarmStart(
                parameters=parameters.astype(np.float32),
                template_id=template_by_label[template_label],
                mode_id=mode_by_name[mode_name],
            )
        )
    return tuple(rows)


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
        ingress_gain_max=args.ingress_gain_max,
        device=args.device,
        hand_edit_fraction=args.hand_edit_fraction,
        success_lift_height=args.success_lift_height,
        success_tail_steps=args.success_tail_steps,
        maximum_object_speed=args.maximum_object_speed,
        maximum_object_angular_speed=args.maximum_object_angular_speed,
        minimum_tail_contact_fraction=args.minimum_tail_contact_fraction,
        minimum_tail_grasp_fraction=args.minimum_tail_grasp_fraction,
        minimum_flat_thumb_fraction=args.minimum_flat_thumb_fraction,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    config.validate()
    warm_starts = _load_warm_starts(
        args.warm_start,
        object_id=args.object_id,
        templates=templates,
        mode_names=grasp_modes,
    )
    _write_json(
        output / "config.json",
        {
            "pipeline": "graspm3-lite-temporal",
            "object_id": args.object_id,
            "search": asdict(config),
            "grasp_modes": list(config.grasp_modes),
            "warm_starts": [str(path) for path in args.warm_start],
            "mode_definitions": [
                {
                    "name": mode.name,
                    "family": mode.mode_family,
                    "description": mode.description,
                    "objective": mode.objective_name,
                    "score_weights": list(mode.score_weights),
                    "table_assisted": mode.table_assisted,
                    "close_power_by_actuator": list(
                        mode.close_power_by_actuator
                        if mode.close_power_by_actuator is not None
                        else (mode.close_power,) * 6
                    ),
                    "ingress_scale": mode.ingress_scale,
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
        f"modes={','.join(config.grasp_modes)} warm_starts={len(warm_starts)} "
        f"device={config.device}",
        flush=True,
    )

    env = MjWarpGraspM3LiteEnv(args.object_id, templates, config)
    shape_summary = env.shape_summary()
    print(
        f"[geometry] family={shape_summary['family']} "
        f"extents_mm={','.join(f'{1000.0 * value:.1f}' for value in shape_summary['extents'])} "
        f"flatness={shape_summary['flatness_ratio']:.2f}",
        flush=True,
    )
    try:
        search = TemporalCEMSearch(
            env,
            config,
            seed=args.seed,
            warm_starts=warm_starts,
        )

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
                f"contact={100.0 * row['best_tail_contact_fraction']:.0f}% "
                f"grasp={100.0 * row['best_tail_grasp_fraction']:.0f}% "
                f"omega_min={row['lowest_tail_max_angular_speed']:.3f}rad/s "
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
            replay = strict_replay_manifest(
                directory,
                profile=FINAL_PROFILE,
                success_lift_height=args.success_lift_height,
                maximum_object_speed=args.maximum_object_speed,
                maximum_object_angular_speed=args.maximum_object_angular_speed,
                verify_tail=args.verify_tail,
                use_cache=False,
            )
            candidate.trajectory.metadata.update(
                {
                    "c_mujoco_verified": bool(replay.success),
                    "verification_status": replay.verification_status,
                    "strict_replay_profile": FINAL_PROFILE,
                    "c_mujoco_final_lift": float(replay.final_lift),
                    "c_mujoco_max_lift": float(replay.max_lift),
                    "c_mujoco_tail_min_lift": float(replay.tail_min_lift),
                    "c_mujoco_tail_max_speed": float(replay.tail_max_speed),
                    "c_mujoco_tail_max_angular_speed": float(
                        replay.tail_max_angular_speed
                    ),
                    "c_mujoco_tail_contact_fraction": float(
                        replay.tail_contact_fraction
                    ),
                    "c_mujoco_tail_grasp_fraction": float(
                        replay.tail_grasp_fraction
                    ),
                    "c_mujoco_tail_opposition_mean": float(
                        replay.tail_opposition_mean
                    ),
                    "c_mujoco_quality_score": float(replay.quality_score),
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
            candidate_rank = (
                not candidate.reference_schedule,
                replay.quality_score,
                candidate.score,
            )
            if verified is None or candidate_rank > (
                not verified.reference_schedule,
                float(verified.trajectory.metadata.get("c_mujoco_quality_score", 0.0)),
                verified.score,
            ):
                verified = candidate

    if verified is not None:
        verified.trajectory.save(output / "best_trajectory")
        status = "FINAL_VERIFIED"
        print(
            f"[verified] template={verified.trajectory.metadata.get('template_label')} "
            f"lift={verified.trajectory.metadata.get('c_mujoco_final_lift', 0.0):.3f}m",
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
            else "NO_FINAL_VERIFIED"
        )
        print("[verified] no candidate passed strict final C MuJoCo replay", flush=True)

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
        "geometry": shape_summary,
        "verification_profile": FINAL_PROFILE,
        "warm_starts": [str(path) for path in args.warm_start],
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
