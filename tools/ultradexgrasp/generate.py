"""Generate and execute a complete UltraDexGrasp demonstration episode."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from source.envs.manipulation import make_lift_env
from source.ultradexgrasp.catalog import load_object_geometry
from source.ultradexgrasp.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from source.ultradexgrasp.contracts import PIPELINE_NAME
from source.ultradexgrasp.executor import (
    execute_grasp,
    rank_candidates_for_scene,
)
from source.ultradexgrasp.hand_surrogate import load_or_calibrate_surrogate
from source.ultradexgrasp.synthesizer import synthesize_grasps


def _slug(object_id: str) -> str:
    return object_id.replace(":", "_").replace("/", "_")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", default="ycb:002_master_chef_can")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--optimization-steps", type=int)
    parser.add_argument("--max-execution-candidates", type=int, default=8)
    parser.add_argument("--synthesis-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_execution_candidates <= 0:
        raise ValueError("--max-execution-candidates must be positive.")
    pipeline = load_pipeline_config(args.config)
    synthesis = pipeline.synthesis
    overrides = {"seed": args.seed}
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed_count is not None:
        overrides["seed_count"] = args.seed_count
    if args.optimization_steps is not None:
        overrides["optimization_steps"] = args.optimization_steps
    synthesis = replace(synthesis, **overrides)
    synthesis.validate()

    output = args.output or Path("outputs/ultradexgrasp") / _slug(args.object_id) / (
        f"seed_{args.seed:04d}"
    )
    if (output / "manifest.json").exists() and not args.overwrite:
        raise FileExistsError(f"Episode already exists at {output}; pass --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)

    print(f"[calibrate] cache={pipeline.surrogate_cache}", flush=True)
    surrogate = load_or_calibrate_surrogate(
        pipeline.surrogate_cache,
        **pipeline.surrogate_options,
    )
    print(
        f"[geometry] object={args.object_id} points={pipeline.surface_points}",
        flush=True,
    )
    geometry = load_object_geometry(
        args.object_id,
        target_size=pipeline.target_size,
        surface_points=pipeline.surface_points,
        seed=args.seed,
    )

    def progress(step: int, total: int, metrics: dict[str, float]) -> None:
        print(
            f"[optimize] {step:4d}/{total} loss={metrics['loss']:.3f} "
            f"contact={1000.0 * metrics['contact_distance']:.2f}mm "
            f"penetration={1000.0 * metrics['maximum_penetration']:.2f}mm "
            f"force={metrics['force_residual']:.3f}",
            flush=True,
        )

    candidates = synthesize_grasps(geometry, surrogate, synthesis, progress=progress)
    valid_candidates = tuple(
        candidate for candidate in candidates if bool(candidate.metrics.get("valid", 0.0))
    )
    archive_payload: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": PIPELINE_NAME,
        "object_id": args.object_id,
        "seed": args.seed,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    _write_json(output / "candidates.json", archive_payload)
    if args.synthesis_only:
        print(f"[done] candidates={output / 'candidates.json'}", flush=True)
        return 0 if valid_candidates else 2
    if not valid_candidates:
        print("[failed] synthesis returned no valid candidate", flush=True)
        return 2

    rank_env = make_lift_env(
        task_config={"object_id": args.object_id},
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
    )
    try:
        observation, _ = rank_env.reset(seed=args.seed)
        ranked = rank_candidates_for_scene(
            rank_env,
            valid_candidates,
            observation["object_pos"],
            observation["object_quat"],
            pregrasp_distance=pipeline.execution.pregrasp_distance,
        )
    finally:
        rank_env.close()

    archive_payload["reachability"] = [
        {
            "seed_index": result.candidate.seed_index,
            "score": result.score,
            "maximum_position_error": result.maximum_position_error,
            "maximum_orientation_error": result.maximum_orientation_error,
        }
        for result in ranked
    ]
    _write_json(output / "candidates.json", archive_payload)

    attempts: list[dict[str, Any]] = []
    for rank, result in enumerate(ranked[: args.max_execution_candidates]):
        candidate = result.candidate
        print(
            f"[execute] rank={rank} seed_index={candidate.seed_index} "
            f"reachability={result.score:.3f}",
            flush=True,
        )
        episode = execute_grasp(
            candidate,
            seed=args.seed,
            config=pipeline.execution,
            render_mode="human" if args.render else None,
        )
        episode.metadata["reachability"] = {
            "rank": rank,
            "score": result.score,
            "maximum_position_error": result.maximum_position_error,
            "maximum_orientation_error": result.maximum_orientation_error,
        }
        attempt = {
            "rank": rank,
            "seed_index": candidate.seed_index,
            "success": episode.success,
            "terminal_stage": episode.terminal_stage,
            "failure_reason": episode.failure_reason,
        }
        attempts.append(attempt)
        if episode.success:
            manifest = episode.save(output)
            _write_json(
                output / "run.json",
                {
                    "schema_version": 1,
                    "pipeline": PIPELINE_NAME,
                    "success": True,
                    "manifest": manifest.name,
                    "attempts": attempts,
                },
            )
            print(f"[success] manifest={manifest}", flush=True)
            return 0
        episode.save(output / "attempts" / f"rank_{rank:02d}_seed_{candidate.seed_index:03d}")
        print(f"[retry] {episode.failure_reason}", flush=True)

    _write_json(
        output / "run.json",
        {
            "schema_version": 1,
            "pipeline": PIPELINE_NAME,
            "success": False,
            "attempts": attempts,
        },
    )
    print(f"[failed] no execution succeeded; details={output / 'run.json'}", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
