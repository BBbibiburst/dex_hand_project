"""Run DexEvolve on one recorded GraspQP/Ultra candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.envs.manipulation import make_lift_env
from source.ultradexgrasp.contracts import DemonstrationEpisode
from source.ultradexgrasp.catalog import load_object_geometry
from source.ultradexgrasp.config import load_pipeline_config
from source.ultradexgrasp.dexevolve import DexEvolveConfig, evolve_candidate
from source.ultradexgrasp.hand_surrogate import load_or_calibrate_surrogate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--offspring", type=int, default=6)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = DemonstrationEpisode.load(args.manifest)
    pipeline = load_pipeline_config()
    geometry = load_object_geometry(
        source.object_id,
        target_size=pipeline.target_size,
        maximum_horizontal_diameter=pipeline.maximum_horizontal_diameter,
        surface_points=pipeline.surface_points,
        seed=args.seed,
    )
    surrogate = load_or_calibrate_surrogate(pipeline.surrogate_cache, **pipeline.surrogate_options)
    config = DexEvolveConfig(
        population_size=args.population,
        offspring=args.offspring,
        generations=args.generations,
        seed=args.seed,
    )
    env = make_lift_env(
        task_config={"object_id": source.object_id, "terminate_on_success": False},
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
        episode_length=pipeline.execution.maximum_steps + 20,
    )
    try:
        archive, history = evolve_candidate(
            source.candidate,
            environment=env,
            geometry=geometry,
            surrogate=surrogate,
            execution=pipeline.execution,
            config=config,
        )
    finally:
        env.close()
    args.output.mkdir(parents=True, exist_ok=True)
    best = archive[0]
    if best.episode is not None:
        best.episode.metadata["dexevolve"] = {
            "source_manifest": str(args.manifest.resolve()),
            "fitness": best.fitness,
            "history": list(history),
        }
        manifest = best.episode.save(args.output)
    else:
        manifest = None
    summary = {
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "success": best.success,
        "best_fitness": best.fitness,
        "best_candidate": best.candidate.to_dict(),
        "manifest": None if manifest is None else manifest.name,
        "history": list(history),
    }
    (args.output / "dexevolve.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[dexevolve] success={best.success} fitness={best.fitness:.3f} output={args.output}",
        flush=True,
    )
    return 0 if best.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
