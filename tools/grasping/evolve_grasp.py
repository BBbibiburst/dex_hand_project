"""Refine a GraspQP seed with DexEvolve-style MuJoCo evolution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from source.grasping.dexevolve import EvolutionConfig, evolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--offspring", type=int, default=16)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--seed-value", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.seed.read_text(encoding="utf-8"))
    config = EvolutionConfig(
        population_size=args.population_size,
        offspring=args.offspring,
        generations=args.generations,
        jobs=args.jobs,
        seconds=args.seconds,
        seed=args.seed_value,
    )
    archive, history = evolve(payload, config)
    best = archive[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(best.payload, indent=2), encoding="utf-8")
    report = args.report or args.output.with_suffix(".evolution.json")
    report.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "summary": {
                    "archive": len(archive),
                    "stable": sum(item.stable for item in archive),
                    "best_fitness": best.fitness,
                    "best_stable": best.stable,
                },
                "history": history,
                "individuals": [
                    {"fitness": item.fitness, "stable": item.stable, "metrics": item.metrics}
                    for item in archive
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"stable={sum(item.stable for item in archive)}/{len(archive)} output={args.output}")


if __name__ == "__main__":
    main()
