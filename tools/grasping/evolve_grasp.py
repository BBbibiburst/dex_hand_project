"""Refine a geometric grasp seed with DexEvolve-style MuJoCo evolution."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from source.grasping.dexevolve import EvolutionConfig, evolve
from source.grasping.standalone_validator import validate_grasp_payload_trajectory


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
    selected = None
    trajectory_errors = []
    for individual in archive:
        if not individual.direct_hold_stable:
            continue
        candidate = dict(individual.payload)
        try:
            validation = validate_grasp_payload_trajectory(
                candidate,
                seconds=args.seconds,
            )
        except Exception as exc:  # noqa: BLE001 - try the next evolved candidate
            trajectory_errors.append(str(exc))
            continue
        if validation.trajectory_hold_stable:
            candidate.update(
                direct_hold_stable=True,
                trajectory_collision_free=True,
                trajectory_hold_stable=True,
                validation_stage="trajectory_hold_stable",
            )
            selected = (candidate, validation)
            break
    report = args.report or args.output.with_suffix(".evolution.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "summary": {
                    "archive": len(archive),
                    "direct_hold_stable": sum(item.direct_hold_stable for item in archive),
                    "trajectory_hold_stable": selected is not None,
                    "best_fitness": best.fitness,
                    "best_direct_hold_stable": best.direct_hold_stable,
                    "trajectory_validation_errors": trajectory_errors[:8],
                },
                "history": history,
                "individuals": [
                    {
                        "fitness": item.fitness,
                        "direct_hold_stable": item.direct_hold_stable,
                        "metrics": item.metrics,
                    }
                    for item in archive
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if selected is None:
        raise RuntimeError(
            "Evolution found no grasp that passes executable approach/closure/hold "
            f"validation; diagnostic report: {report}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(selected[0], indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"trajectory_hold_stable=True output={args.output} report={report}")


if __name__ == "__main__":
    main()
