"""Search and physics-validate grasps for catalogue objects."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.grasping.constants import DEFAULT_GRIP_PRELOAD
from source.workflows.grasp_benchmark import GraspBenchmarkConfig, run_grasp_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Use the standard GraspQP -> DexEvolve -> MuJoCo full-catalog preset.",
    )
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--joint-candidates", type=int, default=128)
    parser.add_argument("--surface-anchors", type=int, default=24)
    parser.add_argument("--rolls-per-anchor", type=int, default=8)
    parser.add_argument("--coarse-keep", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--support-margin", type=float, default=0.008)
    parser.add_argument("--generator", choices=("heuristic", "graspqp"), default="heuristic")
    parser.add_argument("--graspqp-iterations", type=int, default=120)
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Run DexEvolve-style MuJoCo refinement after GraspQP generation.",
    )
    parser.add_argument("--evolution-population", type=int, default=32)
    parser.add_argument("--evolution-offspring", type=int, default=16)
    parser.add_argument("--evolution-generations", type=int, default=20)
    parser.add_argument("--evolution-jobs", type=int, default=4)
    parser.add_argument("--evolution-seconds", type=float, default=1.5)
    parser.add_argument("--evolution-dir", type=Path)
    parser.add_argument("--search-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument(
        "--end-effector", choices=("dex_hand", "pika_gripper"), default="dex_hand"
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--grip-preload", type=float, default=DEFAULT_GRIP_PRELOAD)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = vars(args)
    full_pipeline = values.pop("full_pipeline")
    if full_pipeline:
        preset = {
            "dataset": "all",
            "generator": "graspqp",
            "graspqp_iterations": 40,
            "points": 1024,
            "joint_candidates": 64,
            "surface_anchors": 12,
            "rolls_per_anchor": 4,
            "coarse_keep": 12,
            "top_k": 4,
            "search_attempts": 5,
            "seconds": 3.0,
            "jobs": 2,
            "evolve": True,
            "evolution_population": 32,
            "evolution_offspring": 16,
            "evolution_generations": 20,
            "evolution_jobs": 8,
            "evolution_seconds": 1.5,
        }
        values.update(preset)
        if values["output"] is None:
            values["output"] = Path("configs/grasps/dex_hand/full_pipeline_benchmark.json")
        if values["config_dir"] is None:
            values["config_dir"] = Path("configs/grasps/dex_hand/graspqp_seeds")
        if values["evolution_dir"] is None:
            values["evolution_dir"] = Path("configs/grasps/dex_hand/dexevolve")
    raise SystemExit(run_grasp_benchmark(GraspBenchmarkConfig(**values)))


if __name__ == "__main__":
    main()
