"""Configuration and catalogue selection for grasp benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from source.grasping.constants import DEFAULT_GRIP_PRELOAD

@dataclass
class GraspBenchmarkConfig:
    """Configuration for a catalogue-wide grasp search and validation run."""

    dataset: str = "all"
    object_ids: list[str] | None = None
    limit: int | None = None
    points: int = 2048
    joint_candidates: int = 128
    surface_anchors: int = 24
    rolls_per_anchor: int = 8
    coarse_keep: int = 24
    top_k: int = 8
    support_margin: float = 0.008
    generator: str = "heuristic"
    graspqp_iterations: int = 120
    evolve: bool = False
    evolution_population: int = 32
    evolution_offspring: int = 16
    evolution_generations: int = 20
    evolution_jobs: int = 4
    evolution_seconds: float = 1.5
    evolution_backend: str = "cpu"
    mjwarp_device: str = "cuda:0"
    mjwarp_batch_size: int = 32
    mjwarp_nconmax: int = 128
    mjwarp_njmax: int = 512
    evolution_dir: Path | None = None
    search_attempts: int = 3
    target_lift_candidates: int = 1
    maximum_saved_candidates: int = 24
    maximum_robot_candidates_per_attempt: int = 6
    task_conditioned_search: bool = False
    task_scene_attempts: int = 12
    task_rotations_per_distance: int = 4
    task_pull_step: float = 0.05
    task_maximum_pull: float = 0.15
    maximum_object_seconds: float = 2700.0
    seed: int = 0
    target_size: float = 0.09
    end_effector: str = "dex_hand"
    seconds: float = 3.0
    settle_seconds: float = 0.8
    grip_preload: float = DEFAULT_GRIP_PRELOAD
    jobs: int = 1
    reuse: bool = False
    validate_robot_lift: bool = False
    pilot: bool = False
    pilot_min_results: int = 4
    pilot_min_lift_rate: float = 0.25
    pilot_max_repeated_failure: int = 3
    resume: bool = False
    retry_incomplete: bool = False
    config_dir: Path | None = None
    output: Path | None = None


def _selected_ids(args: "GraspBenchmarkConfig") -> list[str]:
    from source.envs.manipulation.object_catalog import object_ids

    available = object_ids(None if args.dataset == "all" else args.dataset)
    if args.object_ids:
        unknown = sorted(set(args.object_ids) - set(available))
        if unknown:
            raise ValueError(f"Objects outside selected catalogue: {unknown}")
        selected = list(dict.fromkeys(args.object_ids))
    else:
        selected = list(available)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        selected = selected[: args.limit]
    return selected
